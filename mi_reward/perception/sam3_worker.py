"""SAM3 worker that materializes task-object masks for base trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}.")
        records.append(item)
    if not records:
        raise ValueError(f"No base trajectories found in {path}.")
    return records


def _resolve_frames(record: dict[str, Any], root: Path) -> list[Path]:
    values = record.get("frames")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"Base record {record.get('base_id', '<unknown>')} requires at least two frames.")
    frames = [Path(str(value)) if Path(str(value)).is_absolute() else (root / str(value)).resolve() for value in values]
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Base record references missing frames: {missing[:3]}")
    return frames


def _checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [path / "sam3.pt", path / "model.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SAM3 checkpoint not found under {path}. Expected sam3.pt.")


def _mask_array(output: dict[str, Any], threshold: float, size: tuple[int, int]) -> np.ndarray:
    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    if hasattr(masks, "detach"):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"SAM3 returned unsupported mask shape {masks.shape}.")
    if scores is not None:
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        score_values = np.asarray(scores).reshape(-1)
        if score_values.shape[0] == masks.shape[0]:
            masks = masks[score_values >= threshold]
    if masks.shape[0] == 0:
        return np.zeros((size[1], size[0]), dtype=bool)
    union = np.any(masks > 0, axis=0)
    if union.shape != (size[1], size[0]):
        union = np.asarray(
            Image.fromarray(union.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
        ) > 0
    return union


class Sam3ImageSegmenter:
    """Small pinned-API adapter around the official SAM3 image processor."""

    def __init__(self, checkpoint: Path, device: str, score_threshold: float):
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError("SAM3 is not installed in .venv. Run requirements/install.sh --all.") from exc
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(self.model)
        self.score_threshold = score_threshold

    def segment(self, image: Image.Image, prompt: str) -> np.ndarray:
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return _mask_array(output, self.score_threshold, image.size)


def run_worker(
    request_path: Path,
    result_path: Path,
    checkpoint_root: Path,
    *,
    device: str = "cuda",
    score_threshold: float = 0.35,
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    segmenter = Sam3ImageSegmenter(_checkpoint(checkpoint_root), device, score_threshold)
    generated: list[dict[str, Any]] = []
    for record_index, record in enumerate(_read_jsonl(input_path)):
        frames = _resolve_frames(record, input_path.parent)
        prompts = record.get("object_prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(f"Record {record.get('base_id', record_index)} has no object_prompts.")
        record_id = str(record.get("base_id") or record.get("traj_id") or f"record_{record_index:06d}")
        root = output_path.parent / "sam3" / f"record_{record_index:06d}"
        composite_root = root / "composite"
        composite_root.mkdir(parents=True, exist_ok=True)
        object_roots = {str(key): root / "objects" / str(key) for key in prompts}
        for directory in object_roots.values():
            directory.mkdir(parents=True, exist_ok=True)
        for frame_index, frame_path in enumerate(frames):
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
                composite = np.zeros((image.height, image.width), dtype=bool)
                for object_id, prompt in prompts.items():
                    mask = segmenter.segment(image, str(prompt))
                    if not mask.any():
                        raise RuntimeError(
                            f"SAM3 found no {prompt!r} in {record_id} frame {frame_index}; adjust object_prompts."
                        )
                    composite |= mask
                    Image.fromarray(mask.astype(np.uint8) * 255).save(
                        object_roots[str(object_id)] / f"{frame_index:06d}.png"
                    )
                Image.fromarray(composite.astype(np.uint8) * 255).save(composite_root / f"{frame_index:06d}.png")
        updated = dict(record)
        controls = dict(updated.get("control_artifacts") or {})
        controls.update(
            {
                "mask_root": str(composite_root.resolve()),
                "segmentation_root": str((root / "objects").resolve()),
            }
        )
        updated["control_artifacts"] = controls
        metadata = dict(updated.get("metadata") or {})
        metadata["sam3"] = {
            "checkpoint": str(_checkpoint(checkpoint_root).resolve()),
            "score_threshold": score_threshold,
            "object_prompts": {str(key): str(value) for key, value in prompts.items()},
        }
        updated["metadata"] = metadata
        generated.append(updated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"records_path": str(output_path)}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synchronized task masks with the official SAM3 package.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint-root", default=".venv/models/sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1].")
    run_worker(
        Path(args.request),
        Path(args.result),
        Path(args.checkpoint_root).resolve(),
        device=args.device,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
