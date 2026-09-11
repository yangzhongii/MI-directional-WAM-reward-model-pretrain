"""Optional SAM 2.1/SAM 3 segmentation for base trajectories.

SAM 2.1 is the default backend because its code and checkpoints are public and
its video predictor is a good fit for offline robot trajectories. Unlike SAM 3,
SAM 2.1 does not accept text prompts: records must provide first-frame point or
box prompts in ``segmentation_prompts``. When the downstream MuJoCo stage will
render its own masks, ``--allow-missing-prompts`` can pass records through
without inventing a source-image mask.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


SAM2_CONFIG_CHECKPOINTS = {
    "configs/sam2.1/sam2.1_hiera_t.yaml": "sam2.1_hiera_tiny.pt",
    "configs/sam2.1/sam2.1_hiera_s.yaml": "sam2.1_hiera_small.pt",
    "configs/sam2.1/sam2.1_hiera_b+.yaml": "sam2.1_hiera_base_plus.pt",
    "configs/sam2.1/sam2.1_hiera_l.yaml": "sam2.1_hiera_large.pt",
}


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
    frames = [
        Path(str(value)) if Path(str(value)).is_absolute() else (root / str(value)).resolve() for value in values
    ]
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Base record references missing frames: {missing[:3]}")
    return frames


def _checkpoint(path: Path, backend: str, model_config: str) -> Path:
    if path.is_file():
        return path
    if backend == "sam2":
        checkpoint_name = SAM2_CONFIG_CHECKPOINTS.get(model_config)
        if checkpoint_name is None:
            raise ValueError(
                f"Unsupported SAM 2.1 model config {model_config!r}; expected one of {sorted(SAM2_CONFIG_CHECKPOINTS)}."
            )
        candidate = path / checkpoint_name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {candidate}")
    candidates = [path / "sam3.pt", path / "model.pt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SAM 3 checkpoint not found under {path}. Expected sam3.pt.")


def _segmentation_prompts(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = record.get("segmentation_prompts")
    if raw is None and isinstance(record.get("metadata"), dict):
        raw = record["metadata"].get("segmentation_prompts")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("segmentation_prompts must be an object keyed by object ID.")
    prompts: dict[str, dict[str, Any]] = {}
    for object_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"segmentation_prompts[{object_id!r}] must be an object.")
        frame_index = int(value.get("frame_index", 0))
        if frame_index != 0:
            raise ValueError("SAM 2.1 prompts currently must initialize frame_index 0 for full-trajectory propagation.")
        box = value.get("box")
        points = value.get("points")
        labels = value.get("labels")
        if box is None and points is None:
            raise ValueError(f"segmentation_prompts[{object_id!r}] requires `box` or `points`.")
        if box is not None and (not isinstance(box, list) or len(box) != 4):
            raise ValueError(f"segmentation_prompts[{object_id!r}].box must be [x0, y0, x1, y1].")
        if points is not None:
            if (
                not isinstance(points, list)
                or not points
                or any(not isinstance(point, list) or len(point) != 2 for point in points)
            ):
                raise ValueError(f"segmentation_prompts[{object_id!r}].points must contain [x, y] pairs.")
            if labels is None:
                labels = [1] * len(points)
            if not isinstance(labels, list) or len(labels) != len(points):
                raise ValueError(f"segmentation_prompts[{object_id!r}].labels must match points.")
        prompts[str(object_id)] = {
            "frame_index": frame_index,
            "box": box,
            "points": points,
            "labels": labels,
            "normalized": bool(value.get("normalized", False)),
        }
    return prompts


def _scaled_prompt(
    prompt: dict[str, Any], size: tuple[int, int]
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    width, height = size
    scale = np.asarray([width, height], dtype=np.float32)
    box = None if prompt["box"] is None else np.asarray(prompt["box"], dtype=np.float32)
    points = None if prompt["points"] is None else np.asarray(prompt["points"], dtype=np.float32)
    labels = None if prompt["labels"] is None else np.asarray(prompt["labels"], dtype=np.int32)
    if prompt["normalized"]:
        if box is not None:
            box = box * np.asarray([width, height, width, height], dtype=np.float32)
        if points is not None:
            points = points * scale
    return box, points, labels


def _mask_from_logits(logits: Any, size: tuple[int, int]) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    values = np.asarray(logits)
    while values.ndim > 2:
        values = values[0]
    mask = values > 0.0
    if mask.shape != (size[1], size[0]):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
        ) > 0
    return mask


class Sam2VideoSegmenter:
    """Pinned adapter around the official SAM 2.1 video predictor."""

    def __init__(self, checkpoint: Path, model_config: str, device: str):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2.1 is not installed. Run requirements/install.sh --generalization-data --sam-backend sam2."
            ) from exc
        self.torch = torch
        self.device = device
        self.predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=device)

    def track(
        self,
        frames: list[Path],
        prompts: dict[str, dict[str, Any]],
        staging_root: Path,
    ) -> dict[str, list[np.ndarray]]:
        with Image.open(frames[0]) as first:
            size = first.size
        masks = {object_id: [np.zeros((size[1], size[0]), dtype=bool) for _ in frames] for object_id in prompts}
        with TemporaryDirectory(prefix="sam2-video-", dir=staging_root) as temporary:
            video_dir = Path(temporary)
            for index, frame in enumerate(frames):
                with Image.open(frame) as source:
                    source.convert("RGB").save(video_dir / f"{index:06d}.jpg", quality=95)
            inference_context = self.torch.inference_mode()
            autocast_context = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with inference_context, autocast_context:
                state = self.predictor.init_state(video_path=str(video_dir))
                object_ids = list(prompts)
                for numeric_id, object_id in enumerate(object_ids, start=1):
                    box, points, labels = _scaled_prompt(prompts[object_id], size)
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=numeric_id,
                        points=points,
                        labels=labels,
                        box=box,
                    )
                for frame_index, output_ids, output_logits in self.predictor.propagate_in_video(state):
                    for output_index, numeric_id in enumerate(output_ids):
                        if hasattr(numeric_id, "item"):
                            numeric_id = numeric_id.item()
                        object_id = object_ids[int(numeric_id) - 1]
                        masks[object_id][int(frame_index)] = _mask_from_logits(output_logits[output_index], size)
        return masks


class Sam3TextSegmenter:
    """Compatibility adapter for the previous text-prompt SAM 3 backend."""

    def __init__(self, checkpoint: Path, device: str, score_threshold: float):
        from mi_reward.perception.sam3_worker import Sam3ImageSegmenter

        self.segmenter = Sam3ImageSegmenter(checkpoint, device, score_threshold)

    def track(
        self,
        frames: list[Path],
        prompts: dict[str, str],
        staging_root: Path,
    ) -> dict[str, list[np.ndarray]]:
        del staging_root
        masks = {object_id: [] for object_id in prompts}
        for frame in frames:
            with Image.open(frame) as source:
                image = source.convert("RGB")
                for object_id, prompt in prompts.items():
                    masks[object_id].append(self.segmenter.segment(image, prompt))
        return masks


def _write_masks(
    masks: dict[str, list[np.ndarray]],
    root: Path,
    record_id: str,
) -> tuple[Path, Path]:
    object_root = root / "objects"
    composite_root = root / "composite"
    composite_root.mkdir(parents=True, exist_ok=True)
    frame_count = len(next(iter(masks.values())))
    for object_id, object_masks in masks.items():
        if len(object_masks) != frame_count:
            raise ValueError(f"Segmentation frame count mismatch for {object_id!r} in {record_id}.")
        directory = object_root / object_id
        directory.mkdir(parents=True, exist_ok=True)
        for frame_index, mask in enumerate(object_masks):
            if not mask.any():
                raise RuntimeError(f"Segmentation found no {object_id!r} in {record_id} frame {frame_index}.")
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{frame_index:06d}.png")
    for frame_index in range(frame_count):
        composite = np.any([object_masks[frame_index] for object_masks in masks.values()], axis=0)
        Image.fromarray(composite.astype(np.uint8) * 255).save(composite_root / f"{frame_index:06d}.png")
    return composite_root, object_root


def run_worker(
    request_path: Path,
    result_path: Path,
    checkpoint_root: Path,
    *,
    backend: str = "sam2",
    model_config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
    device: str = "cuda",
    score_threshold: float = 0.35,
    allow_missing_prompts: bool = False,
) -> None:
    if backend not in {"sam2", "sam3", "none"}:
        raise ValueError(f"Unsupported segmentation backend: {backend!r}.")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    records = _read_jsonl(input_path)
    segmenter: Sam2VideoSegmenter | Sam3TextSegmenter | None = None
    checkpoint: Path | None = None
    generated: list[dict[str, Any]] = []
    for record_index, record in enumerate(
        tqdm(records, desc=f"Segmentation ({backend})", unit="trajectory")
    ):
        updated = dict(record)
        record_id = str(record.get("base_id") or record.get("traj_id") or f"record_{record_index:06d}")
        metadata = dict(updated.get("metadata") or {})
        if backend == "none":
            metadata["segmentation"] = {"backend": "none", "status": "skipped"}
            updated["metadata"] = metadata
            generated.append(updated)
            continue
        frames = _resolve_frames(record, input_path.parent)
        if backend == "sam2":
            prompts: dict[str, Any] = _segmentation_prompts(record)
            if not prompts:
                if not allow_missing_prompts:
                    raise ValueError(
                        f"Record {record_id} has no segmentation_prompts. SAM 2.1 requires first-frame boxes or points; "
                        "text-only object_prompts are supported only by SAM 3."
                    )
                metadata["segmentation"] = {
                    "backend": "sam2",
                    "status": "skipped-missing-prompts",
                    "reason": "downstream simulator must provide masks",
                }
                updated["metadata"] = metadata
                generated.append(updated)
                continue
        else:
            raw_prompts = record.get("object_prompts")
            if not isinstance(raw_prompts, dict) or not raw_prompts:
                raise ValueError(f"Record {record_id} has no object_prompts for SAM 3.")
            prompts = {str(key): str(value) for key, value in raw_prompts.items()}
        if segmenter is None:
            checkpoint = _checkpoint(checkpoint_root, backend, model_config)
            segmenter = (
                Sam2VideoSegmenter(checkpoint, model_config, device)
                if backend == "sam2"
                else Sam3TextSegmenter(checkpoint, device, score_threshold)
            )
        root = output_path.parent / "segmentation" / f"record_{record_index:06d}"
        root.mkdir(parents=True, exist_ok=True)
        masks = segmenter.track(frames, prompts, root)
        composite_root, object_root = _write_masks(masks, root, record_id)
        controls = dict(updated.get("control_artifacts") or {})
        controls.update({"mask_root": str(composite_root.resolve()), "segmentation_root": str(object_root.resolve())})
        updated["control_artifacts"] = controls
        metadata["segmentation"] = {
            "backend": backend,
            "status": "completed",
            "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
            "model_config": model_config if backend == "sam2" else None,
            "score_threshold": score_threshold if backend == "sam3" else None,
            "object_ids": list(prompts),
        }
        updated["metadata"] = metadata
        generated.append(updated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"records_path": str(output_path)}), encoding="utf-8")


def _load_backend_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Segmentation install config not found: {path}. "
            "Run requirements/install.sh --generalization-data first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Segmentation install config must be a JSON object: {path}")
    required = {"backend", "checkpoint_root", "model_config"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Segmentation install config is missing {sorted(missing)}: {path}")
    return {key: str(payload[key]) for key in required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synchronized task masks with SAM 2.1 or SAM 3.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--backend-config",
        help="Installer-written JSON selecting the backend, checkpoint root, and SAM 2.1 model config.",
    )
    parser.add_argument("--backend", choices=("sam2", "sam3", "none"), default="sam2")
    parser.add_argument("--checkpoint-root", default=".venv/models/sam2")
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--allow-missing-prompts", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1].")
    if args.backend_config:
        backend_config = _load_backend_config(Path(args.backend_config).resolve())
        args.backend = backend_config["backend"]
        args.checkpoint_root = backend_config["checkpoint_root"]
        args.model_config = backend_config["model_config"]
    run_worker(
        Path(args.request),
        Path(args.result),
        Path(args.checkpoint_root).resolve(),
        backend=args.backend,
        model_config=args.model_config,
        device=args.device,
        score_threshold=args.score_threshold,
        allow_missing_prompts=args.allow_missing_prompts,
    )


if __name__ == "__main__":
    main()
