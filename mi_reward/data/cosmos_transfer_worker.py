"""Cosmos Transfer2.5 stage worker for scene-level trajectory variation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


_RESAMPLING = getattr(Image, "Resampling", Image)
_VARIANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RECORD_PATH_FIELDS = (
    "action_path",
    "robot_state_path",
    "object_state_path",
    "relation_path",
    "context_video_path",
)
_CONTROL_PATH_FIELDS = ("mask_root", "depth_root", "segmentation_root", "edge_root")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-4000:]}")


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError("Install imageio-ffmpeg in .venv or provide a system ffmpeg binary.")


def _resolved(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_no}.")
        records.append(payload)
    if not records:
        raise ValueError(f"No input records found in {path}.")
    return records


def _load_variants(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list) or not variants:
        raise ValueError("Scene variant config requires a non-empty `variants` list.")
    seen_ids: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict) or not variant.get("variant_id") or not variant.get("prompt"):
            raise ValueError("Every scene variant requires `variant_id` and `prompt`.")
        variant_id = variant["variant_id"]
        prompt = variant["prompt"]
        if not isinstance(variant_id, str) or not _VARIANT_ID_PATTERN.fullmatch(variant_id):
            raise ValueError("Scene variant IDs may contain only letters, digits, dots, underscores, and hyphens.")
        if variant_id in seen_ids:
            raise ValueError(f"Duplicate scene variant ID: {variant_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Scene variant {variant_id!r} requires a non-empty string prompt.")
        control_weight = float(variant.get("control_weight", 1.0))
        if not 0.0 <= control_weight <= 1.0:
            raise ValueError(f"Scene variant {variant_id!r} control_weight must be in [0, 1].")
        seen_ids.add(variant_id)
    return variants


def _prepare_staging(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.png"):
        stale.unlink()


def _encode_frames(frames: list[Path], output: Path, fps: int) -> tuple[int, int]:
    ffmpeg = _ffmpeg()
    staging = output.parent / f"{output.stem}_rgb"
    _prepare_staging(staging)
    size: tuple[int, int] | None = None
    for index, source in enumerate(frames):
        if not source.is_file():
            raise FileNotFoundError(f"Input frame not found: {source}")
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            size = size or rgb.size
            if rgb.size != size:
                rgb = rgb.resize(size, _RESAMPLING.BILINEAR)
            rgb.save(staging / f"{index:06d}.png")
    assert size is not None
    _run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(staging / "%06d.png"), "-pix_fmt", "yuv420p", str(output),
    ])
    return size


def _encode_depth(depth_root: Path, output: Path, frame_count: int, size: tuple[int, int], fps: int) -> None:
    ffmpeg = _ffmpeg()
    paths = sorted(depth_root.glob("*.npy"))
    if len(paths) != frame_count:
        raise ValueError(f"Expected {frame_count} depth maps in {depth_root}, found {len(paths)}.")
    arrays = [np.asarray(np.load(path), dtype=np.float32).squeeze() for path in paths]
    finite_items = [item[np.isfinite(item)] for item in arrays if np.isfinite(item).any()]
    finite = np.concatenate(finite_items) if finite_items else np.asarray([], dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
    scale = max(float(hi - lo), 1e-6)
    staging = output.parent / f"{output.stem}_maps"
    _prepare_staging(staging)
    for index, array in enumerate(arrays):
        normalized = np.clip((np.nan_to_num(array, nan=float(hi)) - lo) / scale, 0.0, 1.0)
        image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L").resize(size, _RESAMPLING.BILINEAR)
        image.save(staging / f"{index:06d}.png")
    _run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(staging / "%06d.png"), "-pix_fmt", "yuv420p", str(output),
    ])


def _encode_mask(mask_root: Path, output: Path, frame_count: int, size: tuple[int, int], fps: int) -> None:
    ffmpeg = _ffmpeg()
    paths = sorted(path for path in mask_root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) != frame_count:
        raise ValueError(f"Expected {frame_count} masks in {mask_root}, found {len(paths)}.")
    staging = output.parent / f"{output.stem}_maps"
    _prepare_staging(staging)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            foreground = np.asarray(image.convert("L").resize(size, _RESAMPLING.NEAREST)) > 0
            # Transfer applies its depth control in white mask regions. Keep the
            # robot/task foreground constrained and leave the background free
            # to follow the scene prompt.
            control_mask = np.where(foreground, 255, 0).astype(np.uint8)
            Image.fromarray(control_mask, mode="L").save(staging / f"{index:06d}.png")
    _run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(staging / "%06d.png"), "-pix_fmt", "yuv420p", str(output),
    ])


def _extract_frames(video: Path, output_dir: Path, expected: int) -> list[str]:
    ffmpeg = _ffmpeg()
    _prepare_staging(output_dir)
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), str(output_dir / "%06d.png")])
    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != expected:
        raise RuntimeError(f"Cosmos Transfer returned {len(frames)} frames; expected {expected} synchronized frames.")
    return [str(path.resolve()) for path in frames]


def _controls(record: dict[str, Any], base: Path) -> tuple[Path, Path]:
    controls = record.get("control_artifacts") or {}
    if not isinstance(controls, dict):
        raise ValueError("control_artifacts must be an object.")
    depth_value = controls.get("depth_root")
    mask_value = controls.get("mask_root")
    if not isinstance(depth_value, str) or not depth_value or not isinstance(mask_value, str) or not mask_value:
        raise FileNotFoundError("Scene transfer requires non-empty depth_root and mask_root paths.")
    depth_root = _resolved(depth_value, base)
    mask_root = _resolved(mask_value, base)
    if not depth_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Scene transfer requires synchronized depth_root and mask_root directories.")
    return depth_root, mask_root


def _absolute_record_paths(record: dict[str, Any], base: Path) -> dict[str, Any]:
    """Keep copied sidecars valid when output JSONL moves to another directory."""

    updated = dict(record)
    for key in _RECORD_PATH_FIELDS:
        value = updated.get(key)
        if isinstance(value, str) and value:
            updated[key] = str(_resolved(value, base))
    controls = updated.get("control_artifacts")
    if isinstance(controls, dict):
        resolved_controls = dict(controls)
        for key in _CONTROL_PATH_FIELDS:
            value = resolved_controls.get(key)
            if isinstance(value, str) and value:
                resolved_controls[key] = str(_resolved(value, base))
        updated["control_artifacts"] = resolved_controls
    return updated


def run_worker(
    request_path: Path,
    result_path: Path,
    transfer_repo: Path,
    variants_path: Path,
    python: str,
    fps: int,
    checkpoint_path: str | None,
    num_gpus: int,
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_records = Path(request["input_records"]).resolve()
    output_records = Path(request["output_records"]).resolve()
    base = input_records.parent
    variants = _load_variants(variants_path)
    if not (transfer_repo / "examples" / "inference.py").is_file():
        raise FileNotFoundError(f"Cosmos Transfer2.5 source not found: {transfer_repo}")

    output_root = output_records.parent / "scene_variants"
    inference_output = output_root / "inference"
    prepared: list[dict[str, Any]] = []
    seen_trajectories: set[str] = set()
    for record_index, record in enumerate(_read_jsonl(input_records)):
        traj_id = record.get("traj_id")
        if not isinstance(traj_id, str) or not traj_id:
            raise ValueError(f"Record {record_index} requires a non-empty traj_id.")
        if traj_id in seen_trajectories:
            raise ValueError(f"Duplicate input traj_id: {traj_id}")
        seen_trajectories.add(traj_id)
        frame_values = record.get("frames")
        if not isinstance(frame_values, list) or len(frame_values) < 2:
            raise ValueError(f"Record {record_index} requires at least two trajectory frames.")
        source_frames = [_resolved(str(path), base) for path in frame_values]
        depth_root, mask_root = _controls(record, base)
        for variant in variants:
            variant_id = str(variant["variant_id"])
            seed = int(variant.get("seed", record_index))
            sample_name = f"record_{record_index:06d}_{variant_id}"
            work = output_root / sample_name
            work.mkdir(parents=True, exist_ok=True)
            input_video, depth_video, mask_video = work / "input.mp4", work / "depth.mp4", work / "mask.mp4"
            size = _encode_frames(source_frames, input_video, fps)
            _encode_depth(depth_root, depth_video, len(source_frames), size, fps)
            _encode_mask(mask_root, mask_video, len(source_frames), size, fps)
            spec = {
                "name": sample_name,
                "prompt": str(variant["prompt"]),
                "video_path": str(input_video.resolve()),
                "seed": seed,
                "max_frames": len(source_frames),
                "keep_input_resolution": True,
                "depth": {
                    "control_path": str(depth_video.resolve()),
                    "mask_path": str(mask_video.resolve()),
                    "control_weight": float(variant.get("control_weight", 1.0)),
                },
            }
            spec_path = work / "transfer_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            expected_video = inference_output / f"{sample_name}.mp4"
            expected_video.unlink(missing_ok=True)
            prepared.append(
                {
                    "record": record,
                    "traj_id": traj_id,
                    "variant": variant,
                    "seed": seed,
                    "sample_name": sample_name,
                    "work": work,
                    "input_video": input_video,
                    "spec_path": spec_path,
                    "expected_video": expected_video,
                    "frame_count": len(source_frames),
                    "depth_root": depth_root,
                    "mask_root": mask_root,
                }
            )

    inference_args = [
        "examples/inference.py",
        "-i",
        *[str(item["spec_path"]) for item in prepared],
        "-o",
        str(inference_output),
        "--model",
        "depth",
    ]
    if num_gpus > 1:
        torchrun = str(Path(python).with_name("torchrun"))
        command = [torchrun, "--nproc_per_node", str(num_gpus), *inference_args]
    else:
        command = [python, *inference_args]
    if checkpoint_path:
        command.extend(["--checkpoint-path", checkpoint_path])
    _run(command, cwd=transfer_repo)

    generated: list[dict[str, Any]] = []
    for item in prepared:
        video = item["expected_video"]
        if not video.is_file():
            raise RuntimeError(f"Cosmos Transfer output was not created: {video}")
        record = item["record"]
        traj_id = item["traj_id"]
        variant = item["variant"]
        variant_id = str(variant["variant_id"])
        work = item["work"]
        mask_root = item["mask_root"]
        depth_root = item["depth_root"]
        input_video = item["input_video"]
        seed = item["seed"]
        updated = _absolute_record_paths(record, base)
        updated["traj_id"] = f"{traj_id}/scene-{variant_id}"
        updated["parent_traj_id"] = traj_id
        updated["frames"] = _extract_frames(video, work / "frames", int(item["frame_count"]))
        updated["scene_variant"] = {
            "variant_id": variant_id,
            "prompt": str(variant["prompt"]),
            "segmentation_path": str(mask_root.resolve()),
            "depth_path": str(depth_root.resolve()),
            "preserve_mask_path": str(mask_root.resolve()),
            "model_id": "Cosmos-Transfer2.5-2B/depth",
            "seed": seed,
            "source_video_path": str(input_video.resolve()),
        }
        generated.append(updated)

    output_records.parent.mkdir(parents=True, exist_ok=True)
    output_records.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"records_path": str(output_records)}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene variants with the official Cosmos Transfer2.5 CLI.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--transfer-repo", default=".venv/src/cosmos-transfer2.5")
    parser.add_argument("--variants", default="mi_reward/configs/scene_variants.yaml")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be at least 1")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    python = str(Path(args.python).resolve())
    checkpoint_path = args.checkpoint_path
    if checkpoint_path and "://" not in checkpoint_path:
        checkpoint_path = str(Path(checkpoint_path).resolve())
    run_worker(
        Path(args.request), Path(args.result), Path(args.transfer_repo).resolve(),
        Path(args.variants).resolve(), python, args.fps, checkpoint_path, args.num_gpus,
    )


if __name__ == "__main__":
    main()
