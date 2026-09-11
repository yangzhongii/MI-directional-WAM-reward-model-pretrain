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
from tqdm.auto import tqdm


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
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


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


def _frame_paths(
    record: dict[str, Any],
    base: Path,
    field: str,
    *,
    fallback: str | None = None,
) -> list[Path]:
    values = record.get(field)
    if values is None and fallback is not None:
        values = record.get(fallback)
    if not isinstance(values, list) or len(values) < 2:
        return []
    paths = [_resolved(str(value), base) for value in values]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Missing {field} frame for {record.get('traj_id')}: {paths[:2]}")
    return paths


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
    paths = sorted(path for path in depth_root.iterdir() if path.suffix.lower() in {".npy", ".npz"})
    if len(paths) != frame_count:
        raise ValueError(f"Expected {frame_count} depth maps in {depth_root}, found {len(paths)}.")
    arrays: list[np.ndarray] = []
    for path in paths:
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if not loaded.files:
                    raise ValueError(f"Depth archive is empty: {path}")
                value = loaded[loaded.files[0]]
            finally:
                loaded.close()
        else:
            value = loaded
        arrays.append(np.asarray(value, dtype=np.float32).squeeze())
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
    records = _read_jsonl(input_records)
    if not (transfer_repo / "examples" / "inference.py").is_file():
        raise FileNotFoundError(f"Cosmos Transfer2.5 source not found: {transfer_repo}")

    output_root = output_records.parent / "scene_variants"
    inference_output = output_root / "inference"
    prepared: list[dict[str, Any]] = []
    seen_trajectories: set[str] = set()
    prepare_progress = tqdm(
        total=len(records) * len(variants),
        desc="Cosmos Transfer controls",
        unit="candidate",
    )
    for record_index, record in enumerate(records):
        traj_id = record.get("traj_id")
        if not isinstance(traj_id, str) or not traj_id:
            raise ValueError(f"Record {record_index} requires a non-empty traj_id.")
        if traj_id in seen_trajectories:
            raise ValueError(f"Duplicate input traj_id: {traj_id}")
        seen_trajectories.add(traj_id)
        third_frames = _frame_paths(record, base, "third_view_frames", fallback="frames")
        wrist_frames = _frame_paths(record, base, "wrist_view_frames")
        if not third_frames:
            raise ValueError(f"Record {record_index} requires at least two trajectory frames.")
        dual_view = bool(wrist_frames)
        if dual_view and len(third_frames) != len(wrist_frames):
            raise ValueError(
                f"Dual-view Transfer requires synchronized streams for {traj_id}: "
                f"{len(third_frames)} != {len(wrist_frames)}."
            )
        depth_root: Path | None = None
        mask_root: Path | None = None
        if not dual_view:
            # Backward-compatible path for the older generalization pipeline.
            depth_root, mask_root = _controls(record, base)
        for variant in variants:
            variant_id = str(variant["variant_id"])
            seed = int(variant.get("seed", record_index))
            sample_name = f"record_{record_index:06d}_{variant_id}"
            work = output_root / sample_name
            work.mkdir(parents=True, exist_ok=True)
            spec_paths: list[Path] = []
            expected_videos: dict[str, Path] = {}
            input_videos: dict[str, Path] = {}
            if dual_view:
                for view_name, view_frames in (("third_view", third_frames), ("wrist_view", wrist_frames)):
                    view_sample = f"{sample_name}__{view_name}"
                    input_video = work / f"{view_name}_input.mp4"
                    _encode_frames(view_frames, input_video, fps)
                    spec = {
                        "name": view_sample,
                        "prompt": str(variant["prompt"]),
                        "video_path": str(input_video.resolve()),
                        "seed": seed,
                        "max_frames": len(view_frames),
                        "keep_input_resolution": True,
                        # Public Transfer2.5 can derive Canny control directly
                        # from RGB. This preserves task geometry without the
                        # private robot-multiview checkpoint or extra depth/SAM
                        # artifacts that LIBERO demonstrations do not provide.
                        "edge": {
                            "control_weight": float(variant.get("control_weight", 1.0)),
                        },
                    }
                    spec_path = work / f"{view_name}_transfer_spec.json"
                    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                    expected_video = inference_output / f"{view_sample}.mp4"
                    expected_video.unlink(missing_ok=True)
                    spec_paths.append(spec_path)
                    expected_videos[view_name] = expected_video
                    input_videos[view_name] = input_video
            else:
                assert depth_root is not None and mask_root is not None
                input_video, depth_video, mask_video = work / "input.mp4", work / "depth.mp4", work / "mask.mp4"
                size = _encode_frames(third_frames, input_video, fps)
                _encode_depth(depth_root, depth_video, len(third_frames), size, fps)
                _encode_mask(mask_root, mask_video, len(third_frames), size, fps)
                spec = {
                    "name": sample_name,
                    "prompt": str(variant["prompt"]),
                    "video_path": str(input_video.resolve()),
                    "seed": seed,
                    "max_frames": len(third_frames),
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
                spec_paths.append(spec_path)
                expected_videos["third_view"] = expected_video
                input_videos["third_view"] = input_video
            prepared.append(
                {
                    "record": record,
                    "traj_id": traj_id,
                    "variant": variant,
                    "seed": seed,
                    "sample_name": sample_name,
                    "work": work,
                    "dual_view": dual_view,
                    "input_videos": input_videos,
                    "spec_paths": spec_paths,
                    "expected_videos": expected_videos,
                    "frame_count": len(third_frames),
                    "depth_root": depth_root,
                    "mask_root": mask_root,
                }
            )
            prepare_progress.update(1)
    prepare_progress.close()

    spec_paths = [str(path) for item in prepared for path in item["spec_paths"]]
    inference_args = ["examples/inference.py", "-i", *spec_paths, "-o", str(inference_output)]
    if not any(bool(item["dual_view"]) for item in prepared):
        # Preserve the previous depth-controlled generalization path. The new
        # LIBERO dual-view branch uses public on-the-fly edge control instead.
        inference_args.extend(["--model", "depth"])
    if num_gpus > 1:
        torchrun = str(Path(python).with_name("torchrun"))
        command = [torchrun, "--nproc_per_node", str(num_gpus), *inference_args]
    else:
        command = [python, *inference_args]
    if checkpoint_path:
        command.extend(["--checkpoint-path", checkpoint_path])
    print(f"[transfer] Starting inference for {len(spec_paths)} view samples", flush=True)
    _run(command, cwd=transfer_repo)

    generated: list[dict[str, Any]] = []
    for item in tqdm(prepared, desc="Cosmos Transfer export", unit="candidate"):
        expected_videos = dict(item["expected_videos"])
        missing_outputs = [str(video) for video in expected_videos.values() if not video.is_file()]
        if missing_outputs:
            raise RuntimeError(f"Cosmos Transfer output was not created: {missing_outputs[0]}")
        record = item["record"]
        traj_id = item["traj_id"]
        variant = item["variant"]
        variant_id = str(variant["variant_id"])
        work = item["work"]
        mask_root = item["mask_root"]
        depth_root = item["depth_root"]
        input_videos = dict(item["input_videos"])
        seed = item["seed"]
        updated = _absolute_record_paths(record, base)
        updated["traj_id"] = f"{traj_id}/scene-{variant_id}"
        updated["parent_traj_id"] = traj_id
        if item["dual_view"]:
            third_frames = _extract_frames(
                expected_videos["third_view"], work / "third_view_frames", int(item["frame_count"])
            )
            wrist_frames = _extract_frames(
                expected_videos["wrist_view"], work / "wrist_view_frames", int(item["frame_count"])
            )
            if len(third_frames) != len(wrist_frames):
                raise RuntimeError(
                    f"Transferred dual views are not synchronized for {traj_id}/scene-{variant_id}."
                )
            updated["source"] = "cosmos_transfer_scene"
            updated["frames"] = third_frames
            updated["third_view_frames"] = third_frames
            updated["wrist_view_frames"] = wrist_frames
            metadata = dict(updated.get("metadata") or {})
            parent_verification = updated.get("verification") or {}
            parent_checks = (
                parent_verification.get("checks", {})
                if isinstance(parent_verification, dict)
                else {}
            )
            parent_checks = parent_checks if isinstance(parent_checks, dict) else {}
            authoritative_outcome = bool(
                parent_checks.get("authoritative_outcome", updated.get("task_outcome") is not None)
            )
            measured_kinematics = bool(
                parent_checks.get("measured_kinematics", updated.get("robot_state_path"))
            )
            metadata.update(
                {
                    "visual_source": "cosmos_transfer",
                    "scene_variant_id": variant_id,
                    "scene_transfer_mode": "paired_public_edge_control",
                    "scene_transfer_input_videos": {
                        key: str(value.resolve()) for key, value in input_videos.items()
                    },
                }
            )
            updated["metadata"] = metadata
            updated["verification"] = {
                "status": "accepted",
                "checks": {
                    "dual_view_sync": True,
                    "shared_scene_variant": True,
                    "label_preserving_visual_only": True,
                    "authoritative_outcome": authoritative_outcome,
                    "measured_kinematics": measured_kinematics,
                },
                "reasons": [],
            }
            updated["scene_variant"] = {
                "variant_id": variant_id,
                "prompt": str(variant["prompt"]),
                "model_id": "Cosmos-Transfer2.5-2B/edge",
                "seed": seed,
                "source_video_path": str(input_videos["third_view"].resolve()),
            }
        else:
            assert mask_root is not None and depth_root is not None
            video = expected_videos["third_view"]
            input_video = input_videos["third_view"]
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
    result_path.write_text(
        json.dumps(
            {
                "records_path": str(output_records),
                "records": len(generated),
                "dual_view_records": sum(bool(item["dual_view"]) for item in prepared),
                "view_samples": len(spec_paths),
            }
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene variants with the official Cosmos Transfer2.5 CLI.")
    parser.add_argument("--request")
    parser.add_argument("--result")
    parser.add_argument("--input-records")
    parser.add_argument("--output-records")
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
    if args.request:
        if args.input_records or args.output_records:
            parser.error("Use either --request or --input-records/--output-records, not both.")
        if not args.result:
            parser.error("--result is required with --request.")
        request_path = Path(args.request)
        result_path = Path(args.result)
    else:
        if not args.input_records or not args.output_records:
            parser.error("--input-records and --output-records are required without --request.")
        output_records = Path(args.output_records).resolve()
        request_path = output_records.with_suffix(".request.json")
        result_path = Path(args.result).resolve() if args.result else output_records.with_suffix(".report.json")
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(
                {
                    "input_records": str(Path(args.input_records).resolve()),
                    "output_records": str(output_records),
                }
            ),
            encoding="utf-8",
        )
    # Keep the virtual-environment interpreter path intact. ``resolve()``
    # follows ``bin/python`` symlinks to /usr/bin/python*, which bypasses the
    # venv site-packages in subprocesses.
    python = str(Path(args.python).expanduser().absolute())
    checkpoint_path = args.checkpoint_path
    if checkpoint_path and "://" not in checkpoint_path:
        checkpoint_path = str(Path(checkpoint_path).resolve())
    run_worker(
        request_path, result_path, Path(args.transfer_repo).resolve(),
        Path(args.variants).resolve(), python, args.fps, checkpoint_path, args.num_gpus,
    )


if __name__ == "__main__":
    main()
