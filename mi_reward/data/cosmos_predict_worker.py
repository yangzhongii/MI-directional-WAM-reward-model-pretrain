"""Cosmos Predict2.5 robot/action-conditioned rollout worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
        raise ValueError(f"No MuJoCo records found in {path}.")
    return records


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted((path / "robot" / "action-cond").glob("*_ema_bf16.pt"))
    if not matches:
        matches = sorted(path.rglob("*_ema_bf16.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Cosmos robot/action-cond EMA checkpoint under {path}, found {len(matches)}."
        )
    return matches[0]


def _frames(record: dict[str, Any], root: Path) -> list[Path]:
    values = record.get("frames")
    if not isinstance(values, list) or not values:
        raise ValueError(f"MuJoCo record {record.get('traj_id', '<unknown>')} has no frames.")
    paths = [_resolve(str(value), root) for value in values]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("MuJoCo record references a missing rendered frame.")
    return paths


def _actions(record: dict[str, Any], root: Path) -> np.ndarray:
    value = record.get("action_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"MuJoCo record {record.get('traj_id', '<unknown>')} has no action_path.")
    path = _resolve(value, root)
    if not path.is_file():
        raise FileNotFoundError(f"Action sidecar is missing: {path}")
    actions = np.asarray(np.load(path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] not in (7, 8):
        raise ValueError(f"Cosmos action conditioning expects [T, 7] or [T, 8], got {actions.shape}.")
    return actions


def _resize(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    height, width = resolution
    return np.asarray(Image.fromarray(image).convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def _synchronized_frames(video: np.ndarray, count: int) -> np.ndarray:
    if video.ndim != 4 or video.shape[-1] not in (3, 4):
        raise ValueError(f"Cosmos returned unsupported video shape {video.shape}.")
    if len(video) < 2:
        raise ValueError("Cosmos returned fewer than two frames.")
    indices = np.linspace(0, len(video) - 1, num=count).round().astype(int)
    return np.asarray(video[indices, ..., :3], dtype=np.uint8)


def _save_frames(frames: np.ndarray, root: Path) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, frame in enumerate(frames):
        path = root / f"{index:06d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path.resolve()))
    return paths


def _save_video(frames: np.ndarray, path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio and imageio-ffmpeg are required in .venv.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def _generate(
    model: Any,
    initial_frame: np.ndarray,
    actions: np.ndarray,
    *,
    chunk_size: int,
    guidance: int,
    seed: int,
) -> np.ndarray:
    current = initial_frame
    chunks: list[np.ndarray] = []
    for start in range(0, len(actions), chunk_size):
        chunk = actions[start : start + chunk_size]
        if len(chunk) < chunk_size:
            padding = np.zeros((chunk_size - len(chunk), actions.shape[1]), dtype=actions.dtype)
            chunk = np.concatenate([chunk, padding], axis=0)
        current, video = model.step_inference(current, action=chunk, guidance=guidance, seed=seed + start)
        chunks.append(np.asarray(video))
    if not chunks:
        raise ValueError("No action chunks were generated.")
    return np.concatenate([chunks[0], *[item[1:] for item in chunks[1:]]], axis=0)


def run_worker(
    request_path: Path,
    result_path: Path,
    cosmos_repo: Path,
    checkpoint_root: Path,
    *,
    experiment: str,
    resolution: tuple[int, int],
    chunk_size: int,
    guidance: int,
    num_candidates: int,
    context_parallel_size: int,
    fps: int,
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    if not (cosmos_repo / "cosmos_predict2").is_dir():
        raise FileNotFoundError(f"Pinned Cosmos Predict2.5 source is missing: {cosmos_repo}")
    checkpoint = _checkpoint(checkpoint_root)
    sys.path.insert(0, str(cosmos_repo))
    try:
        from cosmos_predict2._src.predict2.action.inference.inference_pipeline import ActionVideo2WorldInference
    except ImportError as exc:
        raise RuntimeError("Cosmos Predict2.5 is not installed in .venv. Run requirements/install.sh --all.") from exc
    model = ActionVideo2WorldInference(
        experiment_name=experiment,
        ckpt_path=str(checkpoint),
        s3_credential_path="",
        context_parallel_size=context_parallel_size,
        distilled=False,
    )
    rank = int(os.environ.get("RANK", "0"))
    generated: list[dict[str, Any]] = []
    try:
        for record_index, record in enumerate(_read_jsonl(input_path)):
            physical_frames = _frames(record, input_path.parent)
            actions = _actions(record, input_path.parent)
            with Image.open(physical_frames[0]) as image:
                initial = _resize(np.asarray(image.convert("RGB")), resolution)
            physical_id = str(record["traj_id"])
            for candidate_index in range(num_candidates):
                seed = int(record.get("generation_seed", 0)) + candidate_index
                video = _generate(
                    model,
                    initial,
                    actions,
                    chunk_size=chunk_size,
                    guidance=guidance,
                    seed=seed,
                )
                video = _synchronized_frames(video, len(physical_frames))
                root = output_path.parent / "predict" / f"record_{record_index:06d}" / f"candidate_{candidate_index:03d}"
                video_path = root / "rollout.mp4"
                if rank == 0:
                    frame_paths = _save_frames(video, root / "frames")
                    _save_video(video, video_path, fps)
                else:
                    frame_paths = [str((root / "frames" / f"{index:06d}.png").resolve()) for index in range(len(video))]
                updated = dict(record)
                updated.update(
                    {
                        "traj_id": f"{physical_id}/predict-{candidate_index:03d}",
                        "parent_traj_id": physical_id,
                        "frames": frame_paths,
                        "generation_seed": seed,
                        "generator": "cosmos_predict2_5_robot_action_cond",
                        "model_id": "Cosmos-Predict2.5-2B/robot/action-cond",
                        "context_video_path": str(video_path.resolve()),
                    }
                )
                generated.append(updated)
    finally:
        model.cleanup()
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"records_path": str(output_path)}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate action-conditioned visual rollouts with Cosmos Predict2.5.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--cosmos-repo", default=".venv/src/cosmos-predict2.5")
    parser.add_argument("--checkpoint-root", default=".venv/models/cosmos-predict2.5")
    parser.add_argument("--experiment", default="ac_reason_embeddings_rectified_flow_2b_256_320")
    parser.add_argument("--resolution", default="256,320", help="Height,width expected by the action-conditioned model.")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    try:
        resolution = tuple(int(value) for value in args.resolution.split(","))
    except ValueError as exc:
        raise SystemExit("--resolution must be HEIGHT,WIDTH") from exc
    if len(resolution) != 2 or min(resolution) < 1:
        parser.error("--resolution must be HEIGHT,WIDTH with positive values.")
    if min(args.chunk_size, args.num_candidates, args.context_parallel_size, args.fps) < 1:
        parser.error("Chunk size, candidate count, context parallel size, and fps must be positive.")
    run_worker(
        Path(args.request),
        Path(args.result),
        Path(args.cosmos_repo).resolve(),
        Path(args.checkpoint_root).resolve(),
        experiment=args.experiment,
        resolution=(resolution[0], resolution[1]),
        chunk_size=args.chunk_size,
        guidance=args.guidance,
        num_candidates=args.num_candidates,
        context_parallel_size=args.context_parallel_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
