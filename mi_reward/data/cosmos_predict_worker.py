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
from tqdm.auto import tqdm

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


def _actions(record: dict[str, Any], root: Path, *, frame_count: int) -> np.ndarray:
    value = record.get("action_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"MuJoCo record {record.get('traj_id', '<unknown>')} has no action_path.")
    path = _resolve(value, root)
    if not path.is_file():
        raise FileNotFoundError(f"Action sidecar is missing: {path}")
    actions = np.asarray(np.load(path), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] not in (7, 8):
        raise ValueError(f"Cosmos action conditioning expects [T-1, 7] or [T-1, 8], got {actions.shape}.")
    if actions.shape[0] != frame_count - 1:
        raise ValueError(
            "Cosmos action/frame alignment requires exactly T-1 transition actions; "
            f"got {actions.shape[0]} actions for {frame_count} physical frames."
        )
    return actions


def _resize(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    height, width = resolution
    return np.asarray(Image.fromarray(image).convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def _reference_image_metrics(generated: np.ndarray, references: np.ndarray) -> dict[str, float | None]:
    error = generated.astype(np.float64) - references.astype(np.float64)
    mse = np.mean(np.square(error), axis=(1, 2, 3))
    psnr = np.full_like(mse, np.inf, dtype=np.float64)
    nonzero = mse > 0
    psnr[nonzero] = 20.0 * np.log10(255.0) - 10.0 * np.log10(mse[nonzero])
    mean_ssim: float | None = None
    try:
        from skimage.metrics import structural_similarity

        min_side = min(int(generated.shape[1]), int(generated.shape[2]))
        win_size = min(7, min_side if min_side % 2 else min_side - 1)
        if win_size < 3:
            return {
                "mean_psnr_db": float(np.mean(psnr)),
                "final_psnr_db": float(psnr[-1]),
                "mean_ssim": None,
            }
        scores = [
            structural_similarity(
                reference,
                prediction,
                channel_axis=-1,
                data_range=255,
                win_size=win_size,
            )
            for reference, prediction in zip(references.astype(np.uint8), generated.astype(np.uint8))
        ]
        mean_ssim = float(np.mean(scores))
    except ImportError:
        pass
    return {
        "mean_psnr_db": float(np.mean(psnr)),
        "final_psnr_db": float(psnr[-1]),
        "mean_ssim": mean_ssim,
    }


def _visual_consistency(
    video: np.ndarray,
    physical_frames: list[Path],
    resolution: tuple[int, int],
    *,
    max_first_frame_mae: float,
    max_mean_pixel_mae: float,
    max_final_pixel_mae: float,
    max_motion_ratio: float,
    max_endpoint_change_mae: float,
    min_endpoint_change_cosine: float,
    min_motion_support_iou: float,
    motion_pixel_threshold: float,
) -> dict[str, Any]:
    reference_items: list[np.ndarray] = []
    for path in physical_frames:
        with Image.open(path) as image:
            reference_items.append(_resize(np.asarray(image.convert("RGB")), resolution))
    references = np.stack(reference_items, axis=0).astype(np.float32)
    generated = np.asarray(video[..., :3], dtype=np.float32)
    if generated.shape != references.shape:
        raise ValueError(
            f"Visual consistency requires aligned generated/reference frames, got "
            f"{generated.shape} and {references.shape}."
        )
    frame_mae = np.abs(generated - references).mean(axis=(1, 2, 3))
    generated_motion = np.abs(np.diff(generated, axis=0)).mean(axis=(1, 2, 3))
    reference_motion = np.abs(np.diff(references, axis=0)).mean(axis=(1, 2, 3))
    motion_ratio = float(generated_motion.mean() / max(float(reference_motion.mean()), 1e-6))
    reference_delta = references[-1] - references[0]
    generated_delta = generated[-1] - generated[0]
    reference_change = np.abs(reference_delta).mean(axis=2)
    generated_change = np.abs(generated_delta).mean(axis=2)
    reference_support = reference_change >= motion_pixel_threshold
    generated_support = generated_change >= motion_pixel_threshold
    support_union = int(np.logical_or(reference_support, generated_support).sum())
    support_intersection = int(np.logical_and(reference_support, generated_support).sum())
    motion_support_iou = 1.0 if support_union == 0 else support_intersection / support_union
    reference_vector = reference_delta.reshape(-1).astype(np.float64)
    generated_vector = generated_delta.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(reference_vector) * np.linalg.norm(generated_vector))
    if denominator > 1e-12:
        endpoint_change_cosine = float(np.dot(reference_vector, generated_vector) / denominator)
    else:
        endpoint_change_cosine = float(
            np.linalg.norm(reference_vector) <= 1e-12 and np.linalg.norm(generated_vector) <= 1e-12
        )
    metrics = {
        "first_frame_mae": float(frame_mae[0]),
        "mean_pixel_mae": float(frame_mae.mean()),
        "final_pixel_mae": float(frame_mae[-1]),
        "generated_motion_mae": float(generated_motion.mean()),
        "reference_motion_mae": float(reference_motion.mean()),
        "motion_ratio": motion_ratio,
        "endpoint_change_mae": float(np.abs(generated_change - reference_change).mean()),
        "endpoint_change_cosine": endpoint_change_cosine,
        "motion_support_iou": float(motion_support_iou),
        "reference_motion_support": float(reference_support.mean()),
        "generated_motion_support": float(generated_support.mean()),
        "frame_count": int(len(video)),
        **_reference_image_metrics(generated, references),
    }
    thresholds = {
        "max_first_frame_mae": float(max_first_frame_mae),
        "max_mean_pixel_mae": float(max_mean_pixel_mae),
        "max_final_pixel_mae": float(max_final_pixel_mae),
        "max_motion_ratio": float(max_motion_ratio),
        "max_endpoint_change_mae": float(max_endpoint_change_mae),
        "min_endpoint_change_cosine": float(min_endpoint_change_cosine),
        "min_motion_support_iou": float(min_motion_support_iou),
        "motion_pixel_threshold": float(motion_pixel_threshold),
    }
    checks = {
        "first_frame": metrics["first_frame_mae"] <= thresholds["max_first_frame_mae"],
        "mean_appearance": metrics["mean_pixel_mae"] <= thresholds["max_mean_pixel_mae"],
        "final_appearance": metrics["final_pixel_mae"] <= thresholds["max_final_pixel_mae"],
        "motion_magnitude": metrics["motion_ratio"] <= thresholds["max_motion_ratio"],
        "endpoint_change": metrics["endpoint_change_mae"] <= thresholds["max_endpoint_change_mae"],
        "endpoint_direction": metrics["endpoint_change_cosine"] >= thresholds["min_endpoint_change_cosine"],
        "motion_location": metrics["motion_support_iou"] >= thresholds["min_motion_support_iou"],
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "thresholds": thresholds,
        "reference": "mujoco_render",
        "version": "pixel_motion_endpoint_v2",
    }


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
        real_chunk = actions[start : start + chunk_size]
        real_length = len(real_chunk)
        model_chunk = real_chunk
        if real_length < chunk_size:
            padding = np.zeros((chunk_size - real_length, actions.shape[1]), dtype=actions.dtype)
            model_chunk = np.concatenate([real_chunk, padding], axis=0)
        current, video = model.step_inference(
            current,
            action=model_chunk,
            guidance=guidance,
            seed=seed + start,
        )
        video = np.asarray(video)
        if video.ndim != 4 or video.shape[-1] not in (3, 4) or len(video) < real_length + 1:
            raise ValueError(
                f"Cosmos returned {video.shape}; at least {real_length + 1} RGB(A) frames are required."
            )
        # A chunk with N actions represents N+1 states. Keep the first state
        # only for the first chunk, and discard frames generated for zero
        # padding instead of resampling them into the physical timeline.
        synchronized = video[: real_length + 1, ..., :3]
        chunks.append(synchronized if not chunks else synchronized[1:])
    if not chunks:
        raise ValueError("No action chunks were generated.")
    result = np.concatenate(chunks, axis=0)
    expected = len(actions) + 1
    if len(result) != expected:
        raise ValueError(f"Cosmos timeline produced {len(result)} frames for {len(actions)} actions; expected {expected}.")
    return np.asarray(result, dtype=np.uint8)


def run_worker(
    request_path: Path,
    result_path: Path,
    cosmos_repo: Path,
    checkpoint_root: Path,
    *,
    adapted_checkpoint: Path | None,
    tokenizer: Path,
    empty_reason_embedding: Path,
    experiment: str,
    resolution: tuple[int, int],
    chunk_size: int,
    guidance: int,
    num_candidates: int,
    context_parallel_size: int,
    fps: int,
    max_first_frame_mae: float,
    max_mean_pixel_mae: float,
    max_final_pixel_mae: float,
    max_motion_ratio: float,
    max_endpoint_change_mae: float,
    min_endpoint_change_cosine: float,
    min_motion_support_iou: float,
    motion_pixel_threshold: float,
    inconsistent_policy: str,
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    if not (cosmos_repo / "cosmos_predict2").is_dir():
        raise FileNotFoundError(f"Pinned Cosmos Predict2.5 source is missing: {cosmos_repo}")
    checkpoint = adapted_checkpoint if adapted_checkpoint is not None else _checkpoint(checkpoint_root)
    for label, path in (
        ("Cosmos checkpoint", checkpoint),
        ("Wan2.1 tokenizer", tokenizer),
        ("empty Reason1 embedding", empty_reason_embedding),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    sys.path.insert(0, str(cosmos_repo))
    try:
        from world_model.inference.cosmos_action import build_action_inference
    except ImportError as exc:
        raise RuntimeError("Cosmos Predict2.5 is not installed in .venv. Run requirements/install.sh --all.") from exc
    records = _read_jsonl(input_path)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[predict] Loading Cosmos Predict checkpoint: {checkpoint}", flush=True)
    if context_parallel_size != 1:
        raise ValueError("The project LoRA inference wrapper currently supports context_parallel_size=1.")
    model = build_action_inference(
        experiment=experiment,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        empty_reason_embedding=empty_reason_embedding,
        use_lora=adapted_checkpoint is not None,
    )
    if rank == 0:
        print("[predict] Model loaded; starting video generation", flush=True)
    progress = tqdm(
        total=len(records) * num_candidates,
        desc="Cosmos Predict",
        unit="candidate",
        disable=rank != 0,
    )
    generated: list[dict[str, Any]] = []
    rejected = 0
    try:
        for record_index, record in enumerate(records):
            physical_frames = _frames(record, input_path.parent)
            actions = _actions(record, input_path.parent, frame_count=len(physical_frames))
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
                visual_consistency = _visual_consistency(
                    video,
                    physical_frames,
                    resolution,
                    max_first_frame_mae=max_first_frame_mae,
                    max_mean_pixel_mae=max_mean_pixel_mae,
                    max_final_pixel_mae=max_final_pixel_mae,
                    max_motion_ratio=max_motion_ratio,
                    max_endpoint_change_mae=max_endpoint_change_mae,
                    min_endpoint_change_cosine=min_endpoint_change_cosine,
                    min_motion_support_iou=min_motion_support_iou,
                    motion_pixel_threshold=motion_pixel_threshold,
                )
                root = output_path.parent / "predict" / f"record_{record_index:06d}" / f"candidate_{candidate_index:03d}"
                video_path = root / "rollout.mp4"
                use_physical = not visual_consistency["passed"] and inconsistent_policy == "use-physical"
                if rank == 0:
                    frame_paths = _save_frames(video, root / "frames")
                    _save_video(video, video_path, fps)
                    root.mkdir(parents=True, exist_ok=True)
                    action_path = root / "actions.npy"
                    np.save(action_path, actions)
                else:
                    frame_paths = [str((root / "frames" / f"{index:06d}.png").resolve()) for index in range(len(video))]
                    action_path = root / "actions.npy"
                if not visual_consistency["passed"] and inconsistent_policy == "reject":
                    rejected += 1
                    progress.update(1)
                    continue
                updated = dict(record)
                metadata = dict(updated.get("metadata") or {})
                metadata.update(
                    {
                        "visual_source": "mujoco" if use_physical else "cosmos_predict",
                        "cosmos_prediction_path": str(video_path.resolve()),
                        "cosmos_prediction_frames": frame_paths,
                        "cosmos_checkpoint": str(checkpoint.resolve()),
                        "cosmos_adapted": adapted_checkpoint is not None,
                    }
                )
                common = {
                    "traj_id": f"{physical_id}/predict-{candidate_index:03d}",
                    "parent_traj_id": physical_id,
                    "generation_seed": seed,
                    "candidate_index": candidate_index,
                    "action_path": str(action_path.resolve()),
                    "visual_consistency": visual_consistency,
                    "metadata": metadata,
                }
                if use_physical:
                    # The simulator is the physical source of truth. Keep a
                    # failed Cosmos prediction under logs for audit, but never
                    # let hallucinated pixels replace its synchronized rollout.
                    common.update(
                        {
                            "frames": [str(path.resolve()) for path in physical_frames],
                            "generator": "mujoco_physical_completion_cosmos_gated",
                            "model_id": str(record.get("model_id", "MuJoCo")),
                        }
                    )
                else:
                    common.update(
                        {
                            "frames": frame_paths,
                            "generator": (
                                "cosmos_predict2_5_robot_action_cond_lora"
                                if adapted_checkpoint is not None
                                else "cosmos_predict2_5_robot_action_cond"
                            ),
                            "model_id": (
                                "Cosmos-Predict2.5-2B/robot/action-cond+project-lora"
                                if adapted_checkpoint is not None
                                else "Cosmos-Predict2.5-2B/robot/action-cond"
                            ),
                            "context_video_path": str(video_path.resolve()),
                        }
                    )
                updated.update(common)
                generated.append(updated)
                progress.update(1)
    finally:
        progress.close()
        model.cleanup()
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        passed = sum(bool(item["visual_consistency"]["passed"]) for item in generated)
        result_path.write_text(
            json.dumps(
                {
                    "records_path": str(output_path),
                    "attempted": len(records) * num_candidates,
                    "accepted": len(generated),
                    "cosmos_passed": passed,
                    "cosmos_rejected": rejected,
                    "physical_fallbacks": len(generated) - passed,
                    "inconsistent_policy": inconsistent_policy,
                }
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate action-conditioned visual rollouts with Cosmos Predict2.5.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--cosmos-repo", default=".venv/src/cosmos-predict2.5")
    parser.add_argument("--checkpoint-root", default=".venv/models/cosmos-predict2.5")
    parser.add_argument("--adapted-checkpoint", help="Exported model_ema_bf16.pt from world_model LoRA training.")
    parser.add_argument("--tokenizer", default=".venv/models/cosmos-predict2.5/tokenizer.pth")
    parser.add_argument(
        "--empty-reason-embedding",
        default=".venv/models/cosmos-predict2.5/robot/action-cond/cr1_empty_string_text_embeddings.pt",
    )
    parser.add_argument("--experiment", default="ac_reason_embeddings_rectified_flow_2b_256_320")
    parser.add_argument("--resolution", default="256,320", help="Height,width expected by the action-conditioned model.")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--guidance", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-first-frame-mae", type=float, default=12.0)
    parser.add_argument("--max-mean-pixel-mae", type=float, default=45.0)
    parser.add_argument("--max-final-pixel-mae", type=float, default=55.0)
    parser.add_argument("--max-motion-ratio", type=float, default=8.0)
    parser.add_argument("--max-endpoint-change-mae", type=float, default=25.0)
    parser.add_argument("--min-endpoint-change-cosine", type=float, default=0.10)
    parser.add_argument("--min-motion-support-iou", type=float, default=0.20)
    parser.add_argument("--motion-pixel-threshold", type=float, default=12.0)
    parser.add_argument(
        "--inconsistent-policy",
        choices=("reject", "use-physical"),
        default="use-physical",
        help="Reject bad Cosmos pixels or keep the synchronized MuJoCo frames as the canonical rollout.",
    )
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
        adapted_checkpoint=None if args.adapted_checkpoint is None else Path(args.adapted_checkpoint).resolve(),
        tokenizer=Path(args.tokenizer).resolve(),
        empty_reason_embedding=Path(args.empty_reason_embedding).resolve(),
        experiment=args.experiment,
        resolution=(resolution[0], resolution[1]),
        chunk_size=args.chunk_size,
        guidance=args.guidance,
        num_candidates=args.num_candidates,
        context_parallel_size=args.context_parallel_size,
        fps=args.fps,
        max_first_frame_mae=args.max_first_frame_mae,
        max_mean_pixel_mae=args.max_mean_pixel_mae,
        max_final_pixel_mae=args.max_final_pixel_mae,
        max_motion_ratio=args.max_motion_ratio,
        max_endpoint_change_mae=args.max_endpoint_change_mae,
        min_endpoint_change_cosine=args.min_endpoint_change_cosine,
        min_motion_support_iou=args.min_motion_support_iou,
        motion_pixel_threshold=args.motion_pixel_threshold,
        inconsistent_policy=args.inconsistent_policy,
    )


if __name__ == "__main__":
    main()
