"""Compare base and adapted Cosmos action models on held-out LIBERO episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image
from tqdm.auto import tqdm

from world_model.inference.cosmos_action import build_action_inference


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _video_frames(path: Path, frame_ids: list[int], resolution: tuple[int, int]) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(reader.get_data(index), dtype=np.uint8)[..., :3] for index in frame_ids]
    finally:
        reader.close()
    height, width = resolution
    return np.stack([
        np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))
        for frame in frames
    ])


def _metrics(generated: np.ndarray, reference: np.ndarray, motion_threshold: float = 12.0) -> dict[str, float | None]:
    generated_f = generated.astype(np.float64)
    reference_f = reference.astype(np.float64)
    error = generated_f - reference_f
    frame_mse = np.square(error).mean(axis=(1, 2, 3))
    frame_psnr = np.where(
        frame_mse > 0,
        20.0 * np.log10(255.0) - 10.0 * np.log10(np.maximum(frame_mse, 1e-12)),
        np.inf,
    )
    generated_delta = generated_f[-1] - generated_f[0]
    reference_delta = reference_f[-1] - reference_f[0]
    denominator = float(np.linalg.norm(generated_delta) * np.linalg.norm(reference_delta))
    endpoint_cosine = (
        float(np.vdot(generated_delta.reshape(-1), reference_delta.reshape(-1)) / denominator)
        if denominator > 1e-12 else 1.0
    )
    generated_support = np.abs(generated_delta).mean(axis=2) >= motion_threshold
    reference_support = np.abs(reference_delta).mean(axis=2) >= motion_threshold
    union = int(np.logical_or(generated_support, reference_support).sum())
    support_iou = 1.0 if union == 0 else float(np.logical_and(generated_support, reference_support).sum() / union)
    mean_ssim: float | None = None
    try:
        from skimage.metrics import structural_similarity

        min_side = min(int(generated.shape[1]), int(generated.shape[2]))
        win_size = min(7, min_side if min_side % 2 else min_side - 1)
        if win_size >= 3:
            mean_ssim = float(np.mean([
                structural_similarity(
                    ref,
                    pred,
                    channel_axis=-1,
                    data_range=255,
                    win_size=win_size,
                )
                for ref, pred in zip(reference, generated)
            ]))
    except ImportError:
        pass
    return {
        "mean_psnr_db": float(np.mean(frame_psnr)),
        "final_psnr_db": float(frame_psnr[-1]),
        "mean_ssim": mean_ssim,
        "mean_mae": float(np.abs(error).mean()),
        "final_mae": float(np.abs(error[-1]).mean()),
        "endpoint_change_cosine": endpoint_cosine,
        "motion_support_iou": support_iou,
    }


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return None if not values else float(np.mean(values))


def _aggregate(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
    keys = (
        "mean_psnr_db", "final_psnr_db", "mean_ssim", "mean_mae", "final_mae",
        "endpoint_change_cosine", "motion_support_iou",
    )
    return {"episodes": len(items), **{key: _mean(items, key) for key in keys}}


def _quality_gate(base: dict[str, Any], adapted: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "psnr": adapted["mean_psnr_db"] - base["mean_psnr_db"] >= float(thresholds.get("min_psnr_gain_db", 0.0)),
        "final_mae": adapted["final_mae"] <= base["final_mae"] * float(thresholds.get("max_final_mae_ratio", 1.0)),
        "endpoint_direction": adapted["endpoint_change_cosine"] - base["endpoint_change_cosine"]
        >= float(thresholds.get("min_endpoint_cosine_gain", 0.0)),
        "motion_location": adapted["motion_support_iou"] - base["motion_support_iou"]
        >= float(thresholds.get("min_motion_iou_gain", 0.0)),
    }
    if base.get("mean_ssim") is not None and adapted.get("mean_ssim") is not None:
        checks["ssim"] = adapted["mean_ssim"] - base["mean_ssim"] >= float(thresholds.get("min_ssim_gain", 0.0))
    return {"passed": bool(all(checks.values())), "checks": checks, "thresholds": thresholds}


def _samples(dataset_root: Path, *, max_episodes: int, fps_downsample_ratio: int, chunk_size: int,
             resolution: tuple[int, int], cosmos_repo: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(cosmos_repo))
    from cosmos_predict2._src.predict2.action.inference.inference import get_action_sequence_from_states

    annotations = sorted((dataset_root / "annotation" / "test").glob("*.json"))[:max_episodes]
    samples: list[dict[str, Any]] = []
    for annotation_path in annotations:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        actions = np.asarray(get_action_sequence_from_states(
            annotation,
            fps_downsample_ratio=fps_downsample_ratio,
            gripper_scale=1.0,
        ), dtype=np.float32)
        if len(actions) < chunk_size:
            continue
        actions = actions[:chunk_size]
        frame_ids = [index * fps_downsample_ratio for index in range(chunk_size + 1)]
        main_video = dataset_root / str(annotation["videos"][0]["video_path"])
        reference = _video_frames(main_video, frame_ids, resolution)
        samples.append({
            "episode_id": str(annotation.get("episode_id", annotation_path.stem)),
            "actions": actions,
            "reference": reference,
            "annotation": str(annotation_path),
        })
    if not samples:
        raise ValueError("No held-out episode has enough frames for one action chunk.")
    return samples


def preflight(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    model = dict(config["model"])
    dataset = dict(config["dataset"])
    required = {
        "base_checkpoint": _resolve(root, model["base_checkpoint"]),
        "adapted_checkpoint": _resolve(root, model["adapted_checkpoint"]),
        "tokenizer": _resolve(root, model["tokenizer"]),
        "empty_reason_embedding": _resolve(root, model["empty_reason_embedding"]),
        "cosmos_repo": _resolve(root, model["source"]),
        "dataset": _resolve(root, dataset["root"]),
    }
    errors = [f"{name} is missing: {path}" for name, path in required.items() if not path.exists()]
    if required["dataset"].exists() and not list((required["dataset"] / "annotation" / "test").glob("*.json")):
        errors.append(f"test annotations are missing below {required['dataset']}")
    report = {"status": "ready" if not errors else "blocked", "paths": {k: str(v) for k, v in required.items()}, "errors": errors}
    if errors:
        raise RuntimeError("Cosmos action validation preflight failed:\n- " + "\n- ".join(errors))
    return report


def run(config_path: Path, root: Path, *, max_episodes_override: int | None = None) -> dict[str, Any]:
    preflight_report = preflight(config_path, root)
    config = _load_config(config_path)
    model_config = dict(config["model"])
    dataset_config = dict(config["dataset"])
    evaluation = dict(config["evaluation"])
    paths = {key: Path(value) for key, value in preflight_report["paths"].items()}
    output = _resolve(root, evaluation["output"])
    output.mkdir(parents=True, exist_ok=True)
    resolution = tuple(int(value) for value in dataset_config.get("resolution", [256, 320]))
    samples = _samples(
        paths["dataset"],
        max_episodes=int(max_episodes_override or evaluation.get("max_episodes", 8)),
        fps_downsample_ratio=int(dataset_config.get("fps_downsample_ratio", 5)),
        chunk_size=int(dataset_config.get("num_action_per_chunk", 12)),
        resolution=(resolution[0], resolution[1]),
        cosmos_repo=paths["cosmos_repo"],
    )
    predictions: dict[str, dict[str, np.ndarray]] = {sample["episode_id"]: {} for sample in samples}
    per_mode: dict[str, list[dict[str, Any]]] = {}
    for mode, use_lora, checkpoint in (
        ("base", False, paths["base_checkpoint"]),
        ("adapted", True, paths["adapted_checkpoint"]),
    ):
        inference = build_action_inference(
            experiment=str(model_config.get("experiment", "ac_reason_embeddings_rectified_flow_2b_256_320")),
            checkpoint=checkpoint,
            tokenizer=paths["tokenizer"],
            empty_reason_embedding=paths["empty_reason_embedding"],
            use_lora=use_lora,
        )
        mode_metrics: list[dict[str, Any]] = []
        try:
            for index, sample in enumerate(tqdm(samples, desc=f"Cosmos held-out {mode}", unit="episode")):
                _, generated = inference.step_inference(
                    sample["reference"][0],
                    action=sample["actions"],
                    guidance=int(evaluation.get("guidance", 7)),
                    seed=int(evaluation.get("seed", 0)) + index,
                )
                generated = np.asarray(generated[..., :3], dtype=np.uint8)
                if generated.shape != sample["reference"].shape:
                    raise ValueError(f"{mode} returned {generated.shape}, expected {sample['reference'].shape}")
                predictions[sample["episode_id"]][mode] = generated
                metrics = {"episode_id": sample["episode_id"], **_metrics(generated, sample["reference"])}
                mode_metrics.append(metrics)
                mode_dir = output / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(mode_dir / f"{sample['episode_id']}.mp4", generated, fps=4, codec="libx264", quality=8)
        finally:
            inference.cleanup()
            del inference
            import torch

            torch.cuda.empty_cache()
        per_mode[mode] = mode_metrics
    for sample in samples:
        episode_id = sample["episode_id"]
        comparison = np.concatenate(
            (sample["reference"], predictions[episode_id]["base"], predictions[episode_id]["adapted"]), axis=2
        )
        comparison_dir = output / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(comparison_dir / f"{episode_id}.mp4", comparison, fps=4, codec="libx264", quality=8)
    aggregate = {mode: _aggregate(values) for mode, values in per_mode.items()}
    report = {
        "status": "complete",
        "episodes": len(samples),
        "aggregate": aggregate,
        "quality_gate": _quality_gate(aggregate["base"], aggregate["adapted"], dict(evaluation.get("quality_gate", {}))),
        "per_episode": per_mode,
        "comparison_layout": "reference | base | adapted",
        "output": str(output),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--max-episodes", type=int, help="Override the configured episode count for a smoke run.")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    config = _resolve(root, args.config)
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive.")
    report = (
        preflight(config, root)
        if args.action == "preflight"
        else run(config, root, max_episodes_override=args.max_episodes)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
