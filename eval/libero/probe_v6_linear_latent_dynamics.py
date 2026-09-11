"""Small v6 diagnostic: cross-state linear dynamics in frozen LaWAM latents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser.parse_args()


def _image(root: Path, relative: str) -> np.ndarray:
    return np.asarray(Image.open(root / relative).convert("RGB"), dtype=np.uint8)


def _eef_delta(candidate: dict, center: dict) -> np.ndarray:
    return np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float64) - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float64)


def _metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    residual = float(np.sum((predicted - actual) ** 2))
    variance = float(np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - residual / max(variance, 1e-12))
    denom = np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1)
    valid = denom > 1e-12
    cosine = np.sum(predicted[valid] * actual[valid], axis=1) / denom[valid]
    return {"r2": r2, "delta_cosine_median": float(np.median(cosine)), "delta_cosine_mean": float(np.mean(cosine)), "count": int(len(actual))}


def _fit_linear(actions: np.ndarray, deltas: np.ndarray, ridge: float) -> np.ndarray:
    if ridge == 0.0:
        return np.linalg.lstsq(actions, deltas, rcond=None)[0]
    return np.linalg.solve(actions.T @ actions + ridge * np.eye(actions.shape[1]), actions.T @ deltas)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    first_by_demo = {}
    for row in rows:
        first_by_demo.setdefault(row["source_demo"], row)
    selected = [first_by_demo[key] for key in sorted(first_by_demo, key=lambda value: int(value.split("_")[1]))]
    extractor = LaWAMLAMFeatureExtractor(args.lam_config, args.lam_checkpoint, args.dino_model, args.device, strict=True)
    samples = []
    for row in selected:
        candidates = [item for item in row["candidates"] if not item["excluded_reasons"]]
        center = next(item for item in candidates if item["candidate_id"] == "center")
        language = row["language"]
        center_z = extractor.extract_image_tokens(_image(args.collection_dir, center["images"]["wrist"]), language).float().mean(dim=0).numpy()
        for candidate in candidates:
            if candidate["candidate_id"] == "center":
                continue
            latent = extractor.extract_image_tokens(_image(args.collection_dir, candidate["images"]["wrist"]), language).float().mean(dim=0).numpy()
            samples.append(
                {
                    "demo": row["source_demo"],
                    "kind": candidate["kind"],
                    "action": _eef_delta(candidate, center),
                    "delta": latent - center_z,
                }
            )
        print(f"EXTRACTED {row['source_demo']} samples={len(candidates)-1}", flush=True)
    folds = []
    demos = sorted(first_by_demo, key=lambda value: int(value.split("_")[1]))
    for heldout_demo in demos:
        train = [item for item in samples if item["demo"] != heldout_demo and item["kind"] != "heldout"]
        test = [item for item in samples if item["demo"] == heldout_demo and item["kind"] == "heldout"]
        x = np.stack([item["action"] for item in train])
        y = np.stack([item["delta"] for item in train])
        xt = np.stack([item["action"] for item in test])
        yt = np.stack([item["delta"] for item in test])
        weights = _fit_linear(x, y, args.ridge)
        folds.append({"heldout_demo": heldout_demo, **_metrics(xt @ weights, yt)})
    aggregate_pred, aggregate_actual = [], []
    for fold in folds:
        heldout_demo = fold["heldout_demo"]
        train = [item for item in samples if item["demo"] != heldout_demo and item["kind"] != "heldout"]
        test = [item for item in samples if item["demo"] == heldout_demo and item["kind"] == "heldout"]
        x, y = np.stack([i["action"] for i in train]), np.stack([i["delta"] for i in train])
        weights = _fit_linear(x, y, args.ridge)
        aggregate_pred.append(np.stack([i["action"] for i in test]) @ weights)
        aggregate_actual.append(np.stack([i["delta"] for i in test]))
    local_folds, local_pred, local_actual = [], [], []
    for demo in demos:
        train = [item for item in samples if item["demo"] == demo and item["kind"] != "heldout"]
        test = [item for item in samples if item["demo"] == demo and item["kind"] == "heldout"]
        x, y = np.stack([i["action"] for i in train]), np.stack([i["delta"] for i in train])
        xt, yt = np.stack([i["action"] for i in test]), np.stack([i["delta"] for i in test])
        weights = _fit_linear(x, y, args.ridge)
        prediction = xt @ weights
        local_folds.append({"demo": demo, **_metrics(prediction, yt)})
        local_pred.append(prediction)
        local_actual.append(yt)
    report = {
        "protocol": "v6_linear_lawam_dynamics_smoke_v2",
        "representation": "mean_pooled_frozen_lawam",
        "split": "leave_one_initial_state_out; train_axis_cross_test_heldout",
        "ridge": args.ridge,
        "train_samples_per_fold": len([item for item in samples if item["demo"] != demos[0] and item["kind"] != "heldout"]),
        "test_samples_per_fold": len([item for item in samples if item["demo"] == demos[0] and item["kind"] == "heldout"]),
        "state_independent_cross_state": {
            "folds": folds,
            "aggregate": _metrics(np.concatenate(aggregate_pred), np.concatenate(aggregate_actual)),
            "macro_r2": float(np.mean([fold["r2"] for fold in folds])),
            "macro_delta_cosine_median": float(np.mean([fold["delta_cosine_median"] for fold in folds])),
        },
        "state_specific_local_oracle": {
            "folds": local_folds,
            "aggregate": _metrics(np.concatenate(local_pred), np.concatenate(local_actual)),
            "macro_r2": float(np.mean([fold["r2"] for fold in local_folds])),
            "macro_delta_cosine_median": float(np.mean([fold["delta_cosine_median"] for fold in local_folds])),
        },
        "status": "DIAGNOSTIC_ONLY",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
