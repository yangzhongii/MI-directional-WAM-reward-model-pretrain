"""Pipeline-v6 oracle-vs-predicted MI decomposition on frozen v5 anchors.

For each actual held-out LIBERO transition, this compares Dame MI from its
rendered next wrist latent (oracle) with MI from the Test-1b dynamics-predicted
next latent.  Both branches use the identical frozen control projection, goal,
normalization, anchors, and actual EEF displacement coordinate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mi_reward.control.mi_action_field import fit_fixed_normalization, normalize_tokens
from mi_reward.models.action_conditioned_latent import ActionConditionedControlModel
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.training.train_v6_control_dynamics import _cache_name, _cache_tokens


def _candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["candidate_id"]: item for item in row["candidates"] if not item["excluded_reasons"]}


def _eef_delta(candidate: dict[str, Any], center: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float64) - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float64), dtype=torch.float32)


def _eef_matrix(candidates: dict[str, dict[str, Any]], label: str, center: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.stack([(_eef_delta(candidates[f"{label}_plus_{axis}"], center) - _eef_delta(candidates[f"{label}_minus_{axis}"], center)) / 2 for axis in range(3)], dim=1).to(dtype=dtype)
    if torch.linalg.matrix_rank(matrix) < 3:
        raise RuntimeError(f"{label} EEF displacement matrix is rank deficient")
    return matrix


def _physical_descent(candidates: dict[str, dict[str, Any]], center: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    eef_matrix = _eef_matrix(candidates, "primary", center, dtype)
    distance_delta = torch.tensor([(float(candidates[f"primary_plus_{axis}"]["physical_after"]["eef_object_distance"]) - float(candidates[f"primary_minus_{axis}"]["physical_after"]["eef_object_distance"])) / 2 for axis in range(3)], dtype=dtype)
    return -(torch.linalg.pinv(eef_matrix.T) @ distance_delta)


def _metric(scores: list[float], progress: list[float]) -> dict[str, float | None]:
    left, right = np.asarray(scores), np.asarray(progress)
    if len(left) < 2:
        return {"r2": None, "spearman": None, "sign_accuracy": None}
    variance = float(np.sum((right - right.mean()) ** 2))
    r2 = None if variance <= 1e-12 else float(1 - np.sum((left - right) ** 2) / variance)
    rank_left, rank_right = np.argsort(np.argsort(left)), np.argsort(np.argsort(right))
    spearman = float(np.corrcoef(rank_left, rank_right)[0, 1]) if np.std(rank_left) and np.std(rank_right) else None
    nonzero = np.abs(right) > 1e-9
    sign = None if not np.any(nonzero) else float(np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero])))
    return {"r2": r2, "spearman": spearman, "sign_accuracy": sign}


def _ranking_pass(metric: dict[str, float | None]) -> bool:
    return all(metric[key] is not None and float(metric[key]) >= threshold for key, threshold in (("r2", 0.50), ("spearman", 0.60), ("sign_accuracy", 0.80)))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("mi_reward/configs/lawam_control_latent_delta.yaml"))
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _token(cache: Path, relative: str) -> torch.Tensor:
    return torch.load(cache / _cache_name(relative), map_location="cpu", weights_only=True).float()


def _mi(tokens: torch.Tensor, reference: torch.Tensor, normalization: Any, estimator: DameSoftHistogramMI) -> torch.Tensor:
    tokens, _ = normalize_tokens(tokens, normalization)
    reference, _ = normalize_tokens(reference, normalization)
    return estimator(tokens, reference)


def _load_model(path: Path, cfg: dict[str, Any], input_dim: int, device: torch.device) -> ActionConditionedControlModel:
    model = ActionConditionedControlModel(input_dim=input_dim, control_dim=int(cfg["representation"]["control_dim"]), action_dim=int(cfg["dynamics"]["action_dim"]), hidden_dim=int(cfg["dynamics"]["hidden_dim"]), action_scale=float(cfg["dynamics"]["action_scale_m"]), spatial_heads=int(cfg["dynamics"]["spatial_heads"])).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    cfg = yaml.safe_load(args.config.read_text())
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    frozen = {int(value) for value in cfg["split"]["frozen_test_demos"]}
    source_demos = {int(str(row["source_demo"]).removeprefix("demo_")) for row in rows}
    if len(rows) != 20 or source_demos != frozen:
        raise RuntimeError("Requires the frozen v5 collection: five states and 20 anchors.")
    cache_rows = []
    for row in rows:
        cache_rows.extend({"current": item["images"]["wrist"], "next": item["images"]["wrist"], "language": row["language"]} for item in row["candidates"] if not item["excluded_reasons"])
        cache_rows.append({"current": row["reference"]["images"]["wrist"], "next": row["reference"]["images"]["wrist"], "language": row["language"]})
    _cache_tokens(cache_rows, args.collection_dir, args.feature_cache_dir, args)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    input_dim = int(_token(args.feature_cache_dir, cache_rows[0]["current"]).shape[-1])
    model = _load_model(args.checkpoint, cfg, input_dim, device)
    estimator = DameSoftHistogramMI(num_bins=8, spline_order=3, normalization="none", channel_mode="channelwise").to(device)
    per_anchor: list[dict[str, Any]] = []
    oracle_scores: list[float] = []
    predicted_scores: list[float] = []
    physical_progress: list[float] = []
    oracle_cosines: list[float] = []
    predicted_cosines: list[float] = []
    branch_gradient_cosines: list[float] = []
    oracle_descent: list[bool] = []
    predicted_descent: list[bool] = []
    for row in rows:
        candidates = _candidate_map(row)
        required = ["center", *[f"{scale}_{sign}_{axis}" for scale in ("primary", "secondary") for sign in ("plus", "minus") for axis in range(3)]]
        missing = [name for name in required if name not in candidates]
        if missing:
            per_anchor.append({"anchor_id": row["anchor_id"], "status": "INVALID_PROTOCOL", "missing": missing})
            continue
        center_candidate = candidates["center"]
        with torch.no_grad():
            center = model.project(_token(args.feature_cache_dir, center_candidate["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
            reference = model.project(_token(args.feature_cache_dir, row["reference"]["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
        normalization = fit_fixed_normalization(reference, [center], fit_anchor_ids=(row["anchor_id"],))
        with torch.no_grad():
            center_predicted_mi = _mi(model.dynamics(center.unsqueeze(0), torch.zeros((1, 3), device=device)).squeeze(0), reference, normalization, estimator)
            center_oracle_mi = _mi(center, reference, normalization, estimator)

        def finite_difference(branch: str) -> torch.Tensor:
            matrix = _eef_matrix(candidates, "primary", center_candidate, torch.float32).to(device)
            values = []
            for axis in range(3):
                for sign in ("plus", "minus"):
                    candidate = candidates[f"primary_{sign}_{axis}"]
                    with torch.no_grad():
                        if branch == "oracle":
                            latent = model.project(_token(args.feature_cache_dir, candidate["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
                        else:
                            latent = model.dynamics(center.unsqueeze(0), _eef_delta(candidate, center_candidate).to(device).unsqueeze(0)).squeeze(0)
                        values.append(_mi(latent, reference, normalization, estimator))
            return torch.linalg.pinv(matrix.T) @ ((torch.stack(values[::2]) - torch.stack(values[1::2])) / 2)

        oracle_gradient, predicted_gradient = finite_difference("oracle"), finite_difference("predicted")
        physical_direction = _physical_descent(candidates, center_candidate, oracle_gradient.dtype).to(device)
        oracle_cosine = float(F.cosine_similarity(oracle_gradient[None], physical_direction[None]).item())
        predicted_cosine = float(F.cosine_similarity(predicted_gradient[None], physical_direction[None]).item())
        branch_cosine = float(F.cosine_similarity(oracle_gradient[None], predicted_gradient[None]).item())
        per_oracle, per_predicted, per_progress = [], [], []
        for name, candidate in candidates.items():
            if not name.startswith("heldout_"):
                continue
            action = _eef_delta(candidate, center_candidate).to(device)
            with torch.no_grad():
                actual = model.project(_token(args.feature_cache_dir, candidate["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
                predicted = model.dynamics(center.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
                per_oracle.append(float(_mi(actual, reference, normalization, estimator) - center_oracle_mi))
                per_predicted.append(float(_mi(predicted, reference, normalization, estimator) - center_predicted_mi))
            per_progress.append(-(float(candidate["physical_after"]["eef_object_distance"]) - float(center_candidate["physical_after"]["eef_object_distance"])))
        oracle_scores.extend(per_oracle); predicted_scores.extend(per_predicted); physical_progress.extend(per_progress)
        oracle_cosines.append(oracle_cosine); predicted_cosines.append(predicted_cosine); branch_gradient_cosines.append(branch_cosine)
        oracle_descent.append(oracle_cosine > 0); predicted_descent.append(predicted_cosine > 0)
        per_anchor.append({"anchor_id": row["anchor_id"], "source_demo": row["source_demo"], "status": "VALID", "action_coordinate": "actual_eef_delta_xyz", "heldout_count": len(per_oracle), "oracle": {"mi_center": float(center_oracle_mi), "finite_difference_gradient": oracle_gradient.tolist(), "physical_direction_cosine": oracle_cosine, "heldout_vs_physical_progress": _metric(per_oracle, per_progress)}, "predicted": {"mi_center": float(center_predicted_mi), "finite_difference_gradient": predicted_gradient.tolist(), "physical_direction_cosine": predicted_cosine, "heldout_vs_physical_progress": _metric(per_predicted, per_progress)}, "oracle_predicted_fd_gradient_cosine": branch_cosine})
        print(f"ANCHOR {row['anchor_id']} oracle_physical={oracle_cosine:.6f} predicted_physical={predicted_cosine:.6f} oracle_predicted_fd={branch_cosine:.6f}", flush=True)
    oracle_heldout = _metric(oracle_scores, physical_progress)
    predicted_heldout = _metric(predicted_scores, physical_progress)
    fidelity = _metric(predicted_scores, oracle_scores)
    oracle_physical_pass = bool(oracle_cosines) and float(np.median(oracle_cosines)) >= .70 and float(np.mean(oracle_descent)) >= .70
    predicted_physical_pass = bool(predicted_cosines) and float(np.median(predicted_cosines)) >= .70 and float(np.mean(predicted_descent)) >= .70
    oracle_ranking_pass, predicted_ranking_pass = _ranking_pass(oracle_heldout), _ranking_pass(predicted_heldout)
    fidelity_pass = _ranking_pass(fidelity)
    if not oracle_ranking_pass:
        verdict = "ORACLE_MI_RANKING_FAIL"
    elif not oracle_physical_pass:
        verdict = "ORACLE_VISUAL_RANKING_NOT_PHYSICAL"
    elif not fidelity_pass:
        verdict = "PREDICTOR_MI_FIDELITY_FAIL"
    elif not predicted_ranking_pass or not predicted_physical_pass:
        verdict = "PREDICTED_MI_FAIL_AFTER_ORACLE_PASS"
    else:
        verdict = "BOTH_ORACLE_AND_PREDICTED_PASS"
    report = {"protocol": "v6_oracle_vs_predicted_mi_decomposition_v1", "benchmark": rows[0]["suite"], "task_id": rows[0]["task_id"], "checkpoint": str(args.checkpoint), "anchors": per_anchor, "heldout_action_count": len(physical_progress), "oracle_actual_next_latent": {"heldout_vs_physical_progress": oracle_heldout, "ranking_pass": oracle_ranking_pass, "physical_direction": {"cosine_median": float(np.median(oracle_cosines)), "distance_descent_fraction": float(np.mean(oracle_descent)), "pass": oracle_physical_pass}}, "predicted_next_latent": {"heldout_vs_physical_progress": predicted_heldout, "ranking_pass": predicted_ranking_pass, "physical_direction": {"cosine_median": float(np.median(predicted_cosines)), "distance_descent_fraction": float(np.mean(predicted_descent)), "pass": predicted_physical_pass}}, "predicted_vs_oracle_mi": {"heldout_score_fidelity": fidelity, "ranking_pass": fidelity_pass, "finite_difference_gradient_cosine_median": float(np.median(branch_gradient_cosines))}, "verdict": verdict, "scope": "Frozen Test-1b checkpoint and frozen v5 20-anchor/24-held-out protocol. Oracle uses actual rendered next latent; predicted uses F(z_t, actual EEF displacement). No Hessian, planner, or privileged signal enters either MI score."}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "oracle": report["oracle_actual_next_latent"], "predicted": report["predicted_next_latent"], "fidelity": report["predicted_vs_oracle_mi"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
