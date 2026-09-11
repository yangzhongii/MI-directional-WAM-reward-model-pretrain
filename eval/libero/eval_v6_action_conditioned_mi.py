"""Pipeline-v6 Test 2: frozen action-conditioned control latent Dame MI on v5 anchors."""
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
from mi_reward.training.train_v6_control_dynamics import _cache_tokens
from mi_reward.training.train_v6_control_dynamics import _cache_name


def _candidate_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["candidate_id"]: item for item in record["candidates"] if not item["excluded_reasons"]}


def _eef_delta(candidate: dict[str, Any], center: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float64)
        - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float64),
        dtype=torch.float32,
    )


def _eef_probe_matrix(candidates: dict[str, dict[str, Any]], label: str, center: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.stack([
        (_eef_delta(candidates[f"{label}_plus_{dim}"], center) - _eef_delta(candidates[f"{label}_minus_{dim}"], center)) / 2.0
        for dim in range(3)
    ], dim=1).to(dtype=dtype)
    if torch.linalg.matrix_rank(matrix) < 3:
        raise RuntimeError(f"{label} EEF displacement matrix is rank deficient.")
    return matrix


def _physical_descent_direction(candidates: dict[str, dict[str, Any]], center: dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    eef_matrix = _eef_probe_matrix(candidates, "primary", center, dtype)
    distance_delta = torch.tensor([
        (float(candidates[f"primary_plus_{dim}"]["physical_after"]["eef_object_distance"]) - float(candidates[f"primary_minus_{dim}"]["physical_after"]["eef_object_distance"])) / 2.0
        for dim in range(3)
    ], dtype=dtype)
    return -(torch.linalg.pinv(eef_matrix.T) @ distance_delta)


def _metric(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    if len(left) < 2:
        return {"r2": None, "spearman": None, "sign_accuracy": None}
    residual = float(np.sum((left - right) ** 2))
    variance = float(np.sum((right - right.mean()) ** 2))
    r2 = None if variance <= 1e-12 else float(1.0 - residual / variance)
    ranks_left, ranks_right = np.argsort(np.argsort(left)), np.argsort(np.argsort(right))
    spearman = float(np.corrcoef(ranks_left, ranks_right)[0, 1]) if np.std(ranks_left) and np.std(ranks_right) else None
    nonzero = np.abs(right) > 1e-9
    sign = None if not np.any(nonzero) else float(np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero])))
    return {"r2": r2, "spearman": spearman, "sign_accuracy": sign}


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


def _mi(tokens: torch.Tensor, reference: torch.Tensor, normalization, estimator: DameSoftHistogramMI) -> torch.Tensor:
    candidate_norm, _ = normalize_tokens(tokens, normalization)
    reference_norm, _ = normalize_tokens(reference, normalization)
    return estimator(candidate_norm, reference_norm)


def _load_model(path: Path, cfg: dict, input_dim: int, device: torch.device) -> ActionConditionedControlModel:
    model = ActionConditionedControlModel(
        input_dim=input_dim,
        control_dim=int(cfg["representation"]["control_dim"]),
        action_dim=int(cfg["dynamics"]["action_dim"]),
        hidden_dim=int(cfg["dynamics"]["hidden_dim"]),
        action_scale=float(cfg["dynamics"]["action_scale_m"]),
        spatial_heads=int(cfg["dynamics"]["spatial_heads"]),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def _predicted_tokens(model: ActionConditionedControlModel, control: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return model.dynamics(control, action.reshape(1, -1)).squeeze(0)


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    cfg = yaml.safe_load(args.config.read_text())
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    frozen = {int(value) for value in cfg["split"]["frozen_test_demos"]}
    if {int(str(row["source_demo"]).removeprefix("demo_")) for row in rows} != frozen or len(rows) != 20:
        raise RuntimeError("Test 2 requires exactly the frozen v5 five-state, 20-anchor collection.")
    cache_rows = []
    for row in rows:
        for candidate in row["candidates"]:
            if not candidate["excluded_reasons"]:
                cache_rows.append({"current": candidate["images"]["wrist"], "next": candidate["images"]["wrist"], "language": row["language"]})
        cache_rows.append({"current": row["reference"]["images"]["wrist"], "next": row["reference"]["images"]["wrist"], "language": row["language"]})
    _cache_tokens(cache_rows, args.collection_dir, args.feature_cache_dir, args)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, cfg, int(_token(args.feature_cache_dir, cache_rows[0]["current"]).shape[-1]), device)
    estimator = DameSoftHistogramMI(num_bins=8, spline_order=3, normalization="none", channel_mode="channelwise").to(device)
    per_anchor, analytic_fd_cosines, heldout_prediction, heldout_observed = [], [], [], []
    physical_cosines, descent_flags, cross_eps = [], [], []
    for row in rows:
        candidates = _candidate_map(row)
        required = ["center", *[f"{scale}_{sign}_{dim}" for scale in ("primary", "secondary") for sign in ("plus", "minus") for dim in range(3)]]
        missing = [key for key in required if key not in candidates]
        if missing:
            per_anchor.append({"anchor_id": row["anchor_id"], "status": "INVALID_PROTOCOL", "missing": missing})
            continue
        with torch.no_grad():
            control = model.project(_token(args.feature_cache_dir, candidates["center"]["images"]["wrist"]).unsqueeze(0).to(device))
            reference = model.project(_token(args.feature_cache_dir, row["reference"]["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
        center = control.squeeze(0)
        normalization = fit_fixed_normalization(reference, [center], fit_anchor_ids=(row["anchor_id"],))
        zero = torch.zeros(3, device=device, requires_grad=True)
        mi_center = _mi(_predicted_tokens(model, control, zero), reference, normalization, estimator)
        analytic = torch.autograd.grad(mi_center, zero)[0].detach()
        center_candidate = candidates["center"]
        def fd_gradient(label: str) -> torch.Tensor:
            eef_matrix = _eef_probe_matrix(candidates, label, center_candidate, dtype=torch.float32).to(device)
            values = []
            for dim in range(3):
                for sign in ("plus", "minus"):
                    action = _eef_delta(candidates[f"{label}_{sign}_{dim}"], center_candidate).to(device)
                    with torch.no_grad():
                        values.append(_mi(_predicted_tokens(model, control, action), reference, normalization, estimator))
            plus = torch.stack(values[::2])
            minus = torch.stack(values[1::2])
            return torch.linalg.pinv(eef_matrix.T) @ ((plus - minus) / 2.0)
        primary_fd = fd_gradient("primary")
        secondary_fd = fd_gradient("secondary")
        analytic_fd_cosine = float(F.cosine_similarity(analytic[None], primary_fd[None]).item())
        cross_epsilon = float(F.cosine_similarity(primary_fd[None], secondary_fd[None]).item())
        physical = _physical_descent_direction(candidates, center_candidate, analytic.dtype).to(device)
        physical_cosine = float(F.cosine_similarity(analytic[None], physical[None]).item())
        predictions, observed = [], []
        for key, candidate in candidates.items():
            if not key.startswith("heldout_"):
                continue
            action = _eef_delta(candidate, center_candidate).to(device)
            prediction = float(torch.dot(analytic, action))
            with torch.no_grad():
                observed_tokens = model.project(_token(args.feature_cache_dir, candidate["images"]["wrist"]).unsqueeze(0).to(device)).squeeze(0)
                observed_mi = _mi(observed_tokens, reference, normalization, estimator)
            predictions.append(prediction)
            observed.append(float(observed_mi - mi_center.detach()))
        heldout_prediction.extend(predictions)
        heldout_observed.extend(observed)
        analytic_fd_cosines.append(analytic_fd_cosine)
        cross_eps.append(cross_epsilon)
        physical_cosines.append(physical_cosine)
        descent_flags.append(physical_cosine > 0.0)
        per_anchor.append({
            "anchor_id": row["anchor_id"], "source_demo": row["source_demo"], "status": "VALID",
            "action_coordinate": "actual_eef_delta_xyz", "mi_center": float(mi_center.detach()),
            "analytic_gradient": analytic.tolist(), "finite_difference_gradient": primary_fd.tolist(),
            "mi_analytic_fd_cosine": analytic_fd_cosine, "cross_epsilon_fd_cosine": cross_epsilon,
            "physical_descent_direction": physical.tolist(), "mi_physical_direction_cosine": physical_cosine,
            "mi_ascent_reduces_eef_object_distance": physical_cosine > 0.0,
            "heldout_count": len(predictions), "heldout": _metric(np.asarray(predictions), np.asarray(observed)),
        })
        print(f"ANCHOR {row['anchor_id']} analytic_fd={analytic_fd_cosine:.6f} physical={physical_cosine:.6f}", flush=True)
    heldout = _metric(np.asarray(heldout_prediction), np.asarray(heldout_observed))
    derivative_pass = bool(analytic_fd_cosines) and float(np.median(analytic_fd_cosines)) >= 0.90
    heldout_pass = bool(heldout_prediction) and heldout["r2"] is not None and float(heldout["r2"]) >= 0.50 and heldout["spearman"] is not None and float(heldout["spearman"]) >= 0.60 and heldout["sign_accuracy"] is not None and float(heldout["sign_accuracy"]) >= 0.80
    physical_pass = bool(physical_cosines) and float(np.median(physical_cosines)) >= 0.70 and float(np.mean(descent_flags)) >= 0.70
    report = {
        "protocol": "v6_action_conditioned_mi_test2_v1", "checkpoint": str(args.checkpoint),
        "benchmark": rows[0]["suite"], "task_id": rows[0]["task_id"], "anchors": per_anchor,
        "mi_analytic_finite_difference": {"cosine_median": float(np.median(analytic_fd_cosines)), "pass": derivative_pass},
        "cross_epsilon": {"finite_difference_cosine_median": float(np.median(cross_eps))},
        "heldout_action_mi": {"count": len(heldout_prediction), **heldout, "pass": heldout_pass},
        "physical_direction": {"cosine_median": float(np.median(physical_cosines)), "distance_descent_fraction": float(np.mean(descent_flags)), "pass": physical_pass},
        "gates": {"derivative_pass": derivative_pass, "heldout_pass": heldout_pass, "physical_direction_pass": physical_pass},
        "status": "PASS" if derivative_pass and heldout_pass and physical_pass else "NO_GO",
        "scope": "Frozen Test-1b control model; Dame MI on predicted next control tokens. No Hessian or planner is evaluated.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gates": report["gates"], "heldout": report["heldout_action_mi"], "physical": report["physical_direction"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
