"""Evaluate Pipeline-v5 MI action geometry from collected local probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mi_reward.control.mi_action_field import (
    estimate_jacobian,
    fit_fixed_normalization,
    mi_and_gradient,
    pullback_gradient,
)
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("wrist", "agentview"), default="wrist")
    parser.add_argument("--representation", choices=("lam", "raw_rgb"), default="lam")
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _token(extractor: LaWAMLAMFeatureExtractor, root: Path, relative: str, language: str) -> torch.Tensor:
    image = np.asarray(Image.open(root / relative).convert("RGB"), dtype=np.uint8)
    return extractor.extract_image_tokens(image, language).float()


def _raw_rgb_tokens(root: Path, relative: str) -> torch.Tensor:
    """Return corresponding RGB pixels as `[H*W,3]` Dame samples."""
    image = np.asarray(Image.open(root / relative).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(image.reshape(-1, 3).copy()).float().div_(255.0)


def _eef_delta(candidate: dict, center: dict) -> torch.Tensor:
    return torch.tensor(
        np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float64)
        - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float64),
        dtype=torch.float32,
    )


def _eef_probe_matrix(candidates: dict[str, dict], label: str, center: dict, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.stack([
        (_eef_delta(candidates[f"{label}_plus_{dim}"], center) - _eef_delta(candidates[f"{label}_minus_{dim}"], center)) / 2.0
        for dim in range(3)
    ], dim=1).to(dtype=dtype)
    if torch.linalg.matrix_rank(matrix) < 3:
        raise RuntimeError(f"{label} EEF displacement matrix is rank deficient: {matrix.tolist()}")
    return matrix


def _physical_descent_direction(candidates: dict[str, dict], center: dict, dtype: torch.dtype) -> torch.Tensor:
    """Return the privileged direction that locally decreases EEF-object distance."""
    eef_matrix = _eef_probe_matrix(candidates, "primary", center, dtype)
    distance_delta = torch.tensor([
        (
            float(candidates[f"primary_plus_{dim}"]["physical_after"]["eef_object_distance"])
            - float(candidates[f"primary_minus_{dim}"]["physical_after"]["eef_object_distance"])
        ) / 2.0
        for dim in range(3)
    ], dtype=dtype)
    return -(torch.linalg.pinv(eef_matrix.T) @ distance_delta)


def _axis_geometry(
    tokens: dict[str, torch.Tensor], candidates: dict[str, dict], label: str, center: dict,
    reference: torch.Tensor, normalization, estimator: DameSoftHistogramMI,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return d(token)/d(actual EEF xyz), pullback gradient, and direct FD gradient."""
    plus = torch.stack([tokens[f"{label}_plus_{dim}"] for dim in range(3)])
    minus = torch.stack([tokens[f"{label}_minus_{dim}"] for dim in range(3)])
    eef_matrix = _eef_probe_matrix(candidates, label, center, plus.dtype)  # [EEF, probe-axis]
    visual_matrix = ((plus - minus) / 2.0).reshape(3, -1).T  # [Z, probe-axis]
    jacobian = visual_matrix @ torch.linalg.pinv(eef_matrix)  # dz / d(actual EEF xyz)
    _, gradient_z, _ = mi_and_gradient(tokens["center"], reference, normalization, estimator=estimator)
    pullback = pullback_gradient(jacobian, gradient_z.reshape(-1))
    mi_plus = torch.stack([mi_and_gradient(plus[i], reference, normalization, estimator=estimator)[0] for i in range(3)])
    mi_minus = torch.stack([mi_and_gradient(minus[i], reference, normalization, estimator=estimator)[0] for i in range(3)])
    direct = torch.linalg.pinv(eef_matrix.T) @ ((mi_plus - mi_minus) / 2.0)
    return jacobian, pullback, direct


def _candidate_map(record: dict) -> dict[str, dict]:
    return {item["candidate_id"]: item for item in record["candidates"] if not item["excluded_reasons"]}


def _metric(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    if len(left) < 2:
        return {"r2": None, "spearman": None, "sign_accuracy": None}
    residual = float(np.sum((left - right) ** 2))
    variance = float(np.sum((right - right.mean()) ** 2))
    r2 = None if variance <= 1e-12 else float(1.0 - residual / variance)
    ranks_left = np.argsort(np.argsort(left))
    ranks_right = np.argsort(np.argsort(right))
    spearman = float(np.corrcoef(ranks_left, ranks_right)[0, 1]) if np.std(ranks_left) and np.std(ranks_right) else None
    nonzero = np.abs(right) > 1e-9
    sign = None if not np.any(nonzero) else float(np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero])))
    return {"r2": r2, "spearman": spearman, "sign_accuracy": sign}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Collection manifest has no anchors.")
    extractor = None if args.representation == "raw_rgb" else LaWAMLAMFeatureExtractor(
        lam_config_path=args.lam_config, lam_ckpt_path=args.lam_checkpoint,
        vision_model_id=args.dino_model, device=args.device, strict=True,
    )
    estimator = DameSoftHistogramMI(num_bins=8, spline_order=3, normalization="none", channel_mode="channelwise")
    per_anchor = []
    all_cosines, stability_cosines, stability_relative_errors, heldout_pred, heldout_actual = [], [], [], [], []
    physical_cosines, physical_descent_aligned = [], []
    for record in rows:
        candidates = _candidate_map(record)
        required = ["center", *[f"{scale}_{sign}_{dim}" for scale in ("primary", "secondary") for sign in ("plus", "minus") for dim in range(3)]]
        missing = [key for key in required if key not in candidates]
        if missing:
            per_anchor.append({"anchor_id": record["anchor_id"], "status": "INVALID_PROTOCOL", "missing": missing})
            continue
        language = record["language"]
        encode = (lambda relative: _raw_rgb_tokens(args.collection_dir, relative)) if extractor is None else (lambda relative: _token(extractor, args.collection_dir, relative, language))
        reference = encode(record["reference"]["images"][args.view])
        tokens = {key: encode(item["images"][args.view]) for key, item in candidates.items()}
        center = tokens["center"]
        normalization = fit_fixed_normalization(reference, [center], fit_anchor_ids=(record["anchor_id"],))
        mi_center, gradient_z, saturation = mi_and_gradient(center, reference, normalization, estimator=estimator)
        center_candidate = candidates["center"]
        jacobian, gradient_pullback, gradient_fd = _axis_geometry(tokens, candidates, "primary", center_candidate, reference, normalization, estimator)
        secondary_jacobian, _, _ = _axis_geometry(tokens, candidates, "secondary", center_candidate, reference, normalization, estimator)
        cosine = float(torch.nn.functional.cosine_similarity(gradient_pullback[None], gradient_fd[None]).item())
        stability_cosine = float(torch.nn.functional.cosine_similarity(jacobian.reshape(1, -1), secondary_jacobian.reshape(1, -1)).item())
        stability_relative = float(torch.linalg.vector_norm(jacobian - secondary_jacobian) / torch.linalg.vector_norm(jacobian).clamp_min(1e-12))
        all_cosines.append(cosine)
        stability_cosines.append(stability_cosine)
        stability_relative_errors.append(stability_relative)
        physical_descent = _physical_descent_direction(candidates, center_candidate, gradient_pullback.dtype)
        physical_cosine = float(torch.nn.functional.cosine_similarity(gradient_pullback[None], physical_descent[None]).item())
        physical_cosines.append(physical_cosine)
        physical_descent_aligned.append(physical_cosine > 0.0)
        valid_heldout = [item for key, item in candidates.items() if key.startswith("heldout_")]
        predictions, actuals = [], []
        for item in valid_heldout:
            action = _eef_delta(item, center_candidate).to(dtype=gradient_pullback.dtype)
            value = mi_and_gradient(tokens[item["candidate_id"]], reference, normalization, estimator=estimator)[0]
            predictions.append(float(torch.dot(gradient_pullback, action)))
            actuals.append(float(value - mi_center))
        heldout_pred.extend(predictions)
        heldout_actual.extend(actuals)
        per_anchor.append({
            "anchor_id": record["anchor_id"], "status": "VALID",
            "mi_center": float(mi_center), "saturation_ratio": saturation,
            "jacobian_shape": list(jacobian.shape), "action_coordinate": "actual_eef_delta_xyz",
            "gradient_pullback": gradient_pullback.tolist(), "gradient_finite_difference": gradient_fd.tolist(),
            "gradient_cosine": cosine, "primary_secondary_jacobian_cosine": stability_cosine,
            "primary_secondary_jacobian_relative_error": stability_relative,
            "physical_descent_direction": physical_descent.tolist(), "physical_descent_cosine": physical_cosine,
            "mi_ascent_reduces_eef_object_distance": physical_cosine > 0.0,
            "heldout_count": len(predictions), "heldout": _metric(np.asarray(predictions), np.asarray(actuals)),
        })
        print(f"ANCHOR {record['anchor_id']} cosine={cosine:.6f} heldout={len(predictions)}", flush=True)
    heldout = _metric(np.asarray(heldout_pred), np.asarray(heldout_actual))
    derivative_pass = bool(all_cosines) and float(np.median(all_cosines)) >= 0.90
    stability_pass = bool(stability_cosines) and float(np.median(stability_cosines)) >= 0.90
    heldout_complete = len(heldout_actual) >= 24
    heldout_pass = (
        heldout_complete
        and heldout["r2"] is not None and float(heldout["r2"]) >= 0.50
        and heldout["spearman"] is not None and float(heldout["spearman"]) >= 0.60
        and heldout["sign_accuracy"] is not None and float(heldout["sign_accuracy"]) >= 0.80
    )
    physical_pass = (
        bool(physical_cosines)
        and float(np.median(physical_cosines)) >= 0.70
        and float(np.mean(physical_descent_aligned)) >= 0.70
    )
    status = "PASS" if derivative_pass and stability_pass and heldout_pass and physical_pass else ("NO_GO" if derivative_pass and heldout_complete else "INCOMPLETE")
    report = {
        "benchmark": rows[0]["suite"], "task_id": rows[0]["task_id"], "protocol": "v5_local_geometry_eval_v1",
        "view": args.view, "representation": args.representation, "anchors": per_anchor,
        "derivative": {"gradient_cosine_median": float(np.median(all_cosines)) if all_cosines else None},
        "cross_epsilon": {"jacobian_cosine_median": float(np.median(stability_cosines)) if stability_cosines else None, "jacobian_relative_error_median": float(np.median(stability_relative_errors)) if stability_relative_errors else None},
        "heldout_mi": {"count": len(heldout_actual), **heldout},
        "physical_direction": {
            "eef_object_descent_cosine_median": float(np.median(physical_cosines)) if physical_cosines else None,
            "mi_ascent_reduces_distance_fraction": float(np.mean(physical_descent_aligned)) if physical_descent_aligned else None,
            "anchors_evaluated": len(physical_cosines),
            "pass": physical_pass,
        },
        "gates": {"derivative_pass": derivative_pass, "cross_epsilon_pass": stability_pass, "heldout_complete": heldout_complete, "heldout_pass": heldout_pass, "physical_direction_pass": physical_pass},
        "status": status,
        "scope": f"{args.representation} representation with simulator finite-difference Jacobian and privileged EEF-object direction. Hessian is not evaluated by this first-order evaluator.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
