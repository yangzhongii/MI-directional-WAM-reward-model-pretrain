"""Frozen v5 Test-1b evaluation: action-conditioned latent dynamics only, no MI."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mi_reward.models.action_conditioned_latent import ActionConditionedControlModel
from mi_reward.training.train_v6_control_dynamics import _cache_name, _cache_tokens, _read_rows


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-collection-dir", type=Path, required=True)
    parser.add_argument("--train-collection-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("mi_reward/configs/lawam_control_latent_delta.yaml"))
    parser.add_argument("--train-feature-cache-dir", type=Path, required=True)
    parser.add_argument("--test-feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge", type=float, default=1e-4)
    return parser.parse_args()


def _token(cache: Path, relative: str) -> torch.Tensor:
    return torch.load(cache / _cache_name(relative), map_location="cpu", weights_only=True).float()


def _metric(predicted_next: torch.Tensor, target_next: torch.Tensor, predicted_delta: torch.Tensor, target_delta: torch.Tensor) -> dict[str, float | int]:
    residual = float((predicted_next - target_next).square().sum())
    variance = float((target_next - target_next.mean()).square().sum())
    cosine = F.cosine_similarity(predicted_delta.flatten(1), target_delta.flatten(1), dim=1)
    return {
        "count": int(len(target_next)),
        "next_latent_r2": float(1.0 - residual / max(variance, 1e-12)),
        "delta_cosine_mean": float(cosine.mean()),
        "delta_cosine_median": float(cosine.median()),
        "delta_cosine_variance": float(cosine.var(unbiased=False)),
    }


def _load_model(checkpoint_path: Path, cfg: dict[str, Any], input_dim: int, device: torch.device) -> ActionConditionedControlModel:
    model = ActionConditionedControlModel(
        input_dim=input_dim,
        control_dim=int(cfg["representation"]["control_dim"]),
        action_dim=int(cfg["dynamics"]["action_dim"]),
        hidden_dim=int(cfg["dynamics"]["hidden_dim"]),
        action_scale=float(cfg["dynamics"]["action_scale_m"]),
        spatial_heads=int(cfg["dynamics"]["spatial_heads"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _fit_linear_baseline(rows: list[dict[str, Any]], cache: Path, model: ActionConditionedControlModel, device: torch.device, ridge: float) -> torch.Tensor:
    actions, deltas = [], []
    with torch.no_grad():
        for row in rows:
            current = _token(cache, row["current"]).unsqueeze(0).to(device)
            nxt = _token(cache, row["next"]).unsqueeze(0).to(device)
            delta = (model.project(nxt) - model.project(current)).flatten().cpu()
            actions.append(torch.tensor(row["action"], dtype=torch.float32))
            deltas.append(delta)
    x, y = torch.stack(actions).double(), torch.stack(deltas).double()
    return torch.linalg.solve(x.T @ x + ridge * torch.eye(3, dtype=torch.float64), x.T @ y).float()


def _test_rows(collection: Path, allowed_demos: set[int]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    anchors: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    for line in (collection / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        demo = int(str(row["source_demo"]).removeprefix("demo_"))
        if demo not in allowed_demos:
            continue
        candidates = [item for item in row["candidates"] if not item["excluded_reasons"]]
        center = next(item for item in candidates if item["candidate_id"] == "center")
        anchors[row["anchor_id"]] = {"demo": demo, "center": center, "candidates": candidates, "language": row["language"]}
        for candidate in candidates:
            if candidate["candidate_id"] == "center":
                continue
            action = np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float32) - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float32)
            samples.append({"anchor_id": row["anchor_id"], "demo": demo, "language": row["language"], "current": center["images"]["wrist"], "next": candidate["images"]["wrist"], "action": action.tolist(), "candidate_id": candidate["candidate_id"]})
    return samples, anchors


def _cross_epsilon(anchor: dict[str, Any], cache: Path, model: ActionConditionedControlModel, device: torch.device) -> float | None:
    candidates = {item["candidate_id"]: item for item in anchor["candidates"]}
    current = _token(cache, anchor["center"]["images"]["wrist"]).unsqueeze(0).to(device)
    with torch.no_grad():
        control = model.project(current)
        pairs = []
        for dim in range(3):
            for sign in ("plus", "minus"):
                primary, secondary = candidates.get(f"primary_{sign}_{dim}"), candidates.get(f"secondary_{sign}_{dim}")
                if primary is None or secondary is None:
                    continue
                a_primary = torch.tensor(np.asarray(primary["physical_after"]["eef_pos"], dtype=np.float32) - np.asarray(anchor["center"]["physical_after"]["eef_pos"], dtype=np.float32), device=device).unsqueeze(0)
                a_secondary = torch.tensor(np.asarray(secondary["physical_after"]["eef_pos"], dtype=np.float32) - np.asarray(anchor["center"]["physical_after"]["eef_pos"], dtype=np.float32), device=device).unsqueeze(0)
                d_primary = model.dynamics.predict_delta(control, a_primary).flatten(1)
                d_secondary = model.dynamics.predict_delta(control, a_secondary).flatten(1)
                pairs.append(float(F.cosine_similarity(d_primary, d_secondary, dim=1).item()))
    return None if not pairs else float(np.mean(pairs))


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    cfg = yaml.safe_load(args.config.read_text())
    frozen_demos = {int(value) for value in cfg["split"]["frozen_test_demos"]}
    train_demos = {int(value) for value in cfg["split"]["train_demos"]}
    if frozen_demos & train_demos:
        raise ValueError("Frozen test demos overlap training demos.")
    train_rows = _read_rows(args.train_collection_dir, train_demos)
    test_rows, anchors = _test_rows(args.test_collection_dir, frozen_demos)
    if len(anchors) != 20:
        raise RuntimeError(f"Expected exactly 20 frozen anchors, found {len(anchors)}.")
    _cache_tokens(test_rows, args.test_collection_dir, args.test_feature_cache_dir, args)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    input_dim = int(_token(args.test_feature_cache_dir, test_rows[0]["current"]).shape[-1])
    model = _load_model(args.checkpoint, cfg, input_dim, device)
    linear_weight = _fit_linear_baseline(train_rows, args.train_feature_cache_dir, model, device, args.ridge)
    per_anchor: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for anchor_id, anchor in anchors.items():
            rows = [row for row in test_rows if row["anchor_id"] == anchor_id]
            current = _token(args.test_feature_cache_dir, anchor["center"]["images"]["wrist"]).unsqueeze(0).to(device)
            control = model.project(current)
            targets, actions = [], []
            for row in rows:
                targets.append(model.project(_token(args.test_feature_cache_dir, row["next"]).unsqueeze(0).to(device)).squeeze(0).cpu())
                actions.append(torch.tensor(row["action"], dtype=torch.float32))
            target = torch.stack(targets)
            action = torch.stack(actions)
            start = control.cpu().expand(len(rows), -1, -1)
            target_delta = target - start
            learned_delta = model.dynamics.predict_delta(control.expand(len(rows), -1, -1), action.to(device)).cpu()
            linear_delta = (action @ linear_weight).reshape_as(target_delta)
            zero_delta = torch.zeros_like(target_delta)
            per_anchor[anchor_id] = {
                "demo": anchor["demo"],
                "count": len(rows),
                "learned_spatial_delta": _metric(start + learned_delta, target, learned_delta, target_delta),
                "linear_action_delta": _metric(start + linear_delta, target, linear_delta, target_delta),
                "static_next_state": _metric(start, target, zero_delta, target_delta),
                "zero_action_model": _metric(start + zero_delta, target, zero_delta, target_delta),
                "cross_epsilon_predicted_delta_cosine": _cross_epsilon(anchor, args.test_feature_cache_dir, model, device),
            }
            print(f"ANCHOR {anchor_id} count={len(rows)}", flush=True)
    states: dict[str, dict[str, Any]] = {}
    for demo in sorted(frozen_demos):
        rows = [value for value in per_anchor.values() if value["demo"] == demo]
        state: dict[str, Any] = {"anchors": len(rows), "models": {}}
        for key in ("learned_spatial_delta", "linear_action_delta", "static_next_state", "zero_action_model"):
            state["models"][key] = {
                metric: float(np.mean([item[key][metric] for item in rows]))
                for metric in ("next_latent_r2", "delta_cosine_mean", "delta_cosine_median", "delta_cosine_variance")
            }
            state["models"][key]["anchor_r2_variance"] = float(np.var([item[key]["next_latent_r2"] for item in rows]))
            state["models"][key]["anchor_delta_cosine_median_variance"] = float(np.var([item[key]["delta_cosine_median"] for item in rows]))
        cross = [item["cross_epsilon_predicted_delta_cosine"] for item in rows if item["cross_epsilon_predicted_delta_cosine"] is not None]
        state["cross_epsilon_predicted_delta_cosine"] = {"mean": float(np.mean(cross)), "variance": float(np.var(cross)), "anchors": len(cross)}
        states[f"demo_{demo}"] = state
    learned_state_medians = [state["models"]["learned_spatial_delta"]["delta_cosine_median"] for state in states.values()]
    report = {
        "protocol": "v6_test1b_frozen_v5_heldout_v1",
        "checkpoint": str(args.checkpoint),
        "test_collection": str(args.test_collection_dir),
        "train_collection_for_linear_baseline": str(args.train_collection_dir),
        "anchors": 20,
        "frozen_initial_states": sorted(frozen_demos),
        "no_mi": True,
        "linear_baseline_ridge": args.ridge,
        "per_initial_state": states,
        "per_anchor": per_anchor,
        "cross_state_learned_delta_cosine_median": {"mean": float(np.mean(learned_state_medians)), "variance": float(np.var(learned_state_medians)), "minimum": float(np.min(learned_state_medians))},
        "status": "TEST1B_HELDOUT_ONLY_NO_MI",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["cross_state_learned_delta_cosine_median"], indent=2), flush=True)


if __name__ == "__main__":
    main()
