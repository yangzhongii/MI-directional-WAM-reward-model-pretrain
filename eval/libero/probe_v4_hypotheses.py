"""Frozen, training-free diagnostics for the Pipeline-v4 decision gate.

This script reads the v3 teacher checkpoint and cached episode features.  It
does not update weights or calibration.  It tests three questions:

1. How sensitive is one-step direction classification to gamma and an
   additive offset of the visual potential?
2. Does the visual critic identify task-level goals on held-out episodes?
3. Does the visual term improve the action-only score when gamma is one?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/teacher_smoke5/information_teacher_v3.pt"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/mi_reward/v4_decision_gate/frozen_probe_v1"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def signed_metrics(scores: np.ndarray, directions: np.ndarray) -> dict[str, float | int | None]:
    keep = directions != 0
    values = scores[keep]
    labels = directions[keep]
    forward = labels > 0
    regression = labels < 0
    forward_recall = float(np.mean(values[forward] > 0)) if np.any(forward) else None
    regression_recall = float(np.mean(values[regression] < 0)) if np.any(regression) else None
    balanced = (
        None
        if forward_recall is None or regression_recall is None
        else float((forward_recall + regression_recall) / 2.0)
    )
    return {
        "non_neutral": int(keep.sum()),
        "forward": int(forward.sum()),
        "regression": int(regression.sum()),
        "forward_recall": forward_recall,
        "regression_recall": regression_recall,
        "balanced_accuracy": balanced,
    }


def _balanced_accuracy_for_tasks(rows: list[dict[str, Any]], tasks: list[int], score_key: str) -> float:
    scores: list[float] = []
    directions: list[int] = []
    for task in tasks:
        row = next(item for item in rows if int(item["task_id"]) == int(task))
        scores.extend(row[score_key])
        directions.extend(row["direction"])
    metric = signed_metrics(np.asarray(scores), np.asarray(directions))["balanced_accuracy"]
    return float("nan") if metric is None else float(metric)


def paired_task_bootstrap(
    rows: list[dict[str, Any]], left: str, right: str, *, draws: int, seed: int
) -> dict[str, float]:
    tasks = sorted(int(row["task_id"]) for row in rows)
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(tasks, size=len(tasks), replace=True).tolist()
        differences[index] = _balanced_accuracy_for_tasks(rows, sampled, left) - _balanced_accuracy_for_tasks(
            rows, sampled, right
        )
    differences = differences[np.isfinite(differences)]
    return {
        "draws_retained": int(len(differences)),
        "mean": float(np.mean(differences)),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def load_models(path: Path) -> tuple[VisualPointwiseInformationCritic, ConditionalActionInformationCritic, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    visual = VisualPointwiseInformationCritic(payload["visual_dim"], payload["visual_dim"]).eval()
    action = ConditionalActionInformationCritic(payload["action_dim"], payload["visual_dim"], payload["visual_dim"]).eval()
    visual.load_state_dict(payload["visual_critic"])
    action.load_state_dict(payload["action_critic"])
    return visual, action, payload


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    visual_critic, action_critic, checkpoint = load_models(args.checkpoint)
    episodes = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.feature_dir.glob("task_*_demo_*.pt"))
    ]
    train = [episode for episode in episodes if episode["split"] == "train"]
    validation = [episode for episode in episodes if episode["split"] == "validation"]
    training_goals = [(int(ep["task_id"]), str(ep["demo_name"]), ep["goal"].float()) for ep in train]
    beta = float(checkpoint.get("beta", 1.0))

    rows: list[dict[str, Any]] = []
    retrieval_correct = 0
    retrieval_total = 0
    same_task_margins: list[float] = []
    for episode in validation:
        task_id = int(episode["task_id"])
        states = episode["visual"].float()
        actions = episode["action"].float()
        directions = episode["direction"].cpu().numpy().astype(np.int64)
        own_goal = episode["goal"].float()
        own_phi = visual_critic(states, own_goal.expand(len(states), -1))
        own_psi = action_critic(actions, states[:-1], own_goal.expand(len(actions), -1))

        candidate_phi = []
        for goal_task, _, goal in training_goals:
            candidate_phi.append(visual_critic(states, goal.expand(len(states), -1)))
        candidate_matrix = torch.stack(candidate_phi, dim=1)
        candidate_tasks = np.asarray([item[0] for item in training_goals])
        best = candidate_matrix.argmax(dim=1).cpu().numpy()
        retrieval_correct += int(np.sum(candidate_tasks[best] == task_id))
        retrieval_total += len(states)
        same = candidate_matrix[:, candidate_tasks == task_id].mean(dim=1)
        cross = candidate_matrix[:, candidate_tasks != task_id].mean(dim=1)
        same_task_margins.extend((same - cross).cpu().tolist())

        task_candidates = candidate_matrix[:, candidate_tasks == task_id]
        phi_task_mean = task_candidates.mean(dim=1)
        phi_task_max = task_candidates.max(dim=1).values
        phi_task_logmeanexp = torch.logsumexp(task_candidates, dim=1) - math.log(task_candidates.shape[1])

        score_map: dict[str, torch.Tensor] = {"psi_only": beta * own_psi}
        for gamma in (1.0, 0.995, 0.99):
            score_map[f"visual_own_gamma_{gamma:g}"] = gamma * own_phi[1:] - own_phi[:-1]
            score_map[f"combined_own_gamma_{gamma:g}"] = (
                gamma * own_phi[1:] - own_phi[:-1] + beta * own_psi
            )
        for name, phi in (
            ("task_mean", phi_task_mean),
            ("task_max", phi_task_max),
            ("task_logmeanexp", phi_task_logmeanexp),
        ):
            score_map[f"visual_{name}_gamma_1"] = phi[1:] - phi[:-1]
        for offset in (-10.0, -5.0, 0.0, 5.0, 10.0):
            shifted = own_phi + offset
            score_map[f"visual_gamma_099_offset_{offset:+g}"] = 0.99 * shifted[1:] - shifted[:-1]

        rows.append(
            {
                "task_id": task_id,
                "demo_name": str(episode["demo_name"]),
                "direction": directions.tolist(),
                **{key: value.cpu().tolist() for key, value in score_map.items()},
            }
        )

    score_keys = [key for key in rows[0] if key not in {"task_id", "demo_name", "direction"}]
    aggregate: dict[str, Any] = {}
    all_directions = np.concatenate([np.asarray(row["direction"]) for row in rows])
    for key in score_keys:
        values = np.concatenate([np.asarray(row[key]) for row in rows])
        aggregate[key] = signed_metrics(values, all_directions)

    comparisons = {
        "combined_gamma1_minus_psi": paired_task_bootstrap(
            rows, "combined_own_gamma_1", "psi_only", draws=args.bootstrap_draws, seed=args.seed
        ),
        "visual_gamma1_minus_gamma099": paired_task_bootstrap(
            rows, "visual_own_gamma_1", "visual_own_gamma_0.99", draws=args.bootstrap_draws, seed=args.seed + 1
        ),
        "task_max_minus_own_goal": paired_task_bootstrap(
            rows, "visual_task_max_gamma_1", "visual_own_gamma_1", draws=args.bootstrap_draws, seed=args.seed + 2
        ),
    }
    result = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "feature_dir": str(args.feature_dir),
            "training": False,
            "calibration_changed": False,
            "validation_episodes": len(validation),
            "task_clusters": len({int(ep["task_id"]) for ep in validation}),
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "warning": "Direction labels are the existing physical heuristic and are not independent annotations.",
        },
        "goal_retrieval": {
            "heldout_state_top1_training_goal_same_task_accuracy": retrieval_correct / retrieval_total,
            "heldout_states": retrieval_total,
            "same_task_minus_cross_task_logit_mean": float(np.mean(same_task_margins)),
            "same_task_minus_cross_task_logit_median": float(np.median(same_task_margins)),
        },
        "aggregate": aggregate,
        "paired_task_cluster_bootstrap": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# Pipeline-v4 frozen decision probe",
        "",
        "No weights, calibration, labels, or cached features were changed.",
        "",
        "## Key metrics",
        "",
        "| Score | Balanced accuracy | Forward recall | Regression recall |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "visual_own_gamma_0.99",
        "visual_own_gamma_1",
        "psi_only",
        "combined_own_gamma_0.99",
        "combined_own_gamma_1",
        "visual_task_mean_gamma_1",
        "visual_task_max_gamma_1",
        "visual_task_logmeanexp_gamma_1",
    ):
        metric = aggregate[key]
        summary.append(
            f"| `{key}` | {metric['balanced_accuracy']:.4f} | "
            f"{metric['forward_recall']:.4f} | {metric['regression_recall']:.4f} |"
        )
    summary.extend(
        [
            "",
            "## Goal sensitivity",
            "",
            f"- Same-task top-1 goal retrieval: {result['goal_retrieval']['heldout_state_top1_training_goal_same_task_accuracy']:.4f}",
            f"- Mean same-task minus cross-task logit: {result['goal_retrieval']['same_task_minus_cross_task_logit_mean']:.4f}",
            "",
            "## Paired task-cluster differences",
            "",
        ]
    )
    for key, value in comparisons.items():
        summary.append(
            f"- `{key}`: mean={value['mean']:.4f}, 95% CI "
            f"[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"
        )
    summary.extend(
        [
            "",
            "These metrics use only five task clusters and heuristic directions. They are a decision probe, not a final benchmark.",
            "",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
