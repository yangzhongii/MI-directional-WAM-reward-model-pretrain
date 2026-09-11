"""Evaluate a trained Pipeline-v3 information teacher on cached LIBERO features.

This evaluator is intentionally lightweight: it never replays MuJoCo or
re-extracts LaWAM features.  It loads cached held-out episodes and reports
direction accuracy for cosine, visual TD information, conditional action
information, and a sweep over the action weight beta in

    D_t = gamma * phi(t+1) - phi(t) + beta * psi_a.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def direction_accuracy(score: torch.Tensor, direction: torch.Tensor) -> tuple[int, int]:
    keep = direction != 0
    if not bool(keep.any()):
        return 0, 0
    correct = ((score[keep] > 0) == (direction[keep] > 0)).sum().item()
    return int(correct), int(keep.sum().item())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    visual_dim = int(payload["visual_dim"])
    action_dim = int(payload["action_dim"])
    gamma = float(payload.get("gamma", 0.99))

    visual_critic = VisualPointwiseInformationCritic(visual_dim, visual_dim).to(device)
    action_critic = ConditionalActionInformationCritic(action_dim, visual_dim, visual_dim).to(device)
    visual_critic.load_state_dict(payload["visual_critic"])
    action_critic.load_state_dict(payload["action_critic"])
    visual_critic.eval()
    action_critic.eval()

    caches = sorted(args.cache_dir.glob("task_*_demo_10.pt"))
    if not caches:
        raise FileNotFoundError(f"No held-out cache files found under {args.cache_dir}")

    aggregate: dict[str, list[int]] = {
        "cosine": [0, 0],
        "delta_phi": [0, 0],
        "psi": [0, 0],
    }
    beta_totals = {float(beta): [0, 0] for beta in args.betas}
    episodes: list[dict[str, object]] = []

    for cache_path in caches:
        episode = torch.load(cache_path, map_location="cpu", weights_only=False)
        visual = episode["visual"].to(device)
        action = episode["action"].to(device)
        goal = episode["goal"].to(device).unsqueeze(0)
        direction = episode["direction"].to(device)
        goal_frames = goal.expand(visual.shape[0], -1)

        phi = visual_critic(visual, goal_frames)
        delta_phi = gamma * phi[1:] - phi[:-1]
        psi = action_critic(action, visual[:-1], goal.expand(action.shape[0], -1))
        cosine = F.cosine_similarity(visual, goal_frames, dim=-1)
        delta_cosine = cosine[1:] - cosine[:-1]

        row: dict[str, object] = {
            "task_id": int(episode["task_id"]),
            "demo_name": str(episode["demo_name"]),
            "forward": int((direction > 0).sum().item()),
            "neutral": int((direction == 0).sum().item()),
            "regression": int((direction < 0).sum().item()),
        }
        for name, score in (("cosine", delta_cosine), ("delta_phi", delta_phi), ("psi", psi)):
            correct, count = direction_accuracy(score, direction)
            aggregate[name][0] += correct
            aggregate[name][1] += count
            row[f"{name}_direction_accuracy"] = None if count == 0 else correct / count

        beta_rows: dict[str, float | None] = {}
        for beta in args.betas:
            score = delta_phi + float(beta) * psi
            correct, count = direction_accuracy(score, direction)
            beta_totals[float(beta)][0] += correct
            beta_totals[float(beta)][1] += count
            beta_rows[str(float(beta))] = None if count == 0 else correct / count
        row["beta_sweep"] = beta_rows
        episodes.append(row)

    aggregate_report = {
        name: (None if count == 0 else correct / count)
        for name, (correct, count) in aggregate.items()
    }
    beta_report = {
        str(beta): (None if count == 0 else correct / count)
        for beta, (correct, count) in beta_totals.items()
    }
    valid = [(beta, value) for beta, value in ((float(k), v) for k, v in beta_report.items()) if value is not None]
    best_beta, best_accuracy = max(valid, key=lambda item: item[1])
    report = {
        "checkpoint": str(args.checkpoint),
        "gamma": gamma,
        "aggregate": aggregate_report,
        "beta_sweep": beta_report,
        "best_beta": best_beta,
        "best_direction_accuracy": best_accuracy,
        "episodes": episodes,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
