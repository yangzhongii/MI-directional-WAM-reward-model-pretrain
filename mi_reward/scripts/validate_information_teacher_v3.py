"""Small reproducible sanity checks for the Pipeline-v3 information teacher.

This script is intentionally diagnostic rather than a training entry point.
It checks two things:

1. On a synthetic distribution with known goal-related structure, the
   contrastive density-ratio critics recover a progress-sensitive visual
   potential and goal-directed transition evidence.
2. On extracted LIBERO privileged sidecars, successful demonstrations contain
   physically measurable toward / neutral / regression transitions that can
   support same-task hard-negative construction.

Passing this script is necessary but not sufficient evidence for the paper
claim; the real Gate-1 benchmark remains a held-out LIBERO experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
    compute_information_teacher_evidence,
    conditional_action_information_loss,
    visual_information_loss,
)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    return float(torch.corrcoef(torch.stack((x_rank, y_rank)))[0, 1].item())


def synthetic_sanity(*, seed: int = 7, steps: int = 700) -> dict[str, float | str]:
    """Fit the v3 critics on a controlled distribution with known direction."""

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_count = 5000
    latent_dim = 8

    goal = torch.randn(sample_count, latent_dim)
    goal = goal / goal.norm(dim=1, keepdim=True).clamp_min(1e-6)
    progress = torch.rand(sample_count, 1)
    visual = progress * goal + 0.35 * torch.randn(sample_count, latent_dim)

    # The joint / goal-conditioned action distribution is intentionally
    # goal-directed.  The conditional marginal negative holds (v,g) fixed and
    # substitutes an equally plausible but goal-inconsistent transition.
    # This is the synthetic analogue of a same-state toward-vs-away hard pair.
    toward_visual_next = (progress + 0.08).clamp(0.0, 1.0) * goal
    toward_visual_next = toward_visual_next + 0.35 * torch.randn(sample_count, latent_dim)
    away_visual_next = (progress - 0.08).clamp(0.0, 1.0) * goal
    away_visual_next = away_visual_next + 0.35 * torch.randn(sample_count, latent_dim)

    toward_action = goal + 0.25 * torch.randn(sample_count, latent_dim)
    away_action = -goal + 0.25 * torch.randn(sample_count, latent_dim)

    permutation = torch.randperm(sample_count)
    train_indices = permutation[:4000]
    validation_indices = permutation[4000:]

    visual_critic = VisualPointwiseInformationCritic(
        latent_dim,
        latent_dim,
        projection_dim=32,
        hidden_dim=64,
    ).to(device)
    action_critic = ConditionalActionInformationCritic(
        latent_dim,
        latent_dim,
        latent_dim,
        projection_dim=32,
        hidden_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(visual_critic.parameters()) + list(action_critic.parameters()),
        lr=2e-3,
    )

    for _ in range(int(steps)):
        indices = train_indices[torch.randint(0, len(train_indices), (256,))]
        v = visual[indices].to(device)
        g = goal[indices].to(device)
        a = toward_action[indices].to(device)
        a_negative = away_action[indices].to(device)
        g_negative = g[torch.randperm(len(g))]
        loss = visual_information_loss(visual_critic, v, g, g_negative)
        loss = loss + conditional_action_information_loss(
            action_critic,
            a,
            v,
            g,
            a_negative,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        indices = validation_indices
        toward_evidence = compute_information_teacher_evidence(
            visual_critic,
            action_critic,
            visual_t=visual[indices].to(device),
            visual_t1=toward_visual_next[indices].to(device),
            goal=goal[indices].to(device),
            action_t=toward_action[indices].to(device),
            gamma=1.0,
            beta=0.35,
        )
        away_evidence = compute_information_teacher_evidence(
            visual_critic,
            action_critic,
            visual_t=visual[indices].to(device),
            visual_t1=away_visual_next[indices].to(device),
            goal=goal[indices].to(device),
            action_t=away_action[indices].to(device),
            gamma=1.0,
            beta=0.35,
        )
        phi = toward_evidence.phi_visual_t.cpu()
        psi_toward = toward_evidence.conditional_action_information.cpu()
        psi_away = away_evidence.conditional_action_information.cpu()
        delta_toward = toward_evidence.delta_phi_visual.cpu()
        delta_away = away_evidence.delta_phi_visual.cpu()
        score_toward = toward_evidence.directional_score.cpu()
        score_away = away_evidence.directional_score.cpu()
        true_progress = progress[indices, 0]

    psi_pair_accuracy = float((psi_toward > psi_away).float().mean().item())
    delta_pair_accuracy = float((delta_toward > delta_away).float().mean().item())
    score_pair_accuracy = float((score_toward > score_away).float().mean().item())

    return {
        "device": str(device),
        "phi_progress_spearman": _spearman(phi, true_progress),
        "psi_toward_over_away_pair_accuracy": psi_pair_accuracy,
        "delta_phi_toward_over_away_pair_accuracy": delta_pair_accuracy,
        "combined_D_toward_over_away_pair_accuracy": score_pair_accuracy,
        "mean_psi_toward": float(psi_toward.mean().item()),
        "mean_psi_away": float(psi_away.mean().item()),
        "mean_delta_toward": float(delta_toward.mean().item()),
        "mean_delta_away": float(delta_away.mean().item()),
        "mean_D_toward": float(score_toward.mean().item()),
        "mean_D_away": float(score_away.mean().item()),
    }


def sidecar_direction_stats(path: Path, *, epsilon: float = 5e-4) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    toward = 0
    neutral = 0
    regression = 0
    grasp_toward = 0
    grasp_regression = 0
    deltas: list[float] = []

    for left, right in zip(frames[:-1], frames[1:]):
        delta = float(right["object_goal_distance_xy"]) - float(left["object_goal_distance_xy"])
        deltas.append(delta)
        if delta < -epsilon:
            toward += 1
            if bool(left["grasped"]):
                grasp_toward += 1
        elif delta > epsilon:
            regression += 1
            if bool(left["grasped"]):
                grasp_regression += 1
        else:
            neutral += 1

    return {
        "path": str(path),
        "demo_name": payload.get("demo_name"),
        "transitions": len(deltas),
        "toward": toward,
        "neutral": neutral,
        "regression": regression,
        "grasp_toward": grasp_toward,
        "grasp_regression": grasp_regression,
        "mean_distance_delta_xy": sum(deltas) / max(len(deltas), 1),
        "first_success": next(
            (frame["frame_index"] for frame in frames if frame["environment_success"]),
            None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--epsilon", type=float, default=5e-4)
    args = parser.parse_args()

    report = {
        "synthetic": synthetic_sanity(seed=args.seed, steps=args.steps),
        "libero_sidecars": [
            sidecar_direction_stats(path, epsilon=args.epsilon) for path in args.sidecar
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
