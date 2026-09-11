"""Held-out smoke test for Pipeline-v3 physical consistency and P/U/N.

This evaluator compares three supervision adapters on cached LIBERO features:

1. ``D-only``: P/U/N from the information score and a calibrated dead band;
2. ``weak physical``: adds only grasp/contact/success event consistency;
3. ``full physical``: additionally uses privileged task-relation direction.

The full gate deliberately uses privileged relation evidence, so its comparison
to the same physical direction labels is a pipeline sanity check rather than an
independent scientific validation.  The weak gate is reported separately to
show how much correction comes from non-distance physical events alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task
from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
)
from mi_reward.scoring.teacher_consistency_v3 import (
    N_LABEL,
    P_LABEL,
    U_LABEL,
    PhysicalConsistencyEvidence,
    information_direction,
    project_teacher_label,
    project_teacher_label_veto,
)


LABELS = (P_LABEL, U_LABEL, N_LABEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--neutral-quantile", type=float, default=0.75)
    parser.add_argument("--alias-quantile", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _truth_label(direction: int) -> str:
    if direction > 0:
        return P_LABEL
    if direction < 0:
        return N_LABEL
    return U_LABEL


def _d_only_label(score: float, deadband: float) -> str:
    direction = information_direction(score, deadband=deadband)
    return P_LABEL if direction > 0 else N_LABEL if direction < 0 else U_LABEL


def _macro_f1(truth: list[str], pred: list[str]) -> float:
    scores: list[float] = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def _metrics(truth: list[str], pred: list[str], alias_mask: list[bool]) -> dict[str, float | int | None]:
    count = len(truth)
    predicted_positive = sum(label == P_LABEL for label in pred)
    true_positive = sum(t == P_LABEL and p == P_LABEL for t, p in zip(truth, pred))
    false_positive = sum(t != P_LABEL and p == P_LABEL for t, p in zip(truth, pred))
    nonforward = sum(t != P_LABEL for t in truth)
    alias_nonforward = sum(a and t != P_LABEL for a, t in zip(alias_mask, truth))
    alias_fp = sum(a and t != P_LABEL and p == P_LABEL for a, t, p in zip(alias_mask, truth, pred))
    return {
        "count": count,
        "macro_f1": _macro_f1(truth, pred),
        "positive_precision": None if predicted_positive == 0 else true_positive / predicted_positive,
        "positive_recall": None if not any(t == P_LABEL for t in truth) else true_positive / sum(t == P_LABEL for t in truth),
        "predicted_positive": predicted_positive,
        "false_positive": false_positive,
        "nonforward_false_positive_rate": None if nonforward == 0 else false_positive / nonforward,
        "alias_nonforward_count": alias_nonforward,
        "alias_false_positive": alias_fp,
        "alias_nonforward_false_positive_rate": None if alias_nonforward == 0 else alias_fp / alias_nonforward,
        "unclear_rate": sum(label == U_LABEL for label in pred) / max(count, 1),
    }


@torch.no_grad()
def _score_episode(
    episode: dict[str, object],
    visual_critic: VisualPointwiseInformationCritic,
    action_critic: ConditionalActionInformationCritic,
    *,
    device: torch.device,
    gamma: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    visual = episode["visual"].to(device)
    action = episode["action"].to(device)
    goal = episode["goal"].to(device).unsqueeze(0)
    phi = visual_critic(visual, goal.expand(visual.shape[0], -1))
    psi = action_critic(action, visual[:-1], goal.expand(action.shape[0], -1))
    score = gamma * phi[1:] - phi[:-1] + beta * psi
    consecutive_similarity = F.cosine_similarity(visual[:-1], visual[1:], dim=-1)
    return score.cpu(), consecutive_similarity.cpu()


def _load_physical_frames(root: Path, suite: str, task_id: int, demo_name: str) -> list[dict[str, object]]:
    import h5py

    demo_path, bddl_path, _ = resolve_libero_task(root, suite, task_id)
    with h5py.File(demo_path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        states = np.asarray(demo["states"], dtype=np.float64)
        rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float64)
        dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.uint8)
    privileged = extract_privileged_records(
        bddl_path=bddl_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    return list(privileged["frames"])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    visual_dim = int(checkpoint["visual_dim"])
    action_dim = int(checkpoint["action_dim"])
    gamma = float(checkpoint.get("gamma", 0.99))
    beta = float(args.beta)

    visual_critic = VisualPointwiseInformationCritic(visual_dim, visual_dim).to(device)
    action_critic = ConditionalActionInformationCritic(action_dim, visual_dim, visual_dim).to(device)
    visual_critic.load_state_dict(checkpoint["visual_critic"])
    action_critic.load_state_dict(checkpoint["action_critic"])
    visual_critic.eval()
    action_critic.eval()

    caches = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(args.cache_dir.glob("task_*.pt"))]
    train = [episode for episode in caches if str(episode.get("split")) == "train"]
    validation = [episode for episode in caches if str(episode.get("split")) == "validation"]
    if not train or not validation:
        raise RuntimeError("Both cached train and validation episodes are required.")

    neutral_abs_scores: list[torch.Tensor] = []
    train_similarity: list[torch.Tensor] = []
    for episode in train:
        score, similarity = _score_episode(
            episode,
            visual_critic,
            action_critic,
            device=device,
            gamma=gamma,
            beta=beta,
        )
        direction = episode["direction"].cpu()
        neutral = direction == 0
        if bool(neutral.any()):
            neutral_abs_scores.append(score[neutral].abs())
        train_similarity.append(similarity)
    if neutral_abs_scores:
        deadband_source = torch.cat(neutral_abs_scores)
    else:
        deadband_source = torch.cat([
            _score_episode(ep, visual_critic, action_critic, device=device, gamma=gamma, beta=beta)[0].abs()
            for ep in train
        ])
    deadband = float(torch.quantile(deadband_source, float(args.neutral_quantile)).item())
    alias_threshold = float(torch.quantile(torch.cat(train_similarity), float(args.alias_quantile)).item())

    truth_all: list[str] = []
    d_only_all: list[str] = []
    weak_all: list[str] = []
    full_all: list[str] = []
    event_veto_all: list[str] = []
    relation_veto_all: list[str] = []
    alias_all: list[bool] = []
    episode_rows: list[dict[str, object]] = []
    root = Path.cwd().resolve()

    for episode in validation:
        score, similarity = _score_episode(
            episode,
            visual_critic,
            action_critic,
            device=device,
            gamma=gamma,
            beta=beta,
        )
        direction = episode["direction"].cpu()
        frames = _load_physical_frames(root, args.suite, int(episode["task_id"]), str(episode["demo_name"]))
        evidences = [PhysicalConsistencyEvidence.from_frames(left, right) for left, right in zip(frames[:-1], frames[1:])]
        length = min(len(score), len(direction), len(evidences), len(similarity))

        truth = [_truth_label(int(value)) for value in direction[:length].tolist()]
        d_only = [_d_only_label(float(value), deadband) for value in score[:length].tolist()]
        weak = [
            project_teacher_label(float(value), evidence, deadband=deadband, use_relation_direction=False).label
            for value, evidence in zip(score[:length].tolist(), evidences[:length])
        ]
        full = [
            project_teacher_label(float(value), evidence, deadband=deadband, use_relation_direction=True).label
            for value, evidence in zip(score[:length].tolist(), evidences[:length])
        ]
        event_veto = [
            project_teacher_label_veto(
                float(value), evidence, deadband=deadband, use_relation_direction=False
            ).label
            for value, evidence in zip(score[:length].tolist(), evidences[:length])
        ]
        relation_veto = [
            project_teacher_label_veto(
                float(value), evidence, deadband=deadband, use_relation_direction=True
            ).label
            for value, evidence in zip(score[:length].tolist(), evidences[:length])
        ]
        alias = [bool(value >= alias_threshold) for value in similarity[:length].tolist()]

        d_only_fp = [i for i, (t, p) in enumerate(zip(truth, d_only)) if t != P_LABEL and p == P_LABEL]
        row = {
            "task_id": int(episode["task_id"]),
            "demo_name": str(episode["demo_name"]),
            "D_only": _metrics(truth, d_only, alias),
            "weak_physical": _metrics(truth, weak, alias),
            "full_physical": _metrics(truth, full, alias),
            "event_veto": _metrics(truth, event_veto, alias),
            "relation_veto": _metrics(truth, relation_veto, alias),
            "D_only_false_positive_candidates": len(d_only_fp),
            "weak_corrected_fraction": None if not d_only_fp else sum(weak[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "full_corrected_fraction": None if not d_only_fp else sum(full[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "event_veto_corrected_fraction": None if not d_only_fp else sum(event_veto[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "relation_veto_corrected_fraction": None if not d_only_fp else sum(relation_veto[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
        }
        episode_rows.append(row)
        truth_all.extend(truth)
        d_only_all.extend(d_only)
        weak_all.extend(weak)
        full_all.extend(full)
        event_veto_all.extend(event_veto)
        relation_veto_all.extend(relation_veto)
        alias_all.extend(alias)

    d_only_fp = [i for i, (t, p) in enumerate(zip(truth_all, d_only_all)) if t != P_LABEL and p == P_LABEL]
    report = {
        "pipeline": "v3_information_plus_privileged_consistency",
        "beta": beta,
        "gamma": gamma,
        "deadband": deadband,
        "deadband_calibration": {
            "source": "train-neutral |D| quantile",
            "quantile": float(args.neutral_quantile),
        },
        "visual_alias_threshold": alias_threshold,
        "visual_alias_calibration": {
            "source": "train consecutive dual-view cosine-similarity quantile",
            "quantile": float(args.alias_quantile),
        },
        "aggregate": {
            "D_only": _metrics(truth_all, d_only_all, alias_all),
            "weak_physical": _metrics(truth_all, weak_all, alias_all),
            "full_physical": _metrics(truth_all, full_all, alias_all),
            "event_veto": _metrics(truth_all, event_veto_all, alias_all),
            "relation_veto": _metrics(truth_all, relation_veto_all, alias_all),
            "D_only_false_positive_candidates": len(d_only_fp),
            "weak_corrected_fraction": None if not d_only_fp else sum(weak_all[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "full_corrected_fraction": None if not d_only_fp else sum(full_all[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "event_veto_corrected_fraction": None if not d_only_fp else sum(event_veto_all[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
            "relation_veto_corrected_fraction": None if not d_only_fp else sum(relation_veto_all[i] != P_LABEL for i in d_only_fp) / len(d_only_fp),
        },
        "episodes": episode_rows,
        "note": "full_physical uses privileged relation direction also present in physical direction labels; treat as pipeline sanity, not independent paper evidence",
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

