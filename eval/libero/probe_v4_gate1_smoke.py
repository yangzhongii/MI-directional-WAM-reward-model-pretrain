"""Small, auditable Pipeline-v4 Gate-1 feasibility experiment.

The experiment uses frozen pooled LaWAM features from Pipeline v3 but never
loads or updates the v3 teacher.  For each of five LIBERO-Spatial tasks:

* demo_0 is a fixed task-level reference episode;
* demo_1 trains small visual heads;
* demo_10 is a held-out test episode.

It compares a cosine difference, a goal-aware direct direction classifier, a
no-goal direct classifier, and a goal-aware scalar potential.  Goal-aware
heads are also evaluated with masked, task-permuted, and initial-state goals.
This is a smoke experiment because the labels are the existing physics
heuristic and only one test initial state is available per task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LABEL_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_LABEL = torch.tensor([-1, 0, 1], dtype=torch.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/mi_reward/v4_decision_gate/gate1_smoke_v1"),
    )
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--reference-demo", type=int, default=0)
    parser.add_argument("--train-demo", type=int, default=1)
    parser.add_argument("--test-demo", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neutral-weight", type=float, default=0.1)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stamp(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Split:
    z0: torch.Tensor
    z1: torch.Tensor
    y: torch.Tensor
    task: torch.Tensor
    goal: torch.Tensor


class DirectHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_goal: bool) -> None:
        super().__init__()
        self.use_goal = use_goal
        self.state = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        if use_goal:
            self.goal = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            fused_dim = hidden_dim * 6
        else:
            fused_dim = hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3)
        )

    def forward(self, z0: torch.Tensor, z1: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        h0 = self.state(z0)
        h1 = self.state(z1)
        pieces = [h0, h1, h1 - h0]
        if self.use_goal:
            hg = self.goal(goal)
            pieces.extend((hg, h0 * hg, h1 * hg))
        return self.classifier(torch.cat(pieces, dim=-1))


class PotentialHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.goal = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def potential(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        hz = self.state(z)
        hg = self.goal(goal)
        return self.score(torch.cat((hz, hg, hz * hg, torch.abs(hz - hg)), dim=-1)).squeeze(-1)

    def forward(self, z0: torch.Tensor, z1: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.potential(z1, goal) - self.potential(z0, goal)


def load_episode(feature_dir: Path, task: int, demo: int) -> dict[str, Any]:
    path = feature_dir / f"task_{task:02d}_demo_{demo}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    visual = payload["visual"].float()
    direction = payload["direction"].long()
    if visual.ndim != 2 or len(visual) != len(direction) + 1:
        raise ValueError(f"Bad feature alignment in {path}: visual={visual.shape}, direction={direction.shape}")
    return payload


def make_splits(args: argparse.Namespace) -> tuple[Split, Split, dict[int, torch.Tensor], dict[int, torch.Tensor], list[Path]]:
    train_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    test_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    goals: dict[int, torch.Tensor] = {}
    initial_goals: dict[int, torch.Tensor] = {}
    paths: list[Path] = []
    for task in args.task_ids:
        reference = load_episode(args.feature_dir, task, args.reference_demo)
        train = load_episode(args.feature_dir, task, args.train_demo)
        test = load_episode(args.feature_dir, task, args.test_demo)
        paths.extend(
            args.feature_dir / f"task_{task:02d}_demo_{demo}.pt"
            for demo in (args.reference_demo, args.train_demo, args.test_demo)
        )
        if len({reference["demo_name"], train["demo_name"], test["demo_name"]}) != 3:
            raise ValueError(f"Task {task}: reference/train/test episodes must differ")
        goals[task] = reference["goal"].float()
        initial_goals[task] = reference["visual"][:5].float().mean(dim=0)
        for payload, target in ((train, train_parts), (test, test_parts)):
            n = len(payload["direction"])
            target.append(
                (
                    payload["visual"][:-1].float(),
                    payload["visual"][1:].float(),
                    payload["direction"].long(),
                    torch.full((n,), task, dtype=torch.long),
                    goals[task].expand(n, -1),
                )
            )

    def merge(parts: list[tuple[torch.Tensor, ...]]) -> Split:
        columns = [torch.cat([part[index] for part in parts], dim=0) for index in range(5)]
        return Split(*columns)

    return merge(train_parts), merge(test_parts), goals, initial_goals, paths


def normalize(train: Split, test: Split, goals: dict[int, torch.Tensor], initial: dict[int, torch.Tensor]) -> tuple[Split, Split, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    states = torch.cat((train.z0, train.z1), dim=0)
    mean = states.mean(dim=0)
    std = states.std(dim=0).clamp_min(1e-5)

    def norm_split(split: Split) -> Split:
        return Split(
            (split.z0 - mean) / std,
            (split.z1 - mean) / std,
            split.y,
            split.task,
            (split.goal - mean) / std,
        )

    return (
        norm_split(train),
        norm_split(test),
        {task: (goal - mean) / std for task, goal in goals.items()},
        {task: (goal - mean) / std for task, goal in initial.items()},
    )


def goal_tensor(split: Split, lookup: dict[int, torch.Tensor]) -> torch.Tensor:
    return torch.stack([lookup[int(task)] for task in split.task.tolist()], dim=0)


def train_direct(
    split: Split,
    *,
    seed: int,
    hidden_dim: int,
    use_goal: bool,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> DirectHead:
    torch.manual_seed(seed)
    model = DirectHead(split.z0.shape[1], hidden_dim, use_goal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    targets = torch.tensor([LABEL_TO_INDEX[int(value)] for value in split.y], dtype=torch.long)
    counts = torch.bincount(targets, minlength=3).float()
    class_weights = (counts.sum() / (3.0 * counts.clamp_min(1))).to(device)
    generator = torch.Generator().manual_seed(seed + 1000)
    for step in range(steps):
        index = torch.randint(len(targets), (min(batch_size, len(targets)),), generator=generator)
        logits = model(split.z0[index].to(device), split.z1[index].to(device), split.goal[index].to(device))
        loss = F.cross_entropy(logits, targets[index].to(device), weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 100 == 0:
            stamp(f"TRAIN direct use_goal={use_goal} seed={seed} step={step + 1}/{steps} loss={loss.item():.6f}")
    return model.eval().cpu()


def train_potential(
    split: Split,
    *,
    seed: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    neutral_weight: float,
    device: torch.device,
) -> PotentialHead:
    torch.manual_seed(seed + 100_000)
    model = PotentialHead(split.z0.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 2000)
    for step in range(steps):
        index = torch.randint(len(split.y), (min(batch_size, len(split.y)),), generator=generator)
        y = split.y[index].to(device)
        delta = model(split.z0[index].to(device), split.z1[index].to(device), split.goal[index].to(device))
        directional = y != 0
        if not torch.any(directional):
            continue
        binary = (y[directional] > 0).float()
        positive = binary.sum().clamp_min(1.0)
        negative = (1.0 - binary).sum().clamp_min(1.0)
        pos_weight = (negative / positive).detach()
        rank_loss = F.binary_cross_entropy_with_logits(delta[directional], binary, pos_weight=pos_weight)
        neutral_loss = delta[~directional].square().mean() if torch.any(~directional) else delta.new_zeros(())
        loss = rank_loss + neutral_weight * neutral_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 100 == 0:
            stamp(
                f"TRAIN potential seed={seed} step={step + 1}/{steps} "
                f"loss={loss.item():.6f} rank={rank_loss.item():.6f} neutral={neutral_loss.item():.6f}"
            )
    return model.eval().cpu()


@torch.inference_mode()
def direct_outputs(model: DirectHead, split: Split, goal: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = model(split.z0, split.z1, goal)
    direction_score = (logits[:, 2] - logits[:, 0]).numpy()
    labels = INDEX_TO_LABEL[logits.argmax(dim=-1)].numpy()
    return direction_score, labels


@torch.inference_mode()
def potential_outputs(model: PotentialHead, split: Split, goal: torch.Tensor) -> np.ndarray:
    return model(split.z0, split.z1, goal).numpy()


def direction_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    keep = truth != 0
    y = truth[keep]
    score = scores[keep]
    forward = y > 0
    regression = y < 0
    forward_recall = float(np.mean(score[forward] > 0)) if np.any(forward) else None
    regression_recall = float(np.mean(score[regression] < 0)) if np.any(regression) else None
    balanced = None if forward_recall is None or regression_recall is None else (forward_recall + regression_recall) / 2
    return {
        "non_neutral": int(keep.sum()),
        "forward": int(forward.sum()),
        "regression": int(regression.sum()),
        "forward_recall": forward_recall,
        "regression_recall": regression_recall,
        "balanced_accuracy": balanced,
        "strict_direction_accuracy": float(np.mean(np.sign(score) == y)) if len(y) else None,
        "exact_ties": int(np.sum(score == 0)),
    }


def three_class_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    labels = (-1, 0, 1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for target, pred in zip(truth, prediction):
        matrix[LABEL_TO_INDEX[int(target)], LABEL_TO_INDEX[int(pred)]] += 1
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(3):
        tp = float(matrix[index, index])
        fn = float(matrix[index].sum() - tp)
        fp = float(matrix[:, index].sum() - tp)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy_three_class": float(np.mean(recalls)),
        "recall": {str(label): recalls[index] for index, label in enumerate(labels)},
        "confusion_matrix_rows_truth_NUP": matrix.tolist(),
    }


def task_metric(task: np.ndarray, truth: np.ndarray, scores: np.ndarray, sampled: np.ndarray) -> float:
    indices = np.concatenate([np.flatnonzero(task == value) for value in sampled])
    result = direction_metrics(scores[indices], truth[indices])["balanced_accuracy"]
    return float("nan") if result is None else float(result)


def paired_bootstrap(
    task: np.ndarray,
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    tasks = np.unique(task)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        difference = task_metric(task, truth, left, sampled) - task_metric(task, truth, right, sampled)
        if np.isfinite(difference):
            values.append(difference)
    array = np.asarray(values)
    return {
        "draws_retained": len(values),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    if len(set(args.task_ids)) < 2:
        raise ValueError("At least two task clusters are required")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    stamp(f"START device={device} tasks={args.task_ids} seeds={args.seeds}")
    train_raw, test_raw, goals_raw, initial_raw, input_paths = make_splits(args)
    train, test, goals, initial_goals = normalize(train_raw, test_raw, goals_raw, initial_raw)
    correct_goal = goal_tensor(test, goals)
    initial_goal = goal_tensor(test, initial_goals)
    tasks = sorted(args.task_ids)
    permuted_lookup = {task: goals[tasks[(index + 1) % len(tasks)]] for index, task in enumerate(tasks)}
    permuted_goal = goal_tensor(test, permuted_lookup)
    masked_goal = torch.zeros_like(correct_goal)
    stamp(
        f"DATA train={len(train.y)} test={len(test.y)} "
        f"train_counts={dict(zip(*np.unique(train.y.numpy(), return_counts=True)))} "
        f"test_counts={dict(zip(*np.unique(test.y.numpy(), return_counts=True)))}"
    )

    raw_correct = torch.stack([goals_raw[int(task)] for task in test_raw.task.tolist()])
    cosine = (
        F.cosine_similarity(test_raw.z1, raw_correct, dim=-1)
        - F.cosine_similarity(test_raw.z0, raw_correct, dim=-1)
    ).numpy()
    outputs: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "direct_correct",
            "direct_masked",
            "direct_permuted",
            "direct_initial_goal",
            "direct_no_goal",
            "potential_correct",
            "potential_masked",
            "potential_permuted",
            "potential_initial_goal",
        )
    }
    direct_classes: list[np.ndarray] = []
    checkpoints: list[str] = []
    for seed in args.seeds:
        direct = train_direct(
            train,
            seed=seed,
            hidden_dim=args.hidden_dim,
            use_goal=True,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        no_goal = train_direct(
            train,
            seed=seed,
            hidden_dim=args.hidden_dim,
            use_goal=False,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        potential = train_potential(
            train,
            seed=seed,
            hidden_dim=args.hidden_dim,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            neutral_weight=args.neutral_weight,
            device=device,
        )
        for name, goal in (
            ("direct_correct", correct_goal),
            ("direct_masked", masked_goal),
            ("direct_permuted", permuted_goal),
            ("direct_initial_goal", initial_goal),
        ):
            score, classes = direct_outputs(direct, test, goal)
            outputs[name].append(score)
            if name == "direct_correct":
                direct_classes.append(classes)
        outputs["direct_no_goal"].append(direct_outputs(no_goal, test, masked_goal)[0])
        for name, goal in (
            ("potential_correct", correct_goal),
            ("potential_masked", masked_goal),
            ("potential_permuted", permuted_goal),
            ("potential_initial_goal", initial_goal),
        ):
            outputs[name].append(potential_outputs(potential, test, goal))
        checkpoint_path = args.output_dir / f"heads_seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "direct": direct.state_dict(),
                "direct_no_goal": no_goal.state_dict(),
                "potential": potential.state_dict(),
                "input_dim": train.z0.shape[1],
                "hidden_dim": args.hidden_dim,
            },
            checkpoint_path,
        )
        checkpoints.append(str(checkpoint_path))

    ensemble = {name: np.mean(np.stack(values), axis=0) for name, values in outputs.items()}
    truth = test.y.numpy()
    task = test.task.numpy()
    # direct_classes already contains semantic labels {-1, 0, +1}; the mode
    # must not be passed through INDEX_TO_LABEL a second time.
    ensemble_classes = torch.mode(torch.tensor(np.stack(direct_classes)), dim=0).values.numpy()
    metrics = {"cosine": direction_metrics(cosine, truth)}
    metrics.update({name: direction_metrics(scores, truth) for name, scores in ensemble.items()})
    metrics["direct_correct_three_class"] = three_class_metrics(ensemble_classes, truth)
    comparisons = {
        "direct_correct_minus_cosine": paired_bootstrap(
            task, truth, ensemble["direct_correct"], cosine, draws=args.bootstrap_draws, seed=71
        ),
        "potential_correct_minus_cosine": paired_bootstrap(
            task, truth, ensemble["potential_correct"], cosine, draws=args.bootstrap_draws, seed=72
        ),
        "direct_correct_minus_no_goal": paired_bootstrap(
            task, truth, ensemble["direct_correct"], ensemble["direct_no_goal"], draws=args.bootstrap_draws, seed=73
        ),
        "direct_correct_minus_masked": paired_bootstrap(
            task, truth, ensemble["direct_correct"], ensemble["direct_masked"], draws=args.bootstrap_draws, seed=74
        ),
        "direct_correct_minus_initial_goal": paired_bootstrap(
            task, truth, ensemble["direct_correct"], ensemble["direct_initial_goal"], draws=args.bootstrap_draws, seed=75
        ),
        "potential_correct_minus_masked": paired_bootstrap(
            task, truth, ensemble["potential_correct"], ensemble["potential_masked"], draws=args.bootstrap_draws, seed=751
        ),
        "potential_correct_minus_permuted": paired_bootstrap(
            task, truth, ensemble["potential_correct"], ensemble["potential_permuted"], draws=args.bootstrap_draws, seed=752
        ),
        "potential_correct_minus_initial_goal": paired_bootstrap(
            task, truth, ensemble["potential_correct"], ensemble["potential_initial_goal"], draws=args.bootstrap_draws, seed=753
        ),
        "direct_correct_minus_potential_correct": paired_bootstrap(
            task, truth, ensemble["direct_correct"], ensemble["potential_correct"], draws=args.bootstrap_draws, seed=76
        ),
    }
    input_hashes = {str(path): sha256(path) for path in input_paths}
    result = {
        "status": "SMOKE_ONLY",
        "protocol": {
            "suite": "libero_spatial",
            "task_ids": args.task_ids,
            "reference_demo": args.reference_demo,
            "train_demo": args.train_demo,
            "test_demo": args.test_demo,
            "reference_definition": "demo_0 successful terminal goal vector; initial hard goal is demo_0 first-five-state mean",
            "episode_overlap": False,
            "seeds": args.seeds,
            "steps": args.steps,
            "hidden_dim": args.hidden_dim,
            "bootstrap": f"{args.bootstrap_draws} task-cluster draws",
            "direction_metric": "P/N only; strict sign, ties wrong; no threshold calibration",
            "warning": (
                "Existing physics-heuristic labels, five task clusters, one test initial state per task, pooled features, "
                "and no set-valued reference. This cannot pass formal Pipeline-v4 Gate 1."
            ),
            "input_sha256": input_hashes,
            "saved_checkpoints": checkpoints,
        },
        "counts": {
            "train": {str(k): int(v) for k, v in zip(*np.unique(train.y.numpy(), return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(truth, return_counts=True))},
        },
        "metrics": metrics,
        "paired_task_cluster_bootstrap": comparisons,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        truth=truth,
        task=task,
        cosine=cosine,
        **ensemble,
    )
    table_names = (
        "cosine",
        "direct_correct",
        "direct_no_goal",
        "direct_masked",
        "direct_permuted",
        "direct_initial_goal",
        "potential_correct",
        "potential_masked",
        "potential_permuted",
        "potential_initial_goal",
    )
    summary = [
        "# Pipeline-v4 Gate-1 smoke",
        "",
        "Status: **SMOKE_ONLY**. This result cannot pass formal Gate 1 because it has one test initial state per task and heuristic labels.",
        "",
        "| Method / control | Balanced accuracy | Forward recall | Regression recall |",
        "|---|---:|---:|---:|",
    ]
    for name in table_names:
        metric = metrics[name]
        summary.append(
            f"| `{name}` | {metric['balanced_accuracy']:.4f} | {metric['forward_recall']:.4f} | {metric['regression_recall']:.4f} |"
        )
    summary.extend(("", "## Paired task-cluster differences", ""))
    for name, value in comparisons.items():
        summary.append(
            f"- `{name}`: mean={value['mean']:.4f}, 95% CI [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"
        )
    summary.extend(
        (
            "",
            "The direct model is goal-grounded only if correct goals outperform the no-goal and hard wrong-goal controls. Cross-task permutation is an identity control, because LIBERO-Spatial tasks share a similar final relation.",
            "",
            "Formal Gate 1 additionally requires independently reviewed challenge labels, at least three initial states per task, set-valued references, and a paired CI lower bound above zero versus cosine.",
            "",
        )
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    stamp("RESULTS")
    print("\n".join(summary), flush=True)
    stamp("COMPLETE")


if __name__ == "__main__":
    main()
