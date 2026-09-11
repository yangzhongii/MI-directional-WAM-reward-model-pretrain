"""Pipeline-v4 Experiment C: action, transition-latent, and state-pair probes.

Each variant uses the same small classifier: a current visual-state encoder, an
auxiliary-evidence encoder, and a common P/U/N head.  Only the auxiliary input
changes: executed simulator action, frozen LaWAM transition latent, or explicit
visual state difference.  Training uses demo_0/1; demo_10 and three held-out
rollout initial states are evaluation only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit_teacher import live_trace_physics
from eval_v4_gate1_multistate import collect_records, paired_bootstrap
from probe_v4_gate1_smoke import LABEL_TO_INDEX, INDEX_TO_LABEL, direction_metrics, three_class_metrics
from mi_reward.data.libero_privileged import classify_physical_transition, resolve_libero_task
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor


VARIANTS = ("executed_action", "transition_latent", "state_pair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("logs/mi_reward/v3_teacher_mainline/features"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-demos", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--test-demo", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--windows-per-trajectory", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stamp(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


@dataclass
class Split:
    z0: torch.Tensor
    z1: torch.Tensor
    executed_action: torch.Tensor
    transition_latent: torch.Tensor
    y: torch.Tensor
    task: torch.Tensor


class EvidenceProbe(nn.Module):
    def __init__(self, state_dim: int, evidence_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.evidence = nn.Sequential(nn.Linear(evidence_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, z0: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((self.state(z0), self.evidence(evidence)), dim=-1))


def raw_actions(task: int, demo: int, count: int) -> torch.Tensor:
    path, _, _ = resolve_libero_task(Path.cwd(), "libero_spatial", task)
    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle["data"][f"demo_{demo}"]["actions"], dtype=np.float32)
    if len(actions) < count:
        raise ValueError(f"task={task} demo={demo}: actions={len(actions)} < transitions={count}")
    return torch.from_numpy(actions[:count])


def load_cached_split(args: argparse.Namespace, demos: list[int]) -> Split:
    parts = []
    for task in args.task_ids:
        for demo in demos:
            path = args.feature_dir / f"task_{task:02d}_demo_{demo}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            visual = payload["visual"].float()
            direction = payload["direction"].long()
            latent = payload["action"].float()
            if len(visual) != len(direction) + 1 or len(latent) != len(direction):
                raise ValueError(f"Cached alignment failed: {path}")
            count = len(direction)
            parts.append((
                visual[:-1], visual[1:], raw_actions(task, demo, count), latent, direction,
                torch.full((count,), task, dtype=torch.long),
            ))
    columns = [torch.cat([part[index] for part in parts], dim=0) for index in range(6)]
    return Split(*columns)


def fit_normalizers(train: Split) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    def fit(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.mean(dim=0), value.std(dim=0).clamp_min(1e-5)
    return {
        "state": fit(torch.cat((train.z0, train.z1), dim=0)),
        "executed_action": fit(train.executed_action),
        "transition_latent": fit(train.transition_latent),
        "state_pair": fit(train.z1 - train.z0),
    }


def normalize(split: Split, norms: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state_mean, state_std = norms["state"]
    z0 = (split.z0 - state_mean) / state_std
    evidence = {
        "executed_action": (split.executed_action - norms["executed_action"][0]) / norms["executed_action"][1],
        "transition_latent": (split.transition_latent - norms["transition_latent"][0]) / norms["transition_latent"][1],
        "state_pair": ((split.z1 - split.z0) - norms["state_pair"][0]) / norms["state_pair"][1],
    }
    return z0, evidence


def train_probe(
    z0: torch.Tensor, evidence: torch.Tensor, y: torch.Tensor, *, seed: int, hidden_dim: int,
    steps: int, batch_size: int, lr: float, weight_decay: float, device: torch.device,
) -> EvidenceProbe:
    torch.manual_seed(seed)
    model = EvidenceProbe(z0.shape[1], evidence.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    targets = torch.tensor([LABEL_TO_INDEX[int(value)] for value in y], dtype=torch.long)
    counts = torch.bincount(targets, minlength=3).float()
    weights = (counts.sum() / (3.0 * counts.clamp_min(1))).to(device)
    generator = torch.Generator().manual_seed(seed + 3000)
    for step in range(steps):
        index = torch.randint(len(y), (min(batch_size, len(y)),), generator=generator)
        logits = model(z0[index].to(device), evidence[index].to(device))
        loss = F.cross_entropy(logits, targets[index].to(device), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 100 == 0:
            stamp(f"TRAIN seed={seed} evidence_dim={evidence.shape[1]} step={step + 1}/{steps} loss={loss.item():.6f}")
    return model.eval().cpu()


@torch.inference_mode()
def ensemble_scores(models: dict[str, list[EvidenceProbe]], z0: torch.Tensor, evidence: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores, classes = {}, {}
    for variant in VARIANTS:
        logits = torch.stack([model(z0, evidence[variant]) for model in models[variant]])
        mean_logits = logits.mean(dim=0)
        scores[variant] = (mean_logits[:, 2] - mean_logits[:, 0]).numpy()
        classes[variant] = INDEX_TO_LABEL[mean_logits.argmax(dim=-1)].numpy()
    return scores, classes


def cached_bootstrap(split: Split, truth: np.ndarray, left: np.ndarray, right: np.ndarray, draws: int, seed: int) -> dict[str, float | int | None]:
    tasks = np.unique(split.task.numpy())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        index = np.concatenate([np.flatnonzero(split.task.numpy() == task) for task in sampled])
        a = direction_metrics(left[index], truth[index])["balanced_accuracy"]
        b = direction_metrics(right[index], truth[index])["balanced_accuracy"]
        if a is not None and b is not None:
            values.append(float(a - b))
    if not values:
        return {"draws_retained": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    values = np.asarray(values)
    return {"draws_retained": len(values), "mean": float(values.mean()), "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975))}


def rollout_split(args: argparse.Namespace, extractor: LaWAMLAMFeatureExtractor) -> tuple[Split, list[dict[str, Any]]]:
    records = collect_records(args.rollout_manifests, set(args.task_ids), args.windows_per_trajectory)
    grouped: dict[tuple[Path, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["manifest"], record["trajectory"]["trajectory_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for (manifest, trajectory_id), group in grouped.items():
        trajectory = group[0]["trajectory"]
        task = int(trajectory["task_index"])
        _, bddl, language = resolve_libero_task(Path.cwd(), "libero_spatial", task)
        selected = sorted({frame for record in group for frame in record["window"]["row"]["provenance"]["history_frame_indices"][-2:]})
        trace = np.load(manifest.parent / trajectory["trace"])
        physical, checks = live_trace_physics(trajectory, trace, selected, bddl)
        if any(not checks[index]["consistent"] for index in selected):
            raise RuntimeError(f"Replay mismatch: {trajectory_id}")
        views = []
        for view in ("agentview", "wrist"):
            paths = [str(manifest.parent / trajectory[view][frame]) for frame in selected]
            views.append(extractor.extract_trajectory_tokens(paths, language).float().mean(dim=1))
        state_by_frame = dict(zip(selected, torch.cat(views, dim=-1)))
        for record in group:
            window = record["window"]
            left, right = window["row"]["provenance"]["history_frame_indices"][-2:]
            latent = extractor.extract_action_latents(
                [str(manifest.parent / trajectory["agentview"][left]), str(manifest.parent / trajectory["agentview"][right])], language
            ).float().mean(dim=1)[0]
            rows.append({
                "z0": state_by_frame[left], "z1": state_by_frame[right],
                "executed_action": torch.tensor(trace["actions"][left], dtype=torch.float32),
                "transition_latent": latent, "direction": classify_physical_transition(physical[left], physical[right]),
                "task": task, "initial_state_cluster": trajectory["pair_id"], "trajectory_id": trajectory_id,
                "source_demo": trajectory["demo_name"], "policy_id": trajectory["policy_id"],
                "window_end": int(window["window_end"]), "images": window["row"]["images"],
            })
        stamp(f"ROLLOUT_FEATURES {trajectory_id} windows={len(group)} total={len(rows)}")
    split = Split(
        torch.stack([row["z0"] for row in rows]), torch.stack([row["z1"] for row in rows]),
        torch.stack([row["executed_action"] for row in rows]), torch.stack([row["transition_latent"] for row in rows]),
        torch.tensor([row["direction"] for row in rows], dtype=torch.long), torch.tensor([row["task"] for row in rows], dtype=torch.long),
    )
    return split, rows


def comparison_set(rows: list[dict[str, Any]], truth: np.ndarray, scores: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, Any]:
    return {
        "transition_minus_executed": paired_bootstrap(rows, truth, scores["transition_latent"], scores["executed_action"], draws, seed),
        "state_pair_minus_executed": paired_bootstrap(rows, truth, scores["state_pair"], scores["executed_action"], draws, seed + 1),
        "transition_minus_state_pair": paired_bootstrap(rows, truth, scores["transition_latent"], scores["state_pair"], draws, seed + 2),
    }


def write_summary(path: Path, title: str, metrics: dict[str, Any], comparisons: dict[str, Any], note: str) -> None:
    lines = [f"# {title}", "", "| Evidence | Balanced accuracy | Forward recall | Regression recall |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        metric = metrics[variant]
        lines.append(f"| `{variant}` | {metric['balanced_accuracy']:.4f} | {metric['forward_recall']:.4f} | {metric['regression_recall']:.4f} |")
    lines.extend(("", "## Paired differences", ""))
    for key, value in comparisons.items():
        lines.append(f"- `{key}`: mean={value['mean']:.4f}, 95% CI [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]")
    lines.extend(("", note, ""))
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train = load_cached_split(args, args.train_demos)
    cached_test = load_cached_split(args, [args.test_demo])
    norms = fit_normalizers(train)
    train_z0, train_evidence = normalize(train, norms)
    test_z0, test_evidence = normalize(cached_test, norms)
    models: dict[str, list[EvidenceProbe]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in args.seeds:
            models[variant].append(train_probe(
                train_z0, train_evidence[variant], train.y, seed=seed, hidden_dim=args.hidden_dim,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, device=device,
            ))
    torch.save({"normalizers": norms, "task_ids": args.task_ids, "train_demos": args.train_demos, "test_demo": args.test_demo}, args.output_dir / "normalizers.pt")
    for variant in VARIANTS:
        for seed, model in zip(args.seeds, models[variant]):
            torch.save({"variant": variant, "seed": seed, "state_dict": model.state_dict(), "state_dim": train_z0.shape[1], "evidence_dim": train_evidence[variant].shape[1], "hidden_dim": args.hidden_dim}, args.output_dir / f"{variant}_seed_{seed}.pt")
    cached_scores, cached_classes = ensemble_scores(models, test_z0, test_evidence)
    cached_truth = cached_test.y.numpy()
    cached_metrics = {variant: direction_metrics(cached_scores[variant], cached_truth) for variant in VARIANTS}
    cached_three_class = {variant: three_class_metrics(cached_classes[variant], cached_truth) for variant in VARIANTS}
    cached_comparisons = {
        "transition_minus_executed": cached_bootstrap(cached_test, cached_truth, cached_scores["transition_latent"], cached_scores["executed_action"], args.bootstrap_draws, args.seed),
        "state_pair_minus_executed": cached_bootstrap(cached_test, cached_truth, cached_scores["state_pair"], cached_scores["executed_action"], args.bootstrap_draws, args.seed + 1),
        "transition_minus_state_pair": cached_bootstrap(cached_test, cached_truth, cached_scores["transition_latent"], cached_scores["state_pair"], args.bootstrap_draws, args.seed + 2),
    }
    write_summary(args.output_dir / "cached_summary.md", "Pipeline-v4 Experiment C cached held-out episodes", cached_metrics, cached_comparisons, "Test episodes are demo_10; labels remain the existing physics heuristic.")
    stamp("CACHED_RESULTS_COMPLETE")
    extractor = LaWAMLAMFeatureExtractor(lam_config_path=args.lam_config, lam_ckpt_path=args.lam_checkpoint, vision_model_id=args.dino_model, device=args.device, strict=True)
    stamp(f"LAM loaded={extractor.using_lam}")
    rollout, rows = rollout_split(args, extractor)
    rollout_z0, rollout_evidence = normalize(rollout, norms)
    rollout_scores, rollout_classes = ensemble_scores(models, rollout_z0, rollout_evidence)
    rollout_truth = rollout.y.numpy()
    rollout_metrics = {variant: direction_metrics(rollout_scores[variant], rollout_truth) for variant in VARIANTS}
    rollout_three_class = {variant: three_class_metrics(rollout_classes[variant], rollout_truth) for variant in VARIANTS}
    rollout_comparisons = comparison_set(rows, rollout_truth, rollout_scores, args.bootstrap_draws, args.seed + 10)
    write_summary(args.output_dir / "SUMMARY.md", "Pipeline-v4 Experiment C multi-state transition evidence", rollout_metrics, rollout_comparisons, "Frozen heads evaluated on three initial states/task; labels are reconstructed physics heuristics.")
    with (args.output_dir / "window_scores.jsonl").open("x") as handle:
        for index, row in enumerate(rows):
            public = {key: value for key, value in row.items() if key not in {"z0", "z1", "executed_action", "transition_latent"}}
            public.update({variant: float(score[index]) for variant, score in rollout_scores.items()})
            handle.write(json.dumps(public) + "\n")
    result = {
        "status": "EXPERIMENT_C_HEURISTIC_VALIDATION",
        "protocol": {"train_demos": args.train_demos, "test_demo": args.test_demo, "tasks": args.task_ids, "seeds": args.seeds, "steps": args.steps, "heads_retrained_on_rollouts": False, "rollout_initial_state_clusters": len({row['initial_state_cluster'] for row in rows}), "rollout_windows": len(rows), "warning": "Physics-heuristic labels are not independent review."},
        "cached": {"metrics": cached_metrics, "three_class": cached_three_class, "comparisons": cached_comparisons},
        "rollout": {"metrics": rollout_metrics, "three_class": rollout_three_class, "comparisons": rollout_comparisons},
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "scores.npz", cached_truth=cached_truth, rollout_truth=rollout_truth, **{f"cached_{key}": value for key, value in cached_scores.items()}, **{f"rollout_{key}": value for key, value in rollout_scores.items()})
    stamp("RESULTS")
    print((args.output_dir / "SUMMARY.md").read_text(), flush=True)
    stamp("COMPLETE")


if __name__ == "__main__":
    main()
