"""Evaluate frozen v4 Gate-1 smoke heads on multi-initial-state rollouts.

No head is trained or selected here.  The script rebuilds the original
training-set normalization, loads the three frozen smoke checkpoints, samples
eight windows from each matched-controller rollout, replays the recorded
actions to recover physical evidence, and evaluates paired goal controls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_teacher import live_trace_physics
from libero_adapter import iter_manifest
from probe_v4_gate1_smoke import DirectHead, PotentialHead, direction_metrics
from mi_reward.data.libero_privileged import classify_physical_transition, resolve_libero_task
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-manifests", type=Path, nargs="+", required=True,
        help="Three rollout manifest.jsonl files from distinct source demos.",
    )
    parser.add_argument(
        "--head-dir", type=Path,
        default=Path("logs/mi_reward/v4_decision_gate/gate1_smoke_v2"),
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/features"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--windows-per-trajectory", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def stamp(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def load_training_normalization(feature_dir: Path, tasks: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    for task in tasks:
        payload = torch.load(
            feature_dir / f"task_{task:02d}_demo_1.pt", map_location="cpu", weights_only=False
        )
        visual = payload["visual"].float()
        states.extend((visual[:-1], visual[1:]))
    stacked = torch.cat(states)
    return stacked.mean(dim=0), stacked.std(dim=0).clamp_min(1e-5)


def load_goals(
    feature_dir: Path, tasks: list[int], mean: torch.Tensor, std: torch.Tensor
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    raw: dict[int, torch.Tensor] = {}
    correct: dict[int, torch.Tensor] = {}
    initial: dict[int, torch.Tensor] = {}
    for task in tasks:
        payload = torch.load(
            feature_dir / f"task_{task:02d}_demo_0.pt", map_location="cpu", weights_only=False
        )
        raw[task] = payload["goal"].float()
        correct[task] = (raw[task] - mean) / std
        initial_raw = payload["visual"][:5].float().mean(dim=0)
        initial[task] = (initial_raw - mean) / std
    return raw, correct, initial


def load_heads(head_dir: Path) -> list[tuple[int, DirectHead, DirectHead, PotentialHead]]:
    models = []
    for path in sorted(head_dir.glob("heads_seed_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        direct = DirectHead(payload["input_dim"], payload["hidden_dim"], True).eval()
        no_goal = DirectHead(payload["input_dim"], payload["hidden_dim"], False).eval()
        potential = PotentialHead(payload["input_dim"], payload["hidden_dim"]).eval()
        direct.load_state_dict(payload["direct"])
        no_goal.load_state_dict(payload["direct_no_goal"])
        potential.load_state_dict(payload["potential"])
        models.append((int(payload["seed"]), direct, no_goal, potential))
    if not models:
        raise FileNotFoundError(f"No frozen head checkpoints in {head_dir}")
    return models


def collect_records(manifests: list[Path], task_ids: set[int], windows: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        raw_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        raw_by_id = {row["trajectory_id"]: row for row in raw_rows if int(row["task_index"]) in task_ids}
        for window in iter_manifest(manifest, "libero_spatial", history_frames=5, windows_per_trajectory=windows):
            trajectory = raw_by_id.get(window["trajectory_id"])
            if trajectory is None:
                continue
            identity = f"{window['trajectory_id']}:{window['window_end']}"
            if identity in seen:
                raise ValueError(f"Duplicate rollout window: {identity}")
            seen.add(identity)
            records.append({"manifest": manifest, "trajectory": trajectory, "window": window})
    return records


def extract_rollout_rows(
    records: list[dict[str, Any]], extractor: LaWAMLAMFeatureExtractor
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Path, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["manifest"], record["trajectory"]["trajectory_id"])
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for (manifest, trajectory_id), group in grouped.items():
        trajectory = group[0]["trajectory"]
        task = int(trajectory["task_index"])
        _, bddl, language = resolve_libero_task(Path.cwd(), "libero_spatial", task)
        selected = sorted(
            {
                frame
                for record in group
                for frame in record["window"]["row"]["provenance"]["history_frame_indices"][-2:]
            }
        )
        trace = np.load(manifest.parent / trajectory["trace"])
        physical, checks = live_trace_physics(trajectory, trace, selected, bddl)
        if any(not checks[index]["consistent"] for index in selected):
            bad = {index: checks[index] for index in selected if not checks[index]["consistent"]}
            raise RuntimeError(f"Replay mismatch for {trajectory_id}: {bad}")
        view_features = []
        for view in ("agentview", "wrist"):
            paths = [str(manifest.parent / trajectory[view][index]) for index in selected]
            tokens = extractor.extract_trajectory_tokens(paths, language).float()
            view_features.append(tokens.mean(dim=1))
        feature_by_frame = dict(zip(selected, torch.cat(view_features, dim=-1)))
        for record in group:
            window = record["window"]
            history = window["row"]["provenance"]["history_frame_indices"]
            left, right = history[-2:]
            direction = classify_physical_transition(physical[left], physical[right])
            rows.append(
                {
                    "task": task,
                    "initial_state_cluster": trajectory["pair_id"],
                    "trajectory_id": trajectory_id,
                    "source_demo": trajectory["demo_name"],
                    "policy_id": trajectory["policy_id"],
                    "episode_success": bool(trajectory["success"]),
                    "window_end": int(window["window_end"]),
                    "frame_t": int(left),
                    "frame_t1": int(right),
                    "direction": int(direction),
                    "z0": feature_by_frame[left],
                    "z1": feature_by_frame[right],
                    "physical_t": physical[left],
                    "physical_t1": physical[right],
                    "images": window["row"]["images"],
                }
            )
        stamp(f"FEATURES {trajectory_id} windows={len(group)} total={len(rows)}")
    return rows


@torch.inference_mode()
def score_rows(
    rows: list[dict[str, Any]],
    heads: list[tuple[int, DirectHead, DirectHead, PotentialHead]],
    mean: torch.Tensor,
    std: torch.Tensor,
    raw_goals: dict[int, torch.Tensor],
    goals: dict[int, torch.Tensor],
    initial_goals: dict[int, torch.Tensor],
) -> dict[str, np.ndarray]:
    z0_raw = torch.stack([row["z0"] for row in rows])
    z1_raw = torch.stack([row["z1"] for row in rows])
    z0 = (z0_raw - mean) / std
    z1 = (z1_raw - mean) / std
    correct = torch.stack([goals[row["task"]] for row in rows])
    initial = torch.stack([initial_goals[row["task"]] for row in rows])
    masked = torch.zeros_like(correct)
    tasks = sorted(goals)
    permuted_lookup = {task: goals[tasks[(index + 1) % len(tasks)]] for index, task in enumerate(tasks)}
    permuted = torch.stack([permuted_lookup[row["task"]] for row in rows])
    raw_correct = torch.stack([raw_goals[row["task"]] for row in rows])
    scores: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed, direct, no_goal, potential in heads:
        direct_correct = direct(z0, z1, correct)
        scores["direct_correct"].append((direct_correct[:, 2] - direct_correct[:, 0]).numpy())
        scores["direct_no_goal"].append(
            (lambda logits: (logits[:, 2] - logits[:, 0]).numpy())(no_goal(z0, z1, masked))
        )
        for name, goal in (
            ("potential_correct", correct),
            ("potential_masked", masked),
            ("potential_permuted", permuted),
            ("potential_initial_goal", initial),
        ):
            scores[name].append(potential(z0, z1, goal).numpy())
        stamp(f"SCORED frozen seed={seed}")
    result = {name: np.mean(np.stack(values), axis=0) for name, values in scores.items()}
    result["cosine"] = (
        F.cosine_similarity(z1_raw, raw_correct, dim=-1)
        - F.cosine_similarity(z0_raw, raw_correct, dim=-1)
    ).numpy()
    return result


def hierarchical_indices(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> np.ndarray:
    tasks = sorted({row["task"] for row in rows})
    by_task: dict[int, list[str]] = {
        task: sorted({row["initial_state_cluster"] for row in rows if row["task"] == task})
        for task in tasks
    }
    sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
    indices: list[int] = []
    for task in sampled_tasks:
        clusters = by_task[int(task)]
        for cluster in rng.choice(clusters, size=len(clusters), replace=True):
            indices.extend(
                index
                for index, row in enumerate(rows)
                if row["task"] == int(task) and row["initial_state_cluster"] == cluster
            )
    return np.asarray(indices, dtype=np.int64)


def paired_bootstrap(
    rows: list[dict[str, Any]], truth: np.ndarray, left: np.ndarray, right: np.ndarray, draws: int, seed: int
) -> dict[str, float | int | None]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = hierarchical_indices(rows, rng)
        left_metric = direction_metrics(left[index], truth[index])["balanced_accuracy"]
        right_metric = direction_metrics(right[index], truth[index])["balanced_accuracy"]
        if left_metric is not None and right_metric is not None:
            values.append(float(left_metric - right_metric))
    if not values:
        return {
            "draws_retained": 0,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
        }
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
    if len(args.rollout_manifests) < 3:
        raise ValueError("At least three distinct initial-state manifests are required")
    args.output_dir.mkdir(parents=True)
    mean, std = load_training_normalization(args.feature_dir, args.task_ids)
    raw_goals, goals, initial_goals = load_goals(args.feature_dir, args.task_ids, mean, std)
    heads = load_heads(args.head_dir)
    records = collect_records(args.rollout_manifests, set(args.task_ids), args.windows_per_trajectory)
    stamp(f"START trajectories={len({r['trajectory']['trajectory_id'] for r in records})} windows={len(records)}")
    extractor = LaWAMLAMFeatureExtractor(
        lam_config_path=args.lam_config,
        lam_ckpt_path=args.lam_checkpoint,
        vision_model_id=args.dino_model,
        device=args.device,
        strict=True,
    )
    stamp(f"LAM loaded={extractor.using_lam}")
    rows = extract_rollout_rows(records, extractor)
    scores = score_rows(rows, heads, mean, std, raw_goals, goals, initial_goals)
    truth = np.asarray([row["direction"] for row in rows])
    metrics = {name: direction_metrics(score, truth) for name, score in scores.items()}
    comparisons = {
        "potential_correct_minus_cosine": paired_bootstrap(
            rows, truth, scores["potential_correct"], scores["cosine"], args.bootstrap_draws, args.seed
        ),
        "potential_correct_minus_masked": paired_bootstrap(
            rows, truth, scores["potential_correct"], scores["potential_masked"], args.bootstrap_draws, args.seed + 1
        ),
        "potential_correct_minus_initial_goal": paired_bootstrap(
            rows, truth, scores["potential_correct"], scores["potential_initial_goal"], args.bootstrap_draws, args.seed + 2
        ),
        "potential_correct_minus_permuted": paired_bootstrap(
            rows, truth, scores["potential_correct"], scores["potential_permuted"], args.bootstrap_draws, args.seed + 3
        ),
        "direct_correct_minus_no_goal": paired_bootstrap(
            rows, truth, scores["direct_correct"], scores["direct_no_goal"], args.bootstrap_draws, args.seed + 4
        ),
        "direct_correct_minus_potential_correct": paired_bootstrap(
            rows, truth, scores["direct_correct"], scores["potential_correct"], args.bootstrap_draws, args.seed + 5
        ),
    }
    clusters = sorted({row["initial_state_cluster"] for row in rows})
    cluster_results = []
    for cluster in clusters:
        index = np.asarray([i for i, row in enumerate(rows) if row["initial_state_cluster"] == cluster])
        cluster_results.append(
            {
                "cluster": cluster,
                "task": rows[int(index[0])]["task"],
                "windows": len(index),
                "potential": direction_metrics(scores["potential_correct"][index], truth[index]),
                "cosine": direction_metrics(scores["cosine"][index], truth[index]),
            }
        )
    review = []
    for index, row in enumerate(rows):
        prediction = 1 if scores["potential_correct"][index] > 0 else -1
        disagreement = prediction != truth[index] and truth[index] != 0
        goal_sensitive = np.sign(scores["potential_correct"][index]) != np.sign(scores["potential_initial_goal"][index])
        if disagreement or goal_sensitive:
            public = {key: value for key, value in row.items() if key not in {"z0", "z1"}}
            public["potential_correct"] = float(scores["potential_correct"][index])
            public["potential_initial_goal"] = float(scores["potential_initial_goal"][index])
            public["cosine"] = float(scores["cosine"][index])
            public["reason"] = {
                "direction_error": bool(disagreement), "goal_sign_change": bool(goal_sensitive)
            }
            review.append(public)
    with (args.output_dir / "review_candidates.jsonl").open("x") as handle:
        for row in review:
            handle.write(json.dumps(row) + "\n")
    gate = {
        "potential_vs_cosine_ci_low_above_zero": comparisons["potential_correct_minus_cosine"]["ci95_low"] > 0,
        "potential_regression_recall_at_least_060": metrics["potential_correct"]["regression_recall"] >= 0.60,
        "correct_vs_masked_ci_low_above_zero": comparisons["potential_correct_minus_masked"]["ci95_low"] > 0,
        "correct_vs_initial_goal_ci_low_above_zero": comparisons["potential_correct_minus_initial_goal"]["ci95_low"] > 0,
        "at_least_three_initial_states_per_task": all(
            len({row["initial_state_cluster"] for row in rows if row["task"] == task}) >= 3
            for task in args.task_ids
        ),
        "independent_reviewed_labels": False,
    }
    result = {
        "status": "MULTISTATE_HEURISTIC_VALIDATION",
        "protocol": {
            "heads": str(args.head_dir),
            "heads_trained_or_tuned": False,
            "manifests": [str(path) for path in args.rollout_manifests],
            "tasks": args.task_ids,
            "trajectories": len({row["trajectory_id"] for row in rows}),
            "initial_state_clusters": len(clusters),
            "windows": len(rows),
            "windows_per_trajectory": args.windows_per_trajectory,
            "bootstrap": "hierarchical task then initial-state cluster paired bootstrap",
            "labels": "existing physics heuristic reconstructed by exact action replay",
            "warning": "Not an independently reviewed challenge set; formal Gate 1 cannot pass yet.",
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate_checks": gate,
        "cluster_results": cluster_results,
        "review_candidates": len(review),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "predictions.npz", truth=truth, **scores)
    summary = [
        "# Pipeline-v4 Gate-1 multi-state frozen-head validation",
        "",
        "Frozen smoke heads were evaluated without training or tuning.",
        "",
        f"- Tasks: {len(args.task_ids)}; initial-state clusters: {len(clusters)}; trajectories: {result['protocol']['trajectories']}; windows: {len(rows)}",
        f"- Physical directions: P={int(np.sum(truth > 0))}, U={int(np.sum(truth == 0))}, N={int(np.sum(truth < 0))}",
        "",
        "| Method / control | Balanced accuracy | Forward recall | Regression recall |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "cosine", "potential_correct", "potential_masked", "potential_permuted",
        "potential_initial_goal", "direct_correct", "direct_no_goal",
    ):
        metric = metrics[name]
        summary.append(
            f"| `{name}` | {metric['balanced_accuracy']:.4f} | {metric['forward_recall']:.4f} | {metric['regression_recall']:.4f} |"
        )
    summary.extend(("", "## Hierarchical paired bootstrap", ""))
    for name, value in comparisons.items():
        summary.append(
            f"- `{name}`: mean={value['mean']:.4f}, 95% CI [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"
        )
    summary.extend(("", "## Gate checks", ""))
    for name, passed in gate.items():
        summary.append(f"- `{name}`: {passed}")
    summary.extend(
        (
            "",
            f"Review candidates saved: {len(review)}.",
            "",
            "Formal Gate 1 remains pending until the selected challenge windows receive independent review.",
            "",
        )
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary))
    stamp("RESULTS")
    print("\n".join(summary), flush=True)
    stamp("COMPLETE")


if __name__ == "__main__":
    main()
