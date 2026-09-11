"""LIBERO trajectory -> existing Qwen inference -> task-matched metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from eval.libero.libero_adapter import SUITES, iter_hdf5, iter_manifest
from mi_reward.evaluation.reward_pair_metrics import summarize_margins


def aggregate_windows(predictions):
    grouped = defaultdict(list)
    for prediction in predictions:
        if prediction["window"] != "early_prefix":
            grouped[(prediction["task_id"], prediction["trajectory_id"])].append(prediction)
    episodes = []
    for group in grouped.values():
        terminal = next(p for p in group if p["window"] == "terminal")
        episode = dict(terminal)
        episode["reward"] = sum(p["reward"] for p in group) / len(group)
        episode["scored_windows"] = len(group)
        episode["window_label_counts"] = dict(Counter(p["label"] for p in group))
        episodes.append(episode)
    return episodes


def compute_metrics(predictions):
    terminal = aggregate_windows(predictions)
    ranking, margins, matched_margins = [], [], []
    for a, b in combinations(terminal, 2):
        if a["task_id"] != b["task_id"]:
            continue
        qa, qb = a.get("quality"), b.get("quality")
        if qa is not None and qb is not None and qa != qb:
            ranking.append(float((a["reward"] - b["reward"]) * (qa - qb) > 0))
        sa, sb = a.get("success"), b.get("success")
        if sa is not None and sb is not None and sa != sb:
            margin = (a["reward"] - b["reward"]) * (1 if sa else -1)
            margins.append(margin)
            if a.get("pair_id") is not None and a["pair_id"] == b.get("pair_id"):
                matched_margins.append(margin)
    mean = lambda values: sum(values) / len(values) if values else None
    metrics = {
        "ranking_accuracy": mean(ranking),
        "success_failure_accuracy": mean([float(m > 0) for m in margins]),
        "reward_margin": mean(margins),
    }
    reasons = {}
    if not ranking:
        reasons["ranking_accuracy"] = "No within-task trajectory pairs with distinct ground-truth quality."
    if not margins:
        for key in ("success_failure_accuracy", "reward_margin"):
            reasons[key] = "No within-task real success/failure trajectory pairs."
    terminals = {(p["task_id"], p["trajectory_id"]): p for p in terminal}
    temporal = [terminals[(p["task_id"], p["trajectory_id"])]["reward"] - p["reward"]
                for p in predictions if p["window"] == "early_prefix"]
    return {
        "metrics": metrics, "unavailable_reasons": reasons,
        "success_failure_details": summarize_margins(margins),
        "matched_initial_state_success_failure": summarize_margins(matched_margins),
        "by_controller": {
            policy: {
                "episodes": sum(p.get("policy_id") == policy for p in terminal),
                "success_rate": mean([float(p["success"]) for p in terminal
                                      if p.get("policy_id") == policy and p.get("success") is not None]),
                "mean_reward": mean([p["reward"] for p in terminal if p.get("policy_id") == policy]),
                "window_label_counts": dict(Counter(p["label"] for p in predictions if p.get("policy_id") == policy)),
            } for policy in sorted({p["policy_id"] for p in terminal if p.get("policy_id") is not None})
        },
        "counts": {"trajectories": len(terminal), "predictions": len(predictions),
                   "ranking_pairs": len(ranking), "success_failure_pairs": len(margins),
                   "successes": sum(p.get("success") is True for p in terminal),
                   "failures": sum(p.get("success") is False for p in terminal)},
        "metric_definitions": {
            "ranking_accuracy": "Pair-weighted within-task accuracy, higher annotated quality preferred.",
            "success_failure_accuracy": "Pair-weighted within-task P(reward_success > reward_failure).",
            "reward_margin": "Mean reward_success - reward_failure over the same pairs.",
            "ties": "Reward ties count as incorrect; equal ground-truth qualities excluded.",
        },
        "smoke_diagnostics": {
            "terminal_label_counts": dict(Counter(p["label"] for p in terminal)),
            "terminal_vs_early_prefix_pairs": len(temporal),
            "terminal_vs_early_prefix_accuracy": mean([float(m > 0) for m in temporal]),
            "terminal_minus_early_prefix_reward": mean(temporal),
            "note": "Temporal proxy only; early prefixes are not labeled failure trajectories.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=SUITES, default="libero_spatial")
    parser.add_argument("--data-root", default=".venv/src/libero/libero/datasets")
    parser.add_argument("--manifest", help="Labeled real-rollout JSONL, instead of official demos")
    parser.add_argument("--checkpoint", default="logs/mi_reward/v3_teacher_mainline/qwen3_vl_reward_v1/model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tasks", type=int, default=1, help="HDF5 task limit; 0=all")
    parser.add_argument("--max-demos", type=int, default=1, help="HDF5 demos per task; 0=all")
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--windows-per-trajectory", type=int, default=1,
                        help="Manifest: evenly spaced contiguous windows, mean reward aggregation")
    parser.add_argument("--prefix-smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Avoid leaving stale completed metrics visible if a run fails.
    if any((output / name).exists() for name in ("results.json", "predictions.jsonl")):
        raise FileExistsError(f"Choose a new output directory: {output}")
    records = list(iter_manifest(args.manifest, args.benchmark, args.history_frames, args.windows_per_trajectory)
                   if args.manifest else iter_hdf5(
                       args.data_root, args.benchmark, output / "images", args.max_tasks,
                       args.max_demos, args.history_frames, args.prefix_smoke))
    if not records:
        raise ValueError("No trajectories selected")
    (output / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"Prepared {len(records)} windows", flush=True)
    from mi_reward.inference.qwen3_vl_reward import Qwen3VLRewardModel
    model = Qwen3VLRewardModel(args.checkpoint)
    predictions = []
    with (output / "predictions.jsonl").open("w") as handle:
        for i, record in enumerate(records):
            result = model.predict_row(record["row"])
            prediction = {k: v for k, v in record.items() if k != "row"}
            prediction.update(label=result.label, reward=result.reward, raw_output=result.raw_output)
            predictions.append(prediction)
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"[{i + 1}/{len(records)}] {record['trajectory_id']} {record['window']}: {result.label} ({result.reward})", flush=True)
    checkpoint = Path(args.checkpoint).resolve()
    (output / "trajectories.jsonl").write_text("".join(json.dumps(p) + "\n" for p in aggregate_windows(predictions)))
    report = {
        "benchmark": args.benchmark,
        "checkpoint": checkpoint.parent.name if checkpoint.name == "model" else checkpoint.name,
        "checkpoint_path": str(checkpoint), "status": "completed",
        "config": vars(args), "inference": "mi_reward.inference.qwen3_vl_reward.Qwen3VLRewardModel.predict_row",
        "trajectory_score": "Mean P/U/N reward across uniformly sampled contiguous history windows (one window means terminal only)",
        **compute_metrics(predictions),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
