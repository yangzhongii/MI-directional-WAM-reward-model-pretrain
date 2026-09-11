"""Test whether the v4 multi-state No-Go depends on physics thresholds.

The model predictions are frozen and loaded from the completed multi-state
evaluation.  This script reconstructs the same 240 windows by exact action
replay, varies only the approach/transport deadband, and reports hierarchical
paired bootstrap intervals.  It does not provide independent human labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from audit_teacher import live_trace_physics
from eval_v4_gate1_multistate import collect_records, paired_bootstrap
from probe_v4_gate1_smoke import direction_metrics
from mi_reward.data.libero_privileged import classify_physical_transition, resolve_libero_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--windows-per-trajectory", type=int, default=8)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=121)
    return parser.parse_args()


def stamp(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def classify_with_multiplier(left: dict[str, Any], right: dict[str, Any], multiplier: float) -> int:
    """Same event priority as the existing heuristic, with scaled distance thresholds."""
    if bool(right["environment_success"]) and not bool(left["environment_success"]):
        return 1
    left_grasped = bool(left["grasped"])
    right_grasped = bool(right["grasped"])
    if left_grasped and not right_grasped:
        return -1
    if right_grasped and not left_grasped:
        return 1
    if not left_grasped and not right_grasped:
        delta = float(left["eef_object_distance_xyz"]) - float(right["eef_object_distance_xyz"])
        epsilon = 7.5e-4 * multiplier
    else:
        delta = float(left["object_goal_distance_xy"]) - float(right["object_goal_distance_xy"])
        epsilon = 5.0e-4 * multiplier
    if delta > epsilon:
        return 1
    if delta < -epsilon:
        return -1
    return 0


def authoritative_event(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, str]:
    if bool(right["environment_success"]) and not bool(left["environment_success"]):
        return 1, "success_acquisition"
    left_grasped = bool(left["grasped"])
    right_grasped = bool(right["grasped"])
    if left_grasped and not right_grasped:
        return -1, "grasp_loss"
    if right_grasped and not left_grasped:
        return 1, "grasp_acquisition"
    return 0, "no_authoritative_event"


def reconstruct_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = collect_records(args.rollout_manifests, set(args.task_ids), args.windows_per_trajectory)
    grouped: dict[tuple[Path, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["manifest"], record["trajectory"]["trajectory_id"])].append(record)
    rows: list[dict[str, Any]] = []
    for (manifest, trajectory_id), group in grouped.items():
        trajectory = group[0]["trajectory"]
        task = int(trajectory["task_index"])
        _, bddl, _ = resolve_libero_task(Path.cwd(), "libero_spatial", task)
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
            raise RuntimeError(f"Replay mismatch: {trajectory_id}")
        for record in group:
            window = record["window"]
            left, right = window["row"]["provenance"]["history_frame_indices"][-2:]
            event_label, event_type = authoritative_event(physical[left], physical[right])
            rows.append(
                {
                    "task": task,
                    "initial_state_cluster": trajectory["pair_id"],
                    "trajectory_id": trajectory_id,
                    "source_demo": trajectory["demo_name"],
                    "policy_id": trajectory["policy_id"],
                    "window_end": int(window["window_end"]),
                    "frame_t": int(left),
                    "frame_t1": int(right),
                    "physical_t": physical[left],
                    "physical_t1": physical[right],
                    "base_direction": classify_physical_transition(physical[left], physical[right]),
                    "event_direction": event_label,
                    "event_type": event_type,
                    "images": window["row"]["images"],
                }
            )
        stamp(f"REPLAY {trajectory_id} windows={len(group)} total={len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    predictions = np.load(args.prediction_file)
    rows = reconstruct_rows(args)
    base = np.asarray([row["base_direction"] for row in rows], dtype=np.int64)
    if not np.array_equal(base, predictions["truth"]):
        mismatch = np.flatnonzero(base != predictions["truth"])
        raise RuntimeError(f"Prediction/window order mismatch at {mismatch[:10].tolist()}")
    score_names = [name for name in predictions.files if name != "truth"]
    score_map = {name: predictions[name] for name in score_names}
    variants: dict[str, Any] = {}
    for index, multiplier in enumerate(args.multipliers):
        truth = np.asarray(
            [classify_with_multiplier(row["physical_t"], row["physical_t1"], multiplier) for row in rows],
            dtype=np.int64,
        )
        name = f"threshold_x{multiplier:g}"
        metrics = {score_name: direction_metrics(score, truth) for score_name, score in score_map.items()}
        comparisons = {
            "potential_correct_minus_cosine": paired_bootstrap(
                rows, truth, score_map["potential_correct"], score_map["cosine"],
                args.bootstrap_draws, args.seed + index * 10,
            ),
            "potential_correct_minus_masked": paired_bootstrap(
                rows, truth, score_map["potential_correct"], score_map["potential_masked"],
                args.bootstrap_draws, args.seed + index * 10 + 1,
            ),
            "potential_correct_minus_initial_goal": paired_bootstrap(
                rows, truth, score_map["potential_correct"], score_map["potential_initial_goal"],
                args.bootstrap_draws, args.seed + index * 10 + 2,
            ),
        }
        variants[name] = {
            "multiplier": multiplier,
            "counts": {"P": int(np.sum(truth > 0)), "U": int(np.sum(truth == 0)), "N": int(np.sum(truth < 0))},
            "changed_vs_x1": int(np.sum(truth != base)),
            "metrics": metrics,
            "comparisons": comparisons,
        }
    event_truth = np.asarray([row["event_direction"] for row in rows], dtype=np.int64)
    event_metrics = {name: direction_metrics(score, event_truth) for name, score in score_map.items()}
    event_types = defaultdict(int)
    for row in rows:
        event_types[row["event_type"]] += 1
    event_comparisons = {
        "potential_correct_minus_cosine": paired_bootstrap(
            rows, event_truth, score_map["potential_correct"], score_map["cosine"],
            args.bootstrap_draws, args.seed + 100,
        ),
        "potential_correct_minus_masked": paired_bootstrap(
            rows, event_truth, score_map["potential_correct"], score_map["potential_masked"],
            args.bootstrap_draws, args.seed + 101,
        ),
    }
    high_confidence = np.asarray(
        [classify_with_multiplier(row["physical_t"], row["physical_t1"], 3.0) for row in rows], dtype=np.int64
    )
    review_rows = []
    for index, row in enumerate(rows):
        if high_confidence[index] == 0:
            continue
        predicted = 1 if score_map["potential_correct"][index] > 0 else -1
        if predicted != high_confidence[index]:
            public = dict(row)
            public.update(
                {
                    "high_confidence_direction": int(high_confidence[index]),
                    **{name: float(score[index]) for name, score in score_map.items()},
                }
            )
            review_rows.append(public)
    with (args.output_dir / "high_confidence_errors.jsonl").open("x") as handle:
        for row in review_rows:
            handle.write(json.dumps(row) + "\n")
    result = {
        "status": "LABEL_SENSITIVITY_ONLY",
        "protocol": {
            "predictions": str(args.prediction_file),
            "predictions_recomputed_or_tuned": False,
            "windows": len(rows),
            "tasks": len(set(row["task"] for row in rows)),
            "initial_state_clusters": len(set(row["initial_state_cluster"] for row in rows)),
            "multipliers": args.multipliers,
            "base_mapping_verified": True,
            "bootstrap": "hierarchical task then initial-state cluster paired bootstrap",
            "warning": "Threshold sensitivity is not independent human annotation.",
        },
        "variants": variants,
        "authoritative_events_only": {
            "counts": {
                "P": int(np.sum(event_truth > 0)), "U": int(np.sum(event_truth == 0)),
                "N": int(np.sum(event_truth < 0)), "types": dict(event_types),
            },
            "metrics": event_metrics,
            "comparisons": event_comparisons,
        },
        "high_confidence_errors": len(review_rows),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = [
        "# Pipeline-v4 label-threshold robustness",
        "",
        "Predictions are frozen; only physical distance thresholds change.",
        "",
        "| Threshold | P/U/N | Potential BA | Potential N recall | Potential-cosine 95% CI | Correct-masked 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in variants.items():
        metric = value["metrics"]["potential_correct"]
        vs_cosine = value["comparisons"]["potential_correct_minus_cosine"]
        vs_mask = value["comparisons"]["potential_correct_minus_masked"]
        counts = value["counts"]
        summary.append(
            f"| `{name}` | {counts['P']}/{counts['U']}/{counts['N']} | "
            f"{metric['balanced_accuracy']:.4f} | {metric['regression_recall']:.4f} | "
            f"[{vs_cosine['ci95_low']:.4f}, {vs_cosine['ci95_high']:.4f}] | "
            f"[{vs_mask['ci95_low']:.4f}, {vs_mask['ci95_high']:.4f}] |"
        )
    event = result["authoritative_events_only"]
    event_metric = event["metrics"]["potential_correct"]
    summary.extend(
        (
            "",
            "## Authoritative events only",
            "",
            f"- Counts P/U/N: {event['counts']['P']}/{event['counts']['U']}/{event['counts']['N']}",
            f"- Potential balanced accuracy: {event_metric['balanced_accuracy']}",
            f"- Potential regression recall: {event_metric['regression_recall']}",
            f"- High-confidence potential errors saved: {len(review_rows)}",
            "",
            "This experiment tests threshold dependence. It does not replace independent visual/event review.",
            "",
        )
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary))
    stamp("RESULTS")
    print("\n".join(summary), flush=True)
    stamp("COMPLETE")


if __name__ == "__main__":
    main()
