from __future__ import annotations

import argparse
import json
from pathlib import Path

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.scoring.trajectory_score import score_trajectory


def _group_by_task(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item.task, []).append(item)
    return grouped


def score_manifest(
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
    gamma: float,
    mi_mode: str,
) -> list[dict[str, object]]:
    store = CachedFeatureStore(feature_root)
    trajectories = read_jsonl(manifest, TrajectoryExample)
    refs_by_task = _group_by_task(read_jsonl(success_refs, SuccessReference))
    scored = []
    for traj in trajectories:
        refs = refs_by_task.get(traj.task, [])
        if not refs:
            continue
        candidate_features = store.load(traj.traj_id)
        ref_scores = [
            score_trajectory(candidate_features, store.load(ref.ref_id), gamma=gamma, mi_mode=mi_mode)
            for ref in refs
        ]
        best = max(ref_scores, key=lambda item: float(item["score_delta"]))
        scored.append(
            {
                "traj_id": traj.traj_id,
                "task": traj.task,
                "score_delta": float(best["score_delta"]),
                "score_mean": float(best["score_mean"]),
                "phi": best["phi"],
            }
        )
    return scored


def build_preference_pairs(scored: list[dict[str, object]], margin: float, top_k: int, bottom_k: int) -> list[PreferencePair]:
    by_task = _group_by_task([type("Scored", (), item) for item in scored])
    pairs: list[PreferencePair] = []
    for task, task_items in by_task.items():
        ranked = sorted(task_items, key=lambda item: float(item.score_delta), reverse=True)
        top = ranked[:top_k]
        bottom = ranked[-bottom_k:] if bottom_k > 0 else []
        for chosen in top:
            for rejected in bottom:
                if chosen.traj_id == rejected.traj_id:
                    continue
                chosen_score = float(chosen.score_delta)
                rejected_score = float(rejected.score_delta)
                if chosen_score > rejected_score + margin:
                    pairs.append(
                        PreferencePair(
                            task=task,
                            chosen_traj_id=str(chosen.traj_id),
                            rejected_traj_id=str(rejected.traj_id),
                            chosen_score=chosen_score,
                            rejected_score=rejected_score,
                            score_type="mi_delta",
                        )
                    )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MI-directional preference pairs from cached features.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--success_refs", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bottom_k", type=int, default=5)
    parser.add_argument("--mi_mode", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi"])
    args = parser.parse_args()

    scored = score_manifest(args.manifest, args.success_refs, args.feature_root, args.gamma, args.mi_mode)
    pairs = build_preference_pairs(scored, margin=args.margin, top_k=args.top_k, bottom_k=args.bottom_k)
    write_jsonl(args.output, pairs)
    report_path = Path(args.output).with_name("score_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"scores": scored, "num_pairs": len(pairs)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
