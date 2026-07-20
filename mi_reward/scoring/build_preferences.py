from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.scoring.trajectory_score import score_trajectory


TEACHER_VERSION_LEGACY = "legacy_v0"
TEACHER_VERSION_DAME_ALIGNED = "dame_aligned_v1"


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


def build_preference_pairs(
    scored: list[dict[str, object]],
    margin: float,
    top_k: int,
    bottom_k: int,
    max_pairs_per_task: int | None = None,
    min_confidence: float = 0.0,
    teacher_version: str = TEACHER_VERSION_LEGACY,
    score_field: str = "score_delta",
    seed: int = 0,
) -> list[PreferencePair]:
    """Build preference pairs with confidence, versioning, and pair-count limits.

    Args:
        scored: list of scored trajectory dicts
        margin: minimum score gap for a valid preference
        top_k: number of top trajectories to use as chosen
        bottom_k: number of bottom trajectories to use as rejected
        max_pairs_per_task: maximum number of pairs per task (None = unlimited)
        min_confidence: minimum confidence for both chosen and rejected
        teacher_version: version tag for the scoring teacher
        score_field: which score field to use for ranking
        seed: random seed for deterministic pair sampling

    Returns:
        list of PreferencePair with extended metadata
    """
    rng = random.Random(seed)
    by_task = _group_by_task([type("Scored", (), item) for item in scored])
    pairs: list[PreferencePair] = []
    for task, task_items in by_task.items():
        ranked = sorted(task_items, key=lambda item: float(getattr(item, score_field, 0.0)), reverse=True)
        top = ranked[:top_k]
        bottom = ranked[-bottom_k:] if bottom_k > 0 else []

        task_pairs = []
        for chosen in top:
            for rejected in bottom:
                if chosen.traj_id == rejected.traj_id:
                    continue
                chosen_score = float(getattr(chosen, score_field, 0.0))
                rejected_score = float(getattr(rejected, score_field, 0.0))
                chosen_conf = float(getattr(chosen, "confidence", 1.0))
                rejected_conf = float(getattr(rejected, "confidence", 1.0))

                if chosen_conf < min_confidence or rejected_conf < min_confidence:
                    continue
                if chosen_score <= rejected_score + margin:
                    continue

                task_pairs.append(
                    PreferencePair(
                        task=task,
                        chosen_traj_id=str(chosen.traj_id),
                        rejected_traj_id=str(rejected.traj_id),
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                        score_type=f"temporally_aligned_dame_mi_{teacher_version}",
                    )
                )

        # Limit pairs per task if requested
        if max_pairs_per_task is not None and len(task_pairs) > max_pairs_per_task:
            task_pairs = rng.sample(task_pairs, max_pairs_per_task)

        pairs.extend(task_pairs)

    return pairs


def build_adjacent_pairs(
    scored: list[dict[str, object]],
    margin: float,
    max_pairs_per_task: int | None = None,
    min_confidence: float = 0.0,
    teacher_version: str = TEACHER_VERSION_DAME_ALIGNED,
    score_field: str = "score_delta",
    seed: int = 0,
) -> list[PreferencePair]:
    """Build preference pairs from adjacent-ranked trajectories.

    Alternative to top-vs-bottom: uses neighbors in the ranking.
    """
    rng = random.Random(seed)
    by_task = _group_by_task([type("Scored", (), item) for item in scored])
    pairs: list[PreferencePair] = []
    for task, task_items in by_task.items():
        ranked = sorted(task_items, key=lambda item: float(getattr(item, score_field, 0.0)), reverse=True)
        task_pairs = []
        for i in range(len(ranked) - 1):
            chosen = ranked[i]
            rejected = ranked[i + 1]
            chosen_score = float(getattr(chosen, score_field, 0.0))
            rejected_score = float(getattr(rejected, score_field, 0.0))
            chosen_conf = float(getattr(chosen, "confidence", 1.0))
            rejected_conf = float(getattr(rejected, "confidence", 1.0))

            if chosen_conf < min_confidence or rejected_conf < min_confidence:
                continue
            if chosen_score <= rejected_score + margin:
                continue

            task_pairs.append(
                PreferencePair(
                    task=task,
                    chosen_traj_id=str(chosen.traj_id),
                    rejected_traj_id=str(rejected.traj_id),
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                    score_type=f"temporally_aligned_dame_mi_{teacher_version}",
                )
            )

        if max_pairs_per_task is not None and len(task_pairs) > max_pairs_per_task:
            task_pairs = rng.sample(task_pairs, max_pairs_per_task)

        pairs.extend(task_pairs)

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
    parser.add_argument("--max_pairs_per_task", type=int, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--pair_mode", default="top_vs_bottom", choices=["top_vs_bottom", "adjacent"])
    parser.add_argument("--teacher_version", default=TEACHER_VERSION_LEGACY)
    parser.add_argument("--score_field", default="score_delta")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mi_mode", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi"])
    args = parser.parse_args()

    scored = score_manifest(args.manifest, args.success_refs, args.feature_root, args.gamma, args.mi_mode)

    if args.pair_mode == "adjacent":
        pairs = build_adjacent_pairs(
            scored, margin=args.margin, max_pairs_per_task=args.max_pairs_per_task,
            min_confidence=args.min_confidence, teacher_version=args.teacher_version,
            score_field=args.score_field, seed=args.seed,
        )
    else:
        pairs = build_preference_pairs(
            scored, margin=args.margin, top_k=args.top_k, bottom_k=args.bottom_k,
            max_pairs_per_task=args.max_pairs_per_task, min_confidence=args.min_confidence,
            teacher_version=args.teacher_version, score_field=args.score_field, seed=args.seed,
        )
    write_jsonl(args.output, pairs)
    report_path = Path(args.output).with_name("score_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"scores": scored, "num_pairs": len(pairs)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
