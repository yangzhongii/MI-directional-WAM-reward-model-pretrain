from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.relations.sequence import load_relation_sequence, relation_progress_potential
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.scoring.directional_potential import DirectionalScoreConfig, score_candidate_trajectory
from mi_reward.scoring.trajectory_score import score_trajectory


TEACHER_VERSION_LEGACY = "legacy_v0"
TEACHER_VERSION_DAME_ALIGNED = "dame_aligned_v1"


def _group_by_task(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item.task, []).append(item)
    return grouped


def _group_by_task_and_goal(items):
    grouped = {}
    for item in items:
        grouped.setdefault((item.task, getattr(item, "goal_ref_id", None)), []).append(item)
    return grouped


def _requires_verification(traj: TrajectoryExample) -> bool:
    return traj.source == "cosmos_action_cond" or traj.candidate_provenance is not None


def score_manifest(
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
    gamma: float,
    mi_mode: str,
    relation_weight: float = 0.0,
    use_token_features: bool = False,
    directional_alignment: bool = False,
    directional_config: DirectionalScoreConfig | None = None,
) -> list[dict[str, object]]:
    if directional_alignment and not use_token_features:
        raise ValueError("Directional alignment requires native token features; pass --token_features.")
    directional_estimator = DameSoftHistogramMI(num_bins=8) if directional_alignment else None
    directional_config = directional_config or DirectionalScoreConfig(gamma=gamma)
    store = CachedFeatureStore(feature_root)
    trajectories = read_jsonl(manifest, TrajectoryExample)
    refs_by_task = _group_by_task(read_jsonl(success_refs, SuccessReference))
    scored = []
    for traj in trajectories:
        if _requires_verification(traj) and not (traj.verification and traj.verification.accepted):
            continue
        refs = refs_by_task.get(traj.task, [])
        if traj.goal_ref_id:
            refs = [ref for ref in refs if ref.ref_id == traj.goal_ref_id]
        if not refs:
            if traj.goal_ref_id:
                raise ValueError(
                    f"Trajectory {traj.traj_id} declares unknown goal_ref_id {traj.goal_ref_id!r} for task {traj.task!r}."
                )
            continue
        feature_id = traj.traj_id + "_tokens" if use_token_features else traj.traj_id
        candidate_features = store.load(feature_id)
        reference_features = {
            ref.ref_id: store.load(ref.ref_id + "_tokens" if use_token_features else ref.ref_id)
            for ref in refs
        }
        if directional_alignment:
            assert directional_estimator is not None
            if candidate_features.ndim != 3 or any(item.ndim != 3 for item in reference_features.values()):
                raise ValueError("Directional alignment requires [T, K, D] candidate and reference tokens.")
            result = score_candidate_trajectory(
                candidate_features,
                reference_features,
                directional_estimator,
                directional_config,
            )
            if result.selected_reference_id is None:
                raise ValueError(f"Directional scoring did not select a success reference for {traj.traj_id}.")
            best_ref = next(ref for ref in refs if ref.ref_id == result.selected_reference_id)
            best = {
                "phi": [float(value) for value in result.phi.detach().cpu().tolist()],
                "score_delta": result.directional_score,
                "score_mean": result.mean_alignment,
                "directional_score": result.directional_score,
                "endpoint_progress": result.endpoint_progress,
                "positive_gain": result.positive_gain,
                "regression_penalty": result.regression_penalty,
                "stage_progress": result.stage_progress,
                "confidence": result.confidence,
            }
        else:
            scored_refs = [
                (
                    ref,
                    score_trajectory(
                        candidate_features,
                        reference_features[ref.ref_id],
                        gamma=gamma,
                        mi_mode=mi_mode,
                    ),
                )
                for ref in refs
            ]
            best_ref, best = max(scored_refs, key=lambda item: float(item[1]["score_delta"]))
        relation_score = 0.0
        relation_phi: list[float] | None = None
        if relation_weight:
            if not traj.relation_path:
                raise ValueError(f"relation_weight requires relation_path for {traj.traj_id}.")
            relation = load_relation_sequence(traj.relation_path)
            relation_values = relation_progress_potential(relation.values, relation.names)
            if relation_values.shape[0] != candidate_features.shape[0]:
                raise ValueError(
                    f"Relation/frame length mismatch while scoring {traj.traj_id}: "
                    f"{relation_values.shape[0]} vs {candidate_features.shape[0]}."
                )
            relation_phi = [float(value) for value in relation_values.tolist()]
            if relation_values.numel() >= 2:
                relation_score = float((gamma * relation_values[1:] - relation_values[:-1]).mean().item())
        scored.append(
            {
                "traj_id": traj.traj_id,
                "task": traj.task,
                "goal_ref_id": best_ref.ref_id,
                "score_delta": float(best["score_delta"]) + relation_weight * relation_score,
                "score_mean": float(best["score_mean"]),
                "phi": best["phi"],
                "relation_phi": relation_phi,
                "relation_score": relation_score,
                "confidence": float(best.get("confidence", 1.0)),
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
    by_task = _group_by_task_and_goal([type("Scored", (), item) for item in scored])
    pairs: list[PreferencePair] = []
    for (task, goal_ref_id), task_items in by_task.items():
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
                        goal_ref_id=goal_ref_id,
                        chosen_confidence=chosen_conf,
                        rejected_confidence=rejected_conf,
                        teacher_version=teacher_version,
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
    by_task = _group_by_task_and_goal([type("Scored", (), item) for item in scored])
    pairs: list[PreferencePair] = []
    for (task, goal_ref_id), task_items in by_task.items():
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
                    goal_ref_id=goal_ref_id,
                    chosen_confidence=chosen_conf,
                    rejected_confidence=rejected_conf,
                    teacher_version=teacher_version,
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
    parser.add_argument("--relation_weight", type=float, default=0.0)
    parser.add_argument("--token_features", action="store_true", help="Score native [T, K, D] LaWAM visual tokens.")
    parser.add_argument(
        "--directional-alignment",
        action="store_true",
        help="Use monotonic MI alignment and directional progress components for preference scores.",
    )
    parser.add_argument("--directional_w_endpoint", type=float, default=1.0)
    parser.add_argument("--directional_w_positive", type=float, default=0.5)
    parser.add_argument("--directional_w_regression", type=float, default=1.0)
    parser.add_argument("--directional_w_stage", type=float, default=1.0)
    parser.add_argument("--directional_w_alignment", type=float, default=0.1)
    args = parser.parse_args()

    scored = score_manifest(
        args.manifest, args.success_refs, args.feature_root, args.gamma, args.mi_mode,
        relation_weight=args.relation_weight, use_token_features=args.token_features,
        directional_alignment=args.directional_alignment,
        directional_config=DirectionalScoreConfig(
            gamma=args.gamma,
            w_endpoint=args.directional_w_endpoint,
            w_positive=args.directional_w_positive,
            w_regression=args.directional_w_regression,
            w_stage=args.directional_w_stage,
            w_alignment=args.directional_w_alignment,
        ),
    )

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
