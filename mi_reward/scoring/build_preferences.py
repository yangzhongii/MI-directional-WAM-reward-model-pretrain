from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.relations.sequence import load_relation_sequence, relation_progress_potential
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.scoring.directional_potential import (
    DirectionalScoreConfig,
    alignment_stage_potential,
    combine_process_potentials,
    score_candidate_trajectory,
    score_potential_curve,
    transition_potential_to_frames,
)
from mi_reward.scoring.trajectory_score import score_trajectory


TEACHER_VERSION_LEGACY = "legacy_v0"
TEACHER_VERSION_DAME_ALIGNED = "dame_aligned_v1"
SCORE_CACHE_VERSION = 2


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


def _resolve_score_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring was requested, but torch.cuda.is_available() is false.")
    return device


def _path_stamp(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _feature_root_stamp(path: str | Path) -> dict[str, int]:
    files = list(Path(path).resolve().glob("*.pt"))
    stats = [item.stat() for item in files]
    return {
        "files": len(stats),
        "bytes": sum(item.st_size for item in stats),
        "latest_mtime_ns": max((item.st_mtime_ns for item in stats), default=0),
    }


def _score_cache_signature(
    *,
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
    gamma: float,
    mi_mode: str,
    relation_weight: float,
    action_weight: float,
    kinematic_weight: float,
    outcome_weight: float,
    success_monotonic_projection: bool,
    success_endpoint_anchor: bool,
    allowed_splits: tuple[str, ...] | None,
    use_token_features: bool,
    directional_alignment: bool,
    directional_config: DirectionalScoreConfig,
) -> str:
    payload = {
        "version": SCORE_CACHE_VERSION,
        "manifest": _path_stamp(manifest),
        "success_refs": _path_stamp(success_refs),
        "features": _feature_root_stamp(feature_root),
        "gamma": gamma,
        "mi_mode": mi_mode,
        "relation_weight": relation_weight,
        "action_weight": action_weight,
        "kinematic_weight": kinematic_weight,
        "outcome_weight": outcome_weight,
        "success_monotonic_projection": success_monotonic_projection,
        "success_endpoint_anchor": success_endpoint_anchor,
        "allowed_splits": allowed_splits,
        "use_token_features": use_token_features,
        "directional_alignment": directional_alignment,
        "directional_config": asdict(directional_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_score_cache(path: Path, signature: str) -> dict[str, dict[str, object]]:
    cached: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # A final partial line can be left by an interrupted process. All
            # complete records before it remain safe to resume.
            continue
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if item.get("signature") == signature and isinstance(score, dict) and score.get("traj_id"):
            cached[str(score["traj_id"])] = score
    return cached


def _append_score_cache(path: Path, signature: str, score: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"signature": signature, "score": score}, ensure_ascii=False) + "\n")


def score_manifest(
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
    gamma: float,
    mi_mode: str,
    relation_weight: float = 0.0,
    action_weight: float = 0.0,
    kinematic_weight: float = 0.0,
    outcome_weight: float = 0.0,
    success_monotonic_projection: bool = True,
    success_endpoint_anchor: bool = True,
    allowed_splits: tuple[str, ...] | None = ("train",),
    use_token_features: bool = False,
    directional_alignment: bool = False,
    directional_config: DirectionalScoreConfig | None = None,
    score_device: str | torch.device = "cpu",
    score_cache: str | Path | None = None,
) -> list[dict[str, object]]:
    if directional_alignment and not use_token_features:
        raise ValueError("Directional alignment requires native token features; pass --token_features.")
    device = _resolve_score_device(score_device)
    directional_estimator = (
        DameSoftHistogramMI(num_bins=8)
        if directional_alignment or action_weight or kinematic_weight
        else None
    )
    directional_config = directional_config or DirectionalScoreConfig(gamma=gamma)
    if directional_estimator is not None:
        directional_estimator = directional_estimator.to(device)
    store = CachedFeatureStore(feature_root)
    trajectories = read_jsonl(manifest, TrajectoryExample)
    refs_by_task = _group_by_task(read_jsonl(success_refs, SuccessReference))
    eligible = [
        traj
        for traj in trajectories
        if (allowed_splits is None or traj.split in allowed_splits)
        and (not _requires_verification(traj) or bool(traj.verification and traj.verification.accepted))
    ]
    cache_path = None if score_cache is None else Path(score_cache)
    signature = _score_cache_signature(
        manifest=manifest,
        success_refs=success_refs,
        feature_root=feature_root,
        gamma=gamma,
        mi_mode=mi_mode,
        relation_weight=relation_weight,
        action_weight=action_weight,
        kinematic_weight=kinematic_weight,
        outcome_weight=outcome_weight,
        success_monotonic_projection=success_monotonic_projection,
        success_endpoint_anchor=success_endpoint_anchor,
        allowed_splits=allowed_splits,
        use_token_features=use_token_features,
        directional_alignment=directional_alignment,
        directional_config=directional_config,
    )
    cached_scores = {} if cache_path is None else _load_score_cache(cache_path, signature)
    reference_tensor_cache: dict[str, torch.Tensor] = {}

    def load_reference(feature_id: str) -> torch.Tensor:
        if feature_id not in reference_tensor_cache:
            reference_tensor_cache[feature_id] = store.load(feature_id).to(device)
        return reference_tensor_cache[feature_id]

    scored: list[dict[str, object]] = []
    progress = tqdm(eligible, desc="Directional MI scoring", unit="trajectory")
    for traj in progress:
        cached = cached_scores.get(traj.traj_id)
        if cached is not None:
            scored.append(cached)
            progress.set_postfix_str(f"resumed={len(scored)}")
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
        candidate_features = store.load(feature_id).to(device)
        reference_features = {
            ref.ref_id: load_reference(ref.ref_id + "_tokens" if use_token_features else ref.ref_id)
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
            visual_progress = alignment_stage_potential(
                result.alignment_path,
                reference_features[result.selected_reference_id].shape[0],
            )
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
            raw_visual = torch.as_tensor(best["phi"], dtype=torch.float32, device=device)
            span = raw_visual.max() - raw_visual.min()
            visual_progress = (
                (raw_visual - raw_visual.min()) / span.clamp_min(1e-8)
                if raw_visual.numel() > 1
                else torch.zeros_like(raw_visual)
            )
        action_score = 0.0
        action_phi: list[float] | None = None
        action_progress: torch.Tensor | None = None
        action_confidence = 1.0
        if action_weight:
            assert directional_estimator is not None
            candidate_actions = store.load(traj.traj_id + "_action_latents").to(device)
            reference_actions = load_reference(best_ref.ref_id + "_action_latents")
            if candidate_actions.ndim != 3 or reference_actions.ndim != 3:
                raise ValueError(
                    f"Action MI requires [T-1,Q,D] caches for {traj.traj_id} and {best_ref.ref_id}."
                )
            if candidate_actions.shape[0] != candidate_features.shape[0] - 1:
                raise ValueError(
                    f"Action/frame length mismatch while scoring {traj.traj_id}: "
                    f"{candidate_actions.shape[0]} actions vs {candidate_features.shape[0]} frames."
                )
            action_result = score_candidate_trajectory(
                candidate_actions,
                {best_ref.ref_id: reference_actions},
                directional_estimator,
                directional_config,
            )
            action_score = float(action_result.directional_score)
            action_phi = [float(value) for value in action_result.phi.detach().cpu().tolist()]
            action_progress = transition_potential_to_frames(
                alignment_stage_potential(action_result.alignment_path, reference_actions.shape[0]),
                candidate_features.shape[0],
            )
            action_confidence = float(action_result.confidence)
        relation_score = 0.0
        relation_phi: list[float] | None = None
        if relation_weight:
            if not traj.relation_path:
                raise ValueError(f"relation_weight requires relation_path for {traj.traj_id}.")
            relation = load_relation_sequence(traj.relation_path)
            relation_values = relation_progress_potential(
                relation.values,
                relation.names,
                task_family=traj.task_family,
            ).to(device)
            if relation_values.shape[0] != candidate_features.shape[0]:
                raise ValueError(
                    f"Relation/frame length mismatch while scoring {traj.traj_id}: "
                    f"{relation_values.shape[0]} vs {candidate_features.shape[0]}."
                )
            relation_phi = [float(value) for value in relation_values.tolist()]
            if relation_values.numel() >= 2:
                relation_score = float((gamma * relation_values[1:] - relation_values[:-1]).mean().item())
        kinematic_score = 0.0
        kinematic_phi: list[float] | None = None
        kinematic_progress: torch.Tensor | None = None
        kinematic_confidence = 1.0
        if kinematic_weight:
            assert directional_estimator is not None
            candidate_kinematics = store.load(traj.traj_id + "_kinematic_latents").to(device)
            reference_kinematics = load_reference(best_ref.ref_id + "_kinematic_latents")
            if candidate_kinematics.ndim != 3 or reference_kinematics.ndim != 3:
                raise ValueError("Kinematic MI requires [T,Q,D] caches.")
            if candidate_kinematics.shape[0] != candidate_features.shape[0]:
                raise ValueError(
                    f"Kinematic/frame length mismatch while scoring {traj.traj_id}: "
                    f"{candidate_kinematics.shape[0]} vs {candidate_features.shape[0]}."
                )
            kinematic_result = score_candidate_trajectory(
                candidate_kinematics,
                {best_ref.ref_id: reference_kinematics},
                directional_estimator,
                directional_config,
            )
            kinematic_score = float(kinematic_result.directional_score)
            kinematic_phi = [float(value) for value in kinematic_result.phi.detach().cpu().tolist()]
            kinematic_progress = alignment_stage_potential(
                kinematic_result.alignment_path,
                reference_kinematics.shape[0],
            )
            kinematic_confidence = float(kinematic_result.confidence)
        task_success = None if traj.task_outcome is None else bool(traj.task_outcome.success)
        outcome_bonus = outcome_weight * float(task_success is True)
        teacher_phi = combine_process_potentials(
            visual_progress,
            action=action_progress,
            kinematic=kinematic_progress,
            relation=(relation_values if relation_weight else None),
            action_weight=action_weight,
            kinematic_weight=kinematic_weight,
            relation_weight=relation_weight,
            task_success=task_success,
            success_monotonic_projection=success_monotonic_projection,
            success_endpoint_anchor=success_endpoint_anchor,
        )
        curve_score = score_potential_curve(teacher_phi, directional_config)
        raw_score = float(curve_score["directional_score"])
        score: dict[str, object] = {
            "traj_id": traj.traj_id,
            "task": traj.task,
            "task_family": traj.task_family,
            "parent_traj_id": (
                traj.parent_traj_id
                or (None if traj.candidate_provenance is None else traj.candidate_provenance.parent_traj_id)
            ),
            "scene_variant_id": (
                None if traj.scene_variant is None else traj.scene_variant.variant_id
            ),
            "instance_target_category": (
                None if traj.instance_variant is None else traj.instance_variant.target_category
            ),
            "instance_variant_id": (
                None if traj.instance_variant is None else traj.instance_variant.variant_id
            ),
            "goal_ref_id": best_ref.ref_id,
            "split": traj.split,
            "task_success": task_success,
            "failure_mode": None if traj.task_outcome is None else traj.task_outcome.failure_mode,
            "candidate_profile": None if traj.task_outcome is None else traj.task_outcome.candidate_profile,
            "score_delta": raw_score + outcome_bonus,
            "raw_score_delta": raw_score,
            "outcome_bonus": outcome_bonus,
            "success_monotonic_projection": success_monotonic_projection,
            "success_endpoint_anchor": success_endpoint_anchor,
            "score_mean": float(best["score_mean"]),
            "phi": best["phi"],
            "visual_score": float(best["score_delta"]),
            "visual_progress_phi": visual_progress.detach().cpu().tolist(),
            "action_phi": action_phi,
            "action_score": action_score,
            "action_progress_phi": (
                None if action_progress is None else action_progress.detach().cpu().tolist()
            ),
            "kinematic_phi": kinematic_phi,
            "kinematic_score": kinematic_score,
            "kinematic_progress_phi": (
                None if kinematic_progress is None else kinematic_progress.detach().cpu().tolist()
            ),
            "relation_phi": relation_phi,
            "relation_score": relation_score,
            "teacher_phi": teacher_phi.detach().cpu().tolist(),
            "teacher_endpoint_progress": curve_score["endpoint_progress"],
            "teacher_positive_gain": curve_score["positive_gain"],
            "teacher_regression_penalty": curve_score["regression_penalty"],
            "confidence": min(
                float(best.get("confidence", 1.0)), action_confidence, kinematic_confidence
            ),
            "directional_process_latent": {
                "visual": "normalized_monotonic_alignment_stage[T]",
                "action": "lawam_normalized_alignment_stage[T]",
                "kinematic": "kinematic_normalized_alignment_stage[T]",
                "relation": "bounded_phase_aware_relation_progress[T]",
                "ranking": (
                    "directional_score(weighted_bounded_teacher_phi) + outcome_anchor"
                ),
            },
        }
        scored.append(score)
        if cache_path is not None:
            _append_score_cache(cache_path, signature, score)
    return scored


def build_outcome_anchored_pairs(
    scored: list[dict[str, object]],
    margin: float,
    max_pairs_per_task: int | None = None,
    min_confidence: float = 0.0,
    teacher_version: str = TEACHER_VERSION_DAME_ALIGNED,
    score_field: str = "score_delta",
    seed: int = 0,
    pair_scope: str = "same_context",
    use_measured_outcomes: bool = True,
) -> list[PreferencePair]:
    """Build measured-outcome and fine-grained pairs within fair contexts.

    ``same_context`` keeps task, split, parent seed, instance and scene fixed.
    It includes success-vs-failure supervision plus score-separated pairs
    within the success and failure sets. Legacy/unscoped records fall back to
    task+goal grouping so external datasets remain usable.
    """

    if pair_scope not in {"same_context", "task"}:
        raise ValueError("pair_scope must be 'same_context' or 'task'.")

    def group_key(item: dict[str, object]) -> tuple[object, ...]:
        base: tuple[object, ...] = (item.get("task"), item.get("goal_ref_id"))
        if pair_scope == "task" or not item.get("parent_traj_id"):
            return base
        return base + (
            item.get("split"),
            item.get("parent_traj_id"),
            item.get("instance_variant_id"),
            item.get("scene_variant_id"),
        )

    def context_id(key: tuple[object, ...]) -> str:
        return "|".join("" if value is None else str(value) for value in key)

    rng = random.Random(seed)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for item in scored:
        grouped.setdefault(group_key(item), []).append(item)
    pairs: list[PreferencePair] = []
    for key, items in grouped.items():
        task, goal_ref_id = str(key[0]), None if key[1] is None else str(key[1])
        ranked = sorted(items, key=lambda item: float(item.get(score_field, 0.0)), reverse=True)
        task_pairs: list[PreferencePair] = []
        for chosen_index, chosen in enumerate(ranked):
            for rejected in ranked[chosen_index + 1 :]:
                pair_chosen, pair_rejected = chosen, rejected
                chosen_score = float(pair_chosen.get(score_field, 0.0))
                rejected_score = float(pair_rejected.get(score_field, 0.0))
                chosen_success = pair_chosen.get("task_success")
                rejected_success = pair_rejected.get("task_success")
                # Measured outcome is the primary anchor even if a noisy raw
                # component happens to reverse the scalar scores.
                if use_measured_outcomes and chosen_success is False and rejected_success is True:
                    pair_chosen, pair_rejected = pair_rejected, pair_chosen
                    chosen_score, rejected_score = rejected_score, chosen_score
                    chosen_success, rejected_success = True, False
                if use_measured_outcomes and chosen_success is True and rejected_success is False:
                    # The measured outcome is authoritative for cross-outcome
                    # pairs. Keep a strictly positive training margin even if
                    # an imperfect process score rates a near miss too highly.
                    chosen_score = max(chosen_score, rejected_score + margin + 1e-6)
                chosen_conf = float(pair_chosen.get("confidence", 1.0))
                rejected_conf = float(pair_rejected.get("confidence", 1.0))
                if min(chosen_conf, rejected_conf) < min_confidence:
                    continue
                if chosen_score <= rejected_score + margin:
                    continue
                comparison_type = "score_only"
                if use_measured_outcomes:
                    comparison_type = (
                        "success_vs_failure"
                        if chosen_success is True and rejected_success is False
                        else "within_success"
                        if chosen_success is True and rejected_success is True
                        else "within_failure"
                        if chosen_success is False and rejected_success is False
                        else "score_only"
                    )
                task_pairs.append(
                    PreferencePair(
                        task=task,
                        chosen_traj_id=str(pair_chosen["traj_id"]),
                        rejected_traj_id=str(pair_rejected["traj_id"]),
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                        score_type=(
                            f"outcome_anchored_directional_mi_{teacher_version}"
                            if use_measured_outcomes
                            else f"score_only_directional_mi_{teacher_version}"
                        ),
                        goal_ref_id=goal_ref_id,
                        chosen_confidence=chosen_conf,
                        rejected_confidence=rejected_conf,
                        teacher_version=teacher_version,
                        split=(
                            None if pair_chosen.get("split") is None else str(pair_chosen["split"])
                        ),
                        context_id=context_id(key),
                        comparison_type=comparison_type,
                    )
                )
        if max_pairs_per_task is not None and len(task_pairs) > max_pairs_per_task:
            task_pairs = rng.sample(task_pairs, max_pairs_per_task)
        pairs.extend(task_pairs)
    return pairs


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


def write_directional_teacher_targets(
    scored: list[dict[str, object]],
    output: str | Path,
    *,
    min_success_endpoint_gain: float = 0.05,
    require_success_monotonic: bool = True,
) -> dict[str, object]:
    """Persist the exact bounded curves used for preference construction."""

    targets: dict[tuple[str, str], torch.Tensor] = {}
    success_gains: list[float] = []
    for item in scored:
        traj_id = str(item["traj_id"])
        goal_ref_id = str(item["goal_ref_id"])
        phi = torch.as_tensor(item.get("teacher_phi"), dtype=torch.float32)
        if phi.ndim != 1 or phi.numel() < 2 or not torch.isfinite(phi).all():
            raise ValueError(f"Invalid directional teacher curve for {traj_id}: {tuple(phi.shape)}")
        if float(phi.min()) < -1e-6 or float(phi.max()) > 1.0 + 1e-6:
            raise ValueError(f"Directional teacher curve leaves [0,1] for {traj_id}.")
        gain = float((phi[-1] - phi[0]).item())
        if item.get("task_success") is True:
            success_gains.append(gain)
            if gain < min_success_endpoint_gain:
                raise ValueError(
                    f"Successful trajectory {traj_id} has endpoint gain {gain:.4f}; "
                    f"required >= {min_success_endpoint_gain:.4f}."
                )
            if require_success_monotonic and bool((phi[1:] + 1e-6 < phi[:-1]).any()):
                raise ValueError(f"Successful trajectory {traj_id} has a regressing teacher curve.")
        targets[(traj_id, goal_ref_id)] = phi

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "target_type": "bounded_directional_process_potential",
        "targets": targets,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path),
        "targets": len(targets),
        "successful_targets": len(success_gains),
        "min_success_endpoint_gain": (
            None if not success_gains else min(success_gains)
        ),
        "mean_success_endpoint_gain": (
            None if not success_gains else sum(success_gains) / len(success_gains)
        ),
    }


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
    parser.add_argument(
        "--pair_mode",
        default="outcome_anchored",
        choices=["outcome_anchored", "score_only", "top_vs_bottom", "adjacent"],
    )
    parser.add_argument(
        "--pair-scope",
        default="same_context",
        choices=["same_context", "task"],
        help="Keep comparisons within parent seed/instance/scene by default.",
    )
    parser.add_argument("--teacher_version", default=TEACHER_VERSION_LEGACY)
    parser.add_argument("--score_field", default="score_delta")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mi_mode", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi"])
    parser.add_argument("--relation_weight", type=float, default=0.0)
    parser.add_argument(
        "--action_weight",
        type=float,
        default=0.0,
        help="Weight of temporally aligned LaWAM action-latent MI in the privileged teacher score.",
    )
    parser.add_argument("--kinematic_weight", type=float, default=0.0)
    parser.add_argument("--outcome_weight", type=float, default=2.0)
    parser.add_argument(
        "--disable-success-monotonic-projection",
        action="store_true",
        help="Ablation only: do not cummax successful privileged-teacher curves.",
    )
    parser.add_argument(
        "--disable-success-endpoint-anchor",
        action="store_true",
        help="Ablation only: do not force successful privileged-teacher endpoints to one.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Manifest split to score; repeat to include more than one. Default: train.",
    )
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
    parser.add_argument(
        "--score-device",
        default="auto",
        help="Device for MI scoring (default: auto, preferring CUDA).",
    )
    parser.add_argument(
        "--mi-pair-chunk-size",
        type=int,
        default=4,
        help="Number of frame pairs evaluated together; lower this if CUDA runs out of memory.",
    )
    parser.add_argument(
        "--score-cache",
        default=None,
        help="Per-trajectory resume cache (default: <output stem>.scores.jsonl).",
    )
    parser.add_argument(
        "--teacher-target-output",
        default=None,
        help="Exact bounded per-frame teacher target file (default: <output stem>.teacher_targets.pt).",
    )
    parser.add_argument("--min-success-endpoint-gain", type=float, default=0.05)
    args = parser.parse_args()
    if args.mi_pair_chunk_size < 1:
        parser.error("--mi-pair-chunk-size must be positive")

    score_cache = args.score_cache or str(
        Path(args.output).with_name(Path(args.output).stem + ".scores.jsonl")
    )

    scored = score_manifest(
        args.manifest, args.success_refs, args.feature_root, args.gamma, args.mi_mode,
        relation_weight=args.relation_weight, action_weight=args.action_weight,
        kinematic_weight=args.kinematic_weight, outcome_weight=args.outcome_weight,
        success_monotonic_projection=not args.disable_success_monotonic_projection,
        success_endpoint_anchor=not args.disable_success_endpoint_anchor,
        allowed_splits=tuple(args.split or ["train"]),
        use_token_features=args.token_features,
        directional_alignment=args.directional_alignment,
        score_device=args.score_device,
        score_cache=score_cache,
        directional_config=DirectionalScoreConfig(
            gamma=args.gamma,
            w_endpoint=args.directional_w_endpoint,
            w_positive=args.directional_w_positive,
            w_regression=args.directional_w_regression,
            w_stage=args.directional_w_stage,
            w_alignment=args.directional_w_alignment,
            mi_pair_chunk_size=args.mi_pair_chunk_size,
        ),
    )

    if args.pair_mode in {"outcome_anchored", "score_only"}:
        pairs = build_outcome_anchored_pairs(
            scored, margin=args.margin, max_pairs_per_task=args.max_pairs_per_task,
            min_confidence=args.min_confidence, teacher_version=args.teacher_version,
            score_field=args.score_field, seed=args.seed, pair_scope=args.pair_scope,
            use_measured_outcomes=args.pair_mode == "outcome_anchored",
        )
    elif args.pair_mode == "adjacent":
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
    teacher_target_output = args.teacher_target_output or str(
        Path(args.output).with_name(Path(args.output).stem + ".teacher_targets.pt")
    )
    target_report = write_directional_teacher_targets(
        scored,
        teacher_target_output,
        min_success_endpoint_gain=args.min_success_endpoint_gain,
        require_success_monotonic=not args.disable_success_monotonic_projection,
    )
    report_path = Path(args.output).with_name("score_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pair_types: dict[str, int] = {}
    pair_splits: dict[str, int] = {}
    for pair in pairs:
        pair_types[str(pair.comparison_type or "unknown")] = pair_types.get(
            str(pair.comparison_type or "unknown"), 0
        ) + 1
        pair_splits[str(pair.split or "unknown")] = pair_splits.get(str(pair.split or "unknown"), 0) + 1
    report_path.write_text(
        json.dumps(
            {
                "scores": scored,
                "num_pairs": len(pairs),
                "pair_scope": args.pair_scope,
                "pair_types": pair_types,
                "pair_splits": pair_splits,
                "teacher_targets": target_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
