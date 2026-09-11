"""Directional information potential scoring with temporal alignment.

Computes a multi-component directional progress score from candidate-reference
MI alignment, decomposing trajectory quality into endpoint progress, positive
gain, regression penalty, stage progress, and mean alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mi_reward.alignment.monotonic_alignment import build_mi_alignment_matrix, monotonic_viterbi_alignment


@dataclass
class DirectionalPotentialResult:
    """Complete result of directional potential scoring for one candidate trajectory."""

    alignment_matrix: torch.Tensor  # [T, S]
    alignment_path: torch.Tensor  # [T] indices into reference
    phi: torch.Tensor  # [T] aligned MI potential per frame

    # Score components
    endpoint_progress: float
    positive_gain: float
    regression_penalty: float
    stage_progress: float
    mean_alignment: float

    # Composite scores
    directional_score: float
    legacy_delta_score: float

    # Metadata
    selected_reference_id: str | None = None
    confidence: float = 1.0


@dataclass
class DirectionalScoreConfig:
    """Configuration for directional score computation."""

    # Score weights
    w_endpoint: float = 1.0
    w_positive: float = 0.5
    w_regression: float = 1.0
    w_stage: float = 1.0
    w_alignment: float = 0.1

    # Legacy delta score
    gamma: float = 0.99

    # Alignment
    alignment_method: str = "monotonic_viterbi"
    max_step: int | None = None
    stay_penalty: float = 0.0
    jump_penalty: float = 0.01
    multi_reference_aggregation: str = "best"  # "best", "logsumexp", "mean"
    mi_pair_chunk_size: int = 1


def alignment_stage_potential(
    alignment_path: torch.Tensor,
    reference_length: int,
) -> torch.Tensor:
    """Convert a monotonic alignment path into bounded reference progress."""

    if alignment_path.ndim != 1:
        raise ValueError(f"alignment_path must be one-dimensional, got {tuple(alignment_path.shape)}")
    if reference_length <= 1:
        return torch.zeros_like(alignment_path, dtype=torch.float32)
    return alignment_path.float().div(float(reference_length - 1)).clamp(0.0, 1.0)


def transition_potential_to_frames(values: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Map ``T-1`` transition progress values onto ``T`` observation frames."""

    if values.ndim != 1 or values.shape[0] != frame_count - 1:
        raise ValueError(
            f"Transition potential must have shape [{frame_count - 1}], got {tuple(values.shape)}"
        )
    framed = torch.empty(frame_count, dtype=values.dtype, device=values.device)
    framed[0] = values[0]
    framed[1:] = values
    return framed


def combine_process_potentials(
    visual: torch.Tensor,
    *,
    action: torch.Tensor | None = None,
    kinematic: torch.Tensor | None = None,
    relation: torch.Tensor | None = None,
    action_weight: float = 0.0,
    kinematic_weight: float = 0.0,
    relation_weight: float = 0.0,
    task_success: bool | None = None,
    success_monotonic_projection: bool = True,
    success_endpoint_anchor: bool = True,
) -> torch.Tensor:
    """Fuse bounded process latents without allowing one channel to explode.

    Every component is interpreted in ``[0, 1]`` and the result is a weighted
    average. Successful offline demonstrations are projected onto a monotonic
    curve and anchored at one at the measured terminal success frame.
    """

    if visual.ndim != 1 or visual.numel() < 2:
        raise ValueError("visual process potential must be a one-dimensional trajectory with T >= 2.")
    total = visual.float().clamp(0.0, 1.0)
    denominator = 1.0
    for name, value, weight in (
        ("action", action, action_weight),
        ("kinematic", kinematic, kinematic_weight),
        ("relation", relation, relation_weight),
    ):
        if weight < 0:
            raise ValueError(f"{name}_weight must be non-negative.")
        if not weight:
            continue
        if value is None or value.ndim != 1 or value.shape != visual.shape:
            shape = None if value is None else tuple(value.shape)
            raise ValueError(f"{name} process potential must match visual shape {tuple(visual.shape)}, got {shape}.")
        total = total + float(weight) * value.float().clamp(0.0, 1.0)
        denominator += float(weight)
    combined = (total / denominator).clamp(0.0, 1.0)
    if task_success is True:
        if success_monotonic_projection:
            combined = torch.cummax(combined, dim=0).values
        if success_endpoint_anchor:
            combined = combined.clone()
            combined[-1] = 1.0
    return combined


def score_potential_curve(phi: torch.Tensor, config: DirectionalScoreConfig) -> dict[str, float]:
    """Score the exact bounded curve that will supervise the student."""

    if phi.ndim != 1 or phi.numel() < 2:
        raise ValueError("Directional teacher curve must be one-dimensional with T >= 2.")
    diffs = phi[1:] - phi[:-1]
    endpoint = float((phi[-1] - phi[0]).item())
    positive = float(F.relu(diffs).mean().item())
    regression = float(F.relu(-diffs).mean().item())
    mean = float(phi.mean().item())
    score = (
        config.w_endpoint * endpoint
        + config.w_positive * positive
        - config.w_regression * regression
        + config.w_stage * endpoint
        + config.w_alignment * mean
    )
    return {
        "directional_score": score,
        "endpoint_progress": endpoint,
        "positive_gain": positive,
        "regression_penalty": regression,
        "stage_progress": endpoint,
        "mean_alignment": mean,
        "legacy_delta_score": float((config.gamma * phi[1:] - phi[:-1]).mean().item()),
    }


def _compute_score_components(
    phi: torch.Tensor,
    alignment_path: torch.Tensor,
    reference_length: int,
    config: DirectionalScoreConfig,
) -> tuple[float, float, float, float, float, float]:
    """Compute directional score components from aligned MI potentials.

    Args:
        phi: [T] aligned MI potential values
        alignment_path: [T] alignment indices into reference
        reference_length: S (number of reference frames)
        config: score weights and parameters

    Returns:
        endpoint_progress, positive_gain, regression_penalty,
        stage_progress, mean_alignment, legacy_delta
    """
    T = phi.numel()
    device = phi.device

    if T < 2:
        phi_mean = float(phi.mean().item()) if T > 0 else 0.0
        return 0.0, 0.0, 0.0, 0.0, phi_mean, 0.0

    # Endpoint progress: Phi_{T-1} - Phi_0
    endpoint_progress = float((phi[-1] - phi[0]).item())

    # Positive gain: mean of relu(Phi_{t+1} - Phi_t)
    diffs = phi[1:] - phi[:-1]
    positive_gain = float(F.relu(diffs).mean().item())

    # Regression penalty: mean of relu(Phi_t - Phi_{t+1})
    regression_penalty = float(F.relu(-diffs).mean().item())

    # Stage progress: (pi(T-1) - pi(0)) / (S - 1)
    if reference_length > 1:
        stage_progress = float(
            (alignment_path[-1].item() - alignment_path[0].item())
            / max(reference_length - 1, 1)
        )
    else:
        stage_progress = 0.0

    # Mean alignment quality
    mean_alignment = float(phi.mean().item())

    # Legacy delta score
    gamma = config.gamma
    legacy_delta = float(((gamma * phi[1:] - phi[:-1]).mean()).item())

    return endpoint_progress, positive_gain, regression_penalty, stage_progress, mean_alignment, legacy_delta


def score_candidate_trajectory(
    candidate_features: torch.Tensor,
    success_references: dict[str, torch.Tensor],
    mi_estimator: nn.Module,
    config: DirectionalScoreConfig,
) -> DirectionalPotentialResult:
    """Score a candidate trajectory against success references.

    Computes temporally-aligned MI potentials, extracts the best
    reference alignment, and returns a detailed directional score.

    Args:
        candidate_features: [T, N, D] candidate token features
        success_references: dict ref_id -> [S, N, D] reference token features
        mi_estimator: DameSoftHistogramMI instance
        config: directional score configuration

    Returns:
        DirectionalPotentialResult with scores and metadata
    """
    if config.multi_reference_aggregation not in {"best", "mean", "logsumexp"}:
        raise ValueError(
            "multi_reference_aggregation must be one of 'best', 'mean', or 'logsumexp'."
        )
    T = candidate_features.shape[0]

    best_result = None
    best_total = float("-inf")
    ref_scores = []

    for ref_id, ref_features in success_references.items():
        S = ref_features.shape[0]

        # 1. Build MI alignment matrix [T, S]
        alignment_matrix = build_mi_alignment_matrix(
            candidate_features,
            ref_features,
            mi_estimator,
            pair_chunk_size=config.mi_pair_chunk_size,
        )

        # 2. Find monotonic alignment path
        if config.alignment_method == "monotonic_viterbi":
            alignment_result = monotonic_viterbi_alignment(
                alignment_matrix,
                max_step=config.max_step,
                stay_penalty=config.stay_penalty,
                jump_penalty=config.jump_penalty,
            )
        elif config.alignment_method == "soft_dtw":
            from mi_reward.alignment.monotonic_alignment import soft_dtw_alignment

            alignment_result = soft_dtw_alignment(alignment_matrix)
        else:
            # Fallback: framewise max
            max_per_frame, path = alignment_matrix.max(dim=1)
            alignment_result = {
                "path": path,
                "aligned_potential": max_per_frame,
                "total_score": max_per_frame.sum(),
            }

        phi = alignment_result["aligned_potential"]
        path = alignment_result["path"]
        total_score = alignment_result["total_score"]

        # 3. Compute score components
        endpoint_progress, positive_gain, regression_penalty, stage_progress, mean_alignment, legacy_delta = (
            _compute_score_components(phi, path, S, config)
        )

        # 4. Weighted directional score
        directional_score = (
            config.w_endpoint * endpoint_progress
            + config.w_positive * positive_gain
            - config.w_regression * regression_penalty
            + config.w_stage * stage_progress
            + config.w_alignment * mean_alignment
        )

        result = DirectionalPotentialResult(
            alignment_matrix=alignment_matrix,
            alignment_path=path,
            phi=phi,
            endpoint_progress=endpoint_progress,
            positive_gain=positive_gain,
            regression_penalty=regression_penalty,
            stage_progress=stage_progress,
            mean_alignment=mean_alignment,
            directional_score=directional_score,
            legacy_delta_score=legacy_delta,
            selected_reference_id=ref_id,
        )

        ref_scores.append((directional_score, result))

        if directional_score > best_total:
            best_total = directional_score
            best_result = result

    if best_result is None and ref_scores:
        best_result = ref_scores[0][1]

    if best_result is not None and ref_scores and config.multi_reference_aggregation != "best":
        scores_tensor = torch.tensor(
            [score for score, _ in ref_scores], device=candidate_features.device, dtype=torch.float32
        )
        if config.multi_reference_aggregation == "mean":
            aggregate_score = float(scores_tensor.mean().item())
        else:
            aggregate_score = float(torch.logsumexp(scores_tensor, dim=0).item())
        # Keep the highest-scoring alignment path as the representative path,
        # but expose the requested multi-reference aggregate as its score.
        best_result.directional_score = aggregate_score

    if best_result is None:
        # No references matched — return zero result
        zero_phi = torch.zeros(T, device=candidate_features.device)
        zero_path = torch.zeros(T, dtype=torch.long, device=candidate_features.device)
        zero_mat = torch.zeros(T, 1, device=candidate_features.device)
        best_result = DirectionalPotentialResult(
            alignment_matrix=zero_mat,
            alignment_path=zero_path,
            phi=zero_phi,
            endpoint_progress=0.0,
            positive_gain=0.0,
            regression_penalty=0.0,
            stage_progress=0.0,
            mean_alignment=0.0,
            directional_score=0.0,
            legacy_delta_score=0.0,
        )

    # Compute confidence from multi-reference agreement
    if len(ref_scores) > 1:
        scores_tensor = torch.tensor([s for s, _ in ref_scores])
        # Higher confidence when one reference clearly dominates
        best_score = scores_tensor.max()
        softmax_weights = F.softmax(scores_tensor, dim=0)
        max_weight = softmax_weights.max()
        # Confidence: consistency across references
        confidence = float(max_weight.item())
        best_result.confidence = confidence
    else:
        best_result.confidence = 1.0

    return best_result


def compute_directional_score_from_pooled(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
    mi_estimator: nn.Module,
    config: DirectionalScoreConfig,
) -> DirectionalPotentialResult:
    """Convenience wrapper when features are already [T, D] pooled vectors.

    Adds a dummy N=1 token dimension before calling score_candidate_trajectory.
    """
    if candidate_features.dim() == 2:
        candidate_features = candidate_features.unsqueeze(1)  # [T, 1, D]

    refs = {}
    if reference_features.dim() == 2:
        reference_features = reference_features.unsqueeze(1)  # [S, 1, D]
    refs["reference_0"] = reference_features

    return score_candidate_trajectory(candidate_features, refs, mi_estimator, config)
