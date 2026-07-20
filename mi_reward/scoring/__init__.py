"""Scoring module exports for MI reward pretraining."""

from mi_reward.scoring.mi_potential import compute_mi
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.scoring.token_correspondence import apply_correspondence
from mi_reward.scoring.directional_potential import (
    DirectionalPotentialResult,
    DirectionalScoreConfig,
    score_candidate_trajectory,
    compute_directional_score_from_pooled,
)
from mi_reward.scoring.trajectory_score import (
    compute_frame_potential,
    compute_delta_score,
    score_trajectory,
)
from mi_reward.scoring.build_preferences import score_manifest, build_preference_pairs

__all__ = [
    "compute_mi",
    "DameSoftHistogramMI",
    "apply_correspondence",
    "DirectionalPotentialResult",
    "DirectionalScoreConfig",
    "score_candidate_trajectory",
    "compute_directional_score_from_pooled",
    "compute_frame_potential",
    "compute_delta_score",
    "score_trajectory",
    "score_manifest",
    "build_preference_pairs",
]
