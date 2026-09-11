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
from mi_reward.scoring.mi_potential_field import (
    MIPotentialField,
    MIBackend,
    BaseLatentEncoder,
    PotentialResult,
    DINOv3LatentEncoder,
    latent_distance_reward,
    cosine_similarity_reward,
    REWARD_FUNCTIONS,
)
from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    InformationTeacherEvidence,
    VisualPointwiseInformationCritic,
    balanced_density_ratio_loss,
    compute_information_teacher_evidence,
    conditional_action_information_loss,
    directional_information_score,
    visual_information_loss,
)


def __getattr__(name: str):
    # Keep the preference helpers available from ``mi_reward.scoring`` without
    # importing the CLI module while ``python -m ...build_preferences`` is
    # starting. Eager import caused runpy's duplicate-module warning.
    if name in {"score_manifest", "build_preference_pairs"}:
        from mi_reward.scoring import build_preferences

        return getattr(build_preferences, name)
    raise AttributeError(name)

__all__ = [
    # Legacy
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
    # New unified API
    "MIPotentialField",
    "MIBackend",
    "BaseLatentEncoder",
    "PotentialResult",
    "DINOv3LatentEncoder",
    "latent_distance_reward",
    "cosine_similarity_reward",
    "REWARD_FUNCTIONS",
    # Canonical v3 information-teacher core
    "VisualPointwiseInformationCritic",
    "ConditionalActionInformationCritic",
    "InformationTeacherEvidence",
    "balanced_density_ratio_loss",
    "visual_information_loss",
    "conditional_action_information_loss",
    "directional_information_score",
    "compute_information_teacher_evidence",
]
