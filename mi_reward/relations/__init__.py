"""Measured geometry and relation utilities for GeoProgress."""

from mi_reward.relations.geometry import RobotState, TaskGeometry, build_relation_descriptor
from mi_reward.relations.sequence import (
    RelationPotentialConfig,
    RelationSequence,
    load_relation_sequence,
    relation_progress_potential,
)

__all__ = [
    "RobotState",
    "TaskGeometry",
    "build_relation_descriptor",
    "RelationPotentialConfig",
    "RelationSequence",
    "load_relation_sequence",
    "relation_progress_potential",
]
