"""Runtime inference utilities for MI potential reward models.

Provides a stable inference contract for loading frozen state-potential
student checkpoints and running batched inference without gradients.
"""
"""Inference wrappers for reward potentials."""

from mi_reward.inference.geoprogress_model import GeoProgressInferenceModel
from mi_reward.inference.potential_model import MIPotentialInferenceModel

__all__ = ["GeoProgressInferenceModel", "MIPotentialInferenceModel"]
