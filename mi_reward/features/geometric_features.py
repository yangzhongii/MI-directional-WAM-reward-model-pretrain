"""Privileged geometric features for Pipeline-v7 diagnostics only.

These functions consume positions recorded by the simulator and must never be
used as an observation-only or deployable reward feature.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def extract_eef_object_goal_relation(sim_state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return non-redundant relative XYZ vectors from a recorded physical state."""
    eef = np.asarray(sim_state["eef_pos"], dtype=np.float64)
    obj = np.asarray(sim_state["task_object_pos"], dtype=np.float64)
    goal = np.asarray(sim_state["goal_object_pos"], dtype=np.float64)
    return {"eef_object": eef - obj, "object_goal": obj - goal}


def extract_geometric_latent(sim_state: dict[str, Any]) -> np.ndarray:
    """Return the non-redundant 6D relation vector ``[eef-object, object-goal]``."""
    relation = extract_eef_object_goal_relation(sim_state)
    return np.concatenate((relation["eef_object"], relation["object_goal"]))


def extract_geometric_delta(sim_state_a: dict[str, Any], sim_state_b: dict[str, Any]) -> np.ndarray:
    """Return the actual privileged geometric-latent delta from A to B."""
    return extract_geometric_latent(sim_state_b) - extract_geometric_latent(sim_state_a)


def geometric_progress_score(sim_state: dict[str, Any]) -> float:
    """Direct privileged pre-contact progress: negative EEF-to-object distance."""
    relation = extract_eef_object_goal_relation(sim_state)
    return -float(np.linalg.norm(relation["eef_object"]))
