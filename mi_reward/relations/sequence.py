"""Loading and scoring synchronized measured relation sequences."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mi_reward.data.schema import RelationDescriptor


@dataclass(frozen=True)
class RelationSequence:
    names: list[str]
    values: torch.Tensor  # [T, R], CPU float32

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError("RelationSequence values must have shape [T, len(names)].")


def _records_from_path(path: str | Path) -> list[Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Relation sequence not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Relation sequence is empty: {source}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("relations", "frames", "records", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"Unsupported relation JSON root in {source}: {type(payload).__name__}")


def load_relation_sequence(path: str | Path) -> RelationSequence:
    """Load ordered ``RelationDescriptor`` records from JSON or JSONL.

    Each entry may be a descriptor directly or ``{"relation": descriptor}``.
    The name order must remain constant across time; this is deliberately a
    hard error because a silent permutation changes reward semantics.
    """

    descriptors: list[RelationDescriptor] = []
    for record in _records_from_path(path):
        if not isinstance(record, dict):
            raise ValueError(f"Relation record in {path} must be a JSON object.")
        payload = record.get("relation", record)
        if not isinstance(payload, dict):
            raise ValueError(f"Relation record in {path} contains a non-object relation.")
        descriptors.append(RelationDescriptor.from_dict(payload))
    if not descriptors:
        raise ValueError(f"Relation sequence is empty: {path}")
    names = descriptors[0].names
    if not names:
        raise ValueError(f"Relation sequence has no named fields: {path}")
    for step, descriptor in enumerate(descriptors[1:], start=1):
        if descriptor.names != names:
            raise ValueError(
                f"Relation field order differs at timestep {step} in {path}; expected {names}, got {descriptor.names}."
            )
    return RelationSequence(
        names=list(names),
        values=torch.tensor([item.values for item in descriptors], dtype=torch.float32),
    )


DEFAULT_PROGRESS_FIELDS = (
    "object_to_goal_distance",
    "ee_to_object_distance",
    "object_goal_orientation_error",
)


@dataclass(frozen=True)
class RelationPotentialConfig:
    """Physical scales for a bounded, phase-aware relation potential."""

    object_goal_distance_scale: float = 0.50
    ee_object_distance_scale: float = 0.25
    orientation_error_scale: float = math.pi
    approach_weight: float = 0.20
    peg_orientation_weight: float = 0.20

    def __post_init__(self) -> None:
        if min(
            self.object_goal_distance_scale,
            self.ee_object_distance_scale,
            self.orientation_error_scale,
        ) <= 0:
            raise ValueError("Relation potential scales must be positive.")
        if not 0.0 <= self.approach_weight < 1.0:
            raise ValueError("approach_weight must be in [0, 1).")
        if not 0.0 <= self.peg_orientation_weight < 1.0:
            raise ValueError("peg_orientation_weight must be in [0, 1).")


def _bounded_progress(error: torch.Tensor, scale: float) -> torch.Tensor:
    return 1.0 - (error.float().abs() / float(scale)).clamp(0.0, 1.0)


def relation_progress_potential(
    relations: torch.Tensor,
    names: list[str],
    fields: tuple[str, ...] = DEFAULT_PROGRESS_FIELDS,
    *,
    task_family: str | None = None,
    config: RelationPotentialConfig | None = None,
) -> torch.Tensor:
    """Return a bounded potential where decreasing task errors mean progress.

    The previous implementation divided every relation by its first-frame
    value. Small initial orientation errors therefore amplified harmless
    motion by thousands of times and dominated visual/action MI. This version
    uses fixed physical units and a stage latch: after grasp/contact, retreat
    motion cannot erase object-to-goal progress merely because the end
    effector moves away from the object.
    """

    if relations.ndim != 2:
        raise ValueError(f"Expected relation tensor [T, R], got {tuple(relations.shape)}")
    index = {name: i for i, name in enumerate(names)}
    selected = [name for name in fields if name in index]
    if not selected:
        raise ValueError(f"No progress fields {fields} appear in relation names {names}.")
    cfg = config or RelationPotentialConfig()

    goal_progress = (
        _bounded_progress(relations[:, index["object_to_goal_distance"]], cfg.object_goal_distance_scale)
        if "object_to_goal_distance" in index
        else torch.zeros(relations.shape[0], dtype=torch.float32, device=relations.device)
    )
    approach_progress = (
        _bounded_progress(relations[:, index["ee_to_object_distance"]], cfg.ee_object_distance_scale)
        if "ee_to_object_distance" in index
        else goal_progress
    )

    family = str(task_family or "").lower()
    goal_quality = goal_progress
    if family == "peg_insertion" and "object_goal_orientation_error" in index:
        orientation = _bounded_progress(
            relations[:, index["object_goal_orientation_error"]], cfg.orientation_error_scale
        )
        goal_quality = (
            (1.0 - cfg.peg_orientation_weight) * goal_progress
            + cfg.peg_orientation_weight * orientation
        )

    if family == "push_shape":
        potential = goal_quality
    elif "object_in_gripper" in index:
        acquired = torch.cummax(
            (relations[:, index["object_in_gripper"]] >= 0.5).float(), dim=0
        ).values
        pre_acquisition = cfg.approach_weight * approach_progress
        post_acquisition = cfg.approach_weight + (1.0 - cfg.approach_weight) * goal_quality
        potential = torch.where(acquired > 0.5, post_acquisition, pre_acquisition)
    else:
        # Generic/legacy relation files do not expose a phase latch. Keep the
        # task goal primary and use approach only as a bounded auxiliary term.
        potential = (
            (1.0 - cfg.approach_weight) * goal_quality
            + cfg.approach_weight * approach_progress
        )

    if "target_satisfied" in index:
        completed = torch.cummax(
            (relations[:, index["target_satisfied"]] >= 0.5).float(), dim=0
        ).values
        potential = torch.maximum(potential, completed)
    return potential.clamp(0.0, 1.0)
