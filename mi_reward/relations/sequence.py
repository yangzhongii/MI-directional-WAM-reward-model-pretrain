"""Loading and scoring synchronized measured relation sequences."""

from __future__ import annotations

import json
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


def relation_progress_potential(
    relations: torch.Tensor,
    names: list[str],
    fields: tuple[str, ...] = DEFAULT_PROGRESS_FIELDS,
) -> torch.Tensor:
    """Return a scale-normalized potential where decreasing errors mean progress.

    This is a deterministic teacher-side signal. It does not turn kinematics
    into a visual latent or invent robot state from generated pixels.
    """

    if relations.ndim != 2:
        raise ValueError(f"Expected relation tensor [T, R], got {tuple(relations.shape)}")
    index = {name: i for i, name in enumerate(names)}
    selected = [index[name] for name in fields if name in index]
    if not selected:
        raise ValueError(f"No progress fields {fields} appear in relation names {names}.")
    errors = relations[:, selected].float()
    scale = errors[0].abs().clamp_min(1e-4)
    return -(errors / scale).mean(dim=-1)
