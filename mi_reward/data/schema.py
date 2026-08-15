from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar


@dataclass(frozen=True)
class RelationDescriptor:
    """Task-relevant geometry measured from synchronized robot observations.

    ``names`` makes stored vectors self-describing, which prevents a reward
    checkpoint from silently consuming a relation vector with a different
    coordinate order.
    """

    names: list[str]
    values: list[float]
    source: str = "measured"

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("RelationDescriptor names and values must have the same length.")

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "RelationDescriptor":
        return cls(
            names=[str(name) for name in item.get("names", [])],
            values=[float(value) for value in item.get("values", [])],
            source=str(item.get("source", "measured")),
        )


@dataclass(frozen=True)
class CandidateProvenance:
    """How an imagined candidate was generated.

    Cosmos outputs pixels only. The paired action and robot-state paths are
    recorded explicitly so the candidate can be checked rather than promoted
    to supervision from visual plausibility alone.
    """

    generator: str
    model_id: str
    parent_traj_id: str
    generation_seed: int | None = None
    action_path: str | None = None
    robot_state_path: str | None = None
    context_video_path: str | None = None
    prompt: str | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "CandidateProvenance":
        return cls(
            generator=str(item.get("generator", "")),
            model_id=str(item.get("model_id", "")),
            parent_traj_id=str(item.get("parent_traj_id", "")),
            generation_seed=(None if item.get("generation_seed") is None else int(item["generation_seed"])),
            action_path=(None if item.get("action_path") is None else str(item["action_path"])),
            robot_state_path=(None if item.get("robot_state_path") is None else str(item["robot_state_path"])),
            context_video_path=(None if item.get("context_video_path") is None else str(item["context_video_path"])),
            prompt=(None if item.get("prompt") is None else str(item["prompt"])),
        )


@dataclass(frozen=True)
class VerificationRecord:
    """Result of deterministic geometry and kinematics checks."""

    status: str
    checks: dict[str, bool]
    reasons: list[str]
    relation: RelationDescriptor | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "VerificationRecord":
        relation = item.get("relation")
        return cls(
            status=str(item.get("status", "unverified")),
            checks={str(name): bool(value) for name, value in dict(item.get("checks", {})).items()},
            reasons=[str(reason) for reason in item.get("reasons", [])],
            relation=RelationDescriptor.from_dict(relation) if isinstance(relation, dict) else None,
        )


@dataclass(frozen=True)
class TrajectoryExample:
    traj_id: str
    task: str
    frames: list[str]
    source: str
    split: str
    metadata: dict[str, Any] | None = None
    robot_state_path: str | None = None
    action_path: str | None = None
    relation_path: str | None = None
    goal_ref_id: str | None = None
    parent_traj_id: str | None = None
    candidate_provenance: CandidateProvenance | None = None
    verification: VerificationRecord | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "TrajectoryExample":
        metadata = dict(item.get("metadata", {}))
        provenance = item.get("candidate_provenance") or metadata.get("candidate_provenance")
        verification = item.get("verification") or metadata.get("verification")
        return cls(
            traj_id=str(item["traj_id"]),
            task=str(item["task"]),
            frames=[str(frame) for frame in item.get("frames", [])],
            source=str(item.get("source", "")),
            split=str(item.get("split", "")),
            metadata=metadata,
            robot_state_path=(None if item.get("robot_state_path") is None else str(item["robot_state_path"])),
            action_path=(None if item.get("action_path") is None else str(item["action_path"])),
            relation_path=(None if item.get("relation_path") is None else str(item["relation_path"])),
            goal_ref_id=(None if item.get("goal_ref_id") is None else str(item["goal_ref_id"])),
            parent_traj_id=(None if item.get("parent_traj_id") is None else str(item["parent_traj_id"])),
            candidate_provenance=(CandidateProvenance.from_dict(provenance) if isinstance(provenance, dict) else None),
            verification=(VerificationRecord.from_dict(verification) if isinstance(verification, dict) else None),
        )


@dataclass(frozen=True)
class SuccessReference:
    ref_id: str
    task: str
    frames: list[str]

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "SuccessReference":
        return cls(
            ref_id=str(item["ref_id"]),
            task=str(item["task"]),
            frames=[str(frame) for frame in item.get("frames", [])],
        )


@dataclass(frozen=True)
class PreferencePair:
    task: str
    chosen_traj_id: str
    rejected_traj_id: str
    chosen_score: float
    rejected_score: float
    score_type: str = "mi_delta"
    goal_ref_id: str | None = None
    chosen_confidence: float = 1.0
    rejected_confidence: float = 1.0
    teacher_version: str = "legacy_v0"

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "PreferencePair":
        return cls(
            task=str(item["task"]),
            chosen_traj_id=str(item["chosen_traj_id"]),
            rejected_traj_id=str(item["rejected_traj_id"]),
            chosen_score=float(item["chosen_score"]),
            rejected_score=float(item["rejected_score"]),
            score_type=str(item.get("score_type", "mi_delta")),
            goal_ref_id=(None if item.get("goal_ref_id") is None else str(item["goal_ref_id"])),
            chosen_confidence=float(item.get("chosen_confidence", 1.0)),
            rejected_confidence=float(item.get("rejected_confidence", 1.0)),
            teacher_version=str(item.get("teacher_version", "legacy_v0")),
        )


T = TypeVar("T")


def read_jsonl(path: str | Path, factory: type[T]) -> list[T]:
    records: list[T] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            records.append(factory.from_dict(payload))  # type: ignore[attr-defined]
    return records


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
