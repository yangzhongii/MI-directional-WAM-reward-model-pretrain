from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar


@dataclass(frozen=True)
class TrajectoryExample:
    traj_id: str
    task: str
    frames: list[str]
    source: str
    split: str
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "TrajectoryExample":
        return cls(
            traj_id=str(item["traj_id"]),
            task=str(item["task"]),
            frames=[str(frame) for frame in item.get("frames", [])],
            source=str(item.get("source", "")),
            split=str(item.get("split", "")),
            metadata=dict(item.get("metadata", {})),
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

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "PreferencePair":
        return cls(
            task=str(item["task"]),
            chosen_traj_id=str(item["chosen_traj_id"]),
            rejected_traj_id=str(item["rejected_traj_id"]),
            chosen_score=float(item["chosen_score"]),
            rejected_score=float(item["rejected_score"]),
            score_type=str(item.get("score_type", "mi_delta")),
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
