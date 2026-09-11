"""Strict contracts for instance-aware rollout artifacts.

This module owns the schema-level checks only. It never estimates poses from
pixels or substitutes generated RGB for physical state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mi_reward.data.schema import ControlArtifacts, InstanceVariantDescriptor, ObjectStateDescriptor


@dataclass(frozen=True)
class ObjectStateFrame:
    frame_index: int
    objects: list[ObjectStateDescriptor]

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("ObjectStateFrame.frame_index must be non-negative.")
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"Duplicate object_id at frame {self.frame_index}: {object_ids}")


@dataclass(frozen=True)
class ObjectStateSequence:
    frames: list[ObjectStateFrame]

    def __post_init__(self) -> None:
        indices = [frame.frame_index for frame in self.frames]
        if not self.frames:
            raise ValueError("ObjectStateSequence must contain at least one frame.")
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("ObjectStateSequence frame indices must be unique and sorted.")

    @property
    def final_objects(self) -> list[ObjectStateDescriptor]:
        return self.frames[-1].objects


def _read_records(path: str | Path) -> list[Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Object-state sequence not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Object-state sequence is empty: {source}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("frames", "states", "records", "object_states"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"Unsupported object-state root in {source}: {type(payload).__name__}")


def _object_from_payload(payload: dict[str, Any]) -> ObjectStateDescriptor:
    # A small compatibility layer keeps external workers free to call position
    # and quaternion by their conventional names while artifacts remain strict.
    normalized = dict(payload)
    if "pose_xyz" not in normalized and "position" in normalized:
        normalized["pose_xyz"] = normalized["position"]
    if "pose_quaternion" not in normalized and "quaternion" in normalized:
        normalized["pose_quaternion"] = normalized["quaternion"]
    if "size_xyz" not in normalized and "size" in normalized:
        normalized["size_xyz"] = normalized["size"]
    return ObjectStateDescriptor.from_dict(normalized)


def load_object_state_sequence(path: str | Path) -> ObjectStateSequence:
    """Load frame-major object state JSON or JSONL.

    Supported frame record form::

        {"frame_index": 0, "objects": [{...}, {...}]}

    A one-object-per-line JSONL form is accepted too and grouped by
    ``frame_index``. Mixing the two forms is rejected to keep provenance
    unambiguous.
    """

    records = _read_records(path)
    frame_records: list[ObjectStateFrame] = []
    per_object: dict[int, list[ObjectStateDescriptor]] = {}
    saw_frame_major = False
    saw_object_major = False
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Object-state record {ordinal} in {path} must be an object.")
        if isinstance(record.get("objects"), list):
            saw_frame_major = True
            frame_index = int(record.get("frame_index", ordinal))
            objects = [_object_from_payload(item) for item in record["objects"] if isinstance(item, dict)]
            if len(objects) != len(record["objects"]):
                raise ValueError(f"Object-state frame {frame_index} in {path} contains a non-object item.")
            frame_records.append(ObjectStateFrame(frame_index=frame_index, objects=objects))
        else:
            saw_object_major = True
            frame_index = int(record.get("frame_index", ordinal))
            per_object.setdefault(frame_index, []).append(_object_from_payload(record))
    if saw_frame_major and saw_object_major:
        raise ValueError(f"Object-state sequence in {path} mixes frame-major and object-major records.")
    if saw_object_major:
        frame_records = [
            ObjectStateFrame(frame_index=index, objects=objects) for index, objects in sorted(per_object.items())
        ]
    return ObjectStateSequence(frames=sorted(frame_records, key=lambda item: item.frame_index))


def _existing_directory(value: str | None) -> bool:
    return value is not None and Path(value).is_dir()


def _artifact_count(root: str | None, suffixes: set[str]) -> int:
    if not _existing_directory(root):
        return 0
    return sum(1 for path in Path(root).rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def validate_instance_artifacts(
    *,
    frames: Iterable[str],
    object_state_path: str | Path | None,
    controls: ControlArtifacts | None,
    instance_variant: InstanceVariantDescriptor | None,
    strict: bool = True,
) -> list[str]:
    """Return stable rejection reasons for malformed instance artifacts."""

    reasons: list[str] = []
    frame_paths = [Path(frame) for frame in frames]
    if not frame_paths or not all(path.is_file() for path in frame_paths):
        reasons.append("missing_candidate_frames")
    if object_state_path is None:
        reasons.append("missing_object_state_path")
        return reasons
    try:
        sequence = load_object_state_sequence(object_state_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        reasons.append("invalid_object_state_sequence")
        return reasons
    if frame_paths and len(sequence.frames) != len(frame_paths):
        reasons.append("object_state_frame_count_mismatch")
    if frame_paths and [frame.frame_index for frame in sequence.frames] != list(range(len(frame_paths))):
        reasons.append("object_state_frame_indices_not_contiguous")
    if instance_variant is None:
        reasons.append("missing_instance_variant")
    elif not any(item.category == instance_variant.target_category for item in sequence.final_objects):
        reasons.append("target_instance_not_present")
    if strict:
        if controls is None:
            reasons.append("missing_control_artifacts")
        else:
            if not _existing_directory(controls.mask_root):
                reasons.append("missing_mask_root")
            elif frame_paths and _artifact_count(
                controls.mask_root, {".png", ".jpg", ".jpeg", ".webp"}
            ) < len(frame_paths):
                reasons.append("missing_mask_artifacts")
            if not _existing_directory(controls.depth_root):
                reasons.append("missing_depth_root")
            elif frame_paths and _artifact_count(controls.depth_root, {".npy", ".npz"}) < len(frame_paths):
                reasons.append("missing_depth_artifacts")
    return reasons
