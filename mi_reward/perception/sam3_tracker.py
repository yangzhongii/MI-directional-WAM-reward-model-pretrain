"""SAM3-compatible instance mask tracking contract.

The repository intentionally does not import SAM3 directly. Different SAM3
releases expose different APIs, while this adapter fixes the artifact contract
consumed by the simulator and Transfer worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class MaskTrackFrame:
    frame_index: int
    object_id: str
    mask_path: str


@dataclass(frozen=True)
class MaskTrackBundle:
    frames: list[MaskTrackFrame]

    def for_object(self, object_id: str) -> list[MaskTrackFrame]:
        return [frame for frame in self.frames if frame.object_id == object_id]


class InstanceTracker(Protocol):
    def track(self, frame_paths: Sequence[str], object_prompts: dict[str, str]) -> MaskTrackBundle:
        ...


class PrecomputedMaskTracker:
    """Load masks written by a SAM3 worker or simulator renderer."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    def track(self, frame_paths: Sequence[str], object_prompts: dict[str, str]) -> MaskTrackBundle:
        del object_prompts
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = payload.get("tracks", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("SAM3 track manifest must be a list or contain a `tracks` list.")
        bundle = MaskTrackBundle(
            frames=[
                MaskTrackFrame(
                    frame_index=int(item["frame_index"]),
                    object_id=str(item["object_id"]),
                    mask_path=str(item["mask_path"]),
                )
                for item in records
                if isinstance(item, dict)
            ]
        )
        expected = set(range(len(frame_paths)))
        actual = {item.frame_index for item in bundle.frames}
        if not expected.issubset(actual):
            raise ValueError("SAM3 track manifest does not cover all input frames.")
        if not all(Path(item.mask_path).is_file() for item in bundle.frames):
            raise FileNotFoundError("SAM3 track manifest references a missing mask.")
        return bundle
