"""Depth artifact contract independent of the selected estimator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class DepthBundle:
    depth_paths: list[str]


class DepthEstimator(Protocol):
    def estimate(self, frame_paths: Sequence[str]) -> DepthBundle:
        ...


class PrecomputedDepthEstimator:
    def __init__(self, depth_paths: Sequence[str]):
        self.depth_paths = [str(Path(path)) for path in depth_paths]

    def estimate(self, frame_paths: Sequence[str]) -> DepthBundle:
        if len(frame_paths) != len(self.depth_paths):
            raise ValueError("Precomputed depth count must match frame count.")
        if not all(Path(path).is_file() for path in self.depth_paths):
            raise FileNotFoundError("Precomputed depth includes a missing file.")
        return DepthBundle(depth_paths=self.depth_paths)
