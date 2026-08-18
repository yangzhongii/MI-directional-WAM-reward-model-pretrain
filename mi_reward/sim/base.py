"""Backend-neutral simulation contracts.

The contracts keep task assets and physics outside the reward code while making
all state needed for verification explicit in exported sidecars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SimulatorConfig:
    model_path: str
    object_body_name: str | None = None
    object_geom_name: str | None = None
    timestep: float | None = None
    camera_name: str | None = None


@dataclass(frozen=True)
class InstanceAsset:
    category: str
    asset_uri: str
    model_path: str | None = None
    mass: float | None = None
    geom_size: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SimulationRollout:
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or self.qvel.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("SimulationRollout tensors must have shape [T, D].")
        if not (len(self.qpos) == len(self.qvel) == len(self.actions)):
            raise ValueError("SimulationRollout tensors must share the same timestep count.")


class SimulatorBackend(ABC):
    """Physical state authority for one task rollout."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        ...

    @abstractmethod
    def apply_instance_variant(self, asset: InstanceAsset) -> None:
        ...

    @abstractmethod
    def rollout(self, actions: np.ndarray) -> SimulationRollout:
        ...

    @abstractmethod
    def render_rgb(self, width: int, height: int) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def model_path(self) -> Path:
        ...

