"""Action proposal contracts for LaWAM or conventional planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    actions: np.ndarray
    seed: int
    source: str

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or self.actions.shape[0] < 1:
            raise ValueError("ActionCandidate.actions must have shape [T, A] with T >= 1.")


class ActionProposer(Protocol):
    def propose(self, num_candidates: int) -> list[ActionCandidate]:
        ...


class NpyActionProposer:
    """Read pre-sampled LaWAM/planner action candidates from `[K, T, A]` NPY."""

    def __init__(self, candidate_path: str | Path, source: str = "lawam_or_planner"):
        self.candidate_path = Path(candidate_path)
        self.source = source

    def propose(self, num_candidates: int) -> list[ActionCandidate]:
        if not self.candidate_path.is_file():
            raise FileNotFoundError(f"Action candidate tensor not found: {self.candidate_path}")
        tensor = np.load(self.candidate_path)
        if tensor.ndim != 3:
            raise ValueError(f"Expected action candidates [K, T, A], got {tuple(tensor.shape)}")
        if num_candidates > tensor.shape[0]:
            raise ValueError(f"Requested {num_candidates} candidates, but only {tensor.shape[0]} are available.")
        return [ActionCandidate(actions=np.asarray(tensor[index]), seed=index, source=self.source) for index in range(num_candidates)]
