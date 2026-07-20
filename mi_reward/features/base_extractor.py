from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        """Extract one frame feature conditioned on the task instruction."""

    def extract_trajectory(self, frame_paths: list[str], task: str) -> torch.Tensor:
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        features = [self.extract_frame(frame_path, task) for frame_path in frame_paths]
        return torch.stack([feature.detach().cpu() for feature in features], dim=0)
