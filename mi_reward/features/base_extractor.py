from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        """Extract one frame feature conditioned on the task instruction.

        Returns:
            torch.Tensor: [D] pooled feature vector
        """

    def extract_trajectory(self, frame_paths: list[str], task: str) -> torch.Tensor:
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        features = [self.extract_frame(frame_path, task) for frame_path in frame_paths]
        return torch.stack([feature.detach().cpu() for feature in features], dim=0)

    def extract_frame_tokens(self, frame_path: str, task: str) -> torch.Tensor:
        """Extract per-token features for one frame.

        Override in subclasses that support token-level extraction.
        Default falls back to pooled feature with a dummy token dimension.

        Returns:
            torch.Tensor: [N, D] token features (N = num_tokens)
        """
        pooled = self.extract_frame(frame_path, task)
        return pooled.unsqueeze(0)  # [1, D]

    def extract_trajectory_tokens(self, frame_paths: list[str], task: str) -> torch.Tensor:
        """Extract per-token features for a trajectory.

        Returns:
            torch.Tensor: [T, N, D] token features
        """
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        token_features = [self.extract_frame_tokens(frame_path, task) for frame_path in frame_paths]
        return torch.stack([feat.detach().cpu() for feat in token_features], dim=0)
