from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np


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

    def extract_image_tokens(self, image: np.ndarray, task: str) -> torch.Tensor:
        """Extract ``[K, D]`` tokens directly from an RGB array.

        Online simulators should not have to write every observation to a
        temporary PNG before reward inference.  Extractors used in a closed
        loop override this method; the explicit error prevents an accidental
        slow disk-backed fallback.
        """

        del image, task
        raise NotImplementedError(f"{type(self).__name__} does not support in-memory RGB observations.")

    def extract_trajectory_tokens(self, frame_paths: list[str], task: str) -> torch.Tensor:
        """Extract per-token features for a trajectory.

        Returns:
            torch.Tensor: [T, N, D] token features
        """
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        token_features = [self.extract_frame_tokens(frame_path, task) for frame_path in frame_paths]
        return torch.stack([feat.detach().cpu() for feat in token_features], dim=0)

    def extract_action_latents(self, frame_paths: list[str], task: str) -> torch.Tensor:
        """Extract transition-level latent actions shaped ``[T-1, Q, D]``.

        Visual-only encoders intentionally do not implement this contract.  A
        caller requesting action latents must use an extractor backed by a
        latent-action model rather than silently substituting visual features.
        """

        del frame_paths, task
        raise NotImplementedError(
            f"{type(self).__name__} does not provide latent actions; use the LaWAM LAM extractor."
        )
