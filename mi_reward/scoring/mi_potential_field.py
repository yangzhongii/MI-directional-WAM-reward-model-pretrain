"""MI Directional Potential Field — unified reward discovery module.

Defines the core information-theoretic reward:
    Phi(s, g) = I(z_s, z_g)           (mutual information potential)
    r_t = gamma * Phi(s_{t+1}, g) - Phi(s_t, g)   (directional reward)

Supports multiple MI backends (gaussian proxy, histogram, Dame B-spline)
and multiple feature sources (DINOv3, LaWAM, pooled).

Reference:
    Dame & Marchand, "Mutual Information-Based Visual Servoing," TRO 2011.
    Ng, Harada & Russell, "Policy Invariance under Reward Transformations," ICML 1999.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn

from mi_reward.alignment.monotonic_alignment import (
    build_mi_alignment_matrix,
    monotonic_viterbi_alignment,
)
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class MIBackend(Enum):
    GAUSSIAN = "gaussian_mi_proxy"
    HISTOGRAM = "histogram_mi"
    DAME_BSPLINE = "dame_bspline"


def _build_backend(backend: MIBackend, num_bins: int = 8) -> nn.Module:
    if backend == MIBackend.DAME_BSPLINE:
        return DameSoftHistogramMI(num_bins=num_bins)
    # For gaussian/histogram, return a simple callable wrapper
    from mi_reward.scoring.mi_potential import compute_mi

    class _SimpleMI(nn.Module):
        def __init__(self, mode: str):
            super().__init__()
            self._mode = mode
        def forward(self, x, y) -> torch.Tensor:
            return compute_mi(x.float().flatten(), y.float().flatten(), mode=self._mode)
    return _SimpleMI(backend.value)


# ---------------------------------------------------------------------------
# Feature encoder interface
# ---------------------------------------------------------------------------

class BaseLatentEncoder(ABC):
    """Encode an image/state batch into a latent representation for MI computation."""

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] or [B, T, C, H, W] → features [B, D] or [B, T, D]"""
        ...

    @abstractmethod
    def encode_trajectory(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, T, C, H, W] → [B, T, D]"""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Directional potential result
# ---------------------------------------------------------------------------

@dataclass
class PotentialResult:
    """Per-timestep MI potential and derived directional reward."""

    phi: torch.Tensor          # [T] MI potential values
    reward: torch.Tensor       # [T-1] r_t = gamma * phi_{t+1} - phi_t
    gamma: float = 0.99

    @property
    def total_progress(self) -> float:
        return float(self.phi[-1].item() - self.phi[0].item())

    @property
    def mean_reward(self) -> float:
        return float(self.reward.mean().item()) if self.reward.numel() > 0 else 0.0


# ---------------------------------------------------------------------------
# Core: MI Potential Field
# ---------------------------------------------------------------------------

class MIPotentialField(nn.Module):
    """Compute MI-based potential Phi(s, g) between state and goal latents.

    Supports per-frame or temporally-aligned potential computation.

    Usage::

        field = MIPotentialField(backend=MIBackend.DAME_BSPLINE)
        phi = field.potential(state_features, goal_features)       # framewise
        result = field.directional_reward(traj_features, goal_features)  # with alignment
    """

    def __init__(
        self,
        backend: MIBackend = MIBackend.DAME_BSPLINE,
        num_bins: int = 8,
        gamma: float = 0.99,
        alignment: bool = True,
        stay_penalty: float = 0.0,
        jump_penalty: float = 0.01,
    ):
        super().__init__()
        self.backend = backend
        self.gamma = gamma
        self.alignment = alignment
        self.stay_penalty = stay_penalty
        self.jump_penalty = jump_penalty
        self._mi = _build_backend(backend, num_bins)

    # ------------------------------------------------------------------
    # Framewise potential
    # ------------------------------------------------------------------

    def potential(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute framewise MI potential: Phi(s_frame, g).

        Args:
            state: [D] or [N, D] current state latent
            goal:  [D] or [N, D] goal latent

        Returns:
            Scalar MI value.
        """
        return self._mi(state, goal)

    def potential_batch(
        self, states: torch.Tensor, goal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-frame potentials for a trajectory against a goal.

        Args:
            states: [T, D] or [T, N, D] trajectory latents
            goal:   [D] or [N, D] goal latent

        Returns:
            phi: [T] potential per timestep.
        """
        T = states.shape[0]
        phi = torch.zeros(T, device=states.device)
        for t in range(T):
            phi[t] = self._mi(states[t], goal)
        return phi

    def potential_aligned(
        self, states: torch.Tensor, goal_trajectory: torch.Tensor,
    ) -> PotentialResult:
        """Compute temporally-aligned MI potentials + directional reward.

        Args:
            states:         [T, N, D] candidate trajectory latents
            goal_trajectory:[S, N, D] goal trajectory latents

        Returns:
            PotentialResult with phi [T] and reward [T-1].
        """
        # Build alignment matrix and find monotonic path
        align_mat = build_mi_alignment_matrix(states, goal_trajectory, self._mi)
        alignment = monotonic_viterbi_alignment(
            align_mat,
            stay_penalty=self.stay_penalty,
            jump_penalty=self.jump_penalty,
        )
        phi = alignment["aligned_potential"]  # [T]

        # Directional reward: r_t = gamma * phi_{t+1} - phi_t
        reward = self._directional_reward(phi)
        return PotentialResult(phi=phi, reward=reward, gamma=self.gamma)

    # ------------------------------------------------------------------
    # Directional reward
    # ------------------------------------------------------------------

    def _directional_reward(self, phi: torch.Tensor) -> torch.Tensor:
        """r_t = gamma * phi_{t+1} - phi_t"""
        if phi.numel() < 2:
            return torch.zeros(0, device=phi.device)
        return self.gamma * phi[1:] - phi[:-1]

    def compute_directional_reward(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step deployment reward.

        r = gamma * I(z_next, z_goal) - I(z_curr, z_goal)

        Args:
            current_state: [D] or [B, D]
            next_state:    [D] or [B, D]
            goal_state:    [D] or [B, D]

        Returns:
            Scalar or [B] reward.
        """
        phi_curr = self.potential(current_state, goal_state)
        phi_next = self.potential(next_state, goal_state)
        return self.gamma * phi_next - phi_curr


# ---------------------------------------------------------------------------
# DINOv3 encoder adapter (for the clean encoder interface)
# ---------------------------------------------------------------------------

class DINOv3LatentEncoder(BaseLatentEncoder):
    """DINOv3 ViT-B/16 as a latent encoder for the MI pipeline.

    Wraps the existing DINOv3FeatureExtractor in the BaseLatentEncoder API.
    """

    def __init__(self, weights_path: str = "weights/dinov3-vitb16-pretrain-lvd1689m", device: str = "cuda"):
        from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
        self._extractor = DINOv3FeatureExtractor(model_path=weights_path, device=device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] → [B, 768]"""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        feats = []
        for b in range(B):
            feats.append(self._extractor.extract_frame(None, ""))  # placeholder
        return torch.stack(feats)

    @torch.no_grad()
    def encode_trajectory(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, T, C, H, W] → [B, T, 768]"""
        raise NotImplementedError("Use frame-level DINOv3FeatureExtractor for trajectories")

    @property
    def feature_dim(self) -> int:
        return 768


# ---------------------------------------------------------------------------
# Ablation reward functions (interchangeable)
# ---------------------------------------------------------------------------

def latent_distance_reward(z_curr: torch.Tensor, z_next: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """r = -||z_next - z_goal|| (negative L2, higher = closer)"""
    d_next = torch.norm(z_next.float() - z_goal.float(), dim=-1)
    d_curr = torch.norm(z_curr.float() - z_goal.float(), dim=-1)
    return d_curr - d_next  # positive if approaching goal


def cosine_similarity_reward(z_curr: torch.Tensor, z_next: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """r = cos(z_next, z_goal) - cos(z_curr, z_goal)"""
    cos_next = torch.nn.functional.cosine_similarity(z_next.float(), z_goal.float(), dim=-1)
    cos_curr = torch.nn.functional.cosine_similarity(z_curr.float(), z_goal.float(), dim=-1)
    return cos_next - cos_curr


# ---------------------------------------------------------------------------
# Reward registry
# ---------------------------------------------------------------------------

REWARD_FUNCTIONS: dict[str, Callable] = {
    "mi_directional": None,  # requires MIPotentialField instance
    "latent_distance": latent_distance_reward,
    "cosine_similarity": cosine_similarity_reward,
}
