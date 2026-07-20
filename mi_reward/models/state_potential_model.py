"""State-potential reward model for distilling Dame-style MI teacher scores.

The student model outputs per-timestep scalar potentials V_theta(o_t, g),
from which trajectory-level progress and deployment rewards are derived.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPooler(nn.Module):
    """Configurable token pooling layer."""

    def __init__(
        self,
        mode: Literal["mean", "attention", "max"] = "mean",
        input_dim: int = 512,
    ):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.Tanh(),
                nn.Linear(input_dim // 4, 1),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool token dimension.

        Args:
            x: [B, T, N, D] or [B, N, D]
            mask: [B, T, N] or [B, N] boolean mask

        Returns:
            [B, T, D] or [B, D]
        """
        if self.mode == "mean":
            if mask is not None:
                mask_f = mask.float().unsqueeze(-1)
                return (x * mask_f).sum(dim=-2) / mask_f.sum(dim=-2).clamp_min(1)
            return x.mean(dim=-2)
        elif self.mode == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=-2).values
        elif self.mode == "attention":
            # Attention weights over tokens
            raw = self.attn(x)  # [..., N, 1]
            if mask is not None:
                raw = raw.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            weights = F.softmax(raw, dim=-2)
            return (x * weights).sum(dim=-2)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")


class StatePotentialRewardModel(nn.Module):
    """State-potential reward model for per-timestep potential prediction.

    Architecture variants:
      - "mlp": temporal MLP over pooled frames
      - "gru": lightweight GRU
      - "transformer": small TransformerEncoder
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        architecture: Literal["mlp", "gru", "transformer"] = "gru",
        token_pooling: Literal["mean", "attention", "max"] = "mean",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_task_conditioning: bool = False,
        task_dim: int | None = None,
    ):
        """
        Args:
            input_dim: feature dimension per token
            hidden_dim: hidden state size
            architecture: temporal model type
            token_pooling: how to pool tokens per frame
            num_layers: number of layers for GRU/transformer
            num_heads: attention heads for transformer
            dropout: dropout rate
            use_task_conditioning: if True, condition on task embedding
            task_dim: dimension of task conditioning vector (ignored if use_task_conditioning=False)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.use_task_conditioning = use_task_conditioning

        # Token pooler
        self.token_pooler = TokenPooler(mode=token_pooling, input_dim=input_dim)

        # Task conditioning projection
        if use_task_conditioning and task_dim is not None:
            self.task_proj = nn.Linear(task_dim, input_dim)
        else:
            self.task_proj = None

        # Temporal model
        if architecture == "mlp":
            self.temporal = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.potential_head = nn.Linear(hidden_dim, 1)
        elif architecture == "gru":
            self.gru = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.potential_head = nn.Linear(hidden_dim, 1)
        elif architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.potential_head = nn.Linear(input_dim, 1)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _pool_tokens(
        self,
        frame_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool token dimension.

        Args:
            frame_features: [B, T, N, D] or [B, T, D]
            mask: [B, T, N] or [B, T] boolean mask

        Returns:
            [B, T, D]
        """
        if frame_features.dim() == 3:
            # Already pooled: [B, T, D]
            return frame_features
        # [B, T, N, D] -> [B, T, D]
        return self.token_pooler(frame_features, mask)

    def forward(
        self,
        frame_or_token_features: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-timestep scalar potentials.

        Args:
            frame_or_token_features: [B, T, D] pooled features or [B, T, N, D] token features
            task_features: [B, task_dim] optional task conditioning

        Returns:
            potentials: [B, T] scalar potential per timestep
        """
        # Pool tokens if needed
        pooled = self._pool_tokens(frame_or_token_features)  # [B, T, D]

        # Task conditioning
        if self.use_task_conditioning and task_features is not None and self.task_proj is not None:
            task_emb = self.task_proj(task_features).unsqueeze(1)  # [B, 1, D]
            pooled = pooled + task_emb

        if self.architecture == "mlp":
            hidden = self.temporal(pooled)  # [B, T, hidden_dim]
            potentials = self.potential_head(hidden).squeeze(-1)  # [B, T]
        elif self.architecture == "gru":
            outputs, _ = self.gru(pooled)  # [B, T, hidden_dim]
            potentials = self.potential_head(outputs).squeeze(-1)  # [B, T]
        elif self.architecture == "transformer":
            outputs = self.transformer(pooled)  # [B, T, D]
            potentials = self.potential_head(outputs).squeeze(-1)  # [B, T]
        else:
            raise RuntimeError(f"Unknown architecture: {self.architecture}")

        return potentials

    def compute_trajectory_score(
        self,
        frame_or_token_features: torch.Tensor,
        task_features: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """Compute trajectory-level score from potential differences.

        predicted_trajectory_score = aggregate_t (gamma * V_{t+1} - V_t)

        Args:
            frame_or_token_features: [B, T, D] or [B, T, N, D]
            task_features: [B, task_dim] optional
            gamma: discount factor

        Returns:
            trajectory_scores: [B]
        """
        potentials = self.forward(frame_or_token_features, task_features)  # [B, T]
        if potentials.shape[1] < 2:
            return potentials.mean(dim=1)
        deltas = gamma * potentials[:, 1:] - potentials[:, :-1]  # [B, T-1]
        return deltas.mean(dim=1)

    def compute_deployment_reward(
        self,
        current_features: torch.Tensor,
        next_features: torch.Tensor,
        task_features: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """Compute deployment reward: r_t = gamma * V(o_{t+1}, g) - V(o_t, g).

        Args:
            current_features: [B, D] or [B, N, D]
            next_features: [B, D] or [B, N, D]
            task_features: [B, task_dim] optional
            gamma: discount

        Returns:
            rewards: [B]
        """
        current_pot = self.forward(
            current_features.unsqueeze(1).unsqueeze(1) if current_features.dim() == 2 else current_features.unsqueeze(1),
            task_features,
        ).squeeze(-1)  # [B]
        next_pot = self.forward(
            next_features.unsqueeze(1).unsqueeze(1) if next_features.dim() == 2 else next_features.unsqueeze(1),
            task_features,
        ).squeeze(-1)  # [B]
        return gamma * next_pot - current_pot
