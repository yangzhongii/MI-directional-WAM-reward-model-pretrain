from __future__ import annotations

import torch
from torch import nn


class TrajectoryRewardHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if trajectory_features.ndim != 3:
            raise ValueError(f"Expected [B, T, D] features, got {tuple(trajectory_features.shape)}")
        if mask is None:
            mask = torch.ones(trajectory_features.shape[:2], dtype=torch.bool, device=trajectory_features.device)
        mask_f = mask.float().unsqueeze(-1)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (trajectory_features * mask_f).sum(dim=1) / lengths
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(trajectory_features.shape[0], device=trajectory_features.device)
        last_pool = trajectory_features[batch_indices, last_indices]
        return self.net(torch.cat([mean_pool, last_pool], dim=-1)).squeeze(-1)
