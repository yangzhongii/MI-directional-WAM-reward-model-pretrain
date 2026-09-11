"""Low-capacity spatial control latent and action-conditioned dynamics for v6.

The frozen LaWAM backbone supplies patch tokens ``[B,K,D]``.  This module
never consumes privileged geometry: its only dynamic input is the measured EEF
displacement used as the candidate-action coordinate.
"""
from __future__ import annotations

import torch
from torch import nn


class ControlLatentProjector(nn.Module):
    """Project channels while retaining the original patch-token positions."""

    def __init__(self, input_dim: int, control_dim: int) -> None:
        super().__init__()
        # A prediction-only objective has the trivial all-zero projection as a
        # solution.  Fixed-scale normalization plus unit-norm projection rows
        # excludes that collapse without using any physical supervision.
        # ``torch.nn.utils.parametrizations.orthogonal`` is not used here: on
        # the project environment's PyTorch build it materializes zero weights
        # after optimizer updates for this rectangular layer.
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.projection = nn.Linear(input_dim, control_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B,K,D] tokens, got {tuple(tokens.shape)}.")
        weight = torch.nn.functional.normalize(self.projection.weight, dim=1)
        return torch.nn.functional.linear(self.norm(tokens), weight)


class ActionEncoder(nn.Module):
    """Bias-free action encoding so the representation is exactly zero at a=0."""

    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2:
            raise ValueError(f"Expected [B,A] actions, got {tuple(action.shape)}.")
        return self.layers(action)


class ActionConditionedLatentDynamics(nn.Module):
    """Token-wise state-conditioned residual dynamics.

    A patch receives its own control feature, a pooled current-state context,
    and a candidate EEF displacement.  The multiplicative action gate makes
    ``F(z, 0) == z`` exactly, which is required by the v6 MI finite-difference
    protocol.
    """

    def __init__(self, control_dim: int, action_dim: int = 3, hidden_dim: int = 64, action_scale: float = 0.003, spatial_heads: int = 4) -> None:
        super().__init__()
        if action_scale <= 0:
            raise ValueError("action_scale must be positive.")
        self.action_scale = float(action_scale)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        if control_dim % spatial_heads:
            raise ValueError("control_dim must be divisible by spatial_heads.")
        self.spatial_mixer = nn.TransformerEncoderLayer(
            d_model=control_dim,
            nhead=spatial_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(control_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, control_dim, bias=False)

    def predict_delta(self, control_tokens: torch.Tensor, eef_delta: torch.Tensor) -> torch.Tensor:
        if control_tokens.ndim != 3:
            raise ValueError(f"Expected [B,K,C] control tokens, got {tuple(control_tokens.shape)}.")
        if eef_delta.shape[0] != control_tokens.shape[0]:
            raise ValueError("Action and token batch sizes must match.")
        mixed_tokens = self.spatial_mixer(control_tokens)
        context = mixed_tokens.mean(dim=1, keepdim=True).expand_as(mixed_tokens)
        state = self.state_encoder(torch.cat((mixed_tokens, context), dim=-1))
        action = self.action_encoder(eef_delta / self.action_scale).unsqueeze(1)
        return self.delta_head(state * action)

    def forward(self, control_tokens: torch.Tensor, eef_delta: torch.Tensor) -> torch.Tensor:
        return control_tokens + self.predict_delta(control_tokens, eef_delta)


class ActionConditionedControlModel(nn.Module):
    """Convenience wrapper for ``P_theta`` followed by ``F_theta``."""

    def __init__(self, input_dim: int, control_dim: int = 64, action_dim: int = 3, hidden_dim: int = 64, action_scale: float = 0.003, spatial_heads: int = 4) -> None:
        super().__init__()
        self.projector = ControlLatentProjector(input_dim, control_dim)
        self.dynamics = ActionConditionedLatentDynamics(control_dim, action_dim, hidden_dim, action_scale, spatial_heads)

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(tokens)

    def forward(self, current_tokens: torch.Tensor, eef_delta: torch.Tensor) -> torch.Tensor:
        return self.dynamics(self.project(current_tokens), eef_delta)
