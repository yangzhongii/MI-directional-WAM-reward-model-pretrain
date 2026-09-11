"""Deployable visual potential distilled from privileged offline teachers."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _compatible_heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class VisualGoalPotential(nn.Module):
    """Predict ``V(o_t, g)`` without action or relation inputs.

    Action latents, robot state and measured relations are privileged teacher
    signals used only while constructing targets. The deployed model consumes
    visual patch tokens and an optional visual goal. A learned null-goal token
    supports benchmarks that do not expose a separate success image.
    """

    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int = 256,
        architecture: Literal["mlp", "gru", "transformer"] = "gru",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        goal_dropout: float = 0.2,
    ):
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("visual_dim must be positive.")
        if not 0.0 <= goal_dropout <= 1.0:
            raise ValueError("goal_dropout must be between zero and one.")
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.goal_dropout = goal_dropout
        heads = _compatible_heads(hidden_dim, num_heads)
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim))
        self.null_goal = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.null_goal, std=0.02)
        self.goal_cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.token_gate = nn.Linear(hidden_dim, 1)
        if architecture == "mlp":
            self.temporal = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
        elif architecture == "gru":
            self.temporal = nn.GRU(
                hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.potential_head = nn.Linear(hidden_dim, 1)

    def _goal_memory(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None,
        goal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state_tokens.shape[0]
        if goal_tokens is None:
            return (
                self.null_goal.expand(batch, 1, self.hidden_dim),
                torch.ones((batch, 1), dtype=torch.bool, device=state_tokens.device),
            )
        if goal_tokens.ndim != 3 or goal_tokens.shape[0] != batch or goal_tokens.shape[-1] != self.visual_dim:
            raise ValueError(f"goal_tokens must be [B,K,{self.visual_dim}], got {tuple(goal_tokens.shape)}")
        memory = self.visual_proj(goal_tokens)
        if goal_mask is None:
            goal_mask = torch.ones(goal_tokens.shape[:2], dtype=torch.bool, device=goal_tokens.device)
        if self.training and self.goal_dropout > 0.0:
            dropped = torch.rand(batch, device=state_tokens.device) < self.goal_dropout
            if dropped.any():
                memory = memory.clone()
                goal_mask = goal_mask.clone()
                memory[dropped] = self.null_goal.expand(int(dropped.sum().item()), memory.shape[1], self.hidden_dim)
                goal_mask[dropped] = False
                goal_mask[dropped, 0] = True
        return memory, goal_mask

    def encode(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return task/goal-conditioned temporal visual states [B,T,H]."""
        if state_tokens.ndim != 4 or state_tokens.shape[-1] != self.visual_dim:
            raise ValueError(f"state_tokens must be [B,T,K,{self.visual_dim}], got {tuple(state_tokens.shape)}")
        batch, steps, patches, _ = state_tokens.shape
        if state_mask is None:
            state_mask = torch.ones((batch, steps), dtype=torch.bool, device=state_tokens.device)
        state = self.visual_proj(state_tokens)
        memory, memory_mask = self._goal_memory(state_tokens, goal_tokens, goal_mask)
        query = state.reshape(batch * steps, patches, self.hidden_dim)
        expanded_memory = memory[:, None].expand(batch, steps, memory.shape[1], self.hidden_dim)
        expanded_memory = expanded_memory.reshape(batch * steps, memory.shape[1], self.hidden_dim)
        expanded_mask = (~memory_mask[:, None].expand(batch, steps, memory.shape[1])).reshape(
            batch * steps, memory.shape[1]
        )
        attended, _ = self.goal_cross_attention(
            query, expanded_memory, expanded_memory, key_padding_mask=expanded_mask, need_weights=False
        )
        token_features = self.fuse_norm(query + attended)
        token_weights = torch.softmax(self.token_gate(token_features).squeeze(-1), dim=-1)
        frames = (token_features * token_weights.unsqueeze(-1)).sum(dim=1).reshape(batch, steps, self.hidden_dim)
        frames = frames.masked_fill(~state_mask.unsqueeze(-1), 0.0)
        if self.architecture == "gru":
            lengths = state_mask.long().sum(dim=1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(frames, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.temporal(packed)  # type: ignore[arg-type]
            hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=steps)
        elif self.architecture == "transformer":
            hidden = self.temporal(frames, src_key_padding_mask=~state_mask)  # type: ignore[operator]
        else:
            hidden = self.temporal(frames)  # type: ignore[operator]
        return hidden.masked_fill(~state_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(state_tokens, goal_tokens, state_mask, goal_mask)
        if state_mask is None:
            state_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        return self.potential_head(hidden).squeeze(-1).masked_fill(~state_mask, 0.0)

    def compute_trajectory_score(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        potentials = self(state_tokens, goal_tokens, state_mask, goal_mask)
        if potentials.shape[1] < 2:
            return potentials[:, 0]
        deltas = gamma * potentials[:, 1:] - potentials[:, :-1]
        if state_mask is None:
            return deltas.mean(dim=1)
        valid = state_mask[:, 1:] & state_mask[:, :-1]
        return (deltas * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def compute_deployment_reward(
        self,
        current_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        current = self(current_tokens.unsqueeze(1), goal_tokens, goal_mask=goal_mask)[:, 0]
        nxt = self(next_tokens.unsqueeze(1), goal_tokens, goal_mask=goal_mask)[:, 0]
        return gamma * nxt - current
