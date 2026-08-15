"""Goal- and relation-conditioned potential model for GeoProgress."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _compatible_heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class GeoProgressPotential(nn.Module):
    """Predict ``V(o_t, g, rho_t)`` from visual patches and measured relations.

    ``rho_t`` is intentionally an explicit measured vector.  It is not a
    second image encoder and is never reconstructed from Cosmos pixels.
    """

    def __init__(
        self,
        visual_dim: int,
        relation_dim: int,
        hidden_dim: int = 256,
        architecture: Literal["mlp", "gru", "transformer"] = "gru",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if visual_dim <= 0 or relation_dim <= 0:
            raise ValueError("visual_dim and relation_dim must both be positive.")
        self.visual_dim = visual_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        heads = _compatible_heads(hidden_dim, num_heads)
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_dim))
        self.relation_proj = nn.Sequential(
            nn.LayerNorm(relation_dim), nn.Linear(relation_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.goal_cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.fuse_norm = nn.LayerNorm(hidden_dim)
        self.token_gate = nn.Linear(hidden_dim, 1)
        if architecture == "mlp":
            self.temporal = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
            )
            output_dim = hidden_dim
        elif architecture == "gru":
            self.temporal = nn.GRU(
                hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
            output_dim = hidden_dim
        elif architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
            output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.potential_head = nn.Linear(output_dim, 1)

    def _validate_inputs(self, state_tokens: torch.Tensor, goal_tokens: torch.Tensor, relations: torch.Tensor) -> None:
        if state_tokens.ndim != 4:
            raise ValueError(f"state_tokens must be [B, T, K, D], got {tuple(state_tokens.shape)}")
        if goal_tokens.ndim != 3:
            raise ValueError(f"goal_tokens must be [B, K, D], got {tuple(goal_tokens.shape)}")
        if relations.ndim != 3:
            raise ValueError(f"relations must be [B, T, R], got {tuple(relations.shape)}")
        if state_tokens.shape[0] != goal_tokens.shape[0] or state_tokens.shape[:2] != relations.shape[:2]:
            raise ValueError("Batch/time dimensions of state tokens, goals, and relations must agree.")
        if state_tokens.shape[-1] != self.visual_dim or goal_tokens.shape[-1] != self.visual_dim:
            raise ValueError("Visual token dimension does not match this model.")
        if relations.shape[-1] != self.relation_dim:
            raise ValueError("Relation dimension does not match this model.")

    def forward(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        relations: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(state_tokens, goal_tokens, relations)
        batch, steps, patches, _ = state_tokens.shape
        if state_mask is None:
            state_mask = torch.ones((batch, steps), dtype=torch.bool, device=state_tokens.device)
        if goal_mask is None:
            goal_mask = torch.ones(goal_tokens.shape[:2], dtype=torch.bool, device=goal_tokens.device)
        state = self.visual_proj(state_tokens)
        goal = self.visual_proj(goal_tokens)
        query = state.reshape(batch * steps, patches, self.hidden_dim)
        memory = goal[:, None].expand(batch, steps, goal.shape[1], self.hidden_dim).reshape(batch * steps, goal.shape[1], self.hidden_dim)
        memory_mask = (~goal_mask[:, None].expand(batch, steps, goal.shape[1])).reshape(batch * steps, goal.shape[1])
        attended, _ = self.goal_cross_attention(query, memory, memory, key_padding_mask=memory_mask, need_weights=False)
        token_features = self.fuse_norm(query + attended)
        token_weights = torch.softmax(self.token_gate(token_features).squeeze(-1), dim=-1)
        visual_frame = (token_features * token_weights.unsqueeze(-1)).sum(dim=1).reshape(batch, steps, self.hidden_dim)
        frames = self.fuse_norm(visual_frame + self.relation_proj(relations))
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
        return self.potential_head(hidden).squeeze(-1).masked_fill(~state_mask, 0.0)

    def compute_trajectory_score(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        relations: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        potentials = self(state_tokens, goal_tokens, relations, state_mask, goal_mask)
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
        goal_tokens: torch.Tensor,
        current_relation: torch.Tensor,
        next_relation: torch.Tensor,
        goal_mask: torch.Tensor | None = None,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        current = self(current_tokens.unsqueeze(1), goal_tokens, current_relation.unsqueeze(1), goal_mask=goal_mask)[:, 0]
        nxt = self(next_tokens.unsqueeze(1), goal_tokens, next_relation.unsqueeze(1), goal_mask=goal_mask)[:, 0]
        return gamma * nxt - current
