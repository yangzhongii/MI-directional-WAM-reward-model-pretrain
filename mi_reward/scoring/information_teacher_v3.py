"""Canonical Pipeline-v3 information-theoretic teacher core.

This module deliberately separates the v3 teacher from the legacy DAME
trajectory-alignment score.  The critics below are *learned contrastive
density-ratio critics*:

    phi_v(v, g) ~= log p(v, g) / (p(v) p(g))

and

    psi_a(a, v, g) ~= log p(a | v, g) / p(a | v).

With balanced positive / negative logistic classification, the optimal logit
is the corresponding log density ratio.  For the conditional action critic,
the caller must construct negatives that approximate ``p(a | v)`` (for
example same-task, visually-near states with a different physical progress
direction).  Random actions from unrelated tasks do *not* satisfy that
contract.

Privileged physical consistency and P/U/N calibration are intentionally not
implemented here; they consume these continuous information scores downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_2d(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B,D] or [D], got {tuple(value.shape)}")
    return value


def _same_batch(*values: torch.Tensor) -> None:
    sizes = {int(value.shape[0]) for value in values}
    if len(sizes) != 1:
        raise ValueError(f"All inputs must share batch size, got {sorted(sizes)}")


class _Projection(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class VisualPointwiseInformationCritic(nn.Module):
    """Contrastive critic for the visual pointwise-information potential.

    A balanced classifier is trained to distinguish samples from ``p(v,g)``
    from samples approximating ``p(v)p(g)``.  Its raw logit is therefore the
    pointwise log-density-ratio estimate used as ``phi_v``.
    """

    def __init__(
        self,
        visual_dim: int,
        goal_dim: int,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.visual = _Projection(visual_dim, projection_dim)
        self.goal = _Projection(goal_dim, projection_dim)
        interaction_dim = projection_dim * 4
        self.head = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        visual = _ensure_2d("visual", visual)
        goal = _ensure_2d("goal", goal)
        _same_batch(visual, goal)
        v = self.visual(visual)
        g = self.goal(goal)
        features = torch.cat((v, g, v * g, torch.abs(v - g)), dim=-1)
        return self.head(features).squeeze(-1)


class ConditionalActionInformationCritic(nn.Module):
    """Contrastive critic for ``PMI(a; g | v)``.

    Positive triples come from ``p(a,v,g)``.  Negative triples must preserve
    the conditioning state/goal while replacing ``a`` with a sample that
    approximates ``p(a|v)``.  The caller is responsible for that conditional
    hard-negative construction.
    """

    def __init__(
        self,
        action_dim: int,
        visual_dim: int,
        goal_dim: int,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action = _Projection(action_dim, projection_dim)
        self.visual = _Projection(visual_dim, projection_dim)
        self.goal = _Projection(goal_dim, projection_dim)
        # a, v, g plus the three pairwise multiplicative interactions.
        interaction_dim = projection_dim * 6
        self.head = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        action: torch.Tensor,
        visual: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        action = _ensure_2d("action", action)
        visual = _ensure_2d("visual", visual)
        goal = _ensure_2d("goal", goal)
        _same_batch(action, visual, goal)
        a = self.action(action)
        v = self.visual(visual)
        g = self.goal(goal)
        features = torch.cat((a, v, g, a * v, a * g, v * g), dim=-1)
        return self.head(features).squeeze(-1)


def balanced_density_ratio_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """Balanced logistic NCE loss whose optimal logit is a log density ratio."""

    positive_logits = positive_logits.reshape(-1)
    negative_logits = negative_logits.reshape(-1)
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        raise ValueError("Both positive and negative critic samples are required.")
    return F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()


def visual_information_loss(
    critic: VisualPointwiseInformationCritic,
    visual: torch.Tensor,
    goal: torch.Tensor,
    negative_goal: torch.Tensor,
) -> torch.Tensor:
    """Train ``phi_v`` using matched goals vs marginal-goal negatives."""

    positive = critic(visual, goal)
    negative = critic(visual, negative_goal)
    return balanced_density_ratio_loss(positive, negative)


def conditional_action_information_loss(
    critic: ConditionalActionInformationCritic,
    action: torch.Tensor,
    visual: torch.Tensor,
    goal: torch.Tensor,
    negative_action_given_visual: torch.Tensor,
) -> torch.Tensor:
    """Train ``psi_a`` with negatives approximating ``p(a|v)``.

    ``negative_action_given_visual`` should come from same-task / visually-near
    states and ideally differ in measured physical progress direction.
    """

    positive = critic(action, visual, goal)
    negative = critic(negative_action_given_visual, visual, goal)
    return balanced_density_ratio_loss(positive, negative)


def directional_information_score(
    phi_t: torch.Tensor,
    phi_t1: torch.Tensor,
    psi_action: torch.Tensor,
    *,
    gamma: float = 0.99,
    beta: float = 1.0,
) -> torch.Tensor:
    """Canonical v3 soft score ``D_t = gamma*phi(t+1)-phi(t)+beta*psi_a``."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1].")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    phi_t, phi_t1, psi_action = torch.broadcast_tensors(phi_t, phi_t1, psi_action)
    return float(gamma) * phi_t1 - phi_t + float(beta) * psi_action


@dataclass(frozen=True)
class InformationTeacherEvidence:
    """Continuous v3 information evidence before privileged consistency gating."""

    phi_visual_t: torch.Tensor
    phi_visual_t1: torch.Tensor
    delta_phi_visual: torch.Tensor
    conditional_action_information: torch.Tensor
    directional_score: torch.Tensor


def compute_information_teacher_evidence(
    visual_critic: VisualPointwiseInformationCritic,
    action_critic: ConditionalActionInformationCritic,
    *,
    visual_t: torch.Tensor,
    visual_t1: torch.Tensor,
    goal: torch.Tensor,
    action_t: torch.Tensor,
    gamma: float = 0.99,
    beta: float = 1.0,
) -> InformationTeacherEvidence:
    """Evaluate the information-only portion of the privileged v3 teacher."""

    phi_t = visual_critic(visual_t, goal)
    phi_t1 = visual_critic(visual_t1, goal)
    psi_action = action_critic(action_t, visual_t, goal)
    delta = float(gamma) * phi_t1 - phi_t
    score = delta + float(beta) * psi_action
    return InformationTeacherEvidence(
        phi_visual_t=phi_t,
        phi_visual_t1=phi_t1,
        delta_phi_visual=delta,
        conditional_action_information=psi_action,
        directional_score=score,
    )
