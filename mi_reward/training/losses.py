"""Multi-objective reward distillation losses.

Provides three complementary loss terms for training state-potential
reward models from Dame-style teacher scores:

1. Potential distillation: align per-frame potentials with teacher Phi
2. Directional-difference distillation: align potential deltas
3. Pairwise trajectory ranking (confidence-weighted Bradley-Terry)

All can be used independently or combined.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss for robust regression."""
    abs_err = (pred - target).abs()
    quadratic = 0.5 * abs_err ** 2
    linear = delta * (abs_err - 0.5 * delta)
    return torch.where(abs_err <= delta, quadratic, linear).mean()


def normalize_tensor(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize to zero mean, unit variance (per-batch)."""
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(eps)
    return (x - mean) / std


def potential_distillation_loss(
    student_potentials: torch.Tensor,
    teacher_phi: torch.Tensor,
    mask: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """L_potential: align normalized student potentials with teacher MI potentials.

    L_potential = Huber(normalize(V_theta(o_t, g)), stop_gradient(normalize(Phi_t)))

    Args:
        student_potentials: [B, T] predicted potentials
        teacher_phi: [B, T] teacher MI potentials (detached)
        mask: [B, T] boolean mask (True = valid)
        huber_delta: Huber loss delta

    Returns:
        scalar loss
    """
    # Normalize per trajectory
    B, T = student_potentials.shape
    student_norm = torch.zeros_like(student_potentials)
    teacher_norm = torch.zeros_like(teacher_phi)

    for b in range(B):
        s_valid = student_potentials[b]
        t_valid = teacher_phi[b]
        if mask is not None:
            valid = mask[b]
            s_valid = s_valid[valid]
            t_valid = t_valid[valid]
        if s_valid.numel() < 2:
            student_norm[b] = student_potentials[b]
            teacher_norm[b] = teacher_phi[b]
        else:
            student_norm[b] = normalize_tensor(student_potentials[b])
            teacher_norm[b] = normalize_tensor(teacher_phi[b].detach())

    loss = huber_loss(student_norm, teacher_norm, delta=huber_delta)

    if mask is not None:
        valid_mask = mask.float()
        loss = (huber_loss(student_norm, teacher_norm, delta=huber_delta) * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    return loss


def directional_difference_loss(
    student_potentials: torch.Tensor,
    teacher_phi: torch.Tensor,
    mask: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """L_direction: align potential differences.

    L_direction = Huber(
        V_theta(o_{t+1}, g) - V_theta(o_t, g),
        stop_gradient(Phi_{t+1} - Phi_t)
    )

    Args:
        student_potentials: [B, T]
        teacher_phi: [B, T] teacher MI potentials (detached)
        mask: [B, T] boolean mask
        huber_delta: Huber delta

    Returns:
        scalar loss
    """
    B, T = student_potentials.shape
    if T < 2:
        return torch.tensor(0.0, device=student_potentials.device, dtype=student_potentials.dtype)

    student_deltas = student_potentials[:, 1:] - student_potentials[:, :-1]  # [B, T-1]
    teacher_deltas = (teacher_phi[:, 1:] - teacher_phi[:, :-1]).detach()  # [B, T-1]

    if mask is not None:
        delta_mask = mask[:, 1:] & mask[:, :-1]  # [B, T-1]
    else:
        delta_mask = None

    if delta_mask is not None:
        # Normalize per trajectory
        loss_sum = 0.0
        count = 0
        for b in range(B):
            valid = delta_mask[b]
            if valid.sum() < 2:
                continue
            s_delta = student_deltas[b][valid]
            t_delta = teacher_deltas[b][valid]
            s_norm = normalize_tensor(s_delta)
            t_norm = normalize_tensor(t_delta)
            loss_sum += huber_loss(s_norm, t_norm, delta=huber_delta) * valid.sum().float()
            count += valid.sum().float()
        return loss_sum / count.clamp_min(1)
    else:
        # Normalize each trajectory
        loss_total = 0.0
        for b in range(B):
            s_norm = normalize_tensor(student_deltas[b])
            t_norm = normalize_tensor(teacher_deltas[b])
            loss_total += huber_loss(s_norm, t_norm, delta=huber_delta)
        return loss_total / B


def pairwise_ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """L_rank: Bradley-Terry pairwise ranking loss.

    L_rank = -log sigmoid(R_theta(chosen) - R_theta(rejected))

    Args:
        chosen_rewards: [B] or scalar
        rejected_rewards: [B] or scalar

    Returns:
        scalar loss
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def confidence_weighted_ranking_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    chosen_confidence: torch.Tensor,
    rejected_confidence: torch.Tensor,
    score_margin: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Confidence-weighted pairwise ranking loss.

    w_ij = sigmoid(score_margin / temperature) * sqrt(chosen_conf * rejected_conf)

    L_rank_weighted = -w_ij * log sigmoid(R(chosen) - R(rejected))

    Args:
        chosen_rewards: [B]
        rejected_rewards: [B]
        chosen_confidence: [B]
        rejected_confidence: [B]
        score_margin: [B] chosen_score - rejected_score
        temperature: sigmoid temperature for margin -> weight
        eps: numerical stability

    Returns:
        scalar weighted loss
    """
    margin_weight = torch.sigmoid(score_margin / temperature)
    confidence_weight = torch.sqrt((chosen_confidence * rejected_confidence).clamp_min(eps))
    weights = margin_weight * confidence_weight

    per_pair_loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    weighted_loss = (weights * per_pair_loss).sum() / weights.sum().clamp_min(eps)
    return weighted_loss


class DistillationLoss(nn.Module):
    """Combined multi-objective distillation loss.

    L_total = lambda_rank * L_rank + lambda_potential * L_potential + lambda_direction * L_direction
    """

    def __init__(
        self,
        lambda_rank: float = 1.0,
        lambda_potential: float = 1.0,
        lambda_direction: float = 0.5,
        confidence_temperature: float = 0.1,
        huber_delta: float = 1.0,
        use_confidence_weights: bool = True,
    ):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_potential = lambda_potential
        self.lambda_direction = lambda_direction
        self.confidence_temperature = confidence_temperature
        self.huber_delta = huber_delta
        self.use_confidence_weights = use_confidence_weights

    def forward(
        self,
        student_potentials_chosen: torch.Tensor,
        student_potentials_rejected: torch.Tensor,
        teacher_phi_chosen: torch.Tensor,
        teacher_phi_rejected: torch.Tensor,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        chosen_confidence: torch.Tensor,
        rejected_confidence: torch.Tensor,
        score_margin: torch.Tensor,
        mask_chosen: torch.Tensor | None = None,
        mask_rejected: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined distillation loss.

        Args:
            student_potentials_chosen: [B, T] potentials for chosen trajectories
            student_potentials_rejected: [B, T] potentials for rejected
            teacher_phi_chosen: [B, T] teacher MI potentials chosen
            teacher_phi_rejected: [B, T] teacher MI potentials rejected
            chosen_rewards: [B] trajectory-level rewards for chosen
            rejected_rewards: [B] trajectory-level rewards for rejected
            chosen_confidence: [B]
            rejected_confidence: [B]
            score_margin: [B]
            mask_chosen: [B, T] optional
            mask_rejected: [B, T] optional

        Returns:
            dict with loss components and total
        """
        # 1. Pairwise ranking
        if self.use_confidence_weights:
            rank_loss = confidence_weighted_ranking_loss(
                chosen_rewards, rejected_rewards,
                chosen_confidence, rejected_confidence, score_margin,
                temperature=self.confidence_temperature,
            )
        else:
            rank_loss = pairwise_ranking_loss(chosen_rewards, rejected_rewards)

        # 2. Potential distillation (average over chosen and rejected)
        pot_loss_chosen = potential_distillation_loss(
            student_potentials_chosen, teacher_phi_chosen, mask_chosen, self.huber_delta
        )
        pot_loss_rejected = potential_distillation_loss(
            student_potentials_rejected, teacher_phi_rejected, mask_rejected, self.huber_delta
        )
        pot_loss = 0.5 * (pot_loss_chosen + pot_loss_rejected)

        # 3. Directional difference distillation
        dir_loss_chosen = directional_difference_loss(
            student_potentials_chosen, teacher_phi_chosen, mask_chosen, self.huber_delta
        )
        dir_loss_rejected = directional_difference_loss(
            student_potentials_rejected, teacher_phi_rejected, mask_rejected, self.huber_delta
        )
        dir_loss = 0.5 * (dir_loss_chosen + dir_loss_rejected)

        # Total
        total = (
            self.lambda_rank * rank_loss
            + self.lambda_potential * pot_loss
            + self.lambda_direction * dir_loss
        )

        return {
            "total": total,
            "rank": rank_loss,
            "potential": pot_loss,
            "direction": dir_loss,
        }
