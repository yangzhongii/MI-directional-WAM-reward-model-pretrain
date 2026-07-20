"""Monotonic temporal alignment for MI potential computation.

Implements Viterbi-style dynamic programming for monotonic alignment
between candidate and reference trajectories, inspired by the temporal
correspondence principles in Dame & Marchand (2011).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mi_alignment_matrix(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
    mi_estimator: nn.Module,
) -> torch.Tensor:
    """Build pairwise MI alignment matrix between candidate and reference frames.

    For each candidate frame t and reference frame s, compute the MI between
    their token features using the provided estimator.

    Args:
        candidate_features: [T, N, D] candidate token features
        reference_features: [S, N, D] reference token features
        mi_estimator: MI estimator (e.g., DameSoftHistogramMI)

    Returns:
        alignment_matrix: [T, S] where A[t, s] = MI(Z_cand[t], Z_ref[s])
    """
    T = candidate_features.shape[0]
    S = reference_features.shape[0]
    alignment = torch.zeros(T, S, device=candidate_features.device, dtype=torch.float32)

    for t in range(T):
        cand_t = candidate_features[t]  # [N, D]
        for s in range(S):
            ref_s = reference_features[s]  # [N, D]
            mi_val = mi_estimator(cand_t, ref_s)
            alignment[t, s] = mi_val

    return alignment


def monotonic_viterbi_alignment(
    alignment_matrix: torch.Tensor,
    max_step: int | None = None,
    stay_penalty: float = 0.0,
    jump_penalty: float = 0.0,
) -> dict:
    """Find monotonic alignment path maximizing cumulative MI score.

    Uses dynamic programming with optional constraints on step size and
    penalties for staying or jumping.

    The path pi satisfies:
      - pi(t + 1) >= pi(t)  (monotonic)
      - pi(t + 1) - pi(t) <= max_step  (optional step constraint)

    Args:
        alignment_matrix: [T, S] pairwise MI matrix
        max_step: maximum allowed step forward (None = no constraint)
        stay_penalty: penalty for staying at same reference frame
        jump_penalty: penalty per step jumped

    Returns:
        dict with:
          path: LongTensor[T] alignment indices into reference
          aligned_potential: FloatTensor[T] MI values along path
          total_score: scalar total cumulative score
    """
    T, S = alignment_matrix.shape
    device = alignment_matrix.device

    # DP table: dp[t, s] = max cumulative score ending at (t, s)
    dp = torch.full((T, S), float("-inf"), device=device)
    # Backpointer: bp[t, s] = previous s index
    bp = torch.zeros((T, S), dtype=torch.long, device=device)

    # Initialize first frame: any reference frame is reachable
    dp[0] = alignment_matrix[0].clone()

    # Forward DP
    for t in range(1, T):
        for s in range(S):
            # Consider all previous reference indices s_prev <= s (monotonic)
            if max_step is not None:
                min_prev_s = max(0, s - max_step)
            else:
                min_prev_s = 0

            if min_prev_s > s:
                continue

            # Extract valid previous states
            prev_scores = dp[t - 1, min_prev_s:s + 1]

            if prev_scores.numel() == 0 or prev_scores.max().item() == float("-inf"):
                continue

            best_prev_idx = prev_scores.argmax()
            best_prev_s = min_prev_s + best_prev_idx.item()
            best_score = prev_scores[best_prev_idx]

            # Apply penalties
            step_size = s - best_prev_s
            penalty = 0.0
            if step_size == 0:
                penalty += stay_penalty
            elif step_size > 1:
                penalty += (step_size - 1) * jump_penalty

            dp[t, s] = best_score + alignment_matrix[t, s] - penalty
            bp[t, s] = best_prev_s

    # Backtrack: find best end state
    total_score, last_s = dp[-1].max(dim=0)
    last_s = last_s.item()

    # If no valid path found, use greedy monotonic
    if total_score.item() == float("-inf"):
        path = _greedy_monotonic_path(alignment_matrix)
        aligned_potential = alignment_matrix[torch.arange(T, device=device), path]
        total_score = aligned_potential.sum()
        return {
            "path": path,
            "aligned_potential": aligned_potential,
            "total_score": total_score,
        }

    # Backtrack
    path_rev = [last_s]
    for t in range(T - 1, 0, -1):
        prev_s = bp[t, path_rev[-1]].item()
        path_rev.append(prev_s)

    path = torch.tensor(list(reversed(path_rev)), dtype=torch.long, device=device)
    aligned_potential = alignment_matrix[torch.arange(T, device=device), path]

    return {
        "path": path,
        "aligned_potential": aligned_potential,
        "total_score": total_score,
    }


def _greedy_monotonic_path(alignment_matrix: torch.Tensor) -> torch.Tensor:
    """Fallback greedy monotonic path when DP fails.

    At each timestep, choose the reference frame with maximum MI
    that satisfies monotonic constraint.
    """
    T, S = alignment_matrix.shape
    device = alignment_matrix.device
    path = torch.zeros(T, dtype=torch.long, device=device)
    current_s = 0

    for t in range(T):
        # Consider all s >= current_s
        if current_s >= S:
            current_s = S - 1
        sub_scores = alignment_matrix[t, current_s:]
        if sub_scores.numel() == 0:
            path[t] = current_s
        else:
            offset = sub_scores.argmax().item()
            current_s = current_s + offset
            path[t] = current_s

    return path


def soft_dtw_alignment(
    alignment_matrix: torch.Tensor,
    temperature: float = 1.0,
    gamma: float = 1.0,
) -> dict:
    """Soft-DTW differentiable monotonic alignment.

    Provides a differentiable alternative to hard Viterbi alignment.
    Should only be used after the DP implementation is verified.

    Args:
        alignment_matrix: [T, S]
        temperature: softmin temperature
        gamma: DTW gamma parameter (0 for classic DTW)

    Returns:
        dict with soft path probabilities and aligned potential
    """
    T, S = alignment_matrix.shape
    device = alignment_matrix.device
    # neg_cost for DTW: maximize MI = minimize -MI
    neg_mi = -alignment_matrix

    # Soft-DTW forward recursion
    # R[t, s] = neg_mi[t, s] + min_gamma(R[t-1, s-1], R[t-1, s], R[t, s-1])
    R = torch.full((T + 1, S + 1), float("inf"), device=device)
    R[0, 0] = 0.0

    for t in range(1, T + 1):
        for s in range(1, S + 1):
            if gamma == 0.0:
                R[t, s] = neg_mi[t - 1, s - 1] + min(R[t - 1, s - 1], R[t - 1, s], R[t, s - 1])
            else:
                # Soft-min with gamma
                vals = torch.stack([R[t - 1, s - 1], R[t - 1, s], R[t, s - 1]])
                min_val = -gamma * torch.logsumexp(-vals / gamma, dim=0)
                R[t, s] = neg_mi[t - 1, s - 1] + min_val

    # Soft backward to get alignment probabilities (simplified)
    # Returns hard path from soft-DTW for simplicity
    path = torch.zeros(T, dtype=torch.long, device=device)
    t, s = T, S
    while t > 0 and s > 0:
        path[t - 1] = s - 1
        if t == 1 and s == 1:
            break
        if t == 1:
            s -= 1
        elif s == 1:
            t -= 1
        else:
            choices = torch.tensor([R[t - 1, s - 1], R[t - 1, s], R[t, s - 1]], device=device)
            move = choices.argmin().item()
            if move == 0:
                t -= 1
                s -= 1
            elif move == 1:
                t -= 1
            else:
                s -= 1

    aligned_potential = alignment_matrix[torch.arange(T, device=device), path]
    total_score = aligned_potential.sum()

    return {
        "path": path,
        "aligned_potential": aligned_potential,
        "total_score": total_score,
    }
