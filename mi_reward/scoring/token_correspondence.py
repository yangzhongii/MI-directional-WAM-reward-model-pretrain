"""Token correspondence strategies for pairing candidate and reference tokens.

Dame's joint histogram pairs corresponding image pixels. For latent visual
tokens, we need explicit correspondence rules. This module provides three
strategies with `same_index` as the default MVP.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def apply_correspondence(
    candidate_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    mode: Literal["same_index", "nearest_cosine", "pooled_window"] = "same_index",
    window_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair candidate and reference tokens according to a correspondence strategy.

    Args:
        candidate_tokens: [N, D] or [T, N, D] candidate token features
        reference_tokens: [S, N, D] or [N, D] reference token features
        mode: correspondence strategy
        window_size: local window size for pooled_window mode

    Returns:
        paired_candidate: [M, D] paired candidate tokens
        paired_reference: [M, D] paired reference tokens
    """
    if mode == "same_index":
        return _same_index_correspondence(candidate_tokens, reference_tokens)
    elif mode == "nearest_cosine":
        return _nearest_cosine_correspondence(candidate_tokens, reference_tokens)
    elif mode == "pooled_window":
        return _pooled_window_correspondence(candidate_tokens, reference_tokens, window_size)
    else:
        raise ValueError(f"Unknown correspondence mode: {mode}")


def _same_index_correspondence(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use tokens at the same spatial/temporal position.

    Args:
        candidate: [N, D] or [T, N, D]
        reference: [N, D] or [S, N, D]

    Returns:
        paired_candidate: [M, D]
        paired_reference: [M, D]
    """
    # Handle temporal dimension
    if candidate.dim() == 3 and reference.dim() == 3:
        # [T, N, D] vs [S, N, D] — take first frame pair as same-index
        # For temporal MI, correspondence is handled by alignment, not here
        T, N_c, D = candidate.shape
        S, N_r, _ = reference.shape
        min_N = min(N_c, N_r)
        # Return flattened pairs matching same spatial indices
        cand_paired = candidate[:, :min_N, :].reshape(-1, D)
        ref_paired = reference[:, :min_N, :].reshape(-1, D)
        return cand_paired, ref_paired
    elif candidate.dim() == 2 and reference.dim() == 2:
        # [N, D] vs [M, D]
        min_N = min(candidate.shape[0], reference.shape[0])
        return candidate[:min_N], reference[:min_N]
    elif candidate.dim() == 3:
        # [T, N, D] vs [N, D] — expand reference to match
        T, N_c, D = candidate.shape
        min_N = min(N_c, reference.shape[0])
        ref_expanded = reference[:min_N].unsqueeze(0).expand(T, -1, -1).reshape(-1, D)
        cand_flat = candidate[:, :min_N, :].reshape(-1, D)
        return cand_flat, ref_expanded
    elif reference.dim() == 3:
        # [N, D] vs [T, N, D]
        N, D = candidate.shape
        T, N_r, _ = reference.shape
        min_N = min(N, N_r)
        cand_expanded = candidate[:min_N].unsqueeze(0).expand(T, -1, -1).reshape(-1, D)
        ref_flat = reference[:, :min_N, :].reshape(-1, D)
        return cand_expanded, ref_flat
    else:
        # Fallback
        return candidate, reference


def _nearest_cosine_correspondence(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match each candidate token to nearest reference token by cosine similarity.

    Uses detached cosine similarity for stability — the matching is treated
    as a fixed correspondence, not a differentiable operation.

    Args:
        candidate: [N, D] candidate tokens
        reference: [M, D] reference tokens

    Returns:
        paired_candidate: [N, D]
        paired_reference: [N, D]
    """
    if candidate.dim() == 3:
        candidate = candidate.reshape(-1, candidate.shape[-1])
    if reference.dim() == 3:
        reference = reference.reshape(-1, reference.shape[-1])

    # Normalize for cosine similarity
    cand_norm = F.normalize(candidate.float(), dim=-1)
    ref_norm = F.normalize(reference.float(), dim=-1)

    # Cosine similarity matrix: [N, M]
    sim = cand_norm @ ref_norm.T

    # Nearest neighbor for each candidate (detached)
    nn_indices = sim.argmax(dim=-1).detach()  # [N]

    paired_ref = reference[nn_indices]  # [N, D]
    return candidate, paired_ref


def _pooled_window_correspondence(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    window_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool local token windows before pairing.

    Args:
        candidate: [N, D] or [T, N, D]
        reference: [M, D] or [S, N, D]
        window_size: pooling window size

    Returns:
        paired_candidate: [N // window_size, D]
        paired_reference: [M // window_size, D]
    """
    if window_size <= 1:
        return _same_index_correspondence(candidate, reference)

    def _pool_tokens(tokens: torch.Tensor, ws: int) -> torch.Tensor:
        """Average pool tokens in windows of size ws along dim 0."""
        N = tokens.shape[0]
        if N < ws:
            return tokens
        # Truncate to multiple of ws
        truncated = tokens[: (N // ws) * ws]
        return truncated.reshape(-1, ws, tokens.shape[-1]).mean(dim=1)

    if candidate.dim() == 3:
        T, N, D = candidate.shape
        cand_pooled = _pool_tokens(candidate.reshape(-1, D), window_size)
        ref_pooled = _pool_tokens(reference.reshape(-1, D) if reference.dim() == 3 else reference, window_size)
        min_len = min(cand_pooled.shape[0], ref_pooled.shape[0])
        return cand_pooled[:min_len], ref_pooled[:min_len]

    cand_pooled = _pool_tokens(candidate, window_size)
    ref_pooled = _pool_tokens(reference, window_size)
    min_len = min(cand_pooled.shape[0], ref_pooled.shape[0])
    return cand_pooled[:min_len], ref_pooled[:min_len]
