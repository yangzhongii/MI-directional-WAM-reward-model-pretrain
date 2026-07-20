"""Baseline scoring methods for ablation and comparison.

Provides inexpensive baselines that can be used to validate the
Dame-style MI potential against simpler similarity measures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_mse_score(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
) -> dict[str, object]:
    """Mean squared error in feature space (lower = more similar).

    Args:
        candidate_features: [T, D] pooled features
        reference_features: [S, D] pooled features

    Returns:
        dict with score (negated MSE, higher = better)
    """
    T = candidate_features.shape[0]
    S = reference_features.shape[0]
    mse_per_frame = []
    for t in range(T):
        # Nearest reference frame MSE
        diffs = candidate_features[t].unsqueeze(0) - reference_features
        mse = (diffs ** 2).mean(dim=-1).min()
        mse_per_frame.append(mse)
    avg_mse = torch.stack(mse_per_frame).mean()
    return {
        "score_delta": float((-avg_mse).item()),  # negate so higher is better
        "score_mean": float((-avg_mse).item()),
        "score_type": "pixel_mse",
    }


def latent_cosine_score(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
) -> dict[str, object]:
    """Mean cosine similarity in feature space.

    Args:
        candidate_features: [T, D]
        reference_features: [S, D]

    Returns:
        dict with cosine similarity score
    """
    cand_norm = F.normalize(candidate_features.float(), dim=-1)
    ref_norm = F.normalize(reference_features.float(), dim=-1)

    # Max cosine per candidate frame
    sim_matrix = cand_norm @ ref_norm.T  # [T, S]
    max_per_frame = sim_matrix.max(dim=1).values
    avg_cosine = max_per_frame.mean()

    return {
        "score_delta": float(avg_cosine.item()),
        "score_mean": float(avg_cosine.item()),
        "score_type": "latent_cosine",
    }


def pooled_correlation_score(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
) -> dict[str, object]:
    """Pooled-feature Pearson correlation.

    Args:
        candidate_features: [T, D]
        reference_features: [S, D]

    Returns:
        dict with correlation score
    """
    T = candidate_features.shape[0]
    S = reference_features.shape[0]

    # Pool reference to single vector
    ref_pooled = reference_features.mean(dim=0)  # [D]
    ref_pooled = (ref_pooled - ref_pooled.mean()) / (ref_pooled.std(unbiased=False) + 1e-8)

    corrs = []
    for t in range(T):
        cand = candidate_features[t]
        cand = (cand - cand.mean()) / (cand.std(unbiased=False) + 1e-8)
        corr = (cand * ref_pooled).mean()
        corrs.append(corr)

    avg_corr = torch.stack(corrs).mean()

    return {
        "score_delta": float(avg_corr.item()),
        "score_mean": float(avg_corr.item()),
        "score_type": "pooled_correlation",
    }


def unaligned_dame_mi_score(
    candidate_features: torch.Tensor,
    reference_features: torch.Tensor,
    mi_estimator,
) -> dict[str, object]:
    """Dame-style MI without temporal alignment (framewise max).

    Args:
        candidate_features: [T, N, D]
        reference_features: [S, N, D]
        mi_estimator: DameSoftHistogramMI instance

    Returns:
        dict with unaligned MI score
    """
    T = candidate_features.shape[0]
    S = reference_features.shape[0]
    mi_per_frame = []
    for t in range(T):
        frame_mi = []
        for s in range(S):
            mi_val = mi_estimator(candidate_features[t], reference_features[s])
            frame_mi.append(mi_val)
        mi_per_frame.append(torch.stack(frame_mi).max())

    phi = torch.stack(mi_per_frame)
    gamma = 0.99
    if phi.numel() >= 2:
        delta = (gamma * phi[1:] - phi[:-1]).mean()
    else:
        delta = torch.tensor(0.0)

    return {
        "score_delta": float(delta.item()),
        "score_mean": float(phi.mean().item()),
        "phi": [float(v.item()) for v in phi],
        "score_type": "unaligned_dame_mi",
    }


SCORING_METHODS = {
    "pixel_mse": pixel_mse_score,
    "latent_cosine": latent_cosine_score,
    "pooled_correlation": pooled_correlation_score,
    "unaligned_dame_mi": unaligned_dame_mi_score,
}
