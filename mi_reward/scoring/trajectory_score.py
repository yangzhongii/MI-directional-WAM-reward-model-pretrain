from __future__ import annotations

import torch

from mi_reward.scoring.mi_potential import compute_mi


def _as_frame_sequence(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim == 2:
        return features
    return features.reshape(features.shape[0], -1)


def compute_frame_potential(candidate_features: torch.Tensor, success_features: torch.Tensor, mi_mode: str) -> torch.Tensor:
    candidate = _as_frame_sequence(candidate_features)
    success = _as_frame_sequence(success_features)
    potentials = []
    for cand_frame in candidate:
        frame_scores = torch.stack([compute_mi(cand_frame, success_frame, mode=mi_mode) for success_frame in success])
        potentials.append(frame_scores.max())
    return torch.stack(potentials)


def compute_delta_score(phi: torch.Tensor, gamma: float) -> float:
    phi = phi.float().flatten()
    if phi.numel() < 2:
        return 0.0
    deltas = gamma * phi[1:] - phi[:-1]
    return float(deltas.mean().item())


def score_trajectory(
    candidate_features: torch.Tensor,
    success_ref_features: torch.Tensor,
    gamma: float,
    mi_mode: str,
) -> dict[str, object]:
    phi = compute_frame_potential(candidate_features, success_ref_features, mi_mode)
    return {
        "phi": [float(value) for value in phi.detach().cpu().tolist()],
        "score_delta": compute_delta_score(phi, gamma),
        "score_mean": float(phi.mean().item()) if phi.numel() else 0.0,
    }
