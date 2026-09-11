from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_vector(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value.reshape(value.shape[0], -1).mean(dim=0)


def _align_vectors(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = _as_vector(x)
    y = _as_vector(y)
    dim = min(x.numel(), y.numel())
    if dim == 0:
        raise ValueError("Cannot compute MI on empty features.")
    return x[:dim], y[:dim]


def _gaussian_mi_proxy(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x, y = _align_vectors(x, y)
    if x.numel() < 2:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    x = (x - x.mean()) / (x.std(unbiased=False) + eps)
    y = (y - y.mean()) / (y.std(unbiased=False) + eps)
    corr = torch.clamp((x * y).mean(), min=-1.0 + eps, max=1.0 - eps)
    return -0.5 * torch.log1p(-(corr * corr))


def _histogram_mi(x: torch.Tensor, y: torch.Tensor, bins: int = 16, sigma: float = 0.08, eps: float = 1e-8) -> torch.Tensor:
    x, y = _align_vectors(x, y)
    x = torch.tanh((x - x.mean()) / (x.std(unbiased=False) + eps))
    y = torch.tanh((y - y.mean()) / (y.std(unbiased=False) + eps))
    centers = torch.linspace(-1.0, 1.0, bins, device=x.device, dtype=x.dtype)
    x_bins = F.softmax(-((x[:, None] - centers[None, :]) ** 2) / sigma, dim=-1)
    y_bins = F.softmax(-((y[:, None] - centers[None, :]) ** 2) / sigma, dim=-1)
    joint = torch.einsum("nb,nc->bc", x_bins, y_bins)
    joint = joint / (joint.sum() + eps)
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    return (joint * (torch.log(joint + eps) - torch.log(px @ py + eps))).sum()


def compute_mi(x: torch.Tensor, y: torch.Tensor, mode: str = "gaussian_mi_proxy") -> torch.Tensor:
    if mode == "gaussian_mi_proxy":
        return _gaussian_mi_proxy(x, y)
    if mode == "histogram_mi":
        return _histogram_mi(x, y)
    raise ValueError(f"Unsupported MI mode: {mode}")
