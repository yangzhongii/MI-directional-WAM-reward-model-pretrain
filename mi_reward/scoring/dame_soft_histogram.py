"""Dame-style B-spline soft-histogram mutual information estimator.

Reference:
  Dame, A. and Marchand, E., "Mutual Information-Based Visual Servoing,"
  IEEE Transactions on Robotics, 2011.

This module provides a numerically stable, differentiable MI estimator using
B-spline kernels for soft histogram binning. It supports channelwise and
flattened modes, robust normalization, and optional normalized MI output.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bspline_kernel(
    x: torch.Tensor,
    centers: torch.Tensor,
    order: int = 3,
    bin_width: float = 1.0,
) -> torch.Tensor:
    """Evaluate B-spline kernel of given order at x relative to centers.

    Args:
        x: [*batch, D] sample values in normalized space
        centers: [num_bins] bin centers
        order: B-spline order (1=linear, 2=quadratic, 3=cubic)
        bin_width: width of each bin

    Returns:
        [*batch, num_bins] soft bin assignments
    """
    x_expanded = x.unsqueeze(-1)  # [*batch, D, 1]
    centers_expanded = centers.view(*([1] * (x.ndim)), -1)  # [1, ..., 1, num_bins]
    diff = (x_expanded - centers_expanded) / bin_width  # [*batch, D, num_bins]
    abs_diff = diff.abs()

    if order == 1:
        # Linear B-spline: 1 - |d| for |d| < 1
        result = torch.clamp(1.0 - abs_diff, min=0.0)
    elif order == 2:
        # Quadratic B-spline
        result = torch.where(
            abs_diff < 0.5,
            0.75 - abs_diff ** 2,
            torch.where(abs_diff < 1.5, 0.5 * (1.5 - abs_diff) ** 2, torch.zeros_like(abs_diff)),
        )
    elif order == 3:
        # Cubic B-spline
        result = torch.where(
            abs_diff < 1.0,
            2.0 / 3.0 - abs_diff ** 2 + 0.5 * abs_diff ** 3,
            torch.where(abs_diff < 2.0, (2.0 - abs_diff) ** 3 / 6.0, torch.zeros_like(abs_diff)),
        )
    else:
        raise ValueError(f"Unsupported B-spline order: {order}")

    return result


def _robust_minmax_normalize(
    x: torch.Tensor,
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust quantile-based min-max normalization.

    Args:
        x: [..., D] input tensor
        low_quantile: lower quantile for robust range
        high_quantile: upper quantile for robust range
        eps: numerical stability

    Returns:
        normalized: [..., D]
        shift: [D] shift values
        scale: [D] scale values
    """
    flat = x.reshape(-1, x.shape[-1])
    q_low = torch.quantile(flat, low_quantile, dim=0)
    q_high = torch.quantile(flat, high_quantile, dim=0)
    shift = q_low
    scale = (q_high - q_low).clamp_min(eps)
    normalized = (x - shift) / scale
    return normalized, shift, scale


class DameSoftHistogramMI(nn.Module):
    """Dame-style B-spline soft-histogram mutual information estimator.

    Implementation follows Dame & Marchand (2011) with adaptations for
    latent token features instead of image pixels.

    Supports two modes:
      - channelwise: compute MI per channel, aggregate
      - flattened: treat all token-channel entries as samples (ablation)
    """

    def __init__(
        self,
        num_bins: int = 8,
        spline_order: int = 3,
        eps: float = 1e-8,
        reduction: Literal["mean", "sum", "none"] = "mean",
        normalization: Literal["robust_minmax", "zscore", "fixed", "none"] = "robust_minmax",
        channel_mode: Literal["channelwise", "flattened"] = "channelwise",
        normalized_mi: bool = False,
        bin_range: tuple[float, float] = (-3.0, 3.0),
    ):
        """
        Args:
            num_bins: number of histogram bins per dimension
            spline_order: B-spline order (1=linear, 2=quadratic, 3=cubic)
            eps: small constant for numerical stability
            reduction: how to aggregate channelwise MI values
            normalization: normalization strategy
            channel_mode: "channelwise" or "flattened"
            normalized_mi: if True, return NMI = MI / sqrt(H(X) * H(Y))
            bin_range: (min, max) of normalized value range for fixed bins
        """
        super().__init__()
        self.num_bins = num_bins
        self.spline_order = spline_order
        self.eps = eps
        self.reduction = reduction
        self.normalization = normalization
        self.channel_mode = channel_mode
        self.normalized_mi = normalized_mi
        self.bin_range = bin_range

        # Register bin centers
        bin_min, bin_max = bin_range
        centers = torch.linspace(bin_min, bin_max, num_bins)
        self.register_buffer("centers", centers)
        self.bin_width = (bin_max - bin_min) / max(num_bins - 1, 1)

        # Running statistics for fixed normalization
        self.register_buffer("running_shift", torch.zeros(1))
        self.register_buffer("running_scale", torch.ones(1))
        self.register_buffer("running_count", torch.zeros(1))

    def _normalize(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        update_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize x and y jointly for valid cross-sample comparison.

        Args:
            x: [..., D]
            y: [..., D]
            update_stats: if True, update running statistics

        Returns:
            x_norm, y_norm: [..., D]
        """
        if self.normalization == "robust_minmax":
            # Concatenate for joint normalization
            combined = torch.cat([x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])], dim=0)
            _, shift, scale = _robust_minmax_normalize(combined, eps=self.eps)
            x_norm = (x - shift) / scale
            y_norm = (y - shift) / scale
        elif self.normalization == "zscore":
            combined = torch.cat([x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])], dim=0)
            mean = combined.mean(dim=0)
            std = combined.std(dim=0, unbiased=False).clamp_min(self.eps)
            x_norm = (x - mean) / std
            y_norm = (y - mean) / std
        elif self.normalization == "fixed":
            if update_stats or self.running_count.item() == 0:
                combined = torch.cat(
                    [x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])], dim=0
                )
                _, shift, scale = _robust_minmax_normalize(combined, eps=self.eps)
                self.running_shift = shift.detach()
                self.running_scale = scale.detach()
                self.running_count.fill_(1)
            shift = self.running_shift.to(x.device)
            scale = self.running_scale.to(x.device)
            x_norm = (x - shift) / scale
            y_norm = (y - shift) / scale
        elif self.normalization == "none":
            x_norm, y_norm = x, y
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        # Clamp to bin range
        bin_min, bin_max = self.bin_range
        x_norm = torch.clamp(x_norm, bin_min, bin_max)
        y_norm = torch.clamp(y_norm, bin_min, bin_max)
        return x_norm, y_norm

    def _compute_histograms(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute soft joint and marginal histograms.

        Args:
            x_norm: [N, D] normalized x samples
            y_norm: [N, D] normalized y samples

        Returns:
            p_xy: [num_bins, num_bins] joint distribution
            p_x: [num_bins] marginal
            p_y: [num_bins] marginal
        """
        # Single-sample path for channelwise: [N, D] where N is num_tokens, D is num_channels
        # For channelwise mode, we iterate over channels
        N, D = x_norm.shape
        eps = self.eps

        # B-spline soft assignment: [N, D, num_bins]
        x_bins = _bspline_kernel(x_norm, self.centers, order=self.spline_order, bin_width=self.bin_width)
        y_bins = _bspline_kernel(y_norm, self.centers, order=self.spline_order, bin_width=self.bin_width)

        # Joint distribution: sum over N of outer product, then average over D
        # [N, D, num_bins, num_bins]
        joint_unnorm = torch.einsum("ndb,ndc->bcd", x_bins, y_bins)  # [num_bins, num_bins]
        joint = joint_unnorm / (joint_unnorm.sum() + eps)

        # Marginals
        p_x = joint.sum(dim=1)  # [num_bins]
        p_y = joint.sum(dim=0)  # [num_bins]

        return joint, p_x, p_y

    def _compute_channelwise_mi(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MI per channel and aggregate.

        Args:
            x_norm: [N, D] normalized x tokens
            y_norm: [N, D] normalized y tokens

        Returns:
            scalar MI value
        """
        N, D = x_norm.shape
        eps = self.eps
        mi_per_channel = []

        for d in range(D):
            x_d = x_norm[:, d:d + 1]  # [N, 1]
            y_d = y_norm[:, d:d + 1]  # [N, 1]

            # B-spline soft assignment for this channel
            x_bins = _bspline_kernel(
                x_d, self.centers, order=self.spline_order, bin_width=self.bin_width
            )  # [N, 1, num_bins]
            y_bins = _bspline_kernel(
                y_d, self.centers, order=self.spline_order, bin_width=self.bin_width
            )  # [N, 1, num_bins]

            # Collapse the singleton channel dim
            x_bins = x_bins.squeeze(1)  # [N, num_bins]
            y_bins = y_bins.squeeze(1)  # [N, num_bins]

            # Joint: [num_bins, num_bins]
            joint_unnorm = torch.einsum("nb,nc->bc", x_bins, y_bins)
            joint = joint_unnorm / (joint_unnorm.sum() + eps)

            p_x = joint.sum(dim=1)  # [num_bins]
            p_y = joint.sum(dim=0)  # [num_bins]

            # MI = sum_ij p_ij * log(p_ij / (p_i * p_j))
            p_x_p_y = p_x.unsqueeze(1) @ p_y.unsqueeze(0)  # [num_bins, num_bins]
            mi = (joint * (torch.log(joint + eps) - torch.log(p_x_p_y + eps))).sum()
            mi_per_channel.append(mi)

        mi_tensor = torch.stack(mi_per_channel)  # [D]

        if self.normalized_mi:
            # Compute entropies for normalization
            # (using the joint already computed; re-derive H from last channel is approximate)
            # For NMI we recompute per channel
            nmi_vals = []
            for d in range(D):
                x_d = x_norm[:, d:d + 1]
                y_d = y_norm[:, d:d + 1]
                x_bins_d = _bspline_kernel(x_d, self.centers, order=self.spline_order, bin_width=self.bin_width).squeeze(1)
                y_bins_d = _bspline_kernel(y_d, self.centers, order=self.spline_order, bin_width=self.bin_width).squeeze(1)

                joint_d = torch.einsum("nb,nc->bc", x_bins_d, y_bins_d)
                joint_d = joint_d / (joint_d.sum() + eps)
                p_x_d = joint_d.sum(dim=1)
                p_y_d = joint_d.sum(dim=0)

                hx = -(p_x_d * torch.log(p_x_d + eps)).sum()
                hy = -(p_y_d * torch.log(p_y_d + eps)).sum()
                mi_d = (joint_d * (torch.log(joint_d + eps) - torch.log(p_x_d.unsqueeze(1) @ p_y_d.unsqueeze(0) + eps))).sum()
                denom = torch.sqrt((hx + eps) * (hy + eps))
                nmi_vals.append(mi_d / denom)

            mi_tensor = torch.stack(nmi_vals)

        if self.reduction == "mean":
            return mi_tensor.mean()
        elif self.reduction == "sum":
            return mi_tensor.sum()
        else:
            return mi_tensor

    def _compute_flattened_mi(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MI treating all (token, channel) entries as samples.

        Args:
            x_norm: [N, D]
            y_norm: [N, D]

        Returns:
            scalar MI
        """
        eps = self.eps
        # Flatten to [N*D, 1]
        x_flat = x_norm.reshape(-1, 1)
        y_flat = y_norm.reshape(-1, 1)

        x_bins = _bspline_kernel(x_flat, self.centers, order=self.spline_order, bin_width=self.bin_width).squeeze(1)
        y_bins = _bspline_kernel(y_flat, self.centers, order=self.spline_order, bin_width=self.bin_width).squeeze(1)

        joint_unnorm = torch.einsum("nb,nc->bc", x_bins, y_bins)
        joint = joint_unnorm / (joint_unnorm.sum() + eps)

        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)
        p_x_p_y = p_x.unsqueeze(1) @ p_y.unsqueeze(0)

        mi = (joint * (torch.log(joint + eps) - torch.log(p_x_p_y + eps))).sum()

        if self.normalized_mi:
            hx = -(p_x * torch.log(p_x + eps)).sum()
            hy = -(p_y * torch.log(p_y + eps)).sum()
            denom = torch.sqrt((hx + eps) * (hy + eps))
            mi = mi / denom

        return mi

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MI between x and y token features.

        Args:
            x: [D] single vector, [N, D] token sequence, or [B, T, N, D] batch
            y: same shape as x

        Returns:
            scalar MI value
        """
        # Shape normalization
        x_orig = x
        y_orig = y

        # Ensure at least [N, D]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, D]
        if y.dim() == 1:
            y = y.unsqueeze(0)  # [1, D]

        # Handle [B, T, N, D] by reshaping to [(B*T*N), D]
        if x.dim() == 4:
            B, T_x, N_x, D_x = x.shape
            x = x.reshape(-1, D_x)
        if y.dim() == 4:
            B_y, T_y, N_y, D_y = y.shape
            y = y.reshape(-1, D_y)

        # Handle [B, N, D] by reshaping to [B*N, D]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        if y.dim() == 3:
            y = y.reshape(-1, y.shape[-1])

        # Verify compatible shapes
        if x.shape[0] != y.shape[0]:
            # Truncate to minimum
            min_len = min(x.shape[0], y.shape[0])
            x = x[:min_len]
            y = y[:min_len]

        D = x.shape[-1]
        if D == 0:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        # Normalize
        x_norm, y_norm = self._normalize(x, y)

        # Compute MI
        if self.channel_mode == "channelwise" and D > 1:
            return self._compute_channelwise_mi(x_norm, y_norm)
        else:
            return self._compute_flattened_mi(x_norm, y_norm)

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, spline_order={self.spline_order}, "
            f"channel_mode={self.channel_mode}, normalized_mi={self.normalized_mi}, "
            f"normalization={self.normalization}"
        )
