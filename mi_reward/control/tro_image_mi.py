"""Image-only Dame--Marchand style cubic-B-spline mutual information.

This module deliberately has no latent normalization, policy, or reward-model
dependency.  Its padded histogram support preserves probability mass at the
intensity boundaries after the literal ``[0, N_c]`` intensity scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TROHistogramAudit:
    current_mass: float
    reference_mass: float
    joint_mass: float
    active_bins: int


def cubic_b_spline(value: torch.Tensor) -> torch.Tensor:
    """Centered cardinal cubic B-spline with support ``[-2, 2]``."""
    absolute = value.abs()
    inner = (4.0 - 6.0 * absolute.square() + 3.0 * absolute.pow(3)) / 6.0
    outer = (2.0 - absolute).clamp_min(0.0).pow(3) / 6.0
    return torch.where(absolute < 1.0, inner, torch.where(absolute < 2.0, outer, torch.zeros_like(value)))


def cubic_b_spline_prime(value: torch.Tensor) -> torch.Tensor:
    """First derivative of the centered cardinal cubic B-spline."""
    absolute = value.abs()
    inner = -2.0 * value + 1.5 * value * absolute
    outer = -0.5 * (2.0 - absolute).clamp_min(0.0).square() * value.sign()
    return torch.where(absolute < 1.0, inner, torch.where(absolute < 2.0, outer, torch.zeros_like(value)))


def cubic_b_spline_second(value: torch.Tensor) -> torch.Tensor:
    """Second derivative of the centered cardinal cubic B-spline."""
    absolute = value.abs()
    inner = -2.0 + 3.0 * absolute
    outer = (2.0 - absolute).clamp_min(0.0)
    return torch.where(absolute < 1.0, inner, torch.where(absolute < 2.0, outer, torch.zeros_like(value)))


class TROImageMI:
    """Literal grayscale image MI with a fixed padded B3 histogram support."""

    def __init__(self, bins: int = 8, padding: int = 2) -> None:
        if bins < 2 or padding < 2:
            raise ValueError("TRO image MI requires bins >= 2 and B3 padding >= 2.")
        self.bins, self.padding = int(bins), int(padding)

    def _weights(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 2:
            raise ValueError(f"Expected one grayscale image [H,W], got {tuple(image.shape)}.")
        scaled = image.to(dtype=torch.float64).clamp(0.0, 255.0) * (self.bins / 255.0)
        knots = torch.arange(-self.padding, self.bins + self.padding + 1, device=image.device, dtype=torch.float64)
        return cubic_b_spline(knots[:, None] - scaled.reshape(1, -1))

    def printed_gradient(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        image_pose_jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate paper (16), (18) using a supplied ``dI/dr`` per pixel.

        ``image_pose_jacobian`` has shape ``[H,W,D]`` and is expressed for the
        unscaled [0,255] image. This function applies the (12) intensity scale.
        """
        if current.shape != reference.shape or image_pose_jacobian.shape[:2] != current.shape:
            raise ValueError("Image and per-pixel pose-Jacobian shapes do not match.")
        current64 = current.to(dtype=torch.float64).clamp(0.0, 255.0)
        scaled = current64 * (self.bins / 255.0)
        knots = torch.arange(-self.padding, self.bins + self.padding + 1, device=current.device, dtype=torch.float64)
        argument = knots[:, None] - scaled.reshape(1, -1)
        current_weight = cubic_b_spline(argument)
        reference_weight = self._weights(reference)
        scaled_jacobian = image_pose_jacobian.reshape(-1, image_pose_jacobian.shape[-1]).to(torch.float64) * (self.bins / 255.0)
        weight_jacobian = -cubic_b_spline_prime(argument)[:, :, None] * scaled_jacobian[None, :, :]
        pixels = current.numel()
        marginal = current_weight.sum(dim=1) / pixels
        joint = (current_weight @ reference_weight.T) / pixels
        joint_jacobian = torch.einsum("ipd,jp->ijd", weight_jacobian, reference_weight) / pixels
        support = joint > 0.0
        coefficient = torch.zeros_like(joint)
        coefficient[support] = 1.0 + torch.log(joint[support] / marginal[:, None].expand_as(joint)[support])
        return torch.einsum("ijd,ij->d", joint_jacobian, coefficient)

    def printed_hessian(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        image_pose_jacobian: torch.Tensor,
        image_pose_hessian: torch.Tensor,
        exact_derivative: bool = False,
    ) -> torch.Tensor:
        """Evaluate the Hessian exactly as printed in paper equation (17).

        The second-probability coefficient is literally ``p_joint / p_current``;
        it is intentionally not replaced by the derivative of equation (16).
        """
        if image_pose_hessian.shape[:2] != current.shape or image_pose_hessian.shape[-2:] != (image_pose_jacobian.shape[-1],) * 2:
            raise ValueError("Per-pixel image Hessian shape does not match the image Jacobian.")
        current64 = current.to(torch.float64).clamp(0.0, 255.0)
        scaled = current64 * (self.bins / 255.0)
        knots = torch.arange(-self.padding, self.bins + self.padding + 1, device=current.device, dtype=torch.float64)
        argument = knots[:, None] - scaled.reshape(1, -1)
        weight = cubic_b_spline(argument)
        reference_weight = self._weights(reference)
        scale = self.bins / 255.0
        jacobian = image_pose_jacobian.reshape(-1, image_pose_jacobian.shape[-1]).to(torch.float64) * scale
        image_hessian = image_pose_hessian.reshape(-1, *image_pose_hessian.shape[-2:]).to(torch.float64) * scale
        weight_jacobian = -cubic_b_spline_prime(argument)[:, :, None] * jacobian[None]
        weight_hessian = cubic_b_spline_second(argument)[:, :, None, None] * torch.einsum("pd,pe->pde", jacobian, jacobian)[None] - cubic_b_spline_prime(argument)[:, :, None, None] * image_hessian[None]
        pixels = current.numel()
        marginal = weight.sum(1) / pixels
        joint = weight @ reference_weight.T / pixels
        dp = torch.einsum("ipd,jp->ijd", weight_jacobian, reference_weight) / pixels
        d2p = torch.einsum("ipde,jp->ijde", weight_hessian, reference_weight) / pixels
        support = joint > 0.0
        curvature = torch.zeros_like(joint)
        curvature[support] = 1.0 / joint[support] - 1.0 / marginal[:, None].expand_as(joint)[support]
        first = torch.einsum("ijd,ije,ij->de", dp, dp, curvature)
        second_coefficient = torch.zeros_like(joint)
        if exact_derivative:
            second_coefficient[support] = 1.0 + torch.log(joint[support] / marginal[:, None].expand_as(joint)[support])
        else:
            second_coefficient[support] = joint[support] / marginal[:, None].expand_as(joint)[support]
        second = torch.einsum("ijde,ij->de", d2p, second_coefficient)
        return first + second

    def __call__(self, current: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, TROHistogramAudit]:
        if current.shape != reference.shape:
            raise ValueError("Current and reference images must have equal shape.")
        weight_current, weight_reference = self._weights(current), self._weights(reference)
        pixels = current.numel()
        marginal_current = weight_current.sum(dim=1) / pixels
        marginal_reference = weight_reference.sum(dim=1) / pixels
        joint = (weight_current @ weight_reference.T) / pixels
        support = joint > 0.0
        ratio = joint[support] / (marginal_current[:, None] * marginal_reference[None, :])[support]
        mi = (joint[support] * ratio.log()).sum()
        audit = TROHistogramAudit(float(marginal_current.sum()), float(marginal_reference.sum()), float(joint.sum()), int(support.sum()))
        return mi, audit
