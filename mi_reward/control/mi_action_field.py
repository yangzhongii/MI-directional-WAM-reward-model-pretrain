"""Local Pipeline-v5 pullback from visual MI to action space.

The visual encoder and simulator are deliberately outside autograd.  Given
frozen patch tokens, this module differentiates the Dame MI scalar with
respect to tokens and maps that derivative to an action coordinate using a
central-finite-difference visual Jacobian.  It does not train a model or
score trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI


@dataclass(frozen=True)
class FixedNormalization:
    """Frozen per-channel token statistics for a single pre-registered run."""

    shift: torch.Tensor
    scale: torch.Tensor
    bin_range: tuple[float, float]
    fit_anchor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.shift.ndim != 1 or self.scale.ndim != 1 or self.shift.shape != self.scale.shape:
            raise ValueError("shift and scale must be matching rank-one tensors.")
        if not torch.isfinite(self.shift).all() or not torch.isfinite(self.scale).all():
            raise ValueError("Fixed normalization statistics must be finite.")
        if torch.any(self.scale <= 0):
            raise ValueError("Fixed normalization scale must be strictly positive.")
        if not self.bin_range[0] < self.bin_range[1]:
            raise ValueError("bin_range must have lower < upper.")


@dataclass(frozen=True)
class LocalActionField:
    """Token-space MI derivatives and their local action-space pullbacks."""

    mi_at_anchor: torch.Tensor
    gradient_z: torch.Tensor
    jacobian_a: torch.Tensor
    gradient_a: torch.Tensor
    hessian_z: torch.Tensor | None
    hessian_pullback_a: torch.Tensor | None


@dataclass(frozen=True)
class TrustRegionStep:
    """Damped second-order and normalized first-order local proposals."""

    gradient_step: torch.Tensor
    newton_step: torch.Tensor | None
    curvature: torch.Tensor
    used_newton: bool
    fallback_reason: str | None


def _as_token_matrix(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(tokens) or tokens.ndim != 2:
        shape = None if not torch.is_tensor(tokens) else tuple(tokens.shape)
        raise ValueError(f"{name} must be a [K,D] token tensor, got {shape}.")
    if tokens.shape[0] < 2 or tokens.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two tokens and one channel.")
    if not torch.is_floating_point(tokens):
        raise ValueError(f"{name} must have floating-point dtype.")
    return tokens


def fit_fixed_normalization(
    reference_tokens: torch.Tensor,
    anchor_neighborhood_tokens: Sequence[torch.Tensor],
    *,
    bin_range: tuple[float, float] = (-3.0, 3.0),
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
    eps: float = 1e-8,
    fit_anchor_ids: Sequence[str] = (),
) -> FixedNormalization:
    """Fit immutable robust channel statistics before evaluating candidates.

    Candidate and held-out action tokens are intentionally not accepted here.
    """

    reference_tokens = _as_token_matrix(reference_tokens, "reference_tokens")
    if not anchor_neighborhood_tokens:
        raise ValueError("anchor_neighborhood_tokens cannot be empty.")
    values = [reference_tokens]
    for index, tokens in enumerate(anchor_neighborhood_tokens):
        tokens = _as_token_matrix(tokens, f"anchor_neighborhood_tokens[{index}]")
        if tokens.shape[1] != reference_tokens.shape[1]:
            raise ValueError("All normalization tokens must have the same channel dimension.")
        values.append(tokens.to(device=reference_tokens.device, dtype=reference_tokens.dtype))
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 1.")
    combined = torch.cat(values, dim=0)
    shift = torch.quantile(combined, low_quantile, dim=0)
    high = torch.quantile(combined, high_quantile, dim=0)
    scale = (high - shift).clamp_min(float(eps))
    return FixedNormalization(
        shift=shift.detach().clone(),
        scale=scale.detach().clone(),
        bin_range=(float(bin_range[0]), float(bin_range[1])),
        fit_anchor_ids=tuple(str(value) for value in fit_anchor_ids),
    )


def normalize_tokens(tokens: torch.Tensor, normalization: FixedNormalization) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize tokens with frozen statistics and return an elementwise clamp mask."""

    tokens = _as_token_matrix(tokens, "tokens")
    if tokens.shape[1] != normalization.shift.numel():
        raise ValueError("Token channel dimension does not match fixed normalization statistics.")
    shift = normalization.shift.to(device=tokens.device, dtype=tokens.dtype)
    scale = normalization.scale.to(device=tokens.device, dtype=tokens.dtype)
    unbounded = (tokens - shift) / scale
    lower, upper = normalization.bin_range
    saturation = (unbounded < lower) | (unbounded > upper)
    return torch.clamp(unbounded, min=lower, max=upper), saturation


def mi_and_gradient(
    tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    normalization: FixedNormalization,
    *,
    estimator: DameSoftHistogramMI | None = None,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return Dame MI, its token gradient, and candidate clamp saturation ratio."""

    tokens = _as_token_matrix(tokens, "tokens")
    reference_tokens = _as_token_matrix(reference_tokens, "reference_tokens")
    if tokens.shape != reference_tokens.shape:
        raise ValueError("tokens and reference_tokens must have matching [K,D] shape.")
    candidate = tokens.detach().clone().requires_grad_(True)
    candidate_norm, saturation = normalize_tokens(candidate, normalization)
    reference_norm, _ = normalize_tokens(reference_tokens.detach(), normalization)
    if estimator is None:
        estimator = DameSoftHistogramMI(
            num_bins=8,
            spline_order=3,
            normalization="none",
            channel_mode="channelwise",
        )
    if estimator.normalization != "none":
        raise ValueError("v5 requires an externally frozen normalization and estimator normalization='none'.")
    estimator = estimator.to(device=candidate.device, dtype=candidate.dtype)
    mi = estimator(candidate_norm, reference_norm)
    gradient = torch.autograd.grad(mi, candidate, create_graph=create_graph)[0]
    return mi, gradient, float(saturation.float().mean().item())


def estimate_jacobian(
    plus_tokens: torch.Tensor,
    minus_tokens: torch.Tensor,
    epsilons: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Estimate ``dz/da`` from aligned central finite-difference token probes."""

    if plus_tokens.ndim != 3 or minus_tokens.ndim != 3 or plus_tokens.shape != minus_tokens.shape:
        raise ValueError("plus_tokens and minus_tokens must be matching [A,K,D] tensors.")
    action_dim, token_count, channel_count = plus_tokens.shape
    if action_dim < 1 or token_count < 2 or channel_count < 1:
        raise ValueError("Token probes must have non-empty [A,K,D] dimensions.")
    epsilon = torch.as_tensor(epsilons, device=plus_tokens.device, dtype=plus_tokens.dtype).reshape(-1)
    if epsilon.numel() != action_dim or torch.any(epsilon <= 0):
        raise ValueError("epsilons must contain one positive value per action dimension.")
    differences = (plus_tokens - minus_tokens) / (2.0 * epsilon[:, None, None])
    return differences.reshape(action_dim, token_count * channel_count).T


def pullback_gradient(jacobian_a: torch.Tensor, gradient_z: torch.Tensor) -> torch.Tensor:
    """Compute ``g_a = J_a^T g_z``."""

    if jacobian_a.ndim != 2:
        raise ValueError("jacobian_a must be [Z,A].")
    gradient_z = gradient_z.reshape(-1)
    if jacobian_a.shape[0] != gradient_z.numel():
        raise ValueError("jacobian_a and gradient_z dimensions do not match.")
    return jacobian_a.T @ gradient_z


def pullback_hessian(jacobian_a: torch.Tensor, hessian_z: torch.Tensor) -> torch.Tensor:
    """Compute the linear-dynamics curvature pullback ``J_a^T H_z J_a``."""

    if hessian_z.ndim != 2 or hessian_z.shape[0] != hessian_z.shape[1]:
        raise ValueError("hessian_z must be square.")
    if jacobian_a.shape[0] != hessian_z.shape[0]:
        raise ValueError("jacobian_a and hessian_z dimensions do not match.")
    return jacobian_a.T @ hessian_z @ jacobian_a


def action_gradient_hessian_from_function(
    objective: Callable[[torch.Tensor], torch.Tensor],
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate a local action objective for Test 0 and FD cross-checks."""

    if action.ndim != 1 or action.numel() < 1:
        raise ValueError("action must be a non-empty rank-one tensor.")
    gradient = torch.autograd.functional.jacobian(objective, action)
    hessian = torch.autograd.functional.hessian(objective, action)
    return gradient, hessian


def construct_local_action_field(
    anchor_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    plus_tokens: torch.Tensor,
    minus_tokens: torch.Tensor,
    epsilons: torch.Tensor | Sequence[float],
    normalization: FixedNormalization,
    *,
    estimator: DameSoftHistogramMI | None = None,
    include_token_hessian: bool = False,
) -> tuple[LocalActionField, float]:
    """Build a v5 local field from pre-collected simulator token probes."""

    anchor_tokens = _as_token_matrix(anchor_tokens, "anchor_tokens")
    reference_tokens = _as_token_matrix(reference_tokens, "reference_tokens")
    mi, gradient_z_matrix, saturation = mi_and_gradient(
        anchor_tokens, reference_tokens, normalization, estimator=estimator, create_graph=include_token_hessian
    )
    jacobian_a = estimate_jacobian(plus_tokens, minus_tokens, epsilons)
    gradient_z = gradient_z_matrix.reshape(-1)
    hessian_z: torch.Tensor | None = None
    hessian_a: torch.Tensor | None = None
    if include_token_hessian:
        def token_objective(flat_tokens: torch.Tensor) -> torch.Tensor:
            candidate = flat_tokens.reshape_as(anchor_tokens)
            candidate_norm, _ = normalize_tokens(candidate, normalization)
            reference_norm, _ = normalize_tokens(reference_tokens, normalization)
            local_estimator = estimator or DameSoftHistogramMI(
                num_bins=8, spline_order=3, normalization="none", channel_mode="channelwise"
            )
            return local_estimator.to(device=candidate.device, dtype=candidate.dtype)(candidate_norm, reference_norm)

        hessian_z = torch.autograd.functional.hessian(token_objective, anchor_tokens.reshape(-1))
        hessian_a = pullback_hessian(jacobian_a, hessian_z)
    return LocalActionField(
        mi_at_anchor=mi.detach(),
        gradient_z=gradient_z.detach(),
        jacobian_a=jacobian_a.detach(),
        gradient_a=pullback_gradient(jacobian_a, gradient_z).detach(),
        hessian_z=None if hessian_z is None else hessian_z.detach(),
        hessian_pullback_a=None if hessian_a is None else hessian_a.detach(),
    ), saturation


def _bounded_step(direction: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-12:
        return torch.zeros_like(direction)
    return direction * min(1.0, float(radius) / float(norm.item()))


def damped_trust_region_step(
    gradient_a: torch.Tensor,
    hessian_a: torch.Tensor,
    *,
    damping: float,
    radius: float,
) -> TrustRegionStep:
    """Form a maximization proposal using damped curvature with safe fallback."""

    gradient_a = gradient_a.reshape(-1)
    if hessian_a.shape != (gradient_a.numel(), gradient_a.numel()):
        raise ValueError("hessian_a must be square with the action dimension.")
    if damping <= 0 or radius <= 0:
        raise ValueError("damping and radius must be positive.")
    gradient_step = _bounded_step(gradient_a, radius)
    curvature = -0.5 * (hessian_a + hessian_a.T)
    matrix = curvature + float(damping) * torch.eye(
        gradient_a.numel(), dtype=gradient_a.dtype, device=gradient_a.device
    )
    try:
        condition = torch.linalg.cond(matrix)
        if not torch.isfinite(condition) or float(condition.item()) > 1e8:
            return TrustRegionStep(gradient_step, None, curvature, False, "ill_conditioned_curvature")
        direction = torch.linalg.solve(matrix, gradient_a)
        if not torch.isfinite(direction).all() or float(torch.dot(gradient_a, direction).item()) <= 0:
            return TrustRegionStep(gradient_step, None, curvature, False, "non_ascent_newton_direction")
        return TrustRegionStep(gradient_step, _bounded_step(direction, radius), curvature, True, None)
    except RuntimeError:
        return TrustRegionStep(gradient_step, None, curvature, False, "singular_curvature")
