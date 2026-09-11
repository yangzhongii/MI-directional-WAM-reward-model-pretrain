from __future__ import annotations

import torch

from mi_reward.control.mi_action_field import (
    action_gradient_hessian_from_function,
    construct_local_action_field,
    estimate_jacobian,
    fit_fixed_normalization,
    mi_and_gradient,
    normalize_tokens,
    pullback_gradient,
    pullback_hessian,
)
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI


def test_fixed_normalization_is_immutable_and_reports_clamping() -> None:
    reference = torch.tensor([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=torch.float64)
    normalization = fit_fixed_normalization(reference, [reference], bin_range=(-10.0, 10.0))
    before_shift, before_scale = normalization.shift.clone(), normalization.scale.clone()
    normalized, saturated = normalize_tokens(torch.tensor([[100.0, -100.0], [0.2, 0.3]], dtype=torch.float64), normalization)
    assert saturated[0].all()
    assert not saturated[1].any()
    assert torch.all(normalized <= 10.0) and torch.all(normalized >= -10.0)
    estimator = DameSoftHistogramMI(normalization="none", channel_mode="channelwise").to(dtype=torch.float64)
    _, _, ratio = mi_and_gradient(reference, reference, normalization, estimator=estimator)
    assert ratio == 0.0
    assert torch.equal(normalization.shift, before_shift)
    assert torch.equal(normalization.scale, before_scale)


def test_dame_mi_gradient_and_hessian_pull_back_through_linear_action_map() -> None:
    torch.manual_seed(13)
    dtype = torch.float64
    tokens, channels, actions = 8, 3, 3
    reference = torch.randn(tokens, channels, dtype=dtype) * 0.15
    base = torch.randn(tokens * channels, dtype=dtype) * 0.04
    jacobian = torch.randn(tokens * channels, actions, dtype=dtype) * 0.03
    action = torch.tensor([0.04, -0.03, 0.02], dtype=dtype)
    anchor = (base + jacobian @ action).reshape(tokens, channels)
    normalization = fit_fixed_normalization(
        reference, [anchor, anchor + 0.01, anchor - 0.01], bin_range=(-20.0, 20.0)
    )
    estimator = DameSoftHistogramMI(num_bins=8, spline_order=3, normalization="none", channel_mode="channelwise").to(dtype=dtype)

    def objective(value: torch.Tensor) -> torch.Tensor:
        candidate, _ = normalize_tokens((base + jacobian @ value).reshape(tokens, channels), normalization)
        ref, _ = normalize_tokens(reference, normalization)
        return estimator(candidate, ref)

    field, saturation = construct_local_action_field(
        anchor,
        reference,
        torch.stack([(anchor.reshape(-1) + jacobian[:, i] * 1e-3).reshape(tokens, channels) for i in range(actions)]),
        torch.stack([(anchor.reshape(-1) - jacobian[:, i] * 1e-3).reshape(tokens, channels) for i in range(actions)]),
        [1e-3] * actions,
        normalization,
        estimator=estimator,
        include_token_hessian=True,
    )
    direct_gradient, direct_hessian = action_gradient_hessian_from_function(objective, action)
    assert saturation == 0.0
    assert torch.allclose(field.gradient_a, direct_gradient, atol=1e-10, rtol=1e-8)
    assert field.hessian_pullback_a is not None
    assert torch.allclose(field.hessian_pullback_a, direct_hessian, atol=1e-10, rtol=1e-8)


def test_jacobian_and_pullback_validate_shapes_and_values() -> None:
    plus = torch.tensor([[[2.0], [4.0]], [[3.0], [5.0]]])
    minus = torch.tensor([[[0.0], [2.0]], [[1.0], [1.0]]])
    jacobian = estimate_jacobian(plus, minus, [1.0, 2.0])
    assert torch.equal(jacobian, torch.tensor([[1.0, 0.5], [1.0, 1.0]]))
    gradient_z = torch.tensor([2.0, -1.0])
    hessian_z = torch.diag(torch.tensor([3.0, 5.0]))
    assert torch.equal(pullback_gradient(jacobian, gradient_z), torch.tensor([1.0, 0.0]))
    assert torch.allclose(pullback_hessian(jacobian, hessian_z), torch.tensor([[8.0, 6.5], [6.5, 5.75]]))
