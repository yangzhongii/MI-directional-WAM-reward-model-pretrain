"""Pipeline-v5 Test 0: MI gradient/Jacobian/Hessian identity check.

The test uses the repository's differentiable Dame B-spline MI estimator and
a known linear latent dynamics map z(a) = z_base + B a.  It verifies that the
action-space gradient and Hessian obtained by differentiating the composite
MI objective agree with their pullbacks through B, and checks both derivatives
against central finite differences.  No robot policy or reward model is
trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/mi_reward/v5_mi_action_geometry/math_identity_v1"),
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--finite-difference-epsilon", type=float, default=3e-3)
    parser.add_argument("--trust-radius", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.1)
    return parser.parse_args()


def maximum_absolute_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(right).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(left - right) / denominator).item())


def central_gradient(function, point: torch.Tensor, epsilon: float) -> torch.Tensor:
    values = []
    for index in range(point.numel()):
        direction = torch.zeros_like(point)
        direction[index] = epsilon
        values.append((function(point + direction) - function(point - direction)) / (2.0 * epsilon))
    return torch.stack(values)


def central_hessian(function, point: torch.Tensor, epsilon: float) -> torch.Tensor:
    size = point.numel()
    result = torch.empty(size, size, dtype=point.dtype)
    center = function(point)
    for left in range(size):
        ei = torch.zeros_like(point)
        ei[left] = epsilon
        result[left, left] = (function(point + ei) - 2.0 * center + function(point - ei)) / epsilon**2
        for right in range(left + 1, size):
            ej = torch.zeros_like(point)
            ej[right] = epsilon
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * epsilon**2)
            result[left, right] = value
            result[right, left] = value
    return result


def bounded_step(direction: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction).clamp_min(1e-12)
    return direction * min(1.0, float(radius) / float(norm.item()))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    dtype = torch.float64
    token_count = int(args.tokens)
    channel_count = int(args.channels)
    action_dim = int(args.action_dim)
    latent_dim = token_count * channel_count

    goal = torch.randn(token_count, channel_count, dtype=dtype).tanh() * 1.5
    jacobian = torch.randn(latent_dim, action_dim, dtype=dtype) * 0.12
    action = torch.linspace(0.3, 0.15, action_dim, dtype=dtype)
    if action_dim > 1:
        action[1] = -0.2
    base = goal.reshape(-1) + torch.randn(latent_dim, dtype=dtype) * 0.08 - jacobian @ action

    estimator = DameSoftHistogramMI(
        num_bins=8,
        spline_order=3,
        normalization="none",
        channel_mode="channelwise",
    ).to(dtype=dtype)

    def latent_objective(value: torch.Tensor) -> torch.Tensor:
        return estimator(value.reshape(token_count, channel_count), goal)

    def action_objective(value: torch.Tensor) -> torch.Tensor:
        return latent_objective(base + jacobian @ value)

    latent = (base + jacobian @ action).detach().requires_grad_(True)
    objective = latent_objective(latent)
    latent_gradient = torch.autograd.grad(objective, latent, create_graph=True)[0]
    action_gradient_pullback = jacobian.T @ latent_gradient
    action_gradient_direct = torch.autograd.functional.jacobian(action_objective, action)

    latent_hessian = torch.autograd.functional.hessian(latent_objective, latent)
    action_hessian_pullback = jacobian.T @ latent_hessian @ jacobian
    action_hessian_direct = torch.autograd.functional.hessian(action_objective, action)

    epsilon = float(args.finite_difference_epsilon)
    action_gradient_finite_difference = central_gradient(action_objective, action, epsilon)
    action_hessian_finite_difference = central_hessian(action_objective, action, epsilon)

    gradient_step = bounded_step(action_gradient_direct, float(args.trust_radius))
    curvature = -0.5 * (action_hessian_direct + action_hessian_direct.T)
    damped_curvature = curvature + float(args.damping) * torch.eye(action_dim, dtype=dtype)
    newton_direction = torch.linalg.solve(damped_curvature, action_gradient_direct)
    newton_step = bounded_step(newton_direction, float(args.trust_radius))
    objective_before = float(action_objective(action).item())
    objective_after_gradient = float(action_objective(action + gradient_step).item())
    objective_after_newton = float(action_objective(action + newton_step).item())

    gradient_pullback_error = maximum_absolute_error(action_gradient_pullback, action_gradient_direct)
    hessian_pullback_error = maximum_absolute_error(action_hessian_pullback, action_hessian_direct)
    gradient_fd_error = relative_error(action_gradient_finite_difference, action_gradient_direct)
    hessian_fd_error = relative_error(action_hessian_finite_difference, action_hessian_direct)
    symmetry_error = maximum_absolute_error(action_hessian_direct, action_hessian_direct.T)

    checks = {
        "gradient_pullback_max_abs_below_1e-10": gradient_pullback_error < 1e-10,
        "hessian_pullback_max_abs_below_1e-9": hessian_pullback_error < 1e-9,
        "gradient_finite_difference_relative_below_1e-4": gradient_fd_error < 1e-4,
        "hessian_finite_difference_relative_below_1e-3": hessian_fd_error < 1e-3,
        "hessian_symmetry_max_abs_below_1e-10": symmetry_error < 1e-10,
        "gradient_step_increases_mi": objective_after_gradient > objective_before,
        "damped_newton_step_increases_mi": objective_after_newton > objective_before,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Numerical identity and implementation check under known linear latent dynamics; not a LIBERO control result.",
        "configuration": {
            "seed": args.seed,
            "token_count": token_count,
            "channel_count": channel_count,
            "action_dim": action_dim,
            "finite_difference_epsilon": epsilon,
            "trust_radius": args.trust_radius,
            "damping": args.damping,
            "mi": "DameSoftHistogramMI(num_bins=8, spline_order=3, normalization=none, channelwise)",
        },
        "errors": {
            "gradient_pullback_max_absolute": gradient_pullback_error,
            "hessian_pullback_max_absolute": hessian_pullback_error,
            "gradient_finite_difference_relative": gradient_fd_error,
            "hessian_finite_difference_relative": hessian_fd_error,
            "hessian_symmetry_max_absolute": symmetry_error,
        },
        "objective": {
            "before": objective_before,
            "after_gradient_step": objective_after_gradient,
            "after_damped_newton_step": objective_after_newton,
            "gradient_gain": objective_after_gradient - objective_before,
            "damped_newton_gain": objective_after_newton - objective_before,
        },
        "action_gradient_direct": action_gradient_direct.detach().tolist(),
        "action_gradient_pullback": action_gradient_pullback.detach().tolist(),
        "action_hessian_direct": action_hessian_direct.detach().tolist(),
        "action_hessian_pullback": action_hessian_pullback.detach().tolist(),
        "checks": checks,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# Pipeline-v5 MI action geometry: mathematical identity check",
        "",
        f"Status: **{report['status']}**",
        "",
        "This checks the repository's differentiable Dame MI under a known linear latent dynamics map. It does not claim that the resulting MI field is useful in LIBERO.",
        "",
        f"- Gradient pullback max absolute error: `{gradient_pullback_error:.3e}`",
        f"- Hessian pullback max absolute error: `{hessian_pullback_error:.3e}`",
        f"- Gradient finite-difference relative error: `{gradient_fd_error:.3e}`",
        f"- Hessian finite-difference relative error: `{hessian_fd_error:.3e}`",
        f"- MI before step: `{objective_before:.9f}`",
        f"- MI after gradient step: `{objective_after_gradient:.9f}`",
        f"- MI after damped Newton step: `{objective_after_newton:.9f}`",
        "",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Pipeline-v5 mathematical identity check failed")


if __name__ == "__main__":
    main()
