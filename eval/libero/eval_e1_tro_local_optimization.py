"""E1-A3: preregistered local free-camera MI optimization pilot.

This experiment does not involve LIBERO robot actuation.  It compares the
scalar-exact Hessian with the literal TRO-2011 equation (17) on the same fixed
synthetic scenes and starts.  Branches are named before results are observed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from mi_reward.control.tro_image_mi import TROImageMI


def gaussian_5x5(image: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Fixed 5x5 Gaussian prefilter required by the E1 protocol."""
    coordinate = torch.arange(-2, 3, dtype=image.dtype, device=image.device)
    kernel_1d = torch.exp(-0.5 * coordinate.square() / (sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    padded = functional.pad(image[None, None], (2, 2, 2, 2), mode="reflect")
    return functional.conv2d(padded, kernel)[0, 0]


def render(pose: torch.Tensor, size: int, scene_id: int) -> torch.Tensor:
    """Differentiable pinhole view of one of three fixed textured planes."""
    dtype, device = pose.dtype, pose.device
    grid = torch.linspace(-0.42, 0.42, size, dtype=dtype, device=device)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    rx, ry, rz = pose[3:]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    rxm = torch.stack((
        torch.stack((torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx))),
        torch.stack((torch.zeros_like(cx), cx, -sx)),
        torch.stack((torch.zeros_like(cx), sx, cx)),
    ))
    rym = torch.stack((
        torch.stack((cy, torch.zeros_like(cy), sy)),
        torch.stack((torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy))),
        torch.stack((-sy, torch.zeros_like(cy), cy)),
    ))
    rzm = torch.stack((
        torch.stack((cz, -sz, torch.zeros_like(cz))),
        torch.stack((sz, cz, torch.zeros_like(cz))),
        torch.stack((torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz))),
    ))
    rays = (rzm @ rym @ rxm) @ torch.stack((x.reshape(-1), y.reshape(-1), torch.ones_like(x).reshape(-1)))
    scale = (1.0 - pose[2]) / rays[2]
    world = pose[:2, None] + rays[:2] * scale
    u, v = world[0], world[1]
    if scene_id == 0:
        image = 127.5 + 45 * torch.sin(19*u + 7*v) + 35 * torch.cos(13*v - 3*u) + 25 * torch.sin(31*u*v)
    elif scene_id == 1:
        image = 126.0 + 42 * torch.cos(23*u - 9*v) + 31 * torch.sin(17*v + 5*u) + 22 * torch.cos(37*u*v + 0.4)
    elif scene_id == 2:
        image = 128.0 + 38 * torch.sin(27*u + 11*v + 0.7) + 34 * torch.cos(21*v - 8*u) + 24 * torch.sin(15*(u*u-v*v))
    else:
        raise ValueError(f"Unsupported scene_id={scene_id}")
    return image.reshape(size, size).clamp(0.0, 255.0)


def clipped_step(step: torch.Tensor, max_translation: float, max_rotation: float) -> torch.Tensor:
    out = step.clone()
    for part, limit in ((slice(0, 3), max_translation), (slice(3, 6), max_rotation)):
        norm = torch.linalg.norm(out[part])
        if norm > limit:
            out[part] *= limit / norm
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--gain", type=float, default=0.35)
    parser.add_argument("--damping", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    torch.set_default_dtype(torch.float64)
    estimator = TROImageMI()
    translation_start = 0.010
    rotation_start = math.radians(1.5)
    max_translation_step = 0.002
    max_rotation_step = math.radians(0.30)
    starts = []
    for axis in range(6):
        pose = torch.zeros(6)
        pose[axis] = (1.0 if axis % 2 == 0 else -1.0) * (translation_start if axis < 3 else rotation_start)
        starts.append(pose)

    trials = []
    scene_audits = []
    identity = torch.eye(6)
    for scene_id in range(3):
        goal = torch.zeros(6)
        reference = gaussian_5x5(render(goal, args.size, scene_id)).detach()

        def scalar(x: torch.Tensor) -> torch.Tensor:
            return estimator(gaussian_5x5(render(x, args.size, scene_id)), reference)[0]

        goal_value = scalar(goal)
        goal_gradient = torch.autograd.functional.jacobian(scalar, goal)
        exact_hstar = torch.autograd.functional.hessian(scalar, goal)
        filtered_render = lambda p: gaussian_5x5(render(p, args.size, scene_id))
        image_jacobian = torch.autograd.functional.jacobian(filtered_render, goal)
        image_hessian = torch.func.jacfwd(torch.func.jacfwd(filtered_render))(goal)
        literal_hstar = estimator.printed_hessian(filtered_render(goal), reference, image_jacobian, image_hessian)
        scene_audits.append({
            "scene_id": scene_id,
            "mi_at_goal": float(goal_value),
            "gradient_norm_at_goal": float(torch.linalg.norm(goal_gradient)),
            "exact_hstar_eigenvalues": torch.linalg.eigvalsh(exact_hstar).tolist(),
            "literal_hstar_eigenvalues": torch.linalg.eigvalsh(literal_hstar).tolist(),
            "literal_exact_relative_error": float(torch.linalg.norm(literal_hstar-exact_hstar) / torch.linalg.norm(exact_hstar).clamp_min(1e-12)),
        })

        methods = {
            "scalar_exact_hstar": exact_hstar,
            "tro_2011_literal_hstar": literal_hstar,
        }
        for start_id, initial in enumerate(starts):
            for method, hstar in methods.items():
                pose = initial.clone()
                history = []
                for step_id in range(args.steps):
                    pose_var = pose.detach().requires_grad_(True)
                    mi = scalar(pose_var)
                    gradient = torch.autograd.grad(mi, pose_var)[0].detach()
                    # Maximization Newton step: -H^{-1} gradient.  Damping is
                    # fixed and identical for both preregistered branches.
                    update = -args.gain * torch.linalg.solve(hstar - args.damping * identity, gradient)
                    update = clipped_step(update, max_translation_step, max_rotation_step)
                    pose = (pose + update).detach()
                    history.append({
                        "step": step_id,
                        "mi": float(mi),
                        "translation_error_m": float(torch.linalg.norm(pose[:3])),
                        "rotation_error_deg": math.degrees(float(torch.linalg.norm(pose[3:]))),
                        "update": update.tolist(),
                    })
                trans_error = float(torch.linalg.norm(pose[:3]))
                rot_error = math.degrees(float(torch.linalg.norm(pose[3:])))
                trials.append({
                    "scene_id": scene_id,
                    "start_id": start_id,
                    "perturbed_axis": start_id,
                    "method": method,
                    "initial_pose": initial.tolist(),
                    "final_pose": pose.tolist(),
                    "final_translation_error_m": trans_error,
                    "final_rotation_error_deg": rot_error,
                    "success": trans_error <= 0.002 and rot_error <= 0.5,
                    "mi_monotone_fraction": sum(history[i+1]["mi"] >= history[i]["mi"] for i in range(len(history)-1)) / max(1, len(history)-1),
                    "history": history,
                })

    summary = {}
    for method in ("scalar_exact_hstar", "tro_2011_literal_hstar"):
        selected = [trial for trial in trials if trial["method"] == method]
        summary[method] = {
            "trials": len(selected),
            "success_rate": sum(t["success"] for t in selected) / len(selected),
            "median_final_translation_error_m": float(torch.tensor([t["final_translation_error_m"] for t in selected]).median()),
            "median_final_rotation_error_deg": float(torch.tensor([t["final_rotation_error_deg"] for t in selected]).median()),
            "mean_mi_monotone_fraction": sum(t["mi_monotone_fraction"] for t in selected) / len(selected),
        }
    report = {
        "protocol": "e1_a3_free_camera_local_optimization_v2",
        "scope": "synthetic_free_camera_no_robot_actuation",
        "branch_selection": "preregistered_from_source_and_derivative_audit_not_selected_by_closed_loop",
        "settings": {
            "scenes": 3,
            "starts_per_scene": 6,
            "steps": args.steps,
            "gain": args.gain,
            "damping": args.damping,
            "translation_start_m": translation_start,
            "rotation_start_deg": math.degrees(rotation_start),
            "max_translation_step_m": max_translation_step,
            "max_rotation_step_deg": math.degrees(max_rotation_step),
            "prefilter": "fixed_5x5_gaussian_reflect_padding_sigma_1.0",
            "success_translation_m": 0.002,
            "success_rotation_deg": 0.5,
        },
        "scene_audits": scene_audits,
        "summary": summary,
        "trials": trials,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"protocol": report["protocol"], "scene_audits": scene_audits, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
