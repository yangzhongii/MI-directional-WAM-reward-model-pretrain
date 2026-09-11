"""E1-A4: decompose the non-zero MI gradient at identical images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.libero.eval_e1_tro_local_optimization import gaussian_5x5, render
from mi_reward.control.tro_image_mi import TROImageMI


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left[None], right[None]).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_default_dtype(torch.float64)
    estimator = TROImageMI()
    goal = torch.zeros(6)
    scenes = []
    for scene_id in range(3):
        filtered_render = lambda pose: gaussian_5x5(render(pose, args.size, scene_id))
        reference = filtered_render(goal).detach()
        image_jacobian = torch.autograd.functional.jacobian(filtered_render, goal)

        current = reference.clone().requires_grad_(True)
        mi = estimator(current, reference)[0]
        pixel_gradient = torch.autograd.grad(mi, current)[0].detach()
        contribution = pixel_gradient[:, :, None] * image_jacobian
        full_pose_gradient = contribution.sum(dim=(0, 1))
        direct_pose_gradient = torch.autograd.functional.jacobian(
            lambda pose: estimator(filtered_render(pose), reference)[0], goal
        )
        border = []
        for margin in (0, 2, 4, 8, 12):
            selected = contribution if margin == 0 else contribution[margin:-margin, margin:-margin]
            gradient = selected.sum(dim=(0, 1))
            border.append({
                "interior_margin_pixels": margin,
                "pose_gradient_norm": float(torch.linalg.norm(gradient)),
                "cosine_to_full": cosine(gradient, full_pose_gradient),
            })

        phases = []
        bin_width = 255.0 / estimator.bins
        for phase in (-0.50, -0.25, 0.0, 0.25, 0.50):
            phased = (reference + phase * bin_width).clamp(0.0, 255.0).requires_grad_(True)
            phased_mi = estimator(phased, phased.detach())[0]
            phased_pixel_gradient = torch.autograd.grad(phased_mi, phased)[0].detach()
            phased_pose_gradient = torch.einsum("hw,hwd->d", phased_pixel_gradient, image_jacobian)
            phases.append({
                "shared_intensity_phase_bins": phase,
                "pixel_gradient_norm": float(torch.linalg.norm(phased_pixel_gradient)),
                "projected_pose_gradient_norm": float(torch.linalg.norm(phased_pose_gradient)),
                "projected_pose_gradient": phased_pose_gradient.tolist(),
            })

        snapped = (reference * estimator.bins / 255.0).round() * bin_width
        snapped_current = snapped.clone().requires_grad_(True)
        snapped_mi = estimator(snapped_current, snapped)[0]
        snapped_pixel_gradient = torch.autograd.grad(snapped_mi, snapped_current)[0].detach()
        snapped_pose_gradient = torch.einsum("hw,hwd->d", snapped_pixel_gradient, image_jacobian)
        scenes.append({
            "scene_id": scene_id,
            "identity_mi": float(mi),
            "identity_pixel_gradient_norm": float(torch.linalg.norm(pixel_gradient)),
            "identity_pixel_gradient_abs_mean": float(pixel_gradient.abs().mean()),
            "pose_gradient_direct": direct_pose_gradient.tolist(),
            "pose_gradient_from_pixels": full_pose_gradient.tolist(),
            "pose_chain_cosine": cosine(direct_pose_gradient, full_pose_gradient),
            "pose_chain_relative_error": float(torch.linalg.norm(direct_pose_gradient-full_pose_gradient) / torch.linalg.norm(direct_pose_gradient).clamp_min(1e-12)),
            "border_decomposition": border,
            "phase_sensitivity": phases,
            "snapped_to_bin_centers": {
                "pixel_gradient_norm": float(torch.linalg.norm(snapped_pixel_gradient)),
                "projected_pose_gradient_norm": float(torch.linalg.norm(snapped_pose_gradient)),
            },
        })
    report = {
        "protocol": "e1_a4_identity_derivative_decomposition_v1",
        "scope": "diagnostic_only_no_gate_tuning",
        "fixed_main_estimator": {"bins": 8, "padding": 2, "prefilter": "5x5 Gaussian sigma 1.0"},
        "scenes": scenes,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    compact = []
    for scene in scenes:
        compact.append({
            "scene_id": scene["scene_id"],
            "pixel_gradient_norm": scene["identity_pixel_gradient_norm"],
            "pose_gradient_norm": float(torch.linalg.norm(torch.tensor(scene["pose_gradient_direct"]))),
            "pose_chain_cosine": scene["pose_chain_cosine"],
            "border_pose_gradient_norms": {str(x["interior_margin_pixels"]): x["pose_gradient_norm"] for x in scene["border_decomposition"]},
            "phase_pose_gradient_norms": {str(x["shared_intensity_phase_bins"]): x["projected_pose_gradient_norm"] for x in scene["phase_sensitivity"]},
            "snapped_pose_gradient_norm": scene["snapped_to_bin_centers"]["projected_pose_gradient_norm"],
        })
    print(json.dumps({"protocol": report["protocol"], "scenes": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
