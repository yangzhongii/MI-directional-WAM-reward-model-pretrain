"""Pipeline-v7 G0: verify frozen physical geometry and direct progress baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mi_reward.features.geometric_features import geometric_progress_score


def _candidates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in row["candidates"] if not candidate["excluded_reasons"]}


def _eef_delta(candidate: dict[str, Any], center: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(np.asarray(candidate["physical_after"]["eef_pos"]) - np.asarray(center["physical_after"]["eef_pos"]), dtype=torch.float64)


def _matrix(candidates: dict[str, dict[str, Any]], label: str, center: dict[str, Any]) -> torch.Tensor:
    result = torch.stack([(_eef_delta(candidates[f"{label}_plus_{axis}"], center) - _eef_delta(candidates[f"{label}_minus_{axis}"], center)) / 2 for axis in range(3)], dim=1)
    if torch.linalg.matrix_rank(result) < 3:
        raise RuntimeError(f"{label} actual EEF matrix is rank deficient")
    return result


def _metric(left: list[float], right: list[float]) -> dict[str, float | None]:
    x, y = np.asarray(left), np.asarray(right)
    residual, variance = float(np.sum((x - y) ** 2)), float(np.sum((y - y.mean()) ** 2))
    ranks_x, ranks_y = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return {"r2": None if variance <= 1e-12 else float(1 - residual / variance), "spearman": None if not np.std(ranks_x) or not np.std(ranks_y) else float(np.corrcoef(ranks_x, ranks_y)[0, 1]), "sign_accuracy": None if not np.any(np.abs(y) > 1e-9) else float(np.mean(np.sign(x[np.abs(y) > 1e-9]) == np.sign(y[np.abs(y) > 1e-9]))) }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 20:
        raise RuntimeError("G0 requires the frozen 20-anchor collection")
    heldout_score: list[float] = []
    heldout_progress: list[float] = []
    per_anchor: list[dict[str, Any]] = []
    primary_cosines: list[float] = []
    cross_epsilon: list[float] = []
    for row in rows:
        candidates = _candidates(row)
        required = ["center", *[f"{scale}_{sign}_{axis}" for scale in ("primary", "secondary") for sign in ("plus", "minus") for axis in range(3)]]
        missing = [name for name in required if name not in candidates]
        if missing:
            raise RuntimeError(f"{row['anchor_id']} is missing {missing}")
        center = candidates["center"]
        center_score = geometric_progress_score(center["physical_after"])

        def gradient(label: str) -> torch.Tensor:
            eef_matrix = _matrix(candidates, label, center)
            score_delta = torch.tensor([(geometric_progress_score(candidates[f"{label}_plus_{axis}"]["physical_after"]) - geometric_progress_score(candidates[f"{label}_minus_{axis}"]["physical_after"])) / 2 for axis in range(3)], dtype=torch.float64)
            return torch.linalg.pinv(eef_matrix.T) @ score_delta

        primary, secondary = gradient("primary"), gradient("secondary")
        relative = np.asarray(center["physical_after"]["eef_pos"], dtype=np.float64) - np.asarray(center["physical_after"]["task_object_pos"], dtype=np.float64)
        physical = torch.tensor(-relative / np.linalg.norm(relative), dtype=torch.float64)
        primary_cosines.append(float(F.cosine_similarity(primary[None], physical[None]).item()))
        cross_epsilon.append(float(F.cosine_similarity(primary[None], secondary[None]).item()))
        scores, progress = [], []
        for name, candidate in candidates.items():
            if name.startswith("heldout_"):
                delta = geometric_progress_score(candidate["physical_after"]) - center_score
                scores.append(delta)
                progress.append(-(float(candidate["physical_after"]["eef_object_distance"]) - float(center["physical_after"]["eef_object_distance"])))
        heldout_score.extend(scores); heldout_progress.extend(progress)
        per_anchor.append({"anchor_id": row["anchor_id"], "source_demo": row["source_demo"], "privileged_input": True, "deployable": False, "heldout_count": len(scores), "heldout_direct_geometry_vs_physical_progress": _metric(scores, progress), "analytic_fd_cosine": float(F.cosine_similarity(primary[None], physical[None]).item()), "cross_epsilon_cosine": float(F.cosine_similarity(primary[None], secondary[None]).item()), "physical_cosine": float(F.cosine_similarity(primary[None], physical[None]).item())})
        print(f"ANCHOR {row['anchor_id']} cross_epsilon={cross_epsilon[-1]:.6f}", flush=True)
    heldout = _metric(heldout_score, heldout_progress)
    passed = bool(heldout["r2"] is not None and heldout["r2"] >= .999999 and min(primary_cosines) >= .999999 and float(np.median(cross_epsilon)) >= .90)
    report = {"protocol": "v7_g0_privileged_geometric_sanity_v1", "branch": "G0_direct_privileged_geometry", "representation_source": "simulator_recorded_eef_object_goal_positions", "privileged_input": True, "deployable": False, "actual_or_predicted": "actual", "heldout_action_count": len(heldout_score), "heldout_direct_geometry_vs_physical_progress": heldout, "analytic_fd_cosine_median": float(np.median(primary_cosines)), "cross_epsilon_cosine_median": float(np.median(cross_epsilon)), "physical_cosine_median": float(np.median(primary_cosines)), "per_anchor": per_anchor, "status": "PASS" if passed else "NO_GO", "next_gate": "G1 requires recorded candidate-level geometric correspondences; point centers alone must not be passed to Dame MI as pseudo-tokens."}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "heldout": heldout, "cross_epsilon_cosine_median": report["cross_epsilon_cosine_median"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
