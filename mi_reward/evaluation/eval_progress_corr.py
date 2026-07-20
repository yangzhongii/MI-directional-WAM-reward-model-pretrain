from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.evaluation.eval_reward_ranking import eval_reward_ranking
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.models.reward_head import TrajectoryRewardHead
from mi_reward.scoring.trajectory_score import score_trajectory


def _corrcoef(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x @ y / denom).item())


def _success_separation(scores: list[float], labels: list[bool]) -> dict[str, float]:
    pos = [score for score, label in zip(scores, labels) if label]
    neg = [score for score, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return {
            "success_mean_score": sum(pos) / max(len(pos), 1) if pos else 0.0,
            "failure_mean_score": sum(neg) / max(len(neg), 1) if neg else 0.0,
            "success_failure_margin": 0.0,
            "success_failure_auc": 0.0,
        }
    correct = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                correct += 1.0
            elif p == n:
                ties += 1.0
    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)
    return {
        "success_mean_score": pos_mean,
        "failure_mean_score": neg_mean,
        "success_failure_margin": pos_mean - neg_mean,
        "success_failure_auc": (correct + 0.5 * ties) / (len(pos) * len(neg)),
    }


def _load_reward_model(ckpt: str | Path, device: torch.device) -> TrajectoryRewardHead:
    checkpoint = torch.load(ckpt, map_location=device)
    config = checkpoint.get("config", {})
    model = TrajectoryRewardHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _score_with_reward_head(model: TrajectoryRewardHead, features: torch.Tensor, device: torch.device) -> float:
    features = features.reshape(features.shape[0], -1).float().unsqueeze(0).to(device)
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        return float(model(features, mask).item())


def _group_refs(refs: list[SuccessReference]) -> dict[str, list[SuccessReference]]:
    grouped: dict[str, list[SuccessReference]] = {}
    for ref in refs:
        grouped.setdefault(ref.task, []).append(ref)
    return grouped


def _score_with_mi(
    example: TrajectoryExample,
    store: CachedFeatureStore,
    refs_by_task: dict[str, list[SuccessReference]],
    gamma: float,
    mi_mode: str,
) -> tuple[float, dict[str, Any]]:
    refs = refs_by_task.get(example.task, [])
    if not refs:
        return 0.0, {"reason": "missing_success_reference"}
    candidate = store.load(example.traj_id)
    scored = [
        score_trajectory(candidate, store.load(ref.ref_id), gamma=gamma, mi_mode=mi_mode)
        for ref in refs
    ]
    best = max(scored, key=lambda item: float(item["score_delta"]))
    return float(best["score_delta"]), best


def eval_progress_corr(
    manifest: str | Path,
    feature_root: str | Path,
    *,
    success_refs: str | Path | None = None,
    ckpt: str | Path | None = None,
    preferences: str | Path | None = None,
    output: str | Path | None = None,
    gamma: float = 0.99,
    mi_mode: str = "gaussian_mi_proxy",
    device: str = "cuda",
) -> dict[str, Any]:
    examples = read_jsonl(manifest, TrajectoryExample)
    store = CachedFeatureStore(feature_root)
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_reward_model(ckpt, device_obj) if ckpt else None
    refs_by_task = _group_refs(read_jsonl(success_refs, SuccessReference)) if success_refs else {}

    records = []
    for example in examples:
        metadata = example.metadata or {}
        if model is not None:
            score = _score_with_reward_head(model, store.load(example.traj_id), device_obj)
            score_detail: dict[str, Any] = {"score_type": "learned_reward"}
        else:
            score, detail = _score_with_mi(example, store, refs_by_task, gamma=gamma, mi_mode=mi_mode)
            score_detail = {"score_type": "mi_delta", **detail}
        progress_value = metadata.get("progress")
        if progress_value is None:
            progress_value = metadata.get("steps")
        records.append(
            {
                "traj_id": example.traj_id,
                "task": example.task,
                "score": float(score),
                "success": metadata.get("success"),
                "progress": progress_value,
                "metadata": metadata,
                "score_detail": score_detail,
            }
        )

    scores = [float(record["score"]) for record in records]
    success_records = [record for record in records if isinstance(record.get("success"), bool)]
    labels = [bool(record["success"]) for record in success_records]
    label_scores = [float(record["score"]) for record in success_records]
    progress_records = [
        record
        for record in records
        if record.get("progress") is not None and isinstance(record.get("progress"), (int, float))
    ]
    report: dict[str, Any] = {
        "num_trajectories": len(records),
        "num_scored": len(scores),
        "score_mean": sum(scores) / max(len(scores), 1) if scores else 0.0,
        "success_labeled_count": len(success_records),
        "success_count": sum(int(label) for label in labels),
        "failure_count": sum(int(not label) for label in labels),
        "progress_labeled_count": len(progress_records),
        "records": records,
    }
    if success_records:
        report.update(_success_separation(label_scores, labels))
    if progress_records:
        report["progress_correlation"] = _corrcoef(
            [float(record["progress"]) for record in progress_records],
            [float(record["score"]) for record in progress_records],
        )
    else:
        report["progress_correlation"] = 0.0
    if preferences and ckpt:
        report["reward_ranking"] = eval_reward_ranking(
            preferences=preferences,
            feature_root=feature_root,
            ckpt=ckpt,
            device=device,
        )
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MI/reward progress correlation on LIBERO or RoboTwin manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--success_refs", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--preferences", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mi_mode", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    eval_progress_corr(**vars(args))


if __name__ == "__main__":
    main()
