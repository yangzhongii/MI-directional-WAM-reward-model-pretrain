from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mi_reward.data.preference_dataset import PreferenceFeatureDataset
from mi_reward.models.reward_head import TrajectoryRewardHead
from mi_reward.training.collator import PreferenceCollator


def _corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x @ y / denom).item())


def eval_reward_ranking(
    preferences: str | Path,
    feature_root: str | Path,
    ckpt: str | Path,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, float]:
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt, map_location=device_obj)
    config = checkpoint.get("config", {})
    model = TrajectoryRewardHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = PreferenceFeatureDataset(preferences, feature_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=PreferenceCollator())
    margins = []
    mi_margins = []
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            chosen_reward = model(batch["chosen_features"].to(device_obj), batch["chosen_mask"].to(device_obj)).cpu()
            rejected_reward = model(batch["rejected_features"].to(device_obj), batch["rejected_mask"].to(device_obj)).cpu()
            margin = chosen_reward - rejected_reward
            margins.append(margin)
            mi_margins.append(batch["chosen_score"] - batch["rejected_score"])
            correct += int((margin > 0).sum().item())
            count += margin.numel()
    reward_margins = torch.cat(margins) if margins else torch.zeros(0)
    score_margins = torch.cat(mi_margins) if mi_margins else torch.zeros(0)
    report = {
        "pairwise_accuracy": correct / max(count, 1),
        "mean_reward_margin": float(reward_margins.mean().item()) if reward_margins.numel() else 0.0,
        "score_correlation": _corrcoef(score_margins, reward_margins),
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pairwise ranking accuracy for an MI reward head.")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    eval_reward_ranking(**vars(args))


if __name__ == "__main__":
    main()
