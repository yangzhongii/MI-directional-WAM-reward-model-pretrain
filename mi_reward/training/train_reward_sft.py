from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mi_reward.data.preference_dataset import PreferenceFeatureDataset
from mi_reward.models.reward_head import TrajectoryRewardHead
from mi_reward.training.collator import PreferenceCollator
from mi_reward.training.loss import pairwise_ranking_loss


def _write_yaml_like(path: Path, config: dict[str, object]) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_reward_sft(
    preferences: str | Path,
    feature_root: str | Path,
    output_dir: str | Path,
    batch_size: int,
    epochs: int,
    lr: float,
    hidden_dim: int = 256,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = PreferenceFeatureDataset(preferences, feature_root)
    if len(dataset) == 0:
        raise ValueError("No preference pairs found.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=PreferenceCollator())
    first = dataset[0]["chosen_features"].reshape(dataset[0]["chosen_features"].shape[0], -1)  # type: ignore[index]
    model = TrajectoryRewardHead(input_dim=first.shape[-1], hidden_dim=hidden_dim).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        global_step = 0
        for epoch in range(epochs):
            total_loss = 0.0
            total_count = 0
            for batch in loader:
                chosen = batch["chosen_features"].to(device_obj)
                rejected = batch["rejected_features"].to(device_obj)
                chosen_mask = batch["chosen_mask"].to(device_obj)
                rejected_mask = batch["rejected_mask"].to(device_obj)
                chosen_rewards = model(chosen, chosen_mask)
                rejected_rewards = model(rejected, rejected_mask)
                loss = pairwise_ranking_loss(chosen_rewards, rejected_rewards)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * chosen.shape[0]
                total_count += chosen.shape[0]
                log_f.write(json.dumps({"step": global_step, "epoch": epoch, "loss": float(loss.item())}) + "\n")
                global_step += 1
            log_f.write(json.dumps({"epoch": epoch, "mean_loss": total_loss / max(total_count, 1)}) + "\n")

    config = {
        "preferences": str(preferences),
        "feature_root": str(feature_root),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "input_dim": model.input_dim,
        "seed": seed,
    }
    _write_yaml_like(out / "train_config.yaml", config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out / "pytorch_model.pt")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MVP trajectory reward head with pairwise ranking SFT.")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_reward_sft(**vars(args))


if __name__ == "__main__":
    main()
