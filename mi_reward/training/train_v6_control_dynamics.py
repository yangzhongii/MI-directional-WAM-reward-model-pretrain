"""Pipeline-v6 Test 1: fit action-conditioned dynamics on frozen LaWAM tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor
from mi_reward.models.action_conditioned_latent import ActionConditionedControlModel


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("mi_reward/configs/lawam_control_latent.yaml"))
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cache_name(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest() + ".pt"


def _read_rows(collection_dir: Path, allowed_demos: set[int]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    samples: list[dict[str, Any]] = []
    for row in rows:
        demo = int(str(row["source_demo"]).removeprefix("demo_"))
        if demo not in allowed_demos:
            continue
        candidates = [candidate for candidate in row["candidates"] if not candidate["excluded_reasons"]]
        center = next((candidate for candidate in candidates if candidate["candidate_id"] == "center"), None)
        if center is None:
            continue
        for candidate in candidates:
            if candidate["candidate_id"] == "center":
                continue
            action = np.asarray(candidate["physical_after"]["eef_pos"], dtype=np.float32) - np.asarray(center["physical_after"]["eef_pos"], dtype=np.float32)
            samples.append({
                "anchor_id": row["anchor_id"],
                "demo": demo,
                "language": row["language"],
                "current": center["images"]["wrist"],
                "next": candidate["images"]["wrist"],
                "action": action.tolist(),
                "kind": candidate["kind"],
            })
    return samples


def _cache_tokens(samples: list[dict[str, Any]], collection_dir: Path, cache_dir: Path, args: argparse.Namespace) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested: dict[str, str] = {}
    for sample in samples:
        requested.setdefault(sample["current"], sample["language"])
        requested.setdefault(sample["next"], sample["language"])
    missing = [(path, language) for path, language in requested.items() if not (cache_dir / _cache_name(path)).is_file()]
    if not missing:
        print(f"FEATURE_CACHE reused={len(requested)}", flush=True)
        return
    extractor = LaWAMLAMFeatureExtractor(args.lam_config, args.lam_checkpoint, args.dino_model, args.device, strict=True)
    for index, (relative_path, language) in enumerate(missing, start=1):
        tokens = extractor.extract_frame_tokens(str(collection_dir / relative_path), language).to(torch.float16)
        torch.save(tokens, cache_dir / _cache_name(relative_path))
        if index % 20 == 0 or index == len(missing):
            print(f"FEATURE_CACHE {index}/{len(missing)}", flush=True)


class _Pairs(Dataset):
    def __init__(self, samples: list[dict[str, Any]], cache_dir: Path):
        self.samples = samples
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.samples[index]
        current = torch.load(self.cache_dir / _cache_name(row["current"]), map_location="cpu", weights_only=True).float()
        target = torch.load(self.cache_dir / _cache_name(row["next"]), map_location="cpu", weights_only=True).float()
        return current, target, torch.tensor(row["action"], dtype=torch.float32)


def _prediction_metrics(model: ActionConditionedControlModel, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    predictions, targets, delta_predictions, delta_targets = [], [], [], []
    zero_error = []
    with torch.no_grad():
        for current, target, action in loader:
            current, target, action = current.to(device), target.to(device), action.to(device)
            current_control = model.project(current)
            predicted = model.dynamics(current_control, action)
            target_control = model.project(target)
            predictions.append(predicted.cpu())
            targets.append(target_control.cpu())
            delta_predictions.append((predicted - current_control).flatten(1).cpu())
            delta_targets.append((target_control - current_control).flatten(1).cpu())
            zero_error.append((model.dynamics(current_control, torch.zeros_like(action)) - current_control).abs().max().cpu())
    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    residual = float((predicted - target).square().sum())
    variance = float((target - target.mean()).square().sum())
    actual_delta = torch.cat(delta_targets)
    predicted_delta = torch.cat(delta_predictions)
    cosine = F.cosine_similarity(predicted_delta, actual_delta, dim=1)
    return {
        "count": int(len(predicted)),
        "next_latent_r2": float(1.0 - residual / max(variance, 1e-12)),
        "delta_cosine_mean": float(cosine.mean()),
        "delta_cosine_median": float(cosine.median()),
        "zero_action_max_abs_error": float(torch.stack(zero_error).max()),
    }


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text())
    split = config["split"]
    train_demos = {int(value) for value in split["train_demos"]}
    validation_demos = {int(value) for value in split["validation_demos"]}
    frozen_test_demos = {int(value) for value in split["frozen_test_demos"]}
    if train_demos & validation_demos or (train_demos | validation_demos) & frozen_test_demos:
        raise ValueError("Train/validation demos must be disjoint from the frozen test demos.")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_rows = _read_rows(args.collection_dir, train_demos)
    validation_rows = _read_rows(args.collection_dir, validation_demos)
    if not train_rows or not validation_rows:
        raise RuntimeError(f"Need valid train and validation samples, got {len(train_rows)} and {len(validation_rows)}.")
    _cache_tokens(train_rows + validation_rows, args.collection_dir, args.feature_cache_dir, args)
    train_loader = DataLoader(_Pairs(train_rows, args.feature_cache_dir), batch_size=int(config["training"]["batch_size"]), shuffle=True)
    validation_loader = DataLoader(_Pairs(validation_rows, args.feature_cache_dir), batch_size=int(config["training"]["batch_size"]), shuffle=False)
    first_current, _, _ = next(iter(train_loader))
    model = ActionConditionedControlModel(
        input_dim=int(first_current.shape[-1]),
        control_dim=int(config["representation"]["control_dim"]),
        action_dim=int(config["dynamics"]["action_dim"]),
        hidden_dim=int(config["dynamics"]["hidden_dim"]),
        action_scale=float(config["dynamics"]["action_scale_m"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    weights = config["training"]["loss"]
    history: list[dict[str, float | int]] = []
    best_r2 = float("-inf")
    args.output_dir.mkdir(parents=True)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for current, target, action in train_loader:
            current, target, action = current.to(device), target.to(device), action.to(device)
            current_control = model.project(current)
            target_control = model.project(target)
            prediction = model.dynamics(current_control, action)
            next_loss = F.l1_loss(prediction, target_control)
            delta_loss = F.l1_loss(prediction - current_control, target_control - current_control)
            loss = float(weights["next_latent_l1"]) * next_loss + float(weights["delta_latent_l1"]) * delta_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(current)
            count += len(current)
        metrics = _prediction_metrics(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), **metrics}
        history.append(record)
        if float(record["next_latent_r2"]) > best_r2:
            best_r2 = float(record["next_latent_r2"])
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, args.output_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record), flush=True)
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    report = {
        "protocol": "v6_control_latent_test1_v1",
        "collection": str(args.collection_dir),
        "feature_cache": str(args.feature_cache_dir),
        "config": config,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation": _prediction_metrics(model, validation_loader, device),
        "history": history,
        "status": "TEST1_ONLY_NO_MI_OR_PHYSICAL_DIRECTION",
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["best_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
