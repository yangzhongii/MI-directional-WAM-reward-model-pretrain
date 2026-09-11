"""Pipeline-v6 Test 1b: train spatial action-conditioned normalized token deltas."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from mi_reward.models.action_conditioned_latent import ActionConditionedControlModel
from mi_reward.training.train_v6_control_dynamics import _Pairs, _cache_tokens, _read_rows


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("mi_reward/configs/lawam_control_latent_delta.yaml"))
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _normalized_delta(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.flatten(1), dim=1, eps=1e-8)


def _metrics(model: ActionConditionedControlModel, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    predicted_next, target_next, predicted_delta, target_delta, zero_errors = [], [], [], [], []
    with torch.no_grad():
        for current, nxt, action in loader:
            current, nxt, action = current.to(device), nxt.to(device), action.to(device)
            control = model.project(current)
            target = model.project(nxt)
            delta = model.dynamics.predict_delta(control, action)
            predicted_next.append((control + delta).cpu())
            target_next.append(target.cpu())
            predicted_delta.append(delta.flatten(1).cpu())
            target_delta.append((target - control).flatten(1).cpu())
            zero_errors.append(model.dynamics.predict_delta(control, torch.zeros_like(action)).abs().max().cpu())
    prediction = torch.cat(predicted_next)
    target = torch.cat(target_next)
    residual = float((prediction - target).square().sum())
    variance = float((target - target.mean()).square().sum())
    pred_delta = torch.cat(predicted_delta)
    true_delta = torch.cat(target_delta)
    cosine = F.cosine_similarity(pred_delta, true_delta, dim=1)
    return {
        "count": int(len(prediction)),
        "next_latent_r2": float(1.0 - residual / max(variance, 1e-12)),
        "delta_cosine_mean": float(cosine.mean()),
        "delta_cosine_median": float(cosine.median()),
        "zero_action_max_abs_error": float(torch.stack(zero_errors).max()),
    }


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    cfg = yaml.safe_load(args.config.read_text())
    split = cfg["split"]
    train_demos = {int(value) for value in split["train_demos"]}
    validation_demos = {int(value) for value in split["validation_demos"]}
    frozen_test_demos = {int(value) for value in split["frozen_test_demos"]}
    if train_demos & validation_demos or (train_demos | validation_demos) & frozen_test_demos:
        raise ValueError("Train/validation demos must be disjoint from frozen test demos.")
    seed = int(cfg["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_rows = _read_rows(args.collection_dir, train_demos)
    validation_rows = _read_rows(args.collection_dir, validation_demos)
    if not train_rows or not validation_rows:
        raise RuntimeError("Need valid train and validation transitions.")
    _cache_tokens(train_rows + validation_rows, args.collection_dir, args.feature_cache_dir, args)
    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(_Pairs(train_rows, args.feature_cache_dir), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(_Pairs(validation_rows, args.feature_cache_dir), batch_size=batch_size, shuffle=False)
    current, _, _ = next(iter(train_loader))
    model = ActionConditionedControlModel(
        input_dim=int(current.shape[-1]),
        control_dim=int(cfg["representation"]["control_dim"]),
        action_dim=int(cfg["dynamics"]["action_dim"]),
        hidden_dim=int(cfg["dynamics"]["hidden_dim"]),
        action_scale=float(cfg["dynamics"]["action_scale_m"]),
        spatial_heads=int(cfg["dynamics"]["spatial_heads"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    weights = cfg["training"]["loss"]
    r2_floor = float(cfg["checkpoint_selection"]["minimum_validation_next_latent_r2"])
    args.output_dir.mkdir(parents=True)
    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_epoch = None
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total, count = 0.0, 0
        for current, nxt, action in train_loader:
            current, nxt, action = current.to(device), nxt.to(device), action.to(device)
            control = model.project(current)
            with torch.no_grad():
                target_delta = model.project(nxt) - control
                target_next = model.project(nxt)
            prediction_delta = model.dynamics.predict_delta(control, action)
            normalized_prediction = _normalized_delta(prediction_delta)
            normalized_target = _normalized_delta(target_delta)
            cosine_loss = 1.0 - (normalized_prediction * normalized_target).sum(dim=1).mean()
            delta_l1 = F.smooth_l1_loss(normalized_prediction, normalized_target)
            reconstruction = F.l1_loss(control + prediction_delta, target_next)
            loss = (
                float(weights["normalized_delta_cosine"]) * cosine_loss
                + float(weights["normalized_delta_smooth_l1"]) * delta_l1
                + float(weights["next_latent_reconstruction"]) * reconstruction
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(current)
            count += len(current)
        metrics = _metrics(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), **metrics}
        history.append(record)
        if float(record["next_latent_r2"]) >= r2_floor and float(record["delta_cosine_median"]) > best_score:
            best_score = float(record["delta_cosine_median"])
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "validation": record}, args.output_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record), flush=True)
    if best_epoch is None:
        raise RuntimeError(f"No validation checkpoint met next-latent R2 >= {r2_floor}.")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    report = {
        "protocol": "v6_control_delta_test1b_v1",
        "collection": str(args.collection_dir),
        "feature_cache": str(args.feature_cache_dir),
        "config": cfg,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "checkpoint_selection": {"minimum_validation_next_latent_r2": r2_floor, "maximize": "validation_delta_cosine_median"},
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation": _metrics(model, validation_loader, device),
        "history": history,
        "status": "TEST1B_ONLY_NO_MI_OR_PHYSICAL_DIRECTION",
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["best_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
