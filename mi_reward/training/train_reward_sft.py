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

    from tqdm import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        global_step = 0
        pbar = tqdm(range(epochs), desc="SFT training", unit="epoch")
        for epoch in pbar:
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
            mean_loss = total_loss / max(total_count, 1)
            log_f.write(json.dumps({"epoch": epoch, "mean_loss": mean_loss}) + "\n")
            pbar.set_postfix({"loss": f"{mean_loss:.4f}"})

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


def train_reward_distill(
    preferences: str | Path,
    feature_root: str | Path,
    output_dir: str | Path,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 1e-4,
    hidden_dim: int = 256,
    architecture: str = "gru",
    lambda_rank: float = 1.0,
    lambda_potential: float = 1.0,
    lambda_direction: float = 0.5,
    gamma: float = 0.99,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, object]:
    """Train StatePotentialRewardModel via directional reward distillation.

    The teacher (MIPotentialField) provides per-frame MI potentials Phi_t.
    The student (StatePotentialRewardModel) predicts per-frame V(o_t, g).
    Distillation loss = rank_loss + potential_loss + direction_loss.
    Deployment reward = gamma * V(o_{t+1}) - V(o_t).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")

    from mi_reward.data.preference_dataset import PreferenceFeatureDataset
    from mi_reward.models.state_potential_model import StatePotentialRewardModel
    from mi_reward.training.collator import PreferenceCollator
    from mi_reward.training.losses import DistillationLoss
    from mi_reward.scoring.mi_potential_field import MIPotentialField, MIBackend

    dataset = PreferenceFeatureDataset(preferences, feature_root)
    if len(dataset) == 0:
        raise ValueError("No preference pairs found.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=PreferenceCollator())

    first = dataset[0]["chosen_features"].reshape(dataset[0]["chosen_features"].shape[0], -1)
    input_dim = first.shape[-1]

    model = StatePotentialRewardModel(
        input_dim=input_dim, hidden_dim=hidden_dim,
        architecture=architecture,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DistillationLoss(
        lambda_rank=lambda_rank,
        lambda_potential=lambda_potential,
        lambda_direction=lambda_direction,
    )

    # Teacher: MI potential field
    mi_field = MIPotentialField(backend=MIBackend.DAME_BSPLINE, gamma=gamma)
    mi_field._mi.to(device_obj)

    from tqdm import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_f:
        global_step = 0
        pbar = tqdm(range(epochs), desc="Distill training", unit="epoch")
        for epoch in pbar:
            total_loss = 0.0
            total_count = 0
            for batch in loader:
                chosen = batch["chosen_features"].to(device_obj)
                rejected = batch["rejected_features"].to(device_obj)
                chosen_mask = batch["chosen_mask"].to(device_obj)
                rejected_mask = batch["rejected_mask"].to(device_obj)

                # Student predictions: per-frame potentials
                student_chosen = model(chosen)  # [B, T]
                student_rejected = model(rejected)

                # Teacher: compute MI potentials
                with torch.no_grad():
                    B = chosen.shape[0]
                    phi_chosen = torch.zeros_like(student_chosen)
                    phi_rejected = torch.zeros_like(student_rejected)
                    for b in range(B):
                        c = chosen[b][chosen_mask[b]]  # [T_valid, D]
                        r = rejected[b][rejected_mask[b]]
                        if c.shape[0] >= 2:
                            phi_chosen[b, :c.shape[0]] = mi_field.potential_batch(c, c[-1])
                        if r.shape[0] >= 2:
                            phi_rejected[b, :r.shape[0]] = mi_field.potential_batch(r, r[-1])

                # Trajectory-level rewards (aggregated potential difference)
                chosen_rewards = model.compute_trajectory_score(chosen, gamma=gamma)
                rejected_rewards = model.compute_trajectory_score(rejected, gamma=gamma)

                # Combined distillation loss (ensure all tensors on same device)
                cuda = chosen.device
                result = loss_fn(
                    student_potentials_chosen=student_chosen,
                    student_potentials_rejected=student_rejected,
                    teacher_phi_chosen=phi_chosen,
                    teacher_phi_rejected=phi_rejected,
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                    chosen_confidence=torch.ones(B, device=cuda),
                    rejected_confidence=torch.ones(B, device=cuda),
                    score_margin=(chosen_rewards - rejected_rewards).to(cuda),
                    mask_chosen=chosen_mask,
                    mask_rejected=rejected_mask,
                )
                loss = result["total"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * B
                total_count += B
                log_f.write(json.dumps({
                    "step": global_step, "epoch": epoch,
                    "loss": float(loss.item()),
                    "rank": float(result["rank"].item()) if torch.is_tensor(result["rank"]) else float(result["rank"]),
                    "potential": float(result["potential"].item()) if torch.is_tensor(result["potential"]) else float(result["potential"]),
                    "direction": float(result["direction"].item()) if torch.is_tensor(result["direction"]) else float(result["direction"]),
                }) + "\n")
                global_step += 1
            mean_loss = total_loss / max(total_count, 1)
            log_f.write(json.dumps({
                "epoch": epoch,
                "mean_loss": mean_loss,
            }) + "\n")
            pbar.set_postfix({"loss": f"{mean_loss:.4f}"})

    config = {
        "preferences": str(preferences), "feature_root": str(feature_root),
        "batch_size": batch_size, "epochs": epochs, "lr": lr,
        "hidden_dim": hidden_dim, "architecture": architecture,
        "input_dim": input_dim, "gamma": gamma, "seed": seed,
        "lambda_rank": lambda_rank, "lambda_potential": lambda_potential,
        "lambda_direction": lambda_direction,
        "model_class": "StatePotentialRewardModel",
    }
    _write_yaml_like(out / "train_config.yaml", config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out / "pytorch_model.pt")
    return config


def train_geoprogress(
    preferences: str | Path,
    feature_root: str | Path,
    output_dir: str | Path,
    manifest: str | Path,
    success_refs: str | Path,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 1e-4,
    hidden_dim: int = 256,
    architecture: str = "gru",
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    lambda_rank: float = 1.0,
    lambda_potential: float = 1.0,
    lambda_direction: float = 0.5,
    relation_weight: float = 1.0,
    mi_backend: str = "gaussian_mi_proxy",
    gamma: float = 0.99,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, object]:
    """Train the verified, goal-conditioned GeoProgress reward potential.

    The goal comes only from a declared successful reference.  Candidate RGB
    is encoded by LaWAM before this function is called; measured relations
    enter separately and unverified Cosmos candidates are rejected by the
    dataset contract.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    from mi_reward.data.geoprogress_dataset import GeoProgressPreferenceDataset
    from mi_reward.models.geoprogress_potential import GeoProgressPotential
    from mi_reward.relations.sequence import relation_progress_potential
    from mi_reward.scoring.mi_potential_field import MIBackend, MIPotentialField
    from mi_reward.training.geoprogress_collator import GeoProgressCollator
    from mi_reward.training.losses import DistillationLoss

    try:
        backend = MIBackend(mi_backend)
    except ValueError as exc:
        choices = ", ".join(item.value for item in MIBackend)
        raise ValueError(f"Unsupported MI backend {mi_backend!r}; choose one of {choices}.") from exc
    dataset = GeoProgressPreferenceDataset(preferences, manifest, success_refs, feature_root)
    if not len(dataset):
        raise ValueError("No GeoProgress preference pairs found.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=GeoProgressCollator())
    first = dataset[0]
    visual_dim = int(first["chosen_tokens"].shape[-1])  # type: ignore[index]
    relation_dim = int(first["chosen_relations"].shape[-1])  # type: ignore[index]
    relation_names = list(first["relation_names"])  # type: ignore[arg-type]
    model = GeoProgressPotential(
        visual_dim=visual_dim,
        relation_dim=relation_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DistillationLoss(
        lambda_rank=lambda_rank,
        lambda_potential=lambda_potential,
        lambda_direction=lambda_direction,
    )
    teacher = MIPotentialField(backend=backend, gamma=gamma).to(device_obj)

    def teacher_potential(
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        goals: torch.Tensor,
        goal_mask: torch.Tensor,
        relations: torch.Tensor,
        names: list[str],
    ) -> torch.Tensor:
        phi = torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device)
        for batch_index in range(tokens.shape[0]):
            length = int(token_mask[batch_index].sum().item())
            if not length:
                continue
            state = tokens[batch_index, :length]
            goal = goals[batch_index, goal_mask[batch_index]]
            visual_phi = teacher.potential_batch(state, goal)
            relation_phi = relation_progress_potential(relations[batch_index, :length], names).to(tokens.device)
            phi[batch_index, :length] = visual_phi + relation_weight * relation_phi
        return phi

    from tqdm import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        global_step = 0
        for epoch in tqdm(range(epochs), desc="GeoProgress training", unit="epoch"):
            total_loss = 0.0
            total_count = 0
            for batch in loader:
                chosen = batch["chosen_tokens"].to(device_obj)  # type: ignore[index]
                rejected = batch["rejected_tokens"].to(device_obj)  # type: ignore[index]
                goals = batch["goal_tokens"].to(device_obj)  # type: ignore[index]
                chosen_relations = batch["chosen_relations"].to(device_obj)  # type: ignore[index]
                rejected_relations = batch["rejected_relations"].to(device_obj)  # type: ignore[index]
                chosen_mask = batch["chosen_mask"].to(device_obj)  # type: ignore[index]
                rejected_mask = batch["rejected_mask"].to(device_obj)  # type: ignore[index]
                goal_mask = batch["goal_mask"].to(device_obj)  # type: ignore[index]
                names = list(batch["relation_names"])  # type: ignore[arg-type]

                student_chosen = model(chosen, goals, chosen_relations, chosen_mask, goal_mask)
                student_rejected = model(rejected, goals, rejected_relations, rejected_mask, goal_mask)
                with torch.no_grad():
                    teacher_chosen = teacher_potential(chosen, chosen_mask, goals, goal_mask, chosen_relations, names)
                    teacher_rejected = teacher_potential(rejected, rejected_mask, goals, goal_mask, rejected_relations, names)
                chosen_rewards = model.compute_trajectory_score(
                    chosen, goals, chosen_relations, chosen_mask, goal_mask, gamma
                )
                rejected_rewards = model.compute_trajectory_score(
                    rejected, goals, rejected_relations, rejected_mask, goal_mask, gamma
                )
                result = loss_fn(
                    student_potentials_chosen=student_chosen,
                    student_potentials_rejected=student_rejected,
                    teacher_phi_chosen=teacher_chosen,
                    teacher_phi_rejected=teacher_rejected,
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                    chosen_confidence=batch["chosen_confidence"].to(device_obj),  # type: ignore[index]
                    rejected_confidence=batch["rejected_confidence"].to(device_obj),  # type: ignore[index]
                    score_margin=batch["score_margin"].to(device_obj),  # type: ignore[index]
                    mask_chosen=chosen_mask,
                    mask_rejected=rejected_mask,
                )
                loss = result["total"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = chosen.shape[0]
                total_loss += float(loss.item()) * count
                total_count += count
                log_f.write(json.dumps({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "rank": float(result["rank"].item()),
                    "potential": float(result["potential"].item()),
                    "direction": float(result["direction"].item()),
                }) + "\n")
                global_step += 1
            log_f.write(json.dumps({"epoch": epoch, "mean_loss": total_loss / max(total_count, 1)}) + "\n")

    config = {
        "model_class": "GeoProgressPotential",
        "preferences": str(preferences),
        "manifest": str(manifest),
        "success_refs": str(success_refs),
        "feature_root": str(feature_root),
        "visual_dim": visual_dim,
        "relation_dim": relation_dim,
        "relation_names": relation_names,
        "hidden_dim": hidden_dim,
        "architecture": architecture,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "mi_backend": backend.value,
        "relation_weight": relation_weight,
        "gamma": gamma,
        "seed": seed,
    }
    _write_yaml_like(out / "train_config.yaml", config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out / "pytorch_model.pt")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train legacy rewards or the verified goal-conditioned GeoProgress potential.")
    parser.add_argument("--mode", default="ranking", choices=["ranking", "distill", "geoprogress"],
                        help="ranking: legacy head; distill: legacy self-goal MI; geoprogress: verified explicit-goal training.")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--architecture", default="gru", choices=["mlp", "gru", "transformer"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--lambda_potential", type=float, default=1.0)
    parser.add_argument("--lambda_direction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest", default=None, help="Trajectory manifest required by --mode geoprogress.")
    parser.add_argument("--success_refs", default=None, help="Success references required by --mode geoprogress.")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--relation_weight", type=float, default=1.0)
    parser.add_argument("--mi_backend", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi", "dame_bspline"])
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != "mode"}
    if args.mode == "geoprogress":
        if not args.manifest or not args.success_refs:
            parser.error("--manifest and --success_refs are required for --mode geoprogress")
        train_geoprogress(**kwargs)
    elif args.mode == "distill":
        for key in ("manifest", "success_refs", "num_layers", "num_heads", "dropout", "relation_weight", "mi_backend"):
            kwargs.pop(key)
        train_reward_distill(**kwargs)
    else:
        for key in ("manifest", "success_refs", "num_layers", "num_heads", "dropout", "relation_weight", "mi_backend"):
            kwargs.pop(key)
        train_reward_sft(**kwargs)


if __name__ == "__main__":
    main()
