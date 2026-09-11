from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mi_reward.data.preference_dataset import PreferenceFeatureDataset
from mi_reward.models.reward_head import TrajectoryRewardHead
from mi_reward.training.collator import PreferenceCollator
from mi_reward.training.loss import pairwise_ranking_loss


TEACHER_TARGET_CACHE_VERSION = 2


def _trajectory_score_from_potentials(
    potentials: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute the model trajectory score without a duplicate forward pass."""

    if potentials.shape[1] < 2:
        return potentials[:, 0]
    deltas = gamma * potentials[:, 1:] - potentials[:, :-1]
    valid = mask[:, 1:] & mask[:, :-1]
    return (deltas * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _teacher_cache_signature(
    *,
    preferences: str | Path,
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
    mi_backend: str,
    gamma: float,
    relation_weight: float,
    action_weight: float,
    kinematic_weight: float,
    teacher_targets_source: str | Path | None = None,
) -> str:
    def file_stamp(value: str | Path) -> tuple[str, int, int]:
        path = Path(value).resolve()
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns

    feature_files = list(Path(feature_root).resolve().glob("*.pt"))
    feature_stats = [path.stat() for path in feature_files]
    payload = {
        "version": TEACHER_TARGET_CACHE_VERSION,
        "preferences": file_stamp(preferences),
        "manifest": file_stamp(manifest),
        "success_refs": file_stamp(success_refs),
        "features": {
            "root": str(Path(feature_root).resolve()),
            "files": len(feature_stats),
            "bytes": sum(item.st_size for item in feature_stats),
            "latest_mtime_ns": max((item.st_mtime_ns for item in feature_stats), default=0),
        },
        "mi_backend": mi_backend,
        "gamma": gamma,
        "relation_weight": relation_weight,
        "action_weight": action_weight,
        "kinematic_weight": kinematic_weight,
        "teacher_targets_source": (
            None if teacher_targets_source is None else file_stamp(teacher_targets_source)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    The goal comes only from a declared successful reference. Candidate RGB is
    encoded by LaWAM before this function is called; the MI teacher aligns each
    candidate with the full success trajectory, while the student consumes the
    successful endpoint and measured relations. Unverified Cosmos candidates
    are rejected by the dataset contract.
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
        goal_trajectories: torch.Tensor,
        goal_trajectory_mask: torch.Tensor,
        relations: torch.Tensor,
        names: list[str],
    ) -> torch.Tensor:
        phi = torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device)
        for batch_index in range(tokens.shape[0]):
            length = int(token_mask[batch_index].sum().item())
            if not length:
                continue
            state = tokens[batch_index, :length]
            goal_length = int(goal_trajectory_mask[batch_index].sum().item())
            if not goal_length:
                continue
            goal_trajectory = goal_trajectories[batch_index, :goal_length]
            visual_phi = teacher.potential_aligned(state, goal_trajectory).phi
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
                goal_trajectories = batch["goal_trajectory_tokens"].to(device_obj)  # type: ignore[index]
                goal_trajectory_mask = batch["goal_trajectory_mask"].to(device_obj)  # type: ignore[index]
                names = list(batch["relation_names"])  # type: ignore[arg-type]

                student_chosen = model(chosen, goals, chosen_relations, chosen_mask, goal_mask)
                student_rejected = model(rejected, goals, rejected_relations, rejected_mask, goal_mask)
                with torch.no_grad():
                    teacher_chosen = teacher_potential(
                        chosen,
                        chosen_mask,
                        goal_trajectories,
                        goal_trajectory_mask,
                        chosen_relations,
                        names,
                    )
                    teacher_rejected = teacher_potential(
                        rejected,
                        rejected_mask,
                        goal_trajectories,
                        goal_trajectory_mask,
                        rejected_relations,
                        names,
                    )
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
        "teacher_alignment": "monotonic_viterbi",
        "relation_weight": relation_weight,
        "lambda_rank": lambda_rank,
        "lambda_potential": lambda_potential,
        "lambda_direction": lambda_direction,
        "gamma": gamma,
        "seed": seed,
    }
    _write_yaml_like(out / "train_config.yaml", config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out / "pytorch_model.pt")
    return config


def train_generalization_reward(
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
    goal_dropout: float = 0.2,
    lambda_rank: float = 1.0,
    lambda_potential: float = 1.0,
    lambda_direction: float = 0.5,
    action_weight: float = 1.0,
    kinematic_weight: float = 0.0,
    relation_weight: float = 1.0,
    mi_backend: str = "dame_bspline",
    gamma: float = 0.99,
    seed: int = 0,
    device: str = "cuda",
    teacher_targets_source: str | Path | None = None,
    train_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = (
        "instance_heldout",
        "scene_heldout",
    ),
) -> dict[str, object]:
    """Distill a privileged visual/action/relation teacher into a visual student.

    The teacher sees LaWAM visual tokens, LaWAM transition latents and measured
    MuJoCo relations. The deployable ``VisualGoalPotential`` sees visual tokens
    plus an optional goal only, so RBM-EVAL never requires action or relation
    sidecars.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    from mi_reward.data.geoprogress_dataset import GeoProgressPreferenceDataset
    from mi_reward.models.visual_goal_potential import VisualGoalPotential
    from mi_reward.relations.sequence import relation_progress_potential
    from mi_reward.scoring.mi_potential_field import MIBackend, MIPotentialField
    from mi_reward.training.geoprogress_collator import GeoProgressCollator
    from mi_reward.training.losses import DistillationLoss

    try:
        backend = MIBackend(mi_backend)
    except ValueError as exc:
        choices = ", ".join(item.value for item in MIBackend)
        raise ValueError(f"Unsupported MI backend {mi_backend!r}; choose one of {choices}.") from exc
    dataset = GeoProgressPreferenceDataset(
        preferences,
        manifest,
        success_refs,
        feature_root,
        require_action_latents=action_weight != 0.0,
        require_kinematic_latents=kinematic_weight != 0.0,
        cache_tokens=True,
        allowed_splits=train_splits,
    )
    if not len(dataset):
        raise ValueError("No verified generalization preference pairs found.")
    validation_dataset = GeoProgressPreferenceDataset(
        preferences,
        manifest,
        success_refs,
        feature_root,
        require_action_latents=False,
        require_kinematic_latents=False,
        cache_tokens=False,
        allowed_splits=validation_splits,
    )
    first = dataset[0]
    visual_dim = int(first["chosen_tokens"].shape[-1])  # type: ignore[index]
    model = VisualGoalPotential(
        visual_dim=visual_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        goal_dropout=goal_dropout,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DistillationLoss(
        lambda_rank=lambda_rank,
        lambda_potential=lambda_potential,
        lambda_direction=lambda_direction,
    )
    from tqdm import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_cache_path = out / "teacher_targets.pt"
    target_signature = _teacher_cache_signature(
        preferences=preferences,
        manifest=manifest,
        success_refs=success_refs,
        feature_root=feature_root,
        mi_backend=backend.value,
        gamma=gamma,
        relation_weight=relation_weight,
        action_weight=action_weight,
        kinematic_weight=kinematic_weight,
        teacher_targets_source=teacher_targets_source,
    )
    teacher_targets: dict[tuple[str, str], torch.Tensor] = {}
    if target_cache_path.is_file():
        payload = torch.load(target_cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("signature") == target_signature:
            raw_targets = payload.get("targets")
            if isinstance(raw_targets, dict):
                teacher_targets = {
                    key: value.float()
                    for key, value in raw_targets.items()
                    if isinstance(key, tuple) and len(key) == 2 and isinstance(value, torch.Tensor)
                }

    loaded_from_source = False
    if not teacher_targets and teacher_targets_source is not None:
        source_path = Path(teacher_targets_source)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Directional teacher target file is missing: {source_path}. "
                "Re-run mi_reward.scoring.build_preferences before training."
            )
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(source_payload, dict) or source_payload.get("version") != 2:
            raise ValueError(
                f"Unsupported directional teacher target payload in {source_path}; expected version 2."
            )
        raw_targets = source_payload.get("targets")
        if not isinstance(raw_targets, dict):
            raise ValueError(f"Directional teacher target payload has no targets mapping: {source_path}")
        teacher_targets = {
            key: value.detach().cpu().float()
            for key, value in raw_targets.items()
            if isinstance(key, tuple) and len(key) == 2 and isinstance(value, torch.Tensor)
        }
        loaded_from_source = True

    teacher: MIPotentialField | None = None

    def compute_teacher_target(traj_id: str, goal_ref_id: str) -> torch.Tensor:
        nonlocal teacher
        if teacher is None:
            teacher = MIPotentialField(
                backend=backend,
                gamma=gamma,
                pair_chunk_size=4 if device_obj.type == "cuda" else 1,
            ).to(device_obj)
        from mi_reward.alignment.monotonic_alignment import (
            build_mi_alignment_matrix,
            monotonic_viterbi_alignment,
        )
        from mi_reward.scoring.directional_potential import (
            alignment_stage_potential,
            combine_process_potentials,
            transition_potential_to_frames,
        )

        def aligned_stage(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            assert teacher is not None
            matrix = build_mi_alignment_matrix(
                values,
                reference,
                teacher._mi,
                pair_chunk_size=teacher.pair_chunk_size,
            )
            alignment = monotonic_viterbi_alignment(
                matrix,
                stay_penalty=teacher.stay_penalty,
                jump_penalty=teacher.jump_penalty,
            )
            return alignment_stage_potential(alignment["path"], reference.shape[0])

        item = dataset.load_teacher_inputs(traj_id, goal_ref_id)
        tokens = item["tokens"].to(device_obj)  # type: ignore[union-attr]
        goal_tokens = item["goal_trajectory_tokens"].to(device_obj)  # type: ignore[union-attr]
        relations = item["relations"].to(device_obj)  # type: ignore[union-attr]
        names = list(item["relation_names"])  # type: ignore[arg-type]
        example = dataset.trajectories[traj_id]
        with torch.inference_mode():
            visual_progress = aligned_stage(tokens, goal_tokens)
            relation_progress = relation_progress_potential(
                relations,
                names,
                task_family=example.task_family,
            ).to(device_obj)
            action_progress = None
            if action_weight:
                actions = item["actions"].to(device_obj)  # type: ignore[union-attr]
                goal_actions = item["goal_actions"].to(device_obj)  # type: ignore[union-attr]
                if actions.shape[0] != tokens.shape[0] - 1:
                    raise ValueError(
                        f"Action/frame teacher mismatch: actions={actions.shape[0]}, frames={tokens.shape[0]}."
                    )
                action_progress = transition_potential_to_frames(
                    aligned_stage(actions, goal_actions),
                    tokens.shape[0],
                )
            kinematic_progress = None
            if kinematic_weight:
                kinematics = item["kinematics"].to(device_obj)  # type: ignore[union-attr]
                goal_kinematics = item["goal_kinematics"].to(device_obj)  # type: ignore[union-attr]
                if kinematics.shape[0] != tokens.shape[0]:
                    raise ValueError(
                        f"Kinematic/frame teacher mismatch: kinematic={kinematics.shape[0]}, "
                        f"frames={tokens.shape[0]}."
                    )
                kinematic_progress = aligned_stage(kinematics, goal_kinematics)
            combined = combine_process_potentials(
                visual_progress,
                action=action_progress,
                kinematic=kinematic_progress,
                relation=relation_progress,
                action_weight=action_weight,
                kinematic_weight=kinematic_weight,
                relation_weight=relation_weight,
                task_success=(
                    None if example.task_outcome is None else bool(example.task_outcome.success)
                ),
            )
        return combined.detach().cpu().float()

    teacher_keys = sorted(set(dataset.teacher_keys()) | set(validation_dataset.teacher_keys()))
    pending_keys = [key for key in teacher_keys if key not in teacher_targets]
    if teacher_targets_source is not None and pending_keys:
        raise ValueError(
            f"Directional teacher target source is missing {len(pending_keys)} required trajectories; "
            f"first missing keys: {pending_keys[:5]}. Rebuild preferences for all configured splits."
        )
    for key in tqdm(pending_keys, desc="Privileged teacher targets", unit="trajectory"):
        teacher_targets[key] = compute_teacher_target(*key)
        temporary_cache = target_cache_path.with_suffix(".tmp")
        torch.save(
            {"signature": target_signature, "targets": teacher_targets},
            temporary_cache,
        )
        temporary_cache.replace(target_cache_path)
    if loaded_from_source:
        temporary_cache = target_cache_path.with_suffix(".tmp")
        torch.save(
            {
                "signature": target_signature,
                "target_type": "bounded_directional_process_potential",
                "targets": teacher_targets,
            },
            temporary_cache,
        )
        temporary_cache.replace(target_cache_path)
    dataset.set_teacher_targets(teacher_targets)
    if len(validation_dataset):
        validation_dataset.set_teacher_targets(teacher_targets)
    del teacher
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=GeoProgressCollator())
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=GeoProgressCollator(),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_selection_score = float("-inf")
    best_epoch = -1
    validation_history: list[dict[str, float | int | None]] = []
    with (out / "train_log.jsonl").open("w", encoding="utf-8", buffering=1) as log_f:
        global_step = 0
        epoch_progress = tqdm(range(epochs), desc="Privileged MI distillation", unit="epoch")
        for epoch in epoch_progress:
            total_loss = 0.0
            total_count = 0
            batch_progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
            )
            for batch in batch_progress:
                chosen = batch["chosen_tokens"].to(device_obj)  # type: ignore[index]
                rejected = batch["rejected_tokens"].to(device_obj)  # type: ignore[index]
                goals = batch["goal_tokens"].to(device_obj)  # type: ignore[index]
                chosen_mask = batch["chosen_mask"].to(device_obj)  # type: ignore[index]
                rejected_mask = batch["rejected_mask"].to(device_obj)  # type: ignore[index]
                goal_mask = batch["goal_mask"].to(device_obj)  # type: ignore[index]
                teacher_chosen = batch["chosen_teacher_phi"].to(device_obj)  # type: ignore[index]
                teacher_rejected = batch["rejected_teacher_phi"].to(device_obj)  # type: ignore[index]

                student_chosen = model(chosen, goals, chosen_mask, goal_mask)
                student_rejected = model(rejected, goals, rejected_mask, goal_mask)
                chosen_rewards = _trajectory_score_from_potentials(
                    student_chosen, chosen_mask, gamma
                )
                rejected_rewards = _trajectory_score_from_potentials(
                    student_rejected, rejected_mask, gamma
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
                batch_progress.set_postfix({"loss": f"{float(loss.item()):.4f}"})
                log_f.write(json.dumps({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "rank": float(result["rank"].item()),
                    "potential": float(result["potential"].item()),
                    "direction": float(result["direction"].item()),
                }) + "\n")
                global_step += 1
            mean_loss = total_loss / max(total_count, 1)
            validation_loss = 0.0
            validation_count = 0
            correct_pairs = 0
            directional_correct = 0
            directional_count = 0
            model.eval()
            with torch.inference_mode():
                for batch in tqdm(
                    validation_loader,
                    desc=f"Held-out {epoch + 1}/{epochs}",
                    unit="batch",
                    leave=False,
                    disable=len(validation_dataset) == 0,
                ):
                    chosen = batch["chosen_tokens"].to(device_obj)  # type: ignore[index]
                    rejected = batch["rejected_tokens"].to(device_obj)  # type: ignore[index]
                    goals = batch["goal_tokens"].to(device_obj)  # type: ignore[index]
                    chosen_mask = batch["chosen_mask"].to(device_obj)  # type: ignore[index]
                    rejected_mask = batch["rejected_mask"].to(device_obj)  # type: ignore[index]
                    goal_mask = batch["goal_mask"].to(device_obj)  # type: ignore[index]
                    teacher_chosen = batch["chosen_teacher_phi"].to(device_obj)  # type: ignore[index]
                    teacher_rejected = batch["rejected_teacher_phi"].to(device_obj)  # type: ignore[index]
                    student_chosen = model(chosen, goals, chosen_mask, goal_mask)
                    student_rejected = model(rejected, goals, rejected_mask, goal_mask)
                    chosen_rewards = _trajectory_score_from_potentials(
                        student_chosen, chosen_mask, gamma
                    )
                    rejected_rewards = _trajectory_score_from_potentials(
                        student_rejected, rejected_mask, gamma
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
                    count = chosen.shape[0]
                    validation_loss += float(result["total"].item()) * count
                    validation_count += count
                    correct_pairs += int((chosen_rewards > rejected_rewards).sum().item())
                    for student_phi, teacher_phi, mask in (
                        (student_chosen, teacher_chosen, chosen_mask),
                        (student_rejected, teacher_rejected, rejected_mask),
                    ):
                        lengths = mask.sum(dim=1).long()
                        last_index = (lengths - 1).clamp_min(0).unsqueeze(1)
                        student_gain = student_phi.gather(1, last_index).squeeze(1) - student_phi[:, 0]
                        teacher_gain = teacher_phi.gather(1, last_index).squeeze(1) - teacher_phi[:, 0]
                        eligible = teacher_gain >= 0.05
                        directional_correct += int(((student_gain > 0) & eligible).sum().item())
                        directional_count += int(eligible.sum().item())
            model.train()
            val_mean_loss = (
                validation_loss / validation_count if validation_count else None
            )
            val_pair_accuracy = (
                correct_pairs / validation_count if validation_count else None
            )
            val_direction_accuracy = (
                directional_correct / directional_count if directional_count else None
            )
            epoch_metrics: dict[str, float | int | None] = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_loss": val_mean_loss,
                "validation_pairs": validation_count,
                "validation_pair_accuracy": val_pair_accuracy,
                "validation_direction_accuracy": val_direction_accuracy,
            }
            validation_history.append(epoch_metrics)
            log_f.write(json.dumps(epoch_metrics) + "\n")
            selection_score = (
                val_pair_accuracy if val_pair_accuracy is not None else -mean_loss
            )
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            postfix = {"loss": f"{mean_loss:.4f}"}
            if val_pair_accuracy is not None:
                postfix["val_acc"] = f"{val_pair_accuracy:.3f}"
            epoch_progress.set_postfix(postfix)

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_report = {
        "selection_metric": (
            "validation_pair_accuracy" if len(validation_dataset) else "negative_train_loss"
        ),
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "train_pairs": len(dataset),
        "validation_pairs": len(validation_dataset),
        "history": validation_history,
    }
    (out / "validation_metrics.json").write_text(
        json.dumps(validation_report, indent=2),
        encoding="utf-8",
    )

    config = {
        "model_class": "VisualGoalPotential",
        "preferences": str(preferences),
        "manifest": str(manifest),
        "success_refs": str(success_refs),
        "feature_root": str(feature_root),
        "visual_dim": visual_dim,
        "hidden_dim": hidden_dim,
        "architecture": architecture,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "goal_dropout": goal_dropout,
        "mi_backend": backend.value,
        "teacher_alignment": "monotonic_viterbi",
        "teacher_target_cache": str(target_cache_path),
        "teacher_targets_source": (
            None if teacher_targets_source is None else str(teacher_targets_source)
        ),
        "teacher_target_type": "bounded_directional_process_potential",
        "teacher_target_count": len(teacher_targets),
        "teacher_inputs": ["visual_tokens", "action_latents", "kinematic_latents", "measured_relations"],
        "deployment_inputs": ["visual_tokens", "optional_goal_tokens"],
        "action_weight": action_weight,
        "kinematic_weight": kinematic_weight,
        "relation_weight": relation_weight,
        "lambda_rank": lambda_rank,
        "lambda_potential": lambda_potential,
        "lambda_direction": lambda_direction,
        "gamma": gamma,
        "seed": seed,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "train_pairs": len(dataset),
        "validation_pairs": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_pair_accuracy": (
            None
            if not validation_history
            else max(
                (
                    item["validation_pair_accuracy"]
                    for item in validation_history
                    if item["validation_pair_accuracy"] is not None
                ),
                default=None,
            )
        ),
    }
    _write_yaml_like(out / "train_config.yaml", config)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out / "pytorch_model.pt")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train legacy rewards or the verified generalization reward potential.")
    parser.add_argument(
        "--mode",
        default="ranking",
        choices=["ranking", "distill", "generalization", "geoprogress"],
        help="generalization: privileged-teacher visual student; geoprogress: legacy relation-dependent student.",
    )
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
    parser.add_argument("--action_weight", type=float, default=1.0)
    parser.add_argument("--kinematic_weight", type=float, default=0.0)
    parser.add_argument("--goal_dropout", type=float, default=0.2)
    parser.add_argument("--mi_backend", default="gaussian_mi_proxy", choices=["gaussian_mi_proxy", "histogram_mi", "dame_bspline"])
    parser.add_argument(
        "--teacher_targets_source",
        default=None,
        help="Version-2 bounded teacher curves emitted by build_preferences.",
    )
    parser.add_argument(
        "--train_split",
        dest="train_splits",
        action="append",
        default=None,
        help="Manifest split used for optimization; repeat for multiple splits.",
    )
    parser.add_argument(
        "--validation_split",
        dest="validation_splits",
        action="append",
        default=None,
        help="Held-out manifest split used for checkpoint selection; repeat for multiple splits.",
    )
    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if k != "mode"}
    kwargs["train_splits"] = tuple(kwargs["train_splits"] or ("train",))
    kwargs["validation_splits"] = tuple(
        kwargs["validation_splits"]
        or ("instance_heldout", "scene_heldout")
    )
    if args.mode in {"generalization", "geoprogress"}:
        if not args.manifest or not args.success_refs:
            parser.error("--manifest and --success_refs are required for --mode generalization")
        if args.mode == "generalization":
            train_generalization_reward(**kwargs)
        else:
            kwargs.pop("action_weight")
            kwargs.pop("kinematic_weight")
            kwargs.pop("goal_dropout")
            kwargs.pop("teacher_targets_source")
            kwargs.pop("train_splits")
            kwargs.pop("validation_splits")
            train_geoprogress(**kwargs)
    elif args.mode == "distill":
        for key in ("manifest", "success_refs", "num_layers", "num_heads", "dropout", "relation_weight", "action_weight", "kinematic_weight", "goal_dropout", "mi_backend", "teacher_targets_source", "train_splits", "validation_splits"):
            kwargs.pop(key)
        train_reward_distill(**kwargs)
    else:
        for key in ("manifest", "success_refs", "num_layers", "num_heads", "dropout", "relation_weight", "action_weight", "kinematic_weight", "goal_dropout", "mi_backend", "teacher_targets_source", "train_splits", "validation_splits"):
            kwargs.pop(key)
        train_reward_sft(**kwargs)


if __name__ == "__main__":
    main()
