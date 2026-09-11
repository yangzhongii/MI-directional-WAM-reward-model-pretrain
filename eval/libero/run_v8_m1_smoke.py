"""Frozen v8 M1 smoke: task-00 cached DINO, physical sidecars, 3 seeds.

This is deliberately a one-task signal check, not the final v8 method gate:
there are no matched visual-shift variants in this cache.  B5 and M1 share
the ranking, stage and physical regression losses; M1 alone adds the
task/stage-conditioned symmetric physical InfoNCE term.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task
from mi_reward.models.visual_goal_potential import VisualGoalPotential


class M1Head(nn.Module):
    def __init__(self, dim: int, physical_dim: int, enabled_mi: bool) -> None:
        super().__init__()
        self.base = VisualGoalPotential(dim, hidden_dim=128, architecture="mlp", num_heads=4, dropout=0.0, goal_dropout=0.0)
        self.stage = nn.Linear(128, 5)
        self.physical = nn.Linear(128, physical_dim)
        self.visual_proj = nn.Linear(128, 64)
        self.physical_proj = nn.Linear(physical_dim, 64)
        self.enabled_mi = enabled_mi

    def forward(self, visual: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.base.encode(visual[:, None, None], goal[:, None])[:, 0]
        return self.base.potential_head(hidden).squeeze(-1), self.stage(hidden), self.physical(hidden), hidden


def stage_and_factor(frame: dict[str, object]) -> tuple[int, list[float], float]:
    success = bool(frame["environment_success"])
    contact = bool(frame["object_goal_contact"])
    grasped = bool(frame["grasped"])
    eef_dist = float(frame["eef_object_distance_xyz"])
    object_dist = float(frame["object_goal_distance_xyz"])
    if success:
        stage = 4
    elif contact:
        stage = 3
    elif grasped:
        stage = 2
    elif eef_dist < 0.04:
        stage = 1
    else:
        stage = 0
    eef = np.asarray(frame["eef_pos"], dtype=np.float32)
    obj = np.asarray(frame["task_object_pos"], dtype=np.float32)
    goal = np.asarray(frame["goal_object_pos"], dtype=np.float32)
    action = np.asarray(frame["action"], dtype=np.float32)
    factors = [eef_dist, object_dist, *(eef - obj).tolist(), *(obj - goal).tolist(), float(action[-1]), float(contact), float(grasped), float(obj[2])]
    # This is only a privileged pair constructor. MI never sees this scalar.
    progress = float(stage * 10.0 - (object_dist if stage >= 2 else eef_dist))
    return stage, factors, progress


def load_demo(root: Path, demo: str, split: str, mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    demo_path, bddl, _ = resolve_libero_task(root, "libero_spatial", 0)
    payload = torch.load(root / "logs/mi_reward/v3_teacher_mainline/features" / f"task_00_{demo}.pt", map_location="cpu", weights_only=False)
    with h5py.File(demo_path, "r") as handle:
        group = handle["data"][demo]
        privileged = extract_privileged_records(bddl_path=bddl, states=group["states"][:], actions=group["actions"][:], rewards=group["rewards"][:], dones=group["dones"][:])
    records = privileged["frames"]
    if len(records) != payload["visual"].shape[0]:
        raise ValueError(f"{demo}: physical/features length mismatch")
    stage, factor, progress = zip(*(stage_and_factor(frame) for frame in records))
    factors = torch.tensor(np.asarray(factor), dtype=torch.float32)
    if mean is not None and std is not None:
        factors = (factors - mean) / std
    return {"visual": payload["visual"].float(), "goal": payload["goal"].float(), "stage": torch.tensor(stage), "physical": factors, "progress": torch.tensor(progress), "split": torch.tensor(0 if split == "train" else 1)}


def infonce(visual: torch.Tensor, physical: torch.Tensor, stage: torch.Tensor, model: M1Head, tau: float = 0.2) -> torch.Tensor:
    q = F.normalize(model.visual_proj(visual), dim=-1)
    k = F.normalize(model.physical_proj(physical), dim=-1)
    losses = []
    for label in stage.unique():
        idx = (stage == label).nonzero().flatten()
        if len(idx) < 2:
            continue
        logits = q[idx] @ k[idx].T / tau
        target = torch.arange(len(idx), device=visual.device)
        losses.extend((F.cross_entropy(logits, target), F.cross_entropy(logits.T, target)))
    return torch.stack(losses).mean() if losses else visual.new_zeros(())


def pair_accuracy(value: torch.Tensor, progress: torch.Tensor) -> float:
    left, right = torch.triu_indices(len(value), len(value), offset=1, device=value.device)
    valid = (progress[left] - progress[right]).abs() > 1e-5
    if not bool(valid.any()):
        return float("nan")
    correct = ((value[left] - value[right]) * (progress[left] - progress[right]) > 0)[valid]
    return float(correct.float().mean())


def probe_r2(train_h: torch.Tensor, train_y: torch.Tensor, test_h: torch.Tensor, test_y: torch.Tensor) -> float:
    x = torch.cat((train_h, torch.ones(len(train_h), 1, device=train_h.device)), dim=1)
    weight = torch.linalg.lstsq(x, train_y).solution
    prediction = torch.cat((test_h, torch.ones(len(test_h), 1, device=test_h.device)), dim=1) @ weight
    return float((1 - (prediction-test_y).square().sum(0) / (test_y-test_y.mean(0)).square().sum(0).clamp_min(1e-8)).mean())


def run(seed: int, enabled_mi: bool, train: dict[str, torch.Tensor], held: dict[str, torch.Tensor], steps: int, device: torch.device) -> dict[str, float]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = M1Head(train["visual"].shape[1], train["physical"].shape[1], enabled_mi).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train = {k: v.to(device) for k, v in train.items() if isinstance(v, torch.Tensor)}
    held = {k: v.to(device) for k, v in held.items() if isinstance(v, torch.Tensor)}
    last = {}
    for step in range(steps):
        index = torch.randint(len(train["visual"]), (min(96, len(train["visual"])),), device=device)
        value, stage_logits, physical_hat, hidden = model(train["visual"][index], train["goal"].expand(len(index), -1))
        other = torch.roll(torch.arange(len(index), device=device), 1)
        rank = F.softplus(-torch.sign(train["progress"][index]-train["progress"][index][other]) * (value-value[other])).mean()
        stage_loss = F.cross_entropy(stage_logits, train["stage"][index])
        physical_loss = F.mse_loss(physical_hat, train["physical"][index])
        mi_loss = infonce(hidden, train["physical"][index], train["stage"][index], model) if enabled_mi else value.new_zeros(())
        loss = rank + 0.5*stage_loss + 0.2*physical_loss + 0.1*mi_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        last = {"loss": float(loss), "rank_loss": float(rank), "stage_loss": float(stage_loss), "physical_loss": float(physical_loss), "cmi_loss": float(mi_loss)}
    with torch.no_grad():
        tr_value, _, _, tr_h = model(train["visual"], train["goal"].expand(len(train["visual"]), -1))
        te_value, stage_logits, _, te_h = model(held["visual"], held["goal"].expand(len(held["visual"]), -1))
        return {**last, "held_pair_accuracy": pair_accuracy(te_value, held["progress"]), "held_stage_accuracy": float((stage_logits.argmax(-1)==held["stage"]).float().mean()), "held_physical_probe_r2": probe_r2(tr_h, train["physical"], te_h, held["physical"]), "train_pair_accuracy": pair_accuracy(tr_value, train["progress"])}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--steps", type=int, default=1000); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    root = Path(__file__).resolve().parents[2]
    raw_train = [load_demo(root, name, "train") for name in ("demo_0", "demo_1")]
    mean = torch.cat([x["physical"] for x in raw_train]).mean(0); std = torch.cat([x["physical"] for x in raw_train]).std(0).clamp_min(1e-4)
    train_parts = [load_demo(root, name, "train", mean, std) for name in ("demo_0", "demo_1")]
    held = load_demo(root, "demo_10", "heldout", mean, std)
    train = {key: torch.cat([part[key] for part in train_parts], dim=0) if key != "goal" else train_parts[0][key] for key in ("visual", "stage", "physical", "progress")}
    train["goal"] = train_parts[0]["goal"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for seed in (7, 17, 27):
        for method, enabled in (("B5_base", False), ("M1_physical_cmi", True)):
            records.append({"seed": seed, "method": method, **run(seed, enabled, train, held, args.steps, device)})
    summary = {method: {metric: float(np.mean([r[metric] for r in records if r["method"] == method])) for metric in ("held_pair_accuracy", "held_stage_accuracy", "held_physical_probe_r2")} for method in ("B5_base", "M1_physical_cmi")}
    report = {"protocol": "v8_m1_minimal_gate_smoke_v1", "scope": "one_task_cached_dino_three_seed_no_visual_shift_variants", "task": "libero_spatial/task-00", "steps": args.steps, "physical_factor_dim": int(mean.numel()), "records": records, "summary": summary, "formal_v8_method_gate": "NOT_EVALUABLE_without_matched_state_visual_shift_and_success_failure_heldout"}
    args.output_dir.mkdir(parents=True); (args.output_dir/"results.json").write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
