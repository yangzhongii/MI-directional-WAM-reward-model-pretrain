"""Train the canonical Pipeline-v3 information teacher on official LIBERO demos.

The first-stage implementation is deliberately narrow and auditable:

* ``v_t`` is the concatenation of pooled synchronized agentview and wrist
  LaWAM visual tokens;
* ``g`` is a short average of the successful terminal dual-view latent;
* ``a_t^lat`` is the pooled LaWAM transition latent inferred from agentview;
* visual marginal negatives use another successful reference from the same
  LIBERO task;
* conditional-action negatives are mined from the same task by nearest visual
  state, while requiring a non-forward privileged physical direction.

Privileged geometry is used only to construct/evaluate hard negatives.  It is
not added to the information score.  The resulting information-only score is

    D_t = gamma * phi_v(t+1) - phi_v(t) + beta * psi_a(t).

Physical consistency / abstention and P/U/N calibration remain downstream
Pipeline-v3 stages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mi_reward.data.libero_privileged import (
    classify_physical_transition,
    extract_privileged_records,
    resolve_libero_task,
)
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor
from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
    conditional_action_information_loss,
    visual_information_loss,
)


@dataclass
class EpisodeFeatures:
    task_id: int
    demo_name: str
    split: str
    visual: torch.Tensor  # [T, 2D]
    action: torch.Tensor  # [T-1, A]
    goal: torch.Tensor  # [2D]
    direction: torch.Tensor  # [T-1], {-1,0,+1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--demos-per-task", type=int, default=3)
    parser.add_argument("--validation-demos", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Calibrated by a held-out 2-task LIBERO smoke on 2026-09-06.  Keep beta
    # explicit in reports/checkpoints; larger experiments should recalibrate it
    # on their dedicated teacher-validation split.
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visual-chunk", type=int, default=16)
    parser.add_argument("--goal-window", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("logs/mi_reward/v3_teacher_mainline/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/mi_reward/v3_teacher_mainline/teacher"))
    parser.add_argument("--lam-config", default=".venv/models/lawam_lam/dino_large_vae.yaml")
    parser.add_argument("--lam-checkpoint", default=".venv/models/lawam_lam/checkpoints/pytorch_model.pt")
    parser.add_argument("--dino-model", default=".venv/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _episode_cache_path(cache_dir: Path, task_id: int, demo_name: str) -> Path:
    return cache_dir / f"task_{task_id:02d}_{demo_name}.pt"


def _extract_visual(
    extractor: LaWAMLAMFeatureExtractor,
    images: np.ndarray,
    task: str,
    chunk: int,
) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    for start in range(0, len(images), chunk):
        part = [np.asarray(image, dtype=np.uint8) for image in images[start : start + chunk]]
        tokens = extractor.extract_trajectory_tokens_images(part, task)
        pooled.append(tokens.float().mean(dim=1))
    return torch.cat(pooled, dim=0)


def _first_success_index(frames: list[dict[str, object]]) -> int | None:
    return next((int(frame["frame_index"]) for frame in frames if bool(frame["environment_success"])), None)


def _extract_episode(
    *,
    extractor: LaWAMLAMFeatureExtractor,
    demo_path: Path,
    bddl_path: Path,
    task_text: str,
    task_id: int,
    demo_name: str,
    split: str,
    cache_path: Path,
    visual_chunk: int,
    goal_window: int,
) -> EpisodeFeatures:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return EpisodeFeatures(**payload)

    import h5py

    with h5py.File(demo_path, "r") as handle:
        demo = handle["data"][demo_name]
        obs = demo["obs"]
        agent = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
        rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float64)
        dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.uint8)

    length = min(len(agent), len(wrist), len(states), len(actions), len(rewards), len(dones))
    agent = agent[:length]
    wrist = wrist[:length]
    states = states[:length]
    actions = actions[:length]
    rewards = rewards[:length]
    dones = dones[:length]

    privileged = extract_privileged_records(
        bddl_path=bddl_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    physical_frames = privileged["frames"]
    directions = torch.tensor(
        [
            classify_physical_transition(left, right)
            for left, right in zip(physical_frames[:-1], physical_frames[1:])
        ],
        dtype=torch.int64,
    )

    agent_visual = _extract_visual(extractor, agent, task_text, visual_chunk)
    wrist_visual = _extract_visual(extractor, wrist, task_text, visual_chunk)
    visual = torch.cat((agent_visual, wrist_visual), dim=-1).float().cpu()
    action_tokens = extractor.extract_action_latents_images(
        [np.asarray(image, dtype=np.uint8) for image in agent],
        task_text,
    )
    action = action_tokens.float().mean(dim=1).cpu()

    if visual.shape[0] != length or action.shape[0] != length - 1 or directions.shape[0] != length - 1:
        raise RuntimeError(
            f"Feature alignment mismatch for task={task_id} {demo_name}: "
            f"visual={tuple(visual.shape)} action={tuple(action.shape)} direction={tuple(directions.shape)}"
        )

    success = _first_success_index(physical_frames)
    goal_start = max(0, (length - goal_window) if success is None else success)
    goal_stop = min(length, goal_start + max(int(goal_window), 1))
    goal = visual[goal_start:goal_stop].mean(dim=0)

    episode = EpisodeFeatures(
        task_id=int(task_id),
        demo_name=str(demo_name),
        split=str(split),
        visual=visual,
        action=action,
        goal=goal,
        direction=directions,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(episode.__dict__, cache_path)
    return episode


def load_episodes(args: argparse.Namespace, extractor: LaWAMLAMFeatureExtractor) -> list[EpisodeFeatures]:
    root = Path.cwd().resolve()
    episodes: list[EpisodeFeatures] = []
    for task_id in args.task_ids:
        demo_path, bddl_path, task_text = resolve_libero_task(root, args.suite, int(task_id))
        import h5py

        with h5py.File(demo_path, "r") as handle:
            names = sorted(handle["data"].keys())[: int(args.demos_per_task)]
        if len(names) <= int(args.validation_demos):
            raise ValueError(
                f"Task {task_id} needs more demos than validation_demos={args.validation_demos}; got {len(names)}"
            )
        validation = set(names[-int(args.validation_demos) :]) if args.validation_demos else set()
        for demo_name in names:
            split = "validation" if demo_name in validation else "train"
            print(f"[features] task={task_id} demo={demo_name} split={split}", flush=True)
            episodes.append(
                _extract_episode(
                    extractor=extractor,
                    demo_path=demo_path,
                    bddl_path=bddl_path,
                    task_text=task_text,
                    task_id=int(task_id),
                    demo_name=demo_name,
                    split=split,
                    cache_path=_episode_cache_path(args.cache_dir, int(task_id), demo_name),
                    visual_chunk=int(args.visual_chunk),
                    goal_window=int(args.goal_window),
                )
            )
    return episodes


def _same_task_other_goal(episodes: list[EpisodeFeatures], episode: EpisodeFeatures) -> torch.Tensor:
    choices = [item.goal for item in episodes if item.task_id == episode.task_id and item.demo_name != episode.demo_name]
    if not choices:
        raise ValueError(f"Task {episode.task_id} needs at least two training demos for same-task goal negatives.")
    return choices[0]


def build_visual_samples(episodes: list[EpisodeFeatures]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = [episode for episode in episodes if episode.split == "train"]
    visual: list[torch.Tensor] = []
    goals: list[torch.Tensor] = []
    negative_goals: list[torch.Tensor] = []
    for episode in train:
        negative_goal = _same_task_other_goal(train, episode)
        visual.append(episode.visual)
        goals.append(episode.goal.unsqueeze(0).expand(episode.visual.shape[0], -1))
        negative_goals.append(negative_goal.unsqueeze(0).expand(episode.visual.shape[0], -1))
    return torch.cat(visual), torch.cat(goals), torch.cat(negative_goals)


def build_action_hard_negatives(
    episodes: list[EpisodeFeatures],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    train = [episode for episode in episodes if episode.split == "train"]
    positive_v: list[torch.Tensor] = []
    positive_g: list[torch.Tensor] = []
    positive_a: list[torch.Tensor] = []
    negative_a: list[torch.Tensor] = []
    per_task: dict[str, int] = {}

    for task_id in sorted({episode.task_id for episode in train}):
        task_episodes = [episode for episode in train if episode.task_id == task_id]
        pos_states: list[torch.Tensor] = []
        pos_actions: list[torch.Tensor] = []
        pos_goals: list[torch.Tensor] = []
        neg_states: list[torch.Tensor] = []
        neg_actions: list[torch.Tensor] = []
        for episode in task_episodes:
            pos_index = torch.nonzero(episode.direction > 0, as_tuple=False).flatten()
            neg_index = torch.nonzero(episode.direction <= 0, as_tuple=False).flatten()
            if pos_index.numel():
                pos_states.append(episode.visual[:-1][pos_index])
                pos_actions.append(episode.action[pos_index])
                pos_goals.append(episode.goal.unsqueeze(0).expand(pos_index.numel(), -1))
            if neg_index.numel():
                neg_states.append(episode.visual[:-1][neg_index])
                neg_actions.append(episode.action[neg_index])
        if not pos_states or not neg_states:
            continue
        pv = torch.cat(pos_states)
        pa = torch.cat(pos_actions)
        pg = torch.cat(pos_goals)
        nv = torch.cat(neg_states)
        na = torch.cat(neg_actions)
        similarity = F.normalize(pv, dim=-1) @ F.normalize(nv, dim=-1).T
        nearest = similarity.argmax(dim=1)
        positive_v.append(pv)
        positive_a.append(pa)
        positive_g.append(pg)
        negative_a.append(na[nearest])
        per_task[str(task_id)] = int(pv.shape[0])

    if not positive_v:
        raise RuntimeError("No same-task forward/non-forward action hard negatives were constructed.")
    return (
        torch.cat(positive_a),
        torch.cat(positive_v),
        torch.cat(positive_g),
        torch.cat(negative_a),
        per_task,
    )


def _direction_accuracy(score: torch.Tensor, direction: torch.Tensor) -> tuple[int, int]:
    keep = direction != 0
    if not bool(keep.any()):
        return 0, 0
    correct = ((score[keep] > 0) == (direction[keep] > 0)).sum().item()
    return int(correct), int(keep.sum().item())


@torch.no_grad()
def evaluate(
    visual_critic: VisualPointwiseInformationCritic,
    action_critic: ConditionalActionInformationCritic,
    episodes: list[EpisodeFeatures],
    *,
    device: torch.device,
    gamma: float,
    beta: float,
) -> dict[str, Any]:
    totals = {"cosine": [0, 0], "delta_phi": [0, 0], "psi": [0, 0], "directional_D": [0, 0]}
    task_rows: list[dict[str, Any]] = []
    for episode in [item for item in episodes if item.split == "validation"]:
        visual = episode.visual.to(device)
        action = episode.action.to(device)
        goal = episode.goal.to(device).unsqueeze(0)
        direction = episode.direction.to(device)
        goal_frames = goal.expand(visual.shape[0], -1)
        phi = visual_critic(visual, goal_frames)
        psi = action_critic(action, visual[:-1], goal.expand(action.shape[0], -1))
        delta_phi = float(gamma) * phi[1:] - phi[:-1]
        directional = delta_phi + float(beta) * psi
        cosine = F.cosine_similarity(visual, goal_frames, dim=-1)
        delta_cosine = cosine[1:] - cosine[:-1]

        row: dict[str, Any] = {
            "task_id": episode.task_id,
            "demo_name": episode.demo_name,
            "forward": int((direction > 0).sum().item()),
            "neutral": int((direction == 0).sum().item()),
            "regression": int((direction < 0).sum().item()),
        }
        for name, score in (
            ("cosine", delta_cosine),
            ("delta_phi", delta_phi),
            ("psi", psi),
            ("directional_D", directional),
        ):
            correct, count = _direction_accuracy(score, direction)
            totals[name][0] += correct
            totals[name][1] += count
            row[f"{name}_direction_accuracy"] = None if count == 0 else correct / count
        task_rows.append(row)

    aggregate = {
        name: (None if count == 0 else correct / count)
        for name, (correct, count) in totals.items()
    }
    return {"aggregate": aggregate, "episodes": task_rows}


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    extractor = LaWAMLAMFeatureExtractor(
        lam_config_path=args.lam_config,
        lam_ckpt_path=args.lam_checkpoint,
        vision_model_id=args.dino_model,
        device=str(device),
        fallback=None,
        strict=True,
    )
    print(f"[model] LaWAM loaded={extractor.using_lam} device={device}", flush=True)

    episodes = load_episodes(args, extractor)
    train_episodes = [episode for episode in episodes if episode.split == "train"]
    validation_episodes = [episode for episode in episodes if episode.split == "validation"]
    if not train_episodes or not validation_episodes:
        raise RuntimeError("Both train and validation episodes are required.")

    visual, goals, negative_goals = build_visual_samples(episodes)
    action, action_visual, action_goals, negative_action, hard_negative_report = build_action_hard_negatives(episodes)
    visual_dim = int(visual.shape[-1])
    action_dim = int(action.shape[-1])
    visual_critic = VisualPointwiseInformationCritic(visual_dim, visual_dim).to(device)
    action_critic = ConditionalActionInformationCritic(action_dim, visual_dim, visual_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(visual_critic.parameters()) + list(action_critic.parameters()),
        lr=float(args.lr),
    )

    visual = visual.to(device)
    goals = goals.to(device)
    negative_goals = negative_goals.to(device)
    action = action.to(device)
    action_visual = action_visual.to(device)
    action_goals = action_goals.to(device)
    negative_action = negative_action.to(device)
    batch_size = int(args.batch_size)
    last_loss = 0.0
    for step in range(int(args.steps)):
        vi = torch.randint(0, visual.shape[0], (min(batch_size, visual.shape[0]),), device=device)
        ai = torch.randint(0, action.shape[0], (min(batch_size, action.shape[0]),), device=device)
        loss_visual = visual_information_loss(
            visual_critic,
            visual[vi],
            goals[vi],
            negative_goals[vi],
        )
        loss_action = conditional_action_information_loss(
            action_critic,
            action[ai],
            action_visual[ai],
            action_goals[ai],
            negative_action[ai],
        )
        loss = loss_visual + loss_action
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(visual_critic.parameters()) + list(action_critic.parameters()),
            5.0,
        )
        optimizer.step()
        last_loss = float(loss.item())
        if step == 0 or (step + 1) % 100 == 0:
            print(
                f"[train] step={step + 1}/{args.steps} loss={loss.item():.4f} "
                f"visual={loss_visual.item():.4f} action={loss_action.item():.4f}",
                flush=True,
            )

    metrics = evaluate(
        visual_critic,
        action_critic,
        episodes,
        device=device,
        gamma=float(args.gamma),
        beta=float(args.beta),
    )
    report = {
        "pipeline": "v3_visual_pmi_plus_conditional_action_pmi",
        "suite": args.suite,
        "task_ids": list(args.task_ids),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "visual_dim": visual_dim,
        "action_dim": action_dim,
        "visual_training_samples": int(visual.shape[0]),
        "action_hard_negative_samples": int(action.shape[0]),
        "action_hard_negatives_by_task": hard_negative_report,
        "gamma": float(args.gamma),
        "beta": float(args.beta),
        "final_training_loss": last_loss,
        "evaluation": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "visual_critic": visual_critic.state_dict(),
            "action_critic": action_critic.state_dict(),
            "visual_dim": visual_dim,
            "action_dim": action_dim,
            "gamma": float(args.gamma),
            "beta": float(args.beta),
            "report": report,
        },
        args.output_dir / "information_teacher_v3.pt",
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
