"""Visual RLPD closed loop using the frozen MI potential as reward."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from mi_reward.closed_loop.libero_demo import demonstration_path, iter_demo_episodes, resize_views
from mi_reward.closed_loop.libero_env import (
    LiberoTaskEnvironment,
    apply_appearance_shift,
    policy_camera_images,
    rlpd_robot_state,
    split_fingerprint,
)
from mi_reward.closed_loop.libero_sac import (
    _goal_tokens,
    _make_extractor,
    _reward_adapter,
    _robot_state_references,
    _validate_splits,
)
from mi_reward.closed_loop.online_potential import PotentialReward
from mi_reward.closed_loop.rlpd import PixelReplayBuffer, RLPDAgent, RLPDConfig, balanced_batch
from mi_reward.worldsample.ppl import PPLConfig, load_synthetic_replay, paced_batch


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {config_path}")
    return config, Path.cwd().resolve()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment(config: dict[str, Any], task_id: int, seed: int) -> LiberoTaskEnvironment:
    benchmark = dict(config["benchmark"])
    policy = dict(config["policy"])
    return LiberoTaskEnvironment(
        str(benchmark.get("suite", "libero_spatial")),
        task_id,
        benchmark_variant=str(benchmark.get("variant", "libero")),
        resolution=int(policy.get("image_size", 128)),
        seed=seed,
        settle_steps=int(benchmark.get("settle_steps", 10)),
        max_episode_steps=benchmark.get("max_episode_steps"),
        camera_names=policy.get("camera_names", ("agentview", "robot0_eye_in_hand")),
    )


def _run_directory(config: dict[str, Any], root: Path, reward_mode: str, seed: int, task_id: int) -> Path:
    output = Path(config.get("output_root", "logs/mi_reward/libero_rlpd"))
    if not output.is_absolute():
        output = root / output
    return output / str(config.get("run_id", "libero_mi_rlpd_v1")) / f"task-{task_id:02d}" / reward_mode / f"seed-{seed}"


def _rlpd_config(config: dict[str, Any], root: Path) -> RLPDConfig:
    values = dict(config["training"].get("rlpd", {}))
    allowed = set(RLPDConfig.__dataclass_fields__)
    values = {key: value for key, value in values.items() if key in allowed}
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(int(value) for value in values["hidden_dims"])
    checkpoint = values.get("encoder_checkpoint")
    if checkpoint:
        path = Path(str(checkpoint)).expanduser()
        values["encoder_checkpoint"] = str(path if path.is_absolute() else (root / path).resolve())
    return RLPDConfig(**values)


def _policy_inputs(config: dict[str, Any], observation: dict[str, Any], appearance: str = "none") -> tuple[np.ndarray, np.ndarray]:
    policy = dict(config["policy"])
    keys = tuple(policy.get("observation_keys", ("agentview_image", "robot0_eye_in_hand_image")))
    images = policy_camera_images(observation, keys)
    if appearance not in {"", "none", "id"}:
        images = np.stack([apply_appearance_shift(image, appearance) for image in images], axis=0)
    images = resize_views(images, int(policy.get("image_size", 128)))
    return images, rlpd_robot_state(observation)


def _demo_policy_views(episode: dict[str, Any], index: int, image_size: int) -> np.ndarray:
    return resize_views(np.stack((episode["main"][index], episode["wrist"][index]), axis=0), image_size)


def _fill_demo_buffer(
    *,
    config: dict[str, Any],
    env: LiberoTaskEnvironment,
    extractor: Any,
    goal_tokens: torch.Tensor | None,
    robot_state_references: tuple[np.ndarray, ...],
    reward_mode: str,
    root: Path,
    buffer: PixelReplayBuffer,
    max_demos: int | None,
    max_transitions: int | None = None,
) -> dict[str, Any]:
    path = demonstration_path(env, root)
    episodes = list(iter_demo_episodes(path, max_demos=max_demos))
    progress = tqdm(episodes, desc=f"Load RLPD demos task={env.task_id}")
    image_size = int(config["policy"].get("image_size", 128))
    transitions = 0
    trend_counts = {"positive": 0, "negative": 0, "unclear": 0}
    for episode in progress:
        adapter = _reward_adapter(
            config, goal_tokens, reward_mode, root, robot_state_references
        )
        if adapter is not None:
            if extractor is None:
                raise RuntimeError("MI demo relabeling requires a visual feature extractor.")
            initial_tokens = extractor.extract_image_tokens(
                np.asarray(episode["main"][0]), env.task_description
            ).float()
            adapter.reset(initial_tokens, np.asarray(episode["states"][0], dtype=np.float32))
        length = len(episode["actions"])
        for index in range(length - 1):
            terminal = index == length - 2 or bool(episode["dones"][index])
            sparse = float(episode["rewards"][index])
            if terminal:
                # These are official successful demonstrations. Some releases
                # place the success reward on the unpaired final action.
                sparse = max(sparse, float(np.max(episode["rewards"][index:])), 1.0)
            if adapter is None:
                reward = float(config["reward"].get("sparse_weight", 1.0)) * sparse
            else:
                next_tokens = extractor.extract_image_tokens(
                    np.asarray(episode["main"][index + 1]), env.task_description
                ).float()
                breakdown = adapter.step(
                    next_tokens,
                    sparse,
                    np.asarray(episode["states"][index + 1], dtype=np.float32),
                )
                reward = breakdown.total
                trend_counts[breakdown.trend_label] += 1
            buffer.add(
                _demo_policy_views(episode, index, image_size),
                np.asarray(episode["states"][index], dtype=np.float32),
                np.asarray(episode["actions"][index], dtype=np.float32),
                reward,
                _demo_policy_views(episode, index + 1, image_size),
                np.asarray(episode["states"][index + 1], dtype=np.float32),
                terminal,
            )
            transitions += 1
            if max_transitions is not None and transitions >= max_transitions:
                return {
                    "path": str(path), "episodes": len(episodes),
                    "transitions": transitions, "trend_counts": trend_counts,
                }
        progress.set_postfix(transitions=transitions)
    return {
        "path": str(path), "episodes": len(episodes),
        "transitions": transitions, "trend_counts": trend_counts,
    }


def _evaluate(
    *,
    config: dict[str, Any],
    env: LiberoTaskEnvironment,
    extractor: Any,
    agent: RLPDAgent,
    goal_tokens: torch.Tensor | None,
    robot_state_references: tuple[np.ndarray, ...],
    reward_mode: str,
    splits: dict[str, dict[str, Any]],
    step: int,
    run_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    evaluation = dict(config.get("evaluation", {}))
    repeats = int(evaluation.get("episodes_per_state", 1))
    save_videos = bool(evaluation.get("save_videos", True))
    max_videos = int(evaluation.get("max_videos_per_split", 3))
    for split_name, split in splits.items():
        appearance = str(split.get("appearance", "none"))
        adapter = _reward_adapter(
            config, goal_tokens, reward_mode, Path.cwd(), robot_state_references
        )
        saved = 0
        for repeat in range(repeats):
            for init_index in split["init_indices"]:
                observation = env.reset(init_index)
                images, state = _policy_inputs(config, observation, appearance)
                if adapter is not None:
                    if extractor is None:
                        raise RuntimeError("MI evaluation requires a visual feature extractor.")
                    tokens = extractor.extract_image_tokens(images[0], env.task_description).float()
                    adapter.reset(tokens, state)
                frames = [images[0]]
                success = False
                mi_return = 0.0
                trend_counts = {"positive": 0, "negative": 0, "unclear": 0}
                for length in range(1, env.max_episode_steps + 1):
                    action = agent.act(images, state, deterministic=True)
                    observation, sparse, success, truncated, _ = env.step(action)
                    images, state = _policy_inputs(config, observation, appearance)
                    if adapter is not None:
                        tokens = extractor.extract_image_tokens(images[0], env.task_description).float()
                        breakdown = adapter.step(tokens, sparse, state)
                        mi_return += breakdown.shaping
                        trend_counts[breakdown.trend_label] += 1
                    if save_videos:
                        frames.append(images[0])
                    if success or truncated:
                        break
                rows.append({
                    "step": step, "split": split_name, "init_state_index": init_index,
                    "repeat": repeat, "success": bool(success), "episode_length": length,
                    "mi_return": mi_return, "shaping_return": mi_return,
                    "trend_counts": trend_counts, "appearance": appearance,
                })
                if save_videos and saved < max_videos:
                    destination = run_dir / "videos" / f"step-{step:08d}" / split_name
                    destination.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(destination / f"init-{init_index:03d}-{'success' if success else 'failure'}.mp4", frames, fps=20)
                    saved += 1
    by_split = {}
    for name, split in splits.items():
        selected = [row for row in rows if row["split"] == name]
        split_trend_counts = {
            label: int(sum(row["trend_counts"][label] for row in selected))
            for label in ("positive", "negative", "unclear")
        }
        by_split[name] = {
            "episodes": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "mean_episode_length": float(np.mean([row["episode_length"] for row in selected])),
            "mean_mi_return": float(np.mean([row["mi_return"] for row in selected])),
            "mean_shaping_return": float(np.mean([row["shaping_return"] for row in selected])),
            "trend_counts": split_trend_counts,
            "split_fingerprint": split_fingerprint(env.suite_name, env.task_id, split["init_indices"], str(split["appearance"])),
        }
    return {"step": step, "reward_mode": reward_mode, "task_id": env.task_id, "splits": by_split, "episodes": rows}


def train(config_path: str | Path, *, reward_mode: str, seed: int, task_id: int, smoke: bool = False) -> Path:
    config, root = _load_config(config_path)
    if smoke:
        config["run_id"] = f"{config.get('run_id', 'libero_mi_rlpd_v1')}_smoke"
        config["benchmark"]["max_episode_steps"] = 4
        for split in config["benchmark"]["evaluation_splits"].values():
            split["init_indices"] = list(split["init_indices"][:1])
        config["training"].update(total_env_steps=6, random_steps=2, batch_size=4, replay_capacity=32, demo_capacity=32, updates_per_step=1, eval_interval=6, checkpoint_interval=6, max_demo_transitions=16)
        config["training"]["max_demos"] = 1
        config["evaluation"].update(save_videos=False, episodes_per_state=1)
        config["training"]["rlpd"].update(num_q_heads=2, critic_subsample_size=2, freeze_backbone=True)
    _seed(seed)
    run_dir = _run_directory(config, root, reward_mode, seed, task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    evaluation_path = run_dir / "evaluation.jsonl"
    if not smoke and (metrics_path.exists() or evaluation_path.exists()) and not (run_dir / "run_report.json").exists():
        raise RuntimeError(f"Partial RLPD run exists at {run_dir}; use a new run_id or archive it before restarting.")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = _environment(config, task_id, seed)
    try:
        train_indices, splits = _validate_splits(config, env)
        uses_mi = reward_mode != "sparse"
        extractor = _make_extractor(config) if uses_mi else None
        if uses_mi:
            goal_tokens, goal_metadata = _goal_tokens(config, env, extractor, root)
            robot_state_references = _robot_state_references(config, env, root)
        else:
            goal_tokens, goal_metadata = None, {"mode": "unused_for_sparse"}
            robot_state_references = ()
        initial = env.reset(train_indices[0])
        images, state = _policy_inputs(config, initial)
        training = dict(config["training"])
        rlpd_config = _rlpd_config(config, root)
        agent = RLPDAgent(images.shape, state.size, env.action_low, env.action_high, rlpd_config, config["runtime"].get("device", "cuda"))
        online = PixelReplayBuffer(int(training.get("replay_capacity", 50_000)), images.shape, state.size, env.action_low.size, seed)
        demos = PixelReplayBuffer(int(training.get("demo_capacity", 20_000)), images.shape, state.size, env.action_low.size, seed + 1)
        demo_report = _fill_demo_buffer(
            config=config, env=env, extractor=extractor, goal_tokens=goal_tokens,
            robot_state_references=robot_state_references,
            reward_mode=reward_mode, root=root, buffer=demos,
            max_demos=training.get("max_demos"), max_transitions=training.get("max_demo_transitions"),
        )
        batch_size = int(training.get("batch_size", 256))
        if demos.size < batch_size - batch_size // 2:
            raise RuntimeError(f"Demo buffer has only {demos.size} transitions for batch_size={batch_size}.")
        ppl_values = dict(training.get("ppl", {}))
        synthetic = None
        ppl_schedule = None
        if bool(ppl_values.get("enabled", False)):
            replay_value = str(ppl_values.get("replay_path", "")).format(task_id=task_id)
            replay_path = Path(replay_value).expanduser()
            if not replay_path.is_absolute():
                replay_path = (root / replay_path).resolve()
            ppl_schedule = PPLConfig(
                max_synthetic_ratio=float(ppl_values.get("max_synthetic_ratio", 0.30)),
                min_synthetic_ratio=float(ppl_values.get("min_synthetic_ratio", 0.05)),
                entropy_reference=float(ppl_values.get("entropy_reference", 4.0)),
                q_shift_tolerance=float(ppl_values.get("q_shift_tolerance", 2.0)),
            )
            synthetic = load_synthetic_replay(
                replay_path,
                image_shape=images.shape,
                state_dim=state.size,
                action_dim=env.action_low.size,
                seed=seed + 2,
            )
        total_steps = int(training.get("total_env_steps", 100_000))
        random_steps = int(training.get("random_steps", 1_000))
        updates_per_step = int(training.get("updates_per_step", 4))
        critic_actor_ratio = int(training.get("critic_actor_ratio", 4))
        eval_interval = int(training.get("eval_interval", 10_000))
        checkpoint_interval = int(training.get("checkpoint_interval", eval_interval))
        rng = np.random.default_rng(seed)
        reward_adapter = _reward_adapter(
            config, goal_tokens, reward_mode, root, robot_state_references
        )
        reward_steps_path = run_dir / "reward_steps.jsonl"
        global_step = 0
        episode = 0
        progress = tqdm(total=total_steps, desc=f"LIBERO RLPD task={task_id} {reward_mode} seed={seed}")
        while global_step < total_steps:
            init_index = int(rng.choice(train_indices))
            observation = env.reset(init_index)
            images, state = _policy_inputs(config, observation)
            if reward_adapter is not None:
                if extractor is None:
                    raise RuntimeError("MI training requires a visual feature extractor.")
                tokens = extractor.extract_image_tokens(images[0], env.task_description).float()
                reward_adapter.reset(tokens, state)
            episode_return = episode_sparse = episode_mi = 0.0
            trend_counts = {"positive": 0, "negative": 0, "unclear": 0}
            reward_step_rows: list[dict[str, Any]] = []
            success = False
            losses: dict[str, float] = {}
            ppl_metrics: dict[str, float | int | bool] = {}
            started = time.monotonic()
            for episode_step in range(1, env.max_episode_steps + 1):
                action = rng.uniform(env.action_low, env.action_high).astype(np.float32) if global_step < random_steps else agent.act(images, state)
                next_observation, sparse, success, truncated, _ = env.step(action)
                next_images, next_state = _policy_inputs(config, next_observation)
                if reward_adapter is None:
                    reward = float(config["reward"].get("sparse_weight", 1.0)) * sparse
                    mi = 0.0
                else:
                    next_tokens = extractor.extract_image_tokens(next_images[0], env.task_description).float()
                    breakdown = reward_adapter.step(next_tokens, sparse, next_state)
                    reward, mi = breakdown.total, breakdown.shaping
                    trend_counts[breakdown.trend_label] += 1
                    if bool(config["reward"].get("log_reward_steps", True)):
                        reward_step_rows.append({
                            "global_step": global_step + 1,
                            "episode": episode + 1,
                            "episode_step": episode_step,
                            "task_id": task_id,
                            "reward_mode": reward_mode,
                            "sparse_reward": breakdown.sparse,
                            "previous_potential": breakdown.previous_potential,
                            "next_potential": breakdown.next_potential,
                            "mi_delta": breakdown.mi_delta,
                            "observed_trend_delta": breakdown.observed_trend_delta,
                            "baseline_trend_delta": breakdown.baseline_trend_delta,
                            "visual_trend_delta": breakdown.visual_trend_delta,
                            "kinematic_progress": breakdown.kinematic_progress,
                            "kinematic_trend_delta": breakdown.kinematic_trend_delta,
                            "directional_consensus": breakdown.directional_consensus,
                            "trend_delta": breakdown.trend_delta,
                            "trend_label": breakdown.trend_label,
                            "trend_reward": breakdown.trend_reward,
                            "success_bonus": breakdown.success_bonus,
                            "shaping_reward": breakdown.shaping,
                            "total_reward": breakdown.total,
                        })
                terminal = bool(success or truncated)
                online.add(images, state, action, reward, next_images, next_state, terminal)
                minimum_online = batch_size // 2
                if global_step >= random_steps and online.size >= minimum_online:
                    for _ in range(updates_per_step):
                        update_actor = (agent.update_step + 1) % critic_actor_ratio == 0
                        update_batch = balanced_batch(online, demos, batch_size)
                        if synthetic is not None and ppl_schedule is not None:
                            probe_size = min(16, online.size)
                            online_probe = online.sample(probe_size)
                            entropy = agent.policy_entropy(online_probe["images"], online_probe["states"])
                            ratio = ppl_schedule.ratio(entropy)
                            candidate_batch, synthetic_count = paced_batch(
                                online, demos, synthetic, batch_size, ratio
                            )
                            accepted = True
                            interval = max(1, int(ppl_values.get("q_check_interval", 100)))
                            q_shift = 0.0
                            if agent.update_step % interval == 0:
                                clear_probe = synthetic.sample_balanced_clear(min(16, synthetic.size))
                                grounded_probe = demos.sample(min(16, demos.size))
                                q_shift = abs(
                                    agent.mean_dataset_q(clear_probe) - agent.mean_dataset_q(grounded_probe)
                                )
                                accepted = q_shift <= ppl_schedule.q_shift_tolerance
                            if accepted:
                                update_batch = candidate_batch
                            ppl_metrics = {
                                "ppl_entropy": entropy,
                                "ppl_synthetic_ratio": ratio,
                                "ppl_synthetic_count": synthetic_count if accepted else 0,
                                "ppl_q_shift": q_shift,
                                "ppl_accepted": accepted,
                            }
                        losses = agent.update(update_batch, update_actor=update_actor)
                images, state = next_images, next_state
                episode_return += reward
                episode_sparse += sparse
                episode_mi += mi
                global_step += 1
                progress.update(1)
                if global_step >= total_steps or terminal:
                    break
            episode += 1
            row = {
                "global_step": global_step, "episode": episode, "task_id": task_id,
                "init_state_index": init_index, "success": bool(success), "episode_length": episode_step,
                "return": episode_return, "sparse_return": episode_sparse, "mi_return": episode_mi,
                "shaping_return": episode_mi, "trend_counts": trend_counts,
                "seconds": time.monotonic() - started, "online_replay": online.size,
                "demo_replay": demos.size, "updates": agent.update_step, **losses,
                **ppl_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if reward_step_rows:
                with reward_steps_path.open("a", encoding="utf-8") as handle:
                    for reward_row in reward_step_rows:
                        handle.write(json.dumps(reward_row) + "\n")
            progress.set_postfix(success=int(success), online=online.size, ret=f"{episode_return:.2f}")
            if global_step % eval_interval < episode_step or global_step >= total_steps:
                report = _evaluate(
                    config=config, env=env, extractor=extractor, agent=agent,
                    goal_tokens=goal_tokens,
                    robot_state_references=robot_state_references,
                    reward_mode=reward_mode, splits=splits, step=global_step,
                    run_dir=run_dir,
                )
                with evaluation_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report) + "\n")
            if global_step % checkpoint_interval < episode_step or global_step >= total_steps:
                agent.save(run_dir / "checkpoints" / "latest.pt", step=global_step, metadata={"task_id": task_id, "reward_mode": reward_mode, "seed": seed})
        progress.close()
        report = {
            "status": "complete", "global_step": global_step, "suite": env.suite_name,
            "task_id": task_id, "reward_mode": reward_mode, "seed": seed,
            "goal": goal_metadata, "demo_buffer": demo_report,
            "synthetic_buffer": None if synthetic is None else {
                "transitions": synthetic.size,
                "pacing": "policy_entropy",
                "max_ratio": ppl_schedule.max_synthetic_ratio if ppl_schedule else None,
                "q_shift_gate": ppl_schedule.q_shift_tolerance if ppl_schedule else None,
            },
            "checkpoint": str(run_dir / "checkpoints" / "latest.pt"),
        }
        (run_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return run_dir
    finally:
        env.close()


def preflight(config_path: str | Path, *, reward_mode: str, seed: int, task_id: int) -> dict[str, Any]:
    config, root = _load_config(config_path)
    env = _environment(config, task_id, seed)
    try:
        train_indices, splits = _validate_splits(config, env)
        uses_mi = reward_mode != "sparse"
        extractor = _make_extractor(config) if uses_mi else None
        if uses_mi:
            goal_tokens, goal = _goal_tokens(config, env, extractor, root)
            robot_state_references = _robot_state_references(config, env, root)
        else:
            goal_tokens, goal = None, {"mode": "unused_for_sparse"}
            robot_state_references = ()
        observation = env.reset(train_indices[0])
        images, state = _policy_inputs(config, observation)
        rlpd_config = _rlpd_config(config, root)
        agent = RLPDAgent(
            images.shape, state.size, env.action_low, env.action_high,
            rlpd_config, config["runtime"].get("device", "cuda"),
        )
        probe_action = agent.act(images, state, deterministic=True)
        ppl_values = dict(config["training"].get("ppl", {}))
        ppl_report: dict[str, Any] = {"enabled": False}
        if bool(ppl_values.get("enabled", False)):
            replay_value = str(ppl_values.get("replay_path", "")).format(task_id=task_id)
            replay_path = Path(replay_value).expanduser()
            if not replay_path.is_absolute():
                replay_path = (root / replay_path).resolve()
            synthetic = load_synthetic_replay(
                replay_path,
                image_shape=images.shape,
                state_dim=state.size,
                action_dim=env.action_low.size,
                seed=seed + 2,
            )
            ppl_report = {
                "enabled": True,
                "replay_path": str(replay_path),
                "transitions": synthetic.size,
                "max_synthetic_ratio": float(ppl_values.get("max_synthetic_ratio", 0.30)),
            }
        potential = None
        adapter = _reward_adapter(
            config, goal_tokens, reward_mode, root, robot_state_references
        )
        if adapter is not None:
            if extractor is None:
                raise RuntimeError("MI preflight requires a visual feature extractor.")
            potential = adapter.reset(
                extractor.extract_image_tokens(images[0], env.task_description).float(),
                state,
            )
        demo = next(iter_demo_episodes(demonstration_path(env, root), max_demos=1))
        return {
            "status": "ready", "algorithm": "visual_rlpd", "reward_backend": "frozen_mi_potential",
            "suite": env.suite_name, "task_id": task_id, "task": env.task_description,
            "policy_image_shape": list(images.shape), "policy_state_dim": int(state.size),
            "action_dim": int(env.action_low.size), "num_q_heads": int(config["training"]["rlpd"].get("num_q_heads", 10)),
            "policy_encoder_checkpoint": rlpd_config.encoder_checkpoint,
            "policy_probe_action_finite": bool(np.isfinite(probe_action).all()),
            "demo_path": str(demonstration_path(env, root)), "demo_transition_count_first_episode": int(len(demo["actions"]) - 1),
            "initial_potential": potential, "goal": goal, "train_indices": list(train_indices),
            "ppl": ppl_report,
            "reward_contract": {
                "mode": reward_mode,
                "history_size": int(config["reward"].get("trend_history_size", 5)),
                "input_interval": int(config["reward"].get("trend_input_interval", 1)),
                "positive": float(config["reward"].get("positive_reward", 1.0)),
                "negative": float(config["reward"].get("negative_reward", -0.2)),
                "unclear": float(config["reward"].get("unclear_reward", 0.0)),
                "success_bonus": float(config["reward"].get("success_bonus", 20.0)),
                "kinematic_enabled": bool(config["reward"].get("kinematic_enabled", False)),
                "kinematic_reference_count": len(robot_state_references),
                "kinematic_positive_threshold": float(
                    config["reward"].get("kinematic_positive_threshold", 0.01)
                ),
                "directional_consensus_steps": int(
                    config["reward"].get("directional_consensus_steps", 1)
                ),
            },
            "evaluation_splits": splits,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("preflight", "train"), default="train")
    parser.add_argument(
        "--reward-mode", choices=tuple(sorted(PotentialReward.MODES)), default="sparse_trend"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(args.config, reward_mode=args.reward_mode, seed=args.seed, task_id=args.task_id), indent=2))
    else:
        print(train(args.config, reward_mode=args.reward_mode, seed=args.seed, task_id=args.task_id, smoke=args.smoke))


if __name__ == "__main__":
    main()
