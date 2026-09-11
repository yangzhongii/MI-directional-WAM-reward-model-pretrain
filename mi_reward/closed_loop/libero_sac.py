"""Train and evaluate a SAC policy with frozen MI potential shaping in LIBERO."""

from __future__ import annotations

import argparse
import csv
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

from mi_reward.closed_loop.libero_env import (
    LiberoTaskEnvironment,
    apply_appearance_shift,
    canonical_camera_image,
    ensure_disjoint_splits,
    proprioception,
    resolve_path,
    split_fingerprint,
)
from mi_reward.closed_loop.online_potential import PotentialReward
from mi_reward.closed_loop.sac import ReplayBuffer, SACAgent, SACConfig
from mi_reward.features.cached_feature_store import make_extractor


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Closed-loop config must be a YAML mapping: {config_path}")
    return config, Path.cwd().resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_extractor(config: dict[str, Any]):
    features = dict(config["features"])
    return make_extractor(
        str(features.get("extractor", "lawam_lam")),
        features.get("dino_checkpoint_root"),
        str(features.get("device", config.get("runtime", {}).get("device", "cuda"))),
        int(features.get("image_size", 224)),
        lam_config_path=features.get("lam_config_path"),
        lam_ckpt_path=features.get("lam_ckpt_path"),
        vision_model_id=features.get("dino_checkpoint_root"),
        strict=bool(features.get("strict", True)),
    )


def _tokens(extractor: Any, observation: dict[str, Any], task: str, appearance: str) -> tuple[torch.Tensor, np.ndarray]:
    image = apply_appearance_shift(canonical_camera_image(observation), appearance)
    tokens = extractor.extract_image_tokens(image, task).float()
    if tokens.ndim != 2:
        raise ValueError(f"Visual extractor must return [K,D] tokens, got {tuple(tokens.shape)}.")
    return tokens, image


def _policy_observation(tokens: torch.Tensor, observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate([tokens.mean(dim=0).numpy(), proprioception(observation)]).astype(np.float32)


def _find_demo_goal_image(env: LiberoTaskEnvironment, demo_index: int, root: Path) -> np.ndarray:
    import h5py
    from libero.libero import get_libero_path

    demonstration = env.suite.get_task_demonstration(env.task_id)
    candidates = [
        Path(get_libero_path("datasets")) / demonstration,
        root / demonstration,
    ]
    dataset_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError(
            f"LIBERO demonstration for task {env.task_id} is missing ({demonstration}). "
            "Download the selected LIBERO suite or set goal.mode to image/null."
        )
    with h5py.File(dataset_path, "r") as handle:
        data = handle["data"]
        demos = sorted(data.keys())
        if not demos:
            raise ValueError(f"No demonstrations found in {dataset_path}.")
        demo = data[demos[int(demo_index) % len(demos)]]
        keys = ("agentview_rgb", "agentview_image", "observation/image")
        for key in keys:
            current: Any = demo.get("obs", demo)
            try:
                for segment in key.split("/"):
                    current = current[segment]
                frame = np.asarray(current[-1])
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    return np.ascontiguousarray(frame)
            except (KeyError, TypeError):
                continue
    raise KeyError(f"No terminal agent-view RGB stream found in {dataset_path}.")


def _goal_tokens(
    config: dict[str, Any], env: LiberoTaskEnvironment, extractor: Any, root: Path
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    goal = dict(config.get("goal", {}))
    mode = str(goal.get("mode", "null"))
    if mode == "null":
        return None, {"mode": "null", "warning": "uses checkpoint null-goal token"}
    if mode == "libero_demo_terminal":
        image = _find_demo_goal_image(env, int(goal.get("demo_index", 0)), root)
        source = f"libero_demo:{env.suite.get_task_demonstration(env.task_id)}"
    elif mode == "image":
        raw = goal.get("image")
        by_task = goal.get("images_by_task", {}) or {}
        raw = by_task.get(str(env.task_id), by_task.get(env.task_id, raw))
        if not raw:
            raise ValueError("goal.mode=image requires goal.image or goal.images_by_task[task_id].")
        path = resolve_path(str(raw), root)
        if not path.is_file():
            raise FileNotFoundError(f"Goal image does not exist: {path}")
        image = imageio.imread(path)
        source = str(path)
    else:
        raise ValueError(f"Unknown goal.mode={mode!r}.")
    tokens = extractor.extract_image_tokens(np.asarray(image), env.task_description).float()
    return tokens, {"mode": mode, "source": source, "token_shape": list(tokens.shape)}


def _robot_state_references(
    config: dict[str, Any], env: LiberoTaskEnvironment, root: Path
) -> tuple[np.ndarray, ...]:
    """Load successful robot-state paths used by online stage alignment."""

    reward = dict(config.get("reward", {}))
    if not bool(reward.get("kinematic_enabled", False)):
        return ()
    from mi_reward.closed_loop.libero_demo import demonstration_path, iter_demo_episodes

    max_demos = int(reward.get("kinematic_reference_demos", 10))
    references = tuple(
        np.asarray(episode["states"], dtype=np.float32)
        for episode in iter_demo_episodes(demonstration_path(env, root), max_demos=max_demos)
    )
    if not references:
        raise ValueError("kinematic_enabled=true but no successful LIBERO state references were loaded.")
    return references


def _reward_adapter(
    config: dict[str, Any], goal_tokens: torch.Tensor | None, reward_mode: str, root: Path,
    robot_state_references: tuple[np.ndarray, ...] | None = None,
) -> PotentialReward | None:
    if reward_mode == "sparse":
        return None
    reward = dict(config["reward"])
    if bool(reward.get("kinematic_enabled", False)) and not robot_state_references:
        raise ValueError(
            "kinematic_enabled=true requires successful robot-state references; "
            "load them with _robot_state_references()."
        )
    checkpoint = resolve_path(str(reward["checkpoint"]), root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MI reward checkpoint is missing: {checkpoint}")
    return PotentialReward(
        str(checkpoint),
        goal_tokens=goal_tokens,
        mode=reward_mode,
        device=str(config.get("runtime", {}).get("device", "cuda")),
        gamma=reward.get("gamma"),
        mi_weight=float(reward.get("mi_weight", 1.0)),
        sparse_weight=float(reward.get("sparse_weight", 1.0)),
        clip_delta=reward.get("clip_delta", 1.0),
        max_history=int(reward.get("max_history", 64)),
        trend_history_size=int(reward.get("trend_history_size", 5)),
        trend_input_interval=int(reward.get("trend_input_interval", 1)),
        trend_baseline_horizon=int(reward.get("trend_baseline_horizon", 16)),
        trend_direction=float(reward.get("trend_direction", 1.0)),
        positive_threshold=float(reward.get("positive_threshold", 0.02)),
        negative_threshold=float(reward.get("negative_threshold", -0.02)),
        positive_reward=float(reward.get("positive_reward", 1.0)),
        negative_reward=float(reward.get("negative_reward", -0.2)),
        unclear_reward=float(reward.get("unclear_reward", 0.0)),
        trend_weight=float(reward.get("trend_weight", 1.0)),
        success_bonus=float(reward.get("success_bonus", 20.0)),
        success_threshold=float(reward.get("success_threshold", 0.5)),
        robot_state_references=robot_state_references,
        kinematic_max_stage_step=int(reward.get("kinematic_max_stage_step", 8)),
        kinematic_initial_match_fraction=float(
            reward.get("kinematic_initial_match_fraction", 0.2)
        ),
        kinematic_advance_margin=float(reward.get("kinematic_advance_margin", 1e-3)),
        kinematic_positive_threshold=float(
            reward.get("kinematic_positive_threshold", 0.01)
        ),
        directional_consensus_steps=int(reward.get("directional_consensus_steps", 1)),
        kinematic_state_weights=reward.get("kinematic_state_weights"),
    )


def _run_directory(config: dict[str, Any], root: Path, reward_mode: str, seed: int, task_id: int) -> Path:
    output_root = resolve_path(str(config.get("output_root", "logs/mi_reward/libero_closed_loop")), root)
    return output_root / str(config.get("run_id", "libero_mi_sac_v1")) / f"task-{task_id:02d}" / reward_mode / f"seed-{seed}"


def _make_env(config: dict[str, Any], task_id: int, seed: int) -> LiberoTaskEnvironment:
    benchmark = dict(config["benchmark"])
    return LiberoTaskEnvironment(
        str(benchmark.get("suite", "libero_spatial")),
        task_id,
        benchmark_variant=str(benchmark.get("variant", "libero")),
        resolution=int(benchmark.get("resolution", 256)),
        seed=seed,
        settle_steps=int(benchmark.get("settle_steps", 10)),
        max_episode_steps=benchmark.get("max_episode_steps"),
    )


def _validate_splits(config: dict[str, Any], env: LiberoTaskEnvironment) -> tuple[tuple[int, ...], dict[str, dict[str, Any]]]:
    benchmark = dict(config["benchmark"])
    train_indices = env.resolve_init_indices(benchmark["train_init_indices"])
    evaluation = dict(benchmark.get("evaluation_splits", {}))
    if not evaluation:
        raise ValueError("benchmark.evaluation_splits cannot be empty.")
    resolved: dict[str, dict[str, Any]] = {}
    for name, raw in evaluation.items():
        item = dict(raw or {})
        item["init_indices"] = env.resolve_init_indices(item["init_indices"])
        item["appearance"] = str(item.get("appearance", "none"))
        resolved[str(name)] = item
    # Evaluation splits may intentionally share states to isolate appearance;
    # none may leak a training initial state.
    for name, item in resolved.items():
        ensure_disjoint_splits({"train": train_indices, name: item["init_indices"]})
    return train_indices, resolved


def _sac_config(raw: dict[str, Any]) -> SACConfig:
    allowed = set(SACConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed}
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(int(value) for value in values["hidden_dims"])
    return SACConfig(**values)


def evaluate(
    *,
    config: dict[str, Any],
    env: LiberoTaskEnvironment,
    extractor: Any,
    agent: SACAgent,
    goal_tokens: torch.Tensor | None,
    reward_mode: str,
    evaluation_splits: dict[str, dict[str, Any]],
    step: int,
    run_dir: Path,
) -> dict[str, Any]:
    evaluation_cfg = dict(config.get("evaluation", {}))
    episodes_per_state = int(evaluation_cfg.get("episodes_per_state", 1))
    save_videos = bool(evaluation_cfg.get("save_videos", True))
    max_videos = int(evaluation_cfg.get("max_videos_per_split", 3))
    rows: list[dict[str, Any]] = []
    for split_name, split in evaluation_splits.items():
        appearance = str(split["appearance"])
        reward_adapter = _reward_adapter(config, goal_tokens, reward_mode, Path.cwd())
        for repeat in range(episodes_per_state):
            for init_index in split["init_indices"]:
                observation = env.reset(init_index)
                tokens, image = _tokens(extractor, observation, env.task_description, appearance)
                policy_obs = _policy_observation(tokens, observation)
                if reward_adapter is not None:
                    reward_adapter.reset(tokens)
                frames = [image]
                episode_mi = 0.0
                success = False
                length = 0
                for length in range(1, env.max_episode_steps + 1):
                    action = agent.act(policy_obs, deterministic=True)
                    next_observation, sparse, success, truncated, _ = env.step(action)
                    next_tokens, next_image = _tokens(extractor, next_observation, env.task_description, appearance)
                    if reward_adapter is not None:
                        episode_mi += reward_adapter.step(next_tokens, sparse).shaping
                    policy_obs = _policy_observation(next_tokens, next_observation)
                    if save_videos and len(frames) < env.max_episode_steps + 1:
                        frames.append(next_image)
                    if success or truncated:
                        break
                row = {
                    "step": step, "split": split_name, "init_state_index": init_index,
                    "repeat": repeat, "success": bool(success), "episode_length": length,
                    "mi_return": episode_mi, "appearance": appearance,
                }
                rows.append(row)
                split_rows = [item for item in rows if item["split"] == split_name]
                if save_videos and len(split_rows) <= max_videos:
                    video_dir = run_dir / "videos" / f"step-{step:08d}" / split_name
                    video_dir.mkdir(parents=True, exist_ok=True)
                    suffix = "success" if success else "failure"
                    imageio.mimwrite(video_dir / f"init-{init_index:03d}-{suffix}.mp4", frames, fps=20)
    by_split: dict[str, dict[str, Any]] = {}
    for name in evaluation_splits:
        selected = [row for row in rows if row["split"] == name]
        by_split[name] = {
            "episodes": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "mean_episode_length": float(np.mean([row["episode_length"] for row in selected])),
            "mean_mi_return": float(np.mean([row["mi_return"] for row in selected])),
            "split_fingerprint": split_fingerprint(env.suite_name, env.task_id, evaluation_splits[name]["init_indices"], str(evaluation_splits[name]["appearance"])),
        }
    return {"step": step, "reward_mode": reward_mode, "task_id": env.task_id, "splits": by_split, "episodes": rows}


def train(
    config_path: str | Path,
    *,
    reward_mode: str,
    seed: int,
    task_id: int,
    resume: bool = False,
    smoke: bool = False,
) -> Path:
    config, root = _load_config(config_path)
    if smoke:
        config["run_id"] = f"{config.get('run_id', 'libero_mi_sac_v1')}_smoke"
        config["benchmark"]["max_episode_steps"] = 4
        for split in config["benchmark"]["evaluation_splits"].values():
            split["init_indices"] = list(split["init_indices"][:1])
        config["training"].update({
            "total_env_steps": 6,
            "random_steps": 2,
            "batch_size": 2,
            "replay_capacity": 32,
            "eval_interval": 6,
            "checkpoint_interval": 6,
        })
        config["evaluation"].update({"save_videos": False, "episodes_per_state": 1})
    _seed_everything(seed)
    run_dir = _run_directory(config, root, reward_mode, seed, task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = _make_env(config, task_id, seed)
    try:
        train_indices, evaluation_splits = _validate_splits(config, env)
        extractor = _make_extractor(config)
        goal_tokens, goal_metadata = _goal_tokens(config, env, extractor, root)
        rng = np.random.default_rng(seed)
        initial_observation = env.reset(int(rng.choice(train_indices)))
        initial_tokens, _ = _tokens(extractor, initial_observation, env.task_description, "none")
        observation_dim = int(initial_tokens.shape[-1] + proprioception(initial_observation).size)
        training = dict(config["training"])
        sac_cfg = _sac_config(dict(training.get("sac", {})))
        latest = run_dir / "checkpoints" / "latest.pt"
        if resume and latest.is_file():
            agent, payload = SACAgent.load(latest, device=str(config.get("runtime", {}).get("device", "cuda")))
            start_step = int(payload["step"])
            if agent.observation_dim != observation_dim:
                raise ValueError("Resume checkpoint observation dimension does not match the configured encoder.")
        else:
            agent = SACAgent(observation_dim, env.action_low, env.action_high, sac_cfg, str(config.get("runtime", {}).get("device", "cuda")))
            start_step = 0
        replay = ReplayBuffer(int(training.get("replay_capacity", 200_000)), observation_dim, env.action_low.size, seed)
        total_steps = int(training.get("total_env_steps", 100_000))
        random_steps = int(training.get("random_steps", 5_000))
        batch_size = int(training.get("batch_size", 256))
        updates_per_step = int(training.get("updates_per_step", 1))
        eval_interval = int(training.get("eval_interval", 10_000))
        checkpoint_interval = int(training.get("checkpoint_interval", eval_interval))
        reward_adapter = _reward_adapter(config, goal_tokens, reward_mode, root)
        metrics_path = run_dir / "train_metrics.jsonl"
        eval_path = run_dir / "evaluation.jsonl"
        episode_index = 0
        global_step = start_step
        progress = tqdm(total=total_steps, initial=start_step, desc=f"LIBERO task={task_id} {reward_mode} seed={seed}")
        next_eval = ((start_step // eval_interval) + 1) * eval_interval
        while global_step < total_steps:
            init_index = int(rng.choice(train_indices))
            observation = env.reset(init_index)
            tokens, _ = _tokens(extractor, observation, env.task_description, "none")
            policy_obs = _policy_observation(tokens, observation)
            if reward_adapter is not None:
                reward_adapter.reset(tokens)
            episode_sparse = 0.0
            episode_mi = 0.0
            episode_return = 0.0
            success = False
            last_losses: dict[str, float] = {}
            episode_start = time.monotonic()
            for episode_step in range(1, env.max_episode_steps + 1):
                if global_step < random_steps:
                    action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
                else:
                    action = agent.act(policy_obs)
                next_observation, sparse, success, truncated, _ = env.step(action)
                next_tokens, _ = _tokens(extractor, next_observation, env.task_description, "none")
                next_policy_obs = _policy_observation(next_tokens, next_observation)
                if reward_adapter is None:
                    total_reward = float(config["reward"].get("sparse_weight", 1.0)) * sparse
                    mi_delta = 0.0
                else:
                    breakdown = reward_adapter.step(next_tokens, sparse)
                    total_reward = breakdown.total
                    mi_delta = breakdown.shaping
                terminal = bool(success or truncated)
                replay.add(policy_obs, action, total_reward, next_policy_obs, terminal)
                if global_step >= random_steps and replay.size >= batch_size:
                    for _ in range(updates_per_step):
                        last_losses = agent.update(replay.sample(batch_size, agent.device))
                policy_obs = next_policy_obs
                episode_sparse += sparse
                episode_mi += mi_delta
                episode_return += total_reward
                global_step += 1
                progress.update(1)
                if global_step >= total_steps or terminal:
                    break
            episode_index += 1
            train_row = {
                "global_step": global_step, "episode": episode_index, "task_id": task_id,
                "init_state_index": init_index, "success": bool(success), "episode_length": episode_step,
                "return": episode_return, "sparse_return": episode_sparse, "mi_return": episode_mi,
                "seconds": time.monotonic() - episode_start, **last_losses,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(train_row) + "\n")
            progress.set_postfix(success=int(success), replay=replay.size, ret=f"{episode_return:.2f}")
            if global_step >= next_eval or global_step >= total_steps:
                report = evaluate(
                    config=config, env=env, extractor=extractor, agent=agent, goal_tokens=goal_tokens,
                    reward_mode=reward_mode, evaluation_splits=evaluation_splits,
                    step=global_step, run_dir=run_dir,
                )
                with eval_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report) + "\n")
                next_eval += eval_interval
            if global_step % checkpoint_interval < episode_step or global_step >= total_steps:
                agent.save(latest, step=global_step, metadata={
                    "suite": env.suite_name, "task_id": task_id, "reward_mode": reward_mode,
                    "seed": seed, "goal": goal_metadata,
                })
        progress.close()
        final_report = {
            "status": "complete", "global_step": global_step, "suite": env.suite_name,
            "task_id": task_id, "reward_mode": reward_mode, "seed": seed,
            "goal": goal_metadata, "checkpoint": str(latest),
            "train_split": list(train_indices), "evaluation_splits": evaluation_splits,
        }
        (run_dir / "run_report.json").write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
        return run_dir
    finally:
        env.close()


def preflight(config_path: str | Path, *, reward_mode: str, seed: int, task_id: int) -> dict[str, Any]:
    config, root = _load_config(config_path)
    env = _make_env(config, task_id, seed)
    try:
        train_indices, eval_splits = _validate_splits(config, env)
        extractor = _make_extractor(config)
        goal_tokens, goal_metadata = _goal_tokens(config, env, extractor, root)
        observation = env.reset(train_indices[0])
        tokens, _ = _tokens(extractor, observation, env.task_description, "none")
        adapter = _reward_adapter(config, goal_tokens, reward_mode, root)
        potential = None if adapter is None else adapter.reset(tokens)
        return {
            "status": "ready", "suite": env.suite_name, "task_id": task_id,
            "task": env.task_description, "initial_states": len(env.initial_states),
            "train_indices": list(train_indices), "evaluation_splits": eval_splits,
            "visual_token_shape": list(tokens.shape), "policy_observation_dim": int(tokens.shape[-1] + proprioception(observation).size),
            "action_dim": int(env.action_low.size), "reward_mode": reward_mode,
            "initial_potential": potential, "goal": goal_metadata,
        }
    finally:
        env.close()


def summarize(config_path: str | Path) -> Path:
    config, root = _load_config(config_path)
    output_root = resolve_path(str(config.get("output_root", "logs/mi_reward/libero_closed_loop")), root) / str(config.get("run_id", "libero_mi_sac_v1"))
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("task-*/*/seed-*/evaluation.jsonl")):
        reports = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not reports:
            continue
        final = reports[-1]
        seed_dir = path.parent
        for split, metrics in final["splits"].items():
            rows.append({
                "task_id": int(seed_dir.parents[1].name.removeprefix("task-")),
                "reward_mode": seed_dir.parent.name,
                "seed": int(seed_dir.name.removeprefix("seed-")),
                "step": final["step"], "split": split, **metrics,
            })
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    fields = ["task_id", "reward_mode", "seed", "step", "split", "episodes", "success_rate", "mean_episode_length", "mean_mi_return", "split_fingerprint"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate: dict[str, Any] = {}
    for mode in sorted({row["reward_mode"] for row in rows}):
        for split in sorted({row["split"] for row in rows}):
            selected = [row for row in rows if row["reward_mode"] == mode and row["split"] == split]
            if selected:
                values = [float(row["success_rate"]) for row in selected]
                aggregate[f"{mode}/{split}"] = {"runs": len(values), "mean_success_rate": float(np.mean(values)), "std_success_rate": float(np.std(values))}
    (output_root / "summary.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("preflight", "train", "summarize"), default="train")
    parser.add_argument("--reward-mode", choices=sorted(PotentialReward.MODES), default="sparse_mi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a six-step end-to-end training/evaluation check.")
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(args.config, reward_mode=args.reward_mode, seed=args.seed, task_id=args.task_id), indent=2))
    elif args.action == "summarize":
        print(summarize(args.config))
    else:
        print(
            train(
                args.config,
                reward_mode=args.reward_mode,
                seed=args.seed,
                task_id=args.task_id,
                resume=args.resume,
                smoke=args.smoke,
            )
        )


if __name__ == "__main__":
    main()
