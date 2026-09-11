"""Audit categorical MI rewards before launching a LIBERO RLPD run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

from mi_reward.closed_loop.libero_demo import demonstration_path, iter_demo_episodes
from mi_reward.closed_loop.libero_rlpd import _environment, _load_config, _policy_inputs
from mi_reward.closed_loop.libero_sac import (
    _goal_tokens,
    _make_extractor,
    _reward_adapter,
    _robot_state_references,
    _validate_splits,
)


LABELS = ("positive", "negative", "unclear")


def _episode_summary(rows: list[dict[str, Any]], source: str, episode: int) -> dict[str, Any]:
    counts = {label: sum(row["trend_label"] == label for row in rows) for label in LABELS}
    steps = len(rows)
    return {
        "source": source,
        "episode": episode,
        "steps": steps,
        "trend_counts": counts,
        "trend_fractions": {
            label: float(counts[label] / steps) if steps else 0.0 for label in LABELS
        },
        "trend_return": float(sum(row["trend_reward"] for row in rows)),
        "mean_trend_reward": float(np.mean([row["trend_reward"] for row in rows])) if rows else 0.0,
        "mean_corrected_delta": float(np.mean([row["trend_delta"] for row in rows])) if rows else 0.0,
        "success_bonus": float(sum(row["success_bonus"] for row in rows)),
    }


def _group_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(item["steps"]) for item in episodes)
    counts = {
        label: sum(int(item["trend_counts"][label]) for item in episodes) for label in LABELS
    }
    return {
        "episodes": len(episodes),
        "steps": total_steps,
        "trend_counts": counts,
        "trend_fractions": {
            label: float(counts[label] / total_steps) if total_steps else 0.0 for label in LABELS
        },
        "mean_episode_trend_return": float(
            np.mean([item["trend_return"] for item in episodes])
        ) if episodes else 0.0,
        "mean_step_trend_reward": float(
            np.mean([item["mean_trend_reward"] for item in episodes])
        ) if episodes else 0.0,
        "episode_details": episodes,
    }


def _auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.5
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return float(wins / (len(positive) * len(negative)))


def _row(breakdown: Any, step: int) -> dict[str, Any]:
    return {
        "step": step,
        "previous_potential": breakdown.previous_potential,
        "next_potential": breakdown.next_potential,
        "observed_trend_delta": breakdown.observed_trend_delta,
        "baseline_trend_delta": breakdown.baseline_trend_delta,
        "visual_trend_delta": breakdown.visual_trend_delta,
        "kinematic_progress": breakdown.kinematic_progress,
        "kinematic_trend_delta": breakdown.kinematic_trend_delta,
        "directional_consensus": breakdown.directional_consensus,
        "trend_delta": breakdown.trend_delta,
        "trend_label": breakdown.trend_label,
        "trend_reward": breakdown.weighted_trend,
        "success_bonus": breakdown.success_bonus,
    }


def _with_identity(rows: list[dict[str, Any]], source: str, episode: int) -> list[dict[str, Any]]:
    return [{"source": source, "episode": episode, **row} for row in rows]


def _threshold_metrics(
    samples: list[dict[str, Any]],
    positive_threshold: float,
    negative_threshold: float,
    positive_reward: float = 1.0,
    negative_reward: float = -0.2,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    episode_scores: dict[str, dict[int, list[float]]] = {}
    for source in ("success_demo", "random", "static"):
        selected = [row for row in samples if row["source"] == source]
        counts = {label: 0 for label in LABELS}
        episode_scores[source] = {}
        for row in selected:
            delta = float(row["trend_delta"])
            if delta >= positive_threshold:
                label, reward = "positive", positive_reward
            elif delta <= negative_threshold:
                label, reward = "negative", negative_reward
            else:
                label, reward = "unclear", 0.0
            counts[label] += 1
            episode_scores[source].setdefault(int(row["episode"]), []).append(float(reward))
        steps = len(selected)
        means = [float(np.mean(values)) for values in episode_scores[source].values()]
        groups[source] = {
            "episodes": len(means),
            "steps": steps,
            "trend_counts": counts,
            "trend_fractions": {
                label: float(counts[label] / steps) if steps else 0.0 for label in LABELS
            },
            "mean_step_trend_reward": float(np.mean(means)) if means else 0.0,
            "episode_mean_rewards": means,
        }
    auc = _auc(
        groups["success_demo"]["episode_mean_rewards"],
        groups["random"]["episode_mean_rewards"],
    )
    groups["success_random_auc"] = auc  # type: ignore[assignment]
    return groups


def _strict_checks(metrics: dict[str, Any], audit_cfg: dict[str, Any]) -> dict[str, bool]:
    success = metrics["success_demo"]
    random = metrics["random"]
    static = metrics["static"]
    return {
        "static_is_unclear": static["trend_fractions"]["unclear"]
        >= float(audit_cfg.get("min_static_unclear_fraction", 0.95)),
        "random_positive_controlled": random["trend_fractions"]["positive"]
        <= float(audit_cfg.get("max_random_positive_fraction", 0.2)),
        "success_positive_gap": (
            success["trend_fractions"]["positive"]
            - random["trend_fractions"]["positive"]
        ) >= float(audit_cfg.get("min_success_positive_gap", 0.15)),
        "success_beats_random_auc": float(metrics["success_random_auc"])
        >= float(audit_cfg.get("min_success_random_auc", 0.7)),
        "success_mean_exceeds_random": success["mean_step_trend_reward"]
        > random["mean_step_trend_reward"],
    }


def _split_samples(
    samples: list[dict[str, Any]], calibration_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration_ids: dict[str, set[int]] = {}
    validation_ids: dict[str, set[int]] = {}
    for source in ("success_demo", "random", "static"):
        episodes = sorted({int(row["episode"]) for row in samples if row["source"] == source})
        if len(episodes) < 2:
            raise ValueError(
                f"Threshold calibration requires at least two {source} episodes; got {len(episodes)}."
            )
        count = min(len(episodes) - 1, max(1, int(round(len(episodes) * calibration_fraction))))
        calibration_ids[source] = set(episodes[:count])
        validation_ids[source] = set(episodes[count:])
    calibration = [
        row for row in samples if int(row["episode"]) in calibration_ids[str(row["source"])]
    ]
    validation = [
        row for row in samples if int(row["episode"]) in validation_ids[str(row["source"])]
    ]
    split = {
        "calibration_episode_ids": {key: sorted(value) for key, value in calibration_ids.items()},
        "validation_episode_ids": {key: sorted(value) for key, value in validation_ids.items()},
    }
    return calibration, validation, split


def _calibrate_thresholds(
    samples: list[dict[str, Any]], audit_cfg: dict[str, Any]
) -> dict[str, Any]:
    calibration, validation, split = _split_samples(
        samples, float(audit_cfg.get("calibration_fraction", 0.4))
    )
    random_deltas = np.asarray(
        [float(row["trend_delta"]) for row in calibration if row["source"] == "random"],
        dtype=np.float64,
    )
    min_threshold = float(audit_cfg.get("min_positive_threshold", 0.005))
    quantiles = np.linspace(0.50, 0.995, 100)
    candidates = sorted({
        max(min_threshold, float(np.quantile(random_deltas, quantile))) for quantile in quantiles
    })
    evaluated: list[dict[str, Any]] = []
    for positive_threshold in candidates:
        negative_threshold = -positive_threshold
        metrics = _threshold_metrics(calibration, positive_threshold, negative_threshold)
        checks = _strict_checks(metrics, audit_cfg)
        success_fraction = metrics["success_demo"]["trend_fractions"]["positive"]
        random_fraction = metrics["random"]["trend_fractions"]["positive"]
        evaluated.append({
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
            "metrics": metrics,
            "checks": checks,
            "accepted": all(checks.values()),
            "selection_key": (
                float(metrics["success_random_auc"]),
                float(success_fraction - random_fraction),
                float(success_fraction),
                -positive_threshold,
            ),
        })
    accepted = [item for item in evaluated if item["accepted"]]
    pool = accepted or evaluated
    selected = max(pool, key=lambda item: item["selection_key"])
    validation_metrics = _threshold_metrics(
        validation, selected["positive_threshold"], selected["negative_threshold"]
    )
    validation_checks = _strict_checks(validation_metrics, audit_cfg)
    status = "passed" if selected["accepted"] and all(validation_checks.values()) else "failed"
    return {
        "status": status,
        "positive_threshold": selected["positive_threshold"],
        "negative_threshold": selected["negative_threshold"],
        "split": split,
        "calibration_metrics": selected["metrics"],
        "calibration_checks": selected["checks"],
        "validation_metrics": validation_metrics,
        "validation_checks": validation_checks,
        "accepted_candidate_count": len(accepted),
        "candidate_count": len(evaluated),
    }


def _write_pilot_config(
    config: dict[str, Any], config_path: Path, calibration: dict[str, Any],
    audit_cfg: dict[str, Any], root: Path, task_id: int,
) -> Path:
    raw = str(
        audit_cfg.get(
            "calibrated_config",
            "mi_reward/configs/libero_rlpd_task0_pilot_calibrated.yaml",
        )
    )
    output = Path(raw).expanduser()
    if not output.is_absolute():
        output = root / output
    if output.resolve() == config_path.resolve():
        raise ValueError("reward_audit.calibrated_config must not overwrite the source config.")
    payload = copy.deepcopy(config)
    payload["run_id"] = f"{config.get('run_id', 'libero_mi_rlpd')}_task{task_id}_pilot"
    payload["benchmark"]["task_ids"] = [task_id]
    payload["reward"]["positive_threshold"] = float(calibration["positive_threshold"])
    payload["reward"]["negative_threshold"] = float(calibration["negative_threshold"])
    payload["reward"]["modes"] = ["sparse", "sparse_trend"]
    payload["reward"]["calibration"] = {
        "source_config": str(config_path),
        "task_id": task_id,
        "validation_status": calibration["status"],
    }
    payload["training"]["seeds"] = [0]
    payload["training"]["total_env_steps"] = int(audit_cfg.get("pilot_env_steps", 10_000))
    payload["training"]["eval_interval"] = int(audit_cfg.get("pilot_eval_interval", 5_000))
    payload["training"]["checkpoint_interval"] = int(
        audit_cfg.get("pilot_checkpoint_interval", 5_000)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def run(
    config_path: str | Path,
    *,
    task_id: int,
    max_demos: int | None = None,
    random_episodes: int | None = None,
    random_max_steps: int | None = None,
    static_steps: int | None = None,
    seed: int = 0,
    calibrate: bool = False,
) -> Path:
    source_config_path = Path(config_path).expanduser().resolve()
    config, root = _load_config(source_config_path)
    audit_cfg = dict(config.get("reward_audit", {}))
    max_demos = int(max_demos if max_demos is not None else audit_cfg.get("max_demos", 10))
    random_episodes = int(
        random_episodes if random_episodes is not None else audit_cfg.get("random_episodes", 10)
    )
    random_max_steps = int(
        random_max_steps if random_max_steps is not None else audit_cfg.get("random_max_steps", 100)
    )
    static_steps = int(static_steps if static_steps is not None else audit_cfg.get("static_steps", 16))
    if min(max_demos, random_episodes, random_max_steps, static_steps) < 1:
        raise ValueError("Reward audit counts must all be positive.")

    output = Path(config.get("output_root", "logs/mi_reward/libero_rlpd_closed_loop"))
    if not output.is_absolute():
        output = root / output
    destination = output / str(config.get("run_id", "libero_mi_rlpd")) / "reward_audit" / f"task-{task_id:02d}"
    destination.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    env = _environment(config, task_id, seed)
    try:
        train_indices, _ = _validate_splits(config, env)
        extractor = _make_extractor(config)
        goal_tokens, goal = _goal_tokens(config, env, extractor, root)
        robot_state_references = _robot_state_references(config, env, root)
        demos = list(iter_demo_episodes(demonstration_path(env, root), max_demos=max_demos))
        success_adapter = _reward_adapter(
            config, goal_tokens, "sparse_trend", root, robot_state_references
        )
        static_adapter = _reward_adapter(
            config, goal_tokens, "trend", root, robot_state_references
        )
        random_adapter = _reward_adapter(
            config, goal_tokens, "trend", root, robot_state_references
        )
        assert success_adapter is not None and static_adapter is not None and random_adapter is not None

        success_episodes: list[dict[str, Any]] = []
        static_episodes: list[dict[str, Any]] = []
        all_samples: list[dict[str, Any]] = []
        for episode_index, episode in enumerate(
            tqdm(demos, desc=f"Reward audit success/static task={task_id}")
        ):
            initial_tokens = extractor.extract_image_tokens(
                np.asarray(episode["main"][0]), env.task_description
            ).float()

            initial_state = np.asarray(episode["states"][0], dtype=np.float32)
            success_adapter.reset(initial_tokens, initial_state)
            success_rows: list[dict[str, Any]] = []
            length = min(len(episode["actions"]) - 1, len(episode["main"]) - 1)
            for index in range(length):
                tokens = extractor.extract_image_tokens(
                    np.asarray(episode["main"][index + 1]), env.task_description
                ).float()
                sparse = 1.0 if index == length - 1 else 0.0
                success_rows.append(_row(success_adapter.step(
                    tokens,
                    sparse,
                    np.asarray(episode["states"][index + 1], dtype=np.float32),
                ), index + 1))
            all_samples.extend(_with_identity(success_rows, "success_demo", episode_index))
            success_episodes.append(_episode_summary(success_rows, "success_demo", episode_index))

            static_adapter.reset(initial_tokens, initial_state)
            static_rows = [
                _row(static_adapter.step(initial_tokens, 0.0, initial_state), step)
                for step in range(1, static_steps + 1)
            ]
            all_samples.extend(_with_identity(static_rows, "static", episode_index))
            static_episodes.append(_episode_summary(static_rows, "static", episode_index))

        random_results: list[dict[str, Any]] = []
        for episode_index in tqdm(range(random_episodes), desc=f"Reward audit random task={task_id}"):
            observation = env.reset(int(rng.choice(train_indices)))
            images, state = _policy_inputs(config, observation)
            tokens = extractor.extract_image_tokens(images[0], env.task_description).float()
            random_adapter.reset(tokens, state)
            rows: list[dict[str, Any]] = []
            for step in range(1, min(random_max_steps, env.max_episode_steps) + 1):
                action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
                observation, sparse, success, truncated, _ = env.step(action)
                images, state = _policy_inputs(config, observation)
                tokens = extractor.extract_image_tokens(images[0], env.task_description).float()
                rows.append(_row(random_adapter.step(tokens, sparse, state), step))
                if success or truncated:
                    break
            all_samples.extend(_with_identity(rows, "random", episode_index))
            random_results.append(_episode_summary(rows, "random", episode_index))

        samples_path = destination / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for sample in all_samples:
                handle.write(json.dumps(sample) + "\n")

        groups = {
            "success_demo": _group_summary(success_episodes),
            "random": _group_summary(random_results),
            "static": _group_summary(static_episodes),
        }
        auc = _auc(
            [item["mean_trend_reward"] for item in success_episodes],
            [item["mean_trend_reward"] for item in random_results],
        )
        current_metrics = _threshold_metrics(
            all_samples,
            float(config["reward"].get("positive_threshold", 0.02)),
            float(config["reward"].get("negative_threshold", -0.02)),
        )
        checks = _strict_checks(current_metrics, audit_cfg)
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "task_id": task_id,
            "task": env.task_description,
            "goal": goal,
            "reward_contract": {
                key: config["reward"].get(key)
                for key in (
                    "trend_history_size", "positive_threshold", "negative_threshold",
                    "trend_direction", "positive_reward", "negative_reward",
                    "unclear_reward", "success_bonus", "kinematic_enabled",
                    "kinematic_reference_demos", "kinematic_max_stage_step",
                    "kinematic_positive_threshold", "directional_consensus_steps",
                )
            },
            "checks": checks,
            "thresholds": {
                key: audit_cfg.get(key)
                for key in (
                    "min_static_unclear_fraction", "max_random_positive_fraction",
                    "min_success_positive_gap", "min_success_random_auc",
                )
            },
            "success_random_auc": auc,
            "groups": groups,
            "samples": str(samples_path),
        }
        if calibrate:
            calibration = _calibrate_thresholds(all_samples, audit_cfg)
            calibration_path = destination / "calibration_report.json"
            if calibration["status"] == "passed":
                pilot_config = _write_pilot_config(
                    config, source_config_path, calibration, audit_cfg, root, task_id
                )
                calibration["pilot_config"] = str(pilot_config)
            calibration_path.write_text(
                json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
            )
            report["precalibration_status"] = report["status"]
            report["status"] = calibration["status"]
            report["calibration"] = calibration
            report["calibration_report"] = str(calibration_path)
        output_path = destination / "report.json"
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"], "report": str(output_path),
            "checks": checks, "success_random_auc": auc,
            "fractions": {
                name: value["trend_fractions"] for name, value in groups.items()
            },
            "calibration": report.get("calibration"),
        }, indent=2))
        return output_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--random-episodes", type=int)
    parser.add_argument("--random-max-steps", type=int)
    parser.add_argument("--static-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate thresholds on isolated episodes and emit a 10k-step pilot config.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_demos = 1
        args.random_episodes = 1
        args.random_max_steps = 6
        args.static_steps = 6
    report_path = run(
        args.config,
        task_id=args.task_id,
        max_demos=args.max_demos,
        random_episodes=args.random_episodes,
        random_max_steps=args.random_max_steps,
        static_steps=args.static_steps,
        seed=args.seed,
        calibrate=args.calibrate,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    final_status = (
        report.get("calibration", {}).get("status")
        if args.calibrate else report.get("status")
    )
    if final_status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
