"""Reusable LIBERO privileged-state replay utilities for Pipeline v3.

The functions in this module expose raw physical evidence only.  They do not
collapse contact / geometry / outcome into a hand-designed reward.  The same
records are consumed by hard-negative construction, teacher consistency, and
held-out physical-aliasing evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def add_libero_source(root: Path) -> None:
    source = root / ".venv" / "src" / "libero"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def resolve_libero_task(root: Path, suite: str, task_id: int) -> tuple[Path, Path, str]:
    """Resolve the official demonstration, BDDL file, and language string."""

    add_libero_source(root)
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"Unknown LIBERO suite {suite!r}; available={sorted(suites)}")
    task_suite = suites[suite]()
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(f"Task id {task_id} is outside [0,{task_suite.n_tasks}).")
    task = task_suite.get_task(task_id)
    demo_rel = Path(task_suite.get_task_demonstration(task_id))
    candidates = (
        root / ".venv" / "src" / "libero" / "libero" / "datasets" / demo_rel,
        root / ".venv" / "datasets" / demo_rel,
        root / demo_rel,
    )
    demo = next((path for path in candidates if path.is_file()), None)
    if demo is None:
        raise FileNotFoundError(demo_rel)
    bddl = (
        root
        / ".venv"
        / "src"
        / "libero"
        / "libero"
        / "libero"
        / "bddl_files"
        / task.problem_folder
        / task.bddl_file
    )
    if not bddl.is_file():
        raise FileNotFoundError(bddl)
    return demo, bddl, str(task.language)


def goal_objects(goal_state: list[list[str]]) -> tuple[str, str, str]:
    """Return ``(predicate, task_object, goal_object)`` for rigid LIBERO goals."""

    for predicate in goal_state:
        if len(predicate) >= 3:
            return str(predicate[0]).lower(), str(predicate[1]), str(predicate[2])
    raise ValueError(f"Expected a binary task goal predicate, got {goal_state!r}")


def _to_list(value: np.ndarray) -> list[float]:
    return np.asarray(value, dtype=np.float64).tolist()


def extract_privileged_records(
    *,
    bddl_path: str | Path,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay saved MuJoCo states and return raw per-frame physical evidence."""

    from libero.libero.envs import OffScreenRenderEnv

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if rewards is None:
        rewards = np.zeros(len(actions), dtype=np.float64)
    if dones is None:
        dones = np.zeros(len(actions), dtype=np.uint8)
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.uint8)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
    )
    try:
        env.reset()
        core = env.env
        predicate, task_object, goal_object = goal_objects(core.parsed_problem["goal_state"])
        if task_object not in core.object_states_dict or goal_object not in core.object_states_dict:
            raise RuntimeError(
                f"Goal objects are unavailable in object_states_dict: {task_object}, {goal_object}; "
                f"available={sorted(core.object_states_dict)}"
            )
        task_state = core.object_states_dict[task_object]
        goal_state = core.object_states_dict[goal_object]
        task_model = core.get_object(task_object)
        length = min(len(states), len(actions), len(rewards), len(dones))
        records: list[dict[str, object]] = []

        for frame_index in range(length):
            env.regenerate_obs_from_state(states[frame_index])
            task_geom = task_state.get_geom_state()
            goal_geom = goal_state.get_geom_state()
            object_pos = np.asarray(task_geom["pos"], dtype=np.float64)
            goal_pos = np.asarray(goal_geom["pos"], dtype=np.float64)
            eef_pos = np.asarray(core._eef_xpos, dtype=np.float64)
            object_goal_delta = object_pos - goal_pos

            try:
                grasped = bool(core._check_grasp(core.robots[0].gripper, task_model))
            except Exception:
                grasped = False
            try:
                object_goal_contact = bool(task_state.check_contact(goal_state))
            except Exception:
                object_goal_contact = False
            try:
                environment_success = bool(core._check_success())
            except Exception:
                environment_success = bool(rewards[frame_index] > 0.0)

            records.append(
                {
                    "frame_index": int(frame_index),
                    "action": _to_list(actions[frame_index]),
                    "task_object_pos": _to_list(object_pos),
                    "task_object_quat": _to_list(task_geom["quat"]),
                    "goal_object_pos": _to_list(goal_pos),
                    "goal_object_quat": _to_list(goal_geom["quat"]),
                    "eef_pos": _to_list(eef_pos),
                    "object_goal_delta_xyz": _to_list(object_goal_delta),
                    "object_goal_distance_xyz": float(np.linalg.norm(object_goal_delta)),
                    "object_goal_distance_xy": float(np.linalg.norm(object_goal_delta[:2])),
                    "eef_object_distance_xyz": float(np.linalg.norm(eef_pos - object_pos)),
                    "grasped": grasped,
                    "object_goal_contact": object_goal_contact,
                    "environment_success": environment_success,
                    "recorded_reward": float(rewards[frame_index]),
                    "recorded_done": bool(dones[frame_index]),
                }
            )

        return {
            "goal_predicate": predicate,
            "task_object": task_object,
            "goal_object": goal_object,
            "frames": records,
        }
    finally:
        env.close()


def classify_physical_transition(
    left: dict[str, object],
    right: dict[str, object],
    *,
    approach_epsilon: float = 7.5e-4,
    transport_epsilon: float = 5e-4,
) -> int:
    """Classify one transition as forward (+1), neutral (0), or regression (-1).

    This helper is intentionally a *sampler label*, not a reward.  It uses
    task-stage predicates to construct same-task hard negatives for the v3
    information critics and to evaluate direction discrimination.

    Priority is categorical rather than weighted:
      1. first measured task success -> forward;
      2. grasp loss -> regression;
      3. grasp acquisition -> forward;
      4. before grasp, EEF approaching / retreating from the task object;
      5. while carrying, task object approaching / retreating from the goal.
    """

    left_success = bool(left.get("environment_success", False))
    right_success = bool(right.get("environment_success", False))
    if right_success and not left_success:
        return 1

    left_grasped = bool(left.get("grasped", False))
    right_grasped = bool(right.get("grasped", False))
    if left_grasped and not right_grasped:
        return -1
    if right_grasped and not left_grasped:
        return 1

    if not left_grasped and not right_grasped:
        delta = float(left["eef_object_distance_xyz"]) - float(right["eef_object_distance_xyz"])
        epsilon = float(approach_epsilon)
    else:
        delta = float(left["object_goal_distance_xy"]) - float(right["object_goal_distance_xy"])
        epsilon = float(transport_epsilon)

    if delta > epsilon:
        return 1
    if delta < -epsilon:
        return -1
    return 0

