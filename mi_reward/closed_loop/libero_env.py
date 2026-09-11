"""Thin LIBERO task adapter with reproducible initial-state partitions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from examples.LIBERO.eval_files.libero_benchmark_adapters import get_benchmark_adapter, quat2axisangle


LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def canonical_camera_image(observation: dict[str, Any], key: str = "agentview_image") -> np.ndarray:
    if key not in observation:
        raise KeyError(f"LIBERO observation has no camera key {key!r}; available={sorted(observation)}")
    image = np.asarray(observation[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera observation {key!r} must be [H,W,3], got {image.shape}.")
    # Official LIBERO HDF5 demonstrations store the same orientation returned
    # by OffScreenRenderEnv.  Flipping here creates a demo/online domain shift
    # and makes both the policy and frozen visual reward observe the wrong view.
    return np.ascontiguousarray(image)


def policy_camera_images(
    observation: dict[str, Any],
    keys: Iterable[str] = ("agentview_image", "robot0_eye_in_hand_image"),
) -> np.ndarray:
    """Return canonical uint8 policy views as ``[V,H,W,3]``."""

    images = [canonical_camera_image(observation, str(key)) for key in keys]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError(f"Policy camera shapes must match, got {[image.shape for image in images]}.")
    return np.stack(images, axis=0)


def proprioception(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"])).astype(np.float32),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[-1:],
        )
    ).astype(np.float32, copy=False)


def rlpd_robot_state(observation: dict[str, Any]) -> np.ndarray:
    """State vector aligned with the official LIBERO demonstration fields."""

    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"])).astype(np.float32),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["robot0_joint_pos"], dtype=np.float32).reshape(-1),
        )
    ).astype(np.float32, copy=False)


def apply_appearance_shift(image: np.ndarray, variant: str) -> np.ndarray:
    """Deterministic camera-only OOD shifts; never presented as semantic OOD."""

    if variant in {"none", "id", ""}:
        return image
    data = image.astype(np.float32) / 255.0
    if variant == "dark_warm":
        data = data * np.asarray([0.72, 0.62, 0.52], dtype=np.float32) + 0.04
    elif variant == "bright_cool":
        data = data * np.asarray([0.80, 0.92, 1.08], dtype=np.float32) + 0.10
    elif variant == "low_contrast":
        data = (data - 0.5) * 0.55 + 0.5
    else:
        raise ValueError(f"Unknown appearance variant: {variant!r}.")
    return np.clip(data * 255.0, 0, 255).astype(np.uint8)


def stable_task_one_hot(task_id: int, task_ids: Iterable[int]) -> np.ndarray:
    ids = tuple(int(value) for value in task_ids)
    if task_id not in ids:
        raise ValueError(f"Task id {task_id} is not in configured task ids {ids}.")
    output = np.zeros(len(ids), dtype=np.float32)
    output[ids.index(task_id)] = 1.0
    return output


class LiberoTaskEnvironment:
    def __init__(
        self,
        suite_name: str,
        task_id: int,
        *,
        benchmark_variant: str = "libero",
        resolution: int = 256,
        seed: int = 0,
        settle_steps: int = 10,
        max_episode_steps: int | None = None,
        camera_names: Iterable[str] | None = None,
    ):
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite {suite_name!r}; available={sorted(benchmark_dict)}")
        self.suite = benchmark_dict[suite_name]()
        if not 0 <= int(task_id) < int(self.suite.n_tasks):
            raise ValueError(f"task_id={task_id} is outside suite range [0,{self.suite.n_tasks}).")
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.task = self.suite.get_task(self.task_id)
        # LIBERO's official NumPy-backed init-state files predate PyTorch 2.6,
        # whose torch.load default changed to weights_only=True. Load only the
        # exact pinned benchmark asset here without weakening the global
        # deserialization default for model checkpoints.
        from libero.libero import get_libero_path

        init_states_path = (
            Path(get_libero_path("init_states"))
            / self.task.problem_folder
            / self.task.init_states_file
        )
        self.initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
        if len(self.initial_states) == 0:
            raise RuntimeError(f"LIBERO task {self.task_id} has no initial states.")
        self.adapter = get_benchmark_adapter(benchmark_variant)
        selected_cameras = None if camera_names is None else tuple(str(name) for name in camera_names)
        self.env, self.task_description = self.adapter.build_env(
            self.task, int(resolution), int(seed), selected_cameras
        )
        self.settle_steps = int(settle_steps)
        self.max_episode_steps = int(max_episode_steps or self.adapter.get_max_steps(suite_name))
        action_spec = getattr(self.env, "action_spec", None)
        if action_spec is None:
            self.action_low = np.full(7, -1.0, dtype=np.float32)
            self.action_high = np.full(7, 1.0, dtype=np.float32)
        else:
            self.action_low = np.asarray(action_spec[0], dtype=np.float32)
            self.action_high = np.asarray(action_spec[1], dtype=np.float32)
        self.episode_step = 0

    def resolve_init_indices(self, requested: Iterable[int]) -> tuple[int, ...]:
        indices = tuple(int(index) for index in requested)
        if not indices:
            raise ValueError("An initial-state split cannot be empty.")
        invalid = [index for index in indices if index < 0 or index >= len(self.initial_states)]
        if invalid:
            raise ValueError(
                f"Initial-state indices {invalid} are invalid for task {self.task_id}; "
                f"the suite provides {len(self.initial_states)} states."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"Initial-state split contains duplicates: {indices}.")
        return indices

    def reset(self, init_state_index: int) -> dict[str, Any]:
        index = self.resolve_init_indices([init_state_index])[0]
        self.env.reset()
        observation = self.env.set_init_state(self.initial_states[index])
        for _ in range(self.settle_steps):
            observation, _, done, _ = self.env.step(LIBERO_DUMMY_ACTION.tolist())
            if done:
                break
        self.episode_step = 0
        return observation

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_low, self.action_high)
        observation, sparse_reward, success, info = self.env.step(action.tolist())
        self.episode_step += 1
        truncated = self.episode_step >= self.max_episode_steps and not bool(success)
        info = dict(info or {})
        info["success"] = bool(success)
        info["sparse_reward"] = float(sparse_reward)
        return observation, float(sparse_reward), bool(success), bool(truncated), info

    def close(self) -> None:
        self.env.close()


def split_fingerprint(suite: str, task_id: int, indices: Iterable[int], appearance: str) -> str:
    payload = f"{suite}:{task_id}:{','.join(str(int(x)) for x in indices)}:{appearance}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_disjoint_splits(splits: dict[str, Iterable[int]]) -> None:
    names = list(splits)
    normalized = {name: set(int(value) for value in splits[name]) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                raise ValueError(f"LIBERO initial-state splits {left!r} and {right!r} overlap: {overlap}.")


def resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (root / value).resolve()
