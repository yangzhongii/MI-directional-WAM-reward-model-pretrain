"""Official LIBERO demonstration reader for the self-contained RLPD loop."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


def demonstration_path(env: Any, root: Path) -> Path:
    relative = Path(env.suite.get_task_demonstration(env.task_id))
    candidates = (
        root / ".venv" / "src" / "libero" / "libero" / "datasets" / relative,
        root / ".venv" / "datasets" / relative,
        root / relative,
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Official LIBERO demonstrations are missing for task {env.task_id}: {relative}"
        )
    return path


def demo_robot_states(observations: h5py.Group) -> np.ndarray:
    """Build the same 15-D state used by online LIBERO observations."""

    required = ("ee_pos", "ee_ori", "gripper_states", "joint_states")
    missing = [key for key in required if key not in observations]
    if missing:
        raise KeyError(f"LIBERO demo is missing state fields: {missing}")
    return np.concatenate(
        [np.asarray(observations[key], dtype=np.float32) for key in required], axis=-1
    )


def resize_views(views: np.ndarray, image_size: int) -> np.ndarray:
    views = np.asarray(views, dtype=np.uint8)
    if views.ndim != 4 or views.shape[-1] != 3:
        raise ValueError(f"Expected views [V,H,W,3], got {views.shape}.")
    if views.shape[1:3] == (image_size, image_size):
        return np.ascontiguousarray(views)
    resized = [
        np.asarray(Image.fromarray(view).resize((image_size, image_size), Image.Resampling.BILINEAR))
        for view in views
    ]
    return np.stack(resized, axis=0).astype(np.uint8, copy=False)


def iter_demo_episodes(
    path: str | Path,
    *,
    max_demos: int | None = None,
) -> Iterator[dict[str, np.ndarray | str]]:
    """Yield successful episodes with two synchronized RGB views."""

    with h5py.File(Path(path), "r") as handle:
        names = sorted(handle["data"].keys())
        if max_demos is not None:
            names = names[: int(max_demos)]
        for name in names:
            demo = handle["data"][name]
            observations = demo["obs"]
            main = np.asarray(observations["agentview_rgb"], dtype=np.uint8)
            wrist = np.asarray(observations["eye_in_hand_rgb"], dtype=np.uint8)
            states = demo_robot_states(observations)
            mujoco_states = np.asarray(demo["states"], dtype=np.float64)
            actions = np.asarray(demo["actions"], dtype=np.float32)
            rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float32)
            dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.bool_)
            length = min(len(main), len(wrist), len(states), len(mujoco_states), len(actions))
            if length < 2:
                continue
            yield {
                "name": name,
                "main": main[:length],
                "wrist": wrist[:length],
                "states": states[:length],
                "mujoco_states": mujoco_states[:length],
                "actions": actions[:length],
                "rewards": rewards[:length],
                "dones": dones[:length],
            }

