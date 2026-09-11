"""Deterministic robot/object-state latent extraction for the privileged teacher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mi_reward.data.instance_schema import load_object_state_sequence


def _records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Robot-state sequence not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Robot-state sequence is empty: {source}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        for key in ("frames", "states", "records", "robot_states"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Unsupported robot-state format: {source}")
    return [dict(item) for item in payload]


def extract_kinematic_latents(
    robot_state_path: str | Path,
    object_state_path: str | Path,
    *,
    output_dim: int = 48,
) -> torch.Tensor:
    """Return synchronized, normalized physical-state tokens ``[T,1,D]``.

    The vector contains joint state, end-effector pose, gripper state and all
    task-object/goal poses in stable object-id order. Padding/truncation gives
    a fixed contract across the three benchmark task families.
    """

    if output_dim < 8:
        raise ValueError("Kinematic latent output_dim must be at least 8.")
    robots = _records(robot_state_path)
    objects = load_object_state_sequence(object_state_path).frames
    if len(robots) != len(objects):
        raise ValueError(
            f"Robot/object state length mismatch: {len(robots)} versus {len(objects)}."
        )
    vectors: list[list[float]] = []
    for index, (robot, object_frame) in enumerate(zip(robots, objects)):
        if int(robot.get("frame_index", index)) != index or object_frame.frame_index != index:
            raise ValueError("Kinematic state frame indices must be contiguous and synchronized.")
        values = [float(value) for value in robot.get("joint_positions", [])]
        values.extend(float(value) for value in robot.get("end_effector_position", []))
        values.extend(float(value) for value in robot.get("end_effector_quaternion", []))
        values.append(float(robot.get("gripper_width", 0.0)))
        for item in sorted(object_frame.objects, key=lambda value: value.object_id):
            values.extend(float(value) for value in item.pose_xyz)
            values.extend(float(value) for value in item.pose_quaternion)
            values.append(float(item.in_gripper))
        values = values[:output_dim] + [0.0] * max(output_dim - len(values), 0)
        vectors.append(values)
    result = torch.tanh(torch.tensor(vectors, dtype=torch.float32)).unsqueeze(1)
    if not torch.isfinite(result).all():
        raise ValueError("Kinematic latent contains NaN or infinity.")
    return result
