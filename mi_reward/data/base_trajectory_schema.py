"""Canonical input contract for generalization data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseTrajectory:
    """One observed trajectory before instance and scene generalization.

    The source may provide only RGB and task text. Physical sidecars are kept
    optional here because they are produced by the planner/MuJoCo stages when
    they are absent from an offline source such as Robometer.
    """

    base_id: str
    source: str
    task: str
    task_family: str
    instruction: str
    frames: list[str]
    initial_frame: str
    goal_frame: str
    goal_ref_id: str
    physical_task_config: str
    object_prompts: dict[str, str]
    split: str = "train"
    action_path: str | None = None
    robot_state_path: str | None = None
    camera_calibration_path: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_id:
            raise ValueError("BaseTrajectory.base_id must be non-empty.")
        if self.source == "simulator_native":
            if self.frames or self.initial_frame or self.goal_frame:
                raise ValueError(
                    "Simulator-native task seeds must defer frames to the MuJoCo stage."
                )
        else:
            if len(self.frames) < 2:
                raise ValueError("BaseTrajectory.frames must contain at least two frames.")
            if self.initial_frame != self.frames[0]:
                raise ValueError("BaseTrajectory.initial_frame must equal frames[0].")
        if not self.goal_ref_id:
            raise ValueError("BaseTrajectory.goal_ref_id must be non-empty.")
        if not self.object_prompts:
            raise ValueError("BaseTrajectory.object_prompts must identify at least one task object.")

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "BaseTrajectory":
        frames = [str(path) for path in item.get("frames", [])]
        return cls(
            base_id=str(item["base_id"]),
            source=str(item["source"]),
            task=str(item["task"]),
            task_family=str(item["task_family"]),
            instruction=str(item.get("instruction", item["task"])),
            frames=frames,
            initial_frame=str(item.get("initial_frame", frames[0] if frames else "")),
            goal_frame=str(item.get("goal_frame", "")),
            goal_ref_id=str(item["goal_ref_id"]),
            physical_task_config=str(item["physical_task_config"]),
            object_prompts={str(key): str(value) for key, value in dict(item.get("object_prompts", {})).items()},
            split=str(item.get("split", "train")),
            action_path=(None if item.get("action_path") is None else str(item["action_path"])),
            robot_state_path=(None if item.get("robot_state_path") is None else str(item["robot_state_path"])),
            camera_calibration_path=(
                None if item.get("camera_calibration_path") is None else str(item["camera_calibration_path"])
            ),
            metadata=dict(item.get("metadata") or {}),
        )
