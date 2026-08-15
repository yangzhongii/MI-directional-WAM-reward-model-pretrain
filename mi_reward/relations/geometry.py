"""Small, dependency-free geometry primitives for robot-grounded reward labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos
from typing import Sequence

from mi_reward.data.schema import RelationDescriptor


def _vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return float(value[0]), float(value[1]), float(value[2])


def _quat4(value: Sequence[float] | None, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly four values.")
    q = tuple(float(component) for component in value)
    norm_sq = sum(component * component for component in q)
    if norm_sq <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    norm = norm_sq ** 0.5
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def _sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]


def _norm(value: tuple[float, float, float]) -> float:
    return sum(component * component for component in value) ** 0.5


def _orientation_error(lhs: tuple[float, float, float, float] | None, rhs: tuple[float, float, float, float] | None) -> float:
    if lhs is None or rhs is None:
        return 0.0
    dot = abs(sum(a * b for a, b in zip(lhs, rhs)))
    return 2.0 * acos(max(-1.0, min(1.0, dot)))


@dataclass(frozen=True)
class RobotState:
    joint_positions: list[float]
    end_effector_position: list[float]
    end_effector_quaternion: list[float] | None = None
    gripper_width: float | None = None


@dataclass(frozen=True)
class TaskGeometry:
    object_position: list[float]
    goal_position: list[float]
    object_quaternion: list[float] | None = None
    goal_quaternion: list[float] | None = None


def build_relation_descriptor(robot: RobotState, geometry: TaskGeometry, source: str = "measured") -> RelationDescriptor:
    """Create an ordered relation vector without conflating it with image similarity."""

    ee = _vec3(robot.end_effector_position, "end_effector_position")
    obj = _vec3(geometry.object_position, "object_position")
    goal = _vec3(geometry.goal_position, "goal_position")
    ee_to_obj = _sub(obj, ee)
    obj_to_goal = _sub(goal, obj)
    ee_q = _quat4(robot.end_effector_quaternion, "end_effector_quaternion")
    obj_q = _quat4(geometry.object_quaternion, "object_quaternion")
    goal_q = _quat4(geometry.goal_quaternion, "goal_quaternion")
    names = [
        "ee_to_object_x", "ee_to_object_y", "ee_to_object_z", "ee_to_object_distance",
        "object_to_goal_x", "object_to_goal_y", "object_to_goal_z", "object_to_goal_distance",
        "ee_object_orientation_error", "object_goal_orientation_error",
    ]
    values = [
        *ee_to_obj, _norm(ee_to_obj),
        *obj_to_goal, _norm(obj_to_goal),
        _orientation_error(ee_q, obj_q), _orientation_error(obj_q, goal_q),
    ]
    if robot.gripper_width is not None:
        names.append("gripper_width")
        values.append(float(robot.gripper_width))
    return RelationDescriptor(names=names, values=values, source=source)
