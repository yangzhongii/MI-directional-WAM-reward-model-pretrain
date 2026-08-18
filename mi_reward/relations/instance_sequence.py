"""Task-aware robot-object relation descriptors for instance rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mi_reward.data.schema import ObjectStateDescriptor, RelationDescriptor
from mi_reward.relations.geometry import RobotState, TaskGeometry, build_relation_descriptor


TASK_REQUIRED_RELATIONS: dict[str, tuple[str, ...]] = {
    "pick_place": ("collision_free", "contact_valid", "target_satisfied"),
    "push_shape": ("collision_free", "contact_valid", "target_satisfied"),
    "peg_insertion": (
        "collision_free",
        "contact_valid",
        "target_satisfied",
        "object_goal_orientation_error",
    ),
}


@dataclass(frozen=True)
class InstanceRelationFlags:
    collision_free: bool
    contact_valid: bool
    target_satisfied: bool


def required_relation_names(task_family: str) -> tuple[str, ...]:
    try:
        return TASK_REQUIRED_RELATIONS[task_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_family for relation checks: {task_family}") from exc


def build_instance_relation_descriptor(
    robot: RobotState,
    task_object: ObjectStateDescriptor,
    goal: ObjectStateDescriptor,
    flags: InstanceRelationFlags,
    *,
    source: str = "simulator",
) -> RelationDescriptor:
    """Build geometric relations plus explicit contact and terminal predicates."""

    base = build_relation_descriptor(
        robot,
        TaskGeometry(
            object_position=task_object.pose_xyz,
            goal_position=goal.pose_xyz,
            object_quaternion=task_object.pose_quaternion,
            goal_quaternion=goal.pose_quaternion,
        ),
        source=source,
    )
    names = [
        *base.names,
        "object_visible",
        "object_in_gripper",
        "collision_free",
        "contact_valid",
        "target_satisfied",
    ]
    values = [
        *base.values,
        float(task_object.visible),
        float(task_object.in_gripper),
        float(flags.collision_free),
        float(flags.contact_valid),
        float(flags.target_satisfied),
    ]
    return RelationDescriptor(names=names, values=values, source=source)


def validate_relation_sequence_for_task(
    descriptors: Iterable[RelationDescriptor],
    task_family: str,
) -> list[str]:
    """Return stable reason codes for relation fields required by one task."""

    records = list(descriptors)
    if not records:
        return ["missing_relation_sequence"]
    required = set(required_relation_names(task_family))
    reasons: list[str] = []
    for step, descriptor in enumerate(records):
        missing = sorted(required.difference(descriptor.names))
        if missing:
            reasons.append(f"missing_task_relations_at_step_{step}")
    return reasons
