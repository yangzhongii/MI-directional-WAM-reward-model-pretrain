"""Geometry/kinematics checks used before a Cosmos candidate becomes supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mi_reward.data.schema import CandidateProvenance, RelationDescriptor, VerificationRecord
from mi_reward.relations.geometry import RobotState


@dataclass(frozen=True)
class FeasibilityConfig:
    workspace_lower: tuple[float, float, float] | None = None
    workspace_upper: tuple[float, float, float] | None = None
    joint_lower: tuple[float, ...] | None = None
    joint_upper: tuple[float, ...] | None = None
    max_ee_to_object_distance: float | None = None
    max_object_to_goal_distance: float | None = None
    max_object_goal_orientation_error: float | None = None
    required_relation_names: tuple[str, ...] = ("object_to_goal_distance",)


def _relation_map(descriptor: RelationDescriptor) -> dict[str, float]:
    return dict(zip(descriptor.names, descriptor.values))


def verify_candidate(
    provenance: CandidateProvenance | None,
    robot_state: RobotState | None,
    relation: RelationDescriptor | None,
    config: FeasibilityConfig,
    *,
    visual_relation: RelationDescriptor | None = None,
    max_visual_relation_error: float | None = None,
    candidate_frames: Sequence[str] | None = None,
) -> VerificationRecord:
    """Reject candidates with missing physical context or violated constraints.

    The verifier never infers joint states from Cosmos pixels. Callers must
    supply an action-conditioned rollout or an observed robot-state sequence.
    """

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if candidate_frames is not None:
        has_frames = bool(candidate_frames) and all(Path(frame).is_file() for frame in candidate_frames)
        checks["candidate_frames"] = has_frames
        if not has_frames:
            reasons.append("missing_candidate_frames")

    has_context = bool(
        provenance
        and provenance.action_path
        and provenance.robot_state_path
        and provenance.parent_traj_id
    )
    checks["action_and_state_context"] = has_context
    if not has_context:
        reasons.append("missing_action_or_robot_state_context")

    checks["relation_available"] = relation is not None
    if relation is None:
        reasons.append("missing_relation_descriptor")
        return VerificationRecord(status="rejected", checks=checks, reasons=reasons)

    relation_values = _relation_map(relation)
    required_ok = all(name in relation_values for name in config.required_relation_names)
    checks["required_relation_fields"] = required_ok
    if not required_ok:
        reasons.append("missing_required_relation_fields")

    if robot_state is not None:
        ee = robot_state.end_effector_position
        workspace_ok = True
        if config.workspace_lower is not None and config.workspace_upper is not None:
            workspace_ok = all(low <= value <= high for value, low, high in zip(ee, config.workspace_lower, config.workspace_upper))
        checks["workspace"] = workspace_ok
        if not workspace_ok:
            reasons.append("end_effector_outside_workspace")

        joints_ok = True
        if config.joint_lower is not None and config.joint_upper is not None:
            joints_ok = (
                len(robot_state.joint_positions) == len(config.joint_lower) == len(config.joint_upper)
                and all(low <= value <= high for value, low, high in zip(robot_state.joint_positions, config.joint_lower, config.joint_upper))
            )
        checks["joint_limits"] = joints_ok
        if not joints_ok:
            reasons.append("joint_limit_violation")
    else:
        checks["workspace"] = False
        checks["joint_limits"] = False
        reasons.append("missing_robot_state")

    def threshold_check(name: str, threshold: float | None) -> None:
        if threshold is None:
            return
        value = relation_values.get(name)
        ok = value is not None and value <= threshold
        checks[name] = ok
        if not ok:
            reasons.append(f"{name}_constraint_failed")

    threshold_check("ee_to_object_distance", config.max_ee_to_object_distance)
    threshold_check("object_to_goal_distance", config.max_object_to_goal_distance)
    threshold_check("object_goal_orientation_error", config.max_object_goal_orientation_error)

    if visual_relation is not None and max_visual_relation_error is not None:
        visual_values = _relation_map(visual_relation)
        shared = sorted(set(relation_values).intersection(visual_values))
        consistency = bool(shared) and max(abs(relation_values[name] - visual_values[name]) for name in shared) <= max_visual_relation_error
        checks["visual_kinematic_relation_consistency"] = consistency
        if not consistency:
            reasons.append("visual_kinematic_relation_mismatch")

    accepted = all(checks.values())
    return VerificationRecord(
        status="accepted" if accepted else "rejected",
        checks=checks,
        reasons=reasons,
        relation=relation,
    )
