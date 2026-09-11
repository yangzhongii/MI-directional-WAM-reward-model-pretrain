"""Instance-level validation layered on top of base feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mi_reward.data.instance_schema import load_object_state_sequence, validate_instance_artifacts
from mi_reward.data.schema import ControlArtifacts, InstanceVariantDescriptor, SimulationProvenance
from mi_reward.relations.instance_sequence import required_relation_names
from mi_reward.relations.sequence import load_relation_sequence


@dataclass(frozen=True)
class InstanceVerificationConfig:
    task_family: str
    max_pose_step: float = 0.25
    require_controls: bool = True
    require_simulation: bool = True


@dataclass(frozen=True)
class InstanceCheckResult:
    checks: dict[str, bool]
    reasons: list[str]
    outcome_checks: dict[str, bool] | None = None
    failure_mode: str = "unknown"

    @property
    def accepted(self) -> bool:
        return all(self.checks.values())


def _task_object_positions(sequence, variant: InstanceVariantDescriptor) -> list[tuple[float, float, float]] | None:
    positions: list[tuple[float, float, float]] = []
    for frame in sequence.frames:
        match = next((item for item in frame.objects if item.object_id == variant.source_object_id), None)
        if match is None:
            matches = [item for item in frame.objects if item.category == variant.target_category]
            if len(matches) != 1:
                return None
            match = matches[0]
        positions.append(tuple(float(value) for value in match.pose_xyz))
    return positions


def _max_step(positions: list[tuple[float, float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    return max(sum((a - b) ** 2 for a, b in zip(lhs, rhs)) ** 0.5 for lhs, rhs in zip(positions, positions[1:]))


def verify_instance_candidate(
    *,
    candidate_frames: Sequence[str],
    object_state_path: str | None,
    relation_path: str | None,
    controls: ControlArtifacts | None,
    instance_variant: InstanceVariantDescriptor | None,
    simulation: SimulationProvenance | None,
    config: InstanceVerificationConfig,
) -> InstanceCheckResult:
    """Check that a candidate's instance sidecars agree with task semantics."""

    checks: dict[str, bool] = {}
    reasons = validate_instance_artifacts(
        frames=candidate_frames,
        object_state_path=object_state_path,
        controls=controls,
        instance_variant=instance_variant,
        strict=config.require_controls,
    )
    checks["instance_artifacts"] = not reasons
    if simulation is None and config.require_simulation:
        checks["simulation_provenance"] = False
        reasons.append("missing_simulation_provenance")
    else:
        checks["simulation_provenance"] = True
    if reasons or object_state_path is None or instance_variant is None:
        return InstanceCheckResult(checks=checks, reasons=sorted(set(reasons)))

    sequence = load_object_state_sequence(object_state_path)
    positions = _task_object_positions(sequence, instance_variant)
    checks["task_object_identity"] = positions is not None
    if positions is None:
        reasons.append("task_object_identity_not_continuous")
    else:
        checks["pose_continuity"] = _max_step(positions) <= config.max_pose_step
        if not checks["pose_continuity"]:
            reasons.append("object_pose_discontinuity")

    if relation_path is None or not Path(relation_path).is_file():
        checks["task_relation_sequence"] = False
        reasons.append("missing_relation_sequence")
        return InstanceCheckResult(checks=checks, reasons=sorted(set(reasons)))
    try:
        relations = load_relation_sequence(relation_path)
    except (FileNotFoundError, ValueError):
        checks["task_relation_sequence"] = False
        reasons.append("invalid_relation_sequence")
        return InstanceCheckResult(checks=checks, reasons=sorted(set(reasons)))
    checks["relation_frame_alignment"] = relations.values.shape[0] == len(candidate_frames)
    if not checks["relation_frame_alignment"]:
        reasons.append("relation_frame_count_mismatch")
    relation_index = {name: index for index, name in enumerate(relations.names)}
    required = required_relation_names(config.task_family)
    missing = [name for name in required if name not in relation_index]
    checks["task_relation_sequence"] = not missing
    if missing:
        reasons.append("missing_task_relation_fields")
        return InstanceCheckResult(checks=checks, reasons=sorted(set(reasons)))

    collision = relations.values[:, relation_index["collision_free"]]
    contact = relations.values[:, relation_index["contact_valid"]]
    terminal = relations.values[-1, relation_index["target_satisfied"]]
    outcome_checks = {
        "collision_free": bool((collision >= 0.5).all().item()),
        "contact_valid": bool((contact >= 0.5).all().item()),
        "target_satisfied": bool(terminal >= 0.5),
    }
    if all(outcome_checks.values()):
        failure_mode = "none"
    elif not outcome_checks["collision_free"]:
        failure_mode = "collision"
    elif not outcome_checks["contact_valid"]:
        failure_mode = "invalid_contact"
    else:
        failure_mode = "goal_not_reached"
    return InstanceCheckResult(
        checks=checks,
        reasons=sorted(set(reasons)),
        outcome_checks=outcome_checks,
        failure_mode=failure_mode,
    )
