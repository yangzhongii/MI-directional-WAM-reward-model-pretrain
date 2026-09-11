"""Ingest official Cosmos robot/action-conditioned outputs into GeoProgress.

The official generator is intentionally kept outside this adapter because its
runtime/configuration changes independently.  This module consumes its RGB
outputs together with the action rollout and measured robot-state sidecars;
it never estimates actions or joints from generated pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mi_reward.data.schema import CandidateProvenance, TrajectoryExample, VerificationRecord, write_jsonl
from mi_reward.relations.geometry import RobotState
from mi_reward.relations.sequence import load_relation_sequence
from mi_reward.verification.feasibility import FeasibilityConfig, verify_candidate


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Candidate record at {path}:{line_no} must be an object.")
            records.append(payload)
    return records


def _frame_paths(record: dict[str, Any], base_dir: Path) -> list[str]:
    if isinstance(record.get("frames"), list):
        return [str((base_dir / item).resolve()) if not Path(item).is_absolute() else str(item) for item in record["frames"]]
    frame_dir = record.get("frame_dir") or record.get("frames_dir")
    if not frame_dir:
        return []
    directory = Path(frame_dir)
    if not directory.is_absolute():
        directory = base_dir / directory
    if not directory.is_dir():
        return []
    return [str(path) for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]


def _resolve_optional(value: Any, base_dir: Path) -> str | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _robot_state_from_record(record: dict[str, Any]) -> RobotState:
    payload = record.get("robot_state", record.get("state", record))
    if not isinstance(payload, dict):
        raise ValueError("Robot state entry must be an object.")
    return RobotState(
        joint_positions=[float(value) for value in payload["joint_positions"]],
        end_effector_position=[float(value) for value in payload["end_effector_position"]],
        end_effector_quaternion=(
            None if payload.get("end_effector_quaternion") is None
            else [float(value) for value in payload["end_effector_quaternion"]]
        ),
        gripper_width=(None if payload.get("gripper_width") is None else float(payload["gripper_width"])),
    )


def load_last_robot_state(path: str | Path | None) -> RobotState | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("states", "robot_states", "frames", "records"):
                if isinstance(payload.get(key), list) and payload[key]:
                    return _robot_state_from_record(payload[key][-1])
            return _robot_state_from_record(payload)
        if isinstance(payload, list) and payload:
            return _robot_state_from_record(payload[-1])
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            return _robot_state_from_record(json.loads(lines[-1]))
    return None


def feasibility_config_from_dict(payload: dict[str, Any]) -> FeasibilityConfig:
    def vec3(name: str) -> tuple[float, float, float] | None:
        value = payload.get(name)
        return None if value is None else tuple(float(item) for item in value)  # type: ignore[return-value]

    def vector(name: str) -> tuple[float, ...] | None:
        value = payload.get(name)
        return None if value is None else tuple(float(item) for item in value)

    required_names = tuple(str(item) for item in payload.get("required_relation_names", []))
    return FeasibilityConfig(
        workspace_lower=vec3("workspace_lower"),
        workspace_upper=vec3("workspace_upper"),
        joint_lower=vector("joint_lower"),
        joint_upper=vector("joint_upper"),
        max_ee_to_object_distance=(None if payload.get("max_ee_to_object_distance") is None else float(payload["max_ee_to_object_distance"])),
        max_object_to_goal_distance=(None if payload.get("max_object_to_goal_distance") is None else float(payload["max_object_to_goal_distance"])),
        max_object_goal_orientation_error=(None if payload.get("max_object_goal_orientation_error") is None else float(payload["max_object_goal_orientation_error"])),
        required_relation_names=required_names or FeasibilityConfig().required_relation_names,
    )


def ingest_action_conditioned_candidates(
    candidate_records: str | Path,
    output_manifest: str | Path,
    feasibility_config: FeasibilityConfig,
    *,
    split: str = "train",
) -> list[TrajectoryExample]:
    """Create a manifest with deterministic acceptance/rejection records.

    Required candidate-record fields: ``traj_id``, ``task``, ``goal_ref_id``,
    ``parent_traj_id``, ``action_path``, ``robot_state_path``, and
    ``relation_path``. Frames can be supplied as ``frames`` or ``frame_dir``.
    Incomplete candidates are written as rejected for auditability, and later
    GeoProgress scoring/training filters them out.
    """

    input_path = Path(candidate_records)
    base_dir = input_path.parent
    examples: list[TrajectoryExample] = []
    for record in _read_records(input_path):
        frames = _frame_paths(record, base_dir)
        action_path = _resolve_optional(record.get("action_path"), base_dir)
        robot_state_path = _resolve_optional(record.get("robot_state_path"), base_dir)
        relation_path = _resolve_optional(record.get("relation_path"), base_dir)
        provenance = CandidateProvenance(
            generator=str(record.get("generator", "cosmos_predict2_5_robot_action_cond")),
            model_id=str(record.get("model_id", "")),
            parent_traj_id=str(record.get("parent_traj_id", "")),
            generation_seed=(None if record.get("generation_seed") is None else int(record["generation_seed"])),
            action_path=action_path,
            robot_state_path=robot_state_path,
            context_video_path=_resolve_optional(record.get("context_video_path"), base_dir),
            prompt=(None if record.get("prompt") is None else str(record["prompt"])),
        )
        relation = None
        if relation_path:
            try:
                relation = load_relation_sequence(relation_path)
                relation_descriptor = None if relation.values.shape[0] == 0 else {
                    "names": relation.names,
                    "values": relation.values[-1].tolist(),
                    "source": "measured",
                }
            except (FileNotFoundError, ValueError):
                relation_descriptor = None
        else:
            relation_descriptor = None
        relation_obj = None
        if relation_descriptor is not None:
            from mi_reward.data.schema import RelationDescriptor

            relation_obj = RelationDescriptor.from_dict(relation_descriptor)
        # A nonempty path string is not evidence. Only existing sidecars are
        # presented to the verifier as action/state context.
        verification_provenance = CandidateProvenance(
            generator=provenance.generator,
            model_id=provenance.model_id,
            parent_traj_id=provenance.parent_traj_id,
            generation_seed=provenance.generation_seed,
            action_path=action_path if action_path and Path(action_path).is_file() else None,
            robot_state_path=robot_state_path if robot_state_path and Path(robot_state_path).is_file() else None,
            context_video_path=provenance.context_video_path,
            prompt=provenance.prompt,
        )
        verification = verify_candidate(
            verification_provenance,
            load_last_robot_state(verification_provenance.robot_state_path),
            relation_obj,
            feasibility_config,
            candidate_frames=frames,
        )
        if not record.get("goal_ref_id"):
            verification = VerificationRecord(
                status="rejected",
                checks={**verification.checks, "goal_reference": False},
                reasons=[*verification.reasons, "missing_goal_ref_id"],
                relation=verification.relation,
            )
        examples.append(
            TrajectoryExample(
                traj_id=str(record["traj_id"]),
                task=str(record["task"]),
                frames=frames,
                source="cosmos_action_cond",
                split=str(record.get("split", split)),
                metadata={"candidate_index": record.get("candidate_index")},
                robot_state_path=robot_state_path,
                action_path=action_path,
                relation_path=relation_path,
                goal_ref_id=(None if record.get("goal_ref_id") is None else str(record["goal_ref_id"])),
                parent_traj_id=provenance.parent_traj_id or None,
                candidate_provenance=provenance,
                verification=verification,
            )
        )
    write_jsonl(output_manifest, examples)
    return examples
