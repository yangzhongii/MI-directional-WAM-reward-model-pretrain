"""Ingest verified scene- and instance-aware rollout candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mi_reward.data.cosmos_action_cond import load_last_robot_state
from mi_reward.data.schema import (
    CandidateProvenance,
    ControlArtifacts,
    InstanceVariantDescriptor,
    RelationDescriptor,
    SceneVariantDescriptor,
    SimulationProvenance,
    TrajectoryExample,
    VerificationRecord,
    write_jsonl,
)
from mi_reward.relations.sequence import load_relation_sequence
from mi_reward.verification.feasibility import FeasibilityConfig, verify_candidate
from mi_reward.verification.instance_checks import InstanceVerificationConfig, verify_instance_candidate


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Candidate record at {source}:{line_no} must be an object.")
        records.append(payload)
    return records


def _resolve(value: Any, base_dir: Path) -> str | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _frames(record: dict[str, Any], base_dir: Path) -> list[str]:
    if isinstance(record.get("frames"), list):
        return [
            str(Path(item) if Path(item).is_absolute() else (base_dir / item).resolve())
            for item in record["frames"]
        ]
    root = record.get("frame_dir") or record.get("frames_dir")
    if not root:
        return []
    directory = Path(str(root))
    if not directory.is_absolute():
        directory = base_dir / directory
    if not directory.is_dir():
        return []
    return [str(path) for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]


def _controls(value: Any, base_dir: Path) -> ControlArtifacts | None:
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    for key in ("mask_root", "depth_root", "segmentation_root", "edge_root"):
        payload[key] = _resolve(payload.get(key), base_dir)
    return ControlArtifacts.from_dict(payload)


def _merge_verification(*records: VerificationRecord) -> VerificationRecord:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    relation = None
    for record in records:
        checks.update(record.checks)
        reasons.extend(record.reasons)
        relation = relation or record.relation
    return VerificationRecord(
        status="accepted" if checks and all(checks.values()) else "rejected",
        checks=checks,
        reasons=sorted(set(reasons)),
        relation=relation,
    )


def ingest_instance_rollout_candidates(
    candidate_records: str | Path,
    output_manifest: str | Path,
    feasibility_config: FeasibilityConfig,
    instance_config: InstanceVerificationConfig | Mapping[str, InstanceVerificationConfig],
    *,
    split: str = "train",
) -> list[TrajectoryExample]:
    """Create a manifest from externally executed instance rollouts.

    Each candidate must include ``task_family``, ``object_state_path``,
    ``controls``, ``instance_variant``, and ``simulation`` in addition to the
    action-conditioned sidecars required by the existing Cosmos ingestion path.
    """

    if isinstance(instance_config, InstanceVerificationConfig):
        verification_configs = {instance_config.task_family: instance_config}
        default_task_family = instance_config.task_family
    else:
        verification_configs = dict(instance_config)
        if not verification_configs:
            raise ValueError("At least one task-family verification config is required.")
        default_task_family = next(iter(verification_configs)) if len(verification_configs) == 1 else None

    input_path = Path(candidate_records)
    base_dir = input_path.parent
    examples: list[TrajectoryExample] = []
    for record in _read_records(input_path):
        frames = _frames(record, base_dir)
        action_path = _resolve(record.get("action_path"), base_dir)
        robot_state_path = _resolve(record.get("robot_state_path"), base_dir)
        relation_path = _resolve(record.get("relation_path"), base_dir)
        object_state_path = _resolve(record.get("object_state_path"), base_dir)
        controls = _controls(record.get("control_artifacts") or record.get("controls"), base_dir)
        record_reasons: list[str] = []
        try:
            instance_variant = (
                InstanceVariantDescriptor.from_dict(record["instance_variant"])
                if isinstance(record.get("instance_variant"), dict)
                else None
            )
        except (KeyError, TypeError, ValueError):
            instance_variant = None
            record_reasons.append("invalid_instance_variant")
        try:
            scene_variant = (
                SceneVariantDescriptor.from_dict(record["scene_variant"])
                if isinstance(record.get("scene_variant"), dict)
                else None
            )
        except (KeyError, TypeError, ValueError):
            scene_variant = None
            record_reasons.append("invalid_scene_variant")
        try:
            simulation = (
                SimulationProvenance.from_dict(record["simulation"])
                if isinstance(record.get("simulation"), dict)
                else None
            )
        except (KeyError, TypeError, ValueError):
            simulation = None
            record_reasons.append("invalid_simulation_provenance")
        task_family = str(record.get("task_family", default_task_family or ""))
        try:
            candidate_config = verification_configs[task_family]
        except KeyError as exc:
            available = ", ".join(sorted(verification_configs))
            raise ValueError(
                f"Candidate {record.get('traj_id', '<unknown>')} has unsupported task_family={task_family!r}; "
                f"configured: {available}."
            ) from exc
        provenance = CandidateProvenance(
            generator=str(record.get("generator", "instance_rollout")),
            model_id=str(record.get("model_id", "")),
            parent_traj_id=str(record.get("parent_traj_id", "")),
            generation_seed=(None if record.get("generation_seed") is None else int(record["generation_seed"])),
            action_path=action_path,
            robot_state_path=robot_state_path,
            context_video_path=_resolve(record.get("context_video_path"), base_dir),
            prompt=(None if record.get("prompt") is None else str(record["prompt"])),
            object_state_path=object_state_path,
            control_artifacts=controls,
            simulation=simulation,
        )
        final_relation: RelationDescriptor | None = None
        if relation_path:
            try:
                relation_sequence = load_relation_sequence(relation_path)
                final_relation = RelationDescriptor(
                    names=relation_sequence.names,
                    values=relation_sequence.values[-1].tolist(),
                    source="simulator",
                )
            except (FileNotFoundError, ValueError):
                final_relation = None
        physical = verify_candidate(
            provenance,
            load_last_robot_state(robot_state_path),
            final_relation,
            feasibility_config,
            candidate_frames=frames,
        )
        instance_result = verify_instance_candidate(
            candidate_frames=frames,
            object_state_path=object_state_path,
            relation_path=relation_path,
            controls=controls,
            instance_variant=instance_variant,
            simulation=simulation,
            config=candidate_config,
        )
        instance_verification = VerificationRecord(
            status="accepted" if instance_result.accepted else "rejected",
            checks=instance_result.checks,
            reasons=sorted(set([*record_reasons, *instance_result.reasons])),
            relation=final_relation,
        )
        if record_reasons:
            instance_verification = VerificationRecord(
                status="rejected",
                checks={**instance_verification.checks, "record_schema": False},
                reasons=instance_verification.reasons,
                relation=instance_verification.relation,
            )
        verification = _merge_verification(physical, instance_verification)
        if not record.get("goal_ref_id"):
            verification = VerificationRecord(
                status="rejected",
                checks={**verification.checks, "goal_reference": False},
                reasons=sorted(set([*verification.reasons, "missing_goal_ref_id"])),
                relation=verification.relation,
            )
        examples.append(
            TrajectoryExample(
                traj_id=str(record["traj_id"]),
                task=str(record["task"]),
                frames=frames,
                source="instance_rollout",
                split=str(record.get("split", split)),
                metadata={"candidate_index": record.get("candidate_index")},
                robot_state_path=robot_state_path,
                action_path=action_path,
                relation_path=relation_path,
                goal_ref_id=(None if record.get("goal_ref_id") is None else str(record["goal_ref_id"])),
                parent_traj_id=provenance.parent_traj_id or None,
                candidate_provenance=provenance,
                verification=verification,
                task_family=task_family,
                object_state_path=object_state_path,
                control_artifacts=controls,
                scene_variant=scene_variant,
                instance_variant=instance_variant,
                simulation=simulation,
            )
        )
    write_jsonl(output_manifest, examples)
    return examples
