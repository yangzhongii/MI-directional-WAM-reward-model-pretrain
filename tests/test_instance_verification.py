"""Acceptance and rejection contracts for rigid instance rollouts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mi_reward.data.cosmos_action_cond import feasibility_config_from_dict
from mi_reward.data.instance_rollout import ingest_instance_rollout_candidates
from mi_reward.data.instance_schema import ObjectStateFrame
from mi_reward.data.schema import InstanceVariantDescriptor, ObjectStateDescriptor, SimulationProvenance
from mi_reward.relations.sequence import RelationSequence
from mi_reward.sim.base import SimulationRollout
from mi_reward.sim.rollout_builder import write_rollout_artifacts
from mi_reward.verification.instance_checks import InstanceVerificationConfig


NAMES = [
    "object_to_goal_distance",
    "ee_to_object_distance",
    "object_goal_orientation_error",
    "collision_free",
    "contact_valid",
    "target_satisfied",
]


def _make_candidate(
    root: Path,
    *,
    collision: bool = False,
    contact: bool = True,
    terminal: bool = True,
    pose_step: float = 0.1,
):
    count = 3
    object_states = [
        ObjectStateFrame(
            frame_index=index,
            objects=[
                ObjectStateDescriptor(
                    object_id="task_object",
                    category="banana",
                    pose_xyz=[pose_step * index, 0.0, 0.1],
                    pose_quaternion=[1.0, 0.0, 0.0, 0.0],
                    size_xyz=[0.04, 0.04, 0.08],
                    in_gripper=index == 1,
                )
            ],
        )
        for index in range(count)
    ]
    values = torch.tensor(
        [
            [0.4, 0.4, 0.2, 1.0, float(contact), 0.0],
            [0.2, 0.2, 0.1, float(not collision), float(contact), 0.0],
            [0.01, 0.1, 0.01, 1.0, float(contact), float(terminal)],
        ],
        dtype=torch.float32,
    )
    artifacts = write_rollout_artifacts(
        root / "candidate",
        frames=np.zeros((count, 12, 12, 3), dtype=np.uint8),
        masks=np.zeros((count, 12, 12), dtype=np.uint8),
        depths=np.zeros((count, 12, 12), dtype=np.float32),
        rollout=SimulationRollout(
            qpos=np.zeros((count, 2), dtype=np.float32),
            qvel=np.zeros((count, 2), dtype=np.float32),
            actions=np.zeros((count, 2), dtype=np.float32),
        ),
        robot_states=[
            {"joint_positions": [0.0, 0.1], "end_effector_position": [0.0, 0.0, 0.2]}
            for _ in range(count)
        ],
        object_states=object_states,
        relations=RelationSequence(names=NAMES, values=values),
        simulation=SimulationProvenance(backend="mujoco", asset_catalog_version="rigid_v1", seed=3),
    )
    variant = InstanceVariantDescriptor(
        source_object_id="task_object",
        source_category="apple",
        target_category="banana",
        asset_uri="builtin://pick_place/banana",
        task_role="task_object",
    )
    record = {
        "traj_id": "instance/pick_place/candidate_0",
        "task": "place the banana in the basket",
        "task_family": "pick_place",
        "frames": artifacts.frame_paths,
        "goal_ref_id": "pick_place/success",
        "parent_traj_id": "recorded/pick_place/episode_0",
        "model_id": "2B/robot/action-cond",
        "action_path": artifacts.action_path,
        "robot_state_path": artifacts.robot_state_path,
        "object_state_path": artifacts.object_state_path,
        "relation_path": artifacts.relation_path,
        "control_artifacts": asdict(artifacts.controls),
        "instance_variant": asdict(variant),
        "simulation": {"backend": "mujoco", "asset_catalog_version": "rigid_v1", "seed": 3},
    }
    records = root / "records.jsonl"
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return records, record, artifacts


def _ingest(records: Path, output: Path):
    return ingest_instance_rollout_candidates(
        records,
        output,
        feasibility_config_from_dict(
            {
                "required_relation_names": ["object_to_goal_distance", "ee_to_object_distance"],
                "max_object_to_goal_distance": 0.1,
            }
        ),
        InstanceVerificationConfig(task_family="pick_place", max_pose_step=0.25),
    )[0]


def test_instance_rollout_accepts_synchronized_valid_sidecars(tmp_path: Path) -> None:
    records, _, _ = _make_candidate(tmp_path)
    example = _ingest(records, tmp_path / "manifest.jsonl")
    assert example.verification is not None
    assert example.verification.accepted
    assert example.source == "instance_rollout"


def test_instance_rollout_keeps_artifact_valid_task_failures_as_supervision(tmp_path: Path) -> None:
    for name, settings, failed_check in (
        ("collision", {"collision": True}, "collision_free"),
        ("contact", {"contact": False}, "contact_valid"),
        ("goal", {"terminal": False}, "target_satisfied"),
    ):
        records, _, _ = _make_candidate(tmp_path / name, **settings)
        example = _ingest(records, tmp_path / name / "manifest.jsonl")
        assert example.verification is not None
        assert example.verification.accepted
        assert example.task_outcome is not None
        assert not example.task_outcome.success
        assert not example.task_outcome.checks[failed_check]


def test_instance_rollout_rejects_pose_and_relation_alignment_failures(tmp_path: Path) -> None:
    records, _, artifacts = _make_candidate(tmp_path / "pose", pose_step=0.8)
    example = _ingest(records, tmp_path / "pose" / "manifest.jsonl")
    assert example.verification is not None
    assert "object_pose_discontinuity" in example.verification.reasons

    records, _, artifacts = _make_candidate(tmp_path / "alignment")
    relation_lines = Path(artifacts.relation_path).read_text(encoding="utf-8").splitlines()
    Path(artifacts.relation_path).write_text("\n".join(relation_lines[:2]) + "\n", encoding="utf-8")
    example = _ingest(records, tmp_path / "alignment" / "manifest.jsonl")
    assert example.verification is not None
    assert "relation_frame_count_mismatch" in example.verification.reasons


def test_instance_rollout_requires_an_existing_action_sidecar(tmp_path: Path) -> None:
    records, _, artifacts = _make_candidate(tmp_path)
    Path(artifacts.action_path).unlink()
    example = _ingest(records, tmp_path / "manifest.jsonl")
    assert example.verification is not None
    assert "missing_action_or_robot_state_context" in example.verification.reasons


def test_cosmos_instance_rollout_requires_visual_consistency_gate(tmp_path: Path) -> None:
    records, record, _ = _make_candidate(tmp_path)
    record["generator"] = "cosmos_predict2_5_robot_action_cond"
    record["visual_consistency"] = {
        "passed": False,
        "checks": {"motion_magnitude": False},
        "metrics": {"motion_ratio": 30.0},
    }
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    example = _ingest(records, tmp_path / "manifest.jsonl")

    assert example.verification is not None
    assert not example.verification.accepted
    assert not example.verification.checks["visual_rollout_consistency"]
    assert "visual_rollout_inconsistent_with_simulation" in example.verification.reasons
    assert example.metadata["visual_consistency"]["metrics"]["motion_ratio"] == 30.0
