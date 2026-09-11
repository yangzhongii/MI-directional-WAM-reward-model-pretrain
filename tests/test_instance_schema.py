"""CPU-only checks for the instance rollout artifact schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mi_reward.data.instance_schema import load_object_state_sequence, validate_instance_artifacts
from mi_reward.data.schema import (
    ControlArtifacts,
    InstanceVariantDescriptor,
    ObjectStateDescriptor,
    SceneVariantDescriptor,
    SimulationProvenance,
    TrajectoryExample,
    read_jsonl,
    write_jsonl,
)
from mi_reward.sim.assets import describe_asset, load_asset


def _state(frame_index: int, category: str = "banana") -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "objects": [
            {
                "object_id": "task_object",
                "category": category,
                "pose_xyz": [0.1 * frame_index, 0.0, 0.1],
                "pose_quaternion": [1.0, 0.0, 0.0, 0.0],
                "size_xyz": [0.04, 0.04, 0.08],
            }
        ],
    }


def _artifacts(root: Path, count: int = 2) -> tuple[list[str], Path, ControlArtifacts]:
    frame_dir, mask_dir, depth_dir = root / "frames", root / "masks", root / "depth"
    for directory in (frame_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True)
    frames: list[str] = []
    for index in range(count):
        frame = frame_dir / f"frame_{index:06d}.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frame)
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / f"frame_{index:06d}.png")
        np.save(depth_dir / f"frame_{index:06d}.npy", np.zeros((8, 8), dtype=np.float32))
        frames.append(str(frame))
    state_path = root / "object_states.jsonl"
    state_path.write_text("".join(json.dumps(_state(index)) + "\n" for index in range(count)), encoding="utf-8")
    return frames, state_path, ControlArtifacts(mask_root=str(mask_dir), depth_root=str(depth_dir))


def _variant() -> InstanceVariantDescriptor:
    return InstanceVariantDescriptor(
        source_object_id="task_object",
        source_category="apple",
        target_category="banana",
        asset_uri="builtin://pick_place/banana",
        task_role="task_object",
    )


def test_object_state_schema_requires_aligned_artifacts(tmp_path: Path) -> None:
    frames, state_path, controls = _artifacts(tmp_path)
    sequence = load_object_state_sequence(state_path)
    assert [frame.frame_index for frame in sequence.frames] == [0, 1]
    assert not validate_instance_artifacts(
        frames=frames,
        object_state_path=state_path,
        controls=controls,
        instance_variant=_variant(),
    )

    state_path.write_text(json.dumps(_state(1)) + "\n" + json.dumps(_state(2)) + "\n", encoding="utf-8")
    reasons = validate_instance_artifacts(
        frames=frames,
        object_state_path=state_path,
        controls=controls,
        instance_variant=_variant(),
    )
    assert "object_state_frame_indices_not_contiguous" in reasons


def test_object_state_schema_rejects_invalid_quaternion_and_duplicate_ids(tmp_path: Path) -> None:
    invalid_quaternion = _state(0)
    invalid_quaternion["objects"][0]["pose_quaternion"] = [0.0, 0.0, 0.0, 0.0]  # type: ignore[index]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([invalid_quaternion]), encoding="utf-8")
    with pytest.raises(ValueError, match="non-zero"):
        load_object_state_sequence(path)

    duplicate = _state(0)
    duplicate["objects"].append(dict(duplicate["objects"][0]))  # type: ignore[index]
    path.write_text(json.dumps([duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate object_id"):
        load_object_state_sequence(path)


def test_extended_manifest_round_trips_without_changing_legacy_fields(tmp_path: Path) -> None:
    example = TrajectoryExample(
        traj_id="instance/pick_place/candidate_0",
        task="place the banana in the basket",
        frames=["frame.png"],
        source="instance_rollout",
        split="train",
        task_family="pick_place",
        object_state_path="object_states.jsonl",
        control_artifacts=ControlArtifacts(mask_root="masks", depth_root="depth"),
        scene_variant=SceneVariantDescriptor(variant_id="table_dark_2", prompt="dark tabletop"),
        instance_variant=_variant(),
        simulation=SimulationProvenance(backend="mujoco", asset_catalog_version="rigid_v1", seed=7),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [example])
    loaded = read_jsonl(manifest, TrajectoryExample)[0]
    assert loaded.instance_variant == example.instance_variant
    assert loaded.scene_variant == example.scene_variant
    assert loaded.simulation == example.simulation


def test_object_state_descriptor_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="three values"):
        ObjectStateDescriptor(
            object_id="banana",
            category="banana",
            pose_xyz=[0.0, 0.0],
            pose_quaternion=[1.0, 0.0, 0.0, 0.0],
            size_xyz=[0.1, 0.1, 0.1],
        )


def test_rigid_asset_catalog_returns_a_simulator_ready_asset() -> None:
    asset = load_asset("pick_place", "banana")
    assert asset.category == "banana"
    assert asset.asset_uri == "builtin://pick_place/banana"
    assert describe_asset("pick_place", "banana").task_role == "target_object"
