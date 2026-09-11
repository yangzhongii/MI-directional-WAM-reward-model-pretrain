from __future__ import annotations

import json
from pathlib import Path

import pytest

from mi_reward.planning.rigid_task_planner import _scene_variants, run_worker
from mi_reward.sim.mujoco_generalization_worker import run_worker as run_simulator_worker


def test_scene_variants_normalize_legacy_strings_and_mappings() -> None:
    variants = _scene_variants(
        {
            "scene_variants": [
                "table_light_1",
                {"variant_id": "table_dark_2", "seed": 2026},
            ]
        }
    )
    assert [item["variant_id"] for item in variants] == ["table_light_1", "table_dark_2"]
    assert all(item["model_id"] == "mujoco_native" for item in variants)


def test_planner_crosses_instance_scene_and_candidate_axes(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    task_config = Path("mi_reward/configs/tasks/pick_place_apple_banana.yaml").resolve()
    input_records = tmp_path / "segmentation_records.jsonl"
    input_records.write_text(
        json.dumps(
            {
                "base_id": "base/episode_0001",
                "task": "pick and place",
                "task_family": "pick_place",
                "physical_task_config": str(task_config),
                "goal_ref_id": "success_0001",
                "metadata": {"simulator_reference_seed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_records = tmp_path / "planner_records.jsonl"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"input_records": str(input_records), "output_records": str(output_records)}),
        encoding="utf-8",
    )

    run_worker(request, tmp_path / "result.json", num_candidates=8, seed=7)

    records = [json.loads(line) for line in output_records.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2 * 2 * 8
    assert {item["scene_variant"]["variant_id"] for item in records} == {
        "table_light_1",
        "table_dark_2",
    }
    assert {item["instance_variant"]["target_category"] for item in records} == {"apple", "banana"}
    assert len({item["traj_id"] for item in records}) == len(records)
    assert all("/scene-" in item["traj_id"] for item in records)
    assert all(Path(item["simulation_spec"]["model_path"]).is_file() for item in records)
    assert {item["instance_variant"]["variant_id"] for item in records} == {
        "apple_train",
        "banana_heldout",
    }
    canonical = [item for item in records if item["metadata"]["simulator_reference_candidate"]]
    assert len(canonical) == 2
    assert {item["instance_variant"]["variant_id"] for item in canonical} == {
        "apple_train", "banana_heldout"
    }
    assert {item["scene_variant"]["variant_id"] for item in canonical} == {"table_light_1"}
    assert {item["split"] for item in records} == {
        "train", "instance_heldout", "scene_heldout", "joint_heldout"
    }
    assert {item["metadata"]["candidate_profile"] for item in records} == {
        "direct_success", "detour_success", "delayed_success", "undershoot",
        "overshoot", "wrong_direction", "failed_grasp", "regress_after_progress",
    }
    contexts: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in records:
        key = (
            item["instance_variant"]["variant_id"],
            item["scene_variant"]["variant_id"],
        )
        contexts.setdefault(key, []).append(item)
    assert all(len(items) == 8 for items in contexts.values())
    for items in contexts.values():
        assert len({tuple(item["simulation_spec"]["initial_object_offset_xyz"]) for item in items}) == 1
        assert len({item["metadata"]["initial_state_seed"] for item in items}) == 1


def test_simulator_emits_measured_success_and_near_miss_outcomes(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    task_config = Path("mi_reward/configs/tasks/pick_place_apple_banana.yaml").resolve()
    base = tmp_path / "base.jsonl"
    base.write_text(json.dumps({
        "base_id": "base/train",
        "task": "pick up the task object and place it in the basket",
        "task_family": "pick_place",
        "physical_task_config": str(task_config),
        "metadata": {"simulator_reference_seed": False},
    }) + "\n", encoding="utf-8")
    planned = tmp_path / "planned.jsonl"
    planner_request = tmp_path / "planner_request.json"
    planner_request.write_text(json.dumps({
        "input_records": str(base), "output_records": str(planned)
    }), encoding="utf-8")
    run_worker(planner_request, tmp_path / "planner_result.json", num_candidates=8, seed=11)
    rows = [json.loads(line) for line in planned.read_text(encoding="utf-8").splitlines()]
    selected = [
        item for item in rows
        if item["instance_variant"]["variant_id"] == "apple_train"
        and item["scene_variant"]["variant_id"] == "table_light_1"
    ]
    simulator_input = tmp_path / "simulator_input.jsonl"
    simulator_input.write_text("".join(json.dumps(item) + "\n" for item in selected), encoding="utf-8")
    simulator_output = tmp_path / "pilot_physical.jsonl"
    simulator_request = tmp_path / "simulator_request.json"
    simulator_request.write_text(json.dumps({
        "input_records": str(simulator_input), "output_records": str(simulator_output)
    }), encoding="utf-8")
    run_simulator_worker(
        simulator_request, tmp_path / "simulator_result.json", width=160, height=128
    )
    outcomes = {
        item["metadata"]["candidate_profile"]: item["task_outcome"]
        for item in (
            json.loads(line) for line in simulator_output.read_text(encoding="utf-8").splitlines()
        )
    }
    assert outcomes["direct_success"]["success"] is True
    assert outcomes["detour_success"]["success"] is True
    assert outcomes["delayed_success"]["success"] is True
    for profile in {
        "undershoot", "overshoot", "wrong_direction", "failed_grasp", "regress_after_progress"
    }:
        assert outcomes[profile]["success"] is False
    assert outcomes["undershoot"]["failure_mode"] == "undershoot"
