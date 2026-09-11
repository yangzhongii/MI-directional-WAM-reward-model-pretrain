from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from mi_reward.data.base_trajectory_schema import BaseTrajectory
from mi_reward.data.schema import SuccessReference, read_jsonl
from mi_reward.data.simulator_native import (
    build_simulator_references,
    prepare_simulator_native,
)
from mi_reward.planning.rigid_task_planner import run_worker as run_planner_worker
from mi_reward.sim.mujoco_generalization_worker import run_worker as run_simulator_worker


TASK_CONFIGS = [
    Path("mi_reward/configs/tasks/pick_place_apple_banana.yaml").resolve(),
    Path("mi_reward/configs/tasks/push_t_to_b.yaml").resolve(),
    Path("mi_reward/configs/tasks/peg_insertion_variants.yaml").resolve(),
]


def test_prepare_simulator_native_is_balanced_across_task_families(tmp_path: Path) -> None:
    output = tmp_path / "base.jsonl"
    report = prepare_simulator_native(
        {"task_configs": [str(path) for path in TASK_CONFIGS], "seeds_per_task": 2},
        output,
        tmp_path / "refs.jsonl",
    )

    records = read_jsonl(output, BaseTrajectory)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.task_family] = counts.get(record.task_family, 0) + 1
        assert record.source == "simulator_native"
        assert record.frames == []
        assert record.goal_ref_id == f"simulator_native/{record.task_family}/success"
    assert counts == {"pick_place": 3, "push_shape": 3, "peg_insertion": 3}
    assert sum(bool(record.metadata["simulator_reference_seed"]) for record in records) == 3
    assert report["records"] == 9


def test_reference_stage_removes_one_independent_success_per_family(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for config_path in TASK_CONFIGS:
        task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        family = str(task_config["task_family"])
        frame_paths = []
        for frame_index in range(2):
            frame = tmp_path / family / f"frame_{frame_index}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"frame")
            frame_paths.append(str(frame))
        relation = tmp_path / family / "relations.jsonl"
        relation.write_text(
            json.dumps({"names": ["target_satisfied"], "values": [1.0]}) + "\n",
            encoding="utf-8",
        )
        for candidate_index in range(3):
            rows.append(
                {
                    "traj_id": f"{family}/candidate-{candidate_index}",
                    "parent_traj_id": (
                        f"{family}/reference-seed"
                        if candidate_index < 2
                        else f"{family}/training-seed"
                    ),
                    "task": str(task_config["instruction"]),
                    "task_family": family,
                    "goal_ref_id": f"simulator_native/{family}/train_instance/success",
                    "instance_variant": {"variant_id": "train_instance"},
                    "physical_task_config": str(config_path),
                    "generation_seed": candidate_index,
                    "frames": frame_paths,
                    "relation_path": str(relation),
                    "metadata": {"simulator_reference_candidate": candidate_index == 0},
                }
            )
    input_records = tmp_path / "physical.jsonl"
    input_records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_records = tmp_path / "candidates.jsonl"
    success_refs = tmp_path / "success_refs.jsonl"

    report = build_simulator_references(input_records, output_records, success_refs)

    retained = [json.loads(line) for line in output_records.read_text(encoding="utf-8").splitlines()]
    references = read_jsonl(success_refs, SuccessReference)
    assert len(retained) == 3
    assert len(references) == 3
    assert {item.ref_id for item in references} == {
        "simulator_native/pick_place/train_instance/success",
        "simulator_native/push_shape/train_instance/success",
        "simulator_native/peg_insertion/train_instance/success",
    }
    assert all("success_reference_source_traj_id" in item["metadata"] for item in retained)
    assert report["training_candidates"] == 3
    assert report["excluded_reference_seed_candidates"] == 6


def test_all_task_instance_reference_rollouts_are_physically_successful(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    base_records = tmp_path / "base.jsonl"
    rows = []
    for config_path in TASK_CONFIGS:
        task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rows.append({
            "base_id": f"simulator_native/{task_config['task_family']}/reference-seed",
            "task": task_config["instruction"],
            "task_family": task_config["task_family"],
            "physical_task_config": str(config_path),
            "metadata": {"simulator_reference_seed": True},
        })
    base_records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    planner_records = tmp_path / "planner.jsonl"
    planner_request = tmp_path / "planner_request.json"
    planner_request.write_text(json.dumps({
        "input_records": str(base_records), "output_records": str(planner_records)
    }), encoding="utf-8")
    run_planner_worker(planner_request, tmp_path / "planner_result.json", num_candidates=1, seed=23)
    candidates = [
        json.loads(line) for line in planner_records.read_text(encoding="utf-8").splitlines()
    ]
    references = [item for item in candidates if item["metadata"]["simulator_reference_candidate"]]
    assert len(references) == 6
    reference_input = tmp_path / "reference_input.jsonl"
    reference_input.write_text(
        "".join(json.dumps(item) + "\n" for item in references), encoding="utf-8"
    )
    physical = tmp_path / "reference_physical.jsonl"
    simulator_request = tmp_path / "simulator_request.json"
    simulator_request.write_text(json.dumps({
        "input_records": str(reference_input), "output_records": str(physical)
    }), encoding="utf-8")
    run_simulator_worker(simulator_request, tmp_path / "simulator_result.json", width=160, height=128)
    physical_rows = [json.loads(line) for line in physical.read_text(encoding="utf-8").splitlines()]
    assert len(physical_rows) == 6
    assert all(item["task_outcome"]["success"] for item in physical_rows)
