"""Balanced simulator-native task seeds and physical success references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mi_reward.data.base_trajectory_schema import BaseTrajectory
from mi_reward.data.schema import SuccessReference, write_jsonl


def _load_task_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task config root must be a mapping: {path}")
    for key in ("task_family", "instruction", "instances", "scene_variants", "simulation", "planner"):
        if payload.get(key) in (None, "", []):
            raise ValueError(f"Task config {path} is missing {key!r}.")
    return payload


def prepare_simulator_native(
    config: dict[str, Any],
    output_records: str | Path,
    success_refs_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Create balanced task seeds; pixels and references are produced later."""

    raw_configs = config.get("task_configs")
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("base_data.simulator_native.task_configs must be a non-empty list.")
    seeds_per_task = int(config.get("seeds_per_task", 1))
    if seeds_per_task < 1:
        raise ValueError("base_data.simulator_native.seeds_per_task must be positive.")
    task_configs = [Path(str(value)).resolve() for value in raw_configs]
    tasks: list[tuple[Path, dict[str, Any]]] = []
    families: set[str] = set()
    instructions: set[str] = set()
    for path in task_configs:
        if not path.is_file():
            raise FileNotFoundError(f"Simulator-native task config is missing: {path}")
        payload = _load_task_config(path)
        family = str(payload["task_family"])
        instruction = str(payload["instruction"])
        if family in families:
            raise ValueError(f"Duplicate simulator-native task family: {family}")
        if instruction in instructions:
            raise ValueError(f"Duplicate simulator-native task instruction: {instruction}")
        families.add(family)
        instructions.add(instruction)
        tasks.append((path, payload))

    output_path = Path(output_records).resolve()
    reference_path = Path(success_refs_path).resolve()
    # One additional seed per task is reserved exclusively for selecting a
    # success reference. None of its instance/scene siblings enter training.
    expected_records = len(tasks) * (seeds_per_task + 1)
    report = {
        "source": "simulator_native",
        "status": "validated" if dry_run else "ready",
        "output_records": str(output_path),
        "success_refs": str(reference_path),
        "references": "deferred_to_reference_stage",
        "records": expected_records,
        "seeds_per_task": seeds_per_task,
        "training_seed_records": len(tasks) * seeds_per_task,
        "reference_seed_records": len(tasks),
        "task_families": sorted(families),
        "task_configs": [str(path) for path in task_configs],
    }
    if dry_run:
        return report

    records: list[BaseTrajectory] = []
    for task_path, payload in tasks:
        family = str(payload["task_family"])
        instruction = str(payload["instruction"])
        simulation = dict(payload["simulation"])
        goal_name = str(simulation.get("goal_body_name", "goal"))
        goal_ref_id = f"simulator_native/{family}/success"
        for seed_index in range(-1, seeds_per_task):
            reference_seed = seed_index == -1
            seed_name = "reference-seed" if reference_seed else f"seed-{seed_index:04d}"
            records.append(
                BaseTrajectory(
                    base_id=f"simulator_native/{family}/{seed_name}",
                    source="simulator_native",
                    task=instruction,
                    task_family=family,
                    instruction=instruction,
                    frames=[],
                    initial_frame="",
                    goal_frame="",
                    goal_ref_id=goal_ref_id,
                    physical_task_config=str(task_path),
                    object_prompts={
                        "task_object": "task object",
                        "goal": goal_name,
                        "robot": "robot gripper",
                    },
                    split=str(config.get("split", "train")),
                    metadata={
                        "simulator_native": True,
                        "task_seed_index": seed_index,
                        "simulator_reference_seed": reference_seed,
                        "physical_sidecars_from_source": False,
                    },
                )
            )
    write_jsonl(output_path, records)
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _terminal_success(record: dict[str, Any]) -> bool:
    relation_path = Path(str(record.get("relation_path", "")))
    if not relation_path.is_file():
        return False
    lines = [line for line in relation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    final = json.loads(lines[-1])
    if not isinstance(final, dict):
        return False
    names = [str(value) for value in final.get("names", [])]
    values = [float(value) for value in final.get("values", [])]
    if "target_satisfied" not in names or len(names) != len(values):
        return False
    return values[names.index("target_satisfied")] >= 0.5


def build_simulator_references(
    input_records: str | Path,
    output_records: str | Path,
    success_refs_path: str | Path,
) -> dict[str, object]:
    """Select one independent physical success reference per task/instance."""

    input_path = Path(input_records).resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No simulator records found in {input_path}.")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("task_family", ""))
        if not family:
            raise ValueError("Simulator candidate has no task_family.")
        config_path = Path(str(row.get("physical_task_config", ""))).resolve()
        task_config = _load_task_config(config_path)
        expected_task = str(task_config["instruction"])
        if family != str(task_config["task_family"]) or str(row.get("task", "")) != expected_task:
            raise ValueError(
                f"Candidate task semantics do not match {config_path}: "
                f"family={family!r}, task={row.get('task')!r}."
            )
        variant = row.get("instance_variant") or {}
        variant_id = str(variant.get("variant_id", "")) if isinstance(variant, dict) else ""
        if not variant_id:
            raise ValueError("Simulator candidate has no instance_variant.variant_id.")
        grouped.setdefault((family, variant_id), []).append(row)

    references: list[SuccessReference] = []
    retained: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    excluded_parent_ids: set[str] = set()
    selected_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    for (family, variant_id), candidates in sorted(grouped.items()):
        eligible = [
            item
            for item in candidates
            if bool((item.get("metadata") or {}).get("simulator_reference_candidate"))
            and _terminal_success(item)
            and isinstance(item.get("frames"), list)
            and len(item["frames"]) >= 2
            and all(Path(str(frame)).is_file() for frame in item["frames"])
        ]
        if not eligible:
            raise ValueError(
                f"Task family {family!r} instance {variant_id!r} has no successful canonical reference candidate."
            )
        selected = min(eligible, key=lambda item: (int(item.get("generation_seed", 0)), str(item["traj_id"])))
        selected_id = str(selected["traj_id"])
        selected_parent_id = str(selected.get("parent_traj_id", ""))
        if not selected_parent_id:
            raise ValueError(f"Reference candidate {selected_id} has no parent_traj_id.")
        selected_ids.add(selected_id)
        excluded_parent_ids.add(selected_parent_id)
        selected_by_group[(family, variant_id)] = selected
        ref_id = str(selected.get("goal_ref_id", ""))
        task = str(selected.get("task", ""))
        if not ref_id or not task:
            raise ValueError(f"Reference candidate {selected_id} has no task/goal_ref_id.")
        if any(str(item.get("goal_ref_id", "")) != ref_id or str(item.get("task", "")) != task for item in candidates):
            raise ValueError(
                f"Task family {family!r} instance {variant_id!r} does not share one task and goal_ref_id."
            )
        references.append(
            SuccessReference(
                ref_id=ref_id,
                task=task,
                frames=[str(value) for value in selected["frames"]],
                task_family=family,
                instance_variant_id=variant_id,
                robot_state_path=str(selected.get("robot_state_path", "")) or None,
                object_state_path=str(selected.get("object_state_path", "")) or None,
                relation_path=str(selected.get("relation_path", "")) or None,
            )
        )

    for row in rows:
        if str(row.get("parent_traj_id", "")) in excluded_parent_ids:
            continue
        metadata = dict(row.get("metadata") or {})
        family = str(row["task_family"])
        variant = row.get("instance_variant") or {}
        variant_id = str(variant.get("variant_id", ""))
        selected = selected_by_group[(family, variant_id)]
        metadata["success_reference_source_traj_id"] = str(selected["traj_id"])
        row["metadata"] = metadata
        retained.append(row)

    output_path = Path(output_records).resolve()
    refs_path = Path(success_refs_path).resolve()
    write_jsonl(output_path, retained)
    write_jsonl(refs_path, references)
    return {
        "records_path": str(output_path),
        "success_refs": str(refs_path),
        "input_candidates": len(rows),
        "training_candidates": len(retained),
        "references": len(references),
        "excluded_reference_seed_candidates": len(rows) - len(retained),
        "task_families": sorted({family for family, _ in grouped}),
        "reference_groups": [f"{family}/{variant}" for family, variant in sorted(grouped)],
        "reference_source_traj_ids": sorted(selected_ids),
    }
