"""Rule-based Cartesian planners for the supported rigid task families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}.")
        records.append(item)
    if not records:
        raise ValueError(f"No SAM3 records found in {path}.")
    return records


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Physical task config must be a mapping: {path}")
    return payload


def _required(mapping: dict[str, Any], name: str) -> Any:
    value = mapping.get(name)
    if value is None or value == "":
        raise ValueError(f"Physical task config is missing {name!r}.")
    return value


def _body_pose(model: Any, data: Any, mujoco: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body does not exist: {name}")
    return np.array(data.xpos[body_id], copy=True), np.array(data.xquat[body_id], copy=True)


def _load_scene(config: dict[str, Any], variant: dict[str, Any]) -> tuple[Any, Any, Any, Path]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed in .venv. Run requirements/install.sh --all.") from exc
    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Physical task config requires a simulation mapping.")
    model_value = variant.get("model_path") or simulation.get("model_path")
    model_path = Path(str(_required({"model_path": model_value}, "model_path"))).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo scene is missing: {model_path}. Supply the task XML and meshes before data generation."
        )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    keyframe = simulation.get("initial_keyframe")
    if keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, str(keyframe))
        if key_id < 0:
            raise ValueError(f"MuJoCo keyframe does not exist: {keyframe}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return mujoco, model, data, model_path


def _phase_points(
    task_family: str,
    object_position: np.ndarray,
    goal_position: np.ndarray,
    start_position: np.ndarray,
    planner: dict[str, Any],
) -> list[tuple[str, np.ndarray, float]]:
    hover = float(planner.get("hover_height", 0.10))
    open_value = float(planner.get("gripper_open", 1.0))
    closed_value = float(planner.get("gripper_closed", 0.0))
    object_offset = np.asarray(planner.get("object_offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    goal_offset = np.asarray(planner.get("goal_offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    obj = object_position + object_offset
    goal = goal_position + goal_offset
    if task_family == "pick_place":
        return [
            ("start", start_position, open_value),
            ("approach", obj + [0.0, 0.0, hover], open_value),
            ("grasp", obj, open_value),
            ("close", obj, closed_value),
            ("lift", obj + [0.0, 0.0, hover], closed_value),
            ("transport", goal + [0.0, 0.0, hover], closed_value),
            ("place", goal, closed_value),
            ("release", goal, open_value),
            ("retreat", goal + [0.0, 0.0, hover], open_value),
        ]
    if task_family == "push_shape":
        delta = goal - obj
        delta[2] = 0.0
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            raise ValueError("Push task object and goal positions are identical.")
        direction = delta / norm
        approach_distance = float(planner.get("approach_distance", 0.08))
        contact_distance = float(planner.get("contact_distance", 0.01))
        z = obj[2] + float(planner.get("push_height_offset", 0.0))
        pre_contact = obj - direction * approach_distance
        pre_contact[2] = z
        contact = obj - direction * contact_distance
        contact[2] = z
        push_end = goal.copy()
        push_end[2] = z
        return [
            ("start", start_position, open_value),
            ("approach", pre_contact, open_value),
            ("contact", contact, closed_value),
            ("push", push_end, closed_value),
            ("retreat", push_end + [0.0, 0.0, hover], open_value),
        ]
    if task_family == "peg_insertion":
        insertion_depth = float(planner.get("insertion_depth", 0.03))
        aligned = goal + [0.0, 0.0, hover]
        inserted = goal + [0.0, 0.0, -insertion_depth]
        return [
            ("start", start_position, open_value),
            ("approach_object", obj + [0.0, 0.0, hover], open_value),
            ("grasp", obj, open_value),
            ("close", obj, closed_value),
            ("lift", obj + [0.0, 0.0, hover], closed_value),
            ("align", aligned, closed_value),
            ("insert", inserted, closed_value),
            ("release", inserted, open_value),
            ("retreat", aligned, open_value),
        ]
    raise ValueError(f"Unsupported task_family: {task_family}")


def _interpolate_plan(
    points: list[tuple[str, np.ndarray, float]],
    quaternion: np.ndarray,
    steps_per_segment: int,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for segment_index, ((_, start, start_gripper), (phase, end, end_gripper)) in enumerate(zip(points, points[1:])):
        for local_index in range(steps_per_segment):
            alpha = float(local_index + 1) / float(steps_per_segment)
            position = (1.0 - alpha) * start + alpha * end
            gripper = (1.0 - alpha) * start_gripper + alpha * end_gripper
            plan.append(
                {
                    "frame_index": len(plan),
                    "segment_index": segment_index,
                    "phase": phase,
                    "ee_position": position.tolist(),
                    "ee_quaternion": quaternion.tolist(),
                    "gripper": float(gripper),
                }
            )
    return plan


def _variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("instances")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Physical task config requires a non-empty instances list.")
    required = {"variant_id", "source_category", "target_category", "asset_uri"}
    for item in variants:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Every instance variant requires {sorted(required)}.")
    return variants


def run_worker(
    request_path: Path,
    result_path: Path,
    *,
    num_candidates: int,
    seed: int,
) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    output_root = output_path.parent / "plans"
    generated: list[dict[str, Any]] = []
    for record_index, record in enumerate(_read_jsonl(input_path)):
        config_path = Path(str(record.get("physical_task_config", ""))).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Physical task config is missing: {config_path}")
        config = _load_yaml(config_path)
        task_family = str(record.get("task_family") or config.get("task_family") or "")
        simulation = config.get("simulation")
        planner = config.get("planner")
        if not isinstance(simulation, dict) or not isinstance(planner, dict):
            raise ValueError(f"{config_path} requires simulation and planner mappings.")
        for variant_index, variant in enumerate(_variants(config)):
            mujoco, model, data, model_path = _load_scene(config, variant)
            object_position, _ = _body_pose(model, data, mujoco, str(_required(simulation, "task_object_body_name")))
            goal_position, _ = _body_pose(model, data, mujoco, str(_required(simulation, "goal_body_name")))
            start_position, start_quaternion = _body_pose(
                model,
                data,
                mujoco,
                str(simulation.get("mocap_body_name") or _required(simulation, "ee_body_name")),
            )
            configured_quaternion = planner.get("ee_quaternion")
            quaternion = (
                start_quaternion
                if configured_quaternion is None
                else np.asarray(configured_quaternion, dtype=np.float64)
            )
            if quaternion.shape != (4,):
                raise ValueError("planner.ee_quaternion must contain [w, x, y, z].")
            quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
            for candidate_index in range(num_candidates):
                candidate_seed = seed + record_index * 10000 + variant_index * 1000 + candidate_index
                rng = np.random.default_rng(candidate_seed)
                jitter = float(planner.get("candidate_jitter_xy", 0.0))
                candidate_goal = goal_position.copy()
                if jitter > 0:
                    candidate_goal[:2] += rng.uniform(-jitter, jitter, size=2)
                points = _phase_points(task_family, object_position, candidate_goal, start_position, planner)
                plan = _interpolate_plan(points, quaternion, int(planner.get("steps_per_segment", 4)))
                if not plan:
                    raise ValueError("Planner produced an empty plan.")
                base_id = str(record.get("base_id") or record.get("traj_id") or f"record_{record_index:06d}")
                variant_id = str(variant["variant_id"])
                traj_id = f"{base_id}/instance-{variant_id}/candidate-{candidate_index:03d}"
                plan_path = output_root / f"record_{record_index:06d}" / variant_id / f"candidate_{candidate_index:03d}.json"
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(json.dumps({"steps": plan}, indent=2), encoding="utf-8")
                updated = dict(record)
                updated.update(
                    {
                        "traj_id": traj_id,
                        "parent_traj_id": base_id,
                        "generation_seed": candidate_seed,
                        "cartesian_plan_path": str(plan_path.resolve()),
                        "instance_variant": {
                            "source_object_id": str(simulation["task_object_body_name"]),
                            "source_category": str(variant["source_category"]),
                            "target_category": str(variant["target_category"]),
                            "asset_uri": str(variant["asset_uri"]),
                            "task_role": str(variant.get("task_role", "task_object")),
                        },
                        "simulation_spec": {
                            **simulation,
                            "model_path": str(model_path),
                            "instance_model_path": (
                                None
                                if variant.get("model_path") is None
                                else str(Path(str(variant["model_path"])).resolve())
                            ),
                            "asset_catalog_version": str(config.get("asset_catalog_version", "rigid_v1")),
                        },
                    }
                )
                generated.append(updated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"records_path": str(output_path)}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan rigid manipulation phases from local MuJoCo task scenes.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_candidates < 1:
        parser.error("--num-candidates must be at least one.")
    run_worker(Path(args.request), Path(args.result), num_candidates=args.num_candidates, seed=args.seed)


if __name__ == "__main__":
    main()
