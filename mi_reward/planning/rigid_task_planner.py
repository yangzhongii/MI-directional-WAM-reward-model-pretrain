"""Rule-based Cartesian planners for the supported rigid task families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm


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
        raise ValueError(f"No segmentation-stage records found in {path}.")
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


def _scene_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("scene_variants")
    if not isinstance(raw, list) or not raw:
        return [{"variant_id": "default", "model_id": "mujoco_native"}]
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        normalized = {"variant_id": item} if isinstance(item, str) else dict(item) if isinstance(item, dict) else None
        if normalized is None or not normalized.get("variant_id"):
            raise ValueError("Every scene variant must be a string or a mapping with variant_id.")
        variant_id = str(normalized["variant_id"])
        if variant_id in seen:
            raise ValueError(f"Duplicate scene variant_id: {variant_id}")
        seen.add(variant_id)
        normalized["variant_id"] = variant_id
        normalized.setdefault("model_id", "mujoco_native")
        variants.append(normalized)
    return variants


def _scene_model_path(
    config: dict[str, Any],
    instance: dict[str, Any],
    scene: dict[str, Any],
) -> Path:
    scene_id = str(scene["variant_id"])
    model_paths = instance.get("scene_model_paths")
    if model_paths is not None and not isinstance(model_paths, dict):
        raise ValueError(f"instance {instance.get('variant_id')} scene_model_paths must be a mapping.")
    model_value = (
        (model_paths or {}).get(scene_id)
        or scene.get("model_path")
        or instance.get("model_path")
        or config.get("simulation", {}).get("model_path")
    )
    return Path(str(_required({"model_path": model_value}, "model_path"))).resolve()


def _load_scene(
    config: dict[str, Any],
    variant: dict[str, Any],
    scene: dict[str, Any],
) -> tuple[Any, Any, Any, Path]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed in .venv. Run requirements/install.sh --all.") from exc
    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Physical task config requires a simulation mapping.")
    model_path = _scene_model_path(config, variant, scene)
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


def _candidate_profiles(planner: dict[str, Any], count: int) -> list[str]:
    raw = planner.get("candidate_profiles", ["direct_success"])
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) and item for item in raw):
        raise ValueError("planner.candidate_profiles must be a non-empty string list.")
    return [str(raw[index % len(raw)]) for index in range(count)]


def _split_name(instance: dict[str, Any], scene: dict[str, Any]) -> str:
    instance_heldout = str(instance.get("split", "train")) != "train"
    scene_heldout = str(scene.get("split", "train")) != "train"
    if instance_heldout and scene_heldout:
        return "joint_heldout"
    if instance_heldout:
        return "instance_heldout"
    if scene_heldout:
        return "scene_heldout"
    return "train"


def _profiled_points(
    task_family: str,
    profile: str,
    object_position: np.ndarray,
    goal_position: np.ndarray,
    start_position: np.ndarray,
    planner: dict[str, Any],
) -> list[tuple[str, np.ndarray, float]]:
    """Create controlled successes and physically measurable near misses."""

    obj = np.array(object_position, copy=True)
    goal = np.array(goal_position, copy=True)
    delta = goal - obj
    if profile == "undershoot":
        goal = obj + 0.55 * delta
    elif profile == "overshoot":
        goal = obj + 1.45 * delta
    elif profile == "wrong_direction":
        goal = obj - 0.35 * delta
    elif profile in {"failed_grasp", "failed_contact"}:
        obj[:2] += np.asarray([0.16, -0.16], dtype=np.float64)

    points = _phase_points(task_family, obj, goal, start_position, planner)
    if profile == "delayed_success":
        wait = np.array(start_position, copy=True)
        points = [points[0], ("wait", wait, points[0][2]), ("wait", wait, points[0][2]), *points[1:]]
    elif profile == "detour_success":
        detour = (object_position + goal_position) * 0.5
        detour[2] = max(float(object_position[2]), float(goal_position[2])) + float(planner.get("hover_height", 0.10))
        detour[:2] += np.asarray([0.10, -0.08])
        anchor = "transport" if task_family == "pick_place" else "align" if task_family == "peg_insertion" else "approach"
        index = next((i for i, item in enumerate(points) if item[0] == anchor), 1)
        points.insert(index, ("detour", detour, points[max(index - 1, 0)][2]))
    elif profile == "regress_after_progress":
        regress = object_position + 0.25 * (goal_position - object_position)
        regress[2] = goal_position[2]
        terminal_phase = "release" if task_family in {"pick_place", "peg_insertion"} else "retreat"
        index = next((i for i, item in enumerate(points) if item[0] == terminal_phase), len(points) - 1)
        points.insert(index, ("regress", regress, points[max(index - 1, 0)][2]))
    return points


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
    # Include the shard output stem so two distributed planner workers sharing
    # one filesystem never overwrite each other's candidate plans.
    output_root = output_path.parent / "plans" / output_path.stem
    records = _read_jsonl(input_path)
    total_candidates = 0
    for record in records:
        config_path = Path(str(record.get("physical_task_config", ""))).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Physical task config is missing: {config_path}")
        task_config = _load_yaml(config_path)
        total_candidates += len(_variants(task_config)) * len(_scene_variants(task_config)) * num_candidates
    progress = tqdm(total=total_candidates, desc="Rigid-task planning", unit="candidate")
    generated: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        config_path = Path(str(record.get("physical_task_config", ""))).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Physical task config is missing: {config_path}")
        config = _load_yaml(config_path)
        task_family = str(record.get("task_family") or config.get("task_family") or "")
        simulation = config.get("simulation")
        planner = config.get("planner")
        if not isinstance(simulation, dict) or not isinstance(planner, dict):
            raise ValueError(f"{config_path} requires simulation and planner mappings.")
        profiles = _candidate_profiles(planner, num_candidates)
        for variant_index, variant in enumerate(_variants(config)):
            for scene_index, scene in enumerate(_scene_variants(config)):
                mujoco, model, data, model_path = _load_scene(config, variant, scene)
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
                context_seed = (
                    seed
                    + record_index * 100000
                    + variant_index * 10000
                    + scene_index * 1000
                )
                initial_rng = np.random.default_rng(context_seed)
                initial_jitter = float(planner.get("initial_object_jitter_xy", 0.03))
                shared_initial_object_offset = np.zeros(3, dtype=np.float64)
                if initial_jitter > 0:
                    shared_initial_object_offset[:2] = initial_rng.uniform(
                        -initial_jitter, initial_jitter, size=2
                    )
                for candidate_index in range(num_candidates):
                    profile = profiles[candidate_index]
                    candidate_seed = context_seed + candidate_index
                    rng = np.random.default_rng(candidate_seed)
                    # Candidate profiles are counterfactual plans from one
                    # shared physical initial state. Candidate-specific
                    # randomness may alter goals/plans, never object spawn.
                    initial_object_offset = shared_initial_object_offset.copy()
                    candidate_object = object_position + initial_object_offset
                    jitter = float(planner.get("candidate_jitter_xy", 0.0))
                    candidate_goal = goal_position.copy()
                    if jitter > 0:
                        candidate_goal[:2] += rng.uniform(-jitter, jitter, size=2)
                    points = _profiled_points(
                        task_family,
                        profile,
                        candidate_object,
                        candidate_goal,
                        start_position,
                        planner,
                    )
                    plan = _interpolate_plan(points, quaternion, int(planner.get("steps_per_segment", 4)))
                    if not plan:
                        raise ValueError("Planner produced an empty plan.")
                    base_id = str(record.get("base_id") or record.get("traj_id") or f"record_{record_index:06d}")
                    variant_id = str(variant["variant_id"])
                    scene_id = str(scene["variant_id"])
                    traj_id = (
                        f"{base_id}/instance-{variant_id}/scene-{scene_id}/candidate-{candidate_index:03d}"
                    )
                    plan_path = (
                        output_root
                        / f"record_{record_index:06d}"
                        / variant_id
                        / scene_id
                        / f"candidate_{candidate_index:03d}_{profile}.json"
                    )
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_path.write_text(json.dumps({"steps": plan}, indent=2), encoding="utf-8")
                    updated = dict(record)
                    metadata = dict(updated.get("metadata") or {})
                    metadata.update(
                        {
                            "instance_variant_id": variant_id,
                            "scene_variant_id": scene_id,
                            "planner_candidate_index": candidate_index,
                            "candidate_profile": profile,
                            "initial_state_seed": context_seed,
                            "expected_task_success": profile in {
                                "direct_success", "detour_success", "delayed_success"
                            },
                            "simulator_reference_candidate": (
                                bool(metadata.get("simulator_reference_seed"))
                                and scene_index == 0
                                and profile == "direct_success"
                            ),
                        }
                    )
                    updated.update(
                        {
                            "traj_id": traj_id,
                            "parent_traj_id": base_id,
                            "generation_seed": candidate_seed,
                            "split": _split_name(variant, scene),
                            "goal_ref_id": f"simulator_native/{task_family}/{variant_id}/success",
                            "cartesian_plan_path": str(plan_path.resolve()),
                            "scene_variant": {
                                "variant_id": scene_id,
                                "split": str(scene.get("split", "train")),
                                "prompt": scene.get("prompt"),
                                "model_id": str(scene.get("model_id", "mujoco_native")),
                                "seed": int(scene.get("seed", candidate_seed)),
                            },
                            "instance_variant": {
                                "variant_id": variant_id,
                                "source_object_id": str(simulation["task_object_body_name"]),
                                "source_category": str(variant["source_category"]),
                                "target_category": str(variant["target_category"]),
                                "asset_uri": str(variant["asset_uri"]),
                                "task_role": str(variant.get("task_role", "task_object")),
                            },
                            "simulation_spec": {
                                **simulation,
                                "initial_object_offset_xyz": initial_object_offset.tolist(),
                                "model_path": str(model_path),
                                "instance_model_path": str(model_path),
                                "asset_catalog_version": str(config.get("asset_catalog_version", "rigid_v1")),
                            },
                            "metadata": metadata,
                        }
                    )
                    generated.append(updated)
                    progress.update(1)

    progress.close()

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
