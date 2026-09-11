"""Execute planned instance variants in headless MuJoCo and export sidecars."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from mi_reward.data.instance_schema import ObjectStateFrame
from mi_reward.data.schema import ObjectStateDescriptor, SimulationProvenance
from mi_reward.relations.geometry import RobotState
from mi_reward.relations.instance_sequence import InstanceRelationFlags, build_instance_relation_descriptor
from mi_reward.relations.sequence import RelationSequence
from mi_reward.sim.base import SimulationRollout
from mi_reward.sim.rollout_builder import write_rollout_artifacts


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
        raise ValueError(f"No planner records found in {path}.")
    return records


def _name_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"MuJoCo object does not exist: {name}")
    return value


def _body_pose(model: Any, data: Any, mujoco: Any, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[body_id], copy=True), np.array(data.xquat[body_id], copy=True), body_id


def _body_size(model: Any, body_id: int) -> list[float]:
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    if count <= 0:
        return [0.01, 0.01, 0.01]
    sizes = np.asarray(model.geom_size[start : start + count], dtype=np.float64)
    return (2.0 * np.max(sizes, axis=0)).tolist()


def _geom_ids(mujoco: Any, model: Any, values: Any, field: str) -> set[int]:
    if values in (None, []):
        return set()
    if not isinstance(values, list):
        raise ValueError(f"simulation_spec.{field} must be a list of geom names.")
    return {_name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, str(name)) for name in values}


def _body_geom_ids(mujoco: Any, model: Any, values: Any, field: str) -> set[int]:
    if values in (None, []):
        return set()
    if not isinstance(values, list):
        raise ValueError(f"simulation_spec.{field} must be a list of body names.")
    result: set[int] = set()
    for name in values:
        body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        start = int(model.body_geomadr[body_id])
        result.update(range(start, start + int(model.body_geomnum[body_id])))
    return result


def _contact_pairs(data: Any) -> set[frozenset[int]]:
    return {
        frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
        for index in range(int(data.ncon))
    }


def _has_cross_contact(pairs: set[frozenset[int]], lhs: set[int], rhs: set[int]) -> bool:
    return any(frozenset((left, right)) in pairs for left in lhs for right in rhs)


def _render(
    renderer: Any,
    data: Any,
    camera: str | None,
    preserve_geoms: set[int],
    scene_option: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
    finally:
        renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        segmentation = np.asarray(renderer.render()).copy()
    finally:
        renderer.disable_segmentation_rendering()
    # MuJoCo segmentation renders ``(object_type, object_id)`` in the first
    # two channels. We need the geom ID (channel 1), not the constant type.
    geom_channel = segmentation[..., 1] if segmentation.ndim == 3 and segmentation.shape[-1] > 1 else segmentation
    if preserve_geoms:
        mask = np.isin(geom_channel, list(preserve_geoms))
    else:
        mask = geom_channel >= 0
    return rgb, depth, mask.astype(np.uint8) * 255


def _plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Cartesian plan contains no steps: {path}")
    return [dict(item) for item in steps if isinstance(item, dict)]


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo ``[w,x,y,z]`` quaternion to a 3x3 rotation matrix."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"Expected a [w,x,y,z] quaternion, got {value.shape}.")
    value = value / max(float(np.linalg.norm(value)), 1e-12)
    w, x, y, z = value
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Match Cosmos/IRASim's Rz*Ry*Rx relative-rotation convention."""

    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy >= 1e-6:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0.0
    return np.asarray([x, y, z], dtype=np.float64)


def _action(
    previous_position: np.ndarray,
    previous_quaternion: np.ndarray,
    current_position: np.ndarray,
    current_quaternion: np.ndarray,
    gripper: float,
    scale: float,
) -> np.ndarray:
    """Create one Bridge-compatible Cosmos action in the prior EE frame."""

    previous_rotation = _quaternion_to_rotation_matrix(previous_quaternion)
    current_rotation = _quaternion_to_rotation_matrix(current_quaternion)
    action = np.zeros(7, dtype=np.float32)
    action[:3] = (previous_rotation.T @ (current_position - previous_position) * scale).astype(np.float32)
    action[3:6] = (_rotation_matrix_to_euler(previous_rotation.T @ current_rotation) * scale).astype(np.float32)
    action[6] = float(gripper)
    return action


def _load_model(spec: dict[str, Any]):
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed in .venv. Run requirements/install.sh --all.") from exc
    model_path = Path(str(spec["model_path"])).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene is missing: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    keyframe = spec.get("initial_keyframe")
    if keyframe:
        key_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_KEY, str(keyframe))
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return mujoco, model, data, model_path


def _execute(record: dict[str, Any], output_dir: Path, width: int, height: int) -> dict[str, Any]:
    spec = record.get("simulation_spec")
    variant = record.get("instance_variant")
    if not isinstance(spec, dict) or not isinstance(variant, dict):
        raise ValueError("Planner record requires simulation_spec and instance_variant mappings.")
    if variant.get("source_category") != variant.get("target_category") and not spec.get("instance_model_path"):
        raise FileNotFoundError(
            "Held-out instance replacement requires instance_variant.model_path in the task YAML; "
            "a builtin:// URI is provenance only and cannot mutate a MuJoCo scene."
        )
    plan_path = Path(str(record.get("cartesian_plan_path", ""))).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Cartesian plan is missing: {plan_path}")
    steps = _plan(plan_path)
    mujoco, model, data, model_path = _load_model(spec)
    mocap_body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, str(spec["mocap_body_name"]))
    mocap_id = int(model.body_mocapid[mocap_body_id])
    if mocap_id < 0:
        raise ValueError(f"Configured mocap body is not mocap-enabled: {spec['mocap_body_name']}")
    gripper_actuator_id = None
    if spec.get("gripper_actuator_name"):
        gripper_actuator_id = _name_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(spec["gripper_actuator_name"])
        )
    object_geoms = _geom_ids(mujoco, model, spec.get("object_geom_names"), "object_geom_names")
    gripper_geoms = _geom_ids(mujoco, model, spec.get("gripper_geom_names"), "gripper_geom_names")
    preserve_geoms = _geom_ids(mujoco, model, spec.get("preserve_geom_names"), "preserve_geom_names")
    preserve_geoms.update(
        _body_geom_ids(mujoco, model, spec.get("preserve_body_names"), "preserve_body_names")
    )
    forbidden_geoms = _geom_ids(mujoco, model, spec.get("forbidden_geom_names"), "forbidden_geom_names")
    require_contact_phases = {str(value) for value in spec.get("require_contact_phases", [])}
    kinematic_task_proxy = bool(spec.get("kinematic_task_proxy", False))
    object_body_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, str(spec["task_object_body_name"]))
    object_joint_id = int(model.body_jntadr[object_body_id])
    if object_joint_id < 0:
        raise ValueError("The task object must have a joint so initial-state variants can be applied.")
    object_qpos_adr = int(model.jnt_qposadr[object_joint_id])
    object_qvel_adr = int(model.jnt_dofadr[object_joint_id])
    if kinematic_task_proxy and model.jnt_type[object_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("kinematic_task_proxy requires the task object to use a free joint.")
    initial_object_offset = np.asarray(spec.get("initial_object_offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    if initial_object_offset.shape != (3,):
        raise ValueError("simulation_spec.initial_object_offset_xyz must contain three values.")
    if np.any(initial_object_offset):
        if model.jnt_type[object_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Initial task-object offsets require a free joint.")
        data.qpos[object_qpos_adr : object_qpos_adr + 3] += initial_object_offset
        data.qvel[object_qvel_adr : object_qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
    settle_steps = int(spec.get("settle_steps", 50))
    control_steps = int(spec.get("steps_per_control", 10))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    if bool(spec.get("hide_full_robot", False)):
        # Menagerie places visual meshes in geom group 2 and collision meshes
        # in group 3. The task table/objects and proxy end-effector use group 0.
        # Hiding groups 2/3 removes the static full Panda from RGB/depth/masks
        # while preserving all physics and the moving proxy gripper.
        scene_option.geomgroup[2] = 0
        scene_option.geomgroup[3] = 0
    frames: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    robot_states: list[dict[str, Any]] = []
    object_states: list[ObjectStateFrame] = []
    relation_records = []
    ee_name = str(spec["ee_body_name"])
    object_name = str(spec["task_object_body_name"])
    goal_name = str(spec["goal_body_name"])
    ee_position, _, _ = _body_pose(model, data, mujoco, ee_name)
    action_scale = float(spec.get("cosmos_action_scale", 20.0))
    goal_distance_threshold = float(spec.get("goal_distance_threshold", 0.05))
    orientation_threshold = float(spec.get("goal_orientation_threshold", 0.35))
    task_family = str(record.get("task_family", ""))
    proxy_attached = False
    proxy_offset = np.zeros(3, dtype=np.float64)
    ee_positions: list[np.ndarray] = []
    ee_quaternions: list[np.ndarray] = []
    gripper_states: list[float] = []
    try:
        for frame_index, step in enumerate(steps):
            target = np.asarray(step["ee_position"], dtype=np.float64)
            quaternion = np.asarray(step["ee_quaternion"], dtype=np.float64)
            if target.shape != (3,) or quaternion.shape != (4,):
                raise ValueError("Every plan step requires ee_position[3] and ee_quaternion[4].")
            phase = str(step.get("phase", ""))
            start_mocap_position = np.array(data.mocap_pos[mocap_id], copy=True)
            data.mocap_quat[mocap_id] = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
            gripper = float(step.get("gripper", 0.0))
            if gripper_actuator_id is not None:
                # Planner gripper values use a normalized [0,1] contract,
                # while the MuJoCo position actuator controls tendon length in
                # metres. Map the normalized command onto the actuator range.
                ctrl_low, ctrl_high = model.actuator_ctrlrange[gripper_actuator_id]
                data.ctrl[gripper_actuator_id] = ctrl_low + gripper * (ctrl_high - ctrl_low)
            current_object_position, _, _ = _body_pose(model, data, mujoco, object_name)
            near_object = float(np.linalg.norm(current_object_position - start_mocap_position)) <= 0.10
            if kinematic_task_proxy:
                if task_family in {"pick_place", "peg_insertion"} and phase == "close" and near_object:
                    if not proxy_attached:
                        proxy_offset = current_object_position - start_mocap_position
                    proxy_attached = True
                # Bind only once the gripper has completed the contact phase.
                # Binding at the start of ``contact`` preserves the earlier
                # pre-contact offset and makes the object overshoot the goal.
                elif task_family == "push_shape" and phase == "push" and near_object:
                    if not proxy_attached:
                        proxy_offset = current_object_position - start_mocap_position
                    proxy_attached = True
                if phase in {"release", "retreat"}:
                    proxy_attached = False
            for control_index in range(control_steps):
                alpha = float(control_index + 1) / float(control_steps)
                data.mocap_pos[mocap_id] = (1.0 - alpha) * start_mocap_position + alpha * target
                if proxy_attached:
                    data.qpos[object_qpos_adr : object_qpos_adr + 3] = data.mocap_pos[mocap_id] + proxy_offset
                    data.qvel[object_qvel_adr : object_qvel_adr + 6] = 0.0
                    mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
            if proxy_attached:
                data.qpos[object_qpos_adr : object_qpos_adr + 3] = target + proxy_offset
                data.qvel[object_qvel_adr : object_qvel_adr + 6] = 0.0
                mujoco.mj_forward(model, data)
            rgb, depth, mask = _render(
                renderer,
                data,
                spec.get("camera_name"),
                preserve_geoms,
                scene_option,
            )
            frames.append(rgb)
            depths.append(depth)
            masks.append(mask)
            qpos.append(np.array(data.qpos, copy=True))
            qvel.append(np.array(data.qvel, copy=True))
            ee_position, ee_quaternion, _ = _body_pose(model, data, mujoco, ee_name)
            ee_positions.append(ee_position)
            ee_quaternions.append(ee_quaternion)
            gripper_states.append(gripper)
            object_position, object_quaternion, object_body_id = _body_pose(model, data, mujoco, object_name)
            goal_position, goal_quaternion, goal_body_id = _body_pose(model, data, mujoco, goal_name)
            pairs = _contact_pairs(data)
            object_contact = _has_cross_contact(pairs, gripper_geoms, object_geoms)
            # The object is expected to rest on the table, so any contact with
            # a forbidden geom is too broad. Only controlled gripper contacts
            # with the table/wall count as a robot collision.
            collision_free = not _has_cross_contact(pairs, gripper_geoms, forbidden_geoms)
            near_object = float(np.linalg.norm(ee_position - object_position)) <= 0.10
            contact_valid = (
                phase not in require_contact_phases
                or object_contact
                or proxy_attached
                or (phase in {"grasp", "contact"} and near_object)
            )
            distance = float(np.linalg.norm(object_position - goal_position))
            quat_dot = abs(float(np.dot(object_quaternion, goal_quaternion)))
            orientation_error = 2.0 * float(np.arccos(np.clip(quat_dot, -1.0, 1.0)))
            target_satisfied = distance <= goal_distance_threshold
            if task_family == "peg_insertion":
                target_satisfied = target_satisfied and orientation_error <= orientation_threshold
            in_gripper = (object_contact or proxy_attached) and phase in {
                "close", "lift", "transport", "place", "align", "insert"
            }
            robot = RobotState(
                joint_positions=np.asarray(data.qpos, dtype=np.float64).tolist(),
                end_effector_position=ee_position.tolist(),
                end_effector_quaternion=ee_quaternion.tolist(),
                gripper_width=gripper,
            )
            task_object = ObjectStateDescriptor(
                object_id=object_name,
                category=str(variant["target_category"]),
                pose_xyz=object_position.tolist(),
                pose_quaternion=object_quaternion.tolist(),
                size_xyz=[float(value) for value in spec.get("task_object_size_xyz", _body_size(model, object_body_id))],
                in_gripper=in_gripper,
            )
            goal = ObjectStateDescriptor(
                object_id=goal_name,
                category=str(spec.get("goal_category", "goal")),
                pose_xyz=goal_position.tolist(),
                pose_quaternion=goal_quaternion.tolist(),
                size_xyz=[float(value) for value in spec.get("goal_size_xyz", _body_size(model, goal_body_id))],
            )
            relation = build_instance_relation_descriptor(
                robot,
                task_object,
                goal,
                InstanceRelationFlags(
                    collision_free=collision_free,
                    contact_valid=contact_valid,
                    target_satisfied=target_satisfied if frame_index == len(steps) - 1 else False,
                ),
            )
            robot_states.append(
                {
                    "frame_index": frame_index,
                    "joint_positions": robot.joint_positions,
                    "end_effector_position": robot.end_effector_position,
                    "end_effector_quaternion": robot.end_effector_quaternion,
                    "gripper_width": robot.gripper_width,
                    "phase": phase,
                }
            )
            object_states.append(ObjectStateFrame(frame_index=frame_index, objects=[task_object, goal]))
            relation_records.append(relation)
    finally:
        renderer.close()

    if len(ee_positions) < 2:
        raise ValueError("Cosmos action conditioning requires at least two synchronized frames.")
    actions = [
        _action(
            ee_positions[index - 1],
            ee_quaternions[index - 1],
            ee_positions[index],
            ee_quaternions[index],
            gripper_states[index],
            action_scale,
        )
        for index in range(1, len(ee_positions))
    ]

    import torch

    relations = RelationSequence(
        names=relation_records[0].names,
        values=torch.tensor([item.values for item in relation_records], dtype=torch.float32),
    )
    simulation = SimulationProvenance(
        backend="mujoco",
        model_path=str(model_path),
        asset_catalog_version=str(spec.get("asset_catalog_version", "rigid_v1")),
        seed=(None if record.get("generation_seed") is None else int(record["generation_seed"])),
    )
    artifacts = write_rollout_artifacts(
        output_dir,
        frames=frames,
        rollout=SimulationRollout(
            qpos=np.stack(qpos),
            qvel=np.stack(qvel),
            actions=np.stack(actions),
        ),
        robot_states=robot_states,
        object_states=object_states,
        relations=relations,
        simulation=simulation,
        masks=masks,
        depths=depths,
    )
    updated = dict(record)
    metadata = dict(updated.get("metadata") or {})
    metadata["kinematic_task_proxy"] = kinematic_task_proxy
    source_controls = updated.get("control_artifacts")
    if isinstance(source_controls, dict):
        metadata["source_segmentation_control_artifacts"] = source_controls
    relation_index = {name: index for index, name in enumerate(relations.names)}
    collision_free = bool((relations.values[:, relation_index["collision_free"]] >= 0.5).all().item())
    contact_valid = bool((relations.values[:, relation_index["contact_valid"]] >= 0.5).all().item())
    target_satisfied = bool(relations.values[-1, relation_index["target_satisfied"]] >= 0.5)
    terminal_distance = float(relations.values[-1, relation_index["object_to_goal_distance"]])
    profile = str(metadata.get("candidate_profile", "unknown"))
    if collision_free and contact_valid and target_satisfied:
        failure_mode = "none"
    elif not collision_free:
        failure_mode = "collision"
    elif not contact_valid:
        failure_mode = profile if profile in {"failed_grasp", "failed_contact"} else "invalid_contact"
    elif profile in {"undershoot", "overshoot", "wrong_direction", "regress_after_progress"}:
        failure_mode = profile
    else:
        failure_mode = "goal_not_reached"
    task_outcome = {
        "success": collision_free and contact_valid and target_satisfied,
        "failure_mode": failure_mode,
        "checks": {
            "collision_free": collision_free,
            "contact_valid": contact_valid,
            "target_satisfied": target_satisfied,
        },
        "candidate_profile": profile,
        "terminal_distance": terminal_distance,
    }
    updated.update(
        {
            "frames": artifacts.frame_paths,
            "action_path": artifacts.action_path,
            "robot_state_path": artifacts.robot_state_path,
            "object_state_path": artifacts.object_state_path,
            "relation_path": artifacts.relation_path,
            "control_artifacts": asdict(artifacts.controls),
            "simulation": asdict(simulation),
            "task_outcome": task_outcome,
            "generator": "mujoco_physical_completion",
            "metadata": metadata,
        }
    )
    return updated


def run_worker(request_path: Path, result_path: Path, *, width: int, height: int) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    input_path = Path(request["input_records"]).resolve()
    output_path = Path(request["output_records"]).resolve()
    records = _read_jsonl(input_path)
    generated: list[dict[str, Any]] = []
    artifact_root = output_path.parent / "mujoco" / output_path.stem
    for index, record in enumerate(tqdm(records, desc="MuJoCo rollouts", unit="candidate")):
        generated.append(_execute(record, artifact_root / f"candidate_{index:06d}", width, height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(item) + "\n" for item in generated), encoding="utf-8")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"records_path": str(output_path)}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute planned rigid-task candidates in headless MuJoCo.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("Render dimensions must be positive.")
    run_worker(Path(args.request), Path(args.result), width=args.width, height=args.height)


if __name__ == "__main__":
    main()
