"""Collect same-anchor LIBERO probes for Pipeline-v5 local MI geometry.

Every candidate begins from a freshly restored recorded MuJoCo state.  This
script only collects RGB and privileged geometric evidence; it does not load a
reward model, estimate MI, or select an action from the observed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from mi_reward.closed_loop.libero_env import canonical_camera_image
from mi_reward.data.libero_privileged import goal_objects, resolve_libero_task


VIEWS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "wrist"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--reference-demo-index", type=int, default=0)
    parser.add_argument("--anchor-demo-indices", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--anchors-per-demo", type=int, default=4)
    parser.add_argument("--reference-frame", type=int, required=True,
                        help="Explicit independently selected pre-grasp reference frame; automatic frame-0 selection is forbidden.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.012)
    parser.add_argument("--epsilon-secondary", type=float, default=0.006)
    parser.add_argument("--heldout-actions", type=int, default=24)
    parser.add_argument("--heldout-radius", type=float, default=0.012)
    parser.add_argument("--gripper-command", type=float, default=-1.0)
    parser.add_argument("--max-object-displacement", type=float, default=2e-4)
    parser.add_argument("--restore-tolerance", type=float, default=1e-8)
    parser.add_argument("--anchor-min-progress", type=float, default=0.25)
    parser.add_argument("--anchor-max-progress", type=float, default=0.60)
    parser.add_argument("--anchor-min-eef-object-distance", type=float, default=0.08)
    parser.add_argument("--anchor-max-eef-object-distance", type=float, default=0.14)
    parser.add_argument("--anchor-target-progress", type=float, default=0.42)
    parser.add_argument("--anchor-target-eef-object-distance", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _physical(env) -> dict[str, Any]:
    core = env.env
    _, task_object, goal_object = goal_objects(core.parsed_problem["goal_state"])
    task_state = core.object_states_dict[task_object]
    goal_state = core.object_states_dict[goal_object]
    task_geom = task_state.get_geom_state()
    goal_geom = goal_state.get_geom_state()
    eef = np.asarray(core._eef_xpos, dtype=np.float64)
    obj = np.asarray(task_geom["pos"], dtype=np.float64)
    goal = np.asarray(goal_geom["pos"], dtype=np.float64)
    try:
        grasped = bool(core._check_grasp(core.robots[0].gripper, core.get_object(task_object)))
    except Exception:
        grasped = False
    try:
        object_goal_contact = bool(task_state.check_contact(goal_state))
    except Exception:
        object_goal_contact = False
    try:
        success = bool(core._check_success())
    except Exception:
        success = bool(env.check_success())
    return {
        "eef_pos": eef.tolist(),
        "task_object_pos": obj.tolist(),
        "goal_object_pos": goal.tolist(),
        "eef_object_distance": float(np.linalg.norm(eef - obj)),
        "object_goal_distance": float(np.linalg.norm(obj - goal)),
        "grasped": grasped,
        "object_goal_contact": object_goal_contact,
        "success": success,
    }


def _save_rgb(observation: dict[str, Any], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for key, name in VIEWS:
        path = directory / f"{name}.png"
        Image.fromarray(canonical_camera_image(observation, key)).save(path)
        paths[name] = str(path)
    return paths


def _restore(env, state: np.ndarray, tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    env.reset()
    observation = env.set_init_state(state)
    physical = _physical(env)
    return observation, physical


def _restore_matches(expected: dict[str, Any], actual: dict[str, Any], tolerance: float) -> tuple[bool, dict[str, float]]:
    deviations = {
        key: float(np.max(np.abs(np.asarray(expected[key]) - np.asarray(actual[key]))))
        for key in ("eef_pos", "task_object_pos", "goal_object_pos")
    }
    same_flags = all(bool(expected[key]) == bool(actual[key]) for key in ("grasped", "object_goal_contact", "success"))
    return max(deviations.values(), default=0.0) <= tolerance and same_flags, deviations


def _candidate_actions(epsilon: float, secondary: float, heldout: int, radius: float, rng: np.random.Generator) -> list[tuple[str, np.ndarray, str]]:
    output: list[tuple[str, np.ndarray, str]] = [("center", np.zeros(3), "center")]
    for label, value in (("primary", epsilon), ("secondary", secondary)):
        for dim in range(3):
            unit = np.zeros(3)
            unit[dim] = value
            output.extend(((f"{label}_plus_{dim}", unit, f"axis_{label}"), (f"{label}_minus_{dim}", -unit, f"axis_{label}")))
    for left in range(3):
        for right in range(left + 1, 3):
            for sign_left, sign_right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                action = np.zeros(3)
                action[left] = sign_left * epsilon
                action[right] = sign_right * epsilon
                output.append((f"cross_{left}_{right}_{sign_left:+d}_{sign_right:+d}", action, "cross_primary"))
    for index in range(heldout):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction).clip(min=1e-12)
        magnitude = radius * rng.uniform(0.15, 1.0)
        output.append((f"heldout_{index:02d}", direction * magnitude, "heldout"))
    return output


def _is_smooth_anchor(physical: dict[str, Any]) -> bool:
    return not bool(physical["grasped"]) and not bool(physical["object_goal_contact"]) and not bool(physical["success"])


def _select_anchor_frames(states: np.ndarray, env, per_demo: int, tolerance: float, args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for index in range(len(states)):
        _, physical = _restore(env, states[index], tolerance)
        progress = index / max(1, len(states) - 1)
        eligible = _is_smooth_anchor(physical) and args.anchor_min_progress <= progress <= args.anchor_max_progress and args.anchor_min_eef_object_distance <= physical["eef_object_distance"] <= args.anchor_max_eef_object_distance
        if eligible:
            distance = abs(progress - args.anchor_target_progress) + abs(physical["eef_object_distance"] - args.anchor_target_eef_object_distance)
            candidates.append((distance, int(index), physical))
    return [(index, physical) for _, index, physical in sorted(candidates)[:per_demo]]


def _run_candidate(env, anchor_state: np.ndarray, anchor_physical: dict[str, Any], action3: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, float]]:
    observation, restored = _restore(env, anchor_state, args.restore_tolerance)
    restore_ok, restore_deviation = _restore_matches(anchor_physical, restored, args.restore_tolerance)
    reasons: list[str] = [] if restore_ok else ["restore_failed"]
    action = np.zeros(7, dtype=np.float64)
    action[:3] = action3
    action[6] = float(args.gripper_command)
    for _ in range(args.steps):
        observation, _, done, _ = env.step(action.tolist())
        if done:
            reasons.append("environment_done")
            break
    after = _physical(env)
    object_displacement = float(np.linalg.norm(np.asarray(after["task_object_pos"]) - np.asarray(anchor_physical["task_object_pos"])))
    if after["grasped"] != anchor_physical["grasped"]:
        reasons.append("grasp_transition")
    if after["object_goal_contact"] != anchor_physical["object_goal_contact"]:
        reasons.append("object_goal_contact_transition")
    if after["success"] != anchor_physical["success"]:
        reasons.append("success_transition")
    metrics = {
        "object_displacement": object_displacement,
        "eef_delta_norm": float(np.linalg.norm(np.asarray(after["eef_pos"]) - np.asarray(anchor_physical["eef_pos"]))),
        "eef_object_distance_delta": float(after["eef_object_distance"] - anchor_physical["eef_object_distance"]),
    }
    return observation, after, sorted(set(reasons)), {**restore_deviation, **metrics}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.steps < 1 or args.anchors_per_demo < 1 or args.epsilon <= 0 or args.epsilon_secondary <= 0:
        raise ValueError("steps, anchors-per-demo, and epsilons must be positive.")
    if args.reference_demo_index in args.anchor_demo_indices:
        raise ValueError("reference-demo-index must be independent of anchor-demo-indices.")
    root = Path.cwd().resolve()
    demo_path, bddl, language = resolve_libero_task(root, args.suite, args.task_id)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "anchors").mkdir()
    rng = np.random.default_rng(args.seed)

    from libero.libero.envs import OffScreenRenderEnv

    with h5py.File(demo_path, "r") as handle:
        names = sorted(handle["data"].keys(), key=lambda name: int(name.removeprefix("demo_")))
        reference_name = f"demo_{args.reference_demo_index}"
        if reference_name not in names:
            raise ValueError(f"Unknown reference demo {reference_name}; available={names[:3]}...")
        reference_states = np.asarray(handle["data"][reference_name]["states"], dtype=np.float64)
        anchor_payloads = {
            demo_index: np.asarray(handle["data"][f"demo_{demo_index}"]["states"], dtype=np.float64)
            for demo_index in args.anchor_demo_indices
            if f"demo_{demo_index}" in names
        }
    if len(anchor_payloads) != len(args.anchor_demo_indices):
        raise ValueError("At least one requested anchor demo does not exist.")

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=128, camera_widths=128)
    records: list[dict[str, Any]] = []
    try:
        reference_frame = int(args.reference_frame)
        if reference_frame <= 0 or reference_frame >= len(reference_states):
            raise ValueError("reference-frame must be an in-range non-initial frame.")
        _, reference_physical = _restore(env, reference_states[reference_frame], args.restore_tolerance)
        reference_progress = reference_frame / max(1, len(reference_states) - 1)
        if not (_is_smooth_anchor(reference_physical) and args.anchor_min_progress <= reference_progress <= args.anchor_max_progress and args.anchor_min_eef_object_distance <= reference_physical["eef_object_distance"] <= args.anchor_max_eef_object_distance):
            raise RuntimeError("Requested reference frame is outside the configured pre-grasp approach band.")
        reference_obs, _ = _restore(env, reference_states[reference_frame], args.restore_tolerance)
        reference_paths = _save_rgb(reference_obs, args.output_dir / "reference")
        reference = {
            "demo_name": reference_name,
            "frame_index": reference_frame,
            "state_sha256": _sha256(reference_states[reference_frame]),
            "physical": reference_physical,
            "images": {key: str(Path(value).relative_to(args.output_dir)) for key, value in reference_paths.items()},
        }

        for demo_index, states in anchor_payloads.items():
            for frame_index, physical in _select_anchor_frames(states, env, args.anchors_per_demo, args.restore_tolerance, args):
                anchor_id = f"demo_{demo_index}_frame_{frame_index:04d}"
                anchor_state = states[frame_index]
                candidates: list[dict[str, Any]] = []
                for candidate_id, action3, kind in _candidate_actions(args.epsilon, args.epsilon_secondary, args.heldout_actions, args.heldout_radius, rng):
                    obs, after, exclusions, metrics = _run_candidate(env, anchor_state, physical, action3, args)
                    paths = _save_rgb(obs, args.output_dir / "frames" / anchor_id / candidate_id)
                    candidates.append({
                        "candidate_id": candidate_id,
                        "kind": kind,
                        "action_command_xyz": action3.tolist(),
                        "images": {key: str(Path(value).relative_to(args.output_dir)) for key, value in paths.items()},
                        "physical_after": after,
                        "metrics": metrics,
                        "excluded_reasons": exclusions,
                    })
                # The anchor can undergo passive settling during the fixed
                # number of simulator steps.  Compare each action to the
                # zero-command center rollout, rather than treating this
                # shared baseline movement as an action-induced contact.
                center = next(candidate for candidate in candidates if candidate["candidate_id"] == "center")
                center_object = np.asarray(center["physical_after"]["task_object_pos"], dtype=np.float64)
                for candidate in candidates:
                    residual = float(np.linalg.norm(np.asarray(candidate["physical_after"]["task_object_pos"]) - center_object))
                    candidate["metrics"]["object_displacement_vs_center"] = residual
                    if candidate["candidate_id"] != "center" and residual > args.max_object_displacement:
                        candidate["excluded_reasons"] = sorted(set(candidate["excluded_reasons"] + ["object_motion_vs_center_above_precontact_limit"]))
                np.savez_compressed(args.output_dir / "anchors" / f"{anchor_id}.npz", anchor_state=anchor_state)
                records.append({
                    "schema_version": "v5_local_geometry_anchor_v1",
                    "suite": args.suite,
                    "task_id": args.task_id,
                    "language": language,
                    "anchor_id": anchor_id,
                    "source_demo": f"demo_{demo_index}",
                    "frame_index": frame_index,
                    "anchor_state_sha256": _sha256(anchor_state),
                    "anchor_physical": physical,
                    "reference": reference,
                    "action_protocol": {"dims": [0, 1, 2], "steps": args.steps, "gripper_command": args.gripper_command, "epsilon": args.epsilon, "epsilon_secondary": args.epsilon_secondary},
                    "candidates_npz": f"anchors/{anchor_id}.npz",
                    "candidates": candidates,
                })
                print(f"ANCHOR {anchor_id} candidates={len(candidates)} excluded={sum(bool(item['excluded_reasons']) for item in candidates)}", flush=True)
    finally:
        env.close()
    with (args.output_dir / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "protocol": "v5_local_geometry_collect_v1",
        "suite": args.suite,
        "task_id": args.task_id,
        "reference": reference,
        "anchors_collected": len(records),
        "candidates_total": sum(len(record["candidates"]) for record in records),
        "candidates_excluded": sum(sum(bool(candidate["excluded_reasons"]) for candidate in record["candidates"]) for record in records),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "collection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
