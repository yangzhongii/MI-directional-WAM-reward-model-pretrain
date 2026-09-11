"""Measure command-to-actual-EEF scale for v5 smooth local probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collect_v5_local_perturbations import _physical, _restore, _restore_matches
from mi_reward.data.libero_privileged import resolve_libero_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands", type=float, nargs="+", default=[0.012, 0.03, 0.06, 0.12])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gripper-command", type=float, default=-1.0)
    parser.add_argument("--max-object-displacement", type=float, default=2e-4)
    parser.add_argument("--restore-tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Collection manifest is empty.")
    root = Path.cwd().resolve()
    _, bddl, _ = resolve_libero_task(root, rows[0]["suite"], int(rows[0]["task_id"]))
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=128, camera_widths=128)
    report = {"protocol": "v5_actual_eef_scale_calibration_v1", "collection": str(args.collection_dir), "commands": args.commands, "steps": args.steps, "anchors": []}
    try:
        for row in rows:
            state = np.load(args.collection_dir / row["candidates_npz"])["anchor_state"]
            _, anchor = _restore(env, state, args.restore_tolerance)
            _, center = _restore(env, state, args.restore_tolerance)
            # Center rollout establishes passive settling for this exact step count.
            zero = np.zeros(7, dtype=np.float64); zero[6] = args.gripper_command
            center_obs = None
            for _ in range(args.steps):
                center_obs, _, _, _ = env.step(zero.tolist())
            center_after = _physical(env)
            entries = []
            for command in args.commands:
                for axis in range(3):
                    for sign in (-1.0, 1.0):
                        _, restored = _restore(env, state, args.restore_tolerance)
                        restore_ok, deviation = _restore_matches(anchor, restored, args.restore_tolerance)
                        action = zero.copy(); action[axis] = sign * command
                        for _ in range(args.steps):
                            env.step(action.tolist())
                        after = _physical(env)
                        eef_delta = np.asarray(after["eef_pos"]) - np.asarray(center_after["eef_pos"])
                        object_motion = float(np.linalg.norm(np.asarray(after["task_object_pos"]) - np.asarray(center_after["task_object_pos"])))
                        smooth = restore_ok and not after["grasped"] and not after["object_goal_contact"] and not after["success"] and object_motion <= args.max_object_displacement
                        entries.append({"command": command, "axis": axis, "sign": int(sign), "actual_eef_delta_xyz_m": eef_delta.tolist(), "actual_eef_delta_norm_mm": float(np.linalg.norm(eef_delta) * 1000), "object_motion_mm": object_motion * 1000, "smooth": smooth, "restore_deviation": deviation})
            report["anchors"].append({"anchor_id": row["anchor_id"], "frame_index": row["frame_index"], "entries": entries})
            print(f"CALIBRATED {row['anchor_id']} entries={len(entries)}", flush=True)
    finally:
        env.close()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
