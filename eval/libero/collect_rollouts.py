"""Collect native simulator controls with measured outcomes, without training.

Replay a held-out demo's actions and an open-gripper intervention from exactly
the same saved initial state. Neither controller's outcome is assumed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mi_reward.closed_loop.libero_env import canonical_camera_image
from mi_reward.data.libero_privileged import resolve_libero_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--demo-index", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-horizon", type=int, default=220)
    args = parser.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    root = Path.cwd()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    demo_name = f"demo_{args.demo_index}"
    teacher_root = root / "logs/mi_reward/v3_teacher_mainline/teacher_export5_v2"
    excluded = set()
    for split in ("train", "validation"):
        for line in (teacher_root / f"qwen_{split}.jsonl").read_text().splitlines():
            provenance = json.loads(line)["provenance"]
            excluded.add((int(provenance["task_id"]), provenance["demo_name"]))
    if any((task, demo_name) in excluded for task in args.task_ids):
        raise ValueError("Requested source episode overlaps teacher train/validation")
    records = []
    with (output / "manifest.jsonl").open("x") as manifest:
        for task_id in args.task_ids:
            demo_path, bddl, language = resolve_libero_task(root, "libero_spatial", task_id)
            with h5py.File(demo_path, "r") as handle:
                demo = handle["data"][demo_name]
                initial_state = np.asarray(demo["states"][0], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float64)
            pair_id = f"task_{task_id:02d}_{demo_name}_seed_{args.seed}"
            for policy_id in ("demo_action_replay", "open_gripper_replay"):
                identity = f"{pair_id}_{policy_id}"
                folder = output / "frames" / identity
                folder.mkdir(parents=True)
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"],
                                         camera_heights=128, camera_widths=128)
                main, wrist, applied, outcomes, rewards, sim_states = [], [], [], [], [], []
                try:
                    env.seed(args.seed)
                    env.reset()
                    obs = env.set_init_state(initial_state)
                    if env.check_success():
                        raise ValueError(f"Already successful initial state: {identity}")

                    def save_frame(observation, index):
                        for key, view, paths in (("agentview_image", "agentview", main),
                                                 ("robot0_eye_in_hand_image", "wrist", wrist)):
                            path = folder / f"{view}_{index:06d}.png"
                            Image.fromarray(canonical_camera_image(observation, key)).save(path)
                            paths.append(str(path.relative_to(output)))

                    save_frame(obs, 0)
                    horizon = max(args.min_horizon, len(actions) + 40)
                    success = False
                    for step in range(horizon):
                        action = actions[step].copy() if step < len(actions) else np.r_[np.zeros(6), actions[-1, -1]]
                        if policy_id == "open_gripper_replay":
                            action[-1] = -1.0
                        obs, reward, done, info = env.step(action.tolist())
                        success = bool(env.check_success())
                        applied.append(action)
                        rewards.append(float(reward))
                        outcomes.append(success)
                        sim_states.append(env.get_sim_state())
                        save_frame(obs, step + 1)
                        if success or done:
                            break
                    trace = output / f"{identity}_trace.npz"
                    np.savez_compressed(trace, initial_state=initial_state, states=sim_states,
                                        actions=applied, environment_success=outcomes, environment_reward=rewards)
                    record = {
                        "benchmark": "libero_spatial", "source": "libero_rollout",
                        "task_id": demo_path.stem.removesuffix("_demo"), "task_index": task_id,
                        "trajectory_id": identity, "pair_id": pair_id, "policy_id": policy_id,
                        "language": language, "agentview": main, "wrist": wrist,
                        "success": success, "quality": None,
                        "outcome_source": "OffScreenRenderEnv.check_success",
                        "termination": "success" if success else ("environment_done" if done else "timeout"),
                        "steps": len(applied), "horizon": horizon, "seed": args.seed,
                        "source_demo": str(demo_path), "demo_name": demo_name,
                        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
                        "teacher_episode_overlap": False, "trace": trace.name,
                        "controller_kind": "scripted_action_replay_control_not_learned_policy",
                    }
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()
                    records.append(record)
                    print(f"{identity}: success={success}, steps={len(applied)}", flush=True)
                finally:
                    env.close()
    summary = {
        "episodes": len(records), "config": vars(args), "status": "completed",
        "by_controller": {policy: {"episodes": sum(r["policy_id"] == policy for r in records),
                                   "successes": sum(r["success"] for r in records if r["policy_id"] == policy)}
                          for policy in ("demo_action_replay", "open_gripper_replay")},
        "scope": "Matched-state native simulator controls; episode-held-out, not a learned-policy benchmark.",
    }
    (output / "collection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
