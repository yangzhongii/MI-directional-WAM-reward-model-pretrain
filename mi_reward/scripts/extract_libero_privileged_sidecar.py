"""Extract raw privileged physical evidence from official LIBERO demos.

This is a Pipeline-v3 data utility, not a reward function.  It replays saved
MuJoCo states and records interpretable physical channels that can later be
used for hard-negative construction, physical consistency, and held-out
aliasing evaluation.

No weighted "physical progress reward" is created here on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    import h5py

    demo_path, bddl_path, language = resolve_libero_task(root, args.suite, args.task_id)
    with h5py.File(demo_path, "r") as handle:
        names = sorted(handle["data"].keys())
        demo_name = names[int(args.demo_index) % len(names)]
        demo = handle["data"][demo_name]
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
        rewards = np.asarray(demo["rewards"], dtype=np.float64)
        dones = np.asarray(demo["dones"], dtype=np.uint8)

    extracted = extract_privileged_records(
        bddl_path=bddl_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    records = extracted["frames"]
    task_object = extracted["task_object"]
    goal_object = extracted["goal_object"]
    predicate = extracted["goal_predicate"]
    payload = {
        "suite": args.suite,
        "task_id": int(args.task_id),
        "language": language,
        "demo_name": demo_name,
        "source": str(demo_path),
        "bddl": str(bddl_path),
        "goal_predicate": predicate,
        "task_object": task_object,
        "goal_object": goal_object,
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": len(records),
                "task_object": task_object,
                "goal_object": goal_object,
                "first_grasp": next((r["frame_index"] for r in records if r["grasped"]), None),
                "first_success": next(
                    (r["frame_index"] for r in records if r["environment_success"]), None
                ),
                "object_goal_distance_xy": {
                    "start": records[0]["object_goal_distance_xy"],
                    "end": records[-1]["object_goal_distance_xy"],
                    "min": min(r["object_goal_distance_xy"] for r in records),
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
