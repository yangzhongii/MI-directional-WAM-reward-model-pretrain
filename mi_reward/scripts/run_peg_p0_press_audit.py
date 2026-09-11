"""Controlled downward press audit for the P1 welded-grasp MuJoCo scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from mi_reward.sim.peg_contact_metrics import contact_force_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="assets/custom_task/scene/scene_peg_round_local_servo.xml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_target")
    mocap_id = int(model.body_mocapid[mocap_body])
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task_object")
    # Lift the welded grasp clear of the table before the downward sweep;
    # the source scene starts with the peg resting on the table by design.
    start = np.array(data.mocap_pos[mocap_id], copy=True)
    start[2] += 0.08
    data.mocap_pos[mocap_id] = start
    for _ in range(250):
        mujoco.mj_step(model, data)
    rows = []
    for i in range(args.steps):
        # 0--100 mm downward travel, followed by settling at each pose.
        data.mocap_pos[mocap_id] = start + np.array([0.0, 0.0, -0.10 * i / max(args.steps - 1, 1)])
        for _ in range(15):
            mujoco.mj_step(model, data)
        force = contact_force_summary(mujoco, model, data)
        rows.append({
            "step": i,
            "commanded_down_mm": 100.0 * i / max(args.steps - 1, 1),
            "mocap_position": data.mocap_pos[mocap_id].tolist(),
            "peg_position": data.xpos[object_body].tolist(),
            **force,
        })
    normal = np.asarray([r["normal_force_n"] for r in rows])
    monotone_fraction = float(np.mean(np.diff(normal) >= -1e-5)) if len(normal) > 1 else 1.0
    report = {
        "stage": "P0_controlled_downward_press",
        "model": args.model,
        "frames": rows,
        "normal_force_monotone_fraction": monotone_fraction,
        "peak_normal_force_n": float(normal.max(initial=0.0)),
        "p0_press_pass": bool(monotone_fraction >= 0.70 and normal.max(initial=0.0) > 0.1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
