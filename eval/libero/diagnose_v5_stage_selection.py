"""Audit LIBERO demo stages before Pipeline-v5 local geometry collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from collect_v5_local_perturbations import _physical
from mi_reward.data.libero_privileged import resolve_libero_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    root = Path.cwd().resolve()
    demo_path, bddl, language = resolve_libero_task(root, args.suite, args.task_id)
    with h5py.File(demo_path, "r") as handle:
        states = {index: np.asarray(handle["data"][f"demo_{index}"]["states"], dtype=np.float64) for index in args.demo_indices}
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=128, camera_widths=128)
    output = {"suite": args.suite, "task_id": args.task_id, "language": language, "stride": args.stride, "demos": {}}
    try:
        for demo_index, trajectory in states.items():
            rows = []
            indices = sorted(set([*range(0, len(trajectory), args.stride), len(trajectory) - 1]))
            for frame in indices:
                env.reset()
                env.set_init_state(trajectory[frame])
                physical = _physical(env)
                rows.append({"frame_index": frame, "progress": frame / max(1, len(trajectory) - 1), **physical})
            output["demos"][f"demo_{demo_index}"] = rows
            print(f"STAGE demo_{demo_index} frames={len(rows)}", flush=True)
    finally:
        env.close()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "stage_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
