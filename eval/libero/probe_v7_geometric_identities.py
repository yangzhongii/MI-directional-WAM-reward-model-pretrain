"""Audit MuJoCo object/goal site and geometry identities for v7 G1 collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mi_reward.data.libero_privileged import goal_objects, resolve_libero_task


def _descendants(model, root: int) -> set[int]:
    output = {root}
    changed = True
    while changed:
        changed = False
        for body, parent in enumerate(model.body_parentid):
            if int(parent) in output and body not in output:
                output.add(body); changed = True
    return output


def _name(model, kind: str, index: int) -> str:
    getter = getattr(model, f"{kind}_id2name", None)
    value = getter(index) if getter else None
    return str(value) if value else f"{kind}_{index}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-index", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    demo_path, bddl, _ = resolve_libero_task(root, args.suite, args.task_id)
    with h5py.File(demo_path, "r") as handle:
        state = np.asarray(handle["data"][f"demo_{args.demo_index}"]["states"][args.frame_index], dtype=np.float64)
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview"], camera_heights=128, camera_widths=128)
    try:
        env.reset(); env.set_init_state(state)
        core, model = env.env, env.env.sim.model
        _, task, goal = goal_objects(core.parsed_problem["goal_state"])
        entities = {}
        for label, name in (("task", task), ("goal", goal)):
            obj = core.get_object(name)
            root_body = model.body_name2id(obj.root_body)
            bodies = _descendants(model, root_body)
            entities[label] = {
                "object_name": name, "root_body": obj.root_body, "body_ids": sorted(bodies),
                "sites": [{"id": i, "name": _name(model, "site", i), "body_id": int(model.site_bodyid[i])} for i in range(model.nsite) if int(model.site_bodyid[i]) in bodies],
                "geoms": [{"id": i, "name": _name(model, "geom", i), "body_id": int(model.geom_bodyid[i]), "type": int(model.geom_type[i]), "data_id": int(model.geom_dataid[i])} for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies],
            }
        report = {"task": entities["task"], "goal": entities["goal"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: {"sites": len(value["sites"]), "geoms": len(value["geoms"])} for key, value in entities.items()}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
