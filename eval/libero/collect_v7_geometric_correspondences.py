"""Augment a frozen LIBERO collection with replayed MuJoCo landmark correspondences.

This does not alter anchors or actions.  It restores every frozen anchor,
replays each recorded candidate action, and saves fixed-identity object and goal
MuJoCo geom centers plus default sites in local and world coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mi_reward.data.libero_privileged import goal_objects, resolve_libero_task


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


def _descendants(model, root: int) -> set[int]:
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body, parent in enumerate(model.body_parentid):
            if int(parent) in bodies and body not in bodies:
                bodies.add(body); changed = True
    return bodies


def _name(model, kind: str, index: int) -> str:
    value = getattr(model, f"{kind}_id2name")(index)
    return str(value) if value else f"{kind}_{index}"


def _identity_table(core, object_name: str, entity: str) -> tuple[int, list[dict[str, Any]]]:
    model = core.sim.model
    root = int(model.body_name2id(core.get_object(object_name).root_body))
    bodies = _descendants(model, root)
    table = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in bodies:
            table.append({"identity": f"{entity}:geom:{_name(model, 'geom', geom_id)}", "kind": "geom", "id": geom_id})
    for site_id in range(model.nsite):
        if int(model.site_bodyid[site_id]) in bodies:
            table.append({"identity": f"{entity}:site:{_name(model, 'site', site_id)}", "kind": "site", "id": site_id})
    if not table:
        raise RuntimeError(f"No MuJoCo geom/site landmarks for {entity}:{object_name}")
    return root, sorted(table, key=lambda point: point["identity"])


def _points(core, root_body: int, table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = core.sim.data
    root_pos = np.asarray(data.body_xpos[root_body], dtype=np.float64)
    root_rotation = np.asarray(data.body_xmat[root_body], dtype=np.float64).reshape(3, 3)
    output = []
    for point in table:
        world = np.asarray(data.geom_xpos[point["id"]] if point["kind"] == "geom" else data.site_xpos[point["id"]], dtype=np.float64)
        output.append({"identity": point["identity"], "kind": point["kind"], "local_xyz": (root_rotation.T @ (world - root_pos)).tolist(), "world_xyz": world.tolist()})
    return output


def _physical_matches(actual: dict[str, Any], expected: dict[str, Any], tolerance: float) -> None:
    for key in ("eef_pos", "task_object_pos", "goal_object_pos"):
        deviation = float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
        if deviation > tolerance:
            raise RuntimeError(f"Replay mismatch {key}: {deviation:.3e} > {tolerance:.3e}")


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 20:
        raise RuntimeError("G1 requires the frozen 20-anchor collection")
    suite, task_id = rows[0]["suite"], int(rows[0]["task_id"])
    root = Path.cwd().resolve()
    _, bddl, _ = resolve_libero_task(root, suite, task_id)
    from libero.libero.envs import OffScreenRenderEnv

    args.output_dir.mkdir(parents=True)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview"], camera_heights=128, camera_widths=128)
    try:
        core = env.env
        _, task_name, goal_name = goal_objects(core.parsed_problem["goal_state"])
        task_root, task_table = _identity_table(core, task_name, "task_object")
        goal_root, goal_table = _identity_table(core, goal_name, "goal_object")
        identities = {"task_object": task_table, "goal_object": goal_table}
        reference_state = np.asarray(np.load(args.collection_dir / rows[0]["candidates_npz"])["anchor_state"], dtype=np.float64)
        # The independent reference is replayed from its demo state in the existing manifest only in later MI evaluation;
        # this collector records candidate states and validates their replay exactly.
        records = []
        local_reference: dict[str, dict[str, np.ndarray]] | None = None
        for row in rows:
            anchor_state = np.asarray(np.load(args.collection_dir / row["candidates_npz"])["anchor_state"], dtype=np.float64)
            candidate_records = []
            for candidate in row["candidates"]:
                env.reset(); env.set_init_state(anchor_state)
                action = np.zeros(7, dtype=np.float64)
                action[:3] = np.asarray(candidate["action_command_xyz"], dtype=np.float64)
                action[6] = float(row["action_protocol"]["gripper_command"])
                for _ in range(int(row["action_protocol"]["steps"])):
                    _, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                actual = {"eef_pos": np.asarray(core._eef_xpos, dtype=np.float64).tolist(), "task_object_pos": np.asarray(core.object_states_dict[task_name].get_geom_state()["pos"], dtype=np.float64).tolist(), "goal_object_pos": np.asarray(core.object_states_dict[goal_name].get_geom_state()["pos"], dtype=np.float64).tolist()}
                _physical_matches(actual, candidate["physical_after"], args.tolerance)
                task_points, goal_points = _points(core, task_root, task_table), _points(core, goal_root, goal_table)
                local = {"task_object": {item["identity"]: np.asarray(item["local_xyz"]) for item in task_points}, "goal_object": {item["identity"]: np.asarray(item["local_xyz"]) for item in goal_points}}
                if local_reference is None:
                    local_reference = local
                for entity in local:
                    for identity, value in local[entity].items():
                        deviation = float(np.max(np.abs(value - local_reference[entity][identity])))
                        if deviation > args.tolerance:
                            raise RuntimeError(f"Local identity drift {identity}: {deviation:.3e} > {args.tolerance:.3e}")
                candidate_records.append({"candidate_id": candidate["candidate_id"], "task_object_points": task_points, "goal_object_points": goal_points})
            records.append({"anchor_id": row["anchor_id"], "source_demo": row["source_demo"], "candidates": candidate_records})
            print(f"ANCHOR {row['anchor_id']} correspondences={len(candidate_records)} task_points={len(task_table)} goal_points={len(goal_table)}", flush=True)
    finally:
        env.close()
    report = {"schema_version": "v7_geometric_correspondence_collection_v1", "base_collection": str(args.collection_dir), "suite": suite, "task_id": task_id, "privileged_input": True, "deployable": False, "landmark_definition": "Fixed MuJoCo geom centers and sites, identified by model name and saved in entity-local and world coordinates.", "point_identities": identities, "anchors": records}
    (args.output_dir / "correspondences.json").write_text(json.dumps(report) + "\n")
    (args.output_dir / "schema.json").write_text(json.dumps({key: value for key, value in report.items() if key != "anchors"}, indent=2) + "\n")
    print(json.dumps({"anchors": len(records), "task_points": len(identities["task_object"]), "goal_points": len(identities["goal_object"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
