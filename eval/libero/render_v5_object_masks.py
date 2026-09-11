"""Replay frozen v5 probes to render aligned task-object instance masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mi_reward.closed_loop.libero_env import canonical_camera_image
from mi_reward.data.libero_privileged import goal_objects, resolve_libero_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("wrist", "agentview"), default="wrist")
    parser.add_argument("--max-rgb-mae", type=float, default=0.0)
    return parser.parse_args()


def _instance_ids(env, task_object: str) -> list[int]:
    names = list(env.env.model.instances_to_ids.keys())
    needle = task_object.lower()
    ids = [index + 1 for index, name in enumerate(names) if needle in str(name).lower()]
    if not ids:
        raise RuntimeError(f"Could not map task object {task_object!r}; instances={names}")
    return ids


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    root = Path.cwd().resolve()
    rows = [json.loads(line) for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Collection manifest has no anchors.")
    suite, task_id = rows[0]["suite"], int(rows[0]["task_id"])
    demo_path, bddl, _ = resolve_libero_task(root, suite, task_id)
    from libero.libero.envs import SegmentationRenderEnv

    camera = "robot0_eye_in_hand" if args.view == "wrist" else "agentview"
    image_key = f"{camera}_image"
    seg_key = f"{camera}_segmentation_instance"
    args.output_dir.mkdir(parents=True)
    report_rows = []
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_segmentations="instance", camera_names=[camera],
        camera_heights=128, camera_widths=128,
    )
    try:
        _, task_object, _ = goal_objects(env.env.parsed_problem["goal_state"])
        ids = _instance_ids(env, task_object)
        reference = rows[0]["reference"]
        with h5py.File(demo_path, "r") as handle:
            reference_state = np.asarray(handle["data"][reference["demo_name"]]["states"][int(reference["frame_index"])], dtype=np.float64)
        env.reset()
        reference_obs = env.set_init_state(reference_state)
        reference_rgb = canonical_camera_image(reference_obs, image_key)
        saved_reference = np.asarray(Image.open(args.collection_dir / reference["images"][args.view]).convert("RGB"), dtype=np.uint8)
        reference_mae = float(np.abs(reference_rgb.astype(np.float32) - saved_reference.astype(np.float32)).mean())
        reference_mask = np.isin(np.asarray(reference_obs[seg_key]).squeeze(), ids).astype(np.uint8) * 255
        reference_path = args.output_dir / "reference_mask.png"
        Image.fromarray(reference_mask).save(reference_path)
        for record in rows:
            state = np.asarray(np.load(args.collection_dir / record["candidates_npz"])["anchor_state"], dtype=np.float64)
            candidate_rows = []
            for candidate in record["candidates"]:
                env.reset()
                observation = env.set_init_state(state)
                action = np.zeros(7, dtype=np.float64)
                action[:3] = np.asarray(candidate["action_command_xyz"], dtype=np.float64)
                action[6] = float(record["action_protocol"]["gripper_command"])
                for _ in range(int(record["action_protocol"]["steps"])):
                    observation, _, done, _ = env.step(action.tolist())
                    if done:
                        break
                replay = canonical_camera_image(observation, image_key)
                saved = np.asarray(Image.open(args.collection_dir / candidate["images"][args.view]).convert("RGB"), dtype=np.uint8)
                mae = float(np.abs(replay.astype(np.float32) - saved.astype(np.float32)).mean())
                mask = np.isin(np.asarray(observation[seg_key]).squeeze(), ids).astype(np.uint8) * 255
                mask_path = args.output_dir / "masks" / record["anchor_id"] / f"{candidate['candidate_id']}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(mask_path)
                ys, xs = np.where(mask > 0)
                candidate_rows.append({
                    "candidate_id": candidate["candidate_id"], "mask": str(mask_path.relative_to(args.output_dir)),
                    "rgb_mae": mae, "aligned": bool(mae <= args.max_rgb_mae), "mask_pixels": int(len(xs)),
                    "centroid_xy": None if len(xs) == 0 else [float(xs.mean()), float(ys.mean())],
                })
            report_rows.append({"anchor_id": record["anchor_id"], "candidates": candidate_rows})
            aligned = sum(item["aligned"] for item in candidate_rows)
            print(f"ANCHOR {record['anchor_id']} aligned={aligned}/{len(candidate_rows)}", flush=True)
    finally:
        env.close()
    report = {
        "protocol": "v5_object_mask_replay_v1", "collection_dir": str(args.collection_dir), "view": args.view,
        "task_object": task_object, "instance_ids": ids, "max_rgb_mae": args.max_rgb_mae,
        "reference": {"mask": str(reference_path.relative_to(args.output_dir)), "rgb_mae": reference_mae,
                      "aligned": bool(reference_mae <= args.max_rgb_mae), "mask_pixels": int(np.count_nonzero(reference_mask))},
        "anchors": report_rows,
    }
    (args.output_dir / "masks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"anchors": len(report_rows), "task_object": task_object}, indent=2), flush=True)


if __name__ == "__main__":
    main()
