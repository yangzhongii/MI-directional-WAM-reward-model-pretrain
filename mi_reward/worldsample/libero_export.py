"""Export official LIBERO demonstrations for Cosmos action-conditioned LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task


def axis_angle_to_rpy(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected axis-angle states [T,3], got {values.shape}.")
    return Rotation.from_rotvec(values).as_euler("xyz").astype(np.float32)


def closed_gripper_state(gripper_qpos: np.ndarray, *, open_width: float = 0.06, closed_width: float = 0.02) -> np.ndarray:
    """Map two-finger qpos to Bridge convention: 0=open, 1=closed."""

    values = np.asarray(gripper_qpos, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"Expected gripper qpos [T,G], got {values.shape}.")
    width = np.abs(values).sum(axis=1)
    if open_width <= closed_width:
        raise ValueError("open_width must be larger than closed_width.")
    state = (open_width - width) / (open_width - closed_width)
    return np.clip(state, 0.0, 1.0).astype(np.float32)


def _dataset_path(root: Path, suite: str, task_id: int) -> tuple[Path, str]:
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"Unknown LIBERO suite {suite!r}; available={sorted(suites)}")
    task_suite = suites[suite]()
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(f"Task id {task_id} is outside [0,{task_suite.n_tasks}).")
    relative = Path(task_suite.get_task_demonstration(task_id))
    candidates = (
        root / ".venv" / "src" / "libero" / "libero" / "datasets" / relative,
        root / ".venv" / "datasets" / relative,
        root / relative,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"LIBERO demonstration is missing: {relative}")
    return source, str(task_suite.get_task(task_id).language)


def _split_names(names: list[str], seed: int) -> dict[str, list[str]]:
    if len(names) < 3:
        raise ValueError("At least three demonstrations are required for train/val/test isolation.")
    shuffled = list(names)
    np.random.default_rng(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.15))
    val_count = max(1, round(len(shuffled) * 0.15))
    if test_count + val_count >= len(shuffled):
        test_count = val_count = 1
    return {
        "train": shuffled[: len(shuffled) - val_count - test_count],
        "val": shuffled[len(shuffled) - val_count - test_count : len(shuffled) - test_count],
        "test": shuffled[len(shuffled) - test_count :],
    }


def export_libero_dataset(
    *,
    source: Path,
    task: str,
    output: Path,
    seed: int = 0,
    fps: int = 20,
    max_demos: int | None = None,
    bddl_path: Path | None = None,
    include_privileged: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    annotations = output / "annotation"
    videos = output / "videos"
    for split in ("train", "val", "test"):
        (annotations / split).mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        names = sorted(handle["data"].keys())
        if max_demos is not None:
            names = names[: int(max_demos)]
        splits = _split_names(names, seed)
        assignment = {name: split for split, split_names in splits.items() for name in split_names}
        progress = tqdm(names, desc="Export LIBERO -> Cosmos", unit="trajectory")
        frame_total = 0
        for name in progress:
            demo = handle["data"][name]
            obs = demo["obs"]
            main = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
            wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
            positions = np.asarray(obs["ee_pos"], dtype=np.float32)
            rotations = axis_angle_to_rpy(np.asarray(obs["ee_ori"], dtype=np.float32))
            gripper = closed_gripper_state(np.asarray(obs["gripper_states"], dtype=np.float32))
            mujoco_states = np.asarray(demo["states"], dtype=np.float64)
            actions = np.asarray(demo["actions"], dtype=np.float64)
            rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float64)
            dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.uint8)
            length = min(
                len(main), len(wrist), len(positions), len(rotations), len(gripper),
                len(mujoco_states), len(actions), len(rewards), len(dones),
            )
            if length < 13:
                continue
            episode_id = f"libero-{source.stem}-{name}"
            main_path = videos / f"{episode_id}-agentview.mp4"
            wrist_path = videos / f"{episode_id}-wrist.mp4"
            imageio.mimwrite(main_path, main[:length], fps=fps, codec="libx264", quality=8)
            imageio.mimwrite(wrist_path, wrist[:length], fps=fps, codec="libx264", quality=8)
            states = np.concatenate((positions[:length], rotations[:length]), axis=1)
            annotation = {
                "episode_id": episode_id,
                "task": task,
                "texts": [task],
                "videos": [
                    {"camera": "agentview", "video_path": str(main_path.relative_to(output))},
                    {"camera": "wrist", "video_path": str(wrist_path.relative_to(output))},
                ],
                "state": states.tolist(),
                "continuous_gripper_state": gripper[:length].tolist(),
                "episode_metadata": {
                    "episode_id": episode_id,
                    "source": str(source),
                    "source_demo": name,
                    "is_eval": assignment[name] != "train",
                    "split": assignment[name],
                    "orientation_conversion": "LIBERO axis-angle to xyz Euler",
                    "gripper_convention": "0=open,1=closed",
                },
            }
            if include_privileged:
                if bddl_path is None:
                    raise ValueError("bddl_path is required when include_privileged=True")
                annotation["privileged"] = extract_privileged_records(
                    bddl_path=bddl_path,
                    states=mujoco_states[:length],
                    actions=actions[:length],
                    rewards=rewards[:length],
                    dones=dones[:length],
                )
            destination = annotations / assignment[name] / f"{episode_id}.json"
            destination.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
            frame_total += length
            progress.set_postfix(split=assignment[name], frames=frame_total)

    report = {
        "status": "complete",
        "format": "cosmos_action_conditioned_bridge_v1",
        "source": str(source),
        "output": str(output),
        "task": task,
        "fps": int(fps),
        "episodes": {split: len(values) for split, values in splits.items()},
        "frames": frame_total,
        "camera_ids": [0, 1],
        "include_privileged": bool(include_privileged),
    }
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--source")
    parser.add_argument("--task")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--bddl")
    parser.add_argument("--include-privileged", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.is_file():
            parser.error(f"--source does not exist: {source}")
        if not args.task:
            parser.error("--task is required with --source.")
        task = args.task
        bddl_path = Path(args.bddl).expanduser().resolve() if args.bddl else None
        if args.include_privileged and (bddl_path is None or not bddl_path.is_file()):
            parser.error("--bddl is required with --source when --include-privileged is set.")
    else:
        source, bddl_path, task = resolve_libero_task(root, args.suite, args.task_id)
    print(json.dumps(export_libero_dataset(
        source=source,
        task=task,
        output=Path(args.output),
        seed=args.seed,
        fps=args.fps,
        max_demos=args.max_demos,
        bddl_path=bddl_path,
        include_privileged=args.include_privileged,
    ), indent=2))


if __name__ == "__main__":
    main()
