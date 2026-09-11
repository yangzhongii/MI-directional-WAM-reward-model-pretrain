"""Export LIBERO demonstrations for action-conditioned Cosmos adaptation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm


def axis_angle_to_rpy(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected axis-angle states [T,3], got {values.shape}.")
    return Rotation.from_rotvec(values).as_euler("xyz").astype(np.float32)


def closed_gripper_state(
    gripper_qpos: np.ndarray,
    *,
    open_width: float = 0.06,
    closed_width: float = 0.02,
) -> np.ndarray:
    """Map two-finger qpos to the action model convention: 0=open, 1=closed."""

    values = np.asarray(gripper_qpos, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"Expected gripper qpos [T,G], got {values.shape}.")
    if open_width <= closed_width:
        raise ValueError("open_width must be larger than closed_width.")
    width = np.abs(values).sum(axis=1)
    return np.clip((open_width - width) / (open_width - closed_width), 0.0, 1.0).astype(np.float32)


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
    boundary = len(shuffled) - val_count - test_count
    return {
        "train": shuffled[:boundary],
        "val": shuffled[boundary : len(shuffled) - test_count],
        "test": shuffled[len(shuffled) - test_count :],
    }


def _prepare_shared_text_embedding(source: Path, annotations: Path) -> Path:
    """Convert the official empty Reason1 tensor once for Dataset_3D.

    Dataset_3D expects one NumPy sidecar beside every annotation. All LIBERO
    episodes share the same task-agnostic empty prompt, so per-episode sidecars
    are symlinks to one 98 MiB float16 array instead of physical copies.
    """

    import torch

    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Empty Reason1 embedding is missing: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {source}, got {type(value).__name__}")
    value = value.squeeze(0)
    if tuple(value.shape) != (512, 100352):
        raise ValueError(f"Expected empty Reason1 embedding [512,100352], got {tuple(value.shape)}")
    destination = annotations / "empty_reason_embedding.npy"
    np.save(destination, value.float().numpy().astype(np.float16))
    return destination


def export_libero_dataset(
    *,
    source: Path,
    task: str,
    output: Path,
    seed: int = 0,
    fps: int = 20,
    max_demos: int | None = None,
    empty_text_embedding: Path | None = None,
) -> dict[str, Any]:
    """Create videos and annotations consumed by Cosmos' Bridge action loader."""

    source = source.resolve()
    output = output.resolve()
    annotations = output / "annotation"
    videos = output / "videos"
    for split in ("train", "val", "test"):
        (annotations / split).mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    shared_embedding = (
        _prepare_shared_text_embedding(empty_text_embedding, annotations)
        if empty_text_embedding is not None
        else None
    )

    exported = {"train": 0, "val": 0, "test": 0}
    frame_total = 0
    with h5py.File(source, "r") as handle:
        names = sorted(handle["data"].keys())
        if max_demos is not None:
            names = names[: int(max_demos)]
        splits = _split_names(names, seed)
        assignment = {name: split for split, values in splits.items() for name in values}
        progress = tqdm(names, desc="Export LIBERO action data", unit="trajectory")
        for name in progress:
            demo = handle["data"][name]
            obs = demo["obs"]
            main = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
            wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
            positions = np.asarray(obs["ee_pos"], dtype=np.float32)
            rotations = axis_angle_to_rpy(np.asarray(obs["ee_ori"], dtype=np.float32))
            gripper = closed_gripper_state(np.asarray(obs["gripper_states"], dtype=np.float32))
            length = min(len(main), len(wrist), len(positions), len(rotations), len(gripper))
            if length < 13:
                continue
            episode_id = f"libero-{source.stem}-{name}"
            main_path = videos / f"{episode_id}-agentview.mp4"
            wrist_path = videos / f"{episode_id}-wrist.mp4"
            imageio.mimwrite(main_path, main[:length], fps=fps, codec="libx264", quality=8)
            imageio.mimwrite(wrist_path, wrist[:length], fps=fps, codec="libx264", quality=8)
            states = np.concatenate((positions[:length], rotations[:length]), axis=1)
            split = assignment[name]
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
                    "is_eval": split != "train",
                    "split": split,
                    "orientation_conversion": "LIBERO axis-angle to xyz Euler",
                    "gripper_convention": "0=open,1=closed",
                },
            }
            annotation_path = annotations / split / f"{episode_id}.json"
            annotation_path.write_text(
                json.dumps(annotation) + "\n", encoding="utf-8"
            )
            if shared_embedding is not None:
                embedding_path = annotation_path.with_suffix(".npy")
                if embedding_path.exists() or embedding_path.is_symlink():
                    embedding_path.unlink()
                embedding_path.symlink_to(os.path.relpath(shared_embedding, embedding_path.parent))
            exported[split] += 1
            frame_total += length
            progress.set_postfix(split=split, frames=frame_total)

    report = {
        "status": "complete",
        "format": "cosmos_action_conditioned_bridge_v1",
        "source": str(source),
        "output": str(output),
        "task": task,
        "fps": int(fps),
        "episodes": exported,
        "frames": frame_total,
        "camera_ids": [0, 1],
        "text_embedding": None if shared_embedding is None else str(shared_embedding),
        "text_embedding_shape": None if shared_embedding is None else [512, 100352],
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
    parser.add_argument(
        "--empty-text-embedding",
        help="Official cr1_empty_string_text_embeddings.pt used by the action checkpoint.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.is_file():
            parser.error(f"--source does not exist: {source}")
        if not args.task:
            parser.error("--task is required with --source.")
        task = args.task
    else:
        source, task = _dataset_path(root, args.suite, args.task_id)
    print(json.dumps(export_libero_dataset(
        source=source,
        task=task,
        output=Path(args.output),
        seed=args.seed,
        fps=args.fps,
        max_demos=args.max_demos,
        empty_text_embedding=None if args.empty_text_embedding is None else Path(args.empty_text_embedding),
    ), indent=2))


if __name__ == "__main__":
    main()
