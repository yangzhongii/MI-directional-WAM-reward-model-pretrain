from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mi_reward.data.schema import TrajectoryExample, write_jsonl

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".gif"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _frame_sort_key(path: Path) -> tuple[str, int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (path.parent.as_posix(), int(digits) if digits else -1, path.name)


def _collect_frames(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.is_dir():
        frames = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return [str(p) for p in sorted(frames, key=_frame_sort_key)]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [str(path)]
    return []


def extract_video_frames(video_path: str | Path, output_dir: str | Path, fps: float = 2.0, overwrite: bool = False) -> list[str]:
    video = Path(video_path)
    output = Path(output_dir)
    existing = _collect_frames(output)
    if existing and not overwrite:
        return existing
    output.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to extract frames from videos.") from exc

    reader = imageio.get_reader(str(video))
    meta = reader.get_meta_data()
    source_fps = float(meta.get("fps") or fps or 1.0)
    stride = max(1, int(round(source_fps / max(float(fps), 1e-6))))
    frame_paths = []
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx % stride != 0:
                continue
            frame_path = output / f"frame_{frame_idx:06d}.png"
            imageio.imwrite(frame_path, frame)
            frame_paths.append(str(frame_path))
    finally:
        reader.close()
    return frame_paths


def _extract_from_video(video_path: str | None, frame_output_root: Path | None, traj_id: str, fps: float, overwrite: bool) -> list[str]:
    if not video_path or frame_output_root is None:
        return []
    video = Path(video_path)
    if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
        return []
    return extract_video_frames(video, frame_output_root / traj_id.replace("/", "__"), fps=fps, overwrite=overwrite)


def build_manifest(frame_root: str | Path, output: str | Path, split: str, source: str) -> list[TrajectoryExample]:
    root = Path(frame_root)
    examples: list[TrajectoryExample] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for traj_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            frames = _collect_frames(traj_dir)
            if frames:
                examples.append(
                    TrajectoryExample(
                        traj_id=f"{task_dir.name}/{traj_dir.name}",
                        task=task_dir.name,
                        frames=frames,
                        source=source,
                        split=split,
                    )
                )
    write_jsonl(output, examples)
    return examples


def _resolve_path(path_value: str | None, base_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    candidate = base_dir / path
    return str(candidate if candidate.exists() else path)


def build_libero_manifest(
    run_dir: str | Path,
    output: str | Path = "dataset/mi_reward/manifests/libero_manifest.jsonl",
    *,
    frame_output_root: str | Path | None = "dataset/mi_reward/extracted_frames/libero",
    fps: float = 2.0,
    split: str = "eval",
    overwrite_frames: bool = False,
) -> list[TrajectoryExample]:
    root = Path(run_dir)
    frame_root = Path(frame_output_root) if frame_output_root else None
    examples: list[TrajectoryExample] = []
    for episode_file in sorted(root.rglob("episodes.jsonl")):
        suite_dir = episode_file.parent
        suite_name = suite_dir.name
        summary = _read_json(suite_dir / "summary.json") if (suite_dir / "summary.json").is_file() else {}
        for record in _read_jsonl(episode_file):
            task_id = int(record.get("task_id", -1))
            episode_id = int(record.get("episode_idx", len(examples)))
            task = str(record.get("task_name") or record.get("task_description") or f"task_{task_id}")
            traj_id = f"libero/{suite_name}/task{task_id:02d}/episode{episode_id:03d}"
            video_path = _resolve_path(record.get("rollout_video_path"), suite_dir)
            frame_dir = Path(video_path).with_name(f"{Path(video_path).stem}_frames") if video_path else None
            frames = _collect_frames(frame_dir) or _extract_from_video(video_path, frame_root, traj_id, fps, overwrite_frames)
            examples.append(
                TrajectoryExample(
                    traj_id=traj_id,
                    task=task,
                    frames=frames,
                    source="libero_eval",
                    split=split,
                    metadata={
                        "suite": suite_name,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "success": bool(record.get("success", False)),
                        "video_path": video_path,
                        "num_actions": record.get("num_actions"),
                        "summary_success_rate": summary.get("total_success_rate"),
                    },
                )
            )
    write_jsonl(output, examples)
    return examples


def _find_robotwin_episode_video(task_dir: Path, episode_id: int) -> str | None:
    candidates = [
        task_dir / "videos" / f"episode{episode_id}.mp4",
        task_dir / "video" / f"episode{episode_id}.mp4",
        task_dir / f"episode{episode_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    matches = sorted(task_dir.rglob(f"*episode{episode_id}*.mp4"))
    return str(matches[0]) if matches else None


def build_robotwin_manifest(
    run_dir: str | Path,
    output: str | Path = "dataset/mi_reward/manifests/robotwin_manifest.jsonl",
    *,
    frame_output_root: str | Path | None = "dataset/mi_reward/extracted_frames/robotwin",
    fps: float = 2.0,
    split: str = "eval",
    overwrite_frames: bool = False,
) -> list[TrajectoryExample]:
    root = Path(run_dir)
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"RoboTwin tasks directory not found: {tasks_dir}")
    frame_root = Path(frame_output_root) if frame_output_root else None
    examples: list[TrajectoryExample] = []
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        for record in summary.get("episodes", []):
            episode_id = int(record.get("episode_id", len(examples)))
            traj_id = f"robotwin/{task_dir.name}/episode{episode_id:03d}"
            video_path = _find_robotwin_episode_video(task_dir, episode_id)
            frame_dir = Path(video_path).with_name(f"{Path(video_path).stem}_frames") if video_path else None
            frames = _collect_frames(frame_dir) or _extract_from_video(video_path, frame_root, traj_id, fps, overwrite_frames)
            examples.append(
                TrajectoryExample(
                    traj_id=traj_id,
                    task=str(summary.get("task_name") or task_dir.name),
                    frames=frames,
                    source="robotwin_eval",
                    split=split,
                    metadata={
                        "task_config": summary.get("task_config"),
                        "episode_id": episode_id,
                        "seed": record.get("seed"),
                        "success": bool(record.get("success", False)),
                        "steps": record.get("steps"),
                        "video_path": video_path,
                        "task_success_rate": summary.get("success_rate"),
                    },
                )
            )
    write_jsonl(output, examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MI reward manifests from frames or LaWAM eval outputs.")
    parser.add_argument("--source_type", choices=["frames", "libero", "robotwin"], default="frames")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--frame_root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", default="generated")
    parser.add_argument("--frame_output_root", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--overwrite_frames", action="store_true")
    args = parser.parse_args()
    if args.source_type == "frames":
        if args.frame_root is None:
            parser.error("--frame_root is required for --source_type frames")
        examples = build_manifest(args.frame_root, args.output, args.split, args.source)
    elif args.source_type == "libero":
        if args.run_dir is None:
            parser.error("--run_dir is required for --source_type libero")
        frame_output_root = args.frame_output_root or "dataset/mi_reward/extracted_frames/libero"
        examples = build_libero_manifest(
            args.run_dir,
            args.output,
            frame_output_root=frame_output_root,
            fps=args.fps,
            split=args.split,
            overwrite_frames=args.overwrite_frames,
        )
    else:
        if args.run_dir is None:
            parser.error("--run_dir is required for --source_type robotwin")
        frame_output_root = args.frame_output_root or "dataset/mi_reward/extracted_frames/robotwin"
        examples = build_robotwin_manifest(
            args.run_dir,
            args.output,
            frame_output_root=frame_output_root,
            fps=args.fps,
            split=args.split,
            overwrite_frames=args.overwrite_frames,
        )
    print(f"Wrote {len(examples)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
