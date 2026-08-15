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


def build_cosmos_manifest(
    task_specs: list[dict[str, str]],
    output_dir: str | Path,
    *,
    num_candidates_per_task: int = 10,
    num_frames: int = 16,
    split: str = "train",
    source: str = "cosmos_predict",
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate candidate futures with a world model and build manifests.

    Uses ``MockFutureGenerator`` by default (no Cosmos dependency).
    To use Cosmos-Predict, pass a ``CosmosPredictGenerator`` instance and
    set ``use_cosmos=True``.

    Args:
        task_specs: list of {"task": str, "init_frame": path} dicts.
        output_dir: root for generated frames + manifest files.
        num_candidates_per_task: how many futures to generate per task.
        num_frames: frames per trajectory.
        split: manifest split name.
        source: manifest source tag.
        seed: RNG seed for deterministic generation.

    Returns:
        (manifest_path, success_refs_path)
    """
    from mi_reward.data.cosmos_generator import MockFutureGenerator, DiffusersCosmosGenerator, save_video_frames
    from mi_reward.data.schema import SuccessReference

    root = Path(output_dir)
    frame_root = root / "frames"
    manifest_dir = root / "manifests"
    frame_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Choose generator: mock or real diffusers (check first task spec for the flag)
    use_real = any(s.get("use_real_cosmos") for s in task_specs)
    if use_real:
        generator = DiffusersCosmosGenerator(
            model_id="weights/cosmos-diffusers-2b",
            variant="post-trained",
            num_steps=10,
            temperature=1.0,
            ref_temperature=0.3,
        )
        print(f"Using DiffusersCosmosGenerator (real Cosmos-Predict2.5 on 4090)")
    else:
        generator = MockFutureGenerator(seed=seed)
        print(f"Using MockFutureGenerator (synthetic, seed={seed})")

    examples: list[TrajectoryExample] = []
    refs: list[SuccessReference] = []

    for spec in task_specs:
        task = spec["task"]
        prompt = spec.get("prompt", task)
        init_path = Path(spec["init_frame"])
        if not init_path.exists():
            raise FileNotFoundError(f"Initial frame not found: {init_path}")
        init_img = np.array(Image.open(init_path).convert("RGB"))
        # Optional goal image for Video2World
        goal_img = None
        if spec.get("goal_frame"):
            gp = Path(spec["goal_frame"])
            if gp.exists():
                goal_img = np.array(Image.open(gp).convert("RGB"))

        # Generate candidates
        videos = generator.generate_candidates(
            init_img, prompt, num_candidates=num_candidates_per_task, num_frames=num_frames,
            goal_image=goal_img,
        )
        for i, video in enumerate(videos):
            traj_id = f"cosmos/{task.replace(' ', '_')}/candidate_{i:03d}"
            out_dir = frame_root / traj_id.replace("/", "__")
            frames = save_video_frames(video, out_dir)
            # Also save as MP4 for visual review
            try:
                import imageio
                imageio.mimsave(str(out_dir / "video.mp4"), video, fps=8)
            except Exception:
                pass
            examples.append(TrajectoryExample(
                traj_id=traj_id, task=task, frames=frames,
                source=source, split=split,
                metadata={"candidate_index": i, "generator": "cosmos_diffusers" if goal_img is not None else "cosmos_image2world"},
            ))

        # Generate success reference
        ref_video = generator.generate_reference(init_img, prompt, num_frames, goal_image=goal_img)
        ref_id = f"cosmos/{task.replace(' ', '_')}/success_ref"
        ref_dir = frame_root / ref_id.replace("/", "__")
        ref_frames = save_video_frames(ref_video, ref_dir)
        try:
            import imageio
            imageio.mimsave(str(ref_dir / "video.mp4"), ref_video, fps=8)
        except Exception:
            pass
        refs.append(SuccessReference(ref_id=ref_id, task=task, frames=ref_frames))

    manifest_path = manifest_dir / "train_manifest.jsonl"
    refs_path = manifest_dir / "success_refs.jsonl"
    write_jsonl(str(manifest_path), examples)
    write_jsonl(str(refs_path), refs)
    return manifest_path, refs_path


import numpy as np  # noqa: E402
from PIL import Image     # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MI reward manifests from frames or LaWAM eval outputs.")
    parser.add_argument("--source_type", choices=["frames", "libero", "robotwin", "cosmos", "cosmos_action_cond"], default="frames")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--frame_root", default=None)
    parser.add_argument("--output", default=None, help="Output manifest path (required for frames/libero/robotwin)")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", default="generated")
    parser.add_argument("--frame_output_root", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--overwrite_frames", action="store_true")
    # Cosmos-specific args
    parser.add_argument("--task_specs", default=None, help="JSONL task specs for --source_type cosmos")
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-cosmos", action="store_true", help="Use real Cosmos-Predict2.5 (not Mock)")
    parser.add_argument("--candidate_records", default=None, help="JSONL external robot/action-conditioned Cosmos outputs.")
    parser.add_argument("--feasibility_config", default=None, help="JSON feasibility constraints for --source_type cosmos_action_cond.")
    args = parser.parse_args()
    if args.source_type == "frames":
        if args.frame_root is None or args.output is None:
            parser.error("--frame_root and --output are required for --source_type frames")
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
    elif args.source_type == "robotwin":
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
    elif args.source_type == "cosmos":
        import json as _json
        if args.task_specs is None or args.output_dir is None:
            parser.error("--task_specs and --output_dir are required for --source_type cosmos")
        # JSONL: read line by line
        task_specs = []
        for line in Path(args.task_specs).read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                task_specs.append(_json.loads(line.strip()))
        out = Path(args.output_dir)
        # Pass --real-cosmos as a special key in each task spec
        if args.real_cosmos:
            for s in task_specs:
                s["use_real_cosmos"] = True
        manifest_path, refs_path = build_cosmos_manifest(
            task_specs, out,
            num_candidates_per_task=args.num_candidates,
            num_frames=args.num_frames,
            split=args.split,
            source=args.source,
            seed=args.seed,
        )
        print(f"Wrote manifest to {manifest_path}")
        print(f"Wrote success refs to {refs_path}")
        return
    elif args.source_type == "cosmos_action_cond":
        if args.candidate_records is None or args.output is None or args.feasibility_config is None:
            parser.error("--candidate_records, --output, and --feasibility_config are required for --source_type cosmos_action_cond")
        from mi_reward.data.cosmos_action_cond import feasibility_config_from_dict, ingest_action_conditioned_candidates

        config_payload = _read_json(Path(args.feasibility_config))
        examples = ingest_action_conditioned_candidates(
            args.candidate_records,
            args.output,
            feasibility_config_from_dict(config_payload),
            split=args.split,
        )
    print(f"Wrote {len(examples)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
