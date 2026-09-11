"""SAM 2.1 zero-shot robustness benchmark on LIBERO-Spatial.

The benchmark deliberately separates *initialization* from *propagation*.
Each task/view receives one first-frame box prompt.  SAM 2.1 is then evaluated
for temporal propagation only, which is the part relevant to deciding whether
domain fine-tuning is necessary for the MI instance-variance pipeline.

Typical workflow:

1. Prepare first frames for manual/grounded box inspection::

     python mi_reward/scripts/sam21_robustness_benchmark.py \
       --output-dir logs/mi_reward/sam21_robustness_spatial10 --prepare-only

2. Fill a boxes JSON with one XYXY box per task/view and run the benchmark::

     python mi_reward/scripts/sam21_robustness_benchmark.py \
       --output-dir logs/mi_reward/sam21_robustness_spatial10 \
       --boxes-json logs/mi_reward/sam21_robustness_spatial10/boxes.json

The model is loaded once and reused for all videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Episode:
    task_id: int
    language: str
    source: Path
    demo_name: str
    agentview: np.ndarray
    wrist: np.ndarray
    obs_keys: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--save-videos", action="store_true")
    return parser.parse_args()


def libero_episode(root: Path, suite: str, task_id: int, demo_index: int) -> Episode:
    import h5py

    libero_source = root / ".venv" / "src" / "libero"
    if libero_source.is_dir() and str(libero_source) not in sys.path:
        sys.path.insert(0, str(libero_source))
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    task_suite = suites[suite]()
    relative = Path(task_suite.get_task_demonstration(task_id))
    candidates = (
        root / ".venv" / "src" / "libero" / "libero" / "datasets" / relative,
        root / ".venv" / "datasets" / relative,
        root / relative,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"LIBERO demonstration is missing: {relative}")

    with h5py.File(source, "r") as handle:
        names = sorted(handle["data"].keys())
        if not names:
            raise ValueError(f"No demonstrations in {source}")
        demo_name = names[int(demo_index) % len(names)]
        obs = handle["data"][demo_name]["obs"]
        agentview = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
        gripper = np.asarray(obs["gripper_states"], dtype=np.float32)
        obs_keys = tuple(sorted(str(key) for key in obs.keys()))

    length = min(len(agentview), len(wrist))
    agentview = agentview[:length]
    wrist = wrist[:length]
    gripper = gripper[:length]

    # Center the robustness clip on the first decisive gripper closing event.
    # This avoids penalizing wrist-view tracking simply because the target is
    # outside the initial field of view, and stresses the actual manipulation
    # phase where occlusion/contact are most challenging.
    width = np.abs(gripper).sum(axis=1)
    closed = np.clip((0.06 - width) / (0.06 - 0.02), 0.0, 1.0)
    candidates = np.flatnonzero(closed >= 0.5)
    center = int(candidates[0]) if len(candidates) else length // 2
    window = min(25, length)
    start = max(0, min(center - window // 2, length - window))
    stop = start + window
    return Episode(
        task_id=int(task_id),
        language=str(task_suite.get_task(task_id).language),
        source=source,
        demo_name=demo_name,
        agentview=agentview[start:stop],
        wrist=wrist[start:stop],
        obs_keys=obs_keys,
    )


def write_rgb_clip(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create prepared clip: {path}")
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def read_rgb_clip(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open prepared clip: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from prepared clip: {path}")
    return np.stack(frames, axis=0)


def load_prepared_episodes(output_dir: Path, task_ids: list[int]) -> list[Episode]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest is missing: {manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_task = {int(record["task_id"]): record for record in records}
    episodes: list[Episode] = []
    for task_id in task_ids:
        record = by_task.get(int(task_id))
        if record is None:
            raise ValueError(f"Prepared manifest does not contain task {task_id}")
        clips = record.get("clips", {})
        agent_path = Path(str(clips.get("agentview", "")))
        wrist_path = Path(str(clips.get("wrist", "")))
        if not agent_path.is_absolute():
            agent_path = output_dir / agent_path
        if not wrist_path.is_absolute():
            wrist_path = output_dir / wrist_path
        agentview = read_rgb_clip(agent_path)
        wrist = read_rgb_clip(wrist_path)
        length = min(len(agentview), len(wrist))
        episodes.append(
            Episode(
                task_id=int(task_id),
                language=str(record["language"]),
                source=Path(str(record["source"])),
                demo_name=str(record["demo_name"]),
                agentview=agentview[:length],
                wrist=wrist[:length],
                obs_keys=tuple(str(value) for value in record.get("obs_keys", [])),
            )
        )
    return episodes


def label_image(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 17), (0, 0, 0), -1)
    cv2.putText(image, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_prepare_outputs(episodes: list[Episode], output_dir: Path, fps: float) -> None:
    first_dir = output_dir / "first_frames"
    clip_dir = output_dir / "clips"
    first_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    agent_cells: list[np.ndarray] = []
    wrist_cells: list[np.ndarray] = []

    for episode in episodes:
        agent_path = first_dir / f"task_{episode.task_id:02d}_agentview.png"
        wrist_path = first_dir / f"task_{episode.task_id:02d}_wrist.png"
        agent_clip = clip_dir / f"task_{episode.task_id:02d}_agentview.mp4"
        wrist_clip = clip_dir / f"task_{episode.task_id:02d}_wrist.mp4"
        cv2.imwrite(str(agent_path), cv2.cvtColor(episode.agentview[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(wrist_path), cv2.cvtColor(episode.wrist[0], cv2.COLOR_RGB2BGR))
        write_rgb_clip(agent_clip, episode.agentview, fps)
        write_rgb_clip(wrist_clip, episode.wrist, fps)
        agent_cells.append(label_image(episode.agentview[0], f"t{episode.task_id:02d} agent"))
        wrist_cells.append(label_image(episode.wrist[0], f"t{episode.task_id:02d} wrist"))
        manifest.append(
            {
                "task_id": episode.task_id,
                "language": episode.language,
                "source": str(episode.source),
                "demo_name": episode.demo_name,
                "frame_count": int(len(episode.agentview)),
                "frame_size_wh": [int(episode.agentview.shape[2]), int(episode.agentview.shape[1])],
                "obs_keys": list(episode.obs_keys),
                "first_frames": {"agentview": str(agent_path), "wrist": str(wrist_path)},
                "clips": {
                    "agentview": str(agent_clip.relative_to(output_dir)),
                    "wrist": str(wrist_clip.relative_to(output_dir)),
                },
            }
        )

    def sheet(cells: list[np.ndarray]) -> np.ndarray:
        rows = []
        for start in range(0, len(cells), 5):
            rows.append(np.concatenate(cells[start : start + 5], axis=1))
        return np.concatenate(rows, axis=0)

    cv2.imwrite(str(output_dir / "first10_agentview.jpg"), sheet(agent_cells))
    cv2.imwrite(str(output_dir / "first10_wrist.jpg"), sheet(wrist_cells))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    boxes_template = {
        str(episode.task_id): {"agentview": None, "wrist": None, "language": episode.language}
        for episode in episodes
    }
    template_path = output_dir / "boxes.template.json"
    if not template_path.exists():
        template_path.write_text(json.dumps(boxes_template, indent=2) + "\n", encoding="utf-8")


def write_video(path: Path, frames_bgr: list[np.ndarray], fps: float, *, gray: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), not gray)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def track_masks(
    predictor: Any,
    frames_rgb: np.ndarray,
    box_xyxy: list[float],
    device: Any,
) -> list[np.ndarray]:
    import torch

    with tempfile.TemporaryDirectory(prefix="sam21_robustness_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, frame_rgb in enumerate(frames_rgb):
            cv2.imwrite(str(temp_path / f"{index:05d}.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        result: dict[int, np.ndarray] = {}
        with torch.inference_mode(), context:
            state = predictor.init_state(video_path=str(temp_path))
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=np.asarray(box_xyxy, dtype=np.float32),
            )
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
                if len(obj_ids) != 1:
                    raise RuntimeError(f"Expected one object, got {list(obj_ids)}")
                result[int(frame_idx)] = (logits[0] > 0.0).detach().cpu().numpy().squeeze().astype(bool)

    missing = sorted(set(range(len(frames_rgb))) - set(result))
    if missing:
        raise RuntimeError(f"Missing propagated masks at frames: {missing[:20]}")
    return [result[index] for index in range(len(frames_rgb))]


def temporal_metrics(masks: list[np.ndarray]) -> dict[str, Any]:
    areas = np.asarray([float(mask.mean()) for mask in masks], dtype=np.float64)
    centers = [centroid(mask) for mask in masks]
    height, width = masks[0].shape
    diagonal = float(math.hypot(width, height))

    area_log_jumps = []
    centroid_jumps = []
    for index in range(1, len(masks)):
        prev_area = max(areas[index - 1], 1e-8)
        area = max(areas[index], 1e-8)
        area_log_jumps.append(float(abs(math.log(area / prev_area))))
        if centers[index - 1] is None or centers[index] is None:
            centroid_jumps.append(float("inf"))
        else:
            dx = centers[index][0] - centers[index - 1][0]
            dy = centers[index][1] - centers[index - 1][1]
            centroid_jumps.append(float(math.hypot(dx, dy) / diagonal))

    nonempty_fraction = float(np.mean(areas > 0.0))
    max_area_log_jump = max(area_log_jumps, default=0.0)
    max_centroid_jump = max(centroid_jumps, default=0.0)
    catastrophic = bool(
        nonempty_fraction < 1.0
        or max_area_log_jump > math.log(2.5)
        or max_centroid_jump > 0.25
        or float(areas.max(initial=0.0)) > 0.70
    )
    return {
        "frame_count": len(masks),
        "nonempty_fraction": nonempty_fraction,
        "area_ratio_mean": float(areas.mean()),
        "area_ratio_min": float(areas.min()),
        "area_ratio_max": float(areas.max()),
        "max_adjacent_area_log_jump": float(max_area_log_jump),
        "max_adjacent_centroid_jump_diag": float(max_centroid_jump),
        "catastrophic_self_consistency_failure": catastrophic,
        "mask_bbox_xyxy": [bbox(mask) for mask in masks],
    }


def review_sheet(frames_rgb: np.ndarray, masks: list[np.ndarray], task_label: str) -> np.ndarray:
    count = len(frames_rgb)
    indices = np.linspace(0, count - 1, num=min(5, count), dtype=int).tolist()
    source_cells: list[np.ndarray] = []
    mask_cells: list[np.ndarray] = []
    overlay_cells: list[np.ndarray] = []
    for idx in indices:
        source = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR)
        mask = masks[idx]
        mask_bgr = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        overlay = source.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 255, 0])).astype(np.uint8)
        for image, row_name in ((source, "src"), (mask_bgr, "mask"), (overlay, "ovr")):
            cv2.putText(
                image,
                f"{task_label} {row_name} f{idx}",
                (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        source_cells.append(source)
        mask_cells.append(mask_bgr)
        overlay_cells.append(overlay)
    return np.concatenate(
        [np.concatenate(source_cells, axis=1), np.concatenate(mask_cells, axis=1), np.concatenate(overlay_cells, axis=1)],
        axis=0,
    )


def load_boxes(path: Path, task_ids: list[int]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for task_id in task_ids:
        task = data.get(str(task_id))
        if not isinstance(task, dict):
            raise ValueError(f"Missing task {task_id} in {path}")
        for camera in ("agentview", "wrist"):
            box_value = task.get(camera)
            if not isinstance(box_value, list) or len(box_value) != 4:
                raise ValueError(f"Task {task_id} camera {camera} needs one [x0,y0,x1,y1] box.")
    return data


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        episodes = [libero_episode(root, args.suite, task_id, args.demo_index) for task_id in args.tasks]
        save_prepare_outputs(episodes, args.output_dir, args.fps)
        print(f"Prepared first frames and manifest at {args.output_dir}")
        return
    episodes = load_prepared_episodes(args.output_dir, args.tasks)
    if args.boxes_json is None:
        raise ValueError("--boxes-json is required unless --prepare-only is set.")

    boxes = load_boxes(args.boxes_json, args.tasks)
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_id} on {device} ...", flush=True)
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id).to(device)

    results: list[dict[str, Any]] = []
    review_dir = args.output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"

    for episode in episodes:
        for camera, frames_rgb in (("agentview", episode.agentview), ("wrist", episode.wrist)):
            box_xyxy = [float(value) for value in boxes[str(episode.task_id)][camera]]
            print(
                f"task={episode.task_id:02d} camera={camera} frames={len(frames_rgb)} box={box_xyxy}",
                flush=True,
            )
            masks = track_masks(predictor, frames_rgb, box_xyxy, device)
            metrics = temporal_metrics(masks)
            record = {
                "task_id": episode.task_id,
                "language": episode.language,
                "demo_name": episode.demo_name,
                "camera": camera,
                "box_xyxy": box_xyxy,
                **{key: value for key, value in metrics.items() if key != "mask_bbox_xyxy"},
            }
            results.append(record)

            sheet = review_sheet(frames_rgb, masks, f"t{episode.task_id:02d}-{camera[:1]}")
            cv2.imwrite(str(review_dir / f"task_{episode.task_id:02d}_{camera}.jpg"), sheet)

            detail = {**record, "mask_bbox_xyxy": metrics["mask_bbox_xyxy"]}
            (review_dir / f"task_{episode.task_id:02d}_{camera}.json").write_text(
                json.dumps(detail, indent=2) + "\n", encoding="utf-8"
            )

            if args.save_videos:
                mask_frames = [(mask.astype(np.uint8) * 255) for mask in masks]
                overlay_frames: list[np.ndarray] = []
                for frame_rgb, mask in zip(frames_rgb, masks):
                    overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 255, 0])).astype(np.uint8)
                    overlay_frames.append(overlay)
                write_video(video_dir / f"task_{episode.task_id:02d}_{camera}_mask.mp4", mask_frames, args.fps, gray=True)
                write_video(video_dir / f"task_{episode.task_id:02d}_{camera}_overlay.mp4", overlay_frames, args.fps)

    passed = sum(not bool(item["catastrophic_self_consistency_failure"]) for item in results)
    summary = {
        "model_id": args.model_id,
        "suite": args.suite,
        "demo_index": args.demo_index,
        "tracks": len(results),
        "self_consistency_passed": passed,
        "self_consistency_pass_rate": passed / max(len(results), 1),
        "criteria": {
            "all_masks_nonempty": True,
            "adjacent_area_ratio_change_less_than": 2.5,
            "adjacent_centroid_jump_less_than_image_diagonal_fraction": 0.25,
            "mask_area_ratio_less_than": 0.70,
            "note": "These are catastrophic-drift screens, not pixel-level ground truth metrics; review sheets remain required.",
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(results[0].keys()) if results else []
    if fieldnames:
        with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
