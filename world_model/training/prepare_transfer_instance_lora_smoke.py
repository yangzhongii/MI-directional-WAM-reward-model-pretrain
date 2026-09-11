#!/usr/bin/env python3
"""Prepare a tiny LIBERO red/white-mug dataset for Transfer2.5 LoRA smoke training.

The script does not copy video bytes. It creates symlinks into the existing
LIBERO dual-view export and writes one caption JSON per linked view.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "logs/world_model/libero_dual_view/annotation/train"
VIDEO_ROOT = ROOT / "logs/world_model/libero_dual_view"
DEFAULT_OUT = ROOT / "logs/world_model/transfer_instance_lora_smoke/dataset"

TASKS = {
    "red": "libero-libero_90-task-84-demo_",
    "white": "libero-libero_90-task-85-demo_",
}
DEMOS_PER_INSTANCE = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", choices=["red", "white", "both"], default="both")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    selected_tasks = TASKS if args.instance == "both" else {args.instance: TASKS[args.instance]}

    videos_dir = out / "videos"
    captions_dir = out / "captions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for instance_name, prefix in selected_tasks.items():
        annotations = sorted(ANNOTATION_DIR.glob(f"{prefix}*.json"))[:DEMOS_PER_INSTANCE]
        if len(annotations) < DEMOS_PER_INSTANCE:
            raise RuntimeError(f"Need {DEMOS_PER_INSTANCE} annotations for {prefix}, got {len(annotations)}")

        for ann_path in annotations:
            record = json.loads(ann_path.read_text())
            task = record["task"]
            for video in record["videos"]:
                camera = video["camera"]
                source = VIDEO_ROOT / video["video_path"]
                if not source.is_file():
                    raise FileNotFoundError(source)

                sample_name = f"{record['episode_id']}__{camera}"
                link = videos_dir / f"{sample_name}.mp4"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(source.resolve())

                caption = {
                    "caption": (
                        f"A LIBERO robot manipulation video from the {camera} camera. "
                        f"The robot performs the task: {task}. "
                        f"Preserve the manipulated {instance_name} mug identity, robot geometry, "
                        "grasp/contact state, scene layout, and motion trajectory."
                    )
                }
                (captions_dir / f"{sample_name}.json").write_text(json.dumps(caption))
                manifest.append(
                    {
                        "sample": sample_name,
                        "instance": instance_name,
                        "episode_id": record["episode_id"],
                        "camera": camera,
                        "task": task,
                        "source_video": str(source.resolve()),
                    }
                )

    (out.parent / f"manifest_{args.instance}.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared_samples={len(manifest)}")
    print(f"dataset_dir={out}")


if __name__ == "__main__":
    main()
