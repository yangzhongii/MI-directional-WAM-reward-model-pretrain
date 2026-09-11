#!/usr/bin/env python3
"""Prepare the first formal LIBERO instance-adaptation set for Transfer2.5.

The goal is not to create physical object-swap ground truth.  Instead, this
dataset exposes Transfer2.5 to several task-equivalent object instances so a
reference-conditioned inference can change object identity while preserving
robot manipulation geometry.

Only the LIBERO train split is linked into the training directory.  Val/test
episodes are recorded separately for held-out replacement evaluation.
Video bytes are never copied; training samples are symlinks to the existing
dual-view export.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = ROOT / "logs/world_model/libero_dual_view"
ANNOTATION_ROOT = EXPORT_ROOT / "annotation"
OUT_ROOT = ROOT / "logs/world_model/transfer_instance_lora_v1"
DATASET_DIR = OUT_ROOT / "dataset"

# Keep the first formal run deliberately conservative: task semantics and
# destination stay fixed inside each family; only the manipulated object
# instance changes.
FAMILIES: dict[str, re.Pattern[str]] = {
    "mug_right_of_caddy": re.compile(
        r"^pick up the (red mug|white mug|yellow and white mug) "
        r"and place it to the right of the caddy$"
    ),
    "grocery_into_basket": re.compile(
        r"^pick up the (alphabet soup|bbq sauce|butter|chocolate pudding|"
        r"cream cheese|cream cheese box|ketchup|milk|orange juice|"
        r"salad dressing|tomato sauce) (?:and place it|and put it) in the basket$"
    ),
}

TRAIN_DEMOS_PER_TASK = 8
HELDOUT_DEMOS_PER_TASK = 2


def iter_annotations(split: str):
    for path in sorted((ANNOTATION_ROOT / split).glob("*.json")):
        record = json.loads(path.read_text())
        yield path, record


def classify(task: str) -> tuple[str, str] | None:
    for family, pattern in FAMILIES.items():
        match = pattern.match(task)
        if match:
            return family, match.group(1)
    return None


def main() -> None:
    videos_dir = DATASET_DIR / "videos"
    captions_dir = DATASET_DIR / "captions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for _, record in iter_annotations(split):
            hit = classify(record["task"])
            if hit is None:
                continue
            family, instance = hit
            grouped[(split, family, record["task"])].append(
                {"record": record, "instance": instance}
            )

    train_manifest: list[dict] = []
    heldout_manifest: list[dict] = []

    # Deterministic per-task sampling keeps future rebuilds identical.
    train_groups = sorted(k for k in grouped if k[0] == "train")
    for _, family, task in train_groups:
        items = sorted(grouped[("train", family, task)], key=lambda x: x["record"]["episode_id"])
        chosen = items[:TRAIN_DEMOS_PER_TASK]
        if not chosen:
            continue

        for item in chosen:
            record = item["record"]
            instance = item["instance"]
            for video in record["videos"]:
                camera = video["camera"]
                source = EXPORT_ROOT / video["video_path"]
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
                        f"The manipulated object instance is {instance}. "
                        "Preserve object pose, robot geometry, grasp/contact state, "
                        "scene layout, camera geometry, and motion trajectory."
                    )
                }
                (captions_dir / f"{sample_name}.json").write_text(json.dumps(caption))
                train_manifest.append(
                    {
                        "sample": sample_name,
                        "family": family,
                        "instance": instance,
                        "task": task,
                        "episode_id": record["episode_id"],
                        "camera": camera,
                        "source_video": str(source.resolve()),
                    }
                )

    # Held-out candidates are never linked into the training dataset.
    for split in ("val", "test"):
        for key in sorted(k for k in grouped if k[0] == split):
            _, family, task = key
            items = sorted(grouped[key], key=lambda x: x["record"]["episode_id"])
            for item in items[:HELDOUT_DEMOS_PER_TASK]:
                record = item["record"]
                heldout_manifest.append(
                    {
                        "split": split,
                        "family": family,
                        "instance": item["instance"],
                        "task": task,
                        "episode_id": record["episode_id"],
                        "videos": record["videos"],
                    }
                )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train_manifest.json").write_text(json.dumps(train_manifest, indent=2))
    (OUT_ROOT / "heldout_manifest.json").write_text(json.dumps(heldout_manifest, indent=2))

    counts = Counter((x["family"], x["instance"]) for x in train_manifest)
    report = {
        "train_view_samples": len(train_manifest),
        "heldout_episodes": len(heldout_manifest),
        "families": sorted({x["family"] for x in train_manifest}),
        "train_counts_by_family_instance": {
            f"{family}/{instance}": count
            for (family, instance), count in sorted(counts.items())
        },
        "train_split_only": True,
        "dual_view": True,
        "video_bytes_copied": False,
    }
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"dataset_dir={DATASET_DIR}")


if __name__ == "__main__":
    main()
