"""Quantify whether a scene variant changes background more than preserved content.

The input preserve mask follows Cosmos guided-generation polarity:
white = task-relevant pixels that should remain close to the source,
black = scene pixels allowed to change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    selected = mask.astype(bool)
    if not np.any(selected):
        return float("nan")
    error = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    return float(error[selected].mean())


def evaluate(source: Path, variant: Path, preserve_mask: Path) -> dict[str, object]:
    source_frames = read_video(source)
    variant_frames = read_video(variant)
    mask_frames = read_video(preserve_mask)
    length = min(len(source_frames), len(variant_frames), len(mask_frames))
    inside: list[float] = []
    outside: list[float] = []
    whole: list[float] = []
    for index in range(length):
        src = source_frames[index]
        var = variant_frames[index]
        if src.shape[:2] != var.shape[:2]:
            var = cv2.resize(var, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_frame = mask_frames[index]
        if mask_frame.shape[:2] != src.shape[:2]:
            mask_frame = cv2.resize(
                mask_frame, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        preserve = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) >= 128
        inside.append(masked_mae(src, var, preserve))
        outside.append(masked_mae(src, var, ~preserve))
        whole.append(float(np.abs(src.astype(np.float32) - var.astype(np.float32)).mean()))

    inside_mean = float(np.nanmean(inside))
    outside_mean = float(np.nanmean(outside))
    return {
        "source": str(source),
        "variant": str(variant),
        "preserve_mask": str(preserve_mask),
        "frames": int(length),
        "preserve_mae_mean": inside_mean,
        "scene_mae_mean": outside_mean,
        "whole_mae_mean": float(np.mean(whole)),
        "scene_to_preserve_mae_ratio": float(outside_mean / max(inside_mean, 1e-6)),
        "preserve_mae_per_frame": inside,
        "scene_mae_per_frame": outside,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--preserve-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.source, args.variant, args.preserve_mask)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
