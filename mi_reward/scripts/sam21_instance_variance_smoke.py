"""Zero-shot SAM 2.1 video-mask smoke test for visual instance variance.

The script deliberately does not use a generative model.  SAM 2.1 tracks the
task object through the source video from one first-frame box prompt, then a
deterministic Lab-space edit turns the segmented red object toward white while
leaving every pixel outside the mask untouched.

Outputs:
  - sam21_mask.mp4: binary tracked mask
  - sam21_overlay.mp4: source video with a mask overlay
  - sam21_red_to_white.mp4: deterministic appearance variant
  - sam21_contact_sheet.jpg: frames 0/6/12/18/24 for quick inspection
  - sam21_metrics.json: per-frame mask area and bounding boxes
  - sam21_masks.npy: boolean [T,H,W] masks
"""

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(77.0, 0.0, 110.0, 38.0),
        help="First-frame object box in source-video pixel coordinates.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float, *, gray: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), not gray)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def white_appearance_variant(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Neutralize chroma and lift luminance only inside ``mask``.

    The original luminance still contributes strongly, so highlights/shadows
    from the source rendering remain visible instead of producing flat white.
    """

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    out_lab = lab.copy().astype(np.float32)
    l_chan = out_lab[..., 0]
    a_chan = out_lab[..., 1]
    b_chan = out_lab[..., 2]

    l_chan[mask] = 0.58 * l_chan[mask] + 0.42 * 235.0
    a_chan[mask] = 128.0 + 0.10 * (a_chan[mask] - 128.0)
    b_chan[mask] = 128.0 + 0.10 * (b_chan[mask] - 128.0)

    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    edited = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    output = frame_bgr.copy()
    output[mask] = edited[mask]
    return output


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def make_contact_sheet(
    source: list[np.ndarray], masks: list[np.ndarray], overlays: list[np.ndarray], edited: list[np.ndarray]
) -> np.ndarray:
    indices = [i for i in (0, 6, 12, 18, 24) if i < len(source)]
    rows: list[np.ndarray] = []
    labels = ("source", "mask", "overlay", "white")
    row_sets = (
        [source[i] for i in indices],
        [cv2.cvtColor((masks[i].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR) for i in indices],
        [overlays[i] for i in indices],
        [edited[i] for i in indices],
    )
    for label, images in zip(labels, row_sets):
        cells = []
        for idx, image in zip(indices, images):
            cell = image.copy()
            cv2.putText(cell, f"{label} f{idx}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    source_frames, fps = read_video(args.input_video)
    height, width = source_frames[0].shape[:2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    box = np.asarray(args.box, dtype=np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_id} on {device} ...", flush=True)
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id).to(device)

    with tempfile.TemporaryDirectory(prefix="sam21_frames_") as frame_dir:
        frame_dir_path = Path(frame_dir)
        for idx, frame in enumerate(source_frames):
            cv2.imwrite(str(frame_dir_path / f"{idx:05d}.jpg"), frame)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        masks_by_frame: dict[int, np.ndarray] = {}
        with torch.inference_mode(), autocast_context:
            state = predictor.init_state(video_path=str(frame_dir_path))
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                box=box,
            )
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                if len(obj_ids) != 1:
                    raise RuntimeError(f"Expected one tracked object, got ids={list(obj_ids)}")
                masks_by_frame[int(frame_idx)] = (mask_logits[0] > 0.0).detach().cpu().numpy().squeeze().astype(bool)

    missing = sorted(set(range(len(source_frames))) - set(masks_by_frame))
    if missing:
        raise RuntimeError(f"SAM 2.1 did not return masks for frames: {missing}")
    masks = [masks_by_frame[i] for i in range(len(source_frames))]
    np.save(args.output_dir / "sam21_masks.npy", np.stack(masks, axis=0))

    mask_frames = [(mask.astype(np.uint8) * 255) for mask in masks]
    overlay_frames: list[np.ndarray] = []
    edited_frames: list[np.ndarray] = []
    for frame, mask in zip(source_frames, masks):
        overlay = frame.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 255, 0])).astype(np.uint8)
        overlay_frames.append(overlay)
        edited_frames.append(white_appearance_variant(frame, mask))

    write_video(args.output_dir / "sam21_mask.mp4", mask_frames, fps, gray=True)
    write_video(args.output_dir / "sam21_overlay.mp4", overlay_frames, fps)
    write_video(args.output_dir / "sam21_red_to_white.mp4", edited_frames, fps)

    contact_sheet = make_contact_sheet(source_frames, masks, overlay_frames, edited_frames)
    cv2.imwrite(str(args.output_dir / "sam21_contact_sheet.jpg"), contact_sheet)

    metrics = {
        "input_video": str(args.input_video),
        "model_id": args.model_id,
        "first_frame_box_xyxy": box.tolist(),
        "fps": fps,
        "frame_count": len(source_frames),
        "frame_size_wh": [width, height],
        "mask_area_ratio": [float(mask.mean()) for mask in masks],
        "mask_bbox_xyxy": [mask_bbox(mask) for mask in masks],
    }
    with (args.output_dir / "sam21_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Done. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
