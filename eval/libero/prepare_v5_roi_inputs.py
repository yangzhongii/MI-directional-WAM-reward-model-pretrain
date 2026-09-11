"""Build fixed visual-region collections for the frozen v5 evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("object_roi", "object_centered_crop", "background_only"), required=True)
    return parser.parse_args()


def transform(image: Image.Image, mask_image: Image.Image, variant: str) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_image) > 0
    if not np.any(mask):
        raise ValueError("Object mask is empty.")
    if variant == "object_centered_crop":
        ys, xs = np.where(mask)
        center_y, center_x = int(np.floor(ys.mean())), int(np.floor(xs.mean()))
        padded = np.pad(rgb, ((32, 32), (32, 32), (0, 0)), mode="reflect")
        return np.ascontiguousarray(padded[center_y:center_y + 64, center_x:center_x + 64])
    if variant == "background_only":
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1) > 0
        selected, count = ~dilated, 4096
    else:
        selected, count = mask, 256
    indices = np.flatnonzero(selected.reshape(-1))
    if len(indices) == 0:
        raise ValueError(f"{variant} selected no pixels.")
    indices = np.resize(indices, count)
    return np.ascontiguousarray(rgb.reshape(-1, 3)[indices].reshape(count, 1, 3))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    report = json.loads((args.mask_dir / "masks.json").read_text())
    masks = {(a["anchor_id"], c["candidate_id"]): args.mask_dir / c["mask"] for a in report["anchors"] for c in a["candidates"]}
    args.output_dir.mkdir(parents=True)
    rows = []
    for line in (args.collection_dir / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        source_reference = record["reference"]["images"]["wrist"]
        reference = transform(Image.open(args.collection_dir / source_reference), Image.open(args.mask_dir / report["reference"]["mask"]), args.variant)
        Image.fromarray(reference).save(args.output_dir / "reference.png")
        record["reference"]["images"]["wrist"] = "reference.png"
        for candidate in record["candidates"]:
            relative = Path("frames") / record["anchor_id"] / f"{candidate['candidate_id']}.png"
            output = args.output_dir / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            transformed = transform(Image.open(args.collection_dir / candidate["images"]["wrist"]), Image.open(masks[(record["anchor_id"], candidate["candidate_id"])]), args.variant)
            Image.fromarray(transformed).save(output)
            candidate["images"]["wrist"] = str(relative)
        rows.append(record)
    (args.output_dir / "manifest.jsonl").write_text("\n".join(json.dumps(record) for record in rows) + "\n")
    (args.output_dir / "collection.json").write_text(json.dumps({"source": str(args.collection_dir), "mask_dir": str(args.mask_dir), "variant": args.variant, "anchors": len(rows)}, indent=2) + "\n")
    print(json.dumps({"variant": args.variant, "anchors": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
