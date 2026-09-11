"""LIBERO generalization evaluation adapter for Qwen3-VL reward models.

This module bridges the existing LIBERO generalization manifests and the
deployable Qwen3-VL reward inference interface.  It intentionally does not
create new training labels; it only converts trajectory artifacts into the
same multimodal row schema used by Robometer evaluation.

Expected input manifest entries may contain:
    - instruction / language
    - agentview and wrist image paths (or frame paths)
    - success label (optional for pure reward inference)
    - provenance metadata

The output rows follow:
    messages + images + provenance

so they can be consumed by qwen3_vl_reward.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _first_present(record: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def build_qwen_row(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a LIBERO generalization trajectory record to Qwen3-VL format."""

    language = _first_present(
        record,
        ["instruction", "language", "task_description", "goal"],
        "",
    )

    images = _first_present(
        record,
        ["images", "image_paths", "frames"],
        [],
    )

    if isinstance(images, dict):
        images = list(images.values())

    return {
        "messages": [
            {
                "role": "user",
                "content": str(language),
            }
        ],
        "images": list(images),
        "provenance": {
            "source": "libero_generalization",
            "trajectory_id": record.get("trajectory_id", record.get("id", "unknown")),
            "history_frame_indices": record.get("history_frame_indices", []),
        },
        "metadata": {
            "success": record.get("success"),
            "variant_type": record.get("variant_type", "unknown"),
            "source": "libero_generalization",
        },
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_manifest(input_path: str, output_path: str) -> None:
    records = load_manifest(input_path)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(build_qwen_row(record)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()

