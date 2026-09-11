"""Shared Qwen3-VL reward-model data utilities for Pipeline v3.

The exported teacher JSONL deliberately keeps privileged diagnostics in a
separate file.  This module consumes only the deployment-compatible Qwen rows:

    task language + dual-view visual history -> Positive / Unclear / Negative

It converts the compact export schema into the structured multimodal chat
format expected by ``Qwen3VLProcessor``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = ("Positive", "Unclear", "Negative")


def read_qwen_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {source}:{line_no}.")
        _validate_row(payload, source=source, line_no=line_no)
        rows.append(payload)
    if not rows:
        raise ValueError(f"Qwen JSONL is empty: {source}")
    return rows


def _validate_row(row: dict[str, Any], *, source: Path, line_no: int) -> None:
    messages = row.get("messages")
    images = row.get("images")
    provenance = row.get("provenance")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Expected two messages at {source}:{line_no}.")
    if not isinstance(images, list) or not images:
        raise ValueError(f"Expected image list at {source}:{line_no}.")
    if not isinstance(provenance, dict):
        raise ValueError(f"Expected provenance at {source}:{line_no}.")
    label = str(messages[-1].get("content", ""))
    if label not in LABELS:
        raise ValueError(f"Unexpected P/U/N label {label!r} at {source}:{line_no}.")
    history = provenance.get("history_frame_indices")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Missing history_frame_indices at {source}:{line_no}.")
    if len(images) != 2 * len(history):
        raise ValueError(
            f"Dual-view image count mismatch at {source}:{line_no}: "
            f"images={len(images)} history={len(history)}"
        )
    forbidden = {
        "privileged_diagnostics",
        "privileged_evidence",
        "directional_score",
        "physical_direction",
        "phi_visual_t",
        "conditional_action_information",
    }
    leaked = forbidden.intersection(row)
    if leaked:
        raise ValueError(f"Privileged/teacher fields leaked into Qwen row at {source}:{line_no}: {sorted(leaked)}")


def row_label(row: dict[str, Any]) -> str:
    return str(row["messages"][-1]["content"])


def label_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row_label(row) for row in rows)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def select_evenly(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Select an evenly spaced deterministic subset for smoke tests."""

    if limit is None or limit <= 0 or limit >= len(rows):
        return list(rows)
    if limit == 1:
        return [rows[len(rows) // 2]]
    indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]
    return [rows[index] for index in indices]


def build_multimodal_messages(
    row: dict[str, Any],
    *,
    project_root: str | Path,
    include_answer: bool,
) -> list[dict[str, Any]]:
    """Convert compact exporter rows to Qwen structured chat messages.

    View identity is supplied as ordinary deployment-compatible text.  No
    privileged state or teacher diagnostics are added.
    """

    root = Path(project_root).resolve()
    images = [Path(str(value)) for value in row["images"]]
    history_count = len(row["provenance"]["history_frame_indices"])
    agent = images[:history_count]
    wrist = images[history_count:]

    def resolve(path: Path) -> str:
        candidate = path if path.is_absolute() else root / path
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return str(candidate.resolve())

    user_text = str(row["messages"][0]["content"])
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": user_text + "\nAgent-view history (oldest to newest):",
        }
    ]
    content.extend({"type": "image", "image": resolve(path)} for path in agent)
    content.append({"type": "text", "text": "Wrist-view history (oldest to newest):"})
    content.extend({"type": "image", "image": resolve(path)} for path in wrist)
    content.append(
        {
            "type": "text",
            "text": "Return exactly one label: Positive, Unclear, or Negative.",
        }
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row_label(row)}],
            }
        )
    return messages


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _debug_cli() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL reward data smoke test")
    parser.add_argument("--jsonl", required=True, help="Qwen JSONL export path")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    print(f"jsonl path: {jsonl_path}")

    rows = read_qwen_jsonl(jsonl_path)
    print(f"total samples: {len(rows)}")
    print(f"label distribution: {label_counts(rows)}")

    samples = select_evenly(rows, args.num_samples)
    print(f"selected samples: {len(samples)}")

    for index, row in enumerate(samples):
        messages = row.get("messages", [])
        images = row.get("images", [])
        print(f"\n[sample {index}]")
        print(f"label: {row_label(row)}")
        print(f"messages: {len(messages)}")
        print(f"images: {len(images)}")
        for image in images:
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = Path(args.project_root) / image_path
            print(f"  image exists: {image_path.is_file()} -> {image_path}")

        # Reuse the deployment conversion path only for validation; no schema changes.
        build_multimodal_messages(
            row,
            project_root=args.project_root,
            include_answer=True,
        )
        print("  multimodal messages: OK")

    print("\nSMOKE TEST PASS")


if __name__ == "__main__":
    _debug_cli()

