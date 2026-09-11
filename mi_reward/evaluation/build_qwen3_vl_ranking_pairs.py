"""Convert teacher preference pairs into the Qwen3-VL ranking contract.

The teacher preference builder deliberately stores trajectory ids and scores,
while the Qwen evaluator consumes full multimodal rows.  This small adapter
keeps the teacher side independent from the reward model side.

The input jsonl is expected to contain ``chosen_traj_id`` and
``rejected_traj_id``.  A trajectory lookup jsonl maps ids to Qwen rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: dict[str, dict[str, Any]] = {}
    with args.trajectory_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["traj_id"])] = row

    output: list[dict[str, Any]] = []
    with args.pairs_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            chosen = rows.get(str(pair["chosen_traj_id"]))
            rejected = rows.get(str(pair["rejected_traj_id"]))
            if chosen is None or rejected is None:
                continue
            output.append(
                {
                    "task": pair.get("task"),
                    "comparison_type": pair.get("comparison_type"),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for item in output:
            handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
