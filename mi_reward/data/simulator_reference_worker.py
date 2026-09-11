"""Pipeline worker that separates MuJoCo success references from candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mi_reward.data.simulator_native import build_simulator_references


def run_worker(request_path: Path, result_path: Path, success_refs: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    report = build_simulator_references(
        request["input_records"],
        request["output_records"],
        success_refs,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent simulator-native success references.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--success-refs", required=True)
    args = parser.parse_args()
    run_worker(Path(args.request), Path(args.result), Path(args.success_refs))


if __name__ == "__main__":
    main()
