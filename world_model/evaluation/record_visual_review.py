"""Record a reproducible human review decision for Cosmos comparison videos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_review(
    *, validation_report: Path, output: Path, decision: str, reviewer: str, notes: str,
) -> dict[str, Any]:
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError("Validation report is not complete.")
    comparison_dir = Path(report["output"]) / "comparisons"
    videos = sorted(comparison_dir.glob("*.mp4"))
    if len(videos) != int(report.get("episodes", 0)):
        raise RuntimeError(
            f"Expected {report.get('episodes', 0)} comparison videos, found {len(videos)} in {comparison_dir}."
        )
    review = {
        "schema_version": 1,
        "passed": decision == "pass",
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "validation_report": str(validation_report.resolve()),
        "comparison_layout": report.get("comparison_layout"),
        "videos": [str(path.resolve()) for path in videos],
        "checklist": {
            "camera_and_first_frame_stable": decision == "pass",
            "action_direction_plausible": decision == "pass",
            "no_material_deformation_teleportation_or_penetration": decision == "pass",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--output")
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    report = Path(args.validation_report).expanduser()
    report = report.resolve() if report.is_absolute() else (root / report).resolve()
    output = Path(args.output).expanduser() if args.output else report.with_name("visual_review.json")
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    print(json.dumps(record_review(
        validation_report=report,
        output=output,
        decision=args.decision,
        reviewer=args.reviewer.strip(),
        notes=args.notes.strip(),
    ), indent=2))


if __name__ == "__main__":
    main()
