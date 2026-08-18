"""Run the configured external stages for an instance-aware data run.

SAM3, Cosmos, and simulator environments evolve independently and can require
different CUDA environments.  This module therefore owns their order and
artifact hand-off, not their model implementations.  Every stage receives a
request JSON and writes a result JSON pointing at its JSONL record output.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


PIPELINE_ORDER = ("sam3", "transfer", "simulator", "planner", "predict")


@dataclass(frozen=True)
class ExternalStage:
    name: str
    command: tuple[str, ...]
    input_records: str | None
    output_records: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExternalStage":
        command = payload.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError("Each external stage requires a non-empty string `command` list.")
        return cls(
            name=str(payload.get("name", "")),
            command=tuple(command),
            input_records=(None if payload.get("input_records") in (None, "") else str(payload["input_records"])),
            output_records=str(payload.get("output_records", "")),
        )


@dataclass(frozen=True)
class StageReport:
    name: str
    input_records: str
    output_records: str
    records: int
    status: str


def _read_jsonl_objects(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Stage record file not found: {path}")
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL produced by stage at {path}:{line_no}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Stage record at {path}:{line_no} must be a JSON object.")
        count += 1
    if count == 0:
        raise ValueError(f"Stage record file is empty: {path}")
    return count


def parse_stages(config: dict[str, Any]) -> list[ExternalStage]:
    execution = config.get("execution", {})
    if execution is None:
        return []
    if not isinstance(execution, dict):
        raise ValueError("`execution` must be a mapping when specified.")
    raw_stages = execution.get("stages", [])
    if raw_stages is None:
        return []
    if not isinstance(raw_stages, list):
        raise ValueError("`execution.stages` must be a list.")
    stages = [ExternalStage.from_dict(item) for item in raw_stages if isinstance(item, dict)]
    if len(stages) != len(raw_stages):
        raise ValueError("Every `execution.stages` entry must be a mapping.")
    names = [stage.name for stage in stages]
    if any(name not in PIPELINE_ORDER for name in names):
        invalid = sorted(set(name for name in names if name not in PIPELINE_ORDER))
        raise ValueError(f"Unsupported external stage name(s): {invalid}; expected a subset of {PIPELINE_ORDER}.")
    if len(names) != len(set(names)):
        raise ValueError("External stage names must be unique.")
    order = [PIPELINE_ORDER.index(name) for name in names]
    if order != sorted(order):
        raise ValueError(f"External stages must use fixed order: {PIPELINE_ORDER}.")
    for stage in stages:
        if not stage.name or not stage.output_records:
            raise ValueError("Each external stage requires `name` and `output_records`.")
        if "{request}" not in stage.command or "{result}" not in stage.command:
            raise ValueError(f"Stage {stage.name!r} command must include {{request}} and {{result}} placeholders.")
    return stages


def _resolved_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_configured_stages(config: dict[str, Any], *, root: str | Path = ".", dry_run: bool = False) -> list[StageReport]:
    """Validate or execute the configured SAM3/Transfer/simulation pipeline.

    The previous stage's JSONL becomes the next stage's input unless an
    explicit ``input_records`` path is supplied.  Workers write
    ``{"records_path": "..."}`` to their result file; the path must match the
    configured output so accidental cross-run hand-offs are rejected.
    """

    base = Path(root).resolve()
    stages = parse_stages(config)
    preparation = config.get("data_preparation", {})
    python_executable = ".venv/bin/python"
    if isinstance(preparation, dict) and preparation.get("python"):
        python_executable = str(preparation["python"])
    reports: list[StageReport] = []
    previous_output: Path | None = None
    for stage in stages:
        input_path = _resolved_path(stage.input_records, base) if stage.input_records else previous_output
        if input_path is None:
            raise ValueError(f"First configured stage {stage.name!r} requires `input_records`.")
        output_path = _resolved_path(stage.output_records, base)
        if dry_run:
            # The first stage must be grounded in a real request manifest.
            # Later implicit inputs are outputs that this dry run deliberately
            # does not create.
            if stage.input_records is not None and not input_path.is_file():
                raise FileNotFoundError(f"Dry run requires stage input records: {input_path}")
            record_count = _read_jsonl_objects(input_path) if input_path.is_file() else 0
            reports.append(
                StageReport(
                    name=stage.name,
                    input_records=str(input_path),
                    output_records=str(output_path),
                    records=record_count,
                    status="validated",
                )
            )
            previous_output = output_path
            continue

        if not input_path.is_file():
            raise FileNotFoundError(f"Stage input records not found for {stage.name}: {input_path}")
        _read_jsonl_objects(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output_path.parent / f"{stage.name}_request.json"
        result_path = output_path.parent / f"{stage.name}_result.json"
        request_path.write_text(
            json.dumps(
                {
                    "stage": stage.name,
                    "input_records": str(input_path.resolve()),
                    "output_records": str(output_path.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Worker commands can legitimately contain JSON or Python dicts.  A
        # broad ``str.format`` would treat those braces as placeholders, so the
        # interface only substitutes its two documented tokens.
        command = [
            item.replace("{python}", python_executable)
            .replace("{request}", str(request_path))
            .replace("{result}", str(result_path))
            for item in stage.command
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{stage.name} worker failed (rc={completed.returncode}): {completed.stderr[-2000:]}")
        if not result_path.is_file():
            raise RuntimeError(f"{stage.name} worker did not write result file: {result_path}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {stage.name} result JSON: {result_path}") from exc
        if not isinstance(result, dict) or not result.get("records_path"):
            raise ValueError(f"{stage.name} result must contain `records_path`.")
        returned_path = _resolved_path(str(result["records_path"]), result_path.parent).resolve()
        if returned_path != output_path.resolve():
            raise ValueError(
                f"{stage.name} returned unexpected records_path {returned_path}; expected {output_path.resolve()}."
            )
        count = _read_jsonl_objects(output_path)
        reports.append(
            StageReport(
                name=stage.name,
                input_records=str(input_path),
                output_records=str(output_path),
                records=count,
                status="completed",
            )
        )
        previous_output = output_path
    return reports


def reports_as_dict(reports: Sequence[StageReport]) -> list[dict[str, object]]:
    return [asdict(report) for report in reports]
