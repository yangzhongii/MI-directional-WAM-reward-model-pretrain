"""Configuration-driven generalization data preparation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mi_reward.data.instance_orchestrator import reports_as_dict, run_configured_stages
from mi_reward.data.base_trajectory_schema import BaseTrajectory
from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for generalization pipeline configuration.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required configuration key: {key}")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, str]:
    paths = config.get("paths")
    verification = config.get("verification")
    if not isinstance(paths, dict) or not isinstance(verification, dict):
        raise ValueError("Configuration requires `paths` and `verification` mappings.")
    primary_task_family = str(_required(config, "task_family"))
    raw_task_families = config.get("task_families", [primary_task_family])
    if (
        not isinstance(raw_task_families, list)
        or not raw_task_families
        or not all(isinstance(item, str) and item for item in raw_task_families)
    ):
        raise ValueError("`task_families` must be a non-empty list of task-family strings.")
    values = {
        "candidate_records": str(_required(paths, "candidate_records")),
        "manifest": str(_required(paths, "manifest")),
        "feasibility_config": str(_required(paths, "feasibility_config")),
        "success_refs": str(_required(paths, "success_refs")),
        "task_family": primary_task_family,
        "task_families": json.dumps(raw_task_families),
        "max_pose_step": str(verification.get("max_pose_step", 0.25)),
        "require_controls": str(bool(verification.get("require_controls", True))),
    }
    values["run_report"] = str(paths.get("run_report") or Path(values["manifest"]).with_suffix(".run_report.json"))
    return values


def _prepare_base_data(config: dict[str, Any], *, dry_run: bool) -> dict[str, object]:
    base_data = config.get("base_data")
    if not isinstance(base_data, dict):
        raise ValueError("Configuration requires a base_data mapping.")
    source = str(_required(base_data, "source"))
    output_records = Path(str(_required(base_data, "output_records"))).resolve()
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration requires paths.")
    success_refs = Path(str(_required(paths, "success_refs"))).resolve()
    if source == "robometer":
        robometer = base_data.get("robometer")
        if not isinstance(robometer, dict):
            raise ValueError("base_data.robometer must be a mapping.")
        if dry_run:
            datasets = robometer.get("datasets")
            if not isinstance(datasets, list) or not datasets:
                raise ValueError("base_data.robometer.datasets must be a non-empty list.")
            return {
                "source": source,
                "status": "validated",
                "output_records": str(output_records),
                "success_refs": str(success_refs),
                "datasets": [str(value) for value in datasets],
            }
        from mi_reward.data.robometer_ingest import ingest_robometer

        return ingest_robometer(robometer, output_records, success_refs)
    if source == "jsonl":
        input_records = Path(str(_required(base_data, "input_records"))).resolve()
        if not input_records.is_file():
            raise FileNotFoundError(f"Base trajectory JSONL is missing: {input_records}")
        _validate_base_records(input_records)
        if input_records != output_records:
            output_records.parent.mkdir(parents=True, exist_ok=True)
            output_records.write_text(input_records.read_text(encoding="utf-8"), encoding="utf-8")
        if not success_refs.is_file():
            raise FileNotFoundError(f"Success-reference JSONL is missing: {success_refs}")
        return {
            "source": source,
            "status": "ready",
            "input_records": str(input_records),
            "output_records": str(output_records),
            "success_refs": str(success_refs),
        }
    raise ValueError("base_data.source must be `robometer` or `jsonl`.")


def _validate_base_records(path: Path) -> int:
    """Validate the external RLinf bridge before any model worker is launched."""

    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid base-record JSONL at {path}:{line_no}.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Base record at {path}:{line_no} must be a JSON object.")
        try:
            BaseTrajectory.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid BaseTrajectory at {path}:{line_no}: {exc}") from exc
        count += 1
    if count == 0:
        raise ValueError(f"Base trajectory JSONL is empty: {path}")
    return count


def validate_training_manifest(path: str | Path, success_refs: str | Path | None = None) -> dict[str, object]:
    """Refuse reward training if accepted generalization data is incomplete."""

    manifest = Path(path)
    references = None
    if success_refs is not None:
        references = {(item.task, item.ref_id) for item in read_jsonl(success_refs, SuccessReference)}
    accepted = 0
    failures: list[str] = []
    for item in read_jsonl(manifest, TrajectoryExample):
        if not (item.verification and item.verification.accepted):
            continue
        accepted += 1
        missing: list[str] = []
        if not item.goal_ref_id:
            missing.append("goal_ref_id")
        elif references is not None and (item.task, item.goal_ref_id) not in references:
            missing.append("declared_goal_reference")
        for name, value in (
            ("action_path", item.action_path),
            ("robot_state_path", item.robot_state_path),
            ("object_state_path", item.object_state_path),
            ("relation_path", item.relation_path),
        ):
            if not value or not Path(value).is_file():
                missing.append(name)
        if not item.frames or not all(Path(frame).is_file() for frame in item.frames):
            missing.append("frames")
        if item.control_artifacts is None:
            missing.append("control_artifacts")
        else:
            if not item.control_artifacts.mask_root or not Path(item.control_artifacts.mask_root).is_dir():
                missing.append("mask_root")
            if not item.control_artifacts.depth_root or not Path(item.control_artifacts.depth_root).is_dir():
                missing.append("depth_root")
        if missing:
            failures.append(f"{item.traj_id}: {', '.join(missing)}")
    if failures:
        raise ValueError("Refusing generalization reward training; accepted candidates are incomplete: " + "; ".join(failures))
    if accepted == 0:
        raise ValueError("Refusing generalization reward training; the manifest has no accepted candidates.")
    return {
        "manifest": str(manifest),
        "success_refs": (None if success_refs is None else str(success_refs)),
        "accepted_candidates": accepted,
        "status": "ready",
    }


def _write_run_report(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and ingest scene/instance generalization rollout workers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-suite", default=None, help="Require the configured task suite before executing workers.")
    parser.add_argument("--validate-training-manifest", default=None, metavar="MANIFEST")
    parser.add_argument(
        "--success-refs",
        default=None,
        help="Success-reference manifest used with --validate-training-manifest.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    values = validate_config(config)
    if args.task_suite is not None and args.task_suite != config.get("task_suite"):
        raise ValueError(
            f"Requested task suite {args.task_suite!r} does not match config task_suite={config.get('task_suite')!r}."
        )
    if args.validate_training_manifest is not None:
        print(json.dumps(validate_training_manifest(args.validate_training_manifest, args.success_refs), indent=2))
        return
    feasibility_path = Path(values["feasibility_config"])
    if not feasibility_path.is_file():
        raise FileNotFoundError(f"Feasibility config not found: {feasibility_path}")
    base_report = _prepare_base_data(config, dry_run=args.dry_run)
    base_output = Path(str(base_report["output_records"])).resolve()
    execution = config.get("execution")
    raw_stages = execution.get("stages") if isinstance(execution, dict) else None
    if isinstance(raw_stages, list) and raw_stages:
        first_input = raw_stages[0].get("input_records") if isinstance(raw_stages[0], dict) else None
        if first_input and Path(str(first_input)).resolve() != base_output:
            raise ValueError("The first worker input_records must equal base_data.output_records.")
    stage_reports = run_configured_stages(
        config,
        dry_run=args.dry_run,
        allow_missing_first_input=args.dry_run and str(base_report.get("source")) == "robometer",
    )
    stages = reports_as_dict(stage_reports)
    if stage_reports and Path(stage_reports[-1].output_records).resolve() != Path(values["candidate_records"]).resolve():
        raise ValueError(
            "The final external stage output must equal paths.candidate_records; "
            f"got {stage_reports[-1].output_records} versus {values['candidate_records']}."
        )
    if args.dry_run:
        print(
            json.dumps(
                {"config": str(config_path), "validated": values, "base_data": base_report, "stages": stages},
                indent=2,
            )
        )
        return
    feasibility_payload = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if not isinstance(feasibility_payload, dict):
        raise ValueError("Feasibility config root must be a JSON object.")
    # The verifier depends on the training stack (including torch).  Keeping
    # this import at the execution boundary lets config validation and worker
    # dry-runs run in a lightweight control environment.
    from mi_reward.data.cosmos_action_cond import feasibility_config_from_dict
    from mi_reward.data.instance_rollout import ingest_instance_rollout_candidates
    from mi_reward.verification.instance_checks import InstanceVerificationConfig

    task_families = [str(item) for item in json.loads(values["task_families"])]
    instance_configs = {
        task_family: InstanceVerificationConfig(
            task_family=task_family,
            max_pose_step=float(values["max_pose_step"]),
            require_controls=values["require_controls"] == "True",
        )
        for task_family in task_families
    }

    examples = ingest_instance_rollout_candidates(
        values["candidate_records"],
        values["manifest"],
        feasibility_config_from_dict(feasibility_payload),
        instance_configs,
    )
    accepted = sum(bool(item.verification and item.verification.accepted) for item in examples)
    rejected = len(examples) - accepted
    _write_run_report(
        values["run_report"],
        {
            "config": str(config_path),
            "candidate_records": values["candidate_records"],
            "manifest": values["manifest"],
            "task_families": task_families,
            "total_candidates": len(examples),
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "stages": stages,
            "base_data": base_report,
        },
    )
    print(f"Wrote {len(examples)} candidates to {values['manifest']} ({accepted} accepted).")


if __name__ == "__main__":
    main()
