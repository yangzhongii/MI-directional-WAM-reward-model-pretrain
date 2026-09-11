"""Configuration-driven generalization data preparation entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from mi_reward.data.instance_orchestrator import parse_stages, reports_as_dict, run_configured_stages
from mi_reward.data.base_trajectory_schema import BaseTrajectory
from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl, write_jsonl


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
        "require_scene_variant": str(bool(verification.get("require_scene_variant", False))),
        "min_accepted_candidates": str(int(verification.get("min_accepted_candidates", 1))),
        "min_acceptance_rate": str(float(verification.get("min_acceptance_rate", 0.0))),
        "require_scene_coverage": str(bool(verification.get("require_scene_coverage", False))),
        "require_task_family_coverage": str(
            bool(verification.get("require_task_family_coverage", False))
        ),
        "require_split_coverage": str(bool(verification.get("require_split_coverage", False))),
    }
    values["run_report"] = str(paths.get("run_report") or Path(values["manifest"]).with_suffix(".run_report.json"))
    return values


def preflight_config(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    reward_config: str | Path = "mi_reward/configs/generalization_reward.yaml",
) -> dict[str, object]:
    """Validate real assets, checkpoints and hardware before launching workers."""

    base = Path(root).resolve()
    errors: list[str] = []
    checks: dict[str, object] = {}

    def resolved(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else base / path

    def require_file(label: str, value: str | Path) -> Path:
        path = resolved(value)
        if not path.is_file():
            errors.append(f"{label} is missing: {path}")
        return path

    def require_dir(label: str, value: str | Path) -> Path:
        path = resolved(value)
        if not path.is_dir():
            errors.append(f"{label} is missing: {path}")
        return path

    preparation = config.get("data_preparation", {})
    if not isinstance(preparation, dict):
        errors.append("data_preparation must be a mapping")
        preparation = {}
    python_path = require_file("pipeline Python", str(preparation.get("python", ".venv/bin/python")))
    if python_path.exists() and not python_path.stat().st_mode & 0o111:
        errors.append(f"pipeline Python is not executable: {python_path}")
    require_file("feasibility config", str(config.get("paths", {}).get("feasibility_config", "")))

    stages = parse_stages(config)
    enabled_names = [stage.name for stage in stages]
    checks["enabled_stages"] = enabled_names
    source_roots = preparation.get("source_roots", {})
    model_roots = preparation.get("model_roots", {})
    if not isinstance(source_roots, dict) or not isinstance(model_roots, dict):
        errors.append("data_preparation source_roots/model_roots must be mappings")
        source_roots, model_roots = {}, {}
    stage_root_keys = {
        "segmentation": "segmentation",
        "predict": "cosmos_predict",
        "transfer": "cosmos_transfer",
    }
    for stage_name, key in stage_root_keys.items():
        if stage_name in enabled_names:
            if key not in source_roots:
                errors.append(f"enabled stage {stage_name} has no source_roots.{key}")
            else:
                require_dir(f"{stage_name} source", str(source_roots[key]))
            if key not in model_roots:
                errors.append(f"enabled stage {stage_name} has no model_roots.{key}")
            else:
                require_dir(f"{stage_name} model root", str(model_roots[key]))

    if "segmentation" in enabled_names:
        segmentation_config = require_file("segmentation backend config", ".venv/models/segmentation.json")
        if segmentation_config.is_file():
            payload = json.loads(segmentation_config.read_text(encoding="utf-8"))
            checkpoint_root = resolved(str(payload.get("checkpoint_root", "")))
            if payload.get("backend") != "none" and not any(checkpoint_root.glob("*.pt")):
                errors.append(f"segmentation checkpoint is missing below {checkpoint_root}")
    if "predict" in enabled_names:
        predict_root = resolved(str(model_roots.get("cosmos_predict", "")))
        if not any(predict_root.glob("robot/action-cond/*_ema_bf16.pt")):
            errors.append(f"Cosmos Predict action-conditioned checkpoint is missing below {predict_root}")
    if "transfer" in enabled_names:
        transfer_stage = next(stage for stage in stages if stage.name == "transfer")
        command = list(transfer_stage.command)
        if "--checkpoint-path" in command:
            index = command.index("--checkpoint-path") + 1
            if index < len(command):
                require_file("Cosmos Transfer checkpoint", command[index])

    base_data = config.get("base_data", {})
    robometer = base_data.get("robometer", {}) if isinstance(base_data, dict) else {}
    if isinstance(base_data, dict) and base_data.get("source") == "robometer" and isinstance(robometer, dict):
        processed_root = resolved(str(robometer.get("processed_root", "")))
        for dataset_name in robometer.get("datasets", []):
            dataset_path = processed_root / str(dataset_name)
            if not dataset_path.is_dir() or not any(dataset_path.iterdir()):
                errors.append(f"configured Robometer dataset is not ready: {dataset_path}")

    task_configs: set[Path] = set()
    simulator_native = base_data.get("simulator_native", {}) if isinstance(base_data, dict) else {}
    if isinstance(base_data, dict) and base_data.get("source") == "simulator_native":
        if not isinstance(simulator_native, dict):
            errors.append("base_data.simulator_native must be a mapping")
        else:
            raw_task_configs = simulator_native.get("task_configs", [])
            if not isinstance(raw_task_configs, list) or not raw_task_configs:
                errors.append("base_data.simulator_native.task_configs must be a non-empty list")
            else:
                task_configs.update(resolved(str(value)) for value in raw_task_configs)
    if isinstance(robometer, dict):
        for rule in robometer.get("task_rules", []):
            if isinstance(rule, dict) and rule.get("physical_task_config"):
                task_configs.add(resolved(str(rule["physical_task_config"])))
    for task_path in sorted(task_configs):
        if not task_path.is_file():
            errors.append(f"physical task config is missing: {task_path}")
            continue
        task_payload = _load_yaml(task_path)
        simulation = task_payload.get("simulation", {})
        if isinstance(simulation, dict) and simulation.get("model_path"):
            require_file(f"MuJoCo scene for {task_path.name}", str(simulation["model_path"]))
        raw_scenes = task_payload.get("scene_variants", [])
        scene_ids: list[str] = []
        if not isinstance(raw_scenes, list) or not raw_scenes:
            errors.append(f"physical task config has no scene_variants: {task_path}")
        else:
            for scene in raw_scenes:
                if isinstance(scene, str):
                    scene_ids.append(scene)
                elif isinstance(scene, dict) and scene.get("variant_id"):
                    scene_ids.append(str(scene["variant_id"]))
                else:
                    errors.append(f"invalid scene variant in {task_path}: {scene!r}")
        if len(scene_ids) != len(set(scene_ids)):
            errors.append(f"duplicate scene variant ids in {task_path}")
        for instance in task_payload.get("instances", []):
            if isinstance(instance, dict) and instance.get("model_path"):
                require_file(
                    f"held-out instance model {instance.get('variant_id', '')}",
                    str(instance["model_path"]),
                )
            if not isinstance(instance, dict):
                continue
            scene_models = instance.get("scene_model_paths")
            if not isinstance(scene_models, dict):
                errors.append(
                    f"instance {instance.get('variant_id', '')} has no scene_model_paths mapping in {task_path}"
                )
                continue
            for scene_id in scene_ids:
                if scene_id not in scene_models:
                    errors.append(
                        f"instance {instance.get('variant_id', '')} has no model for scene {scene_id} in {task_path}"
                    )
                else:
                    require_file(
                        f"scene {scene_id} model for instance {instance.get('variant_id', '')}",
                        str(scene_models[scene_id]),
                    )

    reward_path = resolved(reward_config)
    if reward_path.is_file():
        reward_payload = _load_yaml(reward_path)
        reward_preparation = reward_payload.get("data_preparation", {})
        if isinstance(reward_preparation, dict):
            reward_python = require_file(
                "reward pipeline Python",
                str(reward_preparation.get("python", ".venv-reward/bin/python")),
            )
            if reward_python.exists() and not reward_python.stat().st_mode & 0o111:
                errors.append(f"reward pipeline Python is not executable: {reward_python}")
        features = reward_payload.get("features", {})
        if isinstance(features, dict):
            require_file("LaWAM LAM config", str(features.get("lam_config_path", "")))
            require_file("LaWAM LAM checkpoint", str(features.get("lam_ckpt_path", "")))
            require_dir("DINO vision checkpoint", str(features.get("dino_checkpoint_root", "")))

    gpu_stages = [name for name in enabled_names if name in {"predict", "transfer"}]
    if gpu_stages:
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            gpu_count = 0
        checks["local_cuda_devices"] = gpu_count
        distributed = config.get("execution", {}).get("distributed", {})
        distributed_enabled = isinstance(distributed, dict) and bool(distributed.get("enabled", False))
        required_gpus = 1
        if "transfer" in enabled_names and not distributed_enabled:
            required_gpus = int(config.get("generation", {}).get("transfer_gpus", 1))
        if gpu_count < required_gpus:
            errors.append(f"local CUDA devices: found {gpu_count}, require at least {required_gpus} for {gpu_stages}")

    execution = config.get("execution", {})
    distributed = execution.get("distributed", {}) if isinstance(execution, dict) else {}
    if isinstance(distributed, dict) and bool(distributed.get("enabled", False)):
        transport = str(distributed.get("transport", "shared")).strip().lower()
        if transport not in {"shared", "rsync"}:
            errors.append("distributed transport must be `shared` or `rsync`")
        if transport == "rsync" and shutil.which("rsync") is None:
            errors.append("distributed rsync transport requires local `rsync`")
        raw_nodes = distributed.get("nodes", [])
        node_status: dict[str, str] = {}
        if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
            errors.append("distributed preflight requires at least two configured nodes")
        else:
            for item in raw_nodes:
                if not isinstance(item, dict):
                    errors.append("distributed node entry must be a mapping")
                    continue
                host = str(item.get("host", ""))
                repo_root = Path(str(item.get("repo_root", "")))
                if not host or not repo_root.is_absolute():
                    errors.append(f"distributed node requires host and absolute repo_root: {item!r}")
                    continue
                configured_python = item.get("python")
                python_on_node = (
                    Path(str(configured_python))
                    if configured_python not in (None, "")
                    else repo_root / ".venv/bin/python"
                )
                pipeline_on_node = repo_root / "mi_reward/data/generalization_pipeline.py"
                if host in {"local", "localhost", socket.gethostname()}:
                    ready = (
                        python_on_node.is_file()
                        and bool(python_on_node.stat().st_mode & 0o111)
                        and pipeline_on_node.is_file()
                    )
                    detail = (
                        "ready"
                        if ready
                        else f"missing {python_on_node} or {pipeline_on_node}"
                    )
                else:
                    try:
                        completed = subprocess.run(
                            [
                                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                host, "test", "-x", str(python_on_node),
                                "-a", "-f", str(pipeline_on_node),
                                *(["-a", "-x", "/usr/bin/rsync"] if transport == "rsync" else []),
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        ready = completed.returncode == 0
                        detail = "ready" if ready else (
                            completed.stderr.strip()
                            or f"missing {python_on_node} or {pipeline_on_node}"
                        )
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        ready = False
                        detail = str(exc)
                node_status[host] = detail
                if not ready:
                    errors.append(f"distributed node {host} is not ready: {detail}")
        checks["distributed_nodes"] = node_status

    report: dict[str, object] = {"status": "ready" if not errors else "failed", "checks": checks, "errors": errors}
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return report


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
    if source == "simulator_native":
        simulator_native = base_data.get("simulator_native")
        if not isinstance(simulator_native, dict):
            raise ValueError("base_data.simulator_native must be a mapping.")
        from mi_reward.data.simulator_native import prepare_simulator_native

        return prepare_simulator_native(
            simulator_native,
            output_records,
            success_refs,
            dry_run=dry_run,
        )
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
    raise ValueError("base_data.source must be `simulator_native`, `robometer`, or `jsonl`.")


def validate_candidate_goal_references(
    candidate_records: str | Path,
    success_refs: str | Path,
) -> dict[str, object]:
    """Ensure every candidate points to a same-task reference before scoring."""

    references = {item.ref_id: item for item in read_jsonl(success_refs, SuccessReference)}
    if not references:
        raise ValueError(f"Success-reference manifest is empty: {success_refs}")
    counts: dict[str, int] = {}
    failures: list[str] = []
    candidate_path = Path(candidate_records)
    for line_no, line in enumerate(candidate_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        traj_id = str(item.get("traj_id", f"line-{line_no}"))
        ref_id = str(item.get("goal_ref_id", ""))
        task = str(item.get("task", ""))
        family = str(item.get("task_family", ""))
        reference = references.get(ref_id)
        if reference is None:
            failures.append(f"{traj_id}: unknown goal_ref_id={ref_id!r}")
        elif reference.task != task:
            failures.append(
                f"{traj_id}: candidate task {task!r} != reference task {reference.task!r}"
            )
        else:
            variant = item.get("instance_variant") or {}
            variant_id = str(variant.get("variant_id", "")) if isinstance(variant, dict) else ""
            if reference.instance_variant_id and reference.instance_variant_id != variant_id:
                failures.append(
                    f"{traj_id}: candidate instance {variant_id!r} != reference instance "
                    f"{reference.instance_variant_id!r}"
                )
        counts[family] = counts.get(family, 0) + 1
    if failures:
        raise ValueError("Candidate/reference semantic contract failed: " + "; ".join(failures[:20]))
    if not counts:
        raise ValueError(f"Candidate manifest is empty: {candidate_path}")
    return {"candidate_count": sum(counts.values()), "task_family_counts": counts}


def _validate_base_records(path: Path) -> int:
    """Validate an external simulator/real-robot JSONL before launching workers."""

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
        if item.scene_variant is None:
            missing.append("scene_variant")
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


def _write_split_manifests(manifest: str | Path, examples: list[TrajectoryExample]) -> dict[str, str]:
    root = Path(manifest).resolve().parent / "splits"
    outputs: dict[str, str] = {}
    for split in sorted({item.split for item in examples}):
        path = root / f"{split}.jsonl"
        write_jsonl(path, [item for item in examples if item.split == split])
        outputs[split] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and ingest scene/instance generalization rollout workers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Strictly check data, weights, assets and GPUs, then exit.")
    parser.add_argument(
        "--reward-config",
        default="mi_reward/configs/generalization_reward.yaml",
        help="Reward config whose LAM/DINO paths are included in --preflight.",
    )
    parser.add_argument("--task-suite", default=None, help="Require the configured task suite before executing workers.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Shard configured planner/simulator/model stages across execution.distributed.nodes.",
    )
    parser.add_argument("--validate-training-manifest", default=None, metavar="MANIFEST")
    parser.add_argument(
        "--success-refs",
        default=None,
        help="Success-reference manifest used with --validate-training-manifest.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    if args.distributed:
        execution = config.setdefault("execution", {})
        if not isinstance(execution, dict):
            raise ValueError("Configuration execution section must be a mapping.")
        distributed = execution.setdefault("distributed", {})
        if not isinstance(distributed, dict):
            raise ValueError("Configuration execution.distributed section must be a mapping.")
        distributed["enabled"] = True
    values = validate_config(config)
    if args.task_suite is not None and args.task_suite != config.get("task_suite"):
        raise ValueError(
            f"Requested task suite {args.task_suite!r} does not match config task_suite={config.get('task_suite')!r}."
        )
    if args.validate_training_manifest is not None:
        print(json.dumps(validate_training_manifest(args.validate_training_manifest, args.success_refs), indent=2))
        return
    if args.preflight:
        print(json.dumps(preflight_config(config, root=Path.cwd(), reward_config=args.reward_config), indent=2))
        return
    feasibility_path = Path(values["feasibility_config"])
    if not feasibility_path.is_file():
        raise FileNotFoundError(f"Feasibility config not found: {feasibility_path}")
    print("[pipeline] Preparing base trajectories", flush=True)
    base_report = _prepare_base_data(config, dry_run=args.dry_run)
    print(
        f"[pipeline] Base data ready: {base_report.get('records', base_report.get('status', 'ready'))}",
        flush=True,
    )
    base_output = Path(str(base_report["output_records"])).resolve()
    execution = config.get("execution")
    raw_stages = execution.get("stages") if isinstance(execution, dict) else None
    if isinstance(raw_stages, list) and raw_stages:
        enabled_stages = [item for item in raw_stages if isinstance(item, dict) and bool(item.get("enabled", True))]
        first_input = enabled_stages[0].get("input_records") if enabled_stages else None
        if first_input and Path(str(first_input)).resolve() != base_output:
            raise ValueError("The first worker input_records must equal base_data.output_records.")
    stage_reports = run_configured_stages(
        config,
        dry_run=args.dry_run,
        allow_missing_first_input=(
            args.dry_run
            and str(base_report.get("source")) in {"robometer", "simulator_native"}
        ),
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
    reference_contract = validate_candidate_goal_references(
        values["candidate_records"],
        values["success_refs"],
    )
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

    print("[pipeline] Verifying and ingesting generated candidates", flush=True)
    examples = ingest_instance_rollout_candidates(
        values["candidate_records"],
        values["manifest"],
        feasibility_config_from_dict(feasibility_payload),
        instance_configs,
        require_scene_variant=values["require_scene_variant"] == "True",
    )
    accepted = sum(bool(item.verification and item.verification.accepted) for item in examples)
    rejected = len(examples) - accepted
    split_manifests = _write_split_manifests(values["manifest"], examples)
    observed_task_families = sorted({str(item.task_family) for item in examples if item.task_family})
    task_family_counts: dict[str, dict[str, int]] = {}
    scene_counts: dict[str, dict[str, int]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    outcome_counts: dict[str, int] = {"success": 0, "failure": 0, "missing": 0}
    failure_modes: dict[str, int] = {}
    for item in examples:
        family = str(item.task_family or "missing")
        family_bucket = task_family_counts.setdefault(
            family, {"total": 0, "accepted": 0, "rejected": 0}
        )
        family_bucket["total"] += 1
        family_key = "accepted" if item.verification and item.verification.accepted else "rejected"
        family_bucket[family_key] += 1
        scene_id = item.scene_variant.variant_id if item.scene_variant else "missing"
        bucket = scene_counts.setdefault(scene_id, {"total": 0, "accepted": 0, "rejected": 0})
        bucket["total"] += 1
        key = "accepted" if item.verification and item.verification.accepted else "rejected"
        bucket[key] += 1
        split_bucket = split_counts.setdefault(item.split, {"total": 0, "accepted": 0, "rejected": 0})
        split_bucket["total"] += 1
        split_bucket[key] += 1
        if item.task_outcome is None:
            outcome_counts["missing"] += 1
        elif item.task_outcome.success:
            outcome_counts["success"] += 1
        else:
            outcome_counts["failure"] += 1
            mode = item.task_outcome.failure_mode
            failure_modes[mode] = failure_modes.get(mode, 0) + 1
    _write_run_report(
        values["run_report"],
        {
            "config": str(config_path),
            "candidate_records": values["candidate_records"],
            "manifest": values["manifest"],
            "task_families": task_families,
            "observed_task_families": observed_task_families,
            "total_candidates": len(examples),
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "acceptance_rate": accepted / max(len(examples), 1),
            "task_outcomes": outcome_counts,
            "failure_modes": failure_modes,
            "split_counts": split_counts,
            "split_manifests": split_manifests,
            "acceptance_gates": {
                "min_accepted_candidates": int(values["min_accepted_candidates"]),
                "min_acceptance_rate": float(values["min_acceptance_rate"]),
                "require_scene_coverage": values["require_scene_coverage"] == "True",
                "require_task_family_coverage": (
                    values["require_task_family_coverage"] == "True"
                ),
                "require_split_coverage": values["require_split_coverage"] == "True",
            },
            "task_family_counts": task_family_counts,
            "scene_variants": scene_counts,
            "reference_contract": reference_contract,
            "stages": stages,
            "base_data": base_report,
        },
    )
    min_accepted = int(values["min_accepted_candidates"])
    min_rate = float(values["min_acceptance_rate"])
    acceptance_rate = accepted / max(len(examples), 1)
    uncovered_scenes = sorted(
        scene_id for scene_id, counts in scene_counts.items() if counts["accepted"] == 0
    )
    uncovered_task_families = sorted(
        family
        for family in task_families
        if task_family_counts.get(family, {}).get("accepted", 0) == 0
    )
    required_splits = {"train", "instance_heldout", "scene_heldout", "joint_heldout"}
    uncovered_splits = sorted(
        split for split in required_splits if split_counts.get(split, {}).get("accepted", 0) == 0
    )
    failures: list[str] = []
    if accepted < min_accepted:
        failures.append(f"accepted candidates {accepted} < required {min_accepted}")
    if acceptance_rate < min_rate:
        failures.append(f"acceptance rate {acceptance_rate:.3f} < required {min_rate:.3f}")
    if values["require_scene_coverage"] == "True" and uncovered_scenes:
        failures.append(f"scenes without accepted candidates: {uncovered_scenes}")
    if values["require_task_family_coverage"] == "True" and uncovered_task_families:
        failures.append(
            f"task families without accepted candidates: {uncovered_task_families}"
        )
    if values["require_split_coverage"] == "True" and uncovered_splits:
        failures.append(f"splits without accepted candidates: {uncovered_splits}")
    if failures:
        raise RuntimeError(
            "Generalization data failed acceptance gates: " + "; ".join(failures)
            + f". Inspect {values['run_report']}."
        )
    print(f"Wrote {len(examples)} candidates to {values['manifest']} ({accepted} accepted).")


if __name__ == "__main__":
    main()
