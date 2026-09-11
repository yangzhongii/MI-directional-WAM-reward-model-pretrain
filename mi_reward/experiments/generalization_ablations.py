"""Generate, run, and summarize the rigid-v3 reward ablation matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "visual_only": {
        "action_weight": 0.0,
        "kinematic_weight": 0.0,
        "relation_weight": 0.0,
    },
    # Interaction ablations keep the measured outcome anchor and add exactly
    # one privileged process channel to the deployable visual teacher.  These
    # distinguish useful channel information from harmful cross-channel
    # interactions that cannot be identified by one-at-a-time removals.
    "visual_relation": {
        "action_weight": 0.0,
        "kinematic_weight": 0.0,
        "relation_weight": 1.0,
    },
    "visual_action": {
        "action_weight": 1.0,
        "kinematic_weight": 0.0,
        "relation_weight": 0.0,
    },
    "visual_kinematic": {
        "action_weight": 0.0,
        "kinematic_weight": 1.0,
        "relation_weight": 0.0,
    },
    "no_action": {"action_weight": 0.0},
    "no_kinematic": {"kinematic_weight": 0.0},
    "no_relation": {"relation_weight": 0.0},
    "no_outcome_anchor": {
        "outcome_weight": 0.0,
        "pair_mode": "score_only",
        "success_monotonic_projection": False,
        "success_endpoint_anchor": False,
        "min_success_endpoint_gain": -1.0,
    },
    "no_monotonic_projection": {
        "success_monotonic_projection": False,
    },
}


def _selected_ablations(names: list[str] | None) -> list[str]:
    selected = list(ABLATIONS) if not names else names
    unknown = sorted(set(selected) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown ablation(s) {unknown}; choose from {sorted(ABLATIONS)}.")
    return selected


def build_ablation_config(
    base: dict[str, Any],
    *,
    name: str,
    seed: int,
    output_root: str | Path,
) -> dict[str, Any]:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation {name!r}.")
    config = copy.deepcopy(base)
    run_dir = Path(output_root) / name / f"seed-{seed}"
    preference_dir = run_dir / "data"
    reward_dir = run_dir / "reward_model"
    config["run_id"] = f"{base.get('run_id', 'generalization')}_{name}_seed{seed}"
    config["ablation"] = {
        "name": name,
        "seed": seed,
        "changes": copy.deepcopy(ABLATIONS[name]),
    }
    paths = config.setdefault("paths", {})
    paths["preferences"] = str(preference_dir / "preferences.jsonl")
    paths["teacher_targets"] = str(preference_dir / "teacher_targets.pt")
    paths["output_dir"] = str(reward_dir)
    training = config.setdefault("training", {})
    training["seed"] = seed
    training.update(copy.deepcopy(ABLATIONS[name]))
    evaluation = config.setdefault("evaluation", {})
    evaluation["checkpoint"] = str(reward_dir / "pytorch_model.pt")
    evaluation["output"] = str(reward_dir / "joint_heldout_eval.json")
    evaluation["score_cache"] = str(preference_dir / "preferences.scores.jsonl")
    evaluation["strict_split_isolation"] = True
    return config


def generate_configs(
    base_config: str | Path,
    *,
    config_root: str | Path,
    output_root: str | Path,
    ablations: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Base config must contain a YAML mapping: {base_config}")
    selected = _selected_ablations(ablations)
    selected_seeds = [0, 1, 2] if not seeds else seeds
    root = Path(config_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in selected:
        for seed in selected_seeds:
            config = build_ablation_config(
                base,
                name=name,
                seed=seed,
                output_root=output_root,
            )
            path = root / f"{name}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            paths.append(path)
    return paths


def _pending_configs(configs: list[Path], *, skip_completed: bool) -> list[Path]:
    pending: list[Path] = []
    for index, config_path in enumerate(configs, start=1):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        report = Path(config["evaluation"]["output"])
        print(f"[ablation] {index}/{len(configs)}: {config_path}", flush=True)
        if skip_completed and report.is_file():
            print(f"[ablation] completed report exists, skipping: {report}", flush=True)
            continue
        pending.append(config_path)
    return pending


def _run_one_config(config_path: Path) -> None:
    subprocess.run(
        ["bash", "mi_reward/scripts/run_generalization_reward.sh", "--config", str(config_path)],
        check=True,
    )
    subprocess.run(
        ["bash", "mi_reward/scripts/eval_generalization_reward.sh", "--config", str(config_path)],
        check=True,
    )


def _run_configs(configs: list[Path], *, skip_completed: bool) -> None:
    for config_path in _pending_configs(configs, skip_completed=skip_completed):
        _run_one_config(config_path)


def _repo_relative(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate
    try:
        return candidate.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Distributed ablation path must be inside {repo_root}: {path}") from exc


def _remote_command(host: str, remote_root: str | Path, command: list[str]) -> None:
    rendered = f"cd {shlex.quote(str(remote_root))} && {shlex.join(command)}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, rendered], check=True)


def _rsync(
    source: str,
    destination: str,
    *,
    progress: bool = False,
    compress: bool = True,
) -> None:
    command = ["rsync", "-a", "--human-readable"]
    if compress:
        command.append("-z")
    if progress:
        command.extend(["--info=progress2,name0", "--no-inc-recursive"])
    command.extend([source, destination])
    subprocess.run(command, check=True)


def _prepare_remote_worker(
    *,
    base_config: str | Path,
    config_root: str | Path,
    host: str,
    remote_root: str | Path,
    sync_remote: bool,
) -> None:
    repo_root = Path.cwd().resolve()
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    paths = base["paths"]
    manifest_rel = _repo_relative(paths["manifest"], repo_root)
    refs_rel = _repo_relative(paths["success_refs"], repo_root)
    feature_rel = _repo_relative(paths["feature_root"], repo_root)
    run_root_rel = _repo_relative(base["run_root"], repo_root)
    data_rel = run_root_rel / "data"
    config_rel = _repo_relative(config_root, repo_root)

    _remote_command(host, remote_root, ["mkdir", "-p", str(data_rel), str(config_rel)])
    if sync_remote:
        print(f"[ablation] Syncing source to {host}:{remote_root}", flush=True)
        source_command = [
            "rsync", "-az", "--human-readable",
            "--exclude=.git/", "--exclude=.venv/", "--exclude=.venv-reward/",
            "--exclude=.venv-eval/", "--exclude=.venv-cosmos/", "--exclude=.venv-dist/",
            "--exclude=logs/", "--exclude=__pycache__/", "--exclude=.pytest_cache/",
            "--exclude=.cache/", "--exclude=*.pyc",
            f"{repo_root}/", f"{host}:{remote_root}/",
        ]
        subprocess.run(source_command, check=True)
        print(f"[ablation] Syncing shared rigid-v3 data to {host}", flush=True)
        _rsync(
            f"{(repo_root / data_rel).resolve()}/",
            f"{host}:{Path(remote_root) / data_rel}/",
            progress=True,
            compress=False,
        )
    _rsync(
        f"{(repo_root / config_rel).resolve()}/",
        f"{host}:{Path(remote_root) / config_rel}/",
    )
    prepare = [
        ".venv-reward/bin/python", "-m", "mi_reward.experiments.prepare_remote_ablation",
        "--manifest", str(manifest_rel),
        "--success-refs", str(refs_rel),
        "--feature-root", str(feature_rel),
        "--old-root", str(repo_root),
        "--new-root", str(remote_root),
    ]
    _remote_command(host, remote_root, prepare)
    _remote_command(
        host,
        remote_root,
        [
            ".venv-reward/bin/python", "-c",
            "import torch; assert torch.cuda.is_available(); print('[ablation] remote CUDA ready:', torch.cuda.get_device_name(0))",
        ],
    )


def _run_remote_config(
    config_path: Path,
    *,
    host: str,
    remote_root: str | Path,
) -> None:
    repo_root = Path.cwd().resolve()
    config_rel = _repo_relative(config_path, repo_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["paths"]["output_dir"])
    run_dir = output_dir.parent
    run_rel = _repo_relative(run_dir, repo_root)
    print(f"[ablation][remote] starting {config_path}", flush=True)
    _remote_command(
        host,
        remote_root,
        ["bash", "mi_reward/scripts/run_generalization_reward.sh", "--config", str(config_rel)],
    )
    _remote_command(
        host,
        remote_root,
        ["bash", "mi_reward/scripts/eval_generalization_reward.sh", "--config", str(config_rel)],
    )
    local_run = repo_root / run_rel
    local_run.mkdir(parents=True, exist_ok=True)
    _rsync(f"{host}:{Path(remote_root) / run_rel}/", f"{local_run}/", progress=True)
    print(f"[ablation][remote] completed and collected {run_rel}", flush=True)


def _run_distributed_configs(
    configs: list[Path],
    *,
    skip_completed: bool,
    base_config: str | Path,
    config_root: str | Path,
    remote_host: str,
    remote_root: str | Path,
    sync_remote: bool,
) -> None:
    pending = _pending_configs(configs, skip_completed=skip_completed)
    if not pending:
        print("[ablation] No pending runs.", flush=True)
        return
    _prepare_remote_worker(
        base_config=base_config,
        config_root=config_root,
        host=remote_host,
        remote_root=remote_root,
        sync_remote=sync_remote,
    )
    local_configs = pending[::2]
    remote_configs = pending[1::2]
    print(
        f"[ablation] Experiment sharding: local={len(local_configs)} remote={len(remote_configs)}",
        flush=True,
    )

    def run_local_queue() -> None:
        for path in local_configs:
            print(f"[ablation][local] starting {path}", flush=True)
            _run_one_config(path)

    def run_remote_queue() -> None:
        for path in remote_configs:
            _run_remote_config(
                path,
                host=remote_host,
                remote_root=remote_root,
            )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ablation-worker") as executor:
        local_future = executor.submit(run_local_queue)
        remote_future = executor.submit(run_remote_queue)
        local_future.result()
        remote_future.result()


def _nested_value(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def collect_results(output_root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path(output_root)
    for report_path in sorted(root.glob("*/seed-*/reward_model/joint_heldout_eval.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reward_dir = report_path.parent
        train_config_path = reward_dir / "train_config.yaml"
        validation_path = reward_dir / "validation_metrics.json"
        train_config = (
            yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
            if train_config_path.is_file()
            else {}
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.is_file()
            else {}
        )
        seed_dir = report_path.parents[1]
        seed = int(seed_dir.name.removeprefix("seed-"))
        row = {
            "ablation": seed_dir.parent.name,
            "seed": seed,
            "split_isolation_valid": bool(report.get("split_isolation", {}).get("valid")),
            "best_epoch": validation.get("best_epoch", train_config.get("best_epoch")),
            "validation_pair_accuracy": validation.get("best_selection_score"),
            "test_success_failure_auc": _nested_value(
                report, "trajectory_metrics", "overall", "success_failure_auc"
            ),
            "test_outcome_pair_accuracy": _nested_value(
                report, "pair_metrics", "measured_outcome_pair_accuracy", "value"
            ),
            "test_progress_pearson": _nested_value(
                report, "trajectory_metrics", "overall", "mean_progress_pearson"
            ),
            "test_progress_spearman": _nested_value(
                report, "trajectory_metrics", "overall", "mean_progress_spearman"
            ),
            "test_regression_detection": _nested_value(
                report,
                "trajectory_metrics",
                "overall",
                "regress_after_progress_detection",
                "value",
            ),
            "test_top1_success": _nested_value(
                report, "candidate_selection", "overall", "top1_success_rate", "value"
            ),
            "report": str(report_path),
        }
        rows.append(row)
    return rows


def summarize_results(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    rows = collect_results(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "ablation_runs.csv"
    fieldnames = list(rows[0]) if rows else ["ablation", "seed"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "validation_pair_accuracy",
        "test_success_failure_auc",
        "test_outcome_pair_accuracy",
        "test_progress_pearson",
        "test_progress_spearman",
        "test_regression_detection",
        "test_top1_success",
    ]
    aggregate: dict[str, Any] = {}
    for name in sorted({str(row["ablation"]) for row in rows}):
        experiment_rows = [row for row in rows if row["ablation"] == name]
        metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in experiment_rows if row.get(metric) is not None]
            metrics[metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                "values": values,
            }
        aggregate[name] = {
            "runs": len(experiment_rows),
            "all_split_isolation_valid": all(row["split_isolation_valid"] for row in experiment_rows),
            "metrics": metrics,
        }
    summary = {
        "schema_version": 1,
        "output_root": str(root),
        "run_count": len(rows),
        "ablations": aggregate,
        "runs_csv": str(csv_path),
    }
    (root / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, run, and summarize rigid-v3 reward ablations.")
    parser.add_argument("--config", default="mi_reward/configs/generalization_reward.yaml")
    parser.add_argument(
        "--config-root",
        default="logs/mi_reward/generalization_rigid_v3/results/ablations/configs",
    )
    parser.add_argument(
        "--output-root",
        default="logs/mi_reward/generalization_rigid_v3/results/ablations",
    )
    parser.add_argument(
        "--action",
        choices=["generate", "prepare-remote", "run", "summarize"],
        default="generate",
    )
    parser.add_argument("--ablation", action="append", choices=sorted(ABLATIONS), default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run independent ablation configs concurrently on this host and one SSH worker.",
    )
    parser.add_argument("--remote-host", default="zhonghaoyang@172.16.0.3")
    parser.add_argument(
        "--remote-root",
        default="/mnt/public/zhonghaoyang/MI-directional-WAM-reward-model-pretrain",
    )
    parser.add_argument(
        "--no-sync-remote",
        action="store_true",
        help="Reuse an already prepared remote source/data copy; still sync generated configs.",
    )
    args = parser.parse_args()

    if args.action == "summarize":
        print(json.dumps(summarize_results(args.output_root), ensure_ascii=False, indent=2))
        return
    configs = generate_configs(
        args.config,
        config_root=args.config_root,
        output_root=args.output_root,
        ablations=args.ablation,
        seeds=args.seed,
    )
    print(json.dumps({"generated": [str(path) for path in configs]}, ensure_ascii=False, indent=2))
    if args.action == "prepare-remote":
        _prepare_remote_worker(
            base_config=args.config,
            config_root=args.config_root,
            host=args.remote_host,
            remote_root=args.remote_root,
            sync_remote=not args.no_sync_remote,
        )
        print(json.dumps({"remote_host": args.remote_host, "status": "ready"}, indent=2))
        return
    if args.action == "run":
        if args.distributed:
            _run_distributed_configs(
                configs,
                skip_completed=args.skip_completed,
                base_config=args.config,
                config_root=args.config_root,
                remote_host=args.remote_host,
                remote_root=args.remote_root,
                sync_remote=not args.no_sync_remote,
            )
        else:
            _run_configs(configs, skip_completed=args.skip_completed)
        print(json.dumps(summarize_results(args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
