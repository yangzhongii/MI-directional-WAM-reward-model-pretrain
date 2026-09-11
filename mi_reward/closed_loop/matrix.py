"""Run LIBERO task/reward/seed matrices locally or across two machines."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from mi_reward.closed_loop.libero_sac import summarize
from mi_reward.closed_loop.online_potential import PotentialReward


def _load(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _jobs(config: dict[str, Any], tasks: list[int], modes: list[str], seeds: list[int]) -> list[tuple[int, str, int]]:
    configured_tasks = [int(x) for x in config["benchmark"]["task_ids"]]
    configured_modes = [str(x) for x in config["reward"]["modes"]]
    configured_seeds = [int(x) for x in config["training"]["seeds"]]
    selected_tasks = configured_tasks if not tasks else tasks
    selected_modes = configured_modes if not modes else modes
    selected_seeds = configured_seeds if not seeds else seeds
    unknown_modes = sorted(set(selected_modes) - PotentialReward.MODES)
    if unknown_modes:
        raise ValueError(f"Unknown reward modes: {unknown_modes}")
    return [(task, mode, seed) for task in selected_tasks for mode in selected_modes for seed in selected_seeds]


def _run_command(config_path: str, job: tuple[int, str, int], action: str = "train", resume: bool = False) -> list[str]:
    task, mode, seed = job
    command = [
        "bash", "mi_reward/scripts/run_libero_closed_loop.sh", "--config", config_path,
        "--action", action, "--task-id", str(task), "--reward-mode", mode, "--seed", str(seed),
    ]
    if resume:
        command.append("--resume")
    return command


def _run_dir(config: dict[str, Any], job: tuple[int, str, int], root: Path) -> Path:
    task, mode, seed = job
    output = Path(config.get("output_root", "logs/mi_reward/libero_closed_loop"))
    if not output.is_absolute():
        output = root / output
    return output / str(config.get("run_id", "libero_mi_sac_v1")) / f"task-{task:02d}" / mode / f"seed-{seed}"


def _remote(host: str, remote_root: str, command: list[str]) -> None:
    rendered = f"cd {shlex.quote(remote_root)} && {shlex.join(command)}"
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, rendered], check=True)


def _sync_source(host: str, remote_root: str, root: Path) -> None:
    subprocess.run(
        [
            # Source sync is deliberately non-destructive. In particular,
            # ``--delete-excluded`` erases .venv*, model data, and logs, while
            # even plain ``--delete`` can erase source edits that exist only
            # on the worker. Stale files are harmless for these module paths.
            "rsync", "-az", "--human-readable",
            "--exclude=.git/", "--exclude=.venv*/", "--exclude=logs/",
            "--exclude=__pycache__/", "--exclude=.pytest_cache/", "--exclude=*.pyc",
            f"{root}/", f"{host}:{remote_root}/",
        ],
        check=True,
    )


def _sync_reward_checkpoint(config: dict[str, Any], host: str, remote_root: str, root: Path) -> None:
    checkpoint = Path(config["reward"]["checkpoint"])
    local = checkpoint if checkpoint.is_absolute() else root / checkpoint
    if not local.is_file():
        raise FileNotFoundError(f"Cannot sync missing reward checkpoint: {local}")
    if checkpoint.is_absolute():
        try:
            relative = local.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("Distributed reward checkpoint must be inside the repository.") from exc
    else:
        relative = checkpoint
    _remote(host, remote_root, ["mkdir", "-p", str(relative.parent)])
    subprocess.run(
        ["rsync", "-az", "--human-readable", str(local), f"{host}:{Path(remote_root) / relative}"],
        check=True,
    )


def _collect(host: str, remote_root: str, relative: Path, root: Path) -> None:
    destination = root / relative
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--human-readable", f"{host}:{Path(remote_root) / relative}/", f"{destination}/"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mi_reward/configs/libero_closed_loop.yaml")
    parser.add_argument("--action", choices=("preflight", "run", "summarize"), default="run")
    parser.add_argument("--task-id", type=int, action="append", default=[])
    parser.add_argument("--reward-mode", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-sync-remote", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    config_path = str(Path(args.config))
    config = _load(config_path)
    if args.action == "summarize":
        print(summarize(config_path))
        return
    jobs = _jobs(config, args.task_id, args.reward_mode, args.seed)
    pending = []
    for job in jobs:
        report = _run_dir(config, job, root) / "run_report.json"
        if args.skip_completed and report.is_file():
            print(f"[libero] skip completed: {report}", flush=True)
        else:
            pending.append(job)
    if args.action == "preflight":
        # One full preflight is sufficient for shared dependencies; task-specific
        # goal/demo checks still run for every selected task.
        for job in pending:
            subprocess.run(_run_command(config_path, job, action="preflight"), check=True)
        return
    if not args.distributed:
        for index, job in enumerate(pending, start=1):
            print(f"[libero] {index}/{len(pending)} local job={job}", flush=True)
            subprocess.run(_run_command(config_path, job, resume=args.resume), check=True)
        return

    distributed = dict(config["distributed"])
    host = str(distributed["host"])
    remote_root = str(distributed["remote_root"])
    if bool(distributed.get("sync_remote", True)) and not args.no_sync_remote:
        _sync_source(host, remote_root, root)
        _sync_reward_checkpoint(config, host, remote_root, root)
    try:
        _remote(host, remote_root, ["test", "-x", ".venv-libero/bin/python"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Remote LIBERO runtime is missing on {host}:{remote_root}. "
            "Install it there with: bash requirements/install.sh --libero-closed-loop"
        ) from exc
    local_jobs, remote_jobs = pending[::2], pending[1::2]
    print(f"[libero] experiment sharding local={len(local_jobs)} remote={len(remote_jobs)}", flush=True)

    def local_queue() -> None:
        for job in local_jobs:
            subprocess.run(_run_command(config_path, job, resume=args.resume), check=True)

    def remote_queue() -> None:
        for job in remote_jobs:
            _remote(host, remote_root, _run_command(config_path, job, resume=args.resume))
            run_dir = _run_dir(config, job, root)
            relative = run_dir.relative_to(root)
            _collect(host, remote_root, relative, root)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="libero-worker") as executor:
        futures = [executor.submit(local_queue), executor.submit(remote_queue)]
        for future in futures:
            future.result()
    print(json.dumps({"status": "complete", "local_jobs": local_jobs, "remote_jobs": remote_jobs}))


if __name__ == "__main__":
    main()
