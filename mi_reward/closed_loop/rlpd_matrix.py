"""Run visual LIBERO RLPD experiments locally or shard them across two hosts."""

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


SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=3",
]
RSYNC_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"


def _load(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _jobs(config: dict[str, Any], tasks: list[int], modes: list[str], seeds: list[int]) -> list[tuple[int, str, int]]:
    selected_tasks = tasks or [int(value) for value in config["benchmark"]["task_ids"]]
    selected_modes = modes or [str(value) for value in config["reward"]["modes"]]
    selected_seeds = seeds or [int(value) for value in config["training"]["seeds"]]
    unknown = sorted(set(selected_modes) - PotentialReward.MODES)
    if unknown:
        raise ValueError(f"Unknown reward modes: {unknown}")
    return [(task, mode, seed) for task in selected_tasks for mode in selected_modes for seed in selected_seeds]


def _command(config_path: str, job: tuple[int, str, int], action: str) -> list[str]:
    task, mode, seed = job
    return [
        "bash", "mi_reward/scripts/run_libero_rlpd.sh", "--config", config_path,
        "--action", action, "--task-id", str(task), "--reward-mode", mode, "--seed", str(seed),
    ]


def _run_dir(config: dict[str, Any], job: tuple[int, str, int], root: Path) -> Path:
    task, mode, seed = job
    output = Path(config.get("output_root", "logs/mi_reward/libero_rlpd_closed_loop"))
    if not output.is_absolute():
        output = root / output
    return output / str(config.get("run_id", "libero_mi_rlpd_v1")) / f"task-{task:02d}" / mode / f"seed-{seed}"


def _remote(host: str, remote_root: str, command: list[str]) -> None:
    rendered = f"cd {shlex.quote(remote_root)} && {shlex.join(command)}"
    subprocess.run(["ssh", *SSH_OPTIONS, host, rendered], check=True)


def _sync_source(host: str, remote_root: str, root: Path) -> None:
    subprocess.run([
        "rsync", "-az", "--human-readable", "--timeout=60", "-e", RSYNC_SSH,
        "--exclude=.git/", "--exclude=.venv*/", "--exclude=logs/",
        "--exclude=__pycache__/", "--exclude=.pytest_cache/", "--exclude=*.pyc",
        f"{root}/", f"{host}:{remote_root}/",
    ], check=True)


def _resolve_artifact(raw: str, root: Path) -> tuple[Path, Path]:
    candidate = Path(raw).expanduser()
    local = candidate if candidate.is_absolute() else root / candidate
    if not local.is_file():
        raise FileNotFoundError(f"Distributed artifact is missing: {local}")
    try:
        relative = local.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Distributed artifact must be inside the repository: {local}") from exc
    return local, relative


def _sync_artifact(raw: str, host: str, remote_root: str, root: Path) -> None:
    local, relative = _resolve_artifact(raw, root)
    _remote(host, remote_root, ["mkdir", "-p", str(relative.parent)])
    subprocess.run([
        "rsync", "-az", "--human-readable", "--timeout=60", "-e", RSYNC_SSH,
        str(local), f"{host}:{Path(remote_root) / relative}",
    ], check=True)


def _prepare_remote(config: dict[str, Any], config_path: str, root: Path, no_sync: bool) -> tuple[str, str]:
    distributed = dict(config["distributed"])
    host = str(distributed["host"])
    remote_root = str(distributed["remote_root"])
    if bool(distributed.get("sync_remote", True)) and not no_sync:
        _sync_source(host, remote_root, root)
        _sync_artifact(str(config["reward"]["checkpoint"]), host, remote_root, root)
        _sync_artifact(str(config["training"]["rlpd"]["encoder_checkpoint"]), host, remote_root, root)
    try:
        _remote(host, remote_root, ["test", "-x", ".venv-libero/bin/python"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Remote LIBERO runtime is missing on {host}:{remote_root}; run "
            "bash requirements/install.sh --libero-closed-loop there."
        ) from exc
    return host, remote_root


def _collect(host: str, remote_root: str, run_dir: Path, root: Path) -> None:
    relative = run_dir.relative_to(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-az", "--human-readable", "--timeout=60", "-e", RSYNC_SSH,
        f"{host}:{Path(remote_root) / relative}/", f"{run_dir}/",
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mi_reward/configs/libero_rlpd_closed_loop.yaml")
    parser.add_argument("--action", choices=("preflight", "run", "summarize"), default="run")
    parser.add_argument("--task-id", type=int, action="append", default=[])
    parser.add_argument("--reward-mode", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--remote-only", action="store_true",
        help="Run every selected job on the configured remote worker and collect its results.",
    )
    parser.add_argument("--no-sync-remote", action="store_true")
    args = parser.parse_args()
    if args.remote_only and not args.distributed:
        parser.error("--remote-only requires --distributed")
    root = Path.cwd().resolve()
    config_path = str(Path(args.config))
    config = _load(config_path)
    if args.action == "summarize":
        print(summarize(config_path))
        return
    pending = []
    for job in _jobs(config, args.task_id, args.reward_mode, args.seed):
        report = _run_dir(config, job, root) / "run_report.json"
        if args.skip_completed and report.is_file():
            print(f"[rlpd] skip completed: {report}", flush=True)
        else:
            pending.append(job)
    if not args.distributed:
        for index, job in enumerate(pending, start=1):
            print(f"[rlpd] {index}/{len(pending)} local job={job}", flush=True)
            subprocess.run(_command(config_path, job, "preflight" if args.action == "preflight" else "train"), check=True)
        return
    host, remote_root = _prepare_remote(config, config_path, root, args.no_sync_remote)
    if args.remote_only:
        local_jobs, remote_jobs = [], pending
    else:
        local_jobs, remote_jobs = pending[::2], pending[1::2]
    print(f"[rlpd] experiment sharding local={len(local_jobs)} remote={len(remote_jobs)}", flush=True)
    action = "preflight" if args.action == "preflight" else "train"

    def local_queue() -> None:
        for job in local_jobs:
            subprocess.run(_command(config_path, job, action), check=True)

    def remote_queue() -> None:
        for job in remote_jobs:
            _remote(host, remote_root, _command(config_path, job, action))
            if action == "train":
                _collect(host, remote_root, _run_dir(config, job, root), root)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rlpd-worker") as executor:
        futures = [executor.submit(local_queue), executor.submit(remote_queue)]
        for future in futures:
            future.result()
    print(json.dumps({"status": "complete", "local_jobs": local_jobs, "remote_jobs": remote_jobs}))


if __name__ == "__main__":
    main()
