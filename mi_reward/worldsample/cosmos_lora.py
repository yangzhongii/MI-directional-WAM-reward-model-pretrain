"""Preflight and launch Cosmos Action-Conditioned 2B LoRA adaptation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    if str(config.get("model", {}).get("size", "")) != "2B":
        raise ValueError(
            "This launcher intentionally supports the public 2B action-conditioned checkpoint only; "
            "a public 14B action-conditioned training checkpoint/recipe is not available."
        )
    return config


def preflight(config_path: Path, root: Path, *, nnodes: int) -> dict[str, Any]:
    config = _load(config_path)
    model = dict(config["model"])
    dataset = dict(config["dataset"])
    checkpoint = _resolve(root, str(model["checkpoint"]))
    data_root = _resolve(root, str(dataset["root"]))
    cosmos_repo = root / ".venv" / "src" / "cosmos-predict2.5"
    errors: list[str] = []
    if not checkpoint.is_file():
        errors.append(f"action-conditioned checkpoint missing: {checkpoint}")
    elif checkpoint.suffix != ".pt":
        errors.append("checkpoint must be an explicit consolidated .pt file")
    if not (cosmos_repo / "scripts" / "train.py").is_file():
        errors.append(f"Cosmos source missing: {cosmos_repo}")
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        directory = data_root / "annotation" / split
        counts[split] = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        if counts[split] == 0:
            errors.append(f"Cosmos {split} annotations missing: {directory}")
    camera_ids = list(dataset.get("camera_ids", [0]))
    if not camera_ids or any(int(value) < 0 for value in camera_ids):
        errors.append("dataset.camera_ids must contain non-negative camera indices")
    if nnodes not in (1, 2):
        errors.append("Only one or two one-GPU nodes are supported by this launcher.")
    report = {
        "status": "ready" if not errors else "blocked",
        "model": "Cosmos-Predict2.5-2B robot/action-cond + LoRA",
        "checkpoint": str(checkpoint),
        "checkpoint_gib": round(checkpoint.stat().st_size / 2**30, 2) if checkpoint.is_file() else None,
        "dataset": str(data_root),
        "annotations": counts,
        "camera_ids": camera_ids,
        "nnodes": nnodes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Cosmos LoRA preflight failed:\n- " + "\n- ".join(errors))
    return report


def training_command(
    config_path: Path,
    root: Path,
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str | None = None,
    master_port: int | None = None,
    max_iter: int | None = None,
    save_iter: int | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    config = _load(config_path)
    model = dict(config["model"])
    dataset = dict(config["dataset"])
    training = dict(config["training"])
    distributed = dict(config.get("distributed", {}))
    checkpoint = _resolve(root, str(model["checkpoint"]))
    data_root = _resolve(root, str(dataset["root"]))
    output_root = _resolve(root, str(training["output_root"]))
    cosmos_repo = root / ".venv" / "src" / "cosmos-predict2.5"
    train_ann = data_root / "annotation" / "train"
    val_ann = data_root / "annotation" / "val"
    test_ann = data_root / "annotation" / "test"
    master_addr = master_addr or (
        "127.0.0.1" if nnodes == 1 else str(distributed.get("master_addr", "127.0.0.1"))
    )
    master_port = int(master_port or distributed.get("master_port", 29500))
    fsdp_shards = nnodes
    overrides = [
        f"experiment={model['experiment']}",
        f"checkpoint.load_path={checkpoint}",
        "checkpoint.load_training_state=false",
        f"checkpoint.save_iter={int(save_iter or training.get('save_iter', 200))}",
        f"trainer.max_iter={int(max_iter or training.get('max_iter', 1000))}",
        f"trainer.validation_iter={int(training.get('validation_iter', 200))}",
        f"optimizer.lr={float(training.get('learning_rate', 1e-4))}",
        "model.config.use_lora=true",
        f"model.config.lora_rank={int(model.get('lora_rank', 32))}",
        f"model.config.lora_alpha={int(model.get('lora_alpha', 32))}",
        f"model.config.lora_target_modules='{model.get('lora_target_modules')}'",
        "model.config.init_lora_weights=true",
        f"model.config.fsdp_shard_size={fsdp_shards}",
        "model_parallel.context_parallel_size=1",
        f"dataloader_train.batch_size={int(training.get('batch_size_per_gpu', 1))}",
    ]
    paths = {
        "train_annotation_path": train_ann,
        "val_annotation_path": val_ann,
        "test_annotation_path": test_ann,
        "video_path": data_root,
    }
    for prefix in ("dataloader_train.dataset", "dataloader_train.sampler.dataset"):
        overrides.extend(f"{prefix}.{key}={value}" for key, value in paths.items())
        overrides.extend(
            (
                f"{prefix}.cam_ids={json.dumps(dataset.get('camera_ids', [0]))}",
                f"{prefix}.video_size={json.dumps(dataset.get('resolution', [256, 320]))}",
                f"{prefix}.num_action_per_chunk={int(dataset.get('num_action_per_chunk', 12))}",
                f"{prefix}.fps_downsample_ratio={int(dataset.get('fps_downsample_ratio', 1))}",
            )
        )
    command = [
        str(root / ".venv" / "bin" / "torchrun"),
        f"--nnodes={nnodes}",
        "--nproc_per_node=1",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "scripts/train.py",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        *overrides,
    ]
    env = dict(os.environ)
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    interface = str(distributed.get("interface", "")).strip()
    if interface == "auto":
        route = subprocess.run(
            ["ip", "route", "get", master_addr], capture_output=True, text=True, check=False
        ).stdout.split()
        interface = route[route.index("dev") + 1] if "dev" in route else ""
    if interface:
        env.setdefault("NCCL_SOCKET_IFNAME", interface)
        env.setdefault("GLOO_SOCKET_IFNAME", interface)
    env.setdefault("NCCL_DEBUG", "WARN")
    return command, env, cosmos_repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("preflight", "print", "run"), default="preflight")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--max-iter", type=int, help="Override max iterations; use 1 for a memory smoke test.")
    parser.add_argument("--save-iter", type=int)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="From node 0, launch the matching node-1 process over configured passwordless SSH.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    config_path = _resolve(root, args.config)
    if args.distributed:
        if args.node_rank != 0:
            parser.error("--distributed is an orchestrator flag and must be invoked on node rank 0.")
        args.nnodes = 2
    report = preflight(config_path, root, nnodes=args.nnodes)
    command, env, cwd = training_command(
        config_path,
        root,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        max_iter=args.max_iter,
        save_iter=args.save_iter,
    )
    if args.action == "preflight":
        if args.distributed:
            config = _load(config_path)
            remote = dict(config.get("distributed", {}))
            remote_host = str(remote.get("remote_host", ""))
            remote_root = str(remote.get("remote_root", ""))
            if not remote_host or not remote_root:
                parser.error("distributed.remote_host and remote_root are required.")
            relative_config = config_path.relative_to(root)
            remote_command = (
                f"cd {shlex.quote(remote_root)} && "
                f"bash mi_reward/scripts/train_cosmos_action_lora.sh "
                f"--config {shlex.quote(str(relative_config))} --action preflight --nnodes 2"
            )
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", remote_host, remote_command], check=True
            )
        print(json.dumps(report, indent=2))
        return
    if args.action == "print":
        print(" ".join(shlex.quote(value) for value in command))
        return
    print(json.dumps(report, indent=2), flush=True)
    if not args.distributed:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    config = _load(config_path)
    remote = dict(config.get("distributed", {}))
    remote_host = str(remote.get("remote_host", ""))
    remote_root = str(remote.get("remote_root", ""))
    if not remote_host or not remote_root:
        parser.error("distributed.remote_host and remote_root are required.")
    relative_config = config_path.relative_to(root)
    remote_args = [
        "bash", "mi_reward/scripts/train_cosmos_action_lora.sh",
        "--config", str(relative_config), "--action", "run",
        "--nnodes", "2", "--node-rank", "1",
        "--master-addr", args.master_addr or str(remote.get("master_addr", "172.16.0.6")),
        "--master-port", str(args.master_port or remote.get("master_port", 29500)),
    ]
    if args.max_iter:
        remote_args.extend(("--max-iter", str(args.max_iter)))
    if args.save_iter:
        remote_args.extend(("--save-iter", str(args.save_iter)))
    remote_command = f"cd {shlex.quote(remote_root)} && {shlex.join(remote_args)}"
    remote_process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", remote_host, remote_command]
    )
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        remote_code = remote_process.wait()
        if remote_code:
            raise subprocess.CalledProcessError(remote_code, remote_command)
    finally:
        if remote_process.poll() is None:
            remote_process.terminate()


if __name__ == "__main__":
    main()
