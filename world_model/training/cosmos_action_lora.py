"""Launch NVIDIA's official Cosmos-Predict2.5 LoRA recipe on project LIBERO data.

The project owns the data contract, split isolation and distributed orchestration.
LoRA model/optimizer/scheduler/trainer settings are imported from NVIDIA's official
Cosmos-Predict2.5 recipe rather than reimplemented here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import subprocess
import time
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
            "The supported robot/action-conditioned checkpoint is 2B. A generic 14B video "
            "checkpoint is not an action-conditioned replacement."
        )
    return config


def _route_interface(target: str) -> str:
    """Return the interface used to reach target, if Linux can resolve it."""
    route = subprocess.run(
        ["ip", "route", "get", target], capture_output=True, text=True, check=False
    ).stdout.split()
    return route[route.index("dev") + 1] if "dev" in route else ""


def _auto_interface(*, master_addr: str, node_rank: int, remote_host: str) -> str:
    """Pick the physical inter-node NIC on both ranks.

    Routing rank 0 to its own master address resolves to ``lo``.  Use the
    worker address on rank 0 and the master address on every worker so both
    sides bind NCCL/Gloo to the inter-node network.
    """
    target = master_addr
    if node_rank == 0:
        target = remote_host.rsplit("@", 1)[-1].strip() or master_addr
    return _route_interface(target)


def _terminate_group(process: subprocess.Popen[Any], *, grace_seconds: float = 8.0) -> None:
    """Stop a launcher and all of its local torchrun descendants."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_distributed(
    local_process: subprocess.Popen[Any], remote_process: subprocess.Popen[Any]
) -> None:
    """Fail fast when either rank exits and clean up the other process group."""
    try:
        while True:
            local_code = local_process.poll()
            remote_code = remote_process.poll()
            if local_code not in (None, 0):
                _terminate_group(remote_process)
                raise subprocess.CalledProcessError(local_code, local_process.args)
            if remote_code not in (None, 0):
                _terminate_group(local_process)
                raise subprocess.CalledProcessError(remote_code, remote_process.args)
            if local_code == 0 and remote_code == 0:
                return
            time.sleep(0.5)
    except BaseException:
        _terminate_group(local_process)
        _terminate_group(remote_process)
        raise


def _collect_remote_output(
    *, remote_host: str, remote_root: str, output_setting: str, local_output: Path
) -> None:
    """Collect rank-1 DCP shards when the two nodes do not share storage."""
    configured = Path(output_setting)
    if configured.is_absolute():
        raise ValueError("Distributed output_root must be relative to each project root.")
    remote_output = Path(remote_root) / configured
    local_output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-az", "--ignore-existing",
            f"{remote_host}:{remote_output.as_posix()}/",
            f"{local_output.as_posix()}/",
        ],
        check=True,
    )


def preflight(config_path: Path, root: Path, *, nnodes: int) -> dict[str, Any]:
    config = _load(config_path)
    model = dict(config["model"])
    dataset = dict(config["dataset"])
    checkpoint = _resolve(root, str(model["checkpoint"]))
    tokenizer = _resolve(root, str(model["tokenizer"]))
    data_root = _resolve(root, str(dataset["root"]))
    cosmos_repo = _resolve(root, str(model.get("source", ".venv/src/cosmos-predict2.5")))
    errors: list[str] = []
    if not checkpoint.is_file():
        errors.append(f"action-conditioned checkpoint missing: {checkpoint}")
    elif checkpoint.suffix != ".pt":
        errors.append("checkpoint must be an explicit consolidated .pt file")
    if not tokenizer.is_file():
        errors.append(f"Wan2.1 tokenizer missing: {tokenizer}")
    if not (cosmos_repo / "scripts" / "train.py").is_file():
        errors.append(f"Cosmos source missing: {cosmos_repo}")
    counts: dict[str, int] = {}
    configured_splits = (
        str(dataset.get("train_annotation_split", "train")),
        str(dataset.get("val_annotation_split", "val")),
        str(dataset.get("test_annotation_split", "test")),
    )
    for split in configured_splits:
        directory = data_root / "annotation" / split
        annotation_files = list(directory.glob("*.json")) if directory.is_dir() else []
        counts[split] = len(annotation_files)
        if counts[split] == 0:
            errors.append(f"Cosmos {split} annotations missing: {directory}")
        missing_embeddings = [path.with_suffix(".npy") for path in annotation_files if not path.with_suffix(".npy").is_file()]
        if missing_embeddings:
            errors.append(
                f"Cosmos {split} is missing {len(missing_embeddings)} Reason1 embedding sidecars; "
                "re-export with --empty-text-embedding"
            )
    multiview = bool(dataset.get("multiview", False))
    camera_ids = list(dataset.get("camera_ids", [[0], 1] if multiview else [0]))
    if multiview:
        valid_multiview = (
            len(camera_ids) == 2
            and isinstance(camera_ids[0], list)
            and bool(camera_ids[0])
            and all(int(value) >= 0 for value in camera_ids[0])
            and not isinstance(camera_ids[1], list)
            and int(camera_ids[1]) >= 0
        )
        if not valid_multiview:
            errors.append("multiview camera_ids must be [[third-camera candidates], wrist-camera]")
    elif not camera_ids or any(int(value) < 0 for value in camera_ids):
        errors.append("dataset.camera_ids must contain non-negative camera indices")
    if nnodes not in (1, 2):
        errors.append("This launcher supports one or two one-GPU nodes.")
    report = {
        "status": "ready" if not errors else "blocked",
        "recipe": "nvidia_cosmos_official_lora_with_libero_dual_view",
        "model": "Cosmos-Predict2.5-2B robot/action-cond",
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer),
        "dataset": str(data_root),
        "annotations": counts,
        "camera_ids": camera_ids,
        "multiview": multiview,
        "packed_resolution": [
            int(dataset.get("resolution", [256, 320])[0]),
            int(dataset.get("resolution", [256, 320])[1]) * (2 if multiview else 1),
        ],
        "nnodes": nnodes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Cosmos action LoRA preflight failed:\n- " + "\n- ".join(errors))
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
    output_root_override: str | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    config = _load(config_path)
    model = dict(config["model"])
    dataset = dict(config["dataset"])
    training = dict(config["training"])
    distributed = dict(config.get("distributed", {}))
    checkpoint = _resolve(root, str(model["checkpoint"]))
    tokenizer = _resolve(root, str(model["tokenizer"]))
    data_root = _resolve(root, str(dataset["root"]))
    output_root = _resolve(root, str(output_root_override or training["output_root"]))
    cosmos_repo = _resolve(root, str(model.get("source", ".venv/src/cosmos-predict2.5")))
    master_addr = master_addr or (
        "127.0.0.1" if nnodes == 1 else str(distributed.get("master_addr", "127.0.0.1"))
    )
    master_port = int(master_port or distributed.get("master_port", 29500))
    train_split = str(dataset.get("train_annotation_split", "train"))
    val_split = str(dataset.get("val_annotation_split", "val"))
    test_split = str(dataset.get("test_annotation_split", "test"))
    paths = {
        "train_annotation_path": data_root / "annotation" / train_split,
        "val_annotation_path": data_root / "annotation" / val_split,
        "test_annotation_path": data_root / "annotation" / test_split,
        "video_path": data_root,
    }
    overrides = [
        "experiment=libero_action_lora_official",
        "~dataloader_train.dataloaders",
        "~trainer.callbacks.wandb",
        "~trainer.callbacks.wandb_10x",
        f"checkpoint.save_iter={int(save_iter or training.get('save_iter', 200))}",
        f"trainer.max_iter={int(max_iter or training.get('max_iter', 1000))}",
        f"model.config.fsdp_shard_size={nnodes}",
        "model_parallel.context_parallel_size=1",
        f"dataloader_train.batch_size={int(training.get('batch_size_per_gpu', 1))}",
        "dataloader_val.batch_size=1",
    ]
    command = [
        str(root / ".venv" / "bin" / "torchrun"),
        f"--nnodes={nnodes}",
        "--nproc_per_node=1",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--module",
        "world_model.training.cosmos_official_train_entrypoint",
        "--config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "--",
        *overrides,
    ]
    env = dict(os.environ)
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    env["COSMOS_INTERNAL"] = "0"
    env["COSMOS_WAN2PT1_VAE_PATH"] = str(tokenizer)
    env["COSMOS_ACTION_BASE_CHECKPOINT"] = str(checkpoint)
    env["COSMOS_LIBERO_DATA_ROOT"] = str(data_root)
    env["COSMOS_LIBERO_TRAIN_ANNOTATION"] = str(paths["train_annotation_path"])
    env["COSMOS_LIBERO_VAL_ANNOTATION"] = str(paths["val_annotation_path"])
    env["COSMOS_LIBERO_TEST_ANNOTATION"] = str(paths["test_annotation_path"])
    env["COSMOS_LIBERO_FPS_DOWNSAMPLE"] = str(int(dataset.get("fps_downsample_ratio", 1)))
    env["COSMOS_LIBERO_ACTION_CHUNK"] = str(int(dataset.get("num_action_per_chunk", 12)))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root), str(cosmos_repo), existing_pythonpath) if value
    )
    interface = str(distributed.get("interface", "")).strip()
    if interface == "auto":
        if nnodes == 1:
            interface = "lo"
        else:
            interface = _auto_interface(
                master_addr=master_addr,
                node_rank=node_rank,
                remote_host=str(distributed.get("remote_host", "")),
            )
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
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--save-iter", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    config_path = _resolve(root, args.config)
    if args.distributed:
        if args.node_rank != 0:
            parser.error("--distributed is only valid on node rank 0.")
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
        output_root_override=args.output_root,
    )
    if args.action == "preflight":
        print(json.dumps(report, indent=2))
        return
    if args.action == "print":
        print(" ".join(shlex.quote(value) for value in command))
        return
    print(json.dumps(report, indent=2), flush=True)
    if not args.distributed:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return

    distributed = dict(_load(config_path).get("distributed", {}))
    remote_host = str(distributed.get("remote_host", ""))
    remote_root = str(distributed.get("remote_root", ""))
    if not remote_host or not remote_root:
        parser.error("distributed.remote_host and remote_root are required.")
    relative_config = config_path.relative_to(root)
    selected_master = args.master_addr or str(distributed.get("master_addr", "172.16.0.6"))
    selected_port = str(args.master_port or distributed.get("master_port", 29500))
    remote_args = [
        "bash", "world_model/scripts/train_cosmos_action_lora.sh",
        "--config", str(relative_config), "--action", "run",
        "--nnodes", "2", "--node-rank", "1",
        "--master-addr", selected_master, "--master-port", selected_port,
    ]
    if args.max_iter:
        remote_args.extend(("--max-iter", str(args.max_iter)))
    if args.save_iter:
        remote_args.extend(("--save-iter", str(args.save_iter)))
    if args.output_root:
        remote_args.extend(("--output-root", args.output_root))
    remote_command = f"cd {shlex.quote(remote_root)} && exec {shlex.join(remote_args)}"
    remote_process = subprocess.Popen(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3", remote_host, remote_command,
        ],
        start_new_session=True,
    )
    local_process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    _wait_for_distributed(local_process, remote_process)
    output_setting = str(args.output_root or _load(config_path)["training"]["output_root"])
    _collect_remote_output(
        remote_host=remote_host,
        remote_root=remote_root,
        output_setting=output_setting,
        local_output=_resolve(root, output_setting),
    )


if __name__ == "__main__":
    main()
