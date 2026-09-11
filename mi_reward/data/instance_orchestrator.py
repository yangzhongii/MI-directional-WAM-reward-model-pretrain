"""Run the configured external stages for a generalization data run.

Every stage runs from a configured project Python. This module owns their
order and artifact hand-off while each worker owns one concrete model/runtime.
Every stage receives a request JSON and writes a result JSON pointing at its
JSONL record output. Distributed nodes may either share one filesystem or use
the rsync transport to exchange only the shards and artifacts they need.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


PIPELINE_ORDER = ("segmentation", "planner", "simulator", "reference", "predict", "transfer")
PIPELINE_STAGE_ORDER = {name: index for index, name in enumerate(PIPELINE_ORDER)}
# Accept existing configs while the generic segmentation stage replaces the
# backend-specific historical name.
PIPELINE_STAGE_ORDER["sam3"] = PIPELINE_STAGE_ORDER["segmentation"]
DEFAULT_DISTRIBUTED_STAGES = frozenset({"planner", "simulator", "predict", "transfer"})


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


@dataclass(frozen=True)
class DistributedNode:
    """A node used for embarrassingly parallel stage execution."""

    host: str
    rank: int
    repo_root: str
    gpus: int = 1
    env: tuple[tuple[str, str], ...] = ()
    python: str | None = None


def _distributed_config(
    config: dict[str, Any],
) -> tuple[bool, frozenset[str], list[DistributedNode], str, bool]:
    execution = config.get("execution")
    if not isinstance(execution, dict):
        return False, DEFAULT_DISTRIBUTED_STAGES, [], "shared", False
    raw = execution.get("distributed")
    if raw in (None, False):
        return False, DEFAULT_DISTRIBUTED_STAGES, [], "shared", False
    if not isinstance(raw, dict):
        raise ValueError("`execution.distributed` must be a mapping.")
    enabled = bool(raw.get("enabled", False))
    transport = str(raw.get("transport", "shared")).strip().lower()
    if transport not in {"shared", "rsync"}:
        raise ValueError("`execution.distributed.transport` must be `shared` or `rsync`.")
    sync_project = bool(raw.get("sync_project", transport == "rsync"))
    raw_stages = raw.get("stages", sorted(DEFAULT_DISTRIBUTED_STAGES))
    if not isinstance(raw_stages, list) or not raw_stages or not all(isinstance(item, str) for item in raw_stages):
        raise ValueError("`execution.distributed.stages` must be a non-empty list of stage names.")
    stages = frozenset(raw_stages)
    invalid = stages.difference(DEFAULT_DISTRIBUTED_STAGES)
    if invalid:
        raise ValueError(
            f"Only these stages support sharding: {sorted(DEFAULT_DISTRIBUTED_STAGES)}; got {sorted(invalid)}"
        )
    if not enabled:
        return False, stages, [], transport, sync_project
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        raise ValueError("Enabled distributed execution requires at least two nodes.")
    nodes: list[DistributedNode] = []
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise ValueError("Every distributed node must be a mapping.")
        host = str(item.get("host", "")).strip()
        repo_root = str(item.get("repo_root", "")).strip()
        rank = int(item.get("rank", index))
        gpus = int(item.get("gpus", 1))
        python = item.get("python")
        if python not in (None, "") and not isinstance(python, str):
            raise ValueError("Distributed node `python` must be a path string when specified.")
        if not host or not repo_root:
            raise ValueError("Every distributed node requires `host` and `repo_root`.")
        if rank < 0 or gpus < 1:
            raise ValueError("Distributed node rank must be non-negative and gpus must be positive.")
        raw_env = item.get("env", {})
        if not isinstance(raw_env, dict) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float)) for key, value in raw_env.items()
        ):
            raise ValueError("Distributed node `env` must be a string-keyed mapping.")
        nodes.append(
            DistributedNode(
                host,
                rank,
                repo_root,
                gpus,
                tuple((str(k), str(v)) for k, v in raw_env.items()),
                None if python in (None, "") else str(python),
            )
        )
    ranks = sorted(node.rank for node in nodes)
    if ranks != list(range(len(nodes))):
        raise ValueError(f"Distributed node ranks must be consecutive starting at 0; got {ranks}.")
    return enabled, stages, nodes, transport, sync_project


def _shard_lines(input_path: Path, shard_dir: Path, count: int) -> list[Path]:
    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines(keepends=True) if line.strip()]
    if not lines:
        raise ValueError(f"Cannot shard empty stage input: {input_path}")
    if len(lines) < count:
        raise ValueError(f"Cannot distribute {len(lines)} records across {count} nodes.")
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for rank in range(count):
        start = len(lines) * rank // count
        end = len(lines) * (rank + 1) // count
        path = shard_dir / f"input.rank{rank}.jsonl"
        path.write_text("".join(lines[start:end]), encoding="utf-8")
        paths.append(path)
    return paths


def _is_local_node(node: DistributedNode, local_host: str) -> bool:
    return node.host in {"local", "localhost", local_host}


def _mapped_path(path: Path, source_root: Path, target_root: Path) -> Path:
    try:
        relative = path.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Distributed rsync path must be inside {source_root}: {path}") from exc
    return target_root / relative


def _rewrite_rooted_paths(value: Any, source_root: Path, target_root: Path) -> Any:
    """Recursively map absolute record paths from one repository root to another."""

    if isinstance(value, dict):
        return {key: _rewrite_rooted_paths(item, source_root, target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_rooted_paths(item, source_root, target_root) for item in value]
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return value
    return str(target_root / relative)


def _rewrite_jsonl_roots(source: Path, output: Path, source_root: Path, target_root: Path) -> None:
    rows: list[str] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid distributed JSONL at {source}:{line_no}") from exc
        rows.append(json.dumps(_rewrite_rooted_paths(payload, source_root, target_root)) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(rows), encoding="utf-8")


def _rewrite_json_roots(path: Path, source_root: Path, target_root: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(_rewrite_rooted_paths(payload, source_root, target_root), indent=2),
        encoding="utf-8",
    )


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _walk_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _walk_strings(nested)]
    return [value] if isinstance(value, str) else []


def _referenced_project_files(manifest: Path, project_root: Path) -> list[str]:
    """Return project-relative files referenced by a JSONL shard.

    Only paths rooted in the current project are eligible. This prevents a
    malformed record from making rsync read arbitrary host files.
    """

    root = project_root.resolve()
    files: set[str] = set()
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid distributed input JSONL at {manifest}:{line_no}") from exc
        for raw in _walk_strings(payload):
            path = Path(raw)
            if not path.is_absolute():
                continue
            try:
                relative = path.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            if path.is_file() or path.is_symlink():
                files.add(str(relative))
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() or child.is_symlink():
                        try:
                            files.add(str(child.resolve().relative_to(root)))
                        except (OSError, ValueError):
                            continue
    return sorted(files)


def _run_transport(command: list[str], *, description: str) -> None:
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"{description} failed (rc={completed.returncode}).")


def _sync_project_to_node(base: Path, node: DistributedNode) -> None:
    print(f"[distributed] Syncing project source to {node.host}:{node.repo_root}", flush=True)
    command = [
        "rsync", "-az", "--human-readable", "--info=progress2",
        "--exclude=.git/", "--exclude=.venv/", "--exclude=.venv-reward/",
        "--exclude=.venv-eval/", "--exclude=.venv-cosmos/", "--exclude=.venv-dist/",
        "--exclude=logs/", "--exclude=__pycache__/", "--exclude=.pytest_cache/",
        "--exclude=.cache/", "--exclude=*.pyc", "--exclude=.hf_token",
        "--exclude=.wandb_api_key", "--exclude=.*_token", "--exclude=.*api_key",
        f"{base.resolve()}/", f"{node.host}:{node.repo_root}/",
    ]
    _run_transport(command, description=f"Project sync to {node.host}")


def _push_rsync_stage_inputs(
    *,
    node: DistributedNode,
    base: Path,
    shard_root: Path,
    shard_input: Path,
    request: Path,
    shard_output: Path,
) -> tuple[Path, Path, Path, Path]:
    """Push a rewritten shard plus its referenced artifacts to a remote node."""

    remote_root = Path(node.repo_root)
    remote_input = _mapped_path(shard_input, base, remote_root)
    remote_request = _mapped_path(request, base, remote_root)
    remote_output = _mapped_path(shard_output, base, remote_root)
    remote_result = _mapped_path(
        shard_root / f"result.rank{node.rank}.json",
        base,
        remote_root,
    )
    transport_root = shard_root / ".transport"
    rewritten_input = transport_root / f"input.rank{node.rank}.jsonl"
    rewritten_request = transport_root / f"request.rank{node.rank}.json"
    _rewrite_jsonl_roots(shard_input, rewritten_input, base.resolve(), remote_root)
    rewritten_request.parent.mkdir(parents=True, exist_ok=True)
    rewritten_request.write_text(
        json.dumps(
            {
                "stage": json.loads(request.read_text(encoding="utf-8"))["stage"],
                "input_records": str(remote_input),
                "output_records": str(remote_output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    remote_dirs = sorted({str(remote_input.parent), str(remote_request.parent), str(remote_output.parent)})
    _run_transport(
        ["ssh", node.host, "mkdir", "-p", *remote_dirs],
        description=f"Remote stage directory creation on {node.host}",
    )

    referenced = _referenced_project_files(shard_input, base)
    if referenced:
        files_from = transport_root / f"files.rank{node.rank}.txt"
        files_from.write_text("\n".join(referenced) + "\n", encoding="utf-8")
        _run_transport(
            [
                "rsync", "-az", "--human-readable", "--info=progress2", "--relative",
                f"--files-from={files_from}", f"{base.resolve()}/", f"{node.host}:{remote_root}/",
            ],
            description=f"Input artifact sync to {node.host}",
        )
    _run_transport(
        ["rsync", "-az", str(rewritten_input), f"{node.host}:{remote_input}"],
        description=f"Input shard sync to {node.host}",
    )
    _run_transport(
        ["rsync", "-az", str(rewritten_request), f"{node.host}:{remote_request}"],
        description=f"Request sync to {node.host}",
    )
    return remote_input, remote_request, remote_output, remote_result


def _pull_rsync_stage_outputs(
    *,
    node: DistributedNode,
    base: Path,
    shard_root: Path,
    shard_output: Path,
    shard_result: Path,
) -> None:
    remote_shard_root = _mapped_path(shard_root, base, Path(node.repo_root))
    shard_root.mkdir(parents=True, exist_ok=True)
    _run_transport(
        [
            "rsync", "-az", "--human-readable", "--info=progress2",
            "--exclude=inputs/", "--exclude=request.rank*.json", "--exclude=.transport/",
            f"{node.host}:{remote_shard_root}/", f"{shard_root.resolve()}/",
        ],
        description=f"Stage output sync from {node.host}",
    )
    if not shard_output.is_file() or not shard_result.is_file():
        raise RuntimeError(f"Distributed stage output from {node.host} was not copied back completely.")
    remote_root = Path(node.repo_root)
    _rewrite_jsonl_roots(shard_output, shard_output, remote_root, base.resolve())
    _rewrite_json_roots(shard_result, remote_root, base.resolve())


def _render_stage_command(
    stage: ExternalStage,
    substitutions: dict[str, str],
    *,
    distributed: bool = False,
    node: DistributedNode | None = None,
) -> list[str]:
    command: list[str] = []
    for item in stage.command:
        rendered = item
        for key, value in substitutions.items():
            rendered = rendered.replace("{" + key + "}", value)
        command.append(rendered)
    if distributed and stage.name == "transfer" and node is not None and "--num-gpus" in command:
        gpu_index = command.index("--num-gpus") + 1
        if gpu_index >= len(command):
            raise ValueError("Distributed transfer command has `--num-gpus` without a value.")
        command[gpu_index] = str(node.gpus)
    return command


def _run_on_node(command: list[str], node: DistributedNode, *, local_host: str) -> None:
    env = {key: value for key, value in node.env}
    env.update({"NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "enp131s0")})
    env.update({"GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "enp131s0")})
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    remote_command = f"cd {shlex.quote(node.repo_root)} && {env_prefix} {shlex.join(command)}"
    if _is_local_node(node, local_host):
        completed = subprocess.run(command, cwd=node.repo_root, env={**os.environ, **env}, check=False, text=True)
    else:
        # OpenSSH sends the final argument to the remote login shell. Passing
        # ``bash -lc`` as separate argv items loses the quoting around the
        # compound ``cd && env command`` string and can execute from $HOME.
        completed = subprocess.run(["ssh", node.host, remote_command], check=False, text=True)
    if completed.returncode:
        raise RuntimeError(f"Distributed stage failed on {node.host} (rc={completed.returncode}).")


def _run_distributed_stage(
    stage: ExternalStage,
    input_path: Path,
    output_path: Path,
    *,
    base: Path,
    substitutions: dict[str, str],
    nodes: list[DistributedNode],
    transport: str = "shared",
) -> int:
    """Run a stage independently on contiguous record shards and merge JSONL."""

    shard_root = output_path.parent / ".distributed" / stage.name
    input_shards = _shard_lines(input_path, shard_root / "inputs", len(nodes))
    output_shards = [shard_root / f"output.rank{node.rank}.jsonl" for node in nodes]
    result_shards = [shard_root / f"result.rank{node.rank}.json" for node in nodes]
    def run_one(node: DistributedNode, shard_input: Path, shard_output: Path, shard_result: Path) -> None:
        request = shard_root / f"request.rank{node.rank}.json"
        request.write_text(
            json.dumps(
                {
                    "stage": stage.name,
                    "input_records": str(shard_input.resolve()),
                    "output_records": str(shard_output.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        local_substitutions = dict(substitutions)
        local_host = socket.gethostname()
        request_for_node = request.resolve()
        result_for_node = shard_result.resolve()
        if node.python:
            local_substitutions["python"] = node.python
        elif transport == "rsync" and not _is_local_node(node, local_host):
            local_substitutions["python"] = str(Path(node.repo_root) / ".venv/bin/python")

        if transport == "rsync" and not _is_local_node(node, local_host):
            _, remote_request, _, remote_result = _push_rsync_stage_inputs(
                node=node,
                base=base,
                shard_root=shard_root,
                shard_input=shard_input,
                request=request,
                shard_output=shard_output,
            )
            request_for_node = remote_request
            result_for_node = remote_result

        local_substitutions.update({"request": str(request_for_node), "result": str(result_for_node)})
        command = _render_stage_command(stage, local_substitutions, distributed=True, node=node)
        _run_on_node(command, node, local_host=local_host)
        if transport == "rsync" and not _is_local_node(node, local_host):
            _pull_rsync_stage_outputs(
                node=node,
                base=base,
                shard_root=shard_root,
                shard_output=shard_output,
                shard_result=shard_result,
            )
        if not shard_result.is_file():
            raise RuntimeError(f"Distributed stage did not produce result: {shard_result}")
        result = json.loads(shard_result.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or Path(str(result.get("records_path", ""))).resolve() != shard_output.resolve():
            raise ValueError(f"Distributed {stage.name} rank {node.rank} returned an unexpected records_path.")

    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = [
            pool.submit(run_one, node, shard_input, shard_output, shard_result)
            for node, shard_input, shard_output, shard_result in zip(nodes, input_shards, output_shards, result_shards)
        ]
        for future in futures:
            future.result()
    merged: list[str] = []
    for shard_output in output_shards:
        if not shard_output.is_file():
            raise RuntimeError(f"Distributed stage did not produce output: {shard_output}")
        merged.extend(
            line for line in shard_output.read_text(encoding="utf-8").splitlines(keepends=True) if line.strip()
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(merged), encoding="utf-8")
    return _read_jsonl_objects(output_path)


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
    if not all(isinstance(item, dict) for item in raw_stages):
        raise ValueError("Every `execution.stages` entry must be a mapping.")
    stages = [ExternalStage.from_dict(item) for item in raw_stages if bool(item.get("enabled", True))]
    names = [stage.name for stage in stages]
    if any(name not in PIPELINE_STAGE_ORDER for name in names):
        invalid = sorted(set(name for name in names if name not in PIPELINE_STAGE_ORDER))
        raise ValueError(f"Unsupported external stage name(s): {invalid}; expected a subset of {PIPELINE_ORDER}.")
    canonical_names = ["segmentation" if name == "sam3" else name for name in names]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("External stage names must be unique.")
    order = [PIPELINE_STAGE_ORDER[name] for name in names]
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


def run_configured_stages(
    config: dict[str, Any],
    *,
    root: str | Path = ".",
    dry_run: bool = False,
    allow_missing_first_input: bool = False,
) -> list[StageReport]:
    """Validate or execute the configured segmentation/Transfer/simulation pipeline.

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
    distributed_enabled, distributed_stages, distributed_nodes, distributed_transport, sync_project = (
        _distributed_config(config)
    )
    if distributed_enabled:
        if not distributed_nodes:
            raise ValueError("Distributed execution is enabled but no nodes were configured.")
        if not any(node.host in {"local", "localhost", socket.gethostname()} for node in distributed_nodes):
            raise ValueError("Distributed execution requires one node marked `local` or matching the local hostname.")
        if distributed_transport == "rsync" and sync_project and not dry_run:
            for node in distributed_nodes:
                if not _is_local_node(node, socket.gethostname()):
                    _sync_project_to_node(base, node)
    generation = config.get("generation", {})
    substitutions = {
        "python": python_executable,
    }
    if isinstance(generation, dict):
        substitutions.update({str(key): str(value) for key, value in generation.items()})
    for stage_index, stage in enumerate(stages, start=1):
        input_path = _resolved_path(stage.input_records, base) if stage.input_records else previous_output
        if input_path is None:
            raise ValueError(f"First configured stage {stage.name!r} requires `input_records`.")
        output_path = _resolved_path(stage.output_records, base)
        if dry_run:
            # The first stage must be grounded in a real request manifest.
            # Later implicit inputs are outputs that this dry run deliberately
            # does not create.
            if (
                stage.input_records is not None
                and not input_path.is_file()
                and not (allow_missing_first_input and not reports)
            ):
                raise FileNotFoundError(f"Dry run requires stage input records: {input_path}")
            # Simulator-native/Robometer dry runs validate configuration but do
            # not rewrite the base manifest. Ignore stale outputs from an older
            # run instead of reporting their counts as the new plan.
            record_count = (
                0
                if allow_missing_first_input
                else (_read_jsonl_objects(input_path) if input_path.is_file() else 0)
            )
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
        input_count = _read_jsonl_objects(input_path)
        print(
            f"[pipeline] Stage {stage_index}/{len(stages)}: {stage.name} "
            f"({input_count} input records)",
            flush=True,
        )
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
        if distributed_enabled and stage.name in distributed_stages:
            count = _run_distributed_stage(
                stage,
                input_path,
                output_path,
                base=base,
                substitutions=substitutions,
                nodes=distributed_nodes,
                transport=distributed_transport,
            )
            reports.append(
                StageReport(
                    name=stage.name,
                    input_records=str(input_path),
                    output_records=str(output_path),
                    records=count,
                    status="completed-distributed",
                )
            )
            print(
                f"[pipeline] Stage {stage.name} completed on {len(distributed_nodes)} nodes: "
                f"{count} output records",
                flush=True,
            )
            previous_output = output_path
            continue
        # Worker commands can legitimately contain JSON or Python dicts. A
        # broad ``str.format`` would treat those braces as placeholders, so we
        # substitute only documented request/result and generation keys.
        substitutions.update({"request": str(request_path), "result": str(result_path)})
        command = _render_stage_command(stage, substitutions)
        # Let child workers inherit the terminal. Their tqdm bars and model
        # initialization messages must remain visible during long stages.
        completed = subprocess.run(command, check=False, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{stage.name} worker failed (rc={completed.returncode}); see worker output above.")
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
        print(f"[pipeline] Stage {stage.name} completed: {count} output records", flush=True)
        previous_output = output_path
    return reports


def reports_as_dict(reports: Sequence[StageReport]) -> list[dict[str, object]]:
    return [asdict(report) for report in reports]
