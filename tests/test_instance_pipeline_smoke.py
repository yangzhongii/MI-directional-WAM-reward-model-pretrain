"""Mock-worker smoke tests for the configured instance rollout pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mi_reward.data.instance_orchestrator import (
    DistributedNode,
    _distributed_config,
    _referenced_project_files,
    _rewrite_jsonl_roots,
    _run_on_node,
    parse_stages,
    run_configured_stages,
)
from mi_reward.data.instance_pipeline import (
    validate_candidate_goal_references,
    validate_training_manifest,
)
from mi_reward.data.schema import (
    ControlArtifacts,
    SuccessReference,
    TrajectoryExample,
    VerificationRecord,
    write_jsonl,
)


def test_configured_stage_passes_a_record_manifest_between_workers(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"request_id": "pick_place_0"}) + "\n", encoding="utf-8")
    output = tmp_path / "candidate_records.jsonl"
    # The tiny worker has the same request/result contract as a GPU worker but
    # simply forwards records, keeping this test independent of CUDA software.
    worker = (
        "import json, pathlib, sys; "
        "request=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "pathlib.Path(request['output_records']).write_text(pathlib.Path(request['input_records']).read_text()); "
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({'records_path': request['output_records']}))"
    )
    config = {
        "execution": {
            "stages": [
                {
                    "name": "segmentation",
                    "command": [sys.executable, "-c", worker, "{request}", "{result}"],
                    "input_records": str(source),
                    "output_records": str(output),
                }
            ]
        }
    }
    dry_reports = run_configured_stages(config, dry_run=True)
    assert dry_reports[0].status == "validated"
    reports = run_configured_stages(config)
    assert reports[0].status == "completed"
    assert reports[0].records == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {"request_id": "pick_place_0"}


def test_disabled_optional_stage_is_not_part_of_the_handoff() -> None:
    command = [sys.executable, "-c", "pass", "{request}", "{result}"]
    config = {
        "execution": {
            "stages": [
                {"name": "predict", "command": command, "input_records": "in.jsonl", "output_records": "predict.jsonl"},
                {
                    "name": "transfer",
                    "enabled": False,
                    "command": command,
                    "output_records": "transfer.jsonl",
                },
            ]
        }
    }
    stages = parse_stages(config)
    assert [stage.name for stage in stages] == ["predict"]
    assert stages[-1].output_records == "predict.jsonl"


def test_distributed_stage_shards_and_merges_two_workers(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(4)), encoding="utf-8")
    output = tmp_path / "merged.jsonl"
    worker = (
        "import json, pathlib, sys; "
        "r=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "pathlib.Path(r['output_records']).write_text(pathlib.Path(r['input_records']).read_text()); "
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({'records_path': r['output_records']}))"
    )
    config = {
        "execution": {
            "distributed": {
                "enabled": True,
                "stages": ["planner"],
                "nodes": [
                    {"host": "local", "rank": 0, "repo_root": str(tmp_path), "gpus": 1},
                    {"host": "localhost", "rank": 1, "repo_root": str(tmp_path), "gpus": 1},
                ],
            },
            "stages": [{
                "name": "planner",
                "command": [sys.executable, "-c", worker, "{request}", "{result}"],
                "input_records": str(source),
                "output_records": str(output),
            }],
        }
    }

    reports = run_configured_stages(config)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"index": index} for index in range(4)]
    assert reports[0].status == "completed-distributed"


def test_rsync_transport_rewrites_roots_and_collects_only_project_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact = project / "logs/run/candidate/frames/frame_000.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frame")
    action = project / "logs/run/candidate/actions.npy"
    action.write_bytes(b"actions")
    outside = tmp_path / "outside.txt"
    outside.write_text("do not copy", encoding="utf-8")
    source = project / "shard.jsonl"
    source.write_text(
        json.dumps(
            {
                "frames": [str(artifact)],
                "action_path": str(action),
                "external_path": str(outside),
                "asset_uri": "builtin://pick_place/apple",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    remote_root = Path("/mnt/public/test-project")
    rewritten = project / "rewritten.jsonl"
    _rewrite_jsonl_roots(source, rewritten, project, remote_root)
    row = json.loads(rewritten.read_text(encoding="utf-8"))

    assert row["frames"] == [str(remote_root / "logs/run/candidate/frames/frame_000.png")]
    assert row["action_path"] == str(remote_root / "logs/run/candidate/actions.npy")
    assert row["external_path"] == str(outside)
    assert row["asset_uri"] == "builtin://pick_place/apple"
    assert _referenced_project_files(source, project) == [
        "logs/run/candidate/actions.npy",
        "logs/run/candidate/frames/frame_000.png",
    ]


def test_rsync_distributed_config_accepts_per_node_python() -> None:
    config = {
        "execution": {
            "distributed": {
                "enabled": True,
                "transport": "rsync",
                "sync_project": True,
                "stages": ["planner", "simulator"],
                "nodes": [
                    {
                        "host": "local",
                        "rank": 0,
                        "repo_root": "/workspace/project",
                        "python": "/workspace/project/.venv/bin/python",
                    },
                    {
                        "host": "user@worker",
                        "rank": 1,
                        "repo_root": "/mnt/project",
                        "python": "/opt/worker/bin/python",
                    },
                ],
            }
        }
    }

    enabled, stages, nodes, transport, sync_project = _distributed_config(config)

    assert enabled is True
    assert stages == {"planner", "simulator"}
    assert transport == "rsync"
    assert sync_project is True
    assert nodes[1].python == "/opt/worker/bin/python"


def test_remote_node_executes_compound_command_from_configured_repo(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    node = DistributedNode(
        host="user@worker",
        rank=1,
        repo_root="/mnt/project",
        env=(("MUJOCO_GL", "egl"),),
    )

    _run_on_node(["/opt/python", "-m", "mi_reward.worker"], node, local_host="controller")

    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "user@worker"]
    assert len(command) == 3
    assert command[2].startswith("cd /mnt/project && ")
    assert "MUJOCO_GL=egl" in command[2]
    assert "/opt/python -m mi_reward.worker" in command[2]
    assert kwargs["check"] is False


def test_training_gate_rejects_incomplete_accepted_candidate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            TrajectoryExample(
                traj_id="instance/pick_place/bad",
                task="pick_place",
                frames=[],
                source="instance_rollout",
                split="train",
                goal_ref_id="pick_place/success",
                verification=VerificationRecord(status="accepted", checks={"all": True}, reasons=[]),
                control_artifacts=ControlArtifacts(mask_root="masks", depth_root="depth"),
            )
        ],
    )
    try:
        validate_training_manifest(manifest)
    except ValueError as exc:
        assert "action_path" in str(exc)
        assert "frames" in str(exc)
    else:
        raise AssertionError("incomplete accepted candidate must not reach reward training")


def test_training_gate_rejects_manifest_without_accepted_candidates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            TrajectoryExample(
                traj_id="instance/pick_place/rejected",
                task="pick_place",
                frames=[],
                source="instance_rollout",
                split="train",
                verification=VerificationRecord(status="rejected", checks={"all": False}, reasons=["invalid"]),
            )
        ],
    )
    try:
        validate_training_manifest(manifest)
    except ValueError as exc:
        assert "no accepted candidates" in str(exc)
    else:
        raise AssertionError("an empty accepted set must not reach reward training")


def test_candidate_reference_contract_rejects_cross_task_reference(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "traj_id": "push/candidate-0",
                "task": "push the task object",
                "task_family": "push_shape",
                "goal_ref_id": "pick/success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    references = tmp_path / "success_refs.jsonl"
    write_jsonl(
        references,
        [SuccessReference(ref_id="pick/success", task="pick up the task object", frames=[])],
    )

    try:
        validate_candidate_goal_references(candidates, references)
    except ValueError as exc:
        assert "candidate task" in str(exc)
    else:
        raise AssertionError("cross-task success references must be rejected")
