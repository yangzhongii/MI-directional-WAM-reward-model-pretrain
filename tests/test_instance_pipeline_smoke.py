"""Mock-worker smoke tests for the configured instance rollout pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mi_reward.data.instance_orchestrator import run_configured_stages
from mi_reward.data.instance_pipeline import validate_training_manifest
from mi_reward.data.schema import ControlArtifacts, TrajectoryExample, VerificationRecord, write_jsonl


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
                    "name": "sam3",
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
