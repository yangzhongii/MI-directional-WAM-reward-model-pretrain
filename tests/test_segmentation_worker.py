"""CPU-only tests for segmentation configuration and pass-through behavior."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from mi_reward.perception.segmentation_worker import _load_backend_config, _segmentation_prompts, run_worker


def _request(tmp_path: Path, record: dict[str, object]) -> tuple[Path, Path, Path]:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    request_path.write_text(
        json.dumps({"input_records": str(input_path), "output_records": str(output_path)}),
        encoding="utf-8",
    )
    return request_path, result_path, output_path


def test_loads_installer_backend_config(tmp_path: Path) -> None:
    config = tmp_path / "segmentation.json"
    config.write_text(
        json.dumps(
            {
                "backend": "sam2",
                "checkpoint_root": ".venv/models/sam2",
                "model_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            }
        ),
        encoding="utf-8",
    )

    assert _load_backend_config(config) == {
        "backend": "sam2",
        "checkpoint_root": ".venv/models/sam2",
        "model_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    }


def test_sam2_prompt_defaults_point_labels_to_positive() -> None:
    prompts = _segmentation_prompts(
        {"segmentation_prompts": {"task_object": {"points": [[0.5, 0.5]], "normalized": True}}}
    )

    assert prompts["task_object"]["labels"] == [1]
    assert prompts["task_object"]["normalized"] is True


def test_sam2_can_pass_through_text_only_record_for_mujoco_masks(tmp_path: Path) -> None:
    frame_paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (8, 8), color=(index, index, index)).save(path)
        frame_paths.append(str(path))
    request_path, result_path, output_path = _request(
        tmp_path,
        {
            "base_id": "robometer/example",
            "frames": frame_paths,
            "object_prompts": {"task_object": "apple"},
        },
    )

    run_worker(
        request_path,
        result_path,
        tmp_path / "missing-checkpoint-is-not-loaded",
        backend="sam2",
        allow_missing_prompts=True,
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["metadata"]["segmentation"] == {
        "backend": "sam2",
        "status": "skipped-missing-prompts",
        "reason": "downstream simulator must provide masks",
    }
    assert json.loads(result_path.read_text(encoding="utf-8"))["records_path"] == str(output_path)


def test_none_backend_does_not_require_frames_or_checkpoint(tmp_path: Path) -> None:
    request_path, result_path, output_path = _request(tmp_path, {"base_id": "real/example"})

    run_worker(request_path, result_path, tmp_path / "unused", backend="none")

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["metadata"]["segmentation"] == {"backend": "none", "status": "skipped"}
