"""CPU-only contracts for the Cosmos Transfer scene-variation worker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mi_reward.data import cosmos_transfer_worker as worker


def test_scene_variant_ids_are_unique_and_path_safe(tmp_path: Path) -> None:
    variants = tmp_path / "variants.yaml"
    variants.write_text(
        "variants:\n"
        "  - variant_id: ../escape\n"
        "    prompt: unsafe\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="variant IDs"):
        worker._load_variants(variants)

    variants.write_text(
        "variants:\n"
        "  - variant_id: lab\n"
        "    prompt: first\n"
        "  - variant_id: lab\n"
        "    prompt: second\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        worker._load_variants(variants)


def test_transfer_mask_applies_depth_control_to_sam_foreground(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    source = np.zeros((4, 4), dtype=np.uint8)
    source[1:3, 1:3] = 255
    Image.fromarray(source).save(mask_root / "000000.png")

    monkeypatch.setattr(worker, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(worker, "_run", lambda command, cwd=None: None)
    output = tmp_path / "mask.mp4"
    worker._encode_mask(mask_root, output, frame_count=1, size=(4, 4), fps=16)

    encoded = np.asarray(Image.open(tmp_path / "mask_maps" / "000000.png"))
    assert encoded[1, 1] == 255
    assert encoded[0, 0] == 0


def test_worker_uses_official_cli_and_keeps_sidecars_valid_across_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    transfer_repo = tmp_path / "transfer"
    (transfer_repo / "examples").mkdir(parents=True)
    (transfer_repo / "examples" / "inference.py").write_text("", encoding="utf-8")
    frames = input_root / "frames"
    masks = input_root / "masks"
    depths = input_root / "depths"
    for directory in (frames, masks, depths):
        directory.mkdir(parents=True)
    for index in range(2):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(frames / f"{index:06d}.png")
        Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(masks / f"{index:06d}.png")
        np.save(depths / f"{index:06d}.npy", np.zeros((4, 4), dtype=np.float32))
    for name in ("actions.json", "states.json", "objects.json", "relations.json"):
        (input_root / name).write_text("[]", encoding="utf-8")

    records = input_root / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "traj_id": "predict/pick_place/0",
                "task": "pick_place",
                "frames": ["frames/000000.png", "frames/000001.png"],
                "action_path": "actions.json",
                "robot_state_path": "states.json",
                "object_state_path": "objects.json",
                "relation_path": "relations.json",
                "control_artifacts": {"mask_root": "masks", "depth_root": "depths"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_records = output_root / "records.jsonl"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"input_records": str(records), "output_records": str(output_records)}),
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    variants = tmp_path / "variants.yaml"
    variants.write_text(
        "variants:\n"
        "  - variant_id: lab_light\n"
        "    prompt: A bright laboratory.\n"
        "  - variant_id: lab_dark\n"
        "    prompt: A dark laboratory.\n",
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path | None = None) -> None:
        commands.append(command)
        input_index = command.index("-i")
        output_index = command.index("-o")
        specs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in command[input_index + 1:output_index]]
        inference_output = Path(command[output_index + 1])
        inference_output.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            (inference_output / f"{spec['name']}.mp4").write_bytes(b"video")

    def fake_extract(video: Path, destination: Path, expected: int) -> list[str]:
        destination.mkdir(parents=True, exist_ok=True)
        frame = destination / "000000.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(frame)
        second = destination / "000001.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(second)
        return [str(frame.resolve()), str(second.resolve())]

    monkeypatch.setattr(worker, "_encode_frames", lambda frames, output, fps: (4, 4))
    monkeypatch.setattr(worker, "_encode_depth", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_encode_mask", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_extract_frames", fake_extract)
    monkeypatch.setattr(worker, "_run", fake_run)

    checkpoint = str(tmp_path / "depth.pt")
    worker.run_worker(request, result, transfer_repo, variants, "/venv/bin/python", 16, checkpoint, 2)

    assert commands == [[
        "/venv/bin/torchrun",
        "--nproc_per_node",
        "2",
        "examples/inference.py",
        "-i",
        str(output_root / "scene_variants" / "record_000000_lab_light" / "transfer_spec.json"),
        str(output_root / "scene_variants" / "record_000000_lab_dark" / "transfer_spec.json"),
        "-o",
        str(output_root / "scene_variants" / "inference"),
        "--model",
        "depth",
        "--checkpoint-path",
        checkpoint,
    ]]
    generated = [json.loads(line) for line in output_records.read_text(encoding="utf-8").splitlines()]
    assert [item["traj_id"] for item in generated] == [
        "predict/pick_place/0/scene-lab_light",
        "predict/pick_place/0/scene-lab_dark",
    ]
    assert all(item["parent_traj_id"] == "predict/pick_place/0" for item in generated)
    assert all(item["action_path"] == str((input_root / "actions.json").resolve()) for item in generated)
    assert all(item["control_artifacts"]["mask_root"] == str(masks.resolve()) for item in generated)
    assert all(item["scene_variant"]["model_id"] == "Cosmos-Transfer2.5-2B/depth" for item in generated)
    assert json.loads(result.read_text(encoding="utf-8"))["records_path"] == str(output_records.resolve())
