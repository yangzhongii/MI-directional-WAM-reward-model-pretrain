"""External Cosmos Transfer2.5 worker contract for scene-level variation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from mi_reward.data.schema import ControlArtifacts


@dataclass(frozen=True)
class SceneTransferRequest:
    input_frame_paths: list[str]
    output_dir: str
    prompt: str
    controls: ControlArtifacts
    seed: int
    model_id: str = "Cosmos-Transfer2.5-2B"

    def validate(self) -> None:
        if not self.input_frame_paths or not all(Path(path).is_file() for path in self.input_frame_paths):
            raise FileNotFoundError("SceneTransferRequest requires existing input frames.")
        if not self.controls.mask_root or not Path(self.controls.mask_root).is_dir():
            raise ValueError("SceneTransferRequest requires a mask_root.")
        if not self.controls.depth_root or not Path(self.controls.depth_root).is_dir():
            raise ValueError("SceneTransferRequest requires a depth_root.")


@dataclass(frozen=True)
class TransferResult:
    frame_paths: list[str]
    model_id: str
    seed: int


class CommandTransferWorker:
    """Run a version-pinned worker command using request/result JSON files.

    ``command`` items may use ``{request}`` and ``{result}`` placeholders. The
    worker must write ``{"frames": [...], "model_id": "..."}`` to result.
    """

    def __init__(self, command: list[str]):
        if not command:
            raise ValueError("Transfer worker command must not be empty.")
        self.command = command

    def generate_scene_variant(self, request: SceneTransferRequest) -> TransferResult:
        request.validate()
        output = Path(request.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        request_path, result_path = output / "transfer_request.json", output / "transfer_result.json"
        request_path.write_text(json.dumps(asdict(request)), encoding="utf-8")
        cmd = [part.format(request=str(request_path), result=str(result_path)) for part in self.command]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Transfer worker failed (rc={completed.returncode}): {completed.stderr[-2000:]}")
        if not result_path.is_file():
            raise RuntimeError("Transfer worker did not write its result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frames = [
            str(path if path.is_absolute() else (result_path.parent / path).resolve())
            for item in payload.get("frames", [])
            for path in [Path(str(item))]
        ]
        if len(frames) != len(request.input_frame_paths) or not all(Path(path).is_file() for path in frames):
            raise RuntimeError("Transfer worker result has missing frame paths.")
        return TransferResult(
            frame_paths=frames,
            model_id=str(payload.get("model_id", request.model_id)),
            seed=request.seed,
        )
