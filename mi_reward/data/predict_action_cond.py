"""External Cosmos Predict2.5 robot/action-conditioned worker contract."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActionConditionedPredictRequest:
    initial_image_path: str
    action_path: str
    robot_state_path: str
    output_dir: str
    prompt: str
    seed: int
    num_frames: int
    model_id: str = "2B/robot/action-cond"

    def validate(self) -> None:
        required = (self.initial_image_path, self.action_path, self.robot_state_path)
        if not all(Path(path).is_file() for path in required):
            raise FileNotFoundError("Action-conditioned prediction requires image, action, and robot-state files.")
        if self.num_frames < 1:
            raise ValueError("ActionConditionedPredictRequest.num_frames must be positive.")


@dataclass(frozen=True)
class ActionConditionedPrediction:
    frame_paths: list[str]
    model_id: str
    seed: int


class CommandActionConditionedPredictWorker:
    """Invoke a separately managed official Cosmos action-conditioned worker."""

    def __init__(self, command: list[str]):
        if not command:
            raise ValueError("Predict worker command must not be empty.")
        self.command = command

    def predict(self, request: ActionConditionedPredictRequest) -> ActionConditionedPrediction:
        request.validate()
        output = Path(request.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        request_path, result_path = output / "predict_request.json", output / "predict_result.json"
        request_path.write_text(json.dumps(asdict(request)), encoding="utf-8")
        cmd = [part.format(request=str(request_path), result=str(result_path)) for part in self.command]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Predict worker failed (rc={completed.returncode}): {completed.stderr[-2000:]}")
        if not result_path.is_file():
            raise RuntimeError("Predict worker did not write its result manifest.")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frames = [
            str(path if path.is_absolute() else (result_path.parent / path).resolve())
            for item in payload.get("frames", [])
            for path in [Path(str(item))]
        ]
        if len(frames) != request.num_frames or not all(Path(path).is_file() for path in frames):
            raise RuntimeError("Predict worker result does not match requested frame count.")
        return ActionConditionedPrediction(
            frame_paths=frames,
            model_id=str(payload.get("model_id", request.model_id)),
            seed=request.seed,
        )
