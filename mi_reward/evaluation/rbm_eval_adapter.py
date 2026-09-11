"""Evaluate MI reward checkpoints with the official Robometer/RBM-EVAL metrics.

Robometer remains an unmodified third-party source tree. This adapter uses its
processed-data samplers and metric compilers, and only translates local reward
model predictions into the result dictionaries those compilers consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def _setup_robometer(source_root: Path, processed_root: Path) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Robometer source is missing: {source_root}. Run --reward-eval install first.")
    if not processed_root.exists():
        raise FileNotFoundError(
            f"Processed RBM data is missing: {processed_root}. "
            "Run install.sh --reward-eval --download-eval-data."
        )
    sys.path.insert(0, str(source_root))
    import os

    os.environ["ROBOMETER_PROCESSED_DATASETS_PATH"] = str(processed_root)


class _FeatureEncoder:
    """Batch RGB frames using the same DINO extractor contract as MI training."""

    def __init__(self, cfg: dict[str, Any], model_input_dim: int):
        from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor

        extractor_name = str(cfg.get("extractor", "dino_v3")).lower()
        device_name = str(cfg.get("device", "cuda"))
        self.device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        self.image_size = int(cfg.get("image_size", 224))
        self.batch_size = max(1, int(cfg.get("batch_size", 8)))
        self.extractor_name = extractor_name
        dino = DINOv3FeatureExtractor(
            model_path=str(_path(cfg.get("model_path"))) if cfg.get("model_path") else None,
            device=str(self.device), image_size=self.image_size, strict=False,
        )
        if extractor_name in {"dino_v3", "dino", "dinov3"}:
            if bool(cfg.get("strict", True)) and dino.model is None:
                raise RuntimeError("DINOv3 weights are required in strict mode.")
            self.extractor = dino
            self.model = dino.model
        elif extractor_name in {"lawam_lam", "lam"}:
            from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor

            self.extractor = LaWAMLAMFeatureExtractor(
                lam_config_path=str(_path(cfg.get("lam_config_path"))) if cfg.get("lam_config_path") else None,
                lam_ckpt_path=str(_path(cfg.get("lam_ckpt_path"))) if cfg.get("lam_ckpt_path") else None,
                vision_model_id=str(_path(cfg.get("vision_model_id"))) if cfg.get("vision_model_id") else None,
                device=str(self.device), fallback=dino, strict=bool(cfg.get("strict", True)),
            )
            self.model = None
        else:
            raise ValueError(f"Unsupported feature extractor: {extractor_name}")
        self.model_input_dim = model_input_dim

    @staticmethod
    def _as_tensor(frames: Any) -> torch.Tensor:
        array = np.asarray(frames)
        if array.ndim != 4:
            raise ValueError(f"Expected trajectory frames [T,H,W,C] or [T,C,H,W], got {array.shape}")
        if array.shape[-1] in (1, 3, 4):
            tensor = torch.from_numpy(array[..., :3]).permute(0, 3, 1, 2)
        elif array.shape[1] in (1, 3, 4):
            tensor = torch.from_numpy(array[:, :3])
        else:
            raise ValueError(f"Cannot identify RGB channel dimension in frame array {array.shape}")
        return tensor.float().div(255.0 if tensor.max() > 1.5 else 1.0)

    @torch.inference_mode()
    def encode(self, frames: Any, *, tokens: bool = False) -> torch.Tensor:
        images = self._as_tensor(frames)
        if self.extractor_name in {"lawam_lam", "lam"}:
            import tempfile
            from PIL import Image

            with tempfile.TemporaryDirectory(prefix="rbm_eval_frames_") as tmp:
                paths = []
                for index, image in enumerate(images):
                    path = Path(tmp) / f"{index:05d}.png"
                    array = image.mul(255).byte().permute(1, 2, 0).numpy()
                    Image.fromarray(array).save(path)
                    paths.append(str(path))
                if tokens:
                    result = self.extractor.extract_trajectory_tokens(paths, task="")
                else:
                    result = self.extractor.extract_trajectory(paths, task="")
            if result.shape[-1] != self.model_input_dim:
                raise RuntimeError(
                    f"Feature dimension {result.shape[-1]} does not match reward checkpoint "
                    f"input {self.model_input_dim}."
                )
            return result.float().cpu()
        if self.model is None:
            pooled = images.mean(dim=(-1, -2))
            if self.model_input_dim != pooled.shape[-1]:
                raise RuntimeError(
                    f"Fallback RGB features have dimension {pooled.shape[-1]}, "
                    f"checkpoint expects {self.model_input_dim}."
                )
            return pooled

        outputs: list[torch.Tensor] = []
        for batch in images.split(self.batch_size):
            batch = torch.nn.functional.interpolate(
                batch.to(self.device),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            output = self.model(batch)
            if hasattr(output, "last_hidden_state"):
                value = output.last_hidden_state
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                value = output.pooler_output.unsqueeze(1)
            elif isinstance(output, dict):
                value = output.get("x_norm_clstoken", next(iter(output.values())))
            elif isinstance(output, (tuple, list)):
                value = output[0]
            else:
                value = output
            if value.ndim == 2:
                value = value.unsqueeze(1)
            value = value.float()
            outputs.append(value if tokens else value.mean(dim=1))
        result = torch.cat(outputs, dim=0).cpu()
        if result.shape[-1] != self.model_input_dim:
            raise RuntimeError(
                f"Feature dimension {result.shape[-1]} does not match reward checkpoint "
                f"input {self.model_input_dim}."
            )
        return result


class _RewardAdapter:
    def __init__(self, cfg: dict[str, Any]):
        requested_type = str(cfg.get("model", {}).get("type", "auto")).lower()
        if requested_type in {"qwen3_vl", "qwen", "qwen3-vl"}:
            from mi_reward.inference.qwen3_vl_reward import Qwen3VLRewardModel

            model_cfg = cfg.get("model", {})
            self.model_type = "qwen3_vl"
            self.config = dict(model_cfg)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Qwen3VLRewardModel(
                model_cfg["model_path"],
                device=str(self.device),
            )
            self.encoder = None
            self.aggregation = str(model_cfg.get("preference_aggregation", "last"))
            self.relations = {}
            self.visual_goals = {}
            self._prediction_cache = {}
            return

        checkpoint_path = _path(cfg["paths"]["checkpoint"])
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Reward checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.config = dict(checkpoint.get("config", {}))
        requested_type = str(cfg.get("model", {}).get("type", "auto")).lower()
        aliases = {
            "statepotentialrewardmodel": "state_potential",
            "trajectoryrewardhead": "trajectory_head",
            "geoprogresspotential": "geo_progress",
            "visualgoalpotential": "visual_goal",
        }
        self.model_type = self._infer_type() if requested_type == "auto" else aliases.get(requested_type, requested_type)
        model_cfg = cfg.get("model", {})
        requested_device = str(model_cfg.get("device", "cuda"))
        self.device = torch.device(
            requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model(checkpoint)
        precision = str(model_cfg.get("precision", "fp32"))
        if precision == "fp16":
            self.model = self.model.half()
        elif precision == "bf16":
            self.model = self.model.bfloat16()
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.encoder = _FeatureEncoder(cfg.get("features", {}), self._input_dim())
        self.aggregation = str(model_cfg.get("preference_aggregation", "last"))
        self.relations = self._load_relation_sidecar(cfg.get("geoprogress", {}).get("relation_sidecar"))
        self.visual_goals = self._load_goal_sidecar(cfg.get("visual_goal", {}).get("goal_sidecar"))
        self._prediction_cache: dict[tuple[str, int], np.ndarray] = {}

    def _infer_type(self) -> str:
        if self.config.get("model_class") == "GeoProgressPotential":
            return "geo_progress"
        if self.config.get("model_class") == "StatePotentialRewardModel":
            return "state_potential"
        if self.config.get("model_class") == "VisualGoalPotential":
            return "visual_goal"
        return "trajectory_head"

    def _input_dim(self) -> int:
        if self.model_type in {"geo", "geo_progress", "geoprogress", "visual_goal", "visualgoal"}:
            return int(self.config["visual_dim"])
        return int(self.config.get("input_dim", getattr(self.model, "input_dim", 0)))

    def _build_model(self, checkpoint: dict[str, Any]) -> torch.nn.Module:
        from mi_reward.models.geoprogress_potential import GeoProgressPotential
        from mi_reward.models.reward_head import TrajectoryRewardHead
        from mi_reward.models.state_potential_model import StatePotentialRewardModel
        from mi_reward.models.visual_goal_potential import VisualGoalPotential

        if self.model_type in {"state", "state_potential", "potential"}:
            if self.config.get("use_task_conditioning"):
                raise ValueError(
                    "StatePotentialRewardModel task conditioning needs a configured task encoder; "
                    "RBM adapter cannot guess it."
                )
            model = StatePotentialRewardModel(
                input_dim=int(self.config["input_dim"]),
                hidden_dim=int(self.config.get("hidden_dim", 256)),
                architecture=str(self.config.get("architecture", "gru")),
                token_pooling=str(self.config.get("token_pooling", "mean")),
                num_layers=int(self.config.get("num_layers", 2)),
                num_heads=int(self.config.get("num_heads", 4)),
                dropout=float(self.config.get("dropout", 0.1)),
            )
        elif self.model_type in {"trajectory", "trajectory_head", "legacy"}:
            model = TrajectoryRewardHead(
                input_dim=int(self.config["input_dim"]),
                hidden_dim=int(self.config.get("hidden_dim", 256)),
            )
        elif self.model_type in {"geo", "geo_progress", "geoprogress"}:
            model = GeoProgressPotential(
                visual_dim=int(self.config["visual_dim"]),
                relation_dim=int(self.config["relation_dim"]),
                hidden_dim=int(self.config.get("hidden_dim", 256)),
                architecture=str(self.config.get("architecture", "gru")),
                num_layers=int(self.config.get("num_layers", 2)),
                num_heads=int(self.config.get("num_heads", 4)),
                dropout=float(self.config.get("dropout", 0.1)),
            )
        elif self.model_type in {"visual_goal", "visualgoal"}:
            model = VisualGoalPotential(
                visual_dim=int(self.config["visual_dim"]),
                hidden_dim=int(self.config.get("hidden_dim", 256)),
                architecture=str(self.config.get("architecture", "gru")),
                num_layers=int(self.config.get("num_layers", 2)),
                num_heads=int(self.config.get("num_heads", 4)),
                dropout=float(self.config.get("dropout", 0.1)),
                goal_dropout=float(self.config.get("goal_dropout", 0.2)),
            )
        else:
            raise ValueError(f"Unsupported reward model type: {self.model_type}")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def _load_relation_sidecar(value: str | None) -> dict[str, dict[str, Any]]:
        if not value:
            return {}
        sidecar = _path(value)
        if sidecar is None or not sidecar.exists():
            raise FileNotFoundError(f"GeoProgress relation sidecar not found: {sidecar}")
        records: dict[str, dict[str, Any]] = {}
        with sidecar.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    key = row.get("trajectory_id", row.get("id"))
                    if not key:
                        raise ValueError(f"Missing trajectory_id/id in {sidecar}")
                    records[str(key)] = row
        return records

    @staticmethod
    def _load_goal_sidecar(value: str | None) -> dict[str, dict[str, Any]]:
        if not value:
            return {}
        sidecar = _path(value)
        if sidecar is None or not sidecar.exists():
            raise FileNotFoundError(f"Visual goal sidecar not found: {sidecar}")
        records: dict[str, dict[str, Any]] = {}
        with sidecar.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("trajectory_id", row.get("id", row.get("task")))
                if not key:
                    raise ValueError(f"Missing trajectory_id/id/task in {sidecar}")
                records[str(key)] = row
        return records

    @staticmethod
    def _load_frame_paths(paths: list[str]) -> np.ndarray:
        from PIL import Image

        frames = [np.asarray(Image.open(_path(item) or Path(item)).convert("RGB")) for item in paths]
        return np.stack(frames, axis=0)

    def _geo_inputs(self, traj: Any, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.relations.get(str(traj.id))
        if record is None:
            raise ValueError(
                "GeoProgressPotential requires a relation/goal sidecar row for every RBM trajectory "
                f"(missing id={traj.id}). Configure geoprogress.relation_sidecar."
            )
        relations = torch.as_tensor(record["relations"], dtype=torch.float32)
        if relations.ndim != 2:
            raise ValueError(
                f"Relation row for {traj.id} must be a [T,R] matrix; got {tuple(relations.shape)}"
            )
        if relations.shape[0] > tokens.shape[0]:
            indices = np.rint(np.linspace(0, relations.shape[0] - 1, tokens.shape[0])).astype(int)
            indices[0], indices[-1] = 0, relations.shape[0] - 1
            relations = relations[torch.as_tensor(indices, dtype=torch.long)]
        if relations.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Relation row for {traj.id} has T={relations.shape[0]}, but sampled frames have T={tokens.shape[0]}."
            )
        if relations.shape[1] != int(self.config["relation_dim"]):
            raise ValueError(f"Relation dimension for {traj.id} does not match checkpoint.")
        if record.get("goal_tokens_path"):
            goal_tokens = torch.load(_path(record["goal_tokens_path"]), map_location="cpu").float()
            if goal_tokens.ndim == 3:
                goal_tokens = goal_tokens.reshape(-1, goal_tokens.shape[-1])
        elif record.get("goal_frames"):
            goal_frames = record["goal_frames"]
            if goal_frames and isinstance(goal_frames[0], str):
                goal_frames = self._load_frame_paths(goal_frames)
            goal_tokens = self.encoder.encode(goal_frames, tokens=True).reshape(-1, tokens.shape[-1])
        else:
            raise ValueError(f"GeoProgress sidecar row {traj.id} needs goal_frames or goal_tokens_path.")
        return goal_tokens, relations

    def _visual_goal(self, traj: Any, tokens: torch.Tensor) -> torch.Tensor | None:
        record = self.visual_goals.get(str(traj.id)) or self.visual_goals.get(str(traj.task))
        if record is None:
            return None
        if record.get("goal_tokens_path"):
            goal_tokens = torch.load(_path(record["goal_tokens_path"]), map_location="cpu").float()
            if goal_tokens.ndim == 3:
                goal_tokens = goal_tokens.reshape(-1, goal_tokens.shape[-1])
        elif record.get("goal_frames"):
            goal_frames = record["goal_frames"]
            if goal_frames and isinstance(goal_frames[0], str):
                goal_frames = self._load_frame_paths(goal_frames)
            goal_tokens = self.encoder.encode(goal_frames, tokens=True).reshape(-1, tokens.shape[-1])
        else:
            raise ValueError(f"Visual goal sidecar row for {traj.id} needs goal_frames or goal_tokens_path.")
        return goal_tokens

    def _build_qwen_row_from_trajectory(self, traj: Any) -> dict[str, Any]:
        """Convert a Robometer trajectory into the Pipeline-v3 Qwen input schema.

        Robometer stores raw trajectories (frames + metadata), while the Qwen
        reward model expects language plus a short visual history.  Prefer an
        explicitly exported qwen_row when available, but allow evaluation on
        native trajectories by extracting the common LIBERO-style camera keys.
        """
        metadata = dict(getattr(traj, "metadata", {}) or {})
        language = (
            metadata.get("instruction")
            or metadata.get("task_description")
            or metadata.get("language")
            or metadata.get("task_name")
            or getattr(traj, "task", None)
            or ""
        )
        frames = getattr(traj, "frames", None)
        if frames is None:
            raise ValueError("Qwen3-VL adapter needs trajectory frames.")

        history = list(frames[-5:])
        agent_images, wrist_images = [], []

        def camera(frame, keys):
            return next((frame[key] for key in keys if key in frame and frame[key] is not None), None)

        for frame in history:
            if isinstance(frame, dict):
                agent = camera(frame, ("agentview_image", "agentview", "front"))
                wrist = camera(frame, ("robot0_eye_in_hand_image", "wrist_image", "wrist"))
                if agent is None or wrist is None:
                    raise ValueError("Qwen v3 requires synchronized agentview and wrist; incomplete camera input")
                agent_images.append(agent)
                wrist_images.append(wrist)
            else:
                raise ValueError("Qwen v3 requires dual-view frames; a single-view benchmark needs an explicitly separate protocol")

        images = agent_images + wrist_images
        if not images:
            raise ValueError("Could not extract RGB images from Robometer trajectory frames.")

        # Qwen3-VL data pipeline expects image references (paths or resolvable
        # image objects), not raw numpy arrays. Robometer stores decoded RGB
        # arrays, so materialize them as temporary PNG files first.
        qwen_images = []
        from PIL import Image
        import tempfile

        if not hasattr(self, "_qwen_frame_dir"):
            self._qwen_frame_dir = tempfile.TemporaryDirectory(prefix="rbm_qwen_frames_")

        for index, image in enumerate(images):
            if isinstance(image, (str, Path)):
                qwen_images.append(str(image))
                continue
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            if hasattr(image, "save"):
                path = Path(self._qwen_frame_dir.name) / f"{index:03d}.png"
                image.save(path)
                qwen_images.append(str(path))
            else:
                raise TypeError(f"Unsupported Robometer image type: {type(image)}")

        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Task: {str(language).rstrip('. ')}. Judge recent task progress as Positive, Unclear, or Negative.",
                }
            ],
            "images": qwen_images,
            "metadata": {
                "trajectory_id": str(getattr(traj, "id", "unknown")),
                "source": "robometer",
            },
            "provenance": {
                "source": "robometer",
                "trajectory_id": str(getattr(traj, "id", "unknown")),
                "history_frame_indices": list(
                    range(max(0, len(frames) - len(history)), len(frames))
                ),
            },
        }

    @torch.inference_mode()
    def predict(self, traj: Any) -> np.ndarray:
        if self.model_type == "qwen3_vl":
            metadata = dict(getattr(traj, "metadata", {}) or {})
            row = metadata.get("qwen_row") or self._build_qwen_row_from_trajectory(traj)
            history = row.get("provenance", {}).get("history_frame_indices", [])
            if not history or len(row.get("images", [])) != 2 * len(history):
                raise ValueError("Invalid dual-view Qwen row: image/history count mismatch")
            result = self.model.predict_row(row)
            return np.asarray([result.reward], dtype=np.float32)

        if traj.frames is None:
            raise ValueError(f"Trajectory {traj.id} has no RGB frames.")
        cache_key = (str(traj.id), len(traj.frames))
        if traj.id is not None and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key].copy()
        model_dtype = next(self.model.parameters()).dtype
        if self.model_type in {"visual_goal", "visualgoal"}:
            state_tokens = self.encoder.encode(traj.frames, tokens=True)
            goal_tokens = self._visual_goal(traj, state_tokens)
            output = self.model(
                state_tokens.unsqueeze(0).to(self.device, dtype=model_dtype),
                None if goal_tokens is None else goal_tokens.unsqueeze(0).to(self.device, dtype=model_dtype),
            )[0]
        elif self.model_type in {"geo", "geo_progress", "geoprogress"}:
            state_tokens = self.encoder.encode(traj.frames, tokens=True)
            goal_tokens, relations = self._geo_inputs(traj, state_tokens)
            output = self.model(
                state_tokens.unsqueeze(0).to(self.device, dtype=model_dtype),
                goal_tokens.unsqueeze(0).to(self.device, dtype=model_dtype),
                relations.unsqueeze(0).to(self.device, dtype=model_dtype),
            )[0]
        elif self.model_type in {"trajectory", "trajectory_head", "legacy"}:
            features = self.encoder.encode(traj.frames).to(self.device, dtype=model_dtype)
            output = torch.stack(
                [self.model(features[: index + 1].unsqueeze(0))[0] for index in range(features.shape[0])]
            )
        else:
            features = self.encoder.encode(traj.frames).to(self.device, dtype=model_dtype).unsqueeze(0)
            output = self.model(features)[0]
        prediction = output.float().detach().cpu().numpy()
        if traj.id is not None:
            self._prediction_cache[cache_key] = prediction
        return prediction.copy()

    def score(self, progress: np.ndarray) -> float:
        if self.aggregation == "mean":
            return float(np.mean(progress))
        if self.aggregation == "sum":
            return float(np.sum(progress))
        if self.aggregation == "delta":
            gamma = float(self.config.get("gamma", 0.99))
            return float(np.mean(gamma * progress[1:] - progress[:-1])) if len(progress) > 1 else float(progress[-1])
        return float(progress[-1])


def _trajectory_result(traj: Any, prediction: np.ndarray) -> dict[str, Any]:
    metadata = dict(traj.metadata or {})
    metadata.setdefault("id", traj.id)
    return {
        "progress_pred": prediction,
        "task": traj.task,
        "data_source": traj.data_source,
        "data_gen_strategy": traj.data_gen_strategy,
        "metadata": metadata,
        "id": traj.id,
        "video_path": metadata.get("video_path") if isinstance(metadata.get("video_path"), str) else None,
        "partial_success": traj.partial_success,
        "target_progress": np.asarray(
            traj.target_progress if traj.target_progress is not None else np.zeros(len(prediction))
        ),
        "quality_label": traj.quality_label,
    }


def _preference_result(sample: Any, chosen_score: float, rejected_score: float) -> dict[str, Any]:
    chosen, rejected = sample.chosen_trajectory, sample.rejected_trajectory

    def metadata(traj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "quality_label": traj.quality_label,
            "data_source": traj.data_source,
            "task": traj.task,
            "id": traj.id,
            "video_path": None,
        }
        if traj.partial_success is not None:
            result["partial_success"] = traj.partial_success
        return result

    # Official RBM-EVAL expects preference_pred to be a binary prediction array.
    prediction = np.asarray([float(chosen_score > rejected_score)], dtype=np.float32)
    return {
        "preference_pred": prediction,
        "preference_labels": np.asarray([1.0], dtype=np.float32),
        "is_correct": bool(chosen_score > rejected_score),
        "is_tie": bool(chosen_score == rejected_score),
        "reward_margin": float(chosen_score - rejected_score),
        "task": chosen.task,
        "data_source": chosen.data_source or rejected.data_source,
        "metadata": {"chosen_metadata": metadata(chosen), "rejected_metadata": metadata(rejected)},
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(_path(str(config_path)) or Path(config_path))
    robometer_cfg = config["robometer"]
    _setup_robometer(
        _path(robometer_cfg["source_root"]) or Path(robometer_cfg["source_root"]),
        _path(robometer_cfg["processed_datasets"]) or Path(robometer_cfg["processed_datasets"]),
    )
    from robometer.configs.experiment_configs import DataConfig
    from robometer.data.dataset_types import PreferenceSample
    from robometer.data.datasets.base import resolve_dataset_keys
    from robometer.data.datasets.custom_eval import CustomEvalDataset
    from robometer.evals.compile_results import (
        run_policy_ranking_eval,
        run_quality_preference_eval,
        run_reward_alignment_eval_per_trajectory,
    )
    from tqdm import tqdm

    adapter = _RewardAdapter(config)
    benchmark = config.get("benchmark", {})
    output_dir = _path(config["paths"]["output_dir"]) or ROOT / "results/rbm_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, Any] = {}
    common = {
        "random_seed": int(benchmark.get("random_seed", 42)),
        "pad_frames": bool(benchmark.get("pad_frames", False)),
    }
    for eval_type in benchmark.get("eval_types", []):
        type_metrics: dict[str, Any] = {}
        for dataset_key in benchmark.get("datasets", {}).get(eval_type, []):
            resolved = resolve_dataset_keys([dataset_key], split="eval")
            data_cfg = DataConfig(
                eval_datasets=resolved,
                train_datasets=resolved,
                dataset_type="rbm",
                max_frames=int(benchmark.get("max_frames", 16)),
                min_frames_per_trajectory=int(benchmark.get("min_frames_per_trajectory", 2)),
                load_embeddings=False,
                progress_pred_type="absolute_wrt_total_frames",
                max_success=1.0,
                sample_type_ratio=[1, 1, 1],
                seed=int(benchmark.get("random_seed", 42)),
            )
            sampler_kwargs = dict(common)
            if eval_type == "reward_alignment":
                sampler_kwargs.update(
                    max_trajectories=benchmark.get("reward_alignment_max_trajectories"),
                    use_frame_steps=bool(benchmark.get("use_frame_steps", False)),
                )
            elif eval_type == "policy_ranking":
                sampler_kwargs.update(
                    max_tasks=benchmark.get("policy_ranking_max_tasks"),
                    num_examples_per_quality_pr=benchmark.get("num_examples_per_quality_pr"),
                    use_frame_steps=bool(benchmark.get("use_frame_steps", False)),
                )
            elif eval_type == "quality_preference":
                sampler_kwargs.update(
                    comparisons_per_task=benchmark.get("comparisons_per_task"),
                    max_comparisons=benchmark.get("max_comparisons"),
                )
            else:
                raise ValueError(f"Unsupported RBM eval type: {eval_type}")

            dataset = CustomEvalDataset(eval_type, data_cfg, verbose=True, sampler_kwargs=sampler_kwargs)
            results: list[dict[str, Any]] = []
            for sample in tqdm(dataset, desc=f"RBM-EVAL {eval_type}/{dataset_key}"):
                if isinstance(sample, PreferenceSample):
                    chosen_score = adapter.score(adapter.predict(sample.chosen_trajectory))
                    rejected_score = adapter.score(adapter.predict(sample.rejected_trajectory))
                    results.append(_preference_result(sample, chosen_score, rejected_score))
                else:
                    results.append(_trajectory_result(sample.trajectory, adapter.predict(sample.trajectory)))
            if not results:
                type_metrics[str(dataset_key)] = {"error": "No samples generated"}
                continue

            source = results[0].get("data_source")
            if eval_type == "quality_preference":
                metrics, groups, details = run_quality_preference_eval(results, data_source=source)
                from mi_reward.evaluation.reward_pair_metrics import summarize_margins
                metrics.update(summarize_margins(r["chosen_score"] - r["rejected_score"] for r in results))
            elif eval_type == "policy_ranking":
                metrics, groups, details = run_policy_ranking_eval(
                    results, "absolute_wrt_total_frames", False, 10, source, "kendall"
                )
                from mi_reward.evaluation.reward_pair_metrics import quality_pair_summary
                # Keep official metrics for traceability but distinguish their
                # order-dependent tie handling from the corrected summary.
                metrics = {"official_legacy_metrics": metrics,
                           "strict_pair_metrics": quality_pair_summary(results)}
            else:
                metrics, plots, _, progress_data = run_reward_alignment_eval_per_trajectory(
                    results,
                    "absolute_wrt_total_frames",
                    False,
                    10,
                    source,
                    bool(benchmark.get("use_frame_steps", False)),
                    False,
                    False,
                )
                for plot in plots:
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(plot)
                    except Exception:
                        pass
                groups, details = {}, {"num_trajectories": len(progress_data)}

            slug = str(dataset_key).replace("/", "_")
            type_metrics[slug] = _jsonable(metrics)
            dataset_out = output_dir / eval_type
            dataset_out.mkdir(parents=True, exist_ok=True)
            if bool(config.get("runtime", {}).get("save_raw_results", True)):
                (dataset_out / f"{slug}_results.json").write_text(
                    json.dumps(_jsonable(results), indent=2), encoding="utf-8"
                )
            (dataset_out / f"{slug}_details.json").write_text(
                json.dumps(_jsonable(details), indent=2), encoding="utf-8"
            )
        all_metrics[eval_type] = type_metrics

    report = {
        "checkpoint": str(_path(config["paths"]["checkpoint"])),
        "model_type": adapter.model_type,
        "metrics": all_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official Robometer/RBM-EVAL metrics on an MI reward checkpoint.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(_jsonable(run(args.config)), indent=2))


if __name__ == "__main__":
    main()
