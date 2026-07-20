"""RLinf reward adapter for MI-distilled state-potential models.

DROP-IN LOCATION: rlinf/workers/reward/mi_potential_reward_model.py

Registers model_type "mi_potential" in RLinf's reward model registry.
Loads a frozen StatePotentialRewardModel and returns per-frame potentials
that EnvWorker converts to potential-difference rewards.

Usage in config:
  reward:
    model:
      model_type: "mi_potential"
      model_path: /path/to/mi_student/pytorch_model.pt
      config_path: /path/to/mi_student/dame_mi_base.yaml
      input_mode: "current_frame"
      image_keys: ["main_images"]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MIPotentialRewardModel(nn.Module):
    """RLinf adapter wrapping an MI-distilled state-potential model.

    This is the online inference wrapper. It does NOT run B-spline MI,
    temporal alignment, or pseudo-preference generation. It only runs
    the frozen student StatePotentialRewardModel.

    Reward output type: "state_potential"
    The EnvWorker computes the potential-difference reward.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        device: str = "cuda",
        precision: str = "fp32",
        input_mode: str = "current_frame",
        image_keys: list[str] | None = None,
        task_description: str | None = None,
    ):
        """
        Args:
            model_path: path to pytorch_model.pt checkpoint
            config_path: optional path to YAML config with reward_model section
            device: torch device
            precision: "fp32", "fp16", or "bf16"
            input_mode: "current_frame" or "history_buffer"
            image_keys: list of observation keys to use
            task_description: fixed task description for non-task-conditioned models
        """
        super().__init__()
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.precision = precision
        self.input_mode = input_mode
        self.image_keys = image_keys or ["main_images"]
        self.task_description = task_description

        # Load the inference model lazily (on first forward call)
        self._inference_model: Any = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load the MI potential model."""
        if self._loaded:
            return

        ckpt_path = Path(self.model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MI potential checkpoint not found: {self.model_path}. "
                f"Train the model with mi_reward first, or check the path."
            )

        try:
            from mi_reward.inference.potential_model import MIPotentialInferenceModel

            self._inference_model = MIPotentialInferenceModel.from_pretrained(
                checkpoint_path=self.model_path,
                config_path=self.config_path,
                device=self.device,
                precision=self.precision,
            )
            self._loaded = True
            logger.info(
                "Loaded MI potential model from %s (arch=%s, input_dim=%d)",
                self.model_path,
                self._inference_model.potential_model.architecture,
                self._inference_model.potential_model.input_dim,
            )
        except ImportError as e:
            raise ImportError(
                "mi_reward package is required for MI potential models. "
                "Install with: pip install -e /path/to/MI-directional-WAM-reward-model-pretrain"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MI potential model: {e}") from e

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        task_descriptions: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute state potentials for a batch of observations.

        Args:
            observations: dict with image tensors keyed by image_keys
            task_descriptions: optional task strings per env

        Returns:
            {
                "potential": Tensor[B],
                "confidence": Tensor[B] in [0, 1],
                "valid": BoolTensor[B],
            }
        """
        self._ensure_loaded()

        # Handle task description
        tasks = task_descriptions
        if tasks is None and self.task_description is not None:
            B = self._infer_batch_size(observations)
            tasks = [self.task_description] * B

        result = self._inference_model.predict_potential(observations, tasks)

        # Log validation
        invalid_count = (~result["valid"]).sum().item()
        if invalid_count > 0:
            logger.warning(
                "MI potential model returned %d/%d invalid predictions",
                invalid_count, len(result["valid"]),
            )

        return result

    def _infer_batch_size(self, observations: dict[str, torch.Tensor]) -> int:
        for key in self.image_keys:
            if key in observations:
                return observations[key].shape[0]
        return 1

    def get_reward_output_type(self) -> str:
        """Declare output semantics for reward composition."""
        return "state_potential"


# ---------------------------------------------------------------------------
# RLinf reward model registry integration
# ---------------------------------------------------------------------------
# Add to rlinf/workers/reward/__init__.py or the reward model registry:
#
#   from rlinf.workers.reward.mi_potential_reward_model import MIPotentialRewardModel
#   REWARD_MODEL_REGISTRY["mi_potential"] = MIPotentialRewardModel
#
# Or if RLinf uses a decorator-based registry:
#
#   @register_reward_model("mi_potential")
#   class MIPotentialRewardModel(nn.Module):
#       ...


class MIPotentialRewardModelConfig:
    """Configuration dataclass for MIPotentialRewardModel.

    Maps to the model: section of RLinf Hydra configs.
    """

    model_type: str = "mi_potential"
    model_path: str = ""
    config_path: str | None = None
    precision: str = "fp32"
    device: str = "cuda"
    input_mode: str = "current_frame"
    image_keys: list[str] = ["main_images"]
    task_description: str | None = None
