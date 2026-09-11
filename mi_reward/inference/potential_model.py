"""Stable inference contract for MI state-potential reward models.

Provides MIPotentialInferenceModel, a frozen wrapper around
StatePotentialRewardModel that can be loaded from a checkpoint
and used for batched online inference without gradients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mi_reward.models.state_potential_model import StatePotentialRewardModel


class MIPotentialInferenceModel(nn.Module):
    """Frozen inference wrapper for MI-distilled state-potential models.

    Loads a StatePotentialRewardModel checkpoint and provides
    batched inference with structured output including potentials,
    confidence estimates, and validity flags.

    Usage:
        model = MIPotentialInferenceModel.from_pretrained(
            checkpoint_path="path/to/pytorch_model.pt",
            device="cuda",
        )
        result = model.predict_potential(
            observations={"main_images": image_tensor},
            task_descriptions=["peg insertion"],
        )
        # result["potential"]: Tensor[B], result["confidence"]: Tensor[B]
    """

    def __init__(
        self,
        potential_model: StatePotentialRewardModel,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.potential_model = potential_model
        self.config = config
        self.metadata = metadata or {}
        self.device = torch.device(device)

        # Normalization statistics (optional, from metadata)
        self.potential_mean: float = float(self.metadata.get("potential_mean", 0.0))
        self.potential_std: float = float(self.metadata.get("potential_std", 1.0))

        # Image preprocessing config
        self.input_image_keys: list[str] = self.metadata.get("input_image_keys", ["main_images"])
        self.history_size: int = int(self.metadata.get("history_size", 1))
        self.training_gamma: float = float(self.metadata.get("training_gamma", 0.99))
        self.task_conditioned: bool = bool(self.metadata.get("task_conditioned", False))
        self.output_semantics: str = self.metadata.get("output_semantics", "state_potential")

        # Frozen
        self.potential_model.eval()
        for param in self.potential_model.parameters():
            param.requires_grad = False

        self.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str | torch.device = "cuda",
        precision: str = "fp32",
    ) -> "MIPotentialInferenceModel":
        """Load a pretrained state-potential model from checkpoint.

        Args:
            checkpoint_path: path to pytorch_model.pt
            config_path: optional path to model config YAML
            device: torch device
            precision: "fp32" or "fp16" or "bf16"

        Returns:
            MIPotentialInferenceModel ready for inference
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = checkpoint.get("config", {})

        # Load metadata if available
        metadata = {}
        meta_path = ckpt_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        # Merge YAML config if provided
        if config_path is not None:
            import yaml
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            # YAML overrides checkpoint config for inference params
            config = {**config, **yaml_config.get("reward_model", {})}

        # Build model
        input_dim = int(config.get("input_dim", 512))
        hidden_dim = int(config.get("hidden_dim", 256))
        architecture = config.get("architecture", metadata.get("architecture", "gru"))
        token_pooling = config.get("token_pooling", metadata.get("token_pooling", "mean"))
        num_layers = int(config.get("num_layers", metadata.get("num_layers", 2)))
        num_heads = int(config.get("num_heads", metadata.get("num_heads", 4)))
        dropout = float(config.get("dropout", metadata.get("dropout", 0.1)))
        use_task = config.get("use_task_conditioning", metadata.get("task_conditioned", False))
        task_dim = config.get("task_dim", metadata.get("task_dim", None))

        model = StatePotentialRewardModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            architecture=architecture,
            token_pooling=token_pooling,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_task_conditioning=use_task,
            task_dim=task_dim,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if precision == "fp16":
            model = model.half()
        elif precision == "bf16":
            model = model.bfloat16()

        return cls(
            potential_model=model,
            config=config,
            metadata=metadata,
            device=device,
        )

    @torch.inference_mode()
    def predict_potential(
        self,
        observations: dict[str, torch.Tensor],
        task_descriptions: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict state potentials for a batch of observations.

        Args:
            observations: dict mapping image_keys to tensors
                Each value is [B, ...] where ... depends on the encoder.
                If features are pre-extracted: {"features": [B, D]}
                If raw images with encoder: {"main_images": [B, C, H, W]}
            task_descriptions: optional list of task strings

        Returns:
            {
                "potential": Tensor[B] scalar potential per env,
                "confidence": Tensor[B] confidence in [0, 1],
                "valid": BoolTensor[B] validity flag,
            }
        """
        B = self._infer_batch_size(observations)
        device = self.device

        # Extract features from observations
        features = self._extract_features(observations)  # [B, T, D] or [B, D]

        # Ensure [B, T, D] format
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B, 1, D]

        # Move to device
        features = features.to(device)

        # Task conditioning (if used)
        task_features = None
        if self.task_conditioned and task_descriptions is not None:
            task_features = self._encode_tasks(task_descriptions)
            if task_features is not None:
                task_features = task_features.to(device)

        # Run model
        try:
            potentials = self.potential_model(features, task_features)  # [B, T]
            # Take last timestep potential for single-frame or accumulated
            potential = potentials[:, -1]  # [B]
        except Exception:
            # Return invalid on inference failure
            return {
                "potential": torch.zeros(B, device=device),
                "confidence": torch.zeros(B, device=device),
                "valid": torch.zeros(B, dtype=torch.bool, device=device),
            }

        # Validate output
        valid = self._validate_potential(potential)

        # Confidence: use 1.0 for valid, 0.0 otherwise (no confidence head)
        confidence = valid.float()

        # Mark invalid entries as zero
        potential = torch.where(valid, potential, torch.zeros_like(potential))

        return {
            "potential": potential,
            "confidence": confidence,
            "valid": valid,
        }

    def _infer_batch_size(self, observations: dict[str, torch.Tensor]) -> int:
        """Infer batch size from observations dict."""
        for key in self.input_image_keys:
            if key in observations:
                return observations[key].shape[0]
        # Fallback: use first tensor
        for v in observations.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        raise ValueError("Cannot infer batch size from observations")

    def _extract_features(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations.

        Override or configure this method to match the training preprocessing.
        Default: expects pre-extracted "features" key, or concatenates
        flattened image tensors.
        """
        # If features are already extracted
        if "features" in observations:
            return observations["features"]

        # If images are provided, use a simple fallback (mean-pool RGB)
        # In production, this should use the same encoder as training
        features_list = []
        for key in self.input_image_keys:
            if key in observations:
                img = observations[key].float()  # [B, C, H, W] or [B, H, W, C]
                # Simple mean-pool fallback (REPLACE with actual encoder in production)
                if img.dim() == 4 and img.shape[1] <= 3:
                    # [B, C, H, W] -> mean over spatial dims -> [B, C]
                    f = img.mean(dim=[-2, -1])  # [B, C]
                elif img.dim() == 4:
                    # [B, H, W, C] -> mean over spatial -> [B, C]
                    f = img.float().mean(dim=[1, 2])
                else:
                    f = img.float().flatten(1)
                features_list.append(f)

        if features_list:
            return torch.cat(features_list, dim=-1)  # [B, D]

        raise KeyError(f"No recognized feature keys in observations. Expected: {self.input_image_keys}")

    def _encode_tasks(self, task_descriptions: list[str]) -> torch.Tensor | None:
        """Encode task descriptions. Override for actual task encoder."""
        if self.potential_model.task_proj is None:
            return None
        # Placeholder: zero embedding
        task_dim = self.potential_model.task_proj.in_features
        return torch.zeros(len(task_descriptions), task_dim)

    def _validate_potential(self, potential: torch.Tensor) -> torch.Tensor:
        """Check for NaN, Inf, and extreme values."""
        valid = torch.isfinite(potential)
        # Check for extremely large values (> 1e6)
        valid = valid & (potential.abs() < 1e6)
        return valid  # [B] bool
