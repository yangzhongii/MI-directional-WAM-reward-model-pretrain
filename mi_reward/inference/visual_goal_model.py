"""Inference wrapper for the privileged-teacher visual student."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mi_reward.models.visual_goal_potential import VisualGoalPotential


class VisualGoalInferenceModel(nn.Module):
    """Frozen token-level student that never requires action/relation inputs."""

    def __init__(self, model: VisualGoalPotential, config: dict[str, object], device: str | torch.device = "cuda"):
        super().__init__()
        self.model = model.eval()
        self.config = config
        requested = torch.device(device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> "VisualGoalInferenceModel":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint.get("config", {}))
        if config.get("model_class") != "VisualGoalPotential":
            raise ValueError("Checkpoint is not a VisualGoalPotential checkpoint.")
        model = VisualGoalPotential(
            visual_dim=int(config["visual_dim"]),
            hidden_dim=int(config.get("hidden_dim", 256)),
            architecture=str(config.get("architecture", "gru")),  # type: ignore[arg-type]
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
            goal_dropout=float(config.get("goal_dropout", 0.2)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device=device)

    @torch.inference_mode()
    def predict_potential(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            state_tokens.to(self.device),
            None if goal_tokens is None else goal_tokens.to(self.device),
            None if state_mask is None else state_mask.to(self.device),
            None if goal_mask is None else goal_mask.to(self.device),
        )

    @torch.inference_mode()
    def deployment_reward(
        self,
        current_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        goal_tokens: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.compute_deployment_reward(
            current_tokens.to(self.device),
            next_tokens.to(self.device),
            None if goal_tokens is None else goal_tokens.to(self.device),
            None if goal_mask is None else goal_mask.to(self.device),
            gamma=float(self.config.get("gamma", 0.99)),
        )
