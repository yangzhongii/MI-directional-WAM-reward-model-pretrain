"""Inference wrapper for a trained GeoProgress potential checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from mi_reward.models.geoprogress_potential import GeoProgressPotential


class GeoProgressInferenceModel(nn.Module):
    """Frozen token-level potential used after LaWAM visual encoding.

    The caller supplies the current/next visual tokens, the task's successful
    goal tokens, and synchronized measured relation vectors. Raw Cosmos RGB is
    not part of online reward inference.
    """

    def __init__(self, model: GeoProgressPotential, config: dict[str, object], device: str | torch.device = "cuda"):
        super().__init__()
        self.model = model.eval()
        self.config = config
        requested_device = torch.device(device)
        self.device = requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, device: str | torch.device = "cuda") -> "GeoProgressInferenceModel":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint.get("config", {}))
        if config.get("model_class") != "GeoProgressPotential":
            raise ValueError("Checkpoint is not a GeoProgressPotential checkpoint.")
        model = GeoProgressPotential(
            visual_dim=int(config["visual_dim"]),
            relation_dim=int(config["relation_dim"]),
            hidden_dim=int(config.get("hidden_dim", 256)),
            architecture=str(config.get("architecture", "gru")),  # type: ignore[arg-type]
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device=device)

    @torch.inference_mode()
    def predict_potential(
        self,
        state_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        relations: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            state_tokens.to(self.device),
            goal_tokens.to(self.device),
            relations.to(self.device),
            None if state_mask is None else state_mask.to(self.device),
            None if goal_mask is None else goal_mask.to(self.device),
        )

    @torch.inference_mode()
    def deployment_reward(
        self,
        current_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_relation: torch.Tensor,
        next_relation: torch.Tensor,
        goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.compute_deployment_reward(
            current_tokens.to(self.device),
            next_tokens.to(self.device),
            goal_tokens.to(self.device),
            current_relation.to(self.device),
            next_relation.to(self.device),
            None if goal_mask is None else goal_mask.to(self.device),
            gamma=float(self.config.get("gamma", 0.99)),
        )
