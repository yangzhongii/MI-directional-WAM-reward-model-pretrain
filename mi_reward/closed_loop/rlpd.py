"""Self-contained visual RLPD components for LIBERO.

The implementation mirrors the sample-efficient parts of the reviewed real-
world setup without depending on RLinf: a pretrained CNN policy encoder,
balanced online/demo replay, an ensemble of Q functions, and high-UTD critic
updates. Reward inference remains external and frozen.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


ArrayBatch = dict[str, np.ndarray]
TensorBatch = dict[str, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class RLPDConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    visual_dim: int = 256
    state_latent_dim: int = 64
    spatial_features: int = 4
    num_q_heads: int = 10
    critic_subsample_size: int = 2
    gamma: float = 0.96
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_temperature: float = 0.01
    target_entropy: float | None = None
    grad_clip_norm: float = 10.0
    freeze_backbone: bool = True
    encoder_checkpoint: str | None = None


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class PixelReplayBuffer:
    """CPU uint8 replay for synchronized camera views and robot state."""

    def __init__(
        self,
        capacity: int,
        image_shape: tuple[int, int, int, int],
        state_dim: int,
        action_dim: int,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 4 or self.image_shape[-1] != 3:
            raise ValueError(f"Images must be [V,H,W,3], got {self.image_shape}.")
        self.images = np.empty((capacity, *self.image_shape), dtype=np.uint8)
        self.next_images = np.empty_like(self.images)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_done = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        images: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_images: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.images[index] = np.asarray(images, dtype=np.uint8)
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.next_images[index] = np.asarray(next_images, dtype=np.uint8)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.not_done[index, 0] = 0.0 if done else 1.0
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ArrayBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} transitions, fewer than {batch_size}.")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "images": self.images[indices],
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_images": self.next_images[indices],
            "next_states": self.next_states[indices],
            "not_done": self.not_done[indices],
        }


def balanced_batch(
    online: PixelReplayBuffer,
    demonstrations: PixelReplayBuffer,
    batch_size: int,
) -> ArrayBatch:
    """Sample the canonical RLPD 50/50 online/offline batch."""

    online_size = int(batch_size) // 2
    demo_size = int(batch_size) - online_size
    online_batch = online.sample(online_size)
    demo_batch = demonstrations.sample(demo_size)
    return {
        key: np.concatenate((online_batch[key], demo_batch[key]), axis=0)
        for key in online_batch
    }


class SpatialLearnedPooling(nn.Module):
    def __init__(self, channels: int, height: int, width: int, features: int):
        super().__init__()
        scale = 1.0 / math.sqrt(max(height * width, 1))
        self.kernel = nn.Parameter(torch.randn(channels, height, width, features) * scale)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        weighted = feature_map.unsqueeze(-1) * self.kernel.unsqueeze(0)
        return weighted.sum(dim=(2, 3)).flatten(start_dim=1)


class ResNetViewEncoder(nn.Module):
    def __init__(self, image_hw: tuple[int, int], config: RLPDConfig):
        super().__init__()
        model = resnet18(weights=None)
        if config.encoder_checkpoint:
            checkpoint = Path(config.encoder_checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"CNN encoder checkpoint is missing: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.freeze_backbone = bool(config.freeze_backbone)
        # Shape probing must not update BatchNorm statistics (and a one-element
        # probe at small resolutions is invalid while BatchNorm is training).
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            sample = self.backbone(torch.zeros(1, 3, *image_hw))
        if was_training and not self.freeze_backbone:
            self.backbone.train()
        _, channels, height, width = sample.shape
        self.pool = SpatialLearnedPooling(channels, height, width, config.spatial_features)
        self.projection = nn.Sequential(
            nn.Linear(channels * config.spatial_features, config.visual_dim),
            nn.LayerNorm(config.visual_dim),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return self.projection(self.pool(features))


class MultiViewStateEncoder(nn.Module):
    def __init__(self, num_views: int, image_hw: tuple[int, int], state_dim: int, config: RLPDConfig):
        super().__init__()
        self.views = nn.ModuleList(ResNetViewEncoder(image_hw, config) for _ in range(num_views))
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, config.state_latent_dim),
            nn.LayerNorm(config.state_latent_dim),
            nn.Tanh(),
        )
        self.output_dim = num_views * config.visual_dim + config.state_latent_dim

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != len(self.views):
            raise ValueError(f"Expected images [B,{len(self.views)},3,H,W], got {tuple(images.shape)}.")
        visual = [encoder(images[:, index]) for index, encoder in enumerate(self.views)]
        return torch.cat((*visual, self.state_projection(states)), dim=-1)


class GaussianActor(nn.Module):
    def __init__(self, feature_dim: int, action_low: np.ndarray, action_high: np.ndarray, config: RLPDConfig):
        super().__init__()
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.action_dim = int(low.size)
        self.network = _mlp(feature_dim, config.hidden_dims, self.action_dim * 2)
        self.register_buffer("action_scale", torch.from_numpy((high - low) / 2.0))
        self.register_buffer("action_bias", torch.from_numpy((high + low) / 2.0))

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(features).chunk(2, dim=-1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5.0, 2.0).exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        if deterministic:
            return action, torch.zeros((*action.shape[:-1], 1), device=action.device)
        correction = torch.log(1.0 - squashed.square() + 1e-6)
        return action, (distribution.log_prob(raw) - correction).sum(dim=-1, keepdim=True)


class QEnsemble(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, config: RLPDConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            _mlp(feature_dim + action_dim, config.hidden_dims, 1)
            for _ in range(config.num_q_heads)
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((features, actions), dim=-1)
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


class RLPDAgent:
    def __init__(
        self,
        image_shape: tuple[int, int, int, int],
        state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: RLPDConfig,
        device: str | torch.device = "cuda",
    ):
        requested = torch.device(device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.image_shape = tuple(image_shape)
        self.state_dim = int(state_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        num_views, height, width, channels = self.image_shape
        if channels != 3:
            raise ValueError(f"Expected RGB images, got {self.image_shape}.")
        self.encoder = MultiViewStateEncoder(num_views, (height, width), state_dim, config).to(self.device)
        self.actor = GaussianActor(self.encoder.output_dim, self.action_low, self.action_high, config).to(self.device)
        self.critic = QEnsemble(self.encoder.output_dim, self.action_low.size, config).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device).eval()
        self.target_critic = copy.deepcopy(self.critic).to(self.device).eval()
        for module in (self.target_encoder, self.target_critic):
            for parameter in module.parameters():
                parameter.requires_grad = False
        critic_parameters = list(self.encoder.parameters()) + list(self.critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(config.init_temperature), dtype=torch.float32, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = float(config.target_entropy or -self.action_low.size)
        self.update_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _tensor_batch(self, batch: ArrayBatch) -> TensorBatch:
        output = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
        for key in ("images", "next_images"):
            output[key] = output[key].float().permute(0, 1, 4, 2, 3).div_(255.0)
            mean = output[key].new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
            std = output[key].new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
            output[key] = (output[key] - mean) / std
        return output

    def _single(self, images: np.ndarray, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._tensor_batch({
            "images": np.asarray(images, dtype=np.uint8)[None],
            "next_images": np.asarray(images, dtype=np.uint8)[None],
            "states": np.asarray(state, dtype=np.float32)[None],
        })
        return batch["images"], batch["states"]

    @torch.inference_mode()
    def act(self, images: np.ndarray, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        image_tensor, state_tensor = self._single(images, state)
        features = self.encoder(image_tensor, state_tensor)
        action, _ = self.actor.sample(features, deterministic=deterministic)
        return action[0].cpu().numpy()

    @torch.inference_mode()
    def policy_entropy(self, images: np.ndarray, states: np.ndarray) -> float:
        """Monte-Carlo entropy proxy used only to pace synthetic replay."""

        batch = self._tensor_batch({
            "images": np.asarray(images, dtype=np.uint8),
            "next_images": np.asarray(images, dtype=np.uint8),
            "states": np.asarray(states, dtype=np.float32),
        })
        features = self.encoder(batch["images"], batch["states"])
        _, log_probability = self.actor.sample(features)
        return float((-log_probability).mean().cpu())

    @torch.inference_mode()
    def mean_dataset_q(self, raw_batch: ArrayBatch) -> float:
        batch = self._tensor_batch(raw_batch)
        features = self.encoder(batch["images"], batch["states"])
        return float(self.critic(features, batch["actions"]).mean().cpu())

    def update(self, raw_batch: ArrayBatch, *, update_actor: bool = True) -> dict[str, float]:
        batch = self._tensor_batch(raw_batch)
        features = self.encoder(batch["images"], batch["states"])
        with torch.no_grad():
            next_actor_features = self.encoder(batch["next_images"], batch["next_states"])
            next_actions, next_log_prob = self.actor.sample(next_actor_features)
            target_features = self.target_encoder(batch["next_images"], batch["next_states"])
            target_all = self.target_critic(target_features, next_actions)
            count = min(self.config.critic_subsample_size, target_all.shape[-1])
            indices = torch.randperm(target_all.shape[-1], device=self.device)[:count]
            target_q = target_all.index_select(-1, indices).min(dim=-1, keepdim=True).values
            target = batch["rewards"] + self.config.gamma * batch["not_done"] * (
                target_q - self.alpha.detach() * next_log_prob
            )
        q_values = self.critic(features, batch["actions"])
        critic_loss = F.mse_loss(q_values, target.expand_as(q_values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.critic.parameters()), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "q": float(q_values.mean().detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }
        if update_actor:
            actor_features = self.encoder(batch["images"], batch["states"]).detach()
            for parameter in self.critic.parameters():
                parameter.requires_grad = False
            actions, log_prob = self.actor.sample(actor_features)
            policy_q = self.critic(actor_features, actions).mean(dim=-1, keepdim=True)
            actor_loss = (self.alpha.detach() * log_prob - policy_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad = True
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            metrics.update(actor_loss=float(actor_loss.detach().cpu()), alpha_loss=float(alpha_loss.detach().cpu()))

        with torch.no_grad():
            for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target.lerp_(source, self.config.tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(source, self.config.tau)
        self.update_step += 1
        return metrics

    def save(self, path: str | Path, *, step: int, metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": int(step), "metadata": metadata or {}, "config": dataclasses.asdict(self.config),
            "image_shape": self.image_shape, "state_dim": self.state_dim,
            "action_low": torch.from_numpy(self.action_low), "action_high": torch.from_numpy(self.action_high),
            "encoder": self.encoder.state_dict(), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_encoder": self.target_encoder.state_dict(), "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), "update_step": self.update_step,
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }, destination)
        return destination
