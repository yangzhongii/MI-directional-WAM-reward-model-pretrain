"""Small, dependency-light Soft Actor-Critic implementation for LIBERO.

The policy is intentionally not a contribution of this repository.  It is a
standard continuous-control SAC baseline used to measure whether a frozen MI
potential improves sample efficiency over LIBERO's sparse success reward.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    init_temperature: float = 0.1
    target_entropy: float | None = None
    grad_clip_norm: float = 10.0


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dims: tuple[int, ...]):
        super().__init__()
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if action_low.shape != action_high.shape or np.any(action_high <= action_low):
            raise ValueError("Action bounds must have equal shapes and high > low.")
        self.action_dim = int(action_low.size)
        self.network = _mlp(observation_dim, hidden_dims, self.action_dim * 2)
        self.register_buffer("action_scale", torch.from_numpy((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.from_numpy((action_high + action_low) / 2.0))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(observation).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(raw)
        action = squashed * self.action_scale + self.action_bias
        if deterministic:
            return action, torch.zeros((*action.shape[:-1], 1), device=action.device)
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - squashed.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.q1 = _mlp(observation_dim + action_dim, hidden_dims, 1)
        self.q2 = _mlp(observation_dim + action_dim, hidden_dims, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([observation, action], dim=-1)
        return self.q1(inputs), self.q2(inputs)


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_done = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, next_observation: np.ndarray, done: bool) -> None:
        idx = self.position
        self.observations[idx] = observation
        self.actions[idx] = action
        self.rewards[idx, 0] = float(reward)
        self.next_observations[idx] = next_observation
        self.not_done[idx, 0] = 0.0 if done else 1.0
        self.position = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError(f"Replay contains {self.size} transitions, fewer than batch size {batch_size}.")
        idx = self.rng.integers(0, self.size, size=batch_size)
        arrays = (
            self.observations[idx], self.actions[idx], self.rewards[idx],
            self.next_observations[idx], self.not_done[idx],
        )
        return tuple(torch.as_tensor(value, device=device) for value in arrays)


class SACAgent:
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: SACConfig,
        device: str | torch.device = "cuda",
    ):
        requested = torch.device(device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.observation_dim = int(observation_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.actor = GaussianActor(observation_dim, self.action_low, self.action_high, config.hidden_dims).to(self.device)
        self.critic = TwinCritic(observation_dim, self.action_low.size, config.hidden_dims).to(self.device)
        self.target_critic = TwinCritic(observation_dim, self.action_low.size, config.hidden_dims).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False
        self.log_alpha = nn.Parameter(torch.tensor(np.log(config.init_temperature), device=self.device, dtype=torch.float32))
        self.target_entropy = float(config.target_entropy if config.target_entropy is not None else -self.action_low.size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.inference_mode()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(tensor, deterministic=deterministic)
        return action[0].cpu().numpy()

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        observation, action, reward, next_observation, not_done = batch
        cfg = self.config
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_observation)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = reward + cfg.gamma * not_done * target_value

        q1, q2 = self.critic(observation, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_optimizer.step()

        sampled_action, log_prob = self.actor.sample(observation)
        actor_q1, actor_q2 = self.critic(observation, sampled_action)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q": float(torch.minimum(q1, q2).mean().detach().cpu()),
        }

    def save(self, path: str | Path, *, step: int, metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": int(step), "metadata": metadata or {},
                "observation_dim": self.observation_dim,
                "action_low": torch.from_numpy(self.action_low.copy()),
                "action_high": torch.from_numpy(self.action_high.copy()),
                "config": dataclasses.asdict(self.config),
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cuda") -> tuple["SACAgent", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu")
        config_raw = dict(payload["config"])
        config_raw["hidden_dims"] = tuple(config_raw["hidden_dims"])
        action_low = torch.as_tensor(payload["action_low"]).cpu().numpy()
        action_high = torch.as_tensor(payload["action_high"]).cpu().numpy()
        agent = cls(payload["observation_dim"], action_low, action_high, SACConfig(**config_raw), device)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        return agent, payload
