"""Potential-difference reward shaping for EnvWorker.

DROP-IN LOCATION: rlinf/workers/env/potential_shaping_state.py

Manages per-environment potential caches and computes confidence-gated
potential-difference rewards. Designed for asynchronous multi-env RL.

The EnvWorker owns transition timing, done signals, env IDs, and reset
state. This module provides the cache and delta computation.

INTEGRATION: Add to EnvWorker:
    from rlinf.workers.env.potential_shaping_state import PotentialShapingState

    self._pot_state = PotentialShapingState(config)

    # After env reset:
    potentials = reward_model(obs0)
    self._pot_state.initialize(env_ids, potentials)

    # After step:
    next_potentials = reward_model(obs_next)
    deltas = self._pot_state.compute_delta(env_ids, next_potentials, gamma)
    final_rewards[env_ids] += potential_scale * deltas

    # On done:
    self._pot_state.clear(env_ids)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class PotentialShapingConfig:
    """Configuration for potential-based reward shaping."""

    enabled: bool = True
    gamma: float = 0.99
    scale: float = 1.0
    delta_clip: float = 1.0
    confidence_gate: bool = True
    confidence_combine: str = "min"  # "min" or "geometric_mean"
    min_confidence: float = 0.0
    invalid_behavior: str = "zero_shaping"  # "zero_shaping" or "skip"
    normalize: bool = False
    warmup_steps: int = 0
    use_for_termination: bool = False
    use_for_success: bool = False


class PotentialShapingState:
    """Per-environment potential cache for potential-difference reward.

    Implements the full lifecycle:
      A. After reset: cache V_phi(o_0, g)
      B. After step: compute gamma*V(o_{t+1}) - V(o_t)
      C. Cache V(o_{t+1}) for next transition
      D. On done: clear all state
    """

    def __init__(self, config: PotentialShapingConfig, max_envs: int = 128):
        self.config = config
        self.max_envs = max_envs

        # Per-env caches
        self._potentials: dict[int, float] = {}
        self._confidences: dict[int, float] = {}
        self._valid: dict[int, bool] = {}

        # Warmup counter (shared across envs)
        self._global_step = 0

    def initialize(
        self,
        env_ids: list[int],
        potentials: torch.Tensor,
        confidences: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> None:
        """Cache initial potential after environment reset.

        Called immediately after env.reset() produces o_0.

        Args:
            env_ids: list of environment indices
            potentials: [B] scalar potentials from reward model
            confidences: [B] optional confidence values
            valid: [B] optional validity flags
        """
        for i, env_id in enumerate(env_ids):
            pot = float(potentials[i].item())
            if not self._is_finite(pot):
                logger.warning("Non-finite initial potential for env %d: %.3f", env_id, pot)
                pot = 0.0

            self._potentials[env_id] = pot
            self._confidences[env_id] = float(confidences[i].item()) if confidences is not None else 1.0
            self._valid[env_id] = bool(valid[i].item()) if valid is not None else True

    def compute_delta(
        self,
        env_ids: list[int],
        next_potentials: torch.Tensor,
        next_confidences: torch.Tensor | None = None,
        next_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute potential-difference rewards for a batch of environments.

        delta[i] = gamma * V(o_{t+1}, g) - cached_V(o_t, g)
        If confidence_gate: delta[i] *= confidence[i]

        Args:
            env_ids: list of environment indices
            next_potentials: [B] V(o_{t+1}) from reward model
            next_confidences: [B] confidence values
            next_valid: [B] validity flags

        Returns:
            deltas: [B] potential-difference reward per env
        """
        self._global_step += 1
        B = len(env_ids)
        deltas = torch.zeros(B, device=next_potentials.device, dtype=torch.float32)

        for i, env_id in enumerate(env_ids):
            current_pot = self._potentials.get(env_id, 0.0)
            current_conf = self._confidences.get(env_id, 1.0)
            current_valid = self._valid.get(env_id, True)

            next_pot = float(next_potentials[i].item())
            next_conf = float(next_confidences[i].item()) if next_confidences is not None else 1.0
            next_v = bool(next_valid[i].item()) if next_valid is not None else True

            # Validate next potential
            if not self._is_finite(next_pot) or not next_v:
                if self.config.invalid_behavior == "zero_shaping":
                    deltas[i] = 0.0
                # Still cache the old potential (don't update)
                logger.warning(
                    "Invalid potential for env %d at step %d: %.3f",
                    env_id, self._global_step, next_pot,
                )
                continue

            # Enforce valid previous state
            if not current_valid:
                # Initialize from this observation
                self._potentials[env_id] = next_pot
                self._confidences[env_id] = next_conf
                self._valid[env_id] = True
                deltas[i] = 0.0
                continue

            # Warmup: skip potential shaping for first N steps
            if self._global_step <= self.config.warmup_steps:
                self._potentials[env_id] = next_pot
                self._confidences[env_id] = next_conf
                deltas[i] = 0.0
                continue

            # Compute raw delta
            raw_delta = self.config.gamma * next_pot - current_pot

            # Clip
            clip_val = self.config.delta_clip
            clipped = torch.clamp(torch.tensor(raw_delta), -clip_val, clip_val).item()

            # Confidence gating
            if self.config.confidence_gate:
                if self.config.confidence_combine == "min":
                    conf = min(current_conf, next_conf)
                elif self.config.confidence_combine == "geometric_mean":
                    conf = (current_conf * next_conf) ** 0.5
                else:
                    conf = min(current_conf, next_conf)

                if conf < self.config.min_confidence:
                    conf = 0.0
                clipped *= conf

            # Scale
            deltas[i] = self.config.scale * clipped

            # Update cache for next step
            self._potentials[env_id] = next_pot
            self._confidences[env_id] = next_conf
            self._valid[env_id] = True

        return deltas

    def clear(self, env_ids: list[int]) -> None:
        """Clear cached state for done/reset environments.

        Must be called on episode termination or reset BEFORE initialize().
        """
        for env_id in env_ids:
            self._potentials.pop(env_id, None)
            self._confidences.pop(env_id, None)
            self._valid.pop(env_id, None)

    def get_cached_potential(self, env_id: int) -> float:
        """Get the currently cached potential for an env (for logging)."""
        return self._potentials.get(env_id, 0.0)

    def get_cached_confidence(self, env_id: int) -> float:
        """Get the currently cached confidence for an env (for logging)."""
        return self._confidences.get(env_id, 0.0)

    @staticmethod
    def _is_finite(value: float) -> bool:
        import math
        return math.isfinite(value) and abs(value) < 1e8


# ---------------------------------------------------------------------------
# Integration pseudocode for EnvWorker
# ---------------------------------------------------------------------------
#
# In EnvWorker.__init__:
#   self._pot_state = PotentialShapingState(potential_config)
#
# In EnvWorker._handle_reset(env_ids):
#   obs0 = env.reset()
#   if use_potential_shaping:
#       pot_result = reward_worker.predict(obs0, tasks)
#       self._pot_state.clear(env_ids)  # clear old episode state
#       self._pot_state.initialize(env_ids,
#           pot_result["potential"],
#           pot_result["confidence"],
#           pot_result["valid"])
#
# In EnvWorker._handle_step(env_ids, actions):
#   obs_next, env_reward, done, info = env.step(actions)
#
#   if use_potential_shaping:
#       pot_result = reward_worker.predict(obs_next, tasks)
#       mi_deltas = self._pot_state.compute_delta(
#           env_ids,
#           pot_result["potential"],
#           pot_result["confidence"],
#           pot_result["valid"],
#       )
#       final_reward = env_reward_weight * env_reward + reward_weight * mi_deltas
#   else:
#       final_reward = env_reward
#
#   # Store components in transition info for logging:
#   info["mi_potential_current"] = self._pot_state.get_cached_potential(env_id)
#   info["mi_potential_next"] = pot_result["potential"][i]
#   info["mi_delta_raw"] = raw_delta
#   info["mi_delta_clipped"] = clipped
#   info["mi_confidence"] = conf
#   info["mi_potential_reward"] = mi_deltas[i]
#   info["env_reward_raw"] = env_reward
#   info["final_reward"] = final_reward
#
#   # On done:
#   for env_id in done_env_ids:
#       self._pot_state.clear([env_id])
# ---------------------------------------------------------------------------
