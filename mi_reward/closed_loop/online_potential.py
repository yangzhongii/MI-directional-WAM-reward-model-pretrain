"""Stateful online adapter for ``VisualGoalPotential`` checkpoints."""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from mi_reward.inference.visual_goal_model import VisualGoalInferenceModel


@dataclasses.dataclass(frozen=True)
class RewardBreakdown:
    total: float
    sparse: float
    mi_delta: float
    weighted_mi: float
    observed_trend_delta: float
    baseline_trend_delta: float
    visual_trend_delta: float
    kinematic_progress: float
    kinematic_trend_delta: float
    directional_consensus: bool
    trend_delta: float
    trend_label: str
    trend_reward: float
    weighted_trend: float
    success_bonus: float
    shaping: float
    previous_potential: float
    next_potential: float


class _StreamingPotential:
    """Evaluate one observation at a time while preserving temporal context."""

    def __init__(
        self,
        inference: VisualGoalInferenceModel,
        goal_tokens: torch.Tensor | None,
        max_history: int = 64,
    ):
        self.inference = inference
        self.model = inference.model
        self.device = inference.device
        self.goal_tokens = None if goal_tokens is None else goal_tokens.float().to(self.device)
        if self.goal_tokens is not None:
            if self.goal_tokens.ndim == 2:
                self.goal_tokens = self.goal_tokens.unsqueeze(0)
            if self.goal_tokens.ndim != 3:
                raise ValueError(f"Goal tokens must have shape [K,D] or [1,K,D], got {tuple(self.goal_tokens.shape)}.")
        self.max_history = int(max_history)
        self.hidden: torch.Tensor | None = None
        self.history: deque[torch.Tensor] = deque(maxlen=self.max_history)

    def reset(self) -> None:
        self.hidden = None
        self.history.clear()

    def _validate(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.detach().float().to(self.device)
        if tokens.ndim != 2 or tokens.shape[-1] != self.model.visual_dim:
            raise ValueError(
                f"Online visual tokens must be [K,{self.model.visual_dim}], got {tuple(tokens.shape)}. "
                "Use the same LaWAM/DINO visual encoder that produced the reward training cache."
            )
        return tokens

    def _encode_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        state_tokens = tokens.unsqueeze(0).unsqueeze(0)
        state = self.model.visual_proj(state_tokens)
        memory, memory_mask = self.model._goal_memory(state_tokens, self.goal_tokens, None)
        query = state.reshape(1, tokens.shape[0], self.model.hidden_dim)
        attended, _ = self.model.goal_cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        fused = self.model.fuse_norm(query + attended)
        weights = torch.softmax(self.model.token_gate(fused).squeeze(-1), dim=-1)
        return (fused * weights.unsqueeze(-1)).sum(dim=1).unsqueeze(1)

    @torch.inference_mode()
    def score(self, tokens: torch.Tensor) -> float:
        tokens = self._validate(tokens)
        if self.model.architecture == "gru":
            frame = self._encode_frame(tokens)
            output, self.hidden = self.model.temporal(frame, self.hidden)
            value = self.model.potential_head(output[:, -1]).squeeze()
        elif self.model.architecture == "mlp":
            frame = self._encode_frame(tokens)
            value = self.model.potential_head(self.model.temporal(frame)[:, -1]).squeeze()
        else:
            self.history.append(tokens.cpu())
            sequence = torch.stack(list(self.history), dim=0).unsqueeze(0).to(self.device)
            value = self.model(sequence, self.goal_tokens)[:, -1].squeeze()
        return float(value.detach().cpu())

    @torch.inference_mode()
    def static_potentials(self, tokens: torch.Tensor, horizon: int) -> list[float]:
        """GRU/transformer drift for a no-motion clip used as a trend baseline."""
        tokens = self._validate(tokens)
        sequence = tokens.unsqueeze(0).unsqueeze(0).expand(1, int(horizon), -1, -1)
        goal = self.goal_tokens
        values = self.model(sequence, goal).squeeze(0)
        return [float(value) for value in values.detach().cpu()]


class _ReferenceKinematicProgress:
    """Online monotonic stage alignment against successful robot trajectories.

    This is the deployment counterpart of the proposal's kinematic process
    latent.  It does not declare task success and never exposes simulator
    object state.  It only asks whether the measured robot state is advancing
    along one of the successful demonstration paths.
    """

    def __init__(
        self,
        references: Sequence[np.ndarray | torch.Tensor],
        *,
        max_stage_step: int = 8,
        initial_match_fraction: float = 0.2,
        advance_margin: float = 1e-3,
        state_weights: Sequence[float] | None = None,
    ):
        arrays = [np.asarray(value, dtype=np.float32) for value in references]
        if not arrays:
            raise ValueError("At least one kinematic reference trajectory is required.")
        state_dim = int(arrays[0].shape[-1])
        if any(value.ndim != 2 or value.shape[0] < 2 or value.shape[1] != state_dim for value in arrays):
            raise ValueError("Kinematic references must all have shape [T,D], with T >= 2 and a common D.")
        if max_stage_step < 1:
            raise ValueError("kinematic_max_stage_step must be positive.")
        if not 0.0 < initial_match_fraction <= 1.0:
            raise ValueError("kinematic_initial_match_fraction must be in (0, 1].")
        if advance_margin < 0.0:
            raise ValueError("kinematic_advance_margin must be non-negative.")
        self.references = arrays
        stacked = np.concatenate(arrays, axis=0)
        # Robust per-coordinate scaling prevents joints (radians), Cartesian
        # position (metres), and gripper state from competing by raw units.
        q10, q90 = np.quantile(stacked, (0.1, 0.9), axis=0)
        self.scale = np.maximum((q90 - q10).astype(np.float32), 1e-3)
        if state_weights is None:
            weights = np.ones(state_dim, dtype=np.float32)
            if state_dim >= 6:
                weights[3:6] = 0.5  # axis-angle orientation is noisier than XYZ
            if state_dim > 8:
                weights[8:] = 0.25  # retain joints without dominating EEF motion
        else:
            weights = np.asarray(tuple(state_weights), dtype=np.float32)
            if weights.shape != (state_dim,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
                raise ValueError(f"kinematic_state_weights must contain {state_dim} non-negative values.")
        self.weights = weights / float(weights.sum())
        self.max_stage_step = int(max_stage_step)
        self.initial_match_fraction = float(initial_match_fraction)
        self.advance_margin = float(advance_margin)
        self.reference_index: int | None = None
        self.stage_index: int | None = None

    def _distances(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        residual = (reference - state[None, :]) / self.scale[None, :]
        return np.sum(np.square(residual) * self.weights[None, :], axis=1)

    def _state(self, state: np.ndarray | torch.Tensor) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32).reshape(-1)
        expected = self.references[0].shape[1]
        if value.shape != (expected,) or not np.isfinite(value).all():
            raise ValueError(f"Robot state must be a finite [{expected}] vector, got {value.shape}.")
        return value

    def reset(self, state: np.ndarray | torch.Tensor) -> float:
        value = self._state(state)
        best: tuple[float, int, int] | None = None
        for reference_index, reference in enumerate(self.references):
            limit = max(1, int(np.ceil(len(reference) * self.initial_match_fraction)))
            distances = self._distances(value, reference[:limit])
            stage = int(np.argmin(distances))
            candidate = (float(distances[stage]), reference_index, stage)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        _, self.reference_index, self.stage_index = best
        return self.progress

    @property
    def progress(self) -> float:
        if self.reference_index is None or self.stage_index is None:
            return 0.0
        length = len(self.references[self.reference_index])
        return float(self.stage_index / max(length - 1, 1))

    def score(self, state: np.ndarray | torch.Tensor) -> float:
        if self.reference_index is None or self.stage_index is None:
            raise RuntimeError("Kinematic progress must be reset before score().")
        value = self._state(state)
        reference = self.references[self.reference_index]
        start = self.stage_index
        stop = min(len(reference), start + self.max_stage_step + 1)
        distances = self._distances(value, reference[start:stop])
        offset = int(np.argmin(distances))
        # Staying at the current stage is always legal. Advancement requires a
        # real distance improvement, so repeated/static observations cannot
        # drift forward merely because nearby demonstration states are similar.
        if offset > 0 and float(distances[offset]) + self.advance_margin < float(distances[0]):
            self.stage_index += offset
        return self.progress


class PotentialReward:
    """Combine authoritative sparse success with MI potential shaping.

    Dense modes use the one-step potential difference.  Trend modes mirror the
    Qwen/RLinf VLM reward interface by converting progress over a short history
    window into ``positive``, ``negative`` or ``unclear``.  ``sparse_trend``
    additionally applies an authoritative environment-success bonus.
    """

    MODES = {"sparse", "mi", "sparse_mi", "trend", "sparse_trend"}

    def __init__(
        self,
        checkpoint: str,
        *,
        goal_tokens: torch.Tensor | None = None,
        mode: str = "sparse_mi",
        device: str = "cuda",
        gamma: float | None = None,
        mi_weight: float = 1.0,
        sparse_weight: float = 1.0,
        clip_delta: float | None = 1.0,
        max_history: int = 64,
        trend_history_size: int = 5,
        trend_input_interval: int = 1,
        trend_baseline_horizon: int = 16,
        trend_direction: float = 1.0,
        positive_threshold: float = 0.02,
        negative_threshold: float = -0.02,
        positive_reward: float = 1.0,
        negative_reward: float = -0.2,
        unclear_reward: float = 0.0,
        trend_weight: float = 1.0,
        success_bonus: float = 20.0,
        success_threshold: float = 0.5,
        robot_state_references: Sequence[np.ndarray | torch.Tensor] | None = None,
        kinematic_max_stage_step: int = 8,
        kinematic_initial_match_fraction: float = 0.2,
        kinematic_advance_margin: float = 1e-3,
        kinematic_positive_threshold: float = 0.01,
        directional_consensus_steps: int = 1,
        kinematic_state_weights: Sequence[float] | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown reward mode {mode!r}; expected one of {sorted(self.MODES)}.")
        self.mode = mode
        self.inference = VisualGoalInferenceModel.from_pretrained(checkpoint, device=device)
        self.scorer = _StreamingPotential(self.inference, goal_tokens, max_history=max_history)
        self.gamma = float(self.inference.config.get("gamma", 0.99) if gamma is None else gamma)
        self.mi_weight = float(mi_weight)
        self.sparse_weight = float(sparse_weight)
        self.clip_delta = None if clip_delta is None else float(clip_delta)
        self.trend_history_size = int(trend_history_size)
        self.trend_input_interval = int(trend_input_interval)
        self.trend_baseline_horizon = int(trend_baseline_horizon)
        self.trend_direction = float(trend_direction)
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        self.positive_reward = float(positive_reward)
        self.negative_reward = float(negative_reward)
        self.unclear_reward = float(unclear_reward)
        self.trend_weight = float(trend_weight)
        self.success_bonus_value = float(success_bonus)
        self.success_threshold = float(success_threshold)
        self.kinematic_positive_threshold = float(kinematic_positive_threshold)
        self.directional_consensus_steps = int(directional_consensus_steps)
        if self.trend_history_size < 2:
            raise ValueError("trend_history_size must be at least 2.")
        if self.trend_input_interval < 1:
            raise ValueError("trend_input_interval must be at least 1.")
        if self.trend_baseline_horizon < self.trend_history_size:
            raise ValueError("trend_baseline_horizon must be at least trend_history_size.")
        if self.trend_direction == 0.0:
            raise ValueError("trend_direction must be non-zero.")
        if self.negative_threshold >= self.positive_threshold:
            raise ValueError("negative_threshold must be smaller than positive_threshold.")
        if self.kinematic_positive_threshold < 0.0:
            raise ValueError("kinematic_positive_threshold must be non-negative.")
        if self.directional_consensus_steps < 1:
            raise ValueError("directional_consensus_steps must be positive.")
        self.kinematic = (
            None
            if robot_state_references is None or len(robot_state_references) == 0
            else _ReferenceKinematicProgress(
                robot_state_references,
                max_stage_step=kinematic_max_stage_step,
                initial_match_fraction=kinematic_initial_match_fraction,
                advance_margin=kinematic_advance_margin,
                state_weights=kinematic_state_weights,
            )
        )
        self.previous_potential: float | None = None
        self.potential_history: deque[float] = deque(maxlen=self.trend_history_size)
        self.kinematic_history: deque[float] = deque(maxlen=self.trend_history_size)
        self.consensus_history: deque[float] = deque(maxlen=self.directional_consensus_steps)
        self.static_baseline: list[float] = []
        self.trend_step = 0

    def reset(
        self,
        initial_tokens: torch.Tensor,
        robot_state: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        self.scorer.reset()
        self.previous_potential = self.scorer.score(initial_tokens)
        self.static_baseline = self.scorer.static_potentials(
            initial_tokens, self.trend_baseline_horizon
        )
        self.potential_history.clear()
        self.potential_history.append(self.previous_potential)
        self.kinematic_history.clear()
        if self.kinematic is not None:
            if robot_state is None:
                raise ValueError("robot_state is required when robot-state references are configured.")
            self.kinematic_history.append(self.kinematic.reset(robot_state))
        self.consensus_history.clear()
        self.trend_step = 0
        return self.previous_potential

    def _trend(
        self,
        next_potential: float,
        robot_state: np.ndarray | torch.Tensor | None,
    ) -> tuple[float, float, float, float, float, bool, float, str, float]:
        self.trend_step += 1
        self.potential_history.append(float(next_potential))
        kinematic_progress = 0.0
        if self.kinematic is not None:
            if robot_state is None:
                raise ValueError("robot_state is required when robot-state references are configured.")
            kinematic_progress = self.kinematic.score(robot_state)
            self.kinematic_history.append(kinematic_progress)
        if (
            len(self.potential_history) < self.trend_history_size
            or self.trend_step % self.trend_input_interval != 0
        ):
            return 0.0, 0.0, 0.0, kinematic_progress, 0.0, False, 0.0, "unclear", self.unclear_reward

        observed_delta = float(next_potential - self.potential_history[0])
        baseline_end = min(self.trend_step, len(self.static_baseline) - 1)
        baseline_start = max(0, baseline_end - (self.trend_history_size - 1))
        baseline_delta = self.static_baseline[baseline_end] - self.static_baseline[baseline_start]
        # A recurrent model can increase its output solely because hidden state
        # is warming up. Subtract the same model's no-motion drift so a static
        # clip maps to unclear instead of receiving a false positive reward.
        visual_delta = float(self.trend_direction * (observed_delta - baseline_delta))
        kinematic_delta = 0.0
        consensus = True
        if self.kinematic is not None:
            kinematic_delta = float(kinematic_progress - self.kinematic_history[0])
            consensus = kinematic_delta >= self.kinematic_positive_threshold

        # Positive visual movement is accepted only when measured robot state
        # advances along a successful reference. Visual regression remains a
        # valid negative signal. This preserves potential shaping while making
        # the positive direction robot-grounded rather than appearance-only.
        joint_evidence = visual_delta if consensus or visual_delta <= 0.0 else 0.0
        self.consensus_history.append(joint_evidence)
        if len(self.consensus_history) < self.directional_consensus_steps:
            trend_delta = 0.0
        elif all(value > 0.0 for value in self.consensus_history):
            trend_delta = float(min(self.consensus_history))
        elif all(value < 0.0 for value in self.consensus_history):
            trend_delta = float(max(self.consensus_history))
        else:
            trend_delta = 0.0
        if trend_delta >= self.positive_threshold:
            return observed_delta, baseline_delta, visual_delta, kinematic_progress, kinematic_delta, consensus, trend_delta, "positive", self.positive_reward
        if trend_delta <= self.negative_threshold:
            return observed_delta, baseline_delta, visual_delta, kinematic_progress, kinematic_delta, consensus, trend_delta, "negative", self.negative_reward
        return observed_delta, baseline_delta, visual_delta, kinematic_progress, kinematic_delta, consensus, trend_delta, "unclear", self.unclear_reward

    def step(
        self,
        next_tokens: torch.Tensor,
        sparse_reward: float,
        robot_state: np.ndarray | torch.Tensor | None = None,
    ) -> RewardBreakdown:
        if self.previous_potential is None:
            raise RuntimeError("PotentialReward.reset(initial_tokens) must be called before step().")
        previous = self.previous_potential
        nxt = self.scorer.score(next_tokens)
        delta = self.gamma * nxt - previous
        self.previous_potential = nxt
        if self.clip_delta is not None:
            delta = max(-self.clip_delta, min(self.clip_delta, delta))
        (
            observed_trend_delta,
            baseline_trend_delta,
            visual_trend_delta,
            kinematic_progress,
            kinematic_trend_delta,
            directional_consensus,
            trend_delta,
            trend_label,
            trend_reward,
        ) = self._trend(nxt, robot_state)
        weighted_mi = self.mi_weight * delta
        weighted_trend = self.trend_weight * trend_reward
        weighted_sparse = self.sparse_weight * float(sparse_reward)
        success_bonus = (
            self.success_bonus_value
            if float(sparse_reward) >= self.success_threshold
            else 0.0
        )
        shaping = weighted_trend if self.mode in {"trend", "sparse_trend"} else weighted_mi
        total = {
            "sparse": weighted_sparse,
            "mi": weighted_mi,
            "sparse_mi": weighted_sparse + weighted_mi,
            "trend": weighted_trend,
            "sparse_trend": success_bonus + weighted_trend,
        }[self.mode]
        return RewardBreakdown(
            total=float(total),
            sparse=float(sparse_reward),
            mi_delta=float(delta),
            weighted_mi=float(weighted_mi),
            observed_trend_delta=float(observed_trend_delta),
            baseline_trend_delta=float(baseline_trend_delta),
            visual_trend_delta=float(visual_trend_delta),
            kinematic_progress=float(kinematic_progress),
            kinematic_trend_delta=float(kinematic_trend_delta),
            directional_consensus=bool(directional_consensus),
            trend_delta=float(trend_delta),
            trend_label=trend_label,
            trend_reward=float(trend_reward),
            weighted_trend=float(weighted_trend),
            success_bonus=float(success_bonus),
            shaping=float(shaping),
            previous_potential=float(previous),
            next_potential=float(nxt),
        )
