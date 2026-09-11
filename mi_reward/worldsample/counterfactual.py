"""Local counterfactual action generation around grounded robot trajectories.

The perturbation follows the WorldSample data construction principle: synthetic
videos are conditioned on nearby, executable actions instead of merely changing
the diffusion seed for one fixed action sequence.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class CounterfactualConfig:
    scale_jitter: float = 0.20
    noise_std: float = 0.05
    inactive_epsilon: float = 1e-6
    preserve_gripper: bool = True
    gripper_index: int = -1

    def validate(self) -> None:
        if self.scale_jitter < 0 or self.noise_std < 0 or self.inactive_epsilon < 0:
            raise ValueError("Counterfactual perturbation magnitudes must be non-negative.")


def perturb_action_sequence(
    actions: np.ndarray,
    *,
    seed: int,
    config: CounterfactualConfig,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Perturb a trajectory while preserving its local task intent.

    A single scale is sampled per action dimension and held fixed through time;
    this produces coherent directional alternatives. Gaussian noise is only
    applied to dimensions that are active in the grounded sequence. Discrete
    gripper commands are retained by default.
    """

    config.validate()
    source = np.asarray(actions, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] < 1:
        raise ValueError(f"Expected actions [T,A], got {source.shape}.")
    rng = np.random.default_rng(int(seed))
    active = np.max(np.abs(source), axis=0) > float(config.inactive_epsilon)
    scale = rng.uniform(1.0 - config.scale_jitter, 1.0 + config.scale_jitter, source.shape[1]).astype(np.float32)
    noise = rng.normal(0.0, config.noise_std, source.shape).astype(np.float32)
    noise *= active[None]
    output = source * scale[None] + noise

    gripper_index = int(config.gripper_index) % source.shape[1]
    if config.preserve_gripper:
        output[:, gripper_index] = source[:, gripper_index]
        scale[gripper_index] = 1.0
        noise[:, gripper_index] = 0.0

    if action_low is not None or action_high is not None:
        if action_low is None or action_high is None:
            raise ValueError("action_low and action_high must be provided together.")
        low = np.broadcast_to(np.asarray(action_low, dtype=np.float32), (source.shape[1],))
        high = np.broadcast_to(np.asarray(action_high, dtype=np.float32), (source.shape[1],))
        if np.any(low >= high):
            raise ValueError("Each action_low value must be smaller than action_high.")
        output = np.clip(output, low, high)

    metadata: dict[str, object] = {
        "method": "worldsample_local_scale_and_noise_v1",
        "seed": int(seed),
        "scale_jitter": float(config.scale_jitter),
        "noise_std": float(config.noise_std),
        "inactive_dimensions": np.flatnonzero(~active).tolist(),
        "preserve_gripper": bool(config.preserve_gripper),
        "gripper_index": gripper_index,
        "mean_absolute_delta": float(np.abs(output - source).mean()),
        "max_absolute_delta": float(np.abs(output - source).max()),
    }
    return output.astype(np.float32, copy=False), metadata
