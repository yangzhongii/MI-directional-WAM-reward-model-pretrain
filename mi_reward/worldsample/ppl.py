"""Paced synthetic replay for the LIBERO RLPD loop.

Synthetic transitions are opt-in and must include synchronized policy images,
robot state, action, next state, and a three-way MI label. This deliberately
rejects image-only hallucinations: RLPD must never train on a visual future
paired with an unrelated proprioceptive state.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from mi_reward.closed_loop.rlpd import ArrayBatch, PixelReplayBuffer


LABEL_NEGATIVE = -1
LABEL_UNCLEAR = 0
LABEL_POSITIVE = 1


@dataclasses.dataclass(frozen=True)
class PPLConfig:
    max_synthetic_ratio: float = 0.30
    min_synthetic_ratio: float = 0.05
    entropy_reference: float = 4.0
    q_shift_tolerance: float = 2.0

    def ratio(self, policy_entropy: float) -> float:
        if not 0 <= self.min_synthetic_ratio <= self.max_synthetic_ratio < 0.5:
            raise ValueError("PPL ratios must satisfy 0 <= min <= max < 0.5.")
        reference = max(float(self.entropy_reference), 1e-6)
        fraction = float(np.clip(policy_entropy / reference, 0.0, 1.0))
        return self.min_synthetic_ratio + fraction * (
            self.max_synthetic_ratio - self.min_synthetic_ratio
        )


class LabeledPixelReplayBuffer(PixelReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = np.empty((self.capacity, 1), dtype=np.int8)

    def add_labeled(self, *args, label: int, **kwargs) -> None:
        if int(label) not in (LABEL_NEGATIVE, LABEL_UNCLEAR, LABEL_POSITIVE):
            raise ValueError(f"Unknown three-state MI label: {label}")
        index = self.position
        super().add(*args, **kwargs)
        self.labels[index, 0] = int(label)

    def sample_balanced_clear(self, batch_size: int) -> ArrayBatch:
        positive = np.flatnonzero(self.labels[: self.size, 0] == LABEL_POSITIVE)
        negative = np.flatnonzero(self.labels[: self.size, 0] == LABEL_NEGATIVE)
        if not len(positive) or not len(negative):
            raise ValueError("Synthetic replay needs at least one positive and one negative transition.")
        positive_count = int(batch_size) // 2
        negative_count = int(batch_size) - positive_count
        indices = np.concatenate(
            (
                self.rng.choice(positive, size=positive_count, replace=len(positive) < positive_count),
                self.rng.choice(negative, size=negative_count, replace=len(negative) < negative_count),
            )
        )
        self.rng.shuffle(indices)
        batch = {
            "images": self.images[indices],
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_images": self.next_images[indices],
            "next_states": self.next_states[indices],
            "not_done": self.not_done[indices],
            "mi_labels": self.labels[indices],
        }
        return batch


def load_synthetic_replay(
    path: str | Path,
    *,
    image_shape: tuple[int, int, int, int],
    state_dim: int,
    action_dim: int,
    seed: int,
) -> LabeledPixelReplayBuffer:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PPL synthetic replay is missing: {source}")
    required = (
        "images", "states", "actions", "rewards", "next_images",
        "next_states", "not_done", "mi_labels",
    )
    with np.load(source, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Synthetic replay is missing arrays: {missing}")
        arrays = {key: np.asarray(payload[key]) for key in required}
    count = len(arrays["actions"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("All synthetic replay arrays must have the same leading dimension.")
    expected = {
        "images": tuple(image_shape),
        "states": (state_dim,),
        "actions": (action_dim,),
        "next_images": tuple(image_shape),
        "next_states": (state_dim,),
    }
    for key, shape in expected.items():
        if tuple(arrays[key].shape[1:]) != shape:
            raise ValueError(f"Synthetic {key} must be [N,{','.join(map(str, shape))}], got {arrays[key].shape}.")
    labels = arrays["mi_labels"].reshape(-1)
    unknown = sorted(set(int(value) for value in np.unique(labels)) - {-1, 0, 1})
    if unknown:
        raise ValueError(f"Synthetic replay has invalid MI labels: {unknown}")
    buffer = LabeledPixelReplayBuffer(max(count, 1), image_shape, state_dim, action_dim, seed)
    for index in range(count):
        buffer.add_labeled(
            arrays["images"][index],
            arrays["states"][index],
            arrays["actions"][index],
            float(np.asarray(arrays["rewards"][index]).reshape(-1)[0]),
            arrays["next_images"][index],
            arrays["next_states"][index],
            not bool(np.asarray(arrays["not_done"][index]).reshape(-1)[0]),
            label=int(labels[index]),
        )
    return buffer


def paced_batch(
    online: PixelReplayBuffer,
    demonstrations: PixelReplayBuffer,
    synthetic: LabeledPixelReplayBuffer,
    batch_size: int,
    synthetic_ratio: float,
) -> tuple[ArrayBatch, int]:
    synthetic_size = max(2, int(round(batch_size * synthetic_ratio)))
    synthetic_size = min(synthetic_size, batch_size - 2)
    grounded_size = batch_size - synthetic_size
    online_size = grounded_size // 2
    demo_size = grounded_size - online_size
    batches = (
        online.sample(online_size),
        demonstrations.sample(demo_size),
        synthetic.sample_balanced_clear(synthetic_size),
    )
    keys = ("images", "states", "actions", "rewards", "next_images", "next_states", "not_done")
    return ({key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}, synthetic_size)
