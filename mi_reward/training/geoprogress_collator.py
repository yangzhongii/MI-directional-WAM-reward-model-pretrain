"""Padding for token-level, goal-conditioned GeoProgress preference batches."""

from __future__ import annotations

import torch


def _pad_token_trajectories(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items or any(item.ndim != 3 for item in items):
        raise ValueError("Expected non-empty [T, K, D] token trajectories.")
    max_t = max(item.shape[0] for item in items)
    max_k = items[0].shape[1]
    dim = items[0].shape[2]
    if any(item.shape[1] != max_k or item.shape[2] != dim for item in items):
        raise ValueError("Token grid or embedding dimensions differ within a batch.")
    batch = torch.zeros((len(items), max_t, max_k, dim), dtype=torch.float32)
    mask = torch.zeros((len(items), max_t), dtype=torch.bool)
    for i, item in enumerate(items):
        batch[i, :item.shape[0], :item.shape[1]] = item.float()
        mask[i, :item.shape[0]] = True
    return batch, mask


def _pad_goals(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items or any(item.ndim != 2 for item in items):
        raise ValueError("Expected non-empty [K, D] goal tokens.")
    max_k = items[0].shape[0]
    dim = items[0].shape[1]
    if any(item.shape[0] != max_k or item.shape[1] != dim for item in items):
        raise ValueError("Goal token grid or embedding dimensions differ within a batch.")
    batch = torch.zeros((len(items), max_k, dim), dtype=torch.float32)
    mask = torch.zeros((len(items), max_k), dtype=torch.bool)
    for i, item in enumerate(items):
        batch[i, :item.shape[0]] = item.float()
        mask[i, :item.shape[0]] = True
    return batch, mask


def _pad_relations(items: list[torch.Tensor]) -> torch.Tensor:
    max_t = max(item.shape[0] for item in items)
    dim = items[0].shape[1]
    if any(item.ndim != 2 or item.shape[1] != dim for item in items):
        raise ValueError("Relation dimensions differ within a batch.")
    batch = torch.zeros((len(items), max_t, dim), dtype=torch.float32)
    for i, item in enumerate(items):
        batch[i, :item.shape[0]] = item.float()
    return batch


class GeoProgressCollator:
    def __call__(self, items: list[dict[str, object]]) -> dict[str, object]:
        names = items[0]["relation_names"]
        if any(item["relation_names"] != names for item in items):
            raise ValueError("All GeoProgress samples in a batch must share relation field names.")
        chosen, chosen_mask = _pad_token_trajectories([item["chosen_tokens"] for item in items])  # type: ignore[list-item]
        rejected, rejected_mask = _pad_token_trajectories([item["rejected_tokens"] for item in items])  # type: ignore[list-item]
        goals, goal_mask = _pad_goals([item["goal_tokens"] for item in items])  # type: ignore[list-item]
        return {
            "pairs": [item["pair"] for item in items],
            "relation_names": names,
            "chosen_tokens": chosen,
            "chosen_mask": chosen_mask,
            "rejected_tokens": rejected,
            "rejected_mask": rejected_mask,
            "goal_tokens": goals,
            "goal_mask": goal_mask,
            "chosen_relations": _pad_relations([item["chosen_relations"] for item in items]),  # type: ignore[list-item]
            "rejected_relations": _pad_relations([item["rejected_relations"] for item in items]),  # type: ignore[list-item]
            "chosen_confidence": torch.tensor([item["pair"].chosen_confidence for item in items], dtype=torch.float32),  # type: ignore[union-attr]
            "rejected_confidence": torch.tensor([item["pair"].rejected_confidence for item in items], dtype=torch.float32),  # type: ignore[union-attr]
            "score_margin": torch.tensor([item["pair"].chosen_score - item["pair"].rejected_score for item in items], dtype=torch.float32),  # type: ignore[union-attr]
        }
