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


def _pad_potentials(items: list[torch.Tensor], max_t: int) -> torch.Tensor:
    if not items or any(item.ndim != 1 or item.shape[0] > max_t for item in items):
        raise ValueError("Expected one teacher potential per valid trajectory frame.")
    batch = torch.zeros((len(items), max_t), dtype=torch.float32)
    for index, item in enumerate(items):
        batch[index, :item.shape[0]] = item.float()
    return batch


class GeoProgressCollator:
    def __call__(self, items: list[dict[str, object]]) -> dict[str, object]:
        chosen, chosen_mask = _pad_token_trajectories([item["chosen_tokens"] for item in items])  # type: ignore[list-item]
        rejected, rejected_mask = _pad_token_trajectories([item["rejected_tokens"] for item in items])  # type: ignore[list-item]
        goals, goal_mask = _pad_goals([item["goal_tokens"] for item in items])  # type: ignore[list-item]
        batch: dict[str, object] = {
            "pairs": [item["pair"] for item in items],
            "chosen_tokens": chosen,
            "chosen_mask": chosen_mask,
            "rejected_tokens": rejected,
            "rejected_mask": rejected_mask,
            "goal_tokens": goals,
            "goal_mask": goal_mask,
            "chosen_confidence": torch.tensor([item["pair"].chosen_confidence for item in items], dtype=torch.float32),  # type: ignore[union-attr]
            "rejected_confidence": torch.tensor([item["pair"].rejected_confidence for item in items], dtype=torch.float32),  # type: ignore[union-attr]
            "score_margin": torch.tensor([item["pair"].chosen_score - item["pair"].rejected_score for item in items], dtype=torch.float32),  # type: ignore[union-attr]
        }
        if "chosen_teacher_phi" in items[0]:
            batch.update(
                {
                    "chosen_teacher_phi": _pad_potentials(
                        [item["chosen_teacher_phi"] for item in items], chosen.shape[1]  # type: ignore[list-item]
                    ),
                    "rejected_teacher_phi": _pad_potentials(
                        [item["rejected_teacher_phi"] for item in items], rejected.shape[1]  # type: ignore[list-item]
                    ),
                }
            )
            return batch
        names = items[0]["relation_names"]
        if any(item["relation_names"] != names for item in items):
            raise ValueError("All GeoProgress samples in a batch must share relation field names.")
        goal_trajectories, goal_trajectory_mask = _pad_token_trajectories(
            [item["goal_trajectory_tokens"] for item in items]  # type: ignore[list-item]
        )
        batch.update(
            {
                "relation_names": names,
                "goal_trajectory_tokens": goal_trajectories,
                "goal_trajectory_mask": goal_trajectory_mask,
                "chosen_relations": _pad_relations([item["chosen_relations"] for item in items]),  # type: ignore[list-item]
                "rejected_relations": _pad_relations([item["rejected_relations"] for item in items]),  # type: ignore[list-item]
            }
        )
        if "chosen_action_latents" in items[0]:
            chosen_actions, chosen_action_mask = _pad_token_trajectories(
                [item["chosen_action_latents"] for item in items]  # type: ignore[list-item]
            )
            rejected_actions, rejected_action_mask = _pad_token_trajectories(
                [item["rejected_action_latents"] for item in items]  # type: ignore[list-item]
            )
            goal_actions, goal_action_mask = _pad_token_trajectories(
                [item["goal_action_latents"] for item in items]  # type: ignore[list-item]
            )
            batch.update(
                {
                    "chosen_action_latents": chosen_actions,
                    "chosen_action_mask": chosen_action_mask,
                    "rejected_action_latents": rejected_actions,
                    "rejected_action_mask": rejected_action_mask,
                    "goal_action_latents": goal_actions,
                    "goal_action_mask": goal_action_mask,
                }
            )
        if "chosen_kinematic_latents" in items[0]:
            chosen_kinematics, chosen_kinematic_mask = _pad_token_trajectories(
                [item["chosen_kinematic_latents"] for item in items]  # type: ignore[list-item]
            )
            rejected_kinematics, rejected_kinematic_mask = _pad_token_trajectories(
                [item["rejected_kinematic_latents"] for item in items]  # type: ignore[list-item]
            )
            goal_kinematics, goal_kinematic_mask = _pad_token_trajectories(
                [item["goal_kinematic_latents"] for item in items]  # type: ignore[list-item]
            )
            batch.update(
                {
                    "chosen_kinematic_latents": chosen_kinematics,
                    "chosen_kinematic_mask": chosen_kinematic_mask,
                    "rejected_kinematic_latents": rejected_kinematics,
                    "rejected_kinematic_mask": rejected_kinematic_mask,
                    "goal_kinematic_latents": goal_kinematics,
                    "goal_kinematic_mask": goal_kinematic_mask,
                }
            )
        return batch
