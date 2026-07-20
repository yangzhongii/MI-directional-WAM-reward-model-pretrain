from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
