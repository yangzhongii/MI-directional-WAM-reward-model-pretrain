from __future__ import annotations

import torch


def _pad_sequence(features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(feature.shape[0] for feature in features)
    dim = features[0].reshape(features[0].shape[0], -1).shape[-1]
    batch = torch.zeros((len(features), max_len, dim), dtype=torch.float32)
    mask = torch.zeros((len(features), max_len), dtype=torch.bool)
    for idx, feature in enumerate(features):
        feature = feature.reshape(feature.shape[0], -1).float()
        batch[idx, : feature.shape[0]] = feature
        mask[idx, : feature.shape[0]] = True
    return batch, mask


class PreferenceCollator:
    def __call__(self, items: list[dict[str, object]]) -> dict[str, torch.Tensor | list[object]]:
        chosen, chosen_mask = _pad_sequence([item["chosen_features"] for item in items])  # type: ignore[list-item]
        rejected, rejected_mask = _pad_sequence([item["rejected_features"] for item in items])  # type: ignore[list-item]
        return {
            "pairs": [item["pair"] for item in items],
            "chosen_features": chosen,
            "chosen_mask": chosen_mask,
            "rejected_features": rejected,
            "rejected_mask": rejected_mask,
            "chosen_score": torch.stack([item["chosen_score"] for item in items]),  # type: ignore[list-item]
            "rejected_score": torch.stack([item["rejected_score"] for item in items]),  # type: ignore[list-item]
        }
