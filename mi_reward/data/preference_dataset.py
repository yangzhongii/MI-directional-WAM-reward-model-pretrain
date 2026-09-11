from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mi_reward.data.schema import PreferencePair, read_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore


class PreferenceFeatureDataset(Dataset[dict[str, object]]):
    def __init__(self, preferences_path: str | Path, feature_root: str | Path):
        self.preferences = read_jsonl(preferences_path, PreferencePair)
        self.store = CachedFeatureStore(feature_root)

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.preferences[index]
        chosen = self.store.load(pair.chosen_traj_id).float()
        rejected = self.store.load(pair.rejected_traj_id).float()
        return {
            "pair": pair,
            "chosen_features": chosen,
            "rejected_features": rejected,
            "chosen_score": torch.tensor(pair.chosen_score, dtype=torch.float32),
            "rejected_score": torch.tensor(pair.rejected_score, dtype=torch.float32),
        }
