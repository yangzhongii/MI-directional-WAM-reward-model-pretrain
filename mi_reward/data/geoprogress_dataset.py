"""Dataset contract for verified, goal-conditioned GeoProgress training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.relations.sequence import RelationSequence, load_relation_sequence


def _requires_verification(example: TrajectoryExample) -> bool:
    return example.source == "cosmos_action_cond" or example.candidate_provenance is not None


class GeoProgressPreferenceDataset(Dataset[dict[str, object]]):
    """Pairs of verified trajectories, shared success references, and relations.

    This deliberately does not fall back to pooled features. The new model
    needs native visual tokens plus measured relation sequences, otherwise a
    run should fail rather than silently become the legacy visual baseline.
    """

    def __init__(
        self,
        preferences_path: str | Path,
        manifest_path: str | Path,
        success_refs_path: str | Path,
        feature_root: str | Path,
    ):
        self.preferences = read_jsonl(preferences_path, PreferencePair)
        self.trajectories = {
            item.traj_id: item for item in read_jsonl(manifest_path, TrajectoryExample)
        }
        self.success_refs = {
            item.ref_id: item for item in read_jsonl(success_refs_path, SuccessReference)
        }
        self.store = CachedFeatureStore(feature_root)
        self._relations: dict[str, RelationSequence] = {}
        for pair in self.preferences:
            self._validate_pair(pair)

    def _validate_pair(self, pair: PreferencePair) -> None:
        try:
            chosen = self.trajectories[pair.chosen_traj_id]
            rejected = self.trajectories[pair.rejected_traj_id]
        except KeyError as exc:
            raise ValueError(f"Preference refers to a trajectory absent from the manifest: {exc}") from exc
        for example in (chosen, rejected):
            if _requires_verification(example) and not (example.verification and example.verification.accepted):
                raise ValueError(
                    f"Unverified Cosmos candidate {example.traj_id} cannot be used as GeoProgress supervision."
                )
            if not example.relation_path:
                raise ValueError(f"GeoProgress requires relation_path for {example.traj_id}.")
        goal_ref_id = self._goal_ref_id(pair, chosen, rejected)
        if goal_ref_id not in self.success_refs:
            raise ValueError(f"Unknown goal_ref_id {goal_ref_id!r} in preference pair.")

    @staticmethod
    def _goal_ref_id(pair: PreferencePair, chosen: TrajectoryExample, rejected: TrajectoryExample) -> str:
        goal_ref_id = pair.goal_ref_id or chosen.goal_ref_id
        if not goal_ref_id or chosen.goal_ref_id != goal_ref_id or rejected.goal_ref_id != goal_ref_id:
            raise ValueError(
                "GeoProgress pairs must have one explicit shared goal_ref_id on the pair and both trajectories."
            )
        return goal_ref_id

    def _load_tokens(self, item_id: str) -> torch.Tensor:
        tokens = self.store.load(item_id + "_tokens").float()
        if tokens.ndim != 3 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
            raise ValueError(
                f"GeoProgress requires native token cache [T, K, D] with K >= 2 for {item_id}; got {tuple(tokens.shape)}."
            )
        return tokens

    def _load_relations(self, example: TrajectoryExample, frames: int) -> RelationSequence:
        cached = self._relations.get(example.traj_id)
        if cached is None:
            assert example.relation_path is not None
            cached = load_relation_sequence(example.relation_path)
            self._relations[example.traj_id] = cached
        if cached.values.shape[0] != frames:
            raise ValueError(
                f"Relation/frame length mismatch for {example.traj_id}: "
                f"{cached.values.shape[0]} relations versus {frames} visual frames."
            )
        return cached

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.preferences[index]
        chosen_example = self.trajectories[pair.chosen_traj_id]
        rejected_example = self.trajectories[pair.rejected_traj_id]
        goal_ref_id = self._goal_ref_id(pair, chosen_example, rejected_example)
        chosen = self._load_tokens(chosen_example.traj_id)
        rejected = self._load_tokens(rejected_example.traj_id)
        goal_trajectory = self._load_tokens(goal_ref_id)
        goal = goal_trajectory[-1]
        chosen_rel = self._load_relations(chosen_example, chosen.shape[0])
        rejected_rel = self._load_relations(rejected_example, rejected.shape[0])
        if chosen_rel.names != rejected_rel.names:
            raise ValueError(f"Relation field mismatch inside pair {chosen_example.traj_id}/{rejected_example.traj_id}.")
        return {
            "pair": pair,
            "goal_ref_id": goal_ref_id,
            "chosen_tokens": chosen,
            "rejected_tokens": rejected,
            "goal_tokens": goal,
            "goal_trajectory_tokens": goal_trajectory,
            "chosen_relations": chosen_rel.values,
            "rejected_relations": rejected_rel.values,
            "relation_names": chosen_rel.names,
        }
