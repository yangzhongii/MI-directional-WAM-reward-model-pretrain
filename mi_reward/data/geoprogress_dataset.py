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
        require_action_latents: bool = False,
        require_kinematic_latents: bool = False,
        cache_tokens: bool = False,
        allowed_splits: tuple[str, ...] | None = None,
    ):
        self.trajectories = {
            item.traj_id: item for item in read_jsonl(manifest_path, TrajectoryExample)
        }
        raw_preferences = read_jsonl(preferences_path, PreferencePair)
        self.preferences = [
            pair
            for pair in raw_preferences
            if allowed_splits is None
            or (
                self.trajectories[pair.chosen_traj_id].split in allowed_splits
                and self.trajectories[pair.rejected_traj_id].split in allowed_splits
            )
        ]
        self.success_refs = {
            item.ref_id: item for item in read_jsonl(success_refs_path, SuccessReference)
        }
        self.store = CachedFeatureStore(feature_root)
        self.require_action_latents = require_action_latents
        self.require_kinematic_latents = require_kinematic_latents
        self.cache_tokens = cache_tokens
        self._relations: dict[str, RelationSequence] = {}
        self._token_cache: dict[str, torch.Tensor] = {}
        self._teacher_targets: dict[tuple[str, str], torch.Tensor] | None = None
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
        if self.require_action_latents:
            for item_id in (chosen.traj_id, rejected.traj_id, goal_ref_id):
                path = self.store.path_for(item_id + "_action_latents")
                if not path.is_file():
                    raise FileNotFoundError(f"Privileged action teacher requires action latent cache: {path}")
        if self.require_kinematic_latents:
            for item_id in (chosen.traj_id, rejected.traj_id, goal_ref_id):
                path = self.store.path_for(item_id + "_kinematic_latents")
                if not path.is_file():
                    raise FileNotFoundError(f"Privileged kinematic teacher requires latent cache: {path}")

    @staticmethod
    def _goal_ref_id(pair: PreferencePair, chosen: TrajectoryExample, rejected: TrajectoryExample) -> str:
        goal_ref_id = pair.goal_ref_id or chosen.goal_ref_id
        if not goal_ref_id or chosen.goal_ref_id != goal_ref_id or rejected.goal_ref_id != goal_ref_id:
            raise ValueError(
                "GeoProgress pairs must have one explicit shared goal_ref_id on the pair and both trajectories."
            )
        return goal_ref_id

    def _load_tokens(self, item_id: str) -> torch.Tensor:
        cached = self._token_cache.get(item_id)
        if cached is not None:
            return cached
        tokens = self.store.load(item_id + "_tokens").float()
        if tokens.ndim != 3 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
            raise ValueError(
                f"GeoProgress requires native token cache [T, K, D] with K >= 2 for {item_id}; got {tuple(tokens.shape)}."
            )
        if self.cache_tokens or item_id in self.success_refs:
            self._token_cache[item_id] = tokens
        return tokens

    def teacher_keys(self) -> list[tuple[str, str]]:
        """Unique trajectory/reference combinations required by all pairs."""

        keys: set[tuple[str, str]] = set()
        for pair in self.preferences:
            chosen = self.trajectories[pair.chosen_traj_id]
            rejected = self.trajectories[pair.rejected_traj_id]
            goal_ref_id = self._goal_ref_id(pair, chosen, rejected)
            keys.add((chosen.traj_id, goal_ref_id))
            keys.add((rejected.traj_id, goal_ref_id))
        return sorted(keys)

    def load_teacher_inputs(self, traj_id: str, goal_ref_id: str) -> dict[str, object]:
        """Load one unique privileged-teacher example outside pair batching."""

        example = self.trajectories[traj_id]
        if example.goal_ref_id != goal_ref_id or goal_ref_id not in self.success_refs:
            raise ValueError(f"Invalid teacher key: {traj_id!r}/{goal_ref_id!r}.")
        tokens = self._load_tokens(traj_id)
        goal_tokens = self._load_tokens(goal_ref_id)
        relation = self._load_relations(example, tokens.shape[0])
        item: dict[str, object] = {
            "tokens": tokens,
            "goal_trajectory_tokens": goal_tokens,
            "relations": relation.values,
            "relation_names": relation.names,
        }
        if self.require_action_latents:
            item["actions"] = self._load_action_latents(traj_id, tokens.shape[0])
            item["goal_actions"] = self._load_action_latents(goal_ref_id, goal_tokens.shape[0])
        if self.require_kinematic_latents:
            item["kinematics"] = self._load_kinematic_latents(traj_id, tokens.shape[0])
            item["goal_kinematics"] = self._load_kinematic_latents(goal_ref_id, goal_tokens.shape[0])
        return item

    def set_teacher_targets(self, targets: dict[tuple[str, str], torch.Tensor]) -> None:
        missing = [key for key in self.teacher_keys() if key not in targets]
        if missing:
            raise ValueError(f"Missing precomputed teacher targets: {missing[:5]}")
        for traj_id, goal_ref_id in self.teacher_keys():
            target = targets[(traj_id, goal_ref_id)]
            frame_count = self._load_tokens(traj_id).shape[0]
            if target.ndim != 1 or target.shape[0] != frame_count:
                raise ValueError(
                    f"Teacher target for {traj_id}/{goal_ref_id} must have shape "
                    f"[{frame_count}], got {tuple(target.shape)}."
                )
            if not torch.isfinite(target).all():
                raise ValueError(f"Teacher target for {traj_id}/{goal_ref_id} contains non-finite values.")
            if float(target.min()) < -1e-6 or float(target.max()) > 1.0 + 1e-6:
                raise ValueError(
                    f"Teacher target for {traj_id}/{goal_ref_id} must stay in [0, 1]."
                )
        self._teacher_targets = targets

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

    def _load_action_latents(self, item_id: str, frames: int) -> torch.Tensor:
        actions = self.store.load(item_id + "_action_latents").float()
        if actions.ndim != 3 or actions.shape[0] != frames - 1 or actions.shape[1] < 1:
            raise ValueError(
                f"Action latents for {item_id} must be [T-1,Q,D]; got {tuple(actions.shape)} "
                f"for {frames} visual frames."
            )
        return actions

    def _load_kinematic_latents(self, item_id: str, frames: int) -> torch.Tensor:
        latents = self.store.load(item_id + "_kinematic_latents").float()
        if latents.ndim != 3 or latents.shape[0] != frames or latents.shape[1] < 1:
            raise ValueError(
                f"Kinematic latents for {item_id} must be [T,Q,D]; got {tuple(latents.shape)} "
                f"for {frames} visual frames."
            )
        return latents

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
        if self._teacher_targets is not None:
            return {
                "pair": pair,
                "goal_ref_id": goal_ref_id,
                "chosen_tokens": chosen,
                "rejected_tokens": rejected,
                "goal_tokens": goal,
                "chosen_teacher_phi": self._teacher_targets[(chosen_example.traj_id, goal_ref_id)],
                "rejected_teacher_phi": self._teacher_targets[(rejected_example.traj_id, goal_ref_id)],
            }
        chosen_rel = self._load_relations(chosen_example, chosen.shape[0])
        rejected_rel = self._load_relations(rejected_example, rejected.shape[0])
        if chosen_rel.names != rejected_rel.names:
            raise ValueError(f"Relation field mismatch inside pair {chosen_example.traj_id}/{rejected_example.traj_id}.")
        item: dict[str, object] = {
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
        if self.require_action_latents:
            item.update(
                {
                    "chosen_action_latents": self._load_action_latents(chosen_example.traj_id, chosen.shape[0]),
                    "rejected_action_latents": self._load_action_latents(rejected_example.traj_id, rejected.shape[0]),
                    "goal_action_latents": self._load_action_latents(goal_ref_id, goal_trajectory.shape[0]),
                }
            )
        if self.require_kinematic_latents:
            item.update(
                {
                    "chosen_kinematic_latents": self._load_kinematic_latents(chosen_example.traj_id, chosen.shape[0]),
                    "rejected_kinematic_latents": self._load_kinematic_latents(rejected_example.traj_id, rejected.shape[0]),
                    "goal_kinematic_latents": self._load_kinematic_latents(goal_ref_id, goal_trajectory.shape[0]),
                }
            )
        return item
