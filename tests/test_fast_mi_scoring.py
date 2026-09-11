from __future__ import annotations

import json
from pathlib import Path

import torch

from mi_reward.data.schema import SuccessReference, TrajectoryExample, write_jsonl
from mi_reward.features.cached_feature_store import (
    CachedFeatureStore,
    feature_bundle_is_cached,
    feature_source_signature,
)
from mi_reward.scoring.build_preferences import score_manifest
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.scoring.directional_potential import DirectionalScoreConfig


def test_vectorized_pairwise_mi_matches_scalar_implementation() -> None:
    torch.manual_seed(17)
    candidate = torch.randn(3, 5, 11)
    reference = torch.randn(4, 5, 11)
    estimator = DameSoftHistogramMI(num_bins=8)

    expected = torch.stack(
        [torch.stack([estimator(candidate_t, reference_s) for reference_s in reference])
         for candidate_t in candidate]
    )
    actual = estimator.pairwise_matrix(candidate, reference, pair_chunk_size=3)

    assert actual.shape == (3, 4)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_score_manifest_resumes_per_trajectory_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    references = tmp_path / "references.jsonl"
    feature_root = tmp_path / "features"
    score_cache = tmp_path / "scores.jsonl"
    trajectories = [
        TrajectoryExample(
            traj_id=f"candidate-{index}",
            task="task",
            frames=[],
            source="real",
            split="train",
        )
        for index in range(2)
    ]
    write_jsonl(manifest, trajectories)
    write_jsonl(references, [SuccessReference(ref_id="success", task="task", frames=[])])
    store = CachedFeatureStore(feature_root)
    torch.manual_seed(23)
    for trajectory in trajectories:
        store.save(trajectory.traj_id + "_tokens", torch.randn(3, 4, 6))
    store.save("success_tokens", torch.randn(3, 4, 6))

    kwargs = {
        "manifest": manifest,
        "success_refs": references,
        "feature_root": feature_root,
        "gamma": 0.99,
        "mi_mode": "gaussian_mi_proxy",
        "use_token_features": True,
        "directional_alignment": True,
        "directional_config": DirectionalScoreConfig(mi_pair_chunk_size=2),
        "score_cache": score_cache,
    }
    first = score_manifest(**kwargs)
    line_count = len(score_cache.read_text(encoding="utf-8").splitlines())
    second = score_manifest(**kwargs)

    assert second == first
    assert line_count == 2
    assert len(score_cache.read_text(encoding="utf-8").splitlines()) == line_count
    assert all(json.loads(line)["signature"] for line in score_cache.read_text().splitlines())


def test_complete_feature_bundle_is_detected_without_tensor_load(tmp_path: Path) -> None:
    store = CachedFeatureStore(tmp_path)
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"version-one")
    source_signature = feature_source_signature([str(frame)])
    for suffix in ("", "_tokens", "_action_latents", "_kinematic_latents"):
        store.save("candidate" + suffix, torch.zeros(1))
    store.write_metadata(
        "candidate",
        extractor_name="lawam_lam",
        source_signature=source_signature,
    )

    assert feature_bundle_is_cached(
        store,
        "candidate",
        tokens=True,
        action_latents=True,
        kinematic_latents=True,
        extractor_name="lawam_lam",
        source_signature=source_signature,
    )
    assert not feature_bundle_is_cached(
        store,
        "candidate",
        tokens=True,
        action_latents=True,
        kinematic_latents=True,
        extractor_name="different_extractor",
    )
    frame.write_bytes(b"version-two-is-different")
    changed_signature = feature_source_signature([str(frame)])
    assert changed_signature != source_signature
    assert not feature_bundle_is_cached(
        store,
        "candidate",
        tokens=True,
        action_latents=True,
        kinematic_latents=True,
        extractor_name="lawam_lam",
        source_signature=changed_signature,
    )


def test_trajectory_score_reuses_existing_student_potentials() -> None:
    from mi_reward.models.visual_goal_potential import VisualGoalPotential
    from mi_reward.training.train_reward_sft import _trajectory_score_from_potentials

    torch.manual_seed(31)
    model = VisualGoalPotential(visual_dim=6, hidden_dim=8, num_heads=2, goal_dropout=0.0)
    model.eval()
    tokens = torch.randn(2, 4, 3, 6)
    goals = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    with torch.no_grad():
        potentials = model(tokens, goals, mask)
        expected = model.compute_trajectory_score(tokens, goals, mask, gamma=0.99)
        actual = _trajectory_score_from_potentials(potentials, mask, gamma=0.99)
    assert torch.allclose(actual, expected)
