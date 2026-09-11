from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mi_reward.closed_loop.libero_env import (
    apply_appearance_shift,
    canonical_camera_image,
    ensure_disjoint_splits,
    policy_camera_images,
)
from mi_reward.closed_loop.matrix import _sync_source
from mi_reward.closed_loop.rlpd_matrix import _jobs
from mi_reward.closed_loop.online_potential import PotentialReward
from mi_reward.closed_loop.rlpd import PixelReplayBuffer, RLPDAgent, RLPDConfig, balanced_batch
from mi_reward.closed_loop.sac import ReplayBuffer, SACAgent, SACConfig
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
from mi_reward.models.visual_goal_potential import VisualGoalPotential


def _checkpoint(path: Path, visual_dim: int = 12) -> Path:
    torch.manual_seed(3)
    model = VisualGoalPotential(
        visual_dim=visual_dim,
        hidden_dim=16,
        architecture="gru",
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        goal_dropout=0.0,
    )
    torch.save(
        {
            "config": {
                "model_class": "VisualGoalPotential",
                "visual_dim": visual_dim,
                "hidden_dim": 16,
                "architecture": "gru",
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "goal_dropout": 0.0,
                "gamma": 0.99,
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_online_potential_is_stateful_and_reports_components(tmp_path: Path) -> None:
    reward = PotentialReward(
        str(_checkpoint(tmp_path / "reward.pt")),
        goal_tokens=torch.randn(5, 12),
        mode="sparse_mi",
        device="cpu",
        mi_weight=2.0,
        clip_delta=0.5,
    )
    initial = reward.reset(torch.randn(7, 12))
    first = reward.step(torch.randn(7, 12), sparse_reward=0.0)
    second = reward.step(torch.randn(7, 12), sparse_reward=1.0)
    assert first.previous_potential == pytest.approx(initial)
    assert second.previous_potential == pytest.approx(first.next_potential)
    assert abs(first.mi_delta) <= 0.5
    assert second.total == pytest.approx(1.0 + second.weighted_mi)


def test_sac_update_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = SACConfig(hidden_dims=(32, 32))
    agent = SACAgent(10, np.full(3, -1.0), np.full(3, 1.0), config, "cpu")
    replay = ReplayBuffer(128, 10, 3, seed=4)
    rng = np.random.default_rng(4)
    for _ in range(64):
        replay.add(rng.normal(size=10), rng.uniform(-1, 1, size=3), 0.1, rng.normal(size=10), False)
    metrics = agent.update(replay.sample(32, agent.device))
    assert all(np.isfinite(value) for value in metrics.values())
    action = agent.act(np.zeros(10, dtype=np.float32), deterministic=True)
    assert action.shape == (3,)
    path = agent.save(tmp_path / "sac.pt", step=64, metadata={"test": True})
    restored, payload = SACAgent.load(path, "cpu")
    np.testing.assert_allclose(action, restored.act(np.zeros(10, dtype=np.float32), deterministic=True), atol=1e-6)
    assert payload["step"] == 64


def test_in_memory_fallback_features_and_appearance_shift() -> None:
    image = np.full((32, 48, 3), 128, dtype=np.uint8)
    extractor = DINOv3FeatureExtractor(model_path=None, device="cpu", image_size=24, strict=False)
    tokens = extractor.extract_image_tokens(image, task="")
    assert tokens.shape == (1, 12)
    shifted = apply_appearance_shift(image, "dark_warm")
    assert shifted.shape == image.shape
    assert shifted.dtype == np.uint8
    assert not np.array_equal(shifted, image)


def test_libero_policy_images_preserve_official_demo_orientation() -> None:
    main = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = np.flip(main, axis=1).copy()
    observation = {"agentview_image": main, "robot0_eye_in_hand_image": wrist}
    np.testing.assert_array_equal(canonical_camera_image(observation), main)
    np.testing.assert_array_equal(policy_camera_images(observation), np.stack((main, wrist)))


def test_initial_state_splits_reject_training_leakage() -> None:
    ensure_disjoint_splits({"train": [0, 1, 2], "test": [3, 4]})
    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint_splits({"train": [0, 1, 2], "test": [2, 3]})


def test_remote_source_sync_preserves_excluded_runtime_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def capture(command: list[str], *, check: bool) -> None:
        assert check
        calls.append(command)

    monkeypatch.setattr("mi_reward.closed_loop.matrix.subprocess.run", capture)
    _sync_source("worker", "/remote/project", tmp_path)
    command = calls[0]
    assert "--delete" not in command
    assert "--delete-excluded" not in command
    assert "--exclude=.venv*/" in command


def _pixel_buffer(seed: int, reward: float) -> PixelReplayBuffer:
    buffer = PixelReplayBuffer(16, (2, 32, 32, 3), 5, 3, seed)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        buffer.add(images, rng.normal(size=5), rng.uniform(-1, 1, size=3), reward, images, rng.normal(size=5), False)
    return buffer


def test_rlpd_balanced_replay_is_half_online_half_demo() -> None:
    batch = balanced_batch(_pixel_buffer(1, 0.0), _pixel_buffer(2, 1.0), 8)
    assert batch["images"].shape == (8, 2, 32, 32, 3)
    np.testing.assert_array_equal(batch["rewards"][:4], 0.0)
    np.testing.assert_array_equal(batch["rewards"][4:], 1.0)


def test_rlpd_matrix_builds_non_overlapping_selected_jobs() -> None:
    config = {
        "benchmark": {"task_ids": [0, 1]},
        "reward": {"modes": ["sparse", "sparse_mi"]},
        "training": {"seeds": [0, 1]},
    }
    jobs = _jobs(config, [1], ["sparse_mi"], [0])
    assert jobs == [(1, "sparse_mi", 0)]


def test_visual_rlpd_agent_uses_multiview_images_and_q_ensemble() -> None:
    config = RLPDConfig(
        hidden_dims=(16,), visual_dim=8, state_latent_dim=4, spatial_features=2,
        num_q_heads=3, critic_subsample_size=2, freeze_backbone=True,
    )
    agent = RLPDAgent((2, 32, 32, 3), 5, np.full(3, -1.0), np.full(3, 1.0), config, "cpu")
    batch = balanced_batch(_pixel_buffer(3, 0.0), _pixel_buffer(4, 1.0), 4)
    metrics = agent.update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    action = agent.act(batch["images"][0], batch["states"][0], deterministic=True)
    assert action.shape == (3,)
    assert len(agent.critic.heads) == 3
