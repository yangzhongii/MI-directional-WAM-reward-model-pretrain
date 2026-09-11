from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mi_reward.data.cosmos_predict_worker import _generate, _visual_consistency
from mi_reward.sim.mujoco_generalization_worker import _action


class _FakeChunkModel:
    def __init__(self) -> None:
        self.calls = 0

    def step_inference(self, current, *, action, guidance, seed):
        del current, guidance, seed
        base = self.calls * 100
        self.calls += 1
        video = np.stack(
            [np.full((2, 3, 3), base + index, dtype=np.uint8) for index in range(len(action) + 1)],
            axis=0,
        )
        return video[-1], video


def test_cosmos_action_uses_previous_end_effector_frame() -> None:
    half = np.sqrt(0.5)
    previous_quaternion = np.asarray([half, 0.0, 0.0, half])  # +90 degrees around world z
    action = _action(
        np.asarray([0.0, 0.0, 0.0]),
        previous_quaternion,
        np.asarray([0.0, 1.0, 0.0]),
        previous_quaternion,
        gripper=1.0,
        scale=20.0,
    )

    np.testing.assert_allclose(action[:3], [20.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(action[3:6], 0.0, atol=1e-5)
    assert action[6] == 1.0


def test_cosmos_chunk_padding_is_trimmed_without_timeline_resampling() -> None:
    actions = np.zeros((31, 7), dtype=np.float32)
    video = _generate(
        _FakeChunkModel(),
        np.zeros((2, 3, 3), dtype=np.uint8),
        actions,
        chunk_size=12,
        guidance=7,
        seed=0,
    )

    assert video.shape == (32, 2, 3, 3)
    assert video[:, 0, 0, 0].tolist() == [*range(13), *range(101, 113), *range(201, 208)]


def test_visual_consistency_rejects_large_drift(tmp_path: Path) -> None:
    frame_paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((4, 5, 3), index * 2, dtype=np.uint8)).save(path)
        frame_paths.append(path)
    reference = np.stack(
        [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in frame_paths],
        axis=0,
    )
    accepted = _visual_consistency(
        reference,
        frame_paths,
        (4, 5),
        max_first_frame_mae=1.0,
        max_mean_pixel_mae=1.0,
        max_final_pixel_mae=1.0,
        max_motion_ratio=2.0,
        max_endpoint_change_mae=1.0,
        min_endpoint_change_cosine=0.9,
        min_motion_support_iou=0.9,
        motion_pixel_threshold=1.0,
    )
    drifted = reference.copy()
    drifted[1:] = 255
    rejected = _visual_consistency(
        drifted,
        frame_paths,
        (4, 5),
        max_first_frame_mae=1.0,
        max_mean_pixel_mae=10.0,
        max_final_pixel_mae=10.0,
        max_motion_ratio=2.0,
        max_endpoint_change_mae=10.0,
        min_endpoint_change_cosine=0.9,
        min_motion_support_iou=0.9,
        motion_pixel_threshold=1.0,
    )

    assert accepted["passed"]
    assert not rejected["passed"]
    assert not rejected["checks"]["mean_appearance"]


def test_visual_consistency_rejects_motion_in_the_wrong_region(tmp_path: Path) -> None:
    reference = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    reference[-1, 1:3, 1:3] = 255
    frame_paths = []
    for index, frame in enumerate(reference):
        path = tmp_path / f"motion-{index}.png"
        Image.fromarray(frame).save(path)
        frame_paths.append(path)
    generated = np.zeros_like(reference)
    generated[-1, 5:7, 5:7] = 255

    result = _visual_consistency(
        generated,
        frame_paths,
        (8, 8),
        max_first_frame_mae=1.0,
        max_mean_pixel_mae=30.0,
        max_final_pixel_mae=40.0,
        max_motion_ratio=2.0,
        max_endpoint_change_mae=40.0,
        min_endpoint_change_cosine=0.1,
        min_motion_support_iou=0.2,
        motion_pixel_threshold=12.0,
    )

    assert not result["passed"]
    assert not result["checks"]["endpoint_direction"]
    assert not result["checks"]["motion_location"]
