from __future__ import annotations

import json
from pathlib import Path

from mi_reward.features.kinematic_extractor import extract_kinematic_latents
from mi_reward.scoring.build_preferences import build_outcome_anchored_pairs


def test_kinematic_extractor_synchronizes_robot_and_object_state(tmp_path: Path) -> None:
    robot = tmp_path / "robot.jsonl"
    robot.write_text("".join(json.dumps({
        "frame_index": index,
        "joint_positions": [0.1 * index, 0.2],
        "end_effector_position": [0.0, 0.1, 0.2 + index * 0.01],
        "end_effector_quaternion": [1.0, 0.0, 0.0, 0.0],
        "gripper_width": float(index % 2),
    }) + "\n" for index in range(3)), encoding="utf-8")
    objects = tmp_path / "objects.jsonl"
    objects.write_text("".join(json.dumps({
        "frame_index": index,
        "objects": [{
            "object_id": "task_object",
            "category": "apple",
            "pose_xyz": [0.3 + index * 0.01, 0.0, 0.1],
            "pose_quaternion": [1.0, 0.0, 0.0, 0.0],
            "size_xyz": [0.05, 0.05, 0.05],
        }],
    }) + "\n" for index in range(3)), encoding="utf-8")

    latents = extract_kinematic_latents(robot, objects, output_dim=24)

    assert tuple(latents.shape) == (3, 1, 24)
    assert not latents[0].equal(latents[-1])


def test_outcome_anchored_preferences_choose_success_over_hard_failure() -> None:
    scored = [
        {
            "traj_id": "success",
            "task": "pick",
            "goal_ref_id": "pick/apple/success",
            "task_success": True,
            "score_delta": 0.2,
            "raw_score_delta": 0.2,
            "confidence": 1.0,
        },
        {
            "traj_id": "hard_near_miss",
            "task": "pick",
            "goal_ref_id": "pick/apple/success",
            "task_success": False,
            "score_delta": 0.9,
            "raw_score_delta": 0.9,
            "confidence": 1.0,
        },
    ]

    pairs = build_outcome_anchored_pairs(scored, margin=0.05)

    assert len(pairs) == 1
    assert pairs[0].chosen_traj_id == "success"
    assert pairs[0].rejected_traj_id == "hard_near_miss"


def test_score_only_preferences_do_not_read_measured_outcomes() -> None:
    shared = {
        "task": "pick",
        "goal_ref_id": "pick/apple/success",
        "split": "train",
        "parent_traj_id": "episode-1",
        "instance_variant_id": "apple_train",
        "scene_variant_id": "table_light_1",
        "confidence": 1.0,
    }
    scored = [
        {**shared, "traj_id": "success", "task_success": True, "score_delta": 0.2},
        {**shared, "traj_id": "failure", "task_success": False, "score_delta": 0.9},
    ]

    pairs = build_outcome_anchored_pairs(
        scored,
        margin=0.05,
        use_measured_outcomes=False,
    )

    assert len(pairs) == 1
    assert pairs[0].chosen_traj_id == "failure"
    assert pairs[0].comparison_type == "score_only"


def test_outcome_pairs_stay_within_initial_state_context_and_include_fine_grained_pairs() -> None:
    shared = {
        "task": "pick",
        "goal_ref_id": "pick/apple/success",
        "split": "train",
        "parent_traj_id": "episode-1",
        "instance_variant_id": "apple_train",
        "scene_variant_id": "table_light_1",
        "confidence": 1.0,
    }
    scored = [
        {**shared, "traj_id": "success_fast", "task_success": True, "score_delta": 3.0},
        {**shared, "traj_id": "success_detour", "task_success": True, "score_delta": 2.4},
        {**shared, "traj_id": "near_miss", "task_success": False, "score_delta": 0.8},
        {
            **shared,
            "parent_traj_id": "episode-2",
            "traj_id": "other_initial_state",
            "task_success": False,
            "score_delta": 0.1,
        },
    ]

    pairs = build_outcome_anchored_pairs(scored, margin=0.05, pair_scope="same_context")

    assert len(pairs) == 3
    assert {pair.comparison_type for pair in pairs} == {
        "success_vs_failure",
        "within_success",
    }
    assert all("other_initial_state" not in {pair.chosen_traj_id, pair.rejected_traj_id} for pair in pairs)
    assert all(pair.split == "train" for pair in pairs)
