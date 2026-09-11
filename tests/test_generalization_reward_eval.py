from __future__ import annotations

from mi_reward.evaluation.generalization_reward_eval import (
    _baseline_report,
    _candidate_selection,
    _pair_summary,
    _trajectory_summary,
)
from mi_reward.data.schema import PreferencePair, TaskOutcome, TrajectoryExample


def _example(traj_id: str, *, success: bool, failure_mode: str = "none") -> TrajectoryExample:
    return TrajectoryExample(
        traj_id=traj_id,
        task="task",
        frames=[],
        source="simulator_native",
        split="joint_heldout",
        task_family="pick_place",
        task_outcome=TaskOutcome(
            success=success,
            failure_mode=failure_mode,
            checks={},
        ),
    )


def _record(traj_id: str, score: float, success: bool, failure_mode: str = "none") -> dict[str, object]:
    return {
        "traj_id": traj_id,
        "split": "joint_heldout",
        "task_family": "pick_place",
        "context_id": "context-a",
        "success": success,
        "failure_mode": failure_mode,
        "trajectory_score": score,
        "endpoint_gain": score,
        "positive_transition_ratio": 1.0 if score > 0 else 0.0,
        "progress_pearson": 1.0,
        "progress_spearman": 1.0,
        "regression_detected": failure_mode == "regress_after_progress",
    }


def test_trajectory_metrics_use_measured_success_labels() -> None:
    report = _trajectory_summary([
        _record("success", 2.0, True),
        _record("wrong", -1.0, False, "wrong_direction"),
        _record("regress", -0.5, False, "regress_after_progress"),
    ])
    assert report["success_failure_auc"] == 1.0
    assert report["success_vs_failure_mode_auc"]["wrong_direction"] == 1.0
    assert report["regress_after_progress_detection"]["value"] == 1.0


def test_pair_metrics_are_grouped_by_task_and_type() -> None:
    examples = {
        "success": _example("success", success=True),
        "failure": _example("failure", success=False, failure_mode="undershoot"),
    }
    pair = PreferencePair(
        task="task",
        chosen_traj_id="success",
        rejected_traj_id="failure",
        chosen_score=1.0,
        rejected_score=0.0,
        split="joint_heldout",
        comparison_type="success_vs_failure",
    )
    report = _pair_summary([pair], {"success": 0.8, "failure": 0.2}, examples)
    assert report["teacher_preference_accuracy"]["overall"]["value"] == 1.0
    assert report["teacher_preference_accuracy"]["task/pick_place"]["value"] == 1.0
    assert report["measured_outcome_pair_accuracy"]["value"] == 1.0


def test_candidate_selection_reports_top1_random_and_oracle() -> None:
    report = _candidate_selection([
        _record("failure", 0.9, False, "overshoot"),
        _record("success", 0.8, True),
        _record("failure-2", 0.1, False, "wrong_direction"),
    ], (1, 2))
    overall = report["overall"]
    assert overall["top1_success_rate"]["value"] == 0.0
    assert overall["random_expected_success_rate"] == 1.0 / 3.0
    assert overall["oracle_success_rate"]["value"] == 1.0
    assert overall["success_at_k"]["2"]["value"] == 1.0
    assert overall["selected_failure_modes"] == {"overshoot": 1}


def test_baseline_report_compares_all_candidate_scores() -> None:
    records = [
        _record("success", 0.8, True),
        _record("failure", 0.2, False, "undershoot"),
    ]
    for record in records:
        success = bool(record["success"])
        record.update({
            "pixel_goal_similarity": 1.0 if success else 0.0,
            "visual_token_goal_cosine": 1.0 if success else 0.0,
            "visual_directional_mi": 1.0 if success else 0.0,
            "privileged_process_score": 1.0 if success else 0.0,
            "privileged_teacher_score": 2.0 if success else 0.0,
            "oracle_score": 2.0 if success else 0.0,
        })
    report = _baseline_report(records, (1,))
    assert report["methods"]["student"]["candidate_selection"]["overall"]["top1_success_rate"]["value"] == 1.0
    assert report["methods"]["pixel_goal_similarity"]["success_failure_auc"] == 1.0
    assert report["methods"]["random"]["candidate_selection"]["overall"]["expected_top1_success_rate"] == 0.5
