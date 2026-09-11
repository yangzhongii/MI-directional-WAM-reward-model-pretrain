from __future__ import annotations

import json
from pathlib import Path

from mi_reward.experiments.generalization_ablations import (
    ABLATIONS,
    _repo_relative,
    build_ablation_config,
    generate_configs,
    summarize_results,
)


def _base_config() -> dict[str, object]:
    return {
        "run_id": "rigid_v3",
        "paths": {
            "manifest": "manifest.jsonl",
            "success_refs": "refs.jsonl",
            "feature_root": "features",
            "preferences": "preferences.jsonl",
            "teacher_targets": "targets.pt",
            "output_dir": "reward_model",
        },
        "training": {
            "action_weight": 1.0,
            "kinematic_weight": 1.0,
            "relation_weight": 1.0,
            "outcome_weight": 2.0,
            "directional_alignment": True,
            "validation_splits": ["instance_heldout", "scene_heldout"],
        },
        "evaluation": {"test_splits": ["joint_heldout"]},
    }


def test_ablation_configs_isolate_outputs_and_change_one_component(tmp_path: Path) -> None:
    config = build_ablation_config(
        _base_config(),
        name="no_action",
        seed=2,
        output_root=tmp_path / "runs",
    )
    assert config["training"]["action_weight"] == 0.0
    assert config["training"]["kinematic_weight"] == 1.0
    assert config["training"]["seed"] == 2
    assert config["paths"]["feature_root"] == "features"
    assert "no_action/seed-2" in config["paths"]["output_dir"]
    assert config["evaluation"]["strict_split_isolation"] is True


def test_interaction_ablations_enable_exactly_one_privileged_channel(tmp_path: Path) -> None:
    expected = {
        "visual_relation": (0.0, 0.0, 1.0),
        "visual_action": (1.0, 0.0, 0.0),
        "visual_kinematic": (0.0, 1.0, 0.0),
    }
    for name, weights in expected.items():
        config = build_ablation_config(
            _base_config(),
            name=name,
            seed=1,
            output_root=tmp_path / "runs",
        )
        training = config["training"]
        assert (
            training["action_weight"],
            training["kinematic_weight"],
            training["relation_weight"],
        ) == weights
        assert training["outcome_weight"] == 2.0
        assert training["directional_alignment"] is True


def test_generate_defaults_to_three_seeds_for_every_ablation(tmp_path: Path) -> None:
    import yaml

    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    paths = generate_configs(
        base_path,
        config_root=tmp_path / "configs",
        output_root=tmp_path / "runs",
    )
    assert len(paths) == len(ABLATIONS) * 3
    assert all(path.is_file() for path in paths)


def test_summary_writes_csv_and_aggregates_seeds(tmp_path: Path) -> None:
    for seed, auc in ((0, 0.7), (1, 0.9)):
        reward_dir = tmp_path / "full" / f"seed-{seed}" / "reward_model"
        reward_dir.mkdir(parents=True)
        report = {
            "split_isolation": {"valid": True},
            "trajectory_metrics": {"overall": {
                "success_failure_auc": auc,
                "mean_progress_pearson": 0.5,
                "mean_progress_spearman": 0.4,
                "regress_after_progress_detection": {"value": 0.3},
            }},
            "pair_metrics": {"measured_outcome_pair_accuracy": {"value": 0.8}},
            "candidate_selection": {"overall": {"top1_success_rate": {"value": 1.0}}},
        }
        (reward_dir / "joint_heldout_eval.json").write_text(json.dumps(report), encoding="utf-8")
        (reward_dir / "validation_metrics.json").write_text(
            json.dumps({"best_epoch": 2, "best_selection_score": 0.75}),
            encoding="utf-8",
        )
    summary = summarize_results(tmp_path)
    assert summary["run_count"] == 2
    assert summary["ablations"]["full"]["metrics"]["test_success_failure_auc"]["mean"] == 0.8
    assert (tmp_path / "ablation_runs.csv").is_file()
    assert (tmp_path / "ablation_summary.json").is_file()


def test_repo_relative_preserves_relative_paths_and_maps_absolute_paths(tmp_path: Path) -> None:
    assert _repo_relative("logs/run", tmp_path) == Path("logs/run")
    assert _repo_relative(tmp_path / "logs/run", tmp_path) == Path("logs/run")
