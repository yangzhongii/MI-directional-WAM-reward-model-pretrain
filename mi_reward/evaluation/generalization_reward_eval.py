"""Independent evaluation for the deployable VisualGoalPotential student.

The evaluator deliberately uses measured simulator outcomes and relation
sidecars as ground truth.  Teacher scores are only used to define the existing
preference-pair diagnostic and never define success, task progress, or Top-k
candidate-selection success.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.inference.visual_goal_model import VisualGoalInferenceModel
from mi_reward.relations.sequence import load_relation_sequence, relation_progress_potential


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _wilson_interval(correct: int, count: int, z: float = 1.96) -> list[float] | None:
    if count <= 0:
        return None
    rate = correct / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(correct: int, count: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "count": count,
        "value": correct / count if count else None,
        "ci95_wilson": _wilson_interval(correct, count),
    }


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = x.norm() * y.norm()
    if float(denominator) <= 1e-12:
        return None
    return float((x @ y / denominator).item())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    sorted_values = values[order]
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and float(sorted_values[end]) == float(sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    return _pearson(_average_ranks(x.float()), _average_ranks(y.float()))


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    correct = 0.0
    for pos in positive:
        for neg in negative:
            correct += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return correct / (len(positive) * len(negative))


def _trajectory_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["trajectory_score"]) for record in records]
    labels = [bool(record["success"]) for record in records]
    success = [record for record in records if record["success"]]
    failure = [record for record in records if not record["success"]]
    summary: dict[str, Any] = {
        "count": len(records),
        "success_count": len(success),
        "failure_count": len(failure),
        "success_failure_auc": _auc(scores, labels),
        "success_mean_score": _mean(float(record["trajectory_score"]) for record in success),
        "failure_mean_score": _mean(float(record["trajectory_score"]) for record in failure),
        "success_mean_endpoint_gain": _mean(float(record["endpoint_gain"]) for record in success),
        "failure_mean_endpoint_gain": _mean(float(record["endpoint_gain"]) for record in failure),
        "mean_positive_transition_ratio": _mean(
            float(record["positive_transition_ratio"]) for record in records
        ),
        "mean_progress_pearson": _mean(
            float(record["progress_pearson"])
            for record in records
            if record["progress_pearson"] is not None
        ),
        "mean_progress_spearman": _mean(
            float(record["progress_spearman"])
            for record in records
            if record["progress_spearman"] is not None
        ),
    }
    failure_auc: dict[str, float | None] = {}
    success_scores = [float(record["trajectory_score"]) for record in success]
    for mode in sorted({str(record["failure_mode"]) for record in failure}):
        mode_scores = [
            float(record["trajectory_score"])
            for record in failure
            if record["failure_mode"] == mode
        ]
        failure_auc[mode] = _auc(
            success_scores + mode_scores,
            [True] * len(success_scores) + [False] * len(mode_scores),
        )
    summary["success_vs_failure_mode_auc"] = failure_auc
    regression = [record for record in failure if record["failure_mode"] == "regress_after_progress"]
    summary["regress_after_progress_detection"] = _rate(
        sum(bool(record["regression_detected"]) for record in regression),
        len(regression),
    )
    return summary


def _pair_summary(
    pairs: list[PreferencePair],
    score_by_id: dict[str, float],
    example_by_id: dict[str, TrajectoryExample],
) -> dict[str, Any]:
    counters: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    margins: list[float] = []
    outcome_correct = 0
    outcome_count = 0
    for pair in pairs:
        if pair.chosen_traj_id not in score_by_id or pair.rejected_traj_id not in score_by_id:
            continue
        margin = score_by_id[pair.chosen_traj_id] - score_by_id[pair.rejected_traj_id]
        correct = int(margin > 0.0)
        margins.append(margin)
        chosen = example_by_id[pair.chosen_traj_id]
        rejected = example_by_id[pair.rejected_traj_id]
        task = str(chosen.task_family or chosen.task)
        comparison = str(pair.comparison_type or "unknown")
        for key in ("overall", f"split/{chosen.split}", f"task/{task}", f"type/{comparison}"):
            counters[key][0] += correct
            counters[key][1] += 1
        chosen_success = bool(chosen.task_outcome and chosen.task_outcome.success)
        rejected_success = bool(rejected.task_outcome and rejected.task_outcome.success)
        if chosen_success != rejected_success:
            outcome_count += 1
            predicted_chosen = margin > 0.0
            outcome_correct += int(predicted_chosen == chosen_success)
    return {
        "teacher_preference_accuracy": {
            key: _rate(correct, count) for key, (correct, count) in sorted(counters.items())
        },
        "measured_outcome_pair_accuracy": _rate(outcome_correct, outcome_count),
        "mean_model_margin": _mean(margins),
    }


def _context_key(example: TrajectoryExample) -> str:
    metadata = example.metadata or {}
    instance_id = metadata.get("instance_variant_id")
    if instance_id is None and example.instance_variant is not None:
        instance_id = example.instance_variant.variant_id
    scene_id = metadata.get("scene_variant_id")
    if scene_id is None and example.scene_variant is not None:
        scene_id = example.scene_variant.variant_id
    return "|".join(
        str(value)
        for value in (
            example.task_family or example.task,
            example.goal_ref_id,
            example.split,
            example.parent_traj_id,
            metadata.get("initial_state_seed"),
            instance_id,
            scene_id,
        )
    )


def _candidate_selection(
    records: list[dict[str, Any]],
    top_ks: tuple[int, ...],
    *,
    score_field: str = "trajectory_score",
) -> dict[str, Any]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        contexts[str(record["context_id"])].append(record)

    def summarize(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
        top1_correct = 0
        oracle_correct = 0
        random_expected = 0.0
        topk_correct = {k: 0 for k in top_ks}
        selected_failures: Counter[str] = Counter()
        for candidates in groups:
            ranked = sorted(candidates, key=lambda record: float(record[score_field]), reverse=True)
            selected = ranked[0]
            has_success = any(bool(record["success"]) for record in candidates)
            top1_correct += int(bool(selected["success"]))
            oracle_correct += int(has_success)
            random_expected += sum(bool(record["success"]) for record in candidates) / len(candidates)
            for k in top_ks:
                topk_correct[k] += int(any(bool(record["success"]) for record in ranked[:k]))
            if not selected["success"]:
                selected_failures[str(selected["failure_mode"])] += 1
        count = len(groups)
        return {
            "contexts": count,
            "top1_success_rate": _rate(top1_correct, count),
            "random_expected_success_rate": random_expected / count if count else None,
            "oracle_success_rate": _rate(oracle_correct, count),
            "binary_oracle_regret": (oracle_correct - top1_correct) / count if count else None,
            "success_at_k": {str(k): _rate(topk_correct[k], count) for k in top_ks},
            "selected_failure_modes": dict(sorted(selected_failures.items())),
        }

    grouped = list(contexts.values())
    report: dict[str, Any] = {"overall": summarize(grouped)}
    tasks = sorted({str(record["task_family"]) for record in records})
    report["by_task"] = {
        task: summarize(
            [group for group in grouped if str(group[0]["task_family"]) == task]
        )
        for task in tasks
    }
    report["context_records"] = [
        {
            "context_id": context_id,
            "task_family": candidates[0]["task_family"],
            "candidate_count": len(candidates),
            "selected_traj_id": max(
                candidates, key=lambda record: float(record[score_field])
            )["traj_id"],
            "selected_success": bool(
                max(candidates, key=lambda record: float(record[score_field]))["success"]
            ),
        }
        for context_id, candidates in sorted(contexts.items())
    ]
    return report


def _load_score_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Directional score cache is missing: {source}")
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid score cache JSON at {source}:{line_number}") from exc
        score = payload.get("score", payload) if isinstance(payload, dict) else None
        if isinstance(score, dict) and score.get("traj_id"):
            scores[str(score["traj_id"])] = score
    return scores


def _load_pixel_tensor(path: str | Path, image_size: int = 64) -> torch.Tensor:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def _static_baseline_summary(
    records: list[dict[str, Any]],
    score_field: str,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    scores = [float(record[score_field]) for record in records]
    labels = [bool(record["success"]) for record in records]
    selection = _candidate_selection(records, top_ks, score_field=score_field)
    selection.pop("context_records", None)
    return {
        "score_field": score_field,
        "success_failure_auc": _auc(scores, labels),
        "candidate_selection": selection,
    }


def _baseline_report(
    records: list[dict[str, Any]],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    methods = {
        "student": _static_baseline_summary(records, "trajectory_score", top_ks),
        "pixel_goal_similarity": _static_baseline_summary(records, "pixel_goal_similarity", top_ks),
        "visual_token_goal_cosine": _static_baseline_summary(
            records, "visual_token_goal_cosine", top_ks
        ),
        "visual_directional_mi": _static_baseline_summary(
            records, "visual_directional_mi", top_ks
        ),
        "privileged_process_teacher": _static_baseline_summary(
            records, "privileged_process_score", top_ks
        ),
        "privileged_outcome_teacher": _static_baseline_summary(
            records, "privileged_teacher_score", top_ks
        ),
        "oracle": _static_baseline_summary(records, "oracle_score", top_ks),
    }
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        contexts[str(record["context_id"])].append(record)
    methods["random"] = {
        "success_failure_auc": 0.5,
        "candidate_selection": {
            "overall": {
                "contexts": len(contexts),
                "expected_top1_success_rate": _mean(
                    sum(bool(item["success"]) for item in group) / len(group)
                    for group in contexts.values()
                ),
            }
        },
    }
    return {
        "score_direction": "higher_is_better",
        "privileged_teacher_warning": (
            "Privileged teacher methods consume offline action/state/relation and possibly outcome labels; "
            "they are diagnostic upper bounds, not deployable rewards."
        ),
        "methods": methods,
    }


def _score_batch(
    model: VisualGoalInferenceModel,
    examples: list[TrajectoryExample],
    store: CachedFeatureStore,
    goal_cache: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    trajectories = [store.load(example.traj_id + "_tokens").float() for example in examples]
    if any(example.goal_ref_id is None for example in examples):
        raise ValueError("Every test trajectory must define goal_ref_id.")
    goals: list[torch.Tensor] = []
    for example in examples:
        assert example.goal_ref_id is not None
        if example.goal_ref_id not in goal_cache:
            goal_cache[example.goal_ref_id] = store.load(example.goal_ref_id + "_tokens").float()[-1]
        goals.append(goal_cache[example.goal_ref_id])
    max_steps = max(tokens.shape[0] for tokens in trajectories)
    patches = trajectories[0].shape[1]
    dim = trajectories[0].shape[2]
    if any(tokens.ndim != 3 or tokens.shape[1:] != (patches, dim) for tokens in trajectories):
        raise ValueError("Visual token shapes differ inside an evaluation batch.")
    state = torch.zeros((len(examples), max_steps, patches, dim), dtype=torch.float32)
    mask = torch.zeros((len(examples), max_steps), dtype=torch.bool)
    for index, tokens in enumerate(trajectories):
        state[index, : tokens.shape[0]] = tokens
        mask[index, : tokens.shape[0]] = True
    goal = torch.stack(goals)
    goal_mask = torch.ones(goal.shape[:2], dtype=torch.bool)
    potentials = model.predict_potential(state, goal, mask, goal_mask).cpu()
    return [potentials[index, : tokens.shape[0]].clone() for index, tokens in enumerate(trajectories)]


def evaluate_generalization_reward(
    *,
    manifest: str | Path,
    success_refs: str | Path,
    preferences: str | Path,
    feature_root: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    score_cache: str | Path | None = None,
    test_splits: tuple[str, ...] = ("joint_heldout",),
    batch_size: int = 4,
    device: str = "cuda",
    top_ks: tuple[int, ...] = (1, 3),
    strict_split_isolation: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    all_examples = read_jsonl(manifest, TrajectoryExample)
    examples = [example for example in all_examples if example.split in test_splits]
    if not examples:
        raise ValueError(f"No trajectories found for test splits {test_splits}.")
    references = read_jsonl(success_refs, SuccessReference)
    reference_by_id = {reference.ref_id: reference for reference in references}
    reference_ids = set(reference_by_id)
    unknown_goals = sorted(
        {
            str(example.goal_ref_id)
            for example in examples
            if example.goal_ref_id is None or example.goal_ref_id not in reference_ids
        }
    )
    if unknown_goals:
        raise ValueError(f"Test trajectories contain unknown success references: {unknown_goals[:5]}.")
    missing_outcomes = [example.traj_id for example in examples if example.task_outcome is None]
    missing_relations = [example.traj_id for example in examples if not example.relation_path]
    if missing_outcomes or missing_relations:
        raise ValueError(
            "Independent evaluation requires measured task_outcome and relation_path for every trajectory; "
            f"missing outcomes={missing_outcomes[:3]}, missing relations={missing_relations[:3]}."
        )

    model = VisualGoalInferenceModel.from_pretrained(checkpoint, device=device)
    checkpoint_train = set(str(value) for value in model.config.get("train_splits", []))
    checkpoint_validation = set(str(value) for value in model.config.get("validation_splits", []))
    overlap = sorted(set(test_splits) & (checkpoint_train | checkpoint_validation))
    isolation = {
        "valid": not overlap,
        "test_splits": list(test_splits),
        "checkpoint_train_splits": sorted(checkpoint_train),
        "checkpoint_validation_splits": sorted(checkpoint_validation),
        "overlap": overlap,
    }
    if overlap and strict_split_isolation:
        raise ValueError(
            "Final-test split isolation failed: checkpoint used test split(s) "
            f"{overlap} for training or model selection. Retrain before final evaluation."
        )

    store = CachedFeatureStore(feature_root)
    goal_cache: dict[str, torch.Tensor] = {}
    goal_pixel_cache: dict[str, torch.Tensor] = {}
    directional_scores = _load_score_cache(
        score_cache
        or Path(preferences).with_name(Path(preferences).stem + ".scores.jsonl")
    )
    records: list[dict[str, Any]] = []
    gamma = float(model.config.get("gamma", 0.99))
    for start in tqdm(range(0, len(examples), batch_size), desc="Independent reward test", unit="batch"):
        batch_examples = examples[start : start + batch_size]
        curves = _score_batch(model, batch_examples, store, goal_cache)
        for example, potential in zip(batch_examples, curves):
            assert (
                example.task_outcome is not None
                and example.relation_path is not None
                and example.goal_ref_id is not None
            )
            score_artifact = directional_scores.get(example.traj_id)
            if score_artifact is None:
                raise ValueError(f"Score cache has no baseline record for {example.traj_id}.")
            goal_reference = reference_by_id[example.goal_ref_id]
            if not example.frames or not goal_reference.frames:
                raise ValueError(f"Pixel baseline requires candidate and goal frames for {example.traj_id}.")
            candidate_pixel = _load_pixel_tensor(example.frames[-1])
            if example.goal_ref_id not in goal_pixel_cache:
                goal_pixel_cache[example.goal_ref_id] = _load_pixel_tensor(goal_reference.frames[-1])
            goal_pixel = goal_pixel_cache[example.goal_ref_id]
            pixel_goal_similarity = -float(F.mse_loss(candidate_pixel, goal_pixel).item())
            candidate_tokens = store.load(example.traj_id + "_tokens").float()[-1].mean(dim=0)
            goal_tokens = goal_cache[example.goal_ref_id].mean(dim=0)
            token_goal_cosine = float(
                F.cosine_similarity(candidate_tokens.unsqueeze(0), goal_tokens.unsqueeze(0)).item()
            )
            relation = load_relation_sequence(example.relation_path)
            oracle = relation_progress_potential(
                relation.values,
                relation.names,
                task_family=example.task_family,
            )
            if oracle.shape[0] != potential.shape[0]:
                raise ValueError(
                    f"Potential/relation length mismatch for {example.traj_id}: "
                    f"{potential.shape[0]} versus {oracle.shape[0]}."
                )
            transition_reward = gamma * potential[1:] - potential[:-1]
            oracle_peak = int(torch.argmax(oracle).item())
            regression_detected = bool(
                oracle_peak < potential.numel() - 1
                and float(potential[oracle_peak] - potential[-1]) >= 0.02
            )
            records.append(
                {
                    "traj_id": example.traj_id,
                    "split": example.split,
                    "task": example.task,
                    "task_family": example.task_family or example.task,
                    "context_id": _context_key(example),
                    "success": example.task_outcome.success,
                    "failure_mode": example.task_outcome.failure_mode,
                    "candidate_profile": example.task_outcome.candidate_profile,
                    "terminal_distance": example.task_outcome.terminal_distance,
                    "trajectory_score": float(transition_reward.mean().item()),
                    "terminal_potential": float(potential[-1].item()),
                    "endpoint_gain": float((potential[-1] - potential[0]).item()),
                    "positive_transition_ratio": float((transition_reward > 0).float().mean().item()),
                    "progress_pearson": _pearson(potential, oracle),
                    "progress_spearman": _spearman(potential, oracle),
                    "regression_detected": regression_detected,
                    "pixel_goal_similarity": pixel_goal_similarity,
                    "visual_token_goal_cosine": token_goal_cosine,
                    "visual_directional_mi": float(score_artifact["visual_score"]),
                    "privileged_process_score": float(score_artifact["raw_score_delta"]),
                    "privileged_teacher_score": float(score_artifact["score_delta"]),
                    "oracle_score": (
                        float(example.task_outcome.success) * 2.0
                        - float(example.task_outcome.terminal_distance or 0.0)
                    ),
                }
            )

    example_by_id = {example.traj_id: example for example in examples}
    score_by_id = {str(record["traj_id"]): float(record["trajectory_score"]) for record in records}
    pairs = [
        pair
        for pair in read_jsonl(preferences, PreferencePair)
        if pair.split in test_splits
        and pair.chosen_traj_id in example_by_id
        and pair.rejected_traj_id in example_by_id
    ]
    by_split = {
        split: _trajectory_summary([record for record in records if record["split"] == split])
        for split in test_splits
    }
    tasks = sorted({str(record["task_family"]) for record in records})
    by_task = {
        task: _trajectory_summary([record for record in records if record["task_family"] == task])
        for task in tasks
    }
    report = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "ground_truth": "measured_mujoco_task_outcome_and_relations",
        "split_isolation": isolation,
        "trajectory_metrics": {
            "overall": _trajectory_summary(records),
            "by_split": by_split,
            "by_task": by_task,
        },
        "pair_metrics": _pair_summary(pairs, score_by_id, example_by_id),
        "candidate_selection": _candidate_selection(records, top_ks),
        "candidate_selection_baselines": _baseline_report(records, top_ks),
        "records": records,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    printable = dict(report)
    printable.pop("records")
    printable["candidate_selection"] = dict(report["candidate_selection"])
    printable["candidate_selection"].pop("context_records", None)
    printable["candidate_selection_baselines"] = {
        "methods": {
            name: {
                "success_failure_auc": method.get("success_failure_auc"),
                "top1_success_rate": (
                    method.get("candidate_selection", {})
                    .get("overall", {})
                    .get("top1_success_rate", {})
                    .get("value", method.get("candidate_selection", {}).get("overall", {}).get("expected_top1_success_rate"))
                ),
            }
            for name, method in report["candidate_selection_baselines"]["methods"].items()
        }
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent held-out evaluation for VisualGoalPotential.")
    parser.add_argument("--config", default="mi_reward/configs/generalization_reward.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-split", action="append", dest="test_splits", default=None)
    parser.add_argument("--allow-split-overlap", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = dict(config.get("paths") or {})
    evaluation = dict(config.get("evaluation") or {})
    test_splits = tuple(args.test_splits or evaluation.get("test_splits", ["joint_heldout"]))
    top_ks = tuple(int(value) for value in evaluation.get("top_k", [1, 3]))
    evaluate_generalization_reward(
        manifest=paths["manifest"],
        success_refs=paths["success_refs"],
        preferences=paths["preferences"],
        score_cache=evaluation.get("score_cache"),
        feature_root=paths["feature_root"],
        checkpoint=args.checkpoint or evaluation.get("checkpoint") or str(Path(paths["output_dir"]) / "pytorch_model.pt"),
        output=args.output or evaluation.get("output") or str(Path(paths["output_dir"]) / "joint_heldout_eval.json"),
        test_splits=test_splits,
        batch_size=args.batch_size or int(evaluation.get("batch_size", 4)),
        device=args.device or str(evaluation.get("device", "cuda")),
        top_ks=top_ks,
        strict_split_isolation=bool(evaluation.get("strict_split_isolation", True)) and not args.allow_split_overlap,
    )


if __name__ == "__main__":
    main()
