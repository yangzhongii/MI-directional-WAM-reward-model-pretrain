"""Read-only v3 teacher audit: cached sampling and real rollout window attribution."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mi_reward.data.export_teacher_targets_v3 import _component_disagreement_strength, _score_episode
from mi_reward.scoring.information_teacher_v3 import VisualPointwiseInformationCritic, ConditionalActionInformationCritic
from mi_reward.scoring.teacher_consistency_v3 import PhysicalConsistencyEvidence, project_teacher_label

BASE = Path("logs/mi_reward/v3_teacher_mainline")
LABELS = ("Positive", "Unclear", "Negative")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if len(values) else None


def classification(truth, predictions):
    matrix = {label: {p: 0 for p in LABELS} for label in LABELS}
    for target, predicted in zip(truth, predictions):
        matrix[target][predicted] += 1
    recalls, f1s = {}, []
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[t][label] for t in LABELS)
        recalls[label] = tp / support if support else None
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0)
    nonforward = sum(t != "Positive" for t in truth)
    return {"count": len(truth), "accuracy": sum(a == b for a, b in zip(truth, predictions)) / len(truth) if truth else None,
            "macro_f1": sum(f1s) / 3, "recall": recalls, "confusion_matrix": matrix,
            "balanced_accuracy_present_classes": np.mean([r for r in recalls.values() if r is not None]).item() if truth else None,
            "nonforward_false_positive_rate": sum(t != "Positive" and p == "Positive" for t, p in zip(truth, predictions)) / nonforward if nonforward else None}


def signed_metrics(scores, directions):
    keep = np.asarray(directions) != 0
    truth = ["Positive" if d > 0 else "Negative" for d in np.asarray(directions)[keep]]
    predicted = ["Positive" if s > 0 else "Negative" for s in np.asarray(scores)[keep]]
    result = classification(truth, predicted)
    # This is a binary direction test; do not average in an absent neutral class.
    result["macro_f1_three_labels"] = result["macro_f1"]
    matrix = result["confusion_matrix"]
    result["macro_f1"] = sum(
        2 * matrix[label][label] / max(1, sum(matrix[label].values()) + sum(matrix[t][label] for t in LABELS))
        for label in ("Positive", "Negative")) / 2
    return result


def live_trace_physics(trajectory, trace, selected, bddl):
    """Replay original actions to retain controller/contact state absent in qpos/qvel snapshots."""
    from libero.libero.envs import OffScreenRenderEnv
    from mi_reward.data.libero_privileged import goal_objects
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"],
                             camera_heights=128, camera_widths=128)
    frames = {}
    checks = {}
    try:
        env.seed(trajectory["seed"])
        env.reset()
        env.set_init_state(trace["initial_state"])
        core = env.env
        _, task_object, goal_object = goal_objects(core.parsed_problem["goal_state"])
        task_state = core.object_states_dict[task_object]
        goal_state = core.object_states_dict[goal_object]
        task_model = core.get_object(task_object)
        for step, applied in enumerate(trace["actions"]):
            _, reward, _, _ = env.step(applied.tolist())
            index = step + 1
            if index not in selected:
                continue
            success = bool(env.check_success())
            state_error = float(np.max(np.abs(env.get_sim_state() - trace["states"][step])))
            consistent = state_error < 1e-6 and success == bool(trace["environment_success"][step])
            checks[index] = {"max_state_error": state_error, "success_matches": success == bool(trace["environment_success"][step]),
                             "consistent": consistent}
            task_geom, goal_geom = task_state.get_geom_state(), goal_state.get_geom_state()
            obj = np.asarray(task_geom["pos"], dtype=float)
            goal = np.asarray(goal_geom["pos"], dtype=float)
            eef = np.asarray(core._eef_xpos, dtype=float)
            delta = obj - goal
            frames[index] = {"frame_index": index, "action": applied.tolist(),
                             "task_object_pos": obj.tolist(), "goal_object_pos": goal.tolist(), "eef_pos": eef.tolist(),
                             "object_goal_delta_xyz": delta.tolist(), "object_goal_distance_xyz": float(np.linalg.norm(delta)),
                             "object_goal_distance_xy": float(np.linalg.norm(delta[:2])),
                             "eef_object_distance_xyz": float(np.linalg.norm(eef - obj)),
                             "grasped": bool(core._check_grasp(core.robots[0].gripper, task_model)),
                             "object_goal_contact": bool(task_state.check_contact(goal_state)),
                             "environment_success": success, "recorded_reward": float(reward),
                             "recorded_done": index == len(trace["actions"])}
        return frames, checks
    finally:
        env.close()


def load_teacher():
    payload = torch.load(BASE / "teacher_smoke5/information_teacher_v3.pt", map_location="cpu", weights_only=False)
    visual = VisualPointwiseInformationCritic(payload["visual_dim"], payload["visual_dim"]).eval()
    action = ConditionalActionInformationCritic(payload["action_dim"], payload["visual_dim"], payload["visual_dim"]).eval()
    visual.load_state_dict(payload["visual_critic"])
    action.load_state_dict(payload["action_critic"])
    for model in (visual, action):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    calibration = json.loads((BASE / "teacher_export5_v2/calibration.json").read_text())
    return visual, action, calibration


def project_frozen(score, delta, psi_scaled, evidence, calibration):
    decision = project_teacher_label(score, evidence, deadband=calibration["directional_deadband"], use_relation_direction=True)
    label, reason = decision.label, None
    strength = _component_disagreement_strength(delta, psi_scaled,
              left_deadband=calibration["visual_component_deadband"], right_deadband=calibration["action_component_deadband"])
    threshold = calibration.get("component_disagreement_strength_threshold")
    if not decision.authoritative_override and label in ("Positive", "Negative"):
        if strength > 0 and threshold is not None and strength >= threshold:
            label, reason = "Unclear", "high_confidence_visual_action_disagreement"
        elif decision.confidence < calibration["confidence_threshold"]:
            label, reason = "Unclear", "low_confidence"
    return label, reason, decision


@torch.inference_mode()
def cached_audit(output):
    visual, action, calibration = load_teacher()
    episodes = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted((BASE / "features").glob("task_*_demo_*.pt"))]
    train = [ep for ep in episodes if ep["split"] == "train"]
    validation = [ep for ep in episodes if ep["split"] == "validation"]
    scores, directions = defaultdict(list), []
    all_rows = []
    for ep in validation:
        values = _score_episode(ep, visual, action, device=torch.device("cpu"), gamma=calibration["gamma"], beta=calibration["beta"])
        cosine = F.cosine_similarity(ep["visual"], ep["goal"].unsqueeze(0), dim=-1)
        episode_scores = {"cosine": (cosine[1:] - cosine[:-1]).tolist(), "delta_phi": values["delta_phi"].tolist(),
                          "psi": values["psi"].tolist(), "D": values["directional_score"].tolist()}
        directions.extend(ep["direction"].tolist())
        for name, values_list in episode_scores.items():
            scores[name].extend(values_list)
        for i, direction in enumerate(ep["direction"].tolist()):
            all_rows.append({"task_id": ep["task_id"], "demo_name": ep["demo_name"], "transition_index": i,
                             "sampler_physical_direction": direction, **{name: vals[i] for name, vals in episode_scores.items()}})
    metrics = {name: signed_metrics(values, directions) for name, values in scores.items()}
    metrics["always_forward"] = signed_metrics(np.ones(len(directions)), directions)
    sampling = []
    for task in sorted({ep["task_id"] for ep in train}):
        task_episodes = [ep for ep in train if ep["task_id"] == task]
        states = torch.cat([ep["visual"][:-1] for ep in task_episodes])
        direction = torch.cat([ep["direction"] for ep in task_episodes])
        pos, neg = states[direction > 0], states[direction <= 0]
        similarity = F.normalize(pos, dim=-1) @ F.normalize(neg, dim=-1).T
        best, indices = similarity.max(dim=1)
        visual_positive, visual_negative = [], []
        for ep in task_episodes:
            alternate = next(other for other in task_episodes if other["demo_name"] != ep["demo_name"])
            visual_positive.extend(visual(ep["visual"], ep["goal"].expand(len(ep["visual"]), -1)).tolist())
            visual_negative.extend(visual(ep["visual"], alternate["goal"].expand(len(ep["visual"]), -1)).tolist())
        sampling.append({"task_id": task, "training_demos": [ep["demo_name"] for ep in task_episodes],
                         "direction_counts": dict(Counter(str(d) for d in direction.tolist())),
                         "positive_action_candidates": len(pos), "negative_action_candidates": len(neg),
                         "unique_selected_negatives": len(set(indices.tolist())),
                         "selected_negative_directions": dict(Counter(str(d) for d in direction[direction <= 0][indices].tolist())),
                         "nearest_negative_visual_cosine": quantiles(best.tolist()),
                         "visual_matched_goal_logits": quantiles(visual_positive),
                         "visual_other_successful_episode_goal_logits": quantiles(visual_negative)})
    diagnostics = [json.loads(line) for line in (BASE / "teacher_export5_v2/teacher_diagnostics.jsonl").read_text().splitlines()]
    residuals, label_mismatches = [], []
    for row in diagnostics:
        reconstructed = row["gamma"] * row["phi_visual_t1"] - row["phi_visual_t"] + row["beta"] * row["conditional_action_information"]
        residuals.append(abs(reconstructed - row["directional_score"]))
        frames = row["privileged_diagnostics"]
        evidence = PhysicalConsistencyEvidence.from_frames(frames["frame_t"], frames["frame_t1"])
        label, _, _ = project_frozen(row["directional_score"], row["delta_phi_visual"], row["beta_scaled_action_information"], evidence, calibration)
        if label != row["P/U/N_target"]:
            label_mismatches.append(row["sample_id"])
    report = {"checkpoint": str(BASE / "teacher_smoke5/information_teacher_v3.pt"), "fixed_calibration": calibration,
              "validation_direction_counts": dict(Counter(str(d) for d in directions)), "direction_metrics_excluding_neutral": metrics,
              "sampling_by_task": sampling,
              "formula_reconstruction": {"rows": len(residuals), "max_absolute_residual": max(residuals), "tolerance": 1e-5},
              "label_projection_reproduction": {"rows": len(diagnostics), "mismatches": label_mismatches},
              "distribution_findings": [
                  "Visual negatives are other successful episode goals within the same task, not unconditional marginal goals.",
                  "Action positive population is restricted to physical forward transitions.",
                  "Action negatives are nearest visually matched nonforward transitions, not samples demonstrated to follow p(a|v).",
                  "No sampling-density correction is applied. A density-ratio classifier estimates its constructed distributions; exact target PMI is not established.",
                  "Physical direction used here is the sampler/gate heuristic, not independent ground truth.",
                  "Thresholds remain train-neutral calibrated, not recalibrated on held-out data during this audit."]}
    write_json(output / "cached_teacher_audit.json", report)
    (output / "cached_validation_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    print(json.dumps({"cached_direction_metrics": metrics, "formula": report["formula_reconstruction"],
                      "projection_mismatches": len(label_mismatches)}, indent=2), flush=True)
    if max(residuals) > 1e-5 or label_mismatches:
        raise RuntimeError("Frozen teacher formula/projection reproduction mismatch")


@torch.inference_mode()
def rollout_audit(output):
    from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor
    from mi_reward.data.libero_privileged import resolve_libero_task, classify_physical_transition
    visual, action, calibration = load_teacher()
    torch.manual_seed(7)
    extractor = LaWAMLAMFeatureExtractor(
        lam_config_path=".venv/models/lawam_lam/dino_large_vae.yaml",
        lam_ckpt_path=".venv/models/lawam_lam/checkpoints/pytorch_model.pt",
        vision_model_id=".venv/models/dinov3-vitb16-pretrain-lvd1689m", device="cuda", strict=True)
    print(f"LAM loaded: {extractor.using_lam}; action history={extractor._latent_action_window()}", flush=True)
    if extractor._latent_action_window() != 2:
        raise ValueError("Audit endpoint extraction assumes the frozen two-frame LAM config")
    source = BASE / "validation_audit_v1"
    manifest = [json.loads(line) for line in (source / "rollouts/manifest.jsonl").read_text().splitlines()]
    inputs = [json.loads(line) for line in (source / "validation/inputs.jsonl").read_text().splitlines()]
    predictions = [json.loads(line) for line in (source / "validation/predictions.jsonl").read_text().splitlines()]
    prediction_by_id = {(r["trajectory_id"], r["window_end"]): r for r in predictions}
    grouped = defaultdict(list)
    for record in inputs:
        grouped[record["trajectory_id"]].append(record)
    goal_cache = {}
    rows = []
    with (output / "rollout_window_audit.jsonl").open("x") as handle:
        for trajectory in manifest:
            task = trajectory["task_index"]
            records = grouped[trajectory["trajectory_id"]]
            trace = np.load(source / "rollouts" / trajectory["trace"])
            selected = sorted({i for record in records for i in record["row"]["provenance"]["history_frame_indices"][-2:]})
            _, bddl, language = resolve_libero_task(Path.cwd(), "libero_spatial", task)
            frame_by_id, replay_checks = live_trace_physics(trajectory, trace, selected, bddl)
            # A successful training reference is available for seen tasks. For
            # unseen tasks use the official demo_0 terminal, declared as an
            # offline reference rather than a goal derived from a failed rollout.
            if task not in goal_cache:
                cache = BASE / f"features/task_{task:02d}_demo_0.pt"
                if cache.exists():
                    goal_cache[task] = torch.load(cache, map_location="cpu", weights_only=False)["goal"]
                else:
                    import h5py
                    with h5py.File(trajectory["source_demo"], "r") as demo_handle:
                        demo = demo_handle["data/demo_0"]
                        terminal_index = next((i for i, reward in enumerate(demo["rewards"][:]) if reward > 0), None)
                        if terminal_index is None:
                            raise ValueError("No successful reference goal")
                        views = []
                        for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                            images = list(demo["obs"][camera][terminal_index:terminal_index + 5])
                            views.append(extractor.extract_trajectory_tokens_images(images, language).float().mean(dim=1).mean(dim=0))
                        goal_cache[task] = torch.cat(views)
            goal = goal_cache[task]
            pooled = []
            for camera in ("agentview", "wrist"):
                paths = [str(source / "rollouts" / trajectory[camera][i]) for i in selected]
                pooled.append(extractor.extract_trajectory_tokens(paths, language).float().mean(dim=1))
            state_features = torch.cat(pooled, dim=-1)
            features_by_id = dict(zip(selected, state_features))
            for record in records:
                left, right = record["row"]["provenance"]["history_frame_indices"][-2:]
                action_paths = [str(source / "rollouts" / trajectory["agentview"][i]) for i in (left, right)]
                latent = extractor.extract_action_latents(action_paths, language).float().mean(dim=1)[0]
                episode = {"visual": torch.stack([features_by_id[left], features_by_id[right]]),
                           "action": latent[None], "goal": goal}
                values = _score_episode(episode, visual, action, device=torch.device("cpu"), gamma=calibration["gamma"], beta=calibration["beta"])
                delta, psi, score = (float(values[key][0]) for key in ("delta_phi", "psi", "directional_score"))
                evidence = PhysicalConsistencyEvidence.from_frames(frame_by_id[left], frame_by_id[right])
                teacher, reason, decision = project_frozen(score, delta, calibration["beta"] * psi, evidence, calibration)
                direction = classify_physical_transition(frame_by_id[left], frame_by_id[right])
                predicted = prediction_by_id[(record["trajectory_id"], record["window_end"])]
                row = {
                    "trajectory_id": trajectory["trajectory_id"], "task_id": task, "policy_id": trajectory["policy_id"],
                    "episode_success": trajectory["success"], "window_end": record["window_end"],
                    "physical_replay_consistent": replay_checks[left]["consistent"] and replay_checks[right]["consistent"],
                    "replay_checks": {"left": replay_checks[left], "right": replay_checks[right]},
                    "frame_t": left, "frame_t1": right, "phi_t": float(values["phi"][0]), "phi_t1": float(values["phi"][1]),
                    "delta_phi": delta, "psi": psi, "D": score, "physical_direction": direction,
                    "physical_label": "Positive" if direction > 0 else "Negative" if direction < 0 else "Unclear",
                    "teacher_label": teacher, "qwen_label": predicted["label"], "qwen_reward": predicted["reward"],
                    "teacher_decision": asdict(decision), "abstention_reason": reason, "evidence": asdict(evidence),
                    "physical_frame_t": frame_by_id[left], "physical_frame_t1": frame_by_id[right],
                    "goal_reference": "official demo_0 successful goal; cached training goal for tasks 0-4",
                    "goal_reference_differs_from_source_episode": True,
                    "images": record["row"]["images"],
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print(f"Audited {trajectory['trajectory_id']}: {len(records)} windows; total {len(rows)}", flush=True)
    valid_rows = [r for r in rows if r["physical_replay_consistent"]]
    report = {"windows": len(rows), "physical_replay_valid_windows": len(valid_rows),
              "physical_replay_excluded_windows": len(rows) - len(valid_rows), "fixed_calibration": calibration,
              "scope": "Frozen shared-reference teacher on existing rollout endpoints; no training or threshold tuning.",
              "limitations": ["Physical direction is the existing sampler heuristic, not independent annotated truth.",
                              "Teacher uses successful demo_0 goal; original teacher training used episode-specific goals.",
                              "Qwen input remains only task language and five-frame dual-view history.",
                              "Window labels target the last transition, matching teacher export semantics."]}
    for group_name, selected_rows in (
        ("all", valid_rows), ("failed_episodes", [r for r in valid_rows if not r["episode_success"]]),
        ("seen_tasks_0_4", [r for r in valid_rows if r["task_id"] < 5]),
        ("unseen_tasks_5_9", [r for r in valid_rows if r["task_id"] >= 5])):
        truth = [r["physical_label"] for r in selected_rows]
        report[group_name] = {
            "teacher_vs_physical_heuristic": classification(truth, [r["teacher_label"] for r in selected_rows]),
            "qwen_vs_physical_heuristic": classification(truth, [r["qwen_label"] for r in selected_rows]),
            "qwen_vs_teacher": classification([r["teacher_label"] for r in selected_rows], [r["qwen_label"] for r in selected_rows]),
            "qwen_positive_teacher_nonpositive": sum(r["qwen_label"] == "Positive" and r["teacher_label"] != "Positive" for r in selected_rows),
            "qwen_positive_physical_nonforward": sum(r["qwen_label"] == "Positive" and r["physical_direction"] <= 0 for r in selected_rows),
        }
    write_json(output / "rollout_teacher_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("cached", "rollout"), required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "cached":
        cached_audit(output)
    else:
        rollout_audit(output)


if __name__ == "__main__":
    main()
