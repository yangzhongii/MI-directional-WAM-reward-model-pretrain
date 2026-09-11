"""Export calibrated Pipeline-v3 teacher targets and deployment-safe Qwen JSONL.

This module is intentionally downstream of the trained information critics.
It never retrains ``phi_v`` or ``psi_a`` and never adds privileged state to the
directional score

    D_t = gamma * phi_v(t+1) - phi_v(t) + beta * psi_a(t).

Privileged LIBERO replay is used only to validate / veto / override / abstain
before projecting the teacher judgment to Positive / Unclear / Negative.

Two artifacts are deliberately separated:

* ``qwen_<split>.jsonl`` contains only task language plus dual-view image
  history and the P/U/N assistant target;
* ``teacher_diagnostics.jsonl`` contains the continuous information scores,
  calibration diagnostics, and privileged evidence used to generate labels.

Keeping those files separate makes privileged-state leakage into the Qwen
student input structurally auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task
from mi_reward.scoring.information_teacher_v3 import (
    ConditionalActionInformationCritic,
    VisualPointwiseInformationCritic,
)
from mi_reward.scoring.teacher_consistency_v3 import (
    N_LABEL,
    P_LABEL,
    U_LABEL,
    PhysicalConsistencyEvidence,
    information_direction,
    project_teacher_label,
)


LABELS = (P_LABEL, U_LABEL, N_LABEL)
PRIVILEGED_FRAME_KEYS = (
    "task_object_pos",
    "goal_object_pos",
    "eef_pos",
    "object_goal_delta_xyz",
    "object_goal_distance_xyz",
    "object_goal_distance_xy",
    "eef_object_distance_xyz",
    "grasped",
    "object_goal_contact",
    "environment_success",
    "recorded_reward",
    "recorded_done",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--neutral-quantile", type=float, default=0.75)
    parser.add_argument(
        "--disagreement-quantile",
        type=float,
        default=0.95,
        help=(
            "Train-only quantile used to define unusually strong visual/action "
            "conflict among physically-forward, D-forward training transitions."
        ),
    )
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Cached episode splits to export. Calibration always uses train only.",
    )
    parser.add_argument(
        "--skip-image-export",
        action="store_true",
        help="Keep HDF5 frame references only. Formal Qwen JSONL normally exports JPEG frames.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty teacher export directory: {path}. "
            "Choose a new --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)


def _quantile(source: list[torch.Tensor], quantile: float, fallback: list[torch.Tensor]) -> float:
    values = torch.cat(source if source else fallback)
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), float(quantile)).item())


@torch.no_grad()
def _score_episode(
    episode: dict[str, Any],
    visual_critic: VisualPointwiseInformationCritic,
    action_critic: ConditionalActionInformationCritic,
    *,
    device: torch.device,
    gamma: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    visual = episode["visual"].to(device)
    action = episode["action"].to(device)
    goal = episode["goal"].to(device).unsqueeze(0)
    goal_frames = goal.expand(visual.shape[0], -1)

    phi = visual_critic(visual, goal_frames)
    psi = action_critic(action, visual[:-1], goal.expand(action.shape[0], -1))
    delta_phi = float(gamma) * phi[1:] - phi[:-1]
    action_information = float(beta) * psi
    directional_score = delta_phi + action_information

    result = {
        "phi": phi.detach().cpu(),
        "delta_phi": delta_phi.detach().cpu(),
        "psi": psi.detach().cpu(),
        "action_information": action_information.detach().cpu(),
        "directional_score": directional_score.detach().cpu(),
    }

    # The cached visual state is [agentview || wrist].  Per-view cosine deltas
    # are used only as an abstention diagnostic; they are never added to D_t.
    if visual.shape[-1] % 2 == 0:
        half = visual.shape[-1] // 2
        agent = visual[:, :half]
        wrist = visual[:, half:]
        agent_goal = goal_frames[:, :half]
        wrist_goal = goal_frames[:, half:]
        agent_cos = F.cosine_similarity(agent, agent_goal, dim=-1)
        wrist_cos = F.cosine_similarity(wrist, wrist_goal, dim=-1)
        result["agent_view_delta"] = (agent_cos[1:] - agent_cos[:-1]).detach().cpu()
        result["wrist_view_delta"] = (wrist_cos[1:] - wrist_cos[:-1]).detach().cpu()
    return result


def _load_episode_replay(
    root: Path,
    suite: str,
    task_id: int,
    demo_name: str,
) -> dict[str, Any]:
    import h5py

    demo_path, bddl_path, task_text = resolve_libero_task(root, suite, task_id)
    with h5py.File(demo_path, "r") as handle:
        demo = handle["data"][demo_name]
        obs = demo["obs"]
        agent = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
        actions = np.asarray(demo["actions"], dtype=np.float64)
        states = np.asarray(demo["states"], dtype=np.float64)
        rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float64)
        dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.uint8)

    length = min(len(agent), len(wrist), len(actions), len(states), len(rewards), len(dones))
    agent = agent[:length]
    wrist = wrist[:length]
    actions = actions[:length]
    states = states[:length]
    rewards = rewards[:length]
    dones = dones[:length]
    privileged = extract_privileged_records(
        bddl_path=bddl_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    return {
        "demo_path": demo_path,
        "bddl_path": bddl_path,
        "task_text": task_text,
        "agent": agent,
        "wrist": wrist,
        "privileged": privileged,
    }


def _history_indices(transition_index: int, frame_count: int, window: int) -> list[int]:
    end = min(int(transition_index) + 1, frame_count - 1)
    start = max(0, end - int(window) + 1)
    return list(range(start, end + 1))


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _export_episode_images(
    *,
    output_dir: Path,
    root: Path,
    task_id: int,
    demo_name: str,
    agent: np.ndarray,
    wrist: np.ndarray,
) -> tuple[list[str], list[str]]:
    episode_dir = output_dir / "frames" / f"task_{task_id:02d}" / demo_name
    agent_dir = episode_dir / "agentview"
    wrist_dir = episode_dir / "wrist"
    agent_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)

    agent_paths: list[str] = []
    wrist_paths: list[str] = []
    for index, image in enumerate(agent):
        path = agent_dir / f"{index:06d}.jpg"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, quality=95)
        agent_paths.append(_relative_to_root(path, root))
    for index, image in enumerate(wrist):
        path = wrist_dir / f"{index:06d}.jpg"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, quality=95)
        wrist_paths.append(_relative_to_root(path, root))
    return agent_paths, wrist_paths


def _frame_diagnostic(frame: dict[str, object]) -> dict[str, object]:
    return {key: frame.get(key) for key in PRIVILEGED_FRAME_KEYS}


def _truth_label(direction: int) -> str:
    if direction > 0:
        return P_LABEL
    if direction < 0:
        return N_LABEL
    return U_LABEL


def _macro_f1(truth: list[str], pred: list[str]) -> float:
    scores: list[float] = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, pred))
        fp = sum(t != label and p == label for t, p in zip(truth, pred))
        fn = sum(t == label and p != label for t, p in zip(truth, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def _sanity_metrics(truth: list[str], pred: list[str]) -> dict[str, float | int | None]:
    predicted_positive = sum(value == P_LABEL for value in pred)
    positive_truth = sum(value == P_LABEL for value in truth)
    true_positive = sum(t == P_LABEL and p == P_LABEL for t, p in zip(truth, pred))
    false_positive = sum(t != P_LABEL and p == P_LABEL for t, p in zip(truth, pred))
    nonforward = len(truth) - positive_truth
    return {
        "count": len(truth),
        "macro_f1": _macro_f1(truth, pred),
        "positive_precision": None if not predicted_positive else true_positive / predicted_positive,
        "positive_recall": None if not positive_truth else true_positive / positive_truth,
        "nonforward_false_positive_rate": None if not nonforward else false_positive / nonforward,
        "unclear_rate": sum(value == U_LABEL for value in pred) / max(len(pred), 1),
    }


def _component_disagreement_strength(
    left: float,
    right: float,
    *,
    left_deadband: float,
    right_deadband: float,
) -> float:
    """Return normalized strength of an opposite-sign clear component conflict."""

    left_direction = information_direction(float(left), deadband=float(left_deadband))
    right_direction = information_direction(float(right), deadband=float(right_deadband))
    if left_direction == 0 or right_direction == 0 or left_direction == right_direction:
        return 0.0
    left_strength = abs(float(left)) / max(float(left_deadband), 1e-6)
    right_strength = abs(float(right)) / max(float(right_deadband), 1e-6)
    return float(min(left_strength, right_strength))


def _calibrate(
    train: list[dict[str, Any]],
    scores: dict[int, dict[str, torch.Tensor]],
    *,
    quantile: float,
    disagreement_quantile: float,
) -> dict[str, Any]:
    d_neutral: list[torch.Tensor] = []
    visual_neutral: list[torch.Tensor] = []
    action_neutral: list[torch.Tensor] = []
    agent_neutral: list[torch.Tensor] = []
    wrist_neutral: list[torch.Tensor] = []
    d_all: list[torch.Tensor] = []
    visual_all: list[torch.Tensor] = []
    action_all: list[torch.Tensor] = []
    agent_all: list[torch.Tensor] = []
    wrist_all: list[torch.Tensor] = []

    for episode in train:
        values = scores[id(episode)]
        direction = episode["direction"].cpu()
        length = min(len(direction), len(values["directional_score"]))
        neutral = direction[:length] == 0
        for key, target_neutral, target_all in (
            ("directional_score", d_neutral, d_all),
            ("delta_phi", visual_neutral, visual_all),
            ("action_information", action_neutral, action_all),
        ):
            value = values[key][:length].abs()
            target_all.append(value)
            if bool(neutral.any()):
                target_neutral.append(value[neutral])
        if "agent_view_delta" in values and "wrist_view_delta" in values:
            for key, target_neutral, target_all in (
                ("agent_view_delta", agent_neutral, agent_all),
                ("wrist_view_delta", wrist_neutral, wrist_all),
            ):
                value = values[key][:length].abs()
                target_all.append(value)
                if bool(neutral.any()):
                    target_neutral.append(value[neutral])

    calibration: dict[str, Any] = {
        "neutral_quantile": float(quantile),
        "directional_deadband": _quantile(d_neutral, quantile, d_all),
        "visual_component_deadband": _quantile(visual_neutral, quantile, visual_all),
        "action_component_deadband": _quantile(action_neutral, quantile, action_all),
        "source": "train-neutral absolute-score quantiles",
        # Clear, non-authoritative consistency decisions assign >=0.5 once D
        # is outside the train-calibrated directional deadband.
        "confidence_threshold": 0.5,
        "confidence_threshold_source": "clear-decision boundary induced by train-calibrated deadband",
    }
    if agent_all and wrist_all:
        calibration["agent_view_deadband"] = _quantile(agent_neutral, quantile, agent_all)
        calibration["wrist_view_deadband"] = _quantile(wrist_neutral, quantile, wrist_all)

    # A mere sign disagreement is too sensitive when one information critic is
    # noisier than the other.  Calibrate what counts as an *unusually strong*
    # conflict using training transitions that privileged replay says are
    # forward and whose combined information score D is also clearly forward.
    # This uses no validation/test labels and does not alter D_t itself.
    conflict_strengths: list[float] = []
    for episode in train:
        values = scores[id(episode)]
        direction = episode["direction"].cpu()
        length = min(len(direction), len(values["directional_score"]))
        for index in range(length):
            if int(direction[index]) <= 0:
                continue
            if information_direction(
                float(values["directional_score"][index]),
                deadband=float(calibration["directional_deadband"]),
            ) <= 0:
                continue
            strength = _component_disagreement_strength(
                float(values["delta_phi"][index]),
                float(values["action_information"][index]),
                left_deadband=float(calibration["visual_component_deadband"]),
                right_deadband=float(calibration["action_component_deadband"]),
            )
            if strength > 0.0:
                conflict_strengths.append(strength)
    calibration["component_disagreement_quantile"] = float(disagreement_quantile)
    calibration["component_disagreement_calibration_count"] = len(conflict_strengths)
    calibration["component_disagreement_strength_threshold"] = (
        None
        if not conflict_strengths
        else float(torch.quantile(torch.tensor(conflict_strengths), float(disagreement_quantile)).item())
    )
    calibration["component_disagreement_source"] = (
        "train physical-forward + D-forward conflicts, normalized by train-neutral component deadbands"
    )
    return calibration


def _component_disagreement(
    left: float,
    right: float,
    *,
    left_deadband: float,
    right_deadband: float,
) -> bool:
    return _component_disagreement_strength(
        left,
        right,
        left_deadband=left_deadband,
        right_deadband=right_deadband,
    ) > 0.0


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.neutral_quantile) <= 1.0:
        raise ValueError("--neutral-quantile must be in [0,1].")
    if not 0.0 <= float(args.disagreement_quantile) <= 1.0:
        raise ValueError("--disagreement-quantile must be in [0,1].")
    if int(args.history_window) < 2:
        raise ValueError("--history-window must be at least 2 frames.")
    _ensure_new_output_dir(args.output_dir)

    root = Path.cwd().resolve()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    visual_dim = int(checkpoint["visual_dim"])
    action_dim = int(checkpoint["action_dim"])
    gamma = float(checkpoint.get("gamma", 0.99))
    beta = float(args.beta)

    visual_critic = VisualPointwiseInformationCritic(visual_dim, visual_dim).to(device)
    action_critic = ConditionalActionInformationCritic(action_dim, visual_dim, visual_dim).to(device)
    visual_critic.load_state_dict(checkpoint["visual_critic"])
    action_critic.load_state_dict(checkpoint["action_critic"])
    visual_critic.eval()
    action_critic.eval()

    episodes = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.cache_dir.glob("task_*.pt"))
    ]
    train = [episode for episode in episodes if str(episode.get("split")) == "train"]
    if not train:
        raise RuntimeError("Teacher export requires cached train episodes for calibration.")
    selected = [episode for episode in episodes if str(episode.get("split")) in set(args.splits)]
    if not selected:
        raise RuntimeError(f"No cached episodes found for requested splits {args.splits!r}.")

    scores = {
        id(episode): _score_episode(
            episode,
            visual_critic,
            action_critic,
            device=device,
            gamma=gamma,
            beta=beta,
        )
        for episode in episodes
    }
    calibration = _calibrate(
        train,
        scores,
        quantile=float(args.neutral_quantile),
        disagreement_quantile=float(args.disagreement_quantile),
    )
    _write_json(
        args.output_dir / "calibration.json",
        {
            "pipeline": "v3_information_plus_privileged_consistency_export",
            "gamma": gamma,
            "beta": beta,
            **calibration,
        },
    )

    qwen_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostic_rows: list[dict[str, Any]] = []
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    abstention_counts: dict[str, Counter[str]] = defaultdict(Counter)
    truth_by_split: dict[str, list[str]] = defaultdict(list)
    pred_by_split: dict[str, list[str]] = defaultdict(list)
    authoritative_counts: dict[str, int] = defaultdict(int)
    image_count = 0

    for episode in selected:
        task_id = int(episode["task_id"])
        demo_name = str(episode["demo_name"])
        split = str(episode["split"])
        values = scores[id(episode)]
        replay = _load_episode_replay(root, args.suite, task_id, demo_name)
        frames = list(replay["privileged"]["frames"])
        agent = replay["agent"]
        wrist = replay["wrist"]
        frame_count = min(len(agent), len(wrist), len(episode["visual"]), len(frames))

        if args.skip_image_export:
            agent_paths = [
                f"hdf5://{_relative_to_root(replay['demo_path'], root)}::{demo_name}/obs/agentview_rgb/{i}"
                for i in range(frame_count)
            ]
            wrist_paths = [
                f"hdf5://{_relative_to_root(replay['demo_path'], root)}::{demo_name}/obs/eye_in_hand_rgb/{i}"
                for i in range(frame_count)
            ]
        else:
            agent_paths, wrist_paths = _export_episode_images(
                output_dir=args.output_dir,
                root=root,
                task_id=task_id,
                demo_name=demo_name,
                agent=agent[:frame_count],
                wrist=wrist[:frame_count],
            )
            image_count += len(agent_paths) + len(wrist_paths)

        evidences = [
            PhysicalConsistencyEvidence.from_frames(left, right)
            for left, right in zip(frames[:-1], frames[1:])
        ]
        direction = episode["direction"].cpu()
        transition_count = min(
            frame_count - 1,
            len(direction),
            len(evidences),
            len(values["directional_score"]),
        )

        for index in range(transition_count):
            score = float(values["directional_score"][index])
            delta_phi = float(values["delta_phi"][index])
            psi = float(values["psi"][index])
            action_information = float(values["action_information"][index])
            evidence = evidences[index]
            base = project_teacher_label(
                score,
                evidence,
                deadband=float(calibration["directional_deadband"]),
                use_relation_direction=True,
            )

            visual_action_disagreement_strength = _component_disagreement_strength(
                delta_phi,
                action_information,
                left_deadband=float(calibration["visual_component_deadband"]),
                right_deadband=float(calibration["action_component_deadband"]),
            )
            visual_action_disagreement = bool(visual_action_disagreement_strength > 0.0)
            disagreement_threshold = calibration.get("component_disagreement_strength_threshold")
            high_confidence_visual_action_disagreement = bool(
                visual_action_disagreement
                and disagreement_threshold is not None
                and visual_action_disagreement_strength >= float(disagreement_threshold)
            )
            view_disagreement = False
            agent_view_delta: float | None = None
            wrist_view_delta: float | None = None
            if (
                "agent_view_delta" in values
                and "wrist_view_delta" in values
                and "agent_view_deadband" in calibration
                and "wrist_view_deadband" in calibration
            ):
                agent_view_delta = float(values["agent_view_delta"][index])
                wrist_view_delta = float(values["wrist_view_delta"][index])
                view_disagreement = _component_disagreement(
                    agent_view_delta,
                    wrist_view_delta,
                    left_deadband=float(calibration["agent_view_deadband"]),
                    right_deadband=float(calibration["wrist_view_deadband"]),
                )

            final_label = base.label
            final_confidence = float(base.confidence)
            abstention_reason: str | None = None
            if not base.authoritative_override and base.label in (P_LABEL, N_LABEL):
                if high_confidence_visual_action_disagreement:
                    final_label = U_LABEL
                    final_confidence = 0.0
                    abstention_reason = "high_confidence_visual_action_disagreement"
                elif float(base.confidence) < float(calibration["confidence_threshold"]):
                    final_label = U_LABEL
                    final_confidence = float(base.confidence)
                    abstention_reason = "low_confidence"

            history = _history_indices(index, frame_count, int(args.history_window))
            images = [agent_paths[i] for i in history] + [wrist_paths[i] for i in history]
            sample_id = f"{args.suite}:task{task_id:02d}:{demo_name}:transition{index:06d}"
            prompt = (
                f"Task: {replay['task_text']}. Judge recent task progress as "
                "Positive, Unclear, or Negative."
            )
            qwen_rows[split].append(
                {
                    "sample_id": sample_id,
                    "images": images,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": final_label},
                    ],
                    "provenance": {
                        "suite": args.suite,
                        "task_id": task_id,
                        "demo_name": demo_name,
                        "transition_index": index,
                        "history_frame_indices": history,
                        "view_order": ["agentview", "wrist"],
                    },
                }
            )

            diagnostic_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "suite": args.suite,
                    "task_id": task_id,
                    "demo_name": demo_name,
                    "transition_index": index,
                    "language_instruction": str(replay["task_text"]),
                    "frame_index_t": index,
                    "frame_index_t1": index + 1,
                    "dual_view_history": {
                        "frame_indices": history,
                        "agentview": [agent_paths[i] for i in history],
                        "wrist": [wrist_paths[i] for i in history],
                    },
                    "phi_visual_t": float(values["phi"][index]),
                    "phi_visual_t1": float(values["phi"][index + 1]),
                    "delta_phi_visual": delta_phi,
                    "conditional_action_information": psi,
                    "beta_scaled_action_information": action_information,
                    "directional_score": score,
                    "gamma": gamma,
                    "beta": beta,
                    "information_direction": int(base.information_direction),
                    "physical_direction": int(base.physical_direction),
                    "physical_consistency": str(base.consistency),
                    "raw_confidence": float(base.confidence),
                    "confidence": final_confidence,
                    "authoritative_override": bool(base.authoritative_override),
                    "base_physical_consistency_label": str(base.label),
                    "P/U/N_target": final_label,
                    "abstention_reason": abstention_reason,
                    "visual_action_disagreement": visual_action_disagreement,
                    "visual_action_disagreement_strength": visual_action_disagreement_strength,
                    "high_confidence_visual_action_disagreement": high_confidence_visual_action_disagreement,
                    "dual_view_disagreement": view_disagreement,
                    "dual_view_disagreement_role": "diagnostic_only_not_target_veto",
                    "agent_view_cosine_delta": agent_view_delta,
                    "wrist_view_cosine_delta": wrist_view_delta,
                    "privileged_evidence": asdict(evidence),
                    "privileged_diagnostics": {
                        "frame_t": _frame_diagnostic(frames[index]),
                        "frame_t1": _frame_diagnostic(frames[index + 1]),
                        "goal_predicate": replay["privileged"].get("goal_predicate"),
                        "task_object": replay["privileged"].get("task_object"),
                        "goal_object": replay["privileged"].get("goal_object"),
                        "feasibility": {
                            "available": False,
                            "reason": "official LIBERO demo replay has no separate feasibility checker",
                        },
                    },
                    "calibration": calibration,
                }
            )

            label_counts[split][final_label] += 1
            if abstention_reason is not None:
                abstention_counts[split][abstention_reason] += 1
            if base.authoritative_override:
                authoritative_counts[split] += 1
            truth_by_split[split].append(_truth_label(int(direction[index])))
            pred_by_split[split].append(final_label)

    for split, rows in qwen_rows.items():
        _write_jsonl(args.output_dir / f"qwen_{split}.jsonl", rows)
    _write_jsonl(args.output_dir / "teacher_diagnostics.jsonl", diagnostic_rows)

    qwen_leakage_keys = {
        "phi_visual_t",
        "directional_score",
        "privileged_evidence",
        "privileged_diagnostics",
        "physical_direction",
        "confidence",
    }
    qwen_leakage_found = any(
        bool(qwen_leakage_keys.intersection(row.keys()))
        for rows in qwen_rows.values()
        for row in rows
    )
    report = {
        "pipeline": "v3_teacher_export",
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "suite": args.suite,
        "gamma": gamma,
        "beta": beta,
        "history_window": int(args.history_window),
        "calibration": calibration,
        "exported_episodes": len(selected),
        "exported_transitions": len(diagnostic_rows),
        "exported_image_files": image_count,
        "label_counts": {split: dict(counts) for split, counts in label_counts.items()},
        "abstention_counts": {split: dict(counts) for split, counts in abstention_counts.items()},
        "authoritative_override_counts": dict(authoritative_counts),
        "qwen_privileged_top_level_leakage_found": qwen_leakage_found,
        "qwen_input_contract": "task language + dual-view visual history only",
        "sanity_against_cached_physical_direction": {
            split: _sanity_metrics(truth_by_split[split], pred_by_split[split])
            for split in sorted(pred_by_split)
        },
        "evidence_level": {
            "consistency_eval": "smoke evidence",
            "full_relation_grounded_agreement": "pipeline sanity only",
            "paper_level_gate1": "not yet established; requires simple privileged heuristic baseline on held-out physical-aliasing benchmark",
        },
        "notes": [
            "D_t is information-only; privileged quantities are never numerically added to it.",
            "Qwen JSONL and privileged diagnostics are separate files to prevent student-input leakage.",
            "P/U/N is the supervision/output interface, not the mathematical definition of MI.",
            "Formal Positive targets use strict privileged physical consistency; agreement with the same relation-derived physical labels remains pipeline sanity, not independent paper evidence.",
            "Per-view cosine disagreement is diagnostic only because the current canonical visual critic is a fused dual-view PMI critic, not two independently calibrated per-view PMI critics.",
            "Visual/action disagreement abstains only when its normalized strength exceeds a train-only calibrated high-conflict threshold.",
        ],
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
