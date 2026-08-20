"""Focused contracts for the verified, goal-conditioned GeoProgress path."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from mi_reward.data.cosmos_action_cond import ingest_action_conditioned_candidates
from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.relations.geometry import RobotState, TaskGeometry, build_relation_descriptor
from mi_reward.relations.sequence import load_relation_sequence, relation_progress_potential
from mi_reward.verification.feasibility import FeasibilityConfig


def test_relation_sequence_is_named_and_progress_is_directional() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relations.json"
        path.write_text(json.dumps([
            {"names": ["object_to_goal_distance", "ee_to_object_distance"], "values": [1.0, 0.8]},
            {"names": ["object_to_goal_distance", "ee_to_object_distance"], "values": [0.4, 0.2]},
        ]), encoding="utf-8")
        sequence = load_relation_sequence(path)
        potential = relation_progress_potential(sequence.values, sequence.names)
        assert potential[1] > potential[0]


def test_action_conditioned_ingest_rejects_missing_sidecars() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        records = root / "candidates.jsonl"
        records.write_text(json.dumps({
            "traj_id": "cosmos/task/candidate0",
            "task": "task",
            "goal_ref_id": "task/success",
            "parent_traj_id": "real/task/episode0",
            "model_id": "robot-action-cond-2b",
            "action_path": "missing_actions.json",
            "robot_state_path": "missing_states.json",
            "relation_path": "missing_relations.json",
        }) + "\n", encoding="utf-8")
        output = root / "manifest.jsonl"
        examples = ingest_action_conditioned_candidates(records, output, FeasibilityConfig())
        assert len(examples) == 1
        assert examples[0].verification is not None
        assert examples[0].verification.status == "rejected"
        assert "missing_action_or_robot_state_context" in examples[0].verification.reasons


def test_geometry_descriptor_and_goal_conditioned_model() -> None:
    relation = build_relation_descriptor(
        RobotState(joint_positions=[0.0, 0.2], end_effector_position=[0.0, 0.0, 0.0]),
        TaskGeometry(object_position=[0.1, 0.0, 0.0], goal_position=[0.5, 0.0, 0.0]),
    )
    assert "object_to_goal_distance" in relation.names

    from mi_reward.models.geoprogress_potential import GeoProgressPotential

    torch.manual_seed(0)
    model = GeoProgressPotential(visual_dim=8, relation_dim=3, hidden_dim=16, num_heads=4)
    model.eval()
    state = torch.randn(2, 4, 3, 8)
    relations = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    goal = torch.randn(2, 3, 8)
    output = model(state, goal, relations, state_mask=mask)
    other_goal = goal.clone()
    other_goal[..., 0] += 0.5
    output_other_goal = model(state, other_goal, relations, state_mask=mask)
    assert output.shape == (2, 4)
    assert torch.equal(output[0, 2:], torch.zeros(2))
    assert not torch.allclose(output, output_other_goal)
    reward = model.compute_deployment_reward(
        state[:, 0], state[:, 1], goal, relations[:, 0], relations[:, 1]
    )
    assert reward.shape == (2,)


def test_directional_scoring_uses_all_success_references_for_confidence() -> None:
    from mi_reward.scoring.build_preferences import score_manifest

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.jsonl"
        references = root / "success_refs.jsonl"
        feature_root = root / "features"
        write_jsonl(manifest, [TrajectoryExample(
            traj_id="candidate",
            task="task",
            frames=[],
            source="real",
            split="train",
        )])
        write_jsonl(references, [
            SuccessReference(ref_id="success/a", task="task", frames=[]),
            SuccessReference(ref_id="success/b", task="task", frames=[]),
        ])
        store = CachedFeatureStore(feature_root)
        torch.manual_seed(11)
        candidate = torch.randn(3, 4, 8)
        reference = torch.randn(3, 4, 8)
        store.save("candidate_tokens", candidate)
        store.save("success/a_tokens", reference)
        store.save("success/b_tokens", reference.clone())

        scored = score_manifest(
            manifest,
            references,
            feature_root,
            gamma=0.99,
            mi_mode="gaussian_mi_proxy",
            use_token_features=True,
            directional_alignment=True,
        )

        assert len(scored) == 1
        assert scored[0]["goal_ref_id"] in {"success/a", "success/b"}
        assert abs(float(scored[0]["confidence"]) - 0.5) < 1e-6


def test_geoprogress_cpu_end_to_end_smoke() -> None:
    """Verified candidate records -> token cache -> one CPU training epoch."""
    from mi_reward.training.train_reward_sft import train_geoprogress

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = ["object_to_goal_distance", "ee_to_object_distance"]
        records = []
        for candidate, distances in (("good", [1.0, 0.7, 0.4, 0.1]), ("bad", [1.0, 0.9, 0.8, 0.7])):
            frames = []
            for step in range(4):
                frame = root / f"{candidate}_{step}.png"
                frame.write_bytes(b"smoke")
                frames.append(str(frame))
            action_path = root / f"{candidate}_actions.json"
            state_path = root / f"{candidate}_states.json"
            relation_path = root / f"{candidate}_relations.json"
            action_path.write_text("[]", encoding="utf-8")
            state_path.write_text(json.dumps({"states": [{
                "joint_positions": [0.0, 0.1],
                "end_effector_position": [0.0, 0.0, 0.0],
            }]}), encoding="utf-8")
            relation_path.write_text(json.dumps([
                {"names": names, "values": [distance, 0.5]} for distance in distances
            ]), encoding="utf-8")
            records.append({
                "traj_id": f"cosmos/task/{candidate}",
                "task": "task",
                "frames": frames,
                "goal_ref_id": "task/success",
                "parent_traj_id": "real/task/episode0",
                "model_id": "robot-action-cond-2b",
                "generation_seed": 1 if candidate == "good" else 2,
                "action_path": str(action_path),
                "robot_state_path": str(state_path),
                "relation_path": str(relation_path),
            })
        record_path = root / "records.jsonl"
        record_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        manifest_path = root / "manifest.jsonl"
        examples = ingest_action_conditioned_candidates(record_path, manifest_path, FeasibilityConfig())
        assert all(example.verification and example.verification.accepted for example in examples)

        refs_path = root / "success_refs.jsonl"
        write_jsonl(refs_path, [SuccessReference(ref_id="task/success", task="task", frames=[])])
        feature_root = root / "features"
        store = CachedFeatureStore(feature_root)
        torch.manual_seed(7)
        store.save("cosmos/task/good_tokens", torch.randn(4, 3, 8))
        store.save("cosmos/task/bad_tokens", torch.randn(4, 3, 8))
        store.save("task/success_tokens", torch.randn(4, 3, 8))
        preferences_path = root / "preferences.jsonl"
        write_jsonl(preferences_path, [PreferencePair(
            task="task",
            chosen_traj_id="cosmos/task/good",
            rejected_traj_id="cosmos/task/bad",
            chosen_score=1.0,
            rejected_score=0.0,
            goal_ref_id="task/success",
            teacher_version="geoprogress_smoke",
        )])
        output_dir = root / "output"
        config = train_geoprogress(
            preferences=preferences_path,
            feature_root=feature_root,
            output_dir=output_dir,
            manifest=manifest_path,
            success_refs=refs_path,
            batch_size=1,
            epochs=1,
            lr=1e-3,
            hidden_dim=16,
            num_heads=4,
            device="cpu",
            seed=0,
        )
        assert config["model_class"] == "GeoProgressPotential"
        assert config["teacher_alignment"] == "monotonic_viterbi"
        assert (output_dir / "pytorch_model.pt").is_file()
