"""Focused contracts for the verified, goal-conditioned GeoProgress path."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from mi_reward.data.cosmos_action_cond import ingest_action_conditioned_candidates
from mi_reward.data.schema import PreferencePair, SuccessReference, TrajectoryExample, write_jsonl
from mi_reward.features.cached_feature_store import CachedFeatureStore
from mi_reward.relations.geometry import RobotState, TaskGeometry, build_relation_descriptor
from mi_reward.relations.sequence import load_relation_sequence, relation_progress_potential
from mi_reward.scoring.directional_potential import combine_process_potentials
from mi_reward.verification.feasibility import FeasibilityConfig


def test_lawam_action_latent_extraction_respects_transition_contract(tmp_path: Path) -> None:
    from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor

    class FakeEncoder:
        num_frames = 3

    class FakeLAM:
        encoder = FakeEncoder()

        @staticmethod
        def get_latent_action(**kwargs):
            clip = kwargs["videos"]
            value = clip.mean().expand(1, 2, 5).clone()
            return {"quantized": value}

    extractor = LaWAMLAMFeatureExtractor.__new__(LaWAMLAMFeatureExtractor)
    extractor.lam = FakeLAM()
    values = {f"frame_{index}": float(index) for index in range(4)}
    extractor._load_frame_tensor = lambda path: torch.full((3, 2, 2), values[path])  # type: ignore[method-assign]
    latents = extractor.extract_action_latents(list(values), task="task")
    assert latents.shape == (3, 2, 5)
    assert latents[1].mean() > latents[0].mean()

    store = CachedFeatureStore(tmp_path)
    store.save("trajectory_action_latents", latents)
    metadata = store.write_metadata(
        "trajectory",
        action_latents=latents,
        extractor_name="lawam_lam",
    )
    assert json.loads(metadata.read_text(encoding="utf-8"))["action_latent_shape"] == [3, 2, 5]


def test_lawam_single_frame_normalization_keeps_chw_shape(tmp_path: Path) -> None:
    from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor

    image_path = tmp_path / "frame.png"
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
    extractor = LaWAMLAMFeatureExtractor.__new__(LaWAMLAMFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.lam = type("FakeLAM", (), {"image_hw": (8, 8)})()

    frame = extractor._load_frame_tensor(str(image_path))

    assert frame.shape == (3, 8, 8)
    assert torch.isfinite(frame).all()


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


def test_relation_potential_is_bounded_with_tiny_initial_orientation_error() -> None:
    names = [
        "object_to_goal_distance",
        "ee_to_object_distance",
        "object_goal_orientation_error",
        "object_in_gripper",
        "target_satisfied",
    ]
    values = torch.tensor([
        [0.45, 0.20, 1.0e-7, 0.0, 0.0],
        [0.30, 0.04, 0.20, 1.0, 0.0],
        [0.01, 0.18, 0.01, 1.0, 1.0],
    ])

    potential = relation_progress_potential(values, names, task_family="peg_insertion")

    assert torch.all((potential >= 0.0) & (potential <= 1.0))
    assert potential[-1] == 1.0
    assert potential[-1] > potential[0]


def test_successful_process_teacher_is_monotonic_and_terminally_anchored() -> None:
    combined = combine_process_potentials(
        torch.tensor([0.0, 0.4, 0.2, 0.8]),
        action=torch.tensor([0.0, 0.3, 0.5, 0.7]),
        relation=torch.tensor([0.1, 0.2, 0.1, 0.9]),
        action_weight=1.0,
        relation_weight=1.0,
        task_success=True,
    )

    assert torch.all(combined[1:] >= combined[:-1])
    assert combined[-1] == 1.0


def test_success_projection_can_be_disabled_for_ablation() -> None:
    combined = combine_process_potentials(
        torch.tensor([0.2, 0.8, 0.4]),
        task_success=True,
        success_monotonic_projection=False,
        success_endpoint_anchor=False,
    )

    assert torch.allclose(combined, torch.tensor([0.2, 0.8, 0.4]))


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
        store.save("candidate_action_latents", torch.randn(2, 2, 6))
        store.save("success/a_action_latents", torch.randn(2, 2, 6))
        store.save("success/b_action_latents", torch.randn(2, 2, 6))

        scored = score_manifest(
            manifest,
            references,
            feature_root,
            gamma=0.99,
            mi_mode="gaussian_mi_proxy",
            use_token_features=True,
            directional_alignment=True,
            action_weight=0.5,
        )

        assert len(scored) == 1
        assert scored[0]["goal_ref_id"] in {"success/a", "success/b"}
        assert scored[0]["action_phi"] is not None
        assert len(scored[0]["action_phi"]) == 2
        assert 0.0 < float(scored[0]["confidence"]) <= 0.5


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
        heldout_examples = [
            replace(
                example,
                traj_id=example.traj_id.replace("cosmos/task/", "cosmos/heldout/"),
                split="scene_heldout",
            )
            for example in examples
        ]
        write_jsonl(manifest_path, [*examples, *heldout_examples])

        refs_path = root / "success_refs.jsonl"
        write_jsonl(refs_path, [SuccessReference(ref_id="task/success", task="task", frames=[])])
        feature_root = root / "features"
        store = CachedFeatureStore(feature_root)
        torch.manual_seed(7)
        store.save("cosmos/task/good_tokens", torch.randn(4, 3, 8))
        store.save("cosmos/task/bad_tokens", torch.randn(4, 3, 8))
        store.save("cosmos/heldout/good_tokens", torch.randn(4, 3, 8))
        store.save("cosmos/heldout/bad_tokens", torch.randn(4, 3, 8))
        store.save("task/success_tokens", torch.randn(4, 3, 8))
        store.save("cosmos/task/good_action_latents", torch.randn(3, 2, 6))
        store.save("cosmos/task/bad_action_latents", torch.randn(3, 2, 6))
        store.save("cosmos/heldout/good_action_latents", torch.randn(3, 2, 6))
        store.save("cosmos/heldout/bad_action_latents", torch.randn(3, 2, 6))
        store.save("task/success_action_latents", torch.randn(3, 2, 6))
        store.save("cosmos/task/good_kinematic_latents", torch.randn(4, 1, 10))
        store.save("cosmos/task/bad_kinematic_latents", torch.randn(4, 1, 10))
        store.save("cosmos/heldout/good_kinematic_latents", torch.randn(4, 1, 10))
        store.save("cosmos/heldout/bad_kinematic_latents", torch.randn(4, 1, 10))
        store.save("task/success_kinematic_latents", torch.randn(4, 1, 10))
        preferences_path = root / "preferences.jsonl"
        write_jsonl(preferences_path, [
            PreferencePair(
                task="task",
                chosen_traj_id="cosmos/task/good",
                rejected_traj_id="cosmos/task/bad",
                chosen_score=1.0,
                rejected_score=0.0,
                goal_ref_id="task/success",
                teacher_version="geoprogress_smoke",
                split="train",
            ),
            PreferencePair(
                task="task",
                chosen_traj_id="cosmos/heldout/good",
                rejected_traj_id="cosmos/heldout/bad",
                chosen_score=1.0,
                rejected_score=0.0,
                goal_ref_id="task/success",
                teacher_version="geoprogress_smoke",
                split="scene_heldout",
            ),
        ])
        teacher_targets_source = root / "preferences.teacher_targets.pt"
        torch.save(
            {
                "version": 2,
                "target_type": "bounded_directional_process_potential",
                "targets": {
                    ("cosmos/task/good", "task/success"): torch.tensor([0.0, 0.3, 0.7, 1.0]),
                    ("cosmos/task/bad", "task/success"): torch.tensor([0.0, 0.1, 0.2, 0.25]),
                    ("cosmos/heldout/good", "task/success"): torch.tensor([0.0, 0.2, 0.8, 1.0]),
                    ("cosmos/heldout/bad", "task/success"): torch.tensor([0.0, 0.1, 0.15, 0.2]),
                },
            },
            teacher_targets_source,
        )
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

        from mi_reward.training.train_reward_sft import train_generalization_reward

        student_output = root / "visual_student"
        student_config = train_generalization_reward(
            preferences=preferences_path,
            feature_root=feature_root,
            output_dir=student_output,
            manifest=manifest_path,
            success_refs=refs_path,
            batch_size=1,
            epochs=1,
            lr=1e-3,
            hidden_dim=16,
            num_heads=4,
            action_weight=0.5,
            kinematic_weight=0.5,
            relation_weight=0.5,
            mi_backend="gaussian_mi_proxy",
            goal_dropout=0.5,
            teacher_targets_source=teacher_targets_source,
            device="cpu",
            seed=0,
        )
        assert student_config["model_class"] == "VisualGoalPotential"
        assert student_config["kinematic_weight"] == 0.5
        assert student_config["teacher_target_count"] == 4
        assert student_config["teacher_target_type"] == "bounded_directional_process_potential"
        assert student_config["train_pairs"] == 1
        assert student_config["validation_pairs"] == 1
        validation_metrics = json.loads(
            (student_output / "validation_metrics.json").read_text(encoding="utf-8")
        )
        assert validation_metrics["validation_pairs"] == 1
        teacher_cache = student_output / "teacher_targets.pt"
        assert teacher_cache.is_file()
        cache_mtime = teacher_cache.stat().st_mtime_ns
        resumed_config = train_generalization_reward(
            preferences=preferences_path,
            feature_root=feature_root,
            output_dir=student_output,
            manifest=manifest_path,
            success_refs=refs_path,
            batch_size=1,
            epochs=1,
            lr=1e-3,
            hidden_dim=16,
            num_heads=4,
            action_weight=0.5,
            kinematic_weight=0.5,
            relation_weight=0.5,
            mi_backend="gaussian_mi_proxy",
            goal_dropout=0.5,
            teacher_targets_source=teacher_targets_source,
            device="cpu",
            seed=0,
        )
        assert resumed_config["teacher_target_count"] == 4
        assert teacher_cache.stat().st_mtime_ns == cache_mtime
        checkpoint = torch.load(student_output / "pytorch_model.pt", map_location="cpu")
        from mi_reward.models.visual_goal_potential import VisualGoalPotential

        student = VisualGoalPotential(
            visual_dim=8,
            hidden_dim=16,
            num_heads=4,
            goal_dropout=0.5,
        )
        student.load_state_dict(checkpoint["model_state_dict"])
        student.eval()
        null_goal_output = student(torch.randn(1, 4, 3, 8))
        assert null_goal_output.shape == (1, 4)
        from mi_reward.inference.visual_goal_model import VisualGoalInferenceModel

        inference = VisualGoalInferenceModel.from_pretrained(student_output / "pytorch_model.pt", device="cpu")
        wrapped_output = inference.predict_potential(torch.randn(1, 4, 3, 8))
        assert wrapped_output.shape == (1, 4)
