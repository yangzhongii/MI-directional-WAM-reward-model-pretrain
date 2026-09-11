"""Contract and metric tests; no model loading or simulator needed."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image

from eval.libero.libero_adapter import SUITES, iter_hdf5, iter_manifest
from eval.libero.run_validation import compute_metrics, aggregate_windows
from mi_reward.evaluation.reward_pair_metrics import summarize_margins, quality_pair_summary
from mi_reward.training.qwen3_vl_reward_data_v3 import build_multimodal_messages


class ValidationTests(unittest.TestCase):
    def test_native_suites_view_order_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite in SUITES:
                folder = root / suite
                folder.mkdir()
                with h5py.File(folder / "task_demo.hdf5", "w") as handle:
                    data = handle.create_group("data")
                    data.attrs["problem_info"] = json.dumps({"language_instruction": "move bowl"})
                    obs = data.create_group("demo_0/obs")
                    obs["agentview_rgb"] = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(12)])
                    obs["eye_in_hand_rgb"] = np.stack([np.full((4, 4, 3), 100 + i, np.uint8) for i in range(12)])
                terminal, prefix = list(iter_hdf5(root, suite, root / "images" / suite, history_frames=3, prefix_smoke=True))
                row = terminal["row"]
                self.assertEqual(row["provenance"]["history_frame_indices"], [9, 10, 11])
                self.assertEqual([int(np.asarray(Image.open(p))[0, 0, 0]) for p in row["images"]],
                                 [9, 10, 11, 109, 110, 111])
                messages = build_multimodal_messages(row, project_root=root, include_answer=False)
                self.assertEqual(len(messages), 1)
                self.assertNotIn("success", row)
                self.assertEqual(row["messages"][0]["content"],
                                 "Task: move bowl. Judge recent task progress as Positive, Unclear, or Negative.")
                self.assertIs(terminal["success"], True)
                self.assertIsNone(prefix["success"])
                self.assertEqual(prefix["row"]["provenance"]["history_frame_indices"], [0, 1, 2])

    def test_task_matching_ties_and_missing_labels(self):
        def item(identity, reward, success, quality, task="a"):
            return dict(trajectory_id=identity, task_id=task, window="terminal", reward=reward,
                        success=success, quality=quality, label="Positive")
        records = [item("s", 1, True, 3), item("f", -1, False, 1),
                   item("tie", 1, False, 1), item("other", -1, False, 1, "b")]
        report = compute_metrics(records)
        self.assertEqual(report["metrics"], dict(ranking_accuracy=0.5, success_failure_accuracy=0.5, reward_margin=1.0))
        self.assertEqual(report["counts"]["success_failure_pairs"], 2)
        self.assertEqual(compute_metrics(list(reversed(records)))["metrics"], report["metrics"])
        unavailable = compute_metrics([item("s", 1, True, None)])
        self.assertTrue(all(value is None for value in unavailable["metrics"].values()))

    def test_manifest_relative_paths_and_label_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4)).save(root / "frame.png")
            record = dict(benchmark="libero_spatial", source="libero_rollout", task_id="task",
                          trajectory_id="rollout", language="move bowl", agentview=["frame.png"],
                          wrist=["frame.png"], success=False, quality=1)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            result = list(iter_manifest(manifest, "libero_spatial"))[0]
            self.assertIs(result["success"], False)
            self.assertEqual(result["row"]["images"], [str(root / "frame.png")] * 2)
            record["success"] = "false"
            manifest.write_text(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                list(iter_manifest(manifest, "libero_spatial"))

    def test_constant_reward_does_not_win(self):
        report = summarize_margins([0, 0, 0])
        self.assertEqual(report["strict_accuracy"], 0)
        self.assertEqual(report["tie_rate"], 1)
        self.assertEqual(report["tie_adjusted_accuracy"], 0.5)
        records = [dict(task="a", quality_label=q, progress_pred=[0])
                   for q in ("failure", "successful", "suboptimal")]
        self.assertEqual(quality_pair_summary(records)["quality_ranking"]["strict_accuracy"], 0)
        self.assertEqual(quality_pair_summary(records), quality_pair_summary(list(reversed(records))))

    def test_whole_trajectory_aggregation(self):
        base = dict(task_id="a", trajectory_id="s", label="Positive", success=True,
                    quality=None, pair_id="matched")
        predictions = [dict(base, window="trajectory_window", reward=1),
                       dict(base, window="terminal", reward=-1),
                       dict(base, trajectory_id="f", success=False, window="terminal", reward=-1)]
        self.assertEqual(aggregate_windows(predictions)[0]["reward"], 0)
        report = compute_metrics(predictions)
        self.assertEqual(report["metrics"]["success_failure_accuracy"], 1)
        self.assertEqual(report["matched_initial_state_success_failure"]["reward_margin"], 1)

    def test_robometer_camera_contract_and_ties(self):
        from mi_reward.evaluation.rbm_eval_adapter import _RewardAdapter, _preference_result
        adapter = object.__new__(_RewardAdapter)
        frames = [{"agentview": np.full((4, 4, 3), i, np.uint8),
                   "wrist": np.full((4, 4, 3), 100 + i, np.uint8)} for i in range(6)]
        trajectory = SimpleNamespace(metadata={}, frames=frames, task="move bowl", id="id",
                                     quality_label="successful", data_source="test", partial_success=None)
        try:
            row = adapter._build_qwen_row_from_trajectory(trajectory)
            self.assertEqual(row["provenance"]["history_frame_indices"], [1, 2, 3, 4, 5])
            self.assertEqual([int(np.asarray(Image.open(p))[0, 0, 0]) for p in row["images"]],
                             [1, 2, 3, 4, 5, 101, 102, 103, 104, 105])
            trajectory.frames = [np.zeros((4, 4, 3), np.uint8)]
            with self.assertRaises(ValueError):
                adapter._build_qwen_row_from_trajectory(trajectory)
            trajectory.frames = [{"agentview": np.zeros((4, 4, 3), np.uint8)}]
            with self.assertRaises(ValueError):
                adapter._build_qwen_row_from_trajectory(trajectory)
            sample = SimpleNamespace(chosen_trajectory=trajectory, rejected_trajectory=trajectory)
            self.assertFalse(_preference_result(sample, 0, 0)["is_correct"])
            self.assertTrue(_preference_result(sample, 0, 0)["is_tie"])
        finally:
            if hasattr(adapter, "_qwen_frame_dir"):
                adapter._qwen_frame_dir.cleanup()

    def test_manifest_multiple_contiguous_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4)).save(root / "frame.png")
            record = dict(benchmark="libero_spatial", source="libero_rollout", task_id="a",
                          trajectory_id="t", language="move bowl", agentview=["frame.png"] * 20,
                          wrist=["frame.png"] * 20, success=False)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            windows = list(iter_manifest(manifest, "libero_spatial", 5, 4))
            self.assertEqual([r["window_end"] for r in windows], [5, 10, 15, 20])
            self.assertEqual(windows[0]["row"]["provenance"]["history_frame_indices"], [0, 1, 2, 3, 4])
            self.assertTrue(all(len(r["row"]["images"]) == 10 for r in windows))


if __name__ == "__main__":
    unittest.main()
