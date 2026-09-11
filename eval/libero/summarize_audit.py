"""Write a compact, traceable audit summary and per-task comparison table."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    validation = json.loads((root / "validation/results.json").read_text())
    collection = json.loads((root / "rollouts/collection.json").read_text())
    robometer = json.loads((root / "robometer_rescore.json").read_text())
    trajectories = [json.loads(line) for line in (root / "validation/trajectories.jsonl").read_text().splitlines()]
    manifest = [json.loads(line) for line in (root / "rollouts/manifest.jsonl").read_text().splitlines()]
    assert collection["status"] == validation["status"] == "completed"
    assert collection["episodes"] == len(manifest) == len(trajectories)
    source = {r["trajectory_id"]: r for r in manifest}
    grouped = defaultdict(dict)
    for trajectory in trajectories:
        item = source[trajectory["trajectory_id"]]
        assert trajectory["success"] is item["success"]
        assert item["teacher_episode_overlap"] is False
        grouped[item["task_index"]][item["policy_id"]] = trajectory
    rows = []
    for task, controls in sorted(grouped.items()):
        replay, opened = controls["demo_action_replay"], controls["open_gripper_replay"]
        assert source[replay["trajectory_id"]]["initial_state_sha256"] == source[opened["trajectory_id"]]["initial_state_sha256"]
        rows.append({"task_id": task, "replay_success": replay["success"], "open_gripper_success": opened["success"],
                     "replay_reward": replay["reward"], "open_gripper_reward": opened["reward"],
                     "replay_minus_open_gripper": replay["reward"] - opened["reward"],
                     "eligible_success_failure_pair": replay["success"] != opened["success"]})
    with (root / "task_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": validation["checkpoint"], "benchmark": validation["benchmark"],
        "collection": collection, "libero": validation,
        "robometer_saved_prediction_audit": robometer, "task_comparisons": rows,
        "limitations": [
            "10 tasks, one source initial state per task, two scripted controllers; not learned-policy ranking.",
            "Source episodes excluded from teacher train/validation; task overlap is permitted and declared.",
            "Window-level P/U/N accuracy is not measured: no independent per-window direction annotations.",
            "Quality ranking is null without external quality labels; success/failure uses measured environment outcomes.",
            "Historical Robometer input contract was invalid/unverified; arithmetic rescore cannot repair input observations.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    sf = validation["success_failure_details"]
    quality = next(iter(robometer["quality_preference"].values()))
    ranking = next(iter(robometer["policy_ranking"].values()))["quality_ranking"]
    lines = ["# Pipeline v3 validation audit", "",
             f"Checkpoint: `{validation['checkpoint']}`. All collection/inference/tests ran in tmux train with .venv-libero.", "",
             "## LIBERO spatial", "",
             f"{collection['episodes']} native simulator trajectories; {validation['counts']['predictions']} five-frame dual-view windows.",
             f"Success/failure pairs: {sf['pairs']}; wins/ties/losses: {sf['wins']}/{sf['ties']}/{sf['losses']}.",
             f"Strict accuracy: {sf['strict_accuracy']}; tie-adjusted accuracy: {sf['tie_adjusted_accuracy']}; mean margin: {sf['reward_margin']}.",
             "Trajectory reward is the mean over eight uniformly sampled windows. Quality ranking remains null.", "",
             "| Task | Replay success | Open-gripper success | Replay reward | Open-gripper reward | Difference |",
             "|---|---|---|---|---|---|"]
    lines.extend(f"| {r['task_id']} | {r['replay_success']} | {r['open_gripper_success']} | {r['replay_reward']:.3f} | {r['open_gripper_reward']:.3f} | {r['replay_minus_open_gripper']:.3f} |" for r in rows)
    lines += ["", "## Historical Robometer arithmetic audit", "",
              f"Quality preference: {quality['wins']} wins, {quality['ties']} ties, {quality['losses']} losses / {quality['pairs']} pairs.",
              f"Strict accuracy: {quality['strict_accuracy']:.6f}; tie-adjusted accuracy: {quality['tie_adjusted_accuracy']:.6f}.",
              f"Quality ranking: {ranking['wins']} wins, {ranking['ties']} ties, {ranking['losses']} losses / {ranking['pairs']} pairs.",
              "Old 97.72% preference counted ties as correct. Old 58.13% ranking is not evidence of discrimination.", "",
              "## Limits", "", *[f"- {item}" for item in summary["limitations"]], "",
              "## Reproduction", "", "```bash",
              "bash eval/libero/run_audit.sh <fresh-output-directory>",
              ".venv-libero/bin/python -m eval.libero.summarize_audit --run-dir <output-directory>", "```", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"summary": str(root / "SUMMARY.md"), "libero_metrics": validation["metrics"],
                      "success_failure_details": sf, "by_controller": validation["by_controller"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
