"""Summarize a completed frozen teacher audit without changing its baseline."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--analysis-subdir", default="replay_v2")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    analysis = root / args.analysis_subdir
    if (analysis / "exit_code.txt").read_text().strip() != "0":
        raise RuntimeError("Audit did not complete successfully")
    frozen = json.loads((root / "baseline/verification.json").read_text())
    if not frozen["unchanged"]:
        raise RuntimeError("Frozen baseline changed")
    cached = json.loads((analysis / "cached_teacher_audit.json").read_text())
    rollout = json.loads((analysis / "rollout_teacher_audit.json").read_text())
    rows = [json.loads(line) for line in (analysis / "rollout_window_audit.jsonl").read_text().splitlines()]
    valid = [row for row in rows if row["physical_replay_consistent"]]
    failures = [row for row in valid if not row["episode_success"]]
    positives = [row for row in failures if row["qwen_label"] == "Positive"]
    attribution = {
        "failed_episode_windows": len(failures), "qwen_positive_in_failed_episode": len(positives),
        "qwen_positive_with_physical_forward": sum(row["physical_direction"] > 0 for row in positives),
        "qwen_positive_with_physical_neutral": sum(row["physical_direction"] == 0 for row in positives),
        "qwen_positive_with_physical_regression": sum(row["physical_direction"] < 0 for row in positives),
        "qwen_positive_teacher_labels": dict(Counter(row["teacher_label"] for row in positives)),
        "teacher_positive_physical_nonforward": sum(row["teacher_label"] == "Positive" and row["physical_direction"] <= 0 for row in valid),
    }
    examples = sorted([row for row in positives if row["physical_direction"] <= 0],
                      key=lambda row: (row["physical_direction"], row["task_id"], row["window_end"]))[:12]
    (analysis / "failure_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    keys = ["trajectory_id", "task_id", "policy_id", "episode_success", "window_end", "frame_t", "frame_t1",
            "physical_replay_consistent", "phi_t", "phi_t1", "delta_phi", "psi", "D", "physical_label", "teacher_label", "qwen_label", "abstention_reason"]
    with (analysis / "window_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)
    summary = {"freeze_verification": frozen, "cached_teacher": cached, "rollout_teacher": rollout,
               "failure_window_attribution": attribution,
               "interpretation": [
                   "The implemented D equation and frozen P/U/N projection reproduce existing exported diagnostics.",
                   "Reproduction does not prove the critics estimate the PMI distributions named in v3.",
                   "Constructed same-task/physical-direction-dependent negative distributions remain a mathematical interpretation gap.",
                   "Teacher-vs-physical metrics are not independent teacher validation because its gate uses the same physical evidence.",
                   "Qwen-vs-teacher disagreement is measured with a fixed successful reference; source episode goal was not available for failed trajectories.",
                   "No training, thresholds, formulas, or checkpoint weights were changed."]}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    metrics = cached["direction_metrics_excluding_neutral"]
    lines = ["# v3 冻结 teacher 审计", "",
             f"冻结校验：{frozen['files']} 个源文件/依赖条目，unchanged={frozen['unchanged']}。原文件及权重未修改。", "",
             "## 公式与标签重现", "",
             f"{cached['formula_reconstruction']['rows']} 条记录，公式最大绝对误差 {cached['formula_reconstruction']['max_absolute_residual']:.3g}；标签投影不一致数 {len(cached['label_projection_reproduction']['mismatches'])}。",
             "这验证算术和实现一致性，不证明采样分布满足严格 PMI 定义。", "",
             "## 缓存验证集：排除 neutral 的方向判断", "",
             "| 分数 | 普通准确率 | 类别平衡准确率 | Regression recall | Binary macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    for name, metric in metrics.items():
        lines.append(f"| {name} | {metric['accuracy']:.4f} | {metric['balanced_accuracy_present_classes']:.4f} | {metric['recall']['Negative']:.4f} | {metric['macro_f1']:.4f} |")
    lines += ["", "## 采样分布审计", "",
              "- Visual 正样本：本 episode 的状态与本 episode 成功 goal；负 goal 来自同任务另一成功 episode。不能直接当作无条件边缘分布 p(g)。",
              "- Action 正样本只含物理前进 transition；负样本取同任务非前进状态的视觉最近邻，没有密度修正。不能直接当作 p(a|v)。",
              "- 冻结阈值来自训练集 neutral；本轮没有利用新验证结果调参。", "",
              "## 原生 rollout 逐窗口对照", "",
              f"窗口总数 {len(rows)}；完整动作重放有效 {len(valid)}；排除 {len(rows)-len(valid)}。",
              "首次单帧 state 恢复在 task 2 成功终帧出现 success 不一致，已保留失败日志；正式统计使用原 seed、初始状态、完整动作重放，并核验状态和原始 success。", "",
              "| 分组 | 窗口数 | Qwen 对 teacher macro-F1 | Qwen 对物理启发式 macro-F1 | Qwen 非前进误报 Positive 比例 |",
              "|---|---:|---:|---:|---:|"]
    for name in ("all", "failed_episodes", "seen_tasks_0_4", "unseen_tasks_5_9"):
        group = rollout[name]
        qm = group["qwen_vs_physical_heuristic"]
        lines.append(f"| {name} | {qm['count']} | {group['qwen_vs_teacher']['macro_f1']:.4f} | {qm['macro_f1']:.4f} | {qm['nonforward_false_positive_rate']} |")
    lines += ["", "## 失败窗口归因统计", "", "```json", json.dumps(attribution, indent=2, ensure_ascii=False), "```", "",
              "## 可定位的分歧窗口", ""]
    for row in examples[:6]:
        frame = row["physical_frame_t"]
        next_frame = row["physical_frame_t1"]
        images = row["images"]
        lines += [f"- `{row['trajectory_id']}`，帧 {row['frame_t']} → {row['frame_t1']}：物理={row['physical_label']}，teacher={row['teacher_label']}，Qwen={row['qwen_label']}；D={row['D']:.4f}。",
                  f"  EEF-object 距离 {frame['eef_object_distance_xyz']:.6f} → {next_frame['eef_object_distance_xyz']:.6f}；grasp {frame['grasped']} → {next_frame['grasped']}。",
                  f"  [Agent 末帧]({images[len(images)//2 - 1]}) · [Wrist 末帧]({images[-1]})", ""]
    lines += ["## 解释边界与下一步", "",
              "物理标签仍是已有 sampler/gate 启发式，不能把 teacher 与这些标签的一致率当作独立物理正确率。Qwen 从 5 帧历史判断最后 transition，与 teacher 导出目标对齐。",
              "失败轨迹的局部接近可以是合理 Positive；应优先复核物理非前进而 Qwen Positive 的窗口，以及 teacher/Qwen 分歧。",
              "Teacher 使用同任务成功 demo_0 goal；新任务的 goal 来自官方成功参考，不是失败终局。这与训练时 episode-specific goal 不完全相同，不能将全部分歧直接归因于蒸馏。",
              "下一步应独立审核困难窗口、明确所构造正负分布的统计目标；在确定问题来源前保持此基线冻结。", "",
              "## 复现命令", "", "```bash",
              "bash eval/libero/run_teacher_audit.sh <frozen-run-directory> <fresh-analysis-directory>",
              ".venv-libero/bin/python -m eval.libero.summarize_teacher_audit --run-dir <frozen-run-directory> --analysis-subdir <analysis-subdirectory>",
              "```", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(root / "SUMMARY.md"), "freeze_unchanged": frozen["unchanged"],
                      "rollout_windows": len(rows), "valid_windows": len(valid), "failure_attribution": attribution}, indent=2), flush=True)


if __name__ == "__main__":
    main()
