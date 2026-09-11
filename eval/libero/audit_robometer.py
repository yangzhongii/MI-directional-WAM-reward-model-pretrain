"""Rescore saved Robometer predictions without claiming a new inference run."""

import argparse
import json
from pathlib import Path

from mi_reward.evaluation.reward_pair_metrics import summarize_margins, quality_pair_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="logs/mi_reward/v3_teacher_mainline/rbm_eval_qwen3_vl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"source": args.source, "kind": "saved_prediction_rescore",
              "limitations": ["Original input camera/prompt contract was not verified; this rescore only corrects metric arithmetic.",
                              "Saved Qwen predictions score a final window, not whole-trajectory directional reward."]}
    for kind in ("quality_preference", "policy_ranking"):
        report[kind] = {}
        for path in sorted((Path(args.source) / kind).glob("*_results.json")):
            records = json.loads(path.read_text())
            if kind == "quality_preference":
                margins = [r["chosen_score"] - r["rejected_score"] for r in records]
                summary = summarize_margins(margins)
                summary["legacy_accuracy_including_ties"] = sum(m >= 0 for m in margins) / len(margins) if margins else None
            else:
                summary = quality_pair_summary(records)
            report[kind][path.name] = summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
