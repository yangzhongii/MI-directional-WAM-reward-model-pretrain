"""Pairwise trajectory ranking evaluation for the Qwen3-VL reward model.

Unlike the legacy feature reward-head evaluator, this evaluator consumes the
Pipeline-v3 Qwen deployment contract:

    dual-view trajectory row -> Positive/Unclear/Negative -> scalar reward

The input jsonl is a pair manifest with ``chosen`` and ``rejected`` rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mi_reward.inference.qwen3_vl_reward import Qwen3VLRewardModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    model = Qwen3VLRewardModel(args.model_path)
    pairs: list[dict[str, Any]] = []
    with args.pairs_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                pairs.append(json.loads(line))

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    correct = 0
    margins: list[float] = []
    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        chosen = model.predict_row(pair["chosen"])
        rejected = model.predict_row(pair["rejected"])
        margin = chosen.reward - rejected.reward
        correct += margin > 0
        margins.append(float(margin))
        predictions.append(
            {
                "chosen_label": chosen.label,
                "rejected_label": rejected.label,
                "reward_margin": margin,
            }
        )

    result = {
        "num_pairs": len(pairs),
        "pairwise_accuracy": correct / max(len(pairs), 1),
        "mean_reward_margin": sum(margins) / max(len(margins), 1),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
