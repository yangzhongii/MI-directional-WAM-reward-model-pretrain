"""Promote a validated adapted Cosmos checkpoint into isolated MI v4 configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, old, new) for key, item in value.items()}
    return value


def promote_data_config(
    base: dict[str, Any], *, checkpoint: str, old_run_id: str, new_run_id: str,
    tokenizer: str, empty_reason_embedding: str,
) -> dict[str, Any]:
    config = _replace_strings(copy.deepcopy(base), old_run_id, new_run_id)
    config["run_id"] = new_run_id
    stages = config.get("execution", {}).get("stages", [])
    predict = next((stage for stage in stages if stage.get("name") == "predict"), None)
    if predict is None:
        raise ValueError("Base data config has no predict stage.")
    predict["enabled"] = True
    command = list(predict.get("command", []))
    for flag, value in (
        ("--adapted-checkpoint", checkpoint),
        ("--tokenizer", tokenizer),
        ("--empty-reason-embedding", empty_reason_embedding),
    ):
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command.extend((flag, value))
    predict["command"] = command
    config["paths"]["candidate_records"] = predict["output_records"]
    return config


def promote(
    *, validation_report: Path, base_data_config: Path, base_reward_config: Path,
    output_data_config: Path, output_reward_config: Path, checkpoint: str,
    old_run_id: str, new_run_id: str, tokenizer: str, empty_reason_embedding: str,
    visual_review: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    if not bool(validation.get("quality_gate", {}).get("passed")):
        raise RuntimeError("Adapted Cosmos quality gate did not pass; refusing to create MI v4 configs.")
    review = json.loads(visual_review.read_text(encoding="utf-8"))
    if not bool(review.get("passed")):
        raise RuntimeError("Adapted Cosmos visual review did not pass; refusing to create MI v4 configs.")
    if Path(review.get("validation_report", "")).resolve() != validation_report.resolve():
        raise RuntimeError("Visual review belongs to a different validation report.")
    base_data = yaml.safe_load(base_data_config.read_text(encoding="utf-8"))
    base_reward = yaml.safe_load(base_reward_config.read_text(encoding="utf-8"))
    if not isinstance(base_data, dict) or not isinstance(base_reward, dict):
        raise ValueError("Base data/reward configurations must be YAML mappings.")
    data = promote_data_config(
        base_data,
        checkpoint=checkpoint,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        tokenizer=tokenizer,
        empty_reason_embedding=empty_reason_embedding,
    )
    reward = _replace_strings(copy.deepcopy(base_reward), old_run_id, new_run_id)
    reward["run_id"] = new_run_id
    output_data_config.parent.mkdir(parents=True, exist_ok=True)
    output_reward_config.parent.mkdir(parents=True, exist_ok=True)
    output_data_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    output_reward_config.write_text(yaml.safe_dump(reward, sort_keys=False), encoding="utf-8")
    return {
        "status": "complete",
        "quality_gate": True,
        "visual_review": {
            "passed": True,
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
            "path": str(visual_review),
        },
        "checkpoint": checkpoint,
        "data_config": str(output_data_config),
        "reward_config": str(output_reward_config),
        "run_id": new_run_id,
        "candidate_records": data["paths"]["candidate_records"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument(
        "--visual-review",
        required=True,
        help="JSON review produced by record_cosmos_action_review.sh.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-data-config", default="mi_reward/configs/generalization_data.yaml")
    parser.add_argument("--base-reward-config", default="mi_reward/configs/generalization_reward.yaml")
    parser.add_argument("--output-data-config", default="mi_reward/configs/generalization_data_v4.yaml")
    parser.add_argument("--output-reward-config", default="mi_reward/configs/generalization_reward_v4.yaml")
    parser.add_argument("--old-run-id", default="generalization_rigid_v3")
    parser.add_argument("--new-run-id", default="generalization_rigid_v4")
    parser.add_argument("--tokenizer", default=".venv/models/cosmos-predict2.5/tokenizer.pth")
    parser.add_argument(
        "--empty-reason-embedding",
        default=".venv/models/cosmos-predict2.5/robot/action-cond/cr1_empty_string_text_embeddings.pt",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    report = promote(
        validation_report=resolve(args.validation_report),
        visual_review=resolve(args.visual_review),
        base_data_config=resolve(args.base_data_config),
        base_reward_config=resolve(args.base_reward_config),
        output_data_config=resolve(args.output_data_config),
        output_reward_config=resolve(args.output_reward_config),
        checkpoint=args.checkpoint,
        old_run_id=args.old_run_id,
        new_run_id=args.new_run_id,
        tokenizer=args.tokenizer,
        empty_reason_embedding=args.empty_reason_embedding,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
