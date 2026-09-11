"""Evaluate Qwen3-VL Pipeline-v3 reward predictions on teacher P/U/N JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mi_reward.training.qwen3_vl_reward_data_v3 import (
    LABELS,
    build_multimodal_messages,
    read_qwen_jsonl,
    row_label,
    select_evenly,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(".venv/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument(
        "--data-jsonl",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/teacher_export5_v2/qwen_validation.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--min-pixels", type=int, default=128 * 128)
    parser.add_argument("--max-pixels", type=int, default=128 * 128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_label(text: str) -> str | None:
    stripped = text.strip()
    if stripped in LABELS:
        return stripped
    lowered = stripped.lower()
    hits = [label for label in LABELS if label.lower() in lowered]
    return hits[0] if len(hits) == 1 else None


def _metrics(truth: list[str], pred: list[str | None]) -> dict[str, Any]:
    valid_pred = [value if value in LABELS else "Invalid" for value in pred]
    correct = sum(t == p for t, p in zip(truth, valid_pred))
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truth, valid_pred))
        fp = sum(t != label and p == label for t, p in zip(truth, valid_pred))
        fn = sum(t == label and p != label for t, p in zip(truth, valid_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(t == label for t in truth),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    confusion = {
        truth_label: {
            predicted_label: sum(
                t == truth_label and p == predicted_label for t, p in zip(truth, valid_pred)
            )
            for predicted_label in (*LABELS, "Invalid")
        }
        for truth_label in LABELS
    }
    return {
        "count": len(truth),
        "accuracy": correct / max(len(truth), 1),
        "macro_f1": sum(f1_values) / len(f1_values),
        "invalid_count": sum(value == "Invalid" for value in valid_pred),
        "prediction_counts": dict(Counter(valid_pred)),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
    rows = select_evenly(read_qwen_jsonl(args.data_jsonl), int(args.max_samples))
    project_root = Path.cwd().resolve()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qwen3-VL reward evaluation requires CUDA on this project.")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    truth: list[str] = []
    pred: list[str | None] = []
    image_kwargs = {
        "images_kwargs": {
            "min_pixels": int(args.min_pixels),
            "max_pixels": int(args.max_pixels),
        }
    }
    for index, row in enumerate(rows):
        messages = build_multimodal_messages(row, project_root=project_root, include_answer=False)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **image_kwargs,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        target = row_label(row)
        prediction = _parse_label(text)
        truth.append(target)
        pred.append(prediction)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "task_id": row.get("provenance", {}).get("task_id"),
                "target": target,
                "prediction": prediction,
                "raw_output": text,
            }
        )
        if index == 0 or (index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "target": target,
                        "prediction": prediction,
                        "raw_output": text,
                    }
                ),
                flush=True,
            )

    metrics = _metrics(truth, pred)
    report = {
        "pipeline": "v3_qwen3vl_reward_eval",
        "model_path": str(args.model_path),
        "data_jsonl": str(args.data_jsonl),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "student_input_contract": "task language + dual-view visual history only",
        "metrics": metrics,
    }
    write_json(args.output, report)
    write_jsonl(args.output.with_suffix(".predictions.jsonl"), predictions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

