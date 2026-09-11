"""Fine-tune Qwen3-VL as the observation-only Pipeline-v3 reward model.

This first mainline trainer intentionally avoids hidden dependency changes:
it uses native Transformers Qwen3-VL support and standard AdamW, with the
vision tower frozen by default.  The assistant-only SFT loss predicts exactly
one of Positive / Unclear / Negative from task language plus dual-view history.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from mi_reward.training.qwen3_vl_reward_data_v3 import (
    build_multimodal_messages,
    label_counts,
    read_qwen_jsonl,
    select_evenly,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(".venv/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/teacher_export5_v2/qwen_train.jsonl"),
    )
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/teacher_export5_v2/qwen_validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/mi_reward/v3_teacher_mainline/qwen3vl_2b_sft_v1"),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--validation-loss-samples", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=128 * 128)
    parser.add_argument("--max-pixels", type=int, default=128 * 128)
    parser.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PEFT LoRA adaptation instead of updating the full language model.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _ensure_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _find_qwen3vl_vision_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the Qwen3-VL vision tower across HF/PEFT wrappers.

    Qwen3-VL does not expose the same attribute name as older Qwen-VL
    checkpoints (for example ``visual``).  Avoid hard-coding one internal
    name so LoRA wrapping does not break freezing.
    """
    candidates = (
        "visual",
        "vision_model",
        "vision_tower",
        "model.visual",
        "model.vision_model",
        "model.vision_tower",
    )
    for name in candidates:
        obj: Any = model
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, torch.nn.Module):
            return obj

    # Fallback: inspect module names. This handles future HF renames.
    for name, module in model.named_modules():
        lower = name.lower()
        if any(token in lower for token in ("vision", "visual", "vit")):
            if isinstance(module, torch.nn.Module) and sum(1 for _ in module.parameters()) > 0:
                return module
    return None


def _freeze_qwen3vl_vision(model: torch.nn.Module) -> None:
    vision = _find_qwen3vl_vision_module(model)
    if vision is None:
        raise AttributeError(
            "Cannot locate Qwen3-VL vision module. Inspect named_modules before freezing."
        )
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    vision.eval()


def _processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "images_kwargs": {
            "min_pixels": int(args.min_pixels),
            "max_pixels": int(args.max_pixels),
        }
    }


def _encode_training_row(
    row: dict[str, Any],
    *,
    processor: Any,
    project_root: Path,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    full_messages = build_multimodal_messages(row, project_root=project_root, include_answer=True)
    prompt_messages = build_multimodal_messages(row, project_root=project_root, include_answer=False)
    kwargs = _processor_kwargs(args)
    full = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )
    prompt_length = int(prompt["input_ids"].shape[1])
    full_length = int(full["input_ids"].shape[1])
    if prompt_length >= full_length:
        raise RuntimeError(
            f"Assistant target span is empty for {row.get('sample_id')}: "
            f"prompt={prompt_length} full={full_length}"
        )
    labels = full["input_ids"].clone()
    labels[:, :prompt_length] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return dict(full)


@torch.no_grad()
def _validation_loss(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    project_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> float | None:
    if not rows:
        return None
    selected = select_evenly(rows, int(args.validation_loss_samples))
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for row in selected:
        batch = _encode_training_row(row, processor=processor, project_root=project_root, args=args)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().float().cpu().item()))
    if was_training:
        model.train()
        if bool(args.freeze_vision):
            _freeze_qwen3vl_vision(model)
    return float(sum(losses) / len(losses))


def main() -> None:
    args = parse_args()
    if int(args.gradient_accumulation) < 1:
        raise ValueError("--gradient-accumulation must be positive.")
    if float(args.epochs) <= 0.0:
        raise ValueError("--epochs must be positive.")
    _ensure_new_directory(args.output_dir)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    project_root = Path.cwd().resolve()
    train_rows = read_qwen_jsonl(args.train_jsonl)
    validation_rows = read_qwen_jsonl(args.validation_jsonl)
    train_rows = select_evenly(train_rows, int(args.max_train_samples))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qwen3-VL reward SFT mainline requires CUDA on this project.")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    if bool(args.use_lora):
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=float(args.lora_dropout),
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    if bool(args.freeze_vision):
        _freeze_qwen3vl_vision(model)
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.config.text_config.use_cache = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_params = int(sum(parameter.numel() for parameter in trainable))
    total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    micro_steps = max(1, math.ceil(float(args.epochs) * len(train_rows)))
    optimizer_steps = max(1, math.ceil(micro_steps / int(args.gradient_accumulation)))
    warmup_steps = int(round(float(args.warmup_ratio) * optimizer_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )

    run_config = {
        "pipeline": "v3_qwen3vl_reward_sft",
        "model_path": str(args.model_path),
        "train_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "train_label_counts": label_counts(train_rows),
        "validation_label_counts": label_counts(validation_rows),
        "epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "gradient_accumulation": int(args.gradient_accumulation),
        "freeze_vision": bool(args.freeze_vision),
        "use_lora": bool(args.use_lora),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "student_input_contract": "task language + dual-view visual history only",
    }
    write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2), flush=True)

    model.train()
    if bool(args.freeze_vision):
        _freeze_qwen3vl_vision(model)
    optimizer.zero_grad(set_to_none=True)
    shuffled = list(train_rows)
    rng = random.Random(int(args.seed))
    running_loss = 0.0
    running_micro_steps = 0
    optimizer_step = 0
    history: list[dict[str, Any]] = []

    for micro_step in range(micro_steps):
        if micro_step % len(shuffled) == 0:
            rng.shuffle(shuffled)
        row = shuffled[micro_step % len(shuffled)]
        batch = _encode_training_row(row, processor=processor, project_root=project_root, args=args)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(**batch)
        group_start = (micro_step // int(args.gradient_accumulation)) * int(args.gradient_accumulation)
        group_size = min(int(args.gradient_accumulation), micro_steps - group_start)
        loss = outputs.loss / group_size
        loss.backward()
        running_loss += float(outputs.loss.detach().float().cpu().item())
        running_micro_steps += 1

        should_step = ((micro_step + 1) % int(args.gradient_accumulation) == 0) or (micro_step + 1 == micro_steps)
        if not should_step:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        mean_loss = running_loss / running_micro_steps
        running_loss = 0.0
        running_micro_steps = 0
        record = {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step + 1,
            "train_loss": mean_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if optimizer_step == 1 or optimizer_step % int(args.log_every) == 0 or optimizer_step == optimizer_steps:
            print(json.dumps(record), flush=True)

    validation_loss = _validation_loss(
        model,
        processor,
        validation_rows,
        project_root=project_root,
        args=args,
        device=device,
    )
    model.config.text_config.use_cache = True
    model.save_pretrained(args.output_dir / "model", safe_serialization=True)
    processor.save_pretrained(args.output_dir / "model")
    final_report = {
        **run_config,
        "optimizer_steps": optimizer_step,
        "final_train_loss": None if not history else float(history[-1]["train_loss"]),
        "validation_loss": validation_loss,
        "status": "completed",
    }
    write_json(args.output_dir / "report.json", final_report)
    with (args.output_dir / "train_history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()

