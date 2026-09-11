"""Runtime Qwen3-VL reward inference for Pipeline v3.

Deployment contract:

    task language + dual-view visual history -> Positive / Unclear / Negative

The wrapper deliberately reuses the exact multimodal message builder used by
training/evaluation so that prompt and view ordering do not drift at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mi_reward.training.qwen3_vl_reward_data_v3 import (
    LABELS,
    build_multimodal_messages,
)


DEFAULT_SCORE_MAP: dict[str, float] = {
    "Positive": 1.0,
    "Unclear": 0.0,
    "Negative": -1.0,
}


@dataclass(frozen=True)
class QwenRewardPrediction:
    label: str
    reward: float
    raw_output: str


def _parse_label(text: str) -> str | None:
    stripped = text.strip()
    if stripped in LABELS:
        return stripped
    lowered = stripped.lower()
    hits = [label for label in LABELS if label.lower() in lowered]
    return hits[0] if len(hits) == 1 else None


class Qwen3VLRewardModel:
    """Frozen Qwen3-VL P/U/N reward model for online inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        min_pixels: int = 128 * 128,
        max_pixels: int = 128 * 128,
        max_new_tokens: int = 6,
        score_map: Mapping[str, float] | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        if resolved_device != "cuda":
            raise RuntimeError("Qwen3-VL reward inference requires CUDA for this project.")

        self.device = torch.device(resolved_device)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.score_map = dict(DEFAULT_SCORE_MAP if score_map is None else score_map)
        missing = [label for label in LABELS if label not in self.score_map]
        if missing:
            raise ValueError(f"score_map is missing labels: {missing}")

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def predict_row(
        self,
        row: dict[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> QwenRewardPrediction:
        """Predict one deployment-compatible exported Qwen row."""

        messages = build_multimodal_messages(
            row,
            project_root=project_root,
            include_answer=False,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        raw_output = self.processor.batch_decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        label = _parse_label(raw_output)
        if label is None:
            raise RuntimeError(f"Qwen reward returned no valid P/U/N label: {raw_output!r}")
        return QwenRewardPrediction(
            label=label,
            reward=float(self.score_map[label]),
            raw_output=raw_output,
        )

