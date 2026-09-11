from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mi_reward.features.base_extractor import BaseFeatureExtractor


class DINOv3FeatureExtractor(BaseFeatureExtractor):
    """Local-weight DINO-style extractor with an opt-in smoke-test fallback.

    The class intentionally avoids downloading weights at runtime. If ``model_path``
    points to a TorchScript or ``torch.load``-able module, that model is used.
    Otherwise it returns compact deterministic RGB statistics for smoke tests.
    Production callers must set ``strict=True``; this prevents accidentally
    training a reward model on the fallback representation.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        image_size: int = 224,
        strict: bool = False,
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.strict = strict
        self.model = self._load_model(model_path)
        if self.strict and self.model is None:
            raise RuntimeError(
                "DINOv3 weights are required in strict mode. Provide an existing local model_path."
            )

    def _load_model(self, model_path: str | None) -> nn.Module | None:
        if not model_path:
            return None
        path = Path(model_path)
        if not path.exists():
            return None
        # HF-format directory (contains config.json + model.safetensors)
        if path.is_dir() and (path / "config.json").exists():
            from transformers import AutoModel
            model = AutoModel.from_pretrained(str(path), trust_remote_code=True)
            model.eval().to(self.device)
            return model
        # Single-file checkpoint (.pt / .jit)
        try:
            model = torch.jit.load(str(path), map_location=self.device)
        except Exception:
            loaded = torch.load(path, map_location=self.device)
            model = loaded if isinstance(loaded, nn.Module) else None
        if model is not None:
            model.eval().to(self.device)
        return model

    def _prepare_image(self, image: object) -> torch.Tensor:
        try:
            from PIL import Image
            from torchvision import transforms
        except Exception as exc:
            raise RuntimeError("Pillow and torchvision are required to read image frames.") from exc

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"RGB image must have shape [H,W,3], got {array.shape}.")
            if array.dtype != np.uint8:
                scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array, mode="RGB")
        return transform(pil_image).unsqueeze(0).to(self.device)

    def _load_image(self, frame_path: str) -> torch.Tensor:
        return self._prepare_image(frame_path)

    def _fallback_feature(self, image: torch.Tensor) -> torch.Tensor:
        flat = image.float().flatten(start_dim=2)
        means = flat.mean(dim=-1)
        stds = flat.std(dim=-1)
        mins = flat.min(dim=-1).values
        maxs = flat.max(dim=-1).values
        return torch.cat([means, stds, mins, maxs], dim=-1).squeeze(0)

    @torch.no_grad()
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        return self.extract_frame_tokens(frame_path, task).mean(dim=0)

    @torch.no_grad()
    def extract_frame_tokens(self, frame_path: str, task: str) -> torch.Tensor:
        del task
        image = self._load_image(frame_path)
        return self._extract_tensor_tokens(image)

    @torch.no_grad()
    def extract_image_tokens(self, image: np.ndarray, task: str) -> torch.Tensor:
        del task
        return self._extract_tensor_tokens(self._prepare_image(image))

    @torch.no_grad()
    def extract_trajectory_tokens_images(self, images: list[np.ndarray], task: str) -> torch.Tensor:
        """Extract frozen DINO tokens for an in-memory RGB trajectory in one batch."""
        if not images:
            raise ValueError("Cannot extract an empty trajectory.")
        del task
        if self.model is None:
            return torch.stack([self.extract_image_tokens(image, "") for image in images], dim=0)
        batch = torch.cat([self._prepare_image(image) for image in images], dim=0)
        return self._extract_tensor_tokens(batch)

    def _extract_tensor_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return self._fallback_feature(image).cpu().unsqueeze(0)
        output = self.model(image)
        # HF AutoModel: BaseModelOutputWithPooling → .pooler_output [B, D]
        # HF AutoModel without pooling: .last_hidden_state [B, N, D]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.float().squeeze(0).cpu()
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output.float().squeeze(0).cpu()
        # Legacy: dict with "x_norm_clstoken" key
        if isinstance(output, dict):
            output = output.get("x_norm_clstoken", next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 3:
            return output.float().squeeze(0).cpu()
        if output.ndim == 1:
            return output.float().unsqueeze(0).cpu()
        return output.float().reshape(output.shape[0], -1).cpu()
