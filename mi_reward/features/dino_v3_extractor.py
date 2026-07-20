from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mi_reward.features.base_extractor import BaseFeatureExtractor


class DINOv3FeatureExtractor(BaseFeatureExtractor):
    """Local-weight DINO-style extractor with a deterministic image-stat fallback.

    The class intentionally avoids downloading weights at runtime. If ``model_path``
    points to a TorchScript or ``torch.load``-able module, that model is used.
    Otherwise it returns compact deterministic RGB statistics so smoke tests and
    data plumbing can run without external assets.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu", image_size: int = 224):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str | None) -> nn.Module | None:
        if not model_path:
            return None
        path = Path(model_path)
        if not path.exists():
            return None
        try:
            model = torch.jit.load(str(path), map_location=self.device)
        except Exception:
            loaded = torch.load(path, map_location=self.device)
            model = loaded if isinstance(loaded, nn.Module) else None
        if model is not None:
            model.eval().to(self.device)
        return model

    def _load_image(self, frame_path: str) -> torch.Tensor:
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
        image = Image.open(frame_path).convert("RGB")
        return transform(image).unsqueeze(0).to(self.device)

    def _fallback_feature(self, image: torch.Tensor) -> torch.Tensor:
        flat = image.flatten(start_dim=2)
        means = flat.mean(dim=-1)
        stds = flat.std(dim=-1)
        mins = flat.min(dim=-1).values
        maxs = flat.max(dim=-1).values
        return torch.cat([means, stds, mins, maxs], dim=-1).squeeze(0)

    @torch.no_grad()
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        del task
        image = self._load_image(frame_path)
        if self.model is None:
            return self._fallback_feature(image).cpu()
        output = self.model(image)
        if isinstance(output, dict):
            output = output.get("x_norm_clstoken", next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.float().reshape(output.shape[0], -1).mean(dim=0).cpu()
