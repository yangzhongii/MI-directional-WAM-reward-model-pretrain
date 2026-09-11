from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mi_reward.features.base_extractor import BaseFeatureExtractor
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor


class LaWAMLAMFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor backed by LaWAM's LAM visual encoder.

    LaWAM policy configs load LAM with
    ``latent_action_model.core.lam_model.load_latent_action_model(ckpt, yaml)``.
    This standalone wrapper uses the same loader and then calls
    ``LatentLAMModel.extract_vision_features`` to obtain frozen visual features.
    The extractor exposes both the legacy pooled representation and the native
    ``[T, K, D]`` patch representation.  GeoProgress uses the latter so that
    object/goal relations are not discarded before relation-aware scoring.
    """

    def __init__(
        self,
        lam_config_path: str | None = None,
        lam_ckpt_path: str | None = None,
        vision_model_id: str | None = None,
        device: str = "cuda",
        fallback: BaseFeatureExtractor | None = None,
        strict: bool = False,
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.lam_config_path = lam_config_path
        self.lam_ckpt_path = lam_ckpt_path
        self.vision_model_id = vision_model_id
        self.strict = strict
        # Do not eagerly construct the visual fallback.  In the strict LaWAM
        # path it is never used, and eagerly loading it can unnecessarily
        # require a second (possibly version-sensitive) vision backbone.
        self.fallback = fallback
        self.lam = self._load_lam()
        if self.lam is None and self.fallback is None and not self.strict:
            self.fallback = DINOv3FeatureExtractor(device=str(self.device))
        if self.strict and self.lam is None:
            raise RuntimeError(
                "LaWAM LAM could not be loaded in strict mode. Provide a valid "
                "lam_config_path, lam_ckpt_path, and local vision_model_id."
            )

    def _import_lam_module(self):
        try:
            return importlib.import_module("latent_action_model.core.lam_model")
        except Exception:
            return None

    def _load_yaml(self, path: str | Path) -> dict[str, Any]:
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_lam(self):
        lam_module = self._import_lam_module()
        if lam_module is None or not self.lam_config_path:
            return None

        config_path = Path(self.lam_config_path)
        if not config_path.exists():
            return None

        temp_config_path: Path | None = None
        config_for_load = config_path
        if self.vision_model_id:
            cfg = self._load_yaml(config_path)
            model_cfg = dict(cfg.get("model", cfg) or {})
            model_cfg["vision_model_id"] = self.vision_model_id
            if "model" in cfg:
                cfg["model"] = model_cfg
            else:
                cfg = model_cfg
            import tempfile
            import yaml

            fd, name = tempfile.mkstemp(prefix="mi_reward_lam_", suffix=".yaml")
            try:
                import os

                os.close(fd)
            except OSError:
                pass
            Path(name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            temp_config_path = Path(name)
            config_for_load = temp_config_path

        try:
            if self.lam_ckpt_path and Path(self.lam_ckpt_path).exists():
                model = lam_module.load_latent_action_model(self.lam_ckpt_path, str(config_for_load))
            else:
                cfg = self._load_yaml(config_for_load)
                model_cfg = dict(cfg.get("model", cfg) or {})
                model_cfg.pop("ar_prediction", None)
                model = lam_module.LatentLAMModel(**model_cfg)
            model.to(self.device).eval()
            for param in model.parameters():
                param.requires_grad = False
            return model
        except Exception:
            return None
        finally:
            if temp_config_path is not None:
                try:
                    temp_config_path.unlink()
                except OSError:
                    pass

    @property
    def using_lam(self) -> bool:
        return self.lam is not None

    def _image_hw(self) -> tuple[int, int]:
        if self.lam is None:
            return (224, 224)
        image_hw = getattr(self.lam, "image_hw", (256, 256))
        return int(image_hw[0]), int(image_hw[1])

    def _image_to_tensor(self, image: object) -> torch.Tensor:
        from PIL import Image
        from latent_action_model.data_loader.video_aug import imagenet_normalize_

        height, width = self._image_hw()
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
        pil_image = pil_image.resize((width, height))
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(pil_image.tobytes()))
        data = data.reshape(height, width, 3).permute(2, 0, 1).contiguous()
        tensor = data.float().mul_(1.0 / 255.0).to(self.device)
        # LaWAM's helper stores ImageNet statistics as [1,3,1,1]. It therefore
        # expects an explicit batch dimension even for one frame; passing CHW
        # makes the in-place broadcast expand to NCHW and raises at runtime.
        imagenet_normalize_(tensor.unsqueeze(0))
        return tensor

    def _load_frame_tensor(self, frame_path: str) -> torch.Tensor:
        return self._image_to_tensor(frame_path)

    def _as_tokens(self, features: torch.Tensor, expected_frames: int) -> torch.Tensor:
        """Normalize LAM encoder output to ``[T, K, D]`` for a single batch."""
        if features.ndim == 4:
            if features.shape[0] != 1:
                raise ValueError(f"Expected a single LAM batch, got {tuple(features.shape)}")
            tokens = features[0]
        elif features.ndim == 3:
            # Some custom encoders flatten B and T.  With B=1 this is still
            # unambiguous as long as the leading dimension is the frame count.
            tokens = features
        else:
            raise ValueError(
                "LAM vision encoder must return patch features shaped [B, T, K, D] "
                f"or [T, K, D], got {tuple(features.shape)}."
            )
        if tokens.ndim != 3 or tokens.shape[0] != expected_frames:
            raise ValueError(
                f"Unexpected LAM token shape {tuple(tokens.shape)} for {expected_frames} frames."
            )
        return tokens.detach().cpu()

    @torch.no_grad()
    def extract_frame_tokens(self, frame_path: str, task: str) -> torch.Tensor:
        if self.lam is None:
            if self.fallback is None:
                raise RuntimeError("No visual fallback is configured.")
            return self.fallback.extract_frame_tokens(frame_path, task)
        del task
        videos = self._load_frame_tensor(frame_path).unsqueeze(0).unsqueeze(0)
        return self._as_tokens(self.lam.extract_vision_features(videos), expected_frames=1).squeeze(0)

    @torch.no_grad()
    def extract_image_tokens(self, image: np.ndarray, task: str) -> torch.Tensor:
        if self.lam is None:
            if self.fallback is None:
                raise RuntimeError("No visual fallback is configured.")
            return self.fallback.extract_image_tokens(image, task)
        del task
        videos = self._image_to_tensor(image).unsqueeze(0).unsqueeze(0)
        return self._as_tokens(self.lam.extract_vision_features(videos), expected_frames=1).squeeze(0)

    @torch.no_grad()
    def extract_trajectory_tokens(self, frame_paths: list[str], task: str) -> torch.Tensor:
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        if self.lam is None:
            if self.fallback is None:
                raise RuntimeError("No visual fallback is configured.")
            return self.fallback.extract_trajectory_tokens(frame_paths, task)
        del task
        frames = torch.stack([self._load_frame_tensor(frame_path) for frame_path in frame_paths], dim=0)
        return self._as_tokens(self.lam.extract_vision_features(frames.unsqueeze(0)), expected_frames=len(frame_paths))

    @torch.no_grad()
    def extract_trajectory_tokens_images(self, images: list[np.ndarray], task: str) -> torch.Tensor:
        """Extract ``[T,K,D]`` visual tokens directly from RGB arrays.

        This is the HDF5 / simulator counterpart of ``extract_trajectory_tokens``
        and avoids writing temporary PNG files merely to call the frozen LAM
        encoder.
        """

        if not images:
            raise ValueError("Cannot extract an empty trajectory.")
        if self.lam is None:
            if self.fallback is None:
                raise RuntimeError("No visual fallback is configured.")
            return torch.stack([self.fallback.extract_image_tokens(image, task) for image in images], dim=0)
        del task
        frames = torch.stack([self._image_to_tensor(image) for image in images], dim=0)
        return self._as_tokens(self.lam.extract_vision_features(frames.unsqueeze(0)), expected_frames=len(images))

    def _latent_action_window(self) -> int:
        if self.lam is None:
            return 0
        encoder = getattr(self.lam, "encoder", None)
        return max(int(getattr(encoder, "num_frames", 0) or 0), 2)

    @staticmethod
    def _window_ending_at(frames: torch.Tensor, end: int, length: int) -> torch.Tensor:
        """Return a fixed-size history ending at ``end``, left-padding at t=0."""

        start = max(0, end - length + 1)
        clip = frames[start : end + 1]
        if clip.shape[0] < length:
            padding = clip[:1].expand(length - clip.shape[0], *clip.shape[1:])
            clip = torch.cat([padding, clip], dim=0)
        return clip

    @torch.no_grad()
    def extract_action_latents(self, frame_paths: list[str], task: str) -> torch.Tensor:
        """Infer LaWAM transition latents for a visual trajectory.

        Each output element describes the motion ending at frame ``t+1``.  The
        LAM's configured temporal window is respected; early transitions are
        left-padded with the first observation.  Robot actions and measured
        relations remain separate privileged signals and are never fabricated
        by this extractor.
        """

        del task
        if self.lam is None:
            raise RuntimeError("LaWAM LAM is required to extract action latents; visual fallback is not allowed.")
        if len(frame_paths) < 2:
            raise ValueError("At least two frames are required to extract a latent action.")
        frames = torch.stack([self._load_frame_tensor(path) for path in frame_paths], dim=0)
        return self._extract_action_latents_tensor(frames)

    @torch.no_grad()
    def extract_action_latents_images(self, images: list[np.ndarray], task: str) -> torch.Tensor:
        """Infer LaWAM transition latents directly from RGB arrays."""

        del task
        if self.lam is None:
            raise RuntimeError("LaWAM LAM is required to extract action latents; visual fallback is not allowed.")
        if len(images) < 2:
            raise ValueError("At least two frames are required to extract a latent action.")
        frames = torch.stack([self._image_to_tensor(image) for image in images], dim=0)
        return self._extract_action_latents_tensor(frames)

    def _extract_action_latents_tensor(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[0] < 2:
            raise ValueError(f"Expected RGB frame tensor [T,C,H,W] with T>=2, got {tuple(frames.shape)}")
        window = self._latent_action_window()
        latents: list[torch.Tensor] = []
        for end in range(1, frames.shape[0]):
            clip = self._window_ending_at(frames, end=end, length=window).unsqueeze(0)
            output = self.lam.get_latent_action(
                videos=clip,
                states=None,
                dec_videos=clip,
                predict_future_frame=False,
            )
            quantized = output.get("quantized")
            if not torch.is_tensor(quantized) or quantized.ndim != 3 or quantized.shape[0] != 1:
                shape = None if not torch.is_tensor(quantized) else tuple(quantized.shape)
                raise ValueError(f"LAM quantized action must be [1, Q, D], got {shape}.")
            latents.append(quantized[0].detach().cpu().float())
        return torch.stack(latents, dim=0)

    @torch.no_grad()
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        return self.extract_frame_tokens(frame_path, task).mean(dim=0)

    @torch.no_grad()
    def extract_trajectory(self, frame_paths: list[str], task: str) -> torch.Tensor:
        return self.extract_trajectory_tokens(frame_paths, task).mean(dim=1)


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_feature_cfg(config: dict[str, Any]) -> dict[str, Any]:
    features = dict(config.get("features", config) or {})
    if "lam_yaml_path" in features and "lam_config_path" not in features:
        features["lam_config_path"] = features["lam_yaml_path"]
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test LaWAM LAM feature extraction on one image.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    cfg = _resolve_feature_cfg(_load_config(args.config))
    extractor = LaWAMLAMFeatureExtractor(
        lam_config_path=cfg.get("lam_config_path"),
        lam_ckpt_path=cfg.get("lam_ckpt_path"),
        vision_model_id=cfg.get("vision_model_id"),
        device=cfg.get("device", "cuda"),
        fallback=DINOv3FeatureExtractor(
            model_path=cfg.get("dino_model_path") or cfg.get("vision_model_id"),
            device=cfg.get("device", "cuda"),
            image_size=int(cfg.get("image_size", 224)),
            strict=False,
        ),
        strict=bool(cfg.get("strict", False)),
    )
    feature = extractor.extract_frame(args.image, task="")
    print(
        json.dumps(
            {
                "using_lam": extractor.using_lam,
                "feature_shape": list(feature.shape),
                "feature_dtype": str(feature.dtype),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
