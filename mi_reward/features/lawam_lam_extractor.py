from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from mi_reward.features.base_extractor import BaseFeatureExtractor
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor


class LaWAMLAMFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor backed by LaWAM's LAM visual encoder.

    LaWAM policy configs load LAM with
    ``latent_action_model.core.lam_model.load_latent_action_model(ckpt, yaml)``.
    This standalone wrapper uses the same loader and then calls
    ``LatentLAMModel.extract_vision_features`` to obtain frozen visual features.
    Token features are mean-pooled into one vector per frame for MI scoring.
    """

    def __init__(
        self,
        lam_config_path: str | None = None,
        lam_ckpt_path: str | None = None,
        vision_model_id: str | None = None,
        device: str = "cuda",
        fallback: BaseFeatureExtractor | None = None,
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.lam_config_path = lam_config_path
        self.lam_ckpt_path = lam_ckpt_path
        self.vision_model_id = vision_model_id
        self.fallback = fallback or DINOv3FeatureExtractor(device=str(self.device))
        self.lam = self._load_lam()

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

    def _load_frame_tensor(self, frame_path: str) -> torch.Tensor:
        from PIL import Image
        from latent_action_model.data_loader.video_aug import imagenet_normalize_

        height, width = self._image_hw()
        image = Image.open(frame_path).convert("RGB").resize((width, height))
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        data = data.reshape(height, width, 3).permute(2, 0, 1).contiguous()
        tensor = data.float().mul_(1.0 / 255.0).to(self.device)
        imagenet_normalize_(tensor)
        return tensor

    @torch.no_grad()
    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        if self.lam is None:
            return self.fallback.extract_frame(frame_path, task)
        del task
        videos = self._load_frame_tensor(frame_path).unsqueeze(0).unsqueeze(0)
        features = self.lam.extract_vision_features(videos)
        return features.reshape(1, -1, features.shape[-1]).mean(dim=1).squeeze(0).detach().cpu()

    @torch.no_grad()
    def extract_trajectory(self, frame_paths: list[str], task: str) -> torch.Tensor:
        if self.lam is None:
            return self.fallback.extract_trajectory(frame_paths, task)
        if not frame_paths:
            raise ValueError("Cannot extract an empty trajectory.")
        del task
        frames = torch.stack([self._load_frame_tensor(frame_path) for frame_path in frame_paths], dim=0)
        videos = frames.unsqueeze(0)
        features = self.lam.extract_vision_features(videos)
        return features.reshape(features.shape[1], -1, features.shape[-1]).mean(dim=1).detach().cpu()


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
            model_path=cfg.get("dino_model_path"),
            device=cfg.get("device", "cuda"),
            image_size=int(cfg.get("image_size", 224)),
        ),
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
