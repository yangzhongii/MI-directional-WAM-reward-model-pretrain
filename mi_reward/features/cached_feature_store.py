from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.features.base_extractor import BaseFeatureExtractor
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor


def safe_feature_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", item_id).strip("_") or "unnamed"


class CachedFeatureStore:
    def __init__(self, feature_root: str | Path):
        self.feature_root = Path(feature_root)
        self.feature_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, item_id: str) -> Path:
        return self.feature_root / f"{safe_feature_name(item_id)}.pt"

    def save(self, item_id: str, features: torch.Tensor) -> Path:
        path = self.path_for(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features.detach().cpu(), path)
        return path

    def load(self, item_id: str) -> torch.Tensor:
        path = self.path_for(item_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached features for {item_id}: {path}")
        return torch.load(path, map_location="cpu")

    def get_or_extract(self, item_id: str, frame_paths: list[str], task: str, extractor: BaseFeatureExtractor) -> torch.Tensor:
        path = self.path_for(item_id)
        if path.exists():
            return self.load(item_id)
        features = extractor.extract_trajectory(frame_paths, task)
        self.save(item_id, features)
        return features

    def get_or_extract_tokens(self, item_id: str, frame_paths: list[str], task: str, extractor: BaseFeatureExtractor) -> torch.Tensor:
        """Extract or load token-level features [T, N, D].

        Uses a separate cache path suffixed with '_tokens' to avoid
        collisions with pooled-feature caches.
        """
        token_id = item_id + "_tokens"
        path = self.path_for(token_id)
        if path.exists():
            return self.load(token_id)
        features = extractor.extract_trajectory_tokens(frame_paths, task)
        self.save(token_id, features)
        return features


def make_extractor(
    name: str,
    model_path: str | None,
    device: str,
    image_size: int,
    lam_config_path: str | None = None,
    lam_ckpt_path: str | None = None,
    vision_model_id: str | None = None,
) -> BaseFeatureExtractor:
    if name.lower() in {"lawam_lam", "lam"}:
        return LaWAMLAMFeatureExtractor(
            lam_config_path=lam_config_path,
            lam_ckpt_path=lam_ckpt_path,
            vision_model_id=vision_model_id,
            device=device,
            fallback=DINOv3FeatureExtractor(model_path=model_path, device=device, image_size=image_size),
        )
    return DINOv3FeatureExtractor(model_path=model_path, device=device, image_size=image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache trajectory and success-reference features.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--success_refs", default=None)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--feature_extractor", default=None)
    parser.add_argument("--lam_config_path", default=None)
    parser.add_argument("--lam_ckpt_path", default=None)
    parser.add_argument("--vision_model_id", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    extractor_name = args.feature_extractor or args.extractor or "dino_v3"
    extractor = make_extractor(
        extractor_name,
        args.model_path,
        args.device,
        args.image_size,
        lam_config_path=args.lam_config_path,
        lam_ckpt_path=args.lam_ckpt_path,
        vision_model_id=args.vision_model_id,
    )
    store = CachedFeatureStore(args.feature_root)
    for example in read_jsonl(args.manifest, TrajectoryExample):
        store.get_or_extract(example.traj_id, example.frames, example.task, extractor)
    if args.success_refs:
        for ref in read_jsonl(args.success_refs, SuccessReference):
            store.get_or_extract(ref.ref_id, ref.frames, ref.task, extractor)


if __name__ == "__main__":
    main()
