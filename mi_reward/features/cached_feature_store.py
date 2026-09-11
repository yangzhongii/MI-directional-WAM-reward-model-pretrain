from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.features.base_extractor import BaseFeatureExtractor
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
from mi_reward.features.lawam_lam_extractor import LaWAMLAMFeatureExtractor
from mi_reward.features.kinematic_extractor import extract_kinematic_latents


def _should_extract(example: TrajectoryExample) -> bool:
    """Exclude rejected generated candidates from expensive feature jobs."""

    requires_verification = (
        example.source in {"cosmos_action_cond", "instance_rollout"} or example.candidate_provenance is not None
    )
    return not requires_verification or bool(example.verification and example.verification.accepted)


def safe_feature_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", item_id).strip("_") or "unnamed"


class CachedFeatureStore:
    def __init__(self, feature_root: str | Path):
        self.feature_root = Path(feature_root)
        self.feature_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, item_id: str) -> Path:
        return self.feature_root / f"{safe_feature_name(item_id)}.pt"

    def metadata_path_for(self, item_id: str) -> Path:
        return self.feature_root / f"{safe_feature_name(item_id)}__feature_meta.json"

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

    def get_or_extract(
        self,
        item_id: str,
        frame_paths: list[str],
        task: str,
        extractor: BaseFeatureExtractor,
        *,
        force: bool = False,
    ) -> torch.Tensor:
        path = self.path_for(item_id)
        if path.exists() and not force:
            return self.load(item_id)
        features = extractor.extract_trajectory(frame_paths, task)
        self.save(item_id, features)
        return features

    def get_or_extract_tokens(
        self,
        item_id: str,
        frame_paths: list[str],
        task: str,
        extractor: BaseFeatureExtractor,
        *,
        force: bool = False,
    ) -> torch.Tensor:
        """Extract or load token-level features [T, N, D].

        Uses a separate cache path suffixed with '_tokens' to avoid
        collisions with pooled-feature caches.
        """
        token_id = item_id + "_tokens"
        path = self.path_for(token_id)
        if path.exists() and not force:
            return self.load(token_id)
        features = extractor.extract_trajectory_tokens(frame_paths, task)
        self.save(token_id, features)
        return features

    def get_or_extract_action_latents(
        self,
        item_id: str,
        frame_paths: list[str],
        task: str,
        extractor: BaseFeatureExtractor,
        *,
        force: bool = False,
    ) -> torch.Tensor:
        """Extract or load LaWAM transition latents ``[T-1, Q, D]``."""

        action_id = item_id + "_action_latents"
        path = self.path_for(action_id)
        if path.exists() and not force:
            return self.load(action_id)
        latents = extractor.extract_action_latents(frame_paths, task)
        if latents.ndim != 3 or latents.shape[0] != len(frame_paths) - 1:
            raise ValueError(
                f"Action latent cache for {item_id} must be [T-1,Q,D]; "
                f"got {tuple(latents.shape)} for {len(frame_paths)} frames."
            )
        self.save(action_id, latents)
        return latents

    def get_or_extract_kinematic_latents(
        self,
        item_id: str,
        robot_state_path: str,
        object_state_path: str,
        *,
        output_dim: int = 48,
        force: bool = False,
    ) -> torch.Tensor:
        latent_id = item_id + "_kinematic_latents"
        path = self.path_for(latent_id)
        if path.exists() and not force:
            return self.load(latent_id)
        latents = extract_kinematic_latents(
            robot_state_path,
            object_state_path,
            output_dim=output_dim,
        )
        self.save(latent_id, latents)
        return latents

    def write_metadata(
        self,
        item_id: str,
        *,
        visual_tokens: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        kinematic_latents: torch.Tensor | None = None,
        extractor_name: str,
        source_signature: str | None = None,
    ) -> Path:
        payload: dict[str, object] = {"extractor": extractor_name}
        if source_signature is not None:
            payload["source_signature"] = source_signature
        if visual_tokens is not None:
            payload["visual_token_shape"] = list(visual_tokens.shape)
        if action_latents is not None:
            payload["action_latent_shape"] = list(action_latents.shape)
        if kinematic_latents is not None:
            payload["kinematic_latent_shape"] = list(kinematic_latents.shape)
        path = self.metadata_path_for(item_id)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def feature_source_signature(
    frame_paths: list[str],
    robot_state_path: str | None = None,
    object_state_path: str | None = None,
) -> str:
    """Fingerprint the exact observations and privileged state sidecars."""

    stamps: list[tuple[str, int | None, int | None]] = []
    for raw_path in [*frame_paths, robot_state_path, object_state_path]:
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        if path.is_file():
            stat = path.stat()
            stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
        else:
            stamps.append((str(path), None, None))
    encoded = json.dumps(stamps, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_extractor(
    name: str,
    model_path: str | None,
    device: str,
    image_size: int,
    lam_config_path: str | None = None,
    lam_ckpt_path: str | None = None,
    vision_model_id: str | None = None,
    strict: bool = False,
) -> BaseFeatureExtractor:
    if name.lower() in {"lawam_lam", "lam"}:
        return LaWAMLAMFeatureExtractor(
            lam_config_path=lam_config_path,
            lam_ckpt_path=lam_ckpt_path,
            vision_model_id=vision_model_id,
            device=device,
            # LAM owns the production encoder in this branch. The fallback is
            # only for legacy/non-strict extraction, so it must not reject a
            # valid LAM setup before LAM has a chance to load.
            fallback=DINOv3FeatureExtractor(
                model_path=model_path or vision_model_id,
                device=device,
                image_size=image_size,
                strict=False,
            ),
            strict=strict,
        )
    return DINOv3FeatureExtractor(model_path=model_path, device=device, image_size=image_size, strict=strict)


def feature_bundle_is_cached(
    store: CachedFeatureStore,
    item_id: str,
    *,
    tokens: bool,
    action_latents: bool,
    kinematic_latents: bool,
    extractor_name: str,
    source_signature: str | None = None,
) -> bool:
    """Check a complete cache bundle without loading its large tensors."""

    required_ids = [item_id]
    if tokens:
        required_ids.append(item_id + "_tokens")
    if action_latents:
        required_ids.append(item_id + "_action_latents")
    if kinematic_latents:
        required_ids.append(item_id + "_kinematic_latents")
    if not all(store.path_for(feature_id).is_file() for feature_id in required_ids):
        return False
    if not (tokens or action_latents or kinematic_latents):
        return True
    metadata_path = store.metadata_path_for(item_id)
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("extractor") == extractor_name
        and (
            source_signature is None
            or metadata.get("source_signature") == source_signature
        )
    )


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
    parser.add_argument("--strict", action="store_true", help="Fail instead of using the smoke-test feature fallback.")
    parser.add_argument("--tokens", action="store_true", help="Also cache native visual patch tokens as <traj_id>_tokens.")
    parser.add_argument(
        "--action-latents",
        action="store_true",
        help="Cache LaWAM transition latents as <traj_id>_action_latents (requires --feature_extractor lawam_lam).",
    )
    parser.add_argument(
        "--kinematic-latents",
        action="store_true",
        help="Cache explicit robot/object-state latents as <id>_kinematic_latents.",
    )
    parser.add_argument("--kinematic-dim", type=int, default=48)
    args = parser.parse_args()

    extractor_name = args.feature_extractor or args.extractor or "dino_v3"
    store = CachedFeatureStore(args.feature_root)
    examples = [
        example for example in read_jsonl(args.manifest, TrajectoryExample) if _should_extract(example)
    ]
    refs = read_jsonl(args.success_refs, SuccessReference) if args.success_refs else []

    source_signatures = {
        item.traj_id: feature_source_signature(
            item.frames,
            item.robot_state_path,
            item.object_state_path,
        )
        for item in examples
    }
    source_signatures.update({
        item.ref_id: feature_source_signature(
            item.frames,
            item.robot_state_path,
            item.object_state_path,
        )
        for item in refs
    })

    def missing_bundle(item_id: str) -> bool:
        return not feature_bundle_is_cached(
            store,
            item_id,
            tokens=args.tokens,
            action_latents=args.action_latents,
            kinematic_latents=args.kinematic_latents,
            extractor_name=extractor_name,
            source_signature=source_signatures[item_id],
        )

    def source_changed(item_id: str) -> bool:
        metadata_path = store.metadata_path_for(item_id)
        if not metadata_path.is_file():
            return any(store.path_for(feature_id).is_file() for feature_id in (
                item_id,
                item_id + "_tokens",
                item_id + "_action_latents",
                item_id + "_kinematic_latents",
            ))
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        return metadata.get("source_signature") != source_signatures[item_id]

    pending_examples = [example for example in examples if missing_bundle(example.traj_id)]
    pending_refs = [ref for ref in refs if missing_bundle(ref.ref_id)]
    if not pending_examples and not pending_refs:
        print(
            f"[features] Cache complete: {len(examples)} candidates and {len(refs)} references",
            flush=True,
        )
        return

    print(f"[features] Loading extractor: {extractor_name}", flush=True)
    extractor = make_extractor(
        extractor_name,
        args.model_path,
        args.device,
        args.image_size,
        lam_config_path=args.lam_config_path,
        lam_ckpt_path=args.lam_ckpt_path,
        vision_model_id=args.vision_model_id,
        strict=args.strict,
    )
    print("[features] Extractor loaded", flush=True)
    for example in tqdm(pending_examples, desc="Candidate feature cache", unit="trajectory"):
        force = source_changed(example.traj_id)
        store.get_or_extract(
            example.traj_id, example.frames, example.task, extractor, force=force
        )
        visual_tokens = None
        action_latents = None
        kinematic_latents = None
        if args.tokens:
            visual_tokens = store.get_or_extract_tokens(
                example.traj_id, example.frames, example.task, extractor, force=force
            )
        if args.action_latents:
            action_latents = store.get_or_extract_action_latents(
                example.traj_id, example.frames, example.task, extractor, force=force
            )
        if args.kinematic_latents:
            if not example.robot_state_path or not example.object_state_path:
                raise ValueError(f"Kinematic extraction requires state sidecars for {example.traj_id}.")
            kinematic_latents = store.get_or_extract_kinematic_latents(
                example.traj_id, example.robot_state_path, example.object_state_path,
                output_dim=args.kinematic_dim, force=force,
            )
        if args.tokens or args.action_latents or args.kinematic_latents:
            store.write_metadata(
                example.traj_id,
                visual_tokens=visual_tokens,
                action_latents=action_latents,
                kinematic_latents=kinematic_latents,
                extractor_name=extractor_name,
                source_signature=source_signatures[example.traj_id],
            )
    if pending_refs:
        for ref in tqdm(pending_refs, desc="Reference feature cache", unit="trajectory"):
            force = source_changed(ref.ref_id)
            store.get_or_extract(ref.ref_id, ref.frames, ref.task, extractor, force=force)
            visual_tokens = None
            action_latents = None
            kinematic_latents = None
            if args.tokens:
                visual_tokens = store.get_or_extract_tokens(
                    ref.ref_id, ref.frames, ref.task, extractor, force=force
                )
            if args.action_latents:
                action_latents = store.get_or_extract_action_latents(
                    ref.ref_id, ref.frames, ref.task, extractor, force=force
                )
            if args.kinematic_latents:
                if not ref.robot_state_path or not ref.object_state_path:
                    raise ValueError(f"Kinematic extraction requires state sidecars for reference {ref.ref_id}.")
                kinematic_latents = store.get_or_extract_kinematic_latents(
                    ref.ref_id, ref.robot_state_path, ref.object_state_path,
                    output_dim=args.kinematic_dim, force=force,
                )
            if args.tokens or args.action_latents or args.kinematic_latents:
                store.write_metadata(
                    ref.ref_id,
                    visual_tokens=visual_tokens,
                    action_latents=action_latents,
                    kinematic_latents=kinematic_latents,
                    extractor_name=extractor_name,
                    source_signature=source_signatures[ref.ref_id],
                )


if __name__ == "__main__":
    main()
