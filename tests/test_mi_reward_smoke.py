"""Full-pipeline smoke test for MI reward SFT.

Follows the exact production path with a real GPU encoder:
  images → ResNet18( GPU) → CachedFeatureStore → score_manifest →
  build_preference_pairs → train_reward_sft → verify checkpoint.

Also includes a fast CPU-only path with synthetic features for CI.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

T = 5   # frames per trajectory
D = 512  # ResNet18 feature dim
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# ResNet18 feature extractor (GPU encoder)
# ---------------------------------------------------------------------------

class ResNet18FeatureExtractor:
    """Lightweight ResNet18 encoder implementing the BaseFeatureExtractor
    contract used by CachedFeatureStore.get_or_extract()."""

    def __init__(self, device: str = "cuda"):
        import torchvision.models as models
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Identity()  # strip classification head → 512-D
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._image_size = 224

    def extract_frame(self, frame_path: str, task: str) -> torch.Tensor:
        del task
        from PIL import Image
        from torchvision import transforms
        img = Image.open(frame_path).convert("RGB")
        t = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = t(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze(0)
        return feat.detach().cpu()

    def extract_trajectory(self, frame_paths: list[str], task: str) -> torch.Tensor:
        feats = [self.extract_frame(p, task) for p in frame_paths]
        return torch.stack(feats, dim=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_features(seed: int, trend: float) -> torch.Tensor:
    """Synthetic features with controlled temporal structure.

    Good trajectory (trend>0): each frame is seed*ref_vec + (1-alpha)*noise,
    so correlation with ref increases over time — producing nonzero MI delta.

    Bad trajectory (trend=0): pure random noise, zero correlation with ref.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    ref_vec = torch.randn(D, generator=gen)  # shared direction
    frames = []
    for t in range(T):
        if trend > 0:
            alpha = 0.1 + 0.9 * t / (T - 1)  # 0.1 → 1.0
            frame = alpha * ref_vec + (1.0 - alpha) * torch.randn(D, generator=gen)
        else:
            frame = torch.randn(D, generator=gen) * 0.5
        frames.append(frame)
    return torch.stack(frames)


def _create_test_images(frame_dir: Path, prefix: str, num_frames: int,
                        ref_img: np.ndarray | None = None,
                        noise_decay: bool = False) -> list[str]:
    """Create RGB PNGs and return their paths.

    * noise_decay=True: frame blends from pure noise → ref_img (alpha=t/(T-1)).
      Good trajectories use this to produce growing MI toward the reference.
    * ref_img set, noise_decay=False: frame = ref_img + small noise.
      Reference images use this — all frames are ref_img-like.
    * Neither: pure random image.
    """
    from PIL import Image
    paths = []
    rng = np.random.default_rng(abs(hash(prefix)) % (2**31))
    for i in range(num_frames):
        if ref_img is not None:
            if noise_decay:
                alpha = i / max(num_frames - 1, 1)
                arr = (alpha * ref_img.astype(np.float32) + (1.0 - alpha) * rng.uniform(0, 255, ref_img.shape)).clip(0, 255).astype(np.uint8)
            else:
                noise = rng.uniform(-20, 20, ref_img.shape).astype(np.float32)
                arr = (ref_img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        else:
            arr = rng.uniform(0, 255, (224, 224, 3)).astype(np.uint8)
        p = str(frame_dir / f"{prefix}_{i:03d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Test: GPU-resnet18 pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_with_gpu_encoder():
    """GPU path: real images → ResNet18 encoder → feature cache → scoring → SFT."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available — this smoke test requires a CUDA GPU.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        manifest_dir = tmp / "manifests"
        feature_dir = tmp / "features"
        frame_dir = tmp / "frames"
        pref_dir = tmp / "preferences"
        output_dir = tmp / "output"
        for d in (manifest_dir, feature_dir, frame_dir, pref_dir, output_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: Create real images + manifest + extract features on GPU ----
        from mi_reward.data.schema import SuccessReference, TrajectoryExample, write_jsonl
        from mi_reward.features.cached_feature_store import CachedFeatureStore

        task = "synthetic_task"
        extractor = ResNet18FeatureExtractor(device="cuda")
        store = CachedFeatureStore(str(feature_dir))

        # Shared reference image — good trajectories blend toward it
        ref_img = np.random.default_rng(42).uniform(0, 255, (224, 224, 3)).astype(np.uint8)

        trajectories = []
        for i in range(6):
            traj_id = f"synthetic/task00/episode{i:03d}"
            is_good = i < 3
            # Good trajectories blend toward the same reference image
            # (progressive MI), bad stay random
            frame_paths = _create_test_images(
                frame_dir, f"traj{i}", T,
                ref_img=ref_img if is_good else None,
                noise_decay=is_good,
            )
            trajectories.append(TrajectoryExample(
                traj_id=traj_id, task=task, frames=frame_paths,
                source="synthetic", split="train", metadata={},
            ))
        write_jsonl(str(manifest_dir / "train_manifest.jsonl"), trajectories)

        # Success reference: all frames = ref_img + small noise
        ref_frame_paths = _create_test_images(frame_dir, "ref", T, ref_img=ref_img, noise_decay=False)
        ref = SuccessReference(ref_id="synthetic/task00/success", task=task, frames=ref_frame_paths)
        write_jsonl(str(manifest_dir / "success_refs.jsonl"), [ref])

        # ---- Use CachedFeatureStore.get_or_extract() — the production path ----
        print("  Extracting features on GPU...")
        for traj in trajectories:
            store.get_or_extract(traj.traj_id, traj.frames, traj.task, extractor)
        store.get_or_extract(ref.ref_id, ref.frames, ref.task, extractor)

        # Verify features were cached
        assert store.path_for(trajectories[0].traj_id).exists()
        feat = store.load(trajectories[0].traj_id)
        assert feat.shape == (T, D), f"Expected [{T}, {D}], got {feat.shape}"

        # ---- Step 2: Score trajectories ----
        from mi_reward.scoring.build_preferences import score_manifest

        scored = score_manifest(
            manifest=str(manifest_dir / "train_manifest.jsonl"),
            success_refs=str(manifest_dir / "success_refs.jsonl"),
            feature_root=str(feature_dir),
            gamma=0.99,
            mi_mode="gaussian_mi_proxy",
        )
        assert len(scored) == 6

        # ---- Step 3: Build preference pairs ----
        from mi_reward.scoring.build_preferences import build_preference_pairs
        from mi_reward.data.schema import write_jsonl as wjl

        pairs = build_preference_pairs(scored, margin=0.01, top_k=2, bottom_k=2,
                                        teacher_version="smoke_test_v0",
                                        score_field="score_delta", seed=0)
        assert len(pairs) > 0, "No preference pairs built"
        wjl(str(pref_dir / "train_preferences.jsonl"), pairs)

        # ---- Step 4: Train reward model SFT ----
        from mi_reward.training.train_reward_sft import train_reward_sft

        train_reward_sft(
            preferences=str(pref_dir / "train_preferences.jsonl"),
            feature_root=str(feature_dir),
            output_dir=str(output_dir),
            batch_size=8, epochs=3, lr=1e-3, hidden_dim=128, seed=0, device="cuda",
        )

        # ---- Step 5: Verify ----
        ckpt_path = output_dir / "pytorch_model.pt"
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["config"]["input_dim"] == D

        from mi_reward.models.reward_head import TrajectoryRewardHead
        head = TrajectoryRewardHead(input_dim=D, hidden_dim=128)
        head.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            r = head(torch.randn(1, T, D))
            assert r.shape == (1,) and torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# Test: Fast CPU synthetic pipeline (CI-friendly)
# ---------------------------------------------------------------------------

def test_full_pipeline_cpu_synthetic():
    """CPU path: synthetic features → score → preferences → SFT (fast, no GPU)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest_dir = tmp / "manifests"
        feature_dir = tmp / "features"
        pref_dir = tmp / "preferences"
        output_dir = tmp / "output"
        for d in (manifest_dir, feature_dir, pref_dir, output_dir):
            d.mkdir(parents=True, exist_ok=True)

        from mi_reward.data.schema import SuccessReference, TrajectoryExample, write_jsonl
        from mi_reward.features.cached_feature_store import CachedFeatureStore

        task = "synthetic_task"
        store = CachedFeatureStore(str(feature_dir))
        trajectories = []
        for i in range(6):
            traj_id = f"synthetic/task00/episode{i:03d}"
            trend = 1.0 if i < 3 else 0.0
            store.save(traj_id, _synthetic_features(seed=i, trend=trend))
            trajectories.append(TrajectoryExample(
                traj_id=traj_id, task=task,
                frames=[f"fake/frame_{j:03d}.png" for j in range(T)],
                source="synthetic", split="train", metadata={},
            ))
        write_jsonl(str(manifest_dir / "train_manifest.jsonl"), trajectories)
        ref_id = "synthetic/task00/success"
        # Reference features: mostly the shared "signal" direction so MI can detect it
        gen = torch.Generator(); gen.manual_seed(0)  # same seed as _synthetic_features seed=0 (good traj #0)
        ref_signal = torch.randn(D, generator=gen)
        ref_feats = torch.stack([ref_signal + torch.randn(D, generator=gen) * 0.3 for _ in range(T)])
        store.save(ref_id, ref_feats)
        refs = [SuccessReference(ref_id=ref_id, task=task, frames=[])]
        write_jsonl(str(manifest_dir / "success_refs.jsonl"), refs)

        from mi_reward.scoring.build_preferences import score_manifest
        scored = score_manifest(
            manifest=str(manifest_dir / "train_manifest.jsonl"),
            success_refs=str(manifest_dir / "success_refs.jsonl"),
            feature_root=str(feature_dir), gamma=0.99, mi_mode="gaussian_mi_proxy",
        )
        assert len(scored) == 6
        good = [s["score_delta"] for s in scored[:3]]
        bad = [s["score_delta"] for s in scored[3:]]
        assert max(good) > min(bad), f"good={good}, bad={bad}"

        from mi_reward.scoring.build_preferences import build_preference_pairs
        from mi_reward.data.schema import write_jsonl as wjl

        pairs = build_preference_pairs(scored, margin=0.01, top_k=2, bottom_k=2,
                                        teacher_version="smoke_test_v0",
                                        score_field="score_delta", seed=0)
        assert len(pairs) > 0
        wjl(str(pref_dir / "train_preferences.jsonl"), pairs)

        from mi_reward.training.train_reward_sft import train_reward_sft

        train_reward_sft(
            preferences=str(pref_dir / "train_preferences.jsonl"),
            feature_root=str(feature_dir), output_dir=str(output_dir),
            batch_size=8, epochs=3, lr=1e-3, hidden_dim=128, seed=0, device="cuda",
        )

        ckpt = torch.load(output_dir / "pytorch_model.pt", map_location="cpu")
        from mi_reward.models.reward_head import TrajectoryRewardHead
        head = TrajectoryRewardHead(input_dim=D, hidden_dim=128)
        head.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            r = head(torch.randn(1, T, D))
            assert r.shape == (1,) and torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# Cosmos mock end-to-end test
# ---------------------------------------------------------------------------

def test_cosmos_mock_pipeline():
    """Cosmos mock → extract features → score → SFT (full pipeline, CPU)."""
    from PIL import Image
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ---- Step 0: Create init frame image ----
        from mi_reward.data.cosmos_generator import MockFutureGenerator, save_video_frames

        init_dir = tmp / "init_frames"
        init_dir.mkdir()
        init_img = np.random.default_rng(0).uniform(0, 255, (224, 224, 3)).astype(np.uint8)
        Image.fromarray(init_img).save(str(init_dir / "scene.png"))

        gen = MockFutureGenerator(seed=0)

        # ---- Generate candidates + ref ----
        videos = gen.generate_candidates(init_img, "test task", num_candidates=6, num_frames=T)
        ref_video = gen.generate_reference(init_img, "test task", num_frames=T)

        frame_dir = tmp / "frames"
        frame_dir.mkdir()
        candidates = []
        for i, video in enumerate(videos):
            paths = save_video_frames(video, frame_dir / f"cand_{i}")
            candidates.append(paths)
        ref_paths = save_video_frames(ref_video, frame_dir / "ref")

        # ---- Feature extraction (DINOv3 ViT-B/16, CPU) ----
        from mi_reward.features.cached_feature_store import CachedFeatureStore
        from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
        from mi_reward.data.schema import SuccessReference, TrajectoryExample, write_jsonl

        FEATURE_DIM = 768  # DINOv3 ViT-B/16
        DINO_WEIGHTS = REPO_ROOT / "weights" / "dinov3-vitb16-pretrain-lvd1689m"
        extractor = DINOv3FeatureExtractor(model_path=str(DINO_WEIGHTS), device="cuda")
        feature_dir = tmp / "features"
        store = CachedFeatureStore(str(feature_dir))

        trajectories = []
        for i, paths in enumerate(candidates):
            traj_id = f"cosmos/task/cand_{i}"
            store.get_or_extract(traj_id, paths, "test task", extractor)
            trajectories.append(TrajectoryExample(
                traj_id=traj_id, task="test task", frames=paths,
                source="cosmos_mock", split="train", metadata={},
            ))

        store.get_or_extract("cosmos/task/ref", ref_paths, "test task", extractor)
        refs = [SuccessReference(ref_id="cosmos/task/ref", task="test task", frames=ref_paths)]

        manifest_dir = tmp / "manifests"
        manifest_dir.mkdir()
        write_jsonl(str(manifest_dir / "train_manifest.jsonl"), trajectories)
        write_jsonl(str(manifest_dir / "success_refs.jsonl"), refs)

        # ---- Score ----
        from mi_reward.scoring.build_preferences import score_manifest
        scored = score_manifest(
            manifest=str(manifest_dir / "train_manifest.jsonl"),
            success_refs=str(manifest_dir / "success_refs.jsonl"),
            feature_root=str(feature_dir), gamma=0.99, mi_mode="gaussian_mi_proxy",
        )
        assert len(scored) == 6

        # ---- Preferences ----
        from mi_reward.scoring.build_preferences import build_preference_pairs
        from mi_reward.data.schema import write_jsonl as wjl
        pairs = build_preference_pairs(scored, margin=0.005, top_k=3, bottom_k=3,
                                        teacher_version="cosmos_mock",
                                        score_field="score_delta", seed=0)
        assert len(pairs) > 0, "No preference pairs from Cosmos mock pipeline"
        pref_dir = tmp / "preferences"
        pref_dir.mkdir()
        wjl(str(pref_dir / "train_preferences.jsonl"), pairs)

        # ---- SFT ----
        from mi_reward.training.train_reward_sft import train_reward_sft
        output_dir = tmp / "output"
        output_dir.mkdir()
        train_reward_sft(
            preferences=str(pref_dir / "train_preferences.jsonl"),
            feature_root=str(feature_dir), output_dir=str(output_dir),
            batch_size=8, epochs=3, lr=1e-3, hidden_dim=128, seed=0, device="cuda",
        )

        ckpt = torch.load(output_dir / "pytorch_model.pt", map_location="cpu")
        from mi_reward.models.reward_head import TrajectoryRewardHead
        head = TrajectoryRewardHead(input_dim=FEATURE_DIM, hidden_dim=128)
        head.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            r = head(torch.randn(1, T, FEATURE_DIM))
            assert r.shape == (1,) and torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# build_cosmos_manifest() production API test
# ---------------------------------------------------------------------------

def test_build_cosmos_manifest_full_pipeline():
    """Test build_cosmos_manifest() with DINOv3 → full pipeline."""
    from PIL import Image
    from mi_reward.data.cosmos_generator import MockFutureGenerator
    from mi_reward.data.build_manifest import build_cosmos_manifest
    from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
    from mi_reward.features.cached_feature_store import CachedFeatureStore
    from mi_reward.data.schema import read_jsonl, SuccessReference, TrajectoryExample

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Create init frame
        init_dir = tmp / "init_frames"
        init_dir.mkdir()
        init_img = np.random.default_rng(0).uniform(0, 255, (224, 224, 3)).astype(np.uint8)
        scene_path = init_dir / "scene.png"
        Image.fromarray(init_img).save(str(scene_path))

        task_specs = [{"task": "open the drawer", "init_frame": str(scene_path)}]

        # ---- build_cosmos_manifest (uses MockFutureGenerator internally) ----
        manifest_path, refs_path = build_cosmos_manifest(
            task_specs, tmp / "cosmos_output",
            num_candidates_per_task=6, num_frames=T, seed=0,
        )

        assert manifest_path.exists() and refs_path.exists()

        # ---- Feature extraction with DINOv3 ----
        FEATURE_DIM = 768
        DINO_WEIGHTS = REPO_ROOT / "weights" / "dinov3-vitb16-pretrain-lvd1689m"
        extractor = DINOv3FeatureExtractor(model_path=str(DINO_WEIGHTS), device="cuda")
        feature_dir = tmp / "features"
        store = CachedFeatureStore(str(feature_dir))

        trajectories = read_jsonl(manifest_path, TrajectoryExample)
        refs = read_jsonl(refs_path, SuccessReference)
        assert len(trajectories) == 6, f"Expected 6 trajectories, got {len(trajectories)}"
        assert len(refs) == 1

        for traj in trajectories:
            store.get_or_extract(traj.traj_id, traj.frames, traj.task, extractor)
        for ref in refs:
            store.get_or_extract(ref.ref_id, ref.frames, ref.task, extractor)

        # ---- Score ----
        from mi_reward.scoring.build_preferences import score_manifest
        scored = score_manifest(
            manifest=str(manifest_path), success_refs=str(refs_path),
            feature_root=str(feature_dir), gamma=0.99, mi_mode="gaussian_mi_proxy",
        )
        assert len(scored) == 6

        # ---- SFT ----
        from mi_reward.scoring.build_preferences import build_preference_pairs
        from mi_reward.data.schema import write_jsonl as wjl
        pairs = build_preference_pairs(scored, margin=0.005, top_k=3, bottom_k=3,
                                        teacher_version="cosmos_v1", score_field="score_delta", seed=0)
        assert len(pairs) > 0
        pref_dir = tmp / "preferences"
        pref_dir.mkdir()
        wjl(str(pref_dir / "train_preferences.jsonl"), pairs)

        from mi_reward.training.train_reward_sft import train_reward_sft
        output_dir = tmp / "output"
        output_dir.mkdir()
        train_reward_sft(
            preferences=str(pref_dir / "train_preferences.jsonl"),
            feature_root=str(feature_dir), output_dir=str(output_dir),
            batch_size=8, epochs=3, lr=1e-3, hidden_dim=128, seed=0, device="cuda",
        )

        ckpt = torch.load(output_dir / "pytorch_model.pt", map_location="cpu")
        from mi_reward.models.reward_head import TrajectoryRewardHead
        head = TrajectoryRewardHead(input_dim=FEATURE_DIM, hidden_dim=128)
        head.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            r = head(torch.randn(1, T, FEATURE_DIM))
            assert r.shape == (1,) and torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# Directional reward distillation test
# ---------------------------------------------------------------------------

def test_directional_reward_distillation():
    """Test train_reward_distill() with MIPotentialField directional reward."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from mi_reward.data.schema import SuccessReference, TrajectoryExample, write_jsonl
        from mi_reward.scoring.build_preferences import score_manifest, build_preference_pairs
        from mi_reward.data.schema import write_jsonl as wjl
        from mi_reward.features.cached_feature_store import CachedFeatureStore

        task = "distill_test"
        store = CachedFeatureStore(str(tmp / "features"))
        trajectories = []
        for i in range(6):
            traj_id = f"distill/task/ep{i:03d}"
            trend = 1.0 if i < 3 else 0.0
            store.save(traj_id, _synthetic_features(seed=i, trend=trend))
            trajectories.append(TrajectoryExample(
                traj_id=traj_id, task=task, frames=[], source="synthetic", split="train", metadata={}))
        write_jsonl(str(tmp / "train_manifest.jsonl"), trajectories)
        store.save("distill/task/ref", torch.randn(T, D) * 0.5 + 2.0)
        refs = [SuccessReference(ref_id="distill/task/ref", task=task, frames=[])]
        write_jsonl(str(tmp / "success_refs.jsonl"), refs)

        scored = score_manifest(str(tmp / "train_manifest.jsonl"), str(tmp / "success_refs.jsonl"),
                                str(tmp / "features"), gamma=0.99, mi_mode="gaussian_mi_proxy")
        assert len(scored) == 6
        pairs = build_preference_pairs(scored, margin=0.0, top_k=3, bottom_k=3,
                                        teacher_version="distill_v0", score_field="score_delta", seed=0)
        assert len(pairs) > 0
        wjl(str(tmp / "train_preferences.jsonl"), pairs)

        from mi_reward.training.train_reward_sft import train_reward_distill
        output_dir = tmp / "output"
        output_dir.mkdir()
        config = train_reward_distill(
            preferences=str(tmp / "train_preferences.jsonl"),
            feature_root=str(tmp / "features"), output_dir=str(output_dir),
            batch_size=4, epochs=1, lr=1e-3, hidden_dim=32,
            architecture="gru", gamma=0.99,
            lambda_rank=1.0, lambda_potential=1.0, lambda_direction=0.5,
            seed=0, device="cuda")
        assert config["model_class"] == "StatePotentialRewardModel"
        ckpt = torch.load(output_dir / "pytorch_model.pt", map_location="cpu")
        from mi_reward.models.state_potential_model import StatePotentialRewardModel
        model = StatePotentialRewardModel(input_dim=config["input_dim"], hidden_dim=32, architecture="gru")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pots = model(torch.randn(2, T, D))
            assert pots.shape == (2, T)
            r = model.compute_deployment_reward(torch.randn(2, D), torch.randn(2, D))
            assert r.shape == (2,) and torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# MIPotentialField ablation tests
# ---------------------------------------------------------------------------

def test_mi_potential_field_backends():
    """Test all MI backends produce finite, meaningful directional rewards."""
    from mi_reward.scoring.mi_potential_field import MIPotentialField, MIBackend

    torch.manual_seed(42)
    base = torch.randn(64)
    curr = base + torch.randn(64) * 0.1
    goal = base + 1.0
    nxt = curr + 0.2 * (goal - curr)

    for backend in MIBackend:
        field = MIPotentialField(backend=backend, gamma=0.99)
        r = field.compute_directional_reward(curr, nxt, goal)
        assert torch.isfinite(r), f"Backend {backend.value} produced non-finite reward"


def test_ablation_reward_functions():
    """Test alternative reward functions: latent_distance, cosine_similarity."""
    from mi_reward.scoring.mi_potential_field import (
        latent_distance_reward, cosine_similarity_reward)
    torch.manual_seed(42)
    a = torch.randn(64)
    b = a + 0.5
    goal = a + 1.0
    r_dist = latent_distance_reward(a, b, goal)
    r_cos = cosine_similarity_reward(a, b, goal)
    assert r_dist > 0, f"Approaching goal should give positive distance reward, got {r_dist}"
    assert r_cos > 0, f"Approaching goal should give positive cosine reward, got {r_cos}"
