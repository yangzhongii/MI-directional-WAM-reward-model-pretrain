"""Cosmos-Predict2.5 video generation adapter.

Runs Cosmos via CLI subprocess — no Python API dependency.
Cosmos upgrades or dependency changes don't affect the MI reward pipeline.

Install:  bash requirements/install.sh --mi-cosmos
Weights:  huggingface-cli download nvidia/Cosmos-Predict2.5-2B --local-dir weights/cosmos-predict2.5
Inference: torchrun --nproc_per_node=1 examples/inference.py -i config.json ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class BaseFutureGenerator(ABC):
    """Abstract interface for world-model future generation."""

    @abstractmethod
    def generate_candidates(
        self, init_image: np.ndarray, task_text: str,
        num_candidates: int = 10, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return list of [T, H, W, C] RGB uint8 arrays."""
        ...

    @abstractmethod
    def generate_reference(
        self, init_image: np.ndarray, task_text: str, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return [T, H, W, C] RGB uint8 array."""
        ...


# ---------------------------------------------------------------------------
# Cosmos-Predict2.5 via CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionConditionedRequest:
    """Sidecar contract for official robot/action-conditioned Cosmos runs.

    The generator's RGB frames are only a candidate video.  The paired action
    and robot-state logs are mandatory and are consumed by
    ``cosmos_action_cond.ingest_action_conditioned_candidates``.
    """

    parent_traj_id: str
    action_path: str
    robot_state_path: str
    relation_path: str
    goal_ref_id: str
    generation_seed: int

class CosmosPredictGenerator(BaseFutureGenerator):
    """Cosmos-Predict2.5 via CLI subprocess — no Python API dependency.

    Runs::

        torchrun --nproc_per_node=<gpus> examples/inference.py \\
          -i config.json --model <model> --checkpoint-path <weights> -o output/

    Defaults to one process for small 2B validation. Multi-process launch is
    optional acceleration and is not used by the robot/action-conditioned
    ingestion path.
    """

    def __init__(
        self,
        cosmos_repo: str | None = None,
        weights_dir: str = "weights/cosmos-predict2.5",
        gpus: int = 1,
        model_name: str = "2B/post-trained",
        temperature: float = 1.0,
        ref_temperature: float = 0.3,
        num_steps: int = 35,
    ):
        repo = Path(cosmos_repo) if cosmos_repo else Path(__file__).resolve().parents[3] / ".venv" / "cosmos-predict2.5"
        self.cosmos_repo = Path(repo)
        self.weights_dir = Path(weights_dir).resolve()
        self.gpus = gpus
        self.model_name = model_name
        self.temperature = temperature
        self.ref_temperature = ref_temperature
        self.num_steps = num_steps

    def _generate_one(
        self, init_image: np.ndarray, task_text: str, num_frames: int, temperature: float, seed: int,
    ) -> np.ndarray:
        import tempfile, subprocess, sys, os

        work_dir = Path(tempfile.mkdtemp(prefix="cosmos_gen_"))
        img_path = work_dir / "init_frame.png"
        Image.fromarray(init_image).save(str(img_path))

        # Write inference config JSON
        h, w = init_image.shape[:2]
        config_path = work_dir / "config.json"
        config_path.write_text(json.dumps({
            "name": "cosmos_gen",
            "prompt": task_text,
            "seed": seed,
            "input_image_path": str(img_path),
            "num_output_frames": num_frames,
        }))

        output_dir = work_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Run Cosmos via CLI (subprocess — no Python API coupling)
        inference_script = self.cosmos_repo / "examples" / "inference.py"
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(self.gpus),
            str(inference_script),
            "-i", str(config_path),
            "--model", self.model_name,
            "--checkpoint-path", str(self.weights_dir),
            "-o", str(output_dir),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(self.gpus))}
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.cosmos_repo), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Cosmos CLI failed (rc={result.returncode}):\n{result.stderr[-2000:]}")

        return _load_single_video(output_dir, num_frames)

    def generate_candidates(
        self, init_image: np.ndarray, task_text: str,
        num_candidates: int = 10, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        # The generic CLI path is visual-only. Robot/action-conditioned output
        # is ingested through cosmos_action_cond.py with its sidecar logs.
        del goal_image
        videos = []
        for i in range(num_candidates):
            video = self._generate_one(init_image, task_text, num_frames, self.temperature, seed=i)
            videos.append(video)
        return videos

    def generate_reference(
        self, init_image: np.ndarray, task_text: str, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> np.ndarray:
        del goal_image
        return self._generate_one(init_image, task_text, num_frames, self.ref_temperature, seed=0)


# ---------------------------------------------------------------------------
# Diffusers CPU-offload backend (single 24GB RTX 4090 validation)
# ---------------------------------------------------------------------------

class DiffusersCosmosGenerator(BaseFutureGenerator):
    """Cosmos-Predict2.5 via Diffusers pipeline with CPU offload for 4090.

    Uses CPU offload and attention slicing for small single-GPU validation.
    A standard RTX 4090 has 24 GB; exact memory use depends on resolution,
    frames, and the installed Cosmos build.
    Uses the distilled 2B model (3.9 GB) with DMD2 solver (4 inference steps).

    Requires::

        pip install diffusers>=0.39.0 accelerate

    Usage::

        gen = DiffusersCosmosGenerator(
            model_id="nvidia/Cosmos-Predict2.5-2B",
            variant="distilled",
        )
        video = gen.generate_reference(init_image, "open the drawer")
    """

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Predict2.5-2B",
        variant: str = "distilled",
        device: str = "cuda",
        num_steps: int = 4,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = True,
        enable_attention_slicing: bool = True,
        temperature: float = 1.0,
        ref_temperature: float = 0.3,
    ):
        self.model_id = model_id
        self.variant = variant
        self.device = device
        self.num_steps = num_steps
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_attention_slicing = enable_attention_slicing
        self.temperature = temperature
        self.ref_temperature = ref_temperature
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import Cosmos2_5_PredictBasePipeline

            # Mock safety checker (cosmos_guardrail not available)
            import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as _cp
            class _DummySC:
                def __getattr__(self, _):
                    return lambda *a, **k: True
                def to(self, *a, **k): return self
                def eval(self): return self
                def check_text_safety(self, p): return True
                def check_video_safety(self, video, *a, **k): return video
            _cp.CosmosSafetyChecker = _DummySC

            # Load local diffusers-format weights
            local = str(self.model_id)
            if not Path(local).exists() and self.variant:
                local = str(Path("weights/cosmos-diffusers-2b").resolve())
            self._pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
                local, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, local_files_only=True,
            )
            if self.enable_cpu_offload:
                self._pipe.enable_model_cpu_offload()
            if self.enable_attention_slicing:
                self._pipe.enable_attention_slicing()
        return self._pipe

    def generate_candidates(
        self, init_image: np.ndarray, task_text: str,
        num_candidates: int = 10, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        videos = []
        for i in range(num_candidates):
            videos.append(self._run(init_image, task_text, num_frames, self.temperature, seed=i, goal_image=goal_image))
        return videos

    def generate_reference(
        self, init_image: np.ndarray, task_text: str, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._run(init_image, task_text, num_frames, self.ref_temperature, seed=0, goal_image=goal_image)

    def _run(
        self, init_image: np.ndarray, task_text: str,
        num_frames: int, temperature: float, seed: int = 0,
        goal_image: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch
        pipe = self._get_pipe()
        h, w = init_image.shape[:2]
        h = (h // 16) * 16
        w = (w // 16) * 16
        kwargs = dict(
            prompt=task_text,
            image=Image.fromarray(init_image).convert("RGB"),
            num_frames=num_frames if num_frames > 1 else 1,
            height=h, width=w,
            num_inference_steps=self.num_steps,
            generator=torch.Generator().manual_seed(seed),
        )
        if goal_image is not None:
            kwargs["video"] = Image.fromarray(goal_image).convert("RGB")  # Video2World: last-frame conditioning
        with torch.inference_mode():
            output = pipe(**kwargs)
        # CosmosPipelineOutput.frames is list[Tensor[H,W,C]] or list[Tensor[T,H,W,C]]
        frames = output.frames[0]
        if isinstance(frames, list):
            return np.stack([np.array(f) for f in frames])
        if isinstance(frames, torch.Tensor):
            f = frames.cpu().numpy()
            if f.ndim == 4:   # [T, H, W, C]
                f = f[0] if f.shape[0] == 1 else f
            return f
        return np.array(frames)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _load_single_video(output_dir: Path, expected_frames: int) -> np.ndarray:
    """Load a single generated video from cosmos output directory."""
    import glob
    mp4_files = sorted(glob.glob(str(output_dir / "*.mp4")))
    if mp4_files:
        return _mp4_to_frames(mp4_files[0], expected_frames)
    pngs = sorted(output_dir.rglob("*.png"))
    if len(pngs) >= 2:
        frames = [np.array(Image.open(p)) for p in pngs[:expected_frames]]
        while len(frames) < expected_frames:
            frames.append(frames[-1])
        return np.stack(frames)
    raise FileNotFoundError(f"No output found in {output_dir}")


def _mp4_to_frames(mp4_path: str, max_frames: int) -> np.ndarray:
    """Extract frames from an mp4 video file."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise RuntimeError("imageio is required to read Cosmos-generated MP4 files.")
    reader = imageio.get_reader(mp4_path)
    frames = []
    stride = max(1, reader.count_frames() // max_frames) if hasattr(reader, "count_frames") else 1
    for i, frame in enumerate(reader):
        if i % stride == 0 and len(frames) < max_frames:
            frames.append(frame)
    reader.close()
    while len(frames) < max_frames:
        frames.append(frames[-1])
    return np.stack(frames)


# ---------------------------------------------------------------------------
# MockFutureGenerator — for smoke tests, no GPU needed
# ---------------------------------------------------------------------------

class MockFutureGenerator(BaseFutureGenerator):
    """Deterministic mock for smoke-testing the full pipeline."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def generate_candidates(
        self, init_image: np.ndarray, task_text: str,
        num_candidates: int = 10, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        del task_text, goal_image
        H, W, C = init_image.shape
        target = np.clip(init_image.astype(np.float32) * 1.5 + 30, 0, 255).astype(np.uint8)
        videos = []
        for i in range(num_candidates):
            frames = []
            for t in range(num_frames):
                if i < num_candidates // 2:
                    alpha = 0.1 + 0.4 * t / max(num_frames - 1, 1)
                    noise = self._rng.uniform(-15, 15, (H, W, C)).astype(np.float32)
                    frame = (alpha * target + (1 - alpha) * init_image + noise * (1 - alpha)).clip(0, 255).astype(np.uint8)
                else:
                    noise = self._rng.uniform(-40, 40, (H, W, C)).astype(np.float32)
                    frame = (0.1 * target + 0.9 * init_image + noise).clip(0, 255).astype(np.uint8)
                frames.append(frame)
            videos.append(np.stack(frames))
        return videos

    def generate_reference(
        self, init_image: np.ndarray, task_text: str, num_frames: int = 16,
        goal_image: np.ndarray | None = None,
    ) -> np.ndarray:
        del task_text, goal_image
        H, W, C = init_image.shape
        target = np.clip(init_image.astype(np.float32) * 1.5 + 30, 0, 255).astype(np.uint8)
        frames = []
        for t in range(num_frames):
            alpha = 0.3 + 0.7 * t / max(num_frames - 1, 1)
            noise = self._rng.uniform(-3, 3, (H, W, C)).astype(np.float32)
            frames.append((alpha * target + (1 - alpha) * init_image + noise).clip(0, 255).astype(np.uint8))
        return np.stack(frames)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def save_video_frames(video: np.ndarray, output_dir: str | Path, prefix: str = "frame", ext: str = ".png") -> list[str]:
    """Save [T, H, W, C] as PNG files, return paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in range(video.shape[0]):
        p = out / f"{prefix}_{t:04d}{ext}"
        Image.fromarray(video[t]).save(str(p))
        paths.append(str(p))
    return paths
