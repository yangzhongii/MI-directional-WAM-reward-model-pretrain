"""Project inference wrapper for base and LoRA-adapted Cosmos action models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


VIEW_NAMES = ("third_view", "wrist_view")


def pack_dual_view_image(third_view: np.ndarray, wrist_view: np.ndarray) -> np.ndarray:
    """Pack synchronized camera images using Cosmos' local multiview layout."""

    third = np.asarray(third_view, dtype=np.uint8)
    wrist = np.asarray(wrist_view, dtype=np.uint8)
    if third.ndim != 3 or third.shape[-1] != 3:
        raise ValueError(f"third_view must be [H,W,3], got {third.shape}")
    if wrist.shape != third.shape:
        raise ValueError(f"wrist_view must match third_view, got {wrist.shape} and {third.shape}")
    return np.concatenate((third, wrist), axis=1)


def unpack_dual_view_video(video: np.ndarray) -> dict[str, np.ndarray]:
    """Split a Cosmos multiview canvas into named synchronized RGB streams."""

    value = np.asarray(video)
    if value.ndim != 4 or value.shape[-1] not in (3, 4):
        raise ValueError(f"Packed Cosmos video must be [T,H,W,C], got {value.shape}")
    if value.shape[2] % 2:
        raise ValueError(f"Packed Cosmos video width must be even, got {value.shape[2]}")
    boundary = value.shape[2] // 2
    return {
        "third_view": np.ascontiguousarray(value[:, :, :boundary, :3], dtype=np.uint8),
        "wrist_view": np.ascontiguousarray(value[:, :, boundary:, :3], dtype=np.uint8),
    }


def step_inference_dual_view(
    model,
    third_view: np.ndarray,
    wrist_view: np.ndarray,
    *,
    action: np.ndarray,
    guidance: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run one jointly packed action-conditioned inference step."""

    packed = pack_dual_view_image(third_view, wrist_view)
    next_packed, packed_video = model.step_inference(
        packed, action=action, guidance=guidance, seed=seed
    )
    streams = unpack_dual_view_video(packed_video)
    next_streams = unpack_dual_view_video(np.asarray(next_packed)[None, ...])
    return (
        {name: frames[0] for name, frames in next_streams.items()},
        streams,
    )


def lora_experiment_options(
    *, tokenizer: Path, use_lora: bool, lora_rank: int = 32, lora_alpha: int = 32,
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
) -> list[str]:
    options = [
        f"+model.config.tokenizer.vae_pth={tokenizer}",
        "model.config.text_encoder_config.compute_online=false",
        "model.config.net.sac_config.mode=none",
    ]
    if use_lora:
        options.extend((
            "model.config.use_lora=true",
            f"model.config.lora_rank={int(lora_rank)}",
            f"model.config.lora_alpha={int(lora_alpha)}",
            f"model.config.lora_target_modules='{lora_target_modules}'",
            "model.config.init_lora_weights=true",
        ))
    return options


def load_empty_reason_embedding(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        value = torch.from_numpy(np.load(path))
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if tuple(value.shape) != (1, 512, 100352):
        raise ValueError(f"Expected Reason1 embedding [1,512,100352], got {tuple(value.shape)}")
    return value.to(dtype=torch.bfloat16, device="cpu").contiguous()


def build_action_inference(
    *, experiment: str, checkpoint: Path, tokenizer: Path, empty_reason_embedding: Path,
    use_lora: bool,
):
    """Instantiate NVIDIA's pipeline with project-local assets and LoRA options.

    Importing is delayed so unit tests and CPU preflight do not initialize CUDA.
    """
    from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
        ActionVideo2WorldInference,
    )
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

    class ProjectActionInference(ActionVideo2WorldInference):
        def __init__(self) -> None:
            self.experiment_name = experiment
            self.ckpt_path = str(checkpoint)
            self.s3_credential_path = ""
            self.context_parallel_size = 1
            self.process_group = None
            self.distilled = False
            self.num_steps = 4
            model, config = load_model_from_checkpoint(
                experiment_name=experiment,
                s3_checkpoint_dir=str(checkpoint),
                config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
                load_ema_to_reg=True,
                experiment_opts=lora_experiment_options(tokenizer=tokenizer, use_lora=use_lora),
            )
            self.model = model
            self.config = config
            self.batch_size = 1
            self.neg_t5_embeddings = None
            self._empty_reason_embedding = load_empty_reason_embedding(empty_reason_embedding)

        def _get_data_batch_input(
            self, video: torch.Tensor, prompt: str, num_conditional_frames: int = 1,
            negative_prompt: str = "", use_neg_prompt: bool = False,
        ) -> dict:
            if use_neg_prompt:
                raise ValueError("Project action inference uses the fixed empty Reason1 embedding without CFG text.")
            _, _, _, height, width = video.shape
            return {
                "dataset_name": "video_data",
                "video": video.cuda(),
                "fps": torch.full((1,), 4.0, device="cuda", dtype=torch.bfloat16),
                "padding_mask": torch.zeros(1, 1, height, width, device="cuda", dtype=torch.bfloat16),
                "num_conditional_frames": num_conditional_frames,
                "t5_text_embeddings": self._empty_reason_embedding.cuda(),
            }

    return ProjectActionInference()
