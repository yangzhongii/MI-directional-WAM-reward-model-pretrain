"""Project registration for the official Cosmos-Predict2.5 LoRA recipe on LIBERO.

This module does not reimplement LoRA.  It directly reuses NVIDIA's official
LoRA optimizer, scheduler, trainer and model config, then layers them on top of
the official action-conditioned experiment.  The only project-owned pieces are
the LIBERO dual-view dataloaders and local checkpoint/data paths.
"""

from __future__ import annotations

import copy
import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.configs.action_conditioned.data import get_sampler
from cosmos_predict2._src.predict2.action.datasets.dataset_mv_local import ActionConditionedMultiViewDataset
from cosmos_predict2.experiments.base.cosmos_nemo_assets_lora import (
    _lora_model_config,
    _lora_model_parallel,
    _lora_optimizer,
    _lora_scheduler,
    _lora_trainer,
)


EXPERIMENT_NAME = "libero_action_lora_official"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _dataset(*, mode: str):
    data_root = _required_env("COSMOS_LIBERO_DATA_ROOT")
    return L(ActionConditionedMultiViewDataset)(
        train_annotation_path=_required_env("COSMOS_LIBERO_TRAIN_ANNOTATION"),
        val_annotation_path=_required_env("COSMOS_LIBERO_VAL_ANNOTATION"),
        test_annotation_path=_required_env("COSMOS_LIBERO_TEST_ANNOTATION"),
        video_path=data_root,
        fps_downsample_ratio=int(_required_env("COSMOS_LIBERO_FPS_DOWNSAMPLE")),
        num_action_per_chunk=int(_required_env("COSMOS_LIBERO_ACTION_CHUNK")),
        cam_ids=[[0], 1],
        accumulate_action=False,
        video_size=[256, 320],
        val_start_frame_interval=1,
        mode=mode,
        load_t5_embeddings=True,
        state_key="state",
        gripper_key="continuous_gripper_state",
        gripper_rescale_factor=1.0,
    )


def build_experiment() -> LazyDict:
    train_dataset = _dataset(mode="train")
    val_dataset = _dataset(mode="val")
    train_loader = L(DataLoader)(
        dataset=train_dataset,
        sampler=L(get_sampler)(dataset=train_dataset),
        batch_size=1,
        drop_last=True,
    )
    val_loader = L(DataLoader)(
        dataset=val_dataset,
        sampler=L(get_sampler)(dataset=val_dataset),
        batch_size=1,
        drop_last=True,
    )

    model = copy.deepcopy(_lora_model_config)
    model["config"].update(
        {
            # Action-conditioned Predict2.5 uses one conditional image and a
            # 12-step action chunk.  Keep these action-specific constraints on
            # top of NVIDIA's official LoRA model config.
            "min_num_conditional_frames": 1,
            "max_num_conditional_frames": 1,
            "conditional_frames_probs": None,
            "state_t": 1 + 12 // 4,
            "text_encoder_config": {"compute_online": False},
            "tokenizer": {"vae_pth": _required_env("COSMOS_WAN2PT1_VAE_PATH")},
            "net": {"action_dim": 7, "temporal_compression_ratio": 4},
        }
    )

    checkpoint = {
        "load_path": _required_env("COSMOS_ACTION_BASE_CHECKPOINT"),
        "load_training_state": False,
        "strict_resume": False,
        "save_iter": 200,
        "load_from_object_store": {"enabled": False},
        "save_to_object_store": {"enabled": False},
    }

    trainer = copy.deepcopy(_lora_trainer)
    trainer_callbacks = trainer.get("callbacks", {})
    trainer_callbacks.pop("wandb", None)
    trainer_callbacks.pop("wandb_10x", None)
    # The upstream example validates every few iterations.  For LIBERO this
    # spends most of the wall time on sampling rather than LoRA optimization.
    # Keep the official callbacks and validation implementation, but use a
    # project-scale cadence.
    trainer["validation_iter"] = 200
    # NVIDIA's trainer validates over the full validation dataloader when this
    # is unset.  LIBERO's exported validation split is much larger than the
    # upstream example, so cap each periodic validation to a small smoke set.
    trainer["max_val_iter"] = 20

    return LazyDict(
        dict(
            defaults=[
                "/experiment/ac_reason_embeddings_rectified_flow_2b_256_320",
                "_self_",
            ],
            job=dict(
                project="cosmos_predict2_action_conditioned",
                group="libero_official_lora",
                name="2b_libero_dual_view_official_lora",
                wandb_mode="disabled",
            ),
            optimizer=copy.deepcopy(_lora_optimizer),
            scheduler=copy.deepcopy(_lora_scheduler),
            trainer=trainer,
            model=model,
            model_parallel=copy.deepcopy(_lora_model_parallel),
            checkpoint=checkpoint,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
        ),
        flags={"allow_objects": True},
    )


def register_experiment() -> None:
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=EXPERIMENT_NAME,
        node=build_experiment(),
    )

