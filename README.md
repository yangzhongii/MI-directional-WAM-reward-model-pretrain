# LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies

This repository contains the cleaned training and evaluation code for **LaWAM**,
a **La**tent **W**orld **A**ction **M**odel for robot policies. LaWAM predicts
future observation features in a frozen visual feature space and injects them as
latent visual subgoals for action generation.

## Paper Overview

LaWAM introduces a latent world-model interface for VLA policies. The overview
figure below summarizes the two-stage pipeline: latent world model learning and
LaWAM policy training with latent visual subgoals.

<p align="center">
  <img src="./assets/lawam_overview.png" alt="LaWAM method overview" width="95%">
</p>

## Index

- [Paper Overview](#paper-overview)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Inference](#inference)
  - [LIBERO](#libero-inference)
  - [RoboTwin](#robotwin-inference)
- [SFT Training](#sft-training)
  - [LIBERO](#libero-sft)
  - [RoboTwin](#robotwin-sft)
- [Checkpoint Notes](#checkpoint-notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## File Structure

```text
starVLA/                 Core LaWAM model, dataloaders, training loop, configs
latent_action_model/     LaWM / latent-action model code and utilities
deployment/              Policy server implementations for evaluation
examples/LIBERO/         LIBERO evaluation scripts
examples/Robotwin/       RoboTwin evaluation scripts and native policy adapter
requirements.txt         LaWAM-side Python dependencies
train_lawam.sh
train_lawam_distributed.sh
```

## Environment Setup

Clone the repository into a directory named `LaWAM`, then create the
policy/training environment from that repository root:

```bash
git clone https://github.com/Nemo-1024/LaWAM.git LaWAM
cd LaWAM

conda create -n lawam python=3.10 -y
conda activate lawam

pip install -U pip
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
```

If the local CUDA/PyTorch build is incompatible with `flash-attn==2.8.3`,
install a matching `flash-attn` wheel manually and then re-run
`pip install -e .`.

Quick import check:

```bash
python - <<'PY'
import torch
import starVLA
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
PY
```

## Model Preparation

This step is required before both training and inference.
All commands in this section and the training sections assume the current
directory is the `LaWAM` repository root.

LaWAM always needs:

- Base VLM:
  [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- LaWM/LAM checkpoint and config:
  [lawam_lam](https://huggingface.co/jialei02/lawam_lam)

Downloadable resources used by the released configs:

| Type | Resource | Used for | Local path expected by examples/configs |
| --- | --- | --- | --- |
| Base VLM weights | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Training and inference | `results/Checkpoints/qwen3_weights` |
| LAM checkpoint/config | [lawam_lam](https://huggingface.co/jialei02/lawam_lam) | Training and inference | `latent_action_model/logs/dino_large_vae/lam_release` |
| LaWAM pretraining checkpoint | [lawam_pretrain](https://huggingface.co/jialei02/lawam_pretrain) | LIBERO/RoboTwin SFT initialization | `results/Checkpoints/pretrain/lawam_pretrain` |
| LIBERO SFT checkpoint | [lawam_libero_sft_release](https://huggingface.co/jialei02/lawam_libero_sft_release) | LIBERO benchmark inference | `results/Checkpoints/libero/lawam_libero_sft_release` |
| RoboTwin SFT checkpoint | [lawam_robotwin_sft_release](https://huggingface.co/jialei02/lawam_robotwin_sft_release) | RoboTwin evaluation | `results/Checkpoints/robotwin/lawam_robotwin_sft_release` |
| LIBERO SFT dataset | [libero_merged_no_noops_20hz](https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz) | LIBERO SFT | `dataset/libero_merged_no_noops_20hz` |
| RoboTwin SFT dataset | [robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged) | RoboTwin SFT | `dataset/robotwin_merged` |

Download Qwen3-VL into the path recorded by the provided configs:

```bash
mkdir -p results/Checkpoints/qwen3_weights

hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir results/Checkpoints/qwen3_weights
```

Download the LaWM/LAM checkpoint and YAML config into the paths recorded by the
provided configs:

```bash
hf download jialei02/lawam_lam \
  --local-dir latent_action_model/logs/dino_large_vae/lam_release
```

The policy server loads Qwen3-VL and LAM from the checkpoint config. The
released configs already point to the paths above.

## Inference

Inference uses two environments:

- the `lawam` environment above for policy loading and serving;
- a separate simulator environment for LIBERO or RoboTwin.

Run LIBERO first if you only need one smoke test. RoboTwin setup is separate and
usually heavier.

### LIBERO Inference

#### 1. Install The LIBERO Simulator

Install LIBERO in a separate environment following the official repository:

https://github.com/Lifelong-Robot-Learning/LIBERO

Example layout:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO

# Create the LIBERO simulator environment with Python 3.10, then install
# LIBERO following the official instructions.
conda create -n libero python=3.10 -y
conda activate libero

# Then set:
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero_env/bin/python
```

After completing the official LIBERO installation, install the MuJoCo version
used by this repository in the Python 3.10 LIBERO simulator environment:

```bash
conda activate <libero_env>
pip install mujoco==3.3.2
```

#### 2. Run LIBERO Benchmark

Set the policy checkpoint path. Use a released LIBERO checkpoint if available
from [lawam_libero_sft_release](https://huggingface.co/jialei02/lawam_libero_sft_release),
or a checkpoint produced by [LIBERO SFT](#libero-sft).

```bash
cd LaWAM
conda activate lawam

hf download jialei02/lawam_libero_sft_release \
  --local-dir results/Checkpoints/libero/lawam_libero_sft_release

export CKPT_PATH=results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt
export LIBERO_HOME=/path/to/LIBERO
export LIBERO_PYTHON=/path/to/libero_env/bin/python
export STAR_VLA_PYTHON="$(which python)"

SUITES="libero_10 libero_goal libero_object libero_spatial" \
NUM_TRIALS_PER_TASK=50 \
NUM_WORKERS=4 \
GPU_IDS="0 1 2 3" \
OUTPUT_ROOT=results/eval_runs/libero \
LIBERO_CKPT_ALIAS=lawam_libero_sft \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$CKPT_PATH"
```

Outputs are saved under:

```text
results/eval_runs/libero/<ckpt_alias>/<run_tag>/
  run_meta.json
  suites/<suite_name>/eval.log
```

### RoboTwin Inference

#### 1. Install The RoboTwin Simulator

Install RoboTwin in a separate environment following the official repository:

https://github.com/RoboTwin-Platform/RoboTwin

Example layout:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git ../RoboTwin

# Create and install the RoboTwin simulator environment following the official
# RoboTwin instructions. Then set:
export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python
```

After completing the official RoboTwin installation, install the extra packages
used by this repository in the RoboTwin simulator environment:

```bash
conda activate <robotwin_env>
pip install \
  accelerate==1.5.2 \
  json-numpy==2.1.1 \
  websockets==15.0.1 \
  msgpack==1.1.2 \
  rich==14.2.0 \
  omegaconf==2.3.0
```

#### 2. Run RoboTwin Evaluation

Native policy mode imports the policy adapter directly from this repository:

```bash
cd LaWAM

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=8

hf download jialei02/lawam_robotwin_sft_release \
  --local-dir results/Checkpoints/robotwin/lawam_robotwin_sft_release

bash examples/Robotwin/eval_files/eval_direct.sh \
  lift_pot \
  demo_clean \
  results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt \
  lawam_robotwin_sft \
  0 \
  0
```

Bridge mode starts a LaWAM websocket policy server plus a RoboTwin bridge
process:

```bash
cd LaWAM

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python
export STAR_VLA_PYTHON=/path/to/lawam_env/bin/python
export POLICY_CKPT_PATH=results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt

# Terminal 1: policy server
bash examples/Robotwin/eval_files/run_policy_server.sh "$POLICY_CKPT_PATH" 0 6694

# Terminal 2: RoboTwin bridge and simulator
PORT=6694 ROBOTWIN_NUM_SLOTS=4 ROBOTWIN_REPLAN_STEPS=8 \
bash examples/Robotwin/eval_files/eval.sh lift_pot demo_clean lawam_robotwin_sft 0 0
```

Full RoboTwin benchmark:

```bash
cd LaWAM

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python
export STAR_VLA_PYTHON=/path/to/lawam_env/bin/python

GPU_IDS="0 1 2 3 4 5 6 7" \
NUM_WORKERS=8 \
ROBOTWIN_NUM_SLOTS=2 \
ROBOTWIN_TEST_NUM=100 \
ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1 \
ROBOTWIN_REPLAN_STEPS=8 \
ROBOTWIN_EVAL_ROOT=results/eval_runs/robotwin \
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt \
  demo_clean
```

Outputs are saved under:

```text
results/eval_runs/robotwin/<ckpt_alias>__<task_config>/<run_tag>/
  tasks/<task_name>/run.log
  tasks/<task_name>/summary.json
```

## SFT Training

SFT training uses the same Qwen3-VL and LAM files prepared in
[Model Preparation](#model-preparation). It also needs:

- LaWAM pretraining checkpoint:
  [lawam_pretrain](https://huggingface.co/jialei02/lawam_pretrain)
- benchmark-specific SFT data

Download the pretraining checkpoint:

```bash
mkdir -p results/Checkpoints/pretrain/lawam_pretrain/final_model

hf download jialei02/lawam_pretrain \
  --local-dir results/Checkpoints/pretrain/lawam_pretrain
```

All training is launched through `train_lawam.sh` for a single node
or `train_lawam_distributed.sh` for multi-node jobs. Extra arguments
are forwarded to OmegaConf, so config fields can be overridden with
`--a.b.c value`.

### LIBERO SFT

#### 1. Download LIBERO SFT Data

The preprocessed LIBERO SFT dataset is available at:

[libero_merged_no_noops_20hz](https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz)

This dataset is derived from the public
[IPEC-COMMUNITY/libero-benchmark-dataset](https://huggingface.co/collections/IPEC-COMMUNITY/libero-benchmark-dataset)
release. Compared with the public source, this release merges the four LIBERO
subsets and converts the data to LeRobot 3.0 format.

Download it under the unified dataset root used by the provided configs
(`dataset/`) with the directory name expected by `data_mix: libero`:

```bash
mkdir -p dataset

hf download jialei02/libero_merged_no_noops_20hz \
  --repo-type dataset \
  --local-dir dataset/libero_merged_no_noops_20hz
```

Expected layout:

```text
dataset/
  libero_merged_no_noops_20hz/
    meta/
    data/
    videos/
```

#### 2. Launch LIBERO SFT

```bash
cd LaWAM
conda activate lawam

CONFIG=starVLA/config/training/starvla_train_libero_pre_detach_distill.yaml

bash train_lawam.sh \
  --config_yaml "$CONFIG" \
  --run_id libero_sft_from_pretrain
```

The output checkpoint is written under:

```text
results/Checkpoints/libero/<timestamp>+<run_id>/
```

### RoboTwin SFT

#### 1. Download RoboTwin SFT Data

The preprocessed RoboTwin SFT dataset is available at:

[robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged)

This dataset uses RoboTwin EEF actions and is derived from the lingbot-va
release, specifically
[robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot/tree/main/lerobot_robotwin_eef_aug_500/beat_block_hammer-aloha-agilex_randomized_500-1000).
Compared with that public source, this release converts the data to LeRobot 3.0
format.

The provided RoboTwin SFT config uses `data_mix: robotwin_eef_30hz`, which
expects a dataset directory named `robotwin_eef_all_v30_merged_slow30fps`.
Download the dataset and create that name if needed:

```bash
mkdir -p dataset

hf download jialei02/robotwin_merged \
  --repo-type dataset \
  --local-dir dataset/robotwin_merged

ln -sfn robotwin_merged dataset/robotwin_eef_all_v30_merged_slow30fps
```

Expected layout:

```text
dataset/
  robotwin_merged/
    meta/
    data/
    videos/
  robotwin_eef_all_v30_merged_slow30fps -> robotwin_merged
```

#### 2. Launch RoboTwin SFT

```bash
cd LaWAM
conda activate lawam

CONFIG=starVLA/config/training/starvla_train_robotwin_eef_pretrain.yaml

bash train_lawam.sh \
  --config_yaml "$CONFIG" \
  --run_id robotwin_sft_from_pretrain
```

The output checkpoint is written under:

```text
results/Checkpoints/robotwin/<timestamp>+<run_id>/
```

For multi-node training, use `train_lawam_distributed.sh` with the
same config:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_host> MASTER_PORT=29500 \
bash train_lawam_distributed.sh \
  --config_yaml "$CONFIG"
```

Run the same command on every node and set `NODE_RANK` accordingly.

## Checkpoint Notes

Training checkpoints are regular PyTorch `.pt` files that include the model
state and the merged training config. Evaluation scripts use the checkpoint
config to recover dataset statistics, action normalization, Qwen3-VL source,
and LAM source. When moving checkpoints across machines, make sure these paths
are valid in the new environment.

- LIBERO checkpoints should use `datasets.vla_data.data_mix: libero`.
- RoboTwin EEF checkpoints should use `datasets.vla_data.data_mix:
  robotwin_eef_30hz` or another supported RoboTwin EEF mixture.
- `framework.qwenvl.base_vlm` must point to Qwen3-VL-2B-Instruct or a local
  copy of that model.
- `framework.action_model.lam_ckpt_path` and
  `framework.action_model.lam_yaml_path` must point to a matching LAM checkpoint
  and YAML config.

## Citation

```bibtex
@misc{chen2026lawam,
  title = {LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies},
  author = {Chen, Jialei and Wang, Kai and Chen, Kang and Chen, Shuaihang and Gao, Feng and Tang, Wenhao and Li, Zhiyuan and Liu, Weilin and Yao, Zhuyu and Li, Boxun and Xu, Yuanbo and Yu, Chao},
  journal = {arXiv preprint arXiv:2606.15768},
  year = {2026},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
}
```

## Acknowledgements

This codebase is based on StarVLA and retains its MIT license. It also builds on
open-source robotics and VLM components including LeRobot, Qwen-VL, DINO,
LIBERO, and RoboTwin.
