# MI-Directional Reward Pretraining from World-Model Futures

*A modular mutual-information reward-learning extension built on [RLinf/LaWAM](https://github.com/RLinf/LaWAM).*

<p>
  <a href="https://arxiv.org/abs/2606.15768"><img alt="arXiv" height="24" src="https://img.shields.io/badge/arXiv-2606.15768-b31b1b.svg"></a>
  <a href="https://rlinf.github.io/LaWAM/"><img alt="Project Page" height="24" src="https://img.shields.io/badge/Project_Page-LaWAM-2ea44f.svg"></a>
  <a href="https://nemo-1024.github.io/blogs/lawam/"><img alt="Blog" height="24" src="https://img.shields.io/badge/Blog-LaWAM-0a66c2.svg"></a>
  <br>
  <a href="https://huggingface.co/collections/jialei02/lawam-checkpoints"><img alt="Hugging Face Model Collection" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Collection-f7c843"></a>
  <a href="https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz"><img alt="Hugging Face Dataset - LIBERO" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843"></a>
  <a href="https://huggingface.co/datasets/jialei02/robotwin_merged"><img alt="Hugging Face Dataset - RoboTwin" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843"></a>
</p>

---

## Research Overview

Large world models and latent world-action models can generate or predict multiple
plausible task futures, but generating futures alone does not provide calibrated
reward supervision. This extension studies whether reward-model supervision can be
constructed without task-specific real-robot rollouts or manually labeled robot
reward data, by deriving pseudo-preferences from imagined trajectories.

The high-level pipeline is:

> World-model generated futures → latent feature extraction → mutual-information
> potential scoring → directional progress ranking → pseudo-preference construction
> → reward-model SFT

Given a language-conditioned task, a success reference trajectory, and multiple
candidate imagined trajectories, a frozen visual or world-model encoder produces
latent features. Frame-level mutual-information (MI) potential between candidate
and reference features is aggregated into a trajectory-level directional progress
score. Candidates with larger MI-potential growth are ranked higher, and top-vs-bottom
pairs are converted into preference labels for pairwise reward-model fine-tuning.

The MI potential acts as a **pseudo-supervision generator**, not as an online
controller or an MI-MPC method. "Direction" refers to ranking imagined future
trajectories by MI-potential growth; it does **not** mean directly
differentiating through a generative model to control the robot.

This initial objective is reward pretraining without task-specific real-robot
reward labels. External robot benchmarks or real trajectories are still required
for evaluation and validation. The extension is designed as a modular component
built on the LaWAM codebase and does **not** modify the original policy training
pipeline.

## Architecture

<p align="center">
  <img src="./assets/MI guided protential RLPD.png" alt="MI-directional reward pretraining pipeline overview" width="95%">
</p>

**Stage 1:** World-model future generation produces candidate imagined trajectories
and a success reference trajectory for a given language-conditioned task.
**Stage 2:** MI-potential direction scoring ranks each candidate by its
frame-level MI growth toward the success reference.
**Stage 3:** Pseudo-preference reward-model SFT trains a trajectory reward head
with pairwise ranking loss on the scored candidates.

## Relation to Dame and Marchand's MI Visual Servoing

This work is inspired by:

> Amaury Dame and Eric Marchand, "Mutual Information-Based Visual Servoing,"
> IEEE Transactions on Robotics, 2011.

The original Dame method uses Shannon mutual information as a primary
image-alignment objective, builds differentiable soft histograms using B-spline
kernels, and derives an MI gradient and Hessian with respect to camera pose for
direct visual-servoing control.

**Our adaptation transfers this information-potential principle from camera-pose
optimization to generated-trajectory progress estimation**, without reproducing
the camera controller:

| Dame TRO | This Repository |
|---|---|
| Current image $I(r)$ | Candidate imagined future trajectory |
| Desired image $I^*$ | Success reference trajectory |
| Camera pose $r$ | Candidate trajectory branch |
| MI alignment objective | Temporally aligned latent MI potential |
| B-spline soft histograms | B-spline soft-histogram MI estimator (`DameSoftHistogramMI`) |
| Camera velocity / control law | Pseudo-preference supervision and potential reward distillation |
| MI controller (6-DOF servo) | Reward-potential student model (`StatePotentialRewardModel`) |

**We do not reuse** Dame's camera interaction matrix, Hessian controller, or
six-DOF servo law. "Direction" in this work means the progression of an
imagined trajectory through a successful reference sequence — not
differentiating through a generative model to output robot actions.

Key components adopted from Dame & Marchand (2011):

- **B-spline soft histograms** for differentiable MI estimation between
  latent token features (not pixels).
- **Monotonic temporal alignment** via dynamic programming, analogous to
  how Dame aligns image regions across views.
- **MI as a progress signal** rather than as a direct control objective.

Key components deliberately not adopted:

- Camera interaction matrix and pose Jacobian.
- Hessian-based second-order optimization.
- Six-DOF velocity control law.
- Real-time image-stream processing.

### Citation

```bibtex
@article{dame2011mutual,
  title = {Mutual Information-Based Visual Servoing},
  author = {Dame, Amaury and Marchand, Eric},
  journal = {IEEE Transactions on Robotics},
  volume = {27},
  number = {5},
  pages = {958--969},
  year = {2011},
}
```

## Relationship to LaWAM

This extension is a **modular reward-learning addition**, not a replacement for
LaWAM. The original LaWAM policy pipeline is left intact.

| Component | Original LaWAM | MI Reward Extension |
|---|---|---|
| Main purpose | Latent visual subgoals for action generation | Directional reward supervision |
| Main input | Robot observations and language instructions | Candidate futures and success references |
| Latent representation | Frozen visual feature space (DINO / V-JEPA) | Reused or adapted LaWAM / DINO feature space |
| Main output | Robot action chunks | Scalar reward or trajectory preference |
| Training objective | Policy / latent world-action model training | Pairwise reward ranking (Bradley-Terry) |
| Real-robot data | Used by original training and evaluation pipeline | Not required for pseudo-label construction; needed for validation |

## Method

### Generated Futures and Success References

For each language-conditioned task $g$, a world model or latent world-action
model produces:

- A **success reference** $\tau^+ = (I^+_0, \dots, I^+_T)$ — a trajectory
  known or predicted to complete the task.
- Multiple **candidate trajectories** $\tau_i = (I_{i,0}, \dots, I_{i,T})$ —
  imagined futures that may vary in task progress.

### Latent Feature Extraction

A frozen visual encoder $\mathcal{F}$ (DINOv3 or LaWAM LAM) maps each frame into
a latent feature vector conditioned on the task instruction:

$$z_{i,t} = \mathcal{F}(I_{i,t}, g), \quad z^+_s = \mathcal{F}(I^+_s, g)$$

### MI-Potential Direction Scoring

The frame-level MI potential between a candidate frame and the success reference is:

$$\Phi_{i,t} = \max_s \text{MI}(z_{i,t}, z^+_s)$$

Two MI estimators are provided: a **Gaussian correlation proxy** (default) and a
**soft-histogram estimator**. The trajectory-level directional score aggregates
MI-potential growth across consecutive frames:

$$S^\Delta_i = \frac{1}{T-1} \sum_{t=0}^{T-2} \bigl[\gamma \cdot \Phi_{i,t+1} - \Phi_{i,t}\bigr]$$

A larger $S^\Delta_i$ indicates that the candidate future follows a direction
with stronger progress toward the successful task reference.

### Pseudo-Preference Construction

Candidates are ranked by $S^\Delta_i$. Preference pairs are formed from top-$k$
vs. bottom-$k$ candidates when the score margin exceeds a threshold:

$$S^\Delta_i > S^\Delta_j + \text{margin} \implies \tau_i \succ \tau_j$$

### Reward-Model SFT

A lightweight trajectory reward head $R_\phi$ is trained with pairwise ranking
loss (Bradley-Terry / log-sigmoid):

$$\mathcal{L}_{\text{rank}} = -\log \sigma\bigl(R_\phi(\tau_{\text{chosen}}, g) - R_\phi(\tau_{\text{rejected}}, g)\bigr)$$

The reward head pools per-frame features via mean pooling and last-frame
pooling, then passes the concatenated representation through a small MLP.
This is more precisely **preference-based reward-model fine-tuning**; we use
"SFT" to emphasize the supervised nature of pseudo-label training.

## Repository Status

| Module | Status | Description |
|---|---|---|
| LaWAM backbone | Available | Upstream latent world-action model (starVLA, latent_action_model) |
| Dame B-spline MI (`dame_soft_histogram`) | Implemented | B-spline soft-histogram MI with channelwise/flattened modes and NMI |
| Token correspondence | Implemented | same_index, nearest_cosine, pooled_window strategies |
| Monotonic temporal alignment | Implemented | Viterbi DP alignment with optional soft-DTW fallback |
| Directional potential scoring | Implemented | Multi-component score: endpoint, positive gain, regression penalty, stage progress |
| Legacy delta scoring | Preserved | `gaussian_mi_proxy` and `histogram_mi` retained as ablation modes |
| MI scoring (`mi_potential`) | Available | Gaussian correlation proxy and basic soft-histogram |
| Trajectory delta scoring | Implemented | Frame-potential aggregation with $\gamma$-weighted delta |
| Preference generation | Implemented | Top-$k$ vs. bottom-$k$ and adjacent-ranked modes with confidence and versioning |
| State-potential reward model | Implemented | GRU/MLP/transformer architectures with per-timestep potential output |
| Multi-objective distillation loss | Implemented | Rank + potential + direction losses with confidence weighting |
| Feature extraction (DINOv3) | Implemented | Local-weight DINO extractor with token-level support |
| Feature extraction (LaWAM LAM) | Implemented | Wraps LaWAM `LatentLAMModel.extract_vision_features` |
| Feature caching | Implemented | Per-trajectory `.pt` cache with token-level support |
| LIBERO manifest builder | Implemented | Parses LIBERO eval runs into trajectory manifests |
| RoboTwin manifest builder | Implemented | Parses RoboTwin eval runs into trajectory manifests |
| Baselines | Implemented | Pixel MSE, latent cosine, pooled correlation, unaligned Dame MI |
| Pairwise ranking evaluation | Implemented | Accuracy, reward margin, score correlation |
| Progress correlation evaluation | Implemented | Temporal progress correlation and success/failure AUC |
| RBM-EVAL benchmark adapter | Implemented | Official reward alignment, policy ranking, and quality preference samplers/metrics |
| Shell script wrappers | Implemented | Scripts under `mi_reward/scripts/` |
| Smoke test | Implemented | Synthetic end-to-end pipeline test |
| Unit tests | Implemented | Soft histogram, temporal alignment, reward loss tests |
| Cosmos-Predict data adapter | Implemented | CLI-based adapter (`mi_reward/data/cosmos_generator.py`) |
| MI Potential Field | Implemented | Unified module (`mi_reward/scoring/mi_potential_field.py`) with Dame B-spline + gaussian + histogram backends |
| Directional reward distillation | Implemented | `train_reward_distill()` trains StatePotentialRewardModel with rank + potential + direction loss |
| Ablation reward functions | Implemented | latent_distance, cosine_similarity, static MI, directional MI (interchangeable via `MIBackend`) |
| Task-conditioned feature preprocessing | Planned | Current extractors pass task as unused parameter |
| Downstream RL/RLPD integration | Drop-in adapter implemented | RLinf registry/worker wiring must be applied in the RLinf checkout |
| Real-robot reward validation | Integration scaffold implemented | Requires the local Franka/RLinf deployment and a trained checkpoint |

## Repository Structure

```text
starVLA/                   Core LaWAM model, dataloaders, training loop, configs
latent_action_model/       LaWM / latent-action model code and utilities
deployment/                Policy servers + Franka FCI real-world controller
├── model_server/           HTTP/TCP policy servers (Franka, Isaac-GR00T)
└── realworld/              Franka controller, ROS bridge, client (standalone)
examples/LIBERO/           LIBERO evaluation scripts
examples/Robotwin/         RoboTwin evaluation scripts and native policy adapter
mi_reward/                 MI-directional reward pretraining extension
├── configs/               Central YAML configuration
├── data/                  Schema definitions, manifest builders, Cosmos adapter, datasets
├── features/              Feature extractors (DINOv3, LaWAM LAM) and cache store
├── scoring/               MI potential, trajectory delta scoring, preference builder
├── models/                Trajectory reward head
├── training/              Pairwise ranking loss, collator, SFT training loop
├── evaluation/            Ranking accuracy and progress correlation evaluation
└── scripts/               Shell wrappers for the full pipeline
eval/                      Independent RBM-EVAL launcher and YAML configuration
tests/                     Smoke tests (full pipeline + MI backends + ablation)
requirements/              Automated env setup (--instance-data, --reward-eval, --mi-cosmos, --rlpd)
train_lawam.sh             Single-node LaWAM training entrypoint
train_lawam_distributed.sh Multi-node LaWAM training entrypoint
```

## Installation

### Instance Data Environment

Install the single `.venv` used by Stage 1. All third-party source trees are
placed in `.venv/src`, Python packages are installed into `.venv`, and model
weights go in `.venv/models`.

```bash
bash requirements/install.sh --instance-data
# Add --download-weights after Hugging Face login to fetch SAM3, Cosmos
# Predict/Transfer, DINOv3, and LAM checkpoints into .venv/models.
source .venv/bin/activate
```

The MuJoCo backend is the fixed simulator for the rigid MVP. The data script
sets `MUJOCO_GL=egl` for headless rendering.

Before the non-dry-run data command, fill `execution.stages` in
`mi_reward/configs/instance_geoprogress.yaml` with the release-specific SAM3,
Cosmos Transfer/Predict, simulator, and planner commands. Every command uses
the installed `.venv/bin/python` through the `{python}` token and must produce
the JSONL hand-off required by the orchestrator. An empty stage list is only
the pre-generated-candidate ingest mode; it does not launch model inference.

### RBM-EVAL Environment

Install the complete Robometer/RBM-EVAL source tree into the shared `.venv`
without allowing its incompatible Torch 2.8 dependency pins to replace the
Cosmos/SAM3 stack. The source is checked out at a fixed commit under
`.venv/src/robometer`; optional processed benchmark data is stored under
`.venv/datasets/robometer`. The download option also runs Robometer's official
archive extraction script so the sampler can load the cache directly.

```bash
bash requirements/install.sh --reward-eval
# After Hugging Face login, download the processed RBM-1M cache:
bash requirements/install.sh --reward-eval --download-eval-data
```

Run the independent evaluation stage with one launcher and one YAML file:

```bash
bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml
```

The adapter uses Robometer's official `reward_alignment`, `policy_ranking`, and
`quality_preference` samplers and metric compilers. It supports the local
`StatePotentialRewardModel` and legacy `TrajectoryRewardHead` checkpoints. A
`GeoProgressPotential` checkpoint is accepted only when
`geoprogress.relation_sidecar` supplies measured relations and an explicit
successful goal (`goal_frames` or `goal_tokens_path`) for every benchmark
trajectory. RBM-EVAL videos do not contain those geometry inputs, so missing
sidecars fail loudly instead of producing scores from fabricated zero relations.
Results are written to `results/rbm_eval/mi_reward/metrics.json` plus per-dataset
raw result files.

### LaWAM Backbone

Clone the repository and create the training environment:

```bash
git clone https://github.com/RLinf/LaWAM.git LaWAM
cd LaWAM

# Automated install with uv
bash requirements/install.sh

# Activate
source .venv/bin/activate
```

Or manually:

```bash
conda create -n lawam python=3.10 -y
conda activate lawam
pip install -U pip
pip install -r requirements/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
```

Quick import check:

```bash
python - <<'PY'
import torch
import starVLA
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
PY
```

### MI Reward Extension

The `mi_reward/` package is installed by the same `pip install -e .` command
above. Additional runtime dependencies (`Pillow`, `torchvision`, `imageio`,
`PyYAML`) are already included in `requirements/requirements.txt`.

No additional installation steps are required beyond the LaWAM backbone setup.

## Data Format

### Candidate Trajectory (JSONL manifest)

```json
{
  "traj_id": "libero/LIVING_ROOM_SCENE4/task00/episode000",
  "task": "open the drawer",
  "frames": ["path/to/000.png", "path/to/001.png"],
  "source": "libero_eval",
  "split": "eval",
  "metadata": {
    "suite": "libero_goal",
    "task_id": 0,
    "success": true
  }
}
```

### Success Reference (JSONL)

```json
{
  "ref_id": "libero/LIVING_ROOM_SCENE4/task00/success",
  "task": "open the drawer",
  "frames": ["path/to/success_000.png", "path/to/success_001.png"]
}
```

### Preference Pair (JSONL)

```json
{
  "task": "open the drawer",
  "chosen_traj_id": "libero/LIVING_ROOM_SCENE4/task00/episode003",
  "rejected_traj_id": "libero/LIVING_ROOM_SCENE4/task00/episode007",
  "chosen_score": 0.42,
  "rejected_score": -0.08,
  "score_type": "mi_delta"
}
```

These schemas are defined in `mi_reward/data/schema.py` and used throughout the
pipeline.

## Quick Start

### 1. Build trajectory manifests

```bash
# From LIBERO eval run:
python -m mi_reward.data.build_manifest \
  --source_type libero \
  --run_dir results/eval_runs/libero/<ckpt_alias>/<run_tag> \
  --output dataset/mi_reward/manifests/libero_manifest.jsonl \
  --fps 2.0

# From RoboTwin eval run:
python -m mi_reward.data.build_manifest \
  --source_type robotwin \
  --run_dir results/eval_runs/robotwin/<ckpt_alias>__<task_config>/<run_tag> \
  --output dataset/mi_reward/manifests/robotwin_manifest.jsonl \
  --fps 2.0
```

### GeoProgress: verified action-conditioned Cosmos candidates

The GeoProgress route does not treat image-only Cosmos video as physical
supervision. Run the official robot/action-conditioned Cosmos job externally,
then place one JSONL record per candidate containing its frame directory,
action rollout, measured robot-state sequence, measured relation sequence,
parent trajectory, success-reference ID, model ID, and seed. Ingest it with
deterministic feasibility constraints:

```bash
python -m mi_reward.data.build_manifest \
  --source_type cosmos_action_cond \
  --candidate_records dataset/mi_reward/cosmos_action_cond_records.jsonl \
  --feasibility_config dataset/mi_reward/feasibility.json \
  --output dataset/mi_reward/manifests/cosmos_action_cond_train.jsonl
```

The feasibility JSON may specify workspace/joint limits and required relation
fields. Rejected records remain in the manifest for auditability but are not
scored or used by GeoProgress training.

Use LaWAM's native patch tokens, then score and train with the explicit goal
reference and relation sequence:

```bash
python -m mi_reward.features.cached_feature_store \
  --manifest dataset/mi_reward/manifests/cosmos_action_cond_train.jsonl \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --feature_root dataset/mi_reward/geoprogress_features \
  --feature_extractor lawam_lam \
  --lam_config_path latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml \
  --lam_ckpt_path latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  --vision_model_id weights/dinov3-vitb16-pretrain-lvd1689m \
  --strict --tokens

python -m mi_reward.scoring.build_preferences \
  --manifest dataset/mi_reward/manifests/cosmos_action_cond_train.jsonl \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --feature_root dataset/mi_reward/geoprogress_features \
  --output dataset/mi_reward/preferences/geoprogress_train.jsonl \
  --token_features --relation_weight 1.0 --teacher_version geoprogress_v1

python -m mi_reward.training.train_reward_sft \
  --mode geoprogress \
  --manifest dataset/mi_reward/manifests/cosmos_action_cond_train.jsonl \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --preferences dataset/mi_reward/preferences/geoprogress_train.jsonl \
  --feature_root dataset/mi_reward/geoprogress_features \
  --output_dir results/mi_reward/geoprogress
```

`mi_reward/scripts/run_geoprogress.sh` runs those stages and requires the
LaWAM paths; set `CANDIDATE_RECORDS` and `FEASIBILITY_CONFIG` when the manifest
still needs to be ingested.

### Legacy GeoProgress environment

The original offline GeoProgress path can still use a separate `.venv-rlpd`
environment and pre-generated action-conditioned candidate sidecars. It is
kept for compatibility with the older Cosmos workflow.

```bash
cd /path/to/LaWAM
bash requirements/install.sh --rlpd
source .venv-rlpd/bin/activate
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY
```

Download the frozen visual weights once after Hugging Face login:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitb16-pretrain-lvd1689m
hf download jialei02/lawam_lam \
  --local-dir latent_action_model/logs/dino_large_vae/lam_release
```

Only when the same machine must also run Cosmos generation, use the heavier
Cosmos environment instead. It clones the official repository and installs its
CUDA extras; it is unnecessary for the normal offline GeoProgress workflow.

```bash
bash requirements/install.sh --mi-cosmos
source .venv-mi/bin/activate
```

With the reward environment active and the required local LAM weights present,
run the wrapper:

```bash

export LAM_CONFIG_PATH=latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml
export LAM_CKPT_PATH=latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt
export VISION_MODEL_ID=weights/dinov3-vitb16-pretrain-lvd1689m
export CANDIDATE_RECORDS=dataset/mi_reward/cosmos_action_cond_records.jsonl
export FEASIBILITY_CONFIG=dataset/mi_reward/feasibility.json
bash mi_reward/scripts/run_geoprogress.sh
```

Run the focused CPU smoke test before a GPU job. It creates only temporary
verified sidecars and synthetic token caches, then runs one training epoch; it
does not invoke Cosmos or require a GPU:

```bash
python -m pytest -q tests/test_geoprogress_contracts.py
```

After local weights are present, run the existing broader regression suite:

```bash
python -m pytest -q tests/test_mi_reward_smoke.py
```

### 2. Extract and cache features

```bash
python -m mi_reward.features.cached_feature_store \
  --manifest dataset/mi_reward/manifests/train_manifest.jsonl \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --feature_root dataset/mi_reward/features \
  --feature_extractor lawam_lam \
  --lam_config_path latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml \
  --lam_ckpt_path latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt \
  --vision_model_id weights/dinov3-vitb16-pretrain-lvd1689m
```

If local LAM weights are unavailable, omit `--lam_config_path` and
`--lam_ckpt_path` to fall back to the DINOv3 extractor.

### 3. Score trajectories and build preference pairs

```bash
python -m mi_reward.scoring.build_preferences \
  --manifest dataset/mi_reward/manifests/train_manifest.jsonl \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --feature_root dataset/mi_reward/features \
  --output dataset/mi_reward/preferences/train_preferences.jsonl \
  --gamma 0.99 \
  --margin 0.05 \
  --top_k 5 \
  --bottom_k 5
```

### 4. Train the reward model

```bash
python -m mi_reward.training.train_reward_sft \
  --preferences dataset/mi_reward/preferences/train_preferences.jsonl \
  --feature_root dataset/mi_reward/features \
  --output_dir results/mi_reward/reward_head_mvp \
  --batch_size 16 \
  --epochs 5 \
  --lr 1e-4
```

### 5. Evaluate ranking accuracy

```bash
python -m mi_reward.evaluation.eval_reward_ranking \
  --preferences dataset/mi_reward/preferences/val_preferences.jsonl \
  --feature_root dataset/mi_reward/features \
  --ckpt results/mi_reward/reward_head_mvp/pytorch_model.pt
```

### 6. Evaluate temporal progress correlation

```bash
python -m mi_reward.evaluation.eval_progress_corr \
  --manifest dataset/mi_reward/manifests/libero_manifest.jsonl \
  --feature_root dataset/mi_reward/features \
  --success_refs dataset/mi_reward/manifests/success_refs.jsonl \
  --output results/mi_reward/libero_progress_report.json
```

### Instance-Aware Rigid V1

The instance-aware path keeps scene variation, object replacement, physical
state, and generated RGB separate. It accepts the `pick_place`, `push_shape`,
and `peg_insertion` task families in one `rigid_v1` manifest. Configure any
available external stages in `mi_reward/configs/instance_geoprogress.yaml`;
each worker receives `{request}` and `{result}` paths and must return a JSON
result containing its `records_path`.

The data-preparation flow uses one command:

```bash
# Validate all YAML paths and worker hand-offs without launching models.
bash mi_reward/scripts/prepare_instance_data.sh \
  --config mi_reward/configs/instance_geoprogress.yaml \
  --task-suite rigid_v1 \
  --dry-run

# Run SAM3, Cosmos Transfer/Predict, MuJoCo rollouts, and verification.
bash mi_reward/scripts/prepare_instance_data.sh \
  --config mi_reward/configs/instance_geoprogress.yaml \
  --task-suite rigid_v1
```

After the manifest is written, start the LaWAM/MI reward stage separately:

```bash
bash mi_reward/scripts/run_instance_geoprogress.sh \
  --config mi_reward/configs/instance_geoprogress.yaml
```

`generate_instance_rollouts.sh` remains as a compatibility alias for
`prepare_instance_data.sh`.

### Smoke test

```bash
python -m pytest tests/test_mi_reward_smoke.py
```

### Shell wrappers

Shell wrappers under `mi_reward/scripts/` accept environment variables
(`MANIFEST`, `SUCCESS_REFS`, `FEATURE_ROOT`, `PREFERENCES`, `OUTPUT_DIR`,
etc.) in place of CLI flags:

```bash
MANIFEST=dataset/mi_reward/manifests/train_manifest.jsonl \
SUCCESS_REFS=dataset/mi_reward/manifests/success_refs.jsonl \
FEATURE_ROOT=dataset/mi_reward/features \
bash mi_reward/scripts/extract_features.sh
```

---

## LaWAM Backbone

The sections below document the upstream LaWAM training and evaluation pipeline.
This extension does not modify these components.

### Model Preparation

All commands in this section and the training sections assume the current
directory is the repository root.

LaWAM always needs:

- Base VLM:
  [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- LAM vision encoder:
  [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- LaWM/LAM checkpoint and config:
  [lawam_lam](https://huggingface.co/jialei02/lawam_lam)

Downloadable resources used by the released configs:

| Type | Resource | Used for | Local path expected by examples/configs |
| --- | --- | --- | --- |
| Base VLM weights | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Training and inference | `results/Checkpoints/qwen3_weights` |
| DINOv3 vision encoder weights | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | LAM feature extraction | `weights/dinov3-vitb16-pretrain-lvd1689m` |
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

Download DINOv3 into the path used by the LAM YAML config:

```bash
mkdir -p weights/dinov3-vitb16-pretrain-lvd1689m

hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitb16-pretrain-lvd1689m
```

Download the LaWM/LAM checkpoint and YAML config into the paths recorded by the
provided configs:

```bash
hf download jialei02/lawam_lam \
  --local-dir latent_action_model/logs/dino_large_vae/lam_release
```

The policy server loads Qwen3-VL and LAM from the checkpoint config, then the
LAM YAML loads DINOv3 through `model.vision_model_id`. If your downloaded LAM
YAML still points to a Hugging Face model id or an unavailable absolute path,
set it to:

```yaml
model:
  vision_model_id: weights/dinov3-vitb16-pretrain-lvd1689m
```

### Inference

Inference uses two environments:

- the `lawam` environment above for policy loading and serving;
- a separate simulator environment for LIBERO or RoboTwin.

Run LIBERO first if you only need one smoke test. RoboTwin setup is separate and
usually heavier.

#### LIBERO Inference

**1. Install the LIBERO Simulator**

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

**2. Run LIBERO Benchmark**

Set the policy checkpoint path. Use a released LIBERO checkpoint if available
from [lawam_libero_sft_release](https://huggingface.co/jialei02/lawam_libero_sft_release),
or a checkpoint produced by LIBERO SFT.

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

#### RoboTwin Inference

**1. Install the RoboTwin Simulator**

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

**2. Run RoboTwin Evaluation**

Use the auto evaluation entrypoint for RoboTwin runs. It starts the LaWAM
policy server, launches RoboTwin workers, and writes a resumable run directory.

```bash
cd LaWAM
conda activate lawam

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python

hf download jialei02/lawam_robotwin_sft_release \
  --local-dir results/Checkpoints/robotwin/lawam_robotwin_sft_release

# Single-task smoke test.
ROBOTWIN_TASKS=lift_pot \
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt \
  demo_clean
```

Full RoboTwin benchmark:

```bash
cd LaWAM
conda activate lawam

export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin_env/bin/python

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

### SFT Training

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

#### LIBERO SFT

**1. Download LIBERO SFT Data**

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

**2. Launch LIBERO SFT**

```bash
cd LaWAM
conda activate lawam

bash train_lawam.sh \
  --run_id libero_sft_from_pretrain
```

The output checkpoint is written under:

```text
results/Checkpoints/libero/<timestamp>+<run_id>/
```

#### RoboTwin SFT

**1. Download RoboTwin SFT Data**

The preprocessed RoboTwin SFT dataset is available at:

[robotwin_merged](https://huggingface.co/datasets/jialei02/robotwin_merged)

This dataset uses RoboTwin EEF actions and is derived from the lingbot-va
release, specifically
[robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot/tree/main/lerobot_robotwin_eef_aug_500/beat_block_hammer-aloha-agilex_randomized_500-1000).
Compared with that public source, this release converts the data to LeRobot 3.0
format.

The provided RoboTwin SFT config uses `data_mix: robotwin_merged`, so download
the dataset under `dataset/robotwin_merged`:

```bash
mkdir -p dataset

hf download jialei02/robotwin_merged \
  --repo-type dataset \
  --local-dir dataset/robotwin_merged
```

Expected layout:

```text
dataset/
  robotwin_merged/
    meta/
    data/
    videos/
```

**2. Launch RoboTwin SFT**

Important RoboTwin SFT settings:

- Reproducing the paper results requires a global batch size of 1024. The
  effective global batch size is
  `per_device_batch_size * total_num_gpus * gradient_accumulation_steps`.
  Adjust `datasets.vla_data.per_device_batch_size` in
  `starVLA/config/training/train_robotwin.yaml` for your GPU memory and GPU
  count. If you do not have enough GPUs, increase
  `trainer.gradient_accumulation_steps` to keep the global batch size at 1024.
- For debugging, a 30k-step RoboTwin SFT run is usually enough to reach around
  80% of the reported performance. You can set
  `--trainer.max_train_steps 30000` for a shorter debug run.

```bash
cd LaWAM
conda activate lawam

bash train_lawam.sh \
  starVLA/config/training/train_robotwin.yaml \
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
  starVLA/config/training/train_robotwin.yaml
```

Run the same command on every node and set `NODE_RANK` accordingly.

### Checkpoint Notes

Training checkpoints are regular PyTorch `.pt` files that include the model
state and the merged training config. Evaluation scripts use the checkpoint
config to recover dataset statistics, action normalization, Qwen3-VL source,
and LAM source. When moving checkpoints across machines, make sure these paths
are valid in the new environment.

- LIBERO checkpoints should use `datasets.vla_data.data_mix: libero`.
- RoboTwin EEF checkpoints should use `datasets.vla_data.data_mix:
  robotwin_merged` or another supported RoboTwin EEF mixture.
- `framework.qwenvl.base_vlm` must point to Qwen3-VL-2B-Instruct or a local
  copy of that model.
- `framework.action_model.lam_ckpt_path` and
  `framework.action_model.lam_yaml_path` must point to a matching LAM checkpoint
  and YAML config.

## Evaluation

### Current Metrics

The MI reward extension evaluates the learned reward model along several axes:

| Metric | Module | Description |
|---|---|---|
| Pairwise ranking accuracy | `eval_reward_ranking` | Fraction of preference pairs where the reward model agrees with MI-based ordering |
| Reward margin | `eval_reward_ranking` | Mean difference between chosen and rejected reward scores |
| Score correlation | `eval_reward_ranking` | Pearson correlation between MI delta scores and learned reward margins |
| Success vs. failure separation | `eval_progress_corr` | Mean score gap and AUC between successful and failed trajectories |
| Temporal progress correlation | `eval_progress_corr` | Correlation between learned/MI scores and ground-truth temporal progress |

### Remaining Evaluation Work

The following evaluation categories are planned but not yet implemented:

- Success-versus-near-miss ranking
- Reward calibration against ground-truth task success
- End-to-end downstream policy-learning evaluation (RLPD with learned reward)
- Comparison against pixel-space baselines (LPIPS, DINO/CLIP cosine similarity)
- Comparison against LaWAM latent cosine similarity (without reward-model SFT)
- Comparison against available robot reward models

### RLinf Franka RLPD Integration

The inference model, potential-difference cache, confidence gating, and optional
MI-guided replay sampler are implemented as drop-in RLinf modules. The RLinf
repository itself is not vendored here, so the final registry and worker import
paths must be applied in the local RLinf checkout. Documentation is available at:

- **Franka 真机部署文档:** [docs/rlinf_integration/franka_mi_potential_rlpd.rst](docs/rlinf_integration/franka_mi_potential_rlpd.rst)
- **集成实现报告:** [docs/rlinf_integration/mi_potential_rlpd_implementation_report.md](docs/rlinf_integration/mi_potential_rlpd_implementation_report.md)
- **代码审计:** [docs/rlinf_integration/mi_potential_rlpd_code_audit.md](docs/rlinf_integration/mi_potential_rlpd_code_audit.md)

## Design Principles

- **Modularity.** The `mi_reward/` package is self-contained and does not
  modify the LaWAM policy training loop.
- **No internet dependency at runtime.** Feature extractors use local weights
  and do not download models automatically.
- **Cached latent features.** Features are computed once and stored as `.pt`
  files, decoupling extraction from scoring and training.
- **Configurable feature extractor.** DINOv3 and LaWAM LAM extractors are
  supported through a common `BaseFeatureExtractor` interface.
- **Reproducible pseudo-label generation.** Scoring uses deterministic MI
  estimators with fixed random seeds.
- **Separation of concerns.** Pseudo-label construction and external benchmark
  evaluation are independent pipeline stages.

## Limitations

- A reward model trained only on world-model-generated futures may learn
  biases specific to the generating model.
- MI in latent space is an estimator of task-relevant similarity, not exact
  task semantics.
- Maximizing MI-potential growth does not guarantee physical feasibility of
  the preferred trajectory.
- Generated success references may contain visual or physical artifacts.
- External robot data is still required to test sim-to-real or
  world-to-real generalization of the learned reward.
- MI scores can be affected by encoder choice and the temporal alignment
  between candidate and reference trajectories.
- The current reward head uses simple mean+last pooling; transformer-based
  or temporal-attention architectures may capture richer structure.

## Roadmap

Checked items are implemented and functional.

- [x] LaWAM LAM feature adapter (`lawam_lam_extractor.py`)
- [x] Gaussian correlation MI proxy (`mi_potential.py`)
- [x] Soft-histogram MI estimator (`mi_potential.py`)
- [x] Feature caching (`cached_feature_store.py`)
- [x] Pseudo-preference generation (`build_preferences.py`)
- [x] Pairwise reward-head SFT (`train_reward_sft.py`)
- [x] LIBERO manifest builder (`build_manifest.py`)
- [x] RoboTwin manifest builder (`build_manifest.py`)
- [x] Ranking accuracy evaluation (`eval_reward_ranking.py`)
- [x] Progress correlation evaluation (`eval_progress_corr.py`)
- [ ] Task-conditioned feature preprocessing
- [x] Cosmos-Predict trajectory adapter
- [x] MI Directional Potential Field (unified module, multi-backend)
- [x] Directional reward distillation (rank + potential + direction losses)
- [x] Drop-in downstream RL/RLPD reward adapter and potential shaping state
- [ ] RLinf checkout registry wiring and end-to-end real-robot validation
- [ ] Transformer-based reward head with temporal attention
- [ ] Multi-reference MI aggregation strategies

## Citation

If you use LaWAM in your research, please cite:

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

This codebase is based on StarVLA and retains its MIT license. The MI-directional
reward pretraining extension is developed within the RLinf/LaWAM ecosystem. The
project builds on open-source robotics and VLM components including LeRobot,
Qwen-VL, DINO, LIBERO, and RoboTwin.
