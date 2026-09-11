# LaWAM MI-Directional Reward Model

This repository adds a directional-progress reward model to LaWAM. The model
uses LaWAM visual/action latents, measured task relations, and verified future
rollouts. It does not replace LaWAM or the downstream control policy.

## Offline benchmark pipeline

```text
balanced simulator-native task seeds
  -> rule-based task planner
  -> headless MuJoCo instance/physics completion
  -> one independent MuJoCo success reference per task and instance
  -> manifest validation
  -> LaWAM visual tokens + inferred actions + physical kinematics + measured relations
  -> privileged MI directional teacher
  -> RGB/optional-goal VisualGoalPotential distillation
  -> RBM-EVAL
  -> LIBERO sparse-vs-MI-vs-sparse+MI visual RLPD closed loop
```

Real-robot trajectory collection is deliberately disabled in the default
experiment. The standalone controllers and `base_records.jsonl` import
contract remain available for later hardware work.

Robometer is retained for benchmark evaluation and optional ablations. It is
not the default generation source because its task text and visual goals do not
match the three custom MuJoCo tasks. Cosmos Predict and Transfer are also
optional ablations; the current 2B generated videos do not pass the strict
physical/visual consistency audit and are disabled by default.

All artifacts produced by the default run share one root:
`logs/mi_reward/generalization_rigid_v3/`. This includes intermediate JSONL,
MuJoCo frames, feature caches, reports, checkpoints,
and RBM-EVAL results. Downloaded sources/models/data remain under `.venv`, and
generated reusable MuJoCo scenes remain under `assets/custom_task`.

MuJoCo is the physical and visual source of truth for the default run. One
canonical successful rollout per task family and object instance is removed
from the candidate set and becomes that instance's success reference. Candidate
and reference task/instance identities must match exactly, which prevents
cross-task or cross-instance supervision and reference leakage.

## Quick start

From the repository root:

```bash
bash requirements/install.sh --all \
  --download-weights \
  --download-eval-data \
  --download-sim-assets \
  --use-mirrors
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Prepare verified instance and scene-generalized data.
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --dry-run
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --preflight
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml

# Optional two-node sample sharding (same pipeline, one additional parameter).
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --preflight --distributed
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --distributed

# Fast planner + MuJoCo transport check before a full distributed run.
.venv/bin/python scripts/dist_pipeline_smoke.py

# Train the directional-progress reward model.
bash mi_reward/scripts/run_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml

# Run the independent MuJoCo final test and same-context Top-1 selection.
bash mi_reward/scripts/eval_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml

# Generate or run the isolated multi-seed ablation matrix.
bash mi_reward/scripts/run_generalization_ablations.sh --action generate --seed 0
bash mi_reward/scripts/run_generalization_ablations.sh --action prepare-remote
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --seed 1 --seed 2 --skip-completed --distributed

# After the original seven variants, run the three channel-interaction variants.
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run \
  --ablation visual_relation --ablation visual_action --ablation visual_kinematic \
  --seed 0 --seed 1 --seed 2 --skip-completed \
  --distributed --no-sync-remote

# Evaluate an RGB/feature-compatible checkpoint with RBM-EVAL.
bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml

# Install and preflight the recommended dual-view CNN/RLPD closed loop.
bash requirements/install.sh --libero-closed-loop \
  --download-libero-data --libero-suite libero_spatial --use-mirrors
bash mi_reward/scripts/run_libero_rlpd.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action preflight --task-id 0 --reward-mode sparse_mi --seed 0

# Run visual RLPD with 50/50 online/demo replay on two machines.
bash mi_reward/scripts/run_libero_rlpd_matrix.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action run --skip-completed --distributed
```

The reward launcher scores all configured generalization splits, forms
preferences only between candidates sharing the same initial state, trains on
`train`, and selects checkpoints on the instance/scene held-out splits.
`joint_heldout` is read only by the independent final-test command; that command
rejects checkpoints whose metadata shows train/validation overlap with the test split.
Its exact bounded teacher curves are saved beside the preference JSONL, so
preference construction and potential distillation use one supervision target.

The installer keeps incompatible Python stacks isolated: `.venv` runs
SAM2.1/Cosmos/MuJoCo, `.venv-reward` runs DINOv3/LaWAM/reward training, and
`.venv-eval` runs RBM-EVAL; `.venv-libero` runs visual RLPD (with the old SAC kept as a baseline). Large sources, weights, and datasets remain shared
under `.venv/{src,models,datasets}`. The launch scripts select the correct
environment automatically.

The command above installs all four environments, optional model/evaluation
artifacts, and the MuJoCo scenes. The default data-generation stages themselves
need only MuJoCo; Cosmos and Robometer downloads are not consumed by v3.
Re-running the installer is safe: completed artifacts are
reused and interrupted Hugging Face downloads resume from their local cache.

With `--all`, `--download-weights` installs all weights required by the default
Predict-only pipeline. On an individual target it is stage-specific:
`--generalization-data` downloads SAM2.1/Cosmos Predict, while
`--reward-train` downloads DINOv3/LaWAM LAM. Cosmos Transfer is disabled by
default and downloaded only with `--download-transfer-weights`.

The installer selects `SAM2.1 base-plus` by default and records the runtime
choice in `.venv/models/segmentation.json`. Select another public SAM2.1
size with `--sam2-size tiny|small|base-plus|large`. SAM3 remains an explicit
compatibility option via `--sam-backend sam3` and is no longer downloaded by
default.

`--download-eval-data` downloads only the Robometer datasets listed in
`mi_reward/configs/generalization_data.yaml`. Use repeated
`--eval-dataset <name>` options to select other datasets. The complete
Robometer snapshot is intentionally opt-in via `--all-eval-data` because it
requires hundreds of GiB for both download and extraction.

The rigid-task scenes can be prepared automatically from Google DeepMind's
open MuJoCo Menagerie Panda model. The installer downloads only the Panda
subtree and generates lightweight task geometry locally:

```bash
bash requirements/install.sh --generalization-data --download-sim-assets --use-mirrors
```

The same asset-only step can be rerun without reinstalling Python packages via
`bash requirements/download_mujoco_assets.sh --use-mirrors`. `--dry-run`
checks configuration and worker hand-offs; `--preflight` additionally requires
real datasets, weights, generated task XMLs, and sufficient local GPUs.

LaWAM infers transition latents from synchronized MuJoCo frames through
`get_latent_action()`. A separate deterministic encoder consumes synchronized
robot/object states to produce the physical kinematic latent. Visual MI,
inferred-action MI, kinematic MI, measured-relation progress, and measured task
outcomes form the offline privileged teacher. The saved `VisualGoalPotential`
checkpoint consumes none of these sidecars at deployment; it uses visual tokens
and either an independent goal image or its learned null-goal token.

## Tasks

The templates cover:

- `pick_place`: apple to banana instance replacement;
- `push_shape`: PushT to PushB;
- `peg_insertion`: round peg to square peg/network-style insertion.

Each task uses an independent train and held-out scene generated under
`assets/custom_task/scene`. The `builtin://` asset URI remains provenance; the
actual geometry comes from the generated MJCF.

## Documentation

- [Detailed installation and end-to-end workflow](docs/mi_reward_end_to_end.rst)
- [Privileged MI teacher and visual student design](docs/privileged_mi_teacher_design.md)
- [Chinese offline benchmark runbook](docs/offline_benchmark_runbook.md)
- [Research gaps and validation roadmap](docs/research_gaps_and_validation_roadmap.md)
- [LIBERO visual RLPD + frozen MI closed-loop validation](docs/libero_closed_loop.md)

The detailed RST is the source of truth for dependency versions, local weight
locations, Robometer preprocessing, asset layout, YAML fields, artifact gates,
GPU expectations, and troubleshooting.

## Repository layout

```text
mi_reward/data/       base schemas, Robometer ingest, Cosmos workers, manifests
mi_reward/perception SAM2.1/SAM3 segmentation worker contract
mi_reward/planning/   rigid task planner
mi_reward/sim/        MuJoCo backend and physical sidecar writer
mi_reward/scoring/    MI and directional-progress teachers
mi_reward/training/   reward-model SFT/distillation
mi_reward/scripts/    one-command data and training launchers
eval/                 RBM-EVAL launcher and YAML
requirements/         isolated runtime installer with shared artifact storage
```
