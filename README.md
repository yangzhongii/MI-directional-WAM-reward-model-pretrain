# LaWAM MI-Directional Reward Model

This repository adds a directional-progress reward model to LaWAM. The model
uses LaWAM visual/action latents, measured task relations, and verified future
rollouts. It does not replace LaWAM or RLinf.

## Fixed pipeline

```text
base trajectory (Robometer, or an external RLinf JSONL bridge)
  -> SAM3 masks
  -> rule-based task planner
  -> headless MuJoCo instance/physics completion
  -> Cosmos Predict2.5 robot/action-conditioned visual rollout
  -> Cosmos Transfer2.5 scene variation
  -> manifest validation
  -> LaWAM features + MI directional preferences
  -> reward-model SFT
  -> RBM-EVAL
  -> RLinf real-robot deployment
```

Real-robot trajectory collection is deliberately not implemented here; RLinf
already owns that part. The external RLinf export must be converted to the
`base_records.jsonl` contract described in the detailed guide.

Robometer is a visual base-data source. It normally does not contain the
action, robot-state, camera-calibration, or object-pose sidecars required for
physical verification. The pipeline therefore never treats Robometer pixels as
physical truth: MuJoCo creates those sidecars before Cosmos generation.

## Quick start

From the repository root:

```bash
bash requirements/install.sh --all --download-weights --download-eval-data
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Prepare verified instance and scene-generalized data.
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --dry-run
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml

# Train the directional-progress reward model.
bash mi_reward/scripts/run_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml

# Evaluate an RGB/feature-compatible checkpoint with RBM-EVAL.
bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml
```

The default data configuration points at a Robometer processed dataset and
placeholder MuJoCo scenes under `assets/custom_task/`. Replace those paths
with the task assets available on the machine. A dry-run checks configuration
and worker hand-offs without loading a model.

## Tasks

The templates cover:

- `pick_place`: apple to banana instance replacement;
- `push_shape`: PushT to PushB;
- `peg_insertion`: round peg to square peg/network-style insertion.

Each task requires a real MuJoCo XML scene and target-instance XML/mesh. The
`builtin://` asset URI is provenance only; it is not a downloadable mesh.

## Documentation

- [Detailed installation and end-to-end workflow](docs/mi_reward_end_to_end.rst)
- [RLinf reward deployment integration](docs/rlinf_integration/franka_mi_potential_rlpd.rst)

The detailed RST is the source of truth for dependency versions, local weight
locations, Robometer preprocessing, asset layout, YAML fields, artifact gates,
GPU expectations, and troubleshooting.

## Repository layout

```text
mi_reward/data/       base schemas, Robometer ingest, Cosmos workers, manifests
mi_reward/perception SAM3 worker contract
mi_reward/planning/   rigid task planner
mi_reward/sim/        MuJoCo backend and physical sidecar writer
mi_reward/scoring/    MI and directional-progress teachers
mi_reward/training/   reward-model SFT/distillation
mi_reward/scripts/    one-command data and training launchers
eval/                 RBM-EVAL launcher and YAML
requirements/         shared .venv installer
```
