.. _mi_reward_end_to_end:

============================================================
MI Reward: End-to-End Workflow
============================================================

This document is the operational guide for the complete project:

.. code-block:: text

   environment -> assets -> future data -> reward SFT -> RBM-EVAL -> Franka RLPD

The workflow has two reward-model branches. ``GeoProgressPotential`` is the
instance-aware, relation-conditioned branch. ``StatePotentialRewardModel`` is
the RGB/feature branch intended for RBM-EVAL and the standard RLinf potential
shaping adapter. A GeoProgress checkpoint must not be silently treated as an
RGB-only checkpoint.

Status at a glance
==================

+-------+-----------------------------+----------------------------------------------+
| Stage | Main entry point             | Status                                       |
+=======+=============================+==============================================+
| 1     | ``requirements/install.sh``  | Environment scripts implemented              |
| 2     | Task YAML and assets         | User assets required                         |
| 3     | ``prepare_instance_data.sh`` | Orchestrator implemented; workers external   |
| 4     | ``train_reward_sft.py``      | Reward SFT implemented                        |
| 5     | ``eval/run_rbm_eval.sh``     | Implemented for compatible checkpoints       |
| 6     | RLinf integration             | Adapter scaffold; manual wiring and hardware |
|       |                              | validation remain                             |
+-------+-----------------------------+----------------------------------------------+

Stage 1: Install environments
==============================

Run all commands from the repository root:

.. code-block:: bash

   cd /home/ddt/MI-directional-WAM-reward-model-pretrain

Create the shared environment used by SAM3, Cosmos, MuJoCo, feature extraction,
and instance data preparation:

.. code-block:: bash

   bash requirements/install.sh --instance-data

Download the local model weights after Hugging Face authentication:

.. code-block:: bash

   hf auth login
   bash requirements/install.sh --instance-data --download-weights

The shared environment stores source trees and weights at:

.. code-block:: text

   .venv/src/sam3
   .venv/src/cosmos-predict2.5
   .venv/models/sam3
   .venv/models/cosmos-predict2.5
   .venv/models/cosmos-transfer2.5
   .venv/models/dinov3-vitb16-pretrain-lvd1689m
   .venv/models/lawam_lam

Install the Robometer/RBM-EVAL source into the same ``.venv``. The installer
uses ``--no-deps`` for Robometer itself so its Torch 2.8 dependency pins do not
replace the existing Cosmos/SAM3 stack:

.. code-block:: bash

   bash requirements/install.sh --reward-eval
   bash requirements/install.sh --reward-eval --download-eval-data

The pinned source and processed data are stored at:

.. code-block:: text

   .venv/src/robometer
   .venv/datasets/robometer

Activate the environment for all following local commands:

.. code-block:: bash

   source .venv/bin/activate
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl

The optional ``--mi-cosmos`` and ``--rlpd`` targets create isolated legacy
environments. They are not required for the shared instance-data/RBM-EVAL path.

Stage 2: Prepare custom assets
==============================

The repository does not ship task-specific object and scene assets. For a
custom instance task, provide the following, or provide an external worker that
generates the equivalent artifacts:

+----------------------------+----------+-----------------------------------------+
| Asset                      | Required | Accepted form                           |
+============================+==========+=========================================+
| Initial global-camera RGB  | Yes      | PNG/JPEG frames or video                |
| Successful reference RGB  | Yes      | PNG/JPEG frames or video                |
| Source object geometry     | Yes      | MuJoCo mesh or XML asset                |
| Target object geometry     | Yes      | MuJoCo mesh or XML asset                |
| Object materials           | Usually  | MTL, texture maps, MuJoCo materials     |
| Scene assets               | Yes      | Franka, table, target, lighting, XML    |
| Camera calibration         | Yes      | Resolution, intrinsics, extrinsics      |
| Task/variant metadata      | Yes      | YAML task configuration                 |
+----------------------------+----------+-----------------------------------------+

A real-robot photograph is optional when MuJoCo can render the initial and
successful reference views. If Cosmos is conditioned on a real image, provide
that image and its camera calibration. Initial and successful views must use
the same camera convention.

A recommended asset layout is:

.. code-block:: text

   assets/custom_task/
   ├── scene/scene.xml
   ├── scene/franka.xml
   ├── scene/table.xml
   ├── scene/basket.xml
   ├── objects/source_object/object.obj
   ├── objects/source_object/object.mtl
   ├── objects/source_object/texture.png
   ├── objects/target_object/object.obj
   ├── objects/target_object/object.mtl
   ├── objects/target_object/texture.png
   ├── camera/intrinsics.json
   ├── camera/extrinsics.json
   ├── observations/initial/000000.png
   ├── observations/success/000000.png
   └── task.yaml

The existing task templates cover ``pick_place``, ``push_shape`` and
``peg_insertion`` under ``mi_reward/configs/tasks/``. The main instance config
is ``mi_reward/configs/instance_geoprogress.yaml``.

Stage 3: Generate and verify future data
=========================================

Configure the following paths and task fields in
``mi_reward/configs/instance_geoprogress.yaml``:

* ``paths.candidate_records``
* ``paths.manifest``
* ``paths.success_refs``
* ``paths.feasibility_config``
* ``task_family`` and ``task_families``
* ``data_preparation.sam3`` and ``data_preparation.cosmos``

The normal worker order is:

.. code-block:: text

   SAM3 -> Cosmos Transfer -> MuJoCo -> planner -> Cosmos Predict

Each external worker receives ``{python}``, ``{request}``, and ``{result}``
placeholders and must write a result JSON containing the next ``records_path``.
The last worker must write exactly ``paths.candidate_records``.

The default configuration contains ``execution.stages: []``. This is only the
pre-generated-candidate ingest mode. It does not launch SAM3, Cosmos, MuJoCo,
or a planner. If no candidate JSONL already exists, fill in the release-specific
worker commands before running the non-dry-run command.

Validate configuration and worker hand-offs first:

.. code-block:: bash

   bash mi_reward/scripts/prepare_instance_data.sh \
     --config mi_reward/configs/instance_geoprogress.yaml \
     --task-suite rigid_v1 \
     --dry-run

Run generation and deterministic verification:

.. code-block:: bash

   bash mi_reward/scripts/prepare_instance_data.sh \
     --config mi_reward/configs/instance_geoprogress.yaml \
     --task-suite rigid_v1

The accepted output must include:

.. code-block:: text

   dataset/mi_reward/manifests/instance_rigid_v1.jsonl
   dataset/mi_reward/manifests/instance_rigid_v1_success_refs.jsonl

For every accepted candidate, strict training validation requires:

.. code-block:: text

   frames
   action_path
   robot_state_path
   object_state_path
   relation_path
   goal_ref_id
   control_artifacts.mask_root
   control_artifacts.depth_root

SAM3 masks, MuJoCo depth maps, actions, robot/object states, measured
relations, provenance, and verification records are normally generated by the
workers. If ``paths.candidate_records`` points to pre-generated JSONL, all of
these files must already exist.

Stage 4: Train reward models
============================

GeoProgress branch
------------------

Use this branch for instance-aware, physically verified supervision:

.. code-block:: bash

   bash mi_reward/scripts/run_instance_geoprogress.sh \
     --config mi_reward/configs/instance_geoprogress.yaml

This runs manifest validation, LAM token extraction, relation-aware MI
preference construction, and GeoProgress SFT. The usual checkpoint is:

.. code-block:: text

   results/mi_reward/instance_rigid_v1/pytorch_model.pt

GeoProgress consumes visual tokens, successful goal tokens, and measured
relations. Its relation sidecars must be retained for later inference.

StatePotential branch
---------------------

Use this branch for RGB/feature-only RBM-EVAL and standard RLinf deployment.
First extract pooled features and build preferences using the commands in the
``Quick Start`` section. Then run:

.. code-block:: bash

   source .venv/bin/activate
   python -m mi_reward.training.train_reward_sft \
     --mode distill \
     --preferences dataset/mi_reward/preferences/train_preferences.jsonl \
     --feature_root dataset/mi_reward/features \
     --output_dir results/mi_reward/state_potential \
     --batch_size 16 \
     --epochs 5 \
     --lr 1e-4

The checkpoint is:

.. code-block:: text

   results/mi_reward/state_potential/pytorch_model.pt

The DINOv3 or LAM feature extractor used here must be the same representation
used by the reward model during training.

Stage 5: Run RBM-EVAL
=====================

RBM-EVAL uses official Robometer samplers and metric compilers for:

* reward alignment;
* policy ranking;
* quality preference.

Edit ``eval/configs/rbm_eval.yaml``:

.. code-block:: yaml

   paths:
     checkpoint: results/mi_reward/state_potential/pytorch_model.pt

   model:
     type: auto
     device: cuda
     precision: fp32

   features:
     extractor: dino_v3
     model_path: .venv/models/dinov3-vitb16-pretrain-lvd1689m
     strict: true

The extractor must match training. Use ``lawam_lam`` and fill the LAM paths if
the checkpoint was trained on LAM features.

Run the benchmark:

.. code-block:: bash

   export ROBOMETER_PROCESSED_DATASETS_PATH="$PWD/.venv/datasets/robometer"
   bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml

Outputs are written to:

.. code-block:: text

   results/rbm_eval/mi_reward/metrics.json
   results/rbm_eval/mi_reward/<eval_type>/*_results.json

The standard RBM-1M data is RGB/video data and does not contain MuJoCo
relations. Therefore a GeoProgress checkpoint cannot be used on RBM-EVAL
unless ``geoprogress.relation_sidecar`` contains measured relations and an
explicit successful goal for every sampled trajectory. Do not substitute zero
relations or an arbitrary final frame.

Stage 6: Deploy with Franka RLPD
================================

The current repository provides drop-in RLinf modules, but it does not vendor
the RLinf repository or modify its registry in place. Real deployment requires
the following manual steps:

1. Use a ``StatePotentialRewardModel`` checkpoint, not a plain GeoProgress
   checkpoint, unless RLinf observations include the required relation and goal
   inputs.
2. Create ``metadata.json`` beside ``pytorch_model.pt``. The template and
   checkpoint export procedure are in
   ``docs/rlinf_integration/franka_mi_potential_rlpd.rst``.
3. Copy ``MIPotentialRewardModel`` into the RLinf reward-worker package.
4. Register ``model_type: mi_potential`` in the RLinf reward registry.
5. Add ``PotentialShapingState`` to the EnvWorker and configure image keys,
   encoder paths, and task description.
6. Copy or adapt
   ``docs/rlinf_integration/config/realworld_peginsertion_rlpd_cnn_async_mi_potential.yaml``.
7. Run RLinf dummy mode and verify model loading, potential deltas, replay
   fields, and SAC losses before connecting hardware.
8. Validate on simulation or LIBERO before a Franka run.
9. Start the first hardware run with ``reward_weight: 0.0``. Only increase the
   shaping weight after the sparse environment reward, camera stream, safety
   limits, and emergency stop have been verified.

The detailed Franka launch commands, cache lifecycle, ablations, metrics, and
safety rules are in ``franka_mi_potential_rlpd.rst``. MI shaping must never
control termination, success detection, collision handling, force limits,
workspace limits, or emergency stops.

Artifact gates
==============

Do not start a stage until the preceding artifact exists:

.. code-block:: text

   Stage 1: .venv/bin/python
   Stage 2: assets + task YAML + initial/success RGB
   Stage 3: manifest + success_refs + accepted candidates > 0
   Stage 4: pytorch_model.pt + train_config.yaml
   Stage 5: metrics.json
   Stage 6: RLinf dummy-mode validation, then hardware review

The data-generation workers and the final RLinf registry/hardware integration
are deployment-specific. The repository therefore documents their contracts
but does not claim that the full six-stage process is currently one-command
automatic.
