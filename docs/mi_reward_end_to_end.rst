.. _mi_reward_end_to_end:

============================================================
MI Reward: Installation and End-to-End Workflow
============================================================

This is the operational document for the offline data and reward-model
pipeline. The project uses one shared ``.venv`` for every third-party tool:
SAM3, MuJoCo, Cosmos Predict2.5, Cosmos Transfer2.5, Robometer/RBM-EVAL, and
their local model weights. Real-robot collection is outside this repository;
RLinf remains the source of those records.

The fixed order is:

.. code-block:: text

   base visual trajectory
     -> SAM3 masks
     -> rigid task planner
     -> MuJoCo physical completion
     -> Cosmos Predict action-conditioned rollout
     -> Cosmos Transfer scene variation
     -> verified manifest
     -> LaWAM directional reward SFT
     -> RBM-EVAL
     -> RLinf deployment

The data-generation launcher is ``mi_reward/scripts/generate_generalization_data.sh``
with ``mi_reward/configs/generalization_data.yaml``. The reward-training
launcher is ``mi_reward/scripts/run_generalization_reward.sh`` with
``mi_reward/configs/generalization_reward.yaml``.

Status and boundaries
=====================

The following components are implemented in this repository:

* Robometer processed-dataset ingestion into a canonical base-record JSONL;
* SAM3 text-prompt mask materialization;
* rule-based pick/place, push, and peg-insertion Cartesian phase planning;
* headless MuJoCo rendering and physical sidecar export;
* Cosmos Predict2.5 action-conditioned rollout invocation;
* Cosmos Transfer2.5 depth-control scene variation;
* strict artifact validation, MI directional preference construction, and SFT.

The following inputs are intentionally external:

* real Franka/RLinf teleoperation and its hardware logs;
* the MuJoCo XML/mesh/material files for a user's robot and scene;
* task-specific SAM3 prompts and target-instance geometry;
* a compatible RLinf checkout for the final deployment wiring.

Robometer is not a physical simulator. Its rows may contain RGB, task text,
quality labels, and progress labels, but generally not synchronized actions,
joint states, object poses, or camera calibration. The ingestion stage records
that limitation in metadata. MuJoCo must produce physical sidecars before a
candidate is accepted for directional reward training.

Stage 1: install the shared environment
========================================

Run from the repository root:

.. code-block:: bash

   cd /home/ddt/MI-directional-WAM-reward-model-pretrain
   bash requirements/install.sh --all

The command is idempotent. It creates ``.venv`` and stores source trees at:

.. code-block:: text

   .venv/src/sam3
   .venv/src/cosmos-predict2.5
   .venv/src/cosmos-transfer2.5
   .venv/src/robometer

The default pinned source commits are defined near the top of
``requirements/install.sh``. Do not install these packages into a second
environment for this pipeline. The legacy ``--mi-cosmos`` and ``--rlpd``
targets are retained for compatibility but are not required by the shared
workflow. PyTorch 2.7.0/torchvision 0.22.0 are installed from the CUDA 12.8
index by default; set ``PYTORCH_INDEX_URL`` only when the host needs a
different compatible wheel source.

Authenticate to Hugging Face, then download all weights into ``.venv/models``:

.. code-block:: bash

   source .venv/bin/activate
   hf auth login
   bash requirements/install.sh --generalization-data --download-weights

The weight layout is:

.. code-block:: text

   .venv/models/sam3/sam3.pt
   .venv/models/cosmos-predict2.5/robot/action-cond/*_ema_bf16.pt
   .venv/models/cosmos-transfer2.5/general/depth/*_ema_bf16.pt
   .venv/models/dinov3-vitb16-pretrain-lvd1689m/
   .venv/models/lawam_lam/

Download and extract Robometer processed data when it is needed as a base
source or benchmark:

.. code-block:: bash

   bash requirements/install.sh --reward-eval --download-eval-data

The installer clones the pinned Robometer source below ``.venv/src/robometer``
and extracts its processed caches below ``.venv/datasets/robometer``. The
processed archive is large; select a smaller dataset in the YAML if storage is
limited. The default repository configuration uses a policy-ranking dataset as
an example, not as a claim that it matches every task.

Set headless rendering variables in every shell that runs data preparation:

.. code-block:: bash

   source .venv/bin/activate
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl

Verify the installation without loading a model:

.. code-block:: bash

   .venv/bin/python - <<'PY'
   import mujoco, numpy, torch
   print("mujoco", mujoco.__version__)
   print("torch", torch.__version__, "cuda", torch.cuda.is_available())
   PY

SAM3 checkpoints are gated on Hugging Face access. If the checkpoint is not
present, SAM3 fails explicitly instead of silently downloading to a global
cache.

Stage 2: prepare MuJoCo assets and materials
============================================

The repository does not ship a Franka scene or task meshes. For each task,
provide a scene XML and target-instance XMLs. A minimum layout is:

.. code-block:: text

   assets/custom_task/
   ├── scene/scene.xml
   ├── scene/scene_banana.xml
   ├── scene/scene_push_b.xml
   ├── scene/scene_peg_square.xml
   ├── meshes/franka/...
   ├── meshes/table/...
   ├── meshes/apple/...
   ├── meshes/banana/...
   ├── materials/*.mtl
   ├── textures/*.png
   └── camera/intrinsics.json

The XML must expose the names referenced by the task YAML. The current
templates use ``global_cam``, ``ee_target``, ``panda_hand``, ``task_object``,
the task goal body, ``gripper``, and the listed geom names. Rename the XML
fields or edit the YAML, but keep the names synchronized.

Each held-out instance must have a real ``model_path`` in its task YAML. A
``builtin://`` URI is stored for provenance only and is never interpreted as a
mesh downloader. If a held-out model path is absent, the MuJoCo worker stops
with an error instead of claiming instance generalization.

The three templates are:

* ``mi_reward/configs/tasks/pick_place_apple_banana.yaml``;
* ``mi_reward/configs/tasks/push_t_to_b.yaml``;
* ``mi_reward/configs/tasks/peg_insertion_variants.yaml``.

The templates deliberately point at placeholder XML paths. Replace them before
a real run. The task YAML also controls hover height, insertion depth, gripper
values, contact phases, and goal tolerances.

Stage 3: Robometer base-data ingestion
======================================

Edit ``mi_reward/configs/generalization_data.yaml``. Set
``base_data.robometer.datasets`` to names that exist under
``.venv/datasets/robometer``. ``task_rules`` maps free-form Robometer task text
to one of the three physical task families and supplies SAM3 prompts. A rule
must be specific enough to identify the task object and goal.

Run the configuration check first:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml --dry-run

The dry-run does not require Robometer data or model weights. It checks the YAML
and the complete worker hand-off. To perform ingestion and all later stages:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml

Robometer ingestion writes:

.. code-block:: text

   dataset/mi_reward/base/robometer_base_records.jsonl
   dataset/mi_reward/manifests/generalization_rigid_v1_success_refs.jsonl
   dataset/mi_reward/base/robometer_base_records.report.json

Every base row contains ``base_id``, ``source: robometer``, task family,
instruction, frame paths, initial/goal frames, a success-reference ID, the
physical task YAML, and object prompts. It does not invent actions or robot
states. Rows without a mapped successful reference are reported and skipped.

The official processed cache stores ``frames`` as a path to a compressed
``trajectory_*.npz`` file (with a ``frames`` array), not as a Python list of
PNG files. The ingest worker reads that format directly and also accepts a
video path, image-path list, NumPy frame array, or image/video bytes. It
materializes normalized PNGs below
``dataset/mi_reward/base/robometer_frames`` so all later workers consume the
same file-based contract.

Using RLinf data instead
------------------------

When the external RLinf collector is ready, set ``base_data.source: jsonl`` and
set ``base_data.input_records`` to a JSONL that already follows the base-record
contract. The required fields are:

.. code-block:: json

   {
     "base_id": "rlinf/pick_place/episode_0001",
     "source": "rlinf",
     "task": "pick up the apple and place it in the basket",
     "task_family": "pick_place",
     "instruction": "pick up the apple and place it in the basket",
     "frames": ["frames/000000.png", "frames/000001.png"],
     "initial_frame": "frames/000000.png",
     "goal_frame": "frames/000001.png",
     "goal_ref_id": "rlinf/pick_place/success_0001",
     "physical_task_config": "mi_reward/configs/tasks/pick_place_apple_banana.yaml",
     "object_prompts": {"task_object": "apple", "goal": "basket", "robot": "robot arm"},
     "action_path": "actions.npy",
     "robot_state_path": "robot_states.jsonl"
   }

The success-reference JSONL named by ``paths.success_refs`` must contain the
declared ``goal_ref_id`` and its frame sequence. This bridge lets RLinf remain
the owner of real-world capture without adding a second collector here. For an
external JSONL, use absolute paths or paths relative to the repository root
for frames, sidecars, and the physical task YAML; the launcher runs from that
root.

Stage 4: physical and visual data generation
==============================================

The data launcher executes these workers in this exact order:

.. code-block:: text

   SAM3 -> planner -> MuJoCo -> Cosmos Predict -> Cosmos Transfer

SAM3 reads base frames and text prompts, then writes per-object and composite
masks. The planner reads the task XML and emits auditable phases such as
``approach``, ``grasp``, ``lift``, ``transport``, ``place``, ``push``, ``align``,
and ``insert``. MuJoCo executes that plan headlessly and writes synchronized:

* rendered RGB frames;
* ``actions.npy``;
* robot-state JSONL;
* object-state JSONL;
* measured relation JSONL;
* per-frame masks and depth maps;
* simulation provenance.

Cosmos Predict2.5 uses the MuJoCo action sequence as its conditioning signal.
Its RGB output is a visual rollout, not a replacement action log. The worker
keeps the MuJoCo action/state/relation sidecars attached to every generated
candidate. Cosmos Transfer2.5 then varies scene appearance using synchronized
depth and foreground controls while preserving those physical sidecars. The
original SAM3 controls remain attached as provenance; MuJoCo re-renders the
foreground mask at the simulated camera resolution used by Transfer.

The output JSONL is:

.. code-block:: text

   dataset/mi_reward/generalization_rollouts/rigid_v1_records.jsonl

The default Predict worker uses the official 2B robot/action-conditioned
checkpoint and the default 256x320 action-conditioned experiment. Set the
experiment, resolution, chunk size, and GPU counts in the command entries of
``generalization_data.yaml`` if a different official checkpoint is installed.
Cosmos Transfer's full 2B depth model is a multi-GPU workload; a single 4090
can be used for dry-runs and smaller supported checkpoints, but the configured
``--num-gpus`` must match the machine.

Artifact gates
--------------

Before reward training, every accepted candidate must have:

.. code-block:: text

   frames
   action_path
   robot_state_path
   object_state_path
   relation_path
   goal_ref_id
   control_artifacts.mask_root
   control_artifacts.depth_root
   instance_variant
   simulation

The verifier rejects missing files, frame-count mismatches, discontinuous
object poses, invalid contact/collision flags, and unsatisfied terminal goals.
Rejected rows remain auditable in the manifest but cannot reach reward SFT.

Stage 5: directional reward SFT
================================

After data generation succeeds, train with the separate reward configuration:

.. code-block:: bash

   bash mi_reward/scripts/run_generalization_reward.sh \
     --config mi_reward/configs/generalization_reward.yaml

This performs, in order:

1. strict accepted-candidate validation;
2. LaWAM LAM token extraction and local feature caching;
3. monotonic alignment to the successful reference;
4. MI directional-progress preference construction;
5. ranking, potential, and directional-difference reward SFT.

The normal checkpoint is:

.. code-block:: text

   results/mi_reward/generalization_rigid_v1/pytorch_model.pt

``training.mi_backend`` controls the reward-training teacher backend. The
preference builder has its own ``training.preference_mi_mode`` because its
legacy CLI accepts ``gaussian_mi_proxy`` or ``histogram_mi``; these are not the
same string as the ``dame_bspline`` training backend.

Stage 6: RBM-EVAL
=================

RBM-EVAL is a separate benchmark launcher and YAML. Set the checkpoint and
feature extractor in ``eval/configs/rbm_eval.yaml`` and run:

.. code-block:: bash

   bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml

The adapter calls the pinned Robometer samplers and metric compilers for
reward alignment, policy ranking, and quality preference. An RGB/feature-only
``StatePotentialRewardModel`` checkpoint is the direct compatibility path. A
relation-conditioned GeoProgress checkpoint additionally needs its relation
sidecar configured under ``geoprogress.relation_sidecar``.

Stage 7: RLinf deployment
==========================

The final Franka/RLPD integration remains in the RLinf checkout. This repository
contains the reward adapter and potential-shaping state, but it does not vendor
RLinf or perform hardware collection. Follow
``docs/rlinf_integration/franka_mi_potential_rlpd.rst`` for registry wiring,
checkpoint metadata, dummy mode, and the first hardware run.

The learned MI reward is a shaping signal. It must not replace environment
termination, collision checks, force limits, workspace limits, or emergency
stops. Keep the sparse environment reward active during deployment.

Troubleshooting
===============

``Missing .venv/bin/python``
    Run ``bash requirements/install.sh --all`` and activate ``.venv``.

``SAM3 checkpoint not found``
    Authenticate to Hugging Face and rerun the generalization-data installer
    with ``--download-weights``.

``Robometer processed dataset is missing``
    Download/extract the selected dataset with
    ``--reward-eval --download-eval-data`` and check its directory name.

``Held-out instance replacement requires instance_variant.model_path``
    Add the target MuJoCo XML path to the corresponding task YAML. A
    ``builtin://`` URI alone is intentionally insufficient.

``MuJoCo body/geom does not exist``
    Rename the XML entities or update the task YAML; do not disable the check.

``Cosmos output was not created``
    Verify the pinned Cosmos source, local checkpoint path, GPU count, CUDA
    visibility, and the action-conditioned experiment name.

``Refusing generalization reward training``
    Inspect the run report and rejected rows. The training gate is designed to
    prevent visual-only Cosmos videos or incomplete Robometer rows from being
    used as physical directional supervision.

Static checks
=============

The following checks do not require a GPU or downloaded weights:

.. code-block:: bash

   python3 -m compileall -q mi_reward
   bash -n requirements/install.sh \
     mi_reward/scripts/generate_generalization_data.sh \
     mi_reward/scripts/run_generalization_reward.sh
   .venv/bin/python -m mi_reward.data.generalization_pipeline \
     --config mi_reward/configs/generalization_data.yaml \
     --task-suite rigid_v1 --dry-run
