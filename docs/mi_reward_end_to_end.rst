.. _mi_reward_end_to_end:

============================================================
MI Reward: Installation and End-to-End Workflow
============================================================

This is the operational document for the offline data and reward-model
pipeline. Incompatible Python stacks are isolated: ``.venv`` contains
SAM2.1/Cosmos/MuJoCo, ``.venv-reward`` contains DINOv3/LaWAM/reward training,
``.venv-eval`` contains Robometer/RBM-EVAL, and ``.venv-libero`` contains the
standard closed-loop visual RLPD runtime. Large source trees, checkpoints, and datasets
remain shared under ``.venv``. Real-robot collection is currently disabled;
future hardware logs enter through the generic JSONL contract.

The fixed order is:

.. code-block:: text

   balanced simulator-native task seeds
     -> rigid task planner
     -> MuJoCo physical completion
     -> independent per-task-and-instance success-reference selection
     -> verified manifest
     -> privileged visual/action/kinematic/relation/outcome MI teacher
     -> VisualGoalPotential distillation
     -> RBM-EVAL
     -> LIBERO dual-view CNN/RLPD sparse / MI / sparse+MI closed-loop evaluation

The data-generation launcher is ``mi_reward/scripts/generate_generalization_data.sh``
with ``mi_reward/configs/generalization_data.yaml``. The reward-training
launcher is ``mi_reward/scripts/run_generalization_reward.sh`` with
``mi_reward/configs/generalization_reward.yaml``.

Status and boundaries
=====================

The following components are implemented in this repository:

* balanced simulator-native seed generation for three task families;
* optional Robometer processed-dataset ingestion into a canonical base-record JSONL;
* SAM2.1 first-frame point/box prompting and video mask propagation;
* rule-based pick/place, push, and peg-insertion Cartesian phase planning;
* headless MuJoCo rendering and physical sidecar export;
* optional Cosmos Predict2.5 action-conditioned rollout invocation;
* optional Cosmos Transfer2.5 depth-control scene variation;
* LaWAM visual-token and transition-action-latent extraction;
* synchronized robot/object-state kinematic-latent extraction;
* strict artifact validation, privileged MI preference construction, and
  visual-only deployment-model distillation.

The following inputs are intentionally external:

* real Franka teleoperation and its hardware logs;
* the MuJoCo XML/mesh/material files for a user's robot and scene;
* optional task-specific SAM2.1 point/box prompts and target-instance geometry;
* optional site-specific hardware deployment wiring.

Robometer is not a physical simulator, and its selected task text does not
match the three custom MuJoCo tasks. It remains available for evaluation and
explicit ablations, but is not the default generation source. The v3 default
uses MuJoCo for both synchronized physical sidecars and canonical RGB frames.

Stage 1: install isolated runtime environments
==============================================

Run from the repository root:

.. code-block:: bash

   cd /home/zhonghaoyang/MI-directional-WAM-reward-model-pretrain
   bash requirements/install.sh --all \
     --download-weights \
     --download-eval-data \
     --download-sim-assets \
     --use-mirrors

The command is idempotent. It creates ``.venv``, ``.venv-reward``,
``.venv-eval``, and ``.venv-libero``. Shared source trees stay at:

.. code-block:: text

   .venv/src/sam2
   .venv/src/cosmos-predict2.5
   .venv/src/cosmos-transfer2.5  # optional
   .venv/src/robometer
   .venv/src/libero

This is the complete default installation command: it installs all four
Python environments, downloads SAM2.1/Cosmos Predict, DINOv3/LaWAM LAM, the
Robometer datasets selected by the data YAML, and the generated MuJoCo task
assets. It deliberately does not download optional Cosmos Transfer weights.
Re-running the command reuses completed files and resumes interrupted Hugging
Face local-directory downloads.

The default Cosmos source commits and the selected SAM source ref are defined
near the top of ``requirements/install.sh``. ``SAM2_GIT_REF`` can pin SAM2 to
an exact reviewed commit for a production run. Cosmos pins
``transformers==4.51.3`` while DINOv3 requires a newer architecture registry;
the environments must not be merged. The legacy ``--mi-cosmos`` and ``--rlpd``
targets are retained for compatibility. PyTorch 2.7.0/torchvision 0.22.0 are
installed from the CUDA 12.8 index by default; set ``PYTORCH_INDEX_URL`` only
when the host needs a different compatible wheel source.

The segmentation backend is chosen in the install command. The default is the
public ``SAM2.1 base-plus`` checkpoint:

.. code-block:: bash

   bash requirements/install.sh --generalization-data \
     --sam-backend sam2 --sam2-size base-plus

Valid SAM2.1 sizes are ``tiny``, ``small``, ``base-plus``, and ``large``.
The installer writes the confirmed choice to
``.venv/models/segmentation.json``; the data YAML reads this file at runtime,
so the installed and executed backends cannot silently diverge. ``sam3`` is
retained only as an explicit compatibility backend, and ``none`` disables
source-image segmentation.

Authenticate to Hugging Face, then download data-generation weights into the
shared ``.venv/models`` directory:

.. code-block:: bash

   .venv/bin/hf auth login
   bash requirements/install.sh --generalization-data --download-weights

Install the reward runtime and download DINOv3/LaWAM LAM into the same shared
model directory:

.. code-block:: bash

   bash requirements/install.sh --reward-train --download-weights

If the weights already exist, omit ``--download-weights``. Both installers run
post-install smoke tests; the reward test verifies that ``dinov3_vit`` is
recognized locally and that the LaWAM module imports with Lightning.

Cosmos Transfer is optional and disabled in the default data YAML. Download
its checkpoint only for the Transfer augmentation ablation:

.. code-block:: bash

   bash requirements/install.sh --generalization-data \
     --download-transfer-weights

When the official Hugging Face endpoint is unavailable, route both dependency
and model downloads through the configured mirrors:

.. code-block:: bash

   bash requirements/install.sh --generalization-data \
     --download-weights --use-mirrors
   bash requirements/install.sh --reward-train \
     --download-weights --use-mirrors

Hugging Face model downloads reuse the saved token for gated repositories,
disable Xet by default, use one worker, and resume the local-directory cache
after an interrupted transfer. Gated model repositories use the official
Hugging Face endpoint even with ``--use-mirrors`` so token and license checks
are not hidden by an unsynchronized mirror. ``HF_MODEL_ENDPOINT`` can override
the model endpoint independently when required. The default SAM2.1 checkpoint comes
from Meta's public checkpoint host and supports resumed ``curl`` downloads.

The weight layout is:

.. code-block:: text

   .venv/models/segmentation.json
   .venv/models/sam2/sam2.1_hiera_base_plus.pt
   .venv/models/cosmos-predict2.5/robot/action-cond/*_ema_bf16.pt
   .venv/models/cosmos-transfer2.5/general/depth/*_ema_bf16.pt  # optional
   .venv/models/dinov3-vitb16-pretrain-lvd1689m/
   .venv/models/lawam_lam/

Download and extract Robometer processed data when it is needed as a base
source or benchmark:

.. code-block:: bash

   bash requirements/install.sh --reward-eval --download-eval-data

The installer creates ``.venv-eval``, clones the pinned Robometer source below
``.venv/src/robometer``, and extracts its processed caches below
``.venv/datasets/robometer``. By
default, it downloads only the datasets listed under
``base_data.robometer.datasets`` in
``mi_reward/configs/generalization_data.yaml``. Downloads use one worker,
disable Xet by default, and retry interrupted transfers while preserving the
Hugging Face local-directory cache.

Select another dataset explicitly with a repeatable option:

.. code-block:: bash

   bash requirements/install.sh --reward-eval --download-eval-data \
     --eval-dataset DATASET_NAME --use-mirrors

Downloading every processed dataset is intentionally opt-in because the
snapshot and its extracted contents require hundreds of GiB:

.. code-block:: bash

   bash requirements/install.sh --reward-eval --all-eval-data --use-mirrors

The default repository configuration uses a policy-ranking dataset as an
example, not as a claim that it matches every task.

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

After configuring real task XMLs, run the strict preflight. Unlike
``--dry-run``, this requires datasets, checkpoints, assets, and visible CUDA
devices. It also verifies both ``.venv/bin/python`` and
``.venv-reward/bin/python``:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml --preflight

SAM2.1 checkpoints are public and require no Hugging Face gate. If SAM3 is
explicitly selected, its existing Hugging Face access requirements still
apply.

Stage 2: download and generate MuJoCo assets
================================================

Download the open Franka Panda model from MuJoCo Menagerie and generate the
small task scenes locally:

.. code-block:: bash

   bash requirements/download_mujoco_assets.sh --use-mirrors

The same operation can be included in the data-environment installation:

.. code-block:: bash

   bash requirements/install.sh --generalization-data \
     --download-sim-assets --use-mirrors

Only the Menagerie ``franka_emika_panda`` subtree is fetched. The apple,
banana, Push-T/Push-B, round/square peg, basket, outlines, and insertion
fixture are generated from native MuJoCo primitives. The generated layout is:

.. code-block:: text

   assets/custom_task/
   ├── README.md
   └── scene/
       ├── asset_manifest.json
       ├── panda_menagerie_adapted.xml
       ├── scene.xml
       ├── scene_pick_apple.xml
       ├── scene_banana.xml
       ├── scene_push_t.xml
       ├── scene_push_b.xml
       ├── scene_peg_round.xml
       ├── scene_peg_square.xml
       ├── scene_<object>_table_light_1.xml
       └── scene_<object>_table_dark_2.xml

The unsuffixed files are compatibility aliases. The planner uses the suffixed
files and takes the Cartesian product of base trajectory, instance variant,
scene variant, and candidate profile. With the default two instances, two
scenes, and eight profiles, one base trajectory produces 32 physical
candidates.
``table_light_1`` and ``table_dark_2`` change illumination, table material,
wall, floor, and skybox while preserving robot/object geometry and camera.
This native MuJoCo scene-level loop does not require Cosmos Transfer.

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

The templates point at these generated train/held-out XMLs. The Panda model
remains active for kinematics and collision checks, while its full-arm geoms
are hidden from the fixed tabletop RGB camera. A stable mocap Cartesian proxy
gripper remains visible and owns task contacts; this avoids over-constraining
Menagerie's high-gain joint servos. For deterministic offline candidate generation, the task YAML enables
an explicit ``kinematic_task_proxy``: after a verified close/contact phase the
free task object follows the Cartesian proxy until release. This proxy is
recorded in candidate metadata and must not be described as learned robot
dynamics. The task YAML also controls hover height, insertion depth, gripper
values, contact phases, and goal tolerances.

Stage 3: simulator-native seeds (default)
==========================================

The default ``base_data.source`` is ``simulator_native``. It creates 5 training
seed records plus one dedicated reference seed for each of ``pick_place``,
``push_shape``, and ``peg_insertion``. Seed
records intentionally have no RGB frames: the planner expands every seed over
two instances, two native scenes, and eight controlled profiles, and MuJoCo creates the
synchronized frames and physical sidecars.

After simulation, the ``reference`` worker selects one canonical successful
rollout per task family and instance. Those six rollouts become instance-matched
``SuccessReference`` records, and all 96 descendants of the three reference
parent seeds are removed from the candidate set. This prevents
candidate/reference leakage. With the checked-in defaults, 576 physical
rollouts become 480 candidates plus six references. The manifest writes 120
items into each of ``train``, ``instance_heldout``, ``scene_heldout`` and
``joint_heldout``; preference training reads only ``train``.

Run the configuration checks first:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml --dry-run
   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml --preflight

Optional Robometer base-data ingestion
---------------------------------------

Robometer is not used by the default v3 generation run. For an explicit
ablation, edit ``mi_reward/configs/generalization_data.yaml``, set
``base_data.source: robometer``, and set
``base_data.robometer.datasets`` to names that exist under
``.venv/datasets/robometer``. ``task_rules`` maps free-form Robometer task text
to one of the three physical task families and supplies semantic object names. A rule
must be specific enough to identify the task object and goal.

Run the configuration check first:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml --dry-run

The dry-run does not require Robometer data or model weights. To perform this
optional ingestion and its configured later stages:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml

Robometer ingestion writes:

.. code-block:: text

   logs/mi_reward/generalization_rigid_v3/data/base/robometer_base_records.jsonl
   logs/mi_reward/generalization_rigid_v3/data/manifests/generalization_rigid_v3_success_refs.jsonl
   logs/mi_reward/generalization_rigid_v3/data/base/robometer_base_records.report.json

Every base row contains ``base_id``, ``source: robometer``, task family,
instruction, frame paths, initial/goal frames, a success-reference ID, the
physical task YAML, and object prompts. It does not invent actions or robot
states. Rows without a mapped successful reference are reported and skipped.

The official processed cache stores ``frames`` as a path to a compressed
``trajectory_*.npz`` file (with a ``frames`` array), not as a Python list of
PNG files. The ingest worker reads that format directly and also accepts a
video path, image-path list, NumPy frame array, or image/video bytes. It
materializes normalized PNGs below
``logs/mi_reward/generalization_rigid_v3/data/base/robometer_frames`` so all later workers consume the
same file-based contract.

Using external real-robot data instead
--------------------------------------

When an external robot collector is ready, set ``base_data.source: jsonl`` and
set ``base_data.input_records`` to a JSONL that already follows the base-record
contract. The required fields are:

.. code-block:: json

   {
     "base_id": "real/pick_place/episode_0001",
     "source": "real_robot",
     "task": "pick up the apple and place it in the basket",
     "task_family": "pick_place",
     "instruction": "pick up the apple and place it in the basket",
     "frames": ["frames/000000.png", "frames/000001.png"],
     "initial_frame": "frames/000000.png",
     "goal_frame": "frames/000001.png",
     "goal_ref_id": "real/pick_place/success_0001",
     "physical_task_config": "mi_reward/configs/tasks/pick_place_apple_banana.yaml",
     "object_prompts": {"task_object": "apple", "goal": "basket", "robot": "robot arm"},
     "segmentation_prompts": {
       "task_object": {"box": [120, 80, 310, 360], "frame_index": 0}
     },
     "action_path": "actions.npy",
     "robot_state_path": "robot_states.jsonl"
   }

The success-reference JSONL named by ``paths.success_refs`` must contain the
declared ``goal_ref_id`` and its frame sequence. This bridge keeps real-world
capture independent from reward training. For an
external JSONL, use absolute paths or paths relative to the repository root
for frames, sidecars, and the physical task YAML; the launcher runs from that
root.

Stage 4: physical data generation and reference closure
=========================================================

The data launcher executes these enabled workers in this order:

.. code-block:: text

   planner -> MuJoCo -> reference

SAM2.1 and Cosmos are disabled in the default v3 configuration. When using an
external image-backed ablation, SAM2.1 can read a first-frame point or box from
``segmentation_prompts`` and propagates each object mask through the trajectory.
SAM2.1 cannot infer a mask from ``object_prompts`` text. For Robometer records
that contain only text, the configured ``--allow-missing-prompts`` passes the
record through; this is safe in this pipeline because MuJoCo subsequently
renders the masks and depth maps consumed by Cosmos Transfer. The planner reads
the task XML and emits auditable phases such as
``approach``, ``grasp``, ``lift``, ``transport``, ``place``, ``push``, ``align``,
and ``insert``. MuJoCo executes that plan headlessly and writes synchronized:

* rendered RGB frames;
* ``actions.npy``;
* robot-state JSONL;
* object-state JSONL;
* measured relation JSONL;
* per-frame masks and depth maps;
* simulation provenance.

When explicitly enabled as an ablation, Cosmos Predict2.5 uses the MuJoCo action sequence as its conditioning signal.
For ``T`` synchronized states the worker supplies exactly ``T-1`` transitions.
XYZ and relative rotation are represented in the preceding end-effector frame
and scaled with the official Bridge factor. Its RGB output is a visual proposal,
not a replacement action log. The worker keeps the MuJoCo action/state/relation
sidecars attached to every generated candidate. When enabled, Cosmos Transfer2.5 then varies scene appearance using synchronized
depth and foreground controls while preserving those physical sidecars. The
original source-image segmentation controls remain attached as provenance;
MuJoCo re-renders the foreground mask at the simulated camera resolution used
by Transfer.

Every candidate carries ``scene_variant.variant_id`` through planner, MuJoCo,
Predict, ingestion, and MI scoring. Strict ingestion rejects a missing scene
variant, and ``data_report.json`` reports total/accepted/rejected candidates
per scene. Native scene-level generation is therefore in the default loop;
Transfer remains an optional additional appearance ablation.

MuJoCo is both the physical and visual source of truth in the default run. If
Predict is explicitly enabled, its proposals are checked
against its synchronized rendering using first/mean/final appearance, temporal
motion magnitude, endpoint-change direction, and moving-pixel support overlap.
With the default ``--inconsistent-policy use-physical``, a failed Cosmos video
is retained below the run log for audit while the candidate's canonical
``frames`` fall back to MuJoCo. Therefore a static object, hallucinated arm, or
motion in the wrong image region cannot enter reward SFT as a valid visual
trajectory. Set the policy to ``reject`` only when rejected Cosmos candidates
should remain rejected rather than use the physical rendering.

The generated task scenes use a fixed third-person camera that covers the full
table. The full Panda remains active for simulation but its visual/collision
geom groups are hidden from RGB/depth/mask rendering; only the moving proxy
gripper and task geometry are shown. This avoids both table occlusion and a
duplicate static-arm/proxy-gripper image.

With the default simulator-native configuration, the output JSONL is:

.. code-block:: text

   logs/mi_reward/generalization_rigid_v3/data/generalization_rollouts/physical_candidates.jsonl

To enable Transfer, set its stage ``enabled: true`` and change
``paths.candidate_records`` in both generalization YAML files to
``logs/mi_reward/generalization_rigid_v3/data/generalization_rollouts/rigid_v3_records.jsonl``.

All artifacts of the default run live below
``logs/mi_reward/generalization_rigid_v3``: intermediate records and frames,
Cosmos diagnostic videos, feature caches, reports, reward checkpoints, and
RBM-EVAL output. Installed sources/checkpoints/datasets remain under ``.venv``;
reusable generated MuJoCo scenes remain under ``assets/custom_task``.

The optional Predict worker uses the official 2B robot/action-conditioned
checkpoint and the default 256x320 action-conditioned experiment. Set the
experiment, resolution, chunk size, and GPU counts in the command entries of
``generalization_data.yaml`` if a different official checkpoint is installed.
``execution.distributed`` performs sample sharding: every node loads a complete
model and handles a different JSONL shard. It does not combine two remote GPUs
into one memory pool. With ``transport: rsync``, nodes may use different
repository and log paths: the controller pushes referenced shard artifacts,
pulls generated output, and rewrites paths back to the controller root. Shared
mounts remain available through ``transport: shared``. For Transfer, the
configured per-worker ``--num-gpus`` must be available on each node.
Single-node execution remains the default. Add one option to the same command
to shard planner, simulator, and any enabled Predict/Transfer stages:

.. code-block:: bash

   bash mi_reward/scripts/generate_generalization_data.sh \
     --config mi_reward/configs/generalization_data.yaml \
     --distributed

Use ``--dry-run --distributed`` first, then ``--preflight --distributed`` after
the remote environment is installed; preflight checks non-interactive SSH and
each node's configured ``python``. Run ``.venv/bin/python
scripts/dist_pipeline_smoke.py`` for a small two-node planner + MuJoCo transfer
test before full generation. Reference selection and final manifest
validation stay on the primary node. The generic real-robot JSONL bridge remains
available but ``real_world.enabled`` is false for the current benchmark run.

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
object poses, invalid schemas and inconsistent visual artifacts. Measured
collision, contact and terminal-goal results belong to ``task_outcome`` instead:
artifact-valid failures are retained as near-miss supervision. Artifact-rejected
rows remain auditable in the manifest but cannot reach reward SFT.

Stage 5: directional reward SFT
================================

After data generation succeeds, train with the separate reward configuration:

.. code-block:: bash

   bash mi_reward/scripts/run_generalization_reward.sh \
     --config mi_reward/configs/generalization_reward.yaml

This performs, in order:

1. strict accepted-candidate validation;
2. LaWAM visual-token extraction and local feature caching;
3. LaWAM ``get_latent_action`` transition-latent extraction;
4. synchronized robot/object-state kinematic-latent extraction;
5. bounded visual/action/kinematic monotonic-stage potentials;
6. bounded phase-aware measured-relation potential and measured-outcome anchors;
7. fine-grained preferences within one initial-state/instance/scene context;
8. optimization on ``train`` and checkpoint selection on instance and scene
   held-out splits; ``joint_heldout`` remains final-test only;
9. ranking, potential, and directional-difference distillation into
   ``VisualGoalPotential``.

The preference builder writes ``rigid_v3.teacher_targets.pt`` next to the
preference JSONL. These are the exact ``[0,1]`` curves used to compute the
ranking score, and the trainer consumes the same file. Successful curves are
monotonic and terminally anchored at one. This prevents preference scoring and
potential regression from silently using different teacher definitions.

The normal checkpoint is:

.. code-block:: text

   logs/mi_reward/generalization_rigid_v3/results/reward_model/pytorch_model.pt

Held-out metrics and the selected epoch are written to:

.. code-block:: text

   logs/mi_reward/generalization_rigid_v3/results/reward_model/validation_metrics.json

Planner candidates within one parent/instance/scene context must share one
initial object offset. Regenerate generalization data created before this
contract was introduced; do not mix an old manifest with new preferences.

Feature extraction is cache-aware. A rerun skips every existing pooled visual,
visual-token, action-latent, and kinematic-latent cache and continues with missing trajectories.
Do not delete ``logs/mi_reward/generalization_rigid_v3/data/features`` after a network,
terminal, or unrelated worker interruption. Progress bars are emitted for
candidate features, success-reference features, directional scoring, and
training epochs.

``training.mi_backend`` controls the reward-training teacher backend. The
preference builder has its own ``training.preference_mi_mode`` because its
legacy CLI accepts ``gaussian_mi_proxy`` or ``histogram_mi``; these are not the
same string as the ``dame_bspline`` training backend.
``training.action_weight`` and ``training.relation_weight`` control privileged
teacher components. ``training.kinematic_weight`` controls the explicit physical
state channel and ``training.outcome_weight`` anchors measured success above
hard near misses. They do not add privileged inputs to the saved
student. ``training.goal_dropout`` trains the learned null-goal path used when
a benchmark has no independent goal image.

Stage 6: independent simulator final test
==========================================

After training, evaluate the frozen checkpoint on ``joint_heldout``:

.. code-block:: bash

   bash mi_reward/scripts/eval_generalization_reward.sh \
     --config mi_reward/configs/generalization_reward.yaml

The evaluator uses measured MuJoCo ``task_outcome`` and relation sequences as
ground truth. It reports success/failure AUC, framewise progress correlation,
pair accuracy by task and comparison type, failure-mode breakdowns, and
same-context Top-1/success@k candidate selection. The report is written to:

.. code-block:: text

   logs/mi_reward/generalization_rigid_v3/results/reward_model/joint_heldout_eval.json

The command rejects a checkpoint if its recorded training or validation splits
overlap ``joint_heldout``. ``--allow-split-overlap`` exists only for preliminary
pipeline debugging and leaves ``split_isolation.valid`` false in the report.

The same report contains candidate-selection baselines for random choice,
terminal pixel similarity, visual-token cosine, visual directional MI,
privileged teachers, the deployable student, and the measured oracle. Run the
isolated ten-variant, three-seed training matrix with:

.. code-block:: bash

   bash mi_reward/scripts/run_generalization_ablations.sh \
     --action run --seed 0 --seed 1 --seed 2 --skip-completed

Each run gets separate preferences, teacher targets, checkpoints and reports
below ``logs/mi_reward/generalization_rigid_v3/results/ablations``. The script
also writes ``ablation_runs.csv`` and ``ablation_summary.json``.

The matrix includes three interaction variants: ``visual_relation``,
``visual_action`` and ``visual_kinematic``. They retain the measured outcome
anchor while enabling exactly one privileged process channel, so that
single-channel benefits can be separated from cross-channel interference.

The launcher remains single-host by default. To prepare the configured SSH
worker without starting a run, then shard independent configs across the local
and remote GPUs, use:

.. code-block:: bash

   bash mi_reward/scripts/run_generalization_ablations.sh \
     --action prepare-remote \
     --remote-host zhonghaoyang@172.16.0.3
   bash mi_reward/scripts/run_generalization_ablations.sh \
     --action run --seed 1 --seed 2 --skip-completed \
     --distributed --remote-host zhonghaoyang@172.16.0.3

This is experiment-level parallelism rather than DDP. The first command syncs
the source and rigid-v3 data, rebases absolute manifest paths and validates the
copied feature cache. Remote outputs are copied back before the final summary.
Do not launch it concurrently with a separate single-host matrix targeting the
same ablation/seed directories.

Stage 7: RBM-EVAL
=================

RBM-EVAL is a separate benchmark launcher and YAML. Set the checkpoint and
feature extractor in ``eval/configs/rbm_eval.yaml`` and run:

.. code-block:: bash

   bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml

The adapter calls the pinned Robometer samplers and metric compilers for
reward alignment, policy ranking, and quality preference. The default
``VisualGoalPotential`` checkpoint consumes visual tokens only when
``visual_goal.goal_sidecar`` is null. An independent per-task success-goal
sidecar may be supplied, but the evaluated trajectory endpoint must never be
used as its own goal. Historical relation-conditioned GeoProgress checkpoints
still require ``geoprogress.relation_sidecar``.

Stage 8: LIBERO closed-loop validation
======================================

The recommended closed loop is self-contained and does not require RLinf. It
uses a dual-view ResNet policy, an RLPD 50/50 online-demonstration replay mix, a
10-head Q ensemble, a frozen LaWAM/DINO encoder, and a frozen
``VisualGoalPotential``. DINO/LaWAM plus the potential replace the reference
Qwen VLM reward; they do not replace the trainable policy CNN. The earlier MLP
SAC entry remains only as a low-capacity baseline. Follow
``docs/libero_closed_loop.md`` for installation, preflight,
sparse/MI/sparse+MI comparisons, two-node experiment sharding, and result
aggregation. Real-world controllers remain disabled by default.

The learned MI reward is a shaping signal. It must not replace environment
termination, collision checks, force limits, workspace limits, or emergency
stops. Keep the sparse environment reward active during deployment.

Troubleshooting
===============

``Missing .venv/bin/python``
    Run ``bash requirements/install.sh --generalization-data``. The data
    launcher uses this environment for SAM2.1, MuJoCo, and Cosmos.

``Missing .venv-reward/bin/python``
    Run ``bash requirements/install.sh --reward-train``. Do not install newer
    Transformers into ``.venv`` because that environment follows Cosmos's
    ``transformers==4.51.3`` constraint.

``Missing .venv-eval/bin/python``
    Run ``bash requirements/install.sh --reward-eval``. Robometer source and
    processed data remain shared below ``.venv``.

``Segmentation install config not found``
    Run ``bash requirements/install.sh --generalization-data``. The installer
    creates ``.venv/models/segmentation.json`` even when weights are downloaded
    later.

``SAM 2.1 checkpoint not found``
    Rerun the generalization-data installer with ``--download-weights`` and
    the same ``--sam2-size`` selection.

``GatedRepoError`` or ``403 Client Error`` for ``facebook/sam3``
    This occurs only after explicitly selecting ``--sam-backend sam3``. Network
    retries cannot resolve a rejected gate request. Verify the active
    account with ``hf auth whoami``, obtain access from the repository authors,
    then replace the saved token with ``hf auth login --force``. In regions
    where the official endpoint is unavailable, run the installer with
    ``--use-mirrors`` after the account has been approved.

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

``Preflight failed``
    Fix every path printed in the combined error. The repository intentionally
    ships placeholder MuJoCo paths, so a fresh checkout cannot pass preflight
    until real scene and held-out-instance XML files are configured. Run
    preflight from the workstation terminal rather than a CUDA-isolated CI
    sandbox.

``Action latent cache is missing``
    Run the reward launcher with a valid LaWAM config/checkpoint. The launcher
    passes ``--action-latents`` automatically. Setting ``action_weight: 0`` is
    the explicit visual/relation ablation; the production path never silently
    substitutes visual features for action latents.

``Kinematic latent cache is missing``
    Keep ``robot_state_path`` and ``object_state_path`` on every candidate and
    success reference. The launcher passes ``--kinematic-latents`` whenever
    ``kinematic_weight`` is non-zero.

``output with shape [3,H,W] doesn't match broadcast shape [1,3,H,W]``
    This was caused by passing a CHW image to LaWAM's batch-shaped ImageNet
    normalization helper. The repository now normalizes through an explicit
    one-frame batch view. Update to the current
    ``mi_reward/features/lawam_lam_extractor.py`` and rerun the reward launcher;
    existing valid feature caches are reused automatically.

``Feature extraction stopped halfway``
    Run the same ``run_generalization_reward.sh`` command again. Cache files
    are written per trajectory, so completed trajectories are skipped and only
    missing pooled features, tokens, action latents, or kinematic latents are computed.

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
   .venv-reward/bin/python -m pytest -q
