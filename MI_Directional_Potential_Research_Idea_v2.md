# MI Directional Potential: Robot-Grounded Reward with Scene and Instance Variation

## 1. Fixed Research Idea

### One-sentence summary

Learn a robot-grounded, goal-conditioned directional MI potential from visual and robot-state trajectories, using Cosmos Transfer2.5 for scene-level variation, SAM3 plus a physics simulator for instance-level object variation, and Cosmos Predict2.5 for action-conditioned future visual trajectories.

### Core research question

> Can a reward model remain directionally aligned with executable task progress when the scene background and the task object change, while visual appearance alone is ambiguous about robot pose, contact, and object state?

The primary task is rigid-object pick-and-place. The first concrete case is replacing an apple with a banana and placing it into a basket. Soft-body manipulation is a later stress test, not part of the first paper claim.

## 2. What Is Novel

The contribution is not a new video generator, a generic VLM reward model, or a simulator. The proposed contribution is:

1. A directional task potential in joint visual and robot-grounded state space.
2. A controlled data-generation protocol that separates scene-level variation from instance-level variation.
3. A feasibility-filtered pseudo-preference pipeline for training a deployable reward model.

The central reward is:

\[
\Phi(s,g) = I(z_s; z_g),
\qquad
r_t = \gamma \Phi(s_{t+1},g)-\Phi(s_t,g),
\]

where `s` contains visual observations, robot state, object state, and task relations. The system evaluates progress direction, not only static visual similarity.

## 3. Two Controlled Variation Levels

### 3.1 Scene-level variation

Change the environment while preserving the robot, task object, and task geometry:

- room or workspace background;
- table appearance;
- lighting and visual style;
- distractor objects;
- camera appearance or viewpoint within a calibrated range.

Cosmos Transfer2.5 receives the scene controls, such as depth, segmentation, edge maps, and spatial masks. Its job is visual scene transformation. It is not treated as a source of physical state or executable actions.

### 3.2 Instance-level variation

Change the specific task object while preserving task intent:

- apple to banana;
- object color, size, or shape;
- initial object pose;
- object mass and collision geometry;
- grasp and placement relation.

SAM3 identifies and tracks the object instance in real observations. A physics simulator replaces the object asset, updates its physical parameters, and generates valid object states and action rollouts. Cosmos is not asked to perform exact physical object replacement by text or mask alone.

## 4. Single End-to-End Pipeline

```text
Initial RGB + goal reference + robot state + task specification
                         |
                         v
       SAM3 instance masks and temporal tracking
                         |
                         +--> scene variant -> Cosmos Transfer2.5
                         |
                         +--> object variant -> simulator asset/state swap
                                                  |
                                                  v
                                      planner or LaWAM action chunks
                                                  |
                                                  v
                         Cosmos Predict2.5 robot/action-cond
                         predicts future visual consequences
                                                  |
                                                  v
                state, action, mask, depth, relation, and frame checks
                                                  |
                                                  v
               LaWAM/DINO latent features + MI directional potential
                                                  |
                                                  v
                 chosen/rejected trajectory preference construction
                                                  |
                                                  v
                         reward-model SFT / GeoProgress training
                                                  |
                                                  v
                         real-robot RLPD reward deployment
```

The action-conditioned Predict2.5 model receives an action sequence and predicts its visual future. It is not itself the action planner. LaWAM or a kinematic planner proposes the action chunks; the simulator and feasibility checks determine whether they are executable.

## 5. Pick-and-Place Example

### Inputs

```text
initial_rgb.png
goal_rgb.png
robot_state.json
object_state.json
camera_calibration.json
task_spec.yaml
```

Example task specification:

```yaml
task: pick_and_place
source_object: apple
target_object: banana
goal_container: basket
scene_variant: kitchen_table_v1
num_candidates: 8
```

### Generation

1. SAM3 produces the apple, basket, gripper, and table instance masks.
2. The simulator loads a banana asset and updates pose, size, mass, collision, and grasp geometry.
3. LaWAM or the planner generates multiple candidate action chunks.
4. Predict2.5 predicts a future video for each action-conditioned candidate.
5. Transfer2.5 optionally changes the background while preserving the robot and task regions.
6. The simulator and sidecar data verify pose continuity, contact, collision, workspace limits, and basket placement.
7. The candidate trajectories are encoded in LaWAM latent space and ranked by directional MI progress.

### Candidate record

Each candidate must contain more than RGB frames:

```text
frames/
actions.npy
robot_states.jsonl
object_states.jsonl
masks/
depth/
relations.jsonl
verification.json
candidate_provenance.json
```

Generated pixels are never treated as physical ground truth without the corresponding state and feasibility sidecars.

## 6. Model and System Roles

| Component | Fixed role | Not responsible for |
|---|---|---|
| SAM3 | Instance segmentation and tracking | 6D physics or action planning |
| MuJoCo first, Newton later | Asset replacement, dynamics, collision, state rollout | Photorealistic video generation |
| Cosmos Transfer2.5 | Scene/background and controlled visual transformation | Reliable object pose, contact, or actions |
| LaWAM | Robot-conditioned latent/action policy and latent features | Exact image editing |
| Cosmos Predict2.5 robot/action-cond | Future visual prediction under actions | Direct action generation |
| MI/GeoProgress | Directional progress scoring | Physics simulation |
| Reward head | Cheap deployable reward approximation | Ground-truth task success by itself |

## 7. Training Data and Reward Learning

The data pipeline has four stages:

1. **Rollout generation:** simulator and LaWAM/planner create valid and invalid action candidates under controlled scene and object variants.
2. **Visual prediction:** Predict2.5 produces action-conditioned future visual candidates; Transfer2.5 adds scene-level appearance variants.
3. **Verification and scoring:** state, object relation, contact, and feasibility checks remove invalid candidates. LaWAM/DINO features and the successful reference trajectory produce MI directional scores.
4. **Preference training:** high-progress candidates become chosen examples and low-progress or infeasible candidates become rejected examples. The reward head is trained with pairwise ranking and GeoProgress losses.

The reward model must be evaluated separately from the candidate generator. A candidate that looks plausible but violates object state or contact constraints is a rejected candidate.

## 8. Implementation Plan in the Current Repository

### Existing modules to retain

- `latent_action_model/` and `starVLA/` for LaWAM/LAM.
- `mi_reward/features/` for latent extraction and caching.
- `mi_reward/scoring/` for MI and directional progress.
- `mi_reward/models/` and `mi_reward/training/` for reward training.
- `mi_reward/verification/` for deterministic feasibility checks.

### Modules to add or extend

- `mi_reward/data/instance_schema.py`: object tracks, masks, depth, poses, variants, and sidecar contracts.
- `mi_reward/perception/sam3_tracker.py`: SAM3 segmentation and temporal tracking adapter.
- `mi_reward/sim/mujoco_backend.py`: headless object replacement, rollout, state export, and rendering.
- `mi_reward/data/transfer_generator.py`: Transfer2.5 scene-control adapter.
- `mi_reward/data/predict_action_cond.py`: Predict2.5 action-conditioned adapter.
- `mi_reward/planning/action_proposer.py`: LaWAM/planner candidate action chunks.
- `mi_reward/data/instance_rollout.py`: candidate generation and manifest construction.
- `mi_reward/verification/instance_checks.py`: object identity, pose, contact, collision, and goal checks.

The current generic `scripts/gen_samples.py` and RGB-only Cosmos path are not sufficient for this protocol. They should remain smoke-test paths, not the scientific data path.

## 9. Main Experiments

### Tasks

- Rigid pick-and-place with held-out object instances.
- Held-out backgrounds, lighting, and camera appearance.
- Near-miss grasp, wrong orientation, collision, and failed placement cases.
- One later deformable task in SoftGym/PlasticineLab/Newton as a stress test.

### Required baselines

- pixel or image-distance reward;
- DINO/LaWAM latent cosine reward;
- VIP-style visual reward;
- PROGRESSOR or comparable progress reward;
- visual-only directional MI;
- joint visual plus robot-state directional MI;
- no Transfer variation;
- no Predict2.5 imagined candidates;
- no simulator feasibility filtering.

### Decisive ablation

The central experiment is whether the robot-grounded directional potential separates successful and near-miss trajectories when static visual similarity prefers the wrong trajectory. This must be tested on object identity, object pose, gripper contact, and background shifts.

## 10. Scope Boundaries

- The first paper is about robot-grounded directional reward, not a new Cosmos model.
- Scene-level and instance-level variation are controlled data-generation mechanisms, not separate papers.
- Soft-body simulation is a later stress test, not a first-stage contribution.
- Predict2.5-generated RGB is an auxiliary candidate future, not physical supervision by itself.
- The system must not send Cosmos-generated pixels directly to the robot controller.

## 11. Success Criterion

The project is successful only if, under held-out object and scene variants, the learned reward ranks feasible successful trajectories above visually plausible near-misses, and the distilled reward improves downstream LaWAM/RLPD learning relative to visual-only and latent-similarity baselines.
