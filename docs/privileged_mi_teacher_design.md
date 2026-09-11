# Privileged MI Teacher and Visual Student

## Purpose

The benchmark path separates information available during offline data
construction from information required by the deployed reward model.

```text
verified rollout
  ├─ RGB frames ───────────────> LaWAM visual tokens
  ├─ frame transitions ────────> LaWAM quantized action latents
  └─ MuJoCo object/robot state -> measured relation sequence
                                      │
                                      v
                         privileged directional MI teacher
                                      │ distillation
                                      v
                      VisualGoalPotential(RGB, optional goal)
```

Cosmos Predict uses the numerical MuJoCo action sequence as generation
conditioning. That sidecar is not the action latent. The action latent is
inferred independently from the rendered transition through
`LatentLAMModel.get_latent_action()` and is used to judge whether the visual
motion resembles motion in a verified successful reference.

## Cached representations

For a trajectory with `T` frames, the feature job writes:

```text
<trajectory>.pt                         pooled visual features [T,D]
<trajectory>_tokens.pt                  visual patch tokens [T,K,Dv]
<trajectory>_action_latents.pt          action latents [T-1,Q,Da]
<trajectory>__feature_meta.json         extractor and tensor shapes
```

Action extraction uses the temporal window configured by the installed LAM.
Each latent at index `t` describes the motion ending at visual frame `t+1`.
Early windows are left-padded with the first frame.

## Teacher score

The visual, action and kinematic trajectories are independently aligned to the
declared successful reference with monotonic MI alignment. Their alignment
paths are converted to normalized stage progress in `[0,1]`. Measured
relations use fixed physical scales and a task phase latch, also in `[0,1]`, so
a tiny initial orientation error cannot dominate the teacher.

The fused process target is a weighted average:

```text
Phi_teacher(t) = weighted_mean(
    Phi_visual_stage(t),
    w_action Phi_action_stage(t),
    w_kinematic Phi_kinematic_stage(t),
    w_relation Phi_relation(t)
)
```

For measured successful trajectories, `Phi_teacher` is made monotonic and its
terminal value is anchored at `1`. The scalar directional score is computed
from this exact curve and then receives the measured-outcome anchor:

```text
S_teacher = directional_score(Phi_teacher) + outcome_anchor
```

The preference builder persists the same per-frame curves in
`rigid_v3.teacher_targets.pt`; training reads that file instead of recomputing
a second teacher. Because action latents describe transitions, their first
value is attached to both the initial frame and the first transition endpoint.
This avoids creating an artificial zero-to-positive jump at the start of every
trajectory.

The weights are configured in
`mi_reward/configs/generalization_reward.yaml`:

```yaml
training:
  action_weight: 1.0
  kinematic_weight: 1.0
  relation_weight: 1.0
  pair_scope: same_context
  train_splits: [train]
  validation_splits: [instance_heldout, scene_heldout]
```

Candidate profiles are compared only when parent initial state, instance,
scene and split match. Success-vs-failure pairs are anchored by measured
outcomes; within-success and within-failure pairs retain fine-grained process
ordering. Instance/scene held-out pairs select checkpoints, while
``joint_heldout`` is reserved for the independent final-test evaluator.

## Student and deployment reward

`VisualGoalPotential` consumes visual patch tokens and, when available, visual
goal tokens. During training, `goal_dropout` replaces a fraction of goals with
a learned null-goal token. The same checkpoint can therefore be evaluated on
RBM-EVAL without action/relation sidecars and without manufacturing a future
goal frame from the evaluated trajectory.

The deployment shaping reward is:

```text
r_t = gamma V_theta(o_{t+1}, g) - V_theta(o_t, g)
```

This is potential-based reward distillation, not PPO. PPO/RLPD is a downstream
policy optimizer and is not used to rank offline candidate trajectories.

## Losses

The student is trained with three existing losses:

```text
L = lambda_rank L_rank
  + lambda_potential L_potential
  + lambda_direction L_direction
```

- `L_rank` preserves teacher preference ordering.
- `L_potential` regresses the privileged per-frame potential.
- `L_direction` preserves the sign and magnitude of temporal progress.

## Compatibility

- `--mode generalization` trains `VisualGoalPotential` and is the default
  benchmark path.
- `--mode geoprogress` keeps the historical relation-dependent
  `GeoProgressPotential` for old checkpoints and sidecar experiments.
- `StatePotentialRewardModel` and `TrajectoryRewardHead` remain supported by
  the RBM adapter.

## Required ablations

Use the same data split and seeds for `full`, `visual_only`, `no_action`,
`no_kinematic`, `no_relation`, `no_outcome_anchor`, and
`no_monotonic_projection`. The isolated configs and reports are generated by:

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --seed 0 --seed 1 --seed 2 --skip-completed
```

The independent evaluator also reports non-trained random, pixel-goal,
visual-token cosine, visual directional-MI, privileged-teacher and oracle
candidate-selection baselines.

The student architecture and deployment inputs must remain unchanged across
these runs so improvements can be attributed to privileged supervision.
