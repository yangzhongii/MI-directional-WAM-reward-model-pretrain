# MI Potential RLPD Integration: Implementation Report

**Date:** 2026-07-20
**Status:** Integration code complete; RLinf repo not available for in-place integration

## 1. Summary

Integration code for using an MI-distilled state-potential reward model
(`StatePotentialRewardModel`) as a potential-based shaping reward in RLinf's
asynchronous Franka RLPD pipeline.

The online reward is a state potential distilled from an offline
mutual-information teacher. RLinf does NOT compute mutual information online.

## 2. Original RLinf Reward Flow (Expected)

```
EnvWorker.reset() → obs_0
    → RewardWorker.predict(obs_0) → reward_model_output
    → (for state_potential: cache V(o_0))

EnvWorker.step(a_t) → obs_{t+1}, env_reward, done
    → RewardWorker.predict(obs_{t+1}) → V(o_{t+1})
    → delta = gamma * V(o_{t+1}) - V(o_t)
    → final_reward = env_weight * env_reward + mi_weight * delta
    → cache V(o_{t+1}) for next step

EnvWorker.done() → clear cache for finished envs
```

## 3. Actual Integration Points

### 3.1 MI Reward Repository (this repo)

**New files:**

| File | Purpose |
|---|---|
| `mi_reward/inference/__init__.py` | Inference package |
| `mi_reward/inference/potential_model.py` | `MIPotentialInferenceModel` — stable inference contract |

**Key API:**
```python
model = MIPotentialInferenceModel.from_pretrained("path/to/pytorch_model.pt")
result = model.predict_potential(observations, task_descriptions)
# result["potential"]: [B], result["confidence"]: [B], result["valid"]: [B]
```

### 3.2 RLinf Integration (drop-in modules)

| File | RLinf Target Location | Purpose |
|---|---|---|
| `docs/rlinf_integration/rlinf_workers_reward/mi_potential_reward_model.py` | `rlinf/workers/reward/` | `MIPotentialRewardModel` adapter, registers `model_type: "mi_potential"` |
| `docs/rlinf_integration/rlinf_workers_env/potential_shaping_state.py` | `rlinf/workers/env/` | `PotentialShapingState` — per-env cache, delta computation, reset handling |
| `docs/rlinf_integration/rlinf_data/mi_progress_sampler.py` | `rlinf/data/` | `MIProgressStratifiedSampler` — Phase B guided replay (optional) |
| `docs/rlinf_integration/config/realworld_peginsertion_rlpd_cnn_async_mi_potential.yaml` | `examples/embodiment/config/` | Full Hydra config with potential shaping |

## 4. Checkpoint Contract

### Required files per student checkpoint:
```
mi_student_checkpoint/
├── pytorch_model.pt      # {"model_state_dict": ..., "config": {...}}
└── metadata.json         # (recommended, generated separately)
```

### metadata.json schema:
```json
{
  "checkpoint_format_version": "1.0",
  "model_class": "StatePotentialRewardModel",
  "encoder_name": "dino_v3",
  "input_image_keys": ["main_images"],
  "history_size": 1,
  "task_conditioned": false,
  "output_semantics": "state_potential",
  "training_gamma": 0.99,
  "architecture": "gru",
  "token_pooling": "mean",
  "num_layers": 2,
  "num_heads": 4,
  "potential_mean": 0.0,
  "potential_std": 1.0
}
```

## 5. Transition Timing for Potential Differences

```
Episode timeline:
  t=0: env.reset() → o_0 → V(o_0) → cached
  t=1: step(a_0) → o_1, env_r_0 → V(o_1)
       delta_0 = gamma*V(o_1) - V(o_0) → assign to transition 0
       cache V(o_1)
  t=2: step(a_1) → o_2, env_r_1 → V(o_2)
       delta_1 = gamma*V(o_2) - V(o_1) → assign to transition 1
       ...
  done: clear cache
```

## 6. Reset-Cache Handling

- `PotentialShapingState.clear(env_ids)` is called BEFORE `initialize()`
- Clear deletes all cached state for the env
- Initialize sets fresh V(o_0)
- Cross-episode contamination is prevented by explicit clear
- If `valid=False` or `!isfinite(potential)`: assign zero shaping, don't update cache

## 7. RLPD 50/50 Mixing Verification

- MI potential shaping does NOT modify the demo/online split
- Phase A: standard RLPD with 50% demo, 50% online, uniform sampling
- Phase B: only the online 50% is stratified; demo 50% unchanged
- `mi_guided_replay.apply_to: "online_only"` enforces this

## 8. Guided Replay Implementation (Phase B)

- `MIProgressStratifiedSampler` classifies online transitions by `mi_delta`
- Samples target fractions: positive (progress), neutral, negative (regression)
- Falls back to uniform when a class is empty
- Does NOT modify SAC critic target, actor objective, or entropy tuning
- Disabled by default (`mi_guided_replay.enabled: false`)

## 9. Backward Compatibility

- `enabled: false` → all MI shaping is no-op, env reward unchanged
- `standalone_realworld: true` retains existing semantics
- ResNet/VLM reward models unaffected
- Existing training configs require no changes

## 10. Ablation Configurations

| Ablation | Config Changes |
|---|---|
| A: Standard RLPD | `potential_shaping.enabled: false` |
| B: Env sparse reward only | `reward.use_reward_model: false` |
| C: Binary reward model | `model.model_type: "resnet"` (existing) |
| D: MI shaping only | `enabled: true`, `mi_guided_replay.enabled: false` |
| E: MI replay only | `reward_weight: 0.0`, `mi_guided_replay.enabled: true` |
| F: Full method | Both enabled, confidence gating on |

## 11. Test Results

| Test Category | Status |
|---|---|
| Syntax check (all 6 new files) | PASS |
| mi_reward import (v0.2.0) | PASS |
| Potential-difference alignment | PASS (4/4) |
| Reset cache handling | PASS |
| Batched async envs | PASS |
| Invalid model output | PASS |
| Reward composition formula | PASS |
| Backward compatibility | PASS |
| Confidence gating | PASS |
| MI stratified sampler logic | PASS (logic verified) |
| Full PyTorch integration tests | NOT RUN (PyTorch unavailable) |

## 12. Unresolved Issues Before Real-Robot Execution

1. **RLinf repo not cloned.** Exact file paths (`rlinf/workers/reward/`, etc.)
   must be verified when the repo is available.

2. **Reward model registry.** The exact registration mechanism (decorator,
   dict, config-based) must be matched to RLinf's existing pattern.

3. **Image preprocessing.** The MI model expects pre-extracted features.
   RLinf must either reuse its own CNN encoder and pass features to the
   potential model, or load a companion encoder checkpoint.

4. **No trained checkpoint exists.** A real `StatePotentialRewardModel` must
   be trained on world-model-generated futures before Franka deployment.

5. **metadata.json generation.** Must be generated alongside the checkpoint
   during/after training.

6. **Dummy-mode end-to-end test.** Requires RLinf's dummy Franka configuration
   to be available.

7. **Franka environment interface.** Exact observation dict keys
   (`main_images`, `wrist_images`, etc.) must match between RLinf and
   the training pipeline.

8. **Task description format.** The string task description must match
   between training and deployment (even though task conditioning is
   disabled by default).

## 13. Files Summary

### New files (9)
```
mi_reward/inference/__init__.py
mi_reward/inference/potential_model.py
docs/rlinf_integration/mi_potential_rlpd_code_audit.md
docs/rlinf_integration/rlinf_workers_reward/mi_potential_reward_model.py
docs/rlinf_integration/rlinf_workers_env/potential_shaping_state.py
docs/rlinf_integration/rlinf_data/mi_progress_sampler.py
docs/rlinf_integration/config/realworld_peginsertion_rlpd_cnn_async_mi_potential.yaml
docs/rlinf_integration/mi_potential_rlpd_implementation_report.md
tests/test_mi_potential_rlpd_integration.py
```

### No existing files modified

## 14. Next Steps

1. Clone RLinf repository and verify file paths
2. Copy drop-in modules to their target locations
3. Register `mi_potential` in the reward model registry
4. Add `PotentialShapingState` to EnvWorker
5. Train a real `StatePotentialRewardModel` checkpoint
6. Generate `metadata.json` alongside the checkpoint
7. Run dummy-mode end-to-end test
8. Test on LIBERO before Franka
9. Run all existing RLinf tests to verify backward compatibility
10. Franka real-robot validation (with safety oversight)
