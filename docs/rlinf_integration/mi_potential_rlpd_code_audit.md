# MI Potential RLPD Integration: Code Audit

**Date:** 2026-07-20
**MI Repo:** `MI-directional-WAM-reward-model-pretrain` (v0.2.0)
**RLinf Repo:** NOT AVAILABLE on this machine
**Status:** Integration code prepared as drop-in modules

## RLinf Repository

The RLinf repository (containing `rlinf/workers/`, `examples/embodiment/`,
`rlinf/data/replay_buffer.py`, etc.) is not cloned on this machine.

Expected paths in RLinf (from the task specification):

```
rlinf/workers/reward/          # Reward worker and model registry
rlinf/workers/env/             # EnvWorker
rlinf/workers/rollout/         # Rollout worker
rlinf/data/replay_buffer.py    # TrajectoryReplayBuffer
examples/embodiment/config/    # RLPD configs
  realworld_peginsertion_rlpd_cnn_async.yaml
examples/embodiment/
  run_realworld_async.sh
  train_async.py
```

Known symbols to locate: `reward_model_registry`, `RewardGroup`, `compute_reward`,
`compute_bootstrap_rewards`, `standalone_realworld`, `reward_model_output`,
`env_reward_weight`, `reward_weight`, `TrajectoryReplayBuffer`, `demo_buffer`,
`replay_buffer`, `sample`, `main_images`, `extra_view_images`, `wrist_images`.

## MI Reward Repository (Available)

### Checkpoint Format

```python
# mi_reward/training/train_reward_sft.py, line 84:
torch.save(
    {"model_state_dict": model.state_dict(), "config": config},
    out / "pytorch_model.pt"
)
```

Config dict keys: `preferences`, `feature_root`, `batch_size`, `epochs`, `lr`,
`hidden_dim`, `input_dim`, `seed`.

**Missing from checkpoint:** architecture, token_pooling, num_layers, num_heads,
use_task_conditioning, input normalization stats (mean/std), encoder name.

### Model Architecture

```python
# mi_reward/models/state_potential_model.py
class StatePotentialRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, architecture="gru",
                 token_pooling="mean", num_layers=2, num_heads=4,
                 dropout=0.1, use_task_conditioning=False, task_dim=None)

    def forward(self, frame_or_token_features, task_features=None) -> Tensor:
        # Input:  [B, T, D] pooled or [B, T, N, D] token features
        # Output: [B, T] scalar potentials per timestep
```

### Key Architectural Details

| Property | Value |
|---|---|
| Input requirement | Pre-extracted feature vectors, NOT raw images |
| Pooling | Token pooler (mean/attention/max), then temporal model |
| Temporal model | GRU (default), MLP, or Transformer |
| Output | Scalar potential per frame [B, T] |
| Confidence head | None — confidence from scoring, not inference |
| Task conditioning | Optional, disabled by default |
| Single-frame support | Yes — `forward(x[:, 0:1, :])` returns [B, 1] |

### Image Preprocessing

The model does NOT handle images. Images must be preprocessed by:
- `mi_reward/features/dino_v3_extractor.py` (DINOv3)
- `mi_reward/features/lawam_lam_extractor.py` (LaWAM LAM)

Preprocessing steps:
1. PIL Image → RGB, resize to (image_size, image_size)
2. ToTensor + ImageNet normalization
3. Forward through frozen encoder → mean-pool tokens → [D] vector
4. Stack frames → [T, D]

### Package Importability

```python
import mi_reward  # works, version 0.2.0
from mi_reward.models.state_potential_model import StatePotentialRewardModel
```

### Single-Frame vs Temporal Window

- Model `forward()` requires [B, T, D] with T ≥ 1
- Single frame: T=1 → [B, 1, D] → return [B, 1]
- Multiple frames: T>1 → temporal model (GRU/transformer) processes sequence
- For RLPD online use: T=1 (current frame only) is sufficient and uses the GRU's initial hidden state
- For history-aware: T can be accumulated frames in a sliding window

## Incompatible Assumptions

1. **No image encoder in checkpoint.** The model expects pre-extracted features.
   RLinf must either:
   - Reuse its own visual encoder and pass features to the potential model (preferred)
   - Load a separate encoder checkpoint alongside the potential model

2. **No confidence head.** During online inference, confidence defaults to 1.0
   unless we add a heuristic (e.g., based on feature statistics).

3. **No normalization metadata.** The checkpoint lacks potential mean/std from
   training. We add optional online normalization statistics.

4. **No metadata.json.** We need to create one alongside the checkpoint.

## Files Needing Modification in RLinf

Based on the task specification:

| RLinf File | Modification |
|---|---|
| `rlinf/workers/reward/` (new file) | Add `MIPotentialRewardModel` adapter |
| `rlinf/workers/reward/__init__.py` or registry | Register `mi_potential` model type |
| `rlinf/workers/env/` (EnvWorker) | Add PotentialShapingState, delta computation |
| `rlinf/workers/rollout/` (if reward-model inputs built here) | Ensure image keys passed |
| `rlinf/data/replay_buffer.py` | Optionally store mi_delta, mi_confidence |
| `examples/embodiment/config/` | New MI potential config |
| Rewards combining logic | Add reward_output_type: state_potential |

## Next Steps Before Real-Robot Execution

1. Clone RLinf repository and verify exact file paths
2. Integrate `MIPotentialInferenceModel` as drop-in reward adapter
3. Add `PotentialShapingState` to EnvWorker
4. Test in dummy mode
5. Train a real StatePotentialRewardModel checkpoint on world-model futures
6. Export checkpoint with full metadata
7. Validate on LIBERO before Franka
