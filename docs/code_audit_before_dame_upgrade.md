# Pre-Modification Code Audit: Dame-Inspired Upgrade

**Date:** 2026-07-20
**Repository:** MI-directional-WAM-reward-model-pretrain
**Base:** RLinf/LaWAM

## Existing mi_reward Components

### Scoring (mi_reward/scoring/)

| File | Status | Notes |
|---|---|---|
| `mi_potential.py` | Implemented | `_gaussian_mi_proxy()` — correlation-based MI proxy on pooled vectors. `_histogram_mi()` — basic tanh-binned soft-histogram MI, NOT B-spline. `compute_mi()` — dispatcher. Both operate on pooled [D] vectors, no token-level support. |
| `trajectory_score.py` | Implemented | `compute_frame_potential()` — framewise max MI over reference frames. `compute_delta_score()` — `gamma * phi[t+1] - phi[t]`. `score_trajectory()` — returns phi list, delta score, mean score. |
| `build_preferences.py` | Implemented | `score_manifest()` — scores all trajectories against best reference. `build_preference_pairs()` — top-k vs bottom-k with margin. No confidence, no versioning, no metadata beyond `score_type: mi_delta`. |

### Models (mi_reward/models/)

| File | Status | Notes |
|---|---|---|
| `reward_head.py` | Implemented | `TrajectoryRewardHead` — MLP (Linear+GELU+Dropout+Linear+GELU+Linear). Mean+last pooling. Scalar trajectory reward output. No state-potential output, no GRU/transformer. |

### Training (mi_reward/training/)

| File | Status | Notes |
|---|---|---|
| `loss.py` | Implemented | `pairwise_ranking_loss()` — `-log_sigmoid(chosen - rejected).mean()`. No potential distillation, no direction loss, no confidence weighting. |
| `collator.py` | Implemented | `PreferenceCollator` — pads chosen/rejected features, builds masks. |
| `train_reward_sft.py` | Implemented | Simple training loop. AdamW, no multi-objective support. |

### Features (mi_reward/features/)

| File | Status | Notes |
|---|---|---|
| `base_extractor.py` | Implemented | `BaseFeatureExtractor` — returns pooled [D] per frame. No `extract_frame_tokens()` returning [N, D]. |
| `dino_v3_extractor.py` | Implemented | Returns pooled feature per frame (mean over tokens). |
| `lawam_lam_extractor.py` | Implemented | Returns pooled feature per frame (mean over tokens). Has `extract_vision_features` access but pools immediately. |
| `cached_feature_store.py` | Implemented | Saves/loads `.pt` tensors. No token-level cache distinction. |

### Data (mi_reward/data/)

| File | Status | Notes |
|---|---|---|
| `schema.py` | Implemented | Frozen dataclasses: `TrajectoryExample`, `SuccessReference`, `PreferencePair`. No confidence/metadata fields on `PreferencePair`. |
| `build_manifest.py` | Implemented | LIBERO and RoboTwin manifest builders. |
| `preference_dataset.py` | Implemented | Loads features from cache. Pooled only. |
| `video_dataset.py` | Implemented | `TrajectoryManifestDataset`. |

### Evaluation (mi_reward/evaluation/)

| File | Status | Notes |
|---|---|---|
| `eval_reward_ranking.py` | Implemented | Pairwise accuracy, reward margin, score correlation. |
| `eval_progress_corr.py` | Implemented | Progress correlation, success/failure AUC. |

### Configuration

| File | Status | Notes |
|---|---|---|
| `configs/mi_reward_base.yaml` | Implemented | Basic YAML config with paths, features, scoring, training sections. No MI estimator config, no alignment config, no directional score weights, no loss weights. |

### Tests

| File | Status | Notes |
|---|---|---|
| `tests/test_mi_reward_smoke.py` | Implemented | Synthetic smoke test covering full pipeline. Uses `gaussian_mi_proxy` only. |

## Gaps Identified

1. **No B-spline MI estimator.** Current histogram MI uses tanh-based binning, not B-spline kernels.
2. **No token correspondence.** Features are pooled to [D] before MI computation.
3. **No temporal alignment.** Frame matching is framewise max over reference frames.
4. **No directional score components.** Only legacy delta score exists.
5. **No state-potential model.** Reward head outputs scalar per trajectory, not per-frame potentials.
6. **No distillation losses.** Only pairwise ranking loss exists.
7. **No confidence weighting.** Preference pairs have no confidence scores.
8. **No baseline comparison infrastructure.**
9. **Feature extractors pool tokens.** No `extract_frame_tokens()` returning [N, D].
10. **No alignment module.** `mi_reward/alignment/` does not exist.

## Broken / Placeholder Items

- None found. All modules import cleanly and the smoke test passes.

## Duplications

- None found. Architecture is modular with clear separation.

## Commands Documented vs Implemented

All CLI entry points (`build_manifest`, `cached_feature_store`, `build_preferences`,
`train_reward_sft`, `eval_reward_ranking`, `eval_progress_corr`) have matching
`main()` functions with argparse. Shell wrappers under `mi_reward/scripts/` are
consistent with the Python modules.

## Environment

- Python: 3.10+ (per pyproject.toml)
- PyTorch: 2.6.0 (per requirements.txt)
- No pytest available in current environment; smoke test logic verified via code inspection
- `mi_reward` package imports successfully
