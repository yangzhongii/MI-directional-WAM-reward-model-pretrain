#!/usr/bin/env bash
set -euo pipefail

python -m mi_reward.training.train_reward_sft \
  --preferences "${PREFERENCES:-dataset/mi_reward/preferences/train_preferences.jsonl}" \
  --feature_root "${FEATURE_ROOT:-dataset/mi_reward/features}" \
  --output_dir "${OUTPUT_DIR:-results/mi_reward/reward_head_mvp}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-5}" \
  --lr "${LR:-1e-4}" \
  --device "${DEVICE:-cuda}"
