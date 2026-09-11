#!/usr/bin/env bash
set -euo pipefail

python -m mi_reward.evaluation.eval_reward_ranking \
  --preferences "${PREFERENCES:-dataset/mi_reward/preferences/val_preferences.jsonl}" \
  --feature_root "${FEATURE_ROOT:-dataset/mi_reward/features}" \
  --ckpt "${CKPT:-results/mi_reward/reward_head_mvp/pytorch_model.pt}" \
  --device "${DEVICE:-cuda}"
