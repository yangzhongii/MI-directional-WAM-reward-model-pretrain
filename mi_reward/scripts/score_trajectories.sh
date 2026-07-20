#!/usr/bin/env bash
set -euo pipefail

python -m mi_reward.scoring.build_preferences \
  --manifest "${MANIFEST:-dataset/mi_reward/manifests/train_manifest.jsonl}" \
  --success_refs "${SUCCESS_REFS:-dataset/mi_reward/manifests/success_refs.jsonl}" \
  --feature_root "${FEATURE_ROOT:-dataset/mi_reward/features}" \
  --output "${PREFERENCES:-dataset/mi_reward/preferences/train_preferences.jsonl}" \
  --gamma "${GAMMA:-0.99}" \
  --margin "${MARGIN:-0.05}" \
  --top_k "${TOP_K:-5}" \
  --bottom_k "${BOTTOM_K:-5}" \
  --mi_mode "${MI_MODE:-gaussian_mi_proxy}"
