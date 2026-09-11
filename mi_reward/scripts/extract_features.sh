#!/usr/bin/env bash
set -euo pipefail

python -m mi_reward.features.cached_feature_store \
  --manifest "${MANIFEST:-dataset/mi_reward/manifests/train_manifest.jsonl}" \
  --success_refs "${SUCCESS_REFS:-dataset/mi_reward/manifests/success_refs.jsonl}" \
  --feature_root "${FEATURE_ROOT:-dataset/mi_reward/features}" \
  --feature_extractor "${FEATURE_EXTRACTOR:-lawam_lam}" \
  --lam_config_path "${LAM_CONFIG_PATH:-}" \
  --lam_ckpt_path "${LAM_CKPT_PATH:-}" \
  --vision_model_id "${VISION_MODEL_ID:-}" \
  --model_path "${DINO_MODEL_PATH:-}" \
  --device "${DEVICE:-cuda}"
