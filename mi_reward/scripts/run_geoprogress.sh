#!/usr/bin/env bash
# Verified GeoProgress pipeline. The Cosmos job itself is external: this script
# only ingests RGB candidates with their action/state/relation sidecars.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv-rlpd/bin/activate" ]; then
    source .venv-rlpd/bin/activate
fi

MANIFEST="${MANIFEST:-dataset/mi_reward/manifests/cosmos_action_cond_train.jsonl}"
SUCCESS_REFS="${SUCCESS_REFS:-dataset/mi_reward/manifests/success_refs.jsonl}"
FEATURE_ROOT="${FEATURE_ROOT:-dataset/mi_reward/geoprogress_features}"
PREFERENCES="${PREFERENCES:-dataset/mi_reward/preferences/geoprogress_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/mi_reward/geoprogress}"
DEVICE="${DEVICE:-cuda}"
SCORING_MI_MODE="${SCORING_MI_MODE:-gaussian_mi_proxy}"
MI_BACKEND="${MI_BACKEND:-$SCORING_MI_MODE}"

if [ -n "${CANDIDATE_RECORDS:-}" ]; then
    : "${FEASIBILITY_CONFIG:?FEASIBILITY_CONFIG is required with CANDIDATE_RECORDS}"
    python -m mi_reward.data.build_manifest \
        --source_type cosmos_action_cond \
        --candidate_records "$CANDIDATE_RECORDS" \
        --feasibility_config "$FEASIBILITY_CONFIG" \
        --output "$MANIFEST" \
        --split train
fi

: "${LAM_CONFIG_PATH:?LAM_CONFIG_PATH must point to the LaWAM YAML}"
: "${LAM_CKPT_PATH:?LAM_CKPT_PATH must point to the LaWAM checkpoint}"
: "${VISION_MODEL_ID:?VISION_MODEL_ID must point to local DINOv3 weights}"

python -m mi_reward.features.cached_feature_store \
    --manifest "$MANIFEST" \
    --success_refs "$SUCCESS_REFS" \
    --feature_root "$FEATURE_ROOT" \
    --feature_extractor lawam_lam \
    --lam_config_path "$LAM_CONFIG_PATH" \
    --lam_ckpt_path "$LAM_CKPT_PATH" \
    --vision_model_id "$VISION_MODEL_ID" \
    --device "$DEVICE" --strict --tokens

python -m mi_reward.scoring.build_preferences \
    --manifest "$MANIFEST" \
    --success_refs "$SUCCESS_REFS" \
    --feature_root "$FEATURE_ROOT" \
    --output "$PREFERENCES" \
    --token_features \
    --relation_weight "${RELATION_WEIGHT:-1.0}" \
    --mi_mode "$SCORING_MI_MODE" \
    --gamma "${GAMMA:-0.99}" \
    --margin "${MARGIN:-0.05}" \
    --top_k "${TOP_K:-5}" \
    --bottom_k "${BOTTOM_K:-5}" \
    --teacher_version "${TEACHER_VERSION:-geoprogress_v1}"

python -m mi_reward.training.train_reward_sft \
    --mode geoprogress \
    --manifest "$MANIFEST" \
    --success_refs "$SUCCESS_REFS" \
    --preferences "$PREFERENCES" \
    --feature_root "$FEATURE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "${BATCH_SIZE:-8}" \
    --epochs "${EPOCHS:-20}" \
    --lr "${LR:-1e-4}" \
    --hidden_dim "${HIDDEN_DIM:-256}" \
    --architecture "${ARCHITECTURE:-gru}" \
    --relation_weight "${RELATION_WEIGHT:-1.0}" \
    --mi_backend "$MI_BACKEND" \
    --gamma "${GAMMA:-0.99}" \
    --device "$DEVICE"
