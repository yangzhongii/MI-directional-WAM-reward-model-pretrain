#!/usr/bin/env bash
# Stage 2: train the generalization-aware directional progress reward model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Missing generalization environment. Run: bash requirements/install.sh --generalization-data" >&2
    exit 1
fi

CONFIG="mi_reward/configs/generalization_reward.yaml"
MANIFEST_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --manifest) MANIFEST_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

read_config() { python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
MANIFEST="${MANIFEST_OVERRIDE:-$(read_config "c['paths']['manifest']")}" 
SUCCESS_REFS="$(read_config "c['paths']['success_refs']")"
FEATURE_ROOT="$(read_config "c['paths']['feature_root']")"
PREFERENCES="$(read_config "c['paths']['preferences']")"
OUTPUT_DIR="$(read_config "c['paths']['output_dir']")"
LAM_CONFIG="$(read_config "c['features']['lam_config_path']")"
LAM_CKPT="$(read_config "c['features']['lam_ckpt_path']")"
DINO_ROOT="$(read_config "c['features']['dino_checkpoint_root']")"
RELATION_WEIGHT="$(read_config "c['training']['relation_weight']")"
HIDDEN_DIM="$(read_config "c['training']['hidden_dim']")"
NUM_HEADS="$(read_config "c['training']['num_heads']")"
BATCH_SIZE="$(read_config "c['training']['batch_size']")"
EPOCHS="$(read_config "c['training']['epochs']")"
LR="$(read_config "c['training']['lr']")"
GAMMA="$(read_config "c['training']['gamma']")"
MI_BACKEND="$(read_config "c['training']['mi_backend']")"
LAMBDA_RANK="$(read_config "c['training']['lambda_rank']")"
LAMBDA_POTENTIAL="$(read_config "c['training']['lambda_potential']")"
LAMBDA_DIRECTION="$(read_config "c['training']['lambda_direction']")"
DIRECTIONAL="$(read_config "c['training']['directional_alignment']")"

python -m mi_reward.data.generalization_pipeline --config "$CONFIG" \
    --validate-training-manifest "$MANIFEST" --success-refs "$SUCCESS_REFS"
python -m mi_reward.features.cached_feature_store \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" \
    --feature_extractor lawam_lam --lam_config_path "$LAM_CONFIG" --lam_ckpt_path "$LAM_CKPT" \
    --vision_model_id "$DINO_ROOT" --strict --tokens

PREFERENCE_ARGS=(--manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" \
    --output "$PREFERENCES" --token_features --relation_weight "$RELATION_WEIGHT" \
    --teacher_version generalization_directional_v1 --gamma "$GAMMA" --mi_mode gaussian_mi_proxy)
if [ "$DIRECTIONAL" = "True" ] || [ "$DIRECTIONAL" = "true" ]; then
    PREFERENCE_ARGS+=(--directional-alignment \
        --directional_w_endpoint "$(read_config "c['training']['directional_w_endpoint']")" \
        --directional_w_positive "$(read_config "c['training']['directional_w_positive']")" \
        --directional_w_regression "$(read_config "c['training']['directional_w_regression']")" \
        --directional_w_stage "$(read_config "c['training']['directional_w_stage']")" \
        --directional_w_alignment "$(read_config "c['training']['directional_w_alignment']")")
fi
python -m mi_reward.scoring.build_preferences "${PREFERENCE_ARGS[@]}"
python -m mi_reward.training.train_reward_sft --mode generalization \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --preferences "$PREFERENCES" \
    --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_DIR" --hidden_dim "$HIDDEN_DIM" \
    --num_heads "$NUM_HEADS" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR" \
    --gamma "$GAMMA" --mi_backend "$MI_BACKEND" --relation_weight "$RELATION_WEIGHT" \
    --lambda_rank "$LAMBDA_RANK" --lambda_potential "$LAMBDA_POTENTIAL" \
    --lambda_direction "$LAMBDA_DIRECTION"
