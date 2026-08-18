#!/usr/bin/env bash
# Build LAM features, pseudo-preferences, and an instance-aware GeoProgress reward head.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv-rlpd/bin/activate" ]; then
    source .venv-rlpd/bin/activate
fi

CONFIG="mi_reward/configs/instance_geoprogress.yaml"
MANIFEST_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --manifest) MANIFEST_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

read_config() {
    python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"
}

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

# This makes artifact completeness a hard gate before expensive feature jobs.
python -m mi_reward.data.instance_pipeline --config "$CONFIG" --validate-training-manifest "$MANIFEST" \
    --success-refs "$SUCCESS_REFS"

python -m mi_reward.features.cached_feature_store \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" \
    --feature_extractor lawam_lam --lam_config_path "$LAM_CONFIG" --lam_ckpt_path "$LAM_CKPT" \
    --vision_model_id "$DINO_ROOT" --strict --tokens
python -m mi_reward.scoring.build_preferences \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" --output "$PREFERENCES" \
    --token_features --relation_weight "$RELATION_WEIGHT" --teacher_version instance_geoprogress_v1
python -m mi_reward.training.train_reward_sft --mode geoprogress \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --preferences "$PREFERENCES" \
    --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_DIR" --hidden_dim "$HIDDEN_DIM" \
    --num_heads "$NUM_HEADS" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR"
