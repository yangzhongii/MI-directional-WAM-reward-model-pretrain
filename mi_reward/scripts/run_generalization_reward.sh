#!/usr/bin/env bash
# Stage 2: train the generalization-aware directional progress reward model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv-reward/bin/activate" ]; then
    source .venv-reward/bin/activate
else
    echo "Missing reward environment. Run: bash requirements/install.sh --reward-train" >&2
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
read_list() { python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); [print(v) for v in $1]"; }
MANIFEST="${MANIFEST_OVERRIDE:-$(read_config "c['paths']['manifest']")}" 
SUCCESS_REFS="$(read_config "c['paths']['success_refs']")"
FEATURE_ROOT="$(read_config "c['paths']['feature_root']")"
PREFERENCES="$(read_config "c['paths']['preferences']")"
TEACHER_TARGETS="$(read_config "c['paths'].get('teacher_targets', c['paths']['preferences'].rsplit('.', 1)[0] + '.teacher_targets.pt')")"
OUTPUT_DIR="$(read_config "c['paths']['output_dir']")"
LAM_CONFIG="$(read_config "c['features']['lam_config_path']")"
LAM_CKPT="$(read_config "c['features']['lam_ckpt_path']")"
DINO_ROOT="$(read_config "c['features']['dino_checkpoint_root']")"
RELATION_WEIGHT="$(read_config "c['training']['relation_weight']")"
ACTION_WEIGHT="$(read_config "c['training'].get('action_weight', 1.0)")"
KINEMATIC_WEIGHT="$(read_config "c['training'].get('kinematic_weight', 1.0)")"
OUTCOME_WEIGHT="$(read_config "c['training'].get('outcome_weight', 2.0)")"
SUCCESS_MONOTONIC="$(read_config "c['training'].get('success_monotonic_projection', True)")"
SUCCESS_ENDPOINT_ANCHOR="$(read_config "c['training'].get('success_endpoint_anchor', True)")"
PAIR_MODE="$(read_config "c['training'].get('pair_mode', 'outcome_anchored')")"
PAIR_SCOPE="$(read_config "c['training'].get('pair_scope', 'same_context')")"
MIN_SUCCESS_ENDPOINT_GAIN="$(read_config "c['training'].get('min_success_endpoint_gain', 0.05)")"
GOAL_DROPOUT="$(read_config "c['training'].get('goal_dropout', 0.2)")"
SEED="$(read_config "c['training'].get('seed', 0)")"
HIDDEN_DIM="$(read_config "c['training']['hidden_dim']")"
NUM_HEADS="$(read_config "c['training']['num_heads']")"
BATCH_SIZE="$(read_config "c['training']['batch_size']")"
EPOCHS="$(read_config "c['training']['epochs']")"
LR="$(read_config "c['training']['lr']")"
GAMMA="$(read_config "c['training']['gamma']")"
MI_BACKEND="$(read_config "c['training']['mi_backend']")"
PREFERENCE_MI_MODE="$(read_config "c['training'].get('preference_mi_mode', 'gaussian_mi_proxy')")"
LAMBDA_RANK="$(read_config "c['training']['lambda_rank']")"
LAMBDA_POTENTIAL="$(read_config "c['training']['lambda_potential']")"
LAMBDA_DIRECTION="$(read_config "c['training']['lambda_direction']")"
DIRECTIONAL="$(read_config "c['training']['directional_alignment']")"
mapfile -t SCORE_SPLITS < <(read_list "c['training'].get('score_splits', ['train'])")
mapfile -t TRAIN_SPLITS < <(read_list "c['training'].get('train_splits', ['train'])")
mapfile -t VALIDATION_SPLITS < <(read_list "c['training'].get('validation_splits', ['instance_heldout', 'scene_heldout'])")

echo "[reward] Stage 1/4: validating accepted candidates"
python -m mi_reward.data.generalization_pipeline --config "$CONFIG" \
    --validate-training-manifest "$MANIFEST" --success-refs "$SUCCESS_REFS"
FEATURE_ARGS=(--manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" \
    --feature_extractor lawam_lam --lam_config_path "$LAM_CONFIG" --lam_ckpt_path "$LAM_CKPT" \
    --vision_model_id "$DINO_ROOT" --strict --tokens)
if [[ ! "$ACTION_WEIGHT" =~ ^0+([.]0+)?$ ]]; then
    FEATURE_ARGS+=(--action-latents)
fi
if [[ ! "$KINEMATIC_WEIGHT" =~ ^0+([.]0+)?$ ]]; then
    FEATURE_ARGS+=(--kinematic-latents)
fi
echo "[reward] Stage 2/4: caching visual, action-process and kinematic latents"
python -m mi_reward.features.cached_feature_store "${FEATURE_ARGS[@]}"

PREFERENCE_ARGS=(--manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --feature_root "$FEATURE_ROOT" \
    --output "$PREFERENCES" --token_features --relation_weight "$RELATION_WEIGHT" \
    --action_weight "$ACTION_WEIGHT" --teacher_version privileged_mi_directional_v4 \
    --kinematic_weight "$KINEMATIC_WEIGHT" --outcome_weight "$OUTCOME_WEIGHT" \
    --pair_mode "$PAIR_MODE" --pair-scope "$PAIR_SCOPE" \
    --teacher-target-output "$TEACHER_TARGETS" \
    --min-success-endpoint-gain "$MIN_SUCCESS_ENDPOINT_GAIN" \
    --gamma "$GAMMA" --mi_mode "$PREFERENCE_MI_MODE" --seed "$SEED")
if [ "$SUCCESS_MONOTONIC" = "False" ] || [ "$SUCCESS_MONOTONIC" = "false" ]; then
    PREFERENCE_ARGS+=(--disable-success-monotonic-projection)
fi
if [ "$SUCCESS_ENDPOINT_ANCHOR" = "False" ] || [ "$SUCCESS_ENDPOINT_ANCHOR" = "false" ]; then
    PREFERENCE_ARGS+=(--disable-success-endpoint-anchor)
fi
for split in "${SCORE_SPLITS[@]}"; do
    PREFERENCE_ARGS+=(--split "$split")
done
if [ "$DIRECTIONAL" = "True" ] || [ "$DIRECTIONAL" = "true" ]; then
    PREFERENCE_ARGS+=(--directional-alignment \
        --directional_w_endpoint "$(read_config "c['training']['directional_w_endpoint']")" \
        --directional_w_positive "$(read_config "c['training']['directional_w_positive']")" \
        --directional_w_regression "$(read_config "c['training']['directional_w_regression']")" \
        --directional_w_stage "$(read_config "c['training']['directional_w_stage']")" \
        --directional_w_alignment "$(read_config "c['training']['directional_w_alignment']")")
fi
echo "[reward] Stage 3/4: directional MI scoring and preference construction"
python -m mi_reward.scoring.build_preferences "${PREFERENCE_ARGS[@]}"
echo "[reward] Stage 4/4: privileged-teacher reward distillation"
TRAINING_ARGS=(--mode generalization \
    --manifest "$MANIFEST" --success_refs "$SUCCESS_REFS" --preferences "$PREFERENCES" \
    --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_DIR" --hidden_dim "$HIDDEN_DIM" \
    --num_heads "$NUM_HEADS" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR" \
    --gamma "$GAMMA" --mi_backend "$MI_BACKEND" --relation_weight "$RELATION_WEIGHT" \
    --action_weight "$ACTION_WEIGHT" --goal_dropout "$GOAL_DROPOUT" \
    --kinematic_weight "$KINEMATIC_WEIGHT" --teacher_targets_source "$TEACHER_TARGETS" \
    --lambda_rank "$LAMBDA_RANK" --lambda_potential "$LAMBDA_POTENTIAL" \
    --lambda_direction "$LAMBDA_DIRECTION" --seed "$SEED")
for split in "${TRAIN_SPLITS[@]}"; do
    TRAINING_ARGS+=(--train_split "$split")
done
for split in "${VALIDATION_SPLITS[@]}"; do
    TRAINING_ARGS+=(--validation_split "$split")
done
python -m mi_reward.training.train_reward_sft "${TRAINING_ARGS[@]}"
