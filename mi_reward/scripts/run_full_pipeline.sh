#!/usr/bin/env bash
# Full MI Directional Reward Pipeline
# ====================================
# All params read from mi_reward/configs/train_mi_reward.yaml
#
# Usage:
#   bash mi_reward/scripts/run_full_pipeline.sh                    # mock
#   bash mi_reward/scripts/run_full_pipeline.sh --real-cosmos      # real Cosmos on 4090
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# Activate venv if available
if [ -f ".venv-rlpd/bin/activate" ]; then
    source .venv-rlpd/bin/activate
fi

CONFIG="${CONFIG:-mi_reward/configs/train_mi_reward.yaml}"
COSMOS_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real-cosmos) COSMOS_FLAG="--real-cosmos"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- Read params from YAML ----
_read_yaml() {
    python3 -c "import yaml,sys;c=yaml.safe_load(open('$CONFIG'));print(c$1)" 2>/dev/null
}

NUM_CANDIDATES=$(_read_yaml "['cosmos']['generation']['num_candidates']")
NUM_FRAMES=$(_read_yaml "['cosmos']['generation']['num_frames']")
TOP_K=$(_read_yaml "['preferences']['top_k']")
BOTTOM_K=$(_read_yaml "['preferences']['bottom_k']")
MARGIN=$(_read_yaml "['preferences']['margin']")
SCORE_FIELD=$(_read_yaml "['preferences']['score_field']")
EPOCHS=$(_read_yaml "['training']['epochs']")
BATCH_SIZE=$(_read_yaml "['training']['batch_size']")
LR=$(_read_yaml "['training']['lr']")
HIDDEN_DIM=$(_read_yaml "['reward_model']['hidden_dim']")
ARCH=$(_read_yaml "['reward_model']['architecture']")
GAMMA=$(_read_yaml "['mi_teacher']['gamma']")

# defaults if YAML read fails
NUM_CANDIDATES="${NUM_CANDIDATES:-6}"
NUM_FRAMES="${NUM_FRAMES:-8}"
TOP_K="${TOP_K:-10}"
BOTTOM_K="${BOTTOM_K:-10}"
MARGIN="${MARGIN:-0.0}"
SCORE_FIELD="${SCORE_FIELD:-score_delta}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ARCH="${ARCH:-gru}"
GAMMA="${GAMMA:-0.99}"

# ---- Timestamped run dir ----
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="logs/mi_reward/${TIMESTAMP}"
FEATURE_ROOT="$RUN_DIR/features"
PREFERENCES="$RUN_DIR/preferences/train_preferences.jsonl"
OUTPUT_DIR="$RUN_DIR/reward_head"
TASK_SPECS="${TASK_SPECS:-dataset/mi_reward/task_specs.jsonl}"
MANIFEST_DIR="$RUN_DIR/manifests"
MANIFEST="$MANIFEST_DIR/train_manifest.jsonl"
SUCCESS_REFS="$MANIFEST_DIR/success_refs.jsonl"

mkdir -p "$MANIFEST_DIR" "$(dirname "$PREFERENCES")" "$OUTPUT_DIR"

EST_SEC=$((NUM_CANDIDATES * 12 + 30))
EST_MIN=$((EST_SEC / 60))
echo "=============================================="
echo " MI Directional Reward Pipeline"
echo " Config:    $CONFIG"
echo " Run:       $RUN_DIR"
echo " Cosmos:    ${COSMOS_FLAG:-(mock)}"
echo " Candidates: $NUM_CANDIDATES  ×  ${NUM_FRAMES}frames"
echo " Preferences: top_${TOP_K} × bottom_${BOTTOM_K}"
echo " SFT:       ${EPOCHS} epochs, ${HIDDEN_DIM}d ${ARCH}, lr=${LR}"
echo " Est. time: ~${EST_MIN}min (with real Cosmos)"
echo "=============================================="

# ---- Step 1: Cosmos generation ----
echo ""
echo "[1/4] Cosmos-Predict2.5 generation ($NUM_CANDIDATES candidates)..."
python -m mi_reward.data.build_manifest \
    --source_type cosmos \
    --task_specs "$TASK_SPECS" \
    --output_dir "$RUN_DIR" \
    --num_candidates "$NUM_CANDIDATES" \
    --num_frames "$NUM_FRAMES" \
    $COSMOS_FLAG

# ---- Step 2: Feature extraction ----
echo ""
echo "[2/4] DINOv3 feature extraction..."
python -m mi_reward.features.cached_feature_store \
    --manifest "$MANIFEST" \
    --success_refs "$SUCCESS_REFS" \
    --feature_root "$FEATURE_ROOT"

# ---- Step 3: Scoring + preferences ----
echo ""
echo "[3/4] MI scoring + preference building (top=${TOP_K}, bottom=${BOTTOM_K})..."
python -m mi_reward.scoring.build_preferences \
    --manifest "$MANIFEST" \
    --success_refs "$SUCCESS_REFS" \
    --feature_root "$FEATURE_ROOT" \
    --output "$PREFERENCES" \
    --gamma "$GAMMA" \
    --top_k "$TOP_K" \
    --bottom_k "$BOTTOM_K" \
    --margin "$MARGIN"

# ---- Step 4: SFT training ----
echo ""
echo "[4/4] Reward model SFT (${HIDDEN_DIM}d ${ARCH}, ${EPOCHS} epochs)..."
python -m mi_reward.training.train_reward_sft --mode distill \
    --preferences "$PREFERENCES" \
    --feature_root "$FEATURE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --hidden_dim "$HIDDEN_DIM" \
    --architecture "$ARCH" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --gamma "$GAMMA"

echo ""
echo "=============================================="
echo " DONE!"
echo " Checkpoint: $OUTPUT_DIR/pytorch_model.pt"
echo "=============================================="
