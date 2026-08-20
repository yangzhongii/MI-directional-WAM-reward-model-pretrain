#!/usr/bin/env bash
# LaWAM + MI Reward Extension — automated environment setup with uv.
#
# Usage:
#   bash requirements/install.sh --all             # shared .venv: all data/eval tools
#   bash requirements/install.sh --generalization-data  # .venv: SAM3/Cosmos/MuJoCo data generation
#   bash requirements/install.sh --reward-eval     # .venv: Robometer/RBM-EVAL adapter
#   bash requirements/install.sh --mi-cosmos      # Training env: cosmos + DINO + MI reward (8×A800)
#   bash requirements/install.sh --rlpd           # Inference env: DINOv3 + RewardHead + Ray (NUC + 4090)
#   bash requirements/install.sh --help
#
# The generalization-data environment is the shared `.venv` for Stage 1 and Stage 2.
# `.venv-mi` and `.venv-rlpd` remain legacy isolated environments.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
VENV_DIR=""
PYTHON_VERSION=""
TARGET=""
INSTALL_ALL=0
USE_MIRRORS=0
INSTALL_FLASH_ATTN=1
INSTALL_PROJECT=1
COSMOS_GIT_REF="${COSMOS_GIT_REF:-a2c298b0a3df3778b973fe65e9e58877b292d8a7}"
COSMOS_TRANSFER_GIT_REF="${COSMOS_TRANSFER_GIT_REF:-2ff49d0717af02057ae79bc75c00fbff9da1b4e7}"
SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
SAM3_GIT_REF="${SAM3_GIT_REF:-8f0b7f4d4e7eda2ed606ebde6702c93359ad01da}"
SAM3_MODEL_ID="${SAM3_MODEL_ID:-facebook/sam3}"
COSMOS_PREDICT_MODEL_ID="${COSMOS_PREDICT_MODEL_ID:-nvidia/Cosmos-Predict2.5-2B}"
COSMOS_TRANSFER_MODEL_ID="${COSMOS_TRANSFER_MODEL_ID:-nvidia/Cosmos-Transfer2.5-2B}"
DINO_MODEL_ID="${DINO_MODEL_ID:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
LAM_MODEL_ID="${LAM_MODEL_ID:-jialei02/lawam_lam}"
ROBOMETER_REPO_URL="${ROBOMETER_REPO_URL:-https://github.com/robometer/robometer.git}"
ROBOMETER_GIT_REF="${ROBOMETER_GIT_REF:-352d160389daa964788de1ec933d1925f3a6de4f}"
DOWNLOAD_WEIGHTS=0
DOWNLOAD_EVAL_DATA=0
GITHUB_PREFIX=""
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# ------------------------------------------------------------------
# Mirror helpers
# ------------------------------------------------------------------
setup_mirror() {
    if [ "$USE_MIRRORS" -eq 1 ]; then
        export UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download
        export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
        export HF_ENDPOINT=https://hf-mirror.com
        export GITHUB_PREFIX="https://ghfast.top/"
        git config --global url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/"
        trap 'unset_mirror' EXIT INT TERM HUP
    fi
}

unset_mirror() {
    if [ "$USE_MIRRORS" -eq 1 ]; then
        unset UV_PYTHON_INSTALL_MIRROR
        unset UV_DEFAULT_INDEX
        unset HF_ENDPOINT
        git config --global --unset url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/" || true
        unset GITHUB_PREFIX
    fi
}

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
print_help() {
    cat <<EOF
Usage: bash requirements/install.sh <target> [options]

Targets (mutually exclusive):
    --all                  Install all data-generation and RBM-EVAL tools into .venv.
    --generalization-data  Data-generation environment with SAM3, Cosmos, and MuJoCo.
                           Creates .venv and stores source/checkpoints under .venv/.
    --reward-eval          RBM-EVAL environment in the shared .venv.
                           Clones the pinned Robometer source under .venv/src/.
    --mi-cosmos            Training environment: cosmos + DINOv3 + MI reward pipeline.
                           Creates .venv-mi (Python 3.10, CUDA 12.8, for 8xA800).
    --rlpd                 Inference environment: DINOv3 + RewardHead + Ray.
                           Creates .venv-rlpd (Python 3.10, for NUC + 4090).

Options:
    --use-mirrors          Use mirrors (aliyun/hf-mirror/ghfast) for faster downloads.
    --no-flash-attn        Skip flash-attn (--mi-cosmos only).
    --no-install-project   Skip editable install of the project itself.
    --cosmos-ref <ref>     Cosmos-Predict2.5 git ref (--generalization-data/--mi-cosmos, default: pinned).
    --transfer-ref <ref>   Cosmos-Transfer2.5 git ref (--generalization-data only, default: pinned).
    --sam3-repo <url>      SAM3 git URL (--generalization-data only).
    --download-weights     Download SAM3, Cosmos Predict/Transfer, DINOv3, and LAM weights.
                           Requires Hugging Face access and uses *_MODEL_ID overrides.
    --download-eval-data   Download Robometer processed evaluation datasets into .venv/datasets/robometer.
                           Requires Hugging Face access; can also be run later with the same target.
    PYTORCH_INDEX_URL      Environment override for the CUDA PyTorch wheel index (default: cu128).
    -h, --help             Show this help.
EOF
}

TARGET="${1:-}"
shift || true

if [[ "$TARGET" = "-h" || "$TARGET" = "--help" ]]; then
    print_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)          print_help; exit 0 ;;
        --use-mirrors)       USE_MIRRORS=1; shift ;;
        --no-flash-attn)    INSTALL_FLASH_ATTN=0; shift ;;
        --no-install-project) INSTALL_PROJECT=0; shift ;;
        --cosmos-ref)       COSMOS_GIT_REF="${2:-}"; shift 2 ;;
        --transfer-ref)     COSMOS_TRANSFER_GIT_REF="${2:-}"; shift 2 ;;
        --sam3-repo)        SAM3_REPO_URL="${2:-}"; shift 2 ;;
        --download-weights) DOWNLOAD_WEIGHTS=1; shift ;;
        --download-eval-data) DOWNLOAD_EVAL_DATA=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$TARGET" in
    --all)
        INSTALL_ALL=1
        TARGET="--generalization-data"
        VENV_DIR=".venv"
        PYTHON_VERSION="3.10"
        ;;
    --generalization-data)
        VENV_DIR="${VENV_DIR:-.venv}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    --reward-eval)
        VENV_DIR="${VENV_DIR:-.venv}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    --mi-cosmos)
        VENV_DIR="${VENV_DIR:-.venv-mi}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    --rlpd)
        VENV_DIR="${VENV_DIR:-.venv-rlpd}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    "")
        print_help
        exit 1
        ;;
    *)
        echo "Unknown target: $TARGET (use --all, --generalization-data, --reward-eval, --mi-cosmos, or --rlpd)" >&2
        exit 1
        ;;
esac

setup_mirror

# ------------------------------------------------------------------
# Ensure uv
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[install] Installing uv..."
    if command -v pip &>/dev/null; then
        pip install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "[install] uv version: $(uv --version)"
echo "[install] Target: $TARGET  |  venv: $VENV_DIR  |  python: $PYTHON_VERSION"

# ==================================================================
# Generalization data environment: --generalization-data
# ==================================================================
if [ "$TARGET" = "--generalization-data" ]; then
    COSMOS_DIR="$VENV_DIR/src/cosmos-predict2.5"
    TRANSFER_DIR="$VENV_DIR/src/cosmos-transfer2.5"
    SAM3_DIR="$VENV_DIR/src/sam3"
    MODEL_DIR="$VENV_DIR/models"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$VENV_DIR/src" "$MODEL_DIR"

    if [ ! -d "$COSMOS_DIR/.git" ]; then
        echo "[install] Cloning Cosmos-Predict2.5 into $COSMOS_DIR..."
        git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_DIR"
        git -C "$COSMOS_DIR" checkout "$COSMOS_GIT_REF"
    fi
    if [ ! -d "$TRANSFER_DIR/.git" ]; then
        echo "[install] Cloning Cosmos-Transfer2.5 into $TRANSFER_DIR..."
        git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$TRANSFER_DIR"
        git -C "$TRANSFER_DIR" checkout "$COSMOS_TRANSFER_GIT_REF"
    fi
    if [ ! -d "$SAM3_DIR/.git" ]; then
        echo "[install] Cloning SAM3 into $SAM3_DIR..."
        git clone "$SAM3_REPO_URL" "$SAM3_DIR"
    fi
    if [ -n "$SAM3_GIT_REF" ]; then
        git -C "$SAM3_DIR" checkout "$SAM3_GIT_REF"
    fi

    echo "[install] Installing the generalization-data Python environment..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url "$PYTORCH_INDEX_URL"
    uv pip install \
        mujoco==3.3.2 \
        transformers==5.2.0 \
        numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf einops tqdm rich pytest \
        accelerate safetensors imageio imageio-ffmpeg matplotlib pandas \
        huggingface_hub[cli] datasets mediapy tyro loguru \
        moderngl shapely OpenEXR cattrs natsort
    uv pip install -e "$SAM3_DIR"
    uv pip install -e "$COSMOS_DIR"
    uv pip install -e "$COSMOS_DIR/packages/cosmos-oss[cu128_torch27]"
    uv pip install -e "$TRANSFER_DIR[cu128]"
    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        uv pip install -e "$REPO_ROOT" --no-deps
    fi

    if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
        echo "[install] Downloading model weights into $MODEL_DIR..."
        hf download "$SAM3_MODEL_ID" --local-dir "$MODEL_DIR/sam3"
        hf download "$COSMOS_PREDICT_MODEL_ID" --local-dir "$MODEL_DIR/cosmos-predict2.5"
        hf download "$COSMOS_TRANSFER_MODEL_ID" --local-dir "$MODEL_DIR/cosmos-transfer2.5"
        hf download "$DINO_MODEL_ID" --local-dir "$MODEL_DIR/dinov3-vitb16-pretrain-lvd1689m"
        hf download "$LAM_MODEL_ID" --local-dir "$MODEL_DIR/lawam_lam"
    else
        echo "[install] Weights not downloaded. Re-run with --download-weights after HF login."
    fi

    echo "[install] Data environment ready: source $VENV_DIR/bin/activate"
    echo "[install] MuJoCo headless default: MUJOCO_GL=egl"

# ==================================================================
# RBM-EVAL environment: --reward-eval
# ==================================================================
elif [ "$TARGET" = "--reward-eval" ]; then
    ROBOMETER_DIR="$VENV_DIR/src/robometer"
    EVAL_DATA_DIR="$VENV_DIR/datasets/robometer"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$VENV_DIR/src" "$EVAL_DATA_DIR"

    if [ ! -d "$ROBOMETER_DIR/.git" ]; then
        echo "[install] Cloning Robometer/RBM-EVAL into $ROBOMETER_DIR..."
        git clone "$ROBOMETER_REPO_URL" "$ROBOMETER_DIR"
    fi
    # Robometer's full dependency set pins incompatible Torch/CUDA versions.
    git -C "$ROBOMETER_DIR" fetch --quiet origin "$ROBOMETER_GIT_REF" || true
    git -C "$ROBOMETER_DIR" checkout --quiet --detach "$ROBOMETER_GIT_REF"

    echo "[install] Installing RBM-EVAL adapter dependencies (without Robometer extras)..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url "$PYTORCH_INDEX_URL"
    uv pip install \
        transformers==5.2.0 numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf einops tqdm rich matplotlib imageio \
        huggingface_hub[cli] hatchling scikit-learn seaborn h5py \
        pydantic datasets hydra-core loguru termcolor codetiming \
        wandb tensorboard sentence-transformers decord mediapy tyro loguru
    uv pip install -e "$ROBOMETER_DIR" --no-deps
    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        uv pip install -e "$REPO_ROOT" --no-deps
    fi

    if [ "$DOWNLOAD_EVAL_DATA" -eq 1 ]; then
        echo "[install] Downloading Robometer processed datasets into $EVAL_DATA_DIR..."
        hf download robometer/processed_datasets --repo-type dataset --local-dir "$EVAL_DATA_DIR"
        echo "[install] Extracting Robometer processed dataset archives..."
        ROBOMETER_PROCESSED_DATASETS_PATH="$EVAL_DATA_DIR" \
            bash "$ROBOMETER_DIR/scripts/untar_processed_datasets.sh"
    else
        echo "[install] Evaluation data not downloaded. Re-run with --download-eval-data after HF login."
    fi
    echo "[install] RBM-EVAL source commit: $(git -C "$ROBOMETER_DIR" rev-parse HEAD)"
    echo "[install] RBM-EVAL environment ready: source $VENV_DIR/bin/activate"

# ==================================================================
# Training env: --mi-cosmos
# ==================================================================
elif [ "$TARGET" = "--mi-cosmos" ]; then
    COSMOS_DIR="$REPO_ROOT/.venv/cosmos-predict2.5"

    # ---- git clone cosmos (no venv needed) ----
    if [ ! -d "$COSMOS_DIR/.git" ]; then
        echo "[install] Cloning Cosmos-Predict2.5 (commit $COSMOS_GIT_REF)..."
        mkdir -p "$(dirname "$COSMOS_DIR")"
        git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_DIR"
        git -C "$COSMOS_DIR" checkout "$COSMOS_GIT_REF"
    else
        echo "[install] Cosmos-Predict2.5 already cloned at $COSMOS_DIR"
    fi

    # ---- create venv ----
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"

    # ---- torch 2.7 for CUDA 12.8 (Cosmos requirement) ----
    echo "[install] Installing PyTorch 2.7.0 for CUDA 12.8..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url "$PYTORCH_INDEX_URL"

    # ---- flash-attn ----
    if [ "$INSTALL_FLASH_ATTN" -eq 1 ]; then
        FA_VER="2.7.3"
        echo "[install] Installing flash-attn $FA_VER (prebuilt wheel)..."
        CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
        PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
        TORCH_MM="torch$(python -c 'import torch; v=torch.__version__.split("+")[0]; print(v[:3].replace(".",""))')"
        CXX_ABI=$(python -c 'import torch; print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")')
        FA_WHEEL="flash_attn-${FA_VER}+cu${CUDA_MAJOR}${TORCH_MM}${CXX_ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
        uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHEEL}" 2>/dev/null \
            || uv pip install "flash-attn==${FA_VER}" --no-build-isolation \
            || echo "[install] WARNING: flash-attn install failed (non-fatal)"
    fi

    # ---- core deps (minimal subset of requirements.txt) ----
    echo "[install] Installing core training dependencies..."
    uv pip install \
        transformers==5.2.0 \
        numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf einops tqdm rich pytest \
        lightning accelerate wandb tensorboard \
        huggingface_hub[cli] datasets safetensors \
        imageio matplotlib pandas

    # ---- cosmos ----
    echo "[install] Installing cosmos-predict2.5 + cu128 extras..."
    uv pip install -e "$COSMOS_DIR"
    uv pip install -e "$COSMOS_DIR/packages/cosmos-oss[cu128_torch27]"

    # ---- project ----
    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        echo "[install] Installing project in editable mode..."
        uv pip install -e "$REPO_ROOT" --no-deps
    fi

    echo "[install] Cosmos-Predict2.5: $(python -c 'import cosmos_predict2; print(cosmos_predict2.__file__)' 2>/dev/null || echo 'import OK (check failed)')"
    echo ""
    echo "[install] =========================================="
    echo "[install]  Weight download (run manually after HF login):"
    echo "[install]    hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir weights/dinov3-vitb16-pretrain-lvd1689m"
    echo "[install]    hf download nvidia/Cosmos-Predict2.5-2B --local-dir weights/cosmos-predict2.5"
    echo "[install] =========================================="

# ==================================================================
# Inference env: --rlpd (NUC + 4090)
# ==================================================================
elif [ "$TARGET" = "--rlpd" ]; then
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"

    echo "[install] Installing torch + DINO + reward model deps..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url "$PYTORCH_INDEX_URL"
    uv pip install \
        transformers numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf einops tqdm rich pytest \
        safetensors matplotlib pandas \
        ray

    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        uv pip install -e "$REPO_ROOT" --no-deps
    fi

    echo ""
    echo "[install] =========================================="
    echo "[install]  RLPD inference env ready."
    echo "[install]  Weights needed: DINOv3 at weights/dinov3-vitb16-pretrain-lvd1689m/"
    echo "[install]  Reward checkpoint: copy pytorch_model.pt from training."
    echo "[install]  Deploy on NUC + 4090 (shared .venv-rlpd)."
    echo "[install] =========================================="
fi

# --all is intentionally implemented as two idempotent passes so both source
# trees and their dependencies share the same .venv without maintaining a
# second dependency resolver branch here.
if [ "$INSTALL_ALL" -eq 1 ]; then
    NEXT_ARGS=(--reward-eval)
    if [ "$DOWNLOAD_EVAL_DATA" -eq 1 ]; then
        NEXT_ARGS+=(--download-eval-data)
    fi
    bash "$SCRIPT_DIR/install.sh" "${NEXT_ARGS[@]}"
    exit 0
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "======================================"
echo "  Install complete!"
echo "  Target : $TARGET"
echo "  Venv   : $VENV_DIR"
echo "  Activate: source $VENV_DIR/bin/activate"
echo "======================================"
