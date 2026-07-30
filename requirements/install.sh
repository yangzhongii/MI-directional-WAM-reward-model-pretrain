#!/usr/bin/env bash
# LaWAM + MI Reward Extension — automated environment setup with uv.
#
# Usage:
#   bash requirements/install.sh --mi-cosmos      # Training env: cosmos + DINO + MI reward (8×A800)
#   bash requirements/install.sh --rlpd           # Inference env: DINOv3 + RewardHead + Ray (NUC + 4090)
#   bash requirements/install.sh --help
#
# The two environments are independent (different venvs, different deps).
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
USE_MIRRORS=0
INSTALL_FLASH_ATTN=1
INSTALL_PROJECT=1
COSMOS_GIT_REF="${COSMOS_GIT_REF:-a2c298b0a3df3778b973fe65e9e58877b292d8a7}"
GITHUB_PREFIX=""

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
    --mi-cosmos            Training environment: cosmos + DINOv3 + MI reward pipeline.
                           Creates .venv-mi (Python 3.10, CUDA 12.8, for 8xA800).
    --rlpd                 Inference environment: DINOv3 + RewardHead + Ray.
                           Creates .venv-rlpd (Python 3.10, for NUC + 4090).

Options:
    --use-mirrors          Use mirrors (aliyun/hf-mirror/ghfast) for faster downloads.
    --no-flash-attn        Skip flash-attn (--mi-cosmos only).
    --no-install-project   Skip editable install of the project itself.
    --cosmos-ref <ref>     Cosmos-Predict2.5 git ref (--mi-cosmos only, default: pinned).
    -h, --help             Show this help.
EOF
}

TARGET="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)          print_help; exit 0 ;;
        --use-mirrors)       USE_MIRRORS=1; shift ;;
        --no-flash-attn)    INSTALL_FLASH_ATTN=0; shift ;;
        --no-install-project) INSTALL_PROJECT=0; shift ;;
        --cosmos-ref)       COSMOS_GIT_REF="${2:-}"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$TARGET" in
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
        echo "Unknown target: $TARGET (use --mi-cosmos or --rlpd)" >&2
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
# Training env: --mi-cosmos
# ==================================================================
if [ "$TARGET" = "--mi-cosmos" ]; then
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
    uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

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
        pyyaml omegaconf einops tqdm rich \
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
    uv pip install torch==2.7.0 torchvision==0.22.0
    uv pip install \
        transformers numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf einops tqdm rich \
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
