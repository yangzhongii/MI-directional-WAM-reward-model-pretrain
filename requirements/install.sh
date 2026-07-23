#!/usr/bin/env bash
# LaWAM + MI Reward Extension — automated environment setup with uv.
#
# Usage:
#   bash requirements/install.sh                    # full install (torch + flash-attn + project)
#   bash requirements/install.sh --no-flash-attn    # skip flash-attn
#   bash requirements/install.sh --franka           # also install Franka realworld ROS deps
#   bash requirements/install.sh --help
#
# Requirements: uv (auto-installed if missing), NVIDIA GPU + CUDA toolkit.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
INSTALL_FLASH_ATTN=1
INSTALL_FRANKA=0
INSTALL_PROJECT=1
NO_ROOT=0

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
print_help() {
    cat <<EOF
Usage: bash requirements/install.sh [options]

Options:
    --venv <dir>           Virtual environment directory (default: .venv).
    --python <version>     Python version (default: 3.10).
    --torch <version>      PyTorch version (default: 2.6.0).
    --no-flash-attn        Skip flash-attn installation.
    --franka               Install Franka realworld ROS dependencies.
    --no-install-project   Skip editable install of the project itself.
    --no-root              Skip system dependency checks.
    -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        --venv)       VENV_DIR="${2:-}"; shift 2 ;;
        --python)     PYTHON_VERSION="${2:-}"; shift 2 ;;
        --torch)      TORCH_VERSION="${2:-}"; shift 2 ;;
        --no-flash-attn)  INSTALL_FLASH_ATTN=0; shift ;;
        --franka)          INSTALL_FRANKA=1; shift ;;
        --no-install-project) INSTALL_PROJECT=0; shift ;;
        --no-root)         NO_ROOT=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

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

# ------------------------------------------------------------------
# Check Python version format
# ------------------------------------------------------------------
if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "[install] ERROR: --python must be of form X.Y (got '$PYTHON_VERSION')." >&2
    exit 1
fi

# ------------------------------------------------------------------
# Create / reuse venv
# ------------------------------------------------------------------
PYPROJECT="$REPO_ROOT/pyproject.toml"

if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    echo "[install] Found existing venv at $VENV_DIR; reusing."
else
    echo "[install] Creating venv: $VENV_DIR (python=$PYTHON_VERSION)..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "[install] Python: $(python --version)"

# ------------------------------------------------------------------
# Install PyTorch
# ------------------------------------------------------------------
echo "[install] Installing PyTorch $TORCH_VERSION..."
uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==0.$(( $(echo "$TORCH_VERSION" | cut -d. -f2) + 15 )).$(echo "$TORCH_VERSION" | cut -d. -f3)" \
    "torchaudio==${TORCH_VERSION}"

# ------------------------------------------------------------------
# Install flash-attn (prebuilt wheel)
# ------------------------------------------------------------------
if [[ "$INSTALL_FLASH_ATTN" -eq 1 ]]; then
    FA_VER="2.7.4.post1"
    CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
    PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
    TORCH_MM="torch$(python -c 'import torch; v=torch.__version__.split("+")[0]; print(v[:3].replace(".",""))')"
    CXX_ABI=$(python -c 'import torch; print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")')
    FA_WHEEL="flash_attn-${FA_VER}+cu${CUDA_MAJOR}${TORCH_MM}${CXX_ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHEEL}"

    echo "[install] Installing flash-attn $FA_VER..."
    if uv pip install "$FA_URL" 2>/dev/null; then
        echo "[install] flash-attn installed from prebuilt wheel."
    else
        echo "[install] Prebuilt wheel unavailable; building from source..."
        uv pip install "flash-attn==${FA_VER}" --no-build-isolation
    fi
fi

# ------------------------------------------------------------------
# Install project dependencies
# ------------------------------------------------------------------
echo "[install] Installing dependencies from requirements.txt..."
uv pip install -r "$SCRIPT_DIR/requirements.txt"

# ------------------------------------------------------------------
# Editable install
# ------------------------------------------------------------------
if [[ "$INSTALL_PROJECT" -eq 1 ]]; then
    echo "[install] Installing project in editable mode..."
    uv pip install -e "$REPO_ROOT" --no-deps
fi

# ------------------------------------------------------------------
# Franka realworld extras
# ------------------------------------------------------------------
if [[ "$INSTALL_FRANKA" -eq 1 ]]; then
    echo "[install] Installing Franka realworld dependencies..."
    uv pip install \
        rospkg \
        filelock \
        psutil

    if [[ "$NO_ROOT" -eq 0 ]]; then
        echo "[install] NOTE: ROS + serl_franka_controllers must be installed separately on the Franka PC."
        echo "       See https://github.com/RLinf/serl_franka_controllers for instructions."
    fi
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "======================================"
echo "  Install complete!"
echo "  Activate: source $VENV_DIR/bin/activate"
echo "======================================"
echo ""
echo "Quick smoke test:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m pytest tests/test_realworld_smoke.py -v"
