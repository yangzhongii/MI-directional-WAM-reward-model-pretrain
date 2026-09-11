#!/usr/bin/env bash
# LaWAM + MI Reward Extension — automated environment setup with uv.
#
# Usage:
#   bash requirements/install.sh --all             # install all isolated runtime environments
#   bash requirements/install.sh --generalization-data  # .venv: SAM2.1/Cosmos/MuJoCo data generation
#   bash requirements/install.sh --reward-train    # .venv-reward: DINOv3/LaWAM/reward training
#   bash requirements/install.sh --reward-eval     # .venv-eval: Robometer/RBM-EVAL adapter
#   bash requirements/install.sh --libero-closed-loop # .venv-libero: LIBERO + frozen MI reward + SAC
#   bash requirements/install.sh --mi-cosmos      # Legacy isolated Cosmos training env (8×A800)
#   bash requirements/install.sh --rlpd           # Inference env: DINOv3 + RewardHead + Ray (NUC + 4090)
#   bash requirements/install.sh --help
#
# Python environments are isolated because Cosmos pins transformers==4.51.3,
# while DINOv3/LaWAM require a newer Transformers release. Large sources,
# checkpoints, and datasets remain shared below `.venv/{src,models,datasets}`.
# `.venv-mi` and `.venv-rlpd` remain legacy isolated environments.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
VENV_DIR=""
ARTIFACT_ROOT="${ARTIFACT_ROOT:-.venv}"
PYTHON_VERSION=""
TARGET=""
INSTALL_ALL=0
USE_MIRRORS=0
INSTALL_FLASH_ATTN=1
INSTALL_PROJECT=1
COSMOS_GIT_REF="${COSMOS_GIT_REF:-a2c298b0a3df3778b973fe65e9e58877b292d8a7}"
COSMOS_TRANSFER_GIT_REF="${COSMOS_TRANSFER_GIT_REF:-2ff49d0717af02057ae79bc75c00fbff9da1b4e7}"
SAM_BACKEND="${SAM_BACKEND:-sam2}"
SAM2_SIZE="${SAM2_SIZE:-base-plus}"
SAM2_REPO_URL="${SAM2_REPO_URL:-https://github.com/facebookresearch/sam2.git}"
SAM2_GIT_REF="${SAM2_GIT_REF:-main}"
SAM2_CHECKPOINT_BASE_URL="${SAM2_CHECKPOINT_BASE_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824}"
SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
SAM3_GIT_REF="${SAM3_GIT_REF:-8f0b7f4d4e7eda2ed606ebde6702c93359ad01da}"
SAM3_MODEL_ID="${SAM3_MODEL_ID:-facebook/sam3}"
COSMOS_PREDICT_MODEL_ID="${COSMOS_PREDICT_MODEL_ID:-nvidia/Cosmos-Predict2.5-2B}"
COSMOS_TRANSFER_MODEL_ID="${COSMOS_TRANSFER_MODEL_ID:-nvidia/Cosmos-Transfer2.5-2B}"
DINO_MODEL_ID="${DINO_MODEL_ID:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
LAM_MODEL_ID="${LAM_MODEL_ID:-jialei02/lawam_lam}"
ROBOMETER_REPO_URL="${ROBOMETER_REPO_URL:-https://github.com/robometer/robometer.git}"
ROBOMETER_GIT_REF="${ROBOMETER_GIT_REF:-352d160389daa964788de1ec933d1925f3a6de4f}"
LIBERO_REPO_URL="${LIBERO_REPO_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_GIT_REF="${LIBERO_GIT_REF:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
LIBERO_ARCHIVE_URL="${LIBERO_ARCHIVE_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO/archive/${LIBERO_GIT_REF}.tar.gz}"
RESNET18_CHECKPOINT_URL="${RESNET18_CHECKPOINT_URL:-https://download.pytorch.org/models/resnet18-f37072fd.pth}"
DOWNLOAD_WEIGHTS=0
DOWNLOAD_TRANSFER_WEIGHTS=0
DOWNLOAD_SIM_ASSETS=0
INSTALL_TRANSFER=0
DOWNLOAD_EVAL_DATA=0
DOWNLOAD_ALL_EVAL_DATA=0
DOWNLOAD_LIBERO_DATA=0
EVAL_DATASETS=()
LIBERO_DATASETS=()
GENERALIZATION_DATA_CONFIG="${GENERALIZATION_DATA_CONFIG:-mi_reward/configs/generalization_data.yaml}"
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-1}"
HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-5}"
GITHUB_PREFIX=""
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

configure_segmentation_backend() {
    case "$SAM_BACKEND" in
        sam2)
            case "$SAM2_SIZE" in
                tiny)
                    SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"
                    SAM2_CHECKPOINT_NAME="sam2.1_hiera_tiny.pt"
                    ;;
                small)
                    SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_s.yaml"
                    SAM2_CHECKPOINT_NAME="sam2.1_hiera_small.pt"
                    ;;
                base-plus|base_plus)
                    SAM2_SIZE="base-plus"
                    SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_b+.yaml"
                    SAM2_CHECKPOINT_NAME="sam2.1_hiera_base_plus.pt"
                    ;;
                large)
                    SAM2_MODEL_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
                    SAM2_CHECKPOINT_NAME="sam2.1_hiera_large.pt"
                    ;;
                *)
                    echo "Unsupported --sam2-size '$SAM2_SIZE' (use tiny, small, base-plus, or large)." >&2
                    exit 1
                    ;;
            esac
            ;;
        sam3|none)
            SAM2_MODEL_CONFIG=""
            SAM2_CHECKPOINT_NAME=""
            ;;
        *)
            echo "Unsupported --sam-backend '$SAM_BACKEND' (use sam2, sam3, or none)." >&2
            exit 1
            ;;
    esac
}

download_public_file_with_retry() {
    local url="$1"
    local target="$2"
    local partial="${target}.incomplete"
    local attempt

    mkdir -p "$(dirname "$target")"
    if [ -s "$target" ]; then
        echo "[install] Public checkpoint already present: $target"
        return 0
    fi
    for ((attempt = 1; attempt <= HF_DOWNLOAD_RETRIES; attempt++)); do
        echo "[install] Downloading $url (attempt $attempt/$HF_DOWNLOAD_RETRIES)..."
        if curl --fail --location --continue-at - \
            --connect-timeout 30 --max-time 0 \
            --retry 3 --retry-delay 5 --retry-all-errors \
            --output "$partial" "$url"; then
            mv "$partial" "$target"
            return 0
        fi
        if [ "$attempt" -lt "$HF_DOWNLOAD_RETRIES" ]; then
            echo "[install] Public checkpoint download interrupted; the next attempt will resume." >&2
            sleep 10
        fi
    done
    echo "[install] ERROR: failed to download $url after $HF_DOWNLOAD_RETRIES attempts." >&2
    echo "[install] Partial data was kept at $partial for the next run." >&2
    return 1
}

write_segmentation_config() {
    local model_dir="$1"
    local checkpoint_root="$model_dir/$SAM_BACKEND"
    local config_path="$model_dir/segmentation.json"

    if [ "$SAM_BACKEND" = "none" ]; then
        checkpoint_root="$model_dir/none"
    fi
    python - "$config_path" "$SAM_BACKEND" "$checkpoint_root" "$SAM2_MODEL_CONFIG" "$SAM2_SIZE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "backend": sys.argv[2],
    "checkpoint_root": sys.argv[3],
    "model_config": sys.argv[4],
    "sam2_size": sys.argv[5],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    echo "[install] Segmentation config: $config_path"
    if [ "$SAM_BACKEND" = "sam2" ]; then
        echo "[install] Segmentation backend confirmed: SAM 2.1 $SAM2_SIZE"
    else
        echo "[install] Segmentation backend confirmed: $SAM_BACKEND"
    fi
}

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
        if [ -n "${GITHUB_PREFIX:-}" ]; then
            git config --global --unset url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/" || true
        fi
        unset GITHUB_PREFIX
    fi
}

# Gated model repositories must be queried through the official endpoint so
# Hugging Face can validate the saved token and accepted model license.
# --use-mirrors still applies to Python, GitHub, and dataset downloads;
# HF_MODEL_ENDPOINT remains available as an explicit model-endpoint override.
hf_download_model() {
    local model_id="$1"
    local local_dir="$2"
    local endpoint="${HF_MODEL_ENDPOINT:-https://huggingface.co}"

    echo "[install] Downloading $model_id from $endpoint into $local_dir..."
    (
        export HF_ENDPOINT="$endpoint"
        hf_download_with_retry \
            "$model_id" \
            --local-dir "$local_dir" \
            --max-workers "$HF_DOWNLOAD_MAX_WORKERS"
    )
}

# Public Robometer archives are large enough that Xet reconstruction and
# parallel HTTPS streams are fragile behind common mainland-China proxies.
# Keep retries local to the download step so an interrupted install resumes
# the existing Hugging Face local-dir cache instead of reinstalling packages.
hf_download_with_retry() {
    local attempt
    local error_log
    local repo_id="${1:-unknown}"
    error_log="$(mktemp "${TMPDIR:-/tmp}/mi-reward-hf-download.XXXXXX")"

    for ((attempt = 1; attempt <= HF_DOWNLOAD_RETRIES; attempt++)); do
        if HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
            HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}" \
            HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
            hf download "$@" 2>&1 | tee "$error_log"; then
            rm -f "$error_log"
            return 0
        fi
        if grep -Eiq \
            'GatedRepoError|401 Client Error|403 Client Error|RepositoryNotFoundError|Invalid user token|access .* (denied|rejected)' \
            "$error_log"; then
            echo "[install] ERROR: Hugging Face could not authorize or locate '$repo_id'; this is not retryable." >&2
            echo "[install] Open https://huggingface.co/$repo_id, accept its license, then refresh the CLI token." >&2
            echo "[install] Model endpoint: ${HF_ENDPOINT:-https://huggingface.co}" >&2
            rm -f "$error_log"
            return 1
        fi
        if [ "$attempt" -lt "$HF_DOWNLOAD_RETRIES" ]; then
            echo "[install] Hugging Face download failed (attempt $attempt/$HF_DOWNLOAD_RETRIES); retrying in 10 seconds..." >&2
            sleep 10
        fi
    done
    rm -f "$error_log"
    echo "[install] ERROR: Hugging Face download failed after $HF_DOWNLOAD_RETRIES attempts." >&2
    return 1
}

configured_eval_datasets() {
    python - "$GENERALIZATION_DATA_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

datasets = config.get("base_data", {}).get("robometer", {}).get("datasets", [])
if not isinstance(datasets, list):
    raise SystemExit(f"base_data.robometer.datasets must be a list in {config_path}")
for dataset in datasets:
    if str(dataset).strip():
        print(str(dataset).strip())
PY
}

robometer_dataset_is_ready() {
    local dataset_dir="$1"
    [ -d "$dataset_dir/processed_dataset" ] || \
        { [ -f "$dataset_dir/dataset_info.json" ] && [ -f "$dataset_dir/state.json" ]; }
}

ensure_robometer_compat_path() {
    local data_dir="$1"
    local dataset_name="$2"
    local dataset_dir="$data_dir/$dataset_name"
    local compat_root="$data_dir/processed_datasets"
    local compat_dir="$compat_root/$dataset_name"
    local expected_target
    local current_target

    if ! robometer_dataset_is_ready "$dataset_dir"; then
        return 0
    fi

    mkdir -p "$compat_root"
    expected_target="$(realpath "$dataset_dir")"
    if [ -L "$compat_dir" ]; then
        current_target="$(readlink -f "$compat_dir" || true)"
        if [ "$current_target" != "$expected_target" ]; then
            echo "[install] ERROR: Robometer compatibility link points to the wrong dataset: $compat_dir" >&2
            return 1
        fi
        return 0
    fi
    if [ -e "$compat_dir" ]; then
        echo "[install] ERROR: Robometer compatibility path already exists and is not a link: $compat_dir" >&2
        return 1
    fi

    # Processed Arrow rows store frame paths as
    # ./processed_datasets/<dataset>/frames/*.npz, while BaseDataset discovers
    # dataset indices directly below ROBOMETER_PROCESSED_DATASETS_PATH. Keep
    # one physical copy and expose both layouts.
    ln -s "../$dataset_name" "$compat_dir"
    echo "[install] Robometer compatibility path ready: $compat_dir"
}

ensure_all_robometer_compat_paths() {
    local data_dir="$1"
    local dataset_dir
    local dataset_name

    shopt -s nullglob
    for dataset_dir in "$data_dir"/*; do
        [ -d "$dataset_dir" ] || continue
        dataset_name="$(basename "$dataset_dir")"
        [ "$dataset_name" = "processed_datasets" ] && continue
        ensure_robometer_compat_path "$data_dir" "$dataset_name"
    done
    shopt -u nullglob
}

extract_robometer_dataset() {
    local data_dir="$1"
    local dataset_name="$2"
    local dataset_dir="$data_dir/$dataset_name"
    local nested_dir="$data_dir/processed_datasets/$dataset_name"
    local archive="$data_dir/$dataset_name.tar"
    local -a parts=()

    if robometer_dataset_is_ready "$dataset_dir"; then
        echo "[install] Robometer dataset already extracted: $dataset_name"
        ensure_robometer_compat_path "$data_dir" "$dataset_name"
        return 0
    fi

    echo "[install] Extracting selected Robometer dataset: $dataset_name"
    if [ -f "$archive" ]; then
        tar -xf "$archive" -C "$data_dir"
    else
        shopt -s nullglob
        parts=("$archive".part-*)
        if [ "${#parts[@]}" -eq 0 ]; then
            parts=("$archive".part*)
        fi
        shopt -u nullglob
        if [ "${#parts[@]}" -eq 0 ]; then
            echo "[install] ERROR: no archive found for Robometer dataset '$dataset_name'." >&2
            return 1
        fi
        cat "${parts[@]}" | tar -xf - -C "$data_dir"
    fi

    if [ -d "$nested_dir" ]; then
        if [ -e "$dataset_dir" ]; then
            echo "[install] ERROR: both '$dataset_dir' and '$nested_dir' exist; refusing to overwrite either." >&2
            return 1
        fi
        mv "$nested_dir" "$dataset_dir"
    fi

    if ! robometer_dataset_is_ready "$dataset_dir"; then
        echo "[install] ERROR: extracted dataset has an unexpected layout: $dataset_dir" >&2
        return 1
    fi
    ensure_robometer_compat_path "$data_dir" "$dataset_name"
    echo "[install] Robometer dataset ready: $dataset_dir"
}

install_reward_runtime() {
    echo "[install] Installing isolated DINOv3/LaWAM reward runtime..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url "$PYTORCH_INDEX_URL"
    uv pip install \
        transformers==5.2.0 \
        lightning==2.4.0 jsonargparse==4.27.7 \
        numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf hydra-core einops timm tqdm rich pytest \
        accelerate safetensors imageio imageio-ffmpeg matplotlib pandas \
        huggingface_hub[cli] datasets av wandb tensorboard
    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        uv pip install -e "$REPO_ROOT" --no-deps
    fi
}

smoke_generalization_environment() {
    python - "$SAM_BACKEND" <<'PY'
import importlib
import importlib.metadata
import sys

import mujoco
import torch

backend = sys.argv[1]
version = importlib.metadata.version("transformers")
if version != "4.51.3":
    raise RuntimeError(f"Cosmos requires transformers==4.51.3, found {version}")
importlib.import_module("cosmos_predict2._src.predict2.action.inference.inference_pipeline")
if backend == "sam2":
    importlib.import_module("sam2")
elif backend == "sam3":
    importlib.import_module("sam3")
print(
    "[install] Data-generation smoke test passed:",
    f"torch={torch.__version__}",
    f"transformers={version}",
    f"mujoco={mujoco.__version__}",
)
PY
}

smoke_reward_environment() {
    local model_dir="$1"
    python - "$model_dir" <<'PY'
import importlib
import importlib.metadata
import sys
from pathlib import Path

import lightning
import torch
from transformers import AutoConfig

model_dir = Path(sys.argv[1])
transformers_version = importlib.metadata.version("transformers")
importlib.import_module("latent_action_model.core.lam_model")
dino_root = model_dir / "dinov3-vitb16-pretrain-lvd1689m"
if (dino_root / "config.json").is_file():
    config = AutoConfig.from_pretrained(
        str(dino_root), trust_remote_code=True, local_files_only=True
    )
    if config.model_type != "dinov3_vit":
        raise RuntimeError(f"Unexpected DINOv3 model type: {config.model_type}")
print(
    "[install] Reward smoke test passed:",
    f"torch={torch.__version__}",
    f"transformers={transformers_version}",
    f"lightning={lightning.__version__}",
)
PY
}

smoke_robometer_environment() {
    local robometer_dir="$1"
    local data_dir="$2"
    python - "$robometer_dir" "$data_dir" <<'PY'
import os
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
processed_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source_root))
os.environ["ROBOMETER_PROCESSED_DATASETS_PATH"] = str(processed_root)

from robometer.configs.experiment_configs import DataConfig  # noqa: F401
from robometer.data.datasets.custom_eval import CustomEvalDataset  # noqa: F401
from robometer.evals.compile_results import run_policy_ranking_eval  # noqa: F401

print("[install] RBM-EVAL import smoke test passed")
PY
}

# ------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------
print_help() {
    cat <<EOF
Usage: bash requirements/install.sh <target> [options]

Targets (mutually exclusive):
    --all                  Install all four isolated runtime environments.
    --generalization-data  Data-generation environment with SAM2.1, Cosmos, and MuJoCo.
                           Creates .venv; shared sources/checkpoints remain under .venv/.
    --reward-train         Reward environment with DINOv3, LaWAM LAM, and Lightning.
                           Creates .venv-reward and reuses checkpoints under .venv/models/.
    --reward-eval          RBM-EVAL environment isolated from Cosmos.
                           Creates .venv-eval and reuses .venv/src and .venv/datasets.
    --libero-closed-loop   LIBERO + frozen MI potential + visual RLPD/SAC environment.
                           Creates .venv-libero and reuses reward weights under .venv/models/.
    --mi-cosmos            Legacy Cosmos training environment.
                           Creates .venv-mi (Python 3.10, CUDA 12.8, for 8xA800).
    --rlpd                 Inference environment: DINOv3 + RewardHead + Ray.
                           Creates .venv-rlpd (Python 3.10, for NUC + 4090).

Options:
    --use-mirrors          Use mirrors (aliyun/hf-mirror/ghfast) for faster downloads.
    --no-flash-attn        Skip flash-attn (--mi-cosmos only).
    --no-install-project   Skip editable install of the project itself.
    --cosmos-ref <ref>     Cosmos-Predict2.5 git ref (--generalization-data/--mi-cosmos, default: pinned).
    --transfer-ref <ref>   Cosmos-Transfer2.5 git ref (--generalization-data only, default: pinned).
    --with-transfer        Install optional Cosmos Transfer source/dependencies without weights.
    --sam-backend <name>   Segmentation backend: sam2, sam3, or none (default: sam2).
    --sam2-size <size>     SAM2.1 checkpoint: tiny, small, base-plus, or large (default: base-plus).
    --sam2-repo <url>      SAM2 git URL (--generalization-data only).
    --sam3-repo <url>      SAM3 git URL; used only with --sam-backend sam3.
    --download-weights     Download weights for the selected target:
                           segmenter/Predict for --generalization-data; DINOv3/LAM for
                           --reward-train and --libero-closed-loop. ResNet18 policy weights
                           are prepared automatically by --libero-closed-loop.
                           Gated model repositories use the official Hugging Face endpoint
                           so license/token checks work; uses *_MODEL_ID overrides.
    --download-transfer-weights
                           Also download optional Cosmos Transfer2.5 weights; implies --download-weights.
    --download-sim-assets  Download the open Menagerie Franka Panda asset and generate
                           the six local MuJoCo task scenes (--generalization-data/--all).
    --download-eval-data   Download Robometer processed evaluation datasets into .venv/datasets/robometer.
                           By default, downloads only datasets named in
                           mi_reward/configs/generalization_data.yaml.
    --eval-dataset <name>  Download one specific Robometer dataset. Repeatable;
                           implies --download-eval-data.
    --all-eval-data        Download and extract the complete Robometer snapshot.
                           This is hundreds of GiB and requires substantial extraction space.
    --download-libero-data Download official LIBERO demonstrations for goal images and
                           optional offline initialization (--libero-closed-loop only).
    --libero-suite <name>  LIBERO dataset to download; repeatable. Supported by upstream:
                           libero_spatial, libero_object, libero_goal, libero_100.
    PYTORCH_INDEX_URL      Environment override for the CUDA PyTorch wheel index (default: cu128).
    HF_MODEL_ENDPOINT      Override the model endpoint (default: https://huggingface.co).
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
        --with-transfer)    INSTALL_TRANSFER=1; shift ;;
        --sam-backend)      SAM_BACKEND="${2:-}"; shift 2 ;;
        --sam2-size)        SAM2_SIZE="${2:-}"; shift 2 ;;
        --sam2-repo)        SAM2_REPO_URL="${2:-}"; shift 2 ;;
        --sam3-repo)        SAM3_REPO_URL="${2:-}"; shift 2 ;;
        --download-weights) DOWNLOAD_WEIGHTS=1; shift ;;
        --download-transfer-weights) DOWNLOAD_WEIGHTS=1; DOWNLOAD_TRANSFER_WEIGHTS=1; INSTALL_TRANSFER=1; shift ;;
        --download-sim-assets) DOWNLOAD_SIM_ASSETS=1; shift ;;
        --download-eval-data) DOWNLOAD_EVAL_DATA=1; shift ;;
        --eval-dataset)
            if [ -z "${2:-}" ]; then
                echo "--eval-dataset requires a dataset name" >&2
                exit 1
            fi
            EVAL_DATASETS+=("$2")
            DOWNLOAD_EVAL_DATA=1
            shift 2
            ;;
        --all-eval-data)
            DOWNLOAD_EVAL_DATA=1
            DOWNLOAD_ALL_EVAL_DATA=1
            shift
            ;;
        --download-libero-data) DOWNLOAD_LIBERO_DATA=1; shift ;;
        --libero-suite)
            if [ -z "${2:-}" ]; then
                echo "--libero-suite requires a dataset name" >&2
                exit 1
            fi
            LIBERO_DATASETS+=("$2")
            DOWNLOAD_LIBERO_DATA=1
            shift 2
            ;;
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
    --reward-train)
        VENV_DIR="${VENV_DIR:-.venv-reward}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    --reward-eval)
        VENV_DIR="${VENV_DIR:-.venv-eval}"
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        ;;
    --libero-closed-loop)
        VENV_DIR="${VENV_DIR:-.venv-libero}"
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
        echo "Unknown target: $TARGET (use --all, --generalization-data, --reward-train, --reward-eval, --libero-closed-loop, --mi-cosmos, or --rlpd)" >&2
        exit 1
        ;;
esac

configure_segmentation_backend
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
    COSMOS_DIR="$ARTIFACT_ROOT/src/cosmos-predict2.5"
    TRANSFER_DIR="$ARTIFACT_ROOT/src/cosmos-transfer2.5"
    MODEL_DIR="$ARTIFACT_ROOT/models"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$ARTIFACT_ROOT/src" "$MODEL_DIR"

    if [ ! -d "$COSMOS_DIR/.git" ]; then
        echo "[install] Cloning Cosmos-Predict2.5 into $COSMOS_DIR..."
        git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_DIR"
        git -C "$COSMOS_DIR" checkout "$COSMOS_GIT_REF"
    fi
    if [ "$INSTALL_TRANSFER" -eq 1 ] && [ ! -d "$TRANSFER_DIR/.git" ]; then
        echo "[install] Cloning optional Cosmos-Transfer2.5 into $TRANSFER_DIR..."
        git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$TRANSFER_DIR"
        git -C "$TRANSFER_DIR" checkout "$COSMOS_TRANSFER_GIT_REF"
    fi
    case "$SAM_BACKEND" in
        sam2)
            SAM_DIR="$ARTIFACT_ROOT/src/sam2"
            if [ ! -d "$SAM_DIR/.git" ]; then
                echo "[install] Cloning SAM2.1 into $SAM_DIR..."
                git clone "$SAM2_REPO_URL" "$SAM_DIR"
            fi
            if [ -n "$SAM2_GIT_REF" ]; then
                git -C "$SAM_DIR" checkout "$SAM2_GIT_REF"
            fi
            ;;
        sam3)
            SAM_DIR="$ARTIFACT_ROOT/src/sam3"
            if [ ! -d "$SAM_DIR/.git" ]; then
                echo "[install] Cloning SAM3 into $SAM_DIR..."
                git clone "$SAM3_REPO_URL" "$SAM_DIR"
            fi
            if [ -n "$SAM3_GIT_REF" ]; then
                git -C "$SAM_DIR" checkout "$SAM3_GIT_REF"
            fi
            ;;
        none)
            SAM_DIR=""
            ;;
    esac

    echo "[install] Installing the generalization-data Python environment..."
    uv pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url "$PYTORCH_INDEX_URL"
    uv pip install \
        mujoco==3.3.2 \
        transformers==4.51.3 \
        numpy scipy pillow opencv-python-headless \
        pyyaml omegaconf hydra-core iopath einops tqdm rich pytest \
        accelerate safetensors imageio imageio-ffmpeg matplotlib pandas \
        huggingface_hub[cli] datasets mediapy tyro loguru \
        moderngl shapely OpenEXR cattrs natsort
    if [ -n "$SAM_DIR" ]; then
        uv pip install -e "$SAM_DIR" --no-deps
    fi
    uv pip install -e "$COSMOS_DIR"
    uv pip install -e "$COSMOS_DIR/packages/cosmos-oss[cu128_torch27]"
    if [ "$INSTALL_TRANSFER" -eq 1 ]; then
        uv pip install -e "$TRANSFER_DIR[cu128]"
    else
        echo "[install] Optional Cosmos Transfer package skipped; use --with-transfer to install it."
    fi
    if [ "$INSTALL_PROJECT" -eq 1 ]; then
        uv pip install -e "$REPO_ROOT" --no-deps
    fi

    write_segmentation_config "$MODEL_DIR"

    if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
        echo "[install] Downloading model weights into $MODEL_DIR..."
        case "$SAM_BACKEND" in
            sam2)
                download_public_file_with_retry \
                    "$SAM2_CHECKPOINT_BASE_URL/$SAM2_CHECKPOINT_NAME" \
                    "$MODEL_DIR/sam2/$SAM2_CHECKPOINT_NAME"
                ;;
            sam3)
                hf_download_model "$SAM3_MODEL_ID" "$MODEL_DIR/sam3"
                ;;
            none)
                echo "[install] Segmentation weights disabled by --sam-backend none."
                ;;
        esac
        hf_download_model "$COSMOS_PREDICT_MODEL_ID" "$MODEL_DIR/cosmos-predict2.5"
        if [ "$DOWNLOAD_TRANSFER_WEIGHTS" -eq 1 ]; then
            hf_download_model "$COSMOS_TRANSFER_MODEL_ID" "$MODEL_DIR/cosmos-transfer2.5"
        else
            echo "[install] Optional Cosmos Transfer weights skipped; use --download-transfer-weights to include them."
        fi
    else
        echo "[install] Weights not downloaded. Re-run with --download-weights after HF login."
    fi

    if [ "$DOWNLOAD_SIM_ASSETS" -eq 1 ]; then
        SIM_ASSET_ARGS=()
        if [ "$USE_MIRRORS" -eq 1 ]; then
            SIM_ASSET_ARGS+=(--use-mirrors)
        fi
        bash "$SCRIPT_DIR/download_mujoco_assets.sh" "${SIM_ASSET_ARGS[@]}"
    else
        echo "[install] MuJoCo task assets not downloaded. Use --download-sim-assets when needed."
    fi

    smoke_generalization_environment
    echo "[install] Data environment ready: source $VENV_DIR/bin/activate"
    echo "[install] MuJoCo headless default: MUJOCO_GL=egl"

# ==================================================================
# Reward training environment: --reward-train
# ==================================================================
elif [ "$TARGET" = "--reward-train" ]; then
    MODEL_DIR="$ARTIFACT_ROOT/models"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$MODEL_DIR"

    install_reward_runtime
    if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
        echo "[install] Downloading DINOv3 and LaWAM LAM weights into $MODEL_DIR..."
        hf_download_model "$DINO_MODEL_ID" "$MODEL_DIR/dinov3-vitb16-pretrain-lvd1689m"
        hf_download_model "$LAM_MODEL_ID" "$MODEL_DIR/lawam_lam"
    else
        echo "[install] Reward weights not downloaded. Re-run --reward-train with --download-weights if needed."
    fi

    smoke_reward_environment "$MODEL_DIR"
    echo "[install] Reward training environment ready: source $VENV_DIR/bin/activate"

# ==================================================================
# RBM-EVAL environment: --reward-eval
# ==================================================================
elif [ "$TARGET" = "--reward-eval" ]; then
    ROBOMETER_DIR="$ARTIFACT_ROOT/src/robometer"
    EVAL_DATA_DIR="$ARTIFACT_ROOT/datasets/robometer"
    MODEL_DIR="$ARTIFACT_ROOT/models"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$ARTIFACT_ROOT/src" "$EVAL_DATA_DIR"

    if [ ! -d "$ROBOMETER_DIR/.git" ]; then
        echo "[install] Cloning Robometer/RBM-EVAL into $ROBOMETER_DIR..."
        git clone "$ROBOMETER_REPO_URL" "$ROBOMETER_DIR"
    fi
    # Robometer's full dependency set pins incompatible Torch/CUDA versions.
    git -C "$ROBOMETER_DIR" fetch --quiet origin "$ROBOMETER_GIT_REF" || true
    git -C "$ROBOMETER_DIR" checkout --quiet --detach "$ROBOMETER_GIT_REF"

    install_reward_runtime
    echo "[install] Installing RBM-EVAL adapter dependencies (without Robometer's incompatible CUDA extras)..."
    uv pip install \
        hatchling scikit-learn seaborn h5py pydantic \
        datasets==4.1.1 loguru termcolor codetiming \
        sentence-transformers decord==0.6.0 mediapy tyro num2words
    uv pip install -e "$ROBOMETER_DIR" --no-deps

    if [ "$DOWNLOAD_EVAL_DATA" -eq 1 ]; then
        if [ "$DOWNLOAD_ALL_EVAL_DATA" -eq 1 ]; then
            echo "[install] WARNING: downloading the complete Robometer snapshot into $EVAL_DATA_DIR."
            echo "[install] This requires hundreds of GiB for archives plus extraction space."
            hf_download_with_retry \
                robometer/processed_datasets \
                --repo-type dataset \
                --local-dir "$EVAL_DATA_DIR" \
                --max-workers "$HF_DOWNLOAD_MAX_WORKERS"
            echo "[install] Extracting all Robometer processed dataset archives..."
            ROBOMETER_PROCESSED_DATASETS_PATH="$EVAL_DATA_DIR" \
                bash "$ROBOMETER_DIR/scripts/untar_processed_datasets.sh"
        else
            if [ "${#EVAL_DATASETS[@]}" -eq 0 ]; then
                mapfile -t EVAL_DATASETS < <(configured_eval_datasets)
            fi
            if [ "${#EVAL_DATASETS[@]}" -eq 0 ]; then
                echo "[install] ERROR: no Robometer datasets were selected and none were found in $GENERALIZATION_DATA_CONFIG." >&2
                exit 1
            fi
            for dataset_name in "${EVAL_DATASETS[@]}"; do
                if robometer_dataset_is_ready "$EVAL_DATA_DIR/$dataset_name"; then
                    echo "[install] Robometer dataset already ready: $dataset_name"
                    continue
                fi
                echo "[install] Downloading selected Robometer dataset: $dataset_name"
                hf_download_with_retry \
                    robometer/processed_datasets \
                    --repo-type dataset \
                    --local-dir "$EVAL_DATA_DIR" \
                    --include "*${dataset_name}.tar*" \
                    --max-workers "$HF_DOWNLOAD_MAX_WORKERS"
                extract_robometer_dataset "$EVAL_DATA_DIR" "$dataset_name"
            done
        fi
    else
        echo "[install] Evaluation data not downloaded. Re-run with --download-eval-data after HF login."
    fi
    ensure_all_robometer_compat_paths "$EVAL_DATA_DIR"
    smoke_reward_environment "$MODEL_DIR"
    smoke_robometer_environment "$ROBOMETER_DIR" "$EVAL_DATA_DIR"
    echo "[install] RBM-EVAL source commit: $(git -C "$ROBOMETER_DIR" rev-parse HEAD)"
    echo "[install] RBM-EVAL environment ready: source $VENV_DIR/bin/activate"

# ==================================================================
# LIBERO closed loop: frozen MI potential + SAC
# ==================================================================
elif [ "$TARGET" = "--libero-closed-loop" ]; then
    LIBERO_DIR="$ARTIFACT_ROOT/src/libero"
    MODEL_DIR="$ARTIFACT_ROOT/models"
    LIBERO_CONFIG_DIR="$ARTIFACT_ROOT/libero_config"
    LIBERO_DATA_DIR="$LIBERO_DIR/libero/datasets"

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "[install] Reusing existing venv at $VENV_DIR"
    else
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    fi
    source "$VENV_DIR/bin/activate"
    mkdir -p "$ARTIFACT_ROOT/src" "$MODEL_DIR"
    mkdir -p "$LIBERO_CONFIG_DIR" "$LIBERO_DATA_DIR"
    export LIBERO_CONFIG_PATH="$(realpath "$LIBERO_CONFIG_DIR")"

    if [ -d "$LIBERO_DIR/.git" ]; then
        git -C "$LIBERO_DIR" fetch --quiet origin "$LIBERO_GIT_REF" || true
        git -C "$LIBERO_DIR" checkout --quiet --detach "$LIBERO_GIT_REF"
    elif [ ! -f "$LIBERO_DIR/.libero-source-ref" ] || \
         [ "$(cat "$LIBERO_DIR/.libero-source-ref" 2>/dev/null || true)" != "$LIBERO_GIT_REF" ]; then
        echo "[install] Downloading pinned LIBERO source archive into $LIBERO_DIR..."
        LIBERO_ARCHIVE_TMP="$(mktemp "${TMPDIR:-/tmp}/libero-source.XXXXXX.tar.gz")"
        LIBERO_ARCHIVE_FETCH_URL="$LIBERO_ARCHIVE_URL"
        if [ "$USE_MIRRORS" -eq 1 ]; then
            LIBERO_ARCHIVE_FETCH_URL="${GITHUB_PREFIX}${LIBERO_ARCHIVE_URL}"
        fi
        curl --fail --location --retry 5 --retry-delay 5 --retry-all-errors \
            --connect-timeout 30 --output "$LIBERO_ARCHIVE_TMP" "$LIBERO_ARCHIVE_FETCH_URL"
        rm -rf "$LIBERO_DIR"
        mkdir -p "$LIBERO_DIR"
        tar -xzf "$LIBERO_ARCHIVE_TMP" --strip-components=1 -C "$LIBERO_DIR"
        rm -f "$LIBERO_ARCHIVE_TMP"
        printf '%s\n' "$LIBERO_GIT_REF" > "$LIBERO_DIR/.libero-source-ref"
    fi

    install_reward_runtime
    echo "[install] Installing LIBERO simulation and visual RLPD/SAC runtime..."
    # Keep this environment's modern Torch/Transformers stack. LIBERO's 2023
    # requirements pin Transformers 4.21 and NumPy 1.22 for its BC baselines;
    # neither is used by the direct OffScreenRenderEnv closed loop.
    uv pip install \
        numpy==1.26.4 mujoco==2.3.7 h5py gym==0.25.2 easydict==1.9 \
        robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 \
        future==0.18.2 cloudpickle==2.1.0 thop==0.1.1.post2209072238
    uv pip install -e "$LIBERO_DIR" --no-deps
    # The visual RLPD policy uses an ImageNet-initialized ResNet18 independently
    # of the frozen DINO/LaWAM reward encoder. Prefer the torch cache when it is
    # already present; otherwise use the resumable public-file downloader.
    RESNET18_DIR="$MODEL_DIR/resnet18-imagenet"
    RESNET18_CHECKPOINT="$RESNET18_DIR/resnet18-f37072fd.pth"
    TORCH_RESNET18_CACHE="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/resnet18-f37072fd.pth"
    mkdir -p "$RESNET18_DIR"
    if [ ! -s "$RESNET18_CHECKPOINT" ] && [ -s "$TORCH_RESNET18_CACHE" ]; then
        cp "$TORCH_RESNET18_CACHE" "$RESNET18_CHECKPOINT"
        echo "[install] Reused cached ResNet18 policy weights: $TORCH_RESNET18_CACHE"
    fi
    download_public_file_with_retry "$RESNET18_CHECKPOINT_URL" "$RESNET18_CHECKPOINT"
    python - <<'PY'
import site
from pathlib import Path

override = Path(site.getsitepackages()[0]) / "robosuite" / "macros_private.py"
override.write_text(
    "# Project-local override: avoid the process-global /tmp/robosuite.log.\n"
    "FILE_LOGGING_LEVEL = None\n"
    "CONSOLE_LOGGING_LEVEL = 'WARN'\n",
    encoding="utf-8",
)
print(f"[install] Robosuite file logging disabled through: {override}")
PY
    # LIBERO's setup.py uses find_packages() from one directory above the
    # actual package root. Modern editable-wheel builders can consequently
    # emit an empty package mapping. Add the reviewed source package root
    # explicitly instead of modifying upstream code.
    python - "$LIBERO_DIR" <<'PY'
import site
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
site_root = Path(site.getsitepackages()[0])
pth = site_root / "libero_source_root.pth"
pth.write_text(str(source_root) + "\n", encoding="utf-8")
print(f"[install] LIBERO source path registered: {pth} -> {source_root}")
PY

    python - "$LIBERO_DIR" "$LIBERO_DATA_DIR" "$LIBERO_CONFIG_DIR/config.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1]).resolve()
data = Path(sys.argv[2]).resolve()
config = Path(sys.argv[3]).resolve()
benchmark_root = source / "libero" / "libero"
payload = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str(data),
    "assets": str(benchmark_root / "assets"),
}
missing = [f"{key}:{value}" for key, value in payload.items() if key != "datasets" and not Path(value).exists()]
if missing:
    raise RuntimeError(f"Pinned LIBERO source has an unexpected layout: {missing}")
config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(f"[install] Project-local LIBERO config: {config}")
PY

    if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
        echo "[install] Downloading DINOv3 and LaWAM LAM weights into $MODEL_DIR..."
        hf_download_model "$DINO_MODEL_ID" "$MODEL_DIR/dinov3-vitb16-pretrain-lvd1689m"
        hf_download_model "$LAM_MODEL_ID" "$MODEL_DIR/lawam_lam"
    fi

    if [ "$DOWNLOAD_LIBERO_DATA" -eq 1 ]; then
        if [ "${#LIBERO_DATASETS[@]}" -eq 0 ]; then
            LIBERO_DATASETS=(libero_spatial)
        fi
        for dataset_name in "${LIBERO_DATASETS[@]}"; do
            echo "[install] Downloading official LIBERO dataset: $dataset_name"
            python "$LIBERO_DIR/benchmark_scripts/download_libero_datasets.py" \
                --download-dir "$LIBERO_DATA_DIR" \
                --datasets "$dataset_name" --use-huggingface
        done
    else
        echo "[install] LIBERO demonstrations skipped. Add --download-libero-data for demo-terminal goals."
    fi

    python - "$LIBERO_GIT_REF" <<'PY'
import importlib.metadata
import sys

import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

suites = benchmark.get_benchmark_dict()
required = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
missing = sorted(required - set(suites))
if missing:
    raise RuntimeError(f"LIBERO installation is missing benchmark suites: {missing}")
print(
    "[install] LIBERO closed-loop smoke test passed:",
    f"torch={torch.__version__}",
    f"transformers={importlib.metadata.version('transformers')}",
    f"libero_ref={sys.argv[1]}",
)
PY
    python - "$RESNET18_CHECKPOINT" <<'PY'
import pathlib
import sys

import torch
from torchvision.models import resnet18

checkpoint = pathlib.Path(sys.argv[1])
model = resnet18(weights=None)
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
print(f"[install] Visual RLPD ResNet18 policy weights ready: {checkpoint}")
PY
    echo "[install] LIBERO closed-loop environment ready: source $VENV_DIR/bin/activate"

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
        transformers==4.51.3 \
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
        transformers==5.2.0 numpy scipy pillow opencv-python-headless \
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

# --all is implemented as four idempotent passes. Python dependencies stay
# isolated, while sources, checkpoints, and datasets stay below ARTIFACT_ROOT.
if [ "$INSTALL_ALL" -eq 1 ]; then
    REWARD_ARGS=(--reward-train)
    EVAL_ARGS=(--reward-eval)
    LIBERO_ARGS=(--libero-closed-loop)
    if [ "$USE_MIRRORS" -eq 1 ]; then
        REWARD_ARGS+=(--use-mirrors)
        EVAL_ARGS+=(--use-mirrors)
        LIBERO_ARGS+=(--use-mirrors)
    fi
    if [ "$INSTALL_PROJECT" -eq 0 ]; then
        REWARD_ARGS+=(--no-install-project)
        EVAL_ARGS+=(--no-install-project)
        LIBERO_ARGS+=(--no-install-project)
    fi
    if [ "$DOWNLOAD_WEIGHTS" -eq 1 ]; then
        REWARD_ARGS+=(--download-weights)
        LIBERO_ARGS+=(--download-weights)
    fi
    if [ "$DOWNLOAD_ALL_EVAL_DATA" -eq 1 ]; then
        EVAL_ARGS+=(--all-eval-data)
    elif [ "$DOWNLOAD_EVAL_DATA" -eq 1 ]; then
        EVAL_ARGS+=(--download-eval-data)
        for dataset_name in "${EVAL_DATASETS[@]}"; do
            EVAL_ARGS+=(--eval-dataset "$dataset_name")
        done
    fi
    ARTIFACT_ROOT="$ARTIFACT_ROOT" bash "$SCRIPT_DIR/install.sh" "${REWARD_ARGS[@]}"
    ARTIFACT_ROOT="$ARTIFACT_ROOT" bash "$SCRIPT_DIR/install.sh" "${EVAL_ARGS[@]}"
    if [ "$DOWNLOAD_LIBERO_DATA" -eq 1 ]; then
        LIBERO_ARGS+=(--download-libero-data)
        for dataset_name in "${LIBERO_DATASETS[@]}"; do
            LIBERO_ARGS+=(--libero-suite "$dataset_name")
        done
    fi
    ARTIFACT_ROOT="$ARTIFACT_ROOT" bash "$SCRIPT_DIR/install.sh" "${LIBERO_ARGS[@]}"
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
