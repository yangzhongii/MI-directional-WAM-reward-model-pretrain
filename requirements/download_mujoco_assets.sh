#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

USE_MIRRORS=0
MENAGERIE_REF="${MENAGERIE_REF:-da76818e269b82289eba39808e2fb91d679d6994}"
MENAGERIE_DIR="${MENAGERIE_DIR:-assets/vendor/mujoco_menagerie}"
OFFICIAL_URL="https://github.com:443/google-deepmind/mujoco_menagerie.git"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --use-mirrors) USE_MIRRORS=1; shift ;;
        --menagerie-ref) MENAGERIE_REF="${2:-}"; shift 2 ;;
        -h|--help)
            echo "Usage: bash requirements/download_mujoco_assets.sh [--use-mirrors] [--menagerie-ref REF]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ "$USE_MIRRORS" -eq 1 ]; then
    REPO_URL="https://ghfast.top/https://github.com/google-deepmind/mujoco_menagerie.git"
else
    # The explicit :443 avoids stale global insteadOf rules such as gh-proxy.com.
    REPO_URL="$OFFICIAL_URL"
fi

if [ -e "$MENAGERIE_DIR" ] && [ ! -d "$MENAGERIE_DIR/.git" ]; then
    echo "[assets] ERROR: $MENAGERIE_DIR exists but is not a Git checkout; refusing to overwrite it." >&2
    exit 1
fi

if [ ! -d "$MENAGERIE_DIR/.git" ]; then
    echo "[assets] Cloning MuJoCo Menagerie metadata into $MENAGERIE_DIR..."
    git clone --depth 1 --filter=blob:none --no-checkout "$REPO_URL" "$MENAGERIE_DIR"
fi

git -C "$MENAGERIE_DIR" remote set-url origin "$REPO_URL"
git -C "$MENAGERIE_DIR" sparse-checkout init --cone

for attempt in 1 2 3 4 5; do
    echo "[assets] Fetching Franka Panda asset (attempt $attempt/5)..."
    if git -C "$MENAGERIE_DIR" fetch --depth 1 origin "$MENAGERIE_REF" && \
        git -C "$MENAGERIE_DIR" checkout --detach "$MENAGERIE_REF" && \
        git -C "$MENAGERIE_DIR" sparse-checkout set franka_emika_panda; then
        break
    fi
    if [ "$attempt" -eq 5 ]; then
        echo "[assets] ERROR: failed to download MuJoCo Menagerie after 5 attempts." >&2
        exit 1
    fi
    sleep 5
done

if [ -x .venv/bin/python ]; then
    PYTHON_BIN=.venv/bin/python
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[assets] Generating task scenes from the open Panda model..."
"$PYTHON_BIN" mi_reward/scripts/prepare_mujoco_assets.py \
    --menagerie-root "$MENAGERIE_DIR/franka_emika_panda" \
    --output-root assets/custom_task/scene

echo "[assets] MuJoCo task assets are ready under assets/custom_task/scene."
