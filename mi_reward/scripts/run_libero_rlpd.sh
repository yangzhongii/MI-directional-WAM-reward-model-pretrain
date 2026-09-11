#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ ! -x .venv-libero/bin/python ]; then
    echo "Missing .venv-libero. Run: bash requirements/install.sh --libero-closed-loop" >&2
    exit 1
fi

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.venv/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/logs/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

exec .venv-libero/bin/python -m mi_reward.closed_loop.libero_rlpd "$@"

