#!/usr/bin/env bash
# Stage 1: generate and verify scene/instance generalization data.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Missing data-generation environment. Run: bash requirements/install.sh --generalization-data" >&2
    exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

CONFIG="mi_reward/configs/generalization_data.yaml"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --task-suite) ARGS+=("--task-suite" "$2"); shift 2 ;;
        --dry-run) ARGS+=("--dry-run"); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

python -m mi_reward.data.generalization_pipeline --config "$CONFIG" "${ARGS[@]}"
