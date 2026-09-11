#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != train ]]; then
    echo 'Run inside tmux train.' >&2
    exit 1
fi
OUTPUT="${1:?Supply a fresh output directory}"
export LIBERO_CONFIG_PATH="$PWD/.venv/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/logs/.mplconfig"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"
.venv-libero/bin/python -m unittest eval.libero.test_validation -v
.venv-libero/bin/python -m eval.libero.audit_robometer --output "$OUTPUT/robometer_rescore.json"
.venv-libero/bin/python -m eval.libero.collect_rollouts --output-dir "$OUTPUT/rollouts"
.venv-libero/bin/python -m eval.libero.run_validation \
  --manifest "$OUTPUT/rollouts/manifest.jsonl" --history-frames 5 \
  --windows-per-trajectory 8 --output-dir "$OUTPUT/validation"
