#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != train ]]; then
    echo 'Run this script inside tmux session train.' >&2
    exit 1
fi
PYTHON="${PYTHON:-.venv-libero/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"$PYTHON" -m unittest eval.libero.test_validation -v
"$PYTHON" -m eval.libero.run_validation \
    --benchmark libero_spatial --max-tasks 10 --max-demos 1 --prefix-smoke \
    "$@"
