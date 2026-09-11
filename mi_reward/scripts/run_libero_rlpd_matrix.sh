#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ ! -x .venv-libero/bin/python ]; then
    echo "Missing .venv-libero. Run: bash requirements/install.sh --libero-closed-loop" >&2
    exit 1
fi

exec .venv-libero/bin/python -m mi_reward.closed_loop.rlpd_matrix "$@"

