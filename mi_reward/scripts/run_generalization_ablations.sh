#!/usr/bin/env bash
# Generate/run/summarize the reproducible rigid-v3 reward ablation matrix.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

if [ -f ".venv-reward/bin/activate" ]; then
    source .venv-reward/bin/activate
else
    echo "Missing reward environment. Run: bash requirements/install.sh --reward-train" >&2
    exit 1
fi

exec python -m mi_reward.experiments.generalization_ablations "$@"
