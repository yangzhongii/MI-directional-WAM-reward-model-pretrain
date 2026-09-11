#!/usr/bin/env bash
# Stage 3: independent held-out evaluation of the deployable reward potential.
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

exec python -m mi_reward.evaluation.generalization_reward_eval "$@"
