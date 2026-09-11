#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG="eval/configs/rbm_eval.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash eval/run_rbm_eval.sh [--config eval/configs/rbm_eval.yaml]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

PYTHON="${PYTHON:-$REPO_ROOT/.venv-eval/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON. Run: bash requirements/install.sh --reward-eval" >&2
    exit 1
fi
export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-$REPO_ROOT/.venv/datasets/robometer}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m mi_reward.evaluation.rbm_eval_adapter --config "$CONFIG"
