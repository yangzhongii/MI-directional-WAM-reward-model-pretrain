#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != train ]]; then
    echo 'Run inside tmux train.' >&2
    exit 1
fi
OUTPUT="${1:?Supply the frozen audit run directory}"
ANALYSIS="${2:-$OUTPUT}"
test -f "$OUTPUT/baseline/freeze_manifest.json"
export LIBERO_CONFIG_PATH="$PWD/.venv/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/logs/.mplconfig"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR" "$ANALYSIS/audit_code"
cp eval/libero/audit_teacher.py eval/libero/freeze_baseline.py eval/libero/run_teacher_audit.sh "$ANALYSIS/audit_code/"
sha256sum "$ANALYSIS"/audit_code/* > "$ANALYSIS/audit_code.sha256"
trap '.venv-libero/bin/python -m eval.libero.freeze_baseline --verify --output-dir "$OUTPUT/baseline"' EXIT
.venv-libero/bin/python -m eval.libero.audit_teacher --phase cached --output-dir "$ANALYSIS"
.venv-libero/bin/python -m eval.libero.audit_teacher --phase rollout --output-dir "$ANALYSIS"
