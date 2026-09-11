#!/usr/bin/env bash
# Pipeline-v5 preflight: no simulator rollout, model update, or reward training.
set -euo pipefail

output_dir="${1:-logs/mi_reward/v5_mi_action_geometry/math_check_v1}"
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir"
export MPLCONFIGDIR="$PWD/logs/.mplconfig"

{
  echo "[v5-math] started_at=$(date --iso-8601=seconds)"
  echo "[v5-math] python=$(.venv-libero/bin/python --version)"
  .venv-libero/bin/python eval/libero/probe_v5_mi_action_geometry.py \
    --output-dir "$output_dir/test0_identity"
  .venv-libero/bin/python -m pytest -q tests/test_mi_action_field.py tests/test_fast_mi_scoring.py
} 2>&1 | tee "$output_dir/run.log"

printf '0\n' > "$output_dir/exit_code.txt"
echo "[v5-math] PASS output_dir=$output_dir" | tee -a "$output_dir/run.log"
