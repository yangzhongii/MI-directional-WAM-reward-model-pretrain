#!/usr/bin/env bash
set -uo pipefail

output_dir="${1:-logs/mi_reward/v4_decision_gate/gate1_smoke_v1}"
live_log="${output_dir}.live.log"

if [[ -e "$output_dir" || -e "$live_log" ]]; then
  echo "Refusing to overwrite existing output: $output_dir or $live_log" >&2
  exit 2
fi

mkdir -p "$(dirname "$output_dir")"
export LIBERO_CONFIG_PATH="$PWD/.venv/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/logs/.mplconfig"

.venv-libero/bin/python eval/libero/probe_v4_gate1_smoke.py \
  --output-dir "$output_dir" \
  2>&1 | tee "$live_log"
status=${PIPESTATUS[0]}

if [[ -d "$output_dir" ]]; then
  cp "$live_log" "$output_dir/run.log"
  printf '%s\n' "$status" > "$output_dir/exit_code.txt"
fi

exit "$status"
