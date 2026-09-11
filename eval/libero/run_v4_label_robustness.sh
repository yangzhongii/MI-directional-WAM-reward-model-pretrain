#!/usr/bin/env bash
set -uo pipefail

output_dir="${1:-logs/mi_reward/v4_decision_gate/label_robustness_v1}"
live_log="${output_dir}.live.log"
if [[ -e "$output_dir" || -e "$live_log" ]]; then
  echo "Refusing to overwrite existing output: $output_dir or $live_log" >&2
  exit 2
fi

export LIBERO_CONFIG_PATH="$PWD/.venv/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/logs/.mplconfig"

.venv-libero/bin/python eval/libero/analyze_v4_label_robustness.py \
  --rollout-manifests \
    logs/mi_reward/v3_teacher_mainline/validation_audit_v1/rollouts/manifest.jsonl \
    logs/mi_reward/v4_decision_gate/multistate_rollouts_v1/demo_21/manifest.jsonl \
    logs/mi_reward/v4_decision_gate/multistate_rollouts_v1/demo_22/manifest.jsonl \
  --prediction-file logs/mi_reward/v4_decision_gate/gate1_multistate_v1/predictions.npz \
  --output-dir "$output_dir" \
  2>&1 | tee "$live_log"
status=${PIPESTATUS[0]}

if [[ -d "$output_dir" ]]; then
  cp "$live_log" "$output_dir/run.log"
  printf '%s\n' "$status" > "$output_dir/exit_code.txt"
fi

exit "$status"
