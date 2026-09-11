#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-logs/mi_reward/v4_decision_gate/multistate_rollouts_v1}"
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite existing output: $output_root" >&2
  exit 2
fi

mkdir -p "$output_root"
export LIBERO_CONFIG_PATH="$PWD/.venv/libero_config"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/logs/.mplconfig"

for demo_index in 21 22; do
  .venv-libero/bin/python eval/libero/collect_rollouts.py \
    --output-dir "$output_root/demo_${demo_index}" \
    --task-ids 0 1 2 3 4 \
    --demo-index "$demo_index" \
    --seed "$((2006 + demo_index))" \
    2>&1 | tee "$output_root/demo_${demo_index}.log"
done

printf '0\n' > "$output_root/exit_code.txt"
