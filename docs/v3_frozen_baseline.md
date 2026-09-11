# Frozen v3 baseline for teacher audit

Frozen at the user's request on 2026-09-08. Mainline formulas, teacher/Qwen
weights, teacher calibration, training data, and training procedure remain fixed
while the evaluation audit runs. No retraining or recalibration is part of this
audit. New audit scripts and reports are separate from the frozen mainline.

Baseline manifest:
`logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1/baseline/freeze_manifest.json`

The manifest records 10,985 SHA256 entries, including 3,493 copied read-only
snapshots. Teacher/Qwen models and relevant source/configuration/data files are
snapshotted. Large feature-model dependencies and rollout input artifacts are
hash-pinned. Original source permissions are unchanged. This is a reproducibility
freeze verified by hashes, not a filesystem access-control guarantee.

Canonical artifacts:

- Teacher: `logs/mi_reward/v3_teacher_mainline/teacher_smoke5/information_teacher_v3.pt`
- Calibration/labels: `logs/mi_reward/v3_teacher_mainline/teacher_export5_v2/`
- Student: `logs/mi_reward/v3_teacher_mainline/qwen3_vl_reward_v1/model/`
- Formula: `D = 0.99 * phi(t+1) - phi(t) + 1.0 * psi(a,v,g)`
- Student input: task-language progress prompt plus synchronized five-frame
  agentview/wrist history; no privileged evidence passed to Qwen.

Run all audit commands inside `tmux train`, using `.venv-libero`:

```bash
bash eval/libero/run_teacher_audit.sh \
  logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1 \
  logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1/<fresh-analysis-subdirectory>
```

The runner copies and hashes audit code for the attempt, reproduces cached
teacher scores/labels, audits existing rollout windows, and checks the frozen
source/snapshot hashes on exit. The first attempt stopped on an inconsistent
success predicate after restoring a standalone simulator state. The completed
protocol in `replay_v2` replays original actions from the same seed/initial state,
checks saved simulator states and success, and excludes any inconsistent window
from physical comparisons. This changes the audit's reconstruction method only.

Physical direction is the existing sampler/gate heuristic, not independent
annotator ground truth. Shared successful demo_0 goal references are used for
rollout teacher inference and differ from training's episode-specific goals.
These limitations must accompany results.
