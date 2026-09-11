# LIBERO reward validation

The current teacher/Qwen baseline is frozen for a read-only mathematical and
window-level audit. See `docs/v3_frozen_baseline.md` for artifacts and commands.

Pipeline v3 adapter: native LIBERO RGB trajectories → existing
`Qwen3VLRewardModel.predict_row()` → task-matched trajectory metrics.
No teacher, training, or inference implementation changes.

Use the existing `.venv-libero` environment inside tmux session `train`:

```bash
bash eval/libero/run_smoke.sh --output-dir logs/mi_reward/v3_teacher_mainline/libero_spatial_smoke_v1
```

This first runs contract/metric tests, then scores one official demo from each
of the ten spatial tasks. Each prediction now uses the last five contiguous frames
of both cameras, with all agentview frames preceding all wrist frames, matching
the shared Qwen message builder, and the teacher export's progress-judgment prompt.
The historical smoke_v1/v2 runs used three frames and bare task text; they are
pipeline execution checks only. RGB orientation is preserved as in the existing
native demo reader. The optional early-prefix window ends at 25% of the episode.

Outputs: `inputs.jsonl` (model rows plus separate provenance/labels),
`predictions.jsonl` (per-window P/U/N outputs), exported PNGs, and `results.json`.
Use a fresh output directory for each run to avoid stale or overwritten results.

Supported suite names: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`,
`libero_goal`. Only spatial data is currently installed. For other suites, supply
the existing dataset parent directory with `--data-root`; no automatic download.
Generic invocation (also run within `tmux train`):

```bash
.venv-libero/bin/python -m eval.libero.run_validation \
  --benchmark libero_spatial --data-root .venv/src/libero/libero/datasets \
  --max-tasks 10 --max-demos 1 --output-dir logs/mi_reward/libero_validation_run
```

Limits apply to HDF5 input; 0 means all. A manifest evaluates all listed records.
Native HDF5 input is specifically the official successful demonstration dataset,
not arbitrary policy rollouts. Existing simulator/state loaders were inspected;
the adapter reads only required RGB slices and task language to avoid loading
entire episodes or simulator state for inference.

For real labeled rollouts use `--manifest path/to/rollouts.jsonl`. Each line:

```json
{"benchmark":"libero_spatial","source":"libero_rollout","task_id":"task-name","trajectory_id":"policy-a-episode-0","language":"pick up the bowl","agentview":["frames/agent_0.png","frames/agent_1.png"],"wrist":["frames/wrist_0.png","frames/wrist_1.png"],"success":false,"quality":1}
```

Image lists must be synchronized and chronological; paths are relative to the
manifest. `success` is boolean or null; `quality` is an optional numeric annotation
where larger is better. Labels never enter the model input. Use genuine native
LIBERO rollouts and externally recorded outcomes, not Cosmos/worldsample data.

Metric definitions:

- `ranking_accuracy`: fraction of within-task pairs with distinct annotated
  quality whose rewards have the correct strict ordering (trajectory quality
  ranking, not a policy-level success-rate ranking).
- `success_failure_accuracy`: fraction of within-task real success/failure pairs
  where reward(success) > reward(failure).
- `reward_margin`: mean reward(success) − reward(failure) over those same pairs.

Pairs are pooled across tasks; reward ties count as incorrect independent of
input order. Missing comparison groups produce JSON null with explicit reasons.
Official demos contain successes only, so the three metrics cannot be estimated
from this smoke dataset. `smoke_diagnostics` separately records terminal labels
and terminal-vs-early-prefix preference/margin. These temporal proxies do not
establish success/failure discrimination or quality ranking. Native demonstrations
may overlap teacher sources; this smoke run establishes pipeline execution, not
held-out generalization performance.

## Corrected validation audit

Run inside `tmux train`, using a fresh directory:

```bash
bash eval/libero/run_audit.sh logs/mi_reward/v3_teacher_mainline/validation_audit_v1
```

This tests the evaluator, rescores saved Robometer predictions, collects real
simulator controls, and evaluates eight uniformly spaced five-frame windows per
trajectory with the frozen Qwen checkpoint. The trajectory score is the mean
window reward, fixed before inspecting predictions. `trajectories.jsonl` saves
these aggregates, while `predictions.jsonl` preserves every scored window.

Collection uses `demo_20` initial states, checking exclusion from both teacher
train and validation episode identities. Two action controllers run from each
identical state: original demonstration actions and the same actions with the
gripper commanded open. Simulation advances normally; no saved intermediate
states are injected. Every outcome is measured with `env.check_success()`, never
inferred from controller identity. Full action/state/outcome traces and dual-view
PNGs are saved. Failed original replays remain failures. This is an episode-held-out
scripted control challenge, not a learned-policy benchmark or task-held-out test.
Only one initial state per task is used in this pilot.

`success_failure_details` includes strict wins, ties, losses, tie-adjusted accuracy
(ties receive 0.5), and margin. `matched_initial_state_success_failure` restricts
comparisons to controls sharing their initial state. `by_controller` records
measured success rates, average rewards, and all window labels. Ranking by external
quality remains unavailable unless genuine quality annotations are provided.
An ultimately failed episode can contain real forward-progress windows; its
Positive fraction is not a window-level false-Positive rate. Likewise terminal
reward need not exceed early reward for this directional model.

The Robometer audit only recomputes metrics from saved predictions. Its old
`chosen >= rejected` rule counted every reward tie as correct. The corrected
adapter uses strict comparisons and reports ties; quality ranking additionally
reports order-invariant strict metrics separately from the official legacy
compiler. It now requires actual synchronized agent/wrist frames (or a valid
dual-view exported row), fixes view-major ordering and NumPy camera selection,
uses five frames, and falls back to the trajectory's task for language. Single-view
input is rejected instead of being silently passed off as dual-view data.
Historical predictions are not thereby repaired; the rescore is an audit of old
evidence, not a fresh corrected Robometer inference benchmark.
