# Contact-validity reward pilot

## Scope

This is a custom MuJoCo pilot, not an Isaac Factory benchmark. Isaac Lab/Factory,
PegInsert, GearMesh and NutThread are not installed in the current environment.
The pilot uses the local welded-grasp peg scene only to validate the experiment
interface: RGB features, an MI consistency score, force/torque summaries and a
privileged hard-negative label.

## Experiment

The pilot generated 240 perturbed states and compared four small classifiers:

1. RGB-only;
2. RGB + MI;
3. RGB + F/T summary;
4. RGB + F/T + MI.

Formal output:

```text
logs/mi_reward/contact_validity_pilot_v1/results.json
logs/mi_reward/contact_validity_pilot_v1/live.log
```

| Model | Accuracy | AUROC |
|---|---:|---:|
| RGB-only | 0.675 | 0.5552 |
| RGB + MI | 0.675 | 0.5581 |
| RGB + F/T | 0.675 | 0.5552 |
| RGB + F/T + MI | 0.675 | 0.5581 |

The positive rate was `0.7083`. The four models are effectively tied. This is a
negative/inconclusive pilot, not evidence that F/T cannot improve contact-validity
reward: labels are generated from a single custom scene with simple offset/force
rules, and the features are low-dimensional summaries rather than RGB/proprio/F/T
history sequences.

## Decision

Raw MI is retained only as an auxiliary ablation feature. The main B experiment
requires a real contact-rich dataset with privileged topology/depth/jam/cross-thread
labels and temporal RGB + proprio + F/T history. Once Isaac Factory is available,
the four-model comparison should be rerun on PegInsert, GearMesh and NutThread with
held-out demos and contact-validity metrics.

## Temporal reward-model pilot

An 8-frame pilot then used RGB history, proprioceptive pose, force/torque summaries
and an optional MI score. Formal output:

```text
logs/mi_reward/contact_validity_reward_pilot_v1/results.json
logs/mi_reward/contact_validity_reward_pilot_v1/live.log
```

| Model | Accuracy | AUROC |
|---|---:|---:|
| RGB-only | 0.7667 | 0.8125 |
| RGB + MI | 0.7667 | 0.8183 |
| RGB + F/T | 0.7667 | 0.8125 |
| RGB + F/T + MI | 0.7417 | 0.8245 |

This remains a custom-scene pilot, not a Factory result. MI gives a small AUROC
change in this synthetic setup, while the full model loses fixed-threshold accuracy
despite the highest AUROC. MI therefore remains an auxiliary ablation; the final
claim requires calibrated thresholds, held-out demos and privileged contact labels
on real contact-rich tasks.
## Dataset contract v2 audit

The redesigned collector `mi_reward/scripts/collect_contact_validity_dataset_v2.py`
now uses a fixed pre-contact goal image for `MI(I_t, I_goal)`, structured mutually
exclusive feature groups, trajectory-level records, contact-pair/depth/force labels
and grouped split metadata. A 12-trial smoke run produced:

```text
logs/mi_reward/contact_validity_dataset_v2b/frames.npz
logs/mi_reward/contact_validity_dataset_v2b/metadata.json
```

All 12 trajectories were labeled `jam`; no `valid_entry` or `false_alignment`
examples were generated. This is an environment/trajectory-generation failure, not
reward-model evidence. Training is frozen until the self-built scene can generate all
required outcome classes with contact topology, insertion depth and recoverability
labels. The original static and temporal pilots remain `INVALID_FOR_COMPARISON`.

## Near-contact reward dataset v1

The scene was then changed to a free-peg near-contact configuration and the collector
was restricted to short clips around the socket. The collector records RGB frames,
force/torque summaries, fixed-goal MI, insertion depth and peg/socket contact pairs;
labels are derived from depth, contact and force criteria rather than end-effector
offsets.

```text
logs/mi_reward/peg_near_contact_reward_v3/frames.npz
logs/mi_reward/peg_near_contact_reward_v3/metadata.json
```

The 30-trial smoke contract contains 10 `valid_entry`, 10 `jam` and 10
`false_alignment` clips. This validates the data categories and trial-level storage,
but it is still not a reward-model result. The next experiment can train the four
structured input variants with trajectory-level splits and bootstrap confidence
intervals.
## First structured reward evaluation

The four variants were evaluated with a trajectory split holding out one trial per
class. Output:

```text
logs/mi_reward/peg_near_contact_reward_v4/results.json
```

All four variants obtained AUROC `1.0`, AUPRC `1.0` and balanced accuracy `1.0` on
only three held-out trajectories. This is not positive reward-model evidence: the
test set is too small and the condition construction makes classes visually
separable. It only verifies that corrected feature groups and the trajectory-level
evaluator run end to end. A formal claim requires many held-out trials spanning
initial states, geometry, appearance and friction.
## Hard-negative v5 smoke audit

The paired collector `mi_reward/scripts/collect_contact_hardnegatives_v5.py` keeps the
same visible pre-contact pose while varying hidden friction, yaw and lateral preload.
Output:

```text
logs/mi_reward/contact_hardneg_v5_smoke3/frames.npz
logs/mi_reward/contact_hardneg_v5_smoke3/metadata.json
```

Paired first-frame RGB MAE is `0.0`, so the visual matching contract passes. However,
even the stronger hidden condition (`friction=5.0`, `yaw=1.5°`, preload `20`) produced
`valid_entry=6`, `jam=0`, `false_alignment=0`. The current socket/free-peg scene does
not convert hidden mechanics into contact-validity failures. This is a physics-scene
failure, not reward-model evidence; hard-negative expansion and reward training remain
frozen until clearance/contact geometry yields distinct force, depth and recoverability
outcomes.
