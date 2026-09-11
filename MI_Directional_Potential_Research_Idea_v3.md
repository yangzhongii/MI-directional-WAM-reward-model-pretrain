# MI Directional Potential v3: Robot-Grounded Information Potential for Qwen3-VL Reward Modeling

## 1. Fixed Research Idea

### One-sentence summary

Learn a privileged robot-grounded directional teacher whose information-theoretic core decomposes task progress into a **visual state potential** and **incremental action information**, verify that soft information signal with privileged physical evidence such as contact, kinematics, task relations, outcome, and feasibility, project the resulting teacher judgment into Positive/Unclear/Negative supervision, fine-tune a directly downloaded Qwen3-VL model on dual-view multimodal JSONL, and deploy the resulting Qwen3-VL reward model in RLinf using only visual history and task language.

### Core research question

> Can goal-related information in visual state and latent action be decomposed into a theoretically meaningful directional progress signal, grounded by privileged robot physics during training, and distilled into an observation-only Qwen3-VL reward model that produces reliable P/U/N sparse reward for robot RL?

The first paper focuses on rigid-object manipulation, with LIBERO as the controlled benchmark and Franka/RLinf as the intended real-robot deployment path.

The main scientific contribution is the **robot-grounded directional reward supervision mechanism**. It is not a new Qwen architecture, not a new Cosmos model, and not a new robot policy.

---

## 2. Research Claim

Visual similarity, temporal order, and generic VLM judgment are insufficient for many manipulation transitions.

Typical failure cases include:

- gripper visually near an object but not in contact;
- object visually above a target but not stably placed;
- apparent progress followed by slip or loss of grasp;
- visually similar states with different gripper/object relations;
- identical or near-identical states followed by opposite actions;
- progress that is non-monotonic in time;
- visually similar scene variants with different physical task state.

The paper should defend the following claim:

> A Qwen3-VL reward model supervised by a privileged robot-grounded information teacher should outperform the same Qwen3-VL backbone trained with temporal-order, success-only, visual-only, or simple privileged heuristic supervision, especially on physically aliased, near-miss, and non-monotonic manipulation progress.

---

## 3. Central Design Principle

The v3 teacher is **one privileged robot-grounded teacher with multiple evidence channels**.

Do **not** describe contact, kinematics, relation, and outcome as independent MI teachers.

The teacher has two different parts:

```text
Information-theoretic core
    |
    +-- visual state information
    +-- conditional action information
    |
    v
soft directional score
    |
    v
Privileged physical consistency
    |
    +-- contact / grasp
    +-- kinematics
    +-- object-task relations
    +-- outcome
    +-- feasibility
    |
    v
confidence / consistency
    |
    v
P / U / N supervision target
```

This distinction is central:

- **visual/action latents** provide the information-theoretic directional signal;
- **robot state and physical relations** validate, correct, or abstain from that signal;
- **P/U/N** is the Qwen reward-model output space, not the native mathematical definition of MI.

---

## 4. Variables and Observation Contract

Let:

- \(o_t\): synchronized visual observation at time \(t\);
- \(v_t = E_v(o_t)\): visual latent state;
- \(a_t^{lat}=E_a(o_t,o_{t+1})\) or a latent action extracted from the executed action/transition;
- \(g\): task goal representation, including task language and optionally a goal/reference visual representation;
- \(x_t^{priv}\): privileged robot/object/physics information available only offline.

The deployment observation is dual-view:

```text
agentview history
      +
wrist history
      +
task language
```

For LIBERO:

- `agentview` is the third-person tabletop view;
- `robot0_eye_in_hand` is the wrist view;
- both views are synchronized;
- a short temporal window, e.g. 5 frames, is preferred to a single frame.

Privileged state is used to construct labels but is not given to Qwen at deployment.

---

## 5. Information-Theoretic Core

### 5.1 Why ordinary MI notation is insufficient

Mutual information is a property of random variables and their distributions:

\[
I(V;G)
=
\mathbb E_{p(v,g)}
\left[
\log \frac{p(v,g)}{p(v)p(g)}
\right].
\]

Therefore a single pair of embeddings \((v_t,g)\) should not be written as though it directly has a scalar `MI` value without defining a distribution or density-ratio estimator.

For a per-state reward potential, v3 uses a **pointwise information score** induced by a learned density-ratio / contrastive critic.

### 5.2 Visual pointwise information potential

Define the visual state potential as:

\[
\phi_t^v
=
\operatorname{PMI}(v_t,g)
=
\log
\frac{p(v_t,g)}{p(v_t)p(g)}.
\]

Equivalently:

\[
\phi_t^v
=
\log p(g\mid v_t)-\log p(g).
\]

Interpretation:

> How much does the current visual latent reduce uncertainty about the goal/task state relative to the marginal distribution?

This is the **state-potential** term.

The directional visual progress is:

\[
\Delta\phi_t^v
=
\gamma\phi_{t+1}^v-
\phi_t^v.
\]

The directionality is created by the temporal difference, not by mutual information itself.

### 5.3 Conditional action information

Action latent describes a transition, not a state.

Therefore it should not be treated as another symmetric state potential.

Define the incremental action information as the conditional pointwise information:

\[
\psi_t^a
=
\operatorname{PMI}(a_t^{lat};g\mid v_t)
=
\log
\frac{
 p(a_t^{lat}\mid v_t,g)
}{
 p(a_t^{lat}\mid v_t)
}.
\]

Interpretation:

> Once the current visual state is already known, how much additional goal-directed information is carried by the executed latent action/transition?

This directly addresses the case:

```text
same or similar current state
        |
        +-- action toward goal
        +-- action away from goal
```

which cannot be distinguished by a state-only potential.

### 5.4 Why visual and action information can coexist

The information chain rule gives the theoretical decomposition:

\[
I(V,A;G)
=
I(V;G)
+
I(A;G\mid V).
\]

This motivates the architecture:

```text
visual latent
    -> state information

action latent
    -> incremental information conditioned on visual state
```

rather than naively summing two unconditional MI terms.

This also reduces double counting between visual motion evidence and latent action evidence.

### 5.5 Directional information score

The core soft teacher score is:

\[
D_t
=
\Delta\phi_t^v
+
\beta\psi_t^a.
\]

Expanded:

\[
D_t
=
\gamma\phi_{t+1}^v-
\phi_t^v
+
\beta
\log
\frac{
 p(a_t^{lat}\mid v_t,g)
}{
 p(a_t^{lat}\mid v_t)
}.
\]

The parameter \(\beta\) controls the incremental contribution of action information.

The paper must ablate:

- visual state potential only;
- action information only;
- naive `visual MI + action MI`;
- proposed `visual PMI + conditional action PMI`.

---

## 6. MI Estimation and Hard-Negative Construction

The implementation must not claim to compute exact MI from one embedding pair.

The estimator should be described as one of:

- learned density-ratio estimation;
- a contrastive critic corresponding to a pointwise information score;
- an InfoNCE-style lower-bound / ranking critic with clearly stated limitations;
- another explicitly defined probabilistic estimator.

### 6.1 Critical negative sampling rule

The critic must learn **progress-relevant** information, not task identity or scene similarity.

Preferred hard negatives are:

```text
same task
same object family
same or similar scene
similar RGB appearance
different physical progress
```

Examples:

- grasped vs almost grasped;
- contact vs near-contact;
- object above container vs stably inside container;
- correct vs wrong orientation;
- holding vs slipping;
- approaching goal vs moving away from goal.

Easy negatives from unrelated tasks should not dominate training.

### 6.2 Successful-reference diversity

Action supervision must not force one canonical expert trajectory.

The action term should model a **goal-conditioned action distribution**:

\[
p(a^{lat}\mid v,g)
\]

rather than only similarity to one reference action latent.

This permits multiple successful strategies, grasp orientations, and approach paths.

---

## 7. Privileged Physical Consistency

Information score alone is not physical truth.

Define privileged evidence:

\[
x_t^{priv}
=
\{
contact,
kinematics,
relations,
outcome,
feasibility
\}.
\]

These channels do not need an MI interpretation.

### 7.1 Contact / grasp evidence

Examples:

- gripper-object contact;
- stable grasp;
- grasp loss;
- object motion coupled to gripper motion.

### 7.2 Kinematic evidence

Examples:

- approach;
- grasp closure;
- lift;
- transport;
- insertion depth;
- release;
- stage-consistent motion.

### 7.3 Task-relation evidence

Examples:

- object-goal displacement;
- relative orientation;
- object-in-container relation;
- alignment;
- target-region membership;
- gripper-object relation.

### 7.4 Outcome / feasibility evidence

Authoritative evidence includes:

- measured success;
- measured failure;
- collision;
- impossible state transition;
- workspace violation;
- simulator inconsistency;
- invalid generated candidate.

### 7.5 Consistency gate

Define a confidence / consistency function:

\[
C_t
=
f_{consistency}
(D_t,x_t^{priv}).
\]

The physical evidence should not simply become another arbitrary weighted reward term.

Its primary role is to:

- validate the soft information score;
- override physically impossible positive judgments;
- detect regression invisible in RGB;
- identify teacher disagreement;
- decide when to abstain.

---

## 8. From Teacher Evidence to P/U/N

The privileged teacher produces:

```text
visual pointwise potential phi_v
visual directional delta
conditional action information psi_a
soft directional score D
physical consistency C
trajectory preference
confidence
```

P/U/N is constructed by a supervision adapter.

### Positive

Use `Positive` when:

- \(D_t\) indicates sufficient forward progress;
- physical evidence is consistent with that progress;
- confidence is above threshold;
- no authoritative failure signal contradicts the judgment.

### Negative

Use `Negative` when:

- \(D_t\) indicates regression; or
- grasp/contact is lost in a task-critical stage; or
- task relations move away from success; or
- an authoritative physical/outcome signal identifies failure or regression.

### Unclear

Use `Unclear` when:

- information progress is inside a calibrated dead band;
- visual and action evidence conflict;
- information score and physical evidence conflict;
- different views disagree;
- confidence is low;
- transition is approximately task-neutral;
- evidence is insufficient.

Thus:

\[
U = \text{teacher abstention / uncertainty},
\]

not merely `abs(score) < fixed_threshold` and not a hardcoded tail-of-episode rule.

Thresholds are calibrated on held-out teacher-validation data.

---

## 9. Teacher Preferences and Continuous Diagnostics

The v2 continuous structure is retained for scientific evaluation.

For each transition/window the teacher may save:

```text
phi_visual_t
phi_visual_t1
delta_phi_visual
conditional_action_information
directional_score
physical_consistency
confidence
P/U/N target
```

For trajectory pairs from the same initial state, it may additionally save:

```text
chosen
rejected
preference_margin
teacher_confidence
```

These continuous values are diagnostic and may be used in auxiliary losses or ablations even though the primary Qwen online interface is P/U/N.

---

## 10. Qwen3-VL SFT Data Construction

Qwen3-VL is downloaded directly and fine-tuned on project-generated multimodal JSON/JSONL.

No external robot VQA SFT stage is mandatory.

The core route is:

```text
Qwen3-VL-Instruct
        +
privileged-teacher-labeled JSONL
        |
        v
Qwen3-VL MI Reward Model
```

### 10.1 Qwen input

Qwen receives only deployment-compatible observations:

```text
task instruction
+
agentview temporal window
+
wrist temporal window
+
optional goal/reference image
```

Privileged state is never included in the student input.

### 10.2 Qwen SFT target

The primary assistant target is:

```text
Positive
Unclear
Negative
```

P/U/N is simultaneously:

- the SFT target output space;
- the inference output space of the final reward model.

This does **not** mean MI itself is a ternary classifier.

### 10.3 Conceptual JSONL record

```json
{
  "images": [
    "agent_t0.jpg",
    "agent_t1.jpg",
    "agent_t2.jpg",
    "agent_t3.jpg",
    "agent_t4.jpg",
    "wrist_t0.jpg",
    "wrist_t1.jpg",
    "wrist_t2.jpg",
    "wrist_t3.jpg",
    "wrist_t4.jpg"
  ],
  "messages": [
    {
      "role": "user",
      "content": "Task: place the red mug to the right of the caddy. Judge recent task progress as Positive, Unclear, or Negative."
    },
    {
      "role": "assistant",
      "content": "Positive"
    }
  ]
}
```

The exact schema follows the selected Qwen3-VL finetuning implementation.

---

## 11. Qwen3-VL Reward Model

After LoRA/SFT, Qwen3-VL itself is the reward model.

No JEPA stage is required in the v3 mainline.

Online inference:

```text
agentview history
      +
wrist history
      +
task language
      |
      v
fine-tuned Qwen3-VL
      |
      v
Positive / Unclear / Negative
```

The model must reproduce the privileged teacher judgment using only visual-language evidence.

A later compression model is optional only if online latency requires it.

---

## 12. RLinf Sparse Reward Interface

Qwen output is parsed into sparse scalar reward.

Example:

\[
r_t^{Qwen}
=
\begin{cases}
+1.0,&P\\
0.0,&U\\
-0.2,&N.
\end{cases}
\]

For the first main experiments, use:

\[
r_t
=
r_t^{env}
+
\lambda r_t^{Qwen}.
\]

Environment success remains the terminal authority.

The paper must **not** claim that arbitrary ternary mapping automatically preserves classical potential-based policy invariance.

The theoretical potential exists inside the privileged teacher; the Qwen P/U/N reward is a learned sparse approximation used for deployment.

---

## 13. Optional Cosmos / Transfer / Simulator Branch

The main method must work on verified simulator/real trajectories without Cosmos.

Cosmos is optional data-coverage support.

### 13.1 Predict2.5

Use target-domain-adapted action-conditioned prediction to create additional candidate futures.

Candidate actions come from:

- planner;
- LaWAM;
- policy;
- another explicit proposer.

Cosmos does not propose reward labels.

### 13.2 Transfer2.5

Use Transfer for controlled visual variation:

- scene/background;
- lighting/style;
- bounded camera appearance;
- visual object-instance variation.

Transfer-generated visual changes are not physical truth.

### 13.3 Simulator

The simulator is authoritative for:

- physical state;
- collision/contact;
- object replacement requiring geometry/dynamics changes;
- feasibility;
- executable state transitions.

### 13.4 Truth rule

```text
generated RGB != physical ground truth
```

Generated data enters supervision only after strict verification.

---

## 14. Canonical v3 Pipeline

```text
                           OFFLINE
------------------------------------------------------------------

Verified dual-view manipulation trajectories
  |
  +-- agentview history
  +-- wrist history
  +-- language / goal
  +-- actions / transitions
  +-- robot state
  +-- object state
  +-- contact / grasp
  +-- task relations
  +-- outcome / feasibility
  |
  v
Visual Encoder E_v
  |
  v
visual latent v_t
  |
  +------------------------------+
  |                              |
  v                              v
Visual pointwise information     Latent Action Encoder E_a
potential phi_v                  |
  |                              v
  |                         action latent a_t^lat
  |                              |
  |                              v
  |                    conditional action information
  |                    psi_a = PMI(a; g | v)
  |                              |
  +---------------+--------------+
                  |
                  v
     directional information score

 D_t = gamma * phi_v(t+1) - phi_v(t) + beta * psi_a
                  |
                  v
     privileged physical consistency
  |
  +-- contact / grasp
  +-- kinematics
  +-- task relations
  +-- outcome / feasibility
  |
  v
confidence / consistency gate
  |
  v
P / U / N label construction
  |
  v
Qwen3-VL multimodal JSONL
  |
  v
Qwen3-VL LoRA / SFT
  |
  v
Qwen3-VL MI Reward Model


                           ONLINE
------------------------------------------------------------------

agentview history
      +
wrist history
      +
task instruction
      |
      v
Qwen3-VL MI Reward Model
      |
      v
P / U / N
      |
      v
RLinf reward parser
      |
      +-- P -> positive sparse shaping reward
      +-- U -> zero / abstention
      +-- N -> negative sparse shaping reward
      |
      v
environment reward + Qwen shaping
      |
      v
RLinf / RLPD / PPO / asynchronous robot RL
      |
      v
Franka


                   OPTIONAL DATA-COVERAGE BRANCH
------------------------------------------------------------------

Cosmos Predict / Transfer / simulator counterfactuals
      |
      v
strict physical / visual verification
      |
      v
hard negatives / regression / object / scene variants
      |
      v
same privileged teacher pipeline
```

---

## 15. Relationship to v2

v3 preserves the v2 idea that progress should be represented by a directional potential and that privileged information should supervise a deployable reward model.

The theoretical clarification is:

### v2-style intuition

```text
visual MI
+
action MI
-> directional potential
```

### v3 formalization

```text
visual pointwise information
    -> state potential

action conditional pointwise information
    -> incremental transition information

visual potential difference
+
conditional action information
    -> directional information score

physical privileged evidence
    -> consistency / confidence
```

The chain-rule motivation is:

\[
I(V,A;G)
=
I(V;G)
+
I(A;G\mid V).
\]

This makes visual and action latent roles complementary instead of redundant.

---

## 16. Required Experimental Questions

### Q1. Is the information-theoretic teacher better than simple privileged heuristics?

This is the decisive novelty test.

Keep Qwen and data fixed and compare supervision from:

- temporal ordering;
- state-distance heuristic;
- relation/contact heuristic;
- visual similarity / cosine;
- visual pointwise-information potential only;
- visual PMI + unconditional action MI;
- **visual PMI + conditional action PMI**;
- **full information teacher + physical consistency**.

### Q2. Is conditional action information necessary?

Evaluate cases with similar state but opposite transition direction.

Compare:

\[
\Delta\phi_v
\]

against:

\[
\Delta\phi_v + \beta\psi_a.
\]

### Q3. Does the chain-rule formulation reduce double counting?

Compare:

\[
I(V;G)+I(A;G)
\]

against:

\[
I(V;G)+I(A;G\mid V).
\]

Measure ranking quality and calibration on held-out progress pairs.

### Q4. Does physical consistency resolve visual aliasing?

Build a dedicated challenge set:

- grasp / no-grasp;
- contact / near-contact;
- placed / visually over target;
- holding / slipping;
- correct / wrong orientation;
- recovery after regression;
- physically invalid but visually plausible generated states.

### Q5. Can Qwen distill the privileged teacher?

Evaluate:

- P/U/N macro-F1;
- per-class recall;
- near-miss false Positive rate;
- `Unclear` calibration;
- agreement with teacher directional score;
- pairwise trajectory ordering.

### Q6. Does the reward improve RL?

Compare:

- environment sparse reward;
- sparse + generic Qwen reward;
- sparse + temporal-supervised Qwen;
- sparse + privileged-heuristic Qwen;
- sparse + information-teacher Qwen.

Measure:

- success rate;
- sample efficiency;
- failed-grasp recovery;
- reward hacking;
- cyclic behavior;
- inference latency;
- memory.

### Q7. Does Cosmos augmentation help?

Only after the core teacher is validated, compare:

- no Cosmos;
- base Cosmos;
- target-domain-adapted Cosmos;
- Transfer scene variation;
- Transfer visual-instance variation;
- simulator physical-instance variation.

---

## 17. Physical-Aliasing Benchmark

This benchmark is a core part of the paper, not an optional diagnostic.

Each pair should have:

```text
same task
same or very similar appearance
different privileged physical state
different true progress direction
```

Recommended pairs:

| Pair | Similar visual appearance | Privileged distinction |
|---|---|---|
| grasp / miss | gripper next to object | contact/grasp state |
| over container / placed | object above/in target region | object-in-container relation |
| hold / slip | object near gripper | grasp stability / relative motion |
| touch / insert | tool at hole entrance | insertion depth/contact geometry |
| approach / retreat | same approximate state | latent action direction |
| valid / collision | visually plausible pose | collision/feasibility |

The main claim should be demonstrated strongly on this benchmark.

---

## 18. Training and Leakage Rules

At minimum isolate:

- information-critic train data;
- critic validation data;
- teacher threshold/calibration data;
- Qwen SFT train data;
- Qwen validation data;
- final reward test data;
- final RL evaluation tasks/states;
- held-out object instances;
- held-out scene variants;
- successful references;
- Cosmos adaptation splits when enabled.

Hard negatives used to train the information critic must not leak final physical-aliasing test pairs.

When a task family appears across splits, track:

- episode identity;
- initial state;
- scene identity;
- object instance;
- reference trajectory;
- candidate trajectory ancestry.

---

## 19. Required Baselines

At minimum:

- sparse environment reward;
- pixel/image distance;
- DINO/LaWAM cosine;
- PROGRESSOR-style visual progress;
- zero-shot Qwen3-VL;
- Qwen3-VL with temporal-order supervision;
- Qwen3-VL with simple privileged state/relation heuristic labels;
- visual pointwise-information teacher;
- naive visual MI + action MI;
- proposed visual PMI + conditional action PMI;
- proposed information teacher + privileged physical consistency;
- old `VisualGoalPotential` / GeoProgress baseline;
- reproducible external process/VLM reward baselines where feasible.

The strongest controlled experiment keeps fixed:

```text
same Qwen backbone
same LoRA rank
same visual windows
same instructions
same number of examples
same optimizer budget
```

and changes only the supervision source.

---

## 20. Go / No-Go Research Gates

### Gate 1 — Information teacher

Require:

\[
\text{information teacher + physical consistency}
>
\text{simple privileged heuristic}
\]

on the physical-aliasing benchmark.

If this fails, the MI contribution must be reconsidered.

### Gate 2 — Action term

Require a measurable gain from:

\[
I(A;G\mid V)
\]

on same-state/different-action cases.

If this fails, remove the action-information term rather than keeping it for complexity.

### Gate 3 — Qwen distillation

Require:

\[
Qwen_{information-teacher}
>
Qwen_{temporal/heuristic}
\]

on held-out objects, scenes, near-misses, and regression/recovery.

### Gate 4 — RL

Require stable improvement in success/sample efficiency without obvious reward hacking.

### Gate 5 — Real robot

Only after the first four gates, move to Franka/RLinf real-robot evaluation.

---

## 21. Model and System Roles

| Component | Fixed role | Not responsible for |
|---|---|---|
| Visual encoder | produce state latent \(v_t\) | reward truth |
| Latent action encoder | represent executed transition/action | state potential |
| Visual pointwise-information critic | estimate goal-related state potential | physical validation |
| Conditional action-information critic | estimate incremental goal-directed transition information | contact truth |
| Contact/kinematic/relation/outcome channels | privileged physical consistency and confidence | MI estimation |
| Teacher supervision adapter | convert score + consistency into P/U/N labels | online inference |
| Qwen3-VL | learn observation-only reward prediction | privileged reward truth |
| RLinf parser | map P/U/N to sparse scalar shaping reward | task-progress estimation |
| Simulator | authoritative physical state / feasibility | photorealistic reward model |
| Cosmos Predict2.5 | optional future-data augmentation | action proposal / reward truth |
| Cosmos Transfer2.5 | optional visual augmentation | physical state truth |

---

## 22. Implementation Mapping

### Reusable existing components

- `mi_reward/features/` for visual/action/kinematic features;
- `mi_reward/scoring/` for current directional scoring and preferences;
- `mi_reward/verification/` for physical consistency checks;
- `mi_reward/data/` for trajectories and sidecars;
- `world_model/` for optional Cosmos augmentation;
- RLinf Qwen-VLM/Franka reward infrastructure for deployment.

### Main v3 components to implement

1. explicit visual pointwise-information critic;
2. explicit conditional action-information critic;
3. hard-negative sampler emphasizing physical aliasing;
4. directional score \(D_t\) implementation;
5. privileged consistency/confidence module;
6. P/U/N supervision adapter;
7. dual-view Qwen3-VL JSONL dataset builder;
8. Qwen3-VL LoRA/SFT integration;
9. held-out physical-aliasing evaluation;
10. supervision-source ablations;
11. RLinf loading/config for the fine-tuned reward model.

The existing continuous `VisualGoalPotential` / GeoProgress path remains a baseline, not the final deployment architecture.

---

## 23. Scope Boundaries

- Mutual information itself is symmetric; **directionality comes from temporal potential difference and conditional transition information**.
- Do not claim a single embedding pair has exact MI without an estimator/distribution definition.
- Visual latent is used as a **state potential**.
- Action latent is used as **conditional incremental transition information**, not a second state potential.
- Kinematics/contact/relation/outcome are privileged physical evidence, not automatically MI terms.
- P/U/N is the Qwen SFT/output interface, not the mathematical definition of the teacher.
- Qwen3-VL can be directly downloaded and fine-tuned on project-generated JSONL.
- Qwen is the deployable student, not the reward authority.
- JEPA is not part of the primary v3 pipeline.
- Cosmos is optional augmentation and never physical reward truth.
- Environment terminal success remains authoritative in the initial RL experiments.

---

## 24. Success Criterion

The project succeeds only if:

1. the visual pointwise-information potential captures goal-related state progress better than simple visual similarity;
2. conditional action information improves direction discrimination beyond state potential alone;
3. the proposed conditional formulation outperforms or is better calibrated than naive independent visual/action information summation;
4. privileged physical consistency reduces false Positive judgments on visually aliased states;
5. the complete teacher outperforms simple privileged heuristic teachers under controlled evaluation;
6. Qwen3-VL fine-tuned on the resulting P/U/N JSONL reproduces teacher judgments from dual-view visual history and task language only;
7. the learned Qwen reward improves LIBERO RL over sparse, temporal-Qwen, and privileged-heuristic-Qwen baselines;
8. the same interface can be deployed through RLinf on Franka without privileged state;
9. optional Cosmos augmentation improves coverage without becoming a source of unverified physical truth.

---

## 25. Compact Canonical Pipeline

Use this as the canonical v3 reference:

```text
Verified dual-view robot trajectories
        |
        +-------------------------------+
        |                               |
        v                               v
visual latent v_t                latent action a_t
        |                               |
        v                               v
pointwise visual information     conditional action information
phi_v = PMI(v,g)                 psi_a = PMI(a,g | v)
        |                               |
        +---------------+---------------+
                        |
                        v
        D_t = gamma*phi_v(t+1) - phi_v(t)
                    + beta*psi_a
                        |
                        v
       privileged physical consistency
   contact + kinematics + relations + outcome
                        |
                        v
               confidence / abstention
                        |
                        v
                 P / U / N labels
                        |
                        v
              Qwen3-VL JSONL SFT
                        |
                        v
             Qwen3-VL MI Reward Model
                        |
                        v
                  online P/U/N
                        |
                        v
              RLinf sparse reward
                        |
                        v
                 LIBERO / Franka

Optional:
Cosmos Predict / Transfer / simulator counterfactuals
    -> strict verification
    -> hard-negative / generalization data
    -> same teacher pipeline
```

This document supersedes the earlier v3 formulation that described several independent MI teachers. The canonical theoretical core is now **visual pointwise information as state potential + conditional action information as incremental transition evidence + privileged physical consistency**.

---

## 26. 实施进展与验证审计更新（2026-09-08）

本节记录当前实际产物、验证结果、已发现的问题和下一步工作，不修改前文的数学定义、模型架构或主 pipeline。主线仍为：

```text
Directional-MI Teacher → P/U/N supervision → Qwen3-VL Reward Model → Reward Validation
```

当前应将项目状态表述为：**teacher 与 Qwen 的工程链路已实现，LIBERO 原生验证已运行；reward 有效性及严格 PMI 解释尚未得到充分验证。** “训练完成”不等于“研究主张已成立”。

### 26.1 已完成产物

| 环节 | 当前产物与状态 |
|---|---|
| Teacher checkpoint | `logs/mi_reward/v3_teacher_mainline/teacher_smoke5/information_teacher_v3.pt`；spatial 前五个任务，10 条训练 episode、5 条验证 episode |
| P/U/N 导出 | `logs/mi_reward/v3_teacher_mainline/teacher_export5_v2/`；1,168 条训练窗口、536 条验证窗口，另存 privileged diagnostics |
| Qwen checkpoint | `logs/mi_reward/v3_teacher_mainline/qwen3_vl_reward_v1/model/`；Qwen3-VL-2B-Instruct，freeze vision、无 LoRA、gradient checkpointing |
| LIBERO evaluator | `eval/libero/`；官方 HDF5/真实 rollout manifest → 适配 → 现有 `Qwen3VLRewardModel.predict_row()` → 逐窗口与轨迹指标 |
| Suite 支持 | 接口支持 libero_10、libero_90、libero_spatial、libero_object、libero_goal；实际数据与运行验证目前仅覆盖 spatial |
| 冻结基线 | `logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1/baseline/`；10,985 个文件哈希条目、3,493 个只读快照文件 |

所有本轮验证、采集和统计均在 `tmux train` 中执行，优先使用现有 `.venv-libero`，没有重新创建环境或训练模型。Cosmos/worldsample 未进入主 validation，继续保留为后续泛化与鲁棒性分支。

### 26.2 原有 Robometer 结论的修正

旧报告中的 quality preference accuracy 0.9772、ranking accuracy 0.5813，以及 failure-vs-successful 0.636，**不能继续作为轨迹偏好能力已经成立的证据**。

审计发现 preference 代码使用 `chosen_score >= rejected_score`，把平局计为正确；ranking 的旧计分也受到平局和输入顺序影响。对已保存的逐样本原始分数重算得到：

| 历史预测审计 | 配对数 | 胜 / 平 / 负 | 严格胜率 | 平局计半分准确率 |
|---|---:|---:|---:|---:|
| Quality preference | 395 | 33 / 353 / 9 | 8.35% | 53.04% |
| Quality ranking | 750 | 0 / 750 / 0 | 0% | 50% |
| Failure vs successful | 250 | 0 / 250 / 0 | 0% | 50% |

Ranking 数据的 150 条轨迹得分全部为 0。严格胜率为 0 在这里表示没有严格胜出的配对，不表示所有配对都排反了。Quality preference 的平局率为 89.37%，平均 chosen-minus-rejected margin 为 0.06076。

适配器已修正严格平局计分、双视角顺序、NumPy camera 选择和任务文本回退，并要求真实同步 agent/wrist 输入。历史推理还存在输入契约问题：不能把本次算术重算称为修正输入后的新 Robometer benchmark。单视角数据需要另行明确评估协议，不能冒充双视角数据。

证据：`logs/mi_reward/v3_teacher_mainline/validation_audit_v1/robometer_rescore.json`。

### 26.3 LIBERO smoke 与原生控制测试

最初的 `libero_spatial_smoke_v1/v2` 各运行 10 条成功 demo、20 个末尾/早期窗口，仅证明链路可执行。它们使用 3 帧历史和裸任务文本，与训练的主要 5 帧历史及进展提示不一致；部分 demo_0 与训练来源重叠。这些运行不构成独立 reward 能力验证。

v3 的 P/U/N 表示近期方向，不能要求末尾 reward 必然高于早期 reward。因此旧 smoke 的“末尾优于早期比例 0%、差值 −0.1”仅为历史诊断，不用作模型失效结论。

修正后的原生测试位于 `validation_audit_v1/`：

- 10 个 spatial 任务，每个使用未出现在 teacher 训练/验证中的 `demo_20` 初始状态，seed=2026。
- 从相同初始状态分别执行原始 demo 动作和保持夹爪打开的扰动动作；simulation 正常推进，不注入中途保存状态。
- 每条轨迹的真实结果由 `env.check_success()` 判定，不根据控制器身份预设成功或失败。
- 使用训练提示与 5 帧双视角历史，每条轨迹均匀选择 8 个窗口，轨迹得分为窗口 reward 的平均值。
- 总计 20 条轨迹、160 个窗口，真实成功 6 条、失败 14 条；7 项输入/指标测试通过。

| 指标 | 结果 |
|---|---:|
| 同任务、同初始状态成功/失败配对 | 6 对 |
| 胜 / 平 / 负 | 5 / 0 / 1 |
| success_failure_accuracy | 83.33% |
| reward_margin | 0.14583 |
| ranking_accuracy | null：没有独立质量排序标签 |
| 原始动作 replay 成功率 / 平均 reward | 60% / 0.6000 |
| 开夹爪 replay 成功率 / 平均 reward | 0% / 0.5875 |

这是单初始状态、两种脚本控制的 episode-held-out pilot，不是训练后 policy ranking，也不是独立 task-held-out benchmark。6 对上的 83.33% 不足以证明普遍有效；失败控制组仍获较高平均 reward，需要窗口级分析。局部方向 reward 的平均值与最终成功之间也没有自动成立的理论对应。

证据：`logs/mi_reward/v3_teacher_mainline/validation_audit_v1/SUMMARY.md`、`task_comparisons.csv`、`validation/results.json`。

### 26.4 冻结 teacher 的数学与采样审计

当前 teacher 的实际计算符合：

\[
D_t=0.99\phi_v(v_{t+1},g)-\phi_v(v_t,g)+\psi_a(a_t,v_t,g).
\]

对 1,704 条已导出记录重建该公式，最大绝对误差约为 \(1.0\times10^{-6}\)；使用原有 physical gate、冻结阈值和 abstention 规则重现 P/U/N，标签不一致数为 0。**未发现总公式计算或导出投影漂移。**

但这不等于两个 critic 已严格估计所声明的 PMI。当前实现存在以下统计解释缺口：

1. **Visual 采样。** 正 goal 来自当前 episode 的成功终局；负 goal 来自同任务另一条成功 episode。该构造并非直接从无条件边缘分布 p(g) 抽样，可能学习 episode 与终局的对应关系，尚未证明学到任务进展。
2. **Action 采样。** 正样本只选物理 forward transition；负样本来自同任务 non-forward 状态的视觉最近邻。该分布没有被证明等于 p(a|v)，且未应用采样密度修正。
3. **密度比解释。** 平衡分类器的理想 logit 对应实际构造的正负分布之比；只有相应分布条件成立，才能进一步解释为目标 PMI。采用 contrastive loss 或使用 PMI 模块名称本身不能完成这一论证。
4. **校准差异。** 当前阈值来自训练集 neutral 分位数，与前文要求的 held-out teacher-validation 校准不完全一致。本次审计保持阈值冻结，没有利用新结果调参。

相关实现：`mi_reward/training/train_information_teacher_v3.py` 中的 `build_visual_samples()`、`build_action_hard_negatives()`，以及 `mi_reward/data/export_teacher_targets_v3.py` 中的 `_calibrate()`。

缓存验证集的 536 个 transition 包含 424 个 forward、59 个 neutral、53 个 regression。排除 neutral 后重新统计：

| 分数 | 普通准确率 | 类别平衡准确率 | Regression recall |
|---|---:|---:|---:|
| Cosine 差分 | 63.10% | 66.04% | 69.81% |
| Visual 势能差 | 51.57% | 52.95% | 54.72% |
| Action critic | 80.71% | 68.51% | 52.83% |
| 合并 D | 79.66% | 67.92% | 52.83% |
| 始终预测 forward | 88.89% | 50.00% | 0% |

Action 项有一定方向信号；当前 visual 项未显示对合并分数的增益。普通准确率受类别不平衡影响，不能单独用于支持 teacher 优越性。上述物理方向来自已有 sampler 启发式，仍不是独立人工/物理挑战集的真值评估。

证据：`logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1/replay_v2/cached_teacher_audit.json`。

### 26.5 逐窗口 teacher、物理证据与 Qwen 对照

在同一批 160 个已评分窗口上，重新提取冻结 LAM 特征与 teacher 分数，并从原始 seed、初始状态和完整动作序列重放获得物理证据。160 个窗口全部通过与原保存状态及 success 的一致性核验。

首次仅恢复单帧 MuJoCo state 的尝试在 task 2 的成功终帧出现 success 不一致，审计停止且保留日志；最终统计采用完整动作重放。没有将不一致的还原结果改标成真值。

| 全部 160 个窗口的对照指标 | 结果 |
|---|---:|
| Qwen 对冻结 teacher accuracy | 42.50% |
| Qwen 对冻结 teacher macro-F1 | 0.33485 |
| Qwen 对 teacher-Negative recall | 2.70%（1/37） |
| Qwen 对物理启发式 macro-F1 | 0.39012 |
| Qwen 对物理 regression recall | 0%（0/22） |
| 物理非前进窗口被 Qwen 判为 Positive | 51.25%（41/80） |

失败轨迹共有 112 个被采样窗口，其中 Qwen 输出 Positive 的 66 个窗口可进一步分为：

| 物理启发式方向 | 窗口数 |
|---|---:|
| Forward | 27 |
| Neutral | 25 |
| Regression | 14 |

同一批 66 个 Qwen-Positive 窗口中，冻结 teacher 分别判为 Positive 17 个、Unclear 31 个、Negative 18 个。因此失败轨迹上的高 reward，不能全部解释为合理的局部前进。

例如 task 1 原始动作 replay 的帧 126→127：EEF-object 距离由 0.160144 增至 0.165688，两帧均未 grasp；物理启发式与 teacher 均判 Negative，Qwen 判 Positive。报告保存了对应双视角图像路径、连续 teacher 分数和物理量，便于独立复核。

解释限制：

- Teacher gate 使用了相同物理证据，因此不能将 teacher 与这些物理标签的高一致率作为独立 teacher 正确率。
- 失败轨迹没有自己的成功终局；本次 teacher 使用同任务成功 demo_0 参考 goal。它不同于训练时的 episode-specific goal，不能把全部 teacher/Qwen 分歧直接归因于学生蒸馏。
- Teacher 窗口目标对应末尾 transition，Qwen 使用 5 帧历史。物理标签仍是既有启发式，窗口与事件语义需要独立审核。
- 原 teacher-validation 上的学生 macro-F1 为 0.55071、Negative recall 为 44.44%；新 rollout 对照更差，但两套标签/goal 条件不同，不能直接视作严格同条件性能下降。

证据：`logs/mi_reward/v3_teacher_mainline/teacher_audit_frozen_v1/SUMMARY.md`、`replay_v2/window_comparison.csv`、`replay_v2/failure_examples.json`、`replay_v2/rollout_teacher_audit.json`。

### 26.6 当前判断与下一步优先级

当前证据支持的判断是：**没有发现总公式计算错误，但严格 PMI 的采样解释尚未闭合；visual 势能贡献弱，学生对 Negative 的识别尤其不足；历史评估还存在已确认的计分与输入问题。** 尚不能把全部问题归于数学总公式，也不能声称完整 v3 的有效性已得到证明。

下一步按以下顺序推进：

1. **独立复核困难窗口。** 优先审核物理非前进而 Qwen Positive、teacher Negative 而 Qwen Positive 的样本；区分真实倒退、合理接近、静止、grasp/contact 事件，以及仅末尾 transition 与整段视觉历史之间的语义差异。
2. **明确采样分布及估计目标。** 对照实际正负样本构造，写清 critic 学习的分布比值与 v3 目标 PMI 之间需要的条件；结合成功参考 goal 敏感性诊断，判断当前信号是否主要依赖 episode/goal 对应关系。
3. **分离 teacher 与学生问题。** 固定输入和参考条件，对比信息分数、physical gate、最终 P/U/N 与 Qwen 输出，确认数据覆盖、teacher 标签和学生复现分别贡献了哪些错误。
4. **扩充独立评估覆盖。** 增加初始状态、真实策略 rollout 和困难失败类型，报告类别平衡指标、Negative recall、平局率、配对数及不确定性，再扩展其他四个 suite。
5. **离线证据充分后再考虑受控消融与闭环 RL。** 当前不据这批小样本结果启动主线重训或 RL 部署；Cosmos 继续作为后续泛化分支。

### 26.7 冻结版本与文档更新的关系

冻结基线记录于 `docs/v3_frozen_baseline.md`。最终模型审计结束时，10,985 个文件条目校验为 unchanged，0 个 mismatch。

本节是在该审计完成后，按用户要求追加的进展记录。**本工作文档因此相对于冻结清单中的旧版本发生了预期变化**；冻结快照中的 v3 文档仍保留审计前内容，历史 manifest 不重写。此文档追加不表示解冻模型、公式、校准、训练数据或训练流程。后续若重新校验旧 manifest，应区分本次明确授权的文档变化与其他未授权变化，不能继续声称整个原清单逐字节未变。

主要复现入口（均在 `tmux train` 内）：

```bash
# 原生 rollout 与历史 Robometer 分数审计，使用新输出目录
bash eval/libero/run_audit.sh <fresh-validation-output-directory>

# 使用既有冻结基线运行 teacher 审计，使用新分析目录
bash eval/libero/run_teacher_audit.sh <frozen-run-directory> <fresh-analysis-directory>
```

注意：teacher 审计脚本结束时会校验旧冻结清单；本次文档追加后，该校验会报告工作文档的预期哈希差异。应保留此差异记录或在后续获准的版本管理工作中单独登记文档修订，不覆盖历史冻结基线以掩盖变化。

## 27. 借用 v2 空间表示与成功参考设计的冻结诊断实验

### 27.1 目的与实际执行内容

本次实验检验：在保留 v3 主架构的条件下，保留视觉 patch 的空间信息、增加同任务成功参考，是否比简单池化的视觉相似度更能识别局部进退。它是无需训练的视觉分支诊断，不是完整 v2 的复现，也不是新的 MI teacher。

实验已在 `tmux test` 的 `v2-visual-probe` 窗口使用既有 `.venv-libero` 执行完成，另一个 pane 实时显示 `run.log`。临时脚本为 `/tmp/v2_visual_probe.py`；完成后按要求删除，保留脚本 SHA-256、实验协议、逐窗口分数和日志。退出码为 0，日志记录 `windows=160/160` 和 `PROTECTED_UNCHANGED true; COMPLETE`。

具体处理如下：

1. 复用第 26 节已通过完整动作重放核验的 160 个 rollout 窗口，覆盖 spatial 的 10 个任务、每任务一个初始状态与两种脚本控制器。使用每个 5 帧窗口最后两帧比较局部变化；没有重新采集 rollout。
2. 使用同一冻结 LAM 提取 agentview、wrist 的 patch tokens。每个视角有 256 个 768 维 token；不训练新 backbone 或 critic。
3. 单参考固定使用同任务成功 `demo_0`；多参考固定使用 `demo_0/1/2`，排除 rollout 来源 `demo_20`。每个参考取首个记录成功帧起最多 5 帧的 token 均值；参考选择不根据待评估分数调整。
4. 做 2×2 对照：空间均值池化后计算 cosine，或计算对应位置 patch 的 cosine 再平均；分别配单参考或三参考相似度的算术平均。两路摄像头分数等权平均，不挑选最佳参考。
5. 主分数为 `similarity(t+1) - similarity(t)`，用符号判断进退，零分不算任何一类预测正确；另存 `0.99 * similarity(t+1) - similarity(t)` 作为次要结果。这里的相似度差分不替换 v3 的 visual critic 或总分 D，也不加入 action 项。

没有修改 teacher 数学、P/U/N、校准、Qwen 推理或训练流程，没有重训，也没有引入 Cosmos。实验前后对 teacher 权重、Qwen 权重、校准和关键实现共 8 个文件做 SHA-256 比较，全部一致。这是指定文件的实验保护检查，不代表第 26.7 节的旧完整 manifest 在文档更新后仍全部一致。

### 27.2 结果与不确定性

160 个窗口的既有物理启发式标签为 forward 80、regression 22、neutral 58。以下方向指标排除 neutral，样本数为 102；类别平衡准确率为 forward recall 与 regression recall 的平均值。

| 表示与参考 | 类别平衡准确率 | Forward recall | Regression recall | 相对池化单参考的平衡准确率差：95% CI（百分点） |
|---|---:|---:|---:|---:|
| 池化 + 单参考 | 44.89% | 62.50% | 27.27% | 基线 |
| 池化 + 多参考 | 40.51% | 53.75% | 27.27% | [-12.28, +3.09] |
| 空间 patch + 单参考 | 33.30% | 57.50% | 9.09% | [-20.50, -2.75] |
| 空间 patch + 多参考 | 35.80% | 62.50% | 9.09% | [-19.62, +1.70] |

置信区间采用按任务聚类的配对 bootstrap：10 个任务、有放回抽样 2,000 次、seed=7，避免把同一任务内窗口当作完全独立样本。只有 10 个任务簇且每任务一个初始状态，区间仍不能替代更广泛验证。

**本次没有观察到直接保留对应位置 patch 或增加成功参考带来的改善。** 空间单参考相对池化单参考的差值区间完全为负；另两种改动的区间跨零，没有清晰增益证据。因此，本次结果不支持直接将这个空间相似度方案接入主线或据此回退到 v2。

必须保留以下解释限制：

- 标签仍来自已有物理启发式，不是独立人工标注；两个脚本控制器及同任务窗口存在相关性。
- 对应位置 patch cosine 假设空间对应，移动的 wrist 摄像头尤其可能破坏这一假设。该方法表现差不能证明 token 中没有有用信息。
- 本次没有训练空间 critic，没有测试完整 v2 参考轨迹对齐、严格 MI 估计、Qwen 蒸馏改善或闭环策略收益。
- 本节是 rollout 上的相似度诊断，第 26.4 节是缓存验证集上的冻结 critic 诊断；不能跨数据集直接用两张表计算方法增益。

### 27.3 对 action latent 与 visual latent 的当前判断

更准确的结论是：**当前 action 分支呈现一定方向信号，当前 visual 势能分支的效果偏弱，尚未显示合并增益；还不能把差异定位到 latent 本身。**

第 26.4 节缓存验证集上，action critic 的平衡准确率为 68.51%，visual 势能差为 52.95%，合并 D 为 67.92%。Action 的 regression recall 仍只有 52.83%，因此“有信号”不足以说明它已经是可靠 reward，也没有补齐条件 PMI 的统计解释。

Visual 分支的问题仍可能涉及池化信息损失、正负样本构造、成功 goal 的选择或 critic 学习目标。本次简单 patch 对照未能将这些原因分离，不能据此认定视觉编码器无效。Qwen 当前也没有直接接收 action latent，teacher 的 action 信号能否通过视觉 P/U/N 监督传递给学生，需要单独验证；已有 Negative 识别不足仍是独立问题。

下一步建议保留冻结基线，先固定同一评估集与参考条件，复核困难窗口的方向标签和时间语义，再分别诊断 goal 敏感性与 visual 正负采样。只有确定单一改动及独立评估标准后，再考虑训练消融；本次没有执行这些后续训练。

### 27.4 保存的证据与查看方式

实验目录：`logs/mi_reward/v3_teacher_mainline/v2_visual_probe_v1/`。

- `protocol.json`：特征、参考、分数、bootstrap 定义及已删除脚本的 SHA-256。
- `scores.csv`：全部 160 个窗口的分数与物理方向；`run.log`：逐窗口可见输出、汇总和清理记录。
- `results.json`、`SUMMARY.md`：主结果、次要折扣差分结果及不确定性。
- `protected_before.json`、`protected_after.json`：8 个受保护文件的前后哈希；`exit_code.txt`：退出码 0。

查看保留输出：

```bash
tmux attach -t test
cat logs/mi_reward/v3_teacher_mainline/v2_visual_probe_v1/SUMMARY.md
tail -n 30 logs/mi_reward/v3_teacher_mainline/v2_visual_probe_v1/run.log
```

临时脚本已按要求删除，上述命令用于查看已有证据，不能直接重跑该脚本；重现实验需要根据保存的协议重新实现。此次文档更新只追加本节，原有 pipeline 定义与冻结快照保持原状。
