# Scientific Review: Pipeline v5-A

## 1. 评审信息与范围

| 项目 | 内容 |
|---|---|
| 评审对象 | `MI_Directional_Potential_Research_Idea_v5-a.md` |
| 评审模式 | scientific/full；对研究计划进行论文级科学性审查 |
| 目标场所 | ICLR 2027（根据当前对话上下文推断） |
| 评审材料 | v5-A pipeline、v5 20-anchor raw-RGB results、相关 evaluator 设计、公开 benchmark 页面和相关工作检索结果 |
| 未提供材料 | v5-A 尚无训练后的 reward-model checkpoint、主 benchmark 结果、真机结果、完整论文正文、统计置信区间和最终实现代码 |
| 证据边界 | 因此本报告评估的是研究计划的可发表性和证据闭环，不把拟议实验当作已经完成的结果 |

## 2. 总体结论与关键理由

v5-A 的概念重构是合理的：它把 privileged physical progress 与 observation-only reward model 分开，也诚实保留了 v5 full-frame raw-RGB physical gate 的 `NO_GO`。但按当前文本，它还不是一篇可投稿论文，而是一份方向正确、尚未完成关键验证的实验计划。

决定性问题有三个：

1. **核心 reward model 尚未训练或评估。** 文档定义了 teacher、student、loss 和 benchmark，但没有 student 的实际结果。
2. **MI 的方法贡献尚未建立。** 当前 full-frame MI 的 physical cosine 为 `0.68842`，低于 `0.70` gate；若 ROI 和 reference-conditioned MI 没有带来相对于 no-MI student 的可重复增益，论文的独特贡献会退化为普通 privileged-to-visual reward distillation。
3. **physical teacher 只表示 pre-contact EEF-object distance。** 这不是完整 manipulation reward，也不能直接覆盖 push、place、grasp、contact 或 long-horizon task progress。

当前审稿判断：**weak reject / borderline only after substantial new evidence**。如果现在按 v5-A 的计划直接投稿，最可能的评审意见是“合理的工程路线，但核心方法和实验尚未完成”。如果完成一个严格的 no-MI versus MI student 对照，并在多个 LIBERO distribution shifts 上证明 MI 带来稳定的 physical-progress 或 action-ranking 增益，评价可以明显改善。

## 3. 预审与投稿适配

### 主题适配

**Status: pass.** 视觉 reward learning、机器人 manipulation、privileged supervision 和 representation/action geometry 属于 ICLR 的主题范围。

### 可评审性

**Status: concern.** 当前输入是 pipeline proposal，不是完整 manuscript。缺少训练结果、统计处理、实现细节和完整 related work，因此只能做证据受限的科学评审。

### 当前年度时间信息

ICLR 2027 官方时间表显示 abstract deadline 为 2026-09-18 AoE、paper deadline 为 2026-09-25 AoE；具体提交规则应以官方页面为准：[ICLR 2027 dates](https://iclr.cc/Conferences/2027/Dates)、[ICLR 2027 call for papers](https://www.iclr.cc/Conferences/2027/CallForPapers)。

### Desk rejection risk

**Risk: medium if submitted in the current form.**

原因不是题目不适合，而是当前文档没有构成可审稿的实验论文：主 student 未实现或未报告结果，且 MI 是否带来方法增益仍未知。可以在投稿前修复，但需要新的训练和 benchmark 结果。

## 4. 摘要与贡献拆解

v5-A 的中心问题是：能否用 simulator 中不可部署的 physical progress 生成 teacher label，训练只依赖视觉 observation、reference 和 candidate action 的 reward model，并测试 MI 是否能作为辅助的 reference-consistency 或 directional regularizer。这个定位在第 0 节和第 3 节明确给出（`v5-a:5-15, 66-120`）。

当前计划包含四个潜在贡献：

1. 用 privileged physical progress 监督 observation-only action-conditioned reward model（`v5-a:68-102`）；
2. 使用 pairwise action ranking 让 reward 输出直接服务于 candidate action selection（`v5-a:124-148`）；
3. 用固定的 action-geometry protocol 评价 MI 和 reward model，而不是只报告 scalar correlation（`v5-a:205-233`）；
4. 检验 MI 是否在 no-MI physical-teacher student 之上提供额外收益（`v5-a:58-64, 190-203`）。

其中第 1、2、3 项是已定义的研究计划，第 4 项才是当前最可能形成独特贡献的部分，但尚无结果支持。

## 5. 主要优势

### S1. 正确分离了 teacher、student 和部署边界

`v5-a:68-120` 明确规定 object pose、EEF-object distance 和 simulator state 只在训练标签、验证或 oracle baseline 中使用，部署时 student 只接收 observation、reference 和 action。这避免了把 privileged oracle 误称为 observation-only reward。

### S2. 对 v5 的负结果处理诚实

`v5-a:28-46` 保留了 full-frame raw-RGB 的真实结果，并明确说明 `0.68842` 未达到 physical gate。这比把高 held-out MI R² 直接包装成 physical reward 成功更可靠。

### S3. Pairwise action ranking 与 reward 使用方式一致

`v5-a:134-148` 不只拟合 state-level scalar，而是要求同一个 state 下更有 physical progress 的 candidate action 得到更高分。这是连接 reward model 与 action selection 的必要设计。

### S4. Benchmark 方向具有可复现性潜力

`v5-a:154-203` 固定了 anchor、reference、candidate actions、held-out split 和 physical oracle 的共享协议。LIBERO 官方提供 Spatial、Object、Goal 等受控 distribution shifts；ManiSkill3 也提供视觉任务、任务卡和 state/visual observation 模式，可作为外部 simulator 验证。[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO datasets](https://libero-project.github.io/datasets), [ManiSkill3 tasks](https://maniskill.readthedocs.io/en/latest/tasks/index.html)

## 6. 主要问题与严重程度

### C1 — MAJOR：v5-A 的核心 student 还没有证据

**位置：** `v5-a:64, 87-152, 215-233`。

**证据：** 文档定义了 (r_\theta(o_t,o_g,a_t))、regression loss、ranking loss 和 MI consistency，但没有训练曲线、held-out reward-model 指标、跨 state 结果或闭环选择结果。

**影响：** 当前只能证明有一个合理的 supervision plan，不能证明 observation-only reward model 学到了 physical progress，更不能证明 MI 改善了它。

**修复条件：** 至少实现并比较两个同架构 student：`physical teacher only` 与 `physical teacher + MI auxiliary`，在未参与训练的 candidate actions 和未见 object/scene split 上报告 progress R²、pairwise ranking accuracy、physical direction cosine 和 one-step distance reduction。

### C2 — MAJOR：MI 的独特贡献尚未存在

**位置：** `v5-a:40, 104-120, 188-203, 235-264`。

**证据：** frozen full-frame raw-RGB MI 的 physical cosine 为 `0.68842`，低于文档预设的 `0.70` gate。v5-A 计划把 MI 降为辅助项，但尚未证明加入它比 no-MI student 更好。

**影响：** 如果 no-MI student 和 with-MI student 相同或后者更差，论文的独特方法贡献会消失，只剩下已知范式的 privileged supervision distillation。Dame–Marchand 已经展示过直接最大化图像 MI 来进行视觉伺服，因而“MI 可导并能形成视觉 servo signal”本身不能作为新颖性主张。[Mutual Information-Based Visual Servoing](https://ieeexplore.ieee.org/document/5772938/)

**修复条件：** ROI 或 reference-conditioned MI 必须在预先冻结的 benchmark 上相对于 no-MI student 带来可重复的 physical-progress、action-ranking 或跨分布泛化收益；若没有收益，应删除 MI 作为核心方法的表述。

### C3 — MAJOR：physical teacher 的任务定义过窄

**位置：** `v5-a:68-85, 154-186`。

**证据：** teacher 只用 EEF 到 task object 的距离变化 (d(s_t)-d(s_{t+1}))。当前 v5 还限定在 smooth pre-contact、EEF `delta xyz` 和两步 rollout。

**影响：** 该标签可能鼓励接近物体，却不能区分正确抓取、错误接触、推动方向、放置目标、遮挡或 long-horizon subgoal。对于 articulated-object 或 Goal suite，EEF-object distance 甚至可能与任务进展相反。

**修复条件：** 将 teacher 明确限定为 `pre-contact approach progress`，或为不同任务定义 task-progress oracle，例如 object-to-goal distance、contact/grasp transition、subgoal completion 和 safety exclusion。不能用 EEF-object distance 直接宣称 manipulation reward。

### C4 — MAJOR：benchmark 仍是计划，不是已定义的可复核协议

**位置：** `v5-a:154-186, 205-233`。

**证据：** 12 个 tasks、3–5 个 initial states、4 个 anchors、ManiSkill3 三类任务都只是数量级描述，未固定 task IDs、demo IDs、random seeds、训练/验证/测试划分、统计单位和 task-level aggregation。

**影响：** 评审者无法判断是否存在 state selection、task selection 或 hyperparameter leakage，也无法复现结果。

**修复条件：** 提供 manifest，固定 task IDs、initial states、anchor selection、candidate seed、exclusion rule、训练/测试 split、模型架构和所有 gate。以 task-level macro-average 和 paired bootstrap confidence interval 报告结果。

### C5 — MAJOR：当前开发 split 存在选择偏差风险

**位置：** `v5-a:164-172, 203`；相关证据见 `MI_Directional_Potential_Research_Idea_v5.md:793-814`。

**证据：** v5 的 9.12 记录说明五个 initial states 已在单-anchor probe 中满足 smooth 条件且 physical direction 为正。v5-A 又把当前 20-anchor collection 设为 development split，但没有定义独立的 held-out state/task split。

**影响：** 即使 student 在当前 20 anchors 上成功，也可能只是在预筛选的 smooth states 上有效，不能支持跨 state 或跨 task 泛化。

**修复条件：** 公开所有 state selection 规则；将 v5 的 20 anchors 只用于开发，将未看过 physical cosine 的 states、objects 和 tasks 作为最终测试集。

### C6 — MAJOR：方法边界没有覆盖完整动作空间

**位置：** `v5-a:87-102, 164-170`。

**证据：** current student 只使用实际 EEF `delta xyz`，而测试阶段固定两步 pre-contact translation。没有 rotation、gripper、contact 或 action duration ablation。

**影响：** 结果只能支持 3D pre-contact translational reward，不足以支持通用 robot manipulation reward model 或原始 WAM 全动作空间主张。

**修复条件：** 当前论文明确收缩为 `3D pre-contact action-conditioned progress model`；或者增加 rotation/gripper/contact 分支，并分别报告 discontinuity 和 object-motion exclusion。

### C7 — MAJOR：没有证明 MI consistency 目标与 physical label 不冲突

**位置：** `v5-a:104-120, 150-152`。

**证据：** 文档允许用 held-out MI gain 或 action-pair ordering 作为辅助目标，但没有定义 (mathcal L_{\text{MI-consistency}})，也没有说明当 MI ordering 与 physical ordering 冲突时如何处理。

**影响：** MI auxiliary 可能把 student 拉向视觉变化而不是 physical progress，重现 v5 full-frame 的失败，只是被回归 loss 部分掩盖。

**修复条件：** 明确定义 MI auxiliary 的输入、target、detach 规则和权重搜索范围；报告 MI/physical ordering agreement，并进行 `lambda_MI=0`、固定小权重和 tuned weight 的消融。

## 7. 次要问题与表达风险

1. `v5-a:70` 中先写 `(d_t)`，随后使用 `d(s_t)`；应统一距离变量定义和 notation。
2. `v5-a:76-83` 的 (u_{\text{phys}}) argmax 没有明确 action ball、时间步长和正负 probe 到向量场的拟合方式；应给出可直接执行的算法。
3. “reward model”同时指 (r_\theta(o_t,o_g,a_t)) 和 MI scalar (M_\phi(o_t,o_g)) 的风险仍然存在；全文应固定称呼为 `physical-teacher student`、`MI auxiliary` 和 `oracle`。
4. `v5-a:176-182` 说 ManiSkill3 不重新调 gate，但不同 simulator 的 action scale、camera model 和 task progress 定义可能不同；应预先规定哪些参数跨环境固定，哪些是环境单位转换。
5. `v5-a:184-186` 的真机协议只有概念描述，没有硬件、相机、标定误差、object pose 估计误差和安全停止规则；应作为 optional validation，而不是当前主结果。

## 8. 新颖性与相关工作定位

| Work / source | What it already establishes | Overlap and remaining difference | Consequence / concern ID |
|---|---|---|---|
| Dame & Marchand, *Mutual Information-Based Visual Servoing* | MI can be used as an image-based visual feature and control signal for visual servoing; the original work reports six-DoF control and real-robot experiments | v5-A does not yet show a new MI control law; its possible difference is physical-teacher distillation and a stricter action-geometry benchmark | C2 |
| Privileged sensing / teacher-student RL, e.g. *Privileged Sensing Scaffolds Reinforcement Learning* | Privileged signals can supervise or scaffold an observation-limited learner during training | v5-A applies this idea to an action-conditioned visual reward model and may add MI as an auxiliary signal, but the incremental effect is unmeasured | C1, C2 |
| LIBERO | Standardized suites with controlled spatial/object/goal distribution shifts | v5-A can contribute a counterfactual reward/action-geometry protocol, but using LIBERO alone is not a methodological novelty | C4, C5 |
| ManiSkill3 | Large visual manipulation simulation and task/state infrastructure | External engine validation could strengthen generality, but it is not yet executed | C4 |

Dame–Marchand 的工作表明，不能把“MI 经过 Jacobian 求导后可以用于视觉伺服”作为 v5-A 的核心 novelty；这一点在原始 TRO 已经建立。[TRO record](https://ieeexplore.ieee.org/document/5772938/)

## 9. 方法正确性与主张支撑

| Claim / location | Inspected support | Judgment | Consequence / concern ID |
|---|---|---|---|
| privileged physical progress can supervise an observation-only reward model (`v5-a:68-102`) | Defined label (d(s_t)-d(s_{t+1})), student inputs, and deployment boundary | Plausible, but no student result and task-progress scope is narrow | C1, C3 |
| pairwise ranking corresponds to action selection (`v5-a:124-148`) | Same-state candidate action ordering is explicitly defined | Methodologically sound, pending held-out and closed-loop evidence | C1 |
| MI can be a useful auxiliary signal (`v5-a:48-64, 104-120`) | Current MI self-consistency is strong, but physical cosine is below gate | Hypothesis only; no positive support for reward improvement yet | C2, C7 |
| LIBERO counterfactual benchmark supports generalization (`v5-a:154-182`) | LIBERO provides multiple task suites and controlled shifts; current v5 has one-task 20-anchor evidence | Suitable infrastructure, incomplete benchmark specification and coverage | C4, C5 |
| Hessian/planner should be optional and late (`v5-a:235-264`) | Follows from current physical gate failure | Correct scope control | No concern |

## 10. 实验、证明与可复核性

### 已有证据

v5 已提供有价值的 teacher-level diagnostics: gradient/direct-FD consistency, cross-epsilon stability, held-out MI prediction and physical-direction failure. 这些结果支持“visual MI field 可以高度自洽但不必然物理正确”这一诊断。

### 缺失的决定性证据

1. no-MI physical-teacher student 与 MI-auxiliary student 的同架构对照；
2. action-conditioned reward output 的 held-out ranking；
3. 未见 state、object instance 和 task 的泛化；
4. one-step candidate selection 与闭环 success；
5. 多 seed 或 task-level confidence interval；
6. 物理 teacher 对 push/place/grasp/goal 的任务定义；
7. MI auxiliary 与 physical progress 冲突时的训练行为。

因此当前可复核性是“v5 teacher diagnostic 较好、v5-A student 尚未可复核”。

## 11. 多视角评审与综合意见

以下是单一 agent 基于同一材料的三种角色视角，不是三个独立审稿人的实证共识。

| View | Decisive basis / concern IDs | Stance | What would change it |
|---|---|---|---|
| Best-supported assessment | v5-A 的 teacher/student 边界清楚，v5 负结果诚实，但 student 和 MI 增益没有结果 | Weak reject for submission now | 完成 no-MI vs MI student、跨 task split 和闭环/one-step evidence |
| Strongest favorable reading | 这是一个有潜力的 privileged-to-observation reward protocol，并且 v5 已发现 MI self-consistency 与 physical correctness 的分离 | Borderline if presented as a benchmark/protocol paper | 证明该 protocol 暴露出稳定、可复现、可修复的 MI failure mode，并补足 multiple tasks |
| Strongest critical reading | Privileged distance supervision 是已知范式，MI 目前只增加复杂度且 full-frame physical gate 失败 | Reject if submitted as an MI reward-model method | MI 必须在固定 benchmark 上给 no-MI student 带来显著且可泛化的增益；否则删去 MI 主张 |

**Agreement:** 当前最重要的缺口是实际 reward-model evidence，而不是再推导 Hessian。

**Disagreement:** favorable reading 认为 benchmark/protocol 可能成为贡献；critical reading 认为没有 MI 增益时只剩常规 distillation。

**Decisive accept axis:** MI auxiliary 是否相对于 no-MI student 带来跨分布的 physical-progress/action-ranking improvement。

**Decisive reject axis:** student 只在预筛选 smooth states 上有效，或 MI consistency 提升但 physical progress 不提升。

## 12. 维度评分与置信度

## Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 2 | 3 | `v5-a:5-15, 48-64`; Dame–Marchand 已建立 MI visual servoing；privileged teacher范式已有先例 | 当前没有证明 MI auxiliary 的增量贡献；在多个 split 上优于 no-MI student 后可升至 3–4 |
| Soundness | 3 | 3 | `v5-a:68-152` 的 teacher/student 数学定义清楚；physical label 对完整 manipulation 不充分 | 明确 task-progress oracle、动作空间和 MI loss 后可升至 4 |
| Evidence | 2 | 4 | `v5-a:17-46` 有 v5 实证，但 `v5-a:87-233` 的 student/benchmark 结果缺失 | 完成实际 student、held-out、跨任务和闭环结果后可升至 4 |
| Significance | 3 | 3 | `v5-a:48-64, 235-264` 对“视觉一致性不等于物理进展”有潜在价值 | 需要证明该区分改变 reward-model 设计或带来可泛化收益 |
| Clarity | 4 | 4 | `v5-a:0-10, 66-120, 235-264` 的角色边界和停止条件清楚 | 统一 teacher/reward/MI 术语并补全算法细节后可升至 5 |
| Reproducibility | 3 | 3 | `v5-a:154-203` 给出 candidate、anchor 和 baseline 方向；未固定完整 manifests、seeds、student architecture | 发布 manifests、代码、训练配置和 task-level statistics 后可升至 4 |
| Ethics / Limitations | 3 | 3 | `v5-a:40-46, 184-186, 251-264` 承认 physical gate、真机和部署限制 | 加入仿真到真机风险、失败动作和安全停止说明后可升至 4 |

**Overall:** 4 / 10  | **Scholarly Confidence:** 3 / 5

**Recommendation:** weak-reject for an ICLR submission in the current evidence state

**Verdict:** v5-A 是合理的研究计划，但当前更像“待执行的 reward-model protocol”，不是已经被实验支持的方法论文。若完成 no-MI vs MI student、未见分布测试和 one-step/closed-loop action selection，整体可上调；若 MI 不带来 physical progress 或 ranking 增益，应删除 MI 方法主张，改成 privileged-to-observation reward distillation 或 benchmark/diagnosis paper。

## 13. 作者关键问题与改判条件

1. `physical-teacher student without MI` 和 `with MI auxiliary` 是否使用完全相同的 backbone、参数量、训练步数和数据量？如果不是，MI gain 无法归因。
2. physical teacher 是否只定义 pre-contact approach？如果是，论文是否会明确放弃通用 manipulation reward 的主张？
3. 最终测试 states、objects 和 tasks 是否在所有 ROI/reference variant 选择完成前冻结？
4. MI auxiliary 与 physical teacher 排序冲突的比例是多少？冲突时是否会损害 student？
5. 12-task benchmark 是否有固定 manifest、task-level aggregation 和多 seed 统计？
6. 在闭环 candidate selection 中，reward model 是否真的选择了更好的 action，还是只提高离线 ranking 指标？

## 14. 修改优先级与复审记录

| ID | Priority / severity | Required change | Why it matters | Status / version |
|---|---|---|---|---|
| C1 | P0 / MAJOR | 实现并报告 no-MI 与 MI-auxiliary reward students | 证明 v5-A 是 reward model，而不是计划 | Open / v5-A |
| C2 | P0 / MAJOR | 在冻结 split 上证明 MI 相对 no-MI 的 physical/ranking 泛化增益 | 决定 MI 是否仍是方法贡献 | Open / v5-A |
| C3 | P0 / MAJOR | 将 physical teacher 限定为 pre-contact 或扩展到 task-specific progress | 防止把距离标签过度解释为 manipulation reward | Open / v5-A |
| C4 | P0 / MAJOR | 固定 task/state/anchor manifest、seed、split 和统计方式 | 使 benchmark 可复核 | Open / v5-A |
| C5 | P0 / MAJOR | 使用未看过 physical cosine 的最终测试 split | 排除 state selection bias | Open / v5-A |
| C6 | P1 / MAJOR | 明确 3D translation scope，或加入 rotation/gripper/contact 实验 | 防止通用动作空间过度主张 | Open / v5-A |
| C7 | P1 / MAJOR | 定义 MI consistency loss、权重和冲突分析 | 防止 MI 重新破坏 physical supervision | Open / v5-A |
| M1 | P1 / MINOR | 统一 distance notation 和 (u_{phys}) 的 action-ball 定义 | 提升数学可复核性 | Open / v5-A |
| M2 | P1 / MINOR | 将 ManiSkill3 和真机标为外部验证，补充跨环境单位转换 | 避免外部 benchmark 计划不具可执行性 | Open / v5-A |

### Score-change conditions

| Change | Condition | Likely affected dimensions | Expected movement |
|---|---|---|---|
| Raise score | MI-auxiliary student 在固定跨 task/object split 上显著优于 no-MI student，并改善 physical progress 和 action ranking | Novelty, Evidence, Significance | +1 to +2 overall |
| Raise score | physical teacher 扩展为 task-specific progress，且闭环 candidate selection 有收益 | Soundness, Significance, Evidence | +1 overall |
| Lower score | student 只在开发 anchors 上有效，或 MI 提高视觉一致性却降低 physical progress | Evidence, Soundness | -1 to -2 overall |
| No quick change | 没有完成 reward-model training 和 benchmark，无法靠改写文字修复 | Evidence, Novelty | No meaningful improvement before submission |

