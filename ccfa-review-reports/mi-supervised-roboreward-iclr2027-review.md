# MI-Supervised RoboReward：ICLR 2027 科学评审

**评审日期：** 2026-09-10  
**目标会议：** ICLR 2027，主会  
**评审模式：** Scientific review  
**评审对象：** 当前会话中提出的最新方案：使用 Dame/TRO 风格 MI 对 WAM 生成的候选轨迹进行视觉对齐排序，以排序或离散分数作为伪标签，加入夹爪与物体结果监督，SFT Qwen3-VL 得到 RoboReward，并作为 WorldSample 的合成轨迹 reward labeler。  
**已检查材料：** [Pipeline v7](../MI_Directional_Potential_Research_Idea_v7.md)、[Dame and Marchand TRO](../Mutual_Information-Based_Visual_Servoing.pdf)、v7 G0/G1 数值结果，以及公开的 RoboReward、WorldSample、TrustRoboReward、TOPReward 方法说明。  
**覆盖限制：** 最新方案尚未写入独立方法稿，没有对应实现、数据集或结果；因此评分反映当前论文证据成熟度，而不是未来实现成功后的上限。

## 1. 总体结论

**当前建议：Reject，3/10。**

这个方案已经从被 v7 否定的“MI 直接表征通用物理进度与动作方向”转向一个逻辑上更窄、也更可检验的问题：MI 是否能作为无人工标注的视觉对齐教师，为 reward VLM 生成 ordinal/pairwise supervision。这一转向保留了 MI 的合理用途，也与 RoboReward 和 WorldSample 的数据接口兼容。

当前仍有一个决定性阻断项：新方法依赖的核心标签假设尚未得到任何直接证据。v7 G1 表明当前 Dame-style MI 在真实 fixed-identity geometric correspondence 上与物理进度无关且方向相反，但最新方案尚未完成严格的相机/末端视觉对齐排序实验。因此，现在既不能声称 MI 标签有效，也不能声称蒸馏后的 Qwen3-VL reward 优于原始 RoboReward。若立即投稿，论文会表现为“已知模块的组合，加上尚未验证的伪标签来源”。

## 2. 预审与投稿适配

| 检查项 | 状态 | 依据与影响 |
|---|---|---|
| ICLR 主题适配 | Pass | reward learning、multimodal representation、robot learning 和 synthetic supervision 属于 ICLR 范围。 |
| 科学可评审性 | Concern | 最新方法没有正式算法定义、数据构造协议和实验结果，评审无法核验主张。 |
| 最低完整度 | High risk | 当前 v7 是 falsification pipeline，不是最新 RoboReward 方法稿。 |
| 篇幅、匿名、模板 | Not assessed | 未提供投稿 PDF 或 LaTeX。 |
| 伦理与安全 | Uncertain | world-model hallucination、自动伪标签偏差和真实机器人安全边界尚未讨论。 |

**Desk-reject risk：中等。** 主题合适，但如果以当前材料提交，贡献和证据对应关系不完整，可能在正常评审中快速形成明确拒稿意见。

## 3. 论文主张与贡献拆解

最新方案可以重构为四个主张：

1. 对同一初始状态和同一目标，TRO-style MI 能可靠排序 WAM 候选轨迹的末端视觉对齐程度。
2. 该排序能自动生成 near-miss 的 pairwise/ordinal reward labels，降低 RoboReward 对人工或启发式标注的依赖。
3. 夹爪事件与物体结果监督能够补充 MI 看不到的接触和任务语义。
4. Qwen3-VL 蒸馏这些标签后，能更准确地评价新轨迹，并改善 WorldSample 的合成轨迹选择或下游策略学习。

其中第 1 项是机制前提，第 2 项是数据构造贡献，第 3 项是 reward decomposition，第 4 项是最终价值。当前只有“Qwen3-VL 可被训练为 RoboReward”和“WorldSample 使用 reward model 标注合成轨迹”得到公开工作支持；新方案自己的四项主张均没有直接结果。

## 4. 主要优势

**S1：问题具有实际价值。** RoboReward 需要区分失败、near-miss 和完成状态；自动产生细粒度、可排序标签可降低 reward dataset 的构造成本，并可能改善合成轨迹利用。

**S2：v7 提供了可信的负证据。** G0 证明数据和 evaluator 自洽，G1 在 20 anchors、480 held-out actions 上排除了 RGB、LaWAM、预测器和伪造 token 等混杂因素。这种逐级 falsification 比不断更换表示更有科学价值。

**S3：MI 的角色已经显著收窄。** 将 MI 限定为视觉对齐子奖励，而不是通用物理进度，避免了当前结果已经暴露的概念过度扩张。

**S4：下游接口清楚。** 轨迹视频、任务文本、目标图像和夹爪状态可以作为 Qwen3-VL 输入，输出离散奖励或 pairwise preference；该 reward model 可以替换 WorldSample 的合成轨迹 labeler。

## 5. 主要问题

### C1：MI 伪标签的有效性尚未成立——严重

**涉及主张：** “TRO-style MI 可以排序候选轨迹的视觉势能。”  
**证据：** v7 G1 的 held-out \(R^2=0.0002\)、Spearman \(-0.1934\)、sign accuracy \(0.3438\)，MI/physical-descent cosine median 为 \(-0.6690\)。  
**影响：** 当前可检查证据支持的是“现有 MI objective 不对应物理进度”，而不是“MI 能产生可靠 reward labels”。Qwen3-VL SFT 只能蒸馏教师标签，不能自动修正教师的系统性错误。  
**改判条件：** 完成严格的 6D camera/EEF alignment ordering test，并用独立的 \(SE(3)\) pose error 或人工 preference 证明 MI pairwise ordering 有效。

### C2：当前新颖性主要来自模块组合——严重

**涉及主张：** “MI→Qwen3-VL reward distillation→WorldSample”构成主要方法创新。  
**证据：** Dame TRO 已提供 MI 图像对齐及 6-DOF 控制；RoboReward 已使用 Qwen3-VL 学习 1–5 轨迹奖励；WorldSample 已使用 reward model 标注 world-model 轨迹；TrustRoboReward 已研究 pointwise/pairwise reward consistency；TOPReward 已从 Qwen3-VL 构造机器人时序 reward。  
**影响：** 简单串联这些组件不足以形成 ICLR 级方法差异。  
**改判条件：** 把贡献固定为一个尚未由最近工作解决的机制，例如“跨场景可校准的解析视觉对齐教师”“无人工 near-miss ordinal labeling”或“MI、夹爪事件和物体结果的可验证 reward decomposition”，并证明它解决现有 reward VLM 的具体失败模式。

### C3：存在循环评价风险——严重

**涉及主张：** “Qwen3-VL 成功学习了更好的 reward。”  
**依据：** 如果训练标签和测试标签都由 MI 产生，结果只能证明模型拟合 MI，不能证明 reward 更接近任务质量。  
**影响：** 这是足以否定主要实验结论的评价设计风险。  
**改判条件：** 使用独立的人工标签、simulator privileged outcome、真实 success/near-miss 标签和下游 policy learning 结果评价，MI 只参与训练标签生成。

### C4：第三视角设置与 Dame TRO 的观测几何不同——严重

**涉及主张：** “可以直接仿照 TRO 对第三视角 RGB 使用 MI。”  
**依据：** Dame 的 eye-in-hand 图像随相机/末端位姿整体变化；固定第三视角中桌面背景保持不变，末端和物体只占局部区域。  
**影响：** whole-frame MI 可能主要衡量背景，而不是末端对齐。此问题也使 raw MI 在不同实例和场景间不可直接比较。  
**改判条件：** 明确使用 wrist view、task ROI、spatial correspondence 或经校准的局部 MI，并报告 whole-frame/ROI/wrist-view 消融及跨场景校准。

### C5：夹爪状态不能单独补齐物理结果——中等

**涉及主张：** “MI 加夹爪状态即可形成完整 manipulation reward。”  
**依据：** 相同的闭合命令可能对应空抓、稳定抓取或物体滑落；这些状态需要物体运动或接触结果才能区分。  
**影响：** 对 reaching 成立的 reward decomposition 不会自动扩展到 grasping、insertion 和 contact-rich tasks。  
**改判条件：** 将 reward 明确拆成 visual alignment、gripper event 和 object/contact outcome 三部分，并限定每项可支持的任务范围。

### C6：1–5 SFT 丢失 ordinal 结构——中等

**涉及主张：** “将 MI rank 分箱后直接 SFT 就能保留排序。”  
**依据：** 普通 token cross-entropy 把标签作为离散输出处理，并不天然编码 1 与 2 比 1 与 5 更接近；近期 reward-model 工作也直接关注 pointwise 与 pairwise 的次序冲突。  
**影响：** rank-to-bin 可能引入阈值敏感和次序反转。  
**改判条件：** 保留 pairwise preference loss 或 ordinal loss，并同时评价 pointwise MAE、pairwise accuracy 和跨任务校准。

## 6. 次要问题与表达风险

1. “MI 势能”“动力学判断”“物理进度”“视觉对齐”在当前讨论中多次互换。方法稿必须只保留可操作定义；建议把 MI 输出称为 `visual alignment score`。
2. 最新主线尚未写入 v7。v7 的标题、研究问题、成功标准和结论仍围绕 directional physical progress，无法作为新论文方法描述。
3. 应明确训练时是否提供目标图像、任务文本、短视频、夹爪状态和 world-model uncertainty；这些输入会改变方法主张。
4. 必须定义 raw MI 是仅做同一 anchor 内排序，还是跨场景标定。当前只有组内排序在概念上较安全。

## 7. 新颖性与相关工作

| 工作 | 已有机制 | 与最新方案的重叠及剩余差异 | 评审影响 |
|---|---|---|---|
| [Dame & Marchand, TRO 2011](../Mutual_Information-Based_Visual_Servoing.pdf) | MI 图像对齐、梯度、Hessian、6-DOF camera control | 已覆盖 MI 作为视觉对齐目标；未覆盖 reward-VLM 伪标签蒸馏 | MI 本身不是新意，C2 |
| [RoboReward](https://arxiv.org/abs/2601.00675) | Qwen3-VL 4B/8B 预测 1–5 rollout reward；构造 negatives/near-misses | 最新方案拟改变标签来源，使用 MI 产生 ordinal/pairwise supervision | 必须优于其标签构造，C2/C3 |
| [WorldSample](https://arxiv.org/abs/2607.02431) | world model 生成轨迹，reward model 标注，PPL 选择和调度 | 最新方案可替换 reward labeler，但该接口本身已存在 | 下游接入不是独立创新，C2 |
| [TrustRoboReward](https://arxiv.org/abs/2608.08491) | 处理 pointwise/pairwise reward 次序冲突 | 与 MI 排序后分箱高度相关；MI 可能成为新的 preference 来源 | 需要处理 ordinal consistency，C6 |
| [TOPReward](https://arxiv.org/abs/2602.19313) | 从 Qwen3-VL token probabilities 获得时序 reward | 同样面向低成本 VLM reward，但信号来源不同 | 需要证明 MI teacher 的独特优势，C2 |

检索覆盖为“部分搜索”：已覆盖当前最直接的用户指定工作及近期 reward-VLM 方法，但没有完成所有 synthetic-label、trajectory-preference 和 visual-servo distillation 文献的系统检索。因此 3/5 的新颖性评分是暂定判断。

## 8. 方法正确性与主张支撑

| 主张 | 当前支持 | 判断 | 关联问题 |
|---|---|---|---|
| 当前 evaluator 和 physical-progress 标签正确 | G0 所有指标为 1.0 | 支持充分，但只属于 plumbing sanity | — |
| 当前 MI 表示物理进度或方向 | G1 所有主要 gate 失败，方向 cosine 为负 | 被当前证据否定 | C1 |
| 严格 TRO-style MI 能排序 6D 视觉对齐 | 尚未进行 camera/EEF pose ordering test | 未验证 | C1/C4 |
| MI rank 能产生可靠 RoboReward 标签 | 尚无独立标签比较 | 未验证 | C1/C3 |
| Qwen3-VL 能作为机器人 reward model | RoboReward 公开工作支持工程可行性 | 可行，但不支持本方法增益 | C2 |
| 新 reward model 能改善 WorldSample | 只有接口层兼容性 | 未验证 | C3 |

## 9. 实验与可复核性判断

v7 的 falsification 部分具有良好可复核性：冻结了 anchors/actions、明确了 privileged/deployable 标记、保存 JSON 和日志，并报告了多个相关和方向指标。它是当前材料最强的部分。

最新方案需要按顺序通过三个独立 gates：

1. **Label validity gate：** 在真实 \(SE(3)\) pose error 上评价 MI 的 pairwise accuracy、Spearman/Kendall、top-k precision，并与 RGB-L2、SSIM、LPIPS、DINO/VIP 和直接 pose distance 比较。
2. **Reward distillation gate：** 比较原始 RoboReward SFT、MI-only labels、gripper/object labels、联合 labels，以及 pairwise/ordinal loss；测试数据必须使用独立人工或 privileged 标签。
3. **Downstream utility gate：** 在相同 synthetic-data budget 下比较 WorldSample/PPL baseline 与加入新 reward model 后的 success、critic bias、样本效率和 hallucinated-transition admission rate。

在 Gate 1 通过前训练 Qwen3-VL，会把计算资源投入到尚未确认的伪标签上。

## 10. 单一评审者的三视角综合

| 视角 | 决定性依据 | 立场 | 改判条件 |
|---|---|---|---|
| 最有证据支持的判断 | v7 否定 broad MI reward；新窄命题无直接结果，C1–C3 | Reject | 完成独立 label-validity 和 distillation 证据 |
| 最有利解释 | 解析 MI teacher 可低成本生成 near-miss preferences，并由 VLM 泛化到新场景 | 有发展潜力 | 证明跨实例排序优于通用视觉相似度 baseline |
| 最强批评 | 这是 Dame、RoboReward、WorldSample 和近期 ranking 方法的组合，Qwen 只会复制有偏 MI 标签 | 当前足以拒稿 | 提供新机制、独立真值和下游因果增益 |

综合来看，问题价值和 v7 的诊断质量值得保留，但最新方法尚处于“可检验假设”而不是“得到证据支持的论文贡献”阶段。

## 11. 维度评分与置信度

| 维度 | 分数（1–5） | 置信度（1–5） | 依据 | 扣分与改判条件 |
|---|---:|---:|---|---|
| Novelty | 3 | 3 | Dame、RoboReward、WorldSample、TrustRoboReward、TOPReward 比较 | 组件组合明显；若形成跨场景可校准的解析 preference teacher 并区别于近期方法，可升至 4 |
| Soundness | 2 | 4 | v7 G1 与最新标签假设，C1/C4/C5 | 核心标签有效性未成立；严格 6D alignment test 和 reward decomposition 通过后可升至 3–4 |
| Evidence | 1 | 5 | 最新方法没有直接实验，C1/C3 | 需要独立真值、蒸馏和下游三层证据 |
| Significance | 4 | 3 | 自动 reward labeling 与 synthetic trajectory filtering | 问题重要；若只适用于固定场景 reaching，则降至 3 |
| Clarity | 3 | 4 | v7 与当前会话中的主线变化 | 需冻结“visual alignment teacher”定义并重写主张边界 |
| Reproducibility | 3 | 4 | v7 协议和日志较完整；新方法无规范 | 发布新数据构造、MI estimator、分箱/排序和训练协议后可升至 4 |
| Ethics / Limitations | 3 | 2 | 尚未形成完整限制讨论 | 需讨论伪标签偏差、world-model hallucination 和真实机器人安全使用边界 |

**Overall：3/10**  
**Scholarly Confidence：2/5**  
**Recommendation：Reject**

整体分数由 Evidence=1 和 Soundness=2 决定。它们不是因为实验数量少，而是因为最新方法唯一决定性的因果链——“MI 排序有效，因此 MI 标签能改善 reward VLM”——目前没有直接支持，而且已有 v7 结果对更广泛版本给出了反证。

## 12. 改判条件

| 变化 | 条件 | 受影响维度 | 预期变化 |
|---|---|---|---|
| 提高分数 | MI 在独立 6D pose/alignment 真值上稳定优于随机及基础视觉相似度，并跨场景保持排序 | Soundness、Evidence | Overall 可能提高 1 分 |
| 提高分数 | MI/pairwise 标签训练的 RoboReward 在独立 reward benchmark 上优于原始 RoboReward、同量随机/启发式标签和无 MI 版本 | Novelty、Evidence | Overall 可能提高 1 分 |
| 提高分数 | 下游 WorldSample 在固定数据预算下获得稳定策略增益，并由消融归因到 MI teacher | Significance、Evidence | 从拒稿边界进入 borderline 的必要条件之一 |
| 降低分数 | 严格 alignment gate 仍失败，或仅 whole-frame fixed-background 条件有效 | Soundness、Significance | 降至 2/10，当前 rescue route 基本终止 |
| 无法快速改变 | 仅把 Qwen3-VL 换成更大模型、增加 SFT epoch 或扩大 MI 数据量 | Novelty、Soundness | 不改变核心判断 |

## 13. 修改优先级

| ID | 优先级 / 严重度 | 必须完成的改变 | 原因 | 状态 |
|---|---|---|---|---|
| C1 | P0 / 严重 | 先完成严格 6D camera/EEF alignment ordering test | 决定 MI 是否能成为标签教师 | 未解决 |
| C3 | P0 / 严重 | 建立与 MI 无关的 reward 测试真值 | 避免循环证明 | 未解决 |
| C2 | P1 / 严重 | 明确区别于 RoboReward、TrustRoboReward、TOPReward 的唯一机制贡献 | 决定 ICLR 新颖性 | 未解决 |
| C4 | P1 / 严重 | 固定 wrist/third-view/ROI 设定并验证背景与场景变化 | 决定 TRO 迁移是否成立 | 未解决 |
| C5 | P1 / 中等 | 将 gripper、object/contact outcome 与 MI alignment 分解 | 避免把视觉对齐误称完整任务 reward | 未解决 |
| C6 | P2 / 中等 | 使用 pairwise/ordinal objective 并检查与 1–5 score 的一致性 | 保留排序监督 | 未解决 |

当前最值得执行的单一步骤是 C1。C1 失败则无需训练 Qwen3-VL；C1 通过后，论文才拥有从 v7 负结果转向 MI reward distillation 的可信起点。
