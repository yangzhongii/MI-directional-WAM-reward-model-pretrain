# Pipeline v6：Action-Relevant LaWAM Latent + MI Direction

## 0. 本版本要解决的问题

Pipeline v5 的正式结果表明：当前 raw-RGB 和 LaWAM visual latent 都能较好地预测自身的 MI 变化，但跨初始状态的 physical-direction gate 没有通过。因此，当前问题不能继续解释为“只差一个 Hessian”。

v6 验证一个更具体的假设：

> LaWAM 当前输出的是静态视觉表示，而物理方向需要一个保留空间结构、动作影响和局部状态转移的 controllable latent。

本版本不是重新证明 LaWAM 一般有效，而是对 LaWAM 增加一个最小的 action-relevant latent 分支，检查 MI 是否能在该分支上恢复物理方向。

本版本暂时不做：

- 不微调整个 DINO/LaWAM backbone；
- 不直接把 LAM action latent 与 goal latent 计算 MI；
- 不启动 Hessian、Newton action proposal 或闭环 RL；
- 不使用 privileged object pose 训练主分支；
- 不把实验结果提前解释为新的 reward model。

只有一阶 action-conditioned MI 同时通过 held-out 和 physical-direction gates 后，才允许继续二阶实验。

---

## 1. 研究问题

### Q1：当前失败是否主要来自 latent encoding？

当前 v5 使用 LaWAM 的 DINOv3 penultimate-layer visual patch tokens，并在 MI estimator 中进行 channelwise aggregation。它们具有视觉语义和局部 patch 结构，但没有保证每个维度都能表达：

1. 末端执行器动作；
2. 物体与目标之间的相对关系；
3. 动作后状态的变化；
4. 任务进度方向。

### Q2：加入动作条件后，MI 是否能表达物理方向？

定义视觉表示：

\[
Z_t=E_{\mathrm{LaWAM}}(o_t)\in\mathbb{R}^{K\times D}.
\]

定义机器人动作：

\[
a_t=[\Delta x,\Delta y,\Delta z,\Delta g].
\]

增加一个动作条件 latent dynamics model：

\[
z_t^c=P_\theta(Z_t),
\]

\[
\hat z_{t+1}^c=F_\theta(z_t^c,E_a(a_t)).
\]

其中 \(z_t^c\) 是 controllable visual latent，\(F_\theta\) 预测动作之后的 latent，而不是直接预测物理距离。

MI objective 改为：

\[
M_t(a)=MI(\hat z_{t+1}^c(a),z_g^c),
\]

\[
\Delta M_t(a)=M_t(a)-M_t(0).
\]

最终验证：

\[
\nabla_a M_t(a)
\quad\text{是否与 privileged physical distance-descent direction 对齐。}
\]

### Q3：LAM action latent 应该放在哪里？

当前 `extract_action_latents()` 从观测帧对中推断已发生的 transition latent。它描述的是已经发生的视觉变化，因此不能直接替代候选动作 \(a_t\) 来做规划。

v6 中 LAM action latent 只作为可选的 transition auxiliary target 或诊断分支：

\[
q_t^{\mathrm{LAM}}=E_{\mathrm{LAM\text{-}action}}(o_t,o_{t+1}).
\]

主规划分支仍然必须使用候选动作编码 \(E_a(a_t)\)，否则在执行动作之前无法评价候选动作。

---

## 2. 已有研究依据

v6 不把“action-relevant latent”当作新的空泛概念，而是组合已有的几类表示学习路线。

### 2.1 几何与关键点 latent

- **Deep Spatial Autoencoders for Visuomotor Learning**：学习可用于闭环 visuomotor control 的空间 feature points。
  [论文](https://arxiv.org/abs/1509.06113)
- **Unsupervised Learning of Visual 3D Keypoints for Control**：从多视角图像中学习具有 3D 几何意义的 keypoints，用于机器人控制。
  [ICML/PMLR](https://proceedings.mlr.press/v139/chen21b.html)
- **Dense Object Nets**：学习用于机器人操作的 dense object descriptors，支持 instance-level 和 category-level 对应关系。
  [CoRL/PMLR](https://proceedings.mlr.press/v87/florence18a.html)
- **Transporter Networks**：保持空间结构，把 manipulation 建模为局部区域的 spatial displacement。
  [CoRL/PMLR](https://proceedings.mlr.press/v155/zeng21a.html)

这类工作说明，物理方向通常更容易在具有坐标、对应关系或位移语义的 latent 中表达，而不是在未筛选的通用语义 channel 上直接表达。

### 2.2 任务进度和 reward latent

- **Time-Contrastive Networks**：学习对任务阶段敏感、对视角变化更鲁棒的表示。
  [CVPR workshop](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Sermanet_Time-Contrastive_Networks_Self-Supervised_CVPR_2017_paper.html)
- **Visual Task Progress Estimation**：明确学习用于估计任务 progress/phase 的 embedding。
  [论文](https://arxiv.org/abs/2003.06977)
- **VIP**：用 goal-conditioned value objective 学习可以产生 dense visual reward 的表示。
  [OpenReview](https://openreview.net/pdf?id=YCETybILpw)
- **R3M**：通过时间对比、视频语言对齐和稀疏性约束学习机器人 manipulation 表示。
  [CoRL/PMLR](https://proceedings.mlr.press/v205/nair23a.html)

这些表示适合作为 reward/progress baseline，但不能直接假设它们能够产生可靠的 3D action direction。

### 2.3 Action-conditioned latent dynamics

- **Learning Visual Servoing with Deep Features and Fitted Q-Iteration**：联合使用 learned visual features、predictive dynamics 和 task-specific feature selection，而不是直接把任意视觉 feature 当作 servo feature。
  [论文](https://arxiv.org/abs/1703.11000)
- **SOLD: Slot Object-Centric Latent Dynamics Models**：在 object-centric latent 中学习动作条件的未来状态，并用于 relational manipulation。
  [项目与论文](https://slot-latent-dynamics.github.io/)

这两类工作直接支持 v6 的核心修改：视觉 latent 必须与动作影响或未来状态预测连接起来。

---

## 3. 当前 LaWAM 的具体问题定位

当前代码路径是：

```text
RGB image
  -> LaWAM/DINOv3 extract_vision_features
  -> [K, D] patch tokens
  -> channelwise Dame MI
  -> visual Jacobian / action pullback
```

当前配置使用 `latent_layer_to_use=-2`、`norm_latents=true` 和 per-token LayerNorm。LaWAM 的 action latent 路径虽然存在，但 v5 evaluator 没有调用它。

因此 v5 实际验证的是：

\[
MI(E_v(o_t),E_v(o_g)),
\]

而不是：

\[
MI(F(E_v(o_t),a_t),E_v(o_g)).
\]

这两个问题不是同一个问题。

此外，当前 Dame MI 的 channelwise 模式会对所有 channel 的 MI 做聚合。它保留了 token 维度，但没有显式学习：

- 哪些 token 属于被操作物体；
- 哪些 token 属于末端执行器；
- 哪些 token 对动作变化敏感；
- 哪些 channel 能预测下一时刻状态。

---

## 4. v6 的模型修改

### 4.1 保留 LaWAM backbone

第一阶段冻结现有 LaWAM/DINOv3 backbone：

\[
Z_t=E_{\mathrm{LaWAM}}(o_t).
\]

这样可以避免把失败原因混淆为整个 backbone 被重新训练后改变了分布。

### 4.2 增加 spatial control projection

增加一个低容量投影头：

\[
z_t^c=P_\theta(Z_t).
\]

第一版建议：

- 保留 token 的二维空间位置；
- 使用 masked/attention pooling，而不是直接 flatten 全部 token；
- 使用低秩线性层或小型 MLP；
- 不使用 privileged object pose 作为输入；
- 不先做全量 transformer fusion。

推荐的最小形式：

\[
z_t^c=\operatorname{Pool}_{\theta}(Z_t)\in\mathbb{R}^{d_c},
\qquad d_c\ll K D.
\]

第一轮只改变表示聚合，不改变 LaWAM 主干。

实现约束：若 \(P_\theta\) 与 \(F_\theta\) 只通过下一 latent prediction 联合训练，\(P_\theta\equiv0\) 是平凡解。Test 1 的最小实现因此固定使用无仿射 LayerNorm 与逐行单位范数 channel projection，排除全零/缩放塌缩；该约束只作用于视觉 token，不使用 privileged pose 或 distance。

### 4.3 增加 action-conditioned latent dynamics

动作编码器接收候选 EEF displacement：

\[
e_a=E_a([\Delta x,\Delta y,\Delta z,\Delta g]).
\]

预测下一时刻 controllable latent：

\[
\hat z_{t+1}^c=F_\theta(z_t^c,e_a).
\]

第一版使用 residual 形式：

\[
\hat z_{t+1}^c=z_t^c+G_\theta(z_t^c,e_a).
\]

训练目标使用真实下一帧的 frozen representation：

\[
z_{t+1}^c=P_\theta(Z_{t+1}).
\]

基础 loss：

\[
\mathcal{L}_{\mathrm{next}}
=
\|\hat z_{t+1}^c-z_{t+1}^c\|_1.
\]

可选的变化量 loss：

\[
\mathcal{L}_{\Delta}
=
\|\hat z_{t+1}^c-z_t^c-(z_{t+1}^c-z_t^c)\|_1.
\]

主实验先使用 \(\mathcal{L}_{\mathrm{next}}+\lambda_\Delta\mathcal{L}_{\Delta}\)，不使用 physical distance label。

### 4.4 Action-conditioned MI

目标参考表示为：

\[
z_g^c=P_\theta(Z_g).
\]

对每个候选动作 \(a\)：

\[
M_t(a)=DameMI(\hat z_{t+1}^c(a),z_g^c).
\]

动作变化预测为：

\[
\Delta M_t(a)=M_t(a)-M_t(0).
\]

只有在这个 objective 上重新验证：

\[
g_a=\nabla_aM_t(a)
\]

是否与真实 physical distance descent direction 对齐。

---

## 5. 代码实现范围

### 5.1 新增模块

建议新增：

```text
mi_reward/models/action_conditioned_latent.py
```

包括：

```text
ControlLatentProjector
ActionEncoder
ActionConditionedLatentDynamics
```

该模块只负责 visual token -> controllable latent -> action-conditioned next latent。

### 5.2 修改 LaWAM extractor

修改：

```text
mi_reward/features/lawam_lam_extractor.py
```

新增接口：

```python
extract_visual_tokens(image)
extract_control_latent(image, goal=None)
extract_lam_transition_latent(images)
```

旧的 `extract_image_tokens()` 和 `extract_action_latents()` 保留，保证 v5 能复现。

### 5.3 新 evaluator

新增：

```text
eval/libero/eval_v6_action_conditioned_geometry.py
```

该 evaluator 必须支持：

- `--representation lam_visual`
- `--representation lam_control`
- `--representation lam_action_conditioned`
- `--representation raw_rgb`
- 固定 normalization；
- 相同的 primary/secondary epsilon；
- 相同的 held-out action 集合；
- 相同的 privileged physical-direction evaluator。

### 5.4 建议配置

新增配置文件：

```text
configs/mi_reward/lawam_control_latent.yaml
```

配置中固定：

- control latent dimension；
- pooling 类型；
- action encoder dimension；
- MLP 层数；
- loss 权重；
- optimizer、seed 和 early stopping；
- train/validation/test anchor split。

所有配置必须写入结果 JSON，不能依赖运行时默认值。

---

## 6. 实验数据和划分

### 6.1 数据来源

继续使用当前 LIBERO/simulator 的 RGB observation、语言任务描述和 EEF action。privileged EEF-object distance 只用于最终 evaluation，不进入主模型训练。

### 6.2 训练数据

如果现有 v5 数据中没有独立的 transition training split，需要额外采集局部 pre-contact transitions：

- 多个 initial states；
- 多个 object/goal anchors；
- 1/2/4 mm 左右的 EEF displacement；
- 保持 object static 和 pre-contact 条件；
- 每个 transition 保存 (o_t,a_t,o_{t+1})。

第一轮不要求完整 task rollout，优先覆盖局部 action neighborhood。

### 6.3 严格测试集

v5 的五个 initial states × 四个 anchors × held-out actions 作为冻结测试协议。训练阶段不能使用测试 anchor 的下一帧目标值。

建议划分：

```text
train: 新采集的 transition episodes
validation: 独立 initial states 或独立 anchors
test: v5 冻结的 5 states × 4 anchors
```

如果数据量不足以支持独立训练集，必须把结果标记为 diagnostic proof-of-concept，不能作为泛化实验。

---

## 7. 完整实验过程

### Test 0：v5 复现锁定

目的：确保 v6 的变化来自 representation/dynamics 分支，而不是 evaluator 改动。

执行：

1. 在冻结测试集上重新运行 v5 full-frame LaWAM visual latent；
2. 记录 gradient cosine、cross-epsilon cosine、held-out R2、Spearman、sign accuracy；
3. 记录 physical cosine 和 distance-descent fraction；
4. 保存 v5 baseline JSON；
5. 若 baseline 与 v5 正式记录不一致，停止后续实验并先修复复现问题。

### Test 1：control latent 的预测能力

比较：

1. 当前 LaWAM visual token；
2. mean/attention pooled LaWAM token；
3. 新增 \(z_t^c=P_\theta(Z_t)\)；
4. 新增 action-conditioned \(\hat z_{t+1}^c=F_\theta(z_t^c,a_t)\)。

指标：

- next-latent prediction R2；
- latent delta cosine；
- cross-epsilon prediction consistency；
- 不同 initial state 的性能方差；
- action magnitude 与 latent change 的相关性。

目标：确认新增 latent 是否真的比原始 DINO token 更能表达动作导致的局部变化。

### Test 2：action-conditioned MI direction

对相同测试 actions 计算：

\[
M(a)=MI(\hat z_{t+1}^c(a),z_g^c).
\]

比较以下四个版本：

| Branch | 表示 | 是否 action-conditioned | 用途 |
|---|---|---:|---|
| B0 | v5 LaWAM visual token | 否 | 原始 baseline |
| B1 | pooled control latent | 否 | 测试 pooling 是否改善 |
| B2 | control latent + action-conditioned predictor | 是 | v6 主方法 |
| B3 | LAM transition/action latent 直接对 goal latent | 间接 | 负向诊断，不作为主方法 |

B3 不能作为最终规划分支，因为它依赖已经发生的 transition，不能直接评价尚未执行的候选动作。

### Test 3：physical-direction gate

对 B0、B1、B2 使用完全相同的 privileged evaluator：

- MI gradient 与 physical descent direction cosine；
- MI ascent 是否降低 EEF-object distance；
- held-out action sign accuracy；
- held-out first-order R2；
- action ranking Spearman；
- 跨 initial state 的 direction consistency。

继承 v5 的通过标准：

- cross-epsilon gradient cosine ≥ 0.90；
- held-out sign accuracy ≥ 0.80；
- held-out first-order \(R^2\ge0.50\)；
- Spearman ≥ 0.60；
- physical cosine ≥ 0.70；
- 至少 4/5 initial states 通过方向一致性要求。

### Test 4：representation ablation

只改变一个组件：

| Ablation | 目的 | 如果失败说明什么 |
|---|---|---|
| 去掉 action condition | 测试静态 latent 是否足够 | 静态表征可能是主要问题 |
| 去掉 control projection | 测试全量 token 聚合影响 | LaWAM token 太混杂 |
| 替换为 mean pooling | 测试空间结构是否重要 | spatial attention 可能必要 |
| 只用 LAM action latent | 测试动作 bottleneck 是否能表达目标进度 | action latent 可能只表达动作模式 |
| 使用 privileged geometric latent | 提供上限诊断 | MI 数学本身是否能在结构化表示上工作 |
| 去掉 \(\mathcal{L}_{\Delta}\) | 测试 latent change supervision | 下一状态预测可能不足以保持局部方向 |

privileged geometric latent 只作为 upper bound，不能计入 observation-only 主结果。

### Test 5：Hessian 进入条件

只有 B2 同时通过 Test 2 和 Test 3 后，才对复合目标

\[
M(a)=MI(F_\theta(P_\theta(Z_t),a),P_\theta(Z_g))
\]

计算 action-space Hessian。

Hessian 进入条件保持 v5：

- Hessian 数值对称性通过；
- 二阶预测不低于一阶预测；
- Newton step 的实际 MI gain 不低于 gradient step；
- physical distance improvement 不恶化。

否则只保留一阶 action-conditioned MI，删除 Hessian 主张。

---

## 8. 预期结果和解释

### 结果 A：action-conditioned control latent 通过

表现：B2 在 held-out 和 physical-direction gates 上明显优于 B0。

解释：当前 LaWAM visual encoding 不是完全错误，但静态视觉 latent 缺少动作条件和未来状态建模。主线可以保留为：

> action-conditioned controllable latent 上的 MI directional reward。

下一步：再做 Hessian 与小规模 closed-loop planner。

### 结果 B：latent prediction 改善，但 physical direction 仍失败

解释：新的 latent 能预测视觉变化，但 MI 相关性仍然不能代表物理进度。此时 MI 只能作为 reference-consistency auxiliary signal，reward 主线应转向 privileged physical progress/action-direction distillation。

### 结果 C：只有 privileged geometric latent 通过

解释：MI 的数学形式在结构化几何表示上可能成立，但 RGB/LaWAM encoder 没有恢复所需几何信息。后续应转向 keypoint、dense correspondence 或 object-relative representation，而不是继续修改 MI estimator。

### 结果 D：所有 latent 都失败

解释：问题可能不在初始 encoding，而在 MI 与物理任务进度之间缺少可识别的目标定义，或当前局部 action probe 对任务状态的影响不足。此时停止 MI directional planner 主线，只保留诊断结果。

### 结果 E：VIP/TCN 类 progress latent 通过进度，但方向失败

解释：progress reward 和 action direction 是两个不同问题。可以使用 progress latent 做 scalar reward，同时另建 geometric/action-conditioned branch 表示动作方向。

---

## 9. 成功标准

v6 的最低成功标准不是“MI 数值变大”，而是以下链条同时成立：

```text
LaWAM visual tokens
  -> control-relevant projection
  -> action-conditioned next latent
  -> MI-to-goal change
  -> held-out action ranking
  -> privileged physical direction
```

至少需要：

1. B2 相比 B0 在跨 state physical cosine 上有稳定改善；
2. B2 的 held-out sign/R2/Spearman 达到预注册阈值；
3. 结果不是由单个 anchor 或单个 epsilon 驱动；
4. 训练没有使用 privileged object pose；
5. B2 的提升能够被 action conditioning 或 control projection ablation 解释。

仅有 MI gradient 与 finite-difference gradient 高 cosine，不足以宣布方向正确，因为 v5 已经证明了 MI 自身的一阶数值一致性可以很好，但 physical cosine 仍然失败。

---

## 10. 最终执行顺序

```text
1. 固定并复现 v5 baseline
2. 收集独立的局部 RGB-action transitions
3. 训练 frozen-LaWAM 上的 control projection + one-step dynamics
4. 验证 next-latent prediction
5. 在相同 held-out probes 上计算 action-conditioned MI
6. 运行 physical-direction gate
7. 做 representation/action-conditioning ablation
8. 只有一阶通过后才测试 Hessian
9. 只有 Hessian 有实际增益后才测试小规模 planner
```

本 pipeline 的首要目标不是立刻得到一个新的 reward model，而是确定：

> LaWAM 的失败来自“静态视觉 latent 不适合物理方向”，还是来自“MI 无法把 action-conditioned latent 转换成任务进度方向”。

---

## 11. Test 1a：mean-pooled LaWAM 线性动力学小实验（2026-09-09）

### 11.1 实验定位

在采集新的独立 transition training split 和训练完整的 \(P_\theta/F_\theta\) 之前，先检查一个更弱的必要条件：frozen LaWAM 的 mean-pooled latent 是否已经包含可由实际 EEF displacement 线性预测的局部变化。

该实验是 diagnostic proof-of-concept，不是 B2 主方法验证。它没有训练 controllable projection，也没有实现依赖 \(z_t\) 的 dynamics network，更没有计算 action-conditioned MI 或 physical-direction gate。

### 11.2 数据与划分

- 数据：`five_state_four_anchor_full_v1` 中五个 initial states 的第一个 anchor；
- 状态：`demo_1`、`demo_2`、`demo_4`、`demo_5`、`demo_10`；
- 表示：frozen LaWAM wrist-image tokens 的 channel mean pooling；
- action coordinate：candidate 与 center 之间的实际 EEF displacement；
- 训练动作：每个训练状态的 24 个 axis/cross probes；
- 测试动作：被测试状态的 24 个 held-out probes；
- 模型：无截距 ordinary least squares，\(\Delta z=B\Delta x_{eef}\)。

同时报告两个对照：

1. **state-independent cross-state**：用其余四个状态拟合一个共享 \(B\)，测试完整留出状态；
2. **state-specific local oracle**：在同一个状态内用 axis/cross 拟合 \(B_s\)，测试该状态的 held-out actions。

第二项不是可部署方法，只用于判断失败主要来自状态依赖，还是 mean-pooled latent 本身缺少稳定的局部 action signal。

### 11.3 执行记录

脚本：

```text
eval/libero/probe_v6_linear_latent_dynamics.py
```

在 `tmux train:v6-smoke` 中执行：

```bash
.venv-libero/bin/python eval/libero/probe_v6_linear_latent_dynamics.py \
  --collection-dir logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_full_v1 \
  --output-dir logs/mi_reward/v6_action_conditioned/linear_lawam_dynamics_smoke_v2 \
  --device cuda
```

结果与实时日志：

```text
logs/mi_reward/v6_action_conditioned/linear_lawam_dynamics_smoke_v2/results.json
logs/mi_reward/v6_action_conditioned/linear_lawam_dynamics_smoke_v2.live.log
```

### 11.4 结果

| 对照 | macro next-latent R2 | macro median delta cosine | pooled delta cosine median |
|---|---:|---:|---:|
| state-independent cross-state | -0.8227 | 0.0386 | 0.0355 |
| state-specific local oracle | -0.6799 | 0.2385 | 0.2315 |

五个状态的 state-independent R2 均为负，范围为 `-0.9467` 至 `-0.5244`。state-specific local oracle 的五个 R2 也全部为负，范围为 `-0.8537` 至 `-0.3988`。

### 11.5 判断

该实验未通过 Test 1 的 next-latent prediction 必要条件。局部 oracle 的 cosine 高于共享跨状态模型，说明 action-to-latent mapping 确实依赖当前状态，这与 v6 使用 \(F_\theta(z_t^c,a_t)\) 的设计动机一致；但局部 oracle 仍然很弱，说明 mean pooling 丢失空间结构后，frozen LaWAM latent 没有形成足够稳定的线性可控坐标。

因此，本结果只否定以下简化分支：

```text
mean-pooled frozen LaWAM + state-independent/state-specific linear residual
```

它不否定 v6 的 learned controllable projection 与 state-conditioned nonlinear dynamics。当前不能进入 Test 2、Test 3 或 Hessian。下一步仍应遵循第 10 节：采集与冻结测试 anchors 独立的 RGB-action transitions，保留一组小型空间 tokens，训练低容量 \(P_\theta/F_\theta\)，先验证 Test 1。只有 Test 1 明确通过后，才计算 action-conditioned MI。

---

## 12. Test 1：独立 transition 上的 spatial action-conditioned dynamics（2026-09-09）

### 12.1 数据与实现

独立采集目录：

```text
logs/mi_reward/v6_action_conditioned/transition_trainval_v1
```

- 40 个 anchors、1,960 个 candidates；严格 pre-contact 检查后排除 560 个候选；
- 训练 demos：`3, 6, 7, 8, 9, 11, 12, 13`，共 1,165 个有效非中心 transition；
- validation demos：`14, 15`，共 195 个有效 non-center transition；
- 冻结 v5 测试 demos `1, 2, 4, 5, 10` 未进入采集训练或 validation；
- 输入为 frozen LaWAM wrist tokens \([K,D]=[256,768]\)，control projection 输出保留 256 个空间 tokens；
- action coordinate 为 candidate 相对于 center 的实际 EEF displacement；不使用 object pose、distance 或任何 reward/MI loss。

模型：无仿射 LayerNorm + 可训练逐行单位范数 channel projection（\(D\rightarrow64\)），随后是 token-wise、state-conditioned residual dynamics。该 dynamics 对零 action 严格满足 \(F(z,0)=z\)。

### 12.2 运行修正

第一次实现使用当前环境的 `torch.nn.utils.parametrizations.orthogonal`。在 optimizer update 后，该矩形 projection 被 materialize 为全零权重，导致 \(P_\theta\) 塌缩、target delta 为零；其目录保留为：

```text
logs/mi_reward/v6_action_conditioned/test1_control_dynamics_projection_collapse_invalid_v1
```

这次运行不构成实验结果。修复为逐行单位范数投影后，重新运行并复用同一份 frozen-token cache：

```text
logs/mi_reward/v6_action_conditioned/test1_control_dynamics_v2
```

### 12.3 结果

按原实现的 validation next-latent R2 early stopping，最佳 checkpoint 位于 epoch 14：

| checkpoint selection | next-latent R2 | delta cosine mean | delta cosine median | zero-action error |
|---|---:|---:|---:|---:|
| 最大 next-latent R2，epoch 14 | 0.9800 | 0.1103 | 0.1110 | 0.0 |
| 训练曲线中最大 delta cosine，epoch 59 | 0.7838 | 0.3689 | 0.3770 | 0.0 |

### 12.4 判断

Test 1 得到的是混合证据，而不是通过：模型可以高精度重建下一 latent 的主体，但该指标主要由静态背景和状态恒等项驱动；在真正衡量动作影响的 latent delta 上，最佳 validation cosine 只有 0.3770。

相较于第 11 节 mean-pooled local oracle 的 0.2385，空间 token + state-conditioned dynamics 确实增加了动作相关信号，但提升仍不足以证明 action-conditioned MI 可以获得可靠方向。当前禁止进入 Test 2、Test 3 或 Hessian。下一步应先把 Test 1 checkpoint selection 固定为“最大 validation delta cosine，同时要求 next-latent R2 不低于预设阈值”，并在冻结 v5 test anchors 上只做 next-latent/delta 评估；若该 held-out Test 1 仍远低于稳定方向所需水平，则将 v6 主线降级为诊断结论，不再计算 MI。

---

## 13. Test 1b：normalized latent delta + spatial mixing（2026-09-09）

### 13.1 修改

Test 1b 直接预测动作造成的 control-token 变化：

\[
\delta z_t=P(Z_{t+1})-P(Z_t),\qquad \hat{\delta z}_t=F_\theta(P(Z_t),a_t).
\]

主损失是 flattened token delta 的 cosine loss 与 normalized Smooth-L1；next-latent reconstruction 仅以权重 `0.10` 作为辅助项。dynamics 增加一层跨 token self-attention spatial mixer，之后再与动作 embedding 相乘产生 delta，因此不再是完全独立的 token-wise dynamics。

checkpoint 选择在训练前固定为：validation next-latent \(R^2\ge0.50\) 时，最大化 validation delta cosine median。

### 13.2 结果

训练数据、frozen LaWAM token cache 和 train/validation split 与 Test 1 相同。正式结果：

```text
logs/mi_reward/v6_action_conditioned/test1b_control_delta_v1/results.json
```

| Test | checkpoint | next-latent R2 | delta cosine mean | delta cosine median | zero-action error |
|---|---:|---:|---:|---:|---:|
| Test 1 | epoch 14，按 R2 选择 | 0.9800 | 0.1103 | 0.1110 | 0.0 |
| Test 1 的最佳 delta 曲线点 | epoch 59 | 0.7838 | 0.3689 | 0.3770 | 0.0 |
| Test 1b | epoch 89，按预注册 delta 规则选择 | 0.7871 | 0.4147 | 0.4201 | 0.0 |

### 13.3 判断

Test 1b 相比 Test 1 的最佳 action-delta cosine 增加 `0.0431`，且没有以降低 next-latent R2 到阈值以下为代价，说明 normalized delta 主目标与 spatial mixing 都提供了真实增益。

但 `0.4201` 仍只是中等的 observation-space action-effect prediction，不是 action-to-physical-direction 成功证据。下一步只能在冻结 v5 的 20 个 test anchors 上运行不含 MI 的 Test 1b held-out evaluation，报告 next-latent R2、delta cosine、每个 initial state 的方差和 cross-epsilon consistency。只有该独立测试仍稳定改善，才有资格讨论 action-conditioned MI；当前仍禁止直接计算 MI、physical direction 或 Hessian。

---

## 14. Frozen Test 1b：v5 五状态、20 anchors 的独立评估（2026-09-09）

### 14.1 协议

冻结测试集为 v5 的 `five_state_four_anchor_full_v1`：`demo_1`、`demo_2`、`demo_4`、`demo_5`、`demo_10`，每个 4 anchors，共 20 anchors、每个 anchor 48 个有效非中心 actions。该 evaluator 只执行 latent-delta evaluation，明确不导入或计算 MI、reward、physical direction 或 Hessian。

对照：

1. Test 1b learned spatial action-conditioned delta；
2. 只由实际 EEF displacement 预测 delta 的 state-independent ridge linear action baseline，系数仅在独立 transition train split 拟合；
3. static next-state baseline；
4. learned model 的 zero-action baseline。

cross-epsilon consistency 定义为同一 anchor、同一轴和符号下 primary `0.18` 与 secondary `0.15` action 的 predicted delta cosine。

### 14.2 结果

结果文件：

```text
logs/mi_reward/v6_action_conditioned/test1b_frozen_v5_heldout_v1/results.json
```

| 模型 | mean next-latent R2 | mean delta cosine | median delta cosine |
|---|---:|---:|---:|
| learned spatial delta | 0.7902 | 0.3903 | 0.3934 |
| linear action baseline | 0.7889 | 0.0339 | 0.0320 |
| static next-state | 0.7896 | 0.0000 | 0.0000 |
| zero-action model | 0.7896 | 0.0000 | 0.0000 |

| initial state | learned next-latent R2 | learned delta cosine median | cross-epsilon predicted delta cosine |
|---|---:|---:|---:|
| demo_1 | 0.7773 | 0.3969 | 1.0000 |
| demo_2 | 0.7873 | 0.3816 | 1.0000 |
| demo_4 | 0.7927 | 0.3950 | 1.0000 |
| demo_5 | 0.7942 | 0.4041 | 1.0000 |
| demo_10 | 0.7991 | 0.3893 | 1.0000 |

跨 state learned delta-cosine median 的均值为 `0.3934`、方差为 `5.72e-05`、最小值为 `0.3816`。相对于独立 validation 的 `0.4201`，回落 `0.0267`，没有出现 validation-specific 的 `0.2` 级退化。

### 14.3 判断

Frozen Test 1b 通过进入 Test 2 的前置检查：action-effect signal 明显优于线性和 static/zero baselines，并且在五个 initial states 上稳定、对 primary/secondary epsilon 一致。

高 next-latent R2 同时出现在 static baseline，证实它不能单独作为成功依据；进入条件来自 learned delta 的跨状态稳定提升。该结果只授权下一步计算 action-conditioned MI 并重复 held-out/physical-direction gates，不授权 Hessian 或 planner。

---

## 15. Test 2：frozen action-conditioned MI（2026-09-09）

### 15.1 协议

固定 Test 1b checkpoint：

```text
logs/mi_reward/v6_action_conditioned/test1b_control_delta_v1/best.pt
```

在相同 frozen v5 五 states、20 anchors 上构造：

\[
\hat z_{t+1}(a)=z_t+\hat{\delta z}(z_t,a),
\qquad
M(a)=DameMI(\hat z_{t+1}(a),z_g).
\]

每个 anchor 的 Dame MI normalization 仍只由 reference 和 zero-action anchor control tokens 固定。重新运行 v5 的实际 EEF coordinate、primary/secondary finite difference、24 held-out actions 和 privileged EEF-object descent evaluator；未计算 Hessian 或 planner。

### 15.2 结果

正式结果：

```text
logs/mi_reward/v6_action_conditioned/test2_action_conditioned_mi_v1/results.json
```

| 指标 | 结果 | gate |
|---|---:|---:|
| MI analytic-gradient / finite-difference cosine median | 0.9880 | >= 0.90，通过 |
| cross-epsilon finite-difference cosine median | 0.9748 | >= 0.90，通过 |
| held-out action R2 | -0.0265 | >= 0.50，失败 |
| held-out Spearman | 0.3210 | >= 0.60，失败 |
| held-out sign accuracy | 0.6063 | >= 0.80，失败 |
| MI / physical descent cosine median | 0.5167 | >= 0.70，失败 |
| distance-descent fraction | 0.9500 | >= 0.70，通过 |

总体状态：`NO_GO`。

### 15.3 结论

Test 1b 已证明 learned spatial latent 能稳定预测一部分 action-induced observation change；Test 2 进一步证明 MI 在该 latent 上仍具有良好的内部一阶数值一致性。可是 held-out MI change 与真实 candidate image 的 MI change 不匹配，且 MI gradient 没有达到物理方向 gate。

因此，当前失败不能再归因于静态 LaWAM latent 缺少 action condition。现有 Dame MI 仍不能把 controllable latent 转换为可靠的物理任务进度方向。Pipeline v6 的 action-effect latent 只保留为 diagnostic representation result；不进入 Hessian、Newton proposal 或 planner。若继续 reward 主线，应把 MI 降级为 reference-consistency auxiliary signal，主监督另行采用 privileged physical progress 或 action-direction distillation。

---

## 16. Oracle-vs-predicted MI 分解：实际下一 latent 与预测下一 latent（2026-09-09）

### 16.1 协议

为把 Test 2 的失败归因于 MI 本身还是 dynamics predictor，固定完全相同的 Test 1b checkpoint、v5 五个 frozen initial states、20 anchors、每个 anchor 24 个 held-out actions 和实际 EEF displacement coordinate。每个 anchor 的 goal latent 与 normalization 也保持不变，只改变 MI 的第一输入：

\[
M_A(a)=DameMI(z_{t+1}^{\mathrm{actual}}(a),z_g),
\qquad
M_B(a)=DameMI(\hat z_{t+1}^{\mathrm{predicted}}(a),z_g).
\]

A 使用 candidate 的实际 rendered wrist image 经冻结 control projector 得到的下一 latent；B 使用 `F(z_t, a)`。两者均以 center MI 作相对分数，并以 held-out candidate 的实际 EEF-object distance reduction 作为物理进度。每个 branch 以 primary finite difference 估计 MI 在实际 EEF coordinate 下的方向，并与 privileged physical descent direction 比较。没有 Hessian、planner 或额外 privileged signal 进入 MI。

正式结果：

```text
logs/mi_reward/v6_action_conditioned/oracle_predicted_mi_decomposition_v1/results.json
```

### 16.2 结果

| branch | held-out R2 vs physical progress | Spearman | sign accuracy | physical cosine median | descent fraction |
|---|---:|---:|---:|---:|---:|
| A: actual next latent | 0.0662 | 0.2435 | 0.5979 | 0.4766 | 0.8000 |
| B: predicted next latent | -0.0001 | 0.0956 | 0.5167 | 0.4926 | 0.9000 |

预测与 oracle 的 held-out MI score fidelity 也很低：R2 `-0.0251`、Spearman `0.2159`、sign accuracy `0.5375`；两 branch 的 primary finite-difference gradient cosine median 为 `0.6265`。这说明 predictor error 确实会改变 MI score，但不构成首要失败来源。

### 16.3 判断

结论为 `ORACLE_MI_RANKING_FAIL`。最关键证据是 A 不经过 dynamics predictor 仍无法通过 held-out physical-progress ranking gate（R2 `>=0.50`、Spearman `>=0.60`、sign `>=0.80`），并且其 physical-direction cosine median `0.4766` 低于 `0.70` gate。

因此，Test 2 的失败不能解释为 latent prediction error 对 MI 过度敏感：在 oracle actual-next-latent 条件下，Dame MI / 当前 goal-latent aggregation 本身已经不能提供可靠的物理任务进度排序或方向。B 的额外退化是次要诊断。该分解也排除了“A 的视觉 ranking 通过而仅 physical direction 失败”的解释，因为 A 的 held-out physical-progress ranking 同样失败。

Pipeline v6 不再进行 Hessian、扩大 LaWAM，或继续改动 MI bins、normalization、epsilon 和 evaluator。当前 action-conditioned latent 可保留为 action-effect diagnostic；若回到 reward 主线，MI 只能作为 reference-consistency auxiliary signal，主监督需改由 privileged physical progress 或 action-direction distillation 提供。
