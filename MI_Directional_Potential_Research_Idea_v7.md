# Pipeline v7：Structured Geometric Latents for MI and Physical Progress

## 0. 版本定位

Pipeline v6 已完成 oracle-vs-predicted MI 分解：

- action-conditioned latent 能预测一部分 observation-space action effect；
- MI 的解析梯度和有限差分一致；
- 但 actual next latent 直接计算 MI 时，held-out physical-progress ranking 仍失败；
- 当前 Dame MI + goal-latent aggregation 不能稳定表示物理任务进度方向。

因此，v7 不再继续扩大 LaWAM，也不继续调整 MI bins、normalization、epsilon 或 Hessian。v7 只回答一个新的诊断问题：

> 换成具有明确几何或任务进度含义的 latent 后，MI 是否能够与物理任务进度建立可靠关系？

v7 的核心原则是：先验证 representation，再验证 MI；先验证 oracle，再验证 visual reconstruction；没有通过前一级，不进入下一级。

---

## 1. 研究问题和假设

### Q1：MI 是否至少能在结构化几何 latent 上表征物理进度？

构造 simulator privileged geometric latent：

\[
z_t^{geo}=\left[
p_t^{eef}-p_t^{obj},
p_t^{obj}-p_t^{goal},
p_t^{eef}-p_t^{goal}
\right].
\]

其中第一阶段只使用 XYZ translation，不引入 rotation、contact state 或 full simulator state。

对 candidate action (a) 计算：

\[
M^{geo}(a)=MI(z_{t+1}^{geo}(a),z_g^{geo}).
\]

如果 geometric oracle 也无法预测 physical progress，说明当前 MI objective 与任务进度的关系本身不成立。

### Q2：如果 geometric oracle 通过，RGB 能否恢复相同结构？

定义视觉 structured latent：

\[
z_t^{kp}=E_{kp}(o_t),
\]

其中包含末端执行器、被操作物体和目标区域的 keypoints 或 dense correspondence descriptors。

再定义动作条件预测：

\[
\hat z_{t+1}^{kp}=F_{kp}(z_t^{kp},a_t).
\]

只在 actual visual keypoint latent 通过后，才评价 predicted visual keypoint latent。

### Q3：任务 progress latent 是否适合作为 reward，但不适合作为 direction？

VIP、TCN、R3M 类表示主要表达阶段或目标接近程度。v7 将它们作为 scalar reward baseline：

\[
r_{progress}(o_t,o_g)=-d(z_t^{progress},z_g^{progress}).
\]

它们不自动进入 MI directional branch。这样可以区分：

- reward/progress 是否可行；
- action direction 是否可行；
- MI 是否提供额外信息。

---

## 2. 已有研究依据

v7 使用已有研究中已经用于控制或 reward 的结构化表示，不把“换 latent”理解为随意替换 backbone。

### 2.1 几何和 keypoint 表示

- Deep Spatial Autoencoders 将视觉输入压缩为空间 feature points，并用于闭环 visuomotor learning。
  [论文](https://arxiv.org/abs/1509.06113)
- Unsupervised Learning of Visual 3D Keypoints for Control 学习具有 3D 几何意义的 keypoints，用于机器人控制。
  [ICML/PMLR](https://proceedings.mlr.press/v139/chen21b.html)
- Dense Object Nets 学习用于机器人 manipulation 的 dense object descriptors，并支持对象对应关系。
  [CoRL/PMLR](https://proceedings.mlr.press/v87/florence18a.html)
- Transporter Networks 保持空间结构，把 manipulation 表达为局部 spatial displacement。
  [CoRL/PMLR](https://proceedings.mlr.press/v155/zeng21a.html)

### 2.2 任务进度和视觉 reward 表示

- Time-Contrastive Networks 学习对任务阶段敏感的时序表示。
  [CVPR workshop](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Sermanet_Time-Contrastive_Networks_Self-Supervised_CVPR_2017_paper.html)
- Visual Task Progress Estimation 明确以 task progress/phase 为学习目标。
  [论文](https://arxiv.org/abs/2003.06977)
- VIP 从 goal-conditioned value objective 学习可用于 dense visual reward 的 representation。
  [OpenReview](https://openreview.net/pdf?id=YCETybILpw)
- R3M 使用时间对比、视频语言对齐和稀疏性约束构造机器人 manipulation representation。
  [CoRL/PMLR](https://proceedings.mlr.press/v205/nair23a.html)

---

## 3. Latent 分支定义

### Branch G：Privileged geometric latent

输入：simulator 提供的 EEF、object、goal 的位置。

表示：

\[
z_t^{geo}=[r_{eo},r_{og},r_{eg}],
\]

其中：

\[
r_{eo}=p_t^{eef}-p_t^{obj},
\qquad
r_{og}=p_t^{obj}-p_t^{goal},
\qquad
r_{eg}=p_t^{eef}-p_t^{goal}.
\]

用途：只作为 MI mathematical upper bound，不作为 observation-only 方法。

### Branch K：Visual keypoint/relative latent

输入：RGB observation，以及必要时的独立 keypoint detector 或 instance correspondence module。

表示：

\[
z_t^{kp}=[\hat p_t^{eef},\hat p_t^{obj},\hat p_t^{goal},
\hat p_t^{obj}-\hat p_t^{eef},
\hat p_t^{goal}-\hat p_t^{obj}].
\]

如果目标点不能从 RGB 可靠提取，先只使用 EEF-object relation，并明确记录 goal proxy 的来源。

第一版允许使用 simulator instance mask 作为诊断输入，但必须标记为 `mask-assisted`，不能作为纯 RGB 结果。

### Branch D：Dense descriptor latent

在 object ROI 或 object-centered crop 中提取 dense descriptors：

\[
Z_t^{desc}\in\mathbb{R}^{K\times d}.
\]

保留 descriptor 的空间位置和 object correspondence，不做全局 mean pooling。该分支用于检查空间 correspondence 是否比 DINO semantic patch token 更适合 MI。

### Branch P：Progress/reward latent

使用 VIP/R3M/TCN 类现有 representation，计算 goal-conditioned scalar progress。该分支只作为 reward baseline：

\[
r_t^{P}=-d(E_P(o_t),E_P(o_g)).
\]

不把它的 scalar gradient 直接解释成 physical EEF direction。

---

## 4. 统一的 oracle-vs-predicted 协议

每个 latent 分支都使用相同的两阶段协议。

### A：Actual-next latent

使用 candidate 实际渲染得到的下一帧或 simulator 下一状态：

\[
M_A(a)=MI(z_{t+1}^{actual}(a),z_g).
\]

该分支绕过 dynamics predictor，直接测试：

> 当前 latent 和 MI 是否能表达 physical progress。

### B：Predicted-next latent

使用 action-conditioned dynamics 预测：

\[
\hat z_{t+1}(a)=F(z_t,a),
\qquad
M_B(a)=MI(\hat z_{t+1}(a),z_g).
\]

该分支测试 predictor 是否足够准确。

### 必须保持不变的设置

- v5 的五个 frozen initial states；
- 20 个 anchors；
- 相同的 held-out actions；
- 相同的 actual EEF displacement coordinate；
- 相同的 physical progress 定义；
- 相同的 anchor-level normalization；
- 相同的 finite-difference epsilon；
- 不使用 Hessian、planner 或闭环 RL。

---

## 5. 评价指标和 gates

### 5.1 Progress/ranking 指标

对每个 candidate action，将 latent score 与实际 physical progress 比较：

- held-out \(R^2\)；
- Spearman rank correlation；
- sign accuracy；
- per-anchor variance；
- per-initial-state variance。

继承 v6 gate：

- \(R^2\ge0.50\)；
- Spearman \(\ge0.60\)；
- sign accuracy \(\ge0.80\)。

### 5.2 Direction 指标

- analytic gradient 与 finite-difference gradient cosine；
- cross-epsilon finite-difference cosine；
- 与 privileged physical descent direction 的 cosine；
- distance-descent fraction。

方向 gate：

- analytic/finite-difference cosine (ge0.90)；
- cross-epsilon cosine (ge0.90)；
- physical cosine (ge0.70)；
- 至少 4/5 initial states 方向一致。

### 5.3 Baseline 必须包含

| Baseline | 作用 |
|---|---|
| Static next-state | 检查 next-latent R2 是否被状态恒等项主导 |
| Zero-action model | 检查零动作 identity 是否造成假提升 |
| State-independent linear action model | 检查 learned dynamics 是否超越简单线性映射 |
| Privileged geometric distance | 物理进度上限 |
| Current v5 LaWAM visual latent | 原始 baseline |
| VIP/R3M/TCN-style progress score | reward baseline |

---

## 6. 完整实验流程

### Test 0：协议和 baseline 锁定

1. 复现 v6 oracle MI 结果；
2. 固定五个 frozen states、20 anchors 和 held-out actions；
3. 固定 physical progress 计算方式；
4. 输出所有 baseline 的 JSON；
5. 如果 baseline 与 v6 不一致，停止后续实验。

### Test 1：Privileged geometric latent MI upper bound

分别测试：

1. actual geometric next latent；
2. simulator action-conditioned predicted geometric next latent；
3. geometric distance/progress baseline；
4. MI score 与 physical progress 的 ranking；
5. MI gradient 与 physical direction 的 alignment。

决策：

- Branch G-A 失败：停止 MI directional branch，不进入视觉 keypoint MI；
- Branch G-A 通过、G-B 失败：问题在 dynamics predictor；
- Branch G-A 和 G-B 都通过：进入 visual structured latent。

### Test 2：Visual keypoint latent

第一阶段使用不训练 backbone 的 keypoint/relative representation：

1. 从 RGB 或 mask 提取 EEF/object/goal keypoints；
2. 做 temporal tracking 和 coordinate normalization；
3. 构造 object-relative latent；
4. 分别评价 actual-next 和 predicted-next；
5. 与 Branch G 的结果比较。

如果使用 simulator instance mask，结果写为 `mask-assisted diagnostic`，不能表述为纯视觉方法。

### Test 3：Dense correspondence latent

只有在 Branch G 或 Branch K 有明确正向证据时才运行。

1. 保留 object-centered spatial descriptors；
2. 禁止全局 mean pooling；
3. 测试 descriptor delta prediction；
4. 运行 actual-next MI 和 predicted-next MI；
5. 比较 Dense Object Nets/Transporter 风格空间保留与当前 DINO patch token 的差异。

### Test 4：Progress/reward latent baseline

使用 VIP/R3M/TCN-style representation 计算 goal-conditioned progress score。

该实验只比较：

- scalar reward ranking；
- task progress correlation；
- 是否能作为 reward model 输入。

不要求该分支通过 physical action-direction gate，因为它的研究目标是 progress，而不是显式动作方向。

### Test 5：MI 是否带来额外价值

对通过的 structured latent 比较：

1. latent Euclidean/cosine goal distance；
2. MI score；
3. goal-conditioned progress/value score；
4. privileged physical teacher。

只有 MI 相比简单 geometric/goal-distance baseline 有稳定增益，才能继续保留 MI 作为主方法组件。

---

## 7. 推荐实现范围

### 7.1 新增文件

```text
mi_reward/features/geometric_features.py
mi_reward/features/keypoint_features.py
mi_reward/features/dense_descriptor_features.py
eval/libero/eval_v7_structured_latents.py
```

### 7.2 Geometric feature extractor

`geometric_features.py` 只实现：

```python
extract_eef_object_goal_relation(sim_state)
extract_geometric_latent(sim_state)
extract_geometric_delta(sim_state_a, sim_state_b)
```

该模块只能用于 oracle diagnostic，输出中必须写入：

```text
privileged_input: true
deployable: false
```

### 7.3 Keypoint feature extractor

`keypoint_features.py` 第一版只支持：

- 2D keypoint；
- mask-assisted centroid；
- temporal correspondence；
- object-relative normalization。

不要在第一版同时训练大型 detector、DINO backbone 和 MI estimator。

### 7.4 Evaluator 输出

每个 branch 的 `results.json` 至少包含：

```text
branch
representation_source
privileged_input
actual_or_predicted
mi_normalization
heldout_r2
spearman
sign_accuracy
analytic_fd_cosine
cross_epsilon_cosine
physical_cosine
distance_descent_fraction
per_initial_state
per_anchor
status
```

---

## 8. 结果解释

### 情况 A：Geometric oracle 的 MI 也失败

结论：当前 MI objective 与 physical progress 不匹配。停止 MI directional reward 主线，保留 geometry/progress reward 研究。

### 情况 B：Geometric oracle 通过，visual keypoint 失败

结论：MI 在结构化几何状态上可能成立，但视觉编码或 keypoint tracking 不足。研究重点转为 visual geometric representation。

### 情况 C：Geometric 和 visual keypoint 都通过

结论：可以继续研究结构化 latent 上的 MI directional reward，并比较 MI 与直接 goal distance 的增益。

### 情况 D：Progress latent 通过，但 direction 失败

结论：reward/progress 与 action direction 是两个不同任务。主线应拆分为 progress reward 和 action-effect model。

### 情况 E：MI 不如直接 goal distance

结论：MI 没有提供必要的额外信息，应从主方法中删除，仅保留为 auxiliary consistency signal。

---

## 9. 成功标准

v7 只有在以下条件同时满足时，才能声称 MI 对结构化 latent 有用：

1. geometric actual-next branch 通过 held-out progress ranking；
2. geometric branch 的 physical cosine 达到 `0.70`；
3. visual structured latent 在冻结测试集上接近 geometric oracle；
4. predicted-next branch 没有显著劣于 actual-next branch；
5. MI 优于简单 goal-distance/progress baseline；
6. 结果跨五个 initial states 和 20 个 anchors 稳定。

单独的 analytic gradient/finite-difference 高 cosine 不足以通过 v7，因为 v5 和 v6 已经证明这只能说明 objective 的局部数值实现正确。

---

## 10. 最终执行顺序

```text
1. 固定并复现 v6 oracle MI NO-GO
2. 实现 privileged geometric latent
3. 完成 geometric actual-next MI upper-bound test
4. 若 geometric oracle 失败，停止 MI directional branch
5. 若 geometric oracle 通过，测试 geometric predicted-next dynamics
6. 再测试 RGB keypoint/object-relative latent
7. 再测试 dense correspondence latent
8. 并行运行 VIP/R3M/TCN-style progress baseline
9. 比较 MI、goal distance 和 progress value
10. 只有 MI 有额外增益时，才保留 MI 作为主方法
```

v7 的目标不是强行挽救 MI，而是完成一个可解释的 latent-level falsification：

> 如果连显式几何 latent 上的 MI 都不能恢复 physical progress，那么问题是 MI objective；如果几何 latent 成功而视觉 latent 失败，问题才是 visual encoding。

---

## 11. 实现前的 G0/G1 修正（2026-09-09）

当前 v5 collection 为每个 candidate 保存了 EEF、task-object 和 goal-object 的中心位置，但没有保存 object surface points、姿态、关键点或可对应的几何点集。因此不能把原始 9D `[r_eo, r_og, r_eg]` 直接送入 Dame MI：`r_eg=r_eo+r_og` 存在冗余，且坐标维度不是 MI estimator 所需的 independent correspondence/token sample。把三维坐标伪装成 token 会使失败无法归因于 MI objective。

因此 Test 1 固定为两个 gates：

1. **G0 direct privileged geometry sanity**：仅用非冗余 `[r_eo,r_og]` 和直接 pre-contact score `-||r_eo||`，复核 20 anchors、480 held-out actions、EEF finite difference 和 physical-progress 标签。这个分数是 privileged upper baseline，不是 MI 结果。
2. **G1 correspondence geometric-MI oracle**：只有 G0 通过且补采集每个 candidate 的同一组 object/goal surface points 或 simulator sites 后才开始。每个 point correspondence 组成一个 token，Dame MI 才有明确的样本轴。不得复制中心点或用人为 offsets 制造 pseudo-tokens。

G0 通过只说明数据、坐标和 progress evaluator 自洽；它不授权关于 MI 的正向结论。G1 的 actual-next branch 才是 v7 对 MI objective 的有效 falsification。VIP/R3M/TCN progress baseline 也延后到 G1 决策之后，避免与“逐级 gate”相冲突。

---

## 12. G0：privileged direct-geometry sanity（2026-09-09）

### 12.1 固定协议

在 v5 的五个 frozen initial states、20 anchors、每个 anchor 24 个 held-out actions（总计 480）上运行。直接 privileged score 为：

\[
s_{geo}(x)=-\lVert p^{eef}(x)-p^{obj}(x)\rVert_2.
\]

所有坐标直接来自 collection 中 candidate 的 simulator-recorded `physical_after`；该 branch 标记为 `privileged_input: true`、`deployable: false`。以 candidate 相对 center 的 score change 和冻结 physical-progress 定义比较，并用实际 EEF displacement 的 primary/secondary finite difference 验证方向。

结果和 tmux 日志：

```text
logs/mi_reward/v7_structured_latents/g0_privileged_geometry_sanity_v2/results.json
logs/mi_reward/v7_structured_latents/g0_privileged_geometry_sanity_v2.live.log
```

### 12.2 结果

| 指标 | 值 |
|---|---:|
| held-out actions | 480 |
| direct geometry vs physical-progress R2 | 1.0000 |
| Spearman | 1.0000 |
| sign accuracy | 1.0000 |
| analytic / primary finite-difference cosine median | 1.0000 |
| primary / secondary finite-difference cosine median | 1.0000 |
| physical-direction cosine median | 1.0000 |

### 12.3 判断与下一步

G0 `PASS`：冻结 collection、实际 EEF coordinate、physical-progress label 和 evaluator 都自洽。由于 score 与 pre-contact physical-progress label 是同一 privileged distance 的相对变化，这一结果是必要的 plumbing sanity check，不能被表述为 MI 或 observation-only reward 的成功。

当前 collection 仅有三个中心位置和 anchor simulator state，没有 candidate-level mesh/surface points、site positions、object pose 或稳定几何 correspondences。因此 **不启动 G1**，也不复制 object center 或添加人为 offset 来制造 Dame MI tokens。下一步是扩展 collection schema，在每个 candidate state 保存同一组 object/goal local surface points 经 simulator pose 变换后的世界坐标，或保存固定 MuJoCo sites；随后才对这些真实对应 token 运行 G1 actual-next geometric-MI oracle。

---

## 13. G1：actual-next geometric correspondence MI oracle（2026-09-09）

### 13.1 Correspondence schema

冻结 v5 collection 的 anchors、actions、RGB 和 physical-progress definition 完全不变。新增 replay augmentation：每个 candidate 从同一 anchor state 重放，保存 task-object 的 `41` 个固定 MuJoCo geoms 加 `1` 个 site、goal-object 的 `11` 个 fixed geoms 加 `1` 个 site。每个 landmark 记录 model-name identity、entity、`local_xyz` 和 `world_xyz`。总覆盖为 20 anchors、980 candidates；replay physical positions 与 collection 中记录逐 candidate 校验，且任一 landmark local identity 漂移即停止。

artifact：

```text
logs/mi_reward/v7_structured_latents/g1_geometric_correspondences_v1/correspondences.json
```

G1 只计算 actual-next branch。每个 task-object landmark token 为：

\[
z_i=[p_i^{obj}-p^{eef},\;p_i^{obj}-p^{goal\ site}],
\]

其中各点均是同一 MuJoCo identity 的真实 world point；reference latent 从独立 reference demo state 以同一 identity 重放得到。没有 RGB、LaWAM、dynamics prediction、Hessian 或 planner。

### 13.2 结果

```text
logs/mi_reward/v7_structured_latents/g1_actual_next_geometric_mi_v1/results.json
logs/mi_reward/v7_structured_latents/g1_actual_next_geometric_mi_v1.live.log
```

| 指标 | G1 actual-next geometric MI |
|---|---:|
| held-out actions | 480 |
| held-out R2 vs physical progress | 0.0002 |
| Spearman | -0.1934 |
| sign accuracy | 0.3438 |
| primary/secondary finite-difference cosine median | 1.0000 |
| MI / physical descent cosine median | -0.6690 |

总体状态：`NO_GO`。

### 13.3 结论

G1 在真正的 fixed-identity geometric correspondence tokens 上仍失败，而且不是数值 finite-difference 不稳定：cross-epsilon cosine 为 `0.99999`。rank 与 sign 均低于 gate，physical-direction cosine 为负，说明当前 Dame MI score 在这个 pre-contact geometry setting 下系统性偏离 EEF-object physical descent。

因此，v7 已完成预注册的关键 falsification：MI directional branch 的失败不能归因于 RGB visual encoding、LaWAM spatial latent、action-conditioned predictor 或伪造的几何 token。**停止 MI directional 分支**；不进入 geometric predicted-next、RGB keypoint MI、dense correspondence MI 或 Hessian。后续只可将 MI 保留为 reference-consistency auxiliary diagnostic；reward 主线应转向 G0 已验证的 privileged physical-progress teacher 或 action-direction distillation，并把 observation-only progress representation 作为独立 reward baseline 研究。
