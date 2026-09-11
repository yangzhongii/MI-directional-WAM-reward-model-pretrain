# Pipeline v8：Physically Grounded Mutual Information for Visual Reward Learning

**目标投稿：** ICLR 2027 主会练手稿  
**冻结日期：** 2026-09-10  
**摘要截止：** 2026-09-18 23:59 AoE  
**全文截止：** 2026-09-25 23:59 AoE  
**计算资源：** 单机 RTX 4090 48 GB；不要求真机  
**主任务：** observation-only robot reward learning  
**MI 的最终角色：** training-only physical-information regularizer，以及待比较的 raw-reward baseline

---

## 1. 冻结后的论文问题

本文不再主张：

> raw mutual information、latent MI 或其数值梯度能够直接表示通用 manipulation progress 或动作方向。

现有 v5--v7 和 strict E1 已经给出反证。本文研究一个更窄的问题：

> 当 reward model 使用 RGB 预测任务进度时，训练阶段最大化视觉表示与 privileged physical factors 之间的 task-conditioned mutual information，是否能够减少 instance-level 和 scene-level visual variance，并提高 reward ranking 的泛化能力？

最终部署模型只输入：

```text
task instruction + recent RGB history + optional goal image
```

机器人状态、物体位姿、接触和夹爪状态只在 simulator 训练阶段提供监督，部署时不需要。

---

## 2. 为什么这条线仍然保留 MI

现有结果否定的是以下映射：

\[
I(z_t,z_g) \longrightarrow \text{physical progress or action direction}.
\]

它没有否定下面的问题：

\[
I(Z_{\mathrm{visual}};\Phi_{\mathrm{physical}}\mid C_{\mathrm{task}}),
\]

其中 \(Z_{\mathrm{visual}}\) 和 \(\Phi_{\mathrm{physical}}\) 都是随机变量，由一批配对状态定义；\(C_{\mathrm{task}}\) 是任务或阶段条件。该 MI 只要求视觉表示保留与物理任务有关的信息，不被当作某一帧的 scalar progress。

这个角色符合 MI 的统计定义：

- 正样本是同一物理状态的视觉观测与 physical-factor vector；
- 场景变体与原图共享 physical factor，构成多视图正样本；
- 负样本来自同一任务或同一阶段内物理状态不同的样本；
- reward 的方向和大小仍由独立的物理排序标签决定。

因此 v8 不是把失败的 Dame scalar 换个名字继续使用。Dame-style MI、conditional InfoNCE 和 reward value 分别承担不同角色，并在结果表中分开。

---

## 3. 当前已经拥有的证据

### 3.1 可直接进入论文的负结果

| 证据 | 已有结果 | 支持的结论 |
|---|---:|---|
| v7 actual-next geometric MI | \(R^2=0.0002\), Spearman \(-0.1934\), sign accuracy \(0.3438\) | geometric/latent Dame MI 不能直接作为 physical-progress direction |
| v7 gradient stability | cross-epsilon cosine \(\approx1.0\) | 失败不是 finite-difference 不稳定 |
| v7 physical alignment | median cosine \(-0.6690\) | MI direction 与 EEF-object descent 系统性不一致 |
| E1-A0/A1 | first-order gradient cosine \(>0.99999\) | MI、image Jacobian 和 pose chain 实现正确 |
| E1-A2 | TRO-2011 printed Hessian relative error \(0.8259\) | printed equation (17) 不是 equation (16) 的 exact derivative |
| E1-A3 | exact branch success \(0/18\), MI monotone \(0.9858\) | optimizer 正确提高 MI，但 estimator maximum 偏离 reference pose |
| E1-A4 | pixel-to-pose chain cosine \(1.0\) | localization bias 不来自 pose chain 或符号错误 |

这些结果的作用是建立设计原则：raw MI 不负责 reward direction；MI 只作为 batch-level representation regularizer。

### 3.2 可复用的工程资产

- LIBERO-Spatial 官方演示、仿真器和双视角读取链路；
- 真实 simulator success/failure rollouts；
- DINOv3/LaWAM feature cache；
- `VisualGoalPotential` 的 goal cross-attention 和 GRU；
- Bradley--Terry pairwise ranking loss；
- Cosmos Predict/Transfer 生成轨迹和视觉变体；
- Qwen3-VL-2B reward checkpoint；
- Robometer MetaWorld 与 USC-Koch processed datasets；
- RBM-EVAL adapter 与三类指标。

---

## 4. v8 方法

### 4.1 Observation encoder

第一版冻结 DINOv3 backbone，只训练小型 reward head：

\[
X_t=E_{\mathrm{DINO}}(o_{t-h:t}),\qquad
X_g=E_{\mathrm{DINO}}(o_g).
\]

使用现有 visual-goal cross-attention 聚合 patch token：

\[
u_t=\operatorname{CrossAttn}(X_t,X_g,e_c),
\qquad h_t=\operatorname{GRU}(u_{t-h:t}).
\]

其中 \(e_c\) 是 task-language embedding。若第一轮暂时没有语言 encoder，可以先使用 task-ID embedding，但正式 task-held-out 实验必须换成冻结的文本 embedding。

### 4.2 Physical-factor target

从 simulator state 构造：

\[
\Phi_t=
[d_{ee,obj},\ d_{obj,goal},\ \Delta p_{ee,obj},\
\Delta p_{obj,goal},\ w_{gripper},\ contact,\ grasped,\ h_{obj}].
\]

不同任务只启用可定义的 factor，并保存 validity mask。距离、向量和高度用训练 split 统计量归一化；contact/grasped 使用二值标签。

### 4.3 Stage-conditioned value

阶段 head 预测：

```text
reach / grasp / transport / place-or-insert / terminal
```

使用 mixture-of-experts value：

\[
V_t=\sum_{k=1}^{K}\pi_k(h_t,c)V_k(h_t,c).
\]

这样 reaching、grasping 和 transport 不需要共享同一套单调几何关系。

### 4.4 Task-conditioned physical MI

分别投影视觉状态和 physical factors：

\[
q_i=q(h_i,c_i),\qquad k_i=k(\Phi_i,c_i).
\]

在同任务、同阶段的 batch 内使用 symmetric InfoNCE：

\[
\mathcal L_{\mathrm{CMI}}
=-\frac{1}{2B}\sum_i
\left[
\log\frac{\exp(s(q_i,k_i)/\tau)}{\sum_{j:c_j=c_i}\exp(s(q_i,k_j)/\tau)}
+
\log\frac{\exp(s(k_i,q_i)/\tau)}{\sum_{j:c_j=c_i}\exp(s(k_i,q_j)/\tau)}
\right].
\]

如果同一 physical state 有原始 RGB、agent/wrist view 或通过审计的 Cosmos appearance variant，它们都属于同一 positive set。负样本优先选择视觉相似但 physical progress 不同的 hard negatives。

这里优化的是 conditional MI lower bound，不报告未经证明的精确 MI 数值。

### 4.5 Reward supervision

主 reward 标签来自 simulator physical progress 和真实 outcome，不来自 MI：

\[
\mathcal L_{rank}
=-\log\sigma\left(V(o^+,g,c)-V(o^-,g,c)\right).
\]

同一 task、同一 initial state 内，根据 privileged stage/factor/outcome 构造 \(o^+\succ o^-\)。跨场景 raw scalar 不直接比较。

辅助损失：

\[
\mathcal L =
\mathcal L_{rank}
+0.5\mathcal L_{stage}
+0.2\mathcal L_{physical}
+\lambda_{MI}\mathcal L_{CMI}
+0.1\mathcal L_{variant}.
\]

首轮只扫描：

```text
lambda_MI in {0, 0.03, 0.1, 0.3}
temperature in {0.07, 0.2}
```

其他结构和数据完全相同。选定超参数后固定，再运行三个 seeds。

部署 reward：

\[
r_t=V(o_{t+1},g,c)-V(o_t,g,c).
\]

第一版使用 \(\gamma=1\)，避免 value offset 与 discount 混淆。

---

## 5. 方法与 baseline 矩阵

| ID | 方法 | MI 的角色 | 是否 deployable |
|---|---|---|---|
| B0 | Random / constant | 无 | 是 |
| B1 | RGB SSD / NCC / LPIPS | goal similarity | 是 |
| B2 | DINO cosine / VIP-style distance | latent similarity | 是 |
| B3 | raw RGB Dame MI | scalar reward | 是 |
| B4 | latent Dame MI | scalar reward | 是 |
| B5 | Base task-conditioned value head | 无 MI | 是 |
| B6 | Base + cross-view InfoNCE | observation MI | 是 |
| M1 | Base + task-conditioned physical MI | proposed auxiliary MI | 是；physical branch 训练后删除 |
| M2 | M1 + audited Cosmos positives | proposed variance treatment | 是 |
| O1 | privileged physical score | oracle upper bound | 否 |
| Q1 | existing Qwen3-VL reward | external learned baseline | 是 |

必须保留 B5。没有 B5，就不能把任何提升归因到 MI。

---

## 6. Benchmark 设计

### 6.1 Benchmark A：LIBERO-Spatial reward ranking

当前已安装，作为主 benchmark。

建议规模：

```text
10 tasks
20 held-out initial states per task
5 rollout qualities per state
= 1000 trajectories
```

五类轨迹：

1. successful demo replay；
2. small action noise；
3. medium action noise；
4. open-gripper counterfactual；
5. temporal/action shuffle 或 early termination。

所有 success、stage 和 physical factors 均由 simulator 记录。训练、验证、测试按 episode identity 分离；测试 initial states 不用于构造训练 preference。

主指标：

- matched-state pairwise accuracy；
- success/failure AUROC；
- groupwise Spearman/Kendall；
- failed-trajectory false-positive rate；
- reward margin；
- calibration ECE；
- best-of-K candidate-selection regret。

### 6.2 Benchmark B：LIBERO visual-variance stress test

对 Benchmark A 的相同测试状态生成视觉变体，物理状态和标签保持不变：

- illumination；
- texture/background；
- object color；
- camera perturbation；
- partial occlusion。

优先使用 simulator rendering。现有 Cosmos Transfer 结果只有通过 geometry-preservation audit 的样本才能进入正式结果。

指标：

\[
\mathrm{RewardStd}_{variant},\quad
\mathrm{PairFlipRate},\quad
\Delta\mathrm{AUROC},\quad
\Delta\mathrm{RankingAcc}.
\]

该 benchmark 是 M1/M2 的主要证据，因为 v8 的目标就是减少 scene-level variance。

### 6.3 Benchmark C：Robometer / RBM-EVAL 外部 OOD

本地已有：

- MetaWorld quality/reward-alignment；
- USC-Koch policy ranking。

必须重新运行当前模型推理，不能使用旧版保存预测作为新结果。旧结果存在大量 ties，历史 `0.9772` preference accuracy 经过严格重算后只有 `0.0835` strict accuracy，因此论文只报告 fresh inference。

该 benchmark 只评价 OOD reward generalization，不作为训练集，也不替代 LIBERO 的独立 simulator truth。

### 6.4 可选 Benchmark D：第二个 LIBERO suite

如果 9 月 14 日前可以取得官方数据，优先顺序：

```text
LIBERO-Object > LIBERO-Goal > LIBERO-10
```

只运行 reward ranking 与 visual-shift evaluation，不训练新的大模型。若数据未及时准备，则不阻塞投稿。

---

## 7. 最小主结果表

### 表 1：clean reward quality

| Method | Pair Acc. ↑ | Success AUROC ↑ | Spearman ↑ | Failure FP ↓ | Best-of-K regret ↓ |
|---|---:|---:|---:|---:|---:|
| RGB Dame MI | TBD | TBD | TBD | TBD | TBD |
| Latent Dame MI | TBD | TBD | TBD | TBD | TBD |
| DINO cosine | TBD | TBD | TBD | TBD | TBD |
| Base value | TBD | TBD | TBD | TBD | TBD |
| Base + physical CMI | TBD | TBD | TBD | TBD | TBD |
| Privileged oracle | TBD | TBD | TBD | TBD | TBD |

### 表 2：scene/instance variance

| Method | Reward std. ↓ | Pair flip ↓ | AUROC drop ↓ | Rank drop ↓ |
|---|---:|---:|---:|---:|
| Base value | TBD | TBD | TBD | TBD |
| + cross-view MI | TBD | TBD | TBD | TBD |
| + physical CMI | TBD | TBD | TBD | TBD |
| + physical CMI + variants | TBD | TBD | TBD | TBD |

### 表 3：外部 OOD

| Method | MetaWorld quality pref. ↑ | Reward alignment ↑ | USC-Koch ranking ↑ | Tie rate ↓ |
|---|---:|---:|---:|---:|
| Existing Qwen reward | TBD | TBD | TBD | TBD |
| Base value | TBD | TBD | TBD | TBD |
| Base + physical CMI | TBD | TBD | TBD | TBD |

### 表 4：MI 角色消融

| Variant | Raw MI reward | Conditional negatives | Physical positives | Variant positives | Pair Acc. | Pair flip |
|---|---:|---:|---:|---:|---:|---:|
| Raw Dame | Yes | No | No | No | TBD | TBD |
| Base | No | No | No | No | TBD | TBD |
| Cross-view InfoNCE | No | Yes | No | Yes | TBD | TBD |
| Physical MI | No | Yes | Yes | No | TBD | TBD |
| Full | No | Yes | Yes | Yes | TBD | TBD |

所有 `TBD` 必须从保存的 JSON 聚合，不手工填写。

---

## 8. 最小判定标准

### Method gate

M1 相比 B5 至少满足：

1. 三个 seeds 中至少两个提高 matched-state pairwise accuracy；
2. mean pairwise accuracy 提升至少 2 percentage points；
3. visual-shift pair flip rate 相对下降至少 15%；
4. clean success AUROC 不下降超过 1 percentage point；
5. physical-factor linear probe 明显优于 B5。

只有通过这些条件，摘要才写“MI improves reward robustness”。

### Negative-result fallback

如果 M1 不通过，论文主张改为：

> A controlled study shows that raw image/latent MI and MI representation regularization do not reliably induce manipulation reward; physical supervision and stage structure dominate performance.

此时仍可提交练手稿，但不能写成提出了有效的新 reward model，也不能选择性删除 v7/E1 失败结果。

---

## 9. 两周执行顺序

### 9 月 10--11 日：实现最小 head

1. 在 `VisualGoalPotential` 的旁路新增 task embedding、stage head 和 physical head；
2. 新增 symmetric conditional InfoNCE；
3. 保持 DINO backbone frozen；
4. 用现有 LIBERO-Spatial cache 完成 1000-step overfit 和 held-out smoke；
5. 输出每项 loss、gradient norm 和 representation collapse 指标。

### 9 月 12 日：最小 MI gate

只跑 B5 与 M1：

```text
1 task × existing held-out trajectories × 3 seeds
```

若 M1 连 physical probe 或 pairwise ranking 都没有改善，停止调大模型；检查 negative grouping 和 factor normalization一次，然后冻结结果。

### 9 月 13--15 日：LIBERO 主实验

1. 生成/整理 matched-state rollout qualities；
2. 运行 B1--B6、M1、M2；
3. 运行 clean 和 visual-shift；
4. 生成所有逐轨迹 JSON 和 bootstrap 95% CI。

### 9 月 16 日：OOD 和候选选择

1. fresh Robometer inference；
2. best-of-K candidate selection；
3. 若已有 RLPD 可运行，只做一个 task、三个 seeds 的补充实验。

### 9 月 17 日：冻结主张

根据真实结果在正向方法稿与 negative-result fallback 之间二选一，冻结标题、摘要、方法名和主结果表。

### 9 月 18 日：提交真实摘要

摘要必须与当前结果一致，不提交占位摘要；作者列表在该日冻结。

### 9 月 19--23 日：写作与图表

建议正文结构：

```text
1 Introduction
2 Why pairwise MI is not manipulation progress
3 Physically conditioned MI reward learning
4 Controlled evaluation protocol
5 Results and failure analysis
6 Related work
7 Limitations
```

### 9 月 24 日：完整审计

- 数字与 JSON 一致；
- train/test episode identity 无泄漏；
- fresh inference 与 historical rescore 分开；
- 匿名、页数、引用、supplement 和代码命令检查。

### 9 月 25 日：提交

保留至少 12 小时用于 PDF、OpenReview 和作者资料问题。

---

## 10. 明确不做的内容

在本次截止前不做：

- 继续扩大 strict TRO wrist-camera reproduction；
- 再训练 LaWAM 或 Cosmos foundation model；
- 用 raw MI 生成 Qwen3-VL 伪标签；
- 完整 PPO/SAC benchmark；
- 真机实验；
- 7-DoF MI Jacobian；
- 把 Cosmos 图像当成独立物理真值；
- 把 Robometer 历史 tie-heavy 指标作为正结果。

这些工作都不会解决当前论文最关键的问题：MI 辅助项是否在独立 physical truth 上改善 reward ranking 和 visual robustness。

---

## 11. 论文贡献的允许写法

若 M1/M2 通过 gate，可以写三项贡献：

1. 通过 latent、geometric 和 strict image-space 三层审计，说明 pairwise MI 何时不能作为 manipulation reward；
2. 提出 task-conditioned physical-information regularization，使 observation-only reward representation 在训练时保留任务相关物理因素；
3. 在 LIBERO matched-state ranking、visual-shift 和外部 Robometer OOD 上独立评价 reward quality，并报告 raw MI、无 MI 和不同 MI 角色的完整消融。

不能写：

- MI 表示通用物理进度；
- MI 提供正确动作方向；
- 复现证明 TRO 错误；
- 仅凭离线 correlation 就证明 RL policy improvement；
- Cosmos variant 自动保持物理状态。

---

## 12. 当前项目状态

```text
Broad raw-MI directional reward: falsified
Strict TRO derivative chain: passed
Strict TRO reference localization: failed under current estimator/configuration
Task-conditioned observation-only value head: existing base implementation
Physical conditional MI regularizer: to implement
LIBERO-Spatial simulator benchmark: available
Cosmos visual variants: available, requires preservation audit
Robometer OOD datasets: available, requires fresh inference
Qwen3-VL: baseline only until small-head gate passes
```

当前唯一 P0 是：用现有 cached features 做 B5 versus M1 的最小三 seed 实验。它可以在最短时间内决定 MI 在这篇稿子里是正向方法，还是系统性负结果的一部分。

---

## 12. M1 最小 cached-feature smoke（2026-09-10）

已在 LIBERO-Spatial task-00 的冻结 DINO cache 上执行 B5 versus M1：`demo_0` 和
`demo_1` 为 train，`demo_10` 为 held-out；各方法使用 seeds `7/17/27` 和 1000
steps。两者共享 pairwise physical-progress ranking、stage classification 与 physical-factor
regression；M1 唯一额外项为 task/stage-conditioned symmetric physical InfoNCE。

physical factors 由同一 MuJoCo states replay 构造，维度为 12：EEF-object distance、
object-goal distance、两个相对 displacement、gripper command、goal contact、grasped 和
object height。DINO backbone 未重新训练；physical branch 只在训练/linear-probe 中使用。

正式记录：

```text
logs/mi_reward/v8_physical_mi/m1_minimal_gate_smoke_v1/results.json
logs/mi_reward/v8_physical_mi/m1_minimal_gate_smoke_v1.live.log
```

| Method | Held-out pair acc. | Held-out stage acc. | Held-out physical linear-probe R2 |
|---|---:|---:|---:|
| B5 base | 0.9436 | 0.9355 | 0.0468 |
| M1 physical CMI | 0.9443 | 0.9606 | 0.4420 |

M1 在 seeds 17/27 提高 held-out pair accuracy、在 seed 7 略低；mean improvement
仅 `+0.0007`（`+0.07` percentage points），没有达到第 8 节的 `+2` percentage-point gate。
同时，physical probe 从接近零提高到 `0.4420`，说明 conditional physical MI 确实使视觉
representation 保留更多 held-out physical factor，而该信息尚未转化为当前 easy temporal
ranking 的明显提升。

本实验没有 matched-state visual-shift variants，也没有 success/failure held-out mixture，
所以 pair-flip 与 success-AUROC 两项不可评估。结论冻结为：`REPRESENTATION_SIGNAL_PASS,
REWARD_METHOD_GATE_NOT_YET_PASSED`。下一步不是调大模型，而是用同一 state 的视觉变体构造
hard matched-state pairs，再重跑相同 B5/M1 三-seed gate。

## 13. M1 matched-state appearance-shift gate（2026-09-10）

为补足上一轮 smoke 缺少 visual-shift 的问题，使用同一 LIBERO-Spatial task-00 MuJoCo
state 的 agentview RGB，施加固定 camera-only appearance transforms：`dark_warm`、
`bright_cool` 和 `low_contrast`。这些变体不是 Cosmos 生成图，也没有改变 state、action 或
physical factors。使用本地冻结 DINOv3 ViT-B/16 重新提取四种视图的 features，B5/M1
均使用相同的 3 seeds 和 1000 steps；M1 的 CMI 使用同一 physical state 的多视图作为
positive set。

正式记录：

```text
logs/mi_reward/v8_physical_mi/m1_matched_state_appearance_v1/results.json
logs/mi_reward/v8_physical_mi/m1_matched_state_appearance_v1.live.log
```

| Method | Held-out pair acc. | Physical probe R2 | Mean pair-flip rate |
|---|---:|---:|---:|
| B5 base | 0.8283 | -1.8491 | 0.0358 |
| M1 physical CMI | 0.8012 | -2.5195 | 0.0310 |

M1 的 held-out pair accuracy 比 B5 低 `2.71` percentage points，physical probe 也没有
提升。pair-flip rate 平均只小幅下降，且三个 appearance variant 的方向并不一致。该结果
不满足 M1 的 pair-accuracy gate，也不支持“physical CMI 提高 reward robustness”的主张。

当前 v8 状态更新为：

```text
M1 physical representation signal: inconclusive across cache choices
M1 clean reward gate: failed
M1 matched-state visual-shift gate: failed
M2 Cosmos positives: blocked
```

在此证据下，不继续扩大 M1 到多任务、多 seed 或 M2。v8 的负结果 fallback 应保留为主
解释：physical supervision 可以被 probe 读出，但当前 conditional MI auxiliary 没有稳定
改善 deployable reward ranking。

## 14. M1 corrected single-task protocol（2026-09-10）

按照修正版要求，只在 task-00 单任务执行一次正式实验，不扩展到多任务。实现固定了四种
同一 state 的视觉版本（`none`、`dark_warm`、`bright_cool`、`low_contrast`）同批采样，
并修正了 all-pair、within-stage、hard-pair、pair-flip 的统计。连续 physical factors 使用
逐因子的 ridge-probe R²，二值因素使用 AUROC；每帧保存 value、stage、physical prediction
和 hidden feature。`conditional_mi` 与 `physical_regression_conditional_mi` 的
`lambda_MI` 只在 validation demos（demo_1）上从 `{0.03, 0.1, 0.3}` 选择，held-out
demo_10 只评估一次。

正式记录：

```text
logs/mi_reward/v8_physical_mi/m1_corrected_single_task_v1/results.json
logs/mi_reward/v8_physical_mi/m1_corrected_single_task_v1.live.log
logs/mi_reward/v8_physical_mi/m1_corrected_single_task_v1/predictions/selected_models_frame_outputs.pt
```

三 seed 的 held-out 均值如下（seed-level 结果保存在 JSON）：

| Group | λMI（validation 选择） | All-pair | Within-stage | Hard-pair | dark/bright/low pair-flip | continuous R² mean | binary AUROC (f9/f10) |
|---|---:|---:|---:|---:|---|---:|---|
| rank + stage | 0 | 0.8331 | 0.8299 | 0.6179 | 0.0649 / 0.0438 / 0.0775 | 1.0000 | 0.2021 / 0.5305 |
| + physical regression | 0 | 0.8045 | 0.8021 | 0.6518 | 0.0808 / 0.0413 / 0.0920 | 1.0000 | 0.7081 / 0.6710 |
| + conditional MI | 0.03–0.30 | 0.7931 | 0.7958 | 0.6419 | 0.0798 / 0.0486 / 0.0907 | 1.0000 | 0.4786 / 0.6003 |
| physical regression + conditional MI | 0.10–0.30 | 0.7869 | 0.7905 | 0.6257 | 0.0715 / 0.0416 / 0.0792 | 1.0000 | 0.7465 / 0.6278 |

修正版的主要结论是：validation 上 conditional-MI 组的 all-pair 可以升到约 0.874，
但 held-out all-pair 下降到 0.793；加入 physical regression 后 held-out pair 进一步
下降。physical regression 确实改善了两个二值 physical factor 的 AUROC，尤其联合组的
`f9=0.7465`，但没有改善主 reward ranking。四个视觉版本已进入同一 minibatch，因而这次
结果不再是旧版 pair 统计或 batch 构造问题造成的假象。

需要保留一个诊断限制：continuous R² 在当前 frame-level probe 划分上接近 1，说明同一
state 的多视图重复使 probe 很容易恢复 state-correlated factors；它不能被解释为跨
state 泛化证据。下一轮若继续，应先做按 state/demo 分组的 probe split，再决定是否保留
physical auxiliary；不应据此推进多任务或 Cosmos M2。
