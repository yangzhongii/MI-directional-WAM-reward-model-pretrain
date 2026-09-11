# Pipeline v5-A：Privileged Physical Teacher → Observation-Only Reward Model

## 0. 版本定位

v5-A 将 v5 的目标从“直接证明 MI 可以作为 autonomous approach direction”调整为：

> 使用 privileged physical progress 构造任务正确的 teacher，再训练只依赖视觉观测的 action-conditioned reward model；MI 只作为待验证的 reference-consistency / directional regularizer。

这仍然是 reward-model pipeline，不是 VLA。模型的目标是给定 observation、reference 和候选 action 输出 reward 或 progress score：

\[
r_\theta(o_t,o_g,a_t)\rightarrow \mathbb R.
\]

它不直接以 language 作为输入，也不直接输出 policy action。

## 1. v5 已经得到的事实

当前 v5 的 frozen full-frame raw-RGB branch 使用：

- LIBERO simulator 原始渲染的 wrist RGB；
- 固定 external reference；
- 实际 EEF `delta xyz` 作为 action coordinate；
- Dame-style channelwise soft-histogram MI；
- primary command `0.18`、secondary command `0.15`、两步 rollout；
- 每个 anchor 24 个 held-out actions。

20-anchor 正式结果为：

| 指标 | 结果 | 结论 |
|---|---:|---|
| pullback/direct-FD gradient cosine | `0.99979` | 数学梯度和实现一致 |
| cross-epsilon Jacobian cosine | `0.94752` | 局部 action geometry 稳定 |
| held-out MI R² | `0.96932` | 能预测自身 MI 改变量 |
| held-out Spearman | `0.98578` | MI 增量排序稳定 |
| held-out sign accuracy | `0.94375` | MI 增减判断稳定 |
| physical EEF-object descent cosine | `0.68842` | 未达到 `0.70` gate |
| MI-ascent distance-descent fraction | `0.95000` | 大多数 action 使距离下降 |

因此 frozen full-frame raw-RGB MI 的正式状态是 **NO-GO**：它形成了稳定、可预测的视觉 action field，但还不能被宣称为跨 state 的 physical approach vector field。

这个结果不支持以下表述：

- MI 已经是可靠的 autonomous approach reward；
- MI gradient 已经可以直接驱动 planner 或 Newton update；
- LAM latent 和 raw RGB 都已经完成 reward-model 验证。

## 2. v5-A 的核心假设

任务正确性和视觉 reference consistency 应被分开验证：

\[
\text{physical progress}
\neq
\text{visual MI change}
\]

但 MI 仍可能作为 observation-only reward model 的辅助信号，帮助模型判断：

1. 当前 observation 是否朝 reference 变化；
2. 候选 action 是否产生稳定的视觉进展；
3. reward representation 是否保留局部 action sensitivity。

v5-A 首先验证 privileged teacher 能否被视觉 reward model 学到，再验证加入 MI regularization 是否改善 held-out action ranking 和跨 instance / scene 泛化。

## 3. Teacher、student 和部署边界

### 3.1 Privileged physical teacher

在 simulator 中使用不可部署的物理状态计算训练标签。令 (d_t) 为 EEF 到 task object 的距离，定义：

\[
y_{\text{phys}}(s_t,a_t)=d(s_t)-d(s_{t+1}).
\]

正值表示 action 使机器人朝目标物体接近。对于需要多维方向的评估，可用一组正负轴向 probe 拟合：

\[
u_{\text{phys}}(s_t)
=
\arg\max_{\|u\|=1}
\mathbb E_{\delta a}[y_{\text{phys}}(s_t,\delta a)] .
\]

该 teacher 只用于训练标签、验证和 oracle baseline。部署时不输入 EEF-object distance、object pose 或 simulator state。

### 3.2 Observation-only reward student

student 输入为：

\[
o_t,\quad o_g,\quad a_t,
\]

输出 physical progress prediction：

\[
r_\theta(o_t,o_g,a_t)\approx y_{\text{phys}}(s_t,a_t).

\]

候选 action 使用实际 EEF `delta xyz`，不引入 action latent encoder 作为当前 v5-A 的必要组成部分。

### 3.3 MI auxiliary branch

MI 分支计算：

\[
M_\phi(o_t,o_g).
\]

如果某个视觉输入 variant 尚未通过 physical-direction gate，MI 只能作为诊断量，不能作为主监督。只有在 ROI 或 reference-conditioned variant 通过 physical gate 后，才允许将其加入 student 的辅助损失：

\[
\mathcal L
=
\mathcal L_{\text{phys}}
 +\lambda_{\text{rank}}\mathcal L_{\text{rank}}
 +\lambda_{\text{MI}}\mathcal L_{\text{MI-consistency}}.
\]

## 4. Reward-model 训练目标

### 4.1 Progress regression

对每个 candidate action 预测 privileged physical progress：

\[
\mathcal L_{\text{phys}}
=
\left(r_\theta(o_t,o_g,a_t)-y_{\text{phys}}(s_t,a_t)\right)^2.
\]

### 4.2 Pairwise action ranking

对于同一个 state 的两个 candidate actions (a_i,a_j)，如果：

\[
y_i>y_j,
\]

则要求：

\[
r_\theta(o_t,o_g,a_i)>r_\theta(o_t,o_g,a_j).
\]

这一步直接对应 action selection，而不是只拟合一个 state-level scalar。

### 4.3 MI consistency

只有通过 physical gate 的 MI variant 才能使用 MI consistency。可使用 action-pair ordering 或 held-out MI gain 作为辅助目标，但不得把 MI gain 直接当作 physical progress label。

## 5. Benchmark 设计

### 5.1 主 benchmark：LIBERO counterfactual action geometry

LIBERO 作为主 benchmark，使用 Spatial、Object 和 Goal 三类分布变化：

- Spatial：scene / spatial arrangement variation；
- Object：object instance variation；
- Goal：reference / goal variation。

最小 ICLR 版本选择 12 个 tasks，每个 suite 4 个 task。每个 task 至少包含：

- 3–5 个独立 initial states；
- 每个 state 4 个 smooth pre-contact anchors；
- 每个 anchor primary / secondary probes；
- 24 个 held-out actions；
- RGB、action、privileged physical progress 和 task metadata。

当前 v5 的 20-anchor collection 作为 development split，不应单独被表述为跨任务 benchmark。

### 5.2 外部 simulator 验证

如果时间和算力允许，在 ManiSkill3 中选择：

1. 一个 pick/place task；
2. 一个 push task；
3. 一个 articulated-object task。

只迁移 frozen reward/evaluator protocol，不重新为每个任务调 gate。

### 5.3 真机验证

真机不是 v5-A 的必要前置条件，但可以作为最终 sanity check。最小设置为固定相机、3 个桌面任务、多个 object instances、EEF pose 和 AprilTag/object pose 记录。真机只验证：在 simulator 中通过的 reward model 是否仍能预测 one-step physical progress。

## 6. Baselines 和消融

主实验至少包含：

| 方法 | 作用 |
|---|---|
| Random action score | 下界 |
| Pixel photometric / SSIM score | 普通视觉相似度 baseline |
| LAM latent MI | latent representation baseline |
| Full-frame raw RGB MI | 当前 frozen NO-GO baseline |
| Object ROI MI | 任务区域 variant |
| Privileged physical oracle | 上界，不作为 observation-only 方法 |
| Physical-teacher student without MI | reward-model 主 baseline |
| Physical-teacher student with MI auxiliary | v5-A 方法 |

所有对比使用相同 anchor、reference、candidate actions 和 held-out split。不能根据测试结果挑选 state 或调整 gate。

## 7. 评估指标和通过条件

### 7.1 Teacher-level metrics

- pullback/direct-FD gradient cosine；
- primary/secondary Jacobian cosine；
- held-out MI R²、Spearman、sign accuracy；
- physical direction cosine；
- distance-descent fraction。

### 7.2 Reward-model metrics

- held-out physical-progress R²；
- held-out pairwise action ranking accuracy；
- physical direction cosine；
- one-step distance reduction；
- cross-state、cross-object、cross-goal generalization；
- closed-loop success rate，作为最终应用指标。

### 7.3 Gate

一个 observation-only reward model 只有同时满足以下条件，才能作为主方法：

1. held-out action ranking 明显优于 random 和 visual-similarity baseline；
2. physical direction cosine 达到预注册阈值；
3. one-step physical progress 不低于 privileged-teacher student baseline 的统计范围；
4. 至少在一个未见 object 或未见 scene split 上保持优势。

如果加入 MI 后只提高 MI consistency，却不提高 physical progress 或 action ranking，则 MI 只能保留为分析工具，不能作为方法贡献。

## 8. 当前停止条件

### 路线 A：MI 仍有用

如果 object ROI 或 reference-conditioned MI 在固定 20 anchors 上达到 physical gate，并在额外 task / instance 上保持优势：

```text
ROI/reference-conditioned MI
        ↓
physical-teacher reward distillation
        ↓
observation-only action-conditioned reward model
        ↓
optional Hessian / planner integration
```

### 路线 B：MI 只是辅助信号

如果 full-frame、ROI、crop 和 reference-conditioned MI 都无法通过 physical gate：

```text
privileged physical teacher
        ↓
observation-only reward model
        ↓
MI 作为 auxiliary analysis / regularizer
```

此时不再声称 MI 产生 physical action direction，论文主线转为 privileged-to-observation reward distillation。

## 9. 当前工作顺序

1. 在 v5 的 20 个 anchors 上完成 full-frame、object ROI、object-centered crop 和 background-only 诊断；
2. 选择一个预注册的 ROI/reference-conditioned variant，不能按结果反复调 MI 参数；
3. 训练 physical-teacher student，先不加入 MI；
4. 加入 MI auxiliary，比较 action ranking、physical progress 和泛化；
5. 在 LIBERO 的 Spatial/Object/Goal split 上进行最小跨任务验证；
6. 只有 reward model 通过 physical gate 后，才考虑 Hessian、planner 或闭环 RL；
7. 如果 MI 没有增益，停止扩展 MI 数学主张。

## 10. 当前结论

v5-A 不再把 MI 直接等同于 reward，也不把当前实验包装为 VLA。当前最稳健的科学命题是：

> privileged physical progress 可以作为 observation-only action-conditioned reward model 的 teacher；MI 是否能改善该 reward model，需要通过固定 benchmark 上的 physical progress、action ranking 和跨分布泛化实验单独证明。

当前还没有训练完成的 reward model，也没有完成 benchmark 或真机验证。因此 v5-A 是一个可执行的 reward-model research pipeline，而不是已经完成的方法结果。
