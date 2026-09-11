# Pipeline v5：MI Action-Space Geometry 小实验

## 0. 目的与边界

Pipeline v5 暂时不训练 Qwen、不修改 v3/v4 teacher、不运行闭环 RL。它只验证一个新假设：

> 能否将 Dame–Marchand visual servoing 中的 MI 标量场，通过视觉/动力学 Jacobian 拉回机器人动作空间，并使用 action-space Hessian 构造可验证的局部动作方向？

v5 不再使用

\[
D_t=\gamma\phi_{t+1}-\phi_t+\beta\psi_t
\]

作为理论起点，conditional-action PMI 也不进入本实验。实验对象是一个可微 MI objective、一个明确的动作到视觉变化映射，以及该复合函数的一阶和二阶导数。

## 1. 核心数学

### 1.1 Visual MI objective

令当前视觉表示为 \(z_t\in\mathbb R^n\)，阶段参考为 \(z_g\)，定义：

\[
M(z_t,z_g)=MI(z_t,z_g).
\]

第一版使用仓库现有的 cubic B-spline soft-histogram estimator：

~~~text
mi_reward/scoring/dame_soft_histogram.py::DameSoftHistogramMI
~~~

它对应 TRO 中经过平滑的可微 histogram MI，但第一轮在 latent patch tokens 上计算，因此只能称为 Dame-style latent MI。

### 1.2 Jacobian 与动作梯度

令动作 \(a_t\in\mathbb R^m\)，局部视觉动力学为

\[
z_{t+1}=f(z_t,a_t).
\]

动作 Jacobian、视觉梯度和动作梯度分别为：

\[
J_a=\frac{\partial z_{t+1}}{\partial a_t}\in\mathbb R^{n\times m},
\]

\[
g_z=\nabla_zM(z_{t+1},z_g)\in\mathbb R^n,
\]

\[
g_a=\nabla_aM(f(z_t,a_t),z_g)=J_a^\top g_z\in\mathbb R^m.
\]

\(J_a^\top g_z\) 是 visual MI 与 action direction 之间的数学连接。

### 1.3 Hessian

令

\[
H_z=\nabla_z^2M(z,z_g).
\]

如果局部动力学为线性映射 \(z_{t+1}=z_0+J_aa_t\)，则

\[
H_a=J_a^\top H_zJ_a.
\]

如果 \(f\) 非线性，完整 Hessian 是

\[
H_a
=J_a^\top H_zJ_a
+\sum_{k=1}^{n}
\frac{\partial M}{\partial z_k}\nabla_a^2f_k.
\]

所以 \(J_a^\top H_zJ_a\) 在非线性 dynamics 下只是 pullback 近似。正式实现应优先直接对复合目标 \(M(f(z_t,a),z_g)\) 求 action-space Hessian-vector product。

### 1.4 局部动作提议

二阶局部 MI improvement：

\[
\widehat{\Delta M}(\delta a)
=g_a^\top\delta a+\frac12\delta a^\top H_a\delta a.
\]

定义阻尼曲率矩阵：

\[
K_a=-\frac12(H_a+H_a^\top)+\lambda I,
\]

并使用 trust-region Newton step：

\[
\delta a_N=\eta K_a^{-1}g_a,\qquad \|\delta a_N\|\le\rho.
\]

一阶基线为：

\[
\delta a_G=\eta\frac{g_a}{\|g_a\|}.
\]

Hessian 只有在相对梯度基线产生可复现改善时才保留。

## 2. Test 0：纯数学与实现恒等式

### 2.1 设置

构造已知线性 latent dynamics：

\[
z(a)=z_{base}+Ba.
\]

使用真实 DameSoftHistogramMI 比较：

1. 直接 autograd 动作梯度；
2. pullback 梯度 \(B^\top\nabla_zM\)；
3. 直接 action Hessian；
4. pullback Hessian \(B^\top H_zB\)；
5. 中心有限差分梯度和 Hessian；
6. gradient step 与 damped Newton step 后的 MI。

### 2.2 已执行结果

状态：**PASS**。

| 检查 | 结果 |
|---|---:|
| Gradient pullback 最大绝对误差 | \(0.000\times10^0\) |
| Hessian pullback 最大绝对误差 | \(1.041\times10^{-17}\) |
| Gradient 有限差分相对误差 | \(1.557\times10^{-7}\) |
| Hessian 有限差分相对误差 | \(1.791\times10^{-6}\) |
| 初始 MI | 0.452155657 |
| Gradient step 后 MI | 0.452209217 |
| Damped Newton step 后 MI | 0.452598809 |

这证明在线性 latent dynamics 和当前可微 Dame MI 实现下，

\[
g_a=J_a^\top g_z,\qquad H_a=J_a^\top H_zJ_a
\]

的数学关系与代码实现一致。它不证明 LIBERO 中的 MI 梯度与物理任务方向一致。

实现和输出：

~~~text
eval/libero/probe_v5_mi_action_geometry.py
logs/mi_reward/v5_mi_action_geometry/math_identity_v1/
~~~

复现命令使用新的输出目录：

~~~bash
.venv-libero/bin/python eval/libero/probe_v5_mi_action_geometry.py \
  --output-dir logs/mi_reward/v5_mi_action_geometry/math_identity_reproduction
~~~

## 3. Test 1：LIBERO 局部 action-space MI geometry

### 3.1 只回答三个问题

1. simulator 中的局部视觉 Jacobian 能否预测动作导致的 MI 变化；
2. MI action gradient 是否与 privileged 几何进度方向一致；
3. Hessian 是否比普通 gradient ascent 提供额外改善。

本实验不评价完整 manipulation reward，不包含 Qwen，也不跨越 grasp/contact 不连续点。

### 3.2 限定阶段

第一轮只选择一个 LIBERO task 的 pre-contact approach 阶段：

- 尚未接触和抓取对象；
- gripper command 固定；
- 目标是调整 EEF 相对对象的位置；
- 从至少 5 个 initial states 选取 20 个 anchor states；
- 每个 anchor 的扰动均从完全相同的 simulator state 恢复；
- 产生 contact、grasp 或 success 切换的扰动全部排除并记录原因。

这使局部动力学保持平滑，让 Hessian 假设在第一轮有意义。

### 3.3 阶段参考与视角

每个 anchor 使用独立成功示范中的 pre-grasp aligned reference，不直接使用整个任务的 terminal image。

第一轮分别比较：

- wrist full-frame MI；
- wrist object/gripper ROI MI；
- agentview ROI MI；
- dual-view weighted MI。

ROI 由 simulator object projection 或冻结 mask 提供，只用于诊断。若 full-frame 失败而 ROI 通过，说明背景像素破坏了 MI field。

### 3.4 固定 normalization

当前 robust_minmax 会针对每个图像对重新计算 normalization。局部导数实验不能让 normalization 随候选动作改变，否则梯度会同时包含尺度重定义。

Test 1 必须：

1. 用 reference 与 anchor neighborhood 预拟合 normalization；
2. 在同一 anchor 的全部扰动中冻结 shift、scale 和 histogram bins；
3. 报告 clamp saturation ratio；
4. saturation 过高时扩大固定 bin range，不根据实验结果逐对归一化。

### 3.5 动作空间和时间跨度

第一轮使用三维 EEF translation：

\[
a=(\Delta x,\Delta y,\Delta z)\in\mathbb R^3.
\]

orientation 和 gripper 固定。每个动作持续 3–5 个 simulator steps，使视觉变化超过单帧亚毫米噪声，同时保持在局部 trust region。

三维实验通过后再扩展到 6DoF EEF twist、7DoF joint velocity 和 grasp 后的 object transport。

### 3.6 Jacobian 估计

对每个动作维度施加正负扰动：

\[
J_a[:,i]\approx
\frac{z(a_0+\epsilon e_i)-z(a_0-\epsilon e_i)}{2\epsilon}.
\]

三维动作每个 anchor 需要 6 个 Jacobian rollout，加中心状态。使用两个 \(\epsilon\) 检查局部线性稳定性。

MI 对 token 的梯度由 autograd 得到：

\[
g_z=\nabla_zM(z,z_g),\qquad
g_a^{pullback}=J_a^\top g_z.
\]

同时直接对 simulator action 做有限差分：

\[
g_{a,i}^{FD}\approx
\frac{M(a_0+\epsilon e_i)-M(a_0-\epsilon e_i)}{2\epsilon}.
\]

两者必须独立计算并比较，避免共同实现错误相互抵消。

### 3.7 Action-space Hessian

动作维度只有 3，可以直接估计完整对称 Hessian：

\[
H_{ii}\approx
\frac{M(a+\epsilon e_i)-2M(a)+M(a-\epsilon e_i)}{\epsilon^2},
\]

\[
H_{ij}\approx
\frac{
M(a+\epsilon e_i+\epsilon e_j)
-M(a+\epsilon e_i-\epsilon e_j)
-M(a-\epsilon e_i+\epsilon e_j)
+M(a-\epsilon e_i-\epsilon e_j)
}{4\epsilon^2}.
\]

同时计算 token pullback \(J_a^\top H_zJ_a\)。二者差异反映 dynamics curvature、renderer 非线性、feature encoder 非线性和有限差分误差。

### 3.8 Held-out action 验证

矩阵通过轴向扰动估计后，另采样至少 24 个 held-out 小动作，不参与矩阵估计。

比较：

\[
\widehat{\Delta M}^{(1)}=g_a^\top\delta a,
\]

\[
\widehat{\Delta M}^{(2)}
=g_a^\top\delta a+\frac12\delta a^\top H_a\delta a,
\]

与 simulator 实测：

\[
\Delta M
=M(f(z_t,a_0+\delta a),z_g)-M(f(z_t,a_0),z_g).
\]

报告：

- first/second-order \(R^2\)；
- \(\Delta M\) 符号准确率；
- predicted/actual action ranking Spearman；
- pullback/finite-difference gradient cosine；
- Hessian 特征值、条件数和 damping；
- MI improvement 与 EEF-object distance improvement 的一致率。

## 4. Test 1 预注册通过标准

### 4.1 数学与数值

- \(g_a^{pullback}\) 与 \(g_a^{FD}\) median cosine ≥ 0.90；
- gradient relative error median ≤ 0.20；
- Hessian symmetry error ≤ \(10^{-4}\)，或相对 Frobenius error ≤ 1%；
- 两个 \(\epsilon\) 下 gradient cosine median ≥ 0.90；
- 无 NaN/Inf，clamp saturation 不过度集中。

### 4.2 局部预测

- held-out action 的 \(\Delta M\) sign accuracy ≥ 80%；
- first-order \(R^2\ge0.50\)；
- second-order \(R^2\) 不低于 first-order；
- action ranking Spearman ≥ 0.60。

### 4.3 物理方向

- MI gradient action 使 EEF-object distance 减小的 anchor 比例 ≥ 70%；
- MI gradient 与 privileged distance-descent direction median cosine ≥ 0.70；
- 至少 4/5 initial states 的方向一致率高于随机动作；
- paired initial-state bootstrap 的改善区间不能完全落在零以下。

### 4.4 Hessian 保留条件

只有同时满足以下条件才保留 Hessian：

- damped Newton step 的实际 MI gain 不低于 gradient step；
- physical distance improvement 不显著恶化；
- 至少一个指标有明确 paired 改善，或在相同增益下明显减少动作范数/振荡。

若 Jacobian-gradient 通过而 Hessian 无增益，v5 保留一阶 MI action field，删除二阶主张。

## 5. 失败结果解释

### 5.1 Test 1 实测：wrist full-frame MI local field（2026-09-08）

状态：**NO-GO（当前 wrist full-frame objective）**。

固定 `libero_spatial/task-00`，使用独立 `demo_0` 的 pre-contact reference，
`demo_1`--`demo_5` 各取一个 pre-contact anchor。每个 anchor 从相同保存的 simulator
state 恢复，执行 4 个固定 gripper command steps，并收集：中心状态、两套 epsilon 的
轴向动作、primary-epsilon cross probes、24 个 held-out 3D actions。采集结果为 5 个
anchor、245 个 candidate，排除数为 0；state restore 的 EEF、task-object、goal-object
偏差均为 0。

冻结 LAM wrist patch tokens 使用外部固定 normalization 和
`DameSoftHistogramMI(normalization="none", channelwise)`。结果：

| 指标 | v5 门槛 | 实测 |
|---|---:|---:|
| pullback/direct FD gradient cosine median | >= 0.90 | **0.9981** |
| held-out actions | >= 24 | **120** |
| held-out first-order MI R2 | >= 0.50 | **-0.2490** |
| held-out MI ranking Spearman | >= 0.60 | **-0.0755** |
| held-out MI sign accuracy | >= 0.80 | **0.5167** |
| clamp saturation | 不应集中 | **0.0** |

每个 anchor 的 gradient cosine 均在 `0.9906--0.9984`，因此
`g_a = J_a^T g_z` 与 primary-axis simulator finite difference 的数值链路通过。
但五个 anchor 的 held-out R2 分别为 `-1.597, -1.663, -1.556, -1.351, 0.011`；没有
一个达到局部预测门槛。

这排除了“Jacobian pullback 的一阶实现错误”作为当前失败的主要解释。当前证据只支持：
轴向微扰下的导数可以被计算；它**不支持**该 wrist full-frame MI field 在预登记的
held-out action ball 内预测实际 MI 改变量。可能来源包括局部线性区域过小、视觉
dynamics/renderer 的非线性，或 full-frame MI/reference 不是有用的局部 objective；
本实验不能区分这些解释。

根据 4.2 的 held-out Go 条件，wrist full-frame variant 不进入 Hessian proposal、EEF
physical-direction claim、planner、Qwen fusion 或 RL。`eval_v5_local_geometry.py` 曾只
按 derivative cosine 和 held-out count 标为 PASS，现已修复为同时检查 R2、Spearman 和
sign accuracy；修正后的正式结果为 `NO_GO`。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/libero_spatial_task00_v1/collection.json
logs/mi_reward/v5_mi_action_geometry/libero_spatial_task00_wrist_eval_v2/results.json
~~~

### 5.2 修复阶段与扰动后的复测（2026-09-08）

5.1 的 frame-0 protocol error 已修复后重测。reference 固定为
`demo_0/frame_34`（EEF-object distance `0.0845 m`）；通过小扰动 calibration 的四个
anchors 为 `demo_1/frame_31`、`demo_2/frame_36`、`demo_4/frame_41`、
`demo_5/frame_36`，其距离为 `0.0804--0.0950 m`。使用 2 simulator steps、primary /
secondary command `0.012 / 0.006`、held-out radius `0.012`。196 个 candidates 中 0 个
跨越 object-motion/contact/grasp/success exclusion，留下 96 个 held-out actions。

| 指标 | v5 门槛 | 修复后实测 |
|---|---:|---:|
| pullback/direct FD gradient cosine median | >= 0.90 | **0.9998** |
| held-out first-order MI R2 | >= 0.50 | **0.0620** |
| held-out MI ranking Spearman | >= 0.60 | **0.3633** |
| held-out MI sign accuracy | >= 0.80 | **0.6354** |

相对于 frame-0/4-step run，held-out R2 从 `-0.2490` 改善到 `0.0620`，Spearman 从
`-0.0755` 改善到 `0.3633`，sign accuracy 从 `0.5167` 改善到 `0.6354`。这确认阶段和
扰动协议确实是原失败的重要混杂因素，但修复后仍没有达到 Go 门槛。四个 anchor 中仅
`demo_2/frame_36` 的 R2/Spearman 达到 `0.6648/0.8330`，但 sign accuracy 为 `0.75`；
其余 anchors 的 R2 为负。因此不能把单一 anchor 的局部成功解释为稳定 action field。

结论：当前 wrist full-frame LAM latent MI 在经过正确阶段选择和无物体扰动的更小 action
ball 后仍为 **NO_GO**。Hessian、physical-direction、planner、Qwen fusion 与 RL 继续
冻结。下一轮若继续，只应按 7.4 比较 raw RGB / ROI / latent ROI objective，并报告
primary-secondary token Jacobian stability；不能再把 full-frame wrist variant 作为主线。

正式输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_heldout_v1/collection.json
logs/mi_reward/v5_mi_action_geometry/far_approach_wrist_eval_v1/results.json
~~~

| 结果 | 结论 |
|---|---|
| Pullback 与 direct finite difference 不一致 | Jacobian、normalization、坐标或实现错误 |
| 导数一致，但 MI 与物理方向相反 | MI objective/reference/ROI 没形成正确任务场 |
| 一阶预测通过，二阶失败 | 局部非线性或 Hessian 噪声；保留 gradient |
| Approach 通过，contact 后失败 | 平滑 visual servoing 仅适用于阶段内 |
| Wrist ROI 通过，full-frame 失败 | 需要任务区域选择 |
| 固定 scene 通过，appearance shift 失败 | MI 没有获得所需 scene invariance |
| 所有 MI 版本失败而几何 oracle 通过 | 当前视觉 MI 表示不适合该任务 |

## 6. Instance/scene variance 后续测试

Test 1 先固定 instance 与 scene。通过后保持相同 simulator anchor/action，只改变：

1. object color/material；
2. table/background appearance；
3. illumination；
4. camera 小扰动；
5. distractor。

对同一个物理动作，比较不同 domain 的：

\[
\cos(g_a^{domain1},g_a^{domain2}),
\]

以及 Newton action cosine 和 action ranking 一致率。layout 或目标位置变化必须建立新的阶段参考，不能复用旧场景 reference。

## 7. 实现计划

### 已完成

- eval/libero/probe_v5_mi_action_geometry.py
  - 使用真实 Dame B-spline MI；
  - 验证 gradient/Hessian pullback；
  - 对照中心有限差分；
  - 验证 gradient/Newton step 提高 MI。

### Test 1 待新增

1. mi_reward/control/mi_action_field.py
   - MI token gradient；
   - Jacobian pullback；
   - Hessian-vector product；
   - damping、trust region 和 action proposal。
2. eval/libero/collect_v5_local_perturbations.py
   - 恢复相同 simulator anchor；
   - 生成正负 epsilon 与 held-out 动作；
   - 排除 contact/discrete-event transition；
   - 保存 RGB、tokens、action、state 和 physical delta。
3. eval/libero/eval_v5_local_geometry.py
   - 估计 Jacobian/Hessian；
   - 比较 pullback 与 finite difference；
   - 计算局部预测和物理方向指标。
4. eval/libero/run_v5_local_geometry.sh
   - 固定配置、模型哈希和输出目录；
   - 不更新 reward/Qwen checkpoint。

## 8. Go/No-Go

~~~text
Test 0：数学恒等式和当前 MI autograd
    PASS
      |
Test 1：LIBERO smooth-stage Jacobian
      |
      +-- 导数不一致：修复 Jacobian/normalization/坐标
      |
MI gradient 与 privileged physical direction 一致？
      |
      +-- 否：MI objective/reference 不适合，不进入规划
      |
Hessian 是否优于 gradient-only？
      |
      +-- 否：保留一阶场，删除 Hessian
      |
再测试 appearance/instance/scene gradient invariance
~~~

Test 1 通过前，不修改 v4 的 No-Go 判断，不启动 Qwen、fusion 或 RL。v5 验证的是新的 action-space MI geometry 假设，不能用它重新解释旧实验。

## 9. 更新结论：修复阶段后的归因与 raw-RGB 对照（2026-09-08）

### 9.1 当前正式结论

修复 frame-0 reference/anchor 选择后，wrist full-frame LAM latent MI 的正式判断仍为
**NO-GO**。正确阶段与更严格的无接触扰动使结果明显改善，但改善不足以证明该 objective
在不同 anchor 上形成稳定的局部 action field：

| 指标 | frame-0 run | 修复阶段 run | v5 门槛 |
|---|---:|---:|---:|
| Held-out first-order MI R2 | -0.2490 | 0.0620 | >= 0.50 |
| Held-out ranking Spearman | -0.0755 | 0.3633 | >= 0.60 |
| Held-out sign accuracy | 0.5167 | 0.6354 | >= 0.80 |

阶段修复带来的提升确认旧协议是重要混杂因素，但不能解释全部失败。四个 anchor 中只有
`demo_2/frame_36` 的 R2/Spearman 达到 `0.6648/0.8330`，且 sign accuracy 仍只有
`0.75`；其他三个 anchor 的 R2 为负。因此，单个 anchor 的局部成功不能支持稳定的
full-frame latent MI gradient claim。

### 9.2 动作动力学不是主要问题

利用 primary-axis action 拟合 action-to-EEF 线性映射后，四个 anchor 的 held-out EEF
位移预测 R2 均接近 `1.0`，median direction cosine 也接近 `1.0`。这说明当前失败不应
主要归因于 LIBERO 控制器、state restore 或 action-to-EEF 局部动力学。

失败更可能出现在：

\[
\text{EEF motion}
\rightarrow
\text{rendered RGB}
\rightarrow
\text{LAM tokens}
\rightarrow
\text{latent MI}.
\]

### 9.3 当前扰动仍处于过小的视觉尺度

虽然修复后使用 primary/secondary command `0.012/0.006`，但执行 2 simulator steps 后，
相对于 center rollout 的实际 EEF 位移只有：

- primary-axis：约 `0.10--0.18 mm`；
- secondary-axis：约 `0.05--0.09 mm`；
- held-out median：约 `0.07--0.09 mm`。

因此，这次测试仍处于明显的亚毫米、亚像素区域。动作到 EEF 的映射在该区域是线性的，
但 renderer、图像量化和视觉 encoder 的局部导数可能受信噪比限制。当前结果不能被干净地
解释为“MI objective 本身失败”，因为实际视觉扰动尺度尚未完成 calibration。

在相同采集图像上进行 raw-pixel 检查时，wrist primary/secondary pixel-Jacobian 的各轴
cosine 约为 `0.72--0.79`，held-out pixel-delta 的 origin-based R2 约为
`0.31--0.35`。相较 frame-0 run 已明显改善，但仍未达到预注册的跨 epsilon 稳定性门槛。
agentview 在该尺度下依然不稳定。

### 9.4 Raw RGB MI 离线对照

为区分“MI 本身不适用”和“LAM latent 破坏局部几何”，在不重新运行 simulator 的情况下，
对同一批 wrist 图像计算了 raw-RGB channelwise DameSoftHistogramMI，并使用与 latent
evaluator 相同的 primary-axis pullback 和 held-out actions。结果如下：

| Objective | Pooled R2 | Pooled Spearman | Pooled sign accuracy |
|---|---:|---:|---:|
| LAM latent MI | 0.0620 | 0.3633 | 0.6354 |
| Raw RGB MI | **0.4151** | **0.6488** | **0.7188** |

Raw RGB MI 的逐 anchor 结果为：

| Anchor | R2 | Spearman | Sign accuracy |
|---|---:|---:|---:|
| `demo_1/frame_31` | 0.3014 | 0.6391 | 0.7083 |
| `demo_2/frame_36` | 0.4192 | 0.8148 | 0.7500 |
| `demo_4/frame_41` | **0.6887** | **0.7930** | **0.8750** |
| `demo_5/frame_36` | -0.1205 | 0.2713 | 0.5417 |

Raw RGB MI 尚未整体达到 v5 的 R2 和 sign gate，因此不能标为 PASS；但它在完全相同的状态、
动作和 reference 上显著优于 LAM latent MI。当前证据因此更支持以下归因：

1. 正确阶段下，raw RGB 中已经存在部分可预测的局部 MI 结构；
2. 当前 LAM token 表示或对 latent channel 计算 histogram MI 的方式进一步破坏了该结构；
3. full-frame objective 在不同 anchor 上仍不稳定；
4. 亚毫米视觉扰动继续限制 raw RGB 和 latent objective 的可判定性。

该 raw-RGB 结果是基于既有采集图像的离线诊断，不替代预注册后的正式独立实验。

### 9.5 对 Hessian 和 reward-model 主线的影响

当前仍不进入 Hessian。Hessian 只能描述一个已经具有稳定一阶导数的局部标量场，不能修复
跨 epsilon 不稳定的视觉 Jacobian，也不能让缺少任务语义的 MI objective 自动获得正确
物理方向。

这次结果不否定 observation-only reward-model 方向。它支持更严格的研究命题：

> Reward representation 必须同时保留视觉语义、局部动作可控性和物理任务方向；预训练
> latent similarity 或 raw MI 不能被默认视为满足这些条件。

MI 在后续 reward model 中只能作为待验证的 reference-grounding signal、ROI/token
选择依据或辅助 regularizer。核心监督应检查并校准 reward 对动作的局部变化是否与
privileged physical progress 一致。

### 9.6 下一轮唯一优先队列

1. 以实际 EEF displacement 而不是 policy command 定义 trust region，采集约
   `1/2/4 mm` 的多尺度 probes，同时保持 pre-contact 和 object-static 条件；
2. 将 raw RGB MI 注册为正式 evaluator 分支，并保留逐 anchor prediction/actual arrays；
3. 在进入 MI 前报告 primary/secondary 的 token-Jacobian cosine、relative norm error 和
   token held-out R2，定位 LAM encoder 是否首先破坏局部几何；
4. 在同一批 probes 上比较 raw RGB full-frame、raw RGB ROI、LAM full-frame 和 LAM ROI；
5. 只有某个 objective 同时通过跨 epsilon、一阶 held-out 和 privileged physical-direction
   gate 后，才恢复 Hessian/Newton 测试；
6. 若修复尺度和 ROI 后所有 MI variant 仍失败，而 physical oracle 通过，则将 MI 降级为
   辅助相似度，reward model 主线转为 privileged action-direction supervision。

本轮只有 4 个 initial-state anchors，低于第 3.2 节计划的至少 5 个 initial states 和
20 个 anchors。因此它足以维持当前 variant 的 NO-GO 和指导故障定位，但不能作为任何
正向、跨状态或泛化结论的最终证据。

### 9.7 实际 EEF 尺度校准与约 2 mm raw-RGB 复测（2026-09-09）

为避免将 controller command 错当作物理动作尺度，新增 per-anchor calibration，直接测量
command 到相对 center rollout 的实际 EEF displacement。四个 smooth anchors 在 command
`0.18/0.24/0.36/0.48` 的全部 96 个正负轴向 probes 上均保持 object-static；对应实际
EEF displacement median 分别为 `2.24/2.99/4.48/5.98 mm`。因此 LIBERO 在当前阶段可
提供 2--4 mm 的 smooth local trust region。

evaluator 同时改为在实际 EEF delta xyz 坐标下计算 Jacobian、direct finite difference 和
held-out prediction，并新增 `raw_rgb` representation。约 2 mm run 使用 primary command
`0.18`、secondary command `0.12`、2 steps；四个 anchors 的 primary-axis 实际 EEF 位移
为 `1.51--2.72 mm`，196 个 candidates 和 96 个 held-out actions 均未触发 object-motion、
contact、grasp 或 success exclusion。

| wrist raw RGB Dame MI 指标 | 结果 | v5 门槛 |
|---|---:|---:|
| pullback/direct FD gradient cosine median | **0.9993** | >= 0.90 |
| held-out first-order MI R2 | **0.9598** | >= 0.50 |
| held-out ranking Spearman | **0.9840** | >= 0.60 |
| held-out sign accuracy | **0.9792** | >= 0.80 |
| primary/secondary Jacobian cosine median | **0.8535** | >= 0.90 |
| primary/secondary Jacobian relative error median | 0.6565 | diagnostic |

四个 anchor 的 held-out R2 为 `0.9823/0.9437/0.9507/0.9569`。这首次证明：在正确阶段、
无物体扰动和毫米级真实 EEF displacement 下，**raw RGB Dame MI 可以稳定预测同一
action ball 中未参与导数估计的 MI 改变量**。此前 latent/full-frame failure 不能再归因于
“visual MI 不存在局部 action geometry”。

正式 gate 仍为 **确认中，而非 PASS**：primary `0.18` 与 secondary `0.12` 产生的实际
位移范围并不相同，cross-epsilon Jacobian cosine `0.8535` 未满足预注册 `0.90`。下一轮
固定 primary `0.18`，将 secondary 改为更近的 `0.15`，只验证四个 smooth anchors 的
cross-epsilon stability；不能因优秀 held-out 指标而跳过该确认。只有该项通过，raw RGB
branch 才进入 privileged EEF-object physical-direction test；Hessian 继续冻结。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_eef_scale_v1/results.json
logs/mi_reward/v5_mi_action_geometry/far_approach_eef_scale_v2/results.json
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_heldout_v1/collection.json
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_rawrgb_eef_eval_v1/results.json
~~~

### 9.8 近尺度确认通过：raw-RGB Jacobian 稳定性（2026-09-09）

为将 9.7 中 `0.18` / `0.12` 的尺度差异与局部几何失效区分，保持相同 reference、四个
smooth anchors、两步 rollout、object-static exclusion 和实际 EEF delta action coordinate，
只将 secondary command 收紧为 `0.15`。本轮不重采 held-out action；它是单独、预注册的
cross-epsilon confirmation，而 held-out gate 仍由 9.7 中同一 primary `0.18` 的 96 个独立
action 提供。

采集得到 4/4 anchors、100 个 primary/secondary candidates、0 exclusions。wrist raw-RGB
Dame MI 的每-anchor Jacobian cosine 为 `0.9445/0.9354/0.9471/0.9395`，中位数为 **0.9420**；
relative error 中位数为 **0.3754**。这满足 v5 的 cross-epsilon cosine `>= 0.90` gate，且比
`0.18` / `0.12` 的 `0.8535` 明显稳定。导数 pullback/direct-FD cosine 中位数保持 **0.9993**。

结合 9.7 的独立 held-out 指标（R2 `0.9598`、Spearman `0.9840`、sign `0.9792`），raw-RGB
branch 现已通过 v5 的一阶局部几何证据：视觉 MI 标量差分可经实际 EEF Jacobian pullback
形成一致的局部 action gradient，并能预测独立 action 的 MI 改变量。evaluator 本次显示
`INCOMPLETE` 是预期的 bookkeeping 状态，因为本轮 `heldout_actions=0`；它不是 cross-epsilon
failure，也不推翻前一轮冻结 held-out evidence。

因此下一步进入 **privileged physical-direction gate**：在相同 smooth anchors 下，用 EEF 到
任务物体的物理距离变化定义 oracle direction，检验 raw-RGB MI pullback direction 是否与真实
approach direction 对齐。该 physical gate 未通过前仍不进入 Hessian、Newton update 或 reward
training integration。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_closeepsilon_stability_v1/collection.json
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_closeepsilon_rawrgb_eval_v1/results.json
~~~

### 9.9 privileged physical-direction gate：四个已验证 anchor 通过（2026-09-09）

在 9.8 的同一批 primary axis probes 上，新增 evaluator 直接从候选 rollout 的 privileged
`eef_object_distance` 及实际 EEF displacement 拟合局部 distance gradient，并取其负方向作为
physical descent oracle；没有使用 MI、RGB 或 reward score 构造该 oracle。随后比较 raw-RGB
MI pullback gradient 的 ascent direction 与这个 oracle。

四个 anchor 的 cosine 为 `0.8091/0.8286/0.7685/0.7064`，median 为 **0.7888**；四个 anchor
均满足 MI ascent 使 EEF-object distance 局部下降，比例为 **1.00**。因此在已验证的四个
smooth pre-contact states 上，同时达到：direction cosine median `>= 0.70` 和 descent-aligned
anchor fraction `>= 0.70`。这给出 v5 目前最强的正向证据：raw-RGB Dame MI 不是仅在自身
标量值上自洽，它的 pulled-back local direction 也与 simulator 的 approach 物理进度一致。

这一结果仍是 **provisional PASS，而非完整 Test-1 PASS**。预注册覆盖要求是至少 5 个 initial
states、20 个 anchors，以及至少 4/5 states 的方向一致性；目前只有 4 个 states、4 个 anchors。
此外，9.8 的 close-epsilon run 按设计未重采 held-out actions，故 evaluator 的本次
`INCOMPLETE` 只表示该单一目录不能独立同时含 held-out gate，并不否定 9.7 的冻结 held-out
evidence。下一轮应在新增独立第五 state 上，以相同 reference、2 mm primary / close secondary
probe 和 24 held-out actions 一次性复现三项一阶证据；在覆盖条件满足前，不进入 Hessian 或
planner/Newton。

本轮修改 `eval/libero/eval_v5_local_geometry.py`，使结果 JSON 额外保存每个 anchor 的
`physical_descent_direction`、cosine 和 distance-descent flag，并保存汇总 physical gate；这不
改变 teacher、MI formulation、reward architecture 或训练主线。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_closeepsilon_rawrgb_physical_eval_v1/results.json
~~~

### 9.10 20 个有效 anchor 正式覆盖实验：运行中状态（2026-09-09）

为满足 Test-1 的覆盖要求，启动正式 collection
`far_approach_2mm_20valid_full_v2`。它固定 `demo_0/frame_34` 为独立 reference，并从
`demo_1`--`demo_25` 各取一个 pre-contact anchor，留出冗余以抵消 object-motion exclusion。
每个 anchor 在同一 manifest 中收集 primary `0.18`、close secondary `0.15`、24 个
radius `0.18` held-out actions、wrist raw-RGB MI 所需帧、实际 EEF displacement 和
privileged EEF-object distance；每个请求 anchor 有 49 个 candidates。

截至本次记录，作业仍在 `tmux train:v5full` 中运行，已完成 `demo_1`--`demo_5` 的 5 个
anchor。`demo_1`、`demo_2`、`demo_4`、`demo_5` 均为 `49/49` 未排除；
`demo_3/frame_38` 有 `48/49` candidates 因 object-motion rule 排除，因而将由 evaluator
标记为无效且不计入正式分母。collector 仍保持活跃计算，正在对后续 demo 扫描相同的
smooth-stage 条件。

本节是进度快照，**不报告 aggregate MI、Jacobian、held-out 或 physical-direction 指标**。
这些指标只有在 collection 写出 `collection.json` 后，才会在同一冻结 evaluator 中一次性
计算并成为正式证据。

旧 `far_approach_2mm_rawrgb_eef_eval_v1/results.json` 的 scope metadata 同时已由错误的
`Frozen LAM tokens` 更正为 `Raw RGB pixels`；representation 和全部数值保持不变。

运行目录：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_20valid_full_v2/
~~~

### 9.11 多 initial-state 完整 probe 结果：raw-RGB physical gate NO-GO（2026-09-09）

`far_approach_2mm_20valid_full_v2` 已完成。采集从 25 个独立 demo 请求 anchor，实际选出 24 个
pre-contact anchor、1,176 个 candidates；432 个 candidates 因预注册 object-motion exclusion
被排除。按“center、六个 primary axis 与六个 secondary axis 全部有效”的 evaluator protocol，
最终有 **14 个 valid anchors**、10 个 `INVALID_PROTOCOL` anchors，及 336 个有效 held-out
actions。因此该 run 没有达到至少 20 个有效 anchors 的覆盖要求。

在这 14 个有效 independent states 上，wrist raw-RGB Dame MI 的局部数学/几何项仍然强：

| 指标 | 结果 | v5 门槛 | gate |
|---|---:|---:|---|
| pullback/direct-FD gradient cosine median | **0.99945** | >= 0.90 | pass |
| primary `0.18` / secondary `0.15` Jacobian cosine median | **0.94580** | >= 0.90 | pass |
| held-out MI R2（336 actions） | **0.95757** | >= 0.50 | pass |
| held-out Spearman | **0.96569** | >= 0.60 | pass |
| held-out sign accuracy | **0.91667** | >= 0.80 | pass |
| EEF-object descent cosine median | **0.61108** | >= 0.70 | **fail** |
| MI ascent distance-descent fraction | **0.92857** | >= 0.70 | pass |

所以此前四个 anchor 的正向 physical result 不能推广为跨初始状态结论。raw-RGB MI 在这些 smooth
states 中确实形成了高度自洽、可预测、跨 close epsilon 稳定的一阶 action field；但该 field
的幅度/方向与真实 EEF-to-object approach direction 的总体 cosine 尚不足。正式 evaluator
已将 physical gate 纳入 overall status，结果为 **NO_GO**；此前忽略该 gate 而输出 `PASS` 的
status bug 已修复，原始统计数值不变。

这次 NO-GO 的含义是：不能将 raw-RGB MI pullback 作为跨 state 的 autonomous approach controller，
也不能进入 Hessian、Newton/planner、reward integration 或 RL。后续优先工作不应盲目继续扩大
同一 full-frame objective 的 anchor 数；应先诊断低-cosine states 的 visual/physical mismatch，
再预注册 ROI 或 reference-conditioned physical alignment variant。若该诊断没有产生通过
physical gate 的 objective，v5 应将 MI 降级为局部 consistency/reference signal，而不是动作
direction supervision。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_20valid_full_v2/collection.json
logs/mi_reward/v5_mi_action_geometry/far_approach_2mm_20valid_full_rawrgb_eval_v2/results.json
~~~

### 9.12 冻结 raw-RGB branch；五 state × 四 anchor 一阶正式复验（2026-09-09）

9.11 表明不能把跨 14 个 sparse anchors 的 physical-direction failure 解释为 Hessian 问题，
也不能继续改动 MI 公式来追逐该结果。因此从本节开始冻结 raw-RGB branch：Dame MI 为
`num_bins=8, spline_order=3, normalization="none", channelwise`，reference/center external
fixed normalization、实际 EEF delta xyz action coordinate、primary `0.18`、secondary `0.15`、
两 simulator steps、24 个 held-out actions（radius `0.18`）和当前 physical-distance evaluator
均不得变更。

冻结文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `eval/libero/collect_v5_local_perturbations.py` | `4cbefb65d21bea316decfb264d8c17642a79d6fe0d18f016b0cebeb43764b31b` |
| `eval/libero/eval_v5_local_geometry.py` | `ac4a38642518b62cc1a6733a8fd409c29051c20690f52fbd321473e4ff635761` |
| `mi_reward/scoring/dame_soft_histogram.py` | `ecfb8e8c7a463e757a8a04160e4976e5e5b0ae6e30afb37f79b3ffbcb931dd68` |
| `mi_reward/control/mi_action_field.py` | `4cfbbd5aff26cb6055a2b47bbba5df88269866328432642a305b956d5589656c` |

下一次正式 collection 固定五个独立 initial states：`demo_1`、`demo_2`、`demo_4`、`demo_5`、
`demo_10`。这些 state 已在单-anchor probe 中满足 smooth pre-contact、完整 primary/secondary
probe 且 physical direction 为正。每个 state 选择四个 anchor，总目标为 20 个 anchors；所有
一阶指标和 physical gate 必须在**同一 collection/evaluator**中报告。Hessian 在这次完整
覆盖的 raw-RGB branch 通过前继续冻结。

### 9.13 冻结五 state × 四 anchor 正式一阶结果（2026-09-09）

冻结 collection 已完成并达到覆盖要求：五个独立 initial states（`demo_1,2,4,5,10`）、每个
四个 smooth anchor，共 **20/20 valid anchors**、980 candidates、**0 exclusions**。每个 anchor
具有 primary `0.18`、secondary `0.15`、24 个 held-out actions、raw-RGB MI 输入、实际 EEF
action coordinate 和 privileged EEF-object distance。统一 evaluator 在 480 个 held-out actions
上给出：

| 指标 | 结果 | 门槛 | gate |
|---|---:|---:|---|
| pullback/direct-FD gradient cosine median | **0.99979** | >= 0.90 | pass |
| cross-epsilon Jacobian cosine median | **0.94752** | >= 0.90 | pass |
| held-out MI R2 | **0.96932** | >= 0.50 | pass |
| held-out MI Spearman | **0.98578** | >= 0.60 | pass |
| held-out MI sign accuracy | **0.94375** | >= 0.80 | pass |
| physical EEF-object descent cosine median | **0.68842** | >= 0.70 | **fail** |
| MI-ascent distance-descent fraction | **0.95000** | >= 0.70 | pass |

正式 status 为 **NO_GO**。这不是采集覆盖、Jacobian、MI 一阶预测或物体运动污染问题：完整
20-anchor collection 没有 exclusions，且前三类 gate 均高于门槛。失败的唯一已注册 gate 是
跨 anchor 的 physical direction cosine，距离门槛差 `0.01158`。不能因其接近阈值而将它视为
PASS，也不能开始 Hessian。

该结果细化了当前归因：frozen full-frame raw-RGB Dame MI 是稳定并可预测自身局部增量的
action field，且大多数 anchor 的 ascent 符号会降低 EEF-object distance；但其连续 direction
尚不足以作为跨 state 的物理 approach vector field。下一步应诊断低-cosine anchor 的视觉
background/reference mismatch，形成独立预注册的 ROI 或 reference-conditioned variant；不对
已经冻结的 raw-RGB branch 调整 bin、normalization、epsilon 或 evaluator gate。

正式输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_full_v1/collection.json
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_full_rawrgb_eval_v1/results.json
~~~

### 9.14 预注册视觉区域诊断（冻结 full-frame 后）

9.13 后，full-frame raw-RGB branch、其 Dame MI 参数、normalization、`0.18/0.15` epsilon、
实际 EEF action coordinate、held-out protocol 和 evaluator 均冻结。不能再修改这些项目以
改善 full-frame 结果。下一轮只在 9.13 的**同一 20 anchors / 980 candidates**上改变输入的
视觉区域，并保持同一 reference、MI estimator、finite-difference Jacobian 和 physical oracle。

由于 archived RGB 未附带 object segmentation，mask 将从同一 anchor state 与同一已保存 action
通过 LIBERO `SegmentationRenderEnv` replay 获得。每个 replay RGB 必须与 collection 中冻结 PNG
按逐像素 MAE 核验；不匹配样本将标为 `mask_replay_mismatch`，不能进入任一 ROI variant 分母。
task-object instance ID 从 LIBERO goal-state object name 映射，不使用手工框或语言模型预测。

四个预注册 variants：

| variant | 固定输入定义 |
|---|---|
| `full_frame` | 原始 `128×128` RGB，作为 9.13 的重复性对照 |
| `object_roi` | task-object instance mask 内 RGB；按 raster order 固定重采样 256 pixels，掩码不足时循环采样 |
| `object_centered_crop` | 以 task-object mask centroid 为中心的 `64×64` RGB crop，边界 reflect padding，不做额外 resize |
| `background_only` | task-object mask 的 7-pixel dilation 以外 RGB；按 raster order 固定重采样 4,096 pixels |

每个 variant 同时报告 480-action held-out R2/Spearman/sign、primary-secondary Jacobian cosine、
MI pullback 与 privileged EEF-object descent cosine，以及 distance-descent fraction。ROI 或
object-centered crop 只有在完整 20-anchor coverage 下 cosine median **稳定 `>=0.70`** 且保留
一阶 held-out/cross-epsilon gates 时，才成为后续 reference-conditioned MI 候选。所有 variants
仍失败时，MI 降级为 reference-consistency auxiliary signal；主 reward supervision 改为
privileged physical progress 或 action-direction distillation。Hessian 继续冻结。

### 9.15 ROI 输入核验完成：instance mask replay（2026-09-09）

object-mask replay 已完成。对 9.13 的相同五个 states、20 个 anchors、980 candidates，使用
LIBERO `SegmentationRenderEnv` 从每个保存 anchor state 和原始 action command 重放两步；task
object 为 `akita_black_bowl_1`。所有 **980/980** candidate RGB 与冻结 collection PNG 的逐像素
MAE 均为 `0.0`，独立 `demo_0/frame_34` reference 的 RGB MAE 也为 `0.0`；没有空 object mask。
因此 object ROI、object-centered crop 与 background-only 的后续输入能以 simulator-grounded
instance mask 定义，而非手工框、语言模型或未验证分割器。

这项 replay 只验证视觉区域输入的可复现性，不计算或改变 MI、Jacobian、held-out 或 physical
direction 指标，因而不改变 9.13 的 full-frame `NO_GO`。下一步仍是按照 9.14 固定定义，在同一
20-anchor data 上运行四个 variants 并报告所有 gates。

可复查输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_object_masks_v2/masks.json
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_object_masks_v2/reference_mask.png
~~~

### 9.16 视觉区域诊断结果：全部 ROI variants 未通过（2026-09-09）

在 9.13 的同一 5 states × 4 anchors、480 held-out actions 上，完成 9.14 预注册的四分支比较。
full-frame 使用冻结结果；其余输入由 9.15 中 RGB MAE 为 0 的 simulator instance masks 生成。
MI、normalization、epsilon、实际 EEF action coordinate 和 evaluator gate 均未调整。

| variant | grad cosine | cross-eps cosine | held-out R2 | Spearman | sign | physical cosine | descent fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-frame | **0.9998** | **0.9475** | **0.9693** | **0.9858** | **0.9438** | 0.6884 | **0.95** |
| object ROI | 0.7767 | 0.7617 | -0.2549 | 0.0051 | 0.5500 | -0.4744 | 0.35 |
| object-centered crop | **0.9985** | 0.8946 | **0.6879** | **0.8149** | **0.8083** | 0.1482 | **0.75** |
| background-only | **0.9813** | 0.8244 | -0.3205 | 0.2293 | 0.5708 | **0.8183** | 0.60 |

没有 variant 同时通过 held-out、cross-epsilon 和 physical-direction gates。object ROI 丢失局部
可预测性且方向反向；object-centered crop 的 held-out 指标尚可，但 cross-epsilon `0.8946 < 0.90`
且 physical cosine 仅 `0.1482`；background-only 虽有较高 physical cosine，但 descent fraction、
cross-epsilon 和全部 held-out gates 失败，不能视为可靠动作场。full-frame 仍是四者中最稳定的
MI consistency signal，但 physical cosine `0.6884 < 0.70`，正式结论保持 `NO_GO`。

因此依照 9.14 的预注册决策，**MI 降级为 reference-consistency auxiliary signal**。reward model
的主监督应转为 privileged physical progress 或 action-direction distillation。Hessian、Newton
proposal 和 MI-driven planner 不启动，因为一阶物理方向尚未跨状态通过。

正式输出：

~~~text
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_full_rawrgb_eval_v1/results.json
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_object_roi_eval_v1/results.json
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_object_centered_crop_eval_v1/results.json
logs/mi_reward/v5_mi_action_geometry/five_state_four_anchor_background_only_eval_v1/results.json
~~~
