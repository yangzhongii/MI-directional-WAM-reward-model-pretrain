# Pipeline v5 最新实验诊断

## 1. 总体结论

当前 `wrist full-frame latent MI` 得到 `NO_GO` 是合理的，但该结果不能用于否定 MI action-space geometry 的整体方向。

现有证据表明：

1. 动作到 EEF 位移的局部映射非常稳定；
2. \(g_a=J_a^\top g_z\) 的链式求导和代码实现基本正确；
3. 失败发生在 EEF 运动之后的视觉观测、latent 表示和 MI objective 这一段；
4. 本轮采集实际使用了 episode 第 0 帧，而不是文档要求的 pre-grasp aligned 阶段，因此当前结果还混入了明确的实验协议错误；
5. 在一阶视觉场稳定之前，不应继续计算或使用 Hessian。

更准确的结论是：

> 当前实验否定了“第 0 帧 reference + wrist full-frame LAM latent MI + 当前亚毫米扰动尺度”这一具体配置，但尚未否定 MI 梯度经 Jacobian 拉回动作空间的研究假设。

## 2. 已确认的实验事实

### 2.1 数学恒等式成立

Pipeline v5 的 Test 0 已经验证：在线性 latent dynamics 下，

\[
g_a=J_a^\top g_z,
\qquad
H_a=J_a^\top H_zJ_a
\]

与 autograd 和有限差分一致。因此，当前没有证据表明 Jacobian/Hessian pullback 公式本身写错。

### 2.2 动作到 EEF 的局部动力学正常

使用 primary-axis 动作拟合 action-to-EEF 线性映射，再预测 held-out 动作的 EEF 位移，五个 anchor 的 \(R^2\) 约为：

\[
0.999994,
0.999993,
0.998519,
0.999802,
0.999994.
\]

held-out 预测方向的 median cosine 接近 1。因此，LIBERO 控制器、state restore 和 action-to-EEF Jacobian 不是当前失败的主要来源。

### 2.3 MI 的轴向导数实现基本正确

实测结果为：

| 指标 | 结果 |
|---|---:|
| Pullback/direct finite-difference gradient cosine median | 0.9981 |
| 各 anchor gradient cosine | 0.9906--0.9984 |
| Pullback/FD gradient relative L2 error | 约 5.8%--16.8% |

这说明 token Jacobian、MI token gradient 和 \(J_a^\top g_z\) 的张量链路没有明显实现错误。

但是，pullback 和 direct finite difference 使用了同一组 primary-axis `plus/minus` 图像。这项检查能够排除主要代数错误，却不能证明该 secant derivative 是 \(a=0\) 附近稳定、可泛化的一阶导数。

### 2.4 一阶模型无法预测 held-out MI 变化

| 指标 | 预注册门槛 | 实测 |
|---|---:|---:|
| Held-out first-order MI \(R^2\) | ≥ 0.50 | -0.2490 |
| Held-out ranking Spearman | ≥ 0.60 | -0.0755 |
| Held-out sign accuracy | ≥ 0.80 | 0.5167 |

五个 anchor 的独立 \(R^2\) 分别为：

\[
-1.597,
-1.663,
-1.556,
-1.351,
0.011.
\]

符号准确率接近随机。这说明当前梯度只能拟合用于估计它的轴向端点，不能预测同一 action ball 内的新动作。

## 3. 已确认的协议问题：reference 和 anchor 选成了第 0 帧

采集结果实际使用：

- reference：`demo_0/frame_0`；
- anchors：`demo_1--demo_5/frame_0`；
- reference 的 EEF-object distance：约 0.383 m；
- anchors 的 EEF-object distance：约 0.355--0.377 m。

这些状态显然还处于 episode 初始区域，并不是 pre-grasp aligned 状态。

原因是 `_select_anchor_frames` 从时间序列开头开始检查；当前 `_is_smooth_anchor` 只要求状态没有 grasp、contact 或 success。第 0 帧通常立即满足条件，所以函数会直接选中它。`ref_candidates[-1]` 也无法解决这个问题，因为 `per_demo=1` 时只收集了第一个合法状态。

这一错误带来两个后果：

1. 当前 reference 不是阶段目标，MI 即使形成方向，也可能指向 demo 0 的初始相机姿态；
2. 当前实验不能用于判断 MI gradient 是否与 pre-grasp 物理方向一致。

这个问题主要影响 MI 的任务语义和 physical-direction claim。它本身不能完全解释 held-out MI 的局部预测失败，因为固定任意 reference 后，一个足够光滑的复合目标仍应具有局部可预测性。

## 4. 当前最可能的直接失败源：视觉有限差分不稳定

本轮 `epsilon=0.025` 执行 4 steps 后，primary-axis 动作只造成约 0.8--1.1 mm 的 EEF 相对位移；secondary epsilon 只造成约 0.5--0.6 mm 位移。

在 128×128 RGB 图像上，这接近亚像素运动。wrist camera 随 EEF 一起移动，因此细小相机运动会引起全画面的光栅重采样、边缘遮挡和 uint8 量化变化。

直接对采集图像进行检查后得到：

- primary 与 secondary epsilon 下，同一轴的 raw-pixel Jacobian cosine 大多位于 \(-0.4\) 到 \(0.36\)；
- 使用 primary-axis pixel Jacobian 预测 held-out 图像变化时，五个 wrist anchor 的 origin-based \(R^2\) 均为负；
- held-out pixel-delta prediction 的 median cosine 大约为 \(-0.13\) 到 \(0.11\)；
- agentview 也出现类似现象。

与此同时，action-to-EEF 的 held-out \(R^2\) 接近 1。因此失稳发生在

\[
\text{EEF pose}\rightarrow\text{rendered RGB}\rightarrow\text{visual tokens}
\]

这一段，而不是机器人局部动力学。

当前还没有直接报告 primary/secondary epsilon 下的 LAM-token Jacobian cosine，因此不能进一步把责任严格区分为 renderer 非光滑、视觉编码器非线性，或者两者共同作用。

## 5. Latent MI objective 的额外风险

当前方法对 LAM/DINO patch token 使用 channelwise soft-histogram MI：每个 latent channel 单独建立 joint histogram，再对通道平均。

这在数学上是可微的 MI-like objective，但它与 TRO 中建立在图像强度和像素配准关系上的 MI 不完全等价，存在以下风险：

1. latent channel 没有灰度强度那样明确、稳定的统计含义；
2. patch token 对亚像素输入变化可能高度非线性；
3. channelwise averaging 可能淹没少量与机器人或物体运动有关的通道；
4. full-frame wrist 表示包含大量背景与相机自运动，任务相关局部变化可能只占很小比例；
5. reference 与 candidate 来自不同 demonstration，MI 可能主要反映外观或相机姿态，而不是 EEF-object 几何进度。

不过，当前实验还不能单独证明“latent MI 不可用”，因为 reference 选择和扰动尺度尚未修复。

## 6. 为什么现在不应加入 Hessian

Hessian 能描述一个平滑标量场的局部曲率，但不能修复不稳定的一阶视觉导数。

目前已经观察到：

\[
J_{RGB}(\epsilon_1)
\not\approx
J_{RGB}(\epsilon_2),
\]

并且 first-order held-out prediction 失败。在此条件下，二阶有限差分通常会进一步放大光栅化、encoder 和 MI 估计噪声。即使算出一个数值 Hessian，也很难把它解释为可用于规划的局部曲率。

因此正确顺序应为：

1. 修复阶段和 reference；
2. 找到稳定的一阶视觉 trust region；
3. 验证 held-out MI 一阶预测；
4. 验证 MI gradient 与 privileged physical direction 一致；
5. 最后再比较 Hessian/Newton 与 gradient-only。

## 7. 下一轮最小诊断实验

### 7.1 修复阶段选择

不要再用“未接触”作为唯一 pre-grasp 条件。至少加入：

- EEF-object distance 范围；
- demonstration progress 范围；
- gripper/object 在 wrist 或 agentview 中的可见性；
- reference 与 anchor 属于相同操作阶段；
- 采集前打印并保存 frame index、EEF-object vector 和距离。

reference 应显式选择一个接近 pre-grasp aligned 的 demo 0 帧，而不是自动选择第一个 smooth frame。

### 7.2 用实际位移定义扰动尺度

不要只用 policy command 的 `0.015/0.025` 表示局部半径。应根据执行后的实际 EEF displacement 建立约 2、4、8 mm 的多尺度 probes，并确保不跨越 contact/grasp 边界。

### 7.3 分层定位第一次失稳的位置

在完全相同的 anchor/action 上依次检查：

\[
J_{EEF},\quad
J_{RGB},\quad
J_{token},\quad
\nabla_a MI.
\]

每一层都报告：

- 两套 epsilon 的 Jacobian/gradient cosine；
- relative norm error；
- held-out first-order \(R^2\)；
- sign accuracy；
- action ranking Spearman。

哪一级首先失败，主要问题就位于哪一级。

### 7.4 比较四个最小 objective

在同一批状态和动作上比较：

1. raw RGB full-frame MI；
2. raw RGB object/gripper ROI MI；
3. LAM latent full-frame MI；
4. LAM latent ROI MI。

如果 raw RGB 通过而 latent 失败，问题主要来自 encoder/latent MI；如果 ROI 通过而 full-frame 失败，问题主要来自背景和 wrist camera 自运动；如果全部失败，应重新考虑基于 histogram MI 的当前局部 objective 或改变视觉扰动尺度。

### 7.5 加入物理 oracle

在一阶 MI held-out prediction 通过后，比较：

\[
\cos\left(g_a^{MI},g_a^{distance}\right)
\]

以及沿 MI gradient 执行动作后 EEF-object distance 是否减小。局部 MI 可预测只表示它形成了一个稳定视觉场，并不自动表示该场指向任务进度。

## 8. 当前归因排序

| 优先级 | 问题 | 当前证据 |
|---:|---|---|
| 1 | Reference/anchor 阶段选错 | 已确认，全部为 frame 0 |
| 2 | 当前尺度下视觉有限差分不稳定 | 强证据，跨 epsilon pixel Jacobian 不一致且 held-out pixel prediction 失败 |
| 3 | LAM token 映射在该尺度不够局部线性 | 高度可疑，尚缺 token 跨 epsilon 直接结果 |
| 4 | Full-frame latent MI 没有形成有用任务势能 | 可能，但受前两项混杂 |
| 5 | Jacobian/Hessian 数学公式错误 | 当前基本排除 |

## 9. 对研究方向的最终判断

现在不应把结果解释为“MI 从一开始就是错的”。能够成立的判断是：

> MI 是一个标量 objective；通过 Jacobian 将其梯度拉回动作空间在数学上成立。但一个正确的 pullback 只能传递已有视觉场的方向，不能让错误 reference、不稳定视觉差分或缺少任务语义的 MI 场自动产生正确控制方向。

因此，v5 的核心数学连接可以保留。下一阶段需要验证的是“能否构造稳定且任务相关的视觉 MI field”，而不是继续增加 Hessian 复杂度。如果修复 reference、尺度和 ROI 后 raw RGB 与 latent MI 都无法通过一阶 held-out 与 physical-direction gate，再考虑将 MI 降级为辅助相似度，而不再作为主要动作方向来源。
