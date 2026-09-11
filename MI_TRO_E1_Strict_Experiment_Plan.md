# E1：严格复现 Dame–Marchand TRO 的 6-DOF MI 视觉伺服

## 0. 实验定位

**Mode：Full experiment design**

E1 只回答三个逐级问题：

1. Dame–Marchand TRO 的 image-space MI、解析梯度、完整 Hessian 和 6-DOF 图像 interaction matrix 是否被正确实现？
2. 在论文成立的前提下，即 eye-in-hand 相机观察静态场景，MI 控制律能否恢复 6-DOF 相机目标位姿？
3. 在同一场景、同一参考图像和无接触短程运动中，MI 是否能够对候选末端相机位姿进行有物理意义的排序？

E1 **不直接验证**完整机械臂任务进度、夹爪开合、接触状态、物体搬运、固定第三视角或 Cosmos 生成轨迹。那些变量超出了 TRO 的成像模型。只有 E1 通过，才能继续做 E2 的 reward-ranking 迁移。

主要参考：[Mutual_Information-Based_Visual_Servoing.pdf](Mutual_Information-Based_Visual_Servoing.pdf)。

---

## 1. 为什么必须把 E1 与已有 v5–v7 实验分开

TRO 的状态变量是相机位姿 \(\mathbf r\)，输入是当前 eye-in-hand 图像 \(I(\mathbf r)\) 和目标相机位姿处的参考图像 \(I^*\)。它假设图像变化主要来自**相机运动**：

\[
I(\mathbf r)\longrightarrow \nabla I\longrightarrow \mathbf L_x
\longrightarrow \mathbf L_{MI}\longrightarrow \mathbf v_c.
\]

已有 v5–v7 则把 MI 用在 latent、几何 token 或动作扰动上，并用 physical progress 检验方向。这不是 TRO 的原始变量和雅可比。G1 的 `NO_GO` 否定了“geometric-token MI 可直接作为 physical-progress direction”，但没有复现下面这条 TRO 链路：

\[
\text{grayscale pixels}\rightarrow
\text{soft joint histogram}\rightarrow
\text{analytic image Jacobian}\rightarrow
\text{camera twist}.
\]

因此，E1 是一次独立的 mathematical reproduction。它要把“公式实现错误”和“MI 不适合任务进度”彻底分开。

---

## 2. 严格实验边界

### 2.1 主实验必须满足

- 相机：`robot0_eye_in_hand` / wrist camera；
- 控制量：相机坐标系下 6-DOF twist
  \(\mathbf v_c=[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]^\top\)；
- 场景：静态、有纹理、目标在当前视野中；
- 参考图像：在目标相机位姿 \(\mathbf T^*_{wc}\) 处真实渲染；
- 当前图像：从同一静态 simulator state、不同相机位姿真实渲染；
- 夹爪：全程固定，不评分；
- 物体：全程固定，无抓取、推动和接触；
- 图像：灰度图，主设置 320×240；若现有环境先用 128×128，则必须补做分辨率消融；
- 不使用 LaWAM、DINO、Qwen、Cosmos 或任何视觉 latent。

### 2.2 暂不允许的设置

- 固定 `agentview` 相机直接套用 \(\mathbf L_x\)；
- 把末端或物体 token 当成论文中的像素样本；
- 当前图像是接触前、参考图像是物体已经移动后的成功帧；
- 用 raw MI 跨场景比较候选；
- 用 simulator pose distance 参与 MI 计算。

固定第三视角下，相机没有运动，TRO 的 \(\mathbf L_x\) 不能直接描述机械臂像素的变化。如果以后使用第三视角，需要重新建立 articulated robot/object image Jacobian，属于 E2/E3，而不是 E1。

---

## 3. 符号、坐标系与位姿更新

- \(\mathbf T_{wc}\)：camera-to-world 变换；
- \(\mathbf T_{ce}\)：end-effector 到 camera 的固定手眼外参；
- \(\mathbf r\in SE(3)\)：当前相机位姿；
- \(\mathbf r^*\)：参考图像对应的目标位姿；
- \(\mathbf x=(x,y)\)：归一化/度量图像平面坐标；
- \(Z(\mathbf x)\)：像素对应深度；
- \(N_x\)：参与直方图计算的像素数；
- \(N_c=8\)：论文设置的灰度 histogram resolution；
- \(\phi=B_3\)：三阶 B-spline；
- \(\lambda>0\)：控制增益。

采用 `T_wc` camera-to-world 右乘 body increment 时，必须区分论文 interaction matrix 所使用的 camera velocity 与 pose increment。E1-A1 的 six-axis audit 已确认二者整体反号：

\[
\mathbf T_{wc}^{k+1}
=\mathbf T_{wc}^{k}\operatorname{Exp}(-\widehat{\mathbf v_c}\Delta t),
\]

其中 \(\mathbf v_c\) 是论文 \(\mathbf L_x\) 对应的 camera velocity。若 evaluator 直接以 \(\delta\boldsymbol\xi\) 参数化 `T_wc Exp(+delta xi)`，则 image pose Jacobian 使用 \(-\nabla I\mathbf L_x\)。不得同时在 pose update 和 image Jacobian 两处取负。

必须用一个单轴有限差分测试锁定 simulator 的正负号。若环境以 EEF twist 或关节速度执行，则使用固定手眼外参和机器人 Jacobian：

\[
\mathbf v_c={}^{c}\mathbf V_e\mathbf v_e,
\qquad
\dot{\mathbf q}=\mathbf J_e^\dagger\mathbf v_e.
\]

E1 评价的是相机 6-DOF 控制律，不把 7 个关节分别构造成 MI 势能。

---

## 4. 严格 TRO 图像 MI

### 4.1 预处理

1. RGB 转灰度；
2. 当前图和参考图均使用相同的 5×5 Gaussian filter；
3. 原始强度 \([0,255]\) 按论文式 (12) 缩放：

\[
\bar I(\mathbf r,\mathbf x)
=I(\mathbf r,\mathbf x)\frac{N_c}{255}.
\]

E1 禁止对每一对候选图像重新做 min-max、quantile 或 z-score 归一化，因为它会改变跨位姿的目标函数。

### 4.2 B-spline soft histogram

边缘概率：

\[
p_I(i,\mathbf r)=\frac{1}{N_x}\sum_{\mathbf x}
\phi\!\left(i-\bar I(\mathbf r,\mathbf x)\right),
\]

\[
p_{I^*}(j)=\frac{1}{N_x}\sum_{\mathbf x}
\phi\!\left(j-\bar I^*(\mathbf x)\right).
\]

联合概率：

\[
p_{II^*}(i,j,\mathbf r)=\frac{1}{N_x}\sum_{\mathbf x}
\phi\!\left(i-\bar I(\mathbf r,\mathbf x)\right)
\phi\!\left(j-\bar I^*(\mathbf x)\right).
\]

MI：

\[
MI(\mathbf r)=\sum_{i,j}p_{II^*}(i,j,\mathbf r)
\log\frac{p_{II^*}(i,j,\mathbf r)}{p_I(i,\mathbf r)p_{I^*}(j)}.
\]

实现要求：

- 按相同像素位置 \(\mathbf x\) 构造联合直方图；
- 主结果使用 \(B_3\)，并显式实现 \(\phi'\) 和 \(\phi''\)；
- 用 `float64` 完成 derivative audit；
- 对零概率 bin 使用 active-support mask，不能让任意大的 `eps` 改变曲率；
- 保存 \(\sum_i p_I\)、\(\sum_jp_{I^*}\) 和 \(\sum_{ij}p_{II^*}\) 供审计；
- 论文同时使用“\(N_c=8\) 个 bins”和求和下标 \(0,\ldots,N_c\) 两种表述。主分支固定为 `tro_literal: knots=0..8`，同时报告 `eight_centers` 敏感性分析，不能在看到结果后选择。

当前 `mi_reward/scoring/dame_soft_histogram.py` 是 latent 适配版本，含 pair-dependent normalization、channelwise 聚合和额外重归一化。E1 新建 image-only 实现，不直接复用该类作为主结果。

---

## 5. 论文的 6-DOF 图像 interaction matrix

对每个像素，TRO 使用：

\[
\mathbf L_x=
\begin{bmatrix}
-1/Z & 0 & x/Z & xy & -(1+x^2) & y\\
0 & -1/Z & y/Z & 1+y^2 & -xy & -x
\end{bmatrix}.
\]

图像梯度转到度量图像平面：

\[
\bar\nabla I=
(\bar\nabla I_{x_m},\bar\nabla I_{y_m})
=(p_x\bar\nabla I_x,p_y\bar\nabla I_y),
\]

其中 \(p_x,p_y\) 来自 wrist camera intrinsics。每个像素强度关于 camera twist 的导数为：

\[
\frac{\partial \phi(i-\bar I)}{\partial\mathbf r}
=-\frac{\partial\phi(i-\bar I)}{\partial i}
\bar\nabla I\,\mathbf L_x.
\]

主设置按论文使用常数深度 \(Z=Z_0\)。在 simulator 中另做 true per-pixel depth 对照，以区分深度近似误差和公式实现误差。

---

## 6. 联合概率的一阶和二阶导数

一阶导数：

\[
\frac{\partial p_{II^*}(i,j,\mathbf r)}{\partial\mathbf r}
=\frac{1}{N_x}\sum_{\mathbf x}
\frac{\partial\phi(i-\bar I)}{\partial\mathbf r}
\phi(j-\bar I^*).
\]

二阶导数：

\[
\frac{\partial^2p_{II^*}(i,j,\mathbf r)}{\partial\mathbf r^2}
=\frac{1}{N_x}\sum_{\mathbf x}
\frac{\partial^2\phi(i-\bar I)}{\partial\mathbf r^2}
\phi(j-\bar I^*).
\]

论文式 (20) 的完整二阶项为：

\[
\begin{aligned}
\frac{\partial^2\phi(i-\bar I)}{\partial\mathbf r^2}
=\;&\frac{\partial^2\phi(i-\bar I)}{\partial i^2}
(\bar\nabla I\mathbf L_x)^\top(\bar\nabla I\mathbf L_x)\\
&-\frac{\partial\phi(i-\bar I)}{\partial i}
(\bar\nabla I_{x_m}\mathbf H_x+\bar\nabla I_{y_m}\mathbf H_y)\\
&-\frac{\partial\phi(i-\bar I)}{\partial i}
\mathbf L_x^\top\bar\nabla^2I\mathbf L_x,
\end{aligned}
\]

其中 \(\mathbf H_x,\mathbf H_y\) 是 \(\mathbf L_x\) 两行关于相机运动的导数，\(\bar\nabla^2I\in\mathbb R^{2\times2}\) 是图像 Hessian。实现中必须保留三项；不能只保留类似 Gauss–Newton 的第一项。

---

## 7. MI 梯度、Hessian 和控制律

### 7.1 按 TRO 印刷公式实现

论文式 (16)：

\[
\mathbf L_{MI}
=\sum_{i,j}\frac{\partial p_{II^*}}{\partial\mathbf r}
\left(1+\log\frac{p_{II^*}}{p_I}\right).
\]

论文式 (17)：

\[
\mathbf H_{MI}
=\sum_{i,j}
\left(\frac{\partial p_{II^*}}{\partial\mathbf r}\right)^\top
\frac{\partial p_{II^*}}{\partial\mathbf r}
\left(\frac{1}{p_{II^*}}-\frac{1}{p_I}\right)
+\frac{\partial^2p_{II^*}}{\partial\mathbf r^2}
\left(\frac{p_{II^*}}{p_I}\right).
\]

局部 Newton 控制律，论文式 (15)：

\[
\mathbf v_c=-\lambda\mathbf H_{MI}^{-1}\mathbf L_{MI}^\top.
\]

论文推荐的预条件控制律，式 (21)：

\[
\boxed{
\mathbf v_c=-\lambda(\mathbf H_{MI}^{*})^{-1}\mathbf L_{MI}^\top
}
\]

\(\mathbf H_{MI}^{*}\) 在参考图像处令 \(I=I^*\) 计算一次，随后固定。由于目标是 MI 最大值，信息充足时 \(\mathbf H_{MI}^{*}\) 应为负曲率矩阵。

### 7.2 必须做公式一致性审计

论文印刷的式 (17) 与直接对式 (16) 求导得到的表达式并非逐项同形。E1 不允许静默修改：

- `tro_printed`：逐字实现论文式 (16)–(20)，用于严格复现；
- `exact_derivative`：从同一个 MI scalar 对 \(SE(3)\) perturbation 做自动微分或高精度有限差分，作为数值真值；
- 两者分别记录，不混合选择；
- 若 `tro_printed` 与数值导数不符而 `exact_derivative` 相符，应报告为公式/定义一致性问题，不能写成“MI objective 失败”；
- 若两者都不符，优先检查坐标系、twist 正负号、内参、深度、图像翻转和 B-spline 边界。

这一步是 E1 最关键的防错机制。

---

## 8. E1-0：实现与坐标系 smoke test

### 8.1 数据

- 一个静态纹理平面；
- 5 张不同纹理；
- 每张纹理 3 个参考距离；
- 每个目标位姿分别对 6 个轴做正负微扰。

### 8.2 单轴扰动

平移 epsilon：`[0.25, 0.5, 1.0, 2.0] mm`；

旋转 epsilon：`[0.025, 0.05, 0.1, 0.2] deg`。

有限差分梯度：

\[
L^{FD}_{MI,k}=
\frac{MI(\mathbf T\operatorname{Exp}(\epsilon\hat{\mathbf e}_k))
-MI(\mathbf T\operatorname{Exp}(-\epsilon\hat{\mathbf e}_k))}{2\epsilon}.
\]

有限差分 Hessian：

\[
\mathbf H^{FD}_{MI,:,k}=
\frac{\mathbf L_{MI}(\mathbf T\operatorname{Exp}(\epsilon\hat{\mathbf e}_k))
-\mathbf L_{MI}(\mathbf T\operatorname{Exp}(-\epsilon\hat{\mathbf e}_k))}{2\epsilon}.
\]

同时记录 \(\tfrac12(\mathbf H+\mathbf H^\top)\) 的对称性误差，但不能用对称化掩盖原始错误。

### 8.3 预注册 gate

- analytic/FD gradient cosine median ≥ 0.99；
- 六个轴分别的 gradient sign accuracy ≥ 0.95；
- cross-epsilon gradient cosine median ≥ 0.99；
- Hessian relative Frobenius error median ≤ 0.10；
- \(\|\mathbf L_{MI}(\mathbf r^*)\|_2\) 接近数值噪声；
- \(\mathbf H^*_{MI}\) 的有效方向为负曲率；纹理退化方向单独报告；
- camera positive-twist 的像素运动方向与 \(\mathbf L_x\) 预测一致。

任何一项不通过，停止 E1-1，不调整数据来追结果。

---

## 9. E1-1：6-DOF closed-loop TRO reproduction

### 9.1 数据构造

- 20 个静态 scene/reference pose；
- 每个 reference pose 采样 30 个初始相机位姿；
- 共 600 个 nominal trials；
- 初始图与参考图至少保留 50% 可见内容；
- 所有 initial pose 在运行前固定 seed 和 manifest。

初始误差分三级：

| 难度 | 平移范数 | 旋转范数 | 用途 |
|---|---:|---:|---|
| Local | 0–20 mm | 0–3° | 验证局部数学正确性 |
| Medium | 20–80 mm | 3–10° | 验证预条件 Hessian 的 basin |
| Wide | 80–180 mm | 10–25° | 接近论文展示范围，测试失效边界 |

这些范围是计划值；若 LIBERO wrist 相机工作空间受限，只能在运行前统一缩放并写入 manifest。

### 9.2 方法

Primary：

- `MI-TRO-Hstar`：式 (16)、完整式 (17)–(20)、式 (21)。

必须基线：

| 方法 | 目的 |
|---|---|
| `MI scalar only` | 只看势能曲线，不发控制 |
| `MI + gradient ascent` | 检验仅一阶梯度 |
| `MI + current H`，式 (15) | 检验局部 Newton |
| `MI + H*`，式 (21) | TRO 主方法 |
| `MI + approximated H`，式 (22) | 验证删除二阶概率项的影响 |
| Photometric SSD servo | 论文中的直接视觉伺服基线 |
| Oracle SE(3) controller | simulator 控制和 IK 上限，不参与视觉结论 |

### 9.3 控制约束

- 控制频率、\(\Delta t\)、\(\lambda\)、最大线速度和角速度在 manifest 中固定；
- 对 Hessian 只允许预注册的小对角 damping；
- 所有方法使用相同 velocity clipping 和终止条件；
- 每步保存 MI、\(\mathbf L_{MI}\)、Hessian eigenvalues、commanded/applied twist 和真实位姿误差；
- 若 IK 或碰撞约束改变命令，trial 标记 `controller_confounded=true`，不能计为 MI failure。

### 9.4 指标

- success rate；
- 最终 translation error（m）；
- 最终 rotation error（deg）；
- 达到阈值所需 steps/time；
- MI 是否随控制迭代总体上升；
- path length / SE(3) geodesic length；
- velocity saturation ratio；
- 按 6 个轴和三个难度分层的结果；
- 95% bootstrap confidence interval，按 reference scene 分组重采样。

建议成功阈值：translation ≤ 2 mm 且 rotation ≤ 0.5°，连续 5 步成立。该阈值是 E1 的预注册目标，不是当前结果。

主 gate：

- Local success rate ≥ 95%；
- Medium success rate ≥ 80%；
- `MI-TRO-Hstar` 明显优于 `MI + approximated H`；
- nominal 下至少不劣于 SSD，光照变化下优于 SSD；
- 失败不集中在同一个未校准轴或图像翻转方向。

Wide 只用于画 convergence basin，不设强制通过率。

---

## 10. E1-2：论文式鲁棒性实验

在同一批固定初始位姿上加入：

1. **illumination**：亮度、对比度、gamma、局部阴影；
2. **occlusion**：固定 10%、25%、40% 图像面积遮挡；
3. **depth**：true depth、目标深度常数、带 ±20% 偏差的常数深度；
4. **resolution**：128×128、320×240；
5. **texture**：高纹理、低纹理和单方向纹理。

扰动只改变当前图或运行过程中的观测，不修改 reference pose ground truth。报告每种 corruption 下的 success drop 和 final pose error。低纹理导致 \(\mathbf H^*_{MI}\) 奇异属于预期 failure mode，必须由 eigenvalue/condition number 解释。

---

## 11. E1-3：从视觉伺服到候选排序的最小桥接

该阶段仍然只做**无接触、同场景、同参考图像的短程相机位姿排序**。

### 11.1 候选组

- 每个 reference/anchor 生成 24 个真实 simulator candidate next poses；
- 候选覆盖 6 个 twist 轴的正负方向和组合扰动；
- 使用实际渲染的 wrist 图，不使用 world-model 预测图；
- 每组候选只能在组内排序，raw MI 不能跨 reference scene 比大小。

### 11.2 独立 ground truth

分别报告平移误差和旋转误差。组合排序使用采样尺度归一化的 SE(3) 距离：

\[
d_{SE(3)}^2=
\left(\frac{\|\mathbf t-\mathbf t^*\|_2}{\sigma_t}\right)^2
+\left(\frac{\|\log(\mathbf R^{*\top}\mathbf R)^\vee\|_2}{\sigma_R}\right)^2,
\]

其中 \(\sigma_t,\sigma_R\) 在采样 manifest 中固定，不能用测试结果拟合。

### 11.3 排序分数

- raw endpoint MI：\(MI(I_k,I^*)\)；
- 一步预测增益：\(\Delta MI_k=MI(I_k,I^*)-MI(I_0,I^*)\)；
- TRO local ascent：\(\mathbf L_{MI}\Delta\boldsymbol\xi_k\)；
- 二阶局部增益：

\[
\widehat{\Delta MI}_k=
\mathbf L_{MI}\Delta\boldsymbol\xi_k
+\frac12\Delta\boldsymbol\xi_k^\top\mathbf H^*_{MI}\Delta\boldsymbol\xi_k.
\]

比较 SSD、NCC/SSIM 和 oracle pose distance。LPIPS/DINO 可作为附加视觉排序基线，但不参与 TRO reproduction gate。

### 11.4 指标与 gate

- groupwise Spearman；
- Kendall \(\tau\)；
- pairwise preference accuracy；
- top-1 / top-3 candidate accuracy；
- best-of-K 后的真实 SE(3) error reduction；
- 按位姿误差、纹理和 photometric corruption 分层。

继续进入 E2 的条件：

- nominal pairwise accuracy 的 scene-level bootstrap 95% CI 下界 > 0.5；
- median groupwise Spearman > 0.5；
- MI 在 illumination/occlusion 条件下相对 SSD 有稳定增益；
- `TRO local ascent` 或二阶增益至少不劣于 raw MI；
- 结论在至少 15/20 个 reference scenes 上方向一致。

这里通过只能支持：

> TRO-style MI 能在同一静态场景内排序短程 eye-in-hand 相机位姿。

它仍不能支持“MI 表示 manipulation progress”或“MI 可直接训练通用 RoboReward”。

---

## 12. 消融矩阵

| 维度 | 主设置 | 消融 |
|---|---|---|
| 图像表征 | grayscale pixels | RGB 分通道、现有 latent MI |
| bins | TRO literal \(N_c=8\) | eight centers、16、32 |
| kernel | cubic \(B_3\) | linear、hard histogram |
| filter | Gaussian 5×5 | none、3×3、7×7 |
| depth | constant \(Z_0\) | true depth、±20% |
| Hessian | full \(H^*_{MI}\) | current H、approximate H、identity |
| derivative | analytic TRO | FD、autodiff scalar audit |
| view | wrist | agentview 仅作为预期失配对照 |
| state change | static/pre-contact | contact/post-object-motion 仅作 failure case |

主表只能使用预注册主设置；消融不能反向替换主设置。

---

## 13. 结果表 schema

### 表 1：解析导数审计

| Branch | Axis | Gradient cosine | Sign acc. | Hessian rel. error | Cross-eps cosine | Status |
|---|---|---:|---:|---:|---:|---|
| TRO printed | tx | TBD | TBD | TBD | TBD | TBD |
| Exact derivative | tx | TBD | TBD | TBD | TBD | TBD |
| … | … | … | … | … | … | … |

### 表 2：闭环 6-DOF 收敛

| Method | Difficulty | Success | Final trans. | Final rot. | Steps | Path ratio |
|---|---|---:|---:|---:|---:|---:|
| MI-TRO-Hstar | Local | TBD | TBD | TBD | TBD | TBD |
| MI-TRO-Hstar | Medium | TBD | TBD | TBD | TBD | TBD |
| SSD | Local | TBD | TBD | TBD | TBD | TBD |

### 表 3：鲁棒性

| Method | Condition | Success | Success drop | Final trans. | Final rot. |
|---|---|---:|---:|---:|---:|
| MI-TRO-Hstar | Illumination | TBD | TBD | TBD | TBD |
| SSD | Illumination | TBD | TBD | TBD | TBD |

### 表 4：候选排序

| Score | Spearman | Kendall | Pairwise acc. | Top-1 | Best-of-K reduction |
|---|---:|---:|---:|---:|---:|
| Raw MI | TBD | TBD | TBD | TBD | TBD |
| TRO first-order gain | TBD | TBD | TBD | TBD | TBD |
| TRO second-order gain | TBD | TBD | TBD | TBD | TBD |
| SSD | TBD | TBD | TBD | TBD | TBD |

所有 `TBD` 必须来自日志，不能手工补写。

---

## 14. 实现文件与日志结构

建议新增：

```text
mi_reward/control/tro_image_mi.py
mi_reward/control/tro_image_interaction.py
mi_reward/control/tro_mi_servo.py
eval/libero/collect_e1_tro_static_poses.py
eval/libero/eval_e1_tro_derivatives.py
eval/libero/eval_e1_tro_closed_loop.py
eval/libero/eval_e1_tro_ranking.py
tests/test_tro_image_mi_derivatives.py
```

统一 artifact：

```text
logs/mi_reward/e1_tro_strict/
  manifest.json
  config.yaml
  derivative_audit.json
  closed_loop_results.json
  robustness_results.json
  ranking_results.json
  per_trial.jsonl
  trajectories/
  videos/
```

`per_trial.jsonl` 至少保存：

```text
scene_id
reference_pose
initial_pose
camera_intrinsics
hand_eye_extrinsics
depth_mode
image_size
mi
mi_gradient
hessian
hessian_eigenvalues
commanded_camera_twist
applied_eef_twist
applied_joint_velocity
translation_error
rotation_error
visibility_overlap
termination_reason
controller_confounded
seed
```

---

## 15. 执行顺序和停止条件

```text
1. 建立静态纹理平面，锁定相机坐标系和 twist 正负号
2. 实现 TRO image MI、B3 一阶/二阶导数和 Lx
3. 完成 E1-0 gradient/Hessian finite-difference audit
4. E1-0 通过后运行 Local closed-loop
5. Local 通过后运行 Medium/Wide convergence basin
6. 运行 illumination、occlusion、depth robustness
7. 运行真实渲染的短程 candidate ranking
8. 全部 gate 通过后才进入 Cosmos/WAM candidate ranking
```

停止条件：

- E1-0 失败：只修公式、坐标和数值实现；不讨论 reward model；
- E1-0 通过、闭环失败：检查 overlap、Hessian conditioning、速度映射和 IK confound；
- 闭环成功、排序失败：MI 适合作为控制局部场，但不适合作为候选 reward；
- 排序 nominal 成功、robustness 无增益：MI 没有证明优于更简单的 SSD/NCC；
- 静态 eye-in-hand 全部成功、Cosmos 排序失败：问题位于 world model/生成图，不位于 TRO 数学；
- 接触或物体状态变化后失败：属于模型假设外，不回写为 TRO 复现失败。

---

## 16. E1 通过后才允许的 E2

E2 才可以加入现有 Cosmos 变体和 reward-model 训练：

1. 使用真实 simulator candidate 建立排序真值；
2. 使用同动作的 Cosmos predicted wrist frames 重算 MI 排序；
3. 比较 `real-render ranking → generated-frame ranking` 的保持率；
4. 用 MI/TRO local-gain 产生组内 preference，而不是跨场景 raw reward；
5. 再将 preference 蒸馏给 Qwen3-VL RoboReward；
6. 夹爪、接触和物体任务进度由独立 physical teacher 或 task-progress label 提供。

当前 Cosmos validation 的视觉审计尚未通过，不能用它替代 E1 的真实渲染 ground truth。

---

## 17. 最终判断规则

E1 的最高强度结论应按证据逐级书写：

- **只通过 derivative audit**：TRO 数学链路实现正确；
- **再通过 closed loop**：MI 可用于静态场景的 6-DOF eye-in-hand visual servoing；
- **再通过 candidate ranking**：MI/TRO local gain 可作为同场景、短程、无接触候选的排序信号；
- **再通过 Cosmos transfer**：生成图保留了该排序信号；
- **再通过 reward benchmark**：该信号才有资格成为 observation-based reward model 的组成部分。

这套 E1 不会推翻 v7 的 `NO_GO`。它检验的是一个更窄、与 TRO 数学完全一致的命题，并决定 MI 是否还能作为后续 reward ranking 的局部辅助信号。

---

## 18. 执行前修订：E1-A / E1-B 分离与严格数值边界（2026-09-10）

E1 不从 600 个机器人闭环 trial 开始。先将数学 image-registration、wrist actuation 和闭环控制分离，以避免把手眼外参、IK、robot self-occlusion 或 velocity clipping 误归因为 TRO MI 公式错误。

### 18.1 新执行顺序

```text
E1-A  free-camera static-scene MI scalar / derivative / Hessian audit
E1-B  wrist-camera camera-twist -> EEF-twist -> joint-velocity audit
E1-C  3 scenes x 6 local starts closed-loop pilot
freeze main manifest
E1-1 main closed loop
E1-2 robustness
E1-3 within-scene candidate ranking
```

E1-A 使用不含 robot/grasper self-occlusion 的 calibrated free camera 或显式 mask 后的静态背景；只在该阶段判断 TRO image Jacobian。E1-B 仅验证实际 wrist camera 的 applied pose increment 与 commanded camera body twist，IK/碰撞导致的不一致标为 `controller_confounded`。E1-C 通过后才开始正式闭环规模实验。

### 18.2 Scalar reference 与 histogram 边界

`exact_derivative` 改称 `numerical_scalar_reference`：MuJoCo renderer 不被假定为可微。该 reference 由对最终 MI scalar 的高精度 SE(3) central finite difference 得到；autodiff 只可用于纯 PyTorch B-spline histogram 内部一致性检查。

主 `tro_literal` 必须明确 B3 boundary completion。由于强度缩放到 `[0,N_c]` 后 B3 在边缘仍有 support，histogram summation 采用预注册 padded support `i,j in {-2,...,N_c+2}`，并保存边缘质量与概率和。`eight_centers` 作为独立 sensitivity branch，不得替换主结果。

### 18.3 Gate 修订

true per-pixel depth branch 才使用 gradient cosine `>=0.99`、Hessian relative error `<=0.10` 的公式审计 gate。constant-depth `Z_0` 是 TRO control approximation，单独报告其相对 degradation，不能因深度近似而否定 E1-A mathematical implementation。

reference 处必须报告 MI scalar maximum、gradient norm、raw Hessian symmetry error、negative-curvature effective rank 和 condition number；不得通过对称化或 damping 掩盖原 Hessian。纹理不可观测轴作为预注册 degeneracy 单列，不进入通过 axis 的 median。

E1-3 删除 raw MI 对 SE(3) distance 的未校准 R2 作为 gate。排序主指标为 groupwise Spearman、Kendall、pairwise accuracy、top-k 与 best-of-K true pose-error reduction；若报告 R2，必须以非-held-out probe actions 预注册拟合的 mapping 校准。

E1 成功仍只支持静态、同场景、无接触 eye-in-hand local pose ranking；不能改变 v7 的 geometric-MI `NO_GO`，也不能替代 reward 的 privileged physical-progress/action-direction 主监督。

---

## 19. E1-A0/A1 一阶公式与坐标审计结果（2026-09-10）

本阶段只验证 image MI probability derivative 与论文 6-DOF interaction matrix，不包含 wrist IK、闭环或 reward。PDF 原文式 (12)、(13)、(16)、(18)、(19) 已逐项核对。

### 19.1 Probability derivative

在 64×64 differentiable static textured plane 上比较：

1. 对最终 MI scalar 的 autograd；
2. 对 camera pose 的 central finite difference；
3. 按论文式 (16)/(18) 由每像素真实 pose Jacobian 构造的 `tro_printed` gradient。

结果：autograd/FD cosine `0.9999999999`，Hessian autograd/FD relative error `1.11e-5`；`tro_printed`/autograd cosine `1.0`、relative error `1.35e-15`。三个 probability masses 均为 `1.0`。因此式 (16)/(18) 的实现与同一 scalar 完全一致。

### 19.2 Interaction matrix 与 pose convention

在 240×240 plane 上，将每像素 renderer forward-mode Jacobian 与论文 \(\nabla I L_x\) 比较。论文 \(L_x\) 对 `T_wc Exp(+delta xi)` 的 global cosine 为 `-0.999996`；使用 \(-L_x\) 后为 `0.999996`，relative error `0.00285`，六轴 cosine 均高于 `0.99999`。

将 \(-\nabla I L_x\) 接入式 (16) 后，完整 printed first-order MI gradient 与 scalar autograd 的 cosine 为 `0.999995`、relative error `0.00699`，通过预注册一阶 gate。由此锁定：论文 camera velocity 更新 `T_wc` 时采用 `T_wc Exp(-v_c dt)`；以 `T_wc Exp(+delta xi)` 做 evaluator perturbation 时使用 `-L_x`。

正式结果：

```text
logs/mi_reward/e1_tro_strict/e1_a0_probability_derivative_v3/results.json
logs/mi_reward/e1_tro_strict/e1_a1_interaction_chain_v3/results.json
```

当前证据只授权实现并审计式 (17)/(20) 的完整 Hessian。reference gradient 非零仍需作为 estimator/boundary/discrete-image diagnostic 处理，不再错误归因为 interaction-matrix 符号或式 (16) 实现。

---

## 20. E1-A2：PDF 式 (17) Hessian coefficient isolation（2026-09-10）

PDF 第 963 页已逐式视觉核对。式 (17) 的第二项字面系数确实为：

\[
\frac{\partial^2 p_{II^*}}{\partial r^2}\left(\frac{p_{II^*}}{p_I}\right),
\]

不是直接微分式 (16) 得到的 \(1+\log(p_{II^*}/p_I)\)。实验使用同一 B3 weights、同一 joint-probability Jacobian/Hessian 和同一 synthetic renderer per-pixel pose Hessian，仅替换这一项的系数。

结果：

```text
logs/mi_reward/e1_tro_strict/e1_a2_hessian_formula_isolation_v3/results.json
logs/mi_reward/e1_tro_strict/e1_a2_hessian_formula_isolation_v3.live.log
```

| Hessian branch | relative Frobenius error vs scalar autograd Hessian |
|---|---:|
| scalar central finite difference | 0.000011 |
| exact derivative of equation (16) | 0.000298 |
| PDF printed equation (17) | 0.825922 |

一阶式 (16) 仍与 scalar autograd 一致：relative error `1.35e-15`。两个二阶 probability branches 使用完全相同的 \(\partial p_{II^*}\) 和 \(\partial^2p_{II^*}\)，因此式 (17) 的大误差不能归因于 B3 二阶导数、renderer Hessian 或 finite-difference epsilon；差异被隔离到 printed second-term coefficient。

按第 7.2 节预注册规则，当前状态定义为 `printed-formula/definition inconsistency`。在查明作者实现实际采用 printed 式 (17) 还是式 (16) 的 exact derivative 前，不运行 6-DOF closed loop，也不以任一 Hessian 分支的闭环表现反向选择公式。该结论只涉及 Hessian 定义，不改变已经通过的一阶 \(L_{MI}\) 与 \(L_x\) 链路。

---

## 21. 一手来源审计与后续分支冻结（2026-09-10）

作者 2010 年 ICRA 前序论文 *Improving Mutual Information-Based Visual Servoing* 对 gradient 式 (7) 直接求导，并把“exact Hessian”写为：

\[
\mathbf H=
\sum_{i,j}
\left(\frac{\partial p_{ij}}{\partial\mathbf r}\right)^\top
\frac{\partial p_{ij}}{\partial\mathbf r}
\left(\frac{1}{p_{ij}}-\frac{1}{p_i}\right)
+
\frac{\partial^2p_{ij}}{\partial\mathbf r^2}
\left(1+\log\frac{p_{ij}}{p_i}\right).
\]

这与当前 MI scalar 的自动微分、central finite difference 以及
`exact_derivative=True` 分支一致。TRO 2011 式 (17) 在同一位置印刷为
\(p_{ij}/p_i\)，与前序论文和直接微分不一致。当前公开 ViSP 主仓库中未检索到可用于裁决该差异的 MI visual-servoing 实现；作者个人页提供论文和视频，但没有该控制器源码链接。

因此冻结两个不可互换的分支：

- `scalar_exact`：数学主分支，采用式 (16) 的严格二阶导数；
- `tro_2011_literal`：文献敏感性分支，逐字采用 TRO 2011 式 (17)。

分支身份在看到闭环结果前固定。后续不得把 `tro_2011_literal` 称为 scalar MI 的 exact Hessian，也不得根据闭环表现选择公式。由于 A0 的 synthetic reference 仍有非零梯度，先执行 E1-A3 free-camera local optimization pilot，同时记录目标位姿处 stationarity、MI 最大点偏移和 pose convergence；只有该问题可解释后才进入 wrist actuation E1-B 和正式 E1-C。

### 21.1 E1-A3 未滤波诊断 pilot（非正式 gate）

首轮 3 scenes × 6 local starts 的未滤波诊断结果保存在：

```text
logs/mi_reward/e1_tro_strict/e1_a3_local_optimization_v1/results.json
logs/mi_reward/e1_tro_strict/e1_a3_local_optimization_v1.live.log
```

`scalar_exact_hstar` 的 success rate 为 `0.3333`，median final translation error 为
`4.94 mm`，median final rotation error 为 `0.289 deg`，MI monotone fraction 为
`0.9900`。三个场景的 6 个不同轴初始位姿分别收敛到几乎相同的场景内终点，但 scene 0/1 的终点偏离 reference 约 `16.97 mm`/`4.94 mm`；其终点 MI 均高于 reference MI。这说明 optimizer 确实在最大化当前 scalar，但当前 scalar 的局部极值不严格位于 reference pose。

`tro_2011_literal_hstar` 的 success rate 为 `0.2778`，MI monotone fraction 仅
`0.6125`，与 A2 的 Hessian 定义不一致结果相符。由于该 pilot 未执行第 4.1 节规定的 5×5 Gaussian filter，它不能作为正式 E1 停止证据；下一轮只补齐固定 Gaussian 预处理，其他 controller、starts、bins、padding 和 gate 保持不变。

### 21.2 E1-A3 严格 Gaussian pilot 结果

补齐固定 5×5、sigma 1.0 Gaussian filter 后的结果保存在：

```text
logs/mi_reward/e1_tro_strict/e1_a3_local_optimization_v2/results.json
logs/mi_reward/e1_tro_strict/e1_a3_local_optimization_v2.live.log
```

| Branch | Trials | Success | Median translation | Median rotation | MI monotone fraction |
|---|---:|---:|---:|---:|---:|
| `scalar_exact_hstar` | 18 | 0.0000 | 5.97 mm | 0.346 deg | 0.9858 |
| `tro_2011_literal_hstar` | 18 | 0.0556 | 5.35 mm | 0.269 deg | 0.6368 |

三个 reference 的 exact scalar gradient norm 分别为 `0.1004`、`0.3759`、
`0.1461`，并未因论文规定的 Gaussian filter 降到数值噪声。exact Hessian 在三个场景均为负定，但 reference 不是 estimator 的驻点。因此 `scalar_exact_hstar` 几乎始终提高 MI，却没有恢复真实 reference pose；这不是 controller 符号错误，而是当前 finite-bin Parzen MI scalar 的局部极值与 reference pose 偏离。

按第 18.3 节 stationarity gate，E1-A3 当前为 `NO_GO_FOR_ROBOT_CLOSED_LOOP`。
暂不执行 E1-B/E1-C 或 600-trial E1-1。下一步只允许做不改变主设置的 identity derivative decomposition：分别量化独立像素强度导数、固定边界宽度下的 pose-gradient contribution，以及 B3 histogram phase sensitivity。该诊断用于解释失败来源，不能通过事后选择 bins、padding 或纹理把 gate 改成通过。

### 21.3 E1-A4 identity derivative decomposition

诊断结果保存在：

```text
logs/mi_reward/e1_tro_strict/e1_a4_identity_derivative_decomposition_v1/results.json
logs/mi_reward/e1_tro_strict/e1_a4_identity_derivative_decomposition_v1.live.log
```

三个场景在 `current == reference` 时，MI 对独立 current pixels 的 gradient norm
分别为 `1.36e-4`、`1.47e-4`、`1.49e-4`；这些非零像素导数经过真实 renderer image Jacobian 投影后，准确产生 pose-gradient norm
`0.1004`、`0.3759`、`0.1461`。pixel-chain 与直接 pose autograd cosine 全部为
`1.0`，排除了 pose chain、interaction sign 或 autodiff 路径错误。

移除 2/4/8/12 像素边界后，pose gradient 仍未系统趋近于零；把两张相同图像共同平移
`-0.5` 到 `+0.5` histogram bin，pose-gradient norm 也基本不变；把强度吸附到 bin centers 后仍保留 `0.0966`、`0.3166`、`0.0967`。因此偏差既不是单纯 crop-border 项，也不是简单的 B3 bin-phase 项。证据指向 finite-bin smooth joint-histogram scalar 在 identical images 处本身存在分布式一阶偏差，该偏差再由 image Jacobian 投影成非零 camera-pose command。

E1 严格复现到此得到分层结论：

1. 式 (16) 与 interaction matrix 一阶链路通过；
2. TRO 2011 式 (17) 不是式 (16) 的 exact Hessian，ICRA 2010 与 scalar derivative 支持 `scalar_exact`；
3. 即使采用 exact Hessian 和论文 Gaussian filter，固定 8-bin B3 MI 的 reference stationarity gate 仍失败；
4. 因而不能把当前实现推进到 LIBERO wrist 机器人闭环并把失败计作 actuation 结果。

当前 strict E1 状态冻结为 `MATHEMATICS_AUDITED_OBJECTIVE_LOCALIZATION_GATE_FAILED`。若继续研究，只能另立非严格 modification，例如 reference-gradient centering 或其他 registration objective，并明确与原 TRO reproduction 分表；它们不能回写成 E1 通过，也不能改变 v7 的 MI physical-progress `NO_GO`。
