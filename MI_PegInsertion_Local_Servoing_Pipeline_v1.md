# Pipeline P1：Goal-Centered MI 用于 Peg Insertion 局部视觉对准

**版本日期：2026-09-10**  
**当前状态：可以开始受控 MuJoCo pilot；尚不具备标准 peg-insertion benchmark 条件**

## 1. 结论先行

现有仓库可以做 peg insertion 的第一阶段实验，但不能直接把当前数据生成流程当成可信的 peg-insertion benchmark。

仓库已经具备：

- round/square peg 和明暗桌面变体；
- MuJoCo 场景、RGB/depth/segmentation 渲染；
- 笛卡尔轨迹生成、末端位姿和物体位姿记录；
- 二值接触检测；
- 严格实现并审计过的 Dame--Marchand 图像 MI、一阶图像雅可比和 6-DoF interaction matrix；
- Franka 笛卡尔阻抗控制、末端 Jacobian、外力和外力矩读取接口。

但是当前 peg rollout 有四个会影响结论的问题：

1. `global_cam` 是固定第三视角，不是 TRO 假设的 eye-in-hand 相机；
2. peg 在抓取后通过直接写 `qpos` 跟随 mocap 末端，接触动力学不会真实决定 peg 的运动；
3. 当前 success 主要由终点距离和姿态阈值判断，测试通过不等于 peg 在接触条件下真实插入；
4. round peg 对应的是由四个 box 围成的方形开口，且约有 3 mm 单边间隙，难度偏低，也不是严格的 round-in-round 配合。

因此，新实验不再研究“MI 是否表示整个任务的物理进度”，而研究一个更窄且与 TRO 数学一致的问题：

> 在 peg 已经抓稳、socket 静止、尚未发生接触的局部范围内，经过目标梯度校正的像素 MI，能否提供比 raw MI 更准确的 5/6-DoF 对准方向，并提高进入插入漏斗的概率？

如果该命题通过，再把它接到阻抗/力控插入阶段。若局部无接触 gate 仍失败，就停止把 MI 用作 peg insertion 控制信号。

---

## 2. 与 v7、v8 和严格 E1 的关系

已有结果不能被新实验覆盖：

| 已有证据 | 结论 | 对 P1 的约束 |
|---|---|---|
| v7 actual-next geometric MI：\(R^2=0.0002\)，Spearman \(-0.1934\)，方向 cosine 中位数 \(-0.6690\) | latent/geometric MI 不能直接表示通用 physical-progress direction | P1 不使用 latent MI，不声称全程 reward/progress |
| E1 一阶梯度 cosine \(>0.99999\) | 像素 MI、图像 Jacobian 和 pose chain 的一阶实现正确 | 直接复用该实现和坐标约定 |
| E1 printed Hessian relative error \(0.8259\) | TRO PDF 中打印的二阶式不是一阶式的精确 Hessian | 不把 printed Hessian 作为主控制分支 |
| E1 exact branch 0/18 success，reference gradient 不为零 | raw finite-bin Parzen MI 的极值偏离目标图像对应位姿 | raw MI 只作 baseline；主方法必须显式处理 reference bias |
| v8 corrected M1 | conditional MI 没有稳定超过直接 physical regression | P1 不宣称 MI 优于有监督物理 pose/value head |

P1 是一个新的、独立报告的修改方法，不得写成“严格 TRO reproduction 已通过”。

---

## 3. 当前资产审计

### 3.1 可直接复用

| 资产 | 位置 | 用途 |
|---|---|---|
| Peg 配置 | `mi_reward/configs/tasks/peg_insertion_variants.yaml` | round/square、train/held-out appearance |
| MuJoCo 场景 | `assets/custom_task/scene/scene_peg_*.xml` | 初始几何和 Panda 外观 |
| 资产生成器 | `mi_reward/scripts/prepare_mujoco_assets.py` | 生成独立 P1 场景变体 |
| RGB/depth/mask 渲染 | `mi_reward/sim/mujoco_generalization_worker.py` | 图像、深度和 ROI mask |
| Strict MI | `mi_reward/control/tro_image_mi.py` | 8-bin cubic B-spline MI、梯度、Hessian audit |
| Interaction matrix | `mi_reward/control/tro_image_interaction.py` | 6-DoF point/image Jacobian |
| Franka 接口 | `deployment/realworld/franka_controller.py` | 后续阻抗位姿命令、Jacobian、外力/力矩 |

本次审计执行：

```bash
MUJOCO_GL=osmesa .venv/bin/python -m pytest \
  tests/test_prepare_mujoco_assets.py \
  tests/test_simulator_native.py -q
```

结果为 `4 passed`（分别执行时资产测试 3 个通过、native rollout 测试 1 个通过）。默认 GLFW 和 EGL 在当前 shell 无可用图形上下文，`MUJOCO_GL=osmesa` 可以完成 headless rollout。这是运行环境问题，不是 peg 资产错误。

### 3.2 必须补齐

新建独立场景和代码，不改写 v8 的日志与配置：

1. 在 `panda_hand` 下增加 `wrist_cam`，固定相机外参；
2. 将 peg 以固定 grasp constraint 连接到末端，禁止控制循环直接改 peg 的 `qpos`；
3. 分别建立 round-in-round 和 square-in-square socket；
4. 设置 0.5、1.0、2.0 mm 三档单边 clearance；
5. 用 `mj_contactForce` 或等价接口记录每个接触的法向/切向力，而不只记录 contact boolean；
6. 增加插入深度、最大横向力、卡死和超时判据；
7. 分离接触前对准 controller 与接触后插入 controller。

### 3.3 暂时缺少

- 仓库中没有可用的 Isaac Lab/Isaac Sim 安装；
- 当前 LIBERO 环境不提供标准 peg insertion benchmark；
- Franka 代码中没有已核验的腕部相机流、内外参标定和 peg/socket 工装；
- 当前 MuJoCo worker 没有连续接触力日志与 force-control loop。

所以第一轮应使用自建 MuJoCo pilot。只有 pilot 通过后，才值得迁移到 Isaac Lab Factory/AutoMate 一类标准装配任务或真机。

---

## 4. 研究假设与变量

### H1：局部可辨识性

在固定场景、固定 peg grasp 和正确 socket ROI 下，目标图像附近的 MI 对至少 \(x,y,\theta_x,\theta_y,\theta_z\) 具有可辨识曲率。

### H2：reference-gradient centering

raw MI 在目标位姿的 estimator bias 可以由目标处梯度显式移除，使目标位姿成为修正信号的驻点。

### H3：局部对准收益

修正后的 MI controller 能提高进入 insertion funnel 的概率，但不负责接触后的插入动力学。

### H4：有限的外观鲁棒性

在不改变几何的亮度、对比度和局部遮挡变化下，MI 相对 SSD 的性能下降更小。该假设必须与 ZNCC 等强传统基线比较。

---

## 5. 数学定义

### 5.1 Raw MI

令腕部相机在局部位姿 \(\boldsymbol\xi\in\mathfrak{se}(3)\) 下获得灰度 ROI \(I_{\boldsymbol\xi}\)，目标预接触位姿的参考图像为 \(I_*\)：

\[
S(\boldsymbol\xi)=\operatorname{MI}(I_{\boldsymbol\xi},I_*).
\]

主设置固定使用 E1 已审计的 8-bin、cubic B-spline Parzen estimator。不得在 test split 上选择 bins 或纹理。

### 5.2 目标梯度校正

E1 已证明一般有：

\[
\mathbf b=\left.\nabla_{\boldsymbol\xi}S(\boldsymbol\xi)\right|_{\boldsymbol\xi=0}\neq0.
\]

定义 goal-centered 局部势能：

\[
\widetilde S(\boldsymbol\xi)
=S(\boldsymbol\xi)-\mathbf b^\top\boldsymbol\xi,
\qquad
\nabla\widetilde S(0)=0.
\]

在线控制只需要使用中心化梯度：

\[
\mathbf g_c(\boldsymbol\xi)
=\nabla S(\boldsymbol\xi)-\mathbf b.
\]

其中 \(\mathbf b\) 在参考位姿离线计算并冻结。此操作只保证 reference stationary，不保证它是唯一极大值，因此必须继续检查 Hessian 和局部 basin。

### 5.3 图像到相机速度

完全沿用 E1 已核验的链路：

\[
\frac{\partial I}{\partial\boldsymbol\xi}
=\nabla I\,\mathbf L_x(Z),
\qquad
\mathbf g=\frac{\partial S}{\partial I}
\frac{\partial I}{\partial\boldsymbol\xi}.
\]

坐标扰动、\(T_{wc}\) 更新方向和 \(\mathbf L_x\) 符号必须沿用 E1 的 finite-difference audit，不能在闭环失败后手动翻转符号。

### 5.4 主控制器

先使用一阶、限幅的梯度上升，避免依赖已发现歧义的 printed Hessian：

\[
\Delta\boldsymbol\xi
=\operatorname{clip}(\eta\mathbf D\mathbf g_c,
\Delta\boldsymbol\xi_{\max}),
\]

其中 \(\mathbf D\) 只做平移/旋转单位尺度归一化。第一阶段锁定 \(z\)，控制 5 DoF；通过后再开放 6 DoF。

二阶版本只作为消融：使用 scalar exact Hessian 或数值 Hessian，并进行负定投影和 damping。TRO PDF printed Hessian 只保留为审计 baseline。

### 5.5 Socket ROI

主 ROI 只保留静态 socket/plate 的纹理与边缘：

- mask 掉随相机运动的 peg、手爪和机械臂；
- 深度使用 simulator ground truth 作为主设置；
- constant-depth 与深度噪声作为敏感性实验；
- 参考图像取 peg 尖端位于孔口上方 3--5 mm 的 pre-contact pose，而不是完全插入后的图像。

这样更接近 TRO 的静态场景假设，并避免 peg/gripper 自身纹理主导 MI。

---

## 6. 控制架构

```text
scripted/oracle coarse approach
              |
              v
goal-centered MI local visual servo (pre-contact, 5 DoF)
              |
        funnel-entry gate
              |
              v
Cartesian impedance insertion (z motion + force limits)
              |
              v
success / jam / force-limit termination
```

本研究中的 MI 模块是局部视觉伺服器，也可以被解释为局部候选动作排序器。它不是全程 reward model，也不是 VLA。若以后接 WorldSample/Qwen3-VL，MI 输出只能作为候选轨迹在接触前末段的辅助分数。

---

## 7. 实验阶段

## P0：资产与物理预检

### 目标

证明新场景中的 peg 是由末端 constraint 和 MuJoCo 接触动力学驱动，而不是直接覆盖物体状态。

### 操作

1. 从现有 round/square scene 派生 `scene_peg_*_local_servo.xml`；
2. 增加 wrist camera，验证其位姿随 `panda_hand` 变化；
3. 固定 grasp 后沿 \(z\) 方向压向封闭板面；
4. 检查 peg 不会穿透，接触力非零，末端偏置由 impedance/compliance 决定；
5. 关闭 direct-qpos attachment；
6. headless 主命令固定 `MUJOCO_GL=osmesa`。

### Gate P0

- RGB、depth、segmentation 同步且尺寸一致；
- wrist camera 相对 hand 外参恒定；
- peg/hand 相对变换漂移小于 0.1 mm、0.1 degree；
- 封闭板碰撞测试中无明显穿透；
- 接触力随下压量单调增加；
- 原始 reference rollout 测试继续通过。

任一项失败，不计算 MI。

## P1：参考驻点与导数审计

### 数据

- 3 个 socket texture；
- round/square 两种几何；
- 每个 reference 在 \(x,y,z\) 上做 \(\pm0.1\) mm，在旋转上做 \(\pm0.02\) degree finite difference；
- 同时计算 autograd/analytic gradient、rendered scalar finite difference 和 exact scalar Hessian。

### 比较

1. raw MI gradient \(\mathbf g\)；
2. centered gradient \(\mathbf g_c=\mathbf g-\mathbf b\)；
3. raw printed Hessian；
4. exact/numerical Hessian。

### Gate P1

- analytic 与 rendered finite-difference gradient cosine \(>0.999\)；
- centered reference gradient norm 小于邻域 median gradient norm 的 1%；
- 在受控 5-DoF 子空间中至少 90% reference 的 Hessian 为局部负定或负半定；
- 三个 epsilon 下的控制方向 cosine 中位数 \(>0.99\)。

若 stationarity 被修正但 Hessian 仍系统性不具备目标极大结构，P1 判定 `NO_GO`。

## P2：无接触局部 landscape 与动作排序

### 初始偏移

每个 reference 采样：

- \(x,y\in[-10,10]\) mm；
- roll/pitch \(\in[-3,3]\) degree；
- yaw：round 为 \([-10,10]\) degree，square 为 \([-5,5]\) degree；
- \(z\) 固定在孔口上方 3--5 mm；
- 每个 geometry/appearance/clearance 至少 200 个状态；
- seed 在实验前固定。

每个状态采样 32 个受限局部 twist candidate。candidate 的 ground-truth 好坏由下一步 5-DoF pose error reduction 定义。

### Baselines

- random candidate；
- oracle pose descent（上界）；
- SSD/photometric error；
- ZNCC；
- gradient correlation；
- raw TRO MI；
- goal-centered MI；
- DINO/视觉特征 cosine（辅助现代基线，不参与 MI 调参）。

### 指标

- score 与真实 pose-error reduction 的 groupwise Spearman/Kendall；
- pairwise accuracy；
- top-1/top-4 candidate regret；
- best-of-K 一步 pose-error reduction；
- analytic command 与 oracle descent 的 cosine；
- 各 DoF 的 sign accuracy；
- reference 周围 false maximum 距离。

### Gate P2

goal-centered MI 必须同时满足：

1. command/oracle cosine 中位数 \(>0.70\)；
2. pairwise accuracy \(>0.65\)；
3. top-1 candidate 的 median pose error reduction \(>0\)；
4. 显著优于 raw MI；
5. 主设置不低于 ZNCC 超过 2 个百分点；
6. round 和 square 两类都不能出现系统性反方向。

若只在某一个 texture 或某一个 geometry 上通过，只能报告诊断结果，不能进入方法主张。

## P3：无接触闭环局部对准

### Protocol

- 每个条件 100 个随机初始 offset；
- 最大 30 个 control steps；
- 单步限制：平移 0.5 mm、旋转 0.2 degree；
- \(z\) 在主实验中锁定；
- 满足 \(|x|,|y|<0.5\) mm，roll/pitch/yaw error \(<0.5\) degree 时视为 funnel entry；
- 控制器超时、振荡或离开初始化范围均记失败。

### 指标

- funnel-entry success；
- 最终平移/旋转误差；
- 收敛步数与路径长度；
- MI/pose error 的单调步比例；
- oscillation 和 divergence rate。

### Gate P3

- centered MI success \(\ge80\%\)；
- raw MI success 显著更低，且差值 \(\ge10\) percentage points；
- centered MI 不显著差于 ZNCC；
- 95th-percentile 最终 \(xy\) error \(<1\) mm，角度 error \(<1\) degree。

P3 通过只证明无接触局部对准，不证明插入成功。

## P4：混合视觉-力控插入

### 两阶段策略

1. 用 P3 controller 达到 funnel-entry condition；
2. 冻结视觉横向大动作，使用 Cartesian impedance 沿 \(-z\) 缓慢插入；
3. 横向力超过阈值时停止下压，执行小幅 spiral/search 或直接判 jam；
4. 达到目标深度且横向力未超限时判成功。

第一版使用固定阻抗参数和固定 spiral baseline，不训练 RL/PPO。

### 对照组

- coarse pose + 直接 impedance insertion；
- ZNCC alignment + impedance insertion；
- raw MI alignment + impedance insertion；
- centered MI alignment + impedance insertion；
- oracle pose alignment + impedance insertion。

### 指标

- insertion success rate；
- 达到的 insertion depth；
- peak lateral force/torque；
- force impulse；
- jam rate；
- completion time；
- socket/peg clearance 分层结果。

### Gate P4

相对“直接 impedance insertion”，centered MI 必须：

- success 提升至少 10 percentage points；
- jam rate 下降至少 20%；
- peak lateral force 不增加；
- 在 0.5 mm clearance 上仍有可重复收益。

若只提高视觉误差而不提高插入成功率，论文结论必须停留在 local visual alignment。

## P5：鲁棒性与迁移

只有 P4 通过后执行：

- train：light、round/square、1--2 mm clearance；
- held-out appearance：dark、亮度/对比度、局部阴影；
- held-out camera：焦距 \(\pm5\%\)、外参平移 \(\pm2\) mm、旋转 \(\pm1\) degree；
- depth noise：0、1、2、5 mm；
- partial occlusion：ROI 面积 0%、10%、20%；
- held-out clearance：0.5 mm；
- 后续标准环境：Isaac Lab Factory/AutoMate 或等价标准 assembly benchmark。

迁移到标准 benchmark 时冻结 P1 的 bins、滤波、centering 和 controller gain，只允许按机器人控制频率统一缩放步长。

---

## 8. 消融实验

必须做：

1. raw MI vs goal-centered MI；
2. 8-bin 主设置 vs 16/32-bin sensitivity；
3. socket ROI vs full image；
4. mask peg/gripper vs 不 mask；
5. ground-truth depth vs constant/noisy depth；
6. first-order vs exact-Hessian damped controller；
7. 5-DoF fixed-z vs full 6-DoF；
8. fixed lighting vs appearance shift；
9. MI alignment + insertion vs insertion-only；
10. physical clearance 分层。

不得把事后表现最好的 bin、ROI 或 texture 重新定义为主协议。主设置在 P2 前冻结。

---

## 9. 结果表结构

### Table 1：局部方向与排序

| Method | Spearman ↑ | Pair Acc ↑ | Command Cosine ↑ | Top-1 Error Reduction ↑ | False-Max Dist ↓ |
|---|---:|---:|---:|---:|---:|
| SSD | | | | | |
| ZNCC | | | | | |
| Raw MI | | | | | |
| Centered MI | | | | | |
| Oracle pose | | | | | |

### Table 2：闭环对准

| Method | Funnel Success ↑ | Final XY mm ↓ | Final Angle deg ↓ | Steps ↓ | Divergence ↓ |
|---|---:|---:|---:|---:|---:|

### Table 3：插入结果

| Method | Success ↑ | Depth mm ↑ | Peak Lateral Force ↓ | Jam ↓ | Time ↓ |
|---|---:|---:|---:|---:|---:|

### Table 4：鲁棒性

按 geometry、clearance、appearance、camera perturbation 和 depth noise 分层，禁止只报告聚合平均值。

统计报告使用 bootstrap 95% confidence interval；success/jam 使用 paired seed 和 McNemar 或 paired bootstrap，连续量使用 paired bootstrap。所有失败样本都保留。

---

## 10. 建议新增文件

```text
mi_reward/configs/tasks/peg_insertion_local_servo.yaml
mi_reward/control/goal_centered_mi_servo.py
mi_reward/sim/peg_insertion_local_env.py
mi_reward/sim/peg_contact_metrics.py
mi_reward/scripts/run_peg_p0_asset_audit.py
mi_reward/scripts/run_peg_p1_derivative_audit.py
mi_reward/scripts/run_peg_p2_local_ranking.py
mi_reward/scripts/run_peg_p3_closed_loop.py
mi_reward/scripts/run_peg_p4_hybrid_insertion.py
tests/test_goal_centered_mi_servo.py
tests/test_peg_contact_physics.py
logs/mi_reward/peg_local_servo_p1/
```

建议从现有 `prepare_mujoco_assets.py` 派生场景生成逻辑，但输出新的 XML 名称，避免破坏 v8 数据 provenance。

统一输出每个 trial 的：

```json
{
  "seed": 0,
  "geometry": "round",
  "clearance_mm": 1.0,
  "appearance": "light",
  "initial_pose_error": [0, 0, 0, 0, 0, 0],
  "controller": "goal_centered_mi",
  "gradient": [0, 0, 0, 0, 0, 0],
  "command": [0, 0, 0, 0, 0, 0],
  "final_pose_error": [0, 0, 0, 0, 0, 0],
  "funnel_entered": false,
  "inserted_depth_mm": 0.0,
  "peak_lateral_force_n": 0.0,
  "jam": false,
  "success": false
}
```

---

## 11. 执行顺序与停止规则

### 第一周最小实验

1. P0：建立 wrist camera、固定 grasp 和真实 contact metrics；
2. P1：在 6 个 reference 上完成 derivative/stationarity audit；
3. P2：先跑 round/light/1 mm 的 200-state ranking；
4. 只有 P2 达标才扩展 square 和 appearance；
5. 只有两类几何都达标才跑 P3 closed loop。

### 硬停止规则

- P1 Hessian/landscape 不支持局部目标极大：停止；
- P2 centered MI 仍系统性反方向：停止；
- P2 明显低于 ZNCC 且外观鲁棒性也无优势：停止 MI 主方法；
- P3 无接触 success 低于 80%：不进入接触实验；
- P4 不提升真实 insertion success：不把它写成 peg-insertion 方法。

停止后仍可将结果写成对 MI visual servo objective 的系统 falsification/diagnostic，但不能通过加入 WAM、LaWAM 或 Qwen 来掩盖局部控制 gate 的失败。

---

## 12. 成功后能够主张什么

P2--P4 全部通过时，合理的论文主张是：

> Finite-bin Parzen MI 在目标位姿存在 estimator-induced gradient bias。通过 reference-gradient centering，并将 MI 限定在静态 socket ROI 和接触前局部对准阶段，可以恢复可用的局部视觉控制信号，并提高窄间隙插入的 funnel-entry 与最终成功率。

## 13. P0 当前执行记录（2026-09-10）

先运行了现有资产与 native rollout 预检：

```text
MUJOCO_GL=osmesa .venv/bin/python -m pytest \
  tests/test_prepare_mujoco_assets.py tests/test_simulator_native.py -q
```

结果：`4 passed in 7.19s`。这只说明现有资产生成和 legacy native rollout 可运行，不能替代新的
物理 P0 gate。

随后新增并运行了结构审计脚本：

```text
mi_reward/scripts/run_peg_p0_asset_audit.py
```

正式输出：

```text
logs/mi_reward/peg_local_servo_p1/p0_asset_audit.json
logs/mi_reward/peg_local_servo_p1/p0_asset_audit.log
```

旧 scene 的 P0 判定为 `FAIL`：只有 `global_cam`，使用 `kinematic_task_proxy=true`，没有
contact-force logging 和 clearance 参数。

随后已生成独立的 `scene_peg_round_local_servo.xml` 和
`peg_insertion_local_servo.yaml`。该场景加入了 hand-relative `wrist_cam`、
`peg_grasp_weld`，关闭 qpos proxy，并启用 clearance/contact-force 配置。新场景结构审计
通过；MuJoCo headless load/step 也通过，当前静态 rollout 观测到 `ncon=6`、resultant
contact force `8.3406 N`。接触统计接口位于 `mi_reward/sim/peg_contact_metrics.py`。

正式输出：

```text
logs/mi_reward/peg_local_servo_p1/p0_local_asset_audit.json
logs/mi_reward/peg_local_servo_p1/p0_local_physics.log
logs/mi_reward/peg_local_servo_p1/p0_contact_metrics.log
```

这只完成了 P0 的结构与可加载性子项；尚未宣称“下压时接触力随位移单调增加”这一完整
物理 gate，也尚未计算 P1 MI。下一步是用该独立场景执行受控下压/横向扰动审计，再决定
是否进入 P1 导数审计。

不能主张：

- MI 表示通用机器人任务进度；
- MI 可以替代 privileged physical reward；
- MI 单独解决接触动力学；
- 当前自建 MuJoCo 结果等价于标准 benchmark 或真机结果。

仅完成 P2/P3，适合作为 workshop 级局部视觉伺服研究或完整论文中的一个模块。若要形成更强投稿，至少还需要 P4 的真实插入收益和一个标准仿真 assembly benchmark；真机 Franka 会显著增强证据，但应在腕部相机标定、工装和安全力限完成后再做。

---

## 13. 当前建议

现在先实现 **P0 + P1 + P2 的最小版本**，不要立刻训练 reward model，也不要接 Cosmos、WAM 或 Qwen3-VL。这个最小实验不依赖大模型训练，现有 MuJoCo、MI 数学实现和 4090 机器已经足够。它能最快回答唯一关键问题：经过预注册的 goal-centering 后，MI 在 peg insertion 的局部对准区间里究竟有没有可重复的方向信息。

## 14. P0 controlled downward press result（2026-09-10）

在 `tmux train` 中对 welded-grasp local-servo scene 执行了先抬升、再沿末端 z 轴下压的审计，
并在每个 settled frame 记录接触数量及法向/切向力。期间排除了 legacy Panda mesh 的内部
自碰撞，避免机器人自身碰撞被误计为 peg/socket 接触。

正式输出：

```text
logs/mi_reward/peg_local_servo_p1/p0_press_audit.json
logs/mi_reward/peg_local_servo_p1/p0_press_audit.log
```

结果：

```text
normal_force_monotone_fraction = 1.0
peak_normal_force_n = 23.2894
p0_press_pass = true
```

因此 P0 的结构、可加载性和受控下压接触响应子项均通过，可以进入 P1 reference
stationarity/derivative audit。该结果只验证局部 MuJoCo 接触物理和日志链路，不包含 MI 控制
结论，也不等价于 peg insertion success。

相机姿态修正后重新运行 P1，finite-difference 已显示明显的 pose 响应，但 analytic
interaction-matrix gradient 仍未对齐：

```text
logs/mi_reward/peg_local_servo_p1/p1_derivative_audit_v2.json
epsilon=1e-4 :  0.0535
epsilon=3e-4 : -0.6364
epsilon=1e-3 : -0.8103
```

因此 P1 仍为 `FAIL`。这排除了“wrist camera 完全只看到背景”这一首要故障，但仍需校准
MuJoCo camera frame、depth 单位和实际 `fovy` 对应的 focal length，之后才能判断
`tro_image_interaction.py` 在该场景下是否与渲染扰动一致。

## 17. Synthetic Jacobian sanity check（2026-09-10）

使用固定 metric depth 和已知平滑图像执行了 `image_pose_jacobian()` 的 synthetic audit：

```text
logs/mi_reward/peg_local_servo_p1/tro_synthetic_jacobian_audit.json
logs/mi_reward/peg_local_servo_p1/tro_synthetic_jacobian_audit.log
```

六个 DoF 的最大 analytic/finite-difference absolute error 为 `2.84e-14`，synthetic
arithmetic check 通过。这说明 interaction matrix 的基本实现没有算术错误；它不替代真实
MuJoCo P1 gate，剩余问题仍是 camera twist 与实际渲染扰动的对应关系，以及 finite-bin MI
在真实图像上的数值稳定性。

随后完成了独立 camera geometry audit：socket 在 camera coordinates 中位于有效视场内，
`rgb_std=76.88`，depth 范围为 `0.147--75.381`，有效 depth fraction 为 `0.634`，并从
实际 `fovy=48°` 得到 `fx=fy=143.75`（128 px audit）。使用该内参重跑 P1 后，cosine
仍为 `0.0535/-0.6364/-0.8103`。因此当前环境已具备可见场景和有效 depth，但
interaction-matrix/image-Jacobian 链路仍未对齐；P2 继续保持冻结。

## 16. Camera twist audit（2026-09-10）

独立的 point-projection audit 已完成：

```text
logs/mi_reward/peg_local_servo_p1/camera_twist_audit.json
logs/mi_reward/peg_local_servo_p1/camera_twist_audit.log
```

参考 socket 点在 camera frame 的深度为 `0.1741 m`，投影位于图像中心附近。逐轴 mocap
扰动得到的像素导数数量级与 `fx/depth` 一致，说明渲染投影本身有效。真正的实现错误是
当前 P1 直接把 world/mocap 六轴扰动传给 camera-frame 的 `point_interaction_matrix()`；
没有施加 camera rotation，也没有补偿 wrist camera 相对 hand 的 lever-arm 项。因此
analytic gradient 与 mocap finite difference 使用的不是同一组变量。下一步先实现
`mocap_twist -> camera_twist` 的显式变换，再重跑 P1；在此之前不解释 MI 结论。

随后已接入显式 `mocap_twist -> camera_twist` 变换，并补充初始 `mj_forward`，修复了上一版
analytic gradient 全零的实现问题。最新审计输出：

```text
logs/mi_reward/peg_local_servo_p1/p1_derivative_audit_v6.json
epsilon=1e-4 : -0.2356
epsilon=3e-4 : -0.8108
epsilon=1e-3 :  0.4054
```

analytic gradient 已非零，但仍未达到 P1 gate。这表明坐标变换主链路已经接通，剩余问题是
相机 twist 符号/扰动变量约定与 finite-bin MI 在小扰动下的数值稳定性，需要用独立的
synthetic image/Jacobian audit 继续拆分；当前仍不能把该结果作为 MI objective 的否定证据。

## 15. P1 reference derivative audit result（2026-09-10）

在通过 P0 的 `wrist_cam` welded-grasp scene 上执行了 8-bin cubic-B-spline MI 的 analytic
gradient 与 rendered scalar finite-difference 对照。三个 epsilon 的 cosine 为：

```text
epsilon=1e-4 :  0.0186
epsilon=3e-4 : -0.1800
epsilon=1e-3 : -0.3335
```

正式输出：

```text
logs/mi_reward/peg_local_servo_p1/p1_derivative_audit.json
logs/mi_reward/peg_local_servo_p1/p1_derivative_audit.log
```

`p1_derivative_pass=false`，未达到 P1 要求的 `>0.999`。当前不能把该结果直接解释为 TRO
数学错误；P1 使用了新 wrist-camera 场景和渲染深度，下一步必须审计 camera 坐标、depth
convention、扰动位姿和 image Jacobian 的对应关系。按照停止规则，暂不进入 P2 ranking 或
MI closed-loop，先修复并重新通过导数 audit。
## 18. Full-frame image Jacobian audit（2026-09-10）

真实 MuJoCo 场景的逐像素 finite-difference 与 image Jacobian 对照记录在：

```text
logs/mi_reward/peg_local_servo_p1/scene_image_jacobian_audit.json
logs/mi_reward/peg_local_servo_p1/scene_image_jacobian_audit.log
```

六个 DoF 的 cosine 约为 `-0.25` 到 `0.11`。当前 full-frame 中大量远裁剪背景和渲染边界
变化污染了导数比较，因此下一步必须先使用预注册 socket ROI，并屏蔽无效 depth/background
像素，再重新验证 image Jacobian，之后才重跑 MI derivative。
## 19. ROI image Jacobian audit（2026-09-10）

固定中心 socket ROI 并屏蔽无效 depth 后，逐轴 cosine 仍接近零，记录于
`logs/mi_reward/peg_local_servo_p1/scene_roi_image_jacobian_audit.json`。同时 finite-difference
范数异常大，说明 MuJoCo 离散 rasterizer 在 `1e-4` 位姿扰动下发生像素跳变；原始 raster
finite difference 不是连续图像导数 ground truth。下一步需固定 ROI 后加入可审计的图像平滑
并扫描更大 epsilon，再决定是否继续 MI derivative。

ROI Gaussian smoothing（sigma=1）并将 epsilon 提高到 `1e-2` 后，部分 DoF cosine 有改善，
但仍未达到 P1 gate：`dof0=-0.664`、`dof1=0.373`、`dof2=0.005`、`dof3=-0.071`、
`dof4=-0.395`、`dof5=0.350`。结果保存在
`logs/mi_reward/peg_local_servo_p1/scene_roi_image_jacobian_smooth.json`。这说明 rasterizer
噪声只是部分原因，camera twist 到像素 Jacobian 的映射仍需校准。
