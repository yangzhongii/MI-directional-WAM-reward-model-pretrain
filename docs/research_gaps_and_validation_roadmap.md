# MI Directional Potential：当前缺口与验证路线

> 状态日期：2026-08-31  
> 当前可复现实验：`generalization_rigid_v3`  
> 本文档区分“已经实现”“可作为基线”“正式论文仍缺失”，避免把数据生成完成误写成完整策略闭环。

## 1. 模型的准确定位

当前部署模型 `VisualGoalPotential` 输出逐帧、目标条件状态势能：

```text
V_phi(o_t, g)
```

部署 reward 是势能差：

```text
r_t^MI = gamma * V_phi(o_{t+1}, g) - V_phi(o_t, g)
```

因此论文中应称其为 **goal-conditioned state potential model**、
**dense reward model** 或 **progress potential model**。它没有使用 Bellman target、
policy return 或动作条件 `Q(s,a)` 训练，不能在未加限定时称为传统 `V^pi` critic。

## 2. v3 已经实现的内容

`generalization_rigid_v3` 当前包含：

- 三类刚性操作任务：`pick_place`、`push_shape`、`peg_insertion`；
- instance 变化：apple/banana、T/B、round/square；
- 两种 MuJoCo 外观：明亮实验室桌面、深色工业工作台；
- 同一 parent/instance/scene context 内共享一个实测物理初态；
- 3 个成功 profile 和 5 个 near-miss/失败 profile；
- RGB、depth、mask、action、robot state、object state、relation 和物理结果 sidecar；
- `train`、`instance_heldout`、`scene_heldout`、`joint_heldout` 四个 split；
- bounded visual/action/kinematic/relation directional teacher；
- preference score 与 student distillation 共用同一条 teacher potential curve；
- 真机和双机接口保留，但不进入当前默认实验。

本次数据生成报告：

```text
candidate trajectories: 480
success references:       6
train:                  120
instance_heldout:       120
scene_heldout:          120
joint_heldout:          120
successful outcomes:    180
failed outcomes:        300
artifact acceptance:   100%
```

这里的 `acceptance` 表示文件、物理 sidecar 和验证契约有效，不表示任务成功。
失败轨迹是 reward 学习所需的有效负样本。

## 3. Scene-level generalization 的缺口

### 3.1 当前只能称为 appearance-level scene shift

当前 `table_light_1` 与 `table_dark_2` 只改变：

- skybox 和背景颜色；
- 桌面纹理；
- 地面与墙面颜色；
- 环境光和主光源位置。

当前没有改变：

- peg hole、basket、push target 的真实位置；
- 相机位姿；
- 桌面几何布局；
- 障碍物、干扰物和其他目标；
- 目标方向；
- 场景语义和物体角色。

配置里的 supermarket、fruit stall、parcel sorting 等自然语言 prompt 目前不存在；
已有 prompt 也只是元数据。默认 `transfer_enabled: false`，不会由 Cosmos Transfer
生成语义背景。

因此 v3 论文表述只能是：

```text
appearance-level generalization across lighting and material changes
```

不能表述为：

```text
generalization across semantic environments or spatial layouts
```

### 3.2 缺少 layout-level variation

需要在后续 v4 中增加：

- peg hole 位于左/右/前/后以及不同 yaw；
- basket、分拣箱和 push target 位于不同可达位置；
- 相机高度、方位和焦距的小范围变化；
- 桌面尺寸和局部遮挡变化；
- distractor object、unused hole、工具箱或障碍物；
- 目标与初始物体距离、方向的系统变化。

目标位置必须写入 MuJoCo body，而不是只对 planner endpoint 加 jitter。规划器和
simulator 应从同一个目标 body 读取位姿并计算 relation。

### 3.3 缺少 semantic scene variation

后续可增加但必须控制任务语义不变：

- Pick-place：实验室、仓储分拣台、水果摊、超市理货台；
- Peg insertion：实验室装配台、工业流水线、维修工作台；
- Push：形状分拣板、包装台、工业质检台。

如果从“水果放入篮子”变为“包裹送入传送带入口”，不仅是 scene 变化，也改变了
object role 和 task semantics。正式实验必须将 appearance、layout、instance 和 task
semantic 分成独立轴，否则无法解释性能变化来源。

### 3.4 Scene-specific success reference

v3 的 6 条成功参考按 `task + instance` 建立，因为两个 appearance scene 的目标布局相同。
一旦 v4 改变目标位置或场景布局，成功参考必须改成：

```text
task + instance + scene/layout -> independent success reference
```

否则会出现候选轨迹向右侧目标运动，却与左侧目标成功轨迹对齐的错误 teacher target。

## 4. 数据划分与独立测试缺口

当前 reward 配置将：

```text
instance_heldout + scene_heldout + joint_heldout
```

全部用于 checkpoint selection。它们因此属于 validation，不能同时作为论文最终 test。

### 最小修正

```text
train                  -> optimization
instance_heldout       -> validation
scene_heldout          -> validation
joint_heldout          -> final test only
```

`joint_heldout` 不得参与早停、超参数选择或 checkpoint 选择。

### 更严格的最终方案

新增 parent-seed 级别的 train/validation/test 隔离。所有 appearance、instance 和 layout
条件下，同一个 parent seed 的后代只能属于一个数据阶段，防止相似规划模板跨 split。

## 5. Reward model 内部评测缺口

当前已有训练期 pair accuracy，但仍缺独立的 `VisualGoalPotential` simulator evaluator。
旧版 `eval_progress_corr.py` 只支持 legacy `TrajectoryRewardHead`，不能直接作为当前模型
的正式评测器。

独立 test 至少需要按 task 和 split 报告：

- success/failure ROC-AUC；
- 同初态 preference accuracy；
- 成功轨迹 `V_T - V_0`；
- 逐帧 potential 与 oracle task progress 的 Pearson/Spearman；
- 正向 transition 比例；
- `regress_after_progress` 回退检测率；
- wrong-direction、undershoot、overshoot 的分组准确率；
- 每个任务的均值、方差和置信区间。

最终 test 必须使用 MuJoCo `task_outcome` 和 relation 作为评测标签，不能使用 teacher score
本身作为 ground truth。

## 6. RBM-EVAL 的正确定位

RBM-EVAL 可以保留，但只作为外部 OOD reward benchmark：

- `reward_alignment`：适合检查 potential 是否跟随外部轨迹进度；
- `policy_ranking`：只排序已采集 policy trajectories，不执行策略；
- `quality_preference`：只能检查离线终局偏好；
- 不能证明 closed-loop success rate 或 RL sample efficiency。

当前 RBM 配置还存在：

- `visual_goal.goal_sidecar: null`，所有样本都走 null-goal；
- `preference_aggregation: last`，只比较终点 `V_T`；
- 没有直接评估部署 reward `gamma*V_{t+1}-V_t`；
- 当前 student 没有语言指令输入，不适合无目标图像的 different-task preference。

正式 RBM 结果应：

- 提供独立成功目标图像 sidecar，或明确只报告 null-goal ablation；
- 同时报告 terminal potential、net progress 和 transition reward；
- 将 same-task quality comparison 与 different-task comparison 分开；
- 明确写成 external OOD offline evaluation，不写成 policy evaluation。

## 7. 决策效用与 closed-loop 缺口

### 7.1 最低成本：同初态候选选择

当前每个 context 已有 8 条物理候选，可以直接评估：

```text
same initial state
  -> score 8 candidate trajectories
  -> choose Top-1
  -> use measured MuJoCo task_outcome as success label
```

报告：

- Top-1 success rate；
- success@k；
- oracle regret；
- 各失败模式被选中的比例。

对比随机、pixel/image distance、DINO/LaWAM cosine、visual-only MI、完整 teacher/student
和 oracle。该实验是 offline counterfactual action-candidate selection，不应冒充真正闭环。

### 7.2 真正 closed-loop

✅ 已实现标准 LIBERO `reset/step` 闭环：

```text
policy action
  -> simulator next observation
  -> V_phi(o_t, g), V_phi(o_{t+1}, g)
  -> dense potential shaping
  -> sparse environment success remains authoritative
```

应比较 sparse-only 与 sparse + MI shaping 的：

- success rate；
- environment steps/sample efficiency；
- return curve；
- held-out object/scene success；
- reward hacking 和 collision rate。

当前主闭环采用独立双视角 CNN/RLPD，不依赖 VLA 或 RLinf：官方成功 demo 与 online
replay 按 50/50 采样，策略使用 ResNet18，critic 使用 10-head Q ensemble；冻结
LaWAM/DINO + `VisualGoalPotential` 替代参考配置里的 Qwen VLM reward。三种 reward mode、
严格初态划分、视频、曲线、汇总和两节点实验级分片均已实现。旧 MLP SAC 仅保留作低容量
baseline。代码完成不等于实验结论完成；仍须跑完多任务、多 seed 曲线。

## 8. 外部 benchmark 缺口

优先级建议：

1. 当前三任务 MuJoCo：可控 mechanism study；
2. RBM-EVAL-OOD：外部离线 reward 泛化；
3. LIBERO：单臂标准任务和 closed-loop policy/shaping；
4. RoboTwin：双臂 embodiment，与当前单末端执行器差距较大，后续再做。

若目标是 A 类会议，三类自定义任务加 RBM-EVAL 通常不足以支撑完整策略效用声明；至少应
补一套标准 simulator policy benchmark，优先 LIBERO。

## 9. 必需 baseline 与 ablation

保持同一数据、模型容量、seed 和训练步数，至少比较：

- random candidate selection；
- pixel/image distance reward；
- DINO/LaWAM latent cosine；
- visual-only directional MI；
- visual + relation；
- visual + action；
- visual + kinematic；
- full privileged teacher；
- full teacher without outcome anchor；
- full teacher without monotonic directional projection；
- sparse task reward only；
- sparse task reward + learned MI potential。

当前实现状态：候选排序 baseline 与十组 teacher/student ablation 的自动化代码已经完成；
三 seed 实验尚未全部运行，因此本节不能标记为实验完成。运行入口为：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --seed 0 --seed 1 --seed 2 --skip-completed
```

## 10. 分阶段路线

### P0：保住 v3 基线与测试合法性

1. 保留当前 v3 数据，不覆盖；
2. ✅ 已将 `joint_heldout` 从 checkpoint selection 中移除；
3. 重新训练 split 隔离后的 v3 appearance-level baseline；
4. ✅ 已实现当前 student 的独立 joint-heldout evaluator；
5. ✅ 已实现同初态 Top-1 candidate selection；
6. ✅ 已完成 RBM-EVAL 辅助外部指标；
7. ✅ 已实现 baseline/ablation 自动化，待运行三 seed 矩阵。

独立评估命令为：

```bash
bash mi_reward/scripts/eval_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml
```

评估器按 split/task/pair type/failure mode 输出指标，并以 checkpoint 元数据强制检查
test split 是否参与过训练或选模。旧 checkpoint 即使通过
`--allow-split-overlap` 调试，也会明确写入 `split_isolation.valid: false`。

### P1：v4 layout/semantic scene

1. 使用新的 `run_id` 和输出目录；
2. 扩展参数化 MuJoCo scene specification；
3. 增加目标位姿、相机、障碍物和语义场景；
4. 为每个 scene/layout 建立独立 success reference；
5. 增加 appearance/layout/semantic/joint test；
6. 不将 v3 与 v4 cache 或 manifest 混用。

### P2：标准闭环 benchmark

1. ✅ 接入 LIBERO；
2. ✅ 接入 frozen MI value 的逐步 potential shaping；
3. ✅ 实现 sparse、MI、sparse+MI SAC 对照；
4. ⏳ 跑完三个任务、三个 seed 并汇总学习曲线；
5. ⏳ 评估 LIBERO-PRO semantic/instance OOD 或真机是否必要。

## 11. 当前完成标准

项目不能仅因为数据生成、reward SFT 或 RBM-EVAL 完成而宣称闭环完成。至少满足以下条件
后，才能宣称 simulator reward-learning 闭环成立：

- 独立 test split 未参与训练和选模；
- reward 对真实 progress、success 和回退方向有效；
- 同初态 candidate selection 明显优于随机与视觉相似度；
- 至少一个 closed-loop simulator 中，MI shaping 提高成功率或样本效率；
- 结果包含必要 baseline、ablation、多 seed 和失败模式分析。
