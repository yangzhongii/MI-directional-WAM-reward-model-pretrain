# 原始研究计划与当前实现对照

本文记录 `MI_Directional_Potential_Research_Idea_v2.md` 在当前代码中的落地状态。
当前目标是先完成离线 benchmark 闭环；真机和 Cosmos 生成保留接口，但不进入默认实验。

## 当前默认闭环

```text
3 task families × 5 train seeds
  -> 2 instance levels × 2 scene levels × 8 candidate profiles
  -> MuJoCo RGB/actions/robot state/object state/relations
  -> artifact verification + measured task outcome
  -> 6 independent task-and-instance success references
  -> strict train/instance-heldout/scene-heldout/joint-heldout splits
  -> visual + inferred-action + physical-kinematic + relation MI teacher
  -> measured-outcome-anchored preference pairs
  -> visual-only VisualGoalPotential distillation
  -> independent joint-heldout + Top-1 candidate selection
  -> RBM-EVAL
```

默认数据规模为 576 条物理 rollout；reference parent seed 的 96 条后代全部从候选中
排除，剩余 480 条，四个 split 各 120 条。只有 `train` 进入优化，instance/scene
held-out 用于选模，`joint_heldout` 只进入独立最终测试。

## 与原计划的对应关系

| 原计划要求 | 当前实现 |
|---|---|
| instance-level generalization | 每类任务显式 train/held-out instance，独立 split |
| scene-level generalization | 浅色 train scene 与深色 held-out scene，独立 split |
| joint generalization | `joint_heldout` 同时隔离 instance 与 scene |
| success 与 near-miss 排序 | 3 个成功 profile + 5 个失败/回退 profile |
| 不把坏文件当训练数据 | `verification` 只负责 artifact validity |
| 不把任务失败丢掉 | `task_outcome` 保存 success/contact/collision/goal 标签 |
| visual process latent | LaWAM/DINO visual token 的方向 MI 势能 |
| action process latent | LaWAM 从相邻图像推断的 `[T-1,Q,D]` action latent |
| physical motion latent | robot/object state 编码的 `[T,1,D]` kinematic latent |
| relation potential | MuJoCo 实测 relation sequence |
| preference construction | measured outcome anchor + hardest near-miss MI 排序 |
| 可部署 reward | privileged teacher 蒸馏为视觉 `VisualGoalPotential` |

这里的 LaWAM action latent 与 kinematic latent 不是同一个变量：前者是视觉推断的运动
表征，后者来自仿真/机器人实测状态。两者同时存在，正好对应“视觉上看起来怎么动”和
“机械臂与物体实际上怎么动”。

## 暂不进入默认实验的部分

- 真机：`base_data.source: jsonl`、独立部署代码和 schema 保留；
  `real_world.enabled: false`，当前不采集、不训练真机数据。
- Cosmos Predict/Transfer：worker、配置与一致性验证保留，但默认 disabled；它们只作为
  后续 ablation，不再污染当前物理闭环。
- 双机不会合并为 96 GB 统一显存。当前实现是 sample sharding，两台 48 GB GPU 各运行
  一个完整 worker，处理不同候选 shard。

## 运行命令

单机默认：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml
```

双机模式只追加一个参数：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --distributed
```

双机启动前先检查配置：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --dry-run \
  --distributed
```

默认的 `transport: rsync` 允许两台机器使用不同的项目和日志绝对路径。每个节点在
配置中声明自己的 `repo_root` 和 `python`；主节点自动发送 shard 及其引用 artifact，
再回收并改写远端输出路径。远端准备完成后运行 `--preflight --distributed`，然后用
`.venv/bin/python scripts/dist_pipeline_smoke.py` 做小规模双机实测。
planner、simulator 以及启用后的 Predict/Transfer 会分片；reference 汇总、manifest
校验和 reward 训练仍由主节点执行。

数据完成后训练 reward：

```bash
bash mi_reward/scripts/run_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml
```

训练后先运行独立 `joint_heldout` 测试，再运行 RBM-EVAL：

```bash
bash mi_reward/scripts/eval_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml
```

## 论文表述边界

当前可以声称的是“严格 held-out split 上的离线方向奖励学习与排序”。不能把
`kinematic_task_proxy` 写成学习到的真实动力学，也不能在尚未完成真机实验时声称真机
强化效果。论文 ablation 至少应包含 visual-only、去 action、去 kinematic、去 relation、
去 outcome anchor 与 full teacher。
