# LIBERO 闭环：视觉 RLPD + 冻结 MI Value/Reward

## 1. 当前推荐闭环

论文主实验使用 `libero_rlpd_closed_loop.yaml`，不是早期的纯 MLP SAC。它借鉴标准
RLPD-CNN 配置，但不依赖 RLinf，也不使用 Qwen VLM：

```text
第三视角 RGB + 腕部 RGB + 15D robot state
  -> ImageNet ResNet18 CNN policy encoder
  -> Gaussian actor + 10-head Q ensemble
  -> online replay 与官方成功 demo replay 各采样 50%
  -> LIBERO env.step(action)
  -> 冻结 LaWAM/DINO encoder + VisualGoalPotential
  -> r_MI = gamma * Phi(o_t+1, g) - Phi(o_t, g)
  -> sparse / MI / sparse+MI 三组 RLPD 更新
```

DINO/LaWAM 提供冻结的视觉 token，`VisualGoalPotential` 才是打分模型；它们替代参考
配置中的 Qwen VLM reward。策略本身使用可训练 CNN，避免旧版把 mean-pooled DINO token
直接送入 MLP、且没有 demo replay 的低样本效率问题。

旧的 `libero_closed_loop.yaml` 与 `run_libero_closed_loop*.sh` 仍保留，作为低容量 SAC
baseline，不再作为主闭环。

## 2. 安装

```bash
bash requirements/install.sh --libero-closed-loop \
  --download-libero-data \
  --libero-suite libero_spatial \
  --use-mirrors
```

安装器会准备 `.venv-libero`、官方 LIBERO demo 和 ResNet18 policy 权重。若
`.venv/models` 中缺 DINOv3/LaWAM，再追加 `--download-weights`。默认需要：

```text
.venv/models/resnet18-imagenet/resnet18-f37072fd.pth
.venv/models/dinov3-vitb16-pretrain-lvd1689m/
.venv/models/lawam_lam/
logs/mi_reward/generalization_rigid_v3/results/reward_model/pytorch_model.pt
```

## 3. 预检与 smoke test

先检查双相机、15D 状态、官方 demo、ResNet、DINO/LaWAM、MI checkpoint 和初态隔离：

```bash
bash mi_reward/scripts/run_libero_rlpd.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action preflight \
  --task-id 0 \
  --reward-mode sparse_mi \
  --seed 0
```

再运行 6 个环境步的端到端 smoke。它只验证 replay、reward、actor/critic update、评估和
checkpoint，不用于报告成功率：

```bash
bash mi_reward/scripts/run_libero_rlpd.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action train \
  --task-id 0 \
  --reward-mode sparse_mi \
  --seed 0 \
  --smoke
```

## 4. 先跑 pilot，再跑完整矩阵

先用 task 0、seed 0 比较三种 reward：

```bash
bash mi_reward/scripts/run_libero_rlpd_matrix.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action run \
  --task-id 0 \
  --reward-mode sparse \
  --reward-mode mi \
  --reward-mode sparse_mi \
  --seed 0 \
  --skip-completed
```

确认至少 `sparse_mi` 能学习、曲线和视频正常后，再跑配置中的
3 tasks × 3 reward modes × 3 seeds：

```bash
bash mi_reward/scripts/run_libero_rlpd_matrix.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action run \
  --skip-completed
```

当前 RLPD run 不持久化 replay buffer，也不支持从半途 checkpoint 恢复。已完成 run 可用
`--skip-completed` 跳过；中断的正式 run 应换 `run_id` 或先归档对应目录。

## 5. 两台机器并行

两台 4090 按完整实验 job 分片，不合并显存，也不跨网线逐步同步梯度：

```bash
bash mi_reward/scripts/run_libero_rlpd_matrix.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action run \
  --skip-completed \
  --distributed
```

入口会同步源码、MI reward checkpoint 和 ResNet18 checkpoint，把奇偶 job 分给本机与
`zhonghaoyang@172.16.0.3`，结束后回收远端结果。远端仍需提前安装 `.venv-libero`，并
具备 DINO/LaWAM 权重及相同的 LIBERO demo。只有确认这些内容和源码都已同步时才加
`--no-sync-remote`。

SSH 使用 10 秒连接超时和 keepalive；若 `.3:22` 不可达，入口会明确失败，而不会无限等待。
当本机已经有长任务、只想占用 `.3` 时，在上述命令追加 `--remote-only`；所有选中的
jobs 都会在远端顺序执行并在完成后自动回收。

## 6. 输出与汇总

```text
logs/mi_reward/libero_rlpd_closed_loop/libero_mi_rlpd_v1/
  task-XX/{sparse,mi,sparse_mi}/seed-X/
    resolved_config.yaml
    train_metrics.jsonl
    evaluation.jsonl
    checkpoints/latest.pt
    videos/step-*/{id_heldout,layout_heldout,appearance_heldout}/
    run_report.json
```

```bash
bash mi_reward/scripts/run_libero_rlpd_matrix.sh \
  --config mi_reward/configs/libero_rlpd_closed_loop.yaml \
  --action summarize
```

论文比较 success-rate curve/AUC、达到固定成功率所需环境步数、最终成功率，而不是只看
shaped return。`sparse` 与 `sparse_mi` 是主对照，`mi` 用于检查 reward 是否可独立驱动。

## 7. 泛化与安全边界

`id_heldout` 和 `layout_heldout` 是未参与训练的官方初态/物体位姿；
`appearance_heldout` 是确定性相机外观扰动。它们不能冒充超市、仓库等 semantic scene
generalization，后者仍需 LIBERO-PRO、自定义资产或真机实验。

MI reward 仅作 shaping。环境 success、termination、碰撞和安全约束始终是权威信号。
真机接口继续保留，但默认闭环不启动真机，也不依赖 RLinf。
