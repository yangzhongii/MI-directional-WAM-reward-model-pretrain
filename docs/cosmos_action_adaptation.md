# Cosmos 动作条件世界模型适配方案

## 1. 边界与目的

本项目借鉴 WorldSample 得到一个工程判断：未经目标机器人、相机和任务域适配的通用世界模型，
很难产生可信的接触与动作后果。但本项目不复现 WorldSample，也不采用它的 PPL 作为方法主体。

本项目自己的闭环是：

```text
真实/仿真轨迹与动作
  -> 独立的 Cosmos action-conditioned LoRA 适配
  -> LaWAM/规划器提出多条动作候选
  -> 适配后的 Cosmos 预测每条候选的视觉未来
  -> visual + inferred-action + kinematic + relation directional MI
  -> reward/value model 排序与三状态奖励
  -> LIBERO RLPD 闭环验证
```

代码边界如下：

- `world_model/`：数据转换、Cosmos LoRA recipe、训练与视频保真度验证；
- `mi_reward/`：消费候选轨迹，提取 latent，计算 directional MI，训练 reward/value；
- `mi_reward/closed_loop/`：用冻结 reward/value 做下游 LIBERO RLPD 验证。

Cosmos 不提出动作，也不负责决定动作好坏。WorldSample 的局部动作扰动和 PPL 不属于当前实现。

## 2. 为什么不能直接使用原始 Cosmos

原始 action-conditioned 2B 权重主要来自 Bridge 风格数据。LIBERO 的机械臂外观、桌面、视角、
动作尺度和接触分布均不同。未适配模型生成的物体形状漂移、机械臂运动不跟随动作，不应交给
MI reward 排序，因为输入本身已经失真。

训练采用项目自有的配置和数据契约，但复用 NVIDIA Cosmos 的模型实现、FSDP 与 diffusion
trainer。重新抄写整个 Cosmos trainer 会增加数值不一致和 checkpoint 不兼容风险，不构成论文贡献。

## 3. 数据阶段

先用 LIBERO 官方成功 demonstrations 做第一阶段 domain adaptation；将来加入失败 rollout 和
online rollout 时，继续沿用同一 annotation contract，并保持 train/val/test episode 隔离。

```bash
bash world_model/scripts/prepare_libero_action_data.sh \
  --suite libero_spatial \
  --task-id 0 \
  --output logs/world_model/libero_spatial/task-00/action_dataset \
  --empty-text-embedding \
    .venv/models/cosmos-predict2.5/robot/action-cond/cr1_empty_string_text_embeddings.pt
```

导出内容包括：

- 第三视角 `agentview` 与腕部视角；
- 末端位置与 axis-angle 到 xyz Euler 的确定性转换；
- `0=open, 1=closed` 的 gripper state；
- 固定 seed 的 train/val/test episode 划分；
- MP4 与 Cosmos Bridge action loader 所需 JSON annotation；
- 所有 episode 共享的官方 empty-string Reason1 embedding 软链接，不加载 7B 文本编码器；
- LIBERO 20 Hz 到公开 action recipe 4 fps 的采样比例。

## 4. LoRA 训练

配置位于 `world_model/configs/cosmos_action_lora_2b.yaml`。

预检：

```bash
bash world_model/scripts/train_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_lora_2b.yaml \
  --action preflight
```

先做单步显存与资产测试：

```bash
bash world_model/scripts/train_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_lora_2b.yaml \
  --action run \
  --max-iter 1 \
  --save-iter 1
```

通过后正式训练：

```bash
bash world_model/scripts/train_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_lora_2b.yaml \
  --action run
```

训练同时显式检查 action-conditioned checkpoint 和本地 Wan2.1 tokenizer，避免运行时再次访问
受限仓库。LoRA 输出写入 `logs/world_model/cosmos_action_lora_2b/`。

## 5. 两机模式

两节点 FSDP 的入口保留为一个参数：

```bash
bash world_model/scripts/train_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_lora_2b.yaml \
  --action run \
  --distributed
```

两台机器各运行一个 rank；它们不是一台机器上的 `cuda:0` 和 `cuda:1`。两边必须有相同源码、
Python 环境、数据、action checkpoint 和 tokenizer。普通以太网下 FSDP 通信可能比单卡更慢，
所以只有单卡 OOM 时才用跨机切分；能单卡运行时，两台机器更适合并行不同 task/seed。启动器会让
两边的 NCCL/Gloo 自动绑定物理网卡；任一 rank 失败或收到 Ctrl-C 时会终止两个进程组。训练成功后，
还会通过 rsync 把远端 rank-1 DCP 分片收回本机输出目录，因为两台工作站没有共享文件系统。

## 6. 验证门槛

LoRA 完成不等于可用于 MI。训练结束后先在训练节点导出最终 checkpoint。预检不会写大文件：

```bash
bash world_model/scripts/export_cosmos_action_lora.sh \
  --training-output logs/world_model/cosmos_action_lora_2b_formal \
  --iteration 1000 \
  --action preflight
```

预检通过后执行 NVIDIA 官方 DCP 转换，并默认删除巨大的 `model.pt` 与 FP32 中间文件，只保留
推理需要的 EMA BF16 checkpoint 和 manifest：

```bash
bash world_model/scripts/export_cosmos_action_lora.sh \
  --training-output logs/world_model/cosmos_action_lora_2b_formal \
  --iteration 1000 \
  --action export
```

随后在独立 test episode 上比较 base 与 adapted checkpoint：

```bash
bash world_model/scripts/validate_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_validation.yaml \
  --action preflight

bash world_model/scripts/validate_cosmos_action_lora.sh \
  --config world_model/configs/cosmos_action_validation.yaml \
  --action run
```

验证输出位于 `logs/world_model/cosmos_action_lora_2b_formal/validation/`。每条 held-out episode
包含 `reference | base | adapted` 横向视频，并报告 PSNR、SSIM、MAE、endpoint pixel-change cosine
和 motion-support IoU。base 与 adapted 使用相同首帧、动作和随机种子；推理直接读取官方 empty
Reason1 embedding，不在线加载 7B 文本编码器。

通过标准：

1. 首帧和相机视角保持一致；
2. 预测运动方向与动作方向一致；
3. PSNR、SSIM、LPIPS 优于 base；
4. 机械臂与目标物不出现明显形变、瞬移或穿透；
5. 相同初态下，由规划器提出的不同动作产生可区分的未来；
6. 通过人工视频审查后，才接入 directional MI 与闭环实验。

只有 `validation_report.json` 中 `quality_gate.passed=true` 且人工视频审查通过后，才创建新的
`generalization_rigid_v4`。现有 v3 必须保留为 base-Cosmos/physical-reference 基线，不能覆盖。

MI 的 Cosmos worker 已支持：

```bash
--adapted-checkpoint \
  logs/world_model/cosmos_action_lora_2b_formal/exports/iter_000001000/model_ema_bf16.pt
```

传入该参数时，worker 会构建与训练一致的 LoRA rank/alpha/target modules，并在每条输出记录中写入
`cosmos_checkpoint` 与 `cosmos_adapted=true`；不传时仍保持 base checkpoint 路径。

质量门控通过后，用 promotion 工具生成隔离的 v4 data/reward 配置。工具会在门控失败时直接拒绝，
不会修改 v3：

```bash
bash world_model/scripts/promote_cosmos_action_lora.sh \
  --validation-report \
    logs/world_model/cosmos_action_lora_2b_formal/validation/validation_report.json \
  --checkpoint \
    logs/world_model/cosmos_action_lora_2b_formal/exports/iter_000001000/model_ema_bf16.pt
```

成功后产生：

- `mi_reward/configs/generalization_data_v4.yaml`：启用 adapted Cosmos predict，并将它的输出设为 v4 candidate records；
- `mi_reward/configs/generalization_reward_v4.yaml`：所有 manifest、feature、preference、checkpoint 和结果路径隔离到 v4。

## 7. 14B 与两张 48GB 4090

两张 48GB 卡对 2B action-conditioned LoRA 足够，单卡也大概率可完成 256×320、13 帧、batch 1。

14B LoRA 的关键问题不只是总显存。LoRA 只减少可训练参数和优化器状态，不消除约 28 GB 的
BF16 基座权重、视频 diffusion activation、VAE、通信 buffer 和 checkpoint 峰值。两张 48GB
跨机器使用 FSDP 加 activation checkpoint/offload，理论上可尝试 generic 14B LoRA，但速度慢且
余量有限。更重要的是，当前项目使用的公开 robot/action-conditioned checkpoint 与 recipe 是 2B；
generic 14B 没有等价的机器人 action-conditioning 权重，不能只改 `size: 14B`。

因此当前正式路线是 2B action-conditioned LoRA。只有获得或自行训练 14B action adapter，并先
验证 action-following 后，14B 才进入实验矩阵。

## 8. 已执行的硬件冒烟结果

2026-09-01 已在 `zhonghaoyang@172.16.0.3` 的单张 48GB 4090 上完成真实 one-step：

- train/val/test episode：34/8/8；
- action windows：train 1464，val 240；
- iteration 1 loss：0.1381；
- 含首次加载、forward、backward 和落盘约 74.38 秒；
- checkpoint：`iter_000000001`，模型分片约 13.0GB，优化器分片约 195MB；
- 训练正常输出 `Done with training`，退出后 GPU 回到约 1GB 桌面占用；
- 观测到的模型加载显存约 18.8GB，实际训练期间未 OOM。

该 smoke 只证明配置、数据、反向传播和 checkpoint 链路可运行，不代表模型已经适配完成，
也不能替代 held-out 视频指标与人工物理一致性检查。

参考：

- [WorldSample](https://arxiv.org/abs/2607.02431)：仅作为目标域 world-model adaptation 的动机；
- [Cosmos action-conditioned post-training](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md)；
- [Cosmos LoRA post-training](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_cosmos_nemo_assets_lora.md)。
