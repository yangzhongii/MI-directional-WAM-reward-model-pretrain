# Cosmos 14B 双机推理改造方案

> **归档说明（2026-08-29）**：当前 benchmark 路线不使用 Cosmos 14B，默认使用
> Cosmos Predict2.5 2B action-conditioned，并通过 visual/action/relation privileged
> teacher 蒸馏视觉 student。本文仅保留为历史调研，不是可执行运行文档。当前命令以
> [`offline_benchmark_runbook.md`](offline_benchmark_runbook.md) 为准。

## 1. 目标

将两台局域网机器上的改装 48 GB RTX 4090 组成一个 Cosmos 14B 推理资源，用于本项目的离线数据生成：

```text
Robometer / RLinf base records
    -> SAM3
    -> MuJoCo physical verification
    -> Cosmos Predict2.5 14B action-conditioned rollout
    -> Cosmos Transfer2.5 scene variation
    -> MI directional preferences
    -> reward-model SFT
```

本方案的目标是让两张 GPU 协同完成 Cosmos 14B 推理，而不是把两张 GPU 当成两个独立的 2B 推理任务。

## 2. 技术结论

### 不使用 DDP 解决 Cosmos 14B 显存问题

DDP 会在每张 GPU 上加载完整模型：

```text
GPU 0: 完整 Cosmos 14B
GPU 1: 完整 Cosmos 14B
```

因此 DDP 不会把 48 GB + 48 GB 合并成一个 96 GB 显存空间，不能解决单卡加载 14B 失败的问题。

DDP 只适合：

- Reward Model SFT 的多 GPU 训练；
- 两张 GPU 分别处理不同样本的任务并行。

### 不引入 Ray 或 SGLang

Ray 适合任务调度，SGLang 适合大模型服务和批量请求推理，但二者都不是当前 Cosmos 14B 显存切分的直接解决方案。

本项目的优先路线是：

```text
Cosmos 原生 Context Parallelism
    -> 两机 NCCL
    -> 如果仍然 OOM，再增加模型切分、量化或 CPU offload
```

## 3. 当前代码现状

### 当前硬阻塞：官方 action-conditioned 版本没有可直接使用的 14B 多 GPU方案

对当前 `.venv/src/cosmos-predict2.5` 和项目配置进行审计后发现：

- 项目默认下载的是 `nvidia/Cosmos-Predict2.5-2B`；
- action-conditioned experiment 目录目前只有 2B 配置；
- 源码的 action-conditioned `net.py` 虽然定义了 `COSMOS_V1_14B_NET_MININET`，但这不等于存在可用的 14B robot/action-conditioned checkpoint；
- 官方 Robot Action-Conditioned Inference 文档明确说明 action-conditioned inference 当前不支持 multi-GPU；
- 官方 14B checkpoint 属于普通 pre-trained/post-trained Video2World，并不是当前 pipeline 要求的 `robot/action-cond` checkpoint；
- 当前项目要求 Cosmos Predict 接收 MuJoCo 产生的 action sidecar，并输出与物理轨迹对齐的视频；
- 普通 Cosmos 14B Video2World checkpoint 不能直接替代 action-conditioned 2B checkpoint。

因此，在获得官方支持的 14B robot/action-conditioned checkpoint、experiment 和多 GPU实现之前，不能只把配置中的 `2B` 改成 `14B`。普通 14B Video2World 可以单独增加一条 visual-only 推理路径，但不能直接生成当前项目所需的 action-conditioned physical rollout，否则会破坏 action/state 与视频的语义对应关系。

本项目当前安装的 action-conditioned experiment 文件可以用以下命令核对：

```bash
find .venv/src/cosmos-predict2.5/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

下一步代码改造的前置条件是二选一：

1. 提供官方发布的 14B robot/action-conditioned checkpoint、experiment 配置和对应 Cosmos commit，并确认其支持多 GPU；
2. 明确把项目改为普通 14B Video2World visual-only rollout，再自行设计 action conditioning 映射。该方案不再等价于当前项目的物理动作条件 rollout，默认不采用。

### 已有能力

`mi_reward/data/cosmos_predict_worker.py` 已经有：

```python
context_parallel_size
```

并将其传入：

```python
ActionVideo2WorldInference(
    context_parallel_size=context_parallel_size
)
```

这说明项目已经预留了 Cosmos Context Parallelism 接口。

### 当前缺口

当前实现仍存在以下问题：

1. 默认使用 Cosmos 2B action-conditioned checkpoint；
2. `generalization_data.yaml` 没有暴露 Predict 阶段的 14B 配置；
3. Predict worker 的 `context_parallel_size` 没有从流水线配置传入；
4. worker 当前由流水线作为单个普通 subprocess 启动，不是两机 `torchrun`；
5. 当前 `torchrun --nproc_per_node` 逻辑主要用于本机 Cosmos Transfer；
6. 没有统一的 `MASTER_ADDR`、`MASTER_PORT`、`NNODES`、`NODE_RANK` 配置；
7. 当前工作目录、checkpoint 路径和中间产物路径没有双机一致性约定。

相关文件：

```text
mi_reward/data/cosmos_predict_worker.py
mi_reward/data/cosmos_transfer_worker.py
mi_reward/data/generalization_pipeline.py
mi_reward/data/instance_orchestrator.py
mi_reward/configs/generalization_data.yaml
mi_reward/scripts/generate_generalization_data.sh
```

## 4. 目标架构

> 当前官方 action-conditioned 路径不支持 multi-GPU，因此下面的双机 Context Parallelism 只能在获得兼容的 14B action-conditioned 实现后执行。它不是当前 pinned 版本可以直接启动的命令。

推荐以 `172.16.0.6` 作为主节点，以 `172.16.0.3` 作为从节点：

```text
主节点 172.16.0.6
  rank 0
  MASTER_ADDR=172.16.0.6
  MASTER_PORT=29500
  NCCL_SOCKET_IFNAME=enp131s0

从节点 172.16.0.3
  rank 1
  通过 NCCL 连接主节点

两节点共同运行 Cosmos Predict2.5 14B
context_parallel_size=2
```

两台机器必须拥有：

- 相同的 Cosmos Predict2.5 源码 commit；
- 相同的 Python、PyTorch、CUDA/NCCL 依赖；
- 相同的 14B checkpoint；
- 可访问的输入 action/state/frame 文件；
- 可写的输出目录；
- 互通的 `29500/tcp` 端口。

## 5. 分阶段改造

### 阶段 A：基础环境和通信检查

两台机器分别检查：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

确认主节点地址：

```bash
hostname -I
```

配置有线网卡：

```bash
export NCCL_SOCKET_IFNAME=enp131s0
export GLOO_SOCKET_IFNAME=enp131s0
```

确认从节点可以访问主节点：

```bash
nc -vz 172.16.0.6 29500
```

测试时主节点需要先监听端口；正式启动 DDP/Cosmos 之前要停止临时 `nc` 进程。

### 阶段 B：准备可复现环境

两台机器使用同一个项目版本：

```bash
git rev-parse HEAD
```

两边输出必须一致。

两台机器使用相同 Cosmos 源码版本：

```bash
git -C .venv/src/cosmos-predict2.5 rev-parse HEAD
```

验证 checkpoint：

```bash
find .venv/models/cosmos-predict2.5 -type f -name '*14b*' -o -name '*14B*'
```

如果两台机器没有共享存储，当前 `transport: rsync` 会自动同步源码、JSONL shard 和
其引用的输入 artifact；大模型 checkpoint 仍应预先安装在各节点，并通过节点专用
Python/worker 配置定位。

### 阶段 C：增加 Cosmos Predict 14B 配置

在 `generalization_data.yaml` 中增加明确的 Predict 配置，例如：

```yaml
cosmos_predict:
  model: 14B/post-trained
  experiment: <Cosmos-14B action-conditioned experiment>
  context_parallel_size: 2
  nnodes: 2
  nproc_per_node: 1
  master_addr: 172.16.0.6
  master_port: 29500
  interface: enp131s0
```

执行 stage 的 command 需要显式传入：

```text
--model 14B/post-trained
--context-parallel-size 2
```

实际 `experiment` 名称和 checkpoint 目录必须以安装的 Cosmos Predict2.5 版本为准，不能直接沿用当前 2B experiment 名称。

### 阶段 D：修改 Predict worker

`cosmos_predict_worker.py` 需要支持以下参数：

```text
--model
--context-parallel-size
--nnodes
--node-rank
--nproc-per-node
--master-addr
--master-port
--nccl-interface
```

worker 本身不应在每个 rank 上重复写最终 JSONL。建议保持当前行为：

```text
rank 0：生成并写入 frames、video 和 result manifest
rank 1：只参与模型计算和同步
```

同时需要确认 Cosmos 内部 Context Parallelism 的初始化方式。如果 Cosmos inference API 要求由外层 `torchrun` 设置 distributed environment，则 worker 只负责读取 `RANK`、`WORLD_SIZE` 等环境变量；如果 Cosmos API 自己创建 process group，则不能重复初始化。

### 阶段 E：修改流水线启动方式

当前 `instance_orchestrator.py` 会将每个 stage 作为本地 subprocess 执行。多机 Predict 阶段需要增加一种启动模式：

```text
local-single
local-multi-gpu
multi-node
```

多机模式的目标命令形态为：

主节点：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=172.16.0.6 \
  --master_port=29500 \
  -m mi_reward.data.cosmos_predict_worker ...
```

从节点：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=172.16.0.6 \
  --master_port=29500 \
  -m mi_reward.data.cosmos_predict_worker ...
```

从节点的启动可以通过 SSH 触发，但 SSH 只负责启动进程，不负责传输 GPU 显存。两台机器都必须能看到相同的 request、checkpoint 和输入文件。

### 阶段 F：先做单条样本测试

不要一开始运行完整 Robometer 数据集。先准备一条已经通过 MuJoCo 的 physical record，只运行 Cosmos Predict：

```text
1 条输入记录
1 个 candidate
低分辨率
最短 action chunk
context_parallel_size=2
```

必须验证：

- 两台机器均有 Python/CUDA 进程；
- 两台机器的显存均被使用；
- rank 0 和 rank 1 都成功加入 process group；
- 没有 CUDA OOM；
- 没有 NCCL timeout；
- 只生成一份最终视频和 manifest；
- 输出帧数与 MuJoCo physical record 一致。

## 6. 显存策略

如果 Context Parallelism 仍然让每张 GPU 加载完整 14B，48 GB 可能仍然不足。此时按以下顺序处理：

1. 确认使用 BF16，而不是 FP32；
2. 降低生成分辨率和视频帧数；
3. 启用 Cosmos 支持的 CPU offload；
4. 使用官方或兼容的量化 checkpoint；
5. 改用真正的模型参数切分/模型并行；
6. 训练场景再考虑 FSDP/DeepSpeed + LoRA。

不能用 DDP 解决模型本体放不进单卡的问题。

## 7. 网络和性能风险

Context Parallelism 需要频繁同步。如果两台机器之间只是普通以太网，性能可能明显低于单机 NVLink 或 PCIe 多卡。

应记录：

```text
单样本耗时
GPU 利用率
网络吞吐
NCCL 等待时间
显存峰值
```

如果跨机 Context Parallelism 太慢，离线数据生成可以退回任务分片：

```text
172.16.0.6：处理 records 0 到 N
172.16.0.3：处理 records N 到 M
```

这会增加吞吐，但不会让两张 GPU 合作生成同一个视频。它是性能兜底方案，不是 14B 显存切分方案。

## 8. Reward Model 训练阶段

Cosmos 数据生成成功后，Reward Model SFT 可以单独使用 Accelerate/DDP：

```text
Cosmos Predict：Context Parallelism / 模型并行
Cosmos Transfer：本机多 GPU 或任务分片
MI preference 构建：CPU/本地缓存
Reward Model SFT：Accelerate + DDP
```

Cosmos 14B 的推理并行和 Reward Model 的 DDP 不应混用为同一个 process group。

## 9. 验收标准

改造完成后，以下命令应能在两台机器上协同运行一个最小样本：

```bash
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=172.16.0.6 \
MASTER_PORT=29500 \
NCCL_SOCKET_IFNAME=enp131s0 \
bash <predict-multinode-launcher>
```

远程节点使用 `NODE_RANK=1`。

验收结果：

- 两个 rank 正常初始化；
- 两张 GPU 显存均明显增加；
- Cosmos 14B 成功完成一条 action-conditioned rollout；
- 生成的视频和 frame manifest 只写一份；
- 输出可以被后续 Cosmos Transfer 和 manifest validator 读取；
- 完整流水线不需要 Ray 或 SGLang；
- 如果 14B 仍然 OOM，错误日志能明确指出是模型加载、激活还是通信阶段。

## 10. 最终推荐

本项目采用以下组合：

```text
Cosmos Predict2.5 2B robot/action-cond：单 GPU 或任务分片
Cosmos Transfer2.5 2B：本机多 GPU或任务分片
MI Reward Model：Accelerate + PyTorch DDP
任务调度：先用 SSH/脚本，暂不引入 Ray
推理服务：暂不引入 SGLang
```

当前实现采用任务/样本分片，而不是两台 GPU 协同生成同一个视频。配置入口是
`mi_reward/configs/generalization_data.yaml` 的 `execution.distributed`，默认关闭；启用后，
主节点会通过 SSH 并发启动两个节点的 planner、simulator 以及手工启用的
Predict/Transfer worker，并在主节点合并 JSONL。reference 和最终验证不分片。

默认 `transport: rsync` 允许不同项目路径和 Python 环境；每个节点通过 `repo_root`
与 `python` 声明本机位置。Predict/Transfer 的大模型 checkpoint 仍须在相应节点预先
安装。运行命令仍然是：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --distributed
```

如果未来获得官方支持的 14B robot/action-conditioned checkpoint 和多 GPU实现，再单独扩展 14B 模式；普通 14B Video2World 不进入当前物理动作条件 pipeline。
