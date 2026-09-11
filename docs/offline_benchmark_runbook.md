# 离线 Benchmark 完整运行手册

本文档对应当前默认路线：三类 simulator-native MuJoCo 数据、离线 privileged MI
teacher、RGB/optional-goal student、RBM-EVAL。Cosmos Predict/Transfer 与 Robometer
数据生成只保留为可选 ablation；当前默认路线不需要 Cosmos、Ray、双机、PPO 或真机采集。

## 1. 安装

推荐在仓库根目录用一条命令完成环境、可选权重、评测数据和仿真资产安装：

```bash
bash requirements/install.sh --all \
  --download-weights \
  --download-eval-data \
  --download-sim-assets \
  --use-mirrors
```

数据生成 v3 默认只使用 MuJoCo，因此即使已经安装了 Cosmos，它也不会在默认 stage
中被加载。该命令不下载 Cosmos Transfer。安装脚本是幂等的：重复执行会
复用已经完成的文件，Hugging Face 中断下载会从本地缓存继续。

如果希望分阶段安装，执行：

```bash
bash requirements/install.sh --generalization-data --download-weights --use-mirrors
bash requirements/install.sh --reward-train --download-weights --use-mirrors
bash requirements/install.sh --reward-eval --download-eval-data --use-mirrors
```

三个 Python 环境互不覆盖：

- `.venv`：SAM2.1、Cosmos Predict、MuJoCo；
- `.venv-reward`：DINOv3、LaWAM LAM、reward 训练；
- `.venv-eval`：Robometer/RBM-EVAL。

源码、权重和数据仍统一保存在 `.venv/src`、`.venv/models`、`.venv/datasets`，
所以拆分环境不会重复下载大模型。各 target 的 `--download-weights` 分别下载：

- `--generalization-data`：SAM2.1、Cosmos Predict2.5 2B action-conditioned；
- `--reward-train`：DINOv3、LaWAM LAM。

如果 `.venv/models` 和 `.venv/datasets` 已经下载完成，只迁移 Python 环境即可：

```bash
bash requirements/install.sh --generalization-data --use-mirrors
bash requirements/install.sh --reward-train --use-mirrors
bash requirements/install.sh --reward-eval --use-mirrors
```

默认不下载可选的 Cosmos Transfer 权重。如需 Transfer：

```bash
bash requirements/install.sh --generalization-data \
  --download-transfer-weights --use-mirrors
```

Hugging Face 下载中断后直接重复同一条命令，已有 `.incomplete` 缓存会续传。
不要删除完整 checkpoint。

大陆网络环境下，`--use-mirrors` 负责 Git/Python 包和非 gated 资源的镜像选择。
如果仍需本机代理，应使用终端中的纯文本 URL（不要带 Markdown 方括号）：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7898
```

需要授权的 Hugging Face 模型仍需先在网页接受协议并登录：

```bash
.venv/bin/hf auth login --force
.venv/bin/hf auth whoami
```

`403 GatedRepoError` 表示账号没有模型权限，重试或切换镜像不能解决；
`SSL UNEXPECTED_EOF`、连接被提前关闭通常是代理链路中断，保持缓存并重复同一条
安装命令即可。

## 2. 下载并生成 MuJoCo 资产

推荐直接运行资产下载器。它只下载 MuJoCo Menagerie 的 Franka Panda
子目录，并用 MuJoCo primitive 生成任务物体，不会拉取整个 LIBERO 或
robosuite：

```bash
bash requirements/download_mujoco_assets.sh --use-mirrors
```

也可以在安装数据生成环境时一起执行：

```bash
bash requirements/install.sh \
  --generalization-data \
  --download-sim-assets \
  --use-mirrors
```

生成路径为：

```text
assets/custom_task/scene/scene.xml
assets/custom_task/scene/scene_pick_apple.xml
assets/custom_task/scene/scene_banana.xml
assets/custom_task/scene/scene_push_t.xml
assets/custom_task/scene/scene_push_b.xml
assets/custom_task/scene/scene_peg_round.xml
assets/custom_task/scene/scene_peg_square.xml
```

`scene.xml` 是兼容旧配置的 apple 场景别名。三个任务 YAML 现在分别指向
对应的训练场景和 held-out 场景。生成器会用本机 MuJoCo 编译全部 XML，并
检查 camera、body、geom、actuator 和 keyframe 名称。

## 3. 环境变量

```bash
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

如果本机通过代理连接 Hugging Face，仅在下载阶段设置代理；数据生成和训练本身
不需要 Hugging Face 网络连接。

## 4. 配置检查

轻量检查只验证 YAML 和 stage hand-off：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --dry-run
```

正式运行前必须执行严格检查：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --preflight
```

`--preflight` 只检查当前启用 stage 依赖的内容：

- `.venv` 数据生成 Python 和 `.venv-reward` 奖励训练 Python；
- MuJoCo scene 和 held-out instance XML；
- 每个 instance × scene variant 对应的 MuJoCo XML；
- 奖励阶段使用的 LAM、DINO 路径；
- 仅当手工启用 Predict/Transfer 时，才检查对应 checkpoint 和 GPU。

只有输出 `"status": "ready"` 后再运行数据生成。

## 5. 数据生成

默认启用的顺序为：

```text
simulator-native seed -> planner -> MuJoCo -> reference -> strict verification
```

最终候选文件是：

```text
logs/mi_reward/generalization_rigid_v3/data/generalization_rollouts/physical_candidates.jsonl
```

默认每类任务生成 5 个训练 seed，另加 1 个专用 reference seed。planner 对
`2 instance × 2 native scene × 8 candidate profile` 做笛卡尔积，因此三类任务共
生成 `3 × 6 × 2 × 2 × 8 = 576` 条物理轨迹。reference 阶段为每个
`task family × instance` 选出一条独立成功轨迹，并将三个 reference parent seed
的 96 条近重复轨迹排除，最终得到 480 条候选和 6 条 success reference。
四个严格 split 分别为 `train`、`instance_heldout`、`scene_heldout`、
`joint_heldout`，各 120 条；`train` 用于优化，instance/scene held-out 用于选模，
`joint_heldout` 只允许进入训练完成后的独立最终测试。
两个 scene 分别使用浅色实验台和深色工业台 XML，物理几何与相机保持一致。
RGB 默认以 320×256 写入；LaWAM 会在特征提取时统一 resize，因此没有必要保存
640×480 的重复 PNG。默认验收门限为至少 360 条 artifact accepted 且接受率不低于 75%。
`logs/mi_reward/generalization_rigid_v3/results/data_report.json` 会同时按 task family
和 scene 输出 total/accepted/rejected；任何任务没有 accepted 候选都会失败。

本次运行产生的所有中间数据、视频、特征、报告、checkpoint 和评测结果统一写入
`logs/mi_reward/generalization_rigid_v3/`。`.venv` 只保存安装的源码、权重和下载数据；
`assets/custom_task` 只保存可复用的 MuJoCo scene，不再把一次运行的产物散落到
`dataset/` 和 `results/`。

MuJoCo 同时是默认流程的物理真值与视觉真值。Cosmos 2B 的现有审计结果未通过
endpoint/motion consistency gate，因此 Predict 默认关闭，不再让幻化机械臂、静止物体
或跨任务画面进入 MI 训练。以后如做 Cosmos ablation，必须单独启用 stage 并重新通过
严格视觉一致性门限。

相机固定为完整桌面第三视角；Panda 全臂只参与 MuJoCo 物理，不进入 RGB，画面仅显示
桌面、任务物体、目标和运动夹爪。这样既避免完整机械臂遮挡桌面，也避免静态 Panda
与独立 proxy gripper 同时出现导致 Cosmos 幻觉。

刚体模板启用了显式 `kinematic_task_proxy`：MuJoCo 检查夹爪到达/接触后，任务
物体在 close 到 release 期间跟随 Cartesian proxy，从而稳定生成可审计的离线
候选。八种 profile 包含三种成功路径和五种可测 near-miss。`verification` 只判断
artifact 是否可信，碰撞、接触和终点结果单独写入 `task_outcome`；因此物理失败会
进入 teacher，损坏或错位文件才会被拒绝。该标记会写入 metadata；它是数据生成
控制代理，不应在论文中表述为学习到的机械臂动力学。若 accepted 数量、比例或 split
覆盖不满足配置门限，生成命令
现在会直接失败，而不会把 `0 accepted` 误报为完成。

Robometer 的 USC Koch 文本（移动杯子、把物体扔进垃圾桶）与当前 apple/banana、
PushT/PushB、peg-insertion 仿真任务并不等价，所以它不再作为默认 base trajectory。
相关 ingest 代码仍保留，供 RBM-EVAL 或显式 ablation 使用。

执行：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml
```

默认配置的一次完整验收应检查 `data_report.json`，不能只看命令是否退出：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("logs/mi_reward/generalization_rigid_v3/results/data_report.json")
report = json.loads(path.read_text(encoding="utf-8"))
print("total:", report["total_candidates"])
print("accepted:", report["accepted_candidates"])
print("rejected:", report["rejected_candidates"])
print("acceptance_rate:", report["acceptance_rate"])
print("scenes:", report["scene_variants"])
print("task families:", report["task_family_counts"])
assert report["accepted_candidates"] > 0
assert all(v["accepted"] > 0 for v in report["task_family_counts"].values())
PY
```

旧版 `generalization_rigid_v2` 把 Robometer 任务文字、custom MuJoCo 任务和失败的
Cosmos 画面混在一起，不能作为当前实验结果。必须重新执行 v3 数据生成，并以
`logs/mi_reward/generalization_rigid_v3/results/data_report.json` 为准。

完成后重点检查：

```bash
python -m mi_reward.data.generalization_pipeline \
  --config mi_reward/configs/generalization_reward.yaml \
  --validate-training-manifest \
  logs/mi_reward/generalization_rigid_v3/data/manifests/generalization_rigid_v3.jsonl \
  --success-refs \
  logs/mi_reward/generalization_rigid_v3/data/manifests/generalization_rigid_v3_success_refs.jsonl
```

如要启用 Transfer：

1. 将 `execution.stages` 中 transfer 的 `enabled` 改成 `true`；
2. 将 `paths.candidate_records` 改成
   `logs/mi_reward/generalization_rigid_v3/data/generalization_rollouts/rigid_v3_records.jsonl`；
3. 同步修改 `generalization_reward.yaml` 的 `candidate_records`；
4. 确认每个 worker 的 `--num-gpus` 与单台节点实际 GPU 数一致；
5. 重新执行 `--preflight`。

当前双机实现是样本切分：两台机器各自加载完整 2B 模型并处理不同 JSONL shard，
它不是把两张显存合并成一个 96 GB 显存池。默认配置使用 `transport: rsync`：主节点
在 stage 启动前同步代码、当前 shard 引用的计划/帧/动作/状态，远端完成后自动回收
输出并把 `/mnt/public/...` 路径改写回主节点路径。因此两边不需要相同绝对路径，也
不需要 NFS。已有共享盘时仍可将 transport 改为 `shared`。

默认仍是单机。双机只需给同一条命令追加 `--distributed`；planner、MuJoCo
simulator 以及手工启用的 Predict/Transfer 会被切为两个 shard，reference 汇总和
manifest 验证仍在主节点执行：

```bash
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml \
  --distributed
```

先执行 `--dry-run --distributed`，再执行 `--preflight --distributed`。每个 node 的
`repo_root` 和 `python` 可以不同，但必须是该机器上的绝对路径；preflight 会验证
非交互 SSH、rsync、源码和 Python。正式全量生成前可运行下面的小规模 planner +
MuJoCo 双机测试：

```bash
.venv/bin/python scripts/dist_pipeline_smoke.py
```

当前阶段不启动真机；通用 real-robot JSONL bridge、
`base_data.source: jsonl` 和 `real_world.enabled: false` 接口均保留。

## 6. Reward 训练

启动脚本会自动使用 `.venv-reward`，不要在 `.venv` 中手工升级
`transformers`：Cosmos 固定使用 4.51.3，而 DINOv3 使用新版本。

```bash
bash mi_reward/scripts/run_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml
```

脚本依次执行：

1. 验证 artifact-accepted candidate；
2. 缓存 pooled visual feature；
3. 缓存 `[T,K,D]` visual tokens；
4. 通过 LAM 缓存 `[T-1,Q,D]` action latents；
5. 从同步 robot/object state 缓存 `[T,1,D]` kinematic latent；
6. 将 visual/action/kinematic 的单调对齐阶段和 phase-aware relation 映射到 `[0,1]`；
7. 只在相同 parent 初态、instance、scene 和 split 内构造细粒度偏好；
8. 只用 `train` 优化，并用 instance/scene held-out 选择 checkpoint；
9. 蒸馏 `VisualGoalPotential`。

directional process latent 链路是：DINO visual tokens `[T,K,D]`、LaWAM
action latents `[T-1,Q,D]` 与实测 kinematic latent `[T,1,D]` 分别计算沿成功
参考轨迹对齐的 MI potential，再和实测 relation potential `[T]`、成功/失败 outcome
anchor 加权得到排序 teacher。成功轨迹的 teacher curve 被投影为单调曲线并在终点锚定为
`1`；同一条落盘 curve 同时用于 preference score 和 student distillation，避免两套 teacher
定义发生漂移。student 学习逐帧势能，
部署 reward 为 `gamma*V(next)-V(current)`。打分记录同时保存 scene、instance
和 `visual/action/kinematic/relation phi`，用于确认链路不是只在配置中开启。

输出：

```text
logs/mi_reward/generalization_rigid_v3/data/features/
logs/mi_reward/generalization_rigid_v3/data/preferences/rigid_v3.jsonl
logs/mi_reward/generalization_rigid_v3/data/preferences/rigid_v3.teacher_targets.pt
logs/mi_reward/generalization_rigid_v3/results/reward_model/pytorch_model.pt
logs/mi_reward/generalization_rigid_v3/results/reward_model/train_config.yaml
logs/mi_reward/generalization_rigid_v3/results/reward_model/train_log.jsonl
logs/mi_reward/generalization_rigid_v3/results/reward_model/validation_metrics.json
```

训练完成后先运行内部独立测试，不要直接跳到 RBM-EVAL：

```bash
bash mi_reward/scripts/eval_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml
```

该命令只读取 `joint_heldout`，并以 MuJoCo `task_outcome` 和 measured relation
作为真值，输出 success/failure AUC、逐帧 progress correlation、按任务/失败模式的
统计、同初态 pair accuracy，以及 8 候选 Top-1/success@k。默认报告写入：

```text
logs/mi_reward/generalization_rigid_v3/results/reward_model/joint_heldout_eval.json
```

评估器会读取 checkpoint 中落盘的 `train_splits` 和 `validation_splits`。若
`joint_heldout` 曾参与优化或选模，命令会拒绝生成正式结果。仅调试旧 checkpoint 时可用
`--allow-split-overlap`，但输出中的 `split_isolation.valid` 会保持为 `false`，不得用于论文。

这次数据契约要求每组 candidate 共享同一个 initial object offset。若数据是在该契约加入前
生成的，必须先重新运行 `generate_generalization_data.sh`，再运行 reward 脚本；旧 feature
cache 的签名会失效并自动重算，不要手工拼接新旧 manifest。

特征缓存按 trajectory 独立写入。进程中断或某条样本报错后，不要删除整个
`logs/mi_reward/generalization_rigid_v3/data/features`；修复问题后重复执行同一条 reward 命令，已有的
pooled feature、visual token 和 action latent 会跳过，只补缺失项。阶段 2、阶段 3
和训练 epoch 都有进度条。

如果看到：

```text
output with shape [3,H,W] doesn't match the broadcast shape [1,3,H,W]
```

这是旧版单帧 CHW 张量调用 LaWAM batch normalization 的维度问题，当前代码已经
通过显式单帧 batch view 修复。直接重新运行 reward 命令即可，合法缓存会继续复用。

## 7. RBM-EVAL

默认配置已指向新的 student checkpoint：

```bash
bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml
```

只有内部 `joint_heldout` 报告中 `split_isolation.valid: true` 后，才执行本节。
RBM-EVAL 是外部 OOD 辅助指标，不替代内部 task outcome 和候选决策评估。

输出位于：

```text
logs/mi_reward/generalization_rigid_v3/results/rbm_eval/metrics.json
logs/mi_reward/generalization_rigid_v3/results/rbm_eval/reward_alignment/
logs/mi_reward/generalization_rigid_v3/results/rbm_eval/policy_ranking/
logs/mi_reward/generalization_rigid_v3/results/rbm_eval/quality_preference/
```

`visual_goal.goal_sidecar: null` 时使用训练得到的 null-goal token；如 benchmark
提供每个任务的独立成功目标图，可配置 goal sidecar。禁止直接把被评估轨迹的最后一帧
当作目标，这会造成未来信息泄漏。

## 8. Ablation

独立测试现在会在 `candidate_selection_baselines.methods` 中自动比较：

- random expected selection；
- terminal pixel-to-goal similarity；
- LaWAM/DINO visual-token cosine；
- visual directional MI；
- privileged process teacher；
- outcome-anchored privileged teacher；
- deployable student；
- measured-outcome oracle。

这些 baseline 直接写入现有 `joint_heldout_eval.json`，不需要重新训练。Pixel、token
和 visual MI 是可部署/非 outcome baseline；privileged teacher 使用离线 action/state/relation，
只能作为 upper bound。

训练 ablation 由统一脚本生成，所有实验共享原始 manifest 和 feature cache，但使用独立
preference、teacher target、checkpoint 和报告目录。先生成一个 seed 的配置检查：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action generate --seed 0
```

单个实验试跑：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --ablation visual_only --seed 0
```

完整三 seed 矩阵：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --seed 0 --seed 1 --seed 2 --skip-completed
```

默认命令始终是单机串行，不会连接其他主机。两台 4090 已准备好独立 reward 环境和
rigid-v3 cache 后，可显式追加 `--distributed`，将待运行的独立 config 交替分给本机与
SSH worker；这属于 experiment-level parallelism，不是 DDP，也不会合并显存：

```bash
# 只同步/校验远端，不启动正式实验。
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action prepare-remote \
  --remote-host zhonghaoyang@172.16.0.3

# 双机各自串行跑一半，远端结果自动拉回本机后统一汇总。
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run --seed 1 --seed 2 --skip-completed \
  --distributed \
  --remote-host zhonghaoyang@172.16.0.3
```

默认远端根目录是
`/mnt/public/zhonghaoyang/MI-directional-WAM-reward-model-pretrain`，可用
`--remote-root` 覆盖。第一次会同步源码和 `run_root/data`；后续数据确定未改变时可追加
`--no-sync-remote`，但仍会同步新生成的 YAML。远端准备器会把 manifest 中的绝对路径
改写为远端根目录，并重新签名 feature metadata，防止误判 cache miss 后重新提取 12GB
特征。不要在另一个单机 matrix 仍运行时再启动分布式 matrix，否则两个调度器可能选择
同一 `<ablation>/seed-N`。

矩阵包含 `full`、`visual_only`、`visual_relation`、`visual_action`、
`visual_kinematic`、`no_action`、`no_kinematic`、`no_relation`、
`no_outcome_anchor` 和 `no_monotonic_projection`。三组 `visual_*` interaction
ablation 都保留 outcome anchor，并且分别只增加 relation、action 或 kinematic 一个
privileged process channel，用于识别单通道收益与跨通道冲突。其中 `no_outcome_anchor` 同时关闭
success/failure pair anchor、终点锚定和基于成功标签的单调投影，确保不会暗中读取 outcome。
`no_monotonic_projection` 保留终点 success anchor，只去掉成功曲线的 cummax 投影。

已有七组 3-seed 结果后，只运行新增的九个 interaction runs：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh \
  --action run \
  --ablation visual_relation \
  --ablation visual_action \
  --ablation visual_kinematic \
  --seed 0 --seed 1 --seed 2 \
  --skip-completed \
  --distributed \
  --no-sync-remote
```

单独重新汇总已有结果：

```bash
bash mi_reward/scripts/run_generalization_ablations.sh --action summarize
```

输出统一位于：

```text
logs/mi_reward/generalization_rigid_v3/results/ablations/
  configs/
  <ablation>/seed-<N>/reward_model/joint_heldout_eval.json
  ablation_runs.csv
  ablation_summary.json
```

## 9. 验证代码

```bash
python3 -m compileall -q mi_reward
bash -n requirements/install.sh \
  mi_reward/scripts/generate_generalization_data.sh \
  mi_reward/scripts/run_generalization_reward.sh \
  mi_reward/scripts/eval_generalization_reward.sh \
  mi_reward/scripts/run_generalization_ablations.sh \
  eval/run_rbm_eval.sh
.venv-reward/bin/python -m pytest -q
```

GPU 测试在不可见 CUDA 的 CI/sandbox 中会显示 skipped，这是正常行为；在工作站终端
执行 `--preflight` 才是实际 CUDA 可见性的最终检查。

## 10. 主流程脚本索引

当前论文/刷榜主路线只需要以下入口：

| 阶段 | 脚本 | 说明 |
|---|---|---|
| 安装 | `requirements/install.sh` | 创建三个隔离环境并按 target 下载权重/数据 |
| 仿真资产 | `requirements/download_mujoco_assets.sh` | 下载 Panda 子树并生成任务 MJCF |
| 数据生成 | `mi_reward/scripts/generate_generalization_data.sh` | ingest → segmentation → planner → MuJoCo → Predict → manifest |
| Reward | `mi_reward/scripts/run_generalization_reward.sh` | 验证 → visual/action/kinematic latent → outcome-anchored MI 排序 → student 蒸馏 |
| 内部测试 | `mi_reward/scripts/eval_generalization_reward.sh` | 独立 joint-heldout 指标与同初态 Top-1 选择 |
| Ablation | `mi_reward/scripts/run_generalization_ablations.sh` | 隔离配置 → 多 seed 训练/评估 → JSON/CSV 汇总 |
| Benchmark | `eval/run_rbm_eval.sh` | 运行 RBM-EVAL 三类指标 |

`run_full_pipeline.sh`、`run_geoprogress.sh`、`extract_features.sh`、
`score_trajectories.sh` 和 `train_reward_sft.sh` 是早期实验/分步调试入口，不是当前
`generalization_data.yaml + generalization_reward.yaml` 主路线。`scripts/dist_test.py`
只用于验证两台机器的 PyTorch distributed/NCCL 通信，不参与单机默认训练，也不会
把两台 48 GB 显存合并为一个 96 GB 显存池。

## 11. 从零到榜单的命令顺序

```bash
# 1. 一次性安装
bash requirements/install.sh --all \
  --download-weights --download-eval-data --download-sim-assets --use-mirrors

# 2. 配置与资源检查
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --dry-run
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml --preflight

# 3. 生成并验证候选数据
bash mi_reward/scripts/generate_generalization_data.sh \
  --config mi_reward/configs/generalization_data.yaml

# 4. 提取 latent、构建 MI preference、训练视觉 student
bash mi_reward/scripts/run_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml

# 5. 独立 joint-heldout 最终测试
bash mi_reward/scripts/eval_generalization_reward.sh \
  --config mi_reward/configs/generalization_reward.yaml

# 6. 外部榜单评测
bash eval/run_rbm_eval.sh --config eval/configs/rbm_eval.yaml
```

每一步成功后再进入下一步。第 3 步以 `data_report.json` 的 acceptance gate 为准，
第 4 步以 `logs/mi_reward/generalization_rigid_v3/results/reward_model/pytorch_model.pt`
为准，第 5 步必须确认 `joint_heldout_eval.json` 中的
`split_isolation.valid` 为 `true`，第 6 步以 RBM-EVAL 的 `metrics.json` 为准。
