.. _franka_mi_potential_rlpd:

#############################################################
在 Franka 上使用 MI 势能塑形奖励 (MI Potential Shaping)
#############################################################

.. note::

   本文档参照 `RLinf 真机强化学习文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka.html>`_
   和 `Reward Model 使用指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/extending/reward_model.html>`_
   的结构编写。MI 势能奖励模型是 RLinf 现有 ``resnet`` / ``vlm`` / ``history_vlm`` 奖励模型之外的第四种类型。

----

概述
====

MI 势能塑形奖励 (MI Potential Shaping) 是一种 **离线蒸馏、在线塑形** 的奖励方案：

- **离线阶段**：从世界模型生成的 futures 中，通过 Dame-style 时间对齐互信息评分，蒸馏出一个轻量的 StatePotentialRewardModel :math:`V_\phi(o_t, g)`。
- **在线阶段**：冻结的 :math:`V_\phi` 在 RLPD 训练的每一步计算势能差 :math:`\gamma V_\phi(o_{t+1}) - V_\phi(o_t)`，作为稠密的「进展」塑形信号叠加在环境稀疏奖励上。

**在线推理不运行** 世界模型、B-spline MI、时间对齐或伪偏好生成。只运行冻结的学生模型。

与 RLinf 现有 Reward Model 的对比:

+---------------------+---------------------+---------------------+---------------------------+
| 特性                  | ResNet / VLM        | HistoryVLM          | **MI Potential (本方案)**    |
+---------------------+---------------------+---------------------+---------------------------+
| 输出语义               | 成功概率 (0/1)        | 成功概率 (0/1)        | **状态势能 (标量)**            |
| 奖励计算               | 模型直接输出 reward    | 模型直接输出 reward    | **势能差**: γV(next)-V(curr) |
| 训练数据               | 人工标注 (成功/失败)    | 人工标注 (成功/失败)    | **世界模型自动生成**           |
| 是否需要真实机器人数据   | 是                   | 是                   | **伪标签不需要，验证需要**      |
| 在线推理延迟           | 低                   | 中 (多帧 VLM)        | **极低 (GRU, ~2M 参数)**     |
| 模型注册类型           | resnet / vlm        | history_vlm         | **mi_potential**           |
+---------------------+---------------------+---------------------+---------------------------+

.. warning::

   本方案仍处于实验阶段。严禁在安全关键场景中仅依赖 MI 势能信号控制机器人。
   MI 势能 **不能** 触发 episode 终止、任务成功判定、或绕过碰撞/力/工作空间限制。

----

前置条件
========

1. 已完成 RLinf Franka 基础部署，确认 ``run_realworld_async.sh`` 可以正常启动。
2. 已按照 ``franka.rst`` 完成相机标定和网络配置。
3. **已完成离线 MI 奖励模型训练**，获得以下文件:

   .. code-block:: text

      mi_student_checkpoint/
      ├── pytorch_model.pt      # 训练脚本自动生成
      └── metadata.json         # 手动或脚本生成 (见下文)

4. 确认 `MI-directional-WAM-reward-model-pretrain` 仓库在训练节点上可导入:

   .. code-block:: bash

      cd /path/to/MI-directional-WAM-reward-model-pretrain
      pip install -e .

      python -c "from mi_reward.inference.potential_model import MIPotentialInferenceModel; print('OK')"

----

集群拓扑
========

与标准 Franka RLPD 部署一致:

.. code-block:: text

   ┌────────────────────────────┐     ┌──────────────────────────────┐
   │  控制节点 (Franka PC)        │     │  训练节点 (GPU 服务器)          │
   │                            │     │                              │
   │  • Franka 实时控制 (1 kHz)   │◄───►│  • RLPD 策略训练                │
   │  • 相机采集 (RealSense/ZED)  │ LAN │  • SAC Critic / Actor 更新     │
   │  • 环境奖励计算              │     │  • MI 势能模型推理 (冻结)        │
   │  • 安全监控                  │     │  • Replay Buffer              │
   │                            │     │  • Demo Buffer                │
   │  [无 GPU]                  │     │  [GPU: A100/3090/4090]        │
   └────────────────────────────┘     └──────────────────────────────┘

**MI 势能模型推理在训练节点 GPU 上运行**，控制节点不接触 MI 模型。

----

第一步：导出 MI 学生检查点
==========================

离线训练完成后，在 ``MI-directional-WAM-reward-model-pretrain`` 仓库中生成 ``metadata.json``:

.. code-block:: bash

   cd /path/to/MI-directional-WAM-reward-model-pretrain

   python - << 'PY'
   import json
   from pathlib import Path
   import torch

   ckpt_path = "results/mi_reward/reward_head_mvp/pytorch_model.pt"
   ckpt = torch.load(ckpt_path, map_location="cpu")
   config = ckpt.get("config", {})

   metadata = {
       "checkpoint_format_version": "1.0",
       "model_class": "StatePotentialRewardModel",
       "encoder_name": "dino_v3",
       "input_image_keys": ["main_images"],
       "history_size": 1,
       "task_conditioned": False,
       "output_semantics": "state_potential",
       "training_gamma": 0.99,
       "architecture": "gru",
       "token_pooling": "mean",
       "num_layers": 2,
       "num_heads": 4,
       "hidden_dim": config.get("hidden_dim", 256),
       "input_dim": config.get("input_dim", 512),
       "potential_mean": 0.0,
       "potential_std": 1.0
   }

   out = Path(ckpt_path).parent / "metadata.json"
   out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
   print(f"Written to {out}")
   PY

将检查点目录复制到训练节点:

.. code-block:: bash

   scp -r results/mi_reward/reward_head_mvp/ user@gpu-node:/path/to/checkpoints/mi_potential/

----

第二步：配置 YAML
================

从现有的 Franka RLPD 配置继承，创建 ``realworld_peginsertion_rlpd_cnn_async_mi_potential.yaml``:

.. code-block:: yaml
   :caption: examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_mi_potential.yaml

   # 继承标准 Franka RLPD 配置
   defaults:
     - realworld_peginsertion_rlpd_cnn_async
     - _self_

   # =====================================================================
   # Reward: 启用 MI 势能塑形
   # =====================================================================
   reward:

     # --- 基本设置 ---
     use_reward_model: true
     group_name: "RewardGroup"

     # 必须为 false: 奖励模型运行在 GPU 训练节点
     standalone_realworld: false

     # 按步计算奖励
     reward_mode: "per_step"

     # 声明: 奖励模型返回状态势能，不是二元成功概率
     reward_output_type: "state_potential"

     # 环境奖励权重 — 保留稀疏任务奖励
     env_reward_weight: 1.0

     # MI 势能奖励权重 — 调整塑形强度
     reward_weight: 1.0

     # ===================================================================
     # 势能塑形参数
     # ===================================================================
     potential_shaping:

       enabled: true

       # 势能衰减因子，默认与 SAC gamma 一致
       gamma: ${algorithm.gamma}

       # 势能差缩放因子
       scale: 1.0

       # 裁剪势能差到 [-delta_clip, +delta_clip]
       delta_clip: 1.0

       # 置信度门控
       confidence_gate: true

       # 置信度组合方式: "min" (取最小值) 或 "geometric_mean" (几何平均)
       confidence_combine: "min"

       # 最小置信度阈值
       min_confidence: 0.0

       # 无效预测处理: "zero_shaping" (跳过塑形, 保留环境奖励)
       invalid_behavior: "zero_shaping"

       # 势能归一化 (初始运行建议关闭)
       normalize: false

       # 预热步数 (前 N 步不施加塑形)
       warmup_steps: 0

       # [安全] 势能不能用于终止或成功判定
       use_for_termination: false
       use_for_success: false

     # ===================================================================
     # MI 势能模型配置
     # ===================================================================
     model:
       # 注册为 mi_potential 类型
       model_type: "mi_potential"

       # 训练好的学生检查点路径
       model_path: /path/to/checkpoints/mi_potential/pytorch_model.pt

       # 可选: 训练时的 YAML 配置
       config_path: null

       # 推理精度
       precision: "fp32"
       device: "cuda"

       # 输入模式: "current_frame" (单帧) 或 "history_buffer" (多帧窗口)
       input_mode: "current_frame"

       # 使用哪个/哪些相机图像
       image_keys:
         - main_images

       # 任务描述 (当前模型不使用 task conditioning, 保留字段)
       task_description: "insert the peg into the target hole"

   # =====================================================================
   # Phase B: MI 引导回放采样 (可选, 默认关闭)
   # =====================================================================
   algorithm:
     mi_guided_replay:
       enabled: false
       apply_to: "online_only"
       positive_threshold: 0.02
       negative_threshold: -0.02
       min_confidence: 0.2
       positive_fraction: 0.4
       neutral_fraction: 0.2
       negative_fraction: 0.4
       fallback_to_uniform: true
       max_resample_attempts: 5

   # =====================================================================
   # 数据采集 (与标准 RLinf 完全一致)
   # =====================================================================
   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

----

第三步：注册 MI 势能模型类型
=============================

在 RLinf 的 reward model registry 中注册 ``mi_potential`` 类型。

将 ``docs/rlinf_integration/rlinf_workers_reward/mi_potential_reward_model.py``
复制到 ``rlinf/workers/reward/mi_potential_reward_model.py``，并在 reward worker 的
注册表中添加:

.. code-block:: python

   # rlinf/workers/reward/__init__.py 或对应的 registry 文件

   from rlinf.workers.reward.mi_potential_reward_model import MIPotentialRewardModel

   REWARD_MODEL_REGISTRY["mi_potential"] = MIPotentialRewardModel

----

第四步：启动训练
=================

训练启动命令与标准 Franka RLPD **完全一致**:

.. code-block:: bash

   # 在训练节点上
   cd /path/to/RLinf

   bash examples/embodiment/run_realworld_async.sh \
     realworld_peginsertion_rlpd_cnn_async_mi_potential

   # 在控制节点上 (Franka PC)
   bash examples/embodiment/run_realworld_async.sh \
     realworld_peginsertion_rlpd_cnn_async_mi_potential \
     --control

训练时观察日志中 MI 势能相关指标:

.. code-block:: text

   [RewardWorker] Loaded MI potential model from /path/to/.../pytorch_model.pt
   [RewardWorker] Architecture: gru, input_dim=512, hidden_dim=256
   [EnvWorker]  Potential shaping enabled (gamma=0.99, scale=1.0, clip=1.0)

如果未看到这些日志，检查 ``potential_shaping.enabled`` 是否为 ``true``。

----

第五步：Dummy Mode 验证（先不要启动真机）
========================================

在连接真机之前，建议使用 dummy 模式验证完整流程:

.. code-block:: bash

   # 在训练节点上
   cd /path/to/RLinf

   # 使用 dummy 配置覆盖
   bash examples/embodiment/run_realworld_async.sh \
     realworld_peginsertion_rlpd_cnn_async_mi_potential \
     env=dummy \
     reward.potential_shaping.warmup_steps=10

验证以下几点:

1. RewardWorker 成功加载 MI 模型
2. 每步日志中出现 ``mi_potential_current`` / ``mi_potential_next`` / ``mi_delta_raw``
3. 势能差在合理范围 (不出现 NaN 或 ±1e6)
4. Replay buffer 中存储了 ``mi_delta`` 和 ``mi_confidence`` 字段
5. SAC Critic 和 Actor loss 正常更新

----

在线奖励公式
============

对于每一帧 transition :math:`(o_t, a_t, r^\text{env}_t, o_{t+1}, \text{done}_t)`:

**第一步**: 计算状态势能

.. math::

   V_t &= V_\phi(o_t, g) \\
   V_{t+1} &= V_\phi(o_{t+1}, g)

**第二步**: 计算势能差并裁剪

.. math::

   \Delta_t^\text{raw} &= \gamma \cdot V_{t+1} - V_t \\
   \Delta_t^\text{clip} &= \text{clamp}(\Delta_t^\text{raw}, -\delta_\text{clip}, +\delta_\text{clip})

**第三步**: 置信度门控

.. math::

   c_t &= \min(c_t^\text{cached}, c_{t+1}^\text{next}) \quad \text{(或几何平均)} \\
   \Delta_t^\text{gated} &=
   \begin{cases}
   c_t \cdot \Delta_t^\text{clip} & \text{if confidence\_gate = true} \\
   \Delta_t^\text{clip} & \text{otherwise}
   \end{cases}

**第四步**: 最终奖励

.. math::

   r_t^\text{final} =
   w_\text{env} \cdot r_t^\text{env}
   +
   w_\text{mi} \cdot \lambda_\text{scale} \cdot \Delta_t^\text{gated}

其中 :math:`w_\text{env}` = ``env_reward_weight``, :math:`w_\text{mi}` = ``reward_weight``,
:math:`\lambda_\text{scale}` = ``scale``, :math:`\delta_\text{clip}` = ``delta_clip``。

势能缓存的生命周期:

.. code-block:: text

   Episode 开始:
     env.reset() → o_0 → V_phi(o_0, g) → [缓存]

   每一步 t:
     step(a_t) → o_{t+1} → V_phi(o_{t+1}, g)
     delta_t = γ·V(o_{t+1}) - V(o_t)   ← V(o_t) 来自缓存
     奖励 = env_reward + mi_delta
     V(o_t) ← V(o_{t+1})                ← 更新缓存

   Episode 结束 (done/reset):
     清空缓存 → 新 episode 重新初始化

.. warning::

   上一个 episode 的缓存势能 **绝对不能** 用于下一个 episode。
   ``PotentialShapingState.clear(env_ids)`` 在每次 reset 前强制清空。

----

Ablation 实验配置
==================

以下 ablation 通过修改 YAML 或命令行覆盖即可实现，无需改 Python 代码。

**A. 标准 RLPD (Baseline)**

.. code-block:: yaml

   reward.potential_shaping.enabled: false

**B. 仅环境稀疏奖励**

.. code-block:: yaml

   reward.use_reward_model: false

**C. 标准二元奖励模型 (ResNet)**

.. code-block:: yaml

   reward.model.model_type: "resnet"
   reward.reward_output_type: "success_probability"

**D. MI 势能塑形 only (Phase A)**

.. code-block:: yaml

   reward.potential_shaping.enabled: true
   reward.reward_weight: 1.0
   algorithm.mi_guided_replay.enabled: false

**E. MI 引导回放 only (Phase B)**

.. code-block:: yaml

   reward.potential_shaping.enabled: true
   reward.reward_weight: 0.0            # 塑形权重为零
   algorithm.mi_guided_replay.enabled: true
   # 注意: 即使 reward_weight=0, 仍需计算势能差来标注 transition 的 mi_delta

**F. 完整方案 (MI 塑形 + MI 引导回放)**

.. code-block:: yaml

   reward.potential_shaping.enabled: true
   reward.reward_weight: 1.0
   algorithm.mi_guided_replay.enabled: true
   reward.potential_shaping.confidence_gate: true

----

TensorBoard 监控指标
=====================

训练启动后关注以下新增指标:

**奖励推理**

.. code-block:: text

   reward/mi_potential_mean           # 势能均值 (应稳定)
   reward/mi_potential_std            # 势能标准差
   reward/mi_delta_mean               # 势能差均值 (正值为进展)
   reward/mi_delta_std                # 势能差标准差
   reward/mi_delta_positive_fraction  # 正向进展的 transition 比例
   reward/mi_delta_negative_fraction  # 负向退步的 transition 比例
   reward/mi_confidence_mean          # 平均置信度
   reward/mi_invalid_fraction         # 无效预测比例 (应 ≈ 0)
   reward/mi_inference_latency_ms     # 单次推理延迟
   reward/mi_clipped_fraction         # 被裁剪的势能差比例

**每 episode**

.. code-block:: text

   env/reward_raw                     # 原始环境奖励
   env/reward_mi_shaping              # MI 塑形奖励分量
   env/reward_final                   # 最终组合奖励
   env/episode_mi_progress            # 整个 episode 的净势能进展
   env/start_potential                # episode 初始势能
   env/end_potential                  # episode 最终势能
   env/net_potential_progress         # V_end - V_start

**回放采样 (Phase B)**

.. code-block:: text

   train/replay/mi_positive_fraction   # 正进展采样比例
   train/replay/mi_neutral_fraction    # 中性采样比例
   train/replay/mi_negative_fraction   # 负退步采样比例
   train/replay/mi_low_confidence_fraction  # 低置信度比例

----

常见问题
========

**Q: 训练时提示 `No module named 'mi_reward'`**

A: 确认 MI 仓库已 ``pip install -e`` 到训练节点的 Python 环境中。

**Q: 势能差始终为 0**

A: 检查:
  1. ``potential_shaping.enabled: true``
  2. ``warmup_steps`` 是否过大
  3. ``delta_clip`` 是否设为了 0
  4. 检查点是否正确加载 (看 RewardWorker 日志)

**Q: 势能差出现 NaN**

A: 检查:
  1. ``metadata.json`` 中的 ``input_dim`` 是否与实际特征维度匹配
  2. 图像预处理是否与训练时一致
  3. 模型是否加载成功 (confidence 应为 1, valid 应为 True)

**Q: MI 势能下降时机器人行为变差**

A: 这是预期行为的一部分 — 负向势能差表示「退步」，SAC Critic 会学到避免该动作。
如果持续下降不回升，可:
  1. 降低 ``reward_weight`` (如 0.5)
  2. 增大 ``delta_clip`` 的负向端对称性
  3. 检查环境奖励是否被 MI 塑形淹没

**Q: 能否用 MI 势能直接判断任务成功**

A: 不能。势能是进展信号，不是二元的成功/失败判断器。
成功判定应始终依赖:
  - 环境 ground truth (仿真)
  - 人工标注或传感器 (真机)
  - 环境自身的 binary reward

----

安全注意事项
=============

.. warning::

   1. **MI 势能不能触发 reset 或终止 episode。** 相关配置已硬编码为 ``false``。
   2. **MI 势能不能绕过** 工作空间限制、力限制、碰撞检测、紧急停止、人工干预、episode 超时。
   3. **模型超时/NaN/无效输出时**：记录限速警告，该步 MI 塑形置零，保留环境奖励，不中断控制循环。
   4. **势能差裁剪** (``delta_clip``) 必须保持合理范围 (默认 1.0)。
   5. **模型只加载一次**（worker 初始化时），**永不**在线更新权重。
   6. **推理始终在 eval 模式**（``torch.inference_mode()``）。
   7. 首次在 Franka 上运行时，建议 **先设 ``reward_weight: 0.0``** ，确认日志正常后再逐步增高。

----

数据采集
========

MI 势能模型 **不需要** 额外的真机数据采集。数据采集流程与标准 Franka RLPD 完全一致:

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

采集到的 episodes 可用于:
  - Demo Buffer (RLPD 的 50% 演示数据)
  - 评估 MI 势能模型的真机泛化能力
  - 未来的奖励模型 fine-tuning

**需要注意的是**：MI 势能模型的训练数据来自世界模型生成的 futures，不是真机采集。
真机采集仅用于 RLPD 的策略训练和验证。

----

相关文档
========

- `MI 势能奖励模型离线训练 <https://github.com/yangzhongii/MI-directional-WAM-reward-model-pretrain>`_
- `Franka 真机强化学习 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka.html>`_
- `Reward Model 使用指南 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/extending/reward_model.html>`_
- `在 Franka 上使用灵巧手 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/franka_dexhand.html>`_

----

引用
====

MI 势能塑形方法的灵感来源:

.. code-block:: bibtex

   @article{dame2011mutual,
     title = {Mutual Information-Based Visual Servoing},
     author = {Dame, Amaury and Marchand, Eric},
     journal = {IEEE Transactions on Robotics},
     volume = {27},
     number = {5},
     pages = {958--969},
     year = {2011},
   }

在线奖励的理论基础 (Potential-Based Reward Shaping):

.. code-block:: bibtex

   @inproceedings{ng1999policy,
     title = {Policy invariance under reward transformations:
              Theory and application to reward shaping},
     author = {Ng, Andrew Y and Harada, Daishi and Russell, Stuart},
     booktitle = {ICML},
     year = {1999},
   }
