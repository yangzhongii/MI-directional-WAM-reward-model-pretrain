# Pipeline v4：Privileged Directional Supervision with Goal-Grounded Visual Reward

## 0. 文档状态

本文档是 v3 之后的新研究与实现路线。v3 checkpoint、缓存、校准和审计结果保持冻结；v4 使用新文件名和新输出目录，不覆盖 v3 历史证据。

v4 的基本决定是：

1. 保留训练期 privileged robot physics、P/U/N abstention、双视角历史和 observation-only Qwen 蒸馏。
2. 不再声称当前 visual critic 严格估计 PMI，也不再声称总分由 MI chain rule 推出。
3. 将 LaWAM 的 `(o_t,o_{t+1})` 表示称为 **transition latent**，不把它等同于 executed action。
4. 先通过冻结诊断和小模型实验验证 goal-grounded visual signal，再决定是否训练 Qwen 和进入闭环 RL。
5. TRO 的 mutual-information visual servoing 只作为“参考图像可定义局部对齐目标”的动机，不作为通用语义进度 reward 的现成理论证明。

## 1. 固定研究问题

### 1.1 一句话主张

利用训练期可见的接触、抓取、运动学、对象关系、结果和可行性证据，构造带拒判的局部方向监督；再将该监督蒸馏到只读取双视角视觉历史、任务语言和任务级成功参考的视觉语言 reward model，使其能够识别 manipulation 中的 progress、neutral、regression 和 near-miss。

### 1.2 第一篇论文应回答的问题

> Privileged physical directional supervision 是否能训练出一个 observation-only reward model，使其在未见 episode、初始状态和真实策略 rollout 上，比 temporal order、goal similarity、success-only 和普通 VLM judgment 更可靠地识别 regression 与 near-miss？

### 1.3 不在第一篇论文中主张的内容

- 不声称任意图像和语义 goal 之间的 MI 随任务进度单调增加。
- 不声称 visual 与 transition 两项构成严格的 information chain-rule decomposition。
- 不声称简单对应位置 patch cosine 等价于 TRO 的 MI visual servoing。
- 不在离线方向识别尚未通过前，把闭环 RL 成功率作为主要结果。

## 2. v3 到 v4 的关键修改

| v3 | v4 |
|---|---|
| `visual PMI critic` | `goal-grounded visual potential` 或直接 `visual direction classifier` |
| `conditional action PMI` | `transition evidence scorer`；executed action 另做独立消融 |
| `D = gamma * phi(t+1) - phi(t) + beta * psi` | 默认使用未折扣 visual difference；组合方式必须通过 held-out 消融选择 |
| episode-specific terminal goal | 训练、验证和 rollout 一致的 task-level reference set |
| 同任务另一 episode goal 作为“边缘负样本” | 正确 goal、错误任务 goal、同任务多模态 goal 分开建模 |
| 物理规则参与训练又充当最终评价真值 | 训练标签与独立 challenge-set 审核分离 |
| 先训练 teacher/Qwen 再看闭环 | 先过 Goal、Visual、Fusion、Student 四道 gate |

## 3. v4 的变量与数据契约

令：

- (o_{t-h:t+1})：同步的 agentview 与 wrist 短历史；
- (z_t=E(o_t))：冻结或轻量微调的视觉状态表示；
- (e_t=E_{tr}(o_t,o_{t+1}))：transition latent；
- (a_t)：模拟器或机器人实际执行动作，仅训练期可选；
- (G_q=\{g_q^1,\ldots,g_q^K\})：任务 (q) 的成功参考集合；
- (l_q)：任务语言；
- (x_t^{priv})：接触、抓取、位姿、对象关系、成功和可行性等 privileged 状态；
- (y_t^{priv}\in\{P,U,N\})：由可信物理证据和阶段语义构造的方向标签。

### 3.1 Goal 必须是部署时可获得的

每个任务在 split 前固定 task-level reference set。训练、验证和失败 rollout 均使用相同定义，不再使用“当前 episode 自己未来的成功终局”。

参考 episode 必须与被评估 episode 分离。对未见任务，可以只使用任务语言，或提供单独收集的成功参考；两种设置必须分开报告。

### 3.2 数据划分单位

最小独立单位是 episode/initial state，最终泛化单位是 task。相邻窗口不得跨 split。所有置信区间以 task 或 initial-state cluster bootstrap 计算，不把窗口数当作独立样本数。

## 4. Privileged 方向标签

### 4.1 标签生成

先为每个任务定义阶段关系，例如：

```text
approach object
  -> establish grasp/contact
  -> transport while preserving grasp
  -> satisfy target relation
  -> stable success
```

对每个 transition 计算独立物理变化：

- 目标对象/位置/容器关系是否改善；
- grasp/contact 是否建立、保持或丢失；
- 是否发生 slip、drop、wrong-object、wrong-target 或碰撞；
- 当前阶段是否切换；
- 环境 success 是否首次成立并保持；
- 证据是否足以判断方向。

标签规则：

- **Positive**：至少一个阶段目标明确改善，且无更强的失败证据；
- **Negative**：阶段关系明确恶化、抓取丢失、目标关系破坏或出现不可行结果；
- **Unclear**：变化小于噪声、证据冲突、阶段未知或视觉窗口与物理事件无法对齐。

### 4.2 避免循环评价

训练标签可以由自动 physics labeler 产生，但最终 challenge set 必须独立审核。推荐抽取至少以下六类事件，每类按任务和初始状态分层：

1. grasp 与 near-grasp；
2. contact 与 near-contact；
3. stable placement 与仅位于目标上方；
4. hold 与 slip/drop；
5. 正常 progress 与短暂视觉遮挡；
6. regression 与 recovery。

审核记录保存物理量、双视角帧和标签理由。评价脚本只读取审核结果，不重新调用 teacher gate。

## 5. Visual 分支的两种候选

v4 不预设 scalar potential 一定成立。先并行比较两种低成本候选。

### 5.1 候选 A：直接方向分类

直接预测：

\[
p_v(y_t\mid o_{t-h:t+1},l_q,G_q),\qquad y_t\in\{P,U,N\}.
\]

它允许非单调和阶段切换，不要求存在全局 scalar potential。第一轮使用冻结特征和小 MLP/attention head，避免用大模型容量掩盖目标定义问题。

损失采用 class-balanced cross entropy 或 focal loss；训练采样按 task、事件类型和 P/U/N 平衡。报告 calibrated confidence 和 abstention coverage-risk 曲线。

### 5.2 候选 B：带排序约束的 scalar potential

对每个参考单独计算相似度：

\[
s_k(z_t,g_q^k)=f_\theta(z_t,g_q^k,l_q).
\]

将成功参考看作集合，而不是先平均 reference embedding：

\[
u_t(G_q)=
\tau\log\left(\frac{1}{K}\sum_{k=1}^{K}
\exp(s_k/\tau)\right).
\]

方向分数默认使用：

\[
d_t^v=u_{t+1}(G_q)-u_t(G_q).
\]

不在方向分类中乘 RL discount。对可信 P/N transition 使用 pairwise ranking loss：

\[
\mathcal L_{rank}
=
\log\left(1+\exp\left[-y_t
(u_{t+1}-u_t)/\tau_r\right]\right),
\]

其中 (y_t=+1) 对应 Positive，(y_t=-1) 对应 Negative。Unclear 样本只用于 deadband/calibration，或使用限制差值绝对值的辅助损失。

如果该候选无法显著超过 cosine difference，则不再把 scalar potential 作为论文核心。

## 6. Goal-grounding 训练与诊断

### 6.1 参考聚合

必须比较：

- single fixed reference；
- score-level mean；
- score-level max；
- normalized log-mean-exp；
- 可学习 set attention。

当前简单 mean/max 的冻结结果只说明旧 critic 没有可用 task-level goal grounding，不代表重新训练后的 set model 一定失败。

### 6.2 Goal 对照组

每个窗口至少构造：

- 正确任务 reference set；
- 同 suite 的错误任务 reference set；
- 外观相近但关系不同的 hard goal；
- 正确语言配错误图像参考；
- 错误语言配正确图像参考。

goal sensitivity 不能只测静态 logit。还应检查更换 goal 后方向预测是否按目标语义变化，以及模型是否只依赖背景和任务身份。

### 6.3 防止 episode identity 捷径

- reference episode 与当前 episode 永不重叠；
- 图像增强不能泄露 episode ID；
- 同一状态搭配多个 counterfactual goal；
- 按初始布局、纹理、相机扰动和对象实例分层报告；
- 训练后用 goal masking 检查性能下降幅度。

## 7. Transition 分支的重新定义

### 7.1 三个独立输入版本

分别训练或线性探测：

1. `executed_action_only`：(a_t) 与当前状态；
2. `observation_transition`：(e_t=E_{tr}(o_t,o_{t+1}))；
3. `state_pair`：显式输入 (z_t,z_{t+1})。

三者不得统一称作 action information。第二、第三种已经观察到结果状态，适合称为 transition evidence。

### 7.2 组合规则

先保留以下可审计基线：

\[
d_t^{fusion}=w_v d_t^v+w_e d_t^e,
\]

其中 (w_v,w_e) 只在 validation split 上拟合，并在 test/challenge split 冻结。还应比较：

- visual-only；
- transition-only；
- late-fusion logistic regression；
- learned gated fusion；
- privileged heuristic only。

只有 fusion 相对 transition-only 的 task-cluster paired 95% CI 下界大于 0，才可以声称视觉参考带来互补增益。否则 final teacher 使用更简单的分支。

## 8. P/U/N 与 abstention

模型先输出连续证据或三类 logits，再用独立 validation set 标定：

- P/N 决策阈值；
- Unclear deadband；
- 最低置信度；
- 多分支冲突时的 abstention；
- 每个 task/event 的 coverage-risk。

阈值不得来自训练集 neutral 分位数。禁止在 test rollout 上重新调阈值。

## 9. Qwen 蒸馏

Qwen 输入保持：

```text
task language
+ synchronized agentview history
+ synchronized wrist history
+ optional task-level success references
```

训练修改：

- episode-grouped split；
- P/U/N class-balanced sampler；
- oversample regression、slip、near-miss 和 wrong-relation；
- 控制高度重叠窗口比例；
- 先冻结大部分 backbone，只训练 adapter/head；
- 使用 validation macro-F1、Negative recall 和 calibration early stop；
- 保存 confusion matrix 和每类置信区间。

Qwen 不需要复现 teacher 的连续分数，只需可靠复现方向、拒判和事件语义。若 Qwen 对 Negative 的泛化持续失败，应缩小模型或增加独立困难负例，不继续增加 teacher 数学复杂度。

## 10. 四道 Go/No-Go Gate

### Gate 0：冻结诊断

状态：**已完成，当前 v3 visual critic 为 No-Go。**

已有冻结探针使用 5 个 held-out episode、536 个 transition，其中 non-neutral 为 477：

| 信号 | Balanced accuracy | Forward recall | Regression recall |
|---|---:|---:|---:|
| visual own-goal, gamma=0.99 | 52.95% | 51.18% | 54.72% |
| visual own-goal, gamma=1 | 52.71% | 50.71% | 54.72% |
| transition/psi only | 68.51% | 84.20% | 52.83% |
| combined, gamma=0.99 | 67.92% | 83.02% | 52.83% |
| combined, gamma=1 | 67.92% | 83.02% | 52.83% |

goal diagnostics：

- 从 10 个训练 reference goal 中检索同任务 goal 的 top-1 accuracy 为 **20.70%**；5 个任务的随机水平为 20%。
- same-task goal 相对 cross-task goal 的平均 logit margin 为 **−0.0146**。
- task-level goal mean/max/log-mean-exp 的 visual balanced accuracy 分别为 50.47%、47.76%、47.41%。
- combined gamma=1 相对 transition-only 的 paired task-cluster bootstrap 差值均值为 −0.80 个百分点，95% CI 为 [−7.39, +2.61]。

解释：调 gamma 无法修复 learned visual critic；当前 critic 没有显示 task-level goal grounding，也没有为 transition-only 提供增益。下一步必须重新训练 visual objective，不能继续调旧 critic 的参考聚合方式。

### Gate 1：小模型 visual feasibility

用冻结视觉特征训练候选 A/B，小规模完成以下对照：

- cosine difference；
- direct visual direction classifier；
- ranked scalar potential；
- single 与 set-valued goal；
- goal masking 与 wrong-goal control。

通过条件：

1. 在独立 challenge set 上 balanced accuracy 显著超过 50%；
2. 相对 cosine baseline 的 task-cluster paired 95% CI 下界大于 0；
3. regression recall 不低于 60%；
4. wrong-goal 或 goal masking 会产生预期且可解释的退化；
5. 结果至少在多个 initial states 上成立。

若 direct classifier 通过而 scalar potential 失败，论文使用直接方向模型。若两者都失败，停止 visual-goal 主线，转为纯 privileged temporal reward distillation。

### Gate 2：Fusion

通过条件：fusion 相对 transition-only 的 paired 95% CI 下界大于 0，并且 regression recall、near-miss false-positive rate 至少一项有明确改善，另一项不能显著恶化。

失败时保留更简单的 transition/visual 单分支，不强行组合。

### Gate 3：Student 与闭环资格

Qwen 必须在未参与校准的 rollout challenge set 上达到：

- macro-F1 明显高于 majority 和 temporal-order baseline；
- Negative recall 至少 60%；
- physical non-forward 被判 Positive 的比例不高于 20%；
- 不同 initial states 的结果方向一致；
- success/failure trajectory ranking 使用严格平局规则。

达到后才运行小规模闭环 RL，并检查停滞、循环、遮挡捷径和 reward hacking。

## 11. 推荐的最小突破实验

### Experiment A：Goal-aware direct direction probe

这是当前最高优先级实验，预计只训练小 head，不训练 Qwen。

数据：复用冻结 LaWAM 特征；增加至少 3 个 initial states/task 的 scripted rollout，重点补充 regression。reference 固定使用与 rollout 不重叠的 task-level demos。

模型：

```text
[z_t, z_t1, z_t1-z_t, language/reference-set summary]
    -> small MLP or two-layer attention head
    -> P/U/N logits
```

必须包含两个控制：

1. 去掉 goal 输入；
2. 将 goal 在 task 间随机置换。

如果正确 goal 模型不能稳定优于这两个控制，说明数据中的局部方向主要由视觉运动或 task identity 决定，goal-grounded 贡献不成立。

### Experiment B：Potential ranking probe

使用同一数据训练 scalar potential，只在高置信 P/N transition 上施加排序损失。比较 direct classifier 和 scalar potential。

该实验回答：局部方向是否可由一个状态函数表示。若 direct classifier 明显更好，说明任务存在阶段依赖、滞后或非保守方向信号，应停止强制 scalar potential。

### Experiment C：Executed-action 与 transition-latent 消融

比较真实 (a_t)、LAM transition latent 和显式 state pair。若 transition latent 与 state pair 相当而明显优于 executed action，应将它解释为 outcome-aware transition evidence，不再赋予 conditional-action-information 含义。

## 12. 代码修改顺序

所有 v4 实现使用新模块，避免破坏 v3 冻结基线。

1. 已增加 `eval/libero/probe_v4_hypotheses.py`：冻结 gamma、offset、goal 和 fusion 诊断。
2. 新增 `mi_reward/training/train_direction_teacher_v4.py`：直接方向分类与 potential ranking 两个 head。
3. 新增 `mi_reward/data/build_direction_dataset_v4.py`：task-level references、group split、counterfactual goals 和 event-balanced sampling。
4. 新增 `mi_reward/scoring/direction_teacher_v4.py`：不使用 PMI 命名，输出 visual/transition/fusion evidence。
5. 新增 `mi_reward/evaluation/eval_direction_teacher_v4.py`：独立 challenge labels、task-cluster bootstrap 和 paired ablation。
6. Gate 1/2 通过后，再新增 `train_qwen3_vl_reward_v4.py` 和对应数据导出器。
7. Gate 3 通过后，才为 RLinf 增加 v4 adapter；v3 adapter 保留用于历史复现。

推荐输出目录：

```text
logs/mi_reward/v4_decision_gate/
logs/mi_reward/v4_direction_teacher/
logs/mi_reward/v4_student/
logs/mi_reward/v4_closed_loop/
```

## 13. 当前可复现命令

冻结决策探针：

```bash
.venv-libero/bin/python eval/libero/probe_v4_hypotheses.py \
  --output-dir logs/mi_reward/v4_decision_gate/frozen_probe_reproduction
```

当前输出：

```text
logs/mi_reward/v4_decision_gate/frozen_probe_v1/results.json
logs/mi_reward/v4_decision_gate/frozen_probe_v1/SUMMARY.md
```

探针只读取 v3 checkpoint 和缓存特征，不修改权重、校准或标签。当前只有 5 个 task cluster，且方向仍是既有 physics heuristic，因此结果用于选择下一步，不作为最终论文结论。

## 14. 最终决策树

```text
Goal-aware direct probe 是否超过无 goal / wrong goal control？
    |
    +-- 否：删除 goal-grounded visual contribution
    |       -> privileged temporal direction distillation
    |
    +-- 是：scalar potential 是否同样成立？
            |
            +-- 否：使用 direct transition direction classifier
            |
            +-- 是：保留 goal-grounded potential
                    |
                    +-- fusion 是否显著超过最佳单分支？
                            |
                            +-- 否：使用最佳单分支
                            +-- 是：使用 learned/calibrated fusion
```

这条路线的目的不是继续维护 MI 名称，而是用最少实验确定：视觉 goal、scalar potential 和 transition evidence 中，究竟哪一项在独立数据上真正成立。

## 15. Gate 1 小模型 smoke：第一次验证结果

### 15.1 实验设计

第一次验证只使用冻结的 pooled LaWAM visual features，不加载或更新 v3 teacher、Qwen、校准或数据缓存。对 `libero_spatial` 的 task 0–4 固定如下 episode 级划分：

- `demo_0`：task-level 成功参考；
- `demo_1`：训练小模型；
- `demo_10`：held-out 测试。

三者不重叠。训练集包含 568 个 transition，测试集包含 536 个 transition；采用 3 个固定随机种子并对分数做 ensemble。比较 cosine difference、goal-aware direct P/U/N classifier、no-goal direct classifier 和 goal-aware scalar potential。Goal 控制包括 zero masking、跨 task 置换和同参考 episode 前 5 帧均值构成的 initial-state hard goal。

P/N 方向指标不使用校准阈值：排除 U 后严格按连续分数正负判断，零分记错。置信区间采用 5,000 次 task-cluster paired bootstrap。跨 task 置换在 `libero_spatial` 中只是 task-identity 控制，因为各任务具有相似的最终关系，不能将它当成语义错误 goal 的充分检验。

### 15.2 结果

| 方法或控制 | Balanced accuracy | Forward recall | Regression recall |
|---|---:|---:|---:|
| Cosine difference | 59.79% | 59.20% | 60.38% |
| Direct + correct goal | 76.18% | 93.87% | 58.49% |
| Direct, no goal | 77.48% | 94.58% | 60.38% |
| Direct + masked goal | 75.24% | 90.09% | 60.38% |
| Direct + initial-state hard goal | 73.47% | 79.01% | 67.92% |
| Scalar potential + correct goal | 73.58% | 71.70% | 75.47% |
| Potential + masked goal | 68.40% | 68.87% | 67.92% |
| Potential + initial-state hard goal | 66.27% | 68.40% | 64.15% |

关键 paired difference：

- direct correct-goal 相对 cosine：+16.19 个百分点，95% CI `[+5.43, +27.60]`；
- potential correct-goal 相对 cosine：+13.70 个百分点，95% CI `[+6.78, +20.16]`；
- direct correct-goal 相对 no-goal：−1.34 个百分点，95% CI `[−6.51, +4.19]`；
- potential correct-goal 相对 masked goal：+5.37 个百分点，95% CI `[+1.60, +9.83]`；
- potential correct-goal 相对 initial-state hard goal：+7.71 个百分点，95% CI `[+1.40, +15.22]`；
- direct correct-goal 相对 potential correct-goal：+2.53 个百分点，95% CI `[−6.55, +10.77]`。

Direct 的三分类 accuracy 为 74.63%、macro-F1 为 0.53295，U recall 为 35.59%。这些三分类指标没有单独校准 abstention，因此只用于暴露 U 识别问题。

### 15.3 当前判定

本实验说明 frozen visual state pair 中存在可学习的局部方向信号，但不能将 direct classifier 的提升归因于 goal：no-goal 基线略高，且 direct 的 regression recall 为 58.49%，没有达到 Gate 1 的 60% 门槛。

Scalar potential 在这次 smoke 中是更值得继续验证的 goal-grounded 候选：它显著超过 cosine，regression recall 为 75.47%，正确成功参考也显著超过 masked 和 initial-state hard goal。不过，direct 与 potential 的差异区间跨零；本结果不能证明 scalar potential 优于 direct classifier。

状态保持为 **SMOKE_ONLY / Gate 1 未通过**，原因如下：

1. 每任务只有一个 test episode/initial state；
2. 方向标签仍为既有 physics heuristic，不是独立审核 challenge labels；
3. 只使用 pooled feature 与 single reference，没有完成 set-valued reference 对照；
4. `libero_spatial` 的跨任务 goal 具有相似终态语义，wrong-goal 证据不充分。

下一步冻结本次 head、超参数和判定标准，增加至少 3 个 initial states/task 的 scripted rollout，只扩展测试数据。在新 rollout 上预注册检验 potential 相对 cosine、masked goal 和 initial-state hard goal 的 paired CI，并审核 regression、near-miss 与模型分歧窗口；不得根据新测试结果调参。

### 15.4 复现与审计记录

实现：`eval/libero/probe_v4_gate1_smoke.py` 与 `eval/libero/run_v4_gate1_smoke.sh`。

正式输出：`logs/mi_reward/v4_decision_gate/gate1_smoke_v2/`，包含 `results.json`、`predictions.npz`、3 个 seed 的小 head、`SUMMARY.md`、`run.log` 和退出码 0。运行位于 `tmux test:v4-gate1-smoke`，下方 pane 保留实时日志显示。

第一次输出 `gate1_smoke_v1/` 保留为审计记录。其 P/N 与 bootstrap 指标有效，但三分类 ensemble 曾对已是 `{-1,0,+1}` 的标签做了第二次映射；正式 v2 已修正并增加完整 potential goal controls，v1 目录已用 `SUPERSEDED.md` 标记。该修正不改变本节报告的 P/N 结果。

## 16. Gate 1 多初始状态冻结复现

### 16.1 目的与冻结协议

第 15 节的正向结果可能来自单个 held-out demo，而不是跨初始状态泛化。因此，本轮冻结 `gate1_smoke_v2` 的 3 个 seed checkpoint、训练归一化、reference、超参数和零阈值，不重新训练、不选择 checkpoint，也不根据 rollout 调参。

实验覆盖 task 0–4，每个 task 使用 `demo_20/21/22` 三个不同 initial states。每个状态运行原 demo action replay 和强制 open-gripper replay，形成 15 个 initial-state clusters、30 条轨迹。`demo_20` 复用既有 rollout；`demo_21/22` 新采集但不进入训练。Action replay 共成功 12/15，open-gripper 为 0/15；controller 名称不作为窗口方向标签。

每条轨迹均匀采样 8 个窗口，共 240 个窗口。对每个窗口使用最后一个 transition，并通过 seed、初始状态和完整动作序列重放恢复 contact、grasp、距离与 success。所有选中帧均通过保存状态与 success 一致性检查。标签仍使用既有 physics heuristic，分布为 P=125、U=83、N=32。

模型输入使用与训练相同的双视角 pooled LaWAM state pair。置信区间采用层级 paired bootstrap：先重采样 5 个 task，再在各 task 内重采样 3 个 initial-state clusters；每种对照始终使用相同采样索引。

### 16.2 多状态结果

| 方法或控制 | Balanced accuracy | Forward recall | Regression recall |
|---|---:|---:|---:|
| Cosine difference | 51.80% | 53.60% | 50.00% |
| Potential + correct goal | 54.60% | 59.20% | 50.00% |
| Potential + masked goal | 58.56% | 64.00% | 53.12% |
| Potential + permuted goal | 56.93% | 57.60% | 56.25% |
| Potential + initial-state hard goal | 59.00% | 68.00% | 50.00% |
| Direct + correct goal | 55.05% | 97.60% | 12.50% |
| Direct, no goal | 50.73% | 95.20% | 6.25% |

层级 paired differences：

- potential correct-goal 相对 cosine：+3.04 个百分点，95% CI `[−15.62, +23.36]`；
- potential correct-goal 相对 masked goal：−3.98 个百分点，95% CI `[−9.23, +1.04]`；
- potential correct-goal 相对 initial-state hard goal：−4.06 个百分点，95% CI `[−15.18, +7.14]`；
- potential correct-goal 相对 permuted goal：−2.42 个百分点，95% CI `[−9.26, +2.29]`；
- direct correct-goal 相对 no-goal：+4.45 个百分点，95% CI `[0.00, +13.04]`；
- direct correct-goal 相对 potential correct-goal：+0.45 个百分点，95% CI `[−12.51, +13.44]`。

Potential 在 `demo_20/21/22` 三组 initial states 上的 balanced accuracy 分别为 62.27%、44.39%、56.82%；按 task 分别为 52.76%、60.80%、46.67%、73.08%、45.11%。结果由明显的状态和任务差异驱动，不是稳定泛化改善。

### 16.3 分析与 Gate 判定

第 15 节的 visual 改善没有在多初始状态 rollout 上复现。Potential 未显著超过 cosine，regression recall 从 75.47% 降为 50%；correct goal 也没有优于 masked、permuted 或 initial-state goal。Direct 几乎退化为 forward predictor，regression recall 只有 12.5%。

因此，以下 Gate 检查均失败：

1. potential 相对 cosine 的 CI 下界大于 0：失败；
2. regression recall 至少 60%：失败；
3. correct goal 相对 masked goal 的 CI 下界大于 0：失败；
4. correct goal 相对 initial-state hard goal 的 CI 下界大于 0：失败。

“每 task 至少三个 initial states”已经满足，但“独立审核 challenge labels”仍未满足。当前状态为 **MULTISTATE_HEURISTIC_VALIDATION / visual heads No-Go**。这个判定否定当前 pooled feature、小样本 head 与 goal conditioning 组合，不等价于证明所有 visual representation 或所有 goal-conditioned model 不可行。

相机方向经复核与 HDF5 训练输入一致；完整重放未出现 mismatch；v3 teacher、Qwen 与 3 个冻结 smoke heads 的哈希未变化。没有发现能够解释性能下降的输入翻转、权重漂移或窗口错位。

### 16.4 后续决策前的标签鲁棒性检查

多状态标签仍来自既有单步 physics heuristic。为避免把阈值附近的轻微运动误当作机制反证，下一轮固定全部预测，重新评估不同 approach/transport 阈值和只保留 grasp、success 或大幅距离变化的高置信子集。若 No-Go 在这些标签变体下保持，停止当前 visual-goal head；若结论随阈值翻转，先建立独立审核 challenge set，不进入 Qwen 或 fusion。

输出目录：`logs/mi_reward/v4_decision_gate/gate1_multistate_v1/`。其中 `results.json` 保存完整聚合与 cluster 指标，`review_candidates.jsonl` 保存 86 个预测错误或 goal 符号变化窗口，`run.log` 保存 `tmux test:v4-multistate` 的执行日志，退出码为 0。

## 17. 标签阈值鲁棒性：No-Go 未由单一 deadband 导致

本轮固定多初始状态实验的 240 个预测，不重训、不重新选择模型，只通过完整动作重放改变 approach/transport distance deadband。每个阈值都重新计算 P/U/N，并采用同样的 task→initial-state 层级 paired bootstrap。

| 阈值倍数 | P/U/N | Potential BA | Potential regression recall | Potential−cosine 95% CI | Correct−masked 95% CI |
|---|---:|---:|---:|---:|---:|
| 0.5× | 130/71/39 | 57.82% | 56.41% | [−9.40, +24.37] | [−7.87, +3.10] |
| 1× | 125/83/32 | 54.60% | 50.00% | [−16.37, +22.09] | [−9.15, +0.99] |
| 2× | 122/91/27 | 58.11% | 55.56% | [−18.84, +33.13] | [−9.04, +1.17] |
| 3× | 114/101/25 | 56.70% | 52.00% | [−20.68, +25.13] | [−9.78, +1.22] |
| 5× | 101/119/20 | 57.67% | 50.00% | [−23.43, +27.68] | [−10.71, +2.60] |

结论没有随合理 deadband 范围翻转：Potential 从未显著超过 cosine，regression recall 从未达到 60%，correct goal 相对 masked goal 的区间始终跨零且均值为负。因而当前 visual-goal No-Go 不能归因于单一阈值设置。

只保留 success acquisition、grasp acquisition 和 grasp loss 等 authoritative 事件时，240 个窗口只有 12 个 Positive、0 个 Negative、228 个 Unclear。这个子集无法估计 regression recall；它揭示的主要数据缺口是没有足够的 slip/drop/grasp-loss challenge，而不是 visual head 已在这些事件上被证明失败。已保存 56 个 3× deadband 下的高置信 potential 错误窗口供独立审核。

状态为 **visual-goal No-Go 保持**。后续停止为这版 goal conditioning 调参，进入 Experiment C：在相同小 head 和相同 P/U/N 上，区分 executed action、LAM transition latent 与 explicit visual state pair 的有效性。标签鲁棒性输出位于 `logs/mi_reward/v4_decision_gate/label_robustness_v2/`；`label_robustness_v1/` 的 bootstrap 缺失类别失败记录保留，v2 对该情况报告 null 而不伪造指标。

## 18. Experiment C：executed action、transition latent 与 state pair

### 18.1 受控设计

Experiment C 不使用 goal 输入，也不使用 visual potential。三个候选均以当前双视角 pooled visual state \(z_t\) 为共同输入，并使用相同的 64 维 state encoder、64 维 evidence encoder、融合层和 P/U/N classifier；唯一变化是 evidence：

1. `executed_action`：当前状态与真实执行的 7 维 simulator action \(a_t\)；
2. `transition_latent`：当前状态与冻结 LaWAM 从连续 agentview 两帧提取的 32 维 latent；
3. `state_pair`：当前状态与显式视觉差分 \(z_{t+1}-z_t\)。

训练使用 task 0–4 的 `demo_0/1`，测试 `demo_10`，并固定 3 个 seed、500 steps、class-balanced cross entropy。之后所有九个 head 冻结，直接评估第 16 节的 15 个 initial states、30 条 rollout、240 个窗口；rollout 不进入训练或选择。每个 rollout transition latent 只由该 transition 的两帧 agentview 提取，不把非连续采样帧连接成伪动作。

### 18.2 缓存 held-out 与 rollout 结果

| Evidence | `demo_10` BA | `demo_10` regression recall | 多状态 rollout BA | 多状态 rollout regression recall |
|---|---:|---:|---:|---:|
| Executed action | 76.30% | 56.60% | 51.92% | 6.25% |
| LaWAM transition latent | 70.52% | 45.28% | 51.12% | 6.25% |
| Visual state pair | 76.77% | 58.49% | 57.77% | 18.75% |

缓存 `demo_10` 上，transition latent 相对 executed action 的 paired task CI 为 `[−9.22, −2.76]` 个百分点，说明它在这个窄分布中更差；state pair 相对 executed action 的 CI 为 `[−4.40, +3.87]`，无明确差异。

多初始状态 rollout 的层级 bootstrap 结果为：

- transition latent 相对 executed action：−0.98 个百分点，95% CI `[−8.48, +4.84]`；
- state pair 相对 executed action：+5.60 个百分点，95% CI `[−5.12, +15.84]`；
- transition latent 相对 state pair：−6.56 个百分点，95% CI `[−15.51, +1.76]`。

### 18.3 解释与决策

三个 evidence 都在缓存 held-out demo 上表现较高，但均没有稳定迁移到多初始状态 rollout。实际 action 与 transition latent 在 rollout 上几乎总是预测 forward；state pair 的 regression recall 较高，但仍只有 18.75%，且相对 executed action 的区间跨零。

因此，Experiment C **没有证据支持**以下任一主张：

- LaWAM transition latent 比实际 action 更有方向信息；
- transition latent 是可靠的 outcome-aware transition evidence；
- state pair 已足够作为可泛化的 rollout regression reward。

此处失败不应解释为 executed action 本身“无用”：它在 offline demo 上与 state pair 接近，说明训练/测试分布过窄；跨 initial-state rollout 的共同失效更像是困难 Negative 覆盖、视觉/物理事件可辨识性和分布迁移问题。当前不选择任何一个分支进入 Qwen 蒸馏、fusion 或 RL。

下一步优先补充独立 challenge 数据，特别是 grasp loss、slip、drop、wrong-object、wrong-target 和 recovery；每类应包含多 initial states、双视角帧、物理记录与独立审核理由。只有在这类数据上，最简单的 privileged temporal classifier 能稳定达到 regression recall 门槛，才重新讨论视觉蒸馏或 transition representation。

实现与证据：`eval/libero/probe_v4_transition_evidence.py`、`eval/libero/run_v4_transition_evidence.sh`、`logs/mi_reward/v4_decision_gate/transition_evidence_v1/`。该目录保存 9 个 probe checkpoint、normalizers、缓存与 rollout 总结、240 个逐窗口分数、完整结果和退出码 0。v3 teacher 与 Qwen 权重在实验前后未修改。
