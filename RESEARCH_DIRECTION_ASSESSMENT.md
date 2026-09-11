# MI Directional Potential 研究方向评估

## 结论

当前研究的大方向值得继续，但需要收缩并重构技术主张。

值得保留的核心方向是：利用训练期可见的 privileged robot physics 构造方向性监督，再将该监督蒸馏到只依赖视觉历史和任务语言的 VLM reward model。这一问题真实、具有机器人强化学习价值，也能围绕 physical aliasing、near-miss、regression 和 abstention 建立清楚的实验故事。

当前不宜继续把“visual PMI + conditional action PMI 的 chain-rule 分解”作为已经成立的核心贡献。现有采样没有识别出文档声明的 PMI，组合分数没有优于 action-only 分数，Qwen 在新 rollout 上也尚未学会可靠识别 regression。因此，当前版本应被视为工程链路已经打通、核心科学假设尚未通过验证。

综合判断：

- 研究大方向：7/10，可以继续。
- 当前 MI 数学主张：3/10，需要重构。
- 当前 reward model 的部署成熟度：不足，暂不进入主线闭环 RL。

## 建议保留的部分

1. **Privileged physical teacher**：训练阶段利用 contact、grasp、kinematics、task relations、outcome 和 feasibility 产生监督。
2. **P/U/N 与 abstention**：Positive、Unclear、Negative 比单一连续分数更适合表达证据冲突和不确定性。
3. **双视角时间历史**：agent view、wrist view 与任务语言符合实际部署条件。
4. **Observation-only distillation**：部署模型不读取 privileged state，研究目标清楚。
5. **Physical-aliasing benchmark**：重点评估 grasp/miss、contact/near-contact、placed/over-target、hold/slip、progress/regression 等困难样本。
6. **独立 reward validation**：按 episode、初始状态和任务划分，检验 teacher 与 student 的真实泛化。

## 当前 pipeline 的主要问题

### 1. Visual critic 没有估计所声明的 PMI

文档定义：

\[
\phi(v,g)=\log\frac{p(v,g)}{p(v)p(g)}.
\]

实际训练的正样本是“当前 episode 的状态与当前 episode 的成功终局”，负样本是“同任务另一个 episode 的成功终局”。因此分类器学习的是当前 episode goal 与另一个 episode goal 的构造分布比，而不是上述 joint/product-of-marginals 密度比。

这会允许模型利用 episode identity、背景、相机细节、初始布局等捷径。现有统计与该解释一致：matched-goal logits 与 other-episode-goal logits 可以被极强地区分，但真正用于 progress 的 \(\Delta\phi\) 只有 52.95% balanced accuracy，接近随机方向判断。

相关实现：`mi_reward/training/train_information_teacher_v3.py::build_visual_samples()`。

### 2. Action critic 更接近 privileged direction classifier

当前 action 正样本只来自 `direction > 0` 的 transition，负动作从 `direction <= 0` 的 transition 中选择视觉最近邻。它实际估计的更接近：

\[
\log
\frac{q(a\mid v,g,Y=\mathrm{forward})}
     {q_{\mathrm{mine}}(a\mid v,g,Y\leq 0)},
\]

而不是文档声明的：

\[
\log\frac{p(a\mid v,g)}{p(a\mid v)}.
\]

物理方向既参与 action 正负样本构造，又用于评价 action critic 的方向准确率，因而存在循环监督。当前 \(\psi\) 的方向性能说明它学到了 privileged direction 信号，但不能证明它估计了 conditional PMI。

相关实现：`mi_reward/training/train_information_teacher_v3.py::build_action_hard_negatives()`。

### 3. Chain rule 不能直接推出当前方向分数

以下 pointwise chain rule 可以成立：

\[
\operatorname{PMI}(v,a;g)
=
\operatorname{PMI}(v;g)
+
\operatorname{PMI}(a;g\mid v).
\]

但它不能直接推出：

\[
D_t=\gamma\phi(v_{t+1},g)-\phi(v_t,g)+\beta\psi(a_t,v_t,g).
\]

当 \(\gamma=\beta=1\) 时，当前分数可写为：

\[
D_t=
\operatorname{PMI}(v_{t+1};g)
+\operatorname{PMI}(v_t,a_t;g)
-2\operatorname{PMI}(v_t;g),
\]

它不是一个标准的 chain-rule information quantity。

此外，当前 \(a_t^{lat}\) 从 \((o_t,o_{t+1})\) 图像对提取，已经包含下一状态信息。将它与 \(\phi(v_{t+1},g)-\phi(v_t,g)\) 相加会重复使用 transition outcome，不能据此声称减少 double counting。

### 4. Teacher 与 physical heuristic 的评价存在循环依赖

完整 physical gate 将同一个 relation direction 用作 authoritative evidence，评价标签又由该 physical direction 生成。因此 full teacher 的 Positive precision 1.0、false-positive rate 0 等结果主要是规则结构保证的，只能作为 pipeline sanity check，不能作为 teacher 优于简单 privileged heuristic 的独立证据。

### 5. 验证样本的有效独立单位过少

缓存验证集虽然包含 536 个 transition，但只来自 5 条 episode，每个任务只有一条 `demo_10`。相邻窗口高度相关，不能把 536 当作 536 个独立样本。统计报告应以 trajectory、initial state 或 task 为聚类单位，并报告 cluster bootstrap confidence interval。

### 6. Teacher goal 存在训练与部署条件差异

训练标签使用每条成功 episode 自己的未来成功终局作为 goal。对于失败轨迹和在线 rollout，这个 episode-specific future goal 不存在，只能改用其他成功 demo 的 goal。这会改变 teacher 的条件分布，并可能令训练标签依赖学生部署时不可获得的未来信息。

建议固定 task-level goal definition，例如语言目标、独立成功参考集或 goal prototype，并在训练、验证和 rollout 中保持一致。

### 7. Qwen student 当前没有学会 regression

现有 Qwen 训练集只有 1,168 个高度重叠窗口，其中 Negative 为 157 条，却更新约 17.2 亿参数，未使用 class-balanced sampling 或 class-weighted objective。

在 160 个 fresh rollout 窗口上：

- Qwen 只输出 3 个 Negative；
- 对 22 个 physical regression 的 recall 为 0/22；
- 对 37 个 teacher Negative 的 recall 为 1/37；
- 80 个 physical non-forward 窗口中有 41 个被判为 Positive；
- Qwen 对 teacher 的 macro-F1 为 0.33485。

因此当前最紧迫的问题是数据覆盖和 student learning，而不是继续扩展 MI 公式或接入闭环 RL。

### 8. 历史 Robometer 结果不能作为有效证据

旧 ranking 数据中的 150 条 trajectory 得分全部为 0。历史 0.5813 ranking accuracy 受平局与输入顺序影响；quality preference 的 395 对中有 353 个平局，严格胜率只有 8.35%。这些结果已经适合作为 evaluator bug audit，但不能支持 reward ranking 能力。

## 当前数据对核心机制的判断

缓存验证集中排除 neutral transition 后：

| 信号 | 普通准确率 | Balanced accuracy | Regression recall |
|---|---:|---:|---:|
| Cosine difference | 63.10% | 66.04% | 69.81% |
| Visual \(\Delta\phi\) | 51.57% | 52.95% | 54.72% |
| Action \(\psi\) | 80.71% | 68.51% | 52.83% |
| Combined \(D\) | 79.66% | 67.92% | 52.83% |
| Always forward | 88.89% | 50.00% | 0% |

在 477 个 non-neutral transition 上，\(\psi\) 单独正确而 \(D\) 错误的样本有 25 个，\(D\) 正确而 \(\psi\) 错误的样本有 20 个，配对 exact test 的 \(p=0.55\)。当前数据没有显示组合项相对 action-only 的增益。

这意味着当前 visual information contribution 未通过最基本的消融检验。若论文核心贡献仍然写成 visual PMI 与 conditional action PMI 的互补分解，审稿人很可能直接质疑该机制已被自己的 ablation 否定。

## 建议的新论文主线

近期主线建议改为：

> 利用训练期 privileged physical evidence 构造带 abstention 的 directional supervision，蒸馏出仅依赖视觉和语言、能够识别 near-miss 与 regression 的机器人 reward model。

这个主线具有三个优点：

1. 研究问题与当前已实现系统一致；
2. 每个关键主张都能通过独立 benchmark 和消融直接验证；
3. 不依赖当前尚未闭合的 PMI 统计解释。

在该主线中，MI 应暂时降为候选增强项。只有当严格定义的 information estimator 在独立数据上显著超过 privileged heuristic、visual baseline 和 action-only baseline 时，才重新提升为核心贡献。

## 建议的技术路线

### Phase A：建立可信的非 MI 基线

直接从 privileged physics 构造 directional labels：

- `Positive`：明确接近下一任务关系、获得 grasp/contact、进入 success；
- `Negative`：明确远离目标、丢失 grasp/contact、发生失败或不可行 transition；
- `Unclear`：物理变化不足、证据冲突或阶段定义不明确。

训练和验证必须按 episode、initial state 和 task 分组，禁止相邻窗口跨 split。训练时使用 class-balanced sampling，重点补充 regression、near-miss、slip、wrong-object、wrong-orientation 和 recovery。

### Phase B：验证 student 是否能蒸馏 privileged teacher

固定 teacher、goal definition、history length 和 prompt，比较：

- class-balanced accuracy 和 macro-F1；
- Negative recall；
- non-forward false-positive rate；
- near-miss 与 regression 子集；
- seen/unseen task、object、scene；
- trajectory-level success/failure ranking；
- 不同 initial state 上的 cluster confidence interval。

如果 observation-only Qwen 在独立 rollout 上仍不能识别 Negative，应先解决视觉分辨率、窗口语义、负样本覆盖和模型容量问题。

### Phase C：重新引入 information term

可以选择两条路线之一：

1. **放弃严格 PMI 命名**：将现有模块明确称为 privileged contrastive directional critic，承认其优化目标由构造的正负分布定义。
2. **重新设计严格 estimand**：明确 joint、product-of-marginals 和 conditional denominator 的采样；使用 executed action 或独立 action representation，避免从 \(o_{t+1}\) 提取的 latent 与 state delta 重复。

无论选择哪条路线，都必须在同一个独立 challenge set 上比较：

- privileged heuristic；
- temporal order；
- cosine/state-distance；
- visual term；
- action-only term；
- combined information score；
- combined score + physical gate。

## Go/No-Go 标准

在进入闭环 RL 前，至少要求：

1. Qwen 在独立 rollout 上的 Negative recall 和 non-forward false-positive rate 达到可接受水平；
2. teacher 明显优于简单 privileged heuristic，且评价真值不复用 teacher gate 的输入规则；
3. combined information score 在 trajectory/task clustered uncertainty 下显著优于 action-only；
4. success/failure ranking 使用严格平局规则，并在多个 initial states 和真实 policy rollouts 上成立；
5. reward 加入 RL 后不会诱发循环、停滞、视觉捷径或 reward hacking。

若第 3 条失败，应删除 visual-MI/chain-rule 主张；若第 1 条失败，应暂停闭环 RL，继续修正数据和 student；若第 2 条失败，论文贡献应转向 benchmark、distillation 或 uncertainty calibration，而不是 information teacher。

## 最先执行的三个实验

1. **Heuristic-teacher baseline**：不使用 MI，只用独立 privileged directional labels 训练同一 Qwen，并在 fresh rollout challenge set 上评价。
2. **Goal sensitivity 与 shortcut test**：对同一状态替换同任务 goal、跨任务 goal、背景相近 goal 和 goal prototype，检查 \(\phi\) 是否主要识别 episode identity。
3. **Action representation ablation**：比较 executed action、仅 \((o_t,a_t)\) 表示、当前 transition latent，以及 action-only 与 combined score，检验下一状态信息重复使用问题。

完成这三项后，再决定 MI 是核心方法、辅助模块，还是应从论文主线中移除。

