# MiniMind Track Todo：From-Scratch Tiny LM 的 Tool-Use SFT 与 Agentic RL 边界实验

## 0. Track 定位

这个 MiniMind Track 不是为了把 MiniMind 训练成一个强大的通用聊天模型，而是作为 WebGym-RL 主项目的底层训练补充线。

核心定位：

> 基于自己完成的小规模 MiniMind pretrain checkpoint，继续完成 tokenizer / special tokens 改造、tool-use SFT、小规模 GRPO，并与 Qwen 0.5B/0.6B track 对比，分析 from-scratch tiny LM 在 Agentic RL / Tool-Use 任务中的能力边界。

一句话概括：

```text
Qwen track 负责效果；MiniMind track 负责深度。
```

---

## 1. 和 WebGym-RL 主项目的关系

WebGym-RL 主项目：

```text
WebGym-RL
├── 本地网页环境
├── 工具接口
├── expert trajectories
├── verifier reward
├── SFT / GRPO
└── Base / SFT / GRPO eval
```

MiniMind Track 复用 WebGym-RL 的：

```text
任务数据
expert trajectories
工具调用格式
verifier reward
eval benchmark
metrics
```

MiniMind Track 自己负责：

```text
MiniMind pretrain checkpoint 整理
tokenizer / special tokens 改造
tool-use SFT
小规模 GRPO
和 Qwen track 对比
能力边界分析
```

---

## 2. 总流程

```text
整理 MiniMind pretrain
→ 检查 tokenizer 和 checkpoint
→ 加入 tool-use special tokens
→ 构造 MiniMind 格式 SFT 数据
→ MiniMind tool-use SFT
→ MiniMind SFT eval
→ 接入 WebGym-RL verifier reward
→ MiniMind 小规模 GRPO
→ MiniMind SFT vs GRPO 对比
→ MiniMind vs Qwen 对比
→ 写能力边界分析
```

---

## 3. 推荐目录结构

可以放在 WebGym-RL repo 里面，也可以作为独立目录。

推荐放在主项目下：

```text
WebGym-RL/
├── minimind_track/
│   ├── README.md
│   ├── configs/
│   │   ├── pretrain_config.yaml
│   │   ├── sft_config.yaml
│   │   └── grpo_config.yaml
│   │
│   ├── checkpoints/
│   │   ├── pretrain/
│   │   ├── sft/
│   │   └── grpo/
│   │
│   ├── tokenizer/
│   │   ├── tokenizer_config.json
│   │   └── special_tokens.json
│   │
│   ├── data/
│   │   ├── sft_train.jsonl
│   │   ├── sft_eval.jsonl
│   │   ├── grpo_prompts.jsonl
│   │   └── eval_tasks.jsonl
│   │
│   ├── scripts/
│   │   ├── convert_webgym_to_minimind_sft.py
│   │   ├── add_tool_tokens.py
│   │   ├── train_sft_minimind.py
│   │   ├── train_grpo_minimind.py
│   │   └── eval_minimind.py
│   │
│   ├── outputs/
│   │   ├── eval_reports/
│   │   ├── figures/
│   │   └── failure_cases/
│   │
│   └── notes/
│       ├── pretrain_notes.md
│       ├── sft_notes.md
│       ├── grpo_notes.md
│       └── boundary_analysis.md
```

---

# Version 0：Pretrain 整理版

## 目标

把你已经完成的小 MiniMind pretrain 结果整理成可复现、可展示、可继续训练的状态。

这个版本不追求效果，只追求：

```text
checkpoint 可加载
tokenizer 可用
loss 曲线可展示
训练配置可复现
```

---

## V0 Todo

### 0.1 整理 pretrain checkpoint

- [ ] 找到已经训练好的 MiniMind pretrain checkpoint
- [ ] 确认 checkpoint 可以成功加载
- [ ] 记录模型参数量
- [ ] 记录训练 token 数
- [ ] 记录训练数据来源
- [ ] 记录训练时长和硬件
- [ ] 保存最终 loss
- [ ] 保存 loss 曲线截图或日志

建议记录格式：

```text
model_size: 例如 26M / 64M
training_tokens: xxxM tokens
training_data: xxx
hardware: RTX 3090 / A40 / MIG40 / etc.
train_time: x hours
final_loss: x.xx
```

---

### 0.2 写 pretrain notes

创建：

```text
minimind_track/notes/pretrain_notes.md
```

内容包括：

- [ ] 为什么做 MiniMind pretrain
- [ ] 模型结构简介
- [ ] tokenizer 简介
- [ ] 数据来源
- [ ] 训练配置
- [ ] loss 曲线
- [ ] 已知问题
- [ ] 下一步如何接 tool-use SFT

---

### 0.3 基础生成测试

测试 pretrain-only 模型是否能正常生成文本。

Todo：

- [ ] 写 `eval_minimind_generation.py`
- [ ] 输入简单 prompt
- [ ] 检查输出是否正常
- [ ] 保存 10 条生成样例

测试 prompt 示例：

```text
你好，请介绍一下你自己。
什么是人工智能？
请写一个 Python 函数计算两个数的和。
```

完成标准：

```text
模型可以加载
模型可以生成
生成不报错
结果可保存
```

---

# Version 1：Tool Special Tokens 改造版

## 目标

让 MiniMind tokenizer 和模型格式适配 WebGym-RL 的 tool-use 数据。

重点不是让模型马上变强，而是让它稳定学习工具调用结构。

---

## V1 Todo

### 1.1 设计 tool-use special tokens

建议加入：

```text
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<answer>
</answer>
<observe>
</observe>
```

可选加入：

```text
<think>
</think>
<invalid>
</invalid>
```

Todo：

- [ ] 确定 special token 列表
- [ ] 写入 `special_tokens.json`
- [ ] 更新 tokenizer
- [ ] 确认 token id 不冲突
- [ ] 保存 tokenizer config

---

### 1.2 Resize embedding

加入 special tokens 后，需要调整模型 embedding。

Todo：

- [ ] 加载 MiniMind pretrain checkpoint
- [ ] 加载新 tokenizer
- [ ] resize token embeddings
- [ ] 初始化新增 token embedding
- [ ] 保存新的 init checkpoint

注意：

```text
新增 token 的 embedding 可以随机初始化，也可以用已有相近 token 的均值初始化。
```

---

### 1.3 Special token ablation 设计

后面可以做一个小 ablation：

```text
不加 special tokens
加入 tool special tokens
加入 tool + answer special tokens
```

第一版不一定马上做，但要先保留配置。

Todo：

- [ ] 创建 no-special-token 配置
- [ ] 创建 tool-special-token 配置
- [ ] 创建 full-special-token 配置

---

# Version 2：MiniMind Tool-Use SFT 版

## 目标

用 WebGym-RL 生成的 expert trajectories 对 MiniMind 做 tool-use SFT。

目标是让 MiniMind 学会：

```text
按照工具调用格式输出
根据 observation 继续下一步动作
在简单网页任务中提交答案
```

---

## V2 Todo

### 2.1 复用 WebGym-RL expert trajectories

输入来自 WebGym-RL：

```text
outputs/trajectories/expert_train.jsonl
outputs/trajectories/expert_eval.jsonl
```

Todo：

- [ ] 读取 WebGym-RL expert trajectories
- [ ] 转成 MiniMind SFT 格式
- [ ] 检查每条样本的 tool call JSON
- [ ] 过滤过长样本
- [ ] 保存为 `sft_train.jsonl` 和 `sft_eval.jsonl`

建议第一版数据量：

```text
train: 500–2000 条
eval: 100–300 条
```

---

### 2.2 SFT 数据格式

建议格式：

```text
<user>
找到价格低于100美元且评分最高的蓝牙耳机。
</user>
<tool_call>{"name": "open_page", "arguments": {"page_id": "shop_home_001"}}</tool_call>
<tool_response>Opened shop_home_001.</tool_response>
<tool_call>{"name": "click", "arguments": {"element_id": "filter_price_under_100"}}</tool_call>
<tool_response>Filtered items under 100 dollars.</tool_response>
<tool_call>{"name": "submit_answer", "arguments": {"answer": "SoundCore A20"}}</tool_call>
```

Todo：

- [ ] 确定 MiniMind 训练模板
- [ ] 写格式转换脚本
- [ ] 随机抽查 50 条样本
- [ ] 检查标签 mask 是否正确

---

### 2.3 MiniMind SFT 训练

Todo：

- [ ] 写 `train_sft_minimind.py`
- [ ] 加载 pretrain checkpoint
- [ ] 加载新 tokenizer
- [ ] 设置 max_seq_len
- [ ] 设置 batch size 和 gradient accumulation
- [ ] 跑 1–3 epoch
- [ ] 保存 SFT checkpoint
- [ ] 记录 loss 曲线

建议配置：

```text
max_seq_len: 1024 或 2048
batch_size: 根据显存设置
epoch: 1–3
learning_rate: 从较小值开始
```

完成标准：

```text
SFT loss 正常下降
checkpoint 可以加载
模型能输出 tool_call 格式
```

---

### 2.4 SFT 后基础评估

对比：

```text
MiniMind Pretrain-only
MiniMind + SFT
```

指标：

```text
Tool Call Format Accuracy
Valid JSON Rate
Valid Tool Name Rate
Invalid Tool Call Rate
Simple Task Success Rate
```

Todo：

- [ ] 写 MiniMind rollout runner
- [ ] 跑 pretrain-only eval
- [ ] 跑 SFT eval
- [ ] 输出 eval report
- [ ] 保存失败案例

完成标准：

```text
SFT 后 tool call 格式正确率明显高于 pretrain-only
SFT 后可以完成一部分 Level 1 简单任务
```

---

# Version 3：MiniMind 小规模 GRPO 版

## 目标

在 MiniMind + SFT 的基础上，用 WebGym-RL verifier reward 做小规模 GRPO。

目标不是追求高成功率，而是验证：

```text
rule-based reward 是否能降低 invalid tool call
GRPO 是否能提升简单任务成功率
GRPO 是否能减少无效操作
```

---

## V3 Todo

### 3.1 选择 RL 任务范围

MiniMind 模型较小，第一版只做简单任务。

建议：

```text
Level 1：单页面信息查找
Level 2：筛选 + 排序
```

暂时不要做：

```text
复杂多页面跳转
长链路推理
跨多个页面对比
需要复杂自然语言理解的任务
```

Todo：

- [ ] 从 WebGym-RL eval tasks 中筛选 Level 1 / Level 2
- [ ] 准备 100–500 条 RL prompts
- [ ] 准备 100–200 条固定 eval tasks

---

### 3.2 复用 WebGym-RL reward

Reward 第一版保持简单：

```text
reward = 0
+ 0.3 tool call 格式正确
+ 0.2 tool name 正确
+ 0.3 最终答案正确
- 0.2 invalid action
- 0.05 每次多余操作
```

Todo：

- [ ] 接入 WebGym-RL verifier
- [ ] 实现 MiniMind reward wrapper
- [ ] 保存 reward breakdown
- [ ] 检查 reward 是否合理

---

### 3.3 MiniMind GRPO 训练

建议配置：

```text
group_size: 4
max_steps: 4–6
max_response_tokens: 512
RL prompts: 100–500
training_steps: 小规模即可
```

Todo：

- [ ] 写 `train_grpo_minimind.py`
- [ ] 加载 MiniMind SFT checkpoint
- [ ] 对同一个 prompt 采样多条 trajectory
- [ ] 计算每条 trajectory reward
- [ ] 计算 group relative advantage
- [ ] 更新模型
- [ ] 保存 GRPO checkpoint
- [ ] 记录 reward 曲线

完成标准：

```text
训练可以稳定运行
reward 没有明显崩坏
checkpoint 可以加载评估
```

---

### 3.4 MiniMind GRPO Eval

对比：

```text
MiniMind Pretrain-only
MiniMind + SFT
MiniMind + SFT + GRPO
```

指标：

```text
Tool Call Format Accuracy
Valid JSON Rate
Invalid Tool Call Rate
Task Success Rate
Average Step Count
```

Todo：

- [ ] 固定 eval set
- [ ] 跑 pretrain-only
- [ ] 跑 SFT
- [ ] 跑 SFT + GRPO
- [ ] 输出表格
- [ ] 保存失败案例
- [ ] 画指标对比图

理想结果：

```text
SFT 明显提升格式稳定性
GRPO 进一步降低 invalid action 或提升简单任务成功率
```

即使 Task Success Rate 提升不大，也可以分析：

```text
极小模型的 base capability 限制了复杂 Agentic RL 效果
GRPO 更容易改善格式和操作效率，而不是凭空补齐理解能力
```

---

# Version 4：MiniMind vs Qwen 对比版

## 目标

把 MiniMind Track 和 WebGym-RL 主线打通，做一个有说服力的模型能力对比。

核心问题：

> From-scratch tiny LM 和成熟开源小模型在同一个 Agentic RL 环境中的差距在哪里？

---

## V4 Todo

### 4.1 统一 eval benchmark

所有模型使用同一套 WebGym-RL eval tasks。

对比模型：

```text
MiniMind Pretrain-only
MiniMind + SFT
MiniMind + SFT + GRPO
Qwen 0.5B/0.6B Base
Qwen 0.5B/0.6B + SFT
Qwen 0.5B/0.6B + SFT + GRPO
```

Todo：

- [ ] 固定 eval set
- [ ] 固定 tool interface
- [ ] 固定 reward function
- [ ] 固定 max steps
- [ ] 固定 generation config

---

### 4.2 指标对比

指标：

```text
Task Success Rate
Tool Call Format Accuracy
Valid JSON Rate
Invalid Tool Call Rate
Average Step Count
Level 1 Success Rate
Level 2 Success Rate
Level 3 Success Rate
```

Todo：

- [ ] 生成总表
- [ ] 按任务难度分层统计
- [ ] 画柱状图
- [ ] 画 line chart 或 radar chart
- [ ] 保存所有 eval reports

---

### 4.3 能力边界分析

重点分析：

```text
MiniMind 能否学会工具调用格式？
MiniMind 是否能完成 Level 1 简单任务？
MiniMind 在 Level 2/3 失败的主要原因是什么？
Qwen 相比 MiniMind 的优势来自哪里？
GRPO 对两个模型的提升是否不同？
```

Todo：

- [ ] 收集 MiniMind 失败案例
- [ ] 收集 Qwen 失败案例
- [ ] 比较错误类型分布
- [ ] 写 `boundary_analysis.md`

可能结论：

```text
MiniMind 通过 SFT 可以快速学会工具调用格式，但在多步任务规划和复杂指令理解上受限。
GRPO 能降低 invalid tool call 和无效操作，但无法完全弥补 pretraining 能力不足。
成熟开源小模型在同一 verifier reward 下更容易从 RL 中获得任务成功率提升。
```

---

# Version 5：MiniMind 研究味增强版

## 目标

如果时间充足，进一步探索 tokenizer、模型规模和 RL 效果之间的关系。

这个版本是加分项，不是第一阶段必须完成。

---

## V5 Todo

### 5.1 Special Token Ablation

对比：

```text
MiniMind-SFT-no-special-token
MiniMind-SFT-tool-special-token
MiniMind-SFT-full-special-token
```

指标：

```text
Tool Call Format Accuracy
Valid JSON Rate
Training Loss
Invalid Tool Call Rate
Task Success Rate
```

Todo：

- [ ] 准备三套 tokenizer / template
- [ ] 跑三组小规模 SFT
- [ ] 评估 tool call 格式稳定性
- [ ] 写 ablation 分析

---

### 5.2 Model Size Ablation

如果你有多个 MiniMind 尺寸，可以比较：

```text
MiniMind-very-small
MiniMind-small
MiniMind-MoE
```

Todo：

- [ ] 统一 SFT 数据
- [ ] 统一 eval set
- [ ] 比较不同参数规模下的 tool-use 能力
- [ ] 分析模型规模对 Agentic RL 的影响

---

### 5.3 RL Algorithm Ablation

如果 MiniMind 代码支持，可以比较：

```text
SFT
SFT + DPO
SFT + GRPO
SFT + CISPO
```

Todo：

- [ ] 构造 preference pairs
- [ ] 跑 DPO baseline
- [ ] 跑 GRPO
- [ ] 跑 CISPO
- [ ] 比较不同后训练方法

---

# 最小可行 Todo 总表

如果想马上开工，就按这个清单做：

```text
[ ] 整理 MiniMind pretrain checkpoint
[ ] 记录 pretrain 配置、loss、硬件、训练数据
[ ] 测试 pretrain-only 生成能力
[ ] 设计 tool-use special tokens
[ ] 更新 tokenizer 并 resize embedding
[ ] 从 WebGym-RL expert trajectories 构造 MiniMind SFT 数据
[ ] 跑 MiniMind tool-use SFT
[ ] 评估 MiniMind Pretrain-only vs MiniMind-SFT
[ ] 接入 WebGym-RL verifier reward
[ ] 筛选 Level 1 / Level 2 RL tasks
[ ] 跑 MiniMind 小规模 GRPO
[ ] 评估 MiniMind-SFT vs MiniMind-SFT-GRPO
[ ] 和 Qwen 0.5B/0.6B track 做统一 eval 对比
[ ] 写 MiniMind 能力边界分析
[ ] 更新 WebGym-RL README，把 MiniMind Track 写进去
```

---

# 推荐完成顺序

## 第一阶段：整理已有 pretrain

```text
目标：让 MiniMind pretrain checkpoint 可复现、可加载、可展示。
```

必须完成：

```text
checkpoint 整理
配置记录
loss 曲线
基础生成测试
```

---

## 第二阶段：Tool-Use SFT

```text
目标：让 MiniMind 学会 WebGym-RL 工具调用格式。
```

必须完成：

```text
special tokens
SFT 数据转换
SFT 训练
Pretrain-only vs SFT eval
```

---

## 第三阶段：小规模 GRPO

```text
目标：验证 verifier reward 对 MiniMind 是否有效。
```

必须完成：

```text
Level 1 / Level 2 RL tasks
reward wrapper
GRPO 训练
SFT vs GRPO eval
```

---

## 第四阶段：MiniMind vs Qwen 对比

```text
目标：形成简历和面试中的核心分析。
```

必须完成：

```text
统一 eval set
统一 metrics
对比表格
失败案例分析
能力边界分析
```

---

# 每个版本对应价值

| 版本 | 能否写简历 | 价值 |
|---|---|---|
| Version 0 | 可以作为背景 | 证明你做过 from-scratch pretrain |
| Version 1 | 可以作为技术细节 | 体现 tokenizer / special token 理解 |
| Version 2 | 可以写 | 有 tool-use SFT |
| Version 3 | 推荐写 | 有 MiniMind Agentic RL 闭环 |
| Version 4 | 强烈推荐写 | 和 Qwen 对比，形成能力边界分析 |
| Version 5 | 加分 | 有研究味和 ablation |

---

# 简历描述草稿

## 中文版

**MiniMind Track：从零训练小模型的 Tool-Use SFT 与 Agentic RL 边界实验**

- 基于自训练 MiniMind 小模型整理 pretrain checkpoint、训练配置与 loss 曲线，并在原有 tokenizer 基础上加入 `<tool_call>`、`<tool_response>`、`<answer>` 等 tool-use special tokens，适配 WebGym-RL 多轮工具调用任务格式。
- 复用 WebGym-RL 中规则专家生成的网页导航 trajectories，对 MiniMind 进行 tool-use SFT，使模型学习标准化工具调用格式、工具响应理解和简单网页任务答案提交流程。
- 接入 WebGym-RL 的 rule-based verifier，将工具调用格式、合法工具名、最终答案正确性、无效操作和步数惩罚转化为 reward，并在 MiniMind-SFT checkpoint 上进行小规模 GRPO 后训练。
- 对比 MiniMind Pretrain-only、MiniMind-SFT、MiniMind-SFT-GRPO 以及 Qwen 0.5B/0.6B 系列模型在 Tool Call Format Accuracy、Valid JSON Rate、Invalid Tool Call Rate、Task Success Rate 等指标上的表现，分析 from-scratch tiny LM 在 Agentic RL 任务中的能力边界。

## English Version

**MiniMind Track: Tool-Use SFT and Agentic RL Boundary Study for a From-Scratch Tiny LM**

- Organized a self-trained MiniMind pretraining checkpoint with reproducible training configurations, loss curves, and generation tests; extended the tokenizer with tool-use special tokens such as `<tool_call>`, `<tool_response>`, and `<answer>` for WebGym-RL style multi-turn tool interactions.
- Reused rule-based expert trajectories from WebGym-RL to perform tool-use SFT, enabling the tiny LM to learn structured tool-call generation, observation-conditioned action prediction, and answer submission for simple web navigation tasks.
- Integrated the WebGym-RL rule-based verifier as a programmable reward function, combining tool-call format validity, valid tool names, final answer correctness, invalid action penalties, and step penalties; applied small-scale GRPO on top of the MiniMind-SFT checkpoint.
- Compared MiniMind Pretrain-only, MiniMind-SFT, MiniMind-SFT-GRPO, and Qwen 0.5B/0.6B baselines on Tool Call Format Accuracy, Valid JSON Rate, Invalid Tool Call Rate, and Task Success Rate, analyzing the capability boundary of from-scratch tiny LMs in agentic RL tasks.

---

# 面试讲法

可以这样讲：

> 我的主项目是 WebGym-RL，用 Qwen 小模型验证可验证网页导航任务上的 SFT + GRPO 后训练效果。除此之外，我还做了一个 MiniMind Track，因为我之前已经从零训练了一个 MiniMind 小模型。我把它接入同一套 WebGym-RL 环境，用相同的 expert trajectories、tool interface、verifier reward 和 eval benchmark，继续做 tool-use SFT 和小规模 GRPO。这样我可以比较 from-scratch tiny LM 和成熟开源小模型在 agentic RL 中的差异。实验上，MiniMind 通过 SFT 可以明显提升 tool call 格式稳定性，但在多步任务规划和复杂指令理解上受限；GRPO 可以进一步降低 invalid action 或提升简单任务成功率，但不能完全弥补 pretraining 能力不足。这个 track 主要展示我对 pretraining、tokenizer、SFT、RL 后训练以及模型能力边界的完整理解。

---

# 最终建议

MiniMind Track 的目标不是追求最强效果，而是回答一个成熟的问题：

```text
在同一个可验证 Agentic RL 环境中，from-scratch tiny LM 能学到什么，不能学到什么？
```

这个问题比单纯说“我训练了一个 MiniMind”更有价值，也能很好地和 WebGym-RL 主项目形成组合。

