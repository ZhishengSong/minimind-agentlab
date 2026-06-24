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

当前进度（2026-06-24）：Pretrain V0、tool-use tokenizer adaptation 和 MiniMind tool-use SFT V1 均已完成。服务器独立完成 Epoch1/2/3 SFT；同一 eval100 上 target exact match 依次为 `72% / 96% / 100%`。Epoch3 在完整 637-example next-action eval 上达到 wrapper/JSON/tool-name `100%`，argument/target exact match `628/637 = 98.59%`，因此选为最终 SFT checkpoint。服务器 checkpoint 已下载到本地并通过 SHA256 `77f29f3f5fd812e2fa05ba3afb6af85b0d319dad2de6e7fbff52e72d35e87ce6` 校验；本地 CPU 加载/生成 smoke 和最终 SFT V1 报告也已完成。固定 pretrain val 回归检查显示 loss 从 `2.252283` 升至 `2.569051`、perplexity 从 `9.509420` 升至 `13.053436`，说明专业化带来可测但非崩坏式的通用能力退化。Epoch2 全量 eval、WebNav-RL rollout、best-of-N 和 GRPO 均为可选后续研究。

### 0.1 整理 pretrain checkpoint

- [x] 找到已经训练好的 MiniMind pretrain checkpoint
- [x] 确认 checkpoint 可以成功加载
- [x] 记录模型参数量
- [ ] 记录训练 token 数
- [x] 记录训练数据来源
- [ ] 记录训练时长和硬件
- [x] 保存最终 loss
- [x] 保存 loss 曲线截图或日志

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

- [x] 为什么做 MiniMind pretrain
- [x] 模型结构简介
- [x] tokenizer 简介
- [x] 数据来源
- [x] 训练配置
- [x] loss 曲线
- [x] 已知问题
- [x] 下一步如何接 tool-use SFT

---

### 0.3 基础生成测试

测试 pretrain-only 模型是否能正常生成文本。

Todo：

- [x] 写固定 prompt generation suite
- [x] 输入简单 prompt
- [x] 检查输出是否正常
- [x] 保存 10 条生成样例

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

- [x] 确定 special token 列表
- [x] 写入 `configs/tool_special_tokens.json`
- [x] 更新 tokenizer
- [x] 确认 token id 不冲突
- [x] 保存 tokenizer config

实际结果：

```text
已有单 token:
<tool_call> id 21
</tool_call> id 22
<tool_response> id 23
</tool_response> id 24
<think> id 25
</think> id 26

新增单 token:
<answer> id 6400
</answer> id 6401
<observe> id 6402
</observe> id 6403
```

---

### 1.2 Resize embedding

加入 special tokens 后，需要调整模型 embedding。

Todo：

- [x] 加载 MiniMind pretrain checkpoint
- [x] 加载新 tokenizer
- [x] resize token embeddings
- [x] 初始化新增 token embedding
- [x] 保存新的 init checkpoint

注意：

```text
新增 token 的 embedding 可以随机初始化，也可以用已有相近 token 的均值初始化。
```

实际输出：

```text
outputs/tooluse_init/tokenizer/
outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt
reports/tooluse_init_report.json
docs/step_10_tooluse_init.md
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

- [x] 读取 WebGym-RL expert trajectories
- [x] 转成 MiniMind SFT 格式
- [x] 检查每条样本的 tool call JSON
- [x] 过滤过长样本
- [x] 保存为 MiniMind next-action SFT train/eval JSONL

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

- [x] 确定 MiniMind 训练模板
- [x] 写格式转换脚本
- [ ] 随机抽查 50 条样本
- [x] 检查标签 mask 是否正确

---

### 2.3 MiniMind SFT 训练

Todo：

- [x] 写 `train_sft_minimind.py`
- [x] 加载 pretrain checkpoint
- [x] 加载新 tokenizer
- [x] 设置 max_seq_len
- [x] 设置 batch size 和 gradient accumulation
- [x] 跑 1–3 epoch
- [x] 保存 SFT smoke checkpoint
- [x] 记录 smoke loss 曲线
- [x] 跑 200-step local SFT
- [x] 评估 next-action format accuracy
- [x] 在服务器完成 Epoch1/2/3 独立 SFT
- [x] 选择 Epoch3 作为最终 SFT checkpoint
- [x] 完成 Epoch3 全量 637-example next-action eval

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

当前 200-step 结果：

```text
wrapper_ok: 100.0%
json_ok: 100.0%
valid_tool_name: 100.0%
tool_name_match: 100.0%
target_exact_match: 68.6%
```

当前约 1 epoch rollout 结果：

```text
eval20 success: 2/20
submitted_rate: 100.0%
format_errors: 0
主要问题: click/answer 参数塌缩
```

服务器 SFT checkpoint 对比：

```text
Epoch1 eval100 target exact match: 72%
Epoch2 eval100 target exact match: 96%
Epoch3 eval100 target exact match: 100%
Epoch3 eval637 target exact match: 98.59% (628/637)
```

当前阶段必要性分级：

```text
已完成且必要：Epoch3 全量 eval637、checkpoint 下载、SHA256 校验
已完成收尾：本地 checkpoint load/generation smoke、最终 SFT 报告
已完成建议项：SFT 前后固定 pretrain val loss 对比
可选：Epoch2 全量 eval637、跨项目 rollout、best-of-N、GRPO
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

- [x] 写可选的 MiniMind rollout adapter
- [ ] 跑 pretrain-only eval
- [x] 跑 SFT next-action format eval
- [x] 输出 next-action eval report
- [x] 跑 rollout-level SFT eval
- [x] 输出 rollout-level eval report
- [x] 保存 rollout failure cases

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
[x] 整理 MiniMind pretrain checkpoint
[x] 记录 pretrain 配置、loss、硬件、训练数据
[x] 测试 pretrain-only 生成能力
[x] 设计 tool-use special tokens
[x] 更新 tokenizer 并 resize embedding
[x] 从 WebGym-RL expert trajectories 构造 MiniMind SFT 数据
[x] 跑 MiniMind tool-use SFT
[x] 完成 MiniMind-SFT next-action full eval
[x] 完成 Pretrain/tool-use-init vs MiniMind-SFT pretrain-loss 回归对比
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

