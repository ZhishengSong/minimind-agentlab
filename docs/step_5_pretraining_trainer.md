# Step 5: Pretraining Trainer

This document explains the fifth milestone of MiniMind AgentLab: the base-model pretraining loop.

## Goal

The goal of this step is to connect the model from Step 3 and the data pipeline from Step 4 into a real training loop.

The trainer now supports:

- model initialization from config
- tokenizer and dataset loading
- DataLoader + collator
- AdamW optimizer
- weight decay parameter grouping
- warmup + cosine learning-rate schedule
- gradient accumulation
- gradient clipping
- fp32 / bf16 / fp16 dtype selection
- CPU or CUDA device selection
- NaN / non-finite loss checking
- JSONL metric logging
- token throughput logging
- tiny local smoke training

## Files Involved

```text
scripts/train_pretrain.py
src/train/optim.py
src/train/logger.py
configs/pretrain_tiny.yaml
configs/minimind_tiny.yaml
logs/pretrain_tiny_smoke_metrics.jsonl
```

## Current Training Flow

```text
PretrainConfig
   |
   v
MiniMindConfig
   |
   v
Tokenizer + PretrainDataset + Collator
   |
   v
DataLoader
   |
   v
MiniMindForCausalLM
   |
   v
loss
   |
   v
backward
   |
   v
gradient clipping
   |
   v
optimizer step
   |
   v
scheduler step
   |
   v
JSONL metrics
```

## Tiny Training Config

File:

```text
configs/pretrain_tiny.yaml
```

This config is for local CPU smoke testing.

It points to a tiny model:

```yaml
model_config: configs/minimind_tiny.yaml
```

And uses:

```yaml
device: cpu
dtype: fp32
max_seq_len: 128
batch_size: 2
gradient_accumulation_steps: 1
max_steps: 20
```

This is not meant to train a useful model. It is meant to prove that the training loop is correct.

## Tiny Model Config

File:

```text
configs/minimind_tiny.yaml
```

The tiny model has about 1.21M parameters.

Why add a tiny model?

The 64M model is useful for the real project target, but CPU debugging would be slow. A tiny model lets us run trainer tests quickly on your laptop.

## Optimizer

File:

```text
src/train/optim.py
```

The optimizer is AdamW.

Parameters are split into two groups:

- decay group
- no-decay group

No-decay parameters include:

- bias parameters
- norm parameters
- 1D parameters

Why?

Weight decay is usually useful for large matrix weights, but it is often not applied to bias and normalization parameters.

## Scheduler

The scheduler is warmup + cosine decay.

During warmup:

```text
lr increases linearly from small value to learning_rate
```

After warmup:

```text
lr decays from learning_rate to min_learning_rate following a cosine curve
```

This is a common LLM pretraining schedule.

## Gradient Accumulation

Gradient accumulation lets us simulate a larger batch size.

If:

```text
batch_size = 4
gradient_accumulation_steps = 8
```

Then:

```text
effective_batch_size = 32
```

In the training loop, the loss is divided by `gradient_accumulation_steps`:

```python
loss = output.loss / gradient_accumulation_steps
```

Why divide?

Because we call backward multiple times before one optimizer step. Dividing keeps the final accumulated gradient scale comparable to a single large batch.

## Training Step

One optimizer update does this:

```text
get batch
forward
compute loss
scale loss for gradient accumulation
backward
unscale gradients if fp16
compute grad norm
clip gradients
optimizer step
scheduler step
zero grad
log metrics
```

## Mixed Precision

The trainer supports:

- `fp32`
- `bf16`
- `fp16`

For local CPU smoke tests, we use:

```yaml
dtype: fp32
device: cpu
```

For GPU training later, we can use:

```yaml
dtype: bf16
device: cuda
```

`fp16` uses `GradScaler` on CUDA to avoid underflow.

## Metric Logging

File:

```text
src/train/logger.py
```

Metrics are written to JSONL:

```text
logs/pretrain_tiny_smoke_metrics.jsonl
```

Each line is one training record.

Metrics include:

- `step`
- `loss`
- `lr`
- `tokens`
- `tokens_per_sec`
- `grad_norm`
- `memory_mb`
- `elapsed_sec`

JSONL is useful because it can be appended during training and later loaded into Python, pandas, or plotting tools.

## Smoke Test Command

Use your `pytorch` conda environment:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml
```

The current run completed 20 steps on CPU.

Loss moved from about:

```text
8.85 -> 7.63
```

This proves:

- batches can enter the model
- loss is finite
- backward works
- optimizer updates parameters
- scheduler changes learning rate
- metrics are logged

## Why This Is Not Yet Final Training

The tiny run uses:

- tiny model
- tiny data
- byte tokenizer
- CPU
- 20 steps

It is only a correctness test.

The real 64M smoke run will happen after checkpoint/resume and generation are ready.

## How To Explain This In An Interview

You can say:

```text
I implemented a config-driven pretraining loop that connects the model, tokenizer, streaming JSONL dataset, and collator. The trainer uses AdamW with separate weight-decay groups, warmup plus cosine learning-rate decay, gradient accumulation, gradient clipping, mixed-precision hooks, non-finite loss checks, and JSONL metrics for reproducibility.

Before running expensive GPU training, I added a tiny CPU smoke config and verified that the full forward-backward-optimizer-scheduler loop runs correctly and that loss decreases on a small repeated dataset.
```

## Status

Implemented:

- tiny local training config
- tiny model config
- AdamW optimizer
- warmup cosine scheduler
- gradient accumulation
- gradient clipping
- dtype/device handling
- JSONL logger
- token throughput logging
- CPU training smoke run

Next:

- checkpoint save/load
- resume training
- generation script
- real 64M smoke run

---

# 第五步：Pretraining Trainer 中文说明

这一节解释 MiniMind AgentLab 的第五个里程碑：基础预训练循环。

## 目标

第五步的目标是把第三步实现的模型和第四步实现的数据管线真正接起来，让模型能训练。

当前 trainer 已经支持：

- 从 config 初始化模型
- 加载 tokenizer 和 dataset
- DataLoader + collator
- AdamW optimizer
- weight decay 参数分组
- warmup + cosine 学习率调度
- gradient accumulation
- gradient clipping
- fp32 / bf16 / fp16 dtype 选择
- CPU 或 CUDA device 选择
- NaN / 非有限 loss 检查
- JSONL metrics 日志
- token throughput 统计
- 本地 tiny smoke training

## 相关文件

```text
scripts/train_pretrain.py
src/train/optim.py
src/train/logger.py
configs/pretrain_tiny.yaml
configs/minimind_tiny.yaml
logs/pretrain_tiny_smoke_metrics.jsonl
```

## 当前训练流程

```text
PretrainConfig
   |
   v
MiniMindConfig
   |
   v
Tokenizer + PretrainDataset + Collator
   |
   v
DataLoader
   |
   v
MiniMindForCausalLM
   |
   v
loss
   |
   v
backward
   |
   v
gradient clipping
   |
   v
optimizer step
   |
   v
scheduler step
   |
   v
JSONL metrics
```

## Tiny Training Config

文件：

```text
configs/pretrain_tiny.yaml
```

这个配置专门用于本地 CPU smoke test。

它指向 tiny model：

```yaml
model_config: configs/minimind_tiny.yaml
```

并使用：

```yaml
device: cpu
dtype: fp32
max_seq_len: 128
batch_size: 2
gradient_accumulation_steps: 1
max_steps: 20
```

它不是为了训练出一个有用的模型，而是为了证明训练循环是正确的。

## Tiny Model Config

文件：

```text
configs/minimind_tiny.yaml
```

tiny model 大约有 1.21M 参数。

为什么要加 tiny model？

因为 64M 模型是正式目标，但在 CPU 上调试会比较慢。tiny model 可以让我们在你的电脑上快速验证 trainer。

## Optimizer

文件：

```text
src/train/optim.py
```

optimizer 使用 AdamW。

参数分成两组：

- decay group
- no-decay group

no-decay 包括：

- bias 参数
- norm 参数
- 1D 参数

原因是：weight decay 通常作用在大矩阵权重上，但一般不会加在 bias 和 normalization 参数上。

## Scheduler

学习率调度是 warmup + cosine decay。

warmup 阶段：

```text
学习率从较小值线性升到 learning_rate
```

warmup 结束后：

```text
学习率按 cosine 曲线从 learning_rate 降到 min_learning_rate
```

这是 LLM pretraining 里很常见的学习率策略。

## Gradient Accumulation

gradient accumulation 用来模拟更大的 batch size。

如果：

```text
batch_size = 4
gradient_accumulation_steps = 8
```

那么：

```text
effective_batch_size = 32
```

训练代码里会把 loss 除以 `gradient_accumulation_steps`：

```python
loss = output.loss / gradient_accumulation_steps
```

为什么要除？

因为我们会多次 backward 后才做一次 optimizer step。如果不除，累积出来的梯度会比正常大 batch 梯度大很多。

## 一个训练 step 做什么

一次 optimizer update 包括：

```text
取 batch
forward
计算 loss
按 gradient accumulation 缩放 loss
backward
如果是 fp16，先 unscale gradients
计算 grad norm
gradient clipping
optimizer step
scheduler step
zero grad
记录日志
```

## Mixed Precision

trainer 支持：

- `fp32`
- `bf16`
- `fp16`

本地 CPU smoke test 使用：

```yaml
dtype: fp32
device: cpu
```

之后 GPU 训练可以用：

```yaml
dtype: bf16
device: cuda
```

如果使用 `fp16`，CUDA 上会启用 `GradScaler`，减少 underflow 风险。

## Metrics Logging

文件：

```text
src/train/logger.py
```

训练指标会写到 JSONL：

```text
logs/pretrain_tiny_smoke_metrics.jsonl
```

每一行是一条训练记录。

记录内容包括：

- `step`
- `loss`
- `lr`
- `tokens`
- `tokens_per_sec`
- `grad_norm`
- `memory_mb`
- `elapsed_sec`

JSONL 的好处是训练过程中可以不断追加，后面也方便用 Python、pandas 或画图工具分析。

## Smoke Test 命令

使用你的 `pytorch` conda 环境：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml
```

当前已经在 CPU 上成功跑完 20 step。

loss 大致从：

```text
8.85 -> 7.63
```

这说明：

- batch 能进入模型
- loss 是有限数值
- backward 正常
- optimizer 能更新参数
- scheduler 能改变学习率
- metrics 能写入日志

## 为什么这还不是正式训练

这个 tiny run 使用的是：

- tiny model
- tiny data
- byte tokenizer
- CPU
- 20 steps

它只是正确性测试。

真正的 64M smoke run 应该等 checkpoint/resume 和 generation 完成之后再跑。

## 面试时怎么讲

可以这样说：

```text
我实现了一个 config-driven pretraining loop，把模型、tokenizer、streaming JSONL dataset 和 collator 接了起来。trainer 使用 AdamW，并对 weight decay 参数分组；学习率使用 warmup + cosine decay；训练支持 gradient accumulation、gradient clipping、mixed precision hooks、非有限 loss 检查，以及 JSONL metrics logging。

在正式 GPU 训练前，我先用 tiny CPU config 做 smoke test，验证 forward、backward、optimizer step、scheduler step 和 logging 全链路都能跑通，并且 loss 在小数据上下降。
```

## 当前状态

已完成：

- tiny 本地训练配置
- tiny 模型配置
- AdamW optimizer
- warmup cosine scheduler
- gradient accumulation
- gradient clipping
- dtype/device handling
- JSONL logger
- tokens/sec logging
- CPU training smoke run

下一步：

- checkpoint save/load
- resume training
- generation script
- 正式 64M smoke run
