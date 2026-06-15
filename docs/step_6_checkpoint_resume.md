# Step 6: Checkpoint And Resume

This document explains the sixth milestone of MiniMind AgentLab: checkpointing and training resume.

## Goal

The goal of this step is to make training recoverable.

If training is interrupted, we should not lose progress. A proper checkpoint must save more than model weights.

The checkpoint system now saves:

- model state dict
- optimizer state dict
- scheduler state dict
- GradScaler state dict when fp16 is enabled
- global training step
- train config
- model config
- Python random state
- PyTorch random state
- CUDA random state when CUDA is available
- `latest.pt`
- step-specific checkpoint files

## Files Involved

```text
src/train/checkpoint.py
scripts/train_pretrain.py
checkpoints/pretrain_tiny/latest.pt
checkpoints/pretrain_tiny/pretrain_step_000005.pt
```

## Why Checkpointing Matters

LLM training can run for hours, days, or weeks.

Without checkpointing:

- a crash loses all progress
- experiments cannot be resumed
- generation cannot load trained weights
- debugging long runs is painful
- results are harder to reproduce

Checkpointing is what turns a toy training loop into a real training system.

## What Gets Saved

Each checkpoint is a PyTorch `.pt` file containing:

```python
{
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "train_config": train_config.to_dict(),
    "model_config": model_config.to_dict(),
    "random_state": random_state,
}
```

The `step` tells us where training stopped.

The model state restores weights.

The optimizer state restores AdamW momentum and variance buffers.

The scheduler state restores the learning-rate schedule.

The random state helps resume sampling and dropout behavior more reproducibly.

## latest.pt And Step Checkpoints

The trainer saves two checkpoint paths:

```text
pretrain_step_000005.pt
latest.pt
```

The step checkpoint is a historical snapshot.

`latest.pt` always points to the most recent checkpoint.

This makes resume easy:

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_tiny/latest.pt
```

## Config Copies

The trainer also writes resolved config copies into the checkpoint directory:

```text
checkpoints/pretrain_tiny/model_config.yaml
checkpoints/pretrain_tiny/pretrain_config.yaml
```

This helps reproducibility. Even if the original config file changes later, the run directory still records what was used.

## Resume Flow

```text
build model
build optimizer
build scheduler
build scaler
   |
   v
load checkpoint
   |
   v
restore model weights
restore optimizer state
restore scheduler state
restore scaler state
restore random state
   |
   v
continue from saved step
```

It is important to build the objects first, then load their saved state.

## Verified Resume Test

First run:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml --override max_steps=3 --override save_interval=2
```

This saved:

```text
pretrain_step_000002.pt
pretrain_step_000003.pt
latest.pt
```

Second run:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml --override max_steps=5 --override save_interval=2 --resume checkpoints\pretrain_tiny\latest.pt
```

The output showed:

```text
resumed from checkpoints\pretrain_tiny\latest.pt at step 3
step 0004 ...
step 0005 ...
```

This proves resume starts from step 3 and continues to step 4/5 instead of restarting at step 1.

## How To Explain This In An Interview

You can say:

```text
I implemented checkpointing that saves the model, optimizer, scheduler, mixed-precision scaler, global step, configs, and random states. Resume reconstructs the model and training objects, loads their state dicts, restores the saved step, and continues training from that point. I verified it with a tiny run by training to step 3, saving latest.pt, then resuming to step 5.
```

## Status

Implemented:

- checkpoint save
- `latest.pt`
- step checkpoints
- config copies
- optimizer restore
- scheduler restore
- scaler restore
- random state restore
- `--resume`
- tiny resume test

Next:

- use checkpoints for generation
- run longer smoke training

---

# 第六步：Checkpoint 和 Resume 中文说明

这一节解释 MiniMind AgentLab 的第六个里程碑：checkpoint 保存和训练恢复。

## 目标

第六步的目标是让训练可以中断后继续。

真正的训练系统不能只保存模型权重，还要保存训练状态。

当前 checkpoint 会保存：

- model state dict
- optimizer state dict
- scheduler state dict
- fp16 时的 GradScaler state dict
- global training step
- train config
- model config
- Python random state
- PyTorch random state
- 如果有 CUDA，也保存 CUDA random state
- `latest.pt`
- 按 step 命名的 checkpoint 文件

## 相关文件

```text
src/train/checkpoint.py
scripts/train_pretrain.py
checkpoints/pretrain_tiny/latest.pt
checkpoints/pretrain_tiny/pretrain_step_000005.pt
```

## 为什么 checkpoint 重要

LLM 训练可能跑几个小时、几天，甚至几周。

如果没有 checkpoint：

- 训练崩了就全部重来
- 实验无法恢复
- generation 无法加载训练后的权重
- 长时间训练很难 debug
- 实验复现困难

checkpoint 是从 toy training loop 走向真实训练系统的重要一步。

## checkpoint 里保存什么

每个 checkpoint 是一个 PyTorch `.pt` 文件，大致包含：

```python
{
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "train_config": train_config.to_dict(),
    "model_config": model_config.to_dict(),
    "random_state": random_state,
}
```

`step` 记录训练到了第几步。

model state 恢复模型权重。

optimizer state 恢复 AdamW 的动量和二阶矩估计。

scheduler state 恢复学习率进度。

random state 帮助恢复随机性，提升 resume 的可复现性。

## latest.pt 和 step checkpoint

训练时会保存两种路径：

```text
pretrain_step_000005.pt
latest.pt
```

step checkpoint 是历史快照。

`latest.pt` 永远是最新 checkpoint。

这样 resume 很方便：

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_tiny/latest.pt
```

## 配置副本

训练时还会把 resolved config 保存到 checkpoint 目录：

```text
checkpoints/pretrain_tiny/model_config.yaml
checkpoints/pretrain_tiny/pretrain_config.yaml
```

这样即使原始配置文件后面被改了，这次 run 用过什么配置仍然保留在 run 目录里。

## Resume 流程

```text
先创建 model
先创建 optimizer
先创建 scheduler
先创建 scaler
   |
   v
读取 checkpoint
   |
   v
恢复模型权重
恢复 optimizer 状态
恢复 scheduler 状态
恢复 scaler 状态
恢复 random state
   |
   v
从保存的 step 继续训练
```

注意顺序：先创建对象，再把 checkpoint 里的 state dict 加载进去。

## 已验证的 resume 测试

第一次运行：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml --override max_steps=3 --override save_interval=2
```

保存了：

```text
pretrain_step_000002.pt
pretrain_step_000003.pt
latest.pt
```

第二次运行：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\train_pretrain.py --config configs\pretrain_tiny.yaml --override max_steps=5 --override save_interval=2 --resume checkpoints\pretrain_tiny\latest.pt
```

输出里能看到：

```text
resumed from checkpoints\pretrain_tiny\latest.pt at step 3
step 0004 ...
step 0005 ...
```

这说明它确实从 step 3 接着训练，而不是从 step 1 重新开始。

## 面试时怎么讲

可以这样说：

```text
我实现了完整 checkpoint/resume，保存模型、optimizer、scheduler、mixed-precision scaler、global step、配置和 random states。恢复时先重建模型和训练对象，再加载各自的 state dict，并从保存的 step 继续训练。我用 tiny run 验证过：先训练到 step 3 保存 latest.pt，再 resume 到 step 5，训练确实从 step 4 继续。
```

## 当前状态

已完成：

- checkpoint 保存
- `latest.pt`
- step checkpoint
- config copy
- optimizer 恢复
- scheduler 恢复
- scaler 恢复
- random state 恢复
- `--resume`
- tiny resume 测试

下一步：

- 用 checkpoint 做 generation
- 跑更长 smoke training
