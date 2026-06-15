# Step 8: Tiny End-to-End Validation

This document explains the eighth milestone of MiniMind AgentLab: running the whole local tiny pipeline end to end.

## Goal

The goal of this step is to prove that the core system works as a complete pipeline before using real data or GPU compute.

The tiny E2E validation checks:

- config loading
- model initialization
- model forward/backward
- JSONL batch loading
- training
- checkpoint saving
- resume
- generation

## Files Involved

```text
scripts/run_tiny_e2e.py
configs/minimind_tiny.yaml
configs/pretrain_tiny.yaml
data/samples/pretrain_tiny.jsonl
checkpoints/pretrain_tiny/latest.pt
outputs/generated_sample.txt
```

## Why Tiny E2E Matters

Before renting a GPU server or running a 64M model, we want to know that the code path is correct.

Tiny E2E validation prevents wasting money debugging:

- shape bugs
- tokenizer bugs
- collator bugs
- optimizer bugs
- checkpoint bugs
- resume bugs
- generation bugs

This is a cheap local correctness test.

## What The Script Runs

Command:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\run_tiny_e2e.py --python C:\Users\zhish\anaconda3\envs\pytorch\python.exe
```

The script runs:

```text
1. train_pretrain.py --dry-run
2. inspect_model.py
3. inspect_batch.py
4. train_pretrain.py for a few steps
5. train_pretrain.py --resume
6. generate.py
```

## Verified Result

The script completed successfully locally.

It verified:

- tiny config loads
- tiny model has 426.88K parameters
- batch shape is correct
- training runs
- checkpoint is saved
- resume starts from the saved step
- generation loads the checkpoint and writes output

## How To Explain This In An Interview

You can say:

```text
Before running larger GPU experiments, I built a tiny end-to-end validation script that exercises the entire stack: config loading, model initialization, data loading, training, checkpointing, resume, and generation. This lets me catch integration bugs locally on CPU before spending compute on the 64M run.
```

## Status

Implemented:

- one-command tiny E2E script
- local CPU validation
- checkpoint/resume included
- generation included

Next:

- validate real MiniMind-compatible tokenizer and data
- run 64M smoke training

---

# 第八步：Tiny End-to-End Validation 中文说明

这一节解释 MiniMind AgentLab 的第八个里程碑：本地 tiny 端到端验证。

## 目标

第八步的目标是：在使用真实数据或 GPU 之前，先证明整个核心系统能完整跑通。

tiny E2E 会检查：

- 配置加载
- 模型初始化
- 模型 forward/backward
- JSONL batch 加载
- 训练
- checkpoint 保存
- resume
- generation

## 相关文件

```text
scripts/run_tiny_e2e.py
configs/minimind_tiny.yaml
configs/pretrain_tiny.yaml
data/samples/pretrain_tiny.jsonl
checkpoints/pretrain_tiny/latest.pt
outputs/generated_sample.txt
```

## 为什么 tiny E2E 重要

在租 GPU 服务器或跑 64M 模型之前，我们要先确认代码路径是正确的。

tiny E2E 可以在本地低成本发现：

- shape bug
- tokenizer bug
- collator bug
- optimizer bug
- checkpoint bug
- resume bug
- generation bug

这一步是便宜的本地正确性测试。

## 脚本会跑什么

命令：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\run_tiny_e2e.py --python C:\Users\zhish\anaconda3\envs\pytorch\python.exe
```

脚本依次运行：

```text
1. train_pretrain.py --dry-run
2. inspect_model.py
3. inspect_batch.py
4. train_pretrain.py 训练几步
5. train_pretrain.py --resume
6. generate.py
```

## 已验证结果

脚本已经在本地成功完成。

它验证了：

- tiny config 能加载
- tiny model 有 426.88K 参数
- batch shape 正确
- training 能跑
- checkpoint 能保存
- resume 能从保存 step 继续
- generation 能加载 checkpoint 并输出文件

## 面试时怎么讲

可以这样说：

```text
在跑更大的 GPU 实验之前，我做了一个 tiny end-to-end validation script，覆盖配置加载、模型初始化、数据加载、训练、checkpoint、resume 和 generation。这样可以先在本地 CPU 上发现集成 bug，避免把服务器时间浪费在基础工程错误上。
```

## 当前状态

已完成：

- 一键 tiny E2E 脚本
- 本地 CPU 验证
- 包含 checkpoint/resume
- 包含 generation

下一步：

- 验证真实 MiniMind-compatible tokenizer 和数据
- 跑 64M smoke training
