# Step 2: Configuration System

This document explains the second milestone of MiniMind AgentLab: the configuration system.

## Why We Need A Configuration System

Training a language model has many parameters:

- model size
- number of layers
- number of attention heads
- tokenizer path
- data path
- batch size
- learning rate
- dtype
- checkpoint directory

Hard-coding these values inside Python files makes experiments hard to reproduce. A configuration system lets us describe an experiment in YAML, load it into Python, validate it, and save it with checkpoints later.

In this project, config files are the bridge between experiment design and executable code.

## Files Involved

```text
configs/minimind_64m.yaml
configs/pretrain.yaml
src/model/config.py
src/train/config.py
src/utils/config.py
scripts/train_pretrain.py
```

## Model Config

File:

```text
configs/minimind_64m.yaml
```

This file describes the model architecture.

Important fields:

```yaml
vocab_size: 6400
hidden_size: 512
num_hidden_layers: 12
num_attention_heads: 8
num_key_value_heads: 2
intermediate_size: 2816
max_position_embeddings: 2048
rope_theta: 1000000.0
rms_norm_eps: 1.0e-6
tie_word_embeddings: true
dropout: 0.0
```

These values decide the shape of the Transformer.

For example:

```text
head_dim = hidden_size / num_attention_heads
         = 512 / 8
         = 64
```

For GQA:

```text
num_key_value_groups = num_attention_heads / num_key_value_heads
                     = 8 / 2
                     = 4
```

That means each KV head is shared by 4 query heads.

## MiniMindConfig

File:

```text
src/model/config.py
```

This file defines `MiniMindConfig`, a Python dataclass for model hyperparameters.

Its responsibilities:

- load model config from YAML
- convert YAML values into a structured Python object
- validate model parameters
- expose derived values such as `head_dim`
- save model config later when checkpointing

Example:

```python
model_config = MiniMindConfig.from_yaml("configs/minimind_64m.yaml")
print(model_config.hidden_size)
print(model_config.head_dim)
```

Validation examples:

- `hidden_size` must be divisible by `num_attention_heads`
- `num_attention_heads` must be divisible by `num_key_value_heads`
- `dropout` must be in `[0, 1)`
- `rope_theta` must be positive

This catches bad experiment settings early, before training crashes deep inside attention code.

## Pretrain Config

File:

```text
configs/pretrain.yaml
```

This file describes a pretraining run.

Important fields:

```yaml
run_name: pretrain_minimind_64m
model_config: configs/minimind_64m.yaml
tokenizer_path: data/tokenizer
train_data_path: data/pretrain_t2t_mini.jsonl
output_dir: checkpoints/pretrain
log_dir: logs

batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 3.0e-4
warmup_steps: 100
max_steps: 1000
dtype: bf16
```

This config does not define the model architecture directly. Instead, it points to the model config through:

```yaml
model_config: configs/minimind_64m.yaml
```

This separation is useful because we can reuse the same model config across different training runs.

## PretrainConfig

File:

```text
src/train/config.py
```

This file defines `PretrainConfig`, a dataclass for training hyperparameters.

Its responsibilities:

- load pretraining config from YAML
- validate training settings
- support command-line overrides
- load the linked `MiniMindConfig`
- compute useful derived values

Example:

```python
train_config = PretrainConfig.from_yaml("configs/pretrain.yaml")
model_config = train_config.load_model_config()
```

One useful property is:

```python
effective_batch_size = batch_size * gradient_accumulation_steps
```

With our current config:

```text
effective_batch_size = 4 * 8 = 32
```

This means each optimizer step sees the equivalent of 32 micro-batches, even though each forward pass only uses batch size 4.

## YAML Utilities

File:

```text
src/utils/config.py
```

This file contains shared helper functions:

- `load_yaml`
- `save_yaml`
- `apply_overrides`
- `set_by_dotted_key`
- `parse_scalar`

Normally it uses PyYAML. If PyYAML is not installed, it has a small fallback parser for simple `key: value` files. This keeps our dry-run command usable even in a minimal environment.

## Command-Line Overrides

We can override config values without editing YAML files.

Example:

```bash
python scripts/train_pretrain.py --dry-run --override batch_size=2 --override dtype=fp32
```

This is useful for experiments.

Instead of creating many similar YAML files, we can keep a base config and override only the values we want to change.

## Dry Run

File:

```text
scripts/train_pretrain.py
```

The dry-run mode loads and validates both configs without starting training.

Command:

```bash
python scripts/train_pretrain.py --dry-run
```

Expected output:

```text
run_name: pretrain_minimind_64m
pretrain_config: configs/pretrain.yaml
model_config: configs/minimind_64m.yaml
model hidden_size/layers/heads: 512/12/8
kv_heads: 2, head_dim: 64
batch_size: 4
gradient_accumulation_steps: 8
effective_batch_size: 32
dtype/device: bf16/auto
```

This proves that:

- the training config can be loaded
- the linked model config can be loaded
- validation passes
- derived values are computed correctly
- command-line entry point works

## Current Flow

```text
configs/pretrain.yaml
        |
        v
PretrainConfig.from_yaml()
        |
        v
train_config.model_config
        |
        v
configs/minimind_64m.yaml
        |
        v
MiniMindConfig.from_yaml()
        |
        v
validated Python config objects
```

## How To Explain This In An Interview

You can say:

```text
I separated model architecture config from training-run config. The model config defines Transformer shape parameters such as hidden size, number of layers, attention heads, KV heads, RoPE settings, and embedding tying. The pretraining config defines run-level settings such as data path, tokenizer path, batch size, gradient accumulation, learning rate, dtype, and checkpoint directories.

Both configs are loaded into dataclasses and validated before training. This catches invalid settings early, for example when hidden size is not divisible by the number of attention heads, or when GQA head counts are incompatible. I also support command-line overrides so I can run small experiment variants without editing the base YAML files.
```

## Why This Matters

This step may look simple, but it is important for real training systems.

It gives us:

- reproducibility
- experiment control
- early validation
- cleaner training code
- easier checkpoint metadata
- easier future scaling to multiple model sizes and runs

Without this layer, later code would become full of hard-coded constants.

## Status

Implemented:

- model YAML config
- pretrain YAML config
- model config dataclass
- pretrain config dataclass
- YAML loading and saving helpers
- command-line overrides
- dry-run validation command

Next:

- model implementation
- parameter counting
- saving config copies into checkpoints

---

# 第二步：配置系统中文说明

这一节解释 MiniMind AgentLab 的第二个里程碑：配置系统。

## 为什么需要配置系统

训练语言模型会涉及很多参数，比如：

- 模型大小
- 层数
- attention heads 数量
- tokenizer 路径
- 数据路径
- batch size
- learning rate
- dtype
- checkpoint 保存目录

如果这些参数都硬编码在 Python 文件里，后面做实验会很痛苦：改一次实验就要改代码，也很难复现之前的结果。

所以我们把实验参数放在 YAML 配置文件里，再由 Python 代码读取、校验、使用。这样配置文件就成为了“实验设计”和“可执行代码”之间的桥梁。

## 相关文件

```text
configs/minimind_64m.yaml
configs/pretrain.yaml
src/model/config.py
src/train/config.py
src/utils/config.py
scripts/train_pretrain.py
```

## 模型配置

文件：

```text
configs/minimind_64m.yaml
```

这个文件描述模型结构，也就是 Transformer 长什么样。

当前关键参数是：

```yaml
vocab_size: 6400
hidden_size: 512
num_hidden_layers: 12
num_attention_heads: 8
num_key_value_heads: 2
intermediate_size: 2816
max_position_embeddings: 2048
rope_theta: 1000000.0
rms_norm_eps: 1.0e-6
tie_word_embeddings: true
dropout: 0.0
```

这些值决定模型的 shape。

比如：

```text
head_dim = hidden_size / num_attention_heads
         = 512 / 8
         = 64
```

对于 GQA：

```text
num_key_value_groups = num_attention_heads / num_key_value_heads
                     = 8 / 2
                     = 4
```

意思是：每个 KV head 会被 4 个 Q head 共享。

## MiniMindConfig 的作用

文件：

```text
src/model/config.py
```

这里定义了 `MiniMindConfig`，它是一个 dataclass，用来承载模型超参数。

它的作用是：

- 从 YAML 加载模型配置
- 把 YAML 里的文本参数变成结构化 Python 对象
- 校验模型参数是否合法
- 提供派生属性，比如 `head_dim`
- 后面 checkpoint 时可以保存模型配置

示例：

```python
model_config = MiniMindConfig.from_yaml("configs/minimind_64m.yaml")
print(model_config.hidden_size)
print(model_config.head_dim)
```

它会检查一些关键约束：

- `hidden_size` 必须能被 `num_attention_heads` 整除
- `num_attention_heads` 必须能被 `num_key_value_heads` 整除
- `dropout` 必须在 `[0, 1)` 范围内
- `rope_theta` 必须是正数

这样可以在训练开始前就发现配置错误，而不是等 attention 里面 reshape 崩掉。

## 训练配置

文件：

```text
configs/pretrain.yaml
```

这个文件描述一次 pretraining run 怎么跑。

关键参数包括：

```yaml
run_name: pretrain_minimind_64m
model_config: configs/minimind_64m.yaml
tokenizer_path: data/tokenizer
train_data_path: data/pretrain_t2t_mini.jsonl
output_dir: checkpoints/pretrain
log_dir: logs

batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 3.0e-4
warmup_steps: 100
max_steps: 1000
dtype: bf16
```

注意，训练配置不直接写模型结构，而是通过这一行指向模型配置：

```yaml
model_config: configs/minimind_64m.yaml
```

这样做的好处是：同一个模型结构可以被多个训练实验复用。

## PretrainConfig 的作用

文件：

```text
src/train/config.py
```

这里定义了 `PretrainConfig`，用于承载训练相关参数。

它的作用是：

- 从 YAML 加载训练配置
- 校验训练参数是否合法
- 支持命令行覆盖配置
- 加载它指向的 `MiniMindConfig`
- 计算一些派生值

比如：

```python
train_config = PretrainConfig.from_yaml("configs/pretrain.yaml")
model_config = train_config.load_model_config()
```

一个重要派生值是：

```python
effective_batch_size = batch_size * gradient_accumulation_steps
```

当前配置中：

```text
effective_batch_size = 4 * 8 = 32
```

意思是：每次 forward 的 batch size 是 4，但通过梯度累积，每次 optimizer step 等价于看到了 32 个 micro-batch。

## YAML 工具函数

文件：

```text
src/utils/config.py
```

这里放的是通用配置工具：

- `load_yaml`
- `save_yaml`
- `apply_overrides`
- `set_by_dotted_key`
- `parse_scalar`

正常情况下它会使用 PyYAML。如果环境里没有安装 PyYAML，它也有一个轻量 fallback，可以解析我们当前这种简单的 `key: value` 配置文件。

这个设计让 dry-run 在最小环境下也能跑。

## 命令行覆盖参数

我们可以不改 YAML 文件，直接从命令行覆盖某些配置。

例子：

```bash
python scripts/train_pretrain.py --dry-run --override batch_size=2 --override dtype=fp32
```

这样做适合快速实验。

比如你想试不同 batch size，不需要复制很多份 YAML，只需要基于同一个配置文件临时覆盖。

## Dry Run

文件：

```text
scripts/train_pretrain.py
```

dry-run 模式只加载和校验配置，不开始训练。

命令：

```bash
python scripts/train_pretrain.py --dry-run
```

预期输出类似：

```text
run_name: pretrain_minimind_64m
pretrain_config: configs/pretrain.yaml
model_config: configs/minimind_64m.yaml
model hidden_size/layers/heads: 512/12/8
kv_heads: 2, head_dim: 64
batch_size: 4
gradient_accumulation_steps: 8
effective_batch_size: 32
dtype/device: bf16/auto
```

这说明：

- 训练配置能被读取
- 模型配置能被读取
- 参数校验通过
- 派生值能正确计算
- 命令行入口能工作

## 当前配置流

```text
configs/pretrain.yaml
        |
        v
PretrainConfig.from_yaml()
        |
        v
train_config.model_config
        |
        v
configs/minimind_64m.yaml
        |
        v
MiniMindConfig.from_yaml()
        |
        v
通过校验的 Python 配置对象
```

## 面试时怎么讲

可以这样说：

```text
我把模型结构配置和训练实验配置分开了。模型配置负责定义 Transformer 的结构参数，比如 hidden size、层数、attention heads、KV heads、RoPE 参数和 embedding tying；训练配置负责定义数据路径、tokenizer 路径、batch size、梯度累积、学习率、dtype 和 checkpoint 目录。

两个配置都会被加载成 dataclass，并在训练开始前做合法性校验。比如 hidden size 必须能被 attention heads 整除，GQA 中 attention heads 必须能被 KV heads 整除。我还支持命令行 override，这样可以快速跑不同实验而不用改基础 YAML 文件。
```

## 这一层为什么重要

配置系统看起来不复杂，但对真实训练工程很重要。

它带来的价值是：

- 可复现
- 实验可控
- 提前发现错误
- 训练代码更干净
- checkpoint 可以保存完整配置
- 未来更容易扩展到多个模型大小和多组实验

如果没有这一层，后面的训练代码就会充满硬编码常量，很难维护。

## 当前状态

已完成：

- 模型 YAML 配置
- 训练 YAML 配置
- 模型配置 dataclass
- 训练配置 dataclass
- YAML 加载和保存工具
- 命令行覆盖参数
- dry-run 校验命令

下一步：

- 保存 config copy 到 checkpoint
- 后续扩展更多实验配置
