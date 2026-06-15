# Step 9: Real MiniMind-Compatible Assets

This document explains the ninth milestone of MiniMind AgentLab: preparing real MiniMind-compatible tokenizer and pretraining data.

## Goal

The goal of this step is to move from local byte-tokenizer smoke tests toward real MiniMind-compatible assets.

Step 9 prepares:

- local asset layout
- tokenizer inspection
- data inspection
- vocab-size comparison
- real pretraining config
- clear failure messages when assets are missing

Actual real data files are not committed to Git.

## Files Involved

```text
data/minimind/README.md
configs/pretrain_minimind_local.yaml
scripts/inspect_tokenizer.py
scripts/validate_pretrain_assets.py
configs/minimind_64m.yaml
```

## Expected Local Layout

Put real MiniMind-compatible assets here:

```text
data/minimind/
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── pretrain_t2t_mini.jsonl
```

The exact tokenizer files can vary depending on source format. Hugging Face tokenizer directories and `tokenizer.json` are both supported.

## Why Assets Are Not Tracked

Tokenizer and dataset files may be:

- large
- externally downloaded
- generated
- subject to license or distribution constraints

So `.gitignore` excludes:

```text
data/minimind/tokenizer/
data/minimind/pretrain_t2t_mini.jsonl
```

The repository tracks only the expected layout and validation scripts.

## Real Pretraining Config

File:

```text
configs/pretrain_minimind_local.yaml
```

This config points to:

```yaml
model_config: configs/minimind_64m.yaml
tokenizer_path: data/minimind/tokenizer
train_data_path: data/minimind/pretrain_t2t_mini.jsonl
```

It is the bridge from tiny smoke tests to a real 64M smoke run.

## Tokenizer Inspection

Command:

```bash
python scripts/inspect_tokenizer.py --tokenizer-path data/minimind/tokenizer --model-config configs/minimind_64m.yaml
```

This checks:

- tokenizer can be loaded
- vocab size
- pad token ID
- EOS token ID
- sample encode/decode
- vocab-size compatibility with model config

If tokenizer files are missing, the script prints:

```text
error: Tokenizer not found at data\minimind\tokenizer. Provide a tokenizer path or use allow_byte_fallback=True.
Place tokenizer files under data/minimind/tokenizer or pass --tokenizer-path.
```

## Asset Validation

Command:

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

This checks:

- tokenizer loads
- JSONL data exists
- examples can be read
- text can be tokenized
- collator can build a batch
- attention lengths look reasonable
- labels use `-100` for padding
- first decoded sample can be inspected

If assets are missing, it gives a clear message telling you where to place them.

## Vocab Size

The model config currently uses:

```yaml
vocab_size: 6400
```

The real tokenizer should be checked against this value.

If the tokenizer vocab size differs, there are two options:

1. update `vocab_size` in the model config to match tokenizer vocab size
2. use the tokenizer version intended for this 64M config

For clean training, the model vocabulary and tokenizer vocabulary should match.

## How This Connects To 64M Training

After assets validate, the 64M smoke run command will be:

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml
```

On a GPU machine, override device if needed:

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml --override device=cuda
```

## How To Explain This In An Interview

You can say:

```text
After validating the tiny local pipeline, I prepared a separate real-asset validation step for MiniMind-compatible tokenizer and JSONL pretraining data. The validation scripts check tokenizer loading, vocab compatibility, JSONL parsing, tokenization, dynamic padding, and label masking before launching a 64M training run. This reduces the risk of wasting GPU time on data or tokenizer integration bugs.
```

## Status

Implemented:

- expected local asset layout
- real pretraining config
- tokenizer inspection script
- pretraining asset validation script
- vocab compatibility warning
- user-friendly missing-file errors

Still needed:

- place real tokenizer files under `data/minimind/tokenizer`
- place real `pretrain_t2t_mini.jsonl` under `data/minimind`
- validate assets
- run 64M smoke training

---

# 第九步：真实 MiniMind-Compatible 资源中文说明

这一节解释 MiniMind AgentLab 的第九个里程碑：准备真实 MiniMind-compatible tokenizer 和预训练数据。

## 目标

第九步的目标是：从本地 ByteTokenizer smoke test 过渡到真实 MiniMind-compatible 资源。

Step 9 已经准备了：

- 本地资源目录结构
- tokenizer 检查脚本
- 数据检查脚本
- vocab size 对比
- 真实 pretraining config
- 缺文件时的清晰错误提示

真实数据文件不会提交到 Git。

## 相关文件

```text
data/minimind/README.md
configs/pretrain_minimind_local.yaml
scripts/inspect_tokenizer.py
scripts/validate_pretrain_assets.py
configs/minimind_64m.yaml
```

## 期望的本地目录结构

把真实 MiniMind-compatible 资源放在：

```text
data/minimind/
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── pretrain_t2t_mini.jsonl
```

具体 tokenizer 文件可能因来源不同而不同。当前代码支持 Hugging Face tokenizer 目录，也支持单独的 `tokenizer.json`。

## 为什么不把资源提交到 Git

tokenizer 和 dataset 可能：

- 文件较大
- 来自外部下载
- 是生成文件
- 有 license 或分发限制

所以 `.gitignore` 会排除：

```text
data/minimind/tokenizer/
data/minimind/pretrain_t2t_mini.jsonl
```

仓库只保留目录说明和验证脚本。

## 真实预训练配置

文件：

```text
configs/pretrain_minimind_local.yaml
```

它指向：

```yaml
model_config: configs/minimind_64m.yaml
tokenizer_path: data/minimind/tokenizer
train_data_path: data/minimind/pretrain_t2t_mini.jsonl
```

这是从 tiny smoke test 走向真实 64M smoke run 的配置入口。

## Tokenizer 检查

命令：

```bash
python scripts/inspect_tokenizer.py --tokenizer-path data/minimind/tokenizer --model-config configs/minimind_64m.yaml
```

它会检查：

- tokenizer 能否加载
- vocab size
- pad token ID
- EOS token ID
- sample encode/decode
- tokenizer vocab size 是否和 model config 匹配

如果 tokenizer 文件缺失，会提示：

```text
error: Tokenizer not found at data\minimind\tokenizer. Provide a tokenizer path or use allow_byte_fallback=True.
Place tokenizer files under data/minimind/tokenizer or pass --tokenizer-path.
```

## 资源验证

命令：

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

它会检查：

- tokenizer 能加载
- JSONL 数据存在
- 样本能读取
- text 能 tokenize
- collator 能构造 batch
- attention lengths 是否合理
- padding label 是否是 `-100`
- 第一条样本能 decode 供人工检查

如果缺文件，会明确告诉你应该把文件放在哪里。

## Vocab Size

当前模型配置里：

```yaml
vocab_size: 6400
```

真实 tokenizer 接入后，要检查它的 vocab size 是否等于这个值。

如果不一致，有两个选择：

1. 把 model config 里的 `vocab_size` 改成 tokenizer 的 vocab size
2. 换成和这个 64M config 对应的 tokenizer 版本

正式训练时，最好让模型 vocab size 和 tokenizer vocab size 一致。

## 和 64M 训练怎么连接

资源验证通过后，64M smoke run 命令是：

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml
```

如果在 GPU 机器上，可以覆盖 device：

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml --override device=cuda
```

## 面试时怎么讲

可以这样说：

```text
在 tiny 本地 pipeline 验证通过后，我单独做了真实 MiniMind-compatible tokenizer 和 JSONL pretraining data 的验证步骤。验证脚本会检查 tokenizer 加载、vocab size 匹配、JSONL 解析、tokenization、动态 padding 和 label masking，然后才启动 64M 训练。这样可以减少把 GPU 时间浪费在数据和 tokenizer 集成 bug 上的风险。
```

## 当前状态

已完成：

- 本地资源目录结构
- 真实 pretraining config
- tokenizer 检查脚本
- pretraining asset 验证脚本
- vocab compatibility warning
- 友好的缺文件错误提示

还需要：

- 把真实 tokenizer 文件放到 `data/minimind/tokenizer`
- 把真实 `pretrain_t2t_mini.jsonl` 放到 `data/minimind`
- 运行资源验证
- 跑 64M smoke training
