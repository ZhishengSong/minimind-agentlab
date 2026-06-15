# Step 4: Tokenizer And Data Pipeline

This document explains the fourth milestone of MiniMind AgentLab: tokenizer loading, JSONL pretraining data, and batch collation.

## Goal

The goal of this step is to turn raw JSONL text into tensors that can be passed into `MiniMindForCausalLM`.

The pipeline now supports:

- tokenizer loading
- a temporary byte tokenizer fallback for local smoke tests
- streaming JSONL pretraining dataset
- text field parsing
- tokenization
- max sequence length truncation
- optional EOS appending
- dynamic batch padding
- `input_ids`
- `attention_mask`
- `labels`
- `labels == -100` on padding positions
- batch inspection script

## Files Involved

```text
src/data/tokenizer.py
src/data/pretrain_dataset.py
src/data/collator.py
src/data/__init__.py
scripts/inspect_batch.py
configs/pretrain_tiny.yaml
data/samples/pretrain_tiny.jsonl
```

## Overall Flow

```text
JSONL file
   |
   v
PretrainDataset
   |
   v
tokenizer.encode(text)
   |
   v
list[int] token ids
   |
   v
CausalLMCollator
   |
   v
input_ids / attention_mask / labels
   |
   v
MiniMindForCausalLM
```

## JSONL Format

The expected pretraining data format is one JSON object per line:

```json
{"text": "MiniMind AgentLab is a small language model training project."}
```

For now, the dataset reads the `text` field by default.

Later, if MiniMind-compatible data uses another field name, the field can be changed through the script argument:

```bash
python scripts/inspect_batch.py --text-field text
```

## Tokenizer Wrapper

File:

```text
src/data/tokenizer.py
```

The tokenizer wrapper gives the rest of the data pipeline a simple interface:

```python
tokenizer.encode(text)
tokenizer.decode(token_ids)
tokenizer.pad_token_id
tokenizer.eos_token_id
tokenizer.vocab_size
```

The preferred path is a real tokenizer on disk:

```text
data/tokenizer
```

The loader supports:

- Hugging Face tokenizer directories through `AutoTokenizer`
- low-level `tokenizers` `tokenizer.json`
- a temporary `ByteTokenizer` fallback for smoke tests

## ByteTokenizer Fallback

The `ByteTokenizer` is only for local testing before the real MiniMind-compatible tokenizer is available.

It maps UTF-8 bytes to token IDs:

```text
pad_token_id = 0
eos_token_id = 1
byte value b -> token id b + 2
```

This means it has:

```text
vocab_size = 258
```

It is useful because it lets us test the data pipeline without downloading anything.

It is not the final tokenizer for real training.

## PretrainDataset

File:

```text
src/data/pretrain_dataset.py
```

`PretrainDataset` is an `IterableDataset`.

Why iterable?

Because pretraining datasets can become large. An iterable dataset can stream one line at a time instead of loading the full dataset into memory.

For each JSONL line, it:

1. parses JSON
2. reads the `text` field
3. encodes text into token IDs
4. appends EOS if available
5. truncates to `max_seq_len`
6. yields `{"input_ids": token_ids}`

## CausalLMCollator

File:

```text
src/data/collator.py
```

The collator turns a list of variable-length examples into fixed-size tensors.

Example input:

```text
example 1 length = 5
example 2 length = 3
```

After padding:

```text
input_ids:
[t0, t1, t2, t3, t4]
[t0, t1, t2, PAD, PAD]

attention_mask:
[1, 1, 1, 1, 1]
[1, 1, 1, 0, 0]

labels:
[t0, t1, t2, t3, t4]
[t0, t1, t2, -100, -100]
```

Why labels are copied from input IDs:

The model itself shifts logits and labels internally for next-token prediction.

Why padding labels are `-100`:

PyTorch cross entropy ignores positions where the label is `-100`.

This prevents the model from learning to predict padding tokens.

## Batch Inspection

File:

```text
scripts/inspect_batch.py
```

Run:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\inspect_batch.py
```

Current output:

```text
config: configs/pretrain_tiny.yaml
data_path: data/samples/pretrain_tiny.jsonl
tokenizer: ByteTokenizer
vocab_size: 258
pad_token_id: 0
eos_token_id: 1
num_examples: 2
input_ids shape: (2, 62)
attention_mask shape: (2, 62)
labels shape: (2, 62)
attention lengths: [62, 61]
num ignored labels: 1
first decoded sample:
MiniMind AgentLab is a small language model training project.
```

This proves:

- JSONL can be read
- text can be tokenized
- examples can be padded into a batch
- `attention_mask` marks real tokens
- padding labels are ignored through `-100`
- decode works for inspection

## How This Connects To The Model

The collator returns:

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}
```

These keys match the model forward signature:

```python
model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
```

So Step 4 connects raw data to the model implemented in Step 3.

## How To Explain This In An Interview

You can say:

```text
I implemented a streaming JSONL data pipeline for causal LM pretraining. Each line contains a text field, which is tokenized, optionally appended with EOS, truncated to max sequence length, and yielded as token IDs. The collator dynamically pads each batch, creates attention masks, and sets padding labels to -100 so cross entropy ignores padded positions.

I also added a tokenizer abstraction so the pipeline can use a real MiniMind-compatible tokenizer later, while still supporting a byte-level fallback for local smoke tests before the tokenizer files are available.
```

## Status

Implemented:

- tokenizer loading abstraction
- Hugging Face tokenizer support
- `tokenizers` JSON support
- byte tokenizer fallback
- streaming JSONL dataset
- dynamic padding collator
- labels with `-100` on padding
- tiny sample JSONL
- `inspect_batch.py`

Next:

- use the real MiniMind-compatible tokenizer
- use the real pretraining JSONL
- connect the batch pipeline to the training loop

---

# 第四步：Tokenizer 和数据管线中文说明

这一节解释 MiniMind AgentLab 的第四个里程碑：tokenizer、JSONL 预训练数据、batch collator。

## 目标

第四步的目标是把原始 JSONL 文本变成可以直接喂给 `MiniMindForCausalLM` 的 tensor。

目前数据管线已经支持：

- 加载 tokenizer
- 在没有真实 tokenizer 时，用临时 byte tokenizer 做本地 smoke test
- streaming JSONL dataset
- 读取 `text` 字段
- 文本 tokenize
- 按 `max_seq_len` 截断
- 可选追加 EOS
- batch 内动态 padding
- 生成 `input_ids`
- 生成 `attention_mask`
- 生成 `labels`
- padding 位置的 label 设为 `-100`
- `inspect_batch.py` 检查脚本

## 相关文件

```text
src/data/tokenizer.py
src/data/pretrain_dataset.py
src/data/collator.py
src/data/__init__.py
scripts/inspect_batch.py
configs/pretrain_tiny.yaml
data/samples/pretrain_tiny.jsonl
```

## 整体流程

```text
JSONL 文件
   |
   v
PretrainDataset
   |
   v
tokenizer.encode(text)
   |
   v
token ids
   |
   v
CausalLMCollator
   |
   v
input_ids / attention_mask / labels
   |
   v
MiniMindForCausalLM
```

## JSONL 数据格式

当前默认格式是一行一个 JSON：

```json
{"text": "MiniMind AgentLab is a small language model training project."}
```

每一行都应该有一个 `text` 字段。

后面如果 MiniMind-compatible 数据字段名不同，可以通过参数修改：

```bash
python scripts/inspect_batch.py --text-field text
```

## Tokenizer Wrapper

文件：

```text
src/data/tokenizer.py
```

我们给 tokenizer 做了一层统一接口，让后续数据管线只依赖这几个能力：

```python
tokenizer.encode(text)
tokenizer.decode(token_ids)
tokenizer.pad_token_id
tokenizer.eos_token_id
tokenizer.vocab_size
```

优先使用真实 tokenizer：

```text
data/tokenizer
```

加载器支持：

- Hugging Face tokenizer 目录
- `tokenizers` 包的 `tokenizer.json`
- 临时 `ByteTokenizer` fallback

## ByteTokenizer 是什么

`ByteTokenizer` 只是为了本地测试。

它把 UTF-8 byte 映射成 token id：

```text
pad_token_id = 0
eos_token_id = 1
byte value b -> token id b + 2
```

所以它的 vocab size 是：

```text
258
```

它的作用是：即使我们还没有下载 MiniMind 的正式 tokenizer，也可以先测试数据管线。

它不是最终训练要用的 tokenizer。

## PretrainDataset 的作用

文件：

```text
src/data/pretrain_dataset.py
```

`PretrainDataset` 是一个 `IterableDataset`。

为什么用 iterable？

因为预训练数据以后可能很大，不适合一次性全部读进内存。IterableDataset 可以一行一行 stream 数据。

每一行处理流程是：

1. 解析 JSON
2. 读取 `text` 字段
3. 用 tokenizer 编码成 token IDs
4. 如果有 EOS，就追加 EOS
5. 截断到 `max_seq_len`
6. yield `{"input_ids": token_ids}`

## CausalLMCollator 的作用

文件：

```text
src/data/collator.py
```

collator 的作用是把不同长度的样本 padding 成同一个 batch。

假设两个样本长度不同：

```text
example 1 length = 5
example 2 length = 3
```

padding 后：

```text
input_ids:
[t0, t1, t2, t3, t4]
[t0, t1, t2, PAD, PAD]

attention_mask:
[1, 1, 1, 1, 1]
[1, 1, 1, 0, 0]

labels:
[t0, t1, t2, t3, t4]
[t0, t1, t2, -100, -100]
```

为什么 labels 先复制 input_ids？

因为模型内部会做 shift，用当前位置预测下一个 token。

为什么 padding label 是 `-100`？

因为 PyTorch 的 cross entropy 会忽略 label 为 `-100` 的位置。

这样模型不会被迫学习预测 padding token。

## Batch 检查脚本

文件：

```text
scripts/inspect_batch.py
```

运行：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\inspect_batch.py
```

当前输出类似：

```text
config: configs/pretrain_tiny.yaml
data_path: data/samples/pretrain_tiny.jsonl
tokenizer: ByteTokenizer
vocab_size: 258
pad_token_id: 0
eos_token_id: 1
num_examples: 2
input_ids shape: (2, 62)
attention_mask shape: (2, 62)
labels shape: (2, 62)
attention lengths: [62, 61]
num ignored labels: 1
first decoded sample:
MiniMind AgentLab is a small language model training project.
```

这证明：

- JSONL 能被读取
- 文本能被 tokenize
- batch 能正确 padding
- `attention_mask` 能区分真实 token 和 padding
- padding label 被设成 `-100`
- decode 能用于人工检查

## 和模型怎么接起来

collator 输出：

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}
```

这正好对应模型 forward：

```python
model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
```

所以第四步的意义是：把原始文本数据接到了第三步实现的模型上。

## 面试时怎么讲

可以这样说：

```text
我实现了一个用于 causal LM pretraining 的 streaming JSONL 数据管线。每一行数据包含 text 字段，dataset 会读取文本、tokenize、追加 EOS、按 max sequence length 截断，并产出 token IDs。collator 会在 batch 内动态 padding，生成 attention mask，并把 padding 位置的 labels 设为 -100，让 cross entropy 忽略这些位置。

我还做了 tokenizer abstraction，后面可以接入 MiniMind-compatible tokenizer；在真实 tokenizer 还没准备好时，也可以用 byte-level fallback 做本地 smoke test，先验证数据管线逻辑。
```

## 当前状态

已完成：

- tokenizer 加载抽象
- Hugging Face tokenizer 支持
- `tokenizers` JSON 支持
- byte tokenizer fallback
- streaming JSONL dataset
- 动态 padding collator
- padding label 设为 `-100`
- tiny sample JSONL
- `inspect_batch.py`

下一步：

- 接入真实 MiniMind-compatible tokenizer
- 接入真实 pretraining JSONL
- 把数据管线接入训练 loop
