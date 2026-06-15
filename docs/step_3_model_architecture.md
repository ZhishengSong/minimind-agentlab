# Step 3: Model Architecture

This document explains the third milestone of MiniMind AgentLab: the MiniMind-style causal language model.

## Goal

The goal of this step is to implement a real causal language model, not a toy placeholder.

The model now supports:

- token embeddings
- RMSNorm
- RoPE
- SwiGLU MLP
- grouped-query attention
- pre-norm Transformer blocks
- tied word embeddings
- causal language modeling loss
- `input_ids`
- `attention_mask`
- `position_ids`
- `labels`
- `ignore_index=-100`

## Files Involved

```text
src/model/config.py
src/model/norm.py
src/model/rope.py
src/model/mlp.py
src/model/attention.py
src/model/modeling_minimind.py
src/utils/param_count.py
scripts/inspect_model.py
```

## Current 64M Config

File:

```text
configs/minimind_64m.yaml
```

Current architecture:

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

Important derived values:

```text
head_dim = hidden_size / num_attention_heads
         = 512 / 8
         = 64

num_key_value_groups = num_attention_heads / num_key_value_heads
                     = 8 / 2
                     = 4
```

The inspected parameter count is:

```text
63.06M parameters
```

This is close to the 64M target.

## Overall Forward Flow

```text
input_ids
   |
   v
token embedding
   |
   v
Transformer block x 12
   |
   v
final RMSNorm
   |
   v
lm_head
   |
   v
logits
```

With labels:

```text
logits + labels
   |
   v
shift by one token
   |
   v
cross entropy loss
```

## Tensor Shapes

For a batch of size `B`, sequence length `T`, hidden size `C`, and vocab size `V`:

```text
input_ids:      [B, T]
hidden_states:  [B, T, C]
logits:         [B, T, V]
labels:         [B, T]
loss:           scalar
```

For attention:

```text
q: [B, num_attention_heads, T, head_dim]
k: [B, num_key_value_heads, T, head_dim]
v: [B, num_key_value_heads, T, head_dim]
```

With the current config:

```text
q: [B, 8, T, 64]
k: [B, 2, T, 64]
v: [B, 2, T, 64]
```

After repeating KV heads for GQA:

```text
k: [B, 8, T, 64]
v: [B, 8, T, 64]
```

## RMSNorm

File:

```text
src/model/norm.py
```

RMSNorm normalizes by the root mean square of the hidden dimension:

```text
x_norm = x / sqrt(mean(x^2) + eps)
```

Then it applies a learned scale vector.

Unlike LayerNorm, RMSNorm does not subtract the mean. It is common in modern LLMs because it is simple, stable, and efficient.

## RoPE

File:

```text
src/model/rope.py
```

RoPE means rotary positional embedding.

Instead of adding a learned position vector to token embeddings, RoPE rotates query and key vectors based on token position.

In this project:

- RoPE is applied to Q and K
- V is not rotated
- cos/sin caches are precomputed
- longer sequences can expand the cache
- `position_ids` are supported

Why Q and K only?

Attention scores come from:

```text
Q @ K^T
```

So rotating Q and K changes attention scores according to relative positions.

## SwiGLU MLP

File:

```text
src/model/mlp.py
```

The MLP uses SwiGLU:

```text
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

It has three projections:

- `gate_proj`
- `up_proj`
- `down_proj`

This is the same broad feed-forward pattern used by many modern decoder-only LLMs.

## Grouped-Query Attention

File:

```text
src/model/attention.py
```

Standard multi-head attention uses the same number of Q, K, and V heads.

Grouped-query attention uses:

```text
num_attention_heads > num_key_value_heads
```

In our config:

```text
8 query heads
2 key/value heads
```

Each KV head is shared by 4 Q heads.

Why use GQA?

- fewer K/V projection parameters
- smaller KV cache later during generation
- close to modern LLM architecture choices

## Causal Mask And Padding Mask

The attention module supports causal language modeling.

Causal mask:

```text
token i can only attend to tokens <= i
```

Padding mask:

```text
real tokens are visible
padding tokens are hidden
```

When `attention_mask` is provided, the model combines causal masking and padding masking before calling PyTorch scaled dot-product attention.

## Transformer Block

File:

```text
src/model/modeling_minimind.py
```

Each block uses pre-norm:

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Pre-norm is usually more stable for training deep Transformers because normalization happens before each sublayer.

## Causal LM Loss

The model computes next-token prediction loss.

Given:

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3]
```

The code shifts logits and labels:

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

Padding labels should be `-100`, and PyTorch cross entropy ignores those positions through:

```python
ignore_index=-100
```

## Tied Word Embeddings

The model supports tied embeddings:

```python
lm_head.weight = embed_tokens.weight
```

This means the input token embedding matrix and output vocabulary projection share parameters.

Benefits:

- fewer parameters
- common in language models
- often improves parameter efficiency

## Inspect Command

Use the `pytorch` conda environment:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\inspect_model.py --batch-size 1 --seq-len 16
```

Expected output:

```text
config: configs/minimind_64m.yaml
hidden_size/layers/heads: 512/12/8
kv_heads/head_dim: 2/64
logits shape: (1, 16, 6400)
loss: ...
total params: 63058432 (63.06M)
trainable params: 63058432 (63.06M)
```

This proves:

- model config loads
- model initializes
- forward pass works
- logits shape is correct
- causal LM loss works
- backward pass works
- parameter count is close to 64M

## How To Explain This In An Interview

You can say:

```text
I implemented a decoder-only causal language model in PyTorch using modern small-LLM components: RMSNorm, RoPE, SwiGLU MLP, grouped-query attention, pre-norm Transformer blocks, and tied word embeddings. The model takes input token IDs, applies token embeddings, runs them through 12 Transformer blocks, normalizes the final hidden states, and projects to vocabulary logits.

For training, I implemented next-token prediction by shifting logits and labels and using cross entropy with ignore_index=-100 for padded positions. I also added an inspection script that validates forward and backward passes and reports that the current configuration has about 63M parameters.
```

## Status

Implemented:

- RMSNorm
- RoPE
- SwiGLU MLP
- GQA attention
- Transformer block
- Causal LM
- tied embeddings
- loss calculation
- parameter counting
- model inspection script

Next:

- tokenizer wrapper
- pretraining dataset
- collator
- batch inspection

---

# 第三步：模型结构中文说明

这一节解释 MiniMind AgentLab 的第三个里程碑：MiniMind-style causal language model。

## 目标

第三步的目标是实现一个真正能 forward/backward 的 decoder-only causal LM，而不是空壳或 toy demo。

目前模型已经支持：

- token embedding
- RMSNorm
- RoPE
- SwiGLU MLP
- grouped-query attention
- pre-norm Transformer block
- tied word embeddings
- causal language modeling loss
- `input_ids`
- `attention_mask`
- `position_ids`
- `labels`
- `ignore_index=-100`

## 相关文件

```text
src/model/config.py
src/model/norm.py
src/model/rope.py
src/model/mlp.py
src/model/attention.py
src/model/modeling_minimind.py
src/utils/param_count.py
scripts/inspect_model.py
```

## 当前 64M 模型配置

文件：

```text
configs/minimind_64m.yaml
```

当前结构是：

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

关键派生值：

```text
head_dim = hidden_size / num_attention_heads
         = 512 / 8
         = 64

num_key_value_groups = num_attention_heads / num_key_value_heads
                     = 8 / 2
                     = 4
```

当前检查出来的参数量是：

```text
63.06M parameters
```

这已经非常接近 64M 目标。

## 整体 forward 流程

```text
input_ids
   |
   v
token embedding
   |
   v
12 层 Transformer block
   |
   v
final RMSNorm
   |
   v
lm_head
   |
   v
logits
```

如果传入 `labels`：

```text
logits + labels
   |
   v
错开一位做 next-token prediction
   |
   v
cross entropy loss
```

## 主要 tensor shape

假设 batch size 是 `B`，序列长度是 `T`，hidden size 是 `C`，词表大小是 `V`：

```text
input_ids:      [B, T]
hidden_states:  [B, T, C]
logits:         [B, T, V]
labels:         [B, T]
loss:           scalar
```

attention 里面：

```text
q: [B, num_attention_heads, T, head_dim]
k: [B, num_key_value_heads, T, head_dim]
v: [B, num_key_value_heads, T, head_dim]
```

当前配置下：

```text
q: [B, 8, T, 64]
k: [B, 2, T, 64]
v: [B, 2, T, 64]
```

GQA 重复 KV heads 之后：

```text
k: [B, 8, T, 64]
v: [B, 8, T, 64]
```

## RMSNorm

文件：

```text
src/model/norm.py
```

RMSNorm 的公式是：

```text
x_norm = x / sqrt(mean(x^2) + eps)
```

然后乘上一个可学习的 scale 参数。

它和 LayerNorm 的区别是：RMSNorm 不减均值，只按均方根做缩放。

现代 LLM 经常用 RMSNorm，因为它更简单，计算更轻，也比较稳定。

## RoPE

文件：

```text
src/model/rope.py
```

RoPE 是 rotary positional embedding，也就是旋转位置编码。

它不是把位置 embedding 加到 token embedding 上，而是根据 token 位置旋转 Q 和 K。

在我们的实现里：

- RoPE 作用在 Q 和 K 上
- V 不做 RoPE
- cos/sin 会提前缓存
- 如果序列更长，可以扩展缓存
- 支持 `position_ids`

为什么只作用在 Q/K 上？

因为 attention score 来自：

```text
Q @ K^T
```

旋转 Q 和 K 会让 attention score 带上位置信息，尤其是相对位置信息。

## SwiGLU MLP

文件：

```text
src/model/mlp.py
```

MLP 使用 SwiGLU：

```text
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

它有三个线性层：

- `gate_proj`
- `up_proj`
- `down_proj`

这种结构比普通 FFN 更接近现代 decoder-only LLM 的设计。

## GQA：Grouped-Query Attention

文件：

```text
src/model/attention.py
```

普通 multi-head attention 里，Q/K/V head 数量一样。

GQA 里：

```text
num_attention_heads > num_key_value_heads
```

当前配置：

```text
8 个 query heads
2 个 key/value heads
```

也就是说，每个 KV head 被 4 个 Q head 共享。

为什么用 GQA？

- K/V 投影参数更少
- 以后做 generation 时 KV cache 更小
- 更接近现代 LLM 架构

## Causal Mask 和 Padding Mask

模型支持 causal language modeling。

causal mask 的含义：

```text
第 i 个 token 只能看见自己和它之前的 token
```

padding mask 的含义：

```text
真实 token 可以被看见
padding token 不应该被看见
```

当传入 `attention_mask` 时，模型会把 causal mask 和 padding mask 合在一起，再送进 PyTorch 的 scaled dot-product attention。

## Transformer Block

文件：

```text
src/model/modeling_minimind.py
```

每一层 block 是 pre-norm 结构：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

pre-norm 的好处是训练更稳定，因为每个子层之前都会先归一化。

## Causal LM Loss

模型做的是 next-token prediction。

例子：

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3]
```

代码里会错开 logits 和 labels：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

padding 位置的 label 应该设成 `-100`，然后 cross entropy 通过：

```python
ignore_index=-100
```

忽略这些位置。

## Tied Word Embeddings

模型支持 tied embeddings：

```python
lm_head.weight = embed_tokens.weight
```

意思是输入 token embedding 和输出词表投影共享同一组参数。

好处：

- 参数更少
- LLM 里很常见
- 对小模型更省参数

## 模型检查命令

使用你的 `pytorch` conda 环境：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\inspect_model.py --batch-size 1 --seq-len 16
```

预期输出类似：

```text
config: configs/minimind_64m.yaml
hidden_size/layers/heads: 512/12/8
kv_heads/head_dim: 2/64
logits shape: (1, 16, 6400)
loss: ...
total params: 63058432 (63.06M)
trainable params: 63058432 (63.06M)
```

这说明：

- 模型配置能加载
- 模型能初始化
- forward pass 正常
- logits shape 正确
- causal LM loss 正常
- backward pass 正常
- 参数量接近 64M

## 面试时怎么讲

可以这样说：

```text
我用 PyTorch 从零实现了一个 decoder-only causal LM，包含现代小型 LLM 常见组件：RMSNorm、RoPE、SwiGLU MLP、GQA、pre-norm Transformer blocks 和 tied embeddings。模型输入 token IDs，先经过 embedding，再经过 12 层 Transformer block，最后经过 RMSNorm 和 lm_head 得到 vocabulary logits。

训练目标是 next-token prediction，所以我对 logits 和 labels 做 shift，然后用 cross entropy 计算 causal LM loss，并用 ignore_index=-100 忽略 padding 位置。我还写了 inspect_model 脚本，验证 forward/backward、logits shape 和参数量，目前配置大约是 63M 参数。
```

## 当前状态

已完成：

- RMSNorm
- RoPE
- SwiGLU MLP
- GQA attention
- Transformer block
- Causal LM
- tied embeddings
- loss 计算
- 参数量统计
- 模型检查脚本

下一步：

- tokenizer wrapper
- pretraining dataset
- collator
- batch inspection
