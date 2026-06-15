# Step 7: Generation

This document explains the seventh milestone of MiniMind AgentLab: loading a checkpoint and generating text.

## Goal

The goal of this step is to close the training-to-inference loop.

After training saves a checkpoint, we should be able to:

- load model config from checkpoint
- rebuild the model
- load model weights
- load tokenizer
- encode a prompt
- generate new tokens autoregressively
- decode generated token IDs
- save generated text to `outputs/`

## Files Involved

```text
scripts/generate.py
src/model/modeling_minimind.py
src/data/tokenizer.py
checkpoints/pretrain_tiny/latest.pt
outputs/generated_sample.txt
```

## Generation Flow

```text
checkpoint
   |
   v
model_config + model_state_dict
   |
   v
MiniMindForCausalLM
   |
   v
prompt text
   |
   v
tokenizer.encode(prompt)
   |
   v
autoregressive decoding
   |
   v
tokenizer.decode(token_ids)
   |
   v
generated text
```

## Autoregressive Decoding

The model generates one token at a time.

For each step:

```text
take current token sequence
run model forward
take logits from last position
choose next token
append next token
repeat
```

Current implementation does not use KV cache yet. That is fine for the first version.

## Sampling Methods

The generation script supports:

- greedy decoding
- temperature sampling
- top-k filtering
- top-p filtering

Greedy decoding:

```text
choose token with highest logit
```

Temperature:

```text
higher temperature = more random
lower temperature = more deterministic
```

Top-k:

```text
only sample from the k most likely tokens
```

Top-p:

```text
only sample from the smallest token set whose probability mass exceeds p
```

## Commands

Greedy generation:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\generate.py --checkpoint checkpoints\pretrain_tiny\latest.pt --prompt MiniMind --max-new-tokens 32 --temperature 0
```

Sampled generation:

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\generate.py --checkpoint checkpoints\pretrain_tiny\latest.pt --prompt MiniMind --max-new-tokens 32 --temperature 0.8 --top-k 20 --output outputs\generated_sample_topk.txt
```

## Current Verified Output

Greedy tiny checkpoint output:

```text
MiniMinddddddddddddddddddddddddddddddddd
```

This output is not meaningful yet. That is expected because:

- the model is tiny
- data is tiny
- training only ran a few steps
- tokenizer is the temporary ByteTokenizer

The important thing is that the checkpoint-to-generation pipeline works.

## Why Generation Matters

Generation proves that training artifacts are usable outside the training loop.

It validates:

- checkpoint loading
- model reconstruction
- tokenizer loading
- prompt encoding
- forward pass in inference mode
- autoregressive decoding
- output saving

Without generation, we only know the model can train. With generation, we know the model can also be used.

## How To Explain This In An Interview

You can say:

```text
I added a generation script that loads model config and weights directly from a checkpoint, rebuilds the causal LM, loads the tokenizer, encodes a prompt, and performs autoregressive decoding. It supports greedy decoding, temperature sampling, top-k, and top-p. For the first version I did not implement KV cache, because the goal was to close the checkpoint-to-inference loop before optimizing generation speed.
```

## Status

Implemented:

- checkpoint loading for inference
- tokenizer loading
- prompt encoding
- greedy decoding
- temperature sampling
- top-k
- top-p
- output saving

Next:

- improve generation quality with longer training
- add KV cache later
- test with real MiniMind tokenizer and 64M checkpoint

---

# 第七步：Generation 中文说明

这一节解释 MiniMind AgentLab 的第七个里程碑：从 checkpoint 加载模型并生成文本。

## 目标

第七步的目标是打通训练到推理的闭环。

训练保存 checkpoint 之后，我们应该能够：

- 从 checkpoint 读取 model config
- 重建模型
- 加载模型权重
- 加载 tokenizer
- encode prompt
- 自回归生成新 token
- decode token IDs
- 把生成文本保存到 `outputs/`

## 相关文件

```text
scripts/generate.py
src/model/modeling_minimind.py
src/data/tokenizer.py
checkpoints/pretrain_tiny/latest.pt
outputs/generated_sample.txt
```

## 生成流程

```text
checkpoint
   |
   v
model_config + model_state_dict
   |
   v
MiniMindForCausalLM
   |
   v
prompt text
   |
   v
tokenizer.encode(prompt)
   |
   v
autoregressive decoding
   |
   v
tokenizer.decode(token_ids)
   |
   v
generated text
```

## 自回归生成是什么

模型一次生成一个 token。

每一步：

```text
拿当前已有 token 序列
跑一次模型 forward
取最后一个位置的 logits
选择下一个 token
把新 token 拼到序列后面
重复
```

当前版本还没有 KV cache。第一版这样可以接受，因为我们的目标是先打通推理闭环，不是优化推理速度。

## 支持的采样方式

当前 generation 脚本支持：

- greedy decoding
- temperature sampling
- top-k filtering
- top-p filtering

greedy decoding：

```text
每一步选择 logit 最大的 token
```

temperature：

```text
temperature 越高，生成越随机
temperature 越低，生成越确定
```

top-k：

```text
只从概率最高的 k 个 token 中采样
```

top-p：

```text
只从累计概率超过 p 的最小 token 集合中采样
```

## 运行命令

greedy generation：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\generate.py --checkpoint checkpoints\pretrain_tiny\latest.pt --prompt MiniMind --max-new-tokens 32 --temperature 0
```

采样 generation：

```bash
C:\Users\zhish\anaconda3\envs\pytorch\python.exe scripts\generate.py --checkpoint checkpoints\pretrain_tiny\latest.pt --prompt MiniMind --max-new-tokens 32 --temperature 0.8 --top-k 20 --output outputs\generated_sample_topk.txt
```

## 当前验证输出

tiny checkpoint 的 greedy 输出是：

```text
MiniMinddddddddddddddddddddddddddddddddd
```

这个输出质量不好是正常的，因为：

- 模型很小
- 数据很小
- 只训练了几步
- tokenizer 还是临时 ByteTokenizer

这里重要的不是生成质量，而是 checkpoint-to-generation 的流程已经跑通。

## 为什么 generation 重要

generation 证明训练产物可以离开训练 loop，被真正用于推理。

它验证了：

- checkpoint 加载
- 模型重建
- tokenizer 加载
- prompt encoding
- inference forward
- autoregressive decoding
- 输出保存

没有 generation，我们只能说模型能训练。  
有了 generation，我们可以说模型能被加载和使用。

## 面试时怎么讲

可以这样说：

```text
我实现了一个 generation script，可以直接从 checkpoint 读取模型配置和权重，重建 causal LM，加载 tokenizer，encode prompt，然后做 autoregressive decoding。它支持 greedy decoding、temperature sampling、top-k 和 top-p。第一版还没有实现 KV cache，因为我优先打通 checkpoint 到 inference 的闭环，后续再优化生成速度。
```

## 当前状态

已完成：

- inference checkpoint loading
- tokenizer loading
- prompt encoding
- greedy decoding
- temperature sampling
- top-k
- top-p
- output saving

下一步：

- 通过更长训练提升生成质量
- 后续加入 KV cache
- 用真实 MiniMind tokenizer 和 64M checkpoint 测试
