# MiniMind AgentLab 项目 To-Do List

## 0. 项目总目标

- [ ] 从零搭建一个 MiniMind-style 小型语言模型训练框架
- [ ] 使用正式的小模型结构、正式 tokenizer、正式数据格式，而不是 toy demo
- [ ] 先完成 base model pretraining pipeline
- [ ] 后续扩展到 SFT、tool-use verifier、Agentic RL、reward hacking analysis、SOC-inspired decoding 和 dLLM 小实验
- [ ] 最终形成一个可以写进简历、可以在面试中清楚讲解的 LLM research / engineering project

---

## 1. Repository 与工程结构

- [ ] 新建项目仓库，例如 `minimind-agentlab`
- [ ] 建立标准目录结构：

```text
minimind-agentlab/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
├── src/
│   ├── model/
│   ├── data/
│   ├── train/
│   └── utils/
├── data/
├── checkpoints/
├── logs/
└── outputs/
```

- [ ] 添加 `.gitignore`
- [ ] 排除大文件：
  - [ ] `checkpoints/`
  - [ ] `data/*.jsonl`
  - [ ] `logs/`
  - [ ] `outputs/`
- [ ] 写 `requirements.txt`
- [ ] 写 README 第一版
- [ ] 说明项目不是 fork 改几行，而是 from-scratch MiniMind-style implementation

---

## 2. 配置系统

- [ ] 创建 `configs/minimind_64m.yaml`
- [ ] 创建 `configs/pretrain.yaml`
- [ ] 配置模型参数：
  - [ ] `vocab_size`
  - [ ] `hidden_size`
  - [ ] `num_hidden_layers`
  - [ ] `num_attention_heads`
  - [ ] `num_key_value_heads`
  - [ ] `intermediate_size`
  - [ ] `max_position_embeddings`
  - [ ] `rope_theta`
  - [ ] `rms_norm_eps`
  - [ ] `tie_word_embeddings`
- [ ] 配置训练参数：
  - [ ] batch size
  - [ ] gradient accumulation
  - [ ] learning rate
  - [ ] weight decay
  - [ ] warmup steps
  - [ ] cosine scheduler
  - [ ] max steps
  - [ ] save interval
  - [ ] log interval
  - [ ] sample interval
  - [ ] dtype, e.g. `bf16`
  - [ ] device
- [ ] 写配置读取函数
- [ ] 支持命令行覆盖配置参数

---

## 3. 模型结构实现

目标：实现一个正式的 MiniMind-style causal LM。

### 3.1 基础模块

- [ ] `src/model/config.py`
  - [ ] 定义模型 config dataclass
  - [ ] 支持从 yaml 加载
  - [ ] 支持保存 config 到 checkpoint

- [ ] `src/model/norm.py`
  - [ ] 实现 RMSNorm
  - [ ] 检查 shape 是否正确
  - [ ] 检查数值稳定性

- [ ] `src/model/rope.py`
  - [ ] 实现 RoPE 频率预计算
  - [ ] 实现 `apply_rotary_pos_emb`
  - [ ] 支持长上下文位置编码
  - [ ] 支持不同 `rope_theta`

- [ ] `src/model/mlp.py`
  - [ ] 实现 SwiGLU MLP
  - [ ] 结构：`gate_proj`, `up_proj`, `down_proj`
  - [ ] 激活函数使用 SiLU

### 3.2 Attention

- [ ] `src/model/attention.py`
  - [ ] 实现 causal self-attention
  - [ ] 支持 multi-head attention
  - [ ] 支持 grouped-query attention, GQA
  - [ ] 支持 `num_attention_heads != num_key_value_heads`
  - [ ] 实现 causal mask
  - [ ] 支持 attention mask
  - [ ] 预留 KV cache 接口
  - [ ] 检查输出 shape

### 3.3 Transformer Block

- [ ] 实现 Transformer block
- [ ] 使用 pre-norm 结构
- [ ] attention 前 RMSNorm
- [ ] MLP 前 RMSNorm
- [ ] residual connection
- [ ] dropout 可配置，但第一版可设为 0

### 3.4 Causal LM

- [ ] `src/model/modeling_minimind.py`
  - [ ] token embedding
  - [ ] 多层 Transformer block
  - [ ] final RMSNorm
  - [ ] lm head
  - [ ] 支持 tied word embeddings
  - [ ] 支持 `input_ids`
  - [ ] 支持 `attention_mask`
  - [ ] 支持 `position_ids`
  - [ ] 支持 `labels`
  - [ ] 支持 `ignore_index=-100`
  - [ ] 返回 `loss` 和 `logits`

### 3.5 参数量检查

- [ ] 写 `src/utils/param_count.py`
- [ ] 打印总参数量
- [ ] 打印 trainable 参数量
- [ ] 检查参数量是否接近目标 64M
- [ ] 在 README 中记录模型配置与参数量

---

## 4. Tokenizer 与数据管线

目标：第一版接入 MiniMind-compatible tokenizer，不重新训练 tokenizer。

### 4.1 Tokenizer

- [ ] `src/data/tokenizer.py`
- [ ] 加载 tokenizer 文件
- [ ] 支持 encode
- [ ] 支持 decode
- [ ] 支持 special tokens
- [ ] 检查 vocab size 是否和模型配置一致
- [ ] 保存 tokenizer 路径到 checkpoint/config

### 4.2 Pretrain Dataset

- [ ] `src/data/pretrain_dataset.py`
- [ ] 读取 `pretrain_t2t_mini.jsonl`
- [ ] 每行解析 JSON
- [ ] 读取 `text` 字段
- [ ] 文本转 token ids
- [ ] 支持 max sequence length 截断
- [ ] 支持 iterable dataset，避免一次性全部读入内存
- [ ] 支持 shuffle buffer，后续可加

### 4.3 Collator

- [ ] `src/data/collator.py`
- [ ] batch 内 padding
- [ ] 生成 `input_ids`
- [ ] 生成 `labels`
- [ ] padding 位置 label 设置为 `-100`
- [ ] 生成 `attention_mask`
- [ ] 检查 batch shape
- [ ] 写 `scripts/inspect_batch.py` 查看 batch 是否正确

---

## 5. Pretraining Trainer

目标：正式训练脚本能够运行 64M 模型和 MiniMind-compatible pretraining data。

### 5.1 训练入口

- [ ] `scripts/train_pretrain.py`
- [ ] 支持 `--config`
- [ ] 支持 `--resume`
- [ ] 支持选择 device
- [ ] 支持 bf16 / fp16 / fp32
- [ ] 支持单卡训练
- [ ] 后续预留 DDP 接口

### 5.2 Optimizer 与 Scheduler

- [ ] `src/train/optim.py`
- [ ] AdamW optimizer
- [ ] 设置 weight decay
- [ ] 对 norm / bias 参数可选择不加 weight decay
- [ ] warmup scheduler
- [ ] cosine decay scheduler
- [ ] 支持 min learning rate

### 5.3 Training Loop

- [ ] forward
- [ ] loss 计算
- [ ] backward
- [ ] gradient accumulation
- [ ] gradient clipping
- [ ] optimizer step
- [ ] scheduler step
- [ ] zero grad
- [ ] global step 统计
- [ ] token 数统计
- [ ] loss 平滑统计

### 5.4 Mixed Precision

- [ ] 支持 `torch.autocast`
- [ ] 支持 bf16
- [ ] 如果使用 fp16，加入 GradScaler
- [ ] 检查 NaN loss
- [ ] 出现 NaN 时保存 debug 信息

---

## 6. Checkpoint 与 Resume

目标：训练中断后可以继续，不丢状态。

- [ ] `src/train/checkpoint.py`
- [ ] 保存 model state dict
- [ ] 保存 optimizer state dict
- [ ] 保存 scheduler state dict
- [ ] 保存 global step
- [ ] 保存 config
- [ ] 保存 random state
- [ ] 保存 CUDA random state
- [ ] 保存 tokenizer 信息
- [ ] 保存 `latest.pt`
- [ ] 按 step 保存 `pretrain_step_xxxxxx.pt`
- [ ] 支持从 checkpoint resume
- [ ] resume 后 lr、step、optimizer 状态正确
- [ ] 测试中断重启是否正常

---

## 7. Logging 与实验记录

目标：从第一版开始就保留可复现实验记录。

- [ ] `src/train/logger.py`
- [ ] 写 JSONL 日志
- [ ] 每隔若干 step 记录：
  - [ ] step
  - [ ] loss
  - [ ] learning rate
  - [ ] tokens/sec
  - [ ] GPU memory
  - [ ] elapsed time
  - [ ] grad norm
- [ ] 输出到 `logs/pretrain_metrics.jsonl`
- [ ] 后续可接入 wandb / tensorboard，但第一版不强制
- [ ] 保存训练命令
- [ ] 保存 git commit hash，后续可加
- [ ] 保存完整 config copy

---

## 8. Generation Script

目标：训练过程中和训练后都能生成样例。

- [ ] `scripts/generate.py`
- [ ] 从 checkpoint 加载模型
- [ ] 从 config 重建模型
- [ ] 加载 tokenizer
- [ ] 支持 prompt 输入
- [ ] 支持 `max_new_tokens`
- [ ] 支持 greedy decoding
- [ ] 支持 temperature sampling
- [ ] 支持 top-k
- [ ] 支持 top-p
- [ ] 支持 repetition penalty，后续可加
- [ ] 输出 decoded text
- [ ] 保存 sample 到 `outputs/`
- [ ] 预留 KV cache 接口

---

## 9. 第一版 Pretrain Smoke Run

目标：不是 toy，而是正式模型 + 正式 tokenizer + 正式数据的短训练验证。

- [ ] 使用 64M 模型配置
- [ ] 使用 MiniMind-compatible tokenizer
- [ ] 使用 `pretrain_t2t_mini.jsonl`
- [ ] 跑通 100 step
- [ ] 跑通 500 step
- [ ] 检查 loss 是否合理下降
- [ ] 检查 tokens/sec
- [ ] 检查显存占用
- [ ] 检查 checkpoint 是否保存
- [ ] 检查 resume 是否可用
- [ ] 检查 generate 是否可用
- [ ] 保存 sample outputs
- [ ] 在 README 记录第一轮结果

---

## 10. README 与项目说明

- [ ] README 写项目定位
- [ ] 说明目标：
  - [ ] from-scratch MiniMind-style small LLM
  - [ ] agentic post-training research framework
  - [ ] verifier-based RL
  - [ ] SOC-inspired decoding
  - [ ] dLLM pilot experiment
- [ ] 写安装方式
- [ ] 写数据准备方式
- [ ] 写训练命令
- [ ] 写生成命令
- [ ] 写 checkpoint/resume 方法
- [ ] 写当前进度
- [ ] 写实验表格
- [ ] 写后续计划
- [ ] 说明和原始 MiniMind 的关系：inspired by / compatible with, but independently implemented

---

## 11. 后续模块：SFT

暂时不在第一阶段实现，但需要预留接口。

- [ ] 支持 chat template
- [ ] 支持 instruction tuning 数据格式
- [ ] 支持 assistant-only loss
- [ ] 支持 padding label mask
- [ ] 实现 `scripts/train_sft.py`
- [ ] 对比 pretrain-only 和 SFT 模型输出
- [ ] 建立 SFT eval prompts

---

## 12. 后续模块：Tool-use Verifier

这是项目主创新方向之一。

- [ ] 设计 tool-use benchmark
- [ ] 实现工具：
  - [ ] calculator
  - [ ] date_diff
  - [ ] unit_converter
  - [ ] json_extractor
- [ ] 定义 tool call 格式
- [ ] 实现 tool call parser
- [ ] 实现 trajectory parser
- [ ] 实现 verifier
- [ ] 打分维度：
  - [ ] format score
  - [ ] tool selection score
  - [ ] argument correctness score
  - [ ] final answer correctness score
  - [ ] efficiency score
- [ ] 实现 `scripts/eval_tooluse.py`
- [ ] 输出 tool format accuracy
- [ ] 输出 tool selection accuracy
- [ ] 输出 final answer accuracy
- [ ] 输出 pass rate

---

## 13. 后续模块：Agentic RL / GRPO

这是项目主线研究模块。

- [ ] 实现 rollout
- [ ] 同一 prompt 采样多条 trajectory
- [ ] 用 verifier 给每条 trajectory 打 reward
- [ ] 实现 reward normalization
- [ ] 实现 KL penalty
- [ ] 实现简化版 GRPO update
- [ ] 支持训练日志：
  - [ ] average reward
  - [ ] pass rate
  - [ ] KL
  - [ ] policy loss
  - [ ] entropy
- [ ] 对比 SFT vs GRPO
- [ ] 分析 RL 是否提升 tool-use pass rate

---

## 14. 后续模块：Reward Hacking Analysis

目标：分析 verifier-based RL 的失败模式。

- [ ] 设计 adversarial prompts
- [ ] 检测 fake tool observation
- [ ] 检测 malformed JSON
- [ ] 检测 parser exploit
- [ ] 检测 repeated tool calls
- [ ] 检测 multiple final answers
- [ ] 检测 hallucinated tool result
- [ ] 建立 failure taxonomy
- [ ] 输出 failure case table
- [ ] 写 `reports/reward_hacking.md`

---

## 15. 后续模块：SOC-inspired Decoding

目标：把生成看成 inference-time optimization / sequential decision problem。

- [ ] 实现 best-of-N sampling
- [ ] 实现 verifier-guided reranking
- [ ] 实现 temperature annealing
- [ ] 实现 reward-guided beam search，后续可选
- [ ] 比较 decoding 方法：
  - [ ] greedy
  - [ ] top-p
  - [ ] best-of-4
  - [ ] best-of-8
  - [ ] annealed sampling
  - [ ] verifier-guided reranking
- [ ] 评估指标：
  - [ ] pass rate
  - [ ] latency
  - [ ] average generated tokens
  - [ ] average tool calls
- [ ] 写 `reports/soc_decoding.md`

---

## 16. 后续模块：dLLM 小实验

目标：做一个 AR-to-diffusion language model pilot。

- [ ] 实现 masked LM objective
- [ ] 改 causal attention 为 bidirectional attention
- [ ] 实现 mask schedule
- [ ] 实现 iterative unmask sampling
- [ ] 比较：
  - [ ] full finetune
  - [ ] attention-only finetune
  - [ ] QK-only / LoRA-style finetune
- [ ] 评估：
  - [ ] mask reconstruction accuracy
  - [ ] generation quality
  - [ ] repetition rate
  - [ ] inference speed
- [ ] 写 `reports/dllm_experiment.md`

---

## 17. 最终实验结果整理

- [ ] pretrain loss curve
- [ ] SFT 对比表
- [ ] tool-use benchmark 表
- [ ] SFT vs GRPO 表
- [ ] reward hacking failure table
- [ ] SOC decoding accuracy-latency 表
- [ ] dLLM 小实验结果
- [ ] sample generations
- [ ] demo screenshots
- [ ] final README
- [ ] final technical report
- [ ] final resume bullets

---

## 18. 简历目标表达

最终希望可以写成类似：

```text
MiniMind AgentLab: From-Scratch Small LLM Training and Agentic RL
• Implemented a MiniMind-style 64M language model from scratch in PyTorch, including RoPE attention, RMSNorm, SwiGLU MLP, GQA, tied embeddings, pretraining, checkpointing, and generation.
• Built a MiniMind-compatible pretraining pipeline with JSONL data loading, bf16 training, gradient accumulation, cosine scheduling, checkpoint/resume, and reproducible experiment logging.
• Designed a verifier-based tool-use benchmark for calculator, date reasoning, unit conversion, and JSON extraction tasks, measuring tool format accuracy, tool selection accuracy, final-answer accuracy, and pass rate.
• Implemented GRPO-style agentic RL using verifier rewards and analyzed reward hacking behaviors such as fake tool observations, parser exploits, malformed JSON, and excessive tool calls.
• Explored SOC-inspired inference-time optimization via best-of-N sampling, temperature annealing, and verifier-guided reranking, with a pilot masked-diffusion LM experiment for AR-to-dLM adaptation.
```

---

## 19. 当前最优先完成的文件

优先级最高：

- [ ] `configs/minimind_64m.yaml`
- [ ] `configs/pretrain.yaml`
- [ ] `src/model/config.py`
- [ ] `src/model/norm.py`
- [ ] `src/model/rope.py`
- [ ] `src/model/mlp.py`
- [ ] `src/model/attention.py`
- [ ] `src/model/modeling_minimind.py`
- [ ] `src/data/tokenizer.py`
- [ ] `src/data/pretrain_dataset.py`
- [ ] `src/data/collator.py`
- [ ] `src/train/optim.py`
- [ ] `src/train/checkpoint.py`
- [ ] `src/train/logger.py`
- [ ] `scripts/train_pretrain.py`
- [ ] `scripts/generate.py`

---

## 20. 当前阶段的完成标准

当前阶段完成的标志：

- [ ] 64M 模型可以初始化
- [ ] 参数量统计正确
- [ ] 可以读取 tokenizer
- [ ] 可以读取 pretrain jsonl
- [ ] batch 检查正确
- [ ] 可以开始训练
- [ ] loss 正常记录
- [ ] checkpoint 正常保存
- [ ] resume 正常恢复
- [ ] generate 脚本正常输出
- [ ] README 能让别人复现当前阶段
