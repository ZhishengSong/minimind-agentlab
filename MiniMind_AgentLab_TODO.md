# MiniMind AgentLab TODO

Last updated: 2026-06-17

## Project Goal

Build a from-scratch MiniMind-style small LLM training stack, then extend it toward tool-use evaluation and verifier-based Agentic RL.

Current positioning:

```text
MiniMind AgentLab: From-Scratch Small LLM Training with Verifier-Based Agentic RL
```

Important distinction:

- Phase 1 is the base LLM training stack.
- Phase 2 will be the Agentic RL / verifier-based tool-use research layer.

## External References And Download Links

Official MiniMind repository:

- GitHub: https://github.com/jingyaogong/minimind

MiniMind dataset download pages:

- ModelScope dataset: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
- Hugging Face dataset: https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main

Files needed for our next real-data step:

- `pretrain_t2t_mini.jsonl` around 1.2GB, required for 64M pretraining smoke run
- Optional later: `sft_t2t_mini.jsonl`
- Optional later Agentic RL data: `agent_rl.jsonl`, `agent_rl_math.jsonl`, `rlaif.jsonl`

Local expected layout:

```text
data/minimind/
|-- tokenizer/
|   |-- tokenizer.json
|   `-- tokenizer_config.json
`-- pretrain_t2t_mini.jsonl
```

Current asset status:

- [x] Official MiniMind tokenizer copied from a local MiniMind checkout
- [x] Tokenizer validated against `configs/minimind_64m.yaml`
- [x] Download `pretrain_t2t_mini.jsonl`
- [x] Place it at `data/minimind/pretrain_t2t_mini.jsonl`
- [x] Validate real dataset with `scripts/validate_pretrain_assets.py`

## Phase 1: Base LLM Training Stack

### Step 1: Repository And Engineering Structure

- [x] Initialize local Git repository
- [x] Connect repository to GitHub
- [x] Create standard project layout
- [x] Add `.gitignore`
- [x] Add `requirements.txt`
- [x] Add clean GitHub README
- [x] Add `configs/`
- [x] Add `scripts/`
- [x] Add `src/`
- [x] Add `docs/`
- [x] Add `data/`, `checkpoints/`, `logs/`, `outputs/`

### Step 2: Configuration System

- [x] Add `configs/minimind_64m.yaml`
- [x] Add `configs/minimind_tiny.yaml`
- [x] Add `configs/pretrain.yaml`
- [x] Add `configs/pretrain_tiny.yaml`
- [x] Add `configs/pretrain_minimind_local.yaml`
- [x] Implement `MiniMindConfig`
- [x] Implement `PretrainConfig`
- [x] Implement YAML loading and saving helpers
- [x] Support command-line overrides
- [x] Support dry-run config validation
- [x] Write `docs/step_2_config_system.md`

### Step 3: Model Architecture

- [x] Implement `RMSNorm`
- [x] Implement RoPE cache and `apply_rotary_pos_emb`
- [x] Implement SwiGLU MLP
- [x] Implement grouped-query causal self-attention
- [x] Implement pre-norm Transformer block
- [x] Implement `MiniMindForCausalLM`
- [x] Support tied word embeddings
- [x] Support `input_ids`
- [x] Support `attention_mask`
- [x] Support `position_ids`
- [x] Support `labels`
- [x] Support `ignore_index=-100`
- [x] Implement parameter counting
- [x] Add `scripts/inspect_model.py`
- [x] Validate 64M-style model forward/backward
- [x] Confirm parameter count around 63.06M
- [x] Write `docs/step_3_model_architecture.md`

### Step 4: Tokenizer And Data Pipeline

- [x] Implement tokenizer abstraction
- [x] Support Hugging Face tokenizer directories
- [x] Support `tokenizer.json`
- [x] Add `ByteTokenizer` fallback for local smoke tests
- [x] Implement streaming JSONL `PretrainDataset`
- [x] Implement dynamic padding collator
- [x] Generate `input_ids`
- [x] Generate `attention_mask`
- [x] Generate `labels`
- [x] Set padding labels to `-100`
- [x] Add tiny sample JSONL
- [x] Add `scripts/inspect_batch.py`
- [x] Validate tiny batch inspection
- [x] Write `docs/step_4_data_pipeline.md`

### Step 5: Pretraining Trainer

- [x] Implement AdamW optimizer
- [x] Split weight decay and no-decay parameter groups
- [x] Implement warmup + cosine scheduler
- [x] Implement gradient accumulation
- [x] Implement gradient clipping
- [x] Support fp32 / bf16 / fp16 mode selection
- [x] Support CPU / CUDA device selection
- [x] Add non-finite loss checks
- [x] Implement JSONL metric logging
- [x] Log loss, lr, tokens/sec, grad norm, memory
- [x] Run local tiny CPU training smoke test
- [x] Write `docs/step_5_pretraining_trainer.md`

### Step 6: Checkpoint And Resume

- [x] Implement checkpoint save
- [x] Save model state dict
- [x] Save optimizer state dict
- [x] Save scheduler state dict
- [x] Save scaler state when fp16 is enabled
- [x] Save global step
- [x] Save train config
- [x] Save model config
- [x] Save random states
- [x] Save `latest.pt`
- [x] Save step checkpoints
- [x] Implement `--resume`
- [x] Validate tiny resume from step 3 to step 5
- [x] Write `docs/step_6_checkpoint_resume.md`

### Step 7: Generation

- [x] Implement checkpoint loading for inference
- [x] Rebuild model from checkpoint config
- [x] Load model weights
- [x] Load tokenizer
- [x] Encode prompt
- [x] Implement greedy decoding
- [x] Implement temperature sampling
- [x] Implement top-k
- [x] Implement top-p
- [x] Save generated output to `outputs/`
- [x] Validate tiny checkpoint generation
- [x] Write `docs/step_7_generation.md`

### Step 8: Tiny End-to-End Validation

- [x] Add `scripts/run_tiny_e2e.py`
- [x] Validate config dry-run
- [x] Validate model inspection
- [x] Validate batch inspection
- [x] Validate tiny training
- [x] Validate checkpoint save
- [x] Validate resume
- [x] Validate generation
- [x] Write `docs/step_8_tiny_e2e_validation.md`

### Step 9: Real MiniMind-Compatible Assets

- [x] Clone official MiniMind repository separately
- [x] Copy official tokenizer into `data/minimind/tokenizer`
- [x] Install tokenizer dependencies in a local Python/Conda environment
- [x] Add `scripts/inspect_tokenizer.py`
- [x] Validate tokenizer vocab size is 6400
- [x] Validate tokenizer encode/decode
- [x] Add `scripts/validate_pretrain_assets.py`
- [x] Add real-data config `configs/pretrain_minimind_local.yaml`
- [x] Add `data/minimind/README.md`
- [x] Write `docs/step_9_minimind_assets.md`
- [x] Download `pretrain_t2t_mini.jsonl`
- [x] Validate real JSONL assets

## Immediate Next Steps

Current status after local validation:

- [x] Real MiniMind tokenizer is available locally and matches `vocab_size=6400`
- [x] Real `pretrain_t2t_mini.jsonl` is available locally at `data/minimind/pretrain_t2t_mini.jsonl`
- [x] Real-data asset validation passed
- [x] Local RTX 5070 Ti Laptop GPU 10-step smoke run passed
- [x] Local 100-step smoke run passed
- [x] Local 500-step smoke run passed
- [x] Checkpoint save, resume, and generation all passed on real-data checkpoints
- [x] `reports/pretrain_smoke_run.md` records the real-data smoke results
- [x] Added fixed-slice validation loss script: `scripts/eval_pretrain_loss.py`

Current training decision:

- Local machine is good for validation and short smoke runs.
- Longer runs should preferably move to a rented server if price is reasonable.
- Candidate server target: 1x RTX 5090 32GB, 80GB+ disk preferred, recent PyTorch/CUDA image required.

### Step 10: Real Data Validation

Before any 64M training run:

- [x] Download `pretrain_t2t_mini.jsonl` from ModelScope or Hugging Face
- [x] Put it at `data/minimind/pretrain_t2t_mini.jsonl`
- [x] Run tokenizer check:

```bash
python scripts/inspect_tokenizer.py --tokenizer-path data/minimind/tokenizer --model-config configs/minimind_64m.yaml --strict-vocab-match
```

- [x] Run data validation:

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

- [x] Confirm batch shape
- [x] Confirm decoded sample looks reasonable
- [x] Confirm tokenizer vocab size equals model vocab size
- [x] Confirm labels use `-100` for padding

### Step 11: 64M Smoke Run

This is not final training. This is a pilot run to catch real-data and real-model issues.

- [x] Run 10-step 64M smoke test

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml --override max_steps=10 --override save_interval=5
```

- [x] Confirm loss is finite
- [x] Confirm checkpoint is saved
- [x] Resume from checkpoint
- [x] Generate from checkpoint
- [x] Run 100-step smoke test
- [x] Run 500-step smoke test if speed is acceptable
- [x] Record loss, tokens/sec, memory, checkpoint path, generated sample

### Step 12: Experiment Reporting

- [x] Add `reports/pretrain_smoke_run.md`
- [x] Add training command
- [x] Add hardware information
- [x] Add config summary
- [x] Add loss table
- [x] Add tokens/sec
- [x] Add memory usage
- [x] Add checkpoint/resume result
- [x] Add generation samples
- [x] Add known limitations

### Step 13: Server Training Preparation

Do this only after local/short smoke runs pass.

- [x] Decide GPU target
  - [x] Local GPU is available and validated for short runs
  - [x] Prefer short rental for server pilot
  - [x] Candidate server GPU: RTX 5090 32GB
  - [ ] Confirm rented server PyTorch can see RTX 5090
- [ ] Add server setup notes
- [ ] Test `git clone`
- [ ] Test dependency install
- [ ] Test asset placement
- [ ] Validate tokenizer on server
- [ ] Validate real dataset on server
- [ ] Test 10-step run on server
- [ ] Run 5k-step server pilot
- [ ] Review server tokens/sec, memory, and cost
- [ ] Decide whether to continue to 20k+ steps
- [ ] Evaluate 5k and 50k checkpoints on a fixed validation slice

Recommended server spec:

```text
GPU: 1x RTX 5090 32GB
CPU: 16+ cores
RAM: 64GB+
System disk: 30GB is acceptable
Data disk: 80GB+ preferred, 100GB safer
Image: PyTorch 2.7+ / CUDA 12.8+ or platform image explicitly supporting RTX 5090
```

First server checks:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Server pilot command:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_minimind_local.yaml \
  --override run_name=pretrain_minimind_64m_server5k \
  --override output_dir=checkpoints/pretrain_minimind_64m_server5k \
  --override device=cuda \
  --override max_steps=5000 \
  --override save_interval=500 \
  --override log_interval=50 \
  --override num_workers=2
```

Server pilot success criteria:

- [ ] Loss remains finite
- [ ] Loss continues below the local 500-step result of about `5.26`
- [ ] Checkpoints save correctly
- [ ] Resume works from latest checkpoint
- [ ] Generation script loads server checkpoint
- [ ] Sustained tokens/sec is stable enough to justify server cost
- [ ] Disk usage remains safe with planned checkpoint frequency
- [ ] Fixed validation loss confirms progress despite noisy streaming training loss

Fixed validation command:

```bash
python scripts/eval_pretrain_loss.py \
  --checkpoint checkpoints/pretrain_minimind_64m_server50k/latest.pt \
  --config configs/pretrain_minimind_local.yaml \
  --device cuda \
  --num-examples 1000 \
  --batch-size 4 \
  --max-seq-len 2048 \
  --save-subset data/samples/pretrain_val_1k.jsonl \
  --output reports/eval_server50k_val_1k.json
```

## Phase 2: Agentic RL Research Layer

Not started yet. This is the post-foundation research phase.

### Step 14: Tool-Use Environment

- [ ] Define tool-call text format
- [ ] Implement calculator tool
- [ ] Implement date-diff tool
- [ ] Implement unit-converter tool
- [ ] Implement JSON extractor tool
- [ ] Implement tool execution environment
- [ ] Define trajectory format

### Step 15: Tool Call Parser And Verifier

- [ ] Implement tool-call parser
- [ ] Implement final-answer parser
- [ ] Implement format verifier
- [ ] Implement tool selection verifier
- [ ] Implement argument correctness verifier
- [ ] Implement final answer correctness verifier
- [ ] Implement efficiency penalty

### Step 16: Tool-Use Benchmark

- [ ] Generate synthetic tool-use prompts
- [ ] Add calculator tasks
- [ ] Add date reasoning tasks
- [ ] Add unit conversion tasks
- [ ] Add JSON extraction tasks
- [ ] Add benchmark metrics
  - [ ] format accuracy
  - [ ] tool selection accuracy
  - [ ] argument correctness
  - [ ] final answer accuracy
  - [ ] pass rate

### Step 17: SFT Warmup For Tool Use

- [ ] Build tiny tool-use SFT dataset
- [ ] Add SFT dataset loader
- [ ] Add assistant-only loss masking
- [ ] Add `scripts/train_sft.py`
- [ ] Compare pretrain-only vs SFT outputs

### Step 18: Inference-Time Optimization

- [ ] Implement best-of-N sampling
- [ ] Implement verifier-guided reranking
- [ ] Compare greedy vs top-p vs best-of-N
- [ ] Track pass rate and latency

### Step 19: GRPO-Style Agentic RL

- [ ] Sample multiple trajectories per prompt
- [ ] Score each trajectory with verifier
- [ ] Compute group-relative advantage
- [ ] Add KL penalty against reference model
- [ ] Implement policy update
- [ ] Log reward, pass rate, KL, entropy, policy loss

### Step 20: Reward Hacking Analysis

- [ ] Detect fake tool observations
- [ ] Detect malformed JSON
- [ ] Detect parser exploits
- [ ] Detect repeated tool calls
- [ ] Detect hallucinated tool results
- [ ] Build failure taxonomy
- [ ] Write `reports/reward_hacking.md`

## Final Resume Targets

Base training stack bullet:

```text
Implemented a MiniMind-style 63M causal LM from scratch in PyTorch with RoPE,
GQA, SwiGLU, RMSNorm, tied embeddings, streaming JSONL data loading,
AdamW pretraining, checkpoint/resume, generation, and reproducible tiny E2E validation.
```

Future Agentic RL bullet:

```text
Built a verifier-based tool-use evaluation and Agentic RL pipeline with
automatic trajectory parsing, rule-based rewards, best-of-N reranking,
and GRPO-style policy optimization for small language models.
```
