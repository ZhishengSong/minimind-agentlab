# MiniMind AgentLab TODO

Last updated: 2026-06-22

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
- [x] Completed rented-server training through step 50,000
- [x] Evaluated the 50k checkpoint on the fixed 1k-example validation slice
- [x] Recorded 50k validation loss `2.25228` and perplexity `9.509392`
- [x] Exported full 5k/10k/20k/50k checkpoints from the rented server
- [x] Exported server logs, configs, tokenizer, reports, samples, and environment metadata
- [x] Downloaded and SHA256-verified the artifact archive locally on 2026-06-22

Current training decision:

- Do not extend pretraining again before reviewing the preserved checkpoints.
- Keep the current workstation for source, documentation, and artifact backup; it has no GPU requirement.
- Use the separate GPU machine for checkpoint loading, fixed evaluation, generation, SFT, and RL.
- Finish Pretrain V0, then follow the Track TODO through special tokens and tool-use SFT before GRPO.
- Pretrain V0 is now closed; Version 1 tool-use tokenizer adaptation has produced an init checkpoint.

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
  - [x] Confirm rented server PyTorch/CUDA environment can run training
- [ ] Add reusable server setup notes
- [x] Test `git clone`
- [x] Test dependency install
- [x] Test asset placement
- [x] Validate tokenizer on server
- [x] Validate real dataset on server
- [x] Test 10-step run on server
- [x] Run 5k-step server pilot
- [x] Review server training throughput and stability
- [x] Continue training through step 50,000
- [x] Evaluate the 50k checkpoint on a fixed validation slice
- [x] Preserve 5k/10k/20k/50k checkpoints and supporting artifacts
- [x] Evaluate all preserved checkpoints on the same fixed validation slice
- [x] Run the fixed 10-prompt generation suite
- [x] Write final Pretrain V0 notes and select the SFT base checkpoint

Pretrain V0 closure outputs:

```text
docs/pretrain_v0_notes.md
reports/pretrain_v0_checkpoint_eval.json
reports/pretrain_v0_checkpoint_eval.md
reports/pretrain_v0_generation_suite.jsonl
reports/pretrain_v0_generation_suite.md
```

Current SFT base decision:

```text
minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt
```

Fixed-slice checkpoint comparison on 2026-06-23:

| Checkpoint | Step | Loss | Perplexity |
| --- | ---: | ---: | ---: |
| `pretrain_step_005000.pt` | 5,000 | 2.986131 | 19.808900 |
| `pretrain_step_010000.pt` | 10,000 | 2.703081 | 14.925641 |
| `pretrain_step_020000.pt` | 20,000 | 3.084734 | 21.861646 |
| `pretrain_step_050000.pt` | 50,000 | 2.251902 | 9.505802 |

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

Tool-use tokenizer adaptation and the first complete SFT milestone are finished. Epoch1/2/3 server training completed on 2026-06-24; Epoch3 is the selected checkpoint. RL training has not started.

### Step 14A: Tool-Use Tokenizer Init

- [x] Define tool-use special token list in `configs/tool_special_tokens.json`
- [x] Inspect existing tokenizer tokenization for tool boundary markers
- [x] Confirm `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, and `</think>` already exist as single tokens
- [x] Add `<answer>`, `</answer>`, `<observe>`, and `</observe>` as special tokens
- [x] Resize 50k checkpoint embeddings from vocab 6400 to 6404
- [x] Save tool-use tokenizer to `outputs/tooluse_init/tokenizer`
- [x] Save tool-use init checkpoint to `outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt`
- [x] Validate tokenizer/checkpoint loading and a forward pass with new token IDs
- [x] Validate `scripts/generate.py` can load the tool-use init checkpoint
- [x] Write `docs/step_10_tooluse_init.md`

### Step 14B: MiniMind Tool-Use SFT Data

- [x] Locate WebNav-RL SFT train/eval data
- [x] Implement `scripts/convert_webgym_to_minimind_sft.py`
- [x] Implement `scripts/inspect_minimind_sft.py`
- [x] Convert 800 train trajectories into 2530 next-action examples
- [x] Convert 200 eval trajectories into 637 next-action examples
- [x] Validate tool-call JSON for all assistant targets during conversion
- [x] Tokenize with `outputs/tooluse_init/tokenizer`
- [x] Confirm no converted examples exceed 2048 tokens
- [x] Implement `scripts/train_sft_minimind.py`
- [x] Add `configs/sft_minimind_webnav_smoke.yaml`
- [x] Run SFT dry-run and verify assistant-only supervised tokens
- [x] Run a 3-step SFT smoke on 8 examples
- [x] Save smoke checkpoint at `checkpoints/sft_minimind_webnav_smoke3/latest.pt`
- [x] Verify `scripts/generate.py` can load the SFT smoke checkpoint
- [x] Run configured 20-step SFT smoke on all converted train examples
- [x] Fix low-level tokenizer eos detection for `<|im_end|>`
- [x] Run 200-step local CUDA/bf16 SFT pass
- [x] Add `scripts/generate_from_sft_example.py`
- [x] Add `scripts/eval_minimind_sft_format.py`
- [x] Evaluate 200-step checkpoint on all 637 eval next-action examples
- [x] Confirm 200-step checkpoint reaches 100% wrapper/JSON/tool-name format accuracy
- [x] Confirm 200-step checkpoint reaches 68.6% exact target match on next-action eval
- [x] Write `docs/step_11_minimind_sft_data.md`
- [x] Implement `scripts/run_minimind_webnav_eval.py`
- [x] Run SFT-200step rollout eval20 in WebNav-RL
- [x] Run constrained tooluse-init rollout baseline
- [x] Confirm SFT-200step has 0 rollout format errors on eval20
- [x] Confirm SFT-200step has 100% submitted rate but 0% task success on eval20
- [x] Identify click/answer argument collapse as the main rollout failure mode
- [x] Run approximately one-epoch MiniMind SFT at 320 steps
- [x] Evaluate SFT-epoch1 on rollout eval20
- [x] Confirm SFT-epoch1 improves rollout success to 2/20 while keeping 0 format errors
- [x] Confirm argument collapse remains the main bottleneck after longer SFT
- [x] Write `docs/step_12_minimind_rollout_eval.md`
- [x] Prepare server artifact manifest and SFT sweep runner
- [x] Run independent Epoch1/2/3 SFT experiments on the GPU server
- [x] Evaluate Epoch1 on the same 100-example next-action slice: `72%` exact match
- [x] Evaluate Epoch2 on the same 100-example next-action slice: `96%` exact match
- [x] Evaluate Epoch3 on the same 100-example next-action slice: `100%` exact match
- [x] Evaluate Epoch3 on the full 637-example next-action eval set
- [x] Record Epoch3 full-eval wrapper/JSON/tool-name accuracy: `100%`
- [x] Record Epoch3 full-eval argument/target exact match: `628/637 = 98.59%`
- [x] Select `checkpoints/sft_minimind_webnav_epoch3/latest.pt` as the final SFT checkpoint
- [x] Save Epoch3 checkpoint SHA256: `77f29f3f5fd812e2fa05ba3afb6af85b0d319dad2de6e7fbff52e72d35e87ce6`
- [x] Download and SHA256-verify Epoch3 checkpoint locally at `checkpoints/sft_minimind_webnav_epoch3_server/latest.pt`
- [x] Run Epoch3 closed-loop WebNav-RL eval20: `20/20 = 100%`
- [x] Run Epoch3 closed-loop WebNav-RL eval200: `191/200 = 95.5%`
- [x] Verify eval200 has 100% submission, 0 invalid tool calls, and 0 format errors
- [x] Analyze all 9 rollout failures in `reports/minimind_sft_epoch3_rollout_failure_analysis.md`
- [x] Add optional WebNav page metadata support to the MiniMind rollout adapter
- [x] Audit all 1,868 V2 eval prompts against the 2,048-token context window
- [x] Add message-aware context compaction for 831 over-length V2 decision points
- [x] Run Epoch3 zero-shot rollout on the current 500-task WebNav V2 snapshot: `0/500`
- [x] Confirm protocol transfer but zero valid V2 start-page actions: `open_page` tool `500/500`, valid V2 page ID `0/500`
- [x] Analyze V2 results separately from the frozen V1 benchmark
- [ ] Run an oracle-first-open V2 ablation before any V2 training

### SFT Milestone Evaluation Priority

Required for closing the current SFT milestone:

- [x] Run a held-out full next-action evaluation on the selected checkpoint
- [x] Verify tool-call wrapper, JSON, tool-name, arguments, and exact-target metrics
- [x] Download the selected checkpoint and verify SHA256 locally
- [x] Load the downloaded Epoch3 checkpoint locally and run one forward/generation smoke test
- [x] Write the final SFT V1 report with training configuration, learning curve, eval table, failures, and selected checkpoint

Strongly recommended before claiming that SFT preserves base-model quality:

- [x] Evaluate tool-use init on the fixed 1k pretrain validation slice: loss `2.252283`, perplexity `9.509420`
- [x] Evaluate SFT Epoch3 on the same fixed 1k pretrain validation slice: loss `2.569051`, perplexity `13.053436`
- [x] Report the pretrain-loss delta: `+0.316768` loss and `+3.544016` perplexity; measurable specialization tradeoff, not model collapse

Optional comparisons, not blockers for the current SFT milestone:

- [ ] Run Epoch2 full 637-example evaluation for a complete learning-curve table
- [x] Run additional rollout/environment evaluation in the separate WebNav-RL project
- [ ] Add best-of-N/verifier reranking experiments
- [ ] Start GRPO-style Agentic RL experiments

### Step 14: Tool-Use Environment

- [ ] Define tool-call text format
- [ ] Implement calculator tool
- [ ] Implement date-diff tool
- [ ] Implement unit-converter tool
- [ ] Implement JSON extractor tool
- [ ] Implement tool execution environment
- [ ] Define trajectory format

### Step 15: Tool Call Parser And Verifier

- [x] Implement offline MiniMind SFT tool-call format parser/evaluator
- [x] Implement rollout-level tool-call parser integration through WebNav-RL adapter
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

- [x] Build tool-use SFT dataset
- [x] Add next-action SFT dataset loader
- [x] Add assistant-only loss masking
- [x] Add `scripts/train_sft_minimind.py`
- [x] Train and evaluate the selected Epoch3 SFT checkpoint
- [x] Compare base/tool-use-init vs SFT on the fixed pretrain validation slice

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

Current tool-use SFT bullet:

```text
Adapted the from-scratch 63M LM for structured tool use and reached 98.59%
held-out next-action exact match plus 95.5% success on a fixed 200-task
closed-loop WebNav benchmark, with zero invalid or malformed tool calls.
```
