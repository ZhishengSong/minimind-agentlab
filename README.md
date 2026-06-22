# MiniMind AgentLab

MiniMind AgentLab is a from-scratch small-LLM training project inspired by the public MiniMind ecosystem. The current milestone is a reproducible MiniMind-style causal language model pretraining stack; the next research layer will extend it toward tool-use evaluation and verifier-based Agentic RL.

The goal is not to build the largest possible model. The goal is to build a compact, inspectable training system that can be explained end to end: model architecture, data pipeline, training loop, checkpoint/resume, generation, and later agentic post-training experiments.

## Highlights

- From-scratch PyTorch decoder-only causal LM
- RMSNorm, RoPE, SwiGLU MLP, grouped-query attention, pre-norm Transformer blocks
- 64M-style model config with about 63.06M parameters
- Streaming JSONL pretraining dataset and dynamic padding collator
- Tokenizer abstraction with MiniMind tokenizer support and local byte-tokenizer fallback
- AdamW training loop with warmup + cosine scheduler, gradient accumulation, clipping, and JSONL metrics
- Checkpoint/resume with model, optimizer, scheduler, scaler, configs, step, and random state
- Generation script with greedy, temperature, top-k, and top-p decoding
- Tiny CPU end-to-end validation for cheap local debugging
- Real MiniMind-compatible tokenizer integration hooks

## Project Status

Completed:

- Step 1: repository scaffold
- Step 2: configuration system
- Step 3: model architecture
- Step 4: tokenizer and data pipeline
- Step 5: pretraining trainer
- Step 6: checkpoint and resume
- Step 7: generation
- Step 8: tiny end-to-end validation
- Step 9: MiniMind-compatible tokenizer/data integration hooks

Current status and limitations:

- Local 10/100/500-step smoke runs and rented-server 5k/50k runs have completed.
- The 50k checkpoint reached fixed-slice validation loss `2.25228` and perplexity `9.509392`.
- Full 5k/10k/20k/50k checkpoints, logs, configs, tokenizer, reports, and samples were exported from the rented server and SHA256-verified locally on 2026-06-22.
- Large checkpoints and the real `pretrain_t2t_mini.jsonl` dataset are intentionally not committed to Git.
- The current non-GPU workstation is used for source and artifact management; checkpoint evaluation and post-training will run on a separate GPU machine.
- Phase 2 tool-use SFT and Agentic RL modules have not started yet.

## Architecture

Current 64M-style config:

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

Derived attention shape:

```text
head_dim = 512 / 8 = 64
num_key_value_groups = 8 / 2 = 4
```

Model flow:

```text
input_ids
  -> token embedding
  -> Transformer block x 12
  -> final RMSNorm
  -> lm_head
  -> logits
  -> shifted causal LM loss
```

## Repository Layout

```text
configs/       Model and training configs
data/          Local tokenizer assets and sample data
docs/          Step-by-step learning notes
scripts/       CLI entry points
src/           Project source code
checkpoints/   Local checkpoints, ignored by git
logs/          JSONL training metrics, ignored by git
outputs/       Generated samples, ignored by git
reports/       Future experiment reports
tests/         Future tests
```

## Setup

Use any Python environment with PyTorch and the packages in `requirements.txt`.
If you keep a dedicated Conda environment for this project, activate it first:

```bash
conda activate pytorch
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you need to call a specific Python executable directly, replace `python`
with that executable path in the commands below.

```bash
python -m pip install -r requirements.txt
```

## Quickstart

Run the full tiny local validation:

```bash
python scripts/run_tiny_e2e.py
```

If you want the script to use a specific interpreter for subprocesses:

```bash
python scripts/run_tiny_e2e.py --python path/to/python
```

This runs:

```text
config dry-run
model inspection
batch inspection
tiny training
resume
generation
```

The tiny path uses:

- `configs/minimind_tiny.yaml`
- `configs/pretrain_tiny.yaml`
- `data/samples/pretrain_tiny.jsonl`
- `ByteTokenizer`

It is a correctness test, not a quality training run.

## Useful Commands

Validate config loading:

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --dry-run
```

Inspect the model:

```bash
python scripts/inspect_model.py --config configs/minimind_64m.yaml --batch-size 1 --seq-len 16
```

Inspect one batch:

```bash
python scripts/inspect_batch.py --config configs/pretrain_tiny.yaml
```

Run tiny training:

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
```

Resume from checkpoint:

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_tiny/latest.pt
```

Generate from checkpoint:

```bash
python scripts/generate.py --checkpoint checkpoints/pretrain_tiny/latest.pt --prompt MiniMind --max-new-tokens 32
```

Inspect the MiniMind tokenizer:

```bash
python scripts/inspect_tokenizer.py --tokenizer-path data/minimind/tokenizer --model-config configs/minimind_64m.yaml --strict-vocab-match
```

Validate real pretraining assets after placing the dataset:

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

## Verified Local Results

Tiny model inspection:

```text
hidden_size/layers/heads: 128/2/4
kv_heads/head_dim: 2/32
logits shape: (1, 16, 258)
total params: 426880 (426.88K)
```

64M-style model inspection:

```text
hidden_size/layers/heads: 512/12/8
kv_heads/head_dim: 2/64
logits shape: (1, 16, 6400)
total params: 63058432 (63.06M)
```

Tiny training smoke run:

```text
loss moved from about 5.58 to 5.34 over a short CPU run
checkpoint saved to checkpoints/pretrain_tiny/latest.pt
resume continued from the saved step
generation loaded the checkpoint successfully
```

MiniMind tokenizer inspection:

```text
tokenizer_type: HFTokenizerWrapper
vocab_size: 6400
pad_token_id: 0
eos_token_id: 2
model_vocab_size: 6400
```

64M real-data smoke run:

```text
checkpoint: checkpoints/pretrain_minimind_64m_smoke10/latest.pt
loss moved from 8.8723 at step 1 to 8.5569 at step 10
resume continued to step 12 with loss 8.3415
peak memory observed: about 2.7GB
generation loaded the checkpoint successfully
```

500-step local GPU smoke run:

```text
checkpoint: checkpoints/pretrain_minimind_64m_smoke100/latest.pt
loss moved from 8.8723 at step 1 to 7.0437 at step 100
resume continued to step 500 with loss 5.2633
peak memory observed: about 3.5GB
generation loaded the checkpoint successfully
```

50k rented-server training and evaluation:

```text
checkpoints preserved at steps: 5k, 10k, 20k, 50k
training loss at step 5k: 2.924881
training loss at step 50k: 2.283401
fixed 1k-example validation loss: 2.252280
fixed 1k-example validation perplexity: 9.509392
generation is coherent at the paragraph level but remains repetitive
artifact archive SHA256: e1c453ffa5e95e7059c60aa53d9b6be8f8ce349caea2eaaeaa5c6e67608f9702
```

## Real MiniMind Assets

The official tokenizer files are expected at:

```text
data/minimind/tokenizer/
|-- tokenizer.json
`-- tokenizer_config.json
```

The real pretraining file should be placed at:

```text
data/minimind/pretrain_t2t_mini.jsonl
```

This file is intentionally not committed to git. The public MiniMind README lists it as about 1.2GB and provides download links through ModelScope and Hugging Face.

After placing the file, run:

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

Then the 64M smoke run entry point is:

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml
```

On GPU:

```bash
python scripts/train_pretrain.py --config configs/pretrain_minimind_local.yaml --override device=cuda
```

## Learning Notes

The `docs/` directory contains step-by-step explanations in English and Chinese:

```text
docs/step_2_config_system.md
docs/step_3_model_architecture.md
docs/step_4_data_pipeline.md
docs/step_5_pretraining_trainer.md
docs/step_6_checkpoint_resume.md
docs/step_7_generation.md
docs/step_8_tiny_e2e_validation.md
docs/step_9_minimind_assets.md
```

These notes are written as study material and interview preparation. They explain what each module does, why it exists, and how to describe it clearly.

## Relationship To MiniMind

This project is inspired by the public MiniMind project and uses MiniMind-compatible tokenizer/data conventions. It is not intended to be a fork with minor edits.

The implementation here rebuilds the training stack independently:

- model modules
- config system
- JSONL data pipeline
- optimizer and scheduler
- checkpoint/resume
- generation
- validation scripts

The intent is to learn and demonstrate the engineering internals behind small LLM training, while keeping compatibility with MiniMind-style resources.

## Roadmap

Near term:

- Run fixed-slice evaluation for the preserved 5k/10k/20k/50k checkpoints on the GPU machine
- Run a fixed 10-prompt generation suite and finalize the pretraining V0 report
- Select the 50k checkpoint as the candidate base for tool-use SFT
- Design tool-use special tokens, update the tokenizer, and resize model embeddings

Next research layer:

- WebGym-RL trajectory conversion and assistant-only SFT loss masking
- Tool-use SFT and pretrain-vs-SFT evaluation
- tool-use benchmark
- tool call parser
- rule-based verifier
- best-of-N verifier reranking
- GRPO-style verifier-based Agentic RL
- reward-hacking analysis

## Resume Positioning

Possible project description:

```text
MiniMind AgentLab: From-Scratch Small LLM Training with Verifier-Based Agentic RL
Implemented a MiniMind-style 63M causal LM in PyTorch with RoPE, GQA, SwiGLU,
RMSNorm, tied embeddings, streaming JSONL data loading, mixed-precision-ready
training, checkpoint/resume, generation, and reproducible tiny E2E validation.
```
