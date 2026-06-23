# Pretrain V0 Notes

Last updated: 2026-06-23

## Goal

Pretrain V0 freezes the first usable MiniMind AgentLab foundation checkpoint. The goal is not to claim assistant-quality language ability. The goal is to preserve a reproducible from-scratch 63M causal LM checkpoint that can be loaded, evaluated, sampled, and used as the base for tool-use SFT.

## Current State

- Model: MiniMind-style decoder-only causal LM, about 63.06M parameters.
- Architecture: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, pre-norm Transformer blocks, tied embeddings.
- Tokenizer: MiniMind-compatible tokenizer with `vocab_size=6400`.
- Data: MiniMind `pretrain_t2t_mini.jsonl`.
- Preserved checkpoints: 5k, 10k, 20k, 50k optimizer steps.
- Candidate SFT base: `minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt`.

The 50k checkpoint has already been evaluated on a fixed 1k-example validation slice:

```text
validation_loss: 2.252280
perplexity: 9.509392
predicted_tokens: 212462
dtype: bf16
device: cuda
```

On 2026-06-23, all preserved checkpoints were re-evaluated locally on the same fixed 1k-example slice using the RTX 5070 Ti Laptop GPU:

| Checkpoint | Step | Loss | Perplexity |
| --- | ---: | ---: | ---: |
| `pretrain_step_005000.pt` | 5,000 | 2.986131 | 19.808900 |
| `pretrain_step_010000.pt` | 10,000 | 2.703081 | 14.925641 |
| `pretrain_step_020000.pt` | 20,000 | 3.084734 | 21.861646 |
| `pretrain_step_050000.pt` | 50,000 | 2.251902 | 9.505802 |

The 50k checkpoint remains the best preserved checkpoint by fixed-slice loss. The 20k result is worse than the 5k and 10k checkpoints on this slice, so it should be treated as a checkpoint-specific anomaly to inspect against the training log rather than as the selected base.

This local comparison used the same `TokenizersWrapper` path for all four checkpoints. The earlier server-side 50k report used `HFTokenizerWrapper`, so its token count and loss differ slightly; the local comparison is the preferred table for choosing among preserved checkpoints.

## Artifact Inventory

The preserved archive is local and SHA256-verified:

```text
archive: minimind-50k-artifacts.tar
sha256: e1c453ffa5e95e7059c60aa53d9b6be8f8ce349caea2eaaeaa5c6e67608f9702
```

Important paths:

```text
minimind-50k-artifacts/checkpoints/pretrain_step_005000.pt
minimind-50k-artifacts/checkpoints/pretrain_step_010000.pt
minimind-50k-artifacts/checkpoints/pretrain_step_020000.pt
minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt
minimind-50k-artifacts/logs/
minimind-50k-artifacts/reports/
minimind-50k-artifacts/samples/
minimind-50k-artifacts/tokenizer/
```

## Completed V0 Closure Work

All preserved checkpoints can be evaluated on the same fixed validation slice:

```bash
python scripts/eval_checkpoint_suite.py \
  --device cuda \
  --dtype bf16 \
  --batch-size 4 \
  --num-examples 1000 \
  --save-subset data/samples/pretrain_val_1k.jsonl
```

Expected outputs:

```text
reports/pretrain_v0_checkpoint_eval.json
reports/pretrain_v0_checkpoint_eval.md
```

The fixed 10-prompt generation suite can be run with:

```bash
python scripts/generate_prompt_suite.py \
  --checkpoint minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt \
  --device cuda \
  --max-new-tokens 128 \
  --temperature 0.0
```

Expected outputs:

```text
reports/pretrain_v0_generation_suite.jsonl
reports/pretrain_v0_generation_suite.md
```

Observed generation quality:

- The model can produce coherent Chinese paragraphs and basic answer formats.
- Repetition remains obvious, especially with greedy decoding.
- Simple factual and conceptual answers are plausible but unstable.
- Arithmetic and translation are unreliable.
- The generation suite supports the decision to use this as a tool-use SFT base, not as a finished assistant.

## Known Limitations

- The current workstation can run checkpoint evaluation and generation on CUDA, but longer SFT/RL runs should still use a stable GPU machine.
- The 50k checkpoint is coherent enough for inspection, but it remains repetitive and factually unreliable.
- This checkpoint should be treated as a foundation for a boundary experiment, not as a finished assistant.

## Decision

Use the 50k checkpoint as the base for Version 1 tokenizer special-token adaptation and Version 2 tool-use SFT.

## Next Phase

Version 1 starts the tool-use adaptation:

```text
1. Define tool-use special tokens.
2. Update tokenizer.
3. Resize model embeddings.
4. Save a tool-use init checkpoint.
5. Convert WebGym-RL expert trajectories into MiniMind SFT format.
```
