# Server Experiment Plan

Last updated: 2026-06-23

## Goal

Move MiniMind tool-use experiments from local smoke tests to server-scale sweeps. The local pipeline is already validated:

```text
pretrain -> tool-use tokenizer/init checkpoint -> SFT data -> SFT training -> next-action eval -> WebNav rollout eval
```

The server phase should answer:

```text
Does longer SFT improve rollout success, or does the model remain bottlenecked by argument selection?
```

## Recommended Server

```text
GPU: RTX 4090 / RTX 5090 / A5000 / A6000
VRAM: 24GB+
RAM: 64GB+
Disk: 80GB+ free
CUDA/PyTorch: any working CUDA PyTorch env is fine for this 63M model
```

## Required Inputs

This repo needs these generated/local assets available on the server:

```text
outputs/tooluse_init/tokenizer/
outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt
outputs/minimind_sft/sft_train_next_action.jsonl
outputs/minimind_sft/sft_eval_next_action.jsonl
configs/sft_minimind_webnav_smoke.yaml
```

The WebNav-RL repo must also be present:

```text
D:/job/Program/WebNav-RL
```

On Linux, use any path and pass it with `--webnav-root` and `--tasks`.

Required WebNav-RL file:

```text
tasks/eval_tasks.jsonl
```

## Environment Check

From the MiniMind repo root:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/train_sft_minimind.py --config configs/sft_minimind_webnav_smoke.yaml --dry-run
python scripts/run_server_minimind_sft_sweep.py --dry-run --epochs 1 --eval-limits 20
```

If the server uses local project dependencies like this workstation, set:

```bash
export PYTHONPATH="$PWD/.local_deps"
```

On PowerShell:

```powershell
$env:PYTHONPATH="$PWD\.local_deps"
```

## Main Sweep

Run independent 1/2/3 epoch SFT experiments from the tool-use init checkpoint, then evaluate each checkpoint on eval20/eval50/eval200:

```bash
python scripts/run_server_minimind_sft_sweep.py \
  --epochs 1,2,3 \
  --steps-per-epoch 320 \
  --eval-limits 20,50,200 \
  --device cuda \
  --webnav-root /path/to/WebNav-RL \
  --tasks /path/to/WebNav-RL/tasks/eval_tasks.jsonl
```

On this Windows workstation layout:

```powershell
D:\Program\Anaconda\envs\research\python.exe scripts\run_server_minimind_sft_sweep.py `
  --epochs 1,2,3 `
  --steps-per-epoch 320 `
  --eval-limits 20,50,200 `
  --device cuda `
  --webnav-root D:/job/Program/WebNav-RL `
  --tasks D:/job/Program/WebNav-RL/tasks/eval_tasks.jsonl
```

Outputs:

```text
checkpoints/sft_minimind_webnav_epoch1/
checkpoints/sft_minimind_webnav_epoch2/
checkpoints/sft_minimind_webnav_epoch3/
outputs/server_sweep/
reports/server_sweep/
reports/server_sweep/summary.json
```

## Baselines

Already observed locally:

```text
tooluse-init constrained baseline:
  tasks: 3
  success_rate: 0.0%
  submitted_rate: 0.0%
  format_errors: 9

SFT-200step eval20:
  success_rate: 0.0%
  submitted_rate: 100.0%
  format_errors: 0

SFT-epoch1 eval20:
  success_rate: 10.0%
  submitted_rate: 100.0%
  format_errors: 0
```

The server sweep should produce stronger eval50/eval200 numbers.

## Summary Command

If reports are produced separately, summarize them with:

```bash
python scripts/summarize_rollout_reports.py \
  --reports "reports/server_sweep/*.json" \
  --output-json reports/server_sweep/summary.json \
  --output-md reports/server_sweep/summary.md
```

## Decision Rule

Use rollout success, not training loss, to decide next steps.

Continue SFT/data balancing if:

```text
eval200 success increases materially from epoch1 -> epoch2 -> epoch3
argument collapse weakens
```

Stop longer plain SFT and switch to reranking/RL if:

```text
format remains stable
submitted rate remains high
success stays low
click/answer arguments collapse to frequent items
```

## Likely Next Experiment

If SFT still collapses, run best-of-N verifier reranking:

```text
sample N valid tool calls
execute or statically score candidates
choose candidate with best verifier score
compare greedy vs best-of-N on eval50/eval200
```

This directly targets the current bottleneck: observation-grounded argument selection.
