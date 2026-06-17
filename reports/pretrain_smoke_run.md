# 64M Real-Data Smoke Run

Date: 2026-06-17

## Goal

Validate that the 63M MiniMind-style pretraining stack works with the real MiniMind tokenizer and `pretrain_t2t_mini.jsonl` dataset before running longer experiments.

## Hardware And Environment

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- GPU memory: about 12GB
- PyTorch: 2.11.0+cu128
- Python: `D:\Program\Anaconda\envs\research\python.exe`
- Temporary project-local dependencies: `.local_deps`

## Assets

- Tokenizer: `data/minimind/tokenizer`
- Dataset: `data/minimind/pretrain_t2t_mini.jsonl`
- Dataset size: 1,241,043,656 bytes
- Model config: `configs/minimind_64m.yaml`
- Training config base: `configs/pretrain_minimind_local.yaml`

Validation command:

```bash
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```

Validation result:

```text
tokenizer_vocab_size: 6400
model_vocab_size: 6400
input_ids shape: (4, 449)
attention lengths: [353, 135, 230, 449]
ignored labels: 629
```

## Smoke Training

Command:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_minimind_local.yaml \
  --override run_name=pretrain_minimind_64m_smoke10 \
  --override output_dir=checkpoints/pretrain_minimind_64m_smoke10 \
  --override device=cuda \
  --override max_steps=10 \
  --override save_interval=5 \
  --override log_interval=1 \
  --override num_workers=0
```

Summary:

| Step | Loss | LR | Tokens/sec | Grad norm | Memory MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.8723 | 6.00e-06 | 10199.1 | 2.695 | 1570.19 |
| 5 | 8.8223 | 3.00e-05 | 19178.0 | 2.699 | 2429.15 |
| 10 | 8.5569 | 6.00e-05 | 21578.8 | 2.697 | 2498.26 |

Checkpoint result:

```text
checkpoints/pretrain_minimind_64m_smoke10/pretrain_step_000005.pt
checkpoints/pretrain_minimind_64m_smoke10/pretrain_step_000010.pt
checkpoints/pretrain_minimind_64m_smoke10/latest.pt
```

## Resume Check

Command:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_minimind_local.yaml \
  --resume checkpoints/pretrain_minimind_64m_smoke10/latest.pt \
  --override run_name=pretrain_minimind_64m_smoke10_resume \
  --override output_dir=checkpoints/pretrain_minimind_64m_smoke10 \
  --override device=cuda \
  --override max_steps=12 \
  --override save_interval=1 \
  --override log_interval=1 \
  --override num_workers=0
```

Resume result:

| Step | Loss | LR | Tokens/sec | Grad norm | Memory MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 8.3912 | 6.60e-05 | 9779.3 | 1.939 | 2306.32 |
| 12 | 8.3415 | 7.20e-05 | 4550.5 | 1.772 | 2677.43 |

## Generation Check

Command:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/pretrain_minimind_64m_smoke10/latest.pt \
  --prompt MiniMind \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --output outputs/pretrain_minimind_64m_smoke10_sample.txt
```

Output:

```text
MiniMind，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
```

This output is expected to be low quality after only 12 optimizer steps. The purpose of this check is checkpoint loading and inference compatibility, not language quality.

## Fixes From This Run

- Fixed training log loss averaging during gradient accumulation.
- Fixed checkpoint resume when loading CPU RNG state with `map_location=cuda`.
- Added missing runtime dependencies to `requirements.txt`: `typing_extensions`, `sympy`.

## Next Steps

- Review whether the local GPU is acceptable for longer training, given the slowdown observed after step 300.
- Decide whether the next run should be local or on a server.
- Track thermals, sustained tokens/sec, and generation samples over longer runs.

## 100-Step And 500-Step Follow-Up

The 100-step run completed successfully on the local GPU.

Command:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_minimind_local.yaml \
  --override run_name=pretrain_minimind_64m_smoke100 \
  --override output_dir=checkpoints/pretrain_minimind_64m_smoke100 \
  --override device=cuda \
  --override max_steps=100 \
  --override save_interval=50 \
  --override log_interval=10 \
  --override num_workers=0
```

100-step summary:

| Step | Loss | LR | Tokens/sec | Grad norm | Memory MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.8723 | 6.00e-06 | 9306.4 | 2.695 | 1570.19 |
| 50 | 7.2738 | 3.00e-04 | 19652.1 | 0.535 | 3128.47 |
| 100 | 7.0437 | 3.03e-05 | 17214.7 | 0.833 | 3128.47 |

The run was then resumed to 500 steps.

Command:

```bash
python scripts/train_pretrain.py \
  --config configs/pretrain_minimind_local.yaml \
  --resume checkpoints/pretrain_minimind_64m_smoke100/latest.pt \
  --override run_name=pretrain_minimind_64m_smoke500 \
  --override output_dir=checkpoints/pretrain_minimind_64m_smoke100 \
  --override device=cuda \
  --override max_steps=500 \
  --override save_interval=100 \
  --override log_interval=25 \
  --override num_workers=0
```

500-step summary:

| Step | Loss | LR | Tokens/sec | Grad norm | Memory MB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 | 6.9286 | 2.82e-04 | 17572.3 | 0.749 | 2964.23 |
| 200 | 6.2580 | 2.33e-04 | 17122.4 | 1.375 | 3369.63 |
| 300 | 5.7026 | 1.42e-04 | 12221.9 | 1.247 | 3369.63 |
| 400 | 5.3879 | 6.22e-05 | 2477.0 | 1.243 | 3369.63 |
| 500 | 5.2633 | 3.00e-05 | 6721.5 | 1.368 | 3455.72 |

Checkpoint result:

```text
checkpoints/pretrain_minimind_64m_smoke100/pretrain_step_000100.pt
checkpoints/pretrain_minimind_64m_smoke100/pretrain_step_000200.pt
checkpoints/pretrain_minimind_64m_smoke100/pretrain_step_000300.pt
checkpoints/pretrain_minimind_64m_smoke100/pretrain_step_000400.pt
checkpoints/pretrain_minimind_64m_smoke100/pretrain_step_000500.pt
checkpoints/pretrain_minimind_64m_smoke100/latest.pt
```

Generation after 500 steps:

```text
MiniMind
-
-
-
-
...
```

The 500-step generation is still low quality, but the loss trend, checkpointing, resume path, and inference loading are all functional.
