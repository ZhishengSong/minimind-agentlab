# Step 12: MiniMind WebNav Rollout Eval

Last updated: 2026-06-23

## Goal

Move beyond teacher-forced next-action evaluation and run MiniMind checkpoints inside the WebNav-RL environment. This measures whether the model can produce parseable tool calls, execute actions, observe page text, and submit answers in a closed loop.

## Script

Rollout adapter:

```text
scripts/run_minimind_webnav_eval.py
```

The script loads a MiniMind checkpoint, renders WebNav-RL `messages` with the same SFT template, generates one assistant tool call, strips `<|im_end|>`, and passes the result to WebNav-RL's `run_model_task`.

Default external inputs:

```text
webnav_root: D:/job/Program/WebNav-RL
tasks: D:/job/Program/WebNav-RL/tasks/eval_tasks.jsonl
tokenizer: outputs/tooluse_init/tokenizer
```

## SFT-200Step Eval20

Command:

```bash
python scripts/run_minimind_webnav_eval.py \
  --checkpoint checkpoints/sft_minimind_webnav_200step/latest.pt \
  --limit 20 \
  --device cuda \
  --output outputs/minimind_rollout/sft_200step_eval20_trajectories.jsonl \
  --report reports/minimind_sft_200step_rollout_eval20.json
```

Result:

```text
tasks: 20
success_rate: 0.0%
submitted_rate: 100.0%
invalid_tool_calls: 0
format_errors: 0
avg_steps: 3.05
```

This means the SFT checkpoint can reliably interact with the environment and submit answers, but it is not yet choosing correct page elements or answers.

Observed action collapse:

```text
open_page:
  shop_home: 11
  course_home: 9

click:
  shop_item_006: 11
  course_item_006: 8
  filter_department_computer_science: 1
  course_cs_item_002: 1

submit_answer:
  RunBeat Clip: 11
  BIO105: 8
  CS220: 1
```

The model has learned the workflow shape:

```text
open_page -> click -> submit_answer
```

but often maps many tasks to the same memorized click target. This is the first clear rollout-level boundary result.

## SFT-Epoch1 Eval20

An approximately one-epoch local SFT run was completed from the tool-use init checkpoint:

```bash
python scripts/train_sft_minimind.py \
  --config configs/sft_minimind_webnav_smoke.yaml \
  --override run_name=sft_minimind_webnav_epoch1 \
  --override output_dir=checkpoints/sft_minimind_webnav_epoch1 \
  --override max_steps=320 \
  --override warmup_steps=16 \
  --override log_interval=20 \
  --override save_interval=160
```

Training loss reached about `0.0237` by step 320.

Checkpoint:

```text
checkpoints/sft_minimind_webnav_epoch1/latest.pt
```

Small next-action eval sample:

```text
eval examples: 20
wrapper_ok: 100.0%
json_ok: 100.0%
valid_tool_name: 100.0%
tool_name_match: 100.0%
target_exact_match: 75.0%
```

Full rollout eval20:

```text
tasks: 20
success_rate: 10.0%
submitted_rate: 100.0%
invalid_tool_calls: 0
format_errors: 0
avg_steps: 3.2
```

Successful tasks:

```text
shop_00802
course_00808
```

Action distribution:

```text
open_page:
  shop_home: 11
  course_home: 9

click:
  shop_item_005: 10
  course_item_007: 6
  filter_credits_4: 3
  course_credits_4_item_003: 3
  sort_rating_desc: 1
  shop_rating_item_005: 1

submit_answer:
  NoteTab Air: 10
  BIO260: 9
  PixelPad S: 1
```

Compared with SFT-200step, one-epoch SFT improves rollout task success from `0/20` to `2/20`, but action collapse remains. The collapse center changes from `shop_item_006 -> RunBeat Clip` / `course_item_006 -> BIO105` to `shop_item_005 -> NoteTab Air` / `course_item_007 or BIO260`.

## Tooluse-Init Baseline

The unsupervised tool-use init checkpoint was too slow for a 20-task rollout because it repeatedly produced invalid outputs. A constrained baseline was run instead:

```bash
python scripts/run_minimind_webnav_eval.py \
  --checkpoint outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt \
  --limit 3 \
  --max-steps 3 \
  --max-new-tokens 32 \
  --device cuda \
  --output outputs/minimind_rollout/tooluse_init_eval3_step3_trajectories.jsonl \
  --report reports/minimind_tooluse_init_rollout_eval3_step3.json
```

Result:

```text
tasks: 3
success_rate: 0.0%
submitted_rate: 0.0%
invalid_tool_calls: 9
format_errors: 9
avg_steps: 3.0
```

## Interpretation

The SFT step clearly improves protocol following:

```text
tooluse-init: no parseable tool calls in the constrained baseline
SFT-200step: 0 format errors across 20 rollout tasks
```

One-epoch SFT starts to produce nonzero task success, but success remains low because the model has not learned robust observation-grounded argument selection.

This gives a useful research conclusion:

```text
For a from-scratch 63M LM, tool-call syntax and tool sequencing are learnable with small SFT, while grounded element selection remains the bottleneck.
```

## Next Step

The next practical experiment is one of:

```text
1. Add supervised loss weighting or extra examples for click argument selection.
2. Implement verifier reranking / best-of-N to select among multiple valid tool calls.
3. Continue SFT for 2-3 epochs only if accompanied by eval-based early stopping.
```

The most informative next step is failure-driven reranking or data balancing, because format is now stable and the remaining failure mode is argument grounding rather than syntax.
