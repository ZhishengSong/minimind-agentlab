# Step 11: MiniMind Tool-Use SFT Data

Last updated: 2026-06-23

## Goal

Convert WebNav-RL expert trajectories into MiniMind-ready next-action SFT data, then verify that the tool-use init checkpoint can train with assistant-only labels.

## Source Data

The source WebNav-RL SFT files are:

```text
D:\job\Program\WebNav-RL\training\sft_train.jsonl
D:\job\Program\WebNav-RL\training\sft_eval.jsonl
```

Each source row contains a multi-turn `messages` list with:

```text
user -> assistant tool_call -> tool observation -> assistant tool_call -> ...
```

All assistant messages are validated as:

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

## Conversion Format

The MiniMind conversion uses next-action SFT. Each assistant tool call becomes one training example:

```text
prompt_text = system/user/tool context up to this turn + <|im_start|>assistant\n
target_text = assistant tool call + <|im_end|>\n
text = prompt_text + target_text
```

This keeps examples short and directly trains the model to predict the next tool action from the current observation.

Tool observations are rendered as user-side tool responses:

```text
<|im_start|>user
<tool_response>
...
</tool_response><|im_end|>
```

The later SFT dataset masks all prompt tokens with `-100` and only trains on `target_text`.

## Scripts

Conversion:

```bash
python scripts/convert_webgym_to_minimind_sft.py \
  --input D:\job\Program\WebNav-RL\training\sft_train.jsonl \
  --output outputs/minimind_sft/sft_train_next_action.jsonl \
  --report reports/minimind_sft_train_report.json \
  --tokenizer-path outputs/tooluse_init/tokenizer \
  --max-seq-len 2048
```

Inspection:

```bash
python scripts/inspect_minimind_sft.py \
  --input outputs/minimind_sft/sft_train_next_action.jsonl \
  --tokenizer-path outputs/tooluse_init/tokenizer \
  --show 2
```

SFT smoke training:

```bash
python scripts/train_sft_minimind.py \
  --limit-examples 8 \
  --override max_steps=3 \
  --override save_interval=3 \
  --override log_interval=1 \
  --override output_dir=checkpoints/sft_minimind_webnav_smoke3 \
  --override run_name=sft_minimind_webnav_smoke3
```

## Conversion Results

Train split:

```text
source trajectories: 800
next-action examples: 2530
skipped_for_length: 0
tool_counts:
  open_page: 800
  click: 930
  submit_answer: 800
total token length:
  min: 172
  mean: 485.21
  p50: 569
  p95: 838
  max: 954
```

Eval split:

```text
source trajectories: 200
next-action examples: 637
skipped_for_length: 0
tool_counts:
  open_page: 200
  click: 237
  submit_answer: 200
total token length:
  min: 172
  mean: 487.56
  p50: 568
  p95: 838
  max: 952
```

No examples exceed the current `max_seq_len=2048` conversion limit.

## Smoke Training Result

The 3-step smoke run used 8 converted examples with effective batch size 8.

```text
step 1 loss 2.9140
step 2 loss 2.0270
step 3 loss 1.6000
```

Saved checkpoint:

```text
checkpoints/sft_minimind_webnav_smoke3/latest.pt
```

The checkpoint loads through `scripts/generate.py`. After only 3 updates, the generated output is not yet a valid tool call; this is expected. The smoke validates the data path, assistant-only label masking, training loop, checkpoint save, and inference load path.

## 20-Step Smoke Result

The configured 20-step smoke ran on all 2530 converted train examples with effective batch size 8.

```text
step 1 loss 2.9575
step 10 loss 0.5792
step 20 loss 0.4425
```

Saved checkpoint:

```text
checkpoints/sft_minimind_webnav_smoke/latest.pt
```

At 20 steps, generation already learns the `<tool_call>` wrapper and valid tool names, but click and submit arguments are still often malformed or repeated. This is enough to justify a longer SFT pass.

During this check, the low-level tokenizer wrapper was fixed so `<|im_end|>` is recognized as `eos_token_id=2`. This lets generation stop at the first assistant end token.

## 200-Step SFT Result

A 200-step local CUDA/bf16 run was completed:

```bash
python scripts/train_sft_minimind.py \
  --config configs/sft_minimind_webnav_smoke.yaml \
  --override run_name=sft_minimind_webnav_200step \
  --override output_dir=checkpoints/sft_minimind_webnav_200step \
  --override max_steps=200 \
  --override warmup_steps=10 \
  --override log_interval=10 \
  --override save_interval=100
```

Saved checkpoint:

```text
checkpoints/sft_minimind_webnav_200step/latest.pt
```

The 200-step run was started before fixing the multi-step SFT logging average, so its JSONL loss values are not the preferred source for loss reporting. The checkpoint itself is valid and was evaluated through generation.

Full next-action format evaluation on all 637 eval examples:

```text
wrapper_ok: 100.0%
json_ok: 100.0%
valid_tool_name: 100.0%
tool_name_match: 100.0%
arguments_exact_match: 68.6%
target_exact_match: 68.6%
```

Report:

```text
reports/minimind_sft_200step_format_eval637.json
```

Interpretation:

- SFT successfully teaches the MiniMind checkpoint the tool-call format.
- The model reliably chooses the right tool type for next-action examples.
- Remaining errors are mostly argument selection errors, especially choosing the wrong `element_id` for `click`.
- This is a good boundary-experiment result: format learning is easy, observation-grounded action selection is the harder part.

## Next Step

The next useful step is a rollout-level MiniMind eval against the WebNav-RL environment, not only next-action teacher-forced prompts. That requires a MiniMind generator adapter for WebNav-RL:

```text
MiniMind checkpoint -> WebNav-RL model runner -> verifier summary
```

The first comparison should be:

```text
Pretrain/tooluse-init vs SFT-200step
metrics: format accuracy, valid JSON, invalid tool calls, task success
```
