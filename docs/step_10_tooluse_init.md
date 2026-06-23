# Step 10: Tool-Use Tokenizer And Init Checkpoint

Last updated: 2026-06-23

## Goal

Prepare the 50k MiniMind pretrain checkpoint for tool-use SFT by making the tool boundary markers single tokens and resizing the model embeddings to match the updated tokenizer.

## Token Decision

The planned tool-use token set is recorded in:

```text
configs/tool_special_tokens.json
```

Required tokens:

```text
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<answer>
</answer>
<observe>
</observe>
<think>
</think>
```

The base MiniMind tokenizer already had these as single tokens:

```text
<tool_call>        id 21
</tool_call>       id 22
<tool_response>    id 23
</tool_response>   id 24
<think>            id 25
</think>           id 26
```

The Version 1 preparation added these tokens:

```text
<answer>           id 6400
</answer>          id 6401
<observe>          id 6402
</observe>         id 6403
```

Final tokenizer vocabulary size:

```text
old_vocab_size: 6400
new_vocab_size: 6404
```

## Script

The preparation script is:

```text
scripts/prepare_tooluse_checkpoint.py
```

Default command:

```bash
python scripts/prepare_tooluse_checkpoint.py
```

Default inputs:

```text
base_checkpoint: minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt
base_tokenizer: data/minimind/tokenizer
special_tokens: configs/tool_special_tokens.json
```

Default outputs:

```text
outputs/tooluse_init/tokenizer/
outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt
reports/tooluse_init_report.json
```

## Embedding Resize

The checkpoint uses tied embeddings, so `embed_tokens.weight` and `lm_head.weight` both need the resized vocabulary dimension.

The script initializes new token rows from the mean embedding of existing related special tokens:

```text
<tool_call>
</tool_call>
<tool_response>
</tool_response>
<think>
</think>
<|im_start|>
<|im_end|>
```

A small deterministic noise term is added with seed `42` and `noise_std=0.001`, so the four new token rows are close to existing tool/chat boundary embeddings without being identical.

## Validation

The tool-use init checkpoint was loaded successfully with the updated tokenizer:

```text
tokenizer_vocab_size: 6404
model_vocab_size: 6404
embed_tokens.weight: (6404, 512)
lm_head.weight: (6404, 512)
tied_embeddings_after_load: true
```

Smoke input:

```text
<tool_call>calculator</tool_call><answer>42</answer>
```

Encoded IDs:

```text
[21, 102, 5270, 2991, 22, 6400, 55, 53, 6401]
```

The forward pass completed and produced logits with shape:

```text
(1, 9, 6404)
```

Generation entry point also loaded the init checkpoint successfully:

```bash
python scripts/generate.py \
  --checkpoint outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt \
  --prompt "<tool_call>calculator</tool_call><answer>" \
  --device cuda \
  --max-new-tokens 16 \
  --temperature 0.0 \
  --output outputs/tooluse_init/tooluse_init_smoke.txt
```

The output is not expected to be useful before SFT. This check only validates tokenizer/checkpoint compatibility.

## Next Step

Version 2 starts by converting WebGym-RL expert trajectories into MiniMind SFT samples. The immediate implementation target is:

```text
scripts/convert_webgym_to_minimind_sft.py
```

The first SFT template should use the newly prepared boundary tokens:

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
<tool_response>...</tool_response>
<answer>...</answer>
<|im_end|>
```
