# Server Artifact Manifest

Last updated: 2026-06-23

## Must Copy

These are required to run the server sweep without regenerating local work:

```text
outputs/tooluse_init/tokenizer/tokenizer.json
outputs/tooluse_init/tokenizer/tokenizer_config.json
outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt
outputs/minimind_sft/sft_train_next_action.jsonl
outputs/minimind_sft/sft_eval_next_action.jsonl
configs/sft_minimind_webnav_smoke.yaml
configs/tool_special_tokens.json
```

Required scripts:

```text
scripts/train_sft_minimind.py
scripts/run_minimind_webnav_eval.py
scripts/run_server_minimind_sft_sweep.py
scripts/summarize_rollout_reports.py
```

Required source:

```text
src/
```

Required sibling project:

```text
WebNav-RL/
```

At minimum, WebNav-RL must include:

```text
env/
rollout/
tasks/eval_tasks.jsonl
tools/
```

## Can Regenerate

These are useful but can be regenerated:

```text
reports/minimind_sft_*_report.json
reports/minimind_sft_*_eval*.json
outputs/minimind_rollout/*.jsonl
checkpoints/sft_minimind_webnav_*
```

## Optional Pretrain Archive

Only needed if rebuilding tool-use init checkpoint on the server:

```text
minimind-50k-artifacts/checkpoints/pretrain_step_050000.pt
data/minimind/tokenizer/
configs/tool_special_tokens.json
scripts/prepare_tooluse_checkpoint.py
```

If `outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt` already exists, the original 50k archive is not required for SFT.
