# Step 13: MiniMind Zero-Shot Eval on a WebNav V2 Snapshot

Last updated: 2026-06-24

## Scope

Evaluate the frozen MiniMind SFT Epoch3 checkpoint on the current WebNav-RL V2 snapshot without training on V2 data and without modifying the WebNav-RL project.

This is a preliminary structural-generalization result. It must remain separate from the final V1 result until WebNav V2 is frozen.

## Snapshot Inputs

```text
tasks: /root/autodl-tmp/WebNav-RL/tasks/v2/eval_tasks.jsonl
metadata: /root/autodl-tmp/WebNav-RL/pages/generated_pages_v2/metadata.json
eval tasks: 500
templates: 15
eval layout: eval_c
train/eval element ID overlap: 0
```

The V2 expert verification passed all 3,500 generated tasks, and the V2 data tests passed locally. The generated metadata and eval tasks were copied directly to the server rather than committed to WebNav-RL.

## MiniMind Adapter

`scripts/run_minimind_webnav_eval.py` now accepts an optional `--metadata` path. When present, it creates each episode with the corresponding WebNav `PageStore`; omitting the option preserves V1 behavior.

V2 observations produce longer prompts than the MiniMind context window. An audit of all 1,868 V2 eval decision points found:

```text
raw prompts over 2048 tokens: 831
prompts after message-aware compaction: 0 over 2048
maximum compacted prompt: 2048 tokens
```

Compaction preserves the system prompt, original task, and latest observation. It keeps the latest action when it fits; for 29 extreme prompts it omits the already-executed action text so the full observation remains available. The report records how often compaction was used.

## Server Smoke

```bash
cd /root/autodl-tmp/minimind-agentlab

python -u scripts/run_minimind_webnav_eval.py \
  --checkpoint checkpoints/sft_minimind_webnav_epoch3/latest.pt \
  --webnav-root /root/autodl-tmp/WebNav-RL \
  --tasks /root/autodl-tmp/WebNav-RL/tasks/v2/eval_tasks.jsonl \
  --metadata /root/autodl-tmp/WebNav-RL/pages/generated_pages_v2/metadata.json \
  --limit 20 \
  --device cuda \
  --max-new-tokens 48 \
  --temperature 0 \
  --output outputs/minimind_rollout/epoch3_v2_snapshot_eval20_trajectories.jsonl \
  --report reports/minimind_sft_epoch3_v2_snapshot_eval20.json
```

## Full Eval

Run all 500 tasks only after the smoke confirms that the metadata path and random element IDs execute correctly:

```bash
python -u scripts/run_minimind_webnav_eval.py \
  --checkpoint checkpoints/sft_minimind_webnav_epoch3/latest.pt \
  --webnav-root /root/autodl-tmp/WebNav-RL \
  --tasks /root/autodl-tmp/WebNav-RL/tasks/v2/eval_tasks.jsonl \
  --metadata /root/autodl-tmp/WebNav-RL/pages/generated_pages_v2/metadata.json \
  --limit 500 \
  --device cuda \
  --max-new-tokens 48 \
  --temperature 0 \
  --output outputs/minimind_rollout/epoch3_v2_snapshot_eval500_trajectories.jsonl \
  --report reports/minimind_sft_epoch3_v2_snapshot_eval500.json
```

## Interpretation Boundary

Epoch3 was trained only on V1 trajectories with fixed semantic element IDs. V2 uses held-out layouts and random IDs exposed in observations. This run therefore measures zero-shot transfer from memorized V1 action patterns to observation-grounded V2 navigation. It is not a V2-trained model result.

## Preliminary Result

The complete 500-task snapshot evaluation finished with:

```text
success: 0/500
submitted: 500/500
format_errors: 0
invalid_tool_calls: 2097
average_model_steps: 5.194
```

Every first output used the valid `open_page` tool, but none copied the V2 start page from the instruction. The model emitted `course_home` 263 times and `shop_home` 237 times, both memorized V1 identifiers. Consequently, no real V2 page observation was reached and prompt compaction was never triggered.

This is evidence of protocol transfer without structural identifier transfer. Full diagnosis is in `reports/minimind_sft_epoch3_v2_snapshot_analysis.md`.

The next controlled experiment is an oracle-first-open ablation. It will force only the correct V2 start-page action and then return control to the unchanged Epoch3 model, isolating random element-ID grounding from start-page transfer.

Run the ablation with:

```bash
python -u scripts/run_minimind_webnav_eval.py \
  --checkpoint checkpoints/sft_minimind_webnav_epoch3/latest.pt \
  --webnav-root /root/autodl-tmp/WebNav-RL \
  --tasks /root/autodl-tmp/WebNav-RL/tasks/v2/eval_tasks.jsonl \
  --metadata /root/autodl-tmp/WebNav-RL/pages/generated_pages_v2/metadata.json \
  --oracle-first-open \
  --limit 500 \
  --device cuda \
  --max-new-tokens 48 \
  --temperature 0 \
  --output outputs/minimind_rollout/epoch3_v2_oracle_open_eval500_trajectories.jsonl \
  --report reports/minimind_sft_epoch3_v2_oracle_open_eval500.json
```

This is an ablation, not an autonomous task-success result. Its purpose is to measure behavior after the first V2 observation becomes available.
