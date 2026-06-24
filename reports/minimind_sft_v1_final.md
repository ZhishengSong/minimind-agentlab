# MiniMind Tool-Use SFT V1 Final Report

Date: 2026-06-24

## Scope

This milestone adapts the from-scratch 63M MiniMind checkpoint to structured next-action tool calling. It covers tokenizer adaptation, checkpoint embedding resize, SFT data conversion, assistant-only loss masking, local smoke tests, server Epoch1/2/3 training, full held-out next-action and closed-loop rollout evaluation, checkpoint selection, and local artifact verification.

## Base

```text
pretrain base: 50k MiniMind checkpoint
tool-use init vocab: 6404
tool-use init checkpoint: outputs/tooluse_init/pretrain_step_050000_tooluse_init.pt
```

## Data

```text
source train trajectories: 800
source eval trajectories: 200
next-action train examples: 2530
next-action eval examples: 637
max train length: 954 tokens
max eval length: 952 tokens
```

Each SFT example contains the full system/user/tool context as prompt tokens and one assistant tool call as the supervised target. Prompt labels are masked with `-100`.

## Training

Independent server runs were trained from the same tool-use init checkpoint:

```text
Epoch1: 320 optimizer steps
Epoch2: 640 optimizer steps
Epoch3: 960 optimizer steps
```

Epoch3 finished with training loss about `0.0012` and was selected after held-out evaluation.

## Evaluation

Same eval100 slice:

| Checkpoint | Wrapper/JSON/Tool Name | Arguments Exact | Target Exact |
| --- | ---: | ---: | ---: |
| Epoch1 | 100% | 72% | 72% |
| Epoch2 | 100% | 96% | 96% |
| Epoch3 | 100% | 100% | 100% |

Epoch3 full 637-example evaluation:

```text
wrapper_ok: 637/637 = 100%
json_ok: 637/637 = 100%
valid_tool_name: 637/637 = 100%
tool_name_match: 637/637 = 100%
arguments_exact_match: 628/637 = 98.59%
target_exact_match: 628/637 = 98.59%
format_errors: 0
```

## Closed-Loop Rollout Evaluation

Epoch3 was evaluated greedily on the fixed 200-task WebNav-RL V1 eval set:

```text
task_success: 191/200 = 95.5%
submitted: 200/200 = 100%
invalid_tool_calls: 0
format_errors: 0
average_model_steps: 3.185
course_tasks: 94/94 = 100%
shopping_tasks: 97/106 = 91.5%
```

All nine failures were valid but incorrect shopping clicks. They reduce to three repeated patterns: three `$24` price lookups selected BassFlow Mini instead of TravelMug One, four white-earbud queries selected the blue BassFlow Mini instead of QuietLite Pro, and two under-$100 minimum-price tasks selected the sixth filtered item instead of the seventh. See `reports/minimind_sft_epoch3_rollout_failure_analysis.md`.

## Selected Artifact

Local path:

```text
checkpoints/sft_minimind_webnav_epoch3_server/latest.pt
```

Size:

```text
756857297 bytes (about 722 MiB)
```

SHA256:

```text
77f29f3f5fd812e2fa05ba3afb6af85b0d319dad2de6e7fbff52e72d35e87ce6
```

The downloaded checkpoint hash matches the server hash. A local CPU load/generation smoke test completed successfully and exactly generated the expected `shop_item_007` click action for `shop_00801::turn02`.

## Decision

Epoch3 is the final MiniMind Tool-Use SFT V1 checkpoint. Further plain SFT epochs are not currently justified because training loss is saturated, held-out next-action accuracy is 98.59%, and closed-loop task success is 95.5% with no protocol errors.

## Pretrain Regression Check

Both checkpoints were evaluated on the same fixed 1k-example pretrain validation slice with 212,462 predicted tokens:

| Checkpoint | Loss | Perplexity |
| --- | ---: | ---: |
| Tool-use init | 2.252283 | 9.509420 |
| SFT Epoch3 | 2.569051 | 13.053436 |

Delta:

```text
loss: +0.316768 (+14.1%)
perplexity: +3.544016 (+37.3%)
```

Interpretation: tool-use SFT produces a measurable general-language specialization tradeoff, but the model has not collapsed. This result supports stopping plain SFT at Epoch3 rather than adding more epochs.

Optional future research:

```text
best-of-N verifier reranking
GRPO-style Agentic RL
```
