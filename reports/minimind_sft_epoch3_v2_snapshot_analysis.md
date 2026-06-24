# MiniMind Epoch3 Zero-Shot WebNav V2 Snapshot Analysis

Date: 2026-06-24

## Scope

This is a preliminary zero-shot evaluation of the V1-trained MiniMind SFT Epoch3 checkpoint on the current WebNav-RL V2 snapshot. V2 was not used for MiniMind training. The result must remain separate from the frozen V1 benchmark until WebNav V2 is finalized.

## Result

| Metric | Result |
| --- | ---: |
| Task success | 0/500 (0%) |
| Submitted | 500/500 (100%) |
| Format errors | 0 |
| Invalid tool calls | 2,097 |
| Average model steps | 5.194 |
| Easy / medium / hard success | 0/132, 0/66, 0/302 |

All 15 templates scored 0%. No submitted answer matched any target entity in the V2 eval set.

## First-Action Diagnosis

The model selected the correct tool type on every first step:

```text
open_page: 500/500
```

However, it never copied the V2 start page explicitly provided in the instruction:

```text
course_home: 263
shop_home: 237
valid V2 start page: 0/500
```

The coarse shopping/course domain was correct for 469/500 tasks, so the model recognized most task domains but mapped them to memorized V1 page IDs.

## Invalid-Call Breakdown

```text
unknown V1 start page IDs: 500
other unknown V1 page IDs: 7
submit_answer calls missing the answer argument: 1,590
total invalid tool calls: 2,097
```

After the initial page-open failure, the model repeatedly emitted V1-style IDs under incorrect argument keys, then submitted an old page or element ID as the answer. Common final answers included `course_home`, `shop_item_002`, `shop_item_008`, and `shop_item_006`.

## Context Audit

The report recorded:

```text
prompt_count: 2597
compacted_prompt_count: 0
max_prompt_tokens: 575
```

No prompt reached a real V2 page observation because the first action always failed. Therefore this run does not yet measure whether the model can select or copy random V2 element IDs from visible page content.

## Interpretation

The model transferred the outer tool-call protocol but not the structural identifier policy:

- XML/JSON tool syntax remained stable.
- Tool-type and task-domain recognition mostly transferred.
- Fixed V1 page and element IDs were memorized.
- The model did not follow an explicitly supplied unseen start-page ID.

The V1 `191/200 = 95.5%` result remains valid for the fixed V1 benchmark, but it must not be presented as structural web-navigation generalization.

## Next Experiment

Before any V2 training or GRPO, run an oracle-first-open ablation: force only the first `open_page(task.start_page)` action, then let the unchanged Epoch3 model act from the real V2 homepage observation. This separates start-page ID transfer from random element-ID grounding.

If the oracle-open ablation also fails, the next justified model change is V2 or mixed V1+V2 SFT after WebNav V2 is frozen. GRPO is not justified before the model can produce valid V2 actions.
