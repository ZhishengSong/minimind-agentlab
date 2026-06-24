# MiniMind Epoch3 WebNav V2 Oracle-Open Analysis

Date: 2026-06-24

## Purpose

The zero-shot V2 snapshot run never reached a V2 observation because Epoch3 reused the V1 `shop_home` and `course_home` identifiers. This ablation forces only the first `open_page(task.start_page)` action and then returns control to the unchanged model.

This is a diagnostic ablation, not an autonomous task-success result.

## Result

| Metric | Result |
| --- | ---: |
| Task success | 0/500 (0%) |
| Submitted | 61/500 (12.2%) |
| Invalid tool calls | 3,279 |
| Format errors | 1,890 |
| Average model steps | 7.68 |
| Max-step termination | 439/500 |
| Prompt compaction | 2,044/3,340 model calls |

All easy, medium, and hard tasks failed.

## First Model Decision

After the oracle opened the correct V2 start page, the model's first real decision produced:

```text
format error: 445/500
parseable click: 55/500
successful click: 0/500
```

None of the 1,450 parsed model actions used a random V2 `el_*` identifier, and none succeeded. Parseable outputs continued to use V1 identifiers such as `shop_item_001`.

## Error Breakdown

Across parsed model actions:

```text
submit_answer: 1183
click: 144
open_page: 38
misspelled/unknown tool variants: 85
```

The largest semantic errors were:

```text
submit_answer missing answer: 1122
click missing element_id: 77
```

Parser failures were:

```text
invalid_json: 1691
invalid_wrapper: 189
invalid_schema: 10
```

## Context-Length Diagnosis

Epoch3 SFT used `max_seq_len=1024`, while every first post-open V2 prompt was much longer:

| First model decision | Count | Prompt tokens |
| --- | ---: | ---: |
| Parseable | 55 | 1795-1818, median 1804 |
| Format error | 445 | 1782-2048, median 2000 |

All 232 course prompts failed to parse. The 55 parseable prompts were shopping tasks with shorter observations, but they still selected old V1 IDs.

## Conclusion

The ablation identifies two independent out-of-distribution barriers:

1. V2 observations exceed the sequence lengths seen during SFT, causing severe format degradation.
2. Even when output remains parseable, the model does not copy random visible `el_*` identifiers and falls back to memorized V1 IDs.

Additional zero-shot rollout variants are unlikely to be informative. GRPO is not justified while the policy cannot produce valid V2 actions.

The next model experiment should wait for a frozen WebNav V2 snapshot, then use V2 or mixed V1+V2 next-action SFT with a 2048-token training length. V1 eval200 must remain a regression benchmark during that training.
