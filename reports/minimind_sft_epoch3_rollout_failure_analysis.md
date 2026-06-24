# MiniMind SFT Epoch3 Rollout Failure Analysis

Date: 2026-06-24

## Evaluation Summary

Epoch3 was evaluated greedily on the fixed 200-task WebNav-RL V1 eval set.

| Metric | Result |
| --- | ---: |
| Task success | 191/200 (95.5%) |
| Submitted | 200/200 (100%) |
| Invalid tool calls | 0 |
| Format errors | 0 |
| Average model steps | 3.185 |

All 94 course tasks succeeded. Shopping tasks reached 97/106 (91.5%).

## Failed Tasks

| Task IDs | Template | Count | Expected | Predicted | Wrong click |
| --- | --- | ---: | --- | --- | --- |
| `shop_00839`, `shop_00945`, `shop_00977` | `shopping_price_lookup` | 3 | TravelMug One | BassFlow Mini | `shop_item_002` instead of `shop_item_010` |
| `shop_00880`, `shop_00893`, `shop_00924`, `shop_00995` | `shopping_color_category` | 4 | QuietLite Pro | BassFlow Mini | `shop_item_002` instead of `shop_item_003` |
| `shop_00847`, `shop_00896` | `shopping_under_100_lowest_price` | 2 | TravelMug One | FocusLamp Max | `shop_under_100_item_006` instead of `shop_under_100_item_007` |

Every failure followed the required tool protocol, opened the correct starting page, emitted valid JSON, called valid tools, and submitted an answer. The failure occurred at element selection; the model then consistently submitted the item it had selected.

## Training-Distribution Evidence

The 800-task training split contains:

| Template/target | Training examples |
| --- | ---: |
| Price lookup -> TravelMug One | 5 |
| Price lookup -> BassFlow Mini | 9 |
| Color/category -> QuietLite Pro | 8 |
| Color/category -> BassFlow Mini | 13 |
| Under-100 lowest-price -> TravelMug One | 9 |

The first two failure groups select a more frequent nearby shopping action instead of the lower-frequency target. The final group has correct training coverage but shows a stable one-position error (`item_006` versus `item_007`) after filtering.

## Interpretation

Epoch3 has solved protocol following and most observation-conditioned action selection in this environment. The remaining errors are deterministic argument-selection mistakes concentrated in three shopping patterns, not parser failures or long-horizon control failures.

The benchmark is a fixed, deterministic, in-distribution synthetic WebNav-RL V1 environment. The 95.5% result supports a claim of closed-loop competence on this benchmark, but it must not be described as general web-browsing ability.

## Next Decision

Further plain SFT epochs are not justified. If additional improvement is needed, the most targeted options are:

1. Balance or augment the three failed shopping patterns.
2. Add verifier-guided best-of-N selection for ambiguous click actions.
3. Preserve this run as the SFT baseline before any GRPO experiment.
