# Pretrain V0 Checkpoint Evaluation

All checkpoints were evaluated on the same fixed JSONL slice.

| Checkpoint | Step | Loss | Perplexity | Predicted Tokens | Tokens/sec | Device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pretrain_step_005000.pt | 5000 | 2.986131 | 19.8089 | 211462 | 54230.56 | cuda |
| pretrain_step_010000.pt | 10000 | 2.703081 | 14.925641 | 211462 | 29203.94 | cuda |
| pretrain_step_020000.pt | 20000 | 3.084734 | 21.861646 | 211462 | 16013.79 | cuda |
| pretrain_step_050000.pt | 50000 | 2.251902 | 9.505802 | 211462 | 26540.35 | cuda |
