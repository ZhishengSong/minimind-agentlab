# MiniMind-Compatible Local Assets

Place real MiniMind-compatible local assets here.

Expected layout:

```text
data/minimind/
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── pretrain_t2t_mini.jsonl
```

The data file is intentionally not tracked by Git.

The tokenizer directory is also ignored by Git because tokenizer files can be large or copied from external sources.

Validate the assets after placing them:

```bash
python scripts/inspect_tokenizer.py --tokenizer-path data/minimind/tokenizer --model-config configs/minimind_64m.yaml
python scripts/validate_pretrain_assets.py --config configs/pretrain_minimind_local.yaml
```
