# GitHub TypeScript dataset pipeline

Purpose: build a practical compile-safe TypeScript corpus for LoRA training, aimed at arena tasks that fail on apply/`tsc`/export gates.

Script: `scripts/build_github_ts_dataset.py`

## What the builder does

- Ingests repositories from explicit list (`--repo owner/name`, `--repos-file`) and/or GitHub search (`--discover-query`).
- Applies suitability filters:
  - permissive SPDX license allowlist (default: MIT / Apache-2.0 / BSD / ISC / Unlicense / 0BSD / CC0-1.0),
  - strict TypeScript evidence in `tsconfig*.json` (`compilerOptions.strict=true` or strict key trio),
  - non-trivial app structure heuristic (`package.json` + include-root TS files).
- Extracts TS/TSX file-level rows from include roots (`src,app,components,pages,lib,utils,server,client,packages`) with test/build/vendor path filtering.
- Writes MLX-ready JSONL splits plus metadata/manifest.

## Output layout

Default output directory:

`data/lora/qwen25-coder-7b/github_ts_compile_safe/`

Files:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `samples_metadata.jsonl`
- `manifest.json`

## Basic usage

Run with explicit repositories:

```bash
source .venv/bin/activate
.venv/bin/python scripts/build_github_ts_dataset.py \
  --repo pmndrs/zustand \
  --repo reduxjs/redux-toolkit
```

Run with discovery (best effort; requires network and may hit rate limits if unauthenticated):

```bash
source .venv/bin/activate
export GITHUB_TOKEN=ghp_xxx   # optional but recommended
.venv/bin/python scripts/build_github_ts_dataset.py \
  --discover-query "react state management stars:>800" \
  --discover-query "typescript game ui stars:>200" \
  --discover-per-query 10
```

Offline wiring/smoke fallback:

```bash
source .venv/bin/activate
.venv/bin/python scripts/build_github_ts_dataset.py --synthetic-smoke
```

## Training on this dataset

```bash
source .venv/bin/activate
mlx_lm.lora --train -c training/lora_qwen25_coder_7b.yaml \
  --data data/lora/qwen25-coder-7b/github_ts_compile_safe \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-github-ts
```

## Notes

- GitHub discovery is resilient but not guaranteed; query failures are logged as warnings and explicit `--repo` entries still run.
- `manifest.json` includes accepted/rejected repo reasons so filter tuning is auditable.
- This script intentionally avoids heavy dependencies and uses Python stdlib HTTP/ZIP handling.
