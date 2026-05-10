# Chunked `game_text` dataset (long files preserved)

This file owns chunking behavior only. Canonical data paths live in `docs/DATA_LAYOUT.md`, current run/adaptor interpretation lives in `docs/RUNS.md`, and command examples live in `docs/WORKFLOW.md`.

## Problem

Previously, `scripts/build_lora_dataset.py` emitted **one row per source file** and **truncated** the body to `--max-chars` (default 12 000). **Everything after the cut was discarded**, which hurts large TypeScript / store / game logic files.

## Solution (default)

The builder now **chunks** long bodies into **multiple JSONL rows** per file:

- Each row is still `{"text": "…"}` for `mlx_lm.lora`.
- Headers include **`# path:`**, **`# part: i/n`**, and **`# chars: start-end`** (end exclusive) so chunks are unambiguous.
- **Train / valid / test** are assigned **per file** (stable hash of `seed:path`), then **all chunks** of that file land in the same split — **no leakage** across splits.
- If valid/test row counts fall below mlx-lm-friendly floors, whole files are **moved** from train until floors are met (same idea as the old row-based minimums, but on chunk rows).

## CLI

| Flag | Default | Role |
|------|---------|------|
| `--chunk-chars` | `8000` | Max **body** characters per chunk (header is extra; keeps typical tokenized length closer to `max_seq_length` in YAML). |
| `--chunk-overlap` | `512` | Characters repeated between adjacent chunks (continuity across boundaries; `0` allowed). |
| `--no-chunk` | off | **Legacy:** one row per file, body truncated to `--max-chars` (default 12 000). |

## Export size cap

`scripts/export_repo_for_training.py` default **`EXPORT_MAX_FILE_BYTES`** was raised so fewer whole files are skipped before chunking (see env override in that script).

## Tokenization note

Chunking is **character-based**. Training still applies **`max_seq_length`** in `training/lora_qwen25_coder_7b.yaml`; very dense code can tokenize longer than prose. If you see many `[WARNING] Some sequences are longer` lines, lower `--chunk-chars` (e.g. `6000`) before lowering quality elsewhere.
