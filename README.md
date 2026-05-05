# fallen-empire-lora

Small **separate** project for experimenting with **LoRA fine-tuning** on open code models, using your game repo as training text. Keeps huge weights and Python ML deps out of the main game repository.

## Why this is not inside `fallen-empire`

- Checkpoints and HF downloads are large.
- Python MLX stack vs the game’s Node/TypeScript toolchain.
- You can point data scripts at the game path read-only.

## Prereqs

- macOS, Apple Silicon (tested conceptually for M4 Pro + 24 GB).
- Python **3.11+ recommended for new environments** (`python3 -m venv .venv && source .venv/bin/activate`). The current verified local venv is recorded in `docs/PROJECT_STATE.md`.

## Quick start

1. Install deps: `pip install -r requirements.txt` (or `pip install -r requirements.txt -c requirements.lock.txt` for the recorded reproducibility constraints).
2. Use the **built-in workflow** (auto-selects `.venv/bin/python` when available; most subcommands write `docs/run_history.md` + per-run logs under `benchmarks/results/runs/`):

   ```bash
   source .venv/bin/activate
   .venv/bin/python scripts/ml_workflow.py smoke          # fast sanity: synthetic data + tiny train + benchmark
   export SOURCE_REPO=/Users/natreed/fallen-empire
   .venv/bin/python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest -- --iters 400
   ```

   See **`docs/WORKFLOW.md`** for subcommands (`prepare`, `train`, `benchmark`, `full`, `smoke`, `arena-dashboard`, …).

3. Manual steps (same as workflow internals): export writes `data/raw/repo_text.jsonl`; `build_lora_dataset.py` builds `data/lora/qwen25-coder-7b/game_text/*.jsonl`; **`mlx_lm.lora --train -c training/lora_qwen25_coder_7b.yaml`** trains. Layout details: **`docs/DATA_LAYOUT.md`**. Export hygiene (size caps, secret redaction) is built into `scripts/export_repo_for_training.py`.

## Docs Map

- `docs/PROJECT_STATE.md` — current verified environment, defaults, and known issues.
- `docs/RUNS.md` — current adapter/run recommendations vs historical experiments.
- `docs/WORKFLOW.md` — commands for `scripts/ml_workflow.py`.
- `docs/DATA_LAYOUT.md` — canonical data/checkpoint paths by model lineage.
- `docs/CHUNKED_GAME_TEXT.md` — long-file chunking behavior for `game_text`.
- `docs/SESSION_LOG.md` and `docs/run_history.md` — append-only historical logs.
- `docs/ARENA_PROGRESSION.md` — comparable arena A/B batches; `docs/ARENA_ROADMAP.md` — dashboard HTML, apply KPIs, promotion gate.
- `docs/GAME_ARENA_APPLY_CONTRACT.md` — strict output-shape contract to reduce `no_applyable_changes`.

## Open in Cursor

**File → Open Folder…** → choose `/Users/natreed/fallen-empire-lora` so this chat’s context is ML-only; keep the game in another window.
