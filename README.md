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
2. Use the **built-in workflow** (auto-selects `.venv/bin/python` when available; writes `docs/run_history.md` + per-run logs under `benchmarks/results/runs/` every time):

   ```bash
   source .venv/bin/activate
   .venv/bin/python scripts/ml_workflow.py smoke          # fast sanity: synthetic data + tiny train + benchmark
   export SOURCE_REPO=/Users/natreed/fallen-empire
   .venv/bin/python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-latest -- --iters 400
   ```

   See **`docs/WORKFLOW.md`** for all subcommands (`prepare`, `train`, `benchmark`, `full`, `smoke`).

3. Manual steps (same as workflow internals): export writes `data/raw/repo_text.jsonl`; `build_lora_dataset.py` builds `data/lora/game_text/*.jsonl`; **`mlx_lm.lora --train -c training/lora_qwen_coder.yaml`** trains. Export hygiene (size caps, secret redaction) is built into `scripts/export_repo_for_training.py`.

## Next steps you’ll add

- Optional: synthetic instruction–response pairs (`messages` JSONL) for chat-style SFT.
- Tune LoRA YAML / `--` overrides; compare adapters with `python scripts/ml_workflow.py benchmark --adapter-path …`.

## Open in Cursor

**File → Open Folder…** → choose `/Users/natreed/fallen-empire-lora` so this chat’s context is ML-only; keep the game in another window.
