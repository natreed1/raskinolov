# Training (LoRA + curriculum intent)

**Preferred:** `python scripts/ml_workflow.py` — see **`docs/WORKFLOW.md`**. Every run appends **`docs/run_history.md`** and writes **`benchmarks/results/runs/<id>/`** (manifest, `RUN.md`, logs).

**Browser UI:** `python scripts/train_ui_gradio.py` (default **http://127.0.0.1:7862**) — export, build, train with live logs; use `ml_workflow.py` when you need the same audit trail on disk.

## How this fits together

1. **Game text export** (`scripts/export_repo_for_training.py`) writes `data/raw/repo_text.jsonl` — one JSON object per file (`path`, `text`). Large; gitignored.

2. **LoRA dataset build** (`scripts/build_lora_dataset.py`) turns that into the layout **`mlx_lm.lora` requires**: a directory containing **`train.jsonl`**, **`valid.jsonl`**, and **`test.jsonl`**. Each line is `{"text": "# path\\n\\n<file body>"}` so the model does causal LM on code-shaped text (mlx-lm auto-detects `text` vs `messages`).

3. **Training** (`mlx_lm.lora --train -c training/lora_qwen_coder.yaml`) loads the **base MLX model** from Hugging Face, freezes it, injects **LoRA adapters** into the last `num_layers` linear layers (here **`-1` = all layers**), and minimizes cross-entropy on `train`, checking `valid` on a schedule. Adapter weights + `adapter_config.json` land under **`adapter_path`** (default `checkpoints/fe-lora-latest`).

4. **Inference** — point **`--adapter-path`** (or `ADAPTER_PATH`) at that folder in `scripts/chat_gradio.py` or `scripts/run_game_benchmark.py` to compare **base vs fine-tuned** Albert.

5. **Benchmark evolution** (`scripts/evolve_benchmark_seasons.py`) is **orthogonal**: it evolves *evaluation tasks*, not gradients. Use it to tighten the suite as the model improves.

## Commands

### Real corpus (after export)

```bash
source .venv/bin/activate
export SOURCE_REPO=/path/to/fallen-empire
python scripts/export_repo_for_training.py
python scripts/build_lora_dataset.py --out-dir data/lora/game_text
mlx_lm.lora --train -c training/lora_qwen_coder.yaml
```

### One-liner wrapper

```bash
./scripts/train_lora.sh
./scripts/train_lora.sh --synthetic-smoke --iters 5 --batch-size 1
```

`--synthetic-smoke` is only for **wiring checks** (tiny fake TypeScript snippets), not quality.

### Override YAML from the CLI

Any `mlx_lm.lora` flag can follow the wrapper or be appended to `mlx_lm.lora` directly; **CLI overrides unset YAML fields** (mlx-lm merge rule).

```bash
mlx_lm.lora --train -c training/lora_qwen_coder.yaml --iters 800 --learning-rate 5e-6
```

## Config files

| File | Role |
|------|------|
| `training/lora_qwen_coder.yaml` | Default **Qwen2.5-Coder-1.5B-Instruct-4bit** LoRA hyperparameters |
| `training/evolution_config.json` | Benchmark evolution + **named curriculum seasons** for how you *stage* data or eval later |

## Notes

- **`--mask-prompt`**: for `messages` / prompt-completion data, masks prompt tokens in the loss; not used for plain `text` code LM.
- **`max_seq_length`**: truncate long files; lower if you hit OOM.
- **`grad_checkpoint`**: trades compute for memory (on in YAML).
- **Export hygiene** is built into `scripts/export_repo_for_training.py`: oversize cap (`EXPORT_MAX_FILE_BYTES`, default 400kB), path/name rules for secrets and key material, binary-ish detection, and regex redaction for common token/key patterns. Re-run export after changing ignore rules.

## Human evaluation

Use `scripts/human_eval_ui.py` (default port **7861**) to generate answers on benchmark tasks, score them, and append rows to `benchmarks/results/human_eval.jsonl`.

## Game-side tests

In the **game** repository, `npm run test:ml-cohort` runs `scripts/ml-lora-cohort-guard.ts`. From the ML repo, `./scripts/run_game_ml_tests.sh` runs the same if `SOURCE_REPO` points at the game checkout.
