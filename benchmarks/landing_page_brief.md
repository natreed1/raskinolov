# Fallen Empire Landing Page Brief

## Product Intent

Fallen Empire is a hard-sci-fi strategy simulation game about imperial collapse,
frontier logistics, faction pressure, and systemic consequences. The landing page
should sell the fantasy of commanding a brittle empire where every tactical choice
feeds back into morale, economy, territory, and survival.

## Desired Visual Direction

- Dark, high-contrast sci-fi interface.
- Tactical map / command center mood.
- Clear readable typography; avoid generic fantasy ornament.
- Use sections that could plausibly become a real game website:
  hero, feature cards, simulation systems, factions/world, screenshots/placeholders,
  and a call-to-action.
- Responsive layout for desktop and mobile.

## Evaluation Criteria

Rate outputs by actually viewing the page, not just reading code:

- Visual quality and hierarchy.
- How well the copy understands Fallen Empire.
- Whether the page feels specific rather than generic.
- Responsiveness and interaction polish.
- Ease of using or extending the code.

## Repository Context Excerpts

### README

```markdown
# fallen-empire-lora

Small **separate** project for experimenting with **LoRA fine-tuning** on open code models, using your game repo as training text. Keeps huge weights and Python ML deps out of the main game repository.

## Why this is not inside `fallen-empire`

- Checkpoints and HF downloads are large.
- Python MLX stack vs the game’s Node/TypeScript toolchain.
- You can point data scripts at the game path read-only.

## Prereqs

- macOS, Apple Silicon (tested conceptually for M4 Pro + 24 GB).
- Python **3.11+** (`python3 -m venv .venv && source .venv/bin/activate`).

## Quick start

1. Install deps: `pip install -r requirements.txt`
2. Use the **built-in workflow** (writes `docs/run_history.md` + per-run logs under `benchmarks/results/runs/` every time):

   ```bash
   source .venv/bin/activate
   python scripts/ml_workflow.py smoke          # fast sanity: synthetic data + tiny train + benchmark
   export SOURCE_REPO=/Users/natreed/fallen-empire
   python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-latest -- --iters 400
   ```

   See **`docs/WORKFLOW.md`** for all subcommands (`prepare`, `train`, `benchmark`, `full`, `smoke`).

3. Manual steps (same as workflow internals): export writes `data/raw/repo_text.jsonl`; `build_lora_dataset.py` builds `data/lora/game_text/*.jsonl`; **`mlx_lm.lora --train -c training/lora_qwen_coder.yaml`** trains. Export hygiene (size caps, secret redaction) is built into `scripts/export_repo_for_training.py`.

## Next steps you’ll add

- Optional: synthetic instruction–response pairs (`messages` JSONL) for chat-style SFT.
- Tune LoRA YAML / `--` overrides; compare adapters with `python scripts/ml_workflow.py benchmark --adapter-path …`.

## Open in Cursor

**File → Open Folder…** → choose `/Users/natreed/fallen-empire-lora` so this chat’s context is ML-only; keep the game in another window.

```

### Project State

```markdown
# fallen-empire-lora — project state

Last verified: **2026-04-26** (Apple Silicon macOS; workspace path `/Users/natreed/fallen-empire-lora`). EvalPlus runner/workflow wiring, router dry-run benchmark, and adapter selection report added; EvalPlus dataset download was blocked by GitHub 502 during verification.

## Purpose

Experiment with **LoRA fine-tuning** on open code models using the Fallen Empire game repository as training text, without bloating the main game repo. See repository `README.md` for rationale.

## Environment

| Item | Value |
|------|--------|
| OS | darwin 24.x (from prior session metadata) |
| Python (venv) | **3.9.6** (`/Users/natreed/fallen-empire-lora/.venv`) |
| README recommendation | Python 3.11+ (optional; current venv is 3.9 and works) |

### Pinned Python packages (venv)

Recorded via `pip freeze` on 2026-04-23; `evalplus==0.3.1` added on 2026-04-26:

```
mlx==0.29.3
mlx-lm==0.29.1
mlx-metal==0.29.3
huggingface_hub==0.36.2
transformers==4.57.6
numpy==2.0.2
gradio==4.44.1
gradio_client==1.3.0
evalplus==0.3.1
```

Install source: `requirements.txt`; resolver pins include `gradio>=4.44,<5` for the chat UI.

## Default base model (smoke / planned LoRA base)

| Field | Value |
|-------|--------|
| Hugging Face id | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| Role | Small instruct **code** model, MLX 4-bit, suitable for first downloads and iteration on unified memory |

Alternatives (same org, larger): `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`, `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (set `MODEL` env var for scripts or pass `--model`).

## Verified commands

### Create venv and install

```bash
cd /Users/natreed/fallen-empire-lora
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Smoke test: load + one generation

```bash
source .venv/bin/activate
python scripts/smoke_base_model.py --max-tokens 96
```

**Outcome (2026-04-23):** First full run downloaded 9 files from Hugging Face (~20 s on a fast connection); subsequent load ~0.7 s from cache. Generation ~0.8 s for 96 new tokens. Output contained correct TypeScript `clamp` implementation; end of string may include model-specific control tokens (harmless for smoke; strip in production UI if needed).

### Interactive chat (mlx-lm CLI)

```bash
source .venv/bin/activate
python -m mlx_lm chat --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
```

CLI entry points also installed in `.venv/bin/` (e.g. `mlx_lm.lora`, `mlx_lm.generate`).

### Gradio browser chat (project UI)

```bash
source .venv/bin/activate
pip install -r requirements.txt   # if gradio not installed yet
python scripts/chat_gradio.py
```

Defaults: bind **http://127.0.0.1:7860**, same base model as smoke script, `concurrency_limit=1` (one generation at a time). Uses `stream_generate` and yields cumulative text for token streaming. Gradio title and default system prompt name the assistant **Albert**.

Useful flags: `--port`, `--model`, `--adapter-path` (LoRA dir when you have one), `--max-tokens`, `--temp`, `--share` (temporary public Gradio URL). Environment mirrors: `MODEL`, `ADAPTER_PATH`, `SYSTEM_PROMPT`, `MAX_TOKENS`, `TEMP`.

**Outcome (2026-04-23):** `--help` and import-time test for `_history_to_messages` succeeded; full browser run not re-logged here (same model stack as smoke).

## Game-oriented model benchmark + evolution

| Item | Location |
|------|-----------|
| Task definitions (game) | `benchmarks/fallen_empire_tasks.json` (15 tasks; domains include evolution/serialization) |
| Task definitions (general) | `benchmarks/general_coding_tasks.json` (8 tasks; HTTP/SQL/encoding/doc/version trivia; use `--profile general`) |
| Execution benchmark | `scripts/run_evalplus_benchmark.py` (EvalPlus HumanEval+/MBPP+; use `ml_workflow.py evalplus`) |
| Routing benchmark | `benchmarks/task_routing_tasks.json` + `scripts/run_routing_benchmark.py` (local/frontier/hybrid policy dry-run) |
| Shared tier / mutation helpers | `scripts/benchmark_evolution_lib.py` |
| Runner | `scripts/run_game_benchmark.py` (`--profile game|general`, `--tier C|B|A` scales rubric + default decode cap) |
| Season evolution | `scripts/evolve_benchmark_seasons.py` + `training/evolution_config.json` |
| Training notes | `training/README.md` |
| Notes | `benchmarks/README.md` |

Scoring is heuristic (substring rules), intended to regress **base vs LoRA** and prompt changes—not to replace game unit tests.

```bash
source .venv/bin/activate
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --profile general
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5
python scripts/run_routing_benchmark.py
python scripts/run_game_benchmark.py --tier A
python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json
```

**Outcome (2026-04-24):** On `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` with greedy `temp=0` and default `--max-tokens` 512: **game 15/15** and **general 8/8** (no tier). (2026-04-23) Evolved task files are experimental; cap `min_chars` at **320** after mutation/tier scaling to stay decodable.

## Dataset export + hygiene

- Script: `scripts/export_repo_for_training.py`
- Default source: `~/fallen-empire` unless `SOURCE_REPO` is set
- Output: `data/raw/repo_text.jsonl` (gitignored under `data/raw/`)
- **Hygiene:** max file size **400 KB** (override with `EXPORT_MAX_FILE_BYTES`), extra skip dirs (`.turbo`, `credentials`, …), skip credential-like suffixes (`.pem`, `.key`, …), path substring rules (`secret`, `/.env`, …), NUL / control-char heuristic for “binary-ish” text, and **regex redaction** for common API keys / PEM blocks / obvious GitHub+Slack token shapes.

## Human evaluation UI

- Script: `scripts/human_eval_ui.py` (default **http://127.0.0.1:7861**)
- Appends JSONL rows to `benchmarks/results/human_eval.jsonl` (gitignored under `benchmarks/results/`).

## Training UI (Gradio)

- Script: `scripts/tra
```

### ML Workflow

```markdown
# Built-in ML workflow

Use **`scripts/ml_workflow.py`** as the single entry point for repeatable pipelines. **Every invocation** writes:

1. **`benchmarks/results/runs/<run_id>/manifest.json`** — machine-readable steps, per-step exit codes, final exit code, timing, argv, versions.  
2. **`benchmarks/results/runs/<run_id>/RUN.md`** — human-readable summary + links to step logs.  
3. **`benchmarks/results/runs/<run_id>/logs/*.log`** — full stdout/stderr per step.  
4. **`docs/run_history.md`** — one appended table row (committed) so the repo always carries a **durable index** of runs; copy artifacts off-machine if you need long-term archives (the `runs/` tree is gitignored).

Activate the venv first (`source .venv/bin/activate`).

## Subcommands

| Command | Does |
|---------|------|
| `python scripts/ml_workflow.py smoke` | Synthetic JSONL → 4 train iters → benchmark **base** model. Fast CI / sanity check. |
| `python scripts/ml_workflow.py prepare` | `export_repo_for_training.py` + `build_lora_dataset.py` (needs `SOURCE_REPO` + game checkout). |
| `python scripts/ml_workflow.py train` | `mlx_lm.lora --train` only. Optional `--evaluate` to run the benchmark after a successful train. |
| `python scripts/ml_workflow.py benchmark` | `run_game_benchmark.py` only; optional `--adapter-path`, `--profile` `game` or `general`. |
| `python scripts/ml_workflow.py evalplus` | `run_evalplus_benchmark.py` execution-based HumanEval+/MBPP+ subset or full suite. |
| `python scripts/ml_workflow.py full` | **prepare → train → benchmark** on the trained adapter; optional `--bench-profile general` for the generic coding suite on the last step. |
| `python scripts/ml_workflow.py train --evaluate` | Optional `--bench-profile general` with `--evaluate` to run the **general** suite on the new adapter. |

### Examples

```bash
# Quick sanity (no game repo, ~20–30 s typical)
python scripts/ml_workflow.py smoke

# Refresh corpus + LoRA splits only
export SOURCE_REPO=$HOME/fallen-empire
python scripts/ml_workflow.py prepare

# Train (YAML defaults); pass-through flags to mlx_lm.lora after `--`
python scripts/ml_workflow.py train --adapter-path checkpoints/exp-001 -- --iters 200

# Same + automatic benchmark on the new adapter
python scripts/ml_workflow.py train --adapter-path checkpoints/exp-001 --evaluate -- --iters 200

# Resume from an existing adapter and run the general benchmark
python scripts/ml_workflow.py train --evaluate --bench-profile general \
  --adapter-path checkpoints/exp-800 \
  -- --iters 800 --batch-size 1 --val-batches 8 \
  --resume-adapter-file checkpoints/exp-001/adapters.safetensors

# End-to-end non-interactive training pass + eval
python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-latest -- --iters 400

# Benchmark only (base model)
python scripts/ml_workflow.py benchmark

# Benchmark LoRA
python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-latest
python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-latest --profile general

# Execution-based HumanEval+ subset via EvalPlus
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5

# Pick the best non-overfit adapter from recorded artifacts
python scripts/select_best_adapter.py

# Dry-run cost router policy benchmark (no API calls)
python scripts/run_routing_benchmark.py
```

`evalplus` uses the open-source EvalPlus datasets and may download HumanEval+/MBPP+ from GitHub on first run. If GitHub is unavailable, retry later or set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` to a local EvalPlus JSONL file.

## Cost-Aware Routing

`scripts/model_router.py` provides the first local-vs-frontier routing layer:

- `LocalMlxBackend` wraps local `mlx-lm` inference.
- `OpenAICompatibleBackend` calls `FRONTIER_API_BASE_URL` + `FRONTIER_API_KEY` / `OPENAI_API_KEY` with `FRONTIER_MODEL`.
- `RoutingPolicy` deterministically chooses `local`, `frontier`, or `hybrid` and records estimated token cost p
```
