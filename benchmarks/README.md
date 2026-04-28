# Fallen Empire — model benchmark suite

After benchmark or train+benchmark passes, prefer logging them via **`python scripts/ml_workflow.py benchmark`** or **`full`** so **`docs/run_history.md`** and `benchmarks/results/runs/<id>/` stay in sync.

Versioned **task definitions** live here as JSON. The **game** suite (`fallen_empire_tasks.json`) is tuned to this game’s stack (TypeScript, hex `q/r`, Zustand, AI param names). The **general** suite (`general_coding_tasks.json`) uses generic CS trivia (HTTP, SQL, encodings, semver) with a **neutral** system prompt so you can compare **base vs LoRA** without always-on Fallen Empire context. All tasks are scored by **cheap string rules** (`all_contains`, `any_contains`, `none_contains`, `min_chars`).

This measures whether the model follows instructions and uses expected substrings; it does **not** measure code correctness via execution. For open-source execution-based coding checks, use `scripts/run_evalplus_benchmark.py` or `python scripts/ml_workflow.py evalplus`.

## Run

From the repo root with the venv active:

```bash
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --profile general
python scripts/run_game_benchmark.py --tier A
python scripts/run_game_benchmark.py --tasks benchmarks/fallen_empire_tasks.json --output-jsonl benchmarks/results/run.jsonl
python scripts/run_evalplus_benchmark.py --suite humaneval --limit 5 --adapter-path checkpoints/fe-lora-30m
python scripts/run_routing_benchmark.py
```

`python scripts/ml_workflow.py benchmark` defaults to the **game** profile; add `--profile general` for the general suite, or `full` / `train --evaluate` with `--bench-profile general`.

`python scripts/ml_workflow.py evalplus --limit 5 --adapter-path …` records the EvalPlus run in the normal workflow artifacts. EvalPlus executes generated Python code in guarded subprocesses and downloads HumanEval+/MBPP+ datasets on first run.

### Task routing suite

`benchmarks/task_routing_tasks.json` checks the deterministic local/frontier/hybrid routing policy in `scripts/model_router.py`. It is a dry-run benchmark only: no local model load and no frontier API call.

### Tiers (C / B / A)

Same metaphor as the game’s sim league: `--tier` scales `min_chars` expectations and sets a default decode budget (`C` 384 / `B` 512 / `A` 768 tokens). Stricter at the top.

### Evolutionary suite (seasons)

`training/evolution_config.json` drives `scripts/evolve_benchmark_seasons.py`, which mutates a **population** of tasks across **seasons** while keeping **anchor** ids fixed. Default output is a flat JSON array you can pass to `--tasks`.

```bash
python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json
```

See `training/README.md` for curriculum + LoRA default knobs.

### Human evaluation UI

Structured ratings (1–5 sliders + notes) saved as JSONL:

```bash
python scripts/human_eval_ui.py
# default http://127.0.0.1:7861 — use --port to avoid clashing with chat_gradio (7860)
```

### Landing page trial arena

Visual/product trials for comparing local LoRA, base, and Cursor/frontier attempts on standalone websites:

```bash
python scripts/landing_page_arena.py ui
# default http://127.0.0.1:7863
```

The arena writes isolated static-site attempts under `benchmarks/results/landing_page_trials/` and ratings under `benchmarks/results/landing_page_ratings.jsonl`. It uses `benchmarks/landing_page_brief.md` as shared Fallen Empire context and does not edit the game repo.

### Game task arena

Disposable worktree/copy trials for real game-code tasks:

```bash
python scripts/game_task_arena.py create --task-id ui-hud-copy
python scripts/game_task_arena.py packet --trial-id <trial_id> --attempt local
python scripts/game_task_arena.py apply --trial-id <trial_id> --attempt local --input /path/to/model_output.md
python scripts/game_task_arena.py verify --trial-id <trial_id> --attempt local
python scripts/game_task_arena.py grade --trial-id <trial_id> --attempt local --correctness 4
python scripts/game_task_arena.py report
```

Task specs live in `benchmarks/game_task_arena_examples.json`. Results are indexed in `benchmarks/results/game_task_index.jsonl`; per-trial manifests, diffs, command logs, ratings, and training records live under `benchmarks/results/game_task_trials/<trial_id>/`.

### Game-side regression (fallen-empire repo)

TypeScript guard that locks **Biome**, **MAP_SIZE_PRESETS**, **Tile** / **SimResult** shapes, and a **runSimulation** smoke run — aligned with ML benchmark prompts:

```bash
cd /path/to/fallen-empire && npm run test:ml-cohort
```

From this repo:

```bash
./scripts/run_game_ml_tests.sh   # uses SOURCE_REPO or ~/fallen-empire
```

Exit code **1** if any task fails (useful in CI). Machine-written lines under `benchmarks/results/` are gitignored; commit changes to `*_tasks.json` when you adjust rubrics.

## Adding tasks

Append objects to `fallen_empire_tasks.json` (or add another file and pass `--tasks`):

- `id` — stable slug
- `category` — free-form tag for reports
- `prompt` — user message (system prompt is fixed in the runner)
- `expect` — scoring object; all listed constraints must pass
