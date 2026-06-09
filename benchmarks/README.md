# Fallen Empire — model benchmark suite

After benchmark or train+benchmark passes, prefer logging them via **`python scripts/ml_workflow.py benchmark`** or **`full`** so **`docs/run_history.md`** and `benchmarks/results/runs/<id>/` stay in sync.

This file explains benchmark types and how to run them. Use `docs/RUNS.md` for current-vs-historical adapter interpretation, and keep the full row archive in `docs/run_history.md`.

Versioned **task definitions** live here as JSON. The active lexical suite is `specialist_benchmark_tasks.json`, which maps prompts to one or more specialist adapters and is scored by cheap string rules (`all_contains`, `any_contains`, `none_contains`, `min_chars`). Treat this as a specialist regression/smoke signal, not a replacement for arena acceptance.

This measures whether the model follows instructions and uses expected substrings; it does **not** measure game-code task completion via apply/compile/preview. For deterministic game-edit capability, use `scripts/run_arena_acceptance_tests.py` or `python scripts/ml_workflow.py arena-acceptance`, which reports the **Arena Capability Index** from applyability, TypeScript checks, export preservation, preview readiness, retry count, token pressure, and task complexity. For open-source execution-based Python coding checks, use `scripts/run_evalplus_benchmark.py` or `python scripts/ml_workflow.py evalplus`.

## Run

From the repo root with the venv active:

```bash
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --specialist combat_risk --specialist ai_planning_explanation
python scripts/run_game_benchmark.py --tier A
python scripts/run_game_benchmark.py --tasks benchmarks/specialist_benchmark_tasks.json --output-jsonl benchmarks/results/run.jsonl
python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 --task-id loading-screen-polish
python scripts/run_evalplus_benchmark.py --suite humaneval --limit 5 --adapter-path checkpoints/fe-lora-30m
python scripts/run_routing_benchmark.py
```

`python scripts/ml_workflow.py benchmark` defaults to the specialist task suite; use repeated `--specialist <id>` to scope to one or more specialists, or `full` / `train --evaluate` with repeated `--bench-specialist <id>`.

`python scripts/ml_workflow.py evalplus --limit 5 --adapter-path …` records the EvalPlus run in the normal workflow artifacts. EvalPlus executes generated Python code in guarded subprocesses and downloads HumanEval+/MBPP+ datasets on first run.

### Task routing suite

`benchmarks/task_routing_tasks.json` checks the deterministic local/frontier/hybrid routing policy in `scripts/model_router.py`. It is a dry-run benchmark only: no local model load and no frontier API call.

`benchmarks/documentation_testing_agent_eval_tasks_v1.json` checks low-risk documentation/testing-agent prompts that should stay on the local `documentation` adapter. Regenerate it with:

```bash
python scripts/build_documentation_testing_agent_eval_tasks_v1.py
python3 -m unittest tests/test_documentation_testing_agent_benchmark.py -v
```

`benchmarks/documentation_agent_rag_tasks_v1.json` checks whether the local 7B documentation/testing agent can answer repo-practices questions when supplied with retrieved context from `data/rag/documentation_agent_corpus.json`:

```bash
python scripts/ml_workflow.py documentation-rag-benchmark
```

Direct invocation (same script):

```bash
PYTHONPATH=scripts python3 scripts/run_documentation_agent_benchmark.py --use-rag
python3 -m unittest tests/test_documentation_rag.py -v
```

`benchmarks/run_analysis_rag_tasks_v1.json` is a second RAG lane focused on documenting/analyzing runs (`docs/run_history.md`, `docs/SPECIALIZED_RUN_HISTORY.md`, latest `runs/<id>/manifest.json` + `RUN.md`, timeseries):

```bash
python scripts/build_run_analysis_rag_corpus.py
PYTHONPATH=scripts python3 scripts/run_run_analysis_agent_benchmark.py --use-rag
python3 -m unittest tests/test_run_analysis_rag.py -v
```

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

Append objects to `specialist_benchmark_tasks.json` (or add another file and pass `--tasks`):

- `id` — stable slug
- `category` — free-form tag for reports
- `specialists` — array of specialist ids this task should exercise
- `prompt` — user message (system prompt is fixed in the runner)
- `expect` — scoring object; all listed constraints must pass
