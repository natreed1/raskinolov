# Built-in ML workflow

Use `scripts/ml_workflow.py` as the single entry point for repeatable pipelines. Launch it from the repo venv (`.venv/bin/python scripts/ml_workflow.py ...` or an activated venv); the script re-execs into `.venv/bin/python` when that interpreter exists so child steps use the same MLX toolchain.

**Except `arena-dashboard`**, every invocation writes:

1. `benchmarks/results/runs/<run_id>/manifest.json` — machine-readable steps, per-step exit codes, final exit code, timing, argv, versions.
2. `benchmarks/results/runs/<run_id>/RUN.md` — human-readable summary plus links to step logs.
3. `benchmarks/results/runs/<run_id>/logs/*.log` — full stdout/stderr per step.
4. `benchmarks/results/runs/<run_id>/training_trajectory.jsonl` — for training runs, parsed `mlx_lm.lora` points with iteration, train/validation loss, learning rate, throughput, trained tokens, memory, and checkpoint flags when those fields are present in logs. The same summary is embedded in the training step inside `manifest.json`.
5. `docs/run_history.md` — one appended table row (committed) with exit/status, trained adapter, and benchmarked adapter/model so the repo always carries a durable index of runs; copy artifacts off-machine if you need long-term archives (the `runs/` tree is gitignored).

For "which run or adapter should I use now?", start with `docs/RUNS.md`. This workflow doc is the command reference, not the historical run narrative.

Activate the venv first (`source .venv/bin/activate`).

## Subcommands


| Command                                           | Does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python scripts/ml_workflow.py smoke`            | Synthetic JSONL → 4 train iters → benchmark **base** model. Fast CI / sanity check. The smoke adapter is written for wiring validation but is not what the benchmark loads.                                                                                                                                                                                                                              |
| `python scripts/ml_workflow.py prepare`          | `export_repo_for_training.py` + `build_lora_dataset.py` (needs `SOURCE_REPO` + game checkout).                                                                                                                                                                                                                                                                                                           |
| `python scripts/ml_workflow.py train`            | `mlx_lm.lora --train` only. Optional `--evaluate` to run the benchmark after a successful train.                                                                                                                                                                                                                                                                                                         |
| `python scripts/ml_workflow.py benchmark`        | `run_game_benchmark.py` only; optional `--adapter-path`, `--profile` `game` or `general`.                                                                                                                                                                                                                                                                                                                |
| `python scripts/ml_workflow.py evalplus`         | `run_evalplus_benchmark.py` execution-based HumanEval+/MBPP+ subset or full suite.                                                                                                                                                                                                                                                                                                                       |
| `python scripts/ml_workflow.py arena-acceptance` | `run_arena_acceptance_tests.py` + deterministic **Arena Capability Index** from apply/typecheck/export/preview artifacts.                                                                                                                                                                                                                                                                                |
| `python scripts/ml_workflow.py full`             | **prepare → train → benchmark** on the trained adapter; optional `--bench-profile general` for the generic coding suite on the last step.                                                                                                                                                                                                                                                                |
| `python scripts/ml_workflow.py arena-dashboard` | Regenerates `benchmarks/results/arena_dashboard.html` from local `benchmarks/results/runs/*/arena_capability.json` (no run folder, **no `run_history` row** — see **`docs/ARENA_ROADMAP.md`**). |
| `python scripts/ml_workflow.py arena-gate-train` | **Loop:** train on `data/lora/game_task_pairwise` (chat JSONL) then run **headless** `scripts/run_arena_gate_benchmark.py` (`round_final_ok` by default; `--gate preview` starts Next and requires `preview_ready`). Stops when the gate passes or `--max-cycles` is exhausted. Requires existing `train.jsonl` or `--rebuild-dataset` with `benchmarks/results/game_task_pairwise_training_data.jsonl`. |
| `python scripts/ml_workflow.py train --evaluate` | Optional `--bench-profile general` with `--evaluate` to run the **general** suite on the new adapter.                                                                                                                                                                                                                                                                                                    |
| `python scripts/ml_workflow.py adapter-registry` | Write or inspect `training/adapter_registry_v1.json` (taxonomy + promotion metadata + policy versions). |
| `python scripts/ml_workflow.py adapter-datasets` | Build per-adapter datasets with shared anti-overfit corpus and per-adapter manifests. |
| `python scripts/ml_workflow.py loading-screen-dataset` | Build a richer `loading_screen` specialist dataset from pairwise winners + UI transfer rows + shared anchors. |
| `python scripts/ml_workflow.py economy-tooltip-dataset` | Build a richer `economy_tooltip` specialist dataset (core `economy-tooltip` + transfer from loading/HUD samples + shared anchors). |
| `python scripts/ml_workflow.py hud-status-dataset` | Build **HUD specialist** JSONL (`hud_status_specialist`): default **baseline shards** + store/API **guardrails**, **HUD pairwise capped at `--max-core-rows` (default 0)** + substring filter on winners; overrides via `--baseline-shards`, `--no-hud-guardrails`, `--no-filter-pairwise-hud`, `--max-transfer-rows`. |
| `python scripts/ml_workflow.py documentation-dataset` | Build **documentation/run-analysis** JSONL (`documentation_specialist/`): curator draft→canonical-prose pairs for this MLX lab (`ml_workflow`, `PROJECT_STATE`, append-only logs, checkpoints, run manifests, token-routing scope guard); `--min-train-core-rows` upsamples deterministic repeats for small train sets. |
| `python scripts/ml_workflow.py documentation-rag-benchmark` | **Documentation-agent RAG** string tasks (`benchmarks/documentation_agent_rag_tasks_v1.json`, corpus `data/rag/documentation_agent_corpus.json`): loads Qwen2.5-Coder-7B (or `MODEL` / `--model`), optional `ADAPTER_PATH` / `--adapter-path`. Writes per-task JSONL under `benchmarks/results/runs/<run_id>/`, appends **`docs/SPECIALIZED_RUN_HISTORY.md`**, tails **`benchmarks/results/documentation_rag_timeseries.jsonl`**, and a normal **`docs/run_history.md`** row. **Exit code** follows the **with-RAG** pass only; the optional no-RAG baseline uses `--no-fail`. Fast path: `--skip-no-rag-baseline`. |
| `python scripts/ml_workflow.py adapter-gate` | Evaluate adapter promotion gates with avg-gain objective and rollback guards. |
| `python scripts/ml_workflow.py drift-check` | Run adapter/router drift checks and emit automated actions (`warn/freeze/reduce/rollback`). |
| `python scripts/ml_workflow.py control-plane-schedule` | Schedule jobs from worker heartbeat/capability snapshots (multi-Mac scaffold). |
| `python scripts/ml_workflow.py multi-adapter-report` | Publish routing + adapter quality + SLO dashboard artifacts under `benchmarks/results/`. |
| `python scripts/ml_workflow.py routing-prompt-lab` | Capture live or batch prompts, predict adapter + legacy route, and write routing JSONL logs. |
| `python scripts/ml_workflow.py routing-benchmark` | Evaluate route-only or dual-label (adapter + route) routing accuracy with confusion summaries. |
| `python scripts/ml_workflow.py routing-dataset` | Build deterministic train/valid/test routing classifier dataset artifacts + manifest. |


### Examples

```bash
# Quick sanity (no game repo, ~20–30 s typical)
python scripts/ml_workflow.py smoke

# Refresh corpus + LoRA splits only
export SOURCE_REPO=$HOME/fallen-empire
python scripts/ml_workflow.py prepare

# Curated arena “gold” answers → chat SFT JSONL (messages) for mlx_lm.lora --data
python scripts/build_arena_baseline_dataset.py --require-all
# Writes: data/lora/arena_task_baselines/{train,valid,test}.jsonl
# Source answers: data/arena_task_baselines/<task_id>.assistant.txt (see README there)

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
python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest -- --iters 400

# Headless arena gate (local MLX + disposable worktree; needs SOURCE_REPO + GAME_ARENA_ROOT)
python scripts/run_arena_gate_benchmark.py --adapter-path checkpoints/fe-lora-30m --task-id loading-screen-polish

# Broad no-human arena acceptance: compile/export gate + preview-ready pages
python scripts/run_arena_acceptance_tests.py --adapter-path checkpoints/fe-lora-game-text-20260428

# Same, but documented in the workflow run history with Arena Capability Index
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 \
  --task-id loading-screen-polish

# Same with progressive retrieval disabled (baseline vs `auto`); see docs/ARENA_PROGRESSION.md
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 \
  --progressive-context off \
  --task-id loading-screen-polish

# Run the full task catalog (not only core six) for broader coverage
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 \
  --suite all

python scripts/ml_workflow.py arena-dashboard --last 40
# opens benchmarks/results/arena_dashboard.html (from local runs/*/arena_capability.json)

python scripts/arena_promotion_gate.py benchmarks/results/runs/<run_id>/arena_capability.json \
  --min-worst-score 40

# Build pairwise chat SFT focused on apply failures only
python scripts/build_game_task_pairwise_dataset.py \
  --input benchmarks/results/game_task_pairwise_training_data.jsonl \
  --out-dir data/lora/game_task_pairwise_apply_focus \
  --focus-apply-failures \
  --repeat 6

# LoRA train + arena gate loop (pairwise data under data/lora/game_task_pairwise/)
python scripts/ml_workflow.py arena-gate-train --adapter-path checkpoints/fe-lora-arena-gate --data-dir data/lora/game_task_pairwise --max-cycles 15 --iters-per-cycle 80 -- --batch-size 1

# Same loop with in-cycle dataset rebuild filtered to apply failures
python scripts/ml_workflow.py arena-gate-train \
  --adapter-path checkpoints/fe-lora-arena-apply-focus \
  --data-dir data/lora/game_task_pairwise_apply_focus \
  --rebuild-dataset \
  --focus-apply-failures \
  --pairwise-repeat 6 \
  --max-cycles 12 \
  --iters-per-cycle 80 \
  -- --batch-size 1

# Benchmark only (base model)
python scripts/ml_workflow.py benchmark

# Benchmark LoRA
python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest
python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest --profile general

# Execution-based HumanEval+ subset via EvalPlus
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5

# Pick the best non-overfit adapter from recorded artifacts
python scripts/select_best_adapter.py

# Dry-run cost router policy benchmark (no API calls)
python scripts/run_routing_benchmark.py
python scripts/run_routing_benchmark.py --mode both --summary-json benchmarks/results/routing_policy_summary.json
python scripts/ml_workflow.py routing-benchmark --mode both --output-jsonl benchmarks/results/routing_policy_rows.jsonl

# Live prompt routing capture (type your own prompts or pass a file)
python scripts/ml_workflow.py routing-prompt-lab --prompt "Audit save/load auth guardrails" --accepted-for-training
python scripts/ml_workflow.py routing-prompt-lab --prompts-file data/routing/live_batch.txt --source live

# Build supervised routing train/valid/test splits
python scripts/ml_workflow.py routing-dataset \
  --live-jsonl benchmarks/results/routing_prompt_lab/smoke.jsonl \
  --dataset-version routing-v1

# Initialize adapter registry and build per-adapter datasets
python scripts/ml_workflow.py adapter-registry --write-default
python scripts/ml_workflow.py adapter-datasets \
  --sources-dir data/arena_task_baselines \
  --shared-anchor-dir data/lora/shared_general_anchor

# Build loading-screen specialist dataset (core + transfer UI tasks)
python scripts/ml_workflow.py loading-screen-dataset \
  --transfer-task-id hud-status-summary \
  --transfer-task-id economy-tooltip \
  --min-train-core-rows 120

# Build economy-tooltip specialist dataset (pairwise + curated baseline + shallow UI transfers)
python scripts/ml_workflow.py economy-tooltip-dataset \
  --transfer-task-id loading-screen-polish \
  --transfer-task-id hud-status-summary \
  --min-train-core-rows 100

# Evaluate candidate gate + drift and publish report
python scripts/ml_workflow.py adapter-gate \
  --candidate benchmarks/results/candidate_adapter_quality.json \
  --champion benchmarks/results/champion_adapter_quality.json \
  --output benchmarks/results/adapter_gate_result.json
python scripts/ml_workflow.py drift-check \
  --adapter-current benchmarks/results/candidate_adapter_quality.json \
  --adapter-baseline benchmarks/results/champion_adapter_quality.json \
  --router-current benchmarks/results/router_current.json \
  --router-baseline benchmarks/results/router_baseline.json \
  --output benchmarks/results/drift_report.json
python scripts/ml_workflow.py multi-adapter-report \
  --routing-jsonl benchmarks/results/routing_policy_rows.jsonl \
  --gate-json benchmarks/results/adapter_gate_result.json \
  --drift-json benchmarks/results/drift_report.json

# Human visual trial arena for standalone landing pages
python scripts/landing_page_arena.py ui

# Disposable game worktree arena for real feature trials
python scripts/game_task_arena.py ui
```

`evalplus` uses the open-source EvalPlus datasets and may download HumanEval+/MBPP+ from GitHub on first run. If GitHub is unavailable, retry later or set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` to a local EvalPlus JSONL file.

## Benchmark Scoring

`scripts/run_game_benchmark.py` reports two layers:

- **Summary** (`15/15`, `8/8`): legacy substring pass/fail smoke score.
- **Capability Index** (`0–100`): weighted score that combines correctness, instruction following, concision, and speed. This catches adapters that pass substring checks by emitting huge generic answers.

Treat this as a **lexical regression/capability proxy**, not a true game-edit capability score.

`scripts/arena_capability_index.py` and `ml_workflow.py arena-acceptance` report deterministic arena scores from real edit artifacts:

- **Completion:** generated output applied, TypeScript passed, exports were preserved, and preview became ready.
- **Integration:** clean apply/compile/export behavior and fewer repair retries.
- **Efficiency:** elapsed time, token pressure, and retry count.
- **Tiered benchmark label:** low tasks are **Smoke Test**, low-medium/medium are **Standard Dev Benchmark**, and medium-high/high are **High-Reasoning Architecture Benchmark**.
- **Complexity weighting:** harder arena tasks count slightly more based on task metadata.
- **Diversified arena index:** blends task-type-balanced performance, worst-tail robustness, and task-catalog coverage so scores are harder to inflate with a narrow task subset.

Low-only runs are capped as smoke tests so a loading-screen pass cannot masquerade as full arena capability. Use the **Standard Dev Benchmark** and **High-Reasoning Architecture Benchmark** tiers as the primary deterministic signal for “completed meaningful game-dev work,” while keeping the string benchmark beside it as a cheap vocabulary/regression smoke test.

## Cost-Aware Routing

`scripts/model_router.py` keeps backward compatibility (`local`/`frontier`/`hybrid`) while now emitting adapter-aware v1 routing metadata:

- `LocalMlxBackend` wraps local `mlx-lm` inference.
- `OpenAICompatibleBackend` calls `FRONTIER_API_BASE_URL` + `FRONTIER_API_KEY` / `OPENAI_API_KEY` with `FRONTIER_MODEL`.
- `RoutingPolicy` adds `adapter_id`, `execution_tier`, `council_mode`, `confidence`, `ambiguity`, `risk_class`, `complexity`, and `policy_version`.
- `scripts/router/classifier.py` provides deterministic confidence/ambiguity/risk/complexity scoring.
- `scripts/router/policy.py` applies adapter-aware tiering, council modes, and API escalation ladder defaults.

The routing benchmark is policy-only and does not call paid APIs.

For route-only legacy tasks, keep `--mode route` (default). For specialization work, use `--mode both` and provide datasets containing `expected_adapter_id` and `expected_legacy_route` labels.

## Landing Page Trial Arena

Use `scripts/landing_page_arena.py` to test whether local adapters and Cursor/frontier attempts can turn Fallen Empire intent/context into a visually usable standalone website. This does **not** edit the game repo; it writes isolated static-site attempts under `benchmarks/results/landing_page_trials/`.

```bash
# Create/update the shared brief from repo docs
python scripts/landing_page_arena.py brief --overwrite

# Create a trial and a Cursor-ready frontier packet
python scripts/landing_page_arena.py create \
  --task "Create a Fallen Empire landing page with hero, faction cards, simulation systems, screenshots placeholders, and a call-to-action."

# Generate an attempt with the recommended local adapter
python scripts/landing_page_arena.py local-attempt \
  --trial-id <trial_id> \
  --attempt local_300 \
  --adapter-path checkpoints/fe-lora-30m

# Preview an attempt in a browser
python scripts/landing_page_arena.py serve --trial-id <trial_id> --attempt local_300 --port 8091

# Save human ratings after looking at the page
python scripts/landing_page_arena.py rate \
  --trial-id <trial_id> \
  --attempt local_300 \
  --visual-quality 4 \
  --game-fit 4 \
  --copy-quality 4 \
  --responsiveness 4 \
  --interaction-quality 3 \
  --mergeability 3 \
  --cleanup-minutes 20 \
  --notes "Good mood and layout; needs stronger faction section."
```

The Gradio UI (`python scripts/landing_page_arena.py ui`, default **[http://127.0.0.1:7863](http://127.0.0.1:7863)**) is a single IDE-style arena: enter one shared prompt, run split Local/Frontier generation, inspect or edit each lane's `index.html` / `styles.css` / `script.js`, press **Preview Current Code**, then grade and save training data. An advanced paste drawer lets you import Cursor/frontier output into the Frontier lane. Comparison records are written under `benchmarks/results/landing_page_comparisons.jsonl` and `benchmarks/results/landing_page_training_data.jsonl`.

The UI has separate **Local** and **Frontier** lanes. The Local lane calls the MLX adapter directly. The Frontier lane can either call an OpenAI-compatible API or import pasted Cursor/frontier output. Configure API mode before launching the UI:

```bash
export FRONTIER_API_KEY=...
export FRONTIER_MODEL=<your-codex-or-frontier-model-id>
# optional for non-OpenAI providers:
export FRONTIER_API_BASE_URL=https://api.openai.com/v1
python scripts/landing_page_arena.py ui
```

Do not paste API keys into prompts, trial notes, or committed docs.

To save local API settings across launches, create an ignored `.env` file in the repo root:

```bash
cp .env.example .env
# edit .env with your real key/model
python scripts/landing_page_arena.py ui
```

`scripts/landing_page_arena.py` loads `.env` automatically without overwriting variables already exported in your shell.

## Game Task Arena

Use `scripts/game_task_arena.py` when you want models to attempt real game-code tasks in disposable worktrees or copies. This is still an evaluation harness, not a merge bot: it never edits `/Users/natreed/fallen-empire` directly, and cleanup removes only disposable worktrees while preserving indexed result artifacts.

The UI keeps standardized tests front-and-center. `Source repo`, `Worktree root`, and `Base ref` are under advanced settings because they normally do not change. Main flow is a single **Run Full Trial** button: create worktrees, generate/apply both model attempts, start two playable preview links, grade both attempts with task-specific criteria, then complete the trial and optionally delete disposable worktrees. Manual packet/verify/report controls live in an optional details drawer.

Task specs can define `preview_path` next to `preview_command`. The arena starts the game dev server in the disposable worktree, records preview links as `http://127.0.0.1:<port><preview_path>`, and marks the preview `ready` only after that URL returns successfully. Current visual tasks deep-link to game-owned sandbox routes such as `/test-env/loading-screen` and `/test-env/combat-risk-preview`, so opening the preview link immediately shows the standardized test environment instead of requiring manual navigation through the normal game.

The default preview command is `next dev -H 127.0.0.1 -p {port}`. The arena sets `WATCHPACK_POLLING=1`, points `PATH`/`NODE_PATH` at the source checkout's `node_modules`, and symlinks `node_modules` into disposable worktrees when possible so previews do not fail with `next: command not found`. If the requested preview port is already occupied, the arena automatically advances to the next free port and records both `requested_port` and actual `port` in `preview.json`; this prevents stale servers from being mistaken for the current attempt. Generated attempts also write `generation_metrics.json` with elapsed seconds, max tokens, input/output/total token counts, and provider usage when available.

Before starting a preview server, the arena runs `npx tsc --noEmit` as a preview preflight. Invalid model code is marked `preflight_failed`, logged under `logs/preview_preflight_tsc.log`, and no broken HTTP 500 preview server is launched for that attempt.

`game_task_arena.py generate` supports `**--progressive-context auto|on|off`**. In `auto`, non-low-complexity tasks first ask the model which exact files it needs, then append bounded targeted snippets to the final edit prompt. The request and selected files are logged under `attempts/<attempt>/logs/progressive_context.*`. It also supports `**--tsc-retries N**`: after each generation the arena applies to the disposable worktree, runs the same `tsc` check, and on failure runs `git reset --hard HEAD` then regenerates with the compiler log appended to the user prompt (per-round logs under `logs/tsc_round_<n>.log`; the latest round is also copied to `logs/preview_preflight_tsc.log`). For MLX, `**--no-extra-eos**` disables the default Qwen extra EOS id (`151645` / `<|im_end|>`); override with `**GAME_TASK_ARENA_EXTRA_EOS_IDS**` (comma-separated ints, or empty to clear). Optional `**--stop-string TEXT**` (repeatable) or `**GAME_TASK_ARENA_STOP_STRINGS**` (`||`-separated substrings, `\n` for newline) stop decoding once the streamed text contains a match. The Gradio **Generate + Apply** flow enables **one** local `tsc` retry by default for the local lane only.

Unless `**GAME_TASK_ARENA_SKIP_EXPORT_GUARD`** is set to a truthy value, `generate` also compares baseline exported identifiers (from `git ls-files` at round start) against files in `git diff --name-only HEAD`; if a touched file drops a previously exported name, the round fails like a typecheck error and the retry prompt includes **Export preservation** (`generation_metrics.json` records `exports_ok` / `missing_exports` per round).

Attempts with `apply_status` that does not start with `applied` or `wrote` are not previewed when `--start` is used; they are marked `skipped_apply_status:<status>`. This prevents unchanged baseline worktrees from appearing as successful model previews.

For model-edit reliability, task context paths should name the exact `src/app` and `src/components` files rendered by the sandbox. The loading-screen task explicitly includes `/test-env/loading-screen`, `src/components/test/TestEnvironmentShell.tsx`, and `src/components/ui/GameLoadingScreen.tsx`; the generation prompt prefers fenced full-file blocks for small UI edits because partial diff hunks are fragile.

All standardized task prompts should name their `/test-env/...` route, point to likely rendered files/helpers, and require the model to infer schema and visual language from code. Keep the expected Fallen Empire style native to the existing app: dark medieval strategy surfaces, parchment copy, empire-gold accents, restrained danger colors, compact tactical wording, and real TypeScript/Zustand/Next data shapes.

When one side applies and the other fails apply, the arena writes `pairwise_training_signal.json` for that trial and appends `benchmarks/results/game_task_pairwise_training_data.jsonl`. These records preserve the same task context, the applyable output, the rejected output/error, token metrics, and diff artifacts for later preference/SFT data shaping.

Manual grading captures both preference and viability. Use `Winner` plus `Preference strength` for the pairwise label; use failure-mode checkboxes to tag why an attempt lost (`parse/apply failed`, `typecheck/preflight failed`, `no visible change`, `generic/off-theme`, etc.). The arena also stores automated viability (`applied`, `verified`, `preview_ready`) and optional manual confirmations for typecheck/visible change in each `rating.json`.

To train on those records, build chat SFT splits first:

```bash
python scripts/build_game_task_pairwise_dataset.py --repeat 4
python scripts/ml_workflow.py train \
  --adapter-path checkpoints/fe-lora-pairwise-smoke \
  -- --data data/lora/game_task_pairwise --iters 30 --batch-size 1 \
     --val-batches 1 --steps-per-eval 10 --steps-per-report 5 \
     --max-seq-length 2048 --save-every 30
```

This is intentionally a small overfit/format-training pass unless the pairwise corpus has grown substantially.

Fenced file blocks may provide the repo-relative path in the fence info (````tsx path=...`), as a bare info-string path, as the first body comment (`// src/components/...`or`// path: ...`), as a markdown heading immediately before the fence, or as a separate path-marker fence followed by a code fence. The parser resolves` .tsx`before`.ts`and strips path comments before writing, so model outputs do not accidentally create sibling`.ts`files for React components or write path markers as code. Run`python scripts/test_game_task_arena_parser.py` after parser changes.

Local MLX attempts use a smaller context window and are instructed to return fenced full-file blocks, not diffs. The local model has repeatedly failed at partial diff hunk generation on UI tasks; file-block output better isolates whether it understands the component/schema from whether it can emit a valid patch.

Context selection prioritizes exact task files before broad glob matches and README. This matters for local runs with smaller context windows: route/component/helper files should appear early enough for the model to see the real schema before generic project docs.

```bash
# Create local + frontier worktrees from a versioned task spec
python scripts/game_task_arena.py create --task-id loading-screen-polish

# Generate a model/Cursor packet for an attempt
python scripts/game_task_arena.py packet --trial-id <trial_id> --attempt local

# Apply model output as a unified diff or fenced repo-relative file blocks
python scripts/game_task_arena.py apply --trial-id <trial_id> --attempt local --input /path/to/model_output.md

# Run fixed verification commands from the task spec
python scripts/game_task_arena.py verify --trial-id <trial_id> --attempt local

# Record or start a preview URL for playtesting.
# If the task has preview_path, the recorded URL opens that sandbox route.
python scripts/game_task_arena.py preview --trial-id <trial_id> --attempt local --port 5174

# Save human grades and training data
python scripts/game_task_arena.py grade --trial-id <trial_id> --attempt local --correctness 4 --winner local

# Generate the comparison/history report
python scripts/game_task_arena.py report

# Remove disposable worktree while keeping manifests/logs/diffs/ratings
python scripts/game_task_arena.py cleanup --trial-id <trial_id> --attempt local
```

Task specs live in `benchmarks/game_task_arena_examples.json`. Results are filed under `benchmarks/results/game_task_trials/<trial_id>/`, indexed in `benchmarks/results/game_task_index.jsonl`, and summarized in `benchmarks/results/game_task_reports/game_task_summary.md`.

## Exit codes

The workflow exits with the **last failing step’s** exit code (or **0** if everything succeeded). CI should treat non-zero as failure. Manifests include `final_exit_code`, `started_at`, `finished_at`, total `elapsed_s`, `trained_adapter_path`, and `benchmark_adapter_path`; older manifests remain readable by the dashboard.

## Results visualizer

Generate a static dashboard from `benchmarks/results/runs/*/manifest.json` plus `docs/run_history.md`:

```bash
source .venv/bin/activate
python scripts/visualize_results.py
# writes benchmarks/results/run_dashboard.html
```

Use `--out` for a custom file and `--open` to launch it in your browser immediately.

## LoRA Vector Trajectories

Use `scripts/analyze_lora_vector_trajectory.py` when a training run saved intermediate adapter checkpoints and you want to see how the LoRA weights moved as vectors:

```bash
source .venv/bin/activate
python scripts/analyze_lora_vector_trajectory.py \
  --adapter-path checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m \
  --reference-adapter-file checkpoints/fe-lora-30m/adapters.safetensors
```

The analyzer writes `vector_trajectory.jsonl`, `layer_trajectory.jsonl`, `manifest.json`, and `SUMMARY.md` under `benchmarks/results/lora_vector_trajectories/<adapter-name>/`. Whole-adapter rows include vector norm, RMS, delta from the reference adapter, delta from the previous checkpoint, and cosine similarity. Layer rows support heatmaps for where the adapter moved most.

## Training UI (Gradio)

Browser panel to **export + build** and **run `mlx_lm.lora`** with streamed logs (default **[http://127.0.0.1:7862](http://127.0.0.1:7862)**):

```bash
source .venv/bin/activate
python scripts/train_ui_gradio.py
```

This does **not** append `docs/run_history.md` by itself; use `ml_workflow.py` when you want that audit trail.

## Dual RAG lanes for docs + run analysis

Documentation RAG (repo practices/policies):

```bash
python scripts/run_documentation_agent_benchmark.py --use-rag
```

Run-analysis RAG (manifests/history/run summaries):

```bash
python scripts/build_run_analysis_rag_corpus.py
python scripts/run_run_analysis_agent_benchmark.py --use-rag
```

Edit-triggered training prep hook:

- `.cursor/hooks/lab_hook_after_fileedit_training_trigger.py`
- calls `scripts/trigger_doc_training_on_changes.py` on watched edits
- refreshes run-analysis corpus and (cooldown-gated) runs `python scripts/ml_workflow.py documentation-dataset`
- queue log: `data/training_triggers/documentation_training_queue.jsonl`

Useful env controls:

- `FE_LAB_DOC_TRIGGER_MIN_SECONDS` (default `900`)
- `FE_LAB_AUTODOC_DATASET_ON_CHANGE` (`1`/`0`)

## Related docs

- `docs/RUNS.md` — current-vs-historical run interpretation and adapter recommendations.
- `docs/DATA_LAYOUT.md` — canonical data, adapter, and lineage paths.
- `docs/CHUNKED_GAME_TEXT.md` — details for chunked long-file `game_text` rows.
- `training/README.md` — LoRA training concepts and config entry points.
- `benchmarks/README.md` — benchmark suite concepts, human eval UI, game-side tests.
- `docs/PROJECT_STATE.md` — pinned versions, current defaults, and known issues.
