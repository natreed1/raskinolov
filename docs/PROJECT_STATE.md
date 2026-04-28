# fallen-empire-lora — project state

Last verified: **2026-04-27** (Apple Silicon macOS; workspace path `/Users/natreed/fallen-empire-lora`). Latest audited work added a cautious mixed `game_text` + pairwise transcript LoRA run. EvalPlus data is now locally available for the first HumanEval subset tasks, but an EvalPlus run can still fail with Metal OOM when other MLX/UI processes are active.

## Purpose

Experiment with **LoRA fine-tuning** on open code models using the Fallen Empire game repository as training text, without bloating the main game repo. See repository `README.md` for rationale.

## Environment


| Item                        | Value                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| OS                          | darwin 24.x (from prior session metadata)                                                        |
| Python baseline             | **3.11+ recommended for new environments**                                                       |
| Current verified local venv | **3.9.6** (`/Users/natreed/fallen-empire-lora/.venv`; legacy working snapshot)                   |
| Reproducibility constraints | `requirements.lock.txt` records exact top-level ML/UI package pins from the verified environment |


### Pinned Python packages (venv)

Recorded via `pip freeze` on 2026-04-23; `evalplus==0.3.1` added on 2026-04-26:

```
mlx==0.29.3
mlx-lm==0.29.1
mlx-metal==0.29.3
huggingface_hub==0.36.2
transformers==4.57.6
numpy==2.0.2
safetensors==0.7.0
gradio==4.44.1
gradio_client==1.3.0
evalplus==0.3.1
rank-bm25==0.2.2
```

Install source: `requirements.txt`; resolver pins include `gradio>=4.44,<5` for the chat UI. For the recorded environment constraints, run `pip install -r requirements.txt -c requirements.lock.txt`.

## Default base model (smoke / planned LoRA base)


| Field           | Value                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| Hugging Face id | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`                                                       |
| Role            | Small instruct **code** model, MLX 4-bit, suitable for first downloads and iteration on unified memory |


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

Defaults: bind **[http://127.0.0.1:7860](http://127.0.0.1:7860)**, same base model as smoke script, `concurrency_limit=1` (one generation at a time). Uses `stream_generate` and yields cumulative text for token streaming. Gradio title and default system prompt name the assistant **Albert**.

Useful flags: `--port`, `--model`, `--adapter-path` (LoRA dir when you have one), `--max-tokens`, `--temp`, `--share` (temporary public Gradio URL). Environment mirrors: `MODEL`, `ADAPTER_PATH`, `SYSTEM_PROMPT`, `MAX_TOKENS`, `TEMP`.

**Outcome (2026-04-23):** `--help` and import-time test for `_history_to_messages` succeeded; full browser run not re-logged here (same model stack as smoke).

## Game-oriented model benchmark + evolution


| Item                           | Location                                                                                                                          |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Task definitions (game)        | `benchmarks/fallen_empire_tasks.json` (15 tasks; domains include evolution/serialization)                                         |
| Task definitions (general)     | `benchmarks/general_coding_tasks.json` (8 tasks; HTTP/SQL/encoding/doc/version trivia; use `--profile general`)                   |
| Execution benchmark            | `scripts/run_evalplus_benchmark.py` (EvalPlus HumanEval+/MBPP+; use `ml_workflow.py evalplus`)                                    |
| Routing benchmark              | `benchmarks/task_routing_tasks.json` + `scripts/run_routing_benchmark.py` (local/frontier/hybrid policy dry-run)                  |
| Visual human arena             | `scripts/landing_page_arena.py` + `benchmarks/landing_page_brief.md` (standalone landing-page trials; no game repo edits)         |
| Game task arena                | `scripts/game_task_arena.py` + `benchmarks/game_task_arena_examples.json` (disposable worktree/copy trials against the game repo) |
| Shared tier / mutation helpers | `scripts/benchmark_evolution_lib.py`                                                                                              |
| Runner                         | `scripts/run_game_benchmark.py` (`--profile game                                                                                  |
| Season evolution               | `scripts/evolve_benchmark_seasons.py` + `training/evolution_config.json`                                                          |
| Training notes                 | `training/README.md`                                                                                                              |
| Notes                          | `benchmarks/README.md`                                                                                                            |


Scoring is heuristic (substring rules), intended to regress **base vs LoRA** and prompt changes—not to replace game unit tests.

Game task arena task specs now support `preview_path`. Preview URLs are composed as `http://127.0.0.1:<port><preview_path>` and previews are marked `ready` only after the URL responds. The current standardized visual tasks point at game-owned `/test-env/...` sandbox routes in `/Users/natreed/fallen-empire`, including `/test-env/loading-screen` and `/test-env/combat-risk-preview`, so human review opens directly on deterministic test screens.

Preview startup uses `next dev -H 127.0.0.1 -p {port}` rather than the package `npm run dev` wrapper. This avoids the old disposable-worktree failure where `npm run dev -- --host ... --port ...` resolved to `next: command not found`; the arena now reuses source checkout dependencies through `PATH`/`NODE_PATH` and a best-effort `node_modules` symlink. If a requested preview port is occupied, the arena advances to the next free port and records both requested and actual ports so stale servers are not marked as ready for new attempts. Generation artifacts include `generation_metrics.json` plus manifest fields for elapsed seconds and token counts.

Loading-screen task context is explicitly tied to the sandbox route: `/test-env/loading-screen` renders `src/components/test/TestEnvironmentShell.tsx` and `src/components/ui/GameLoadingScreen.tsx`. Keep exact `src/app` / `src/components` paths in task specs; root-only `app/**/`* or `components/**/`* globs are insufficient for this game checkout.

All `benchmarks/game_task_arena_examples.json` prompts now include route-specific sandbox context plus explicit instructions to infer schema/vibe from supplied code. The expected style is the existing Fallen Empire language: dark medieval strategy panels, parchment text, empire-gold accents, restrained red/amber danger states, compact tactical wording, and real `src/store/useGameStore.ts` / `src/types/game.ts` / `src/lib` types rather than invented data shapes.

Pairwise apply outcomes are now training-data candidates. If one model attempt applies and the other does not, `scripts/game_task_arena.py` writes `pairwise_training_signal.json` under the trial directory and appends `benchmarks/results/game_task_pairwise_training_data.jsonl` with winner/loser outputs, apply metadata, errors, token metrics, and the winning diff.

`scripts/build_game_task_pairwise_dataset.py` converts those pairwise records into chat SFT splits under `data/lora/game_task_pairwise/`. First smoke-sized arena-results training run: `20260427-042822_4168da`, command `ml_workflow.py train --adapter-path checkpoints/fe-lora-pairwise-smoke -- --data data/lora/game_task_pairwise --iters 30 ...`, exit **0**, adapter at `checkpoints/fe-lora-pairwise-smoke/`. Treat it as an overfit/format-training smoke adapter until benchmarked in the arena.

Follow-up evaluation: `checkpoints/fe-lora-pairwise-smoke` scored **15/15 game** but **7/8 general**, so it shows overfit/regression risk. `checkpoints/fe-lora-pairwise-r10-from-30m` resumed from `fe-lora-30m` for 10 low-LR iters on non-repeated pairwise data and scored **15/15 game** / **8/8 general**, but a local-only arena smoke still produced non-applyable declaration stubs. Do not promote either pairwise adapter as default without a successful arena edit preview.

Benchmark scoring now includes a **Capability Index** in addition to legacy pass/fail. It combines correctness, instruction following, concision, and speed. First observed example: `checkpoints/fe-lora-pairwise-r10-from-30m` on general benchmark scored **8/8** but only **83.7/100** capability because concision was **11.9/100** (run `20260427-220341_d13b1d`). Use this to flag over-verbose adapters that pass substring checks.

```bash
source .venv/bin/activate
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --profile general
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5
python scripts/run_routing_benchmark.py
python scripts/landing_page_arena.py ui
python scripts/game_task_arena.py ui
python scripts/run_game_benchmark.py --tier A
python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json
```

**Outcome (2026-04-24):** On `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` with greedy `temp=0` and default `--max-tokens` 512: **game 15/15** and **general 8/8** (no tier). (2026-04-23) Evolved task files are experimental; cap `min_chars` at **320** after mutation/tier scaling to stay decodable.

### Visual arena parser / UI notes

As of 2026-04-26, `scripts/landing_page_arena.py ui` presents a single IDE-style workflow: one build prompt fans out to Local and Frontier lanes, generated code appears in editable `index.html` / `styles.css` / `script.js` panes, previews render side by side in sandboxed iframes, and grading writes paired comparison/training rows. Parser behavior is intentionally strict to safe static filenames but tolerant of common model response formats (`path=`, `filename=`, language-only fences, bare fenced filenames, labeled sections, and raw HTML). Unparsed output is preserved in `model_output.md` / `parse_error.html` without replacing `index.html`.

## Dataset export + hygiene

- Script: `scripts/export_repo_for_training.py`
- Default source: `~/fallen-empire` unless `SOURCE_REPO` is set
- Output: `data/raw/repo_text.jsonl` (gitignored under `data/raw/`)
- **Hygiene:** max file size **400 KB** (override with `EXPORT_MAX_FILE_BYTES`), extra skip dirs (`.turbo`, `credentials`, …), skip credential-like suffixes (`.pem`, `.key`, …), path substring rules (`secret`, `/.env`, …), NUL / control-char heuristic for “binary-ish” text, and **regex redaction** for common API keys / PEM blocks / obvious GitHub+Slack token shapes.

## Human evaluation UI

- Script: `scripts/human_eval_ui.py` (default **[http://127.0.0.1:7861](http://127.0.0.1:7861)**)
- Appends JSONL rows to `benchmarks/results/human_eval.jsonl` (gitignored under `benchmarks/results/`).

## Landing page trial arena

- Script: `scripts/landing_page_arena.py` (Gradio default **[http://127.0.0.1:7863](http://127.0.0.1:7863)**)
- Shared context: `benchmarks/landing_page_brief.md`
- Trial artifacts: `benchmarks/results/landing_page_trials/<trial_id>/` plus JSONL indexes under `benchmarks/results/`
- Purpose: compare local LoRA/base/frontier attempts by viewing standalone static pages in a browser, then record visual/product ratings. This is intentionally separate from the game repo and does not mutate `/Users/natreed/fallen-empire`.
- UI: single IDE-style arena. One prompt splits into Local and Frontier attempts, shows editable `index.html` / `styles.css` / `script.js` panes for both, previews current code, then saves grades/training data. **Local** calls `LocalMlxBackend`; **Frontier** can call an OpenAI-compatible API via `FRONTIER_API_KEY` / `OPENAI_API_KEY`, `FRONTIER_MODEL`, and optional `FRONTIER_API_BASE_URL`, or import pasted Cursor/frontier output.
- Token budget: arena Local + Frontier split defaults to **8192** max tokens. If either lane still truncates static-site output, escalate deliberately to **12000**, then **16000** only for larger landing-page trials.
- Local API config: `scripts/landing_page_arena.py` loads ignored repo-root `.env` files; `.env.example` documents keys. Do not commit `.env`.
- Training data: comparison records append to `benchmarks/results/landing_page_comparisons.jsonl` and `benchmarks/results/landing_page_training_data.jsonl`, including task, brief, winner, scores, notes, and generated files.
- Recommended local adapter for first trials: `checkpoints/fe-lora-30m`.

## Training UI (Gradio)

- Script: `scripts/train_ui_gradio.py` (default **[http://127.0.0.1:7862](http://127.0.0.1:7862)**)
- Convenience panel for export + dataset build + `mlx_lm.lora` with live logs; for committed run documentation use `scripts/ml_workflow.py` (see `docs/WORKFLOW.md`).

## Game-side ML cohort tests

- **Game repo:** `scripts/ml-lora-cohort-guard.ts` — `npm run test:ml-cohort`
- **ML repo wrapper:** `scripts/run_game_ml_tests.sh` (honours `SOURCE_REPO`, default `~/fallen-empire`)

## Game task arena

- Script: `scripts/game_task_arena.py` (Gradio default **[http://127.0.0.1:7868](http://127.0.0.1:7868)**)
- Task specs: `benchmarks/game_task_arena_examples.json`
- Standardized task coverage: low-complexity loading/start screen polish, visible HUD/status UI, economy tooltip, combat risk preview, save/load/API guard, and AI planning rationale. Tasks are designed to span UI/backend/combat/economics/planning and prefer changes that are easy to locate quickly in a preview.
- Grading: each standardized task defines five task-specific grading criteria in `benchmarks/game_task_arena_examples.json`; the UI updates local/frontier slider labels when the selected task changes and saved ratings include `rubric_labels` / `rubric_scores`.
- Default source repo: `SOURCE_REPO` or `/Users/natreed/fallen-empire`
- Default disposable tree root: `GAME_ARENA_ROOT` or `~/fallen-empire-arena`
- Results: `benchmarks/results/game_task_trials/<trial_id>/`, append-only `benchmarks/results/game_task_index.jsonl`, training rows in `benchmarks/results/game_task_training_data.jsonl`, summaries under `benchmarks/results/game_task_reports/`
- Purpose: evaluate local/frontier attempts on real game-code tasks in disposable worktrees/copies, run fixed verification commands, record diffs/logs/previews/ratings, compare adapters over time, and clean up worktrees without deleting indexed artifacts.
- UI: standardized task selector is the primary control; `Source repo`, `Worktree root`, and `Base ref` are advanced settings. Main flow creates worktrees, generates/applies both model attempts, starts playable Local/Frontier preview links, records grades, and optionally deletes disposable worktrees. Packet text, manifests, verification output, and reports are hidden in a details drawer.
- Safety: never edit the main game checkout; never run model-suggested shell commands; only run task-spec verification commands; reject file writes outside allowed path globs.

## Known issues / API notes

1. **urllib3 / LibreSSL:** At import time, urllib3 may warn that LibreSSL is older than OpenSSL 1.1.1; training/inference still proceeded.
2. `**mlx-lm` 0.29.1 generation:** `generate()` forwards kwargs to `generate_step()`, which does **not** accept `temp=`. Use `sampler=make_sampler(temp=..., top_p=1.0)` for non-zero temperature; `temp=0` omits sampler (greedy argmax). Implemented in `scripts/smoke_base_model.py` and `scripts/chat_gradio.py`.
3. **EvalPlus dataset/runtime:** First `ml_workflow.py evalplus` verification attempts on 2026-04-26 failed before model load because GitHub returned HTTP 502 for `HumanEvalPlus.jsonl.gz`. On 2026-04-27, HumanEval subset data loaded from the local EvalPlus cache and `fe-lora-pairwise-r10-from-30m` passed HumanEval/0 and HumanEval/1 before Metal OOM (`20260427-045347_05350b`). Retry with fewer concurrent MLX/Gradio processes or lower generation pressure before treating EvalPlus as blocked by model quality.

## Training / LoRA (wired)


| Piece                        | Location                                                                                                                                                                                                                                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Orchestrator (preferred)** | `scripts/ml_workflow.py` — re-execs into `.venv/bin/python` when available; per-run `manifest.json`, `RUN.md`, `logs/`, and training-run `training_trajectory.jsonl` under `benchmarks/results/runs/<id>/`; appends `**docs/run_history.md`** with exit/status and trained vs benchmarked adapter metadata |
| **Training UI**              | `scripts/train_ui_gradio.py` — browser panel (default port **7862**); no automatic `run_history` append                                                                                                                                                                                                    |
| Results visualizer           | `scripts/visualize_results.py` — static HTML dashboard from run manifests + `docs/run_history.md`                                                                                                                                                                                                          |
| LoRA vector analyzer         | `scripts/analyze_lora_vector_trajectory.py` — reads saved adapter safetensors checkpoints and writes whole-adapter / per-layer movement JSONL                                                                                                                                                              |
| Adapter selector             | `scripts/select_best_adapter.py` → `benchmarks/results/adapter_selection.md`; current recommendation: `checkpoints/fe-lora-30m`                                                                                                                                                                            |
| Cost router                  | `scripts/model_router.py` — local MLX vs OpenAI-compatible frontier API routing (`local`, `frontier`, `hybrid`)                                                                                                                                                                                            |
| Visual trial arena           | `scripts/landing_page_arena.py` — standalone landing-page trials, Cursor packets, local attempts, previews, ratings                                                                                                                                                                                        |
| Workflow doc                 | `docs/WORKFLOW.md`                                                                                                                                                                                                                                                                                         |
| Run index (committed)        | `docs/run_history.md`                                                                                                                                                                                                                                                                                      |
| Dataset builder              | `scripts/build_lora_dataset.py` → `data/lora/game_text/{train,valid,test}.jsonl` (`data/lora/` gitignored)                                                                                                                                                                                                 |
| Default LoRA YAML            | `training/lora_qwen_coder.yaml` (Qwen2.5-Coder-1.5B-Instruct-4bit, adamw, rank 16, `num_layers: -1`, grad checkpoint)                                                                                                                                                                                      |
| Wrapper (legacy)             | `scripts/train_lora.sh`                                                                                                                                                                                                                                                                                    |


```bash
source .venv/bin/activate
python scripts/ml_workflow.py smoke
export SOURCE_REPO=$HOME/fallen-empire
python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-latest -- --iters 400
python scripts/visualize_results.py
python scripts/select_best_adapter.py
# lower-level equivalent:
python scripts/export_repo_for_training.py
python scripts/build_lora_dataset.py --out-dir data/lora/game_text
mlx_lm.lora --train -c training/lora_qwen_coder.yaml
# or: ./scripts/train_lora.sh
```

**Outcome (2026-04-23):** `mlx_lm.lora --train` with synthetic JSONL and `--iters 2` completed; wrote `checkpoints/_smoke_lora/adapters.safetensors` (smoke path; normal default in YAML is `checkpoints/fe-lora-latest`).

**Outcome (2026-04-25):** `checkpoints/fe-lora-30m` trained for **300** iterations with `--batch-size 1 --val-batches 1`; training completed in ~19.75 min, final train loss **0.799**, validation loss **0.964**, peak memory **8.598 GB**. Follow-up benchmark run with the venv Python scored **15/15** on the game profile. Artifacts: `benchmarks/results/runs/20260425-202718_459eb4/` (train) and `benchmarks/results/runs/20260425-204711_1efcd0/` (benchmark).

**Outcome (2026-04-26):** `checkpoints/fe-lora-800-from-30m` resumed from `checkpoints/fe-lora-30m/adapters.safetensors` for **800** more iterations with `--batch-size 1 --val-batches 8`; training completed in ~52.97 min, final train loss **0.178**, validation loss **1.943**, best validation reading **1.400** at iteration 150, peak memory **8.598 GB**. Automatic **general** benchmark scored **8/8**; follow-up **game** benchmark scored **14/15**, failing `siege-wall-priority-chain` on missing `wallBuildPriority`. Artifacts: `benchmarks/results/runs/20260426-175720_fb7d5b/` (train + general benchmark) and `benchmarks/results/runs/20260426-185136_24ec8d/` (game benchmark).

**Outcome (2026-04-27):** `data/lora/game_text_pairwise_cautious_text/` mixes all current `game_text` rows with modest pairwise transcript reinforcement (train **153** rows: **141** game text + **3** pairwise rows repeated **4x**; valid **9**; test **8**). `checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m` resumed from `checkpoints/fe-lora-30m/adapters.safetensors` for **160** low-LR iterations (`--learning-rate 3e-6`, `--max-seq-length 2048`, `--val-batches 4`); training completed in **524.81s**, final train loss **0.539**, final validation loss **1.918**, best observed validation loss **0.983** at iter 100, peak memory **4.887 GB**. Benchmarks: **13/15 game** (failed `skirmish-signature` / `axial-hex-distance`) and **8/8 general**. Treat as a negative/diagnostic adapter, not a promotion candidate. Artifacts: `benchmarks/results/runs/20260427-045439_18486b/` (train), `benchmarks/results/runs/20260427-050332_bae6a4/` (game benchmark), `benchmarks/results/runs/20260427-050511_59d1ce/` (general benchmark). Failed audited attempts before the successful run: `20260427-045341_7dff75` (mixed `messages`/`text` schema) and `20260427-045410_759dc8` (Metal OOM at 4096 context).

**Workflow metadata note (2026-04-25):** New `ml_workflow.py` manifests include true workflow `started_at`, `finished_at`, total `elapsed_s`, and `final_exit_code`; `RUN.md` mirrors those fields. Step logs write shell-quoted commands for reproducibility. `scripts/visualize_results.py` reads both old and new manifest shapes and uses `final_exit_code` when available for dashboard status.

**Workflow metadata note (2026-04-26):** New `ml_workflow.py` runs also include `trained_adapter_path` and `benchmark_adapter_path`, and `docs/run_history.md` rows now include `Exit` / `Status`. This makes failed runs and smoke runs clearer: `smoke` trains `checkpoints/_workflow_smoke` for wiring validation but benchmarks the base model.

**Workflow trajectory note (2026-04-27):** Training steps now parse `mlx_lm.lora` logs into `training_trajectory.jsonl` and embed a `training_trajectory` summary in the step manifest. The first dynamics probe, `checkpoints/fe-lora-dynamics-probe-30-from-30m`, resumed from `checkpoints/fe-lora-30m/adapters.safetensors` for **30** low-LR iterations on `data/lora/game_text_pairwise_cautious_text` and produced **5** trajectory points (`iter 0`, initial validation, and iters 10/20/30). Final train loss **0.953**, final validation loss **2.158**, best validation loss **0.649** at iter 20, peak memory **4.814 GB**, trained tokens **39,295**. Outcome labels: **15/15 game** and **8/8 general**. Artifacts: `benchmarks/results/runs/20260427-051140_4baad5/` (train trajectory), `benchmarks/results/runs/20260427-051326_55ab75/` (game), `benchmarks/results/runs/20260427-051502_fc30f6/` (general).

**LoRA vector trajectory note (2026-04-27):** `scripts/analyze_lora_vector_trajectory.py` compares numbered `*_adapters.safetensors` checkpoints as ordered LoRA vectors. First run on `checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m` against reference `checkpoints/fe-lora-30m/adapters.safetensors` analyzed **4** checkpoints, **392** tensors, and **18,464,768** LoRA parameters. Whole-adapter movement from reference grew from **0.386487** at iter 40 to **0.712277** at iter 160 while cosine to reference stayed high (**0.999930 → 0.999763**). Outputs: `benchmarks/results/lora_vector_trajectories/fe-lora-mixed-cautious-text-160s2k-from-30m/`.

## Git / large artifacts

`.gitignore` excludes `.venv/`, `models/`, `checkpoints/`, `data/raw/`, `data/lora/`, `benchmarks/results/`, common weight extensions. Do not commit weights or raw exports.