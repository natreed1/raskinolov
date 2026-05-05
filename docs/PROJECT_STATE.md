# fallen-empire-lora — project state

Last verified: **2026-05-04** (Apple Silicon macOS; workspace path `/Users/natreed/fallen-empire-lora`). Current defaults target the **Qwen2.5-Coder-7B** MLX lineage. See `docs/RUNS.md` for current-vs-historical run interpretation and `docs/run_history.md` for the complete append-only run table. **Auxiliary** documentation-agent RAG evals and other cross-cutting benchmarks are indexed in **`docs/SPECIALIZED_RUN_HISTORY.md`** (with **`benchmarks/results/documentation_rag_timeseries.jsonl`** for trend JSONL).

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
gradio==4.44.1
gradio_client==1.3.0
evalplus==0.3.1
```

Install source: `requirements.txt`; resolver pins include `gradio>=4.44,<5` for the chat UI. For the recorded environment constraints, run `pip install -r requirements.txt -c requirements.lock.txt`.

## Default base model (smoke / LoRA / arena local base)


| Field           | Value                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Hugging Face id | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`                                                                            |
| Role            | Default instruct **code** model, MLX 4-bit, used for current LoRA training, benchmarks, chat UI, and arena local attempts |


Smaller alternatives (same org): `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` (set `MODEL` env var for scripts or pass `--model`, and use a matching adapter for that base).

## Current Run Status


| Item                                                | Current interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current arena adapter default                       | `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` via `scripts/fe_lineage.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Latest deterministic arena gate                     | `20260428-160917_fd9233` scored `arena 1/1` against the current arena adapter                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Arena six-task A/B (progressive) chunk6k            | `20260428-171007_517f3e` (`auto`) vs `20260428-180627_42befd` (`off`): both **1/6** preview passes; ACI **53.4** vs **56.3**. See `**docs/ARENA_PROGRESSION.md`**                                                                                                                                                                                                                                                                                                                                                                                                 |
| Arena six-task A/B (progressive) best-val300        | `20260429-025429_9793ff` (`auto`, **1/6**, ACI **55.2**) vs `20260429-034932_0812a4` (`off`, **0/6**, ACI **47.8**). Adapter `checkpoints/fe-lora-qwen25-coder-7b-best-val300` = iter-**300** weights from `20260428-195934_d451c9`. Off run used worktree `benchmarks/results/arena_worktrees`. Failed quick retry: `20260429-034918_34e0cb` (`PermissionError` on `~/fallen-empire-arena`).                                                                                                                                                                     |
| Latest apply-contract + SFT six-task run            | `20260430-183012_121be1` (`auto`, **3/6**, ACI **68.1**) on `checkpoints/fe-lora-arena-apply-sft`; tiers **100.0 / 35.9 / 78.1** (smoke/std/high). Working attribution: improvement likely comes from combined apply-contract context injection + targeted arena baseline SFT; still below promotion thresholds due to weak worst-task tail.                                                                                                                                                                                                                      |
| Latest reverse-recovery six-task run                | `20260430-231450_487deb` (`auto`, **3/6**, ACI **75.0**) on `checkpoints/fe-lora-arena-reverse-recover` after retraining from the overfit standard-dev adapter on six-task apply-contract baselines. Tiers **100.0 / 45.1 / 85.4**; standard-dev remains the active bottleneck (`hud-status-summary`, `economy-tooltip`).                                                                                                                                                                                                                                         |
| Latest repair-set SFT + six-task check              | Trained `checkpoints/fe-lora-arena-apply-repair-stddev-20260502` on combined repair-set corpus `data/lora/arena_apply_repair_stddev_20260502` (`20260502-190626_aa03bc`, exit 0), then ran six-task acceptance `20260502-192506_b1c69f` (`auto`) and got **0/6**, ACI **52.54**. Applyability improved on most tasks, but final `tsc` remained the gating failure mode.                                                                                                                                                                                           |
| Latest mixed-curriculum + repairs six-task check    | Trained `checkpoints/fe-lora-arena-mixed-repair-20260502` on `data/lora/arena_balanced_with_repairs_20260502` (broad curriculum + boosted HUD/economy repair rows) via `20260502-201848_c45345` (exit 0), then ran six-task acceptance `20260502-211555_ce7aa2` (`auto`) and got **1/6**, ACI **53.33**. `combat-risk-preview` passed; standard-dev still failed with `tsc`/export instability.                                                                                                                                                                   |
| Latest compile-supervision checkpoint sweep + core6 | Trained `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502` on `data/lora/arena_balanced_with_repairs_compile_20260502` (broad+repair corpus plus explicit compile/export repair rows) via `20260502-220731_da066d` (exit 0), then gated checkpoints every 20 iters: `ckpt20` (`std-dev 0/2`, `guard 2/4`), `ckpt40` (`0/2`, `0/4`), `ckpt60` (`0/2`, `0/4`). Core6 confirmation on `ckpt20` (`20260502-234543_08b06c`) reached **2/6**, ACI **61.97** (`loading-screen-polish`, `save-load-api-guard` passed), with standard-dev still `tsc`-limited. |
| Overnight specialist cycle2 (`economy` / HUD / combat) single-task gates | **`economy_tooltip`:** dataset `20260504-035847_b88a85`, train **`20260504-035848_5bfba2`** (exit 0, ~8969 s); arena **`20260504-062823_8b3903`**: **`0/1`** — **apply_ok**, exports_ok, **`tsc` exit 2** both rounds.<br>**`hud_status`:** dataset `20260504-063042_18ea7e`, train **`20260504-063044_ba7a5e`** (exit 0, ~9763 s); arena **`20260504-091330_8c7651`**: **`0/1`** — round0 `tsc` fail + export gap; round1 **`no_applyable_changes`**.<br>**`combat_risk`:** `adapter-datasets` `20260504-091830_32b592`; first train **`20260504-091837_00c86b`** failed (empty **`valid.jsonl`** / **`test.jsonl`** until **`dataset_builder`** fix); rerun train **`20260504-091854_ee9a2d`** (exit 0); arena **`20260504-110442_ee9296`**: **`0/1`** — apply_ok, **`tsc` exit 2** both rounds. **`adapter_registry_v1`** now defaults these three specialists to **`cycle2`** (**shadow**, not promoted on gate pass). |
| Router registry default paths (2026-05-04) | `training/adapter_registry_v1.json` maps **`economy_tooltip`**, **`hud_status`**, and **`combat_risk`** **`adapter_path`** to **`checkpoints/.../cycle2`** so router / Gradio loads the overnight weights; **`promotion_state`** remains **`shadow`** until a documented passing arena gate. |
| Loading-screen specialist lock-down (v2 data)      | Built `data/lora/adapters/loading_screen_specialist` with `scripts/ml_workflow.py loading-screen-dataset` (core loading + UI transfer rows), trained `checkpoints/adapters/loading_screen/cycle2` (`20260503-180149_d2ed85`, 120 iters, exit 0), then evaluated `loading-screen-polish` + transfer set (`20260503-184315_a608df`) at **1/3**: loading-screen passed (`100`), transfer tasks (`hud-status-summary`, `economy-tooltip`) still failed on `tsc`. Registry promotion applied: `training/adapter_registry_v1.json` now maps `loading-screen-polish` to `loading_screen` with `promotion_state: champion` and adapter path `checkpoints/adapters/loading_screen/cycle2`. |
| Documentation/run-analysis specialist (mlx-lab prose) | `python scripts/ml_workflow.py documentation-dataset` builds `data/lora/adapters/documentation_specialist/` (gold draft→canonical pairs; task alias `mlx-lora-docs-normalize`) for this **LoRA lab repo**, not game implementation. Registry **`documentation`** adapter path **`checkpoints/adapters/documentation/cycle3`** (**shadow**). Latest scope-guard dataset: **`20260505-012316_bc96ac`** (21 core rows, 160 train rows). Latest train: **`20260505-012327_c24601`** (exit 0, **40** iters from `cycle2`, final val **1.045**, best val **0.957** at iter 30, test loss **0.781**) wrote **`adapters.safetensors`** (~154MB). Routing eval **`benchmarks/documentation_eval_tasks_v1.json`** (22 prompts) scored **100%** route+adapter locally; regenerate prompts via **`scripts/build_documentation_eval_tasks_v1.py`**; **`python -m unittest discover -s tests`**. Two RAG lanes: **documentation RAG** (`data/rag/documentation_agent_corpus.json`, `run_documentation_agent_benchmark.py`) and **run-analysis RAG** (`data/rag/run_analysis_agent_corpus.json`, `run_run_analysis_agent_benchmark.py` + `build_run_analysis_rag_corpus.py`). |
| Latest 7B chunked train runs                        | Training completed for both Apr 28 7B chunk passes; workflow exit `1` came from one failed post-train lexical game benchmark task, not a training crash                                                                                                                                                                                                                                                                                                                                                                                                           |
| Optional arena SFT corpus                           | `data/arena_task_baselines/*.assistant.txt` + `scripts/build_arena_baseline_dataset.py` → `data/lora/arena_task_baselines/` (messages JSONL for supervised arena outputs). See `data/arena_task_baselines/README.md`.                                                                                                                                                                                                                                                                                                                                             |
| Historical 1.5B adapters                            | Keep for comparison only unless scripts are explicitly pointed at the matching 1.5B base model                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |


Detailed row-by-row run history belongs in `docs/run_history.md`; narrative command context belongs in `docs/SESSION_LOG.md`. **Arena A/B baselines** (same task suite, progressive on vs off) are tracked in `docs/ARENA_PROGRESSION.md`.

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
python -m mlx_lm chat --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
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

### Router supervisor chat (registry adapters + OSS switch)

Supervisor loads **open-weight Qwen MLX** plus registry LoRA entries from **`training/adapter_registry_v1.json`**; policy labels each turn (`local`/`frontier`/`hybrid`) but **generation is always MLX** (no frontier API hook in-app). **`scripts/router_chat_gradio.py`** binds **[http://127.0.0.1:7864](http://127.0.0.1:7864)** by default (chosen to dodge `train_ui_gradio.py`, which owns **7862**). Expand **OSS / routing controls** to toggle **Auto** routing vs **Codebase OSS** (implements `GenerationRequest(force_route='local')` — policy stays on `local`, still selecting specialist adapters via classifier unless you pin **LoRA adapter lock**).

```bash
source .venv/bin/activate
export ROUTER_CHAT_DEFAULT_BACKBONE=codebase_oss   # optional UI default (or `auto`)
export ROUTER_CHAT_DEFAULT_ADAPTER_LOCK=auto      # registry id overrides classifier when not `auto`
python scripts/router_chat_gradio.py
```

Regression guardrails: **`tests/test_router_backbone_controls.py`** (policy parity for forced-local vs security/long-prompt escalation).

### Terminal lab runner + static optimization page

Prefer **non-interactive** orchestration (`scripts/fe_ml_lab_runner.py`) for the recurring “kick off MLX training smoke / full workflow” flows so Composer sessions stay short. Runner events land in **`lab_dashboard/agent_events.jsonl`** (optional heuristic `FE_ML_LAB_SPARED_USD`). Rebuild the deployable KPI bundle with **`python scripts/build_lab_optimization_dashboard.py`** → **`lab_dashboard/index.html`** (+ committed `docs/run_history.md`; optional manual **`lab_dashboard/cursor_usage.jsonl`** ledger; optional **`--cursor-usage`** / **`--agent-events`** overrides for scripted builds). Deploy instructions: **`lab_dashboard/README.md`**. Lightweight MLX Cursor skill **`.cursor/skills/fe-mlx-lab/SKILL.md`** uses **`disable-model-invocation: true`**; **`@fe-mlx-lab`** on MLX/router tasks so unrelated sessions skip that context. Regression tests **`python3 -m unittest discover -s tests -p test_fe_ml_lab_tools.py`**. Optional hooks: **Cursor `stop`** (**`FE_LAB_CURSOR_HOOK_APPEND=1`**) appends **`lab_dashboard/cursor_hook_events.jsonl`**; **`afterShellExecution`** autodoc (enabled in `.cursor/hooks.json`) appends **`lab_dashboard/shell_command_events.jsonl`** and compact test-run rows in **`docs/SPECIALIZED_RUN_HISTORY.md`** with git-change snapshots (`m/u/d` + touched paths). Set **`FE_LAB_REMOTE_INGEST_URL`** and **`FE_LAB_REMOTE_INGEST_TOKEN`** to mirror those events to a private hosted dashboard (`scripts/private_dashboard_server.py`; see **`docs/PRIVATE_DASHBOARD_DEPLOY.md`**). Hooks stay lightweight (no MLX training inside hooks).

## Game-oriented model benchmark + evolution


| Item                           | Location                                                                                                                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task definitions (game)        | `benchmarks/fallen_empire_tasks.json` (15 tasks; domains include evolution/serialization)                                                                                              |
| Task definitions (general)     | `benchmarks/general_coding_tasks.json` (8 tasks; HTTP/SQL/encoding/doc/version trivia; use `--profile general`)                                                                        |
| Arena capability index         | `scripts/arena_capability_index.py` + `scripts/run_arena_acceptance_tests.py` / `ml_workflow.py arena-acceptance` (deterministic apply/typecheck/export/preview task-completion score) |
| Execution benchmark            | `scripts/run_evalplus_benchmark.py` (EvalPlus HumanEval+/MBPP+; use `ml_workflow.py evalplus`)                                                                                         |
| Routing benchmark              | `benchmarks/task_routing_tasks.json` + `scripts/run_routing_benchmark.py` (legacy route + v1 adapter/tier/confidence metadata)                                                          |
| Specialist routing eval sets     | `data/routing/*_eval_prompts_v*.jsonl` (+ `benchmarks/*_eval_tasks_v*.json`; loading_screen / save_load / economy_tooltip / **documentation** v1), mixed suite `benchmarks/mixed_routing_eval_v1.json` (incl. docs shard; + `scripts/build_mixed_routing_eval_v1.py`)                                              |
| Live routing prompt lab        | `scripts/routing_prompt_lab.py` + `benchmarks/results/routing_prompt_lab/*.jsonl` (manual prompt intake with predicted adapter/route lineage rows)                                      |
| Routing training dataset build | `scripts/build_routing_training_dataset.py` + `data/lora/routing_classifier/<dataset_version>/` + `docs/ROUTING_DATASET_CONTRACT.md`                                                     |
| Adapter registry + taxonomy    | `training/adapter_registry_v1.json` + `scripts/adapters/taxonomy.py` + `docs/adapter_taxonomy_v1.md`                                                                                    |
| Per-adapter datasets           | `scripts/adapters/dataset_builder.py` + `data/lora/adapters/<adapter_id>/` + `docs/dataset_contract_v1.md`                                                                             |
| Adapter gates + drift          | `scripts/adapters/gates.py` + `scripts/adapters/drift_monitor.py` + `docs/gate_policy_v1.md`                                                                                            |
| Control plane scaffold         | `scripts/control_plane/*.py` + `ml_workflow.py control-plane-schedule`                                                                                                                   |
| Ops report/dashboard outputs   | `scripts/build_multi_adapter_report.py` + `ml_workflow.py multi-adapter-report` (`benchmarks/results/multi_adapter_dashboard.json`, `benchmarks/results/multi_adapter_report.md`)     |
| Visual human arena             | `scripts/landing_page_arena.py` + `benchmarks/landing_page_brief.md` (standalone landing-page trials; no game repo edits)                                                              |
| Game task arena                | `scripts/game_task_arena.py` + `benchmarks/game_task_arena_examples.json` (disposable worktree/copy trials against the game repo)                                                      |
| Shared tier / mutation helpers | `scripts/benchmark_evolution_lib.py`                                                                                                                                                   |
| Runner                         | `scripts/run_game_benchmark.py` (`--profile game                                                                                                                                       |
| Season evolution               | `scripts/evolve_benchmark_seasons.py` + `training/evolution_config.json`                                                                                                               |
| Training notes                 | `training/README.md`                                                                                                                                                                   |
| Notes                          | `benchmarks/README.md`                                                                                                                                                                 |


`run_game_benchmark.py` scoring is heuristic (substring rules), intended to regress **base vs LoRA** and prompt changes—not to replace game unit tests. Arena scoring is split into **Smoke Test** (low complexity), **Standard Dev Benchmark** (low-medium/medium), and **High-Reasoning Architecture Benchmark** (medium-high/high). The deterministic arena score is the preferred non-human signal for real game-edit completion: it scores apply success, TypeScript checks, export preservation, preview readiness, retry count, token/time efficiency, and task complexity from arena artifacts. Low-only runs are capped as smoke tests so a loading-screen pass is not treated as full arena capability. As of 2026-05-01, arena capability output also includes a **diversified** headline index (`diversified_arena_capability_index`) that blends task-type-balanced performance, worst-tail robustness, and task-catalog coverage.

**Dashboard + gates:** Regenerate `**benchmarks/results/arena_dashboard.html`** with `**python scripts/ml_workflow.py arena-dashboard**` after local six-task runs; gate adapter promotion with `**python scripts/arena_promotion_gate.py …/arena_capability.json**` (see `**docs/ARENA_ROADMAP.md**`). Promotion gating now enforces minimum evaluated-task count and coverage ratio by default (`--min-tasks 6`, `--min-coverage-ratio 1.0`) and can require suite metadata (`--require-suite core6|all|custom`). Acceptance summaries mark explicit `--task-id` subsets as `suite: "custom"` so partial checks are not mislabeled as core-six runs. Progressive `**auto`/off** norms for keyed adapters live in `**scripts/fe_lineage.py`**.

**Multi-adapter v1 baseline (2026-05-03):** routing/classification now emits adapter-aware metadata (`adapter_id`, `execution_tier`, `council_mode`, `confidence`, `ambiguity`, `risk_class`, `complexity`) under policy tag `router_policy_v1`. Locked defaults follow peak-specialist profile with canary steps `5 -> 15 -> 35 -> 60 -> 100`, medium-complexity council size `4`, high-complexity judge path `api_first`, and rollback guards on apply-failure regression (`>=8pp`), latency regression (`>=40%` p95), or safety events (`>2`). **Deterministic-router specialist shortcuts:** `loading_screen`, `save_load_api_guard`, and `economy_tooltip` take local fast-path + suppress generic substring escalations once the classifier wins that adapter; global carve-outs (**`cryptography`**, audits of **authn/authz** without save-domain lexicon) still beat the save specialist when both match.


Routing test+train pipeline now supports dual-label evaluation and dataset generation:

- `scripts/run_routing_benchmark.py --mode both` scores adapter + legacy-route labels and emits confusion summaries.
- `scripts/routing_prompt_lab.py` captures live prompts (vibe-coding style) with predicted adapter/route metadata into JSONL.
- `scripts/build_routing_training_dataset.py` merges benchmark/live/curated labels into deterministic train/valid/test splits with `manifest.json`.
- `scripts/ml_workflow.py` includes `routing-prompt-lab`, `routing-benchmark`, and `routing-dataset` so these runs follow normal workflow artifact + `docs/run_history.md` audit paths.

Game task arena task specs now support `preview_path`. Preview URLs are composed as `http://127.0.0.1:<port><preview_path>` and previews are marked `ready` only after the URL responds. The current standardized visual tasks point at game-owned `/test-env/...` sandbox routes in `/Users/natreed/fallen-empire`, including `/test-env/loading-screen` and `/test-env/combat-risk-preview`, so human review opens directly on deterministic test screens.

Preview startup uses `next dev -H 127.0.0.1 -p {port}` rather than the package `npm run dev` wrapper. This avoids the old disposable-worktree failure where `npm run dev -- --host ... --port ...` resolved to `next: command not found`; the arena now reuses source checkout dependencies through `PATH`/`NODE_PATH` and a best-effort `node_modules` symlink. Generation artifacts include `generation_metrics.json` plus manifest fields for elapsed seconds and token counts.

Progressive context is now enabled by default for non-low-complexity arena generation (`--progressive-context auto`). The model first requests exact files it needs, the harness appends bounded targeted snippets, and logs are written to `logs/progressive_context.json` / `.md`. This is intended to move HUD/economy/combat/architecture tasks away from one static oversized context packet.

As of 2026-05-01, standard-dev tasks (`hud-status-summary`, `economy-tooltip`) use tighter retrieval controls in `scripts/game_task_arena.py`: noisy file exclusion (`.bak`, `.DS_Store`), task-specific priority/allowlisted context paths, deduped packed files with a small cap, and larger reserved body budget for code context (with `package.json` omitted for that path).

Applyability contract is now codified in `docs/GAME_ARENA_APPLY_CONTRACT.md` and injected into arena model packets plus pairwise/baseline SFT prompts. For targeted `no_applyable_changes` training loops, use `build_game_task_pairwise_dataset.py --focus-apply-failures` (or `ml_workflow.py arena-gate-train --rebuild-dataset --focus-apply-failures`).

```bash
source .venv/bin/activate
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --profile general
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5
python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 --task-id loading-screen-polish
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
- Script: `scripts/build_github_ts_dataset.py` (optional TypeScript corpus intake from GitHub repos/search with permissive-license + strict-tsconfig + path filters; default output `data/lora/qwen25-coder-7b/github_ts_compile_safe/`).

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
- Default local adapter: `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` (shared via `scripts/fe_lineage.py`; do not use 1.5B adapters with the 7B base).

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
- Default local adapter: `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` (shared via `scripts/fe_lineage.py`; avoids loading old 1.5B adapters against the 7B base)
- Results: `benchmarks/results/game_task_trials/<trial_id>/`, append-only `benchmarks/results/game_task_index.jsonl`, training rows in `benchmarks/results/game_task_training_data.jsonl`, summaries under `benchmarks/results/game_task_reports/`
- Purpose: evaluate local/frontier attempts on real game-code tasks in disposable worktrees/copies, run fixed verification commands, record diffs/logs/previews/ratings, compare adapters over time, and clean up worktrees without deleting indexed artifacts.
- UI: standardized task selector is the primary control; `Source repo`, `Worktree root`, and `Base ref` are advanced settings. Main flow creates worktrees, generates/applies both model attempts, starts playable Local/Frontier preview links, records grades, and optionally deletes disposable worktrees. Packet text, manifests, verification output, and reports are hidden in a details drawer.
- Safety: never edit the main game checkout; never run model-suggested shell commands; only run task-spec verification commands; reject file writes outside allowed path globs.
- Generation timeout: the UI runs Local and Frontier generation in bounded subprocesses (default **360s**, override with `GAME_TASK_ARENA_GENERATION_TIMEOUT_S` or the UI field). Timeouts are recorded as `generation_timeout` instead of leaving the UI apparently running forever.

## Known issues / API notes

1. **urllib3 / LibreSSL:** At import time, urllib3 may warn that LibreSSL is older than OpenSSL 1.1.1; training/inference still proceeded.
2. `**mlx-lm` 0.29.1 generation:** `generate()` forwards kwargs to `generate_step()`, which does **not** accept `temp=`. Use `sampler=make_sampler(temp=..., top_p=1.0)` for non-zero temperature; `temp=0` omits sampler (greedy argmax). Implemented in `scripts/smoke_base_model.py` and `scripts/chat_gradio.py`.
3. **EvalPlus dataset availability:** First `ml_workflow.py evalplus` verification attempts on 2026-04-26 failed before model load because GitHub returned HTTP 502 for `HumanEvalPlus.jsonl.gz`. Retry later or set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` to a local EvalPlus JSONL.
4. **Qwen2.5 chat EOS under mlx_lm streaming:** Wrapped tokenizers expose `<|endoftext|>` (151643) in `eos_token_ids` only. Assistant turns terminate with `<|im_end|>` (**151645**); if omitted, ``stream_generate`` never stops on IM-end and chats show repeated sentinel strings (looks like a “failed” docs LoRA). After every `mlx_lm.load`, runners call **`scripts/mlx_qwen_stop_tokens.register_qwen_coder_instruct_extra_stops`** (Gradio chats, **`router_chat_gradio.py`**, **`human_eval_ui.py`**, **`LocalMlxBackend`**).
## Training / LoRA (wired)


| Piece                                | Current role                                                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/ml_workflow.py`             | Preferred repeatable entry point; re-execs into `.venv/bin/python`, writes ignored per-run artifacts, and appends `docs/run_history.md`.    |
| `docs/WORKFLOW.md`                   | Command reference for workflow subcommands and examples.                                                                                    |
| `docs/RUNS.md`                       | Current-vs-historical run interpretation and adapter recommendations.                                                                       |
| `docs/run_history.md`                | Append-only committed run index.                                                                                                            |
| `scripts/build_lora_dataset.py`      | Builds the default 7B `game_text` splits; layout details live in `docs/DATA_LAYOUT.md` and chunking details in `docs/CHUNKED_GAME_TEXT.md`. |
| `training/lora_qwen25_coder_7b.yaml` | Current 7B LoRA config. The old `training/lora_qwen_coder.yaml` name is historical only.                                                    |
| `scripts/train_lora.sh`              | Legacy convenience wrapper; prefer `scripts/ml_workflow.py` for documented runs.                                                            |


For historical run outcomes, use `docs/RUNS.md` first, then `docs/run_history.md` for the full table and `docs/SESSION_LOG.md` for command narrative. This file should stay focused on current defaults and durable workflow behavior.

## Git / large artifacts

`.gitignore` excludes `.venv/`, `models/`, `checkpoints/`, `data/raw/`, `data/lora/`, `benchmarks/results/`, common weight extensions. Do not commit weights or raw exports.
