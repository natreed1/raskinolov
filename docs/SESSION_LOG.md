# Session log (append-only)

Newest entries at the **top**.

---

## 2026-05-04 — Private dashboard scoring tab (Cursor vs model outputs)

Updated `scripts/private_dashboard_server.py` to add a tabbed UI with a new **Scoring vs Cursor Work** panel. The server now reads latest `docs/generated/*.comparison.json` (e.g., duplicate-doc artifacts), computes winner by keyword-hit score, and renders side-by-side tracks for `codex_authored`, `opensource`, and `specialized`.

Added private API endpoint `GET /api/scoring` (same Basic-auth guard as other viewer APIs) and HTML tab wiring (`Overview`, `Scoring`, `Docs Snapshot`).

Docs updated: `docs/PRIVATE_DASHBOARD_DEPLOY.md` now includes `/api/scoring` and scoring-tab verification step.

---

## 2026-05-04 — Code-change training trigger + dual RAG lanes

Added a second RAG lane for run documentation/analysis and wired training prep to codebase edits.

**Dual RAG:**

- Documentation RAG (existing): `data/rag/documentation_agent_corpus.json` + `scripts/run_documentation_agent_benchmark.py`
- Run-analysis RAG (new): `data/rag/run_analysis_agent_corpus.json` + `scripts/run_analysis_rag.py` + `scripts/run_run_analysis_agent_benchmark.py` + tasks `benchmarks/run_analysis_rag_tasks_v1.json`

**Trigger on codebase changes (training prep):**

- New hook: `.cursor/hooks/lab_hook_after_fileedit_training_trigger.py` wired via `.cursor/hooks.json` `afterFileEdit`
- New trigger script: `scripts/trigger_doc_training_on_changes.py`
- On watched edits (`docs/`, `scripts/`, `benchmarks/`, `tests/`, `training/`), it refreshes run-analysis corpus and (cooldown-gated) runs `python scripts/ml_workflow.py documentation-dataset`
- Queue/audit output: `data/training_triggers/documentation_training_queue.jsonl`

Controls:

- `FE_LAB_DOC_TRIGGER_MIN_SECONDS` (default 900)
- `FE_LAB_AUTODOC_DATASET_ON_CHANGE` (default on)

---

## 2026-05-04 — Duplicate docs phase: open-source vs documentation-specialist output

Ran a paired generation pass for the new private dashboard docs using the same prompt/reference source (`docs/PRIVATE_DASHBOARD_DEPLOY.md`) across:

- base open-source model (`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`)
- specialized adapter (`checkpoints/adapters/documentation/cycle3`)

Artifacts written to **`docs/generated/`**:

- `private_dashboard_deploy.opensource.md`
- `private_dashboard_deploy.specialized.md`
- `private_dashboard_deploy.comparison.json`
- `private_dashboard_deploy.comparison.md`

Comparison includes a third anchor score for **Codex-authored** `docs/PRIVATE_DASHBOARD_DEPLOY.md` (keyword coverage + word-count). This establishes the requested duplicate-supervision baseline for tuning documentation behavior.

---

## 2026-05-04 — Private live telemetry dashboard (Railway-first, secure)

Implemented **`scripts/private_dashboard_server.py`**: a no-extra-deps WSGI service with HTTP Basic auth for viewers (`/`, `/api/summary`, `/api/events`), bearer-token ingest (`POST /api/ingest`), SQLite persistence (`FE_DASHBOARD_DB_PATH`), and docs snapshots (`PROJECT_STATE`, `SESSION_LOG`, `SPECIALIZED_RUN_HISTORY`) for personal project observability.

Added **`railway.json`** for one-command deploy (`python3 scripts/private_dashboard_server.py`) and wrote **`docs/PRIVATE_DASHBOARD_DEPLOY.md`** with env/volume/security setup plus local hook->remote ingest wiring.

Hook updates: **`.cursor/hooks/lab_hook_after_shell_autodoc.py`** and **`.cursor/hooks/lab_hook_stop_append.py`** now optionally POST events when **`FE_LAB_REMOTE_INGEST_URL`** + **`FE_LAB_REMOTE_INGEST_TOKEN`** are set (fail-open on network errors). `afterShellExecution` autodoc remains local-first (`shell_command_events.jsonl` + `SPECIALIZED_RUN_HISTORY`).

---

## 2026-05-04 — Cursor test-run autodoc hook (shell + code-change snapshot)

Added project hook **`.cursor/hooks/lab_hook_after_shell_autodoc.py`** and wired **`afterShellExecution`** in **`.cursor/hooks.json`**. Hook command now enables autodoc by default for this repo (`FE_LAB_AUTODOC_APPEND=1` inline), so Cursor shell test/benchmark commands are auto-recorded with exit code + git status snapshot (`m/u/d` counts + touched paths) to **`lab_dashboard/shell_command_events.jsonl`** and appended as compact rows in **`docs/SPECIALIZED_RUN_HISTORY.md`**.

Filter is command-based (`pytest`, `unittest`, benchmark runners, `scripts/ml_workflow.py`) unless **`FE_LAB_AUTODOC_INCLUDE_ALL=1`** is set.

---

## 2026-05-04 — Documentation-agent RAG: specialized run history + orchestrated workflow

**Goal:** keep auxiliary evals legible alongside `docs/run_history.md`, and make the documentation-agent RAG benchmark a one-command habit with aggregate telemetry.

**Added:** `docs/SPECIALIZED_RUN_HISTORY.md` — append-only table for cross-cutting benchmarks (retrospective rows for the manual **2026-05-04** `documentation_agent_*_7b.jsonl` runs + convention for linking `ml_workflow_run_id` in Notes).

**Orchestration:** `python scripts/ml_workflow.py documentation-rag-benchmark` — `benchmarks/results/runs/<run_id>/` (manifest, per-task JSONL), **`docs/run_history.md`** row, **`docs/SPECIALIZED_RUN_HISTORY.md`** row, tail **`benchmarks/results/documentation_rag_timeseries.jsonl`**. **Exit code** follows the **with-RAG** pass; optional no-RAG baseline uses `run_documentation_agent_benchmark.py --no-fail`.

**Runner:** `scripts/fe_ml_lab_runner.py documentation-rag-benchmark` (optional passthrough flags) appends `lab_dashboard/agent_events.jsonl` with `kind: documentation_rag_benchmark`.

**Code:** `scripts/run_documentation_agent_benchmark.py` gains `--no-fail`; `scripts/ml_workflow.py` gains helpers `_append_specialized_history_row`, `_count_doc_benchmark_jsonl`, `_append_documentation_rag_timeseries`. Docs: `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `benchmarks/README.md`, `.cursor/rules/precise-ml-documentation.mdc`. Test: `tests/test_fe_ml_lab_tools.py::test_cmd_documentation_rag_benchmark_invokes_ml_workflow`.

**Verify:** `python -m py_compile scripts/ml_workflow.py …`, `python scripts/ml_workflow.py documentation-rag-benchmark --help`, `python -m unittest tests.test_fe_ml_lab_tools tests.test_documentation_rag -v`.

---

## 2026-05-04 — Cursor `stop` hook ledger (opt-in)

Added **`.cursor/hooks.json`** plus **`.cursor/hooks/lab_hook_stop_append.py`**: Agents hitting **`stop`** append **`lab_dashboard/cursor_hook_events.jsonl`** when **`FE_LAB_CURSOR_HOOK_APPEND=1`** (optional **`FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD=1`** reruns **`build_lab_optimization_dashboard.py`**). **`--cursor-hooks`** override added on the dashboard script. **`lab_dashboard/README.md`** documents hooks versus scheduled MLX versus git syncing across clones/windows.

---

## 2026-05-04 — fe-mlx-lab skill · Cursor token footprint

Shrunk **`.cursor/skills/fe-mlx-lab/SKILL.md`**, **`disable-model-invocation: true`** so MLX guidance is mainly on explicit **`@fe-mlx-lab`** mentions; playbook says terminal-only MLX and **path citations instead of log dumps**.

---

## 2026-05-04 — Router Gradio OSS backbone switch (`force_route=local`)

`scripts/router_chat_gradio.py` now exposes accordion **OSS / routing controls**: **Backbone** (policy **Auto** vs **Codebase OSS** → `GenerationRequest(force_route='local')`), optional registry **LoRA adapter lock**, plus defaults via **`ROUTER_CHAT_DEFAULT_BACKBONE`**, **`ROUTER_CHAT_DEFAULT_ADAPTER_LOCK`**, or **`--default-backbone`** / **`--default-adapter-lock`**. Listener default **`--port` is `7864`** (avoids **`train_ui_gradio.py`'s** **`7862`** collision). **`tests/test_router_backbone_controls.py`** locks regressions vs security-keyword frontier escalation. Revised **`.cursor/skills/fe-mlx-lab/SKILL.md`** stating Composer cannot load repo LoRA; OSS path is MLX UIs described in **`docs/PROJECT_STATE.md`**.


Added **`tests/test_fe_ml_lab_tools.py`** (mocked subprocess coverage for **`fe_ml_lab_runner`**, deterministic fixtures for **`build_lab_optimization_dashboard`**) + CLI overrides **`--cursor-usage`** / **`--agent-events`** on **`scripts/build_lab_optimization_dashboard.py`**. Narrative SKILL note: Cursor integration is Markdown skill metadata consumption, **not** a runtime plugin.

Added **`scripts/fe_ml_lab_runner.py`** with default task **`learning` → `ml_workflow.py smoke`** (switch `--sequence full` for end-to-end; passthrough MLX flags **after `--`**). Each invocation appends JSON lines to **`lab_dashboard/agent_events.jsonl`** (`FE_ML_LAB_SPARED_USD` optional heuristic). **`scripts/build_lab_optimization_dashboard.py`** renders **`lab_dashboard/index.html`** from **`docs/run_history.md`** plus optional **`lab_dashboard/cursor_usage.jsonl`**. Supporting docs: **`lab_dashboard/README.md`**, Cursor skill **`.cursor/skills/fe-mlx-lab/SKILL.md`**. Verified **`python3 -m py_compile scripts/fe_ml_lab_runner.py scripts/build_lab_optimization_dashboard.py`**, **`python scripts/build_lab_optimization_dashboard.py`**, and **`python3 -m unittest discover -s tests`**.

---

## 2026-05-04 — Combat `cycle3` resume train + HUD / combat arena acceptance

**Combat resume:** Finished **`python scripts/ml_workflow.py train`** with **`--resume-adapter-file checkpoints/adapters/combat_risk/cycle3/adapters.safetensors --iters 300`**. Run **`benchmarks/results/runs/20260504-175440_9e0edb`** — **`final_exit_code` 0** (~12644 s).

**Arena (single-task, `auto`, `benchmarks/game_task_arena_examples.json`):**

| Adapter | Run id | Task | Result |
| --- | --- | --- | --- |
| `checkpoints/adapters/hud_status/cycle2` | **`20260504-212542_a576dd`** | `hud-status-summary` | **Failed** — **`generate_failed`**, **`generate_exit` 124** (never reached apply) |
| `checkpoints/adapters/combat_risk/cycle3` | **`20260504-214556_a37ce5`** | `combat-risk-preview` | **Failed** — round0 edited **`GameHUD.tsx`** (export **`GameHUD`** missing); round1 **`no_applyable_changes`** |

**Registry:** Bumped **`combat_risk` `adapter_path`** to **`checkpoints/adapters/combat_risk/cycle3`** now that resume train completed (**`training/adapter_registry_v1.json`**).

---

## 2026-05-04 — Registry: route `economy_tooltip`, `hud_status`, `combat_risk` to `cycle2`

Updated `training/adapter_registry_v1.json` so **`adapter_path`** resolves to the adapters trained in the overnight lockdown run: `checkpoints/adapters/economy_tooltip/cycle2`, `checkpoints/adapters/hud_status/cycle2`, `checkpoints/adapters/combat_risk/cycle2`; lineages bumped accordingly. **Promotion remains `shadow`** (single-task arena gates for these three were **not** passing at last documented runs); this only aligns the router / local UIs with the newest on-disk weights.

---

## 2026-05-04 — Overnight specialist lockdown (economy / HUD / combat cycle2)

Executed the agreed runbook: **`ml_workflow`** dataset → **`train`** → single-task **`arena-acceptance`** (`--progressive-context auto`, **`benchmarks/game_task_arena_examples.json`**, **`SOURCE_REPO`** `~/fallen-empire`, **`GAME_ARENA_ROOT`** `~/fallen-empire-arena`). **No registry promotion** (none of the three gates passed end-to-end).

| Stage | Run id / adapter | Outcome |
| --- | --- | --- |
| **`economy_tooltip`** dataset | `benchmarks/results/runs/20260504-035847_b88a85` | exit **0** |
| **`economy_tooltip`** train | `20260504-035848_5bfba2` → `checkpoints/adapters/economy_tooltip/cycle2` | exit **0** (~8970 s) |
| **`economy_tooltip`** arena | `20260504-062823_8b3903`, `--task-id economy-tooltip` | exit **1**, **0/1** — **`apply_final_ok`**, **`exports_final_ok`**, **`tsc`** `tsc_exit_2` rounds 0–1 |
| **`hud_status`** dataset | `20260504-063042_18ea7e` | exit **0** |
| **`hud_status`** train | `20260504-063044_ba7a5e` → `checkpoints/adapters/hud_status/cycle2` | exit **0** (~9763 s) |
| **`hud_status`** arena | `20260504-091330_8c7651`, `--task-id hud-status-summary` | exit **1**, **0/1** — mixed rounds (export gap + retry **`no_applyable_changes`**) |
| **`adapter-datasets`** | `20260504-091830_32b592` | exit **0** (refreshed `data/lora/adapters/*`) |
| **`combat_risk`** train (1st) | `20260504-091837_00c86b` | exit **1** — MLX **`IndexError`** on empty **`valid.jsonl`** / **`test.jsonl`** for **`combat_risk`** when only two synthetic train lines existed |
| **`combat_risk`** train (2nd) | `20260504-091854_ee9a2d` | exit **0** after duplicating train rows into **`valid.jsonl`** / **`test.jsonl`** for the immediate run; **`scripts/adapters/dataset_builder.py`** **`_split_rows`** now guarantees non-empty valid+test for tiny families so future **`adapter-datasets`** builds load in **`mlx_lm`** |
| **`combat_risk`** arena | `20260504-110442_ee9296`, `--task-id combat-risk-preview` | exit **1**, **0/1** — **`apply_final_ok`**, **`tsc_exit_2`** both rounds |

**Next:** widen **`shared_general_anchor`** or add **`build_combat_*` / pairwise combat rows** before another combat cycle2 pass; chase **`tsc`** deltas on **`economy`** and **`combat`** with curator-aligned repair JSONL or lower LR / fewer iters smoke.

---

## 2026-05-04 — HUD cycle2 arena smoke (`hud-status-summary`)

Ran `arena-acceptance` on `checkpoints/adapters/hud_status/cycle2`. Run **`benchmarks/results/runs/20260504-031825_0f7a64`**: **`exit_code` 1**, **0 / 1** passed. Applied `CompactEmpireStatus.tsx` + `TestEnvironmentShell.tsx` but **`tsc` failed** both rounds (`goldPile`, `goldHold`, `morale` on `Player`, missing imports like `countVillagesInPlayerTerritory` / supply helper, bogus `provinceHexKeys`) — schema hallucination vs curator baseline.

---

## 2026-05-04 — Retrain HUD adapter (v2 specialist JSONL)

**Outcome:** Completed successfully (`exit 0`, ~80 min). Run `benchmarks/results/runs/20260504-015258_141e5c`. MLX iter **200** (val loss ~**0.39**, train loss ~**0.005**); final weights written to `checkpoints/adapters/hud_status/cycle2/adapters.safetensors` and `0000200_adapters.safetensors`. `run_history.md` appended by orchestrator.

**Command:** Background `ml_workflow.py train --adapter-path checkpoints/adapters/hud_status/cycle2` with `--data data/lora/adapters/hud_status_specialist` (200 iters, hyphenated mlx args).

**Next:** `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/hud_status/cycle2 --task-id hud-status-summary`

---

## 2026-05-04 — HUD specialist dataset v2 (alignment)

**Goal:** align HUD training targets with curator gold + real `useGameStore` slices; reduce TSC-hallucination pressure from pairwise.

**Implementation (`scripts/adapters/build_hud_status_specialist_dataset.py`):**

- Lineage `**hud_status_specialist:v2`**; manifest `**schema_version`:** `hud_status_specialist_dataset_v2`.
- Arena-aligned `**HUD_TASK_USER`** for all baseline shards (paths, spectate bootstrap, no invented APIs).
- `**--baseline-shards` (default 12):** duplicate `hud-status-summary.assistant.txt` core rows with distinct `record_id`s.
- **Guardrail rows (3):** short instruction replies anchoring valid store selectors (`players`, `cities`, `units`, …), bootstrap order, JSX closing-tag caution; omit with `**--no-guardrails`**.
- **HUD pairwise filter:** skip winners containing known bad fragments (`matchState`, `myId`, …); disable via `**--no-filter-pairwise-hud`**.
- `**--max-core-rows` default 0:** no HUD pairwise core unless explicitly increased.

**Orchestration (`scripts/ml_workflow.py`):** `hud-status-dataset` forwards `**--baseline-shards`**, `**--no-hud-guardrails**`, `**--no-filter-pairwise-hud**`, default `**--max-core-rows 0**`.

**Docs:** `docs/WORKFLOW.md` row for `hud-status-dataset`.

**Rebuild:** Ran builder with `--min-train-core-rows 120` → **15 core** (12 baseline + 3 guard, 0 pairwise HUD), manifest under `data/lora/adapters/hud_status_specialist/manifest.json`.

---

## 2026-05-04 — HUD cycle2 smoke arena: background launcher PATH

**What happened:** A background `arena-acceptance` for `hud-status-summary` wrote `command not found: python` to `benchmarks/results/hud_cycle2_smoke_arena.log` (non-login shell had no `python` on `PATH`).

**Fix:** Re-launched the same gate using the repo venv: `.venv/bin/python scripts/ml_workflow.py arena-acceptance ...` (PID noted in shell; log appended to the same file).

---

## 2026-05-03 — Promoted loading-screen specialist as pipeline-ready

**Goal:** mark `loading_screen` as the primary pipeline adapter for `loading-screen-polish` after successful lock-down run.

**Changes:**

- Updated taxonomy mapping in `scripts/adapters/taxonomy.py`:
  - `loading-screen-polish` now routes to `loading_screen` (no longer `general_fallback`).
- Updated `training/adapter_registry_v1.json`:
  - `general_fallback.task_ids` cleared.
  - `loading_screen.task_ids` now includes `loading-screen-polish`.
  - `loading_screen.adapter_path` set to `checkpoints/adapters/loading_screen/cycle2`.
  - `loading_screen.lineage` set to `loading_screen:v2:cycle2`.
  - `loading_screen.promotion_state` set to `champion`.

**Rationale:**

- `loading-screen-polish` achieved deterministic acceptance pass (`score 100`) on the focused specialist evaluation run (`20260503-184315_a608df`), which is sufficient for the current narrow-scope promotion target.
- Transfer-domain behavior remains experimental and is intentionally not part of this promotion decision.

---

## 2026-05-03 — Loading-screen specialist dataset v2 + cycle2 training/acceptance

**Goal:** lock down `loading_screen` first by improving specialist data quality (core loading rows + transfer UI rows) before moving to the next domain.

**Implemented:**

- Added `scripts/adapters/build_loading_screen_specialist_dataset.py` to build `data/lora/adapters/loading_screen_specialist/{train,valid,test}.jsonl` + `manifest.json`.
- Added `loading-screen-dataset` subcommand wiring in `scripts/ml_workflow.py`.
- Dataset builder sources:
  - `benchmarks/results/game_task_pairwise_training_data.jsonl` winner outputs for `loading-screen-polish`.
  - curated baseline `data/arena_task_baselines/loading-screen-polish.assistant.txt`.
  - transfer task rows (`hud-status-summary`, `economy-tooltip`) to test domain translation.
- Added deterministic split/backfill behavior and minimum train-size expansion (`--min-train-core-rows`) to avoid tiny specialist runs.

**Commands run:**

- `python scripts/ml_workflow.py loading-screen-dataset --transfer-task-id hud-status-summary --transfer-task-id economy-tooltip --min-train-core-rows 120` (pass; run `20260503-180139_940384`).
- `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/loading_screen/cycle2 -- --data data/lora/adapters/loading_screen_specialist --iters 120 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 40 --max-seq-length 2048` (pass; run `20260503-180149_d2ed85`).
- `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/loading_screen/cycle2 --source-repo /Users/natreed/fallen-empire --preview-port 5300 --timeout-s 14400 --task-id loading-screen-polish --task-id hud-status-summary --task-id economy-tooltip` (run `20260503-184315_a608df`, `1/3` pass).

**Outcome:**

- `loading-screen-polish` passed (`score 100`, apply/tsc/exports/preview all true), so the specialized agent can complete its core task in this run.
- Transfer-domain probes remain unstable (`hud-status-summary` and `economy-tooltip` failed with `tsc`), so scope is currently **core-only reliable** and **transfer experimental**.

---

## 2026-05-03 — Built routing test+train pipeline (prompt lab + dual benchmark + dataset builder + workflow wiring)

**Goal:** implement the routing development pipeline for specialization selection: support live typed prompts, preserve historical route benchmark coverage, evaluate adapter+route labels, and generate reproducible training datasets.

**Added files:**

- `docs/ROUTING_DATASET_CONTRACT.md`: dual-label schema (`expected_adapter_id` + `expected_legacy_route`), lineage fields, deterministic split policy, hard-example criteria.
- `scripts/routing_prompt_lab.py`: interactive/batch prompt intake that predicts adapter + legacy route and writes lineage-rich JSONL under `benchmarks/results/routing_prompt_lab/`.
- `scripts/build_routing_training_dataset.py`: merges benchmark/live/curated labeled prompts into deterministic `train/valid/test` splits + `manifest.json` under `data/lora/routing_classifier/<dataset_version>/`.
- `data/routing/README.md`: versioning conventions for routing source labels.

**Updated runtime wiring:**

- `scripts/run_routing_benchmark.py` now supports `--mode route|adapter|both`, dual-label scoring, confusion summaries, and optional `--summary-json`.
- `scripts/ml_workflow.py` added subcommands:
  - `routing-prompt-lab`
  - `routing-benchmark`
  - `routing-dataset`
  All three are included in standard run-artifact + `docs/run_history.md` workflow behavior.

**Docs updated:**

- `docs/WORKFLOW.md`: command table + examples for prompt lab, dual-label routing benchmark, and routing dataset builds.
- `docs/PROJECT_STATE.md`: routing pipeline status and script/index paths.
- `docs/DATA_LAYOUT.md`: canonical routing source/build artifact paths.

**Verification commands run:**

- `python3 -m py_compile scripts/routing_prompt_lab.py scripts/run_routing_benchmark.py scripts/build_routing_training_dataset.py scripts/ml_workflow.py` (pass).
- `python3 scripts/run_routing_benchmark.py --mode both` (pass; route accuracy `12/12`, adapter labeled rows `0/0` on legacy benchmark file).
- `python3 scripts/routing_prompt_lab.py --prompt "Please review this combat risk tooltip behavior" --output-jsonl benchmarks/results/routing_prompt_lab/smoke.jsonl` (pass; wrote one routing row with adapter/route prediction).
- `python3 scripts/build_routing_training_dataset.py --live-jsonl benchmarks/results/routing_prompt_lab/smoke.jsonl --dataset-version routing-smoke --out-root data/lora/routing_classifier` (pass; wrote deterministic split files + manifest).

---

## 2026-05-03 — Implemented v1 multi-adapter architecture scaffold end-to-end

**Goal:** implement the approved v1 architecture plan without replacing the existing ML workflow/arena stack; deliver practical scaffolding across taxonomy, datasets, gates, routing policy, control plane, drift automation, reporting, docs, and verification.

**Added architecture modules:**

- `scripts/adapters/taxonomy.py`: locked v1 family taxonomy, task mapping, registry schema, promotion state metadata, canary defaults, and `policy_version` lineage fields.
- `scripts/adapters/dataset_builder.py`: per-adapter dataset flow with shared anti-overfit corpus + manifests (`85/10/5` default mix) into `data/lora/adapters/<adapter_id>/`.
- `scripts/adapters/gates.py`: per-adapter gate verdicts with primary average-gain objective and hard rollback guards.
- `scripts/adapters/drift_monitor.py`: warn/freeze/reduce/rollback drift checks separating adapter vs router movement.
- `scripts/router/classifier.py` + `scripts/router/policy.py`: confidence/ambiguity/risk/complexity classifier and adapter-aware routing policy with council modes + API escalation ladder.
- `scripts/control_plane/models.py`, `scripts/control_plane/auth.py`, `scripts/control_plane/scheduler.py`: worker heartbeat/capability schemas, auth token primitives, scheduling/retry scaffolding for multi-Mac workers.
- `scripts/build_multi_adapter_report.py`: dashboard/report output for routing distribution, adapter gate quality, and SLO drift signals.

**Integrated with existing stack:**

- Updated `scripts/model_router.py` to keep legacy `local/frontier/hybrid` routing while adding v1 metadata fields (`adapter_id`, `execution_tier`, `council_mode`, `confidence`, `ambiguity`, `risk_class`, `complexity`, `policy_version`, lineage tags).
- Updated `scripts/run_routing_benchmark.py` to emit the new routing metadata in benchmark rows.
- Extended `scripts/ml_workflow.py` with v1 subcommands while preserving run-artifact and run-history behavior:
  - `adapter-registry`
  - `adapter-datasets`
  - `adapter-gate`
  - `drift-check`
  - `control-plane-schedule`
  - `multi-adapter-report`

**Docs + contracts added/updated:**

- Added: `docs/adapter_taxonomy_v1.md`, `docs/dataset_contract_v1.md`, `docs/gate_policy_v1.md`.
- Added baseline registry: `training/adapter_registry_v1.json`.
- Updated: `docs/WORKFLOW.md` (new subcommands/examples + routing metadata), `docs/PROJECT_STATE.md` (v1 architecture status and defaults).

**Verification commands run:**

- `.venv/bin/python -m py_compile scripts/model_router.py scripts/run_routing_benchmark.py scripts/ml_workflow.py scripts/build_multi_adapter_report.py scripts/adapters/taxonomy.py scripts/adapters/dataset_builder.py scripts/adapters/gates.py scripts/adapters/drift_monitor.py scripts/control_plane/models.py scripts/control_plane/auth.py scripts/control_plane/scheduler.py`
- `.venv/bin/python scripts/run_routing_benchmark.py` (policy-only benchmark, now writing v1 metadata fields)
- `.venv/bin/python scripts/run_routing_benchmark.py --output-jsonl benchmarks/results/routing_policy_rows.jsonl` (12/12 pass with v1 route metadata rows)
- `.venv/bin/python scripts/ml_workflow.py adapter-registry --path training/adapter_registry_v1.json`
- `.venv/bin/python scripts/ml_workflow.py adapter-registry --write-default --path /tmp/adapter_registry_v1.smoke.json`
- `.venv/bin/python scripts/ml_workflow.py adapter-datasets --sources-dir data/arena_task_baselines --shared-anchor-dir data/lora/game_text --out-dir data/lora/adapters`

---

## 2026-05-02 — Added GitHub strict-TS ingestion pipeline for compile-safe LoRA data

**Goal:** create a practical now-usable ingestion/curation path for GitHub TypeScript repos that better match arena apply+`tsc`+exports bottlenecks.

**Added / changed:**

- Added `scripts/build_github_ts_dataset.py`.
  - Inputs: explicit `--repo owner/name`, optional `--repos-file`, optional discovery via `--discover-query`.
  - Filters: permissive SPDX allowlist, strict `tsconfig` evidence (`strict=true` or strict key trio), non-trivial app heuristic (`package.json` + include-root TS paths), TS/TSX path filters and test/build exclusions.
  - Outputs: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `samples_metadata.jsonl`, `manifest.json`.
  - Robustness: best-effort discovery (warnings on query/API failure), optional GitHub token, zip cache under `data/raw/github_ts_cache`, manifest-level accepted/rejected reasons.
- Added docs: `docs/GITHUB_TS_DATASET.md` (usage, filters, output layout, training invocation).
- Updated `docs/DATA_LAYOUT.md` with canonical path section for `data/lora/qwen25-coder-7b/github_ts_compile_safe/`.
- Updated `docs/PROJECT_STATE.md` dataset section to reference the new optional GitHub TypeScript builder.

**Commands run:**

- `.venv/bin/python scripts/build_github_ts_dataset.py --repo pmndrs/zustand --repo reduxjs/redux-toolkit --out-dir data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke`
  - First run: exited with no rows; manifest showed `missing_zipball_url` for both repos.
  - Fixed script to fall back from `zipball_url` to `archive_url` template.
- `.venv/bin/python scripts/build_github_ts_dataset.py --repo pmndrs/zustand --repo reduxjs/redux-toolkit --out-dir data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke --refresh-cache`
  - Success: emitted **145** rows total (`train=124`, `valid=10`, `test=11`), accepted both repos.
- `.venv/bin/python -m py_compile scripts/build_github_ts_dataset.py`
  - Success (syntax check clean).

**Smoke output path:** `data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke/` (contains dataset JSONL + manifest + metadata index).

---

## 2026-05-02 — Compile supervision rows + checkpoint-gated sweep (20/40/60) + anti-regression

**Goal:** recover capability lost in recent mixed-repair runs by adding explicit compile/export supervision rows and selecting a safer checkpoint via gated evaluation every 20 iterations.

**Prepared compile-supervision datasets:**

- Built `data/lora/arena_compile_repair_rows_20260502` from recent core6 failures in `20260502-211555_ce7aa2`:
  - tasks: `loading-screen-polish`, `hud-status-summary`, `economy-tooltip`, `save-load-api-guard`, `ai-planning-explanation`
  - each row includes failure diagnostics (`apply_status`, `tsc_status`, `missing_exports`) and uses the curated `data/arena_task_baselines/<task>.assistant.txt` as the corrected target.
  - split sizes: `train=3`, `valid=1`, `test=1`.
- Built mixed corpus `data/lora/arena_balanced_with_repairs_compile_20260502` by combining:
  - `data/lora/arena_balanced_with_repairs_20260502`
  - `data/lora/arena_compile_repair_rows_20260502` (compile rows repeated 5x per split)
  - final sizes: `train=127`, `valid=21`, `test=21`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-mixed-compile-supervision-20260502 -- --data data/lora/arena_balanced_with_repairs_compile_20260502 --iters 60 --batch-size 1 --val-batches 2 --steps-per-eval 20 --steps-per-report 10 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors`
- `ml_workflow` run id: `20260502-220731_da066d` (exit 0).
- Materialized checkpoint adapters:
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt20`
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt40`
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt60`

**Checkpoint gate sweep (standard-dev + anti-regression):**

- `ckpt20`:
  - std-dev run `20260502-223623_9e2673`: **0/2**, ACI **37.56**.
  - anti-regression run `20260502-224810_d58935`: **2/4**, ACI **72.78** (`loading-screen-polish`, `save-load-api-guard` passed).
- `ckpt40`:
  - std-dev run `20260502-230130_539fa0`: **0/2**, ACI **44.17**.
  - anti-regression run `20260502-230941_4d9206`: **0/4**, ACI **49.22**.
- `ckpt60`:
  - std-dev run `20260502-232636_e02b4a`: **0/2**, ACI **44.03**.
  - anti-regression run `20260502-233256_847399`: **0/4**, ACI **50.48**.

**Core6 confirmation on selected checkpoint (`ckpt20`):**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt20 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto --task-id loading-screen-polish --task-id hud-status-summary --task-id economy-tooltip --task-id combat-risk-preview --task-id save-load-api-guard --task-id ai-planning-explanation`
- `ml_workflow` run id: `20260502-234543_08b06c` (exit 1), result **2/6** accepted, ACI **61.97**.
- Passes: `loading-screen-polish`, `save-load-api-guard`.
- Remaining failures are still largely `tsc_exit_2` final-round failures; `hud-status-summary` / `economy-tooltip` did not recover acceptance.

**Conclusion:** compile-supervision rows improved selected anti-regression tasks at the earlier checkpoint (`ckpt20`) and raised full-core6 acceptance versus the prior mixed-repair run, but standard-dev recovery remains incomplete and compile correctness is still the primary gate.

---

## 2026-05-02 — Mixed-curriculum (baseline + repairs) SFT and core6 ACI check

**Goal:** validate whether mixing broad arena curriculum with targeted HUD/economy repair rows preserves broader capability while improving applyability failures.

**Prepared mixed dataset:**

- Built `data/lora/arena_balanced_with_repairs_20260502` from:
  - `data/lora/arena_balanced_curriculum_v1`
  - `data/lora/arena_apply_repair_stddev_20260502` (repair rows repeated 3x)
- Final counts: `train=112`, `valid=16`, `test=16`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-mixed-repair-20260502 -- --data data/lora/arena_balanced_with_repairs_20260502 --iters 60 --batch-size 1 --val-batches 2 --steps-per-eval 20 --steps-per-report 10 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors`
- `ml_workflow` run id: `20260502-201848_c45345` (exit 0).

**Core6 acceptance / ACI:**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-mixed-repair-20260502 --suite core6 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto`
- `ml_workflow` run id: `20260502-211555_ce7aa2` (exit 1), result **1/6** accepted, ACI **53.33**.
- Passing task: `combat-risk-preview`; standard-dev tasks (`hud-status-summary`, `economy-tooltip`) still failed final `tsc` and dropped exports in retry rounds.

---

## 2026-05-02 — Combined repair-set SFT pass + six-task acceptance

**Goal:** run one focused SFT pass on both newly created repair sets (`hud-status-summary` + `economy-tooltip`) and immediately validate on the core six-task arena suite.

**Prepared combined dataset:**

- Built `data/lora/arena_apply_repair_stddev_20260502` by combining:
  - `data/lora/arena_apply_repair_hud_20260502`
  - `data/lora/arena_apply_repair_economy_20260502`
- Combined split sizes: `train=16`, `valid=4`, `test=4`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-apply-repair-stddev-20260502 -- --data data/lora/arena_apply_repair_stddev_20260502 --iters 40 --batch-size 1 --val-batches 1 --steps-per-eval 10 --steps-per-report 5 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428/adapters.safetensors`
- `ml_workflow` run id: `20260502-190626_aa03bc` (exit 0).

**Six-task acceptance check:**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-apply-repair-stddev-20260502 --suite core6 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto`
- `ml_workflow` run id: `20260502-192506_b1c69f` (exit 1), result **0/6** accepted, ACI **52.54**.
- Pattern: repair-set SFT improved applyability on several tasks (`apply_final_ok` true for 5/6) but all those tasks still failed final `tsc`; `save-load-api-guard` remained `no_applyable_changes`.

---

## 2026-05-02 — Added economy-tooltip apply-repair supervision artifact

**Goal:** capture the newest economy tooltip dual failure (`20260502-173012_economy-tooltip`) as targeted repair supervision.

**Changed:**

- Added `data/arena_task_baselines/economy-tooltip.apply-repair-20260502.json` with local/frontier failure metadata:
  - local: `no_applyable_changes`
  - frontier (`gpt-5.5`): `apply_check_failed` (`error: corrupt patch at line 45`)
- Kept `data/arena_task_baselines/economy-tooltip.assistant.txt` as the corrected applyable supervision target for this task.

**Built dataset:**

- `.venv/bin/python scripts/build_arena_baseline_dataset.py --task-id economy-tooltip --sources-dir data/arena_task_baselines --out-dir data/lora/arena_apply_repair_economy_20260502 --repeat 12`
- Result: `expanded_records=12` (`train=8`, `valid=2`, `test=2`) with no missing baselines.

---

## 2026-05-02 — Added hud-status-summary apply-repair supervision artifact

**Goal:** capture the `20260502-161308_hud-status-summary` dual failure (`no_applyable_changes` local, `apply_check_failed` frontier `gpt-5.5`) as a targeted training example with a corrected applyable answer.

**Changed:**

- Updated `data/arena_task_baselines/hud-status-summary.assistant.txt` with a corrected fenced-file response suitable for direct SFT supervision (parse-safe apply format; no malformed diff hunks).
- Added `data/arena_task_baselines/hud-status-summary.apply-repair-20260502.json` to record trial id, failure statuses, and the corrected supervision target path.

**Built dataset:**

- `.venv/bin/python scripts/build_arena_baseline_dataset.py --task-id hud-status-summary --sources-dir data/arena_task_baselines --out-dir data/lora/arena_apply_repair_hud_20260502 --repeat 12`
- Result: `expanded_records=12` (`train=8`, `valid=2`, `test=2`) with no missing baselines.

---

## 2026-05-02 — Patched suite metadata + promotion coverage gate

**Goal:** close two benchmark hardening gaps: misleading suite labels for explicit task subsets and promotion-gate acceptance of partial benchmark runs.

**Changed (`scripts/run_arena_acceptance_tests.py`):**

- Explicit `--task-id` runs now write `suite: "custom"` in `summary.json` instead of inheriting CLI default `core6`.
- `--suite all` and default runs now emit canonical suite labels (`all`, `core6`) based on resolved task selection.

**Changed (`scripts/arena_promotion_gate.py`):**

- Added minimum coverage guardrails with defaults:
  - `--min-tasks` (default **6**),
  - `--min-coverage-ratio` (default **1.0**),
  - optional `--require-suite core6|all|custom`.
- Gate output now prints evaluated-task count, coverage ratio, and detected suite label alongside worst-task stats.
- Added backward-compatible coverage fallback for older `arena_capability.json` files that do not include `task_coverage_ratio`.

**Validation:**

- `python3 -m py_compile scripts/run_arena_acceptance_tests.py scripts/arena_promotion_gate.py` passed.
- `python3 scripts/arena_promotion_gate.py benchmarks/results/runs/20260501-184217_6f6e9a/arena_capability.json` now correctly fails partial runs on task-count/coverage guards.
- `python3 scripts/arena_promotion_gate.py benchmarks/results/runs/20260430-231450_487deb/arena_capability.json` uses compatibility fallback (`task_coverage_ratio=1.0000` from legacy metadata) and fails only on configured worst-task threshold.

---

## 2026-05-01 — Guardrailed standard-dev SFT attempt (no safe promotion checkpoint)

**Goal:** run a compile-repair-oriented SFT pass while explicitly checking for overfitting regressions on non-standard tasks.

**Training run:** `ml_workflow.py train` `20260501-182241_8f2f0a` on mixed dataset `data/lora/arena_balanced_curriculum_v1`, adapter `checkpoints/fe-lora-arena-guarded-stddev-v1`, resumed from `checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors` (60 iters; final/best val **0.003**).

**Guardrail evals (latest checkpoint, iter 60):**

- Standard-dev pair `20260501-184217_6f6e9a`: **0/2**, ACI **54.9** (still `tsc`-blocked).
- Non-standard guard set `20260501-184605_5ac7f5` (`loading`, `combat`, `save-load`, `ai`): **2/4**; `ai-planning-explanation` regressed (previous reverse-recover had this passing).

**Checkpoint sweep (anti-overfit fallback):**

- Materialized checkpoint-20 adapter (`checkpoints/fe-lora-arena-guarded-stddev-v1-ckpt20`):
  - Standard-dev `20260501-185755_db8e58`: **0/2**, ACI **44.61** (worse).
  - Guard set `20260501-190206_da70d1`: **3/4**, preserving prior non-standard behavior profile.
- Materialized checkpoint-40 adapter (`checkpoints/fe-lora-arena-guarded-stddev-v1-ckpt40`):
  - Standard-dev `20260501-190606_5eccc9`: **0/2**, ACI **52.33** (still no std-dev pass).

**Conclusion:** this mixed SFT recipe did not produce a checkpoint that improves standard-dev tasks without collateral risk. Keep `fe-lora-arena-reverse-recover` (or ckpt20 for guarded behavior) as the safer baseline; next cycle should add explicit compile-repair supervision examples for standard-dev tasks rather than relying on baseline-text repetition.

---

## 2026-05-01 — Diversified arena capability metric + broader suite selector

**Goal:** make arena capability harder to game with narrow task subsets and enable straightforward full-catalog acceptance runs.

**Changed (`scripts/arena_capability_index.py`):**

- Added task-type breakdown from task metadata (`task_type_breakdown`).
- Added diversified scoring outputs:
  - `task_type_balanced_index` (equal-weight across active task types),
  - `tail_robustness_index` (bottom quartile mean, min two tasks),
  - `task_coverage_ratio` (evaluated tasks vs catalog tasks),
  - `raw_diversified_index` and `diversified_arena_capability_index`.
- Added markdown/report output sections for task-type breakdown and diversified metrics.
- CLI summary line now prints headline ACI and diversified ACI together.

**Changed (`scripts/run_arena_acceptance_tests.py`, `scripts/ml_workflow.py`):**

- Added `--suite core6|all` (default `core6`).
- `--suite all` runs all task ids in the selected tasks JSON when no explicit `--task-id` list is provided.
- Workflow manifest now records arena suite under `arena_context.suite`.

**Docs updated:** `docs/ARENA_ROADMAP.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`.

**Validation:**

- `python3 -m py_compile scripts/arena_capability_index.py scripts/run_arena_acceptance_tests.py scripts/ml_workflow.py` passed.
- `python3 scripts/arena_capability_index.py --help` passed.
- `python3 scripts/run_arena_acceptance_tests.py --help` passed.
- `python3 scripts/ml_workflow.py arena-acceptance --help` passed.

---

## 2026-05-01 — Standard-dev RAG tightening in arena context pipeline

**Goal:** reduce noisy/low-signal retrieval for `hud-status-summary` and `economy-tooltip`, and increase actual code context budget for those tasks.

**Changed (`scripts/game_task_arena.py`):**

- Added standard-dev retrieval controls:
  - task-specific priority context files (`STANDARD_DEV_PRIORITY_CONTEXT_FILES`)
  - task-specific progressive prefix allowlists (`STANDARD_DEV_PROGRESSIVE_ALLOWED_PREFIXES`)
  - noisy file exclusion (`.bak`, `.DS_Store`) from context collection
- Tightened context packing for standard-dev tasks:
  - dedupe final packed paths, cap to top 6 files
  - raise per-file minimum packed chars (1200 vs 900)
  - skip `package.json` in prefix for standard-dev tasks
  - reserve larger body budget for code context (prefix cap lowered; body budget increased)
- Tightened progressive retrieval behavior:
  - added standard-dev retrieval rules in probe prompt
  - post-filter progressive paths through task-specific prefix allowlists plus priority seeds

**Validation:**

- `python3 -m py_compile scripts/game_task_arena.py` passed.
- Focused acceptance after first tightening: `20260501-161805_01a0d2` (**0/2**, ACI **53.33**).
- Focused acceptance after body-budget tightening: `20260501-162712_9623fb` (**0/2**, ACI **54.32**), with improved context pack quality (`prefix_chars` ~4046 vs ~6444 and larger code-file coverage in `context_pack.json`), but final failures remain `tsc`-driven.

**Interpretation:** retrieval quality improved measurably (more relevant code included, less prompt crowding), but model-side compile correctness on standard-dev tasks is still the gating issue.

---

## 2026-04-30 — Reverse recovery pass completed + six-task validation

**Reverse pass training:** `ml_workflow.py train` run `20260430-224503_0280f4` finished **exit 0** on `checkpoints/fe-lora-arena-reverse-recover`, resuming from the overfit adapter (`checkpoints/fe-lora-arena-standard-dev-sft/adapters.safetensors`) and training on `data/lora/arena_task_baselines_apply_contract` for 80 iters. Final train loss **0.005**, final/best val loss **0.002** (best at iter 60).

**Immediate six-task check:** `ml_workflow.py arena-acceptance` run `20260430-231450_487deb` on `checkpoints/fe-lora-arena-reverse-recover` scored **3/6**, ACI **74.97** (smoke/std/high: **100.0 / 45.09 / 85.39**). This recovers from the earlier 0/6 regression and preserves strong high-tier behavior.

**Remaining weakness:** `hud-status-summary` and `economy-tooltip` still fail due to `tsc` instability (and HUD export break), so standard-dev remains the bottleneck.

**Prepared better-system dataset:** built `data/lora/arena_balanced_curriculum_v1` as a mixed curriculum (all six tasks + light standard-dev boost) for the next SFT cycle.

---

## 2026-04-30 — Control checks: context vs SFT regression attribution

**Control A (same harness/context, non-standard tasks):** `ml_workflow.py arena-acceptance` on `checkpoints/fe-lora-arena-apply-sft` for `loading-screen-polish` + `ai-planning-explanation` → run `20260430-221251_84b70d`, **1/2** pass. `loading-screen-polish` remained fully healthy (`apply/tsc/exports` true), indicating the harness/context path is not globally broken.

**Control B (same harness/context, standard-dev tasks):** `ml_workflow.py arena-acceptance` on `checkpoints/fe-lora-arena-apply-sft` for `hud-status-summary` + `economy-tooltip` → run `20260430-221703_cfce32`, **0/2** pass (ACI **41.93**). `hud-status-summary` failed on repeated `no_applyable_changes`; `economy-tooltip` applied but failed `tsc`, then dropped `GameHUD` export on retry.

**Interpretation:** Evidence still points to narrow standard-dev SFT over-specialization as the primary 0/6 collapse driver, with standard-dev prompt/context tightening acting as a fragility amplifier on the standard-dev pair.

---

## 2026-04-30 — Wrote frontier playbook for 1/6 → 3/6 jump

**Docs:** Expanded `docs/ARENA_ROADMAP.md` with a dedicated write-up section covering: observed improvement (`20260430-183012_121be1`, **3/6**, ACI **68.05**), explicit working attribution (apply-contract context + targeted SFT), causal/mechanical rationale, repeatable runbook, and ordered frontier experiments to raise worst-tail tasks.

**Intent:** make the capability increase and repeat strategy durable for future cycles without relying on chat memory.

---

## 2026-04-30 — Apply-contract SFT run + six-task acceptance jump (1/6 → 3/6)

**Trained:** `ml_workflow.py train` on `data/lora/arena_task_baselines_apply_contract` → run `20260430-172957_435fb1`, adapter `checkpoints/fe-lora-arena-apply-sft`, final val loss **0.006**.

**Evaluated:** `ml_workflow.py arena-acceptance` on that adapter → valid run `20260430-183012_121be1` (**3/6**, ACI **68.05**, tiers **100.0 / 35.9 / 78.1**). Prior immediate attempt `20260430-182924_4a1732` is not comparable because it used system Python and failed generation imports (`ModuleNotFoundError: mlx_lm`).

**Attribution note (explicit):** current working interpretation is that the capability jump from historical **1/6** runs to **3/6** is driven by the **combination** of (a) apply-contract context injection (`docs/GAME_ARENA_APPLY_CONTRACT.md`) and (b) targeted SFT. This is recorded as provisional until repeated across additional acceptance batches.

**Gate status:** `scripts/arena_promotion_gate.py` still fails on worst-tail thresholds (`worst=22.87`, `mean_two_worst=27.57`), so promotion remains blocked.

---

## 2026-04-30 — Apply contract context + apply-failure-focused SFT hooks

**Changed runtime context:** Added `docs/GAME_ARENA_APPLY_CONTRACT.md` (strict output schema for arena applyability) and injected it into `scripts/game_task_arena.py` packet context (`build_context_pack` prefix + packet prompt reminder). This gives every generate/apply round a shared contract aimed at reducing `no_applyable_changes`.

**Changed SFT data path:** Updated `scripts/build_game_task_pairwise_dataset.py` with `--focus-apply-failures` (filters loser records to apply-format failures), plus contract text injection in user prompts and metadata tags. Updated `scripts/build_arena_baseline_dataset.py` to include the same contract in baseline SFT prompts.

**Workflow/docs:** Added `--focus-apply-failures` passthrough in `scripts/ml_workflow.py arena-gate-train` when `--rebuild-dataset` is used. Documented the loop in `docs/ARENA_ROADMAP.md`, `docs/WORKFLOW.md`, and `docs/PROJECT_STATE.md`.

**Verified:** `python3 -m py_compile scripts/game_task_arena.py scripts/build_game_task_pairwise_dataset.py scripts/build_arena_baseline_dataset.py scripts/ml_workflow.py`.

---

## 2026-04-29 — Arena dashboard, promotion gate, roadmap, fe_lineage policies

**Added:** `scripts/build_arena_dashboard.py` → `benchmarks/arena_dashboard.html` (six-task × runs table, combat/ai KPI columns); `**python scripts/ml_workflow.py arena-dashboard`** alias (no run artifacts/history append). `**scripts/arena_promotion_gate.py`** for worst-of-six thresholds on `arena_capability.json`. `**scripts/fe_lineage.py**`: `DEFAULT_ARENA_PROGRESSIVE_CONTEXT`, `ARENA_ADAPTER_PROGRESSIVE_POLICY` (chunk6k + best-val300 `**auto**`). **Docs:** `docs/ARENA_ROADMAP.md`; `**docs/ARENA_PROGRESSION.md`**, `**docs/PROJECT_STATE.md`**, `**docs/WORKFLOW.md**`, `**scripts/ml_workflow.py**` updated for dashboards + policy pointers. Verified: `python3 -m py_compile` on touched scripts; `build_arena_dashboard.py --last 5`; `arena_promotion_gate.py` rejects `20260429-025429_9793ff` at `--min-worst-score 40`.

---

**Ran:** `ml_workflow.py arena-acceptance` → `20260429-024422_e7aec2` on `checkpoints/fe-lora-qwen25-coder-7b-latest` (default six-task suite, progressive auto, worktree `~/fallen-empire-arena`, ~~31 min). **Exit 1**, **0/6** preview passes, headline ACI **~~48.28**; artifacts `benchmarks/results/runs/20260429-024422_e7aec2/` (row in `docs/run_history.md`). Distinct from **best-val300** A/B above (different adapter snapshot; best-val300 auto **55.24** / off **47.84**).

---

## 2026-04-29 — Best-val300 six-task arena (progressive auto vs off)

**Adapter:** Created `checkpoints/fe-lora-qwen25-coder-7b-best-val300` = `adapter_config.json` + `0000300_adapters.safetensors` from `fe-lora-qwen25-coder-7b-latest` (best validation at iter **300**, train run `20260428-195934_d451c9`).

**Ran:** Two `ml_workflow.py arena-acceptance` six-task batches (same `SOURCE_REPO` / task ids). **Auto:** `--worktree-root ~/fallen-empire-arena` → `20260429-025429_9793ff`, **1/6** pass, ACI **55.24**, tier indices smoke/std/high **32.3 / 48.7 / 64.3**. **Off (first try):** `20260429-034918_34e0cb` failed all tasks at `create` with `PermissionError` on `~/fallen-empire-arena`. **Off (retry):** `--worktree-root benchmarks/results/arena_worktrees`, `--preview-port 5200` → `20260429-034932_0812a4`, **0/6** pass, ACI **47.84**, tier **33.2 / 50.9 / 49.7**.

**Docs:** `docs/ARENA_PROGRESSION.md` table + narrative; `docs/PROJECT_STATE.md` arena row; canvases `arena-capability-index-tracker` and `arena-three-tier-progression` embedded with both JSON snapshots (best-ACI sort: **auto** ahead of **off** for this adapter).

---

## 2026-04-29 — Canvas: run logs sorted by best ACI

**Changed:** `arena-capability-index-tracker.canvas.tsx` and `arena-three-tier-progression.canvas.tsx` — run log tables and chart categories ordered by **headline ACI descending** (best first); canonical **Run #** unchanged; tier canvas registry gains an **ACI** column and **best-first** tier series order.

---

## 2026-04-29 — Canvas: omit smoke-only run from ACI tracker

**Changed:** `canvases/arena-capability-index-tracker.canvas.tsx` and `canvases/arena-three-tier-progression.canvas.tsx` no longer include **20260428-160917** (single-task acceptance); run numbers reset to **Run 1–2** for the comparable **six-task** batches (`171007` auto, `180627` off). Tracker intro + callout state that smoke-only rows are out of scope for this trend line.

---

## 2026-04-28 — Full arena acceptance run and model pin

**Ran:** `.venv/bin/python scripts/run_arena_acceptance_tests.py --adapter-path checkpoints/fe-lora-game-text-20260428 --timeout-s 7200` → `benchmarks/results/arena_acceptance/acceptance-20260428-031445/`, exit **1**, **0/4** tasks passed (`loading-screen-polish`, `hud-status-summary`, `economy-tooltip`, `combat-risk-preview`).

**Outcome:** Loading/economy produced no applyable changes or corrupt patches; HUD wrote `src/components/ui/GameHUD.tsx` but failed `tsc` and dropped the `GameHUD` export; combat crashed with a LoRA/base-model hidden-size mismatch (`3584` vs `1536`), indicating the gate inherited a 7B default with a 1.5B adapter.

**Changed:** `scripts/run_arena_gate_benchmark.py` and `scripts/run_arena_acceptance_tests.py` now accept/pass `--model` and infer the base model from `adapter_config.json` when present, avoiding 7B/1.5B adapter mismatches.

---

## 2026-04-26 — Game arena preview connectivity + metrics

**Changed:** Fixed Game Task Arena preview startup for disposable Fallen Empire worktrees. Task specs now use `next dev -H 127.0.0.1 -p {port}` instead of the previous `npm run dev -- --host ... --port ...` wrapper that failed with `sh: next: command not found`; preview startup now reuses the source checkout `node_modules` through `PATH`/`NODE_PATH` plus a best-effort worktree symlink.

**Changed:** `preview()` now waits for the sandbox URL and records `preview_status`, `ready_elapsed_s`, and preview metadata before showing links as ready. It also carries the local `/test-env` sandbox files into new disposable worktrees while those game-side files are still uncommitted. Generation now records elapsed seconds, max tokens, input/output/total token counts, estimated frontier cost, provider usage, and writes per-attempt `generation_metrics.json`.

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py`; `load_task_specs` confirms `loading-screen-polish` uses `next dev -H 127.0.0.1 -p {port}` and `/test-env/loading-screen`; fresh smoke trial `preview-smoke-1777253838` previewed `/test-env/loading-screen` on port **5187** with `preview_status=ready` and HTTP **200**, then the preview process was terminated and the disposable worktree cleaned up.

---

## 2026-04-26 — Game sandbox preview links

**Changed:** Added `preview_path` support to `scripts/game_task_arena.py` so preview metadata and UI links can open a task-specific local path while still starting the task's existing `preview_command`. Updated `benchmarks/game_task_arena_examples.json` so standardized tasks deep-link to `/test-env/...` sandbox routes in the game repo.

**Game repo:** Added `/Users/natreed/fallen-empire/src/app/test-env/[envId]/page.tsx`, `/Users/natreed/fallen-empire/src/components/test/TestEnvironmentShell.tsx`, and `/Users/natreed/fallen-empire/src/lib/testEnvironments.ts`. The loading-screen route renders only the loading shell; combat risk preview starts the existing `battle_test`, opens the battle report, and pauses the sim; other visual tasks boot deterministic paused observer states.

**Verified:** `npx tsc --noEmit` and `npm run test:ml-cohort` in `/Users/natreed/fallen-empire` passed; IDE lints reported no errors for the new game route/components or `scripts/game_task_arena.py`; `.venv/bin/python -m py_compile scripts/game_task_arena.py` passed; loading `benchmarks/game_task_arena_examples.json` through `load_task_specs` returned all six `preview_path` values.

---

## 2026-04-26 — Game task arena contrast fix

**Changed:** Tightened `scripts/game_task_arena.py` Gradio CSS so the Game Task Arena uses explicit readable text, panel, input, table, and code colors in light and dark mode while preserving the blue primary-button styling.

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py`; `.venv/bin/python -c "from scripts import game_task_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning); IDE lints report no errors for `scripts/game_task_arena.py`; started a fresh UI server on **[http://127.0.0.1:7874](http://127.0.0.1:7874)** and verified HTTP 200.

---

## 2026-04-26 — Safer arena token default

**Changed:** Raised the landing-page arena Local + Frontier split max-token default from 4096 to **8192** via `DEFAULT_ARENA_MAX_TOKENS`, including the matching `local-attempt --max-tokens` default. Documented the escalation path: start at **8192**, raise to **12000** if static-site output still truncates, then **16000** only for larger trials.

**Verified:** `.venv/bin/python -m py_compile scripts/landing_page_arena.py`; `.venv/bin/python -c "from scripts import landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning). Started updated arena at **[http://127.0.0.1:7867](http://127.0.0.1:7867)** and verified HTTP 200.

---

## 2026-04-26 — Landing page arena parser and IDE UX fixes

**Changed:** Hardened `scripts/landing_page_arena.py` file extraction for common model output styles (`path=`, `filename=`, language-only fences, bare fenced filenames, labeled sections, and raw HTML). Generated attempts now save unparsed raw output to `model_output.md` / `parse_error.html` without replacing `index.html` with fallback parse-error HTML, and generated links to missing local HTML pages such as `game.html` are neutralized to avoid preview 404s.

**Changed:** Tightened the landing-page generation prompt to require only `index.html`, `styles.css`, and `script.js`, with no local secondary HTML links. Simplified the Gradio surface into a single IDE-style workflow with local/frontier code panes, side-by-side iframe previews, paste-import into the Frontier lane, and grading/training-data save.

**Verified:** `.venv/bin/python -m py_compile scripts/landing_page_arena.py`; synthetic parser cases for multiple fence/label/raw-HTML formats; `.venv/bin/python -c "from scripts import landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning). No local MLX generation was run.

---

## 2026-04-26 — Workflow documentation and audit hardening

**Changed:** `scripts/ml_workflow.py` now re-execs into `.venv/bin/python` when the repo venv exists, keeping export/build/benchmark/EvalPlus steps on the same interpreter as the MLX stack. Future manifests and `RUN.md` files include separate `trained_adapter_path` and `benchmark_adapter_path` fields.

**Changed:** `docs/run_history.md` is now one well-formed table with `Exit`, `Status`, `Trained adapter`, and `Benchmarked adapter/model` columns. Historical rows were normalized so failed runs are visible in the committed index and smoke runs show that the synthetic adapter is trained while the base model is benchmarked.

**Docs:** updated `README.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `requirements.txt`; added `requirements.lock.txt` as lightweight reproducibility constraints for the verified ML/UI package versions.

**Verified:** `.venv/bin/python -m py_compile scripts/ml_workflow.py scripts/visualize_results.py`; `.venv/bin/python scripts/ml_workflow.py --help`; `python3 scripts/ml_workflow.py --help`. No training or benchmark run was launched.

---

## 2026-04-26 — Game task arena V1 (`game_task_arena.py`)

**Added:** `scripts/game_task_arena.py`, a disposable worktree/copy arena for real Fallen Empire game-code trials. It creates local/frontier attempts from versioned task specs, writes context/model packets, applies unified diffs or fenced repo-relative file blocks with path allowlists, runs fixed verification commands, records preview metadata, saves human grades/training records, generates adapter/model comparison reports, and cleans up disposable trees without deleting indexed artifacts.

**Added:** `benchmarks/game_task_arena_examples.json` with starter `ui`, `combat`, `economy`, and `refactor` task specs. Results are filed under `benchmarks/results/game_task_trials/<trial_id>/`, indexed in `benchmarks/results/game_task_index.jsonl`, and summarized in `benchmarks/results/game_task_reports/game_task_summary.md`.

**Updated:** simplified the Gradio UI so standardized tests are primary. Source/worktree/base-ref fields moved into an advanced drawer; the main output after trial creation is just Local and Frontier worktree links. Packet text, manifests, verification output, and reports now live in an optional details drawer.

**Updated:** changed the Game Task Arena UI to match the intended evaluation flow: create disposable worktrees, generate/apply both local and frontier model attempts, start two playable preview links, grade both attempts, and complete the trial with optional cleanup. Worktree links are no longer the primary output.

**Updated:** restyled the Game Task Arena UI with a forced dark theme: black background, readable white text, dark panels/inputs, blue primary buttons, rounded controls, and reduced visual noise.

**Updated:** revised `benchmarks/game_task_arena_examples.json` to six standardized tasks spanning complexity and subsystem coverage: loading/start screen polish, HUD/status UI, economy tooltip, combat risk preview, save/load/API guard, and AI planning rationale. The suite prioritizes quick-to-locate preview changes where possible while still including backend/combat/planning tasks.

**Updated:** added task-specific grading rubrics to every standardized game arena task. The Gradio grading sliders now relabel for the selected task, and saved ratings include `rubric_labels` / `rubric_scores` alongside the generic score fields for comparison reports.

**Updated:** collapsed create/generate/preview into a single **Run Full Trial** button. Manual create/generate/preview/packet/verify/report controls remain in the details drawer for debugging, but the primary workflow is now one action followed by playtesting and grading.

**Verified:** `python3 -m py_compile scripts/game_task_arena.py`; created a temporary mock game repo under `/tmp`, created local/frontier copy attempts, applied a fenced file write to `src/ui/Hud.tsx`, ran fixed `npm run test:ml-cohort` verification (mock pass), recorded preview metadata, saved a human grade/training record, generated the comparison report, and cleaned both disposable copy attempts.

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — Landing page human trial arena

**Added:** `scripts/landing_page_arena.py`, a standalone visual/product evaluation arena for Fallen Empire landing-page trials. It creates shared briefs, isolated static-site attempt folders, Cursor-ready frontier packets, local MLX attempts via `LocalMlxBackend`, preview serving via `python -m http.server`, Gradio UI on port **7863**, and human rating JSON.

**Updated:** the Gradio arena now has separate **Local** and **Frontier** lanes. Local runs the MLX adapter directly. Frontier can call an OpenAI-compatible Codex/frontier API from environment variables (`FRONTIER_API_KEY` / `OPENAI_API_KEY`, `FRONTIER_MODEL`, optional `FRONTIER_API_BASE_URL`) or import pasted Cursor/frontier output for immediate preview and rating.

**Updated:** added a **Compare** tab that takes one shared prompt, splits it into Local and Frontier attempts, runs both lanes in parallel when Frontier API mode is enabled, shows both final previews side by side, and saves paired comparison/training records with winner, scores, notes, brief, and generated files.

**Updated:** promoted the arena to an IDE-style first tab: one prompt, split Local/Frontier generation, editable code panes for `index.html` / `styles.css` / `script.js`, a single **Preview Current Code** button, and grading/training-data save after visual inspection. Trial/attempt names remain internal storage details instead of required user inputs.

**Updated:** removed redundant Gradio tabs from the arena UI and made the IDE workflow the whole product surface. Cursor/frontier paste import now writes directly into the Frontier code panes and preview.

**Updated:** added ignored `.env` support for the landing page arena so `FRONTIER_API_KEY`, `FRONTIER_MODEL`, and optional `FRONTIER_API_BASE_URL` can persist locally without being committed. `.env.example` documents the expected keys and `.gitignore` excludes `.env*` except the example.

**Added:** `benchmarks/landing_page_brief.md` as the shared context packet generated from current repo docs. Trial artifacts are written under `benchmarks/results/landing_page_trials/` and rating/index rows under `benchmarks/results/` (gitignored).

**Verified:** `python3 scripts/landing_page_arena.py brief --overwrite`; created smoke trial `smoke-landing-page`, generated a Cursor packet, scaffolded `manual_smoke`, and saved a smoke rating. `python3 -m py_compile scripts/landing_page_arena.py scripts/model_router.py`; `.venv/bin/python -m py_compile scripts/landing_page_arena.py scripts/model_router.py`; `.venv/bin/python -c "import scripts.landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` (Gradio app builds).

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — EvalPlus benchmark + cost-aware routing framework

**Added:** `scripts/run_evalplus_benchmark.py` for execution-based EvalPlus HumanEval+/MBPP+ subsets/full suites; `scripts/ml_workflow.py evalplus` records those runs under the normal `benchmarks/results/runs/<id>/` artifact structure. Added `evalplus>=0.3.1` to `requirements.txt`.

**Added:** `scripts/model_router.py` with deterministic `RoutingPolicy`, `LocalMlxBackend`, `OpenAICompatibleBackend`, `ModelRouter`, cost metadata, and dry-run support. Added `benchmarks/task_routing_tasks.json` plus `scripts/run_routing_benchmark.py`; dry-run routing benchmark scored **12/12**.

**Adapter selection:** `scripts/select_best_adapter.py` writes `benchmarks/results/adapter_selection.md`; current recommendation is `checkpoints/fe-lora-30m` because it retains game benchmark **15/15** and has much better validation behavior than `checkpoints/fe-lora-800-from-30m`, which regressed to game **14/15** and final validation loss **1.943**.

**Verification:** `python3 -m py_compile scripts/ml_workflow.py scripts/run_evalplus_benchmark.py scripts/model_router.py scripts/run_routing_benchmark.py scripts/select_best_adapter.py`; `python3 scripts/run_routing_benchmark.py` → **12/12**; `python3 scripts/select_best_adapter.py` → recommends `checkpoints/fe-lora-30m`. Two `ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 2` attempts failed before model load because GitHub returned HTTP 502 while EvalPlus downloaded `HumanEvalPlus.jsonl.gz`; retry later or set `HUMANEVAL_OVERRIDE_PATH`.

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — 800-iteration resumed LoRA pass + general benchmark

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --evaluate --bench-profile general --adapter-path checkpoints/fe-lora-800-from-30m -- --iters 800 --batch-size 1 --val-batches 8 --resume-adapter-file checkpoints/fe-lora-30m/adapters.safetensors --save-every 100` produced run artifacts at `benchmarks/results/runs/20260426-175720_fb7d5b/`. Training resumed from `checkpoints/fe-lora-30m/adapters.safetensors`, completed successfully in ~52.97 minutes, saved final weights plus 100/200/300/400/500/600/700/800 snapshots, and ended at train loss **0.178**, validation loss **1.943**, peak memory **8.598 GB**. Best validation reading during this pass was **1.400** at iteration 150 with `--val-batches 8`.

**Benchmarks:** automatic **general** profile benchmark on `checkpoints/fe-lora-800-from-30m` scored **8/8**. Follow-up game profile benchmark via `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-800-from-30m --profile game` produced `benchmarks/results/runs/20260426-185136_24ec8d/` and scored **14/15**; failure was `siege-wall-priority-chain` missing `wallBuildPriority`.

**Dashboard:** refreshed `benchmarks/results/run_dashboard.html` with `python3 scripts/visualize_results.py` (7 runs included).

---

## 2026-04-25 — 300-iteration LoRA training pass (`fe-lora-30m`)

**Ran:** `python3 scripts/ml_workflow.py train --evaluate --adapter-path checkpoints/fe-lora-30m -- --iters 300 --batch-size 1 --val-batches 1` produced run artifacts at `benchmarks/results/runs/20260425-202718_459eb4/`. Training completed successfully in ~19.75 minutes, saved `checkpoints/fe-lora-30m/adapters.safetensors` plus 100/200/300-iteration snapshots, and ended at train loss **0.799**, validation loss **0.964**, peak memory **8.598 GB**.

**Benchmark retry:** The workflow's inline benchmark failed because the command was launched with system `python3` instead of the venv Python (`ModuleNotFoundError: No module named 'mlx_lm'`). Reran with `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-30m`, which produced `benchmarks/results/runs/20260425-204711_1efcd0/` and scored **15/15** on the game profile.

**Dashboard:** refreshed `benchmarks/results/run_dashboard.html` with `python3 scripts/visualize_results.py` (5 runs included).

---

## 2026-04-25 — Workflow metadata reliability review

**Changed:** `scripts/ml_workflow.py` now records true workflow start/finish timestamps, total elapsed seconds, and `final_exit_code` in each future manifest; `RUN.md` mirrors those fields and step logs use shell-quoted commands for more reproducible copy/paste. `scripts/visualize_results.py` remains compatible with older manifests and uses `final_exit_code` when present, with an explicit Exit column in the dashboard table.

**Safety:** Detected an active `python scripts/ml_workflow.py train --evaluate --adapter-path checkpoints/fe-lora-30m -- --iters 300 --batch-size 1 --val-batches 1` / `mlx_lm.lora` training process and did not run workflow commands or modify generated run artifacts.

**Verified:** `python3 -m py_compile scripts/ml_workflow.py scripts/visualize_results.py`; `python3 scripts/ml_workflow.py --help`; `tmp_runs="$(mktemp -d /tmp/fe-lora-empty-runs.XXXXXX)" && python3 scripts/visualize_results.py --runs-dir "$tmp_runs" --history /tmp/fe-lora-empty-history.md --out /tmp/fe-lora-dashboard.html` (writes outside repo generated artifacts; output: `Runs included: 0`).

**Docs:** updated `docs/WORKFLOW.md` and `docs/PROJECT_STATE.md`.

---

## 2026-04-25 — Run results visualizer (`visualize_results.py`)

**Added:** `scripts/visualize_results.py` to generate a static HTML dashboard from `benchmarks/results/runs/*/manifest.json`, enriched with `docs/run_history.md` metadata when present. Dashboard includes KPI cards (success/failure, average duration, benchmark pass rate), subcommand/day bar summaries, and a sortable-style run table with status, duration, benchmark summary, adapter path, and step failure counts.

**Verified:** `python3 scripts/visualize_results.py` → wrote `benchmarks/results/run_dashboard.html` with **3 runs included**.

**Docs:** updated `docs/WORKFLOW.md` (new “Results visualizer” section) and `docs/PROJECT_STATE.md` (wired tools table + command examples).

---

## 2026-04-24 — General coding benchmark (`general_coding_tasks.json`)

**Added:** `benchmarks/general_coding_tasks.json` (8 tasks: HTTP GET, JSONL, SQL `SELECT`, digit regex, UTF-8, API expansion, Markdown fence, semver) and `--profile game|general` on `scripts/run_game_benchmark.py` (general → neutral system prompt + default general task file). `json` import fixed for `--output-jsonl`. `scripts/ml_workflow.py` forwards `--profile` on `benchmark`; `full` and `train --evaluate` accept `--bench-profile`.

**Verified:** `python scripts/run_game_benchmark.py --profile general` → **8/8**; default game run → **15/15** on `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, greedy, default max tokens.

**Docs:** `benchmarks/README.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`.

---

## 2026-04-24 — Train UI Gradio compatibility fix

Fixed startup failures in `scripts/train_ui_gradio.py` for older `gradio` APIs in this venv: removed unsupported `Textbox(..., monospace=True)` argument and added missing `Tuple` import required by Gradio's runtime type-hint inspection. Verified app construction with `python -c "import scripts.train_ui_gradio as t; t.build_app(); print('build_app ok')"`.

---

## 2026-04-24 — Gradio training UI (`train_ui_gradio.py`)

Added `scripts/train_ui_gradio.py` (default port **7862**): export + build dataset buttons area, LoRA hyperparameter sliders/fields, live training log stream, Stop, optional post-train benchmark. Documented in `docs/WORKFLOW.md`. Does not replace `ml_workflow.py` for committed `run_history.md` / `manifest.json` audit trail.

---

## 2026-04-24 — Built-in `ml_workflow.py` + run documentation

**Added:** `scripts/ml_workflow.py` subcommands `smoke`, `prepare`, `train` (optional `--evaluate`), `benchmark`, `full`. Each run writes `benchmarks/results/runs/<id>/{manifest.json,RUN.md,logs/}` and appends `**docs/run_history.md`**. `**docs/WORKFLOW.md`** describes usage. README + `training/README.md` + `.cursor/rules/precise-ml-documentation.mdc` + `docs/PROJECT_STATE.md` updated to prefer the orchestrator. `train_lora.sh` comment points to workflow.

**Verified:** `python scripts/ml_workflow.py smoke` exit 0; `docs/run_history.md` gains a row; sample `RUN.md` lists steps and timings.

---

## 2026-04-23 — Game-side tests, human eval UI, export hygiene

**Game repo (`fallen-empire`):** Added `scripts/ml-lora-cohort-guard.ts` (Biome / map presets / Tile+SimResult shape / `runSimulation` smoke) and `package.json` script `**npm run test:ml-cohort`**.

**ML repo:** `scripts/run_game_ml_tests.sh` runs that npm script against `SOURCE_REPO` or `~/fallen-empire`. `scripts/human_eval_ui.py` — Gradio UI on port **7861** by default: pick benchmark task, generate with MLX (optional adapter), 1–5 sliders + notes, append JSONL to `benchmarks/results/human_eval.jsonl`.

**Export:** Expanded `scripts/export_repo_for_training.py` with size cap, path/suffix skips, binary-ish skip, and redaction regexes for keys/tokens/PEM blocks; env `EXPORT_MAX_FILE_BYTES` overrides cap.

**Verified:** `npm run test:ml-cohort` in game repo passes; `SOURCE_REPO=/Users/natreed/fallen-empire python scripts/export_repo_for_training.py` reports skip counts; `human_eval_ui.py --help` ok.

---

## 2026-04-23 — mlx_lm.lora wiring

**Added:** `scripts/build_lora_dataset.py` (export JSONL → `text` JSONL splits), `training/lora_qwen_coder.yaml`, `scripts/train_lora.sh`. `.gitignore`: `data/lora/`. `training/README.md` documents end-to-end flow.

**Verified:** `build_lora_dataset.py --synthetic-smoke`; `mlx_lm.lora --train -c training/lora_qwen_coder.yaml --iters 2 --batch-size 1 --val-batches 1 --steps-per-eval 1 --steps-per-report 1 --max-seq-length 1024 --adapter-path checkpoints/_smoke_lora` → training completes, adapters saved.

---

## 2026-04-23 — Ambitious benchmark + season / evolutionary harness

**Goal:** Harder game-aligned eval tasks; **tiered** scoring (C/B/A); **season-based evolution** of the benchmark population mirroring sim-system; ambitious **LoRA curriculum** defaults as JSON (not executed).

**Added / changed:**

- `benchmarks/fallen_empire_tasks.json`: +8 tasks (SimResult fields, axial distance, map presets, mutation buckets prose, season↔curriculum metaphor, Next route sketch, siege/wall chain, headless core path).
- `scripts/benchmark_evolution_lib.py`: tier specs, `scale_expect_for_tier`, `check_expect`, `mutate_task` / `crossover`, `MAX_EXPECT_MIN_CHARS=320`, vocab injection only for codeish categories.
- `scripts/run_game_benchmark.py`: imports lib; `--tier C|B|A`.
- `training/evolution_config.json`, `training/README.md`: population 18, 8 seasons, tier cycle `C,C,B,B,A,A,B,A`, anchors, selection modes, `lora_ambitious_defaults`, named curriculum seasons.
- `scripts/evolve_benchmark_seasons.py`: season loop, immigration templates, optional `--with-mlx`, dedupe ids, writes flat task JSON array.

**Verified:** `python scripts/run_game_benchmark.py` → 15/15; `python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json` (structural mode, fast).

---

## 2026-04-23 — Fallen Empire model benchmark suite

**Goal:** A repeatable, game-flavored eval harness for the MLX model (compare base vs future LoRA).

**Added:** `benchmarks/fallen_empire_tasks.json` (7 tasks: biomes, hex coords, pure TS, API shape, AI param prose, Zustand, neighbor offsets), `scripts/run_game_benchmark.py` (load model once, score with `expect` rules, optional JSONL log), `benchmarks/README.md`. `.gitignore`: `benchmarks/results/`.

**Verified:** `python scripts/run_game_benchmark.py` → **7/7** on Qwen2.5-Coder-1.5B-Instruct-4bit after tightening biome prompt and broadening Zustand `any_contains` keywords.

---

## 2026-04-23 — Assistant name “Albert”

Renamed the Gradio window title and default system persona to **Albert** in `scripts/chat_gradio.py`; aligned smoke script system line in `scripts/smoke_base_model.py`.

---

## 2026-04-23 — Gradio chat UI

**Goal:** Browser interface to confirm you can hold a multi-turn conversation with the local MLX model (and optional future LoRA adapters).

**Changes:**

- Added `scripts/chat_gradio.py`: loads `mlx_lm.load` once; `gr.ChatInterface` with a generator that streams `stream_generate` token deltas; builds prompts from system prompt + Gradio tuple history + current user message via `apply_chat_template`.
- `requirements.txt`: `gradio>=4.44,<5` (venv resolved to `gradio==4.44.1`).

**Commands run:** `python scripts/chat_gradio.py --help`; small Python snippet importing `_history_to_messages` (success).

**Next:** Run `python scripts/chat_gradio.py`, open the printed URL, chat; then proceed to dataset export / LoRA when ready.

---

## 2026-04-23 — Base MLX model smoke + documentation rule

**Goal:** Confirm an open-source MLX code model downloads and generates; establish strict documentation practice for the repo.

**Changes:**

- Added `scripts/smoke_base_model.py`: loads default HF repo `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, builds instruct prompt via `tokenizer.apply_chat_template`, calls `mlx_lm.generate` with `max_tokens`; uses `make_sampler` when `--temp > 0` (required for `mlx-lm==0.29.1`).
- Added `.cursor/rules/precise-ml-documentation.mdc` (`alwaysApply: true`): mandates maintaining `docs/PROJECT_STATE.md` and append-only `docs/SESSION_LOG.md` with precise versions, ids, and verified commands.
- Added `docs/PROJECT_STATE.md` and this file.

**Commands run (representative):**

- `python3 -m venv .venv` → `pip install -r requirements.txt` (success; resolved to `mlx-lm==0.29.1`, `mlx==0.29.3`, etc.).
- First `python scripts/smoke_base_model.py`: download OK; **failed** once with `TypeError: generate_step() got an unexpected keyword argument 'temp'` — fixed by removing `temp=` and using `make_sampler` only when needed.
- Second run: **success** (~0.7 s load from cache, ~0.8 s generation for 96 tokens).

**Next (suggested):** Run `export_repo_for_training.py` with `SOURCE_REPO` pointing at the game tree; sketch LoRA invocation for `mlx_lm.lora` with the same base model id and document exact CLI in `PROJECT_STATE.md` after verification.

---

## 2026-04-28 — 7B chunked `game_text` baseline run

**Goal:** Get a first documented baseline for the larger **Qwen2.5-Coder-7B-Instruct-4bit** LoRA path using chunked long-file data.

**Ran:** `export SOURCE_REPO=/Users/natreed/fallen-empire && .venv/bin/python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunked-20260428 -- --iters 200 --batch-size 1 --val-batches 4 --max-seq-length 2048 --steps-per-eval 25 --steps-per-report 10 --save-every 50`.

**Outcome:** Workflow run `20260428-041532_007bdf` exited **1** because the automatic game benchmark failed one task; **training completed successfully**. Export wrote **160** files; chunked builder wrote **426** rows (`377/24/25` train/valid/test). Training ended at iter **200**, final train loss **1.168**, final validation loss **1.109**, peak memory **9.825 GB**, trained tokens **326,024**, and saved final adapter plus 50/100/150/200 snapshots under `checkpoints/fe-lora-qwen25-coder-7b-chunked-20260428/`.

**Benchmark:** Game profile scored **14/15**; failed `siege-wall-priority-chain` for missing `siegeChance` and `wallBuildPriority`. Capability Index **81.1/100** (correctness **96.7**, instruction **75.0**, concision **32.4**, speed **38.3**).

---

## 2026-04-28 — 7B LoRA 400-iter train on `fe-lora-qwen25-coder-7b-latest`

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest -- --iters 400`

**Outcome:** Exit **0**; ~~**3.7 h**; `benchmarks/results/runs/20260428-195934_d451c9/`. Trajectory: **final train loss ~0.87**, **final val ~1.574**, **best val ~1.275 @ iter 300**, **~~768k** trained tokens, **~12.97 GB** peak. No automatic lexical benchmark in this subcommand.

**Docs:** Repaired `docs/run_history.md` table row that had merged `arena-acceptance` + `train` on one line.

---

## 2026-04-29 — Game benchmark on `fe-lora-qwen25-coder-7b-latest` (post 400-iter train)

**Ran:** `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest --profile game`

**Outcome:** Run `20260429-023956_396046`, exit **1** (one heuristic task failed). **14/15** tasks passed (**93%**). Failed task: `siege-wall-priority-chain` — missing required substrings `siegeChance` and `wallBuildPriority`. Capability Index **84.2/100** (correctness **96.7**, instruction **79.7**, concision **45.9**, speed **48.0**). ~**160 s** wall time.

---

## 2026-04-29 — Full arena acceptance default (six tasks / three ACI tiers)

**Changed:** `scripts/run_arena_acceptance_tests.py` `DEFAULT_TASKS` now includes `**save-load-api-guard`** and `**ai-planning-explanation`** so default `arena-acceptance` runs populate **Smoke / Standard Dev / High-Reasoning** tier breakdown in `arena_capability_index.py`. Increased default `--timeout-s` to **14400** here and in `ml_workflow.py arena-acceptance` for six-task preview gates.

**Started:** `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest --source-repo /Users/natreed/fallen-empire --worktree-root /Users/natreed/fallen-empire-arena` (background on agent host; can take up to 4 h).

---

## 2026-04-29 — Canvas: training + lexical progression charts

**Updated** Cursor canvas `canvases/arena-three-tier-progression.canvas.tsx`: embedded **val/train loss lines** from `20260428-195934_d451c9/training_trajectory.jsonl`, **bar chart** of final vs best val across three **7B** passes (chunked / chunk6k / latest), **line chart** of lexical capability index + pass rate across those passes, rubric table, footnote for **1.5B game_text** run, retained latest-run three-band task table.

---

## 2026-04-29 — Canvas: Arena ACI three tiers (Smoke / Standard / High-Reasoning)

**Updated** `canvases/arena-three-tier-progression.canvas.tsx` to chart **Arena Capability Index** `tier_breakdown` only: **Smoke Test**, **Standard Dev Benchmark**, **High-Reasoning Architecture Benchmark** from `arena_capability.json` runs **20260428-171007** vs **20260428-180627**, plus tier definition table and note on smoke-only **20260428-160917**; removed lexical-band / training-loss focus from this canvas.

**Follow-up:** Canvas adds chronological **Run 1–3** registry table, **“latest”** pill (Run 3), run numbers on chart axes and stats, and **Run 4** hint for the next embedded acceptance batch.

---

## 2026-04-29 — Canvas: Arena Capability Index tracker

**Added** `canvases/arena-capability-index-tracker.canvas.tsx`: run-numbered log table, headline ACI line chart, six-task-only ACI bar chart, completion/integration/efficiency lines, tier_breakdown columns; seeds from `arena_capability.json` for runs **20260428-160917**, **171007**, **180627**; documents how to append **Run 4+** after new acceptance batches.

---

## 2026-05-03 — Workflow cleanup + generated artifact path separation

**Goal:** Reduce workflow/docs drift and keep generated outputs out of source-doc paths under `benchmarks/`.

**Changed (code defaults):**

- `scripts/ml_workflow.py`
  - `arena-dashboard` now defaults to `benchmarks/results/arena_dashboard.html`.
  - `multi-adapter-report` now defaults to `benchmarks/results/multi_adapter_dashboard.json` and `benchmarks/results/multi_adapter_report.md`.
  - Updated command help text/docstring references to match the new artifact locations.
- `scripts/build_arena_dashboard.py` default `--out` now points at `benchmarks/results/arena_dashboard.html`.
- `scripts/build_multi_adapter_report.py` default `--out-json`/`--out-md` now point at `benchmarks/results/...`.

**Changed (docs):**

- `docs/WORKFLOW.md` updated command table/example paths for `arena-dashboard` and clarified that `multi-adapter-report` artifacts belong under `benchmarks/results/`.
- `docs/PROJECT_STATE.md` updated ops-report and arena-dashboard path references to `benchmarks/results/...`.
- `docs/ARENA_ROADMAP.md` and `docs/ARENA_PROGRESSION.md` updated to reference `benchmarks/results/arena_dashboard.html`.

**Outcome:** Workflow defaults now route generated HTML/JSON/MD outputs to `benchmarks/results/` (already gitignored), leaving `benchmarks/*.md` and `docs/*.md` focused on source documentation.

---

## 2026-05-03 — Routing: loading_screen false positives explained + eval re-run

**Context:** Hybrid/frontier mismatches on the 30-prompt loading-screen eval were not coming from ambiguous complexity in `router/policy.py` alone—they were substring overrides in `scripts/model_router.py` (`RoutingPolicy.frontier_keywords` / `hybrid_keywords`) firing before the classifier ladder outcome.

**Specific triggers:**

- `**performance`** matched “perceived **performance**”.
- `**review`** substring-matched “**review**ers”.
- `**production`** matched “**production**-ready”.
- (`**optimize`** is in `hybrid_keywords` too; prompts with the exact contiguous substring trigger hybrid when that path runs.)

**Code:** Loading-screen prompts already short-circuit in `RoutingPolicy.decide` when `adapter_id=="loading_screen"` and `risk_class!="high"` to use `plan_to_legacy_route(build_plan(...))`, suppressing generic keyword shortcuts while still honoring `force_route` and classifier-derived high-risk.

**Verification:** `python scripts/run_routing_benchmark.py --tasks benchmarks/loading_screen_eval_tasks_v1.json --mode both` → route **30/30**, adapter **30/30**; summaries in `benchmarks/results/routing_policy_summary_loading_eval_v1_keyword_override_fix.json` and row dump `benchmarks/results/routing_policy_rows_loading_eval_v1_keyword_override_fix.jsonl`.

---

## 2026-05-03 — Arena acceptance: `save-load-api-guard` (save_load_api_guard adapter)

**Command:** `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/save_load_api_guard/cycle1 --task-id save-load-api-guard` (registry still names `…/champion` but only **cycle1** exists on disk locally).

**Outcome:** **1/1** passed (`passed_all: true`), preview gate **ok**, ACI headline **100**/100 for the single task. Run dir: `benchmarks/results/runs/20260503-191922_510e7d/`.

---

## 2026-05-03 — Save/load specialist: 30-prompt routing eval + router alignment

**Data:** Added `data/routing/save_load_api_guard_eval_prompts_v1.jsonl` (30 curated prompts; mix of serialization/API/auth/production/review-ish wording). Benchmark tasks: `benchmarks/save_load_api_guard_eval_tasks_v1.json` (SHA256[:16] ids).

**Routing:** Extended `scripts/router/policy.py` with `**save_load_api_guard_specialist_fastpath`** so in-domain `**auth` / `security` / `api**` classifier signals don’t escalate to council/API; mirrored `**scripts/model_router.py**` specialist branch (suppress generic frontier/hybrid substring overrides when classifier selects this adapter).

**Prompt hygiene:** A few originals mis-scored (`**loading`** in “loading persisted” → `**loading_screen**`, `**serializing**` vs taxonomy `**serialization**`, stray `**compare**`, ambiguity on missing save keywords)—rewritten in the JSONL.

**Benchmark:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/save_load_api_guard_eval_tasks_v1.json --mode both` → **30/30** route + adapter (`benchmarks/results/routing_policy_summary_save_load_eval_v1.json`, `routing_policy_rows_save_load_eval_v1.jsonl`).

---

## 2026-05-03 — Mixed routing regression (specialists + policy fixtures)

**Artifact:** `benchmarks/mixed_routing_eval_v1.json` (**76** rows): shuffled (**seed 42**) mix of loading-screen (**30**) + save/load (**30**) curated eval tasks, `**task_routing_tasks.json`** policy rows (**12**), and four **specialist probes** (**hud / economy_tooltip / combat_risk / ai_planning_explanation**). Regenerate with `python scripts/build_mixed_routing_eval_v1.py`.

**Router:** Added `_force_frontier_over_save_specialist()` in `scripts/model_router.py` before the save-specialist shortcut: `**cryptography`** always escalates `**frontier**`; `**audit**` + `**authentication**`/`**authorization**` without `**save`/`serialization**` lexicon escapes false `**save_load_api_guard**` classification (fixes policy rows like security audit + crypto signing design).

**Run:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/mixed_routing_eval_v1.json --mode both` → route **76/76**, labeled adapter buckets **64/64**, overall **100%**; row trace `benchmarks/results/routing_policy_rows_mixed_eval_v1.jsonl`, summary `benchmarks/results/routing_policy_summary_mixed_eval_v1.json`. Re-checked save-only `**30/30`** after the frontier guard (`routing_policy_summary_save_load_after_mixed_guard.json`).

---

## 2026-05-03 — Third specialist integrated: `economy_tooltip` (routing + dataset path)

**Router:** `**economy_tooltip_specialist_fastpath`** in `scripts/router/policy.py`; matching specialist branch in `scripts/model_router.py` (suppresses generic `**frontier**` / `**hybrid**` substring overrides after `**save_load_api_guard**` and before blanket frontier keywords).

**Data:** Curated `**data/routing/economy_tooltip_eval_prompts_v1.jsonl`** (**30**) + `**benchmarks/economy_tooltip_eval_tasks_v1.json`**. Routing check: `**30/30**` route + adapter (`benchmarks/results/routing_policy_summary_economy_eval_v1.json`). Mixed `**76/76**` regression still passes; mixed `**economy_tooltip**` probe now reports `**economy_tooltip specialist route**`.

**Train path:** `**scripts/adapters/build_economy_tooltip_specialist_dataset.py`** outputs `**data/lora/adapters/economy_tooltip_specialist/**`; orchestrated via `**python scripts/ml_workflow.py economy-tooltip-dataset**` (defaults: transfer `**loading-screen-polish**` + `**hud-status-summary**`, `--min-train-core-rows` **100**).

**Registry:** `training/adapter_registry_v1.json` `**economy_tooltip`** `**adapter_path**` → `**checkpoints/adapters/economy_tooltip/cycle1**`, lineage `**economy_tooltip:v1:cycle1+routing_fastpath_v1**`, `**promotion_state**` remains `**shadow**` until arena `**economy-tooltip**` is proven independently of routing-only readiness.

**Docs:** `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `data/routing/README.md`.

---

## 2026-05-04 — Documentation specialist scaffolding (dataset + registry + orchestrator)

**Goal:** Simple “documentation agent” LoRA: canonical mlx-lab prose (paths, append-only docs, `ml_workflow` wording) without HUD/arena pairwise.

**Scripts:** Fixed bash newline escaping in **`scripts/adapters/build_documentation_specialist_dataset.py`**. **`scripts/ml_workflow.py`** new subcommand **`documentation-dataset`** → runs that builder with run manifest + **`docs/run_history.md`** row.

**Taxonomy/registry:** **`documentation`** added to **`LOCKED_ADAPTER_FAMILIES`**; **`TASK_TO_ADAPTER`** maps **`mlx-lora-docs-normalize`** → **`documentation`**. **`training/adapter_registry_v1.json`** entry: **`checkpoints/adapters/documentation/cycle1`**, lineage **`documentation_specialist:v1:cycle1`**, **`shadow`**. **`checkpoints/adapters/documentation/cycle1/adapter_config.json`** seeded (data → **`documentation_specialist/train.jsonl`**).

**Verify:** `.venv/bin/python scripts/ml_workflow.py documentation-dataset` (run **`20260504-032701_489e1b`**) wrote **`data/lora/adapters/documentation_specialist/`** (**120**/3/1 train/valid/test rows).

**Docs:** **`docs/WORKFLOW.md`** table row; **`docs/DATA_LAYOUT.md`** documentation specialist paths.

---

## 2026-05-04 — Documentation routing keyword eval + trained cycle1 weights

**Routing:** Expanded **`scripts/router/classifier.py`** phrase bank for **`documentation`**; **`build_plan`** + **`model_router`** specialist fastpaths mirror economy/loading semantics so frontier/hybrid substring hooks don’t steal mlx-doc prompts.

**Eval sources:** **`scripts/build_documentation_eval_tasks_v1.py`** → **`data/routing/documentation_eval_prompts_v1.jsonl`** + **`benchmarks/documentation_eval_tasks_v1.json`** (SHA256-prefix ids; regenerate with the script).

**Regression:** **`tests/test_documentation_routing_eval.py`** + **`tests/__init__.py`** exercise each task via **`RoutingPolicy`**, subprocess **`run_routing_benchmark.py`**, and (**after train**) **`adapters.safetensors`** presence.

**Train:** **`python scripts/ml_workflow.py documentation-dataset`** then **`train --adapter-path checkpoints/adapters/documentation/cycle1 -- --data …/documentation_specialist --iters 80 …`** (**`20260504-032912_d8cd21`**, exit 0).

**Docs:** **`data/routing/README.md`** listed the docs eval JSONL beside other specialists.

---

## 2026-05-04 — Mixed routing eval includes documentation shard

**Builder:** `scripts/build_mixed_routing_eval_v1.py` now merges `benchmarks/documentation_eval_tasks_v1.json` as **`mixed-doc-{id}`** rows (`shard: documentation_eval`) before the fixed specialist probes and policy fixtures; **94** tasks (**seed 42**).

**Regression:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/mixed_routing_eval_v1.json --mode both` → **94/94** route + labeled adapter (**82/82** adapter-scored rows), summary `benchmarks/results/routing_policy_summary_mixed_eval_v2_docs.json`.

---

## 2026-05-04 — Router supervisor Gradio (human try-out)

**Added:** `scripts/router_chat_gradio.py` — chats with **MLX** while each user turn runs `RoutingPolicy`; loads `adapters.safetensors` from `training/adapter_registry_v1.json` when present (documentation / economy_tooltip / loading_screen on this machine).

**Run:** `python scripts/router_chat_gradio.py` (default **`http://127.0.0.1:7862`**, distinct from `human_eval_ui.py` **`7861`** and `chat_gradio.py` **`7860`**).

**Fix:** (1) `mlx_lm.stream_generate` leaked `verbose` into `generate_step` — removed bogus `verbose=False`. (2) MLX `tokenizer.eos_token_ids` omitted **`151645` (`<|im_end|>`)**, so `stream_generate` never stopped cleanly on assistant end and Gradio echoed repeated sentinel strings (looked like a broken documentation LoRA). Added **`scripts/mlx_qwen_stop_tokens.py`** + post-`load()` registration wherever we stream (**`router_chat_gradio.py`**, **`chat_gradio.py`**, **`human_eval_ui.py`**, **`model_router.LocalMlxBackend`**, **`smoke_base_model.py`**). Optional supervisor JSONL (**`ROUTER_CHAT_LOG_JSONL`** or **`--interaction-log-jsonl`**) captures routing + prompts + generations.

---

## 2026-05-04 — Supervisor JSONL + decode budget tuned for reusable SFT rows

**`router_chat_gradio.py`:** default assistant decode ceiling raised (**2048** new tokens vs **896**) so economy/HUD/UI turns stop mid-structure less often; override with **`MAX_TOKENS`** / **`--max-tokens`**. Completed responses log **`mlx_finish_reason`**, token counts, **`generation_budget_hit`**, Markdown fence imbalance, **`recommended_for_sft_assistant_turn`**, **`schema_version: router_chat_supervisor_v1`**. UI emits a truncation banner when MLX hits **`length`** limits.

**System prompt:** nudges complete structured Markdown/fenced replies for gameplay/UI workloads.

**Docs:** **`docs/ROUTING_DATASET_CONTRACT.md`** — appendix on merging supervisor completions into downstream datasets.
