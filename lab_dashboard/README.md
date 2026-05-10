# Lab optimization dashboard (static deploy)

This folder is generated and/or fed by lightweight **terminal-side** tooling so MLX work does not consume long Cursor chats.

## Automated tests

`tests/test_fe_ml_lab_tools.py` exercises the Markdown parser + JSONL helpers plus the CLI runner hooks with mocked subprocesses (no GPUs, no MLX). Run locally before merging toolchain changes:

```bash
python3 -m unittest discover -s tests -p test_fe_ml_lab_tools.py -v
```

## Build

From the repo root (venv recommended):

```bash
python scripts/build_lab_optimization_dashboard.py
```

Writes `lab_dashboard/index.html` from committed `docs/run_history.md`, optional ledger files beside this README, and any event log lines created by `scripts/fe_ml_lab_runner.py`.

## Two small “adaptors”

1. **`scripts/fe_ml_lab_runner.py`** — runs standardized `ml_workflow.py` sequences (default **`learning` → smoke**) and appends **`agent_events.jsonl`** here with timing and exit codes. Set **`FE_ML_LAB_SPARED_USD`** to record a heuristic “USD spared by not improvising inside chat” tally (manual bookkeeping).
2. **`scripts/build_lab_optimization_dashboard.py`** — packs run history plus ledgers into a **single-folder static site**.

## Cursor spend log (manual)

Append JSON lines to **`cursor_usage.jsonl`** (tracked or private copy — your choice):

```json
{"ts":"2026-05-04T12:34:56Z","usd":4.87,"category":"composer","note":"Weekly estimate"}
```

The dashboard totals the `usd` field. Cursor does **not** stream billing into this repo automatically.

## Hooks + OSS “background lane”

Composer never attaches MLX checkpoints. Hooks are **lifecycle scripts** Cursor runs beside the IDE (audit/journal/sync), not substitutes for `mlx_lm` inside chat.

### `stop` hook (checked in)

- **`.cursor/hooks.json`** → **`.cursor/hooks/lab_hook_stop_append.py`**
- Appends **`lab_dashboard/cursor_hook_events.jsonl`** when an Agent run exits via **`stop`**.

Turn on logging **explicitly** (otherwise the script no-ops so repos stay quiet):

```bash
export FE_LAB_CURSOR_HOOK_APPEND=1
```

Optional **`FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD=1`** re-runs **`scripts/build_lab_optimization_dashboard.py`** immediately after each stop (HTML regen only).

Heavy MLX training still belongs with **`cron`/`launchd`**, **`ml_workflow.py`**, or a future queue watcher — hooks should stay lightweight.

### `afterShellExecution` hook (checked in)

- **`.cursor/hooks.json`** → **`.cursor/hooks/lab_hook_after_shell_autodoc.py`**
- Records test/benchmark shell commands plus git-change snapshots to:
  - **`lab_dashboard/shell_command_events.jsonl`**
  - **`docs/SPECIALIZED_RUN_HISTORY.md`** (compact append-only row per captured command)

This repo enables it by default in `.cursor/hooks.json` (command prefixes `FE_LAB_AUTODOC_APPEND=1`).

Optional:

- **`FE_LAB_AUTODOC_INCLUDE_ALL=1`** records all shell commands (default filter captures test/benchmark commands such as `pytest`, `unittest`, `ml_workflow.py`, benchmark runners).
- **`FE_LAB_REMOTE_INGEST_URL`** + **`FE_LAB_REMOTE_INGEST_TOKEN`** forward each captured hook event to a remote private dashboard ingest endpoint (`POST /api/ingest`).

### `afterFileEdit` training trigger hook (checked in)

- **`.cursor/hooks.json`** → **`.cursor/hooks/lab_hook_after_fileedit_training_trigger.py`**
- On watched codebase edits (`docs/`, `scripts/`, `benchmarks/`, `tests/`, `training/`), runs:
  - `scripts/generate_change_documentation_capture.py` (**always**, async): open-source + specialized documentation capture for the changed path
  - `scripts/build_run_analysis_rag_corpus.py` (refresh run-analysis RAG corpus; cooldown-gated with dataset refresh)
  - `python scripts/ml_workflow.py documentation-dataset` (documentation training dataset refresh; cooldown-gated)
- Queue/audit rows are appended to:
  - `data/training_triggers/documentation_training_queue.jsonl`
- Capture artifacts are written to:
  - `data/documentation_captures/*.json` + `.opensource.md` + `.specialized.md`

Environment controls:

- `FE_LAB_DOC_TRIGGER_MIN_SECONDS` (default `900`) throttle window between heavy refreshes
- `FE_LAB_AUTODOC_DATASET_ON_CHANGE` (`1`/`0`) enable/disable dataset refresh while still recording triggers
- `FE_LAB_ALWAYS_RUN_OPEN_SOURCE_ON_CHANGE` (`1`/`0`, default on) control always-on open-source+specialized capture on watched edits

### Private hosted dashboard (personal-only)

Use **`scripts/private_dashboard_server.py`** for a private site with:

- HTTP Basic auth for viewers
- Bearer-token ingest endpoint
- SQLite event storage
- live summary + recent command/event feed + docs snapshots

Deployment guide: **`docs/PRIVATE_DASHBOARD_DEPLOY.md`** (Railway-first).

### Syncing clones / Cursor windows / machines

- Same repo checkout on disk → hooks + JSONL instantly shared between windows pointing at one root.
- Different machines/commits → **`git push`/`pull`** the small artifacts you care about (`*.jsonl`, rebuilt `index.html`, `docs/run_history.md`); never expect multi‑GB **`checkpoints/`** blobs in git (`checkpoints/` remain gitignored here).

See **`cursor_hook_events.example.jsonl`** for the row shape Cursor appends while the hook stays enabled.

## Deploy

Upload the **`lab_dashboard/`** directory after building (GitHub Pages: point site root here; Netlify/Vercel: deploy this folder).

For a purely public artifact you can omit `cursor_usage.jsonl`; the workflow table still renders from **`docs/run_history.md`**.

## Example ledger

See `cursor_usage.example.jsonl` for formatting.
