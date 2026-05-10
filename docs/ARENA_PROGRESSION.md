# Arena acceptance progression (comparable runs)

Use this file to track **apples-to-apples** arena acceptance batches: same task IDs, adapter, `SOURCE_REPO`, and gate (`preview`), varying only documented axes (e.g. `**--progressive-context`**).

**Canonical orchestrator:** `python scripts/ml_workflow.py arena-acceptance …` — writes `benchmarks/results/runs/<run_id>/`, `manifest.json` (includes `arena_context.progressive_context` when applicable), and appends `docs/run_history.md`.

Progressive `**auto` vs `off`** pairings preferred for adapters listed in `**scripts/fe_lineage.py**` (`DEFAULT_ARENA_PROGRESSIVE_CONTEXT`, `ARENA_ADAPTER_PROGRESSIVE_POLICY`); regenerate `**benchmarks/results/arena_dashboard.html**` with `**python scripts/ml_workflow.py arena-dashboard**` after batches to compare rows (see `**docs/ARENA_ROADMAP.md**`).

**Full six-task suite** (default acceptance coverage):

`loading-screen-polish`, `hud-status-summary`, `economy-tooltip`, `combat-risk-preview`, `save-load-api-guard`, `ai-planning-explanation`

## Run table


| Date (UTC) | Run ID                   | Progressive context | Pass (preview) | ACI  | Smoke / Std / High tiers | Adapter                                                                                                                                                  |
| ---------- | ------------------------ | ------------------- | -------------- | ---- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-28 | `20260428-171007_517f3e` | `auto`              | 1/6            | 53.4 | 96.6 / 49.5 / 45.1       | `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428`                                                                                                   |
| 2026-04-28 | `20260428-180627_42befd` | `off`               | 1/6            | 56.3 | 96.6 / 51.3 / 49.5       | same                                                                                                                                                     |
| 2026-04-29 | `20260429-025429_9793ff` | `auto`              | 1/6            | 55.2 | 32.3 / 48.7 / 64.3       | `checkpoints/fe-lora-qwen25-coder-7b-best-val300` (iter-300 weights; worktree `~/fallen-empire-arena`, port 5174)                                        |
| 2026-04-29 | `20260429-034932_0812a4` | `off`               | 0/6            | 47.8 | 33.2 / 50.9 / 49.7       | same adapter; worktree `benchmarks/results/arena_worktrees`, port 5200 (retry after `20260429-034918` `PermissionError` on home worktree)                |
| 2026-04-29 | `20260429-034918_34e0cb` | `off`               | 0/6            | 25.0 | 25 / 25 / 25             | same adapter — **invalid** (immediate `PermissionError` on `~/fallen-empire-arena`; superseded by `034932`)                                              |
| 2026-04-30 | `20260430-183012_121be1` | `auto`              | 3/6            | 68.1 | 100.0 / 35.9 / 78.1      | `checkpoints/fe-lora-arena-apply-sft` (SFT on `data/lora/arena_task_baselines_apply_contract`; worktree `benchmarks/results/arena_worktrees`, port 5274) |
| 2026-04-30 | `20260430-231450_487deb` | `auto`              | 3/6            | 75.0 | 100.0 / 45.1 / 85.4      | `checkpoints/fe-lora-arena-reverse-recover` (reverse pass from `fe-lora-arena-standard-dev-sft` on six-task apply-contract baselines; port 5290) |
| 2026-04-30 | `20260430-215539_b451dd` | `auto`              | 0/6            | 52.3 | 36.6 / 54.3 / 54.9       | `checkpoints/fe-lora-arena-standard-dev-sft` (standard-dev-focused SFT from `fe-lora-arena-apply-sft`; six-task eval run after iter-80 checkpoint save) |


**Best-val300 Apr 29:** Progressive **auto** beats **off** on headline ACI (55.2 vs 47.8) with opposite sign vs **chunk6k** Apr 28 (where off led by ~3 pts). Canvas trackers use iter-**300** weights (`best-val300` dir); final iter-400 weights remain on `fe-lora-qwen25-coder-7b-latest`.

**Apr 30 attribution note (working):** the jump from prior **1/6** runs to **3/6** in `20260430-183012_121be1` is currently attributed to the combined change set of (1) adding the explicit apply contract (`docs/GAME_ARENA_APPLY_CONTRACT.md`) into arena context and (2) running targeted arena baseline SFT (`checkpoints/fe-lora-arena-apply-sft`). Treat this as a provisional causal attribution until reproduced across additional batches.

After each new batch, add a row (replace `pending`) and bump “Last verified” context in `docs/PROJECT_STATE.md` if this was a deliberate regression check.

## Baseline command (progressive off)

```bash
cd /Users/natreed/fallen-empire-lora
source .venv/bin/activate
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 \
  --source-repo /Users/natreed/fallen-empire \
  --progressive-context off \
  --timeout-s 14400 \
  --preview-port 5200 \
  --task-id loading-screen-polish \
  --task-id hud-status-summary \
  --task-id economy-tooltip \
  --task-id combat-risk-preview \
  --task-id save-load-api-guard \
  --task-id ai-planning-explanation
```

## Progressive (default / auto) command

Omit `--progressive-context` or pass `--progressive-context auto`.