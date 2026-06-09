# HUD Status Dataset Expansion Plan (Strict Specialist Only)

This runbook expands `hud_status` data without cross-domain contamination and produces a clean specialist training cycle.

## 1) Objective and Pass Criteria

Current active specialist dataset status:

- Dataset: `data/lora/adapters/hud_status_specialist`
- Current scorecard verdict: `quarantine` (low purity/high duplication)

Target for next cycle:

- `transfer_pairwise = 0`
- `skipped_cross_domain_benchmark > 0` (confirm transfer/multidomain prompts were excluded from train synthesis)
- `rejected_foreign_pairwise > 0` (confirm strict filter is active)
- Build manifest split target: at least `train >= 240`
- Post-train benchmark goal: `hud_status` specialist suite pass, with improved instruction + concision stability.

## 2) Expansion Shape (What to Add)

Add **120-180 new HUD-only rows** across two sources:

- **A. Pairwise core rows (70%)**: ~90-120 rows
  - Keep only true HUD prompts:
    - compact chips/labels
    - morale/supply/risk ordering
    - one-line tactical summaries
    - clutter reduction and visibility tradeoffs
  - Reject anything with loading, economy, save/load schema, or AI planning rationale focus.
- **B. Benchmark-synth rows (30%)**: ~30-60 rows
  - Add new `hud_core`, `hud_ui_change`, `hud_constraints` tasks.
  - Keep `hud_transfer` / `hud_multidomain` prompts in benchmark files for eval coverage, but do not use them as train synthesis rows.

Suggested prompt-family balance for new HUD prompts:

- `40%` pure status chips/labels (short outputs)
- `35%` constrained formatting (exact counts/one-line/comma-separated/plain text)
- `25%` UI rationale/change requests (2-3 sentence compact explanations)

## 3) Data Authoring Rules (Hard)

Every new pairwise row intended for HUD training must satisfy:

- `task.id` is `hud-status-summary` (or a HUD-specific id namespace)
- `task.specialists` contains `hud_status`
- `task.prompt` and `winner_output` contain HUD-status semantics (morale/supply/risk/pressure/territory scan language)
- no code-fence/code-block artifacts unless prompt explicitly requests code (HUD set should usually be plain text)
- no cross-domain primary intent (`loading`, `save/load`, `schema`, `economy cause/effect`, `AI council rationale`)

## 4) Expand Benchmark Prompt File (Pure Train-Friendly)

Create a dedicated pure training prompt file from the existing mass suite:

- input: `benchmarks/hud_status_mass_tasks_v1.json`
- output: `benchmarks/hud_status_mass_tasks_v2_pure.json`
- keep categories:
  - `hud_core`
  - `hud_ui_change`
  - `hud_constraints`
- drop categories:
  - `hud_transfer`
  - `hud_multidomain`

Example one-liner:

```bash
python3 -c 'import json,pathlib;src=pathlib.Path("benchmarks/hud_status_mass_tasks_v1.json");dst=pathlib.Path("benchmarks/hud_status_mass_tasks_v2_pure.json");rows=json.loads(src.read_text());keep={"hud_core","hud_ui_change","hud_constraints"};out=[r for r in rows if r.get("category") in keep];dst.write_text(json.dumps(out,indent=2)+"\n");print(f"wrote {len(out)} rows -> {dst}")'
```

## 5) Build Dataset (Strict Mode)

Run strict specialist-only builder:

```bash
python scripts/ml_workflow.py hud-status-dataset \
  --pairwise-jsonl benchmarks/results/game_task_pairwise_training_data.jsonl \
  --benchmark-tasks-json benchmarks/hud_status_mass_tasks_v2_pure.json \
  --out-dir data/lora/adapters/hud_status_specialist_cycle4_pure \
  --max-core-rows 220 \
  --max-benchmark-rows 180 \
  --min-train-core-rows 240
```

Expected manifest checks (`data/lora/adapters/hud_status_specialist_cycle4_pure/manifest.json`):

- `source_counts.transfer_pairwise == 0`
- `strict_specialist_only == true`
- `allow_transfer == false`
- `allow_cross_domain_benchmark == false`
- `source_counts.rejected_foreign_pairwise > 0` (if non-HUD rows exist in input pairwise file)
- `source_counts.skipped_cross_domain_benchmark >= 0`

## 6) Train + Evaluate Cycle

```bash
python scripts/ml_workflow.py train \
  --adapter-path checkpoints/adapters/hud_status/cycle4_pure \
  --evaluate \
  --bench-specialist hud_status \
  -- \
  --data data/lora/adapters/hud_status_specialist_cycle4_pure \
  --iters 80
```

Then run benchmark-only reconfirm:

```bash
python scripts/ml_workflow.py benchmark \
  --adapter-path checkpoints/adapters/hud_status/cycle4_pure \
  --specialist hud_status
```

## 7) Acceptance Gate for Promotion

Promote only if all hold:

- strict dataset manifest checks pass (Section 5)
- specialist benchmark for `hud_status` passes target suite
- no regression on constraint-following prompts (one-line/plain-text/count-limited)
- scorecard classification for `hud_status_specialist_cycle4_pure` no longer `quarantine`

If not met:

- keep checkpoint in shadow
- add another 40-80 targeted HUD prompts only in failed pattern buckets
- repeat build/train/eval loop.

## 8) Explicit Labeling Policy

- Treat any dataset/checkpoint with names containing:
  - `mock_aug`, `ui_merge`, `dual`, `ablation`, `experiment`
  as **test ablation only**.
- Do not use ablation-tagged datasets for specialist promotion decisions.
