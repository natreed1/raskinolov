# Patch Task Bank V1

This document describes the initial patch-derived task bank at:

- `benchmarks/patch_task_bank_v1.json`

The bank is seeded from real `fallen-empire` commits and intended to bootstrap high-quality task authoring for Evaluation System V2.

## What this artifact contains

- 24 patch-derived tasks.
- Source-linked metadata (`source_commit`, `source_subject`, `changed_files`).
- Coverage metadata aligned with V2 (`domain_primary`, `subskill`, `risk`, `complexity`, `task_type`, `expected_multi_domain`).
- Recommended split labels:
  - `train_candidate`
  - `eval_public`
  - `holdout_eval`

## Why patch-derived tasks

- They are grounded in real regressions and integration work.
- They naturally produce multi-file, high-signal tasks.
- They reduce synthetic-task drift by anchoring prompts to observed failure classes.

## Current coverage snapshot

The initial bank is weighted toward high-signal integration domains:

- `save_load_api_guard` and multiplayer/server contracts
- `hud_status` and dense UI/state synchronization
- `combat_risk` and mechanic integration
- `ai_planning_explanation` and shared plan execution logic
- cross-domain tasks with auth/runtime/deploy guardrails

## How to use this bank now

1. Treat `eval_public` as immediate candidate prompts for benchmark expansion.
2. Keep `holdout_eval` out of training data generation.
3. Curate `train_candidate` tasks into supervised data only after de-dup and quality review.
4. Attach deterministic verification commands and preview targets when converting each item into arena-style runnable tasks.

## Task promotion workflow

For each bank item selected into runnable benchmarks:

1. Convert patch summary prompt into a concrete arena task with explicit acceptance criteria.
2. Add task-specific context path constraints and allowed edit paths.
3. Run pilot evals and inspect failure taxonomy.
4. Promote to public benchmark only if discriminative and stable across repeated runs.
5. Reserve a subset as hidden holdouts for promotion decisions.

## Next expansion pass

- Mine additional commits emphasizing:
  - bug-fix commits (`fix:`) and post-deploy regressions,
  - high-risk API/schema patches,
  - cross-domain state sync issues.
- Target >= 80 patch-derived tasks before creating V2 promotion suite subsets.
- Keep at least 25 percent of patch bank in hidden holdout.
