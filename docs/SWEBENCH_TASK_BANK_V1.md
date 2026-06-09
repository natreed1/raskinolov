# SWE-Bench Task Bank V1 (Game-Applicable)

This task bank adapts SWE-Bench issue archetypes to Fallen Empire workflows while keeping prompts implementation-oriented and patch-friendly.

## Sourcing approach

- Used SWE-Bench-style bug families as templates (boundary regressions, schema mismatches, stale state, race conditions, cache invalidation, backward compatibility, partial failure handling).
- Converted each family into a concrete game-facing engineering task with explicit expected behavior and likely touch points in this repo.
- Kept task language in "patch/fix" form instead of open-ended ideation so tasks score cleanly in benchmark lanes.
- Marked all items as adapted archetypes, not verbatim reproductions of benchmark issue text.

## Mapping rationale

- **HUD/UI** tasks target real user-visible regressions (threshold bands, stale refresh, reconnect duplication, deterministic rendering).
- **Economy** tasks target reasoning and data correctness failures (signed deltas, null fallback, ranking determinism, turn cache invalidation).
- **Combat** tasks target high-signal logic bugs (band boundaries, shared modifiers, unknown terrain policy, encounter-state isolation).
- **Save/load/API** tasks target safety-critical persistence behavior (typed contracts, migrations, atomic apply, robust error paths).
- **AI/planning** tasks target rationale correctness under constraints (feasibility gates, template key contracts, regression tests).
- **Multiplayer/runtime/deploy** tasks target operational reliability (reconnect races, timeout configuration drift, worker leaks).
- **Multidomain** tasks enforce consistency across systems where many regressions hide.

## Caveats and decontamination

- These are SWE-Bench-aligned archetypes adapted to this repo, not copied SWE-Bench instances.
- Avoid including exact external issue wording, full stack traces, or known canonical fixes from public benchmark cases.
- Keep future expansions decontaminated by deriving new tasks from bug *families* and repo-specific behavior, not memorized case text.
- Re-run anti-overfit review when adding tasks: reject vague prompts, duplicate failure modes, and tasks passable via generic boilerplate.

## How to use in eval/train splits

- Current split in `benchmarks/swebench_game_applicable_tasks_v1.json` is balanced for iteration:
  - `train`: 15
  - `eval`: 7
  - `holdout`: 3
- Use `train` for adapter iteration and prompt tuning.
- Use `eval` for promotion decisions and gate comparisons.
- Keep `holdout` untouched during authoring/tuning; run only for final confidence and tail-risk checks.
- If converting to execution lanes, preserve split boundaries and keep ids stable for timeseries comparability.

## Organization update (canonical paths)

- Source registry for SWE-bench Verified mapped lane now lives at:
  - `benchmarks/task_bank/sources/swebench_verified_github_candidates_v1.json`
  - mapped tasks: `benchmarks/task_bank/sources/swebench_verified_game_mapped_tasks_v1.json`
- Domain and skill organization now lives at:
  - `benchmarks/task_bank/organization/domain_index_v1.json`
  - `benchmarks/task_bank/organization/skill_index_v1.json`
- Compiled generalist mixed-source eval now lives at:
  - `benchmarks/task_bank/compiled/generalist_eval_v1.json`
- Compiled SWE-only generalist eval now lives at:
  - `benchmarks/task_bank/compiled/generalist_eval_swe_verified_v1.json`
- Canonical game-repo target file hints now live at:
  - `benchmarks/task_bank/organization/domain_file_hints_v1.json`

Important: do not use early `likely_files` values in `benchmarks/swebench_game_applicable_tasks_v1.json` as final routing targets. Treat that file as raw imported prompt storage, and use domain file hints plus curated indices for metric-grade evaluation.
