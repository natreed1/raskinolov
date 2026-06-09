# SWE-Bench Mapped Eval Routing Plan

This document defines routing needed to run SWE-Bench-mapped tasks correctly in the Fallen Empire eval stack.

Task source:
- `benchmarks/task_bank/sources/swebench_verified_game_mapped_tasks_v1.json`

## Why routing is required

SWE-Bench-style tasks are patch-oriented and broad. In this repo, evaluation lanes are split across:
- deterministic game-edit acceptance (`arena-acceptance`),
- specialist benchmark/routing lanes,
- runtime/deploy safety checks.

Without explicit routing, tasks can be scored in the wrong lane and produce misleading metrics.

## Routing contract

For each task, route by `domain_primary` first, then by `task_type`.

### Domain to execution lane

- `hud_status`, `economy`, `army_operations`, `ai_strategy_and_planning`
  - Primary lane: arena-style game-edit validation
  - Required checks: apply success, typecheck/verify, preview readiness when applicable

- `state_perstitence_integrity`
  - Primary lane: arena-style validation + API/serialization guard checks
  - Required checks: schema validation outcomes and no partial state corruption

- `multidomain`
  - Primary lane: arena-style validation with slice scoring
  - Required checks: cross-domain invariant consistency

### Task type to verifier emphasis

- `single_file` / `ui_behavior`
  - Emphasize deterministic preview + visual/label correctness
- `state_logic` / `multi_file`
  - Emphasize consistency across dependent call paths
- `api_schema`
  - Emphasize typed error shape, backward compatibility, and parse stability
- `safety_guard`
  - Emphasize fail-closed behavior and non-crashing degraded paths
- `performance`
  - Emphasize listener/worker lifecycle and non-duplication in loops

## Router implementation guidance

Use this route policy for task execution:

1. Read task metadata (`domain_primary`, `domains_secondary`, `task_type`, `risk`).
2. Choose a primary specialist adapter by `domain_primary`.
3. If `domains_secondary` is non-empty and risk is high, allow multi-agent split/merge path.
4. Force conservative settings for high-risk tasks:
   - bug-check enabled,
   - stricter validation-first sequencing,
   - no acceptance on preview-only success.
5. Record per-task routing metadata:
   - selected adapter id,
   - secondary adapter id (if any),
   - route reason,
   - verification lane used.

## Suggested policy mapping (initial)

- `hud_status` -> `hud_status` specialist adapter
- `economy` -> `economy` specialist adapter
- `army_operations` -> `army_operations` specialist adapter
- `state_perstitence_integrity` -> `state_perstitence_integrity` specialist adapter
- `ai_strategy_and_planning` -> `ai_strategy_and_planning` specialist adapter
- `state_perstitence_integrity` (multiplayer_runtime_deploy subdomain) -> runtime-focused verifier path under persistence/integrity adapter
- `multidomain` -> hierarchical route with primary + secondary adapter

## Runtime/Deploy correctness criteria

For runtime/deploy-scoped tasks (especially those tagged with `multiplayer_runtime_deploy` subdomain metadata), evaluation is considered correct only when all required signals pass for the task's tagged subskill.

Scope pruning note:

- Exclude eval/test-harness infrastructure tasks from runtime/deploy correctness scoring when they are primarily centered on benchmark/arena orchestration scripts (for example `scripts/game_task_arena.py`, `scripts/run_game_benchmark.py`, `scripts/run_arena_acceptance_tests.py`, `scripts/human_eval_ui.py`, `scripts/landing_page_arena.py`).
- Keep runtime/deploy tasks that map to product/game-server behavior (container startup, dependency contract, env precedence, bootstrap sequencing, worker teardown correctness).

Signals enforced in eval:

- module import resolution in container runtime (e.g. shared imports resolve under Dockerized startup)
- dependency presence + lock consistency (manifest and lockfile agree for deploy environments)
- environment variable precedence and defaulting behavior (explicit override > config default > safe fallback)
- service bootstrap success/failure behavior (server starts cleanly; invalid bootstrap path fails with actionable error)
- idempotent teardown and no orphan workers (worker/main-thread transitions do not leak child lifecycles)
- typed error handling for runtime/deploy failures (structured, deterministic failure payloads instead of ambiguous exceptions)

Where these checks are sourced in this repo's eval stack:

- **task acceptance signals**: task-level `acceptance.required_outcomes` in `benchmarks/patch_task_bank_v1.json` (treated as acceptance signal contract for patch-derived runtime/deploy tasks)
- **verify commands**: task-level `acceptance.verify_commands` (for example `npm run test:ml-cohort`) in task manifests
- **lane gating metadata/manifests**:
  - source manifests under `benchmarks/task_bank/sources/*.json` (including SWE-mapped and patch-derived source lanes)
  - compiled routing manifests such as `benchmarks/task_bank/compiled/generalist_eval_swe_verified_v1.json`
  - domain/skill organization indexes in `benchmarks/task_bank/organization/domain_index_v1.json` and `benchmarks/task_bank/organization/skill_index_v1.json`
  - runtime/deploy policy mapping in `benchmarks/task_bank/organization/runtime_deploy_correctness_policy_v1.json`

## Multiplayer correctness criteria

Multiplayer-tagged persistence tasks are validated in a stricter runtime lane because correctness depends on lifecycle and protocol behavior, not just compile/apply success.

Primary pass/fail signals:

- handshake and reconnect behavior
  - reconnect state transitions are idempotent (`reconnect_state_machine_idempotency`)
  - websocket handshake retry path can recover without corrupting session state (`websocket_handshake_recovery`)
- duplicate listener prevention
  - reconnect/rejoin flows do not register duplicate listeners or hooks
- schema compatibility and typed errors
  - snapshot/remap payloads remain back-compatible (`snapshot_schema_compatibility`)
  - incompatible payloads fail with typed, deterministic runtime errors
- timeout and env precedence
  - deploy defaults differ from local defaults where required
  - explicit env/CLI overrides win (`deploy_env_timeout_policy`)
- teardown and leak checks
  - session teardown is idempotent (`session_lifecycle_cleanup`)
  - no orphan workers or leaked background handlers remain
- deployment bootstrap success
  - Railway/service bootstrap contract is satisfied (`railway_service_bootstrap_contract`)
  - invalid bootstrap paths fail fast with actionable errors

Evaluator flow for multiplayer tasks:

1. Select multiplayer task set from `benchmarks/task_bank/sources/multiplayer_runtime_tasks_v1.json`.
2. Resolve per-task correctness expectations from:
   - `acceptance_signals` (SWE-mapped tasks) and/or `acceptance.required_outcomes` (patch tasks),
   - `verify_commands` / `acceptance.verify_commands`.
3. Route by `domain_primary` (`state_perstitence_integrity`) + `multiplayer_runtime_deploy` subdomain tag into runtime/deploy verifier lane using:
   - `benchmarks/task_bank/organization/domain_index_v1.json`,
   - `benchmarks/task_bank/organization/skill_index_v1.json`,
   - `benchmarks/task_bank/organization/runtime_deploy_correctness_policy_v1.json`.
4. Gate inclusion and final scoring through compiled manifests (`benchmarks/task_bank/compiled/*.json`) and emit typed `failure_mode` + lane metadata.

## Core gameplay correctness criteria

Core gameplay tasks prioritize deterministic simulation and player-facing behavior consistency.

Primary pass/fail signals:

- state machine transition integrity (`state_machine_transition_integrity`)
  - transition guards prevent invalid phase hops
  - repeated actions do not create impossible state combinations
- combat modifier consistency (`combat_modifier_consistency`)
  - equivalent combat contexts produce stable modifier/risk outputs
  - no silent precision drift between related combat paths
- input controller event ordering (`input_controller_event_ordering`)
  - pointer/keyboard/controller event sequencing is deterministic
  - drag/select/click sequences do not race into stale state
- deterministic preview state isolation (`deterministic_preview_state_isolation`)
  - preview state does not mutate canonical simulation state
  - repeated previews produce equivalent outputs for equivalent inputs

Evaluator flow:

1. Select from `benchmarks/task_bank/sources/core_gameplay_tasks_v1.json`.
2. Read acceptance contract from task metadata (`acceptance_signals` or `acceptance.required_outcomes`).
3. Execute `verify_commands`/`acceptance.verify_commands` and lane checks.
4. Score pass/fail with typed failure modes, then aggregate into per-domain metrics.

## Economy/research/progression correctness criteria

Economy/research/progression tasks prioritize numeric semantics, tick ordering, and schema-safe progression updates.

Primary pass/fail signals:

- signed delta semantics (`signed_delta_semantics`)
  - positive/negative resource deltas preserve sign and intent in both logic and UI
- progression tick ordering (`progression_tick_ordering`)
  - progression side effects run in deterministic order each tick
  - no skipped or double-applied progression effects
- research schema migration compatibility (`research_schema_migration_compatibility`)
  - migration and hydration preserve required research/progression fields
  - incompatible payloads emit typed migration errors
- resource projection cache invalidation (`resource_projection_cache_invalidation`)
  - derived economy projections refresh when dependencies change
  - stale cache paths do not leak outdated resource signals

Evaluator flow:

1. Select from `benchmarks/task_bank/sources/economy_research_progression_tasks_v1.json`.
2. Apply task-level acceptance contracts (`acceptance_signals` / `acceptance.required_outcomes`).
3. Run lane verify commands and schema/type guards.
4. Record pass/fail and typed failure modes for domain and subskill slices.

## How a task is tested

Plain-language lifecycle for one task:

1. **Prompt + context provided to model**
   - The model receives the task prompt text, domain/subskill metadata, and expected file hints from the source manifest entry.
2. **Expected code changes**
   - The model is expected to edit files implied by task metadata (`changed_files` for patch tasks, `canonical_file_hints`/domain hints for mapped tasks), not unrelated harness infrastructure.
3. **Acceptance checks**
   - Evaluator validates declared acceptance contract:
     - SWE-mapped tasks: `acceptance_signals`
     - patch-derived tasks: `acceptance.required_outcomes`
4. **Verify and lane gating**
   - Evaluator runs `verify_commands` / `acceptance.verify_commands`, then applies lane-specific checks selected from routing metadata (`domain_index`, `skill_index`, and correctness policy manifests).
5. **Pass/fail + metrics contribution**
   - A task passes only if acceptance + verify + lane checks pass.
   - Results emit `acceptance_passed`, `verify_ok`, `failure_mode`, and routing metadata, then roll up into overall, per-domain, per-task-type, and high-risk metrics.

## Scoring outputs needed for final metrics

For each mapped task, emit:
- `route_selected`
- `adapter_selected`
- `apply_ok`
- `verify_ok`
- `preview_ok` (if applicable)
- `acceptance_passed`
- `failure_mode` (typed)
- `elapsed_s`
- `risk`
- `domain_primary`
- `task_type`

These fields are sufficient to compute:
- overall pass rate,
- per-domain pass rates,
- per-task-type pass rates,
- high-risk slice pass rate,
- worst-tail robustness metrics.

## Integration checklist

- Register source: `benchmarks/task_bank/sources/swebench_verified_github_candidates_v1.json`
- Use task list: `benchmarks/task_bank/sources/swebench_verified_game_mapped_tasks_v1.json`
- Add curated multiplayer source: `benchmarks/task_bank/sources/multiplayer_runtime_tasks_v1.json`
- Add references to domain/skill indices when curating next compiled benchmark revision.
- Keep this lane decontaminated from training holdouts.

