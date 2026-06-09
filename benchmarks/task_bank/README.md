# Task Bank Organization (Systematic Eval Build)

This folder is the canonical organization layer for task sourcing and evaluation assembly.

It separates task origin from evaluation usage, so we can iterate quality without losing lineage.

## Source buckets

- `sources/swebench_verified_github_candidates_v1.json`
  - SWE-bench Verified sourced lane mapped from real `instance_id` entries.
  - References `sources/swebench_verified_game_mapped_tasks_v1.json`.
- `sources/fallen_empire_repo_candidates_v1.json`
  - Tasks mined from `fallen-empire` git patches/commits.
- `sources/custom_author_tasks_v1.json`
  - Net-new hand-authored tasks we add directly.
- `sources/multiplayer_runtime_tasks_v1.json`
  - Curated multiplayer/runtime/deploy product-behavior lane.
  - References stable task IDs from patch-derived and SWE-verified mapped sources.
  - Explicitly excludes eval/test-harness infrastructure tasks.
- `sources/core_gameplay_tasks_v1.json`
  - Curated core gameplay task lane (`army_operations` / `hud_status` focus).
  - Uses stable lineage refs back to patch and SWE-verified mapped sources.
- `sources/economy_research_progression_tasks_v1.json`
  - Curated economy/research/progression task lane.
  - Uses stable lineage refs and granular progression/economy subskills.

## Organization layers

- `organization/domain_index_v1.json`
  - Organizes task IDs by domain across all sources.
- `organization/skill_index_v1.json`
  - Organizes task IDs by subskill/failure family across all sources.
- `organization/subskill_taxonomy_v1.json`
  - Controlled canonical subskill taxonomy.
  - Includes alias-to-canonical mapping for collapsed near-duplicate labels.
- `organization/domain_file_hints_v1.json`
  - Canonical game-repo file hints by domain.
  - Use this instead of legacy `likely_files` from early imported drafts.
- `organization/user_facing_task_requirements_v1.json`
  - Required metadata contract for user-facing tasks.
  - Enforces `acceptance_signals` + `hard_constraints` + `compile_commands` for user-facing prompts.

## Compiled eval

- `compiled/generalist_eval_v1.json`
  - First compiled mixed-source benchmark list for generalist evaluation.
  - Uses task references (`source_file` + `task_id`) instead of duplicating prompts.

## Operating workflow

1. Ingest candidates into one of the source buckets.
2. Validate metadata quality and de-duplicate.
3. Assign domain + skill index references.
4. Curate a compiled eval manifest from indexed candidates.
5. Run eval lanes and promote only tasks that are discriminative and stable.
6. Run `python3 scripts/validate_user_facing_task_contracts.py` before promoting user-facing tasks.

