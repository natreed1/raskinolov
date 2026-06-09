# Evaluation System V2

Status: draft proposal for replacing the current "core6 + ad hoc extensions" process with a coverage-driven evaluation program.

## Why change

The current system has strong foundations (deterministic arena acceptance, routing benchmark, execution benchmark), but promotion decisions are still too sensitive to small task sets. V2 addresses this by:

- expanding curated task coverage across domains and subskills,
- separating fast smoke gating from promotion-grade gating,
- adding hidden holdouts and tail-risk checks,
- making ablation experiments first-class and reproducible.

## Design goals

- Improve confidence that a promoted adapter generalizes beyond a narrow suite.
- Preserve fast feedback loops for daily iteration.
- Detect regressions in high-risk domains early.
- Make task coverage and quality auditable.

## Evaluation architecture

Run four lanes every cycle, each with a different role.

### Lane A: Sentinel smoke (fast)

- Purpose: immediate breakage detection while iterating.
- Suite: current core six arena tasks.
- Runtime: short, repeatable.
- Decision role: block obvious regressions only.

### Lane B: Coverage acceptance (promotion-critical)

- Purpose: broad capability and integration validation.
- Suite: expanded arena catalog with balanced domain and subskill slices.
- Decision role: primary promotion gate.

### Lane C: General coding robustness

- Purpose: guard against generic code generation collapse.
- Suite: EvalPlus subset/full (`humaneval`, `mbpp`) with stable config.
- Decision role: secondary blocker if severe drop is observed.

### Lane D: Routing and control-plane quality

- Purpose: validate adapter/route decisions and high-risk misroute behavior.
- Suite: routing benchmark + promotion gate thresholds.
- Decision role: blocker for router-related promotions.

## Task taxonomy (required metadata)

Every new task should include explicit metadata so coverage and slice metrics can be computed deterministically.

- `domain_primary`: one of `hud_status`, `hud_status`, `economy`, `army_operations`, `state_perstitence_integrity`, `ai_strategy_and_planning`, `documentation`, `routing`.
- `domains_secondary`: optional list for cross-domain tasks.
- `subskill`: concrete capability being tested (see section below).
- `complexity`: `low`, `low_medium`, `medium`, `medium_high`, `high`.
- `risk`: `low`, `medium`, `high`.
- `task_type`: `single_file`, `multi_file`, `api_schema`, `ui_behavior`, `state_logic`, `performance`, `safety_guard`.
- `requires_preview`: boolean.
- `requires_typecheck`: boolean (default true for game-edit tasks).
- `expected_multi_domain`: boolean.
- `anti_overfit_notes`: brief note describing what makes shortcutting harder.

## Initial subskill catalog

Use this as the first pass for subskill-by-subskill expansion.

- `hud_status`
  - UX hierarchy and readability polish
  - progress/state truthfulness
  - asset/perf-safe loading behavior
- `hud_status`
  - state-to-UI mapping correctness
  - concise tactical summarization
  - visual consistency with existing panel language
- `economy`
  - causal explanation correctness
  - formula/field binding correctness
  - tooltip brevity and clarity under constraints
- `army_operations`
  - risk computation wiring and modifier coverage
  - uncertainty communication
  - report readability under dense data
- `state_perstitence_integrity`
  - schema validation and compatibility
  - serialization integrity and fallback behavior
  - API response typing/error discipline
- `ai_strategy_and_planning`
  - action rationale grounded in real action types
  - parameter/value explanation correctness
  - concise strategic narrative under strict limits
- `multidomain`
  - economy-to-combat linkage
  - HUD + AI rationale coherence
  - save/load + UI state restoration correctness

## Suite construction rules

- Keep `core6` as sentinel only.
- Create a promotion suite with at least:
  - 120 total tasks,
  - minimum 12 tasks per primary domain,
  - at least 30 percent multidomain tasks,
  - at least 20 percent high-risk tasks,
  - balanced complexity distribution (no tier below 15 percent).
- Maintain an unseen holdout suite:
  - 25 to 40 tasks,
  - never used for prompt/dataset authoring,
  - sampled from every domain and from multidomain.

## Scoring and gates

Promotion should pass all of the following:

- Coverage acceptance pass rate above threshold on the promotion suite.
- Minimum pass rate on every domain slice (no single-domain collapse).
- Minimum pass rate on high-risk slice.
- Minimum pass rate on multidomain slice.
- Tail robustness: worst-N slice above threshold (for example worst 10 percent tasks).
- No severe regression versus currently promoted adapter on:
  - sentinel core six,
  - routing high-risk misroute metrics,
  - EvalPlus pass rate floor.

Recommended first thresholds (tune after two weeks of data):

- overall promotion suite pass rate >= 0.72
- per-domain pass rate >= 0.60
- high-risk pass rate >= 0.58
- multidomain pass rate >= 0.55
- worst-10-percent slice >= 0.40
- router high-risk misroute count <= current policy cap

## Ablation protocol (required for process changes)

Any material process/model/data change should be evaluated through controlled ablations.

### Ablation principles

- Change one factor at a time.
- Keep task suites fixed for comparability.
- Run at least two repeats when runtime allows.
- Record all run configs and outputs in run artifacts.

### Candidate factors

- dataset composition (single-domain vs mixed vs multidomain-heavy),
- context policy (`progressive-context auto` vs `off`),
- bug-check loop on/off,
- routing selection mode and hierarchy knobs,
- specialist routing policy vs baseline policy.

### Required ablation report fields

- hypothesis,
- factor changed and fixed controls,
- run ids,
- suite-level metrics,
- per-slice deltas,
- failure-mode deltas (`no_applyable_changes`, `verify_failed`, `preview_failed`, typecheck failures),
- promotion decision and rationale.

## Data quality policy for tasks

Every added task should satisfy:

- clear observable acceptance criteria,
- grounded file/path references,
- realistic constraints (no invented schema),
- bounded scope with explicit non-goals,
- anti-shortcut design (task cannot pass via generic boilerplate).

Reject tasks that are:

- redundant with existing tasks,
- ambiguous to score,
- too open-ended for deterministic gate usage,
- overfitted to one exact phrasing pattern.

## Execution plan (subskill-by-subskill)

For each subskill:

1. Define capability intent (what good output demonstrates).
2. Draft 6 to 10 candidate tasks.
3. Curate to 4 high-quality tasks with varied complexity.
4. Add metadata and scoring notes.
5. Run pilot lane (small batch) and inspect failure taxonomy.
6. Promote only tasks that are discriminative and stable.

Suggested order:

1. `state_perstitence_integrity` (high risk, strong payoff)
2. `economy`
3. `army_operations`
4. `hud_status`
5. `ai_strategy_and_planning`
6. `hud_status`
7. multidomain packs

## Near-term implementation checklist

- Add a task metadata contract doc for arena tasks (`schema_version`, required fields).
- Build a task coverage report script (domain/subskill/risk/complexity heatmap).
- Add slice-aware gate script that consumes acceptance summary JSON and task metadata.
- Add an ablation runner template and report schema.
- Start subskill expansion with one domain this week and publish first coverage dashboard snapshot.
- Use organized task-bank structure under `benchmarks/task_bank/`:
  - Source registries: `sources/swebench_verified_github_candidates_v1.json`, `sources/fallen_empire_repo_candidates_v1.json`, `sources/custom_author_tasks_v1.json`
  - Domain/skill indices: `organization/domain_index_v1.json`, `organization/skill_index_v1.json`
  - Canonical file hints: `organization/domain_file_hints_v1.json`
  - Compiled mixed benchmark: `compiled/generalist_eval_v1.json`
- Seed patch-derived tasks from real game-repo commits (initial seed: `benchmarks/patch_task_bank_v1.json`, process notes in `docs/PATCH_TASK_BANK_V1.md`).

