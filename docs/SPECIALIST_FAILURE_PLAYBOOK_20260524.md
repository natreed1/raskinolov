# Specialist Failure Playbook (Cloud 121-Task Runs)

Date: 2026-05-24

## Scope

This playbook summarizes failure patterns from the completed cloud 121-task runs for:

- `loading_screen`
- `hud_status`
- `economy_tooltip`
- `combat_risk`
- `save_load_api_guard`
- `ai_planning_explanation`

Each specialist was evaluated with `single_specialist_local` over the full 121-task manifest.

## High-level failure mix

- Total failed specialist rows analyzed: `495`
- `typescript_compile_or_type_error`: `419` (`84.6%` of failures)
- `no_applyable_changes`: `76` (`15.4%` of failures)

Interpretation:

- Most losses are not missing context alone; they are generated changes that fail repository compile/verify.
- Secondary issue is output shape non-compliance (fenced snippets/prose/JSON rather than applyable patch contract).

## Top recurring verify signatures (ranked)

The following signatures dominate compile/verify failures:

1. `Module '../src/core/gameCore' has no exported member 'SimResult'.` (`73`)
2. `Module '../lib/ai' has no exported member 'planAiTurn'.` (`62`)
3. `Module '../src/types/game' has no exported member 'Biome'.` (`57`)
4. `Module '../lib/gameLoop' has no exported member 'processEconomyTurn'.` (`43`)
5. `Module '../lib/military' has no exported member 'movementTick'.` (`35`)
6. `Module '../lib/combat' has no exported member 'MoraleState'.` (`34`)
7. `Biome import/export mismatch variant` (`27`)
8. `Module '../lib/battalionTraining' has no exported member 'advanceBattalionTrainingOrders'.` (`12`)
9. `Biome declared locally but not exported` (`11`)
10. `Module '../lib/applyAiPlan' has no exported member 'applyAiInstantBuilds'.` (`6`)
11. `MAP_SIZE_PRESETS export/import mismatch` (`6`)
12. `Module '@/lib/aiTactics' has no exported member 'computeArmyComposition'.` (`6`)
13. `File ... is not a module` (`6`)
14. `Module '../lib/commanders' has no exported member 'rollCommanderIdentity'.` (`6`)
15. `Module '@/lib/siegeRecruitment' has no exported member 'siegeCompositionAllowsRecruit'.` (`6`)
16. `Module '../lib/contestedZone' has no exported member 'applyContestedZonePayout'.` (`6`)
17. `Object literal has unknown property 'playerId'` (`5`)
18. `Expected 4 arguments, but got 2` (`4`)
19. `Property 'size' does not exist on type 'Unit'` (`3`)
20. Type-safety/return typing violations (`implicit any`, missing return) (`2+`)

Pattern summary:

- Repeated **non-existent symbol imports** and **stale export assumptions** are the dominant class.
- Secondary pattern: **type contract breakage** in economy/state flows.

## Output-shape failures (`no_applyable_changes`)

- Count: `76`
- Dominant shape: `fenced_non_contract_output` (`76/76`)

Typical first-line patterns:

- ```

```
- `````json`
- Markdown/code answers that look plausible but are not parseable/applyable under arena apply contract.

## Concrete failure modes

1. **Import/export hallucination under broad edits**
  - Model references symbols not exported by current codebase.
  - Most common in `gameCore` and shared `types/game` imports.
2. **Type contract drift**
  - Object shapes and function signatures changed without coordinated call-site updates.
3. **Patch contract non-compliance**
  - Returns code snippets or JSON instead of a valid applyable patch/file output shape.
4. **Over-broad edits in high-fanout modules**
  - Editing `gameCore`/`types` without dependency-aware checks causes cascading compile failures.

## Targeted remediation plan

### A) Prompt and decoding controls (immediate)

- Force strict apply-contract format for all non-low tasks:
  - explicit first line requirements
  - hard negative examples showing invalid fenced-code-only responses
- Add anti-hallucination instruction:
  - "Only import symbols that already exist; if unsure, inspect and preserve existing exports."

### B) Training data repairs (high leverage)

- Build a focused repair corpus from failing trials:
  - `(bad_output -> corrected_output)` pairs for top 20 recurring signatures
  - include both compile-fix and apply-shape-fix pairs
- Add "export-safe edit" examples:
  - preserve import lists unless symbol existence is verified in-file.
- Add no-apply contract drills:
  - turn fenced snippets/prose into applyable diff/fenced-file contract outputs.

### C) Inference-time guardrail loop

- Before finalize, run one local "symbol validation rewrite" pass:
  - detect non-existent imports/exports in generated patch
  - auto-rewrite to existing symbols or minimal-safe change
- Keep bug-check loop enabled; add checks keyed to top signature list above.

### D) Evaluation instrumentation

- Split `verify_failed_unknown` into explicit sublabels from log text:
  - `verify_ts_missing_export`
  - `verify_ts_signature_mismatch`
  - `verify_ts_object_shape_mismatch`
  - `verify_output_not_applyable`
- Track these per specialist so promotion decisions are based on causal failure buckets.

## Prioritized next experiments

1. **Repair-set fine-tune v1**: top-20 TS signature fixes + no-apply contract fixes.
2. **Guarded inference v1**: one-pass symbol/import sanity rewrite before apply.
3. **A/B rerun (121 tasks)**:
  - baseline current specialist
  - +repair-set adapter
  - +repair-set adapter + guardrail loop

Success criterion:

- Reduce `typescript_compile_or_type_error` share from `84.6%` to `<60%`
- Reduce `no_applyable_changes` from `76` to `<20` across the 6-run specialist panel.

