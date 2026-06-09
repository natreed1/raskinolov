# Lambda Council Behavior Baselines 2026-05-26

Behavior baseline extracted from the recovered Lambda council traces.

**Interpretation:** these are persona/behavior baselines from a single base model run. Lineage fields come from `training/adapter_registry_v1.json`; this artifact does not prove per-participant LoRA weights were loaded.

- Source rows: `benchmarks/results/lambda_recovery/council121-20260526/council_conversation_eval_rows_council121-20260526.jsonl`
- Source summary: `benchmarks/results/lambda_recovery/council121-20260526/council_conversation_eval_summary_council121-20260526.json`
- Rows: 121

## Ranked Behaviors

| Base expert | Lineage | Seen | Winner tasks | Winner mentions | Avg score | Behavior markers |
|---|---|---:|---:|---:|---:|---|
| `loading_screen` | `loading_screen:v2:cycle2` | 52 | 52 | 90 | 0.7803 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `hud_status` | `hud_status:v3:cycle3` | 25 | 25 | 37 | 0.7785 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `ai_planning_explanation` | `ai_planning_explanation:v1:champion` | 18 | 18 | 28 | 0.7795 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, implementation snippets |
| `combat_risk` | `combat_risk:v2_tsc_resume:cycle4` | 12 | 12 | 16 | 0.7834 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `save_load_api_guard` | `save_load_api_guard:v1:champion` | 7 | 7 | 12 | 0.7782 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `economy_tooltip` | `economy_tooltip:v1:cycle2+routing_fastpath_v1` | 19 | 4 | 5 | 0.7849 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, implementation snippets |
| `precise_short` | `generalist_profile` | 121 | 27 | 27 | 0.7037 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `sliding_window` | `generalist_profile` | 121 | 27 | 27 | 0.6990 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |
| `wide_compressed` | `generalist_profile` | 121 | 0 | 0 | 0.6961 | explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note |

## Top Specialist Examples

### `loading_screen`

Lineage: `loading_screen:v2:cycle2`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note, implementation snippets, verification/testing reminder, domain-specific concise UI copy

- `patch-node-path-runtime-resolution-01` winner `loading_screen::assertive` score `0.7839`: **Recommendation:** To ensure that bare imports resolve reliably in your production container without moving source files, consider adding an alias resolution strategy in your TypeScript configuration (`tsconfig.json`). This can be done using the `paths` option to map import paths to their actual locations. For example: ```json { "compilerOptions": { "baseUrl": ".", "paths": { "@shared/*": ["app/src/shared/*"] } } } ``` This approach allows you to use aliases like `@shared/utils` instead of rela
- `patch-test-env-registry-contract-04` winner `loading_screen::assertive` score `0.7879`: **Recommendation:** To define a typed registry for deterministic test environments, I propose creating a structured data format using JSON. This format will include the following fields: 1. `id`: A unique identifier for each test environment. 2. `title`: A brief name or title for the test environment. 3. `description`: A detailed description of the test environment. 4. `boot_mode`: The method used to start the test environment (e.g., cold boot, warm boot). 5. `seed` (optional): A random seed val

### `hud_status`

Lineage: `hud_status:v3:cycle3`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note, verification/testing reminder, domain-specific concise UI copy

- `patch-multiplayer-snapshot-types-09` winner `hud_status::assertive` score `0.7717`: **Recommendation:** To address the issue of multiplayer snapshots occasionally desyncing with store/UI expectations, we need to implement a comprehensive solution that focuses on patch snapshot ingestion and type alignment between the HUD and simulation state. Here’s a detailed plan: ### Step 1: Define a Robust Synchronization Protocol Develop a new synchronization protocol that ensures all data structures and types are consistently aligned across the HUD and simulation state. This protocol shou
- `patch-multiplayer-snapshot-types-09` winner `hud_status::cautious` score `0.7701`: **Recommendation:** To address the issue of multiplayer snapshots occasionally desyncing with store/UI expectations, we need to ensure that the patch snapshot ingestion process aligns types correctly between the HUD and simulation state. Here is a revised and more detailed approach: 1. **Type Alignment Verification:** - **Files Needed:** `SnapshotIngestion.cpp`, `HUDState.h`, `SimulationState.h` - **Action:** Implement a function `verifyTypeAlignment` in `SnapshotIngestion.cpp` that checks if th

### `ai_planning_explanation`

Lineage: `ai_planning_explanation:v1:champion`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, implementation snippets, verification/testing reminder, domain-specific concise UI copy

- `patch-ai-plan-shared-application-05` winner `ai_planning_explanation::assertive` score `0.7717`: **Recommendation:** To address the divergence between AI plan execution during training and runtime, I recommend introducing a shared plan-application module named `UnifiedPlanExecutor`. This module will encapsulate the logic for executing plans consistently across both phases. Here’s how we can implement this: 1. **Create the UnifiedPlanExecutor Module:** - Develop a new Python file, e.g., `unified_plan_executor.py`, which contains the `UnifiedPlanExecutor` class. - The class should have method
- `patch-ai-plan-shared-application-05` winner `ai_planning_explanation::cautious` score `0.7701`: **Recommendation:** To address the divergence between AI plan execution during training and runtime, I recommend introducing a shared plan-application module named `UnifiedPlanExecutor`. This module will encapsulate the logic for executing plans consistently across both phases. Here’s how we can implement this: 1. **Create the UnifiedPlanExecutor Module:** - Develop a new Python file, e.g., `unified_plan_executor.py`, which contains the `UnifiedPlanExecutor` class. - The class should have method

### `combat_risk`

Lineage: `combat_risk:v2_tsc_resume:cycle4`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note, implementation snippets, verification/testing reminder, domain-specific concise UI copy

- `patch-ai-move-target-eligibility-28` winner `combat_risk::assertive` score `0.7858`: **Recommendation:** To address the issue of patching AI move-target selection to handle unit-state eligibility based on strategy policy and food constraints, I propose the following approach: 1. **Integrate Strategy Policies and Food Constraints:** - Modify the `MoveTargetSelection` class by adding a method `evaluateUnitEligibility`. - This method will take into account both the current state of the unit (e.g., starving, moving) and the strategy policy to determine if the unit can be assigned a 
- `patch-archer-max-range-auto-positioning-49` winner `combat_risk::assertive` score `0.7717`: **Recommendation:** To ensure archer units maintain their maximum effective attack range during engagements while adhering to legal max-range standoff positions, updating when fronts shift, and without breaking melee pathing or engagement resolution determinism, the following steps should be taken: 1. **Create a New Positioning Module:** - Develop a new module named `ArcherRangeMaintainer` that will manage the positioning logic for archer units. - This module should be integrated into the existi

### `save_load_api_guard`

Lineage: `save_load_api_guard:v1:champion`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, override/escalation note, implementation snippets, verification/testing reminder, domain-specific concise UI copy

- `patch-node-path-runtime-resolution-01` winner `save_load_api_guard::assertive` score `0.7839`: **Recommendation:** To ensure that bare imports resolve reliably in your production container without moving source files, consider adding an alias resolution strategy in your TypeScript configuration (`tsconfig.json`). This can be done using the `paths` option to map import paths to their actual locations. For example: ```json { "compilerOptions": { "baseUrl": ".", "paths": { "@shared/*": ["app/src/shared/*"] } } } ``` This approach allows you to use aliases like `@shared/utils` instead of rela
- `patch-builder-automation-rebalance-15` winner `save_load_api_guard::assertive` score `0.7852`: **Recommendation:** To address the stale builder automation decisions following recent building/progression updates, I propose the following comprehensive approach: 1. **Automated Policy Review and Update:** - Develop an automated script that periodically reviews the current automation policies against the latest game updates. - The script should identify discrepancies between the existing policies and the updated game mechanics. 2. **Dynamic Contextual Patching:** - Implement a dynamic contextu

### `economy_tooltip`

Lineage: `economy_tooltip:v1:cycle2+routing_fastpath_v1`

Behavior markers: explicit recommendation framing, risk callouts, needed-file inventory, stepwise action plan, implementation snippets, verification/testing reminder, domain-specific concise UI copy

- `patch-hud-economy-mixed-panel-26` winner `economy_tooltip::assertive` score `0.7885`: **Recommendation:** To integrate economy-relevant cues into the HUD while maintaining tactical readability and avoiding visual drift, I propose the following steps: 1. **Define Requirements:** - Identify key economy-related metrics to be displayed (e.g., upkeep costs, pressure levels, readiness context). - Ensure these metrics do not interfere with the current army stack UX. 2. **Design Modular Overlay:** - Create a modular overlay system that can dynamically display economy-related cues based o
- `patch-worker-wage-productivity-balance-47` winner `economy_tooltip::assertive` score `0.7717`: **Recommendation:** To implement worker wages in the patch economy dynamics, consider the following detailed plan: 1. **Define Wage-Productivity Function:** - Create a new file named `wage_productivity.py`. - Define a function `calculate_productivity(wage)` that returns an increase in productivity based on the wage. The function should have a balanced curve to prevent excessive exploitation. For example: ```python def calculate_productivity(wage): return min(1.0, wage / 100) * wage ``` 2. **Modi
