# Router Task Taxonomy (Measured Specialist Roles)

Router labels still use stable `adapter_id` values for checkpoint paths and historical compatibility, but user-facing specialist roles were renamed after the 2026-05-30 full crossdomain 121-task baseline. Prefer the measured role names in UI, docs, and routing explanations.

| Stable adapter ID | Measured role name | Accurate routing surface |
|---|---|---|
| `loading_screen` | `ui_surface_composer` | UI/HUD visual surfaces, loading/title screens, panel themes, sprite overlays, isometric display, render contracts |
| `hud_status` | `ui_state_signals` | HUD/status signals, resource/status indicators, state exposure, indicator logic, UI contracts, input flow |
| `economy_tooltip` | `resource_ui_projection` | Resource projection UI, economy-facing tooltips, cost display, production signals, resource cache, signed deltas |
| `combat_risk` | `army_ui_flow` | Army UI/order flow, formation flow, movement UI, selection flow, tactical controls, combat-preview presentation |
| `save_load_api_guard` | `state_contract_guard` | State persistence, serialization, snapshot schema, request validation, identity integrity, API/state contracts |
| `ai_planning_explanation` | `crossdomain_state_patch` | Generic crossdomain state patches, schema updates, shared logic, repair candidates, fallback patching |
| `economistRL` | `economistRL` | Experimental RL lane for hard economy simulation, market dynamics, food/population feedback, labor productivity, spoilage, and progressive upkeep |
| `documentation` | `documentation` | Project docs normalization, workflow/run-history/process documentation tasks |
| `general_fallback` | `general_fallback` | Tasks that do not match a specialist family strongly enough |

## Adapter-first route defaults

- Local-first: `loading_screen` / `ui_surface_composer`, `hud_status` / `ui_state_signals`, `economy_tooltip` / `resource_ui_projection`, `documentation`, `general_fallback`
- Hybrid: `combat_risk` / `army_ui_flow`, `save_load_api_guard` / `state_contract_guard`, `ai_planning_explanation` / `crossdomain_state_patch`, `economistRL`
- Frontier: reserved for future specialist adapters requiring direct escalation

Global policy overrides still apply:

- high-risk keywords (`security`, `auth`, `cryptography`, `billing`, etc.) => `frontier`
- long prompt token threshold => `frontier`

## Demoted Specialist Identities

The crossdomain baseline did not support these as primary specialist identities for the older adapters. `economy_simulation` and `market_dynamics` are now explicit `economistRL` experiment targets; the rest remain hard/escalation tags or retraining targets:

- `economy_simulation` (`economistRL` experimental)
- `market_dynamics` (`economistRL` experimental)
- `ai_strategy`
- `planner_policy`
- `combat_math`
- `targeting_policy`
- `multidomain_systems`

