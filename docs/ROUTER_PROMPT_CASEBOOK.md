# Router Prompt Casebook

This casebook feeds both routing training curation and RAG retrieval for router debugging.

## Keyword-to-adapter examples

- `loading screen`, `splash screen`, `title screen`, `crest`, `banner` => `loading_screen`
- `hud`, `status panel`, `status overlay` => `hud_status`
- `economy`, `tooltip`, `income`, `upkeep`, `net income` => `economy_tooltip`
- `combat risk`, `battle risk`, `enemy stats versus` => `combat_risk`
- `save/load`, `autosave`, `serialization`, `auth guard` => `save_load_api_guard`
- `ai planning`, `decision rationale`, `why ai` => `ai_planning_explanation`
- `project_state`, `session_log`, `run_history`, `workflow` => `documentation`

## Prompt cases (gold labels)

### Case: Loading polish

- Prompt: "Polish the loading screen visuals and improve title legibility with medieval styling."
- Expected adapter: `loading_screen`
- Expected route: `local`

### Case: Economy explanation

- Prompt: "Improve the economy tooltip wording so players understand income, upkeep, and net gold."
- Expected adapter: `economy_tooltip`
- Expected route: `local`

### Case: Combat preview

- Prompt: "Add combat risk previewing with enemy-vs-player stat comparison before attack confirmation."
- Expected adapter: `combat_risk`
- Expected route: `hybrid`

### Case: Save/load API guardrails

- Prompt: "Audit save/load endpoints for authorization and serialization safety before release."
- Expected adapter: `save_load_api_guard`
- Expected route: `hybrid`

### Case: Documentation normalization

- Prompt: "Normalize PROJECT_STATE and SESSION_LOG entries and keep ml_workflow usage docs consistent."
- Expected adapter: `documentation`
- Expected route: `local`

### Case: Security architecture escalation

- Prompt: "Design authentication and cryptography controls for production save signing and key rotation."
- Expected adapter: `general_fallback`
- Expected route: `frontier` (high-risk override)

