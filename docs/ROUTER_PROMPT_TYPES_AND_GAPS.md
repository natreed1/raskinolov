# Router Prompt Types And Gaps (2026-05-18)

This note summarizes prompt families observed from workflow/routing history, structural gaps, and the new routing artifacts created for gap-fill training.

## Prompt Types Users Ask

- `ui_hierarchy`: loading/HUD readability, visual order, dense UI scanability.
- `constraint_rewrite`: strict style constraints (one sentence, no bullets, short lines, terse chips).
- `systems_explainer`: economy/combat/planning rationale and cause-effect explanation.
- `api_guardrails`: save/load auth, schema validation, serialization safety.
- `api_contract`: response shape, migration versioning, error contract design.
- `planning_rationale`: AI intent + fallback branch + tradeoff explanation.
- `planning_review`: review planner output and propose safer branch.
- `docs_normalization`: align `PROJECT_STATE`, `WORKFLOW`, and run-history wording.
- `docs_summary`: summarize run results and promotion blockers.
- `run_analysis`: infer failure themes from run history/log snippets.
- `mechanical_edit`: rename/boilerplate/small deterministic code edits.
- `high_risk_security`: cryptography/authentication/production safety escalation.
- `high_risk_architecture`: billing/privacy/migration architecture work.
- `incident_response`: distributed data-loss/race-condition triage.
- `multi_intent_explainer`: prompts combining multiple specialist intents.
- `multi_agent_mixed`: prompts that benefit from split subtasks and merge.

## Structural Gaps Found

- Coverage was shallow for curated routing labels (`8` rows in `router_cases_v1`).
- Adapter labels were under-sampled for cross-domain and constraint-heavy prompts.
- Legacy route labels were skewed to local/hybrid and had limited explicit frontier escalation rows.
- Similarity prototype seeds were imbalanced by adapter family.
- Skill extraction v1 retained only one stable skill under strict thresholds, limiting manifold interpretability for routing actions.

## New Nodes, Manifolds, And Training Data Added

- New curated routing cases: `data/routing/router_cases_v2.jsonl` (`32` rows across prompt types above).
- New manifold prototype set: `data/routing/manifold_prototype_prompts_v2.json`.
- Router prototype ingestion update: `scripts/model_router.py` now loads `manifold_prototype_prompts_v2.json` and supports `.jsonl` prototype sources.
- New extracted skill/manifold artifacts:
  - `data/routing/skills_v2.json`
  - `data/routing/specialist_skill_profiles_v2.json`
  - `data/routing/skill_manifolds_v2.json`
- New routing classifier dataset/version:
  - `data/lora/routing_classifier/router-v3-gapfill-20260518/manifest.json`
  - rows: `52` (`44` train, `5` valid, `3` test)
- New trained classifier artifact:
  - `training/router_classifier_v2/manifest.json`
  - vocab: `350`
  - train/valid/test accuracy: `1.0` / `0.6` / `0.6667`

## Initial Outcome Snapshot

- Routing benchmark with v2 classifier in hybrid mode:
  - summary: `benchmarks/results/routing_policy_summary_v3_classifier.json`
  - route accuracy: `12/12`
  - adapter accuracy: `12/12`
  - overall accuracy: `12/12`

## Follow-on Build + Train Directions

- Expand curated rows for low-support labels in v2 splits (especially `combat_risk`, `hud_status`, `save_load_api_guard` cross-domain blends).
- Add a mixed-task benchmark file focused on multi-intent prompts and run promotion gate checks against it.
- Promote `router_classifier_v2` behind a controlled env gate (`ROUTER_ADAPTER_SELECTION_MODE=hybrid`, `ROUTER_CLASSIFIER_DIR=training/router_classifier_v2`) before default switch.
- Keep periodic manifold extraction runs with the same thresholds used for `skills_v2` to track stability drift over new benchmark cycles.
