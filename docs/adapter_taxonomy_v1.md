# Adapter Taxonomy v1

This document locks the task-family contract for the v1 multi-adapter architecture.

## Frozen families

- `general_fallback`
- `documentation`
- `loading_screen`
- `hud_status`
- `economy_tooltip`
- `combat_risk`
- `save_load_api_guard`
- `ai_planning_explanation`

## First-wave specialized rollout

1. `hud_status`
2. `economy_tooltip`
3. `combat_risk`
4. `save_load_api_guard`
5. `ai_planning_explanation`

`loading_screen` is promoted in the adapter registry (`loading-screen-polish`); older notes may still reference `general_fallback` for loading-only traffic.

## Task mapping

- `mlx-lora-docs-normalize` -> `documentation` (MLX/LoRA lab documentation steward; routing optional)
- `loading-screen-polish` -> `loading_screen`
- `hud-status-summary` -> `hud_status`
- `economy-tooltip` -> `economy_tooltip`
- `combat-risk-preview` -> `combat_risk`
- `save-load-api-guard` -> `save_load_api_guard`
- `ai-planning-explanation` -> `ai_planning_explanation`

## Mixed-intent rule

Routing computes primary and secondary candidates. If confidence margin is narrow (high ambiguity), the router escalates to council mode before direct API escalation, except hard safety overrides.

## Registry schema highlights

`training/adapter_registry_v1.json` stores:

- `adapter_id`, `task_family`, `task_ids`
- `base_model`, `adapter_path`, `lineage`
- `promotion_state` (`shadow`, `canary`, `champion`, `frozen`, `rollback`)
- `policy_version`, `council_thresholds_version`
- canary policy defaults (`5,15,35,60,100` traffic progression)

