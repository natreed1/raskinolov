# EconomistRL sandbox: extend the game, add task variables

## Intent

Each economistRL sandbox is a **testing extension** of Fallen Empire — not a parallel toy economy.

| Layer | What it is |
|-------|------------|
| **Game core** | Real `City`, `Player`, `Unit`, `processEconomyTurn`, `computeEmpireIncomeStatement`, constants from `@/types/game` |
| **Task extension** | Fields that do not exist in production yet (`taskExt.marketPriceGold`, projection invalidation counters, …) |
| **Task mechanic** | `src/lib/economistRl/<task_slug>/mechanic.ts` — refines one tick: call game entry points and/or update `taskExt` with clamps |
| **Vitest** | Asserts on **game paths** (`cities[0].storage.food`, `cities[0].population`, `players[0].gold`) or `taskExt` where fictional |

Flat names in `simulation_spec.scenario.initial_state` (`storageFood`, `currentStock`, …) are **authoring aliases** only. They are normalized when building fixtures (`economist_rl_field_names.py`).

## Integration modes (`economist_rl_game_sandbox.py`)

| Mode | Subsections | State shape | Default tick |
|------|-------------|-------------|--------------|
| `game_turn` | food, inventory, labor | `{ cities, units, players, tiles, territory, turn }` | `processEconomyTurn(...)` |
| `hybrid_city` | market, upkeep, cache, adversarial | `{ city, players, taskExt, turn }` | Game `City` + bounded `taskExt` update |
| `lab_scalar` | generalist | flat scalars | lab-only |

Shared helpers: `src/lib/economistRl/envs/shared/fixture.ts` (`minimalCity`, `minimalPlayer`, `recordsToMaps`, `cloneCity`).

## What is *not* mirrored yet

- **Full map**: sandboxes use empty `tiles` / `territory` unless a task seeds them — production farms need territory for full `computeCityProductionRate`.
- **Private phases**: `populationGrowthPhase` is not exported; food/inventory/labor sandboxes run the full `processEconomyTurn` cycle instead of isolating one phase.
- **Military upkeep**: upkeep hybrid tasks use `taskExt` + `players[].gold`; they do not call `upkeepTick` from `military.ts` yet.
- **Market price**: no `marketPriceGold` on `City` in production — hybrid mode keeps it in `taskExt` while stock pressure uses `city.storage.goods` / `storageCap.goods`.

## Regenerating starters

```bash
ECONOMIST_RL_SOURCE_REPO=~/fallen-empire \
  python scripts/adapters/tag_economist_rl_task_execution.py \
  --in benchmarks/economistRL_tasks_v2_coding.json \
  --out benchmarks/economistRL_tasks_v3_execution.json \
  --refresh-all
```

## Rollout guidance for models

1. Read env `fixture.ts` / `types.ts` for the subsection — that is the graded contract.
2. Import from `@/types/game` and `@/lib/gameLoop` when the reference stub does.
3. Update `City` / `Player` / `taskExt` in the tick return value; do not loop over fictional `state.cities` Record maps unless the task explicitly adds them in `taskExt`.
