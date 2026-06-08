#!/usr/bin/env python3
"""Per-subsection TypeScript sandbox environments for economistRL vitest integration.

Sandboxes **extend** Fallen Empire: real ``City`` / ``Player`` types and ``processEconomyTurn``
where the mechanic exists in ``gameLoop.ts``. Task-only variables live in ``taskExt`` or fixtures.
See ``economist_rl_game_sandbox`` and ``docs/SANDBOX_GAME_FIELD_ALIGNMENT.md``.
"""

from __future__ import annotations

import re
from typing import Any

from economist_rl_field_names import (
    SUBSECTION_CANONICAL_DEFAULTS,
    canonicalize_field_list,
    canonicalize_state,
)
from economist_rl_game_sandbox import SHARED_ENV_DIR

ECONOMY_SUBSECTIONS = frozenset(
    {
        "food_population_feedback",
        "market_elasticity_pricing",
        "labor_wage_productivity",
        "inventory_storage_spoilage",
        "upkeep_progressive_costs",
        "resource_projection_cache_integrity",
        "adversarial_multidomain_economy",
    }
)

GENERALIST_SUBSECTION = "generalist_bounded"


def _slug(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return token or "mechanic"


def resolve_subsection(task: dict[str, Any]) -> str:
    raw = str(task.get("subsection") or task.get("subskill") or "").strip()
    slug = _slug(raw)
    if slug in ECONOMY_SUBSECTIONS:
        return slug
    if "food" in slug and "population" in slug:
        return "food_population_feedback"
    if "market" in slug or "elastic" in slug or "price" in slug:
        return "market_elasticity_pricing"
    if "labor" in slug or "wage" in slug:
        return "labor_wage_productivity"
    if "inventory" in slug or "spoil" in slug or "storage" in slug:
        return "inventory_storage_spoilage"
    if "upkeep" in slug:
        return "upkeep_progressive_costs"
    if "cache" in slug or "projection" in slug:
        return "resource_projection_cache_integrity"
    if "adversarial" in slug or "multidomain" in slug:
        return "adversarial_multidomain_economy"
    return GENERALIST_SUBSECTION


def env_dir_for_subsection(subsection: str) -> str:
    return f"src/lib/economistRl/envs/{_slug(subsection)}"


def env_file_paths(subsection: str) -> list[str]:
    from economist_rl_game_sandbox import SHARED_FIXTURE, sandbox_integration_mode

    base = env_dir_for_subsection(subsection)
    if sandbox_integration_mode(subsection) == "lab_scalar":
        return [f"{base}/types.ts", f"{base}/runTicks.ts", f"{base}/index.ts"]
    return [
        SHARED_FIXTURE,
        f"{SHARED_ENV_DIR}/runAppliedSimCli.ts",
        f"{base}/types.ts",
        f"{base}/fixture.ts",
        f"{base}/runTicks.ts",
        f"{base}/index.ts",
    ]


def tick_function_name(task: dict[str, Any]) -> str:
    subsection = resolve_subsection(task)
    parts = [p for p in subsection.split("_") if p]
    return "apply" + "".join(p.title() for p in parts) + "Tick"


def _state_fields(task: dict[str, Any], *, fallback: list[str]) -> list[str]:
    sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
    fields = canonicalize_field_list([str(f).strip() for f in sim.get("relevant_state") or [] if str(f).strip()])
    return fields[:10] if fields else fallback


def _scenario_seed(task: dict[str, Any]) -> dict[str, float]:
    sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
    scenario = (sim.get("scenario") or {}).get("initial_state") if isinstance(sim.get("scenario"), dict) else {}
    if not isinstance(scenario, dict):
        return {}
    return canonicalize_state(scenario)


def _ts_object_literal(values: dict[str, float], *, indent: str = "  ") -> str:
    if not values:
        return f"{indent}scalar: 1,"
    lines = [f"{indent}{k}: {v}," for k, v in values.items()]
    return "\n".join(lines)


def _env_types_ts(subsection: str, fields: list[str], seeds: dict[str, float]) -> str:
    type_name = "".join(p.title() for p in subsection.split("_")) + "State"
    field_lines = "\n".join(f"  {name}: number;" for name in fields)
    scarcity = dict(seeds)
    surplus = dict(seeds)
    if subsection == "market_elasticity_pricing":
        scarcity.update({"storageGoods": 40, "storageCapGoods": 100, "marketPriceGold": 10, "demand": 1.0})
        surplus.update({"storageGoods": 160, "storageCapGoods": 100, "marketPriceGold": 10, "demand": 1.0})
    elif subsection == "food_population_feedback":
        scarcity.update({"storageFood": 40, "population": 100, "appliedBirthRate": 0.04})
        surplus.update({"storageFood": 200, "population": 100, "appliedBirthRate": 0.04})
    elif subsection == "labor_wage_productivity":
        scarcity.update({"playerGold": 80, "wage": 0.5, "expectedWage": 1.0, "cityMorale": 0.4, "productivity": 0.7})
        surplus.update({"playerGold": 800, "wage": 1.1, "expectedWage": 1.0, "cityMorale": 0.85, "productivity": 1.1})
    elif subsection == "inventory_storage_spoilage":
        scarcity.update(
            {"storageFood": 30, "storageCapFood": 100, "preservationModifier": 0.7, "foodDecayPerTick": 0.08}
        )
        surplus.update(
            {"storageFood": 140, "storageCapFood": 200, "preservationModifier": 0.95, "foodDecayPerTick": 0.03}
        )
    elif subsection == "upkeep_progressive_costs":
        scarcity.update({"playerGold": 60, "garrisonUnitCount": 24, "upkeepPerUnit": 3, "supplyDistance": 8})
        surplus.update({"playerGold": 900, "garrisonUnitCount": 8, "upkeepPerUnit": 1.5, "supplyDistance": 2})
    elif subsection == "resource_projection_cache_integrity":
        scarcity.update({"resourceProjectionCacheDirty": 1, "invalidationScopeCount": 4, "resourceProjectionValid": 0})
        surplus.update({"resourceProjectionCacheDirty": 1, "invalidationScopeCount": 0, "resourceProjectionValid": 0})
    seed_block = _ts_object_literal(seeds or {fields[0]: 1.0 for fields in fields[:1]})
    scar_block = _ts_object_literal(scarcity)
    sur_block = _ts_object_literal(surplus)
    extra_constants = ""
    if subsection == "market_elasticity_pricing":
        extra_constants = """
export const PRICE_FLOOR = 5;
export const PRICE_CAP = 25;
export const MULT_FLOOR = 0.75;
export const MULT_CAP = 1.35;
export const MAX_PRICE_STEP = 1.25;
export const PRICE_SMOOTH = 0.22;
"""
    elif subsection == "food_population_feedback":
        extra_constants = """
export const POP_BIRTH_RATE = 0.12;
export const POP_RESERVE_FOOD_FLOOR = 50;
export const POP_UNSAFE_FOOD_BUFFER_SPAN = 100;
export const FOOD_PRODUCED_PER_TICK = 16;
export const POP_FOOD_CONSUMED_PER_CAPITA = 0.18;
"""
    return f"""// economistRL sandbox environment: {subsection}
export type {type_name} = {{
{field_lines}
}};

export function clamp(value: number, lo: number, hi: number): number {{
  return Math.max(lo, Math.min(hi, value));
}}

export function clamp01(value: number): number {{
  return clamp(value, 0, 1);
}}
{extra_constants}
export const DEFAULT_SEED: {type_name} = {{
{seed_block}
}};

export const SCARCITY_SEED: {type_name} = {{
{scar_block}
}};

export const SURPLUS_SEED: {type_name} = {{
{sur_block}
}};
"""


def _env_run_ticks_ts(subsection: str) -> str:
    type_name = "".join(p.title() for p in subsection.split("_")) + "State"
    return f"""// economistRL sandbox environment: {subsection}
import type {{ {type_name} }} from './types';

export type TickFn = (state: {type_name}) => {type_name};

export function runTicks(
  fn: TickFn,
  initial: {type_name},
  tickCount = 20,
): {type_name}[] {{
  const trace: {type_name}[] = [{{ ...initial }}];
  let state = {{ ...initial }};
  for (let i = 0; i < tickCount; i += 1) {{
    state = fn({{ ...state }});
    trace.push({{ ...state }});
  }}
  return trace;
}}
"""


def _env_index_ts(subsection: str) -> str:
    return f"export * from './types';\nexport * from './runTicks';\n"


def env_starter_files(task: dict[str, Any]) -> list[dict[str, str]]:
    from economist_rl_game_sandbox import env_files_for_subsection, sandbox_integration_mode

    subsection = resolve_subsection(task)
    if sandbox_integration_mode(subsection) != "lab_scalar":
        return [{"path": p, "content": c} for p, c in env_files_for_subsection(subsection, task)]
    fields = _fields_for_subsection(task, subsection)
    seeds = _scenario_seed(task)
    return [
        {"path": p, "content": c}
        for p, c in [
            (env_file_paths(subsection)[0], _env_types_ts(subsection, fields, seeds)),
            (env_file_paths(subsection)[1], _env_run_ticks_ts(subsection)),
            (env_file_paths(subsection)[2], _env_index_ts(subsection)),
        ]
    ]


def _fields_for_subsection(task: dict[str, Any], subsection: str) -> list[str]:
    fallback = SUBSECTION_CANONICAL_DEFAULTS.get(
        subsection, SUBSECTION_CANONICAL_DEFAULTS[GENERALIST_SUBSECTION]
    )
    return _state_fields(task, fallback=fallback)


def _relative_import(from_file: str, to_file: str) -> str:
    from pathlib import Path

    depth = len(Path(from_file).parent.parts)
    prefix = "/".join([".."] * depth)
    stem = Path(to_file).with_suffix("").as_posix()
    return f"{prefix}/{stem}"


def mechanic_stub_body(task: dict[str, Any], lib_file: str) -> str:
    from economist_rl_game_sandbox import (
        game_turn_mechanic_tick_impl,
        hybrid_mechanic_tick_impl,
        sandbox_integration_mode,
    )

    subsection = resolve_subsection(task)
    task_id = str(task.get("id") or "task")
    fn = tick_function_name(task)
    type_name = "".join(p.title() for p in subsection.split("_")) + "State"
    signals = ", ".join(_signals(task)[:6]) or "bounded feedback"
    mode = sandbox_integration_mode(subsection)

    if mode == "game_turn":
        tick_impl, _extra = game_turn_mechanic_tick_impl(subsection)
        types_path = _relative_import(lib_file, f"src/lib/economistRl/envs/{subsection}/types.ts")
        shared_path = _relative_import(lib_file, "src/lib/economistRl/envs/shared/fixture.ts")
        game_loop_path = _relative_import(lib_file, "src/lib/gameLoop.ts")
        extra_import = (
            f"import {{ processEconomyTurn }} from '{game_loop_path}';\n"
            f"import {{ recordsToMaps }} from '{shared_path}';\n"
            f"import type {{ {type_name} }} from '{types_path}';\n"
        )
        return f"""// economistRL task mechanic: {task_id}
// Signals: {signals}
{extra_import}

export type EconomistRlState = {type_name};

export function {fn}(state: EconomistRlState): EconomistRlState {{{tick_impl}
}}
"""

    if mode == "hybrid_city":
        tick_impl, extra_import = hybrid_mechanic_tick_impl(subsection)
        types_path = _relative_import(lib_file, f"src/lib/economistRl/envs/{subsection}/types.ts")
        if subsection == "market_elasticity_pricing":
            extra_import = (
                f"import {{ clamp, MULT_CAP, MULT_FLOOR, PRICE_CAP, PRICE_FLOOR, PRICE_SMOOTH, "
                f"type {type_name} }} from '{types_path}';\n"
            )
        else:
            extra_import = f"import type {{ {type_name} }} from '{types_path}';\n"
        return f"""// economistRL task mechanic: {task_id}
// Signals: {signals}
{extra_import}

export type EconomistRlState = {type_name};

export function {fn}(state: EconomistRlState): EconomistRlState {{{tick_impl}
}}
"""

    env_types = _relative_import(lib_file, env_file_paths(subsection)[0])
    fields = _fields_for_subsection(task, subsection)
    state_lines = "\n".join(f"  {name}: number;" for name in fields)

    if subsection == "market_elasticity_pricing":
        tick_impl = f"""
  const pressure = state.storageCapGoods / Math.max(1, state.storageGoods);
  const multiplier = clamp(1 + 0.18 * (pressure - 1), MULT_FLOOR, MULT_CAP);
  const target = clamp(state.marketPriceGold * multiplier * Math.max(0.01, state.demand), PRICE_FLOOR, PRICE_CAP);
  const marketPriceGold = clamp(
    state.marketPriceGold + PRICE_SMOOTH * (target - state.marketPriceGold),
    PRICE_FLOOR,
    PRICE_CAP,
  );
  const step = Math.abs(marketPriceGold - state.marketPriceGold);
  if (step > MAX_PRICE_STEP) {{
    const sign = marketPriceGold >= state.marketPriceGold ? 1 : -1;
    return {{
      ...state,
      marketPriceGold: clamp(state.marketPriceGold + sign * MAX_PRICE_STEP, PRICE_FLOOR, PRICE_CAP),
      scarcityPressure: Math.max(0, pressure - 1),
      surplusPressure: Math.max(0, 1 - pressure),
    }};
  }}
  return {{
    ...state,
    marketPriceGold,
    scarcityPressure: Math.max(0, pressure - 1),
    surplusPressure: Math.max(0, 1 - pressure),
  }};"""
        extra_import = (
            f"import {{ clamp, MULT_CAP, MULT_FLOOR, MAX_PRICE_STEP, PRICE_CAP, PRICE_FLOOR, "
            f"PRICE_SMOOTH, type {type_name} }} from '{env_types}';"
        )
    elif subsection == "food_population_feedback":
        tick_impl = """
  const bufferRatio = clamp01(
    (state.storageFood - POP_RESERVE_FOOD_FLOOR) / Math.max(1, POP_UNSAFE_FOOD_BUFFER_SPAN),
  );
  const appliedBirthRate = Math.max(0, POP_BIRTH_RATE * bufferRatio);
  const foodConsumed = state.population * POP_FOOD_CONSUMED_PER_CAPITA;
  let storageFood = state.storageFood + FOOD_PRODUCED_PER_TICK - foodConsumed;
  if (storageFood < 0) {
    storageFood = 0;
  }
  const population = state.population + state.population * appliedBirthRate;
  return { ...state, storageFood, population, appliedBirthRate };"""
        extra_import = (
            f"import {{ clamp01, FOOD_PRODUCED_PER_TICK, POP_BIRTH_RATE, POP_FOOD_CONSUMED_PER_CAPITA, "
            f"POP_RESERVE_FOOD_FLOOR, POP_UNSAFE_FOOD_BUFFER_SPAN, type {type_name} }} from '{env_types}';"
        )
    elif subsection == "labor_wage_productivity":
        tick_impl = """
  const wageGap = state.expectedWage - state.wage;
  const cityMorale = clamp01(state.cityMorale + 0.05 * wageGap);
  const productivity = clamp(state.productivity + 0.08 * cityMorale - 0.04 * Math.max(0, wageGap), 0.2, 1.5);
  const totalEmployed = state.totalEmployed * productivity;
  const wage = clamp(state.wage + 0.1 * wageGap, 0.1, 3);
  return { ...state, cityMorale, productivity, totalEmployed, wage };"""
        extra_import = f"import {{ clamp, clamp01, type {type_name} }} from '{env_types}';"
    elif subsection == "inventory_storage_spoilage":
        tick_impl = """
  const decay = state.foodDecayPerTick * (1 - clamp01(state.preservationModifier));
  const afterDecay = Math.max(0, state.storageFood * (1 - decay));
  const storageFood = Math.min(afterDecay, state.storageCapFood);
  return { ...state, storageFood };"""
        extra_import = f"import {{ clamp01, type {type_name} }} from '{env_types}';"
    elif subsection == "upkeep_progressive_costs":
        tick_impl = """
  const distanceFactor = 1 + 0.12 * state.supplyDistance;
  const upkeepTotal = state.garrisonUnitCount * state.upkeepPerUnit * distanceFactor;
  const netGoldDelta = -upkeepTotal;
  const playerGold = Math.max(0, state.playerGold + netGoldDelta);
  return { ...state, upkeepTotal, netGoldDelta, playerGold };"""
        extra_import = f"import {{ type {type_name} }} from '{env_types}';"
    elif subsection == "resource_projection_cache_integrity":
        tick_impl = """
  let invalidationScopeCount = state.invalidationScopeCount;
  let resourceProjectionCacheDirty = state.resourceProjectionCacheDirty;
  if (invalidationScopeCount > 0) {
    invalidationScopeCount = Math.max(0, invalidationScopeCount - 1);
  } else if (resourceProjectionCacheDirty > 0) {
    resourceProjectionCacheDirty = 0;
  }
  const stillDirty = resourceProjectionCacheDirty > 0 || invalidationScopeCount > 0;
  const resourceProjectionValid = stillDirty ? 0 : 1;
  return {
    ...state,
    resourceProjectionValid,
    resourceProjectionCacheDirty: stillDirty ? 1 : 0,
    invalidationScopeCount,
  };"""
        extra_import = f"import {{ type {type_name} }} from '{env_types}';"
    elif subsection == "adversarial_multidomain_economy":
        tick_impl = """
  const foodConsumed = state.population * 0.15;
  let storageFood = Math.max(0, state.storageFood + 12 - foodConsumed);
  const pressure = state.storageCapGoods / Math.max(1, state.storageGoods);
  const marketPriceGold = clamp(state.marketPriceGold * (1 + 0.05 * (pressure - 1)), 5, 25);
  const upkeepTotal = state.garrisonUnitCount * 1.5;
  const playerGold = Math.max(0, state.playerGold - upkeepTotal);
  return { ...state, storageFood, marketPriceGold, playerGold, upkeepTotal };"""
        extra_import = f"import {{ clamp, type {type_name} }} from '{env_types}';"
    else:
        tick_impl = """
  const pressure = state.pressure ?? 1;
  const boundedSignal = clamp01((state.scalar ?? 1) * pressure);
  return { ...state, scalar: boundedSignal, boundedSignal };"""
        extra_import = f"import {{ clamp01, type {type_name} }} from '{env_types}';"

    return f"""// economistRL task mechanic: {task_id}
// Signals: {signals}
{extra_import}

export type EconomistRlState = {type_name};

export function {fn}(state: EconomistRlState): EconomistRlState {{{tick_impl}
}}
"""


def _wrap_test_stub(
    task_id: str,
    check_slug: str,
    env_ticks: str,
    extra: str,
    mechanic_import: str,
    fn: str,
    body: str,
) -> str:
    return f"""// economistRL targeted test: {task_id}
import {{ describe, it, expect }} from 'vitest';
import {{ runTicks }} from '{env_ticks}';
{extra}import {{ {fn} }} from '{mechanic_import}';

describe('economistRL {task_id}', () => {{
  it('{check_slug}', () => {{{body}
  }});
}});
"""


def test_stub_body(task: dict[str, Any], *, lib_file: str, test_file: str) -> str:
    """Vitest stub: one ``it('goal_*')`` per ``simulation_spec.goals`` entry."""
    from economist_rl_vitest_goals import render_vitest_stub

    return render_vitest_stub(task, lib_file=lib_file, test_file=test_file)


def _signals(task: dict[str, Any]) -> list[str]:
    static = task.get("static_code_mechanics") if isinstance(task.get("static_code_mechanics"), dict) else {}
    out = [str(s) for s in static.get("flexible_signals") or [] if str(s).strip()]
    for mech in (task.get("expect") or {}).get("mechanics") or []:
        if isinstance(mech, dict):
            out.extend(str(k) for k in mech.get("keywords") or [])
    return out[:12]


def _outcome_check_names(task: dict[str, Any]) -> list[str]:
    targeted = task.get("targeted_tests") if isinstance(task.get("targeted_tests"), dict) else {}
    checks = [str(item).strip() for item in targeted.get("outcome_checks") or [] if str(item).strip()]
    return checks or ["bounded deterministic updates"]


def sandbox_starter_bodies(task: dict[str, Any], *, lib_file: str, test_file: str) -> list[dict[str, str]]:
    """Env package + mechanic + vitest for one sandbox task."""
    out = env_starter_files(task)
    out.append({"path": lib_file, "content": mechanic_stub_body(task, lib_file)})
    if test_file:
        out.append({"path": test_file, "content": test_stub_body(task, lib_file=lib_file, test_file=test_file)})
    return out
