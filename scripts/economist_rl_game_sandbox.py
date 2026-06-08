#!/usr/bin/env python3
"""Game-extending economistRL sandbox TypeScript generators.

Sandboxes **import and run** Fallen Empire types and economy entry points where they
exist (``City``, ``processEconomyTurn``, ``computeEmpireIncomeStatement``, …). Task-only
fields live in ``taskExt`` or scenario-derived fixtures — not a parallel flat economy.

Authoring: ``simulation_spec.scenario.initial_state`` may still use flat aliases
(``storageFood``, …); :mod:`economist_rl_field_names` normalizes them when building seeds.
"""

from __future__ import annotations

from typing import Any

from economist_rl_field_names import canonicalize_state

SHARED_ENV_DIR = "src/lib/economistRl/envs/shared"
SHARED_FIXTURE = f"{SHARED_ENV_DIR}/fixture.ts"

# Full economy cycle via production gameLoop (population, storage, production, …).
GAME_TURN_SUBSECTIONS = frozenset(
    {
        "food_population_feedback",
        "inventory_storage_spoilage",
        "labor_wage_productivity",
    }
)

# Real ``City`` / ``Player`` plus task-only extensions (no full turn).
HYBRID_CITY_SUBSECTIONS = frozenset(
    {
        "market_elasticity_pricing",
        "upkeep_progressive_costs",
        "resource_projection_cache_integrity",
        "adversarial_multidomain_economy",
    }
)


def sandbox_integration_mode(subsection: str) -> str:
    if subsection in GAME_TURN_SUBSECTIONS:
        return "game_turn"
    if subsection in HYBRID_CITY_SUBSECTIONS:
        return "hybrid_city"
    return "lab_scalar"


def shared_fixture_ts() -> str:
    return """// Shared fixtures: minimal Fallen Empire entities for economistRL sandboxes.
import type { City, Player, Tile, TerritoryInfo, Unit } from '@/types/game';

export const SANDBOX_PLAYER_ID = 'economistrl-player-1';
export const SANDBOX_CITY_ID = 'economistrl-city-1';

const EMPTY_STORAGE = {
  food: 0,
  goods: 0,
  guns: 0,
  gunsL2: 0,
  iron: 0,
  stone: 0,
  wood: 0,
  refinedWood: 0,
};

export function minimalPlayer(gold = 500): Player {
  return {
    id: SANDBOX_PLAYER_ID,
    name: 'Sandbox Empire',
    color: '#38bdf8',
    gold,
    taxRate: 0.1,
    foodPriority: 'civilians_first',
    isHuman: true,
  };
}

export function minimalCity(
  overrides: Partial<City> & { storageFood?: number; storageCapFood?: number; storageGoods?: number } = {},
): City {
  const { storageFood, storageCapFood, storageGoods, storage, storageCap, ...rest } = overrides;
  return {
    id: SANDBOX_CITY_ID,
    name: 'Sandbox City',
    q: 0,
    r: 0,
    ownerId: SANDBOX_PLAYER_ID,
    population: 100,
    morale: 0.7,
    storage: {
      ...EMPTY_STORAGE,
      ...(storage ?? {}),
      ...(storageFood != null ? { food: storageFood } : {}),
      ...(storageGoods != null ? { goods: storageGoods } : {}),
    },
    storageCap: {
      ...EMPTY_STORAGE,
      food: storageCapFood ?? 200,
      goods: 50,
      guns: 50,
      gunsL2: 0,
      iron: 50,
      stone: 50,
      wood: 50,
      refinedWood: 50,
      ...(storageCap ?? {}),
    },
    buildings: [],
    ...rest,
  };
}

export function cloneCity(city: City): City {
  return {
    ...city,
    storage: { ...city.storage },
    storageCap: { ...city.storageCap },
    buildings: city.buildings.map(b => ({ ...b })),
  };
}

export function recordsToMaps(
  tiles: Record<string, Tile>,
  territory: Record<string, TerritoryInfo>,
): { tiles: Map<string, Tile>; territory: Map<string, TerritoryInfo> } {
  return {
    tiles: new Map(Object.entries(tiles)),
    territory: new Map(Object.entries(territory)),
  };
}

"""


def run_applied_sim_cli_ts() -> str:
    return """// Execute applied mechanic.ts with env runTicks + SCARCITY/SURPLUS seeds (stdout JSON).
import * as fs from 'fs';
import { pathToFileURL } from 'node:url';

type SimConfig = {
  mechanicAbs: string;
  runTicksAbs: string;
  fixtureAbs: string;
  tickFunction: string;
  tickCount: number;
};

function serialize(row: unknown): Record<string, unknown> {
  return JSON.parse(JSON.stringify(row ?? {}));
}

async function main() {
  const configPath = process.argv[2];
  if (!configPath) {
    console.error('usage: runAppliedSimCli.ts <config.json>');
    process.exit(2);
  }
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8')) as SimConfig;
  const mechanic = await import(pathToFileURL(config.mechanicAbs).href);
  const ticksMod = await import(pathToFileURL(config.runTicksAbs).href);
  const fixture = await import(pathToFileURL(config.fixtureAbs).href);
  const tickFn = mechanic[config.tickFunction];
  if (typeof tickFn !== 'function') {
    throw new Error(`missing tick function ${config.tickFunction}`);
  }
  const runTicks = ticksMod.runTicks as (fn: (s: unknown) => unknown, seed: unknown, n: number) => unknown[];
  const scarcity = runTicks(tickFn, fixture.SCARCITY_SEED, config.tickCount);
  const surplus = runTicks(tickFn, fixture.SURPLUS_SEED, config.tickCount);
  const n = config.tickCount;
  const payload = {
    tickCount: n,
    scarcityStart: serialize(scarcity[0]),
    scarcityEnd: serialize(scarcity[n]),
    surplusStart: serialize(surplus[0]),
    surplusEnd: serialize(surplus[n]),
  };
  process.stdout.write(`${JSON.stringify(payload)}\\n`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
"""


def _type_name(subsection: str) -> str:
    return "".join(p.title() for p in subsection.split("_")) + "State"


def _scenario_nums(task: dict[str, Any]) -> dict[str, float]:
    sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
    scenario = (sim.get("scenario") or {}).get("initial_state") if isinstance(sim.get("scenario"), dict) else {}
    if not isinstance(scenario, dict):
        return {}
    return canonicalize_state(scenario)


def _num(scenario: dict[str, float], key: str, default: float) -> float:
    return float(scenario.get(key, default))


def game_turn_types_ts(subsection: str) -> str:
    type_name = _type_name(subsection)
    return f"""// economistRL game-extending sandbox: {subsection}
import type {{ City, Player, Tile, TerritoryInfo, Unit }} from '@/types/game';

/** One economy tick via ``processEconomyTurn`` (production gameLoop). */
export type {type_name} = {{
  cities: City[];
  units: Unit[];
  players: Player[];
  tiles: Record<string, Tile>;
  territory: Record<string, TerritoryInfo>;
  turn: number;
}};
"""


def game_turn_fixture_ts(subsection: str, task: dict[str, Any]) -> str:
    type_name = _type_name(subsection)
    scenario = _scenario_nums(task)
    storage_food = _num(scenario, "storageFood", 140.0)
    storage_cap = _num(scenario, "storageCapFood", 200.0)
    population = int(_num(scenario, "population", 100.0))
    player_gold = int(_num(scenario, "playerGold", 500.0))
    scar_morale = 0.35
    sur_morale = 0.85
    if subsection == "food_population_feedback":
        scar_food, sur_food = 40.0, 200.0
        scar_extra = sur_extra = ""
    elif subsection == "inventory_storage_spoilage":
        scar_food, sur_food = 30.0, 140.0
        storage_cap = _num(scenario, "storageCapFood", 100.0)
        scar_extra = sur_extra = ""
    elif subsection == "labor_wage_productivity":
        scar_food, sur_food = storage_food, storage_food
        scar_extra = f", morale: {scar_morale}"
        sur_extra = f", morale: {sur_morale}"
    else:
        scar_food, sur_food = storage_food * 0.4, storage_food * 1.4
        scar_extra = sur_extra = ""
    return f"""// economistRL fixtures: {subsection}
import type {{ {type_name} }} from './types';
import {{ minimalCity, minimalPlayer, SANDBOX_CITY_ID }} from '../shared/fixture';

function baseTurnState(city: ReturnType<typeof minimalCity>): {type_name} {{
  return {{
    cities: [city],
    units: [],
    players: [minimalPlayer({player_gold})],
    tiles: {{}},
    territory: {{}},
    turn: 0,
  }};
}}

export const DEFAULT_SEED: {type_name} = baseTurnState(
  minimalCity({{ population: {population}, storageFood: {storage_food}, storageCapFood: {storage_cap} }}),
);

export const SCARCITY_SEED: {type_name} = baseTurnState(
  minimalCity({{ population: {population}, storageFood: {scar_food}, storageCapFood: {storage_cap}{scar_extra} }}),
);

export const SURPLUS_SEED: {type_name} = baseTurnState(
  minimalCity({{ population: {population}, storageFood: {sur_food}, storageCapFood: {storage_cap}{sur_extra} }}),
);

export function cloneGameTurnState(state: {type_name}): {type_name} {{
  return {{
    ...state,
    cities: state.cities.map(c => ({{
      ...c,
      storage: {{ ...c.storage }},
      storageCap: {{ ...c.storageCap }},
      buildings: c.buildings.map(b => ({{ ...b }})),
    }})),
    units: state.units.map(u => ({{ ...u }})),
    players: state.players.map(p => ({{ ...p }})),
    tiles: {{ ...state.tiles }},
    territory: {{ ...state.territory }},
  }};
}}

export function primaryCity(state: {type_name}) {{
  const city = state.cities.find(c => c.id === SANDBOX_CITY_ID) ?? state.cities[0];
  if (!city) throw new Error('sandbox missing city');
  return city;
}}
"""


def game_turn_run_ticks_ts(subsection: str) -> str:
    type_name = _type_name(subsection)
    return f"""// economistRL runTicks: {subsection}
import type {{ {type_name} }} from './types';
import {{ cloneGameTurnState }} from './fixture';

export type TickFn = (state: {type_name}) => {type_name};

export function runTicks(fn: TickFn, initial: {type_name}, tickCount = 20): {type_name}[] {{
  const trace: {type_name}[] = [cloneGameTurnState(initial)];
  let state = cloneGameTurnState(initial);
  for (let i = 0; i < tickCount; i += 1) {{
    state = fn(cloneGameTurnState(state));
    trace.push(cloneGameTurnState(state));
  }}
  return trace;
}}
"""


def hybrid_types_ts(subsection: str) -> str:
    type_name = _type_name(subsection)
    ext_blocks = {
        "market_elasticity_pricing": """
  marketPriceGold: number;
  demand: number;
  scarcityPressure: number;
  surplusPressure: number;
""",
        "upkeep_progressive_costs": """
  garrisonUnitCount: number;
  upkeepPerUnit: number;
  supplyDistance: number;
  upkeepTotal: number;
  netGoldDelta: number;
""",
        "resource_projection_cache_integrity": """
  resourceProjectionValid: number;
  resourceProjectionCacheDirty: number;
  invalidationScopeCount: number;
  marketPriceDirty: number;
  buildingCompletionDirty: number;
""",
        "adversarial_multidomain_economy": """
  marketPriceGold: number;
  garrisonUnitCount: number;
  upkeepTotal: number;
""",
    }
    ext = ext_blocks.get(subsection, "\n  scalar: number;\n")
    return f"""// economistRL hybrid sandbox (City + task extensions): {subsection}
import type {{ City, Player }} from '@/types/game';

export type {type_name}TaskExt = {{{ext}
}};

export type {type_name} = {{
  city: City;
  players: Player[];
  taskExt: {type_name}TaskExt;
  turn: number;
}};
"""


def hybrid_fixture_ts(subsection: str, task: dict[str, Any]) -> str:
    type_name = _type_name(subsection)
    scenario = _scenario_nums(task)
    player_gold = int(_num(scenario, "playerGold", 500.0))
    population = int(_num(scenario, "population", 100.0))

    if subsection == "market_elasticity_pricing":
        default_ext = "marketPriceGold: 10, demand: 1, scarcityPressure: 0, surplusPressure: 0"
        scar_ext = "marketPriceGold: 10, demand: 1, scarcityPressure: 0, surplusPressure: 0"
        sur_ext = scar_ext
        scar_city = "minimalCity({ storageFood: 80, storageGoods: 40, storageCapFood: 200 })"
        sur_city = "minimalCity({ storageFood: 80, storageGoods: 160, storageCapFood: 200 })"
    elif subsection == "upkeep_progressive_costs":
        default_ext = "garrisonUnitCount: 12, upkeepPerUnit: 2, supplyDistance: 4, upkeepTotal: 0, netGoldDelta: 0"
        scar_ext = "garrisonUnitCount: 24, upkeepPerUnit: 3, supplyDistance: 8, upkeepTotal: 0, netGoldDelta: 0"
        sur_ext = "garrisonUnitCount: 8, upkeepPerUnit: 1.5, supplyDistance: 2, upkeepTotal: 0, netGoldDelta: 0"
        scar_city = f"minimalCity({{ population: {population} }})"
        sur_city = scar_city
    elif subsection == "resource_projection_cache_integrity":
        default_ext = (
            "resourceProjectionValid: 0, resourceProjectionCacheDirty: 1, "
            "invalidationScopeCount: 2, marketPriceDirty: 0, buildingCompletionDirty: 0"
        )
        scar_ext = (
            "resourceProjectionValid: 0, resourceProjectionCacheDirty: 1, "
            "invalidationScopeCount: 4, marketPriceDirty: 0, buildingCompletionDirty: 0"
        )
        sur_ext = (
            "resourceProjectionValid: 0, resourceProjectionCacheDirty: 1, "
            "invalidationScopeCount: 0, marketPriceDirty: 0, buildingCompletionDirty: 0"
        )
        scar_city = sur_city = f"minimalCity({{ population: {population} }})"
    else:
        default_ext = "marketPriceGold: 10, garrisonUnitCount: 12, upkeepTotal: 0"
        scar_ext = sur_ext = default_ext
        scar_city = f"minimalCity({{ population: {population}, storageFood: 40 }})"
        sur_city = f"minimalCity({{ population: {population}, storageFood: 160 }})"

    return f"""// economistRL hybrid fixtures: {subsection}
import type {{ {type_name}, {type_name}TaskExt }} from './types';
import {{ cloneCity, minimalCity, minimalPlayer }} from '../shared/fixture';

function baseHybrid(city: ReturnType<typeof minimalCity>, taskExt: {type_name}TaskExt): {type_name} {{
  return {{ city, players: [minimalPlayer({player_gold})], taskExt, turn: 0 }};
}}

export const DEFAULT_SEED: {type_name} = baseHybrid(
  {sur_city if subsection == 'adversarial_multidomain_economy' else scar_city},
  {{ {default_ext} }},
);

export const SCARCITY_SEED: {type_name} = baseHybrid({scar_city}, {{ {scar_ext} }});
export const SURPLUS_SEED: {type_name} = baseHybrid({sur_city}, {{ {sur_ext} }});

export function cloneHybridState(state: {type_name}): {type_name} {{
  return {{
    turn: state.turn,
    players: state.players.map(p => ({{ ...p }})),
    city: cloneCity(state.city),
    taskExt: {{ ...state.taskExt }},
  }};
}}
"""


def hybrid_run_ticks_ts(subsection: str) -> str:
    type_name = _type_name(subsection)
    return f"""// economistRL runTicks: {subsection}
import type {{ {type_name} }} from './types';
import {{ cloneHybridState }} from './fixture';

export type TickFn = (state: {type_name}) => {type_name};

export function runTicks(fn: TickFn, initial: {type_name}, tickCount = 20): {type_name}[] {{
  const trace: {type_name}[] = [cloneHybridState(initial)];
  let state = cloneHybridState(initial);
  for (let i = 0; i < tickCount; i += 1) {{
    state = fn(cloneHybridState(state));
    trace.push(cloneHybridState(state));
  }}
  return trace;
}}
"""


def game_turn_mechanic_tick_impl(subsection: str) -> tuple[str, str]:
    type_name = _type_name(subsection)
    tick_impl = f"""
  const {{ tiles, territory }} = recordsToMaps(state.tiles, state.territory);
  const result = processEconomyTurn(
    state.cities,
    state.units,
    state.players,
    tiles,
    territory,
    state.turn + 1,
    1.0,
  );
  return {{
    ...state,
    cities: result.cities,
    units: result.units,
    players: result.players,
    turn: state.turn + 1,
  }};"""
    extra_import = (
        f"import {{ processEconomyTurn }} from '@/lib/gameLoop';\n"
        f"import {{ recordsToMaps }} from '../../shared/fixture';\n"
        f"import type {{ {type_name} }} from '{_relative_env_import(subsection, 'types')}';"
    )
    return tick_impl, extra_import


def _relative_env_import(subsection: str, file_stem: str) -> str:
    return f"../envs/{subsection}/{file_stem}"


def hybrid_mechanic_tick_impl(subsection: str) -> tuple[str, str]:
    type_name = _type_name(subsection)
    rel = _relative_env_import(subsection, "types")
    if subsection == "market_elasticity_pricing":
        tick_impl = """
  const pressure = state.city.storageCap.goods / Math.max(1, state.city.storage.goods);
  const multiplier = clamp(1 + 0.18 * (pressure - 1), MULT_FLOOR, MULT_CAP);
  const target = clamp(
    state.taskExt.marketPriceGold * multiplier * Math.max(0.01, state.taskExt.demand),
    PRICE_FLOOR,
    PRICE_CAP,
  );
  const marketPriceGold = clamp(
    state.taskExt.marketPriceGold + PRICE_SMOOTH * (target - state.taskExt.marketPriceGold),
    PRICE_FLOOR,
    PRICE_CAP,
  );
  return {
    ...state,
    turn: state.turn + 1,
    taskExt: {
      ...state.taskExt,
      marketPriceGold,
      scarcityPressure: Math.max(0, pressure - 1),
      surplusPressure: Math.max(0, 1 - pressure),
    },
  };"""
        extra_import = (
            f"import {{ clamp, MULT_CAP, MULT_FLOOR, PRICE_CAP, PRICE_FLOOR, PRICE_SMOOTH }} from '{rel}';\n"
            f"import type {{ {type_name} }} from '{rel}';"
        )
    elif subsection == "upkeep_progressive_costs":
        tick_impl = """
  const distanceFactor = 1 + 0.12 * state.taskExt.supplyDistance;
  const upkeepTotal = state.taskExt.garrisonUnitCount * state.taskExt.upkeepPerUnit * distanceFactor;
  const netGoldDelta = -upkeepTotal;
  const players = state.players.map(p => ({
    ...p,
    gold: Math.max(0, p.gold + netGoldDelta),
  }));
  return {
    ...state,
    players,
    turn: state.turn + 1,
    taskExt: { ...state.taskExt, upkeepTotal, netGoldDelta },
  };"""
        extra_import = f"import type {{ {type_name} }} from '{rel}';"
    elif subsection == "resource_projection_cache_integrity":
        tick_impl = """
  let invalidationScopeCount = state.taskExt.invalidationScopeCount;
  let resourceProjectionCacheDirty = state.taskExt.resourceProjectionCacheDirty;
  if (invalidationScopeCount > 0) {
    invalidationScopeCount = Math.max(0, invalidationScopeCount - 1);
  } else if (resourceProjectionCacheDirty > 0) {
    resourceProjectionCacheDirty = 0;
  }
  const stillDirty = resourceProjectionCacheDirty > 0 || invalidationScopeCount > 0;
  const resourceProjectionValid = stillDirty ? 0 : 1;
  return {
    ...state,
    turn: state.turn + 1,
    taskExt: {
      ...state.taskExt,
      invalidationScopeCount,
      resourceProjectionCacheDirty: stillDirty ? 1 : 0,
      resourceProjectionValid,
    },
  };"""
        extra_import = f"import type {{ {type_name} }} from '{rel}';"
    else:
        tick_impl = """
  const foodConsumed = state.city.population * 0.15;
  const city = {
    ...state.city,
    storage: {
      ...state.city.storage,
      food: Math.max(0, state.city.storage.food + 12 - foodConsumed),
    },
  };
  const pressure = city.storageCap.goods / Math.max(1, city.storage.goods);
  const marketPriceGold = Math.min(25, Math.max(5, state.taskExt.marketPriceGold * (1 + 0.05 * (pressure - 1))));
  const upkeepTotal = state.taskExt.garrisonUnitCount * 1.5;
  const players = state.players.map(p => ({ ...p, gold: Math.max(0, p.gold - upkeepTotal) }));
  return {
    ...state,
    city,
    players,
    turn: state.turn + 1,
    taskExt: { ...state.taskExt, marketPriceGold, upkeepTotal },
  };"""
        extra_import = f"import type {{ {type_name} }} from '{rel}';"
    return tick_impl, extra_import


def game_turn_test_body(subsection: str, fn: str) -> str:
    if subsection == "food_population_feedback":
        return f"""
    const safe = runTicks({fn}, SURPLUS_SEED, 20);
    const unsafe = runTicks({fn}, SCARCITY_SEED, 20);
    expect(unsafe[20].cities[0].population).not.toBe(unsafe[0].cities[0].population);
    expect(safe[20].cities[0].population).not.toBe(safe[0].cities[0].population);
    expect(safe[20].cities[0].population).toBeGreaterThanOrEqual(unsafe[20].cities[0].population);
    expect(safe[20].cities[0].storage.food).toBeGreaterThanOrEqual(0);
    expect(unsafe[20].cities[0].storage.food).toBeGreaterThanOrEqual(0);"""
    if subsection == "inventory_storage_spoilage":
        return f"""
    const cramped = runTicks({fn}, SCARCITY_SEED, 5);
    const roomy = runTicks({fn}, SURPLUS_SEED, 5);
    expect(cramped[5].cities[0].storage.food).not.toBe(cramped[0].cities[0].storage.food);
    expect(roomy[5].cities[0].storage.food).not.toBe(roomy[0].cities[0].storage.food);
    expect(roomy[5].cities[0].storage.food).toBeGreaterThanOrEqual(cramped[5].cities[0].storage.food);"""
    if subsection == "labor_wage_productivity":
        return f"""
    const low = runTicks({fn}, SCARCITY_SEED, 10);
    const high = runTicks({fn}, SURPLUS_SEED, 10);
    expect(low[10].cities[0].morale).not.toBe(low[0].cities[0].morale);
    expect(high[10].cities[0].morale).not.toBe(high[0].cities[0].morale);
    expect(high[10].cities[0].morale).toBeGreaterThanOrEqual(low[10].cities[0].morale);"""
    return f"""
    const trace = runTicks({fn}, DEFAULT_SEED, 10);
    expect(trace[10].cities[0].morale).toBeDefined();
    expect(trace[10].turn).toBe(10);"""


def hybrid_test_body(subsection: str, fn: str) -> str:
    if subsection == "market_elasticity_pricing":
        return f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, 20);
    const surplus = runTicks({fn}, SURPLUS_SEED, 20);
    expect(scarcity[20].taskExt.marketPriceGold).not.toBe(scarcity[0].taskExt.marketPriceGold);
    expect(surplus[20].taskExt.marketPriceGold).not.toBe(surplus[0].taskExt.marketPriceGold);
    expect(scarcity[20].taskExt.marketPriceGold).toBeGreaterThan(surplus[20].taskExt.marketPriceGold);
    expect(scarcity[20].city.storage.goods).toBeGreaterThanOrEqual(0);"""
    if subsection == "upkeep_progressive_costs":
        return f"""
    const strained = runTicks({fn}, SCARCITY_SEED, 5);
    const funded = runTicks({fn}, SURPLUS_SEED, 5);
    expect(strained[5].players[0].gold).not.toBe(strained[0].players[0].gold);
    expect(funded[5].players[0].gold).not.toBe(funded[0].players[0].gold);
    expect(strained[5].players[0].gold).toBeLessThanOrEqual(funded[5].players[0].gold);"""
    if subsection == "resource_projection_cache_integrity":
        return f"""
    const dirty = runTicks({fn}, SCARCITY_SEED, 8);
    const clean = runTicks({fn}, SURPLUS_SEED, 2);
    expect(dirty[8].taskExt.resourceProjectionValid).not.toBe(dirty[0].taskExt.resourceProjectionValid);
    expect(clean[2].taskExt.resourceProjectionValid).not.toBe(clean[0].taskExt.resourceProjectionValid);
    expect(dirty[4].taskExt.resourceProjectionValid).toBe(0);
    expect(clean[2].taskExt.resourceProjectionValid).toBeGreaterThan(dirty[4].taskExt.resourceProjectionValid);"""
    return f"""
    const trace = runTicks({fn}, DEFAULT_SEED, 20);
    const moved =
      trace[20].city.storage.food !== trace[0].city.storage.food ||
      trace[20].taskExt.marketPriceGold !== trace[0].taskExt.marketPriceGold ||
      trace[20].players[0].gold !== trace[0].players[0].gold;
    expect(moved).toBe(true);"""


def env_files_for_subsection(subsection: str, task: dict[str, Any]) -> list[tuple[str, str]]:
    base = f"src/lib/economistRl/envs/{subsection}"
    mode = sandbox_integration_mode(subsection)
    shared = [
        (SHARED_FIXTURE, shared_fixture_ts()),
        (f"{SHARED_ENV_DIR}/runAppliedSimCli.ts", run_applied_sim_cli_ts()),
    ]
    if mode == "game_turn":
        return [
            *shared,
            (f"{base}/types.ts", game_turn_types_ts(subsection)),
            (f"{base}/fixture.ts", game_turn_fixture_ts(subsection, task)),
            (f"{base}/runTicks.ts", game_turn_run_ticks_ts(subsection)),
            (f"{base}/index.ts", "export * from './types';\nexport * from './fixture';\nexport * from './runTicks';\n"),
        ]
    if mode == "hybrid_city":
        extra = ""
        if subsection == "market_elasticity_pricing":
            extra = """
export const PRICE_FLOOR = 5;
export const PRICE_CAP = 25;
export const MULT_FLOOR = 0.75;
export const MULT_CAP = 1.35;
export const MAX_PRICE_STEP = 1.25;
export const PRICE_SMOOTH = 0.22;

export function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}
"""
        types_content = hybrid_types_ts(subsection) + extra
        return [
            *shared,
            (f"{base}/types.ts", types_content),
            (f"{base}/fixture.ts", hybrid_fixture_ts(subsection, task)),
            (f"{base}/runTicks.ts", hybrid_run_ticks_ts(subsection)),
            (f"{base}/index.ts", "export * from './types';\nexport * from './fixture';\nexport * from './runTicks';\n"),
        ]
    return []
