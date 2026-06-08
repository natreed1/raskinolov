#!/usr/bin/env python3
"""Map task-bank ``simulation_spec.goals`` to executable Vitest ``it()`` blocks.

Each goal becomes one weighted test case so Vitest partial credit aligns with
task-bank goal weights (not only generic subsection templates).
"""

from __future__ import annotations

import re
from typing import Any

from economist_rl_game_sandbox import sandbox_integration_mode
from economist_rl_sandbox_envs import env_file_paths, resolve_subsection, tick_function_name


def goal_it_title(goal_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(goal_name).strip()).strip("_").lower()
    return f"goal_{token[:56] or 'behavior'}"


def _slug(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return token or "goal"


def simulation_goals(task: dict[str, Any]) -> list[dict[str, Any]]:
    sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
    goals: list[dict[str, Any]] = []
    for raw in sim.get("goals") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        goals.append(
            {
                "name": name,
                "weight": float(raw.get("weight") or 1.0),
                "target": str(raw.get("target") or ""),
                "benchmark": str(raw.get("benchmark") or ""),
            }
        )
    return goals


def _tick_count(task: dict[str, Any], *, default: int = 20) -> int:
    sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
    return int(sim.get("tick_count") or default)


def _goal_it_block(goal_name: str, task: dict[str, Any], fn: str) -> str | None:
    """Return Vitest body for a single simulation_spec goal (inside one ``it``)."""
    subsection = resolve_subsection(task)
    n = _tick_count(task)
    name = goal_name.lower()
    mode = subsection

    if mode == "food_population_feedback":
        if "birth" in name and ("taper" in name or "floor" in name or "buffer" in name):
            return f"""
    const safe = runTicks({fn}, SURPLUS_SEED, {n});
    const unsafe = runTicks({fn}, SCARCITY_SEED, {n});
    expect(unsafe[{n}].cities[0].population).not.toBe(unsafe[0].cities[0].population);
    expect(safe[{n}].cities[0].population).toBeGreaterThanOrEqual(unsafe[{n}].cities[0].population);
    expect(safe[{n}].cities[0].storage.food).toBeGreaterThanOrEqual(0);"""
        if "food" in name and ("negative" in name or "silent" in name):
            return f"""
    const unsafe = runTicks({fn}, SCARCITY_SEED, {n});
    expect(unsafe[{n}].cities[0].storage.food).toBeGreaterThanOrEqual(0);"""
        if "oscill" in name or "spike" in name or "smooth" in name:
            return f"""
    const trace = runTicks({fn}, SURPLUS_SEED, {n});
    const deltas: number[] = [];
    for (let i = 1; i <= {n}; i += 1) {{
      deltas.push(Math.abs(trace[i].cities[0].population - trace[i - 1].cities[0].population));
    }}
    const maxDelta = Math.max(...deltas);
    const sorted = [...deltas].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
    expect(maxDelta).toBeLessThanOrEqual(Math.max(1, 2 * median));"""

    if mode == "market_elasticity_pricing":
        if "scarcity" in name and "price" in name:
            return f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, {n});
    expect(scarcity[{n}].taskExt.marketPriceGold).not.toBe(scarcity[0].taskExt.marketPriceGold);"""
        if "surplus" in name and "price" in name:
            return f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, {n});
    const surplus = runTicks({fn}, SURPLUS_SEED, {n});
    expect(surplus[{n}].taskExt.marketPriceGold).not.toBe(surplus[0].taskExt.marketPriceGold);
    expect(scarcity[{n}].taskExt.marketPriceGold).toBeGreaterThan(surplus[{n}].taskExt.marketPriceGold);"""
        if "bound" in name or "floor" in name or "cap" in name:
            return f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, {n});
    expect(scarcity[{n}].taskExt.marketPriceGold).toBeGreaterThanOrEqual(PRICE_FLOOR);
    expect(scarcity[{n}].taskExt.marketPriceGold).toBeLessThanOrEqual(PRICE_CAP);"""
        if "smooth" in name or "oscill" in name or "jump" in name:
            return f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, {n});
    const steps: number[] = [];
    for (let i = 1; i <= {n}; i += 1) {{
      steps.push(Math.abs(scarcity[i].taskExt.marketPriceGold - scarcity[i - 1].taskExt.marketPriceGold));
    }}
    expect(Math.max(...steps)).toBeLessThanOrEqual(1.25);"""

    if mode == "labor_wage_productivity":
        if "morale" in name or "wage" in name:
            return f"""
    const low = runTicks({fn}, SCARCITY_SEED, 10);
    const high = runTicks({fn}, SURPLUS_SEED, 10);
    expect(high[10].cities[0].morale).toBeGreaterThanOrEqual(low[10].cities[0].morale);"""
        if "productivity" in name:
            return f"""
    const low = runTicks({fn}, SCARCITY_SEED, 10);
    const high = runTicks({fn}, SURPLUS_SEED, 10);
    expect(high[10].cities[0].morale).not.toBe(low[10].cities[0].morale);"""

    if mode == "inventory_storage_spoilage":
        if "decay" in name or "spoil" in name:
            return f"""
    const cramped = runTicks({fn}, SCARCITY_SEED, 5);
    expect(cramped[5].cities[0].storage.food).not.toBe(cramped[0].cities[0].storage.food);"""
        if "capacity" in name or "preserv" in name:
            return f"""
    const cramped = runTicks({fn}, SCARCITY_SEED, 5);
    const roomy = runTicks({fn}, SURPLUS_SEED, 5);
    expect(roomy[5].cities[0].storage.food).toBeGreaterThanOrEqual(cramped[5].cities[0].storage.food);"""

    if mode == "upkeep_progressive_costs":
        if "gold" in name or "upkeep" in name or "afford" in name:
            return f"""
    const strained = runTicks({fn}, SCARCITY_SEED, 5);
    const funded = runTicks({fn}, SURPLUS_SEED, 5);
    expect(strained[5].players[0].gold).toBeLessThanOrEqual(funded[5].players[0].gold);"""
        if "distance" in name or "supply" in name:
            return f"""
    const strained = runTicks({fn}, SCARCITY_SEED, 5);
    expect(strained[5].taskExt.upkeepTotal).toBeGreaterThan(0);"""

    if mode == "resource_projection_cache_integrity":
        if "stale" in name or "invalid" in name or "dirty" in name:
            return f"""
    const dirty = runTicks({fn}, SCARCITY_SEED, 8);
    expect(dirty[8].taskExt.resourceProjectionValid).not.toBe(dirty[0].taskExt.resourceProjectionValid);"""
        if "recompute" in name or "projection" in name or "cache" in name:
            return f"""
    const dirty = runTicks({fn}, SCARCITY_SEED, 8);
    const clean = runTicks({fn}, SURPLUS_SEED, 2);
    expect(clean[2].taskExt.resourceProjectionValid).toBeGreaterThan(dirty[4].taskExt.resourceProjectionValid);"""

    if mode == "adversarial_multidomain_economy":
        return f"""
    const trace = runTicks({fn}, DEFAULT_SEED, {n});
    expect(trace[{n}].city.storage.food).toBeGreaterThanOrEqual(0);
    expect(trace[{n}].taskExt.marketPriceGold).toBeGreaterThanOrEqual(5);"""

    return None


def _fallback_goal_block(goal_name: str, task: dict[str, Any], fn: str) -> str:
    """Executable assert for goals without a dedicated template (incl. generalist rubric goals)."""
    subsection = resolve_subsection(task)
    mode = sandbox_integration_mode(subsection)
    n = min(_tick_count(task), 10)
    name = goal_name.lower()

    if subsection == "labor_wage_productivity" and "recovery" in name:
        return f"""
    const trace = runTicks({fn}, SCARCITY_SEED, {n});
    const deltas: number[] = [];
    for (let i = 1; i <= {n}; i += 1) {{
      deltas.push(Math.abs(trace[i].cities[0].morale - trace[i - 1].cities[0].morale));
    }}
    expect(Math.max(...deltas)).toBeLessThanOrEqual(25);"""

    if subsection == "upkeep_progressive_costs" and ("large" in name or "army" in name or "force" in name):
        return f"""
    const strained = runTicks({fn}, SCARCITY_SEED, {n});
    expect(strained[{n}].taskExt.upkeepTotal).toBeGreaterThan(0);
    expect(strained[{n}].players[0].gold).toBeLessThanOrEqual(strained[0].players[0].gold);"""

    if mode == "game_turn":
        return f"""
    const safe = runTicks({fn}, SURPLUS_SEED, {n});
    const unsafe = runTicks({fn}, SCARCITY_SEED, {n});
    expect(safe[{n}].cities[0].storage.food).toBeGreaterThanOrEqual(0);
    expect(unsafe[{n}].cities[0].storage.food).toBeGreaterThanOrEqual(0);"""

    if mode == "hybrid_city":
        seed = "SCARCITY_SEED"
        if subsection == "market_elasticity_pricing":
            return f"""
    const trace = runTicks({fn}, {seed}, {n});
    expect(trace[{n}].taskExt.marketPriceGold).toBeGreaterThanOrEqual(0);
    expect(trace[{n}].city.storage.goods).toBeGreaterThanOrEqual(0);"""
        return f"""
    const trace = runTicks({fn}, {seed}, {n});
    expect(trace[{n}]).toBeDefined();"""

    return f"""
    const trace = runTicks({fn}, DEFAULT_SEED, {n});
    expect(trace[{n}]).toBeDefined();
    expect(trace[{n}].boundedSignal).toBeGreaterThanOrEqual(0);"""


def _subsection_baseline_assertions(task: dict[str, Any], fn: str) -> list[tuple[str, str, float]]:
    """Fallback goals when simulation_spec.goals missing — subsection templates."""
    subsection = resolve_subsection(task)
    n = _tick_count(task)
    if subsection == "market_elasticity_pricing":
        return [
            (
                "scarcity_and_surplus_diverge",
                f"""
    const scarcity = runTicks({fn}, SCARCITY_SEED, {n});
    const surplus = runTicks({fn}, SURPLUS_SEED, {n});
    expect(scarcity[{n}].taskExt.marketPriceGold).not.toBe(scarcity[0].taskExt.marketPriceGold);
    expect(scarcity[{n}].taskExt.marketPriceGold).toBeGreaterThan(surplus[{n}].taskExt.marketPriceGold);""",
                1.0,
            ),
        ]
    if subsection == "food_population_feedback":
        return [
            (
                "food_buffer_affects_population",
                f"""
    const safe = runTicks({fn}, SURPLUS_SEED, {n});
    const unsafe = runTicks({fn}, SCARCITY_SEED, {n});
    expect(safe[{n}].cities[0].population).toBeGreaterThanOrEqual(unsafe[{n}].cities[0].population);""",
                1.0,
            ),
        ]
    return [("bounded_behavior", f"    const trace = runTicks({fn}, DEFAULT_SEED, 5);\n    expect(trace[5]).toBeDefined();", 1.0)]


def goal_vitest_cases(task: dict[str, Any], fn: str) -> list[dict[str, Any]]:
    """One Vitest ``it`` per task-bank goal with optional weight metadata."""
    goals = simulation_goals(task)
    cases: list[dict[str, Any]] = []
    for goal in goals:
        gname = str(goal["name"])
        body = _goal_it_block(gname, task, fn) or _fallback_goal_block(gname, task, fn)
        cases.append({"name": gname, "weight": float(goal["weight"]), "body": body})
    if cases:
        return cases
    for name, body, weight in _subsection_baseline_assertions(task, fn):
        cases.append({"name": name, "weight": weight, "body": body})
    return cases


def build_goal_based_vitest_body(task: dict[str, Any], *, lib_file: str, test_file: str) -> tuple[str, str]:
    """Return (extra_imports, test_body) for full vitest file content."""
    from economist_rl_game_sandbox import sandbox_integration_mode
    from economist_rl_sandbox_envs import _relative_import, env_file_paths

    task_id = str(task.get("id") or "task")
    fn = tick_function_name(task)
    subsection = resolve_subsection(task)
    paths = env_file_paths(subsection)
    mode = sandbox_integration_mode(subsection)
    if mode != "lab_scalar":
        seed_import_path = paths[3]
        ticks_import_path = paths[4]
        types_import_path = paths[2]
    else:
        seed_import_path = paths[0]
        ticks_import_path = paths[1]
        types_import_path = paths[0]

    env_seeds = _relative_import(test_file, seed_import_path)
    env_ticks = _relative_import(test_file, ticks_import_path)
    extra = f"import {{ SCARCITY_SEED, SURPLUS_SEED }} from '{env_seeds}';\n"
    if subsection == "market_elasticity_pricing":
        types_path = _relative_import(test_file, types_import_path)
        extra = (
            f"import {{ PRICE_CAP, PRICE_FLOOR }} from '{types_path}';\n"
            f"import {{ SCARCITY_SEED, SURPLUS_SEED }} from '{env_seeds}';\n"
        )
    elif mode == "lab_scalar":
        extra = f"import {{ DEFAULT_SEED }} from '{env_seeds}';\n"

    goal_lines = []
    for case in goal_vitest_cases(task, fn):
        title = goal_it_title(str(case["name"]))
        goal_lines.append(f"  it('{title}', () => {{{case['body']}\n  }});")

    body = "\n".join(goal_lines)
    header = f"// Goals from simulation_spec: {len(simulation_goals(task))} loaded into vitest\n"
    return extra, header + body


def render_vitest_stub(task: dict[str, Any], *, lib_file: str, test_file: str) -> str:
    """Full Vitest file: one ``it()`` per ``simulation_spec.goals`` entry."""
    from economist_rl_sandbox_envs import _relative_import, tick_function_name

    task_id = str(task.get("id") or "task")
    fn = tick_function_name(task)
    mechanic_import = _relative_import(test_file, lib_file)
    subsection = resolve_subsection(task)
    paths = env_file_paths(subsection)
    mode = sandbox_integration_mode(subsection)
    if mode != "lab_scalar":
        ticks_import_path = paths[4]
    else:
        ticks_import_path = paths[1]
    env_ticks = _relative_import(test_file, ticks_import_path)
    extra, body = build_goal_based_vitest_body(task, lib_file=lib_file, test_file=test_file)
    return f"""// economistRL targeted test: {task_id}
import {{ describe, it, expect }} from 'vitest';
import {{ runTicks }} from '{env_ticks}';
{extra}import {{ {fn} }} from '{mechanic_import}';

describe('economistRL {task_id}', () => {{
{body}
}});
"""


def sync_targeted_tests_from_goals(task: dict[str, Any]) -> dict[str, Any]:
    """Ensure targeted_tests.outcome_checks lists simulation goal names for reporting."""
    targeted = dict(task.get("targeted_tests") or {}) if isinstance(task.get("targeted_tests"), dict) else {}
    goals = simulation_goals(task)
    if goals:
        targeted["outcome_checks"] = [g["name"] for g in goals]
        targeted["goal"] = (
            "Vitest encodes simulation_spec.goals (one it() per goal; partial credit by goal weight)."
        )
    return targeted
