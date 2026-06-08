#!/usr/bin/env python3
"""Reference 20-tick simulators for economistRL task subsections.

These are **lab-side toy environments**, not the Fallen Empire game engine.
Tasks may describe mechanics that do not exist in production code; each env
scores whether a rollout describes bounded, testable behavior that would pass
the task's simulation_spec goals in a focused 20-tick trace.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from typing import Any, Callable

SimFn = Callable[[dict[str, Any], str], dict[str, Any]]

PLACEHOLDER_TOKENS = frozenset(
    {
        "task_defined_seed_value",
        "task_seed",
        "scenario_seed",
        "placeholder",
        "tbd",
    }
)

FIELD_DEFAULTS: dict[str, float] = {
    "population": 100.0,
    "storageFood": 140.0,
    "storageCapFood": 160.0,
    "foodProducedPerTick": 16.0,
    "popFoodConsumedPerCapita": 0.18,
    "appliedBirthRate": 0.035,
    "birthPressure": 0.02,
    "popReserveFoodFloor": 50.0,
    "popUnsafeFoodBufferSpan": 100.0,
    "storageGoods": 50.0,
    "storageCapGoods": 100.0,
    "marketPriceGold": 10.0,
    "demand": 1.0,
    "surplusStorageGoods": 150.0,
    "wage": 0.8,
    "expectedWage": 1.0,
    "cityMorale": 0.75,
    "productivity": 1.0,
    "totalEmployed": 100.0,
    "playerGold": 500.0,
    "warehouseLevel": 1.0,
    "preservationModifier": 0.85,
    "foodDecayPerTick": 0.06,
    "garrisonUnitCount": 12.0,
    "upkeepPerUnit": 2.0,
    "supplyDistance": 4.0,
    "upkeepTotal": 100.0,
    "netGoldDelta": 12.0,
    "resourceProjectionValid": 1.0,
    "resourceProjectionCacheDirty": 0.0,
    "marketPriceDirty": 10.0,
    "buildingCompletionDirty": 1.0,
    "upkeepModifierDirty": 1.0,
    "workerReassignmentDirty": 6.0,
    "invalidationScopeCount": 0.0,
    "pressure": 1.0,
    "scalar": 1.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _task_seed(task: dict[str, Any]) -> int:
    token = str(task.get("id") or task.get("subskill") or "economistRL")
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def _jitter(base: float, task: dict[str, Any], field: str, span: float = 0.08) -> float:
    mix = (_task_seed(task) ^ hash(field)) % 1000
    factor = 1.0 + ((mix / 1000.0) - 0.5) * span
    return float(base) * factor


def _coerce_numeric(raw: Any, *, field: str, task: dict[str, Any]) -> float:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        nums = [_coerce_numeric(v, field=f"{field}.{k}", task=task) for k, v in raw.items()]
        return float(sum(nums) / max(1, len(nums)))
    if isinstance(raw, list):
        nums = [_coerce_numeric(v, field=field, task=task) for v in raw]
        return float(sum(nums) / max(1, len(nums)))
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped.lower() in PLACEHOLDER_TOKENS:
            return _jitter(FIELD_DEFAULTS.get(field, 1.0), task, field)
        try:
            return float(stripped)
        except ValueError:
            return _jitter(FIELD_DEFAULTS.get(field, 1.0), task, field)
    return _jitter(FIELD_DEFAULTS.get(field, 1.0), task, field)


def resolve_scenario_state(task: dict[str, Any]) -> dict[str, float]:
    """Resolve simulation_spec.scenario.initial_state into numeric env seeds."""
    from economist_rl_field_names import canonicalize_state

    scenario = (task.get("simulation_spec") or {}).get("scenario", {}).get("initial_state") or {}
    if not isinstance(scenario, dict):
        return normalize_scenario_state(task, {"scalar": _jitter(1.0, task, "scalar")})
    resolved: dict[str, float] = {}
    for key, raw in canonicalize_state(scenario).items():
        resolved[key] = _coerce_numeric(raw, field=key, task=task)
    if "surplusStorageGoods" not in resolved and "storageCapGoods" in resolved:
        resolved["surplusStorageGoods"] = resolved["storageCapGoods"] * 1.5
    if "popReserveFoodFloor" not in resolved:
        resolved["popReserveFoodFloor"] = FIELD_DEFAULTS["popReserveFoodFloor"]
    if "popUnsafeFoodBufferSpan" not in resolved:
        resolved["popUnsafeFoodBufferSpan"] = FIELD_DEFAULTS["popUnsafeFoodBufferSpan"]
    return normalize_scenario_state(task, resolved)


def _normalize_rate(field: str, value: float) -> float:
    lower = field.lower()
    if value <= 1.0:
        return value
    if "rate" in lower or "decay" in lower or "spoil" in lower:
        return min(0.5, value / 100.0)
    return value


def normalize_scenario_state(task: dict[str, Any], state: dict[str, float]) -> dict[str, float]:
    """Apply subsection-specific seed shaping so toy envs exercise the intended pressure."""
    subsection = str(task.get("subsection") or "")
    out = {key: _normalize_rate(key, val) for key, val in state.items()}

    if subsection == "inventory_storage_spoilage":
        capacity = out.get("storageCapFood", FIELD_DEFAULTS["storageCapFood"])
        warehouse = out.get("warehouseLevel", FIELD_DEFAULTS["warehouseLevel"])
        effective_capacity = capacity * max(1.0, warehouse)
        stock = out.get("storageFood", FIELD_DEFAULTS["storageFood"])
        out["storageCapFood"] = effective_capacity
        out["storageFood"] = max(stock, effective_capacity * 1.45)
        out.setdefault("foodDecayPerTick", FIELD_DEFAULTS["foodDecayPerTick"])
        out.setdefault("preservationModifier", FIELD_DEFAULTS["preservationModifier"])

    elif subsection == "market_elasticity_pricing":
        out.setdefault("storageGoods", FIELD_DEFAULTS["storageGoods"])
        out.setdefault("storageCapGoods", FIELD_DEFAULTS["storageCapGoods"])
        if out.get("storageGoods", 0.0) >= out.get("storageCapGoods", 100.0):
            out["storageGoods"] = out["storageCapGoods"] * 0.5

    elif subsection == "food_population_feedback":
        out.setdefault("storageFood", FIELD_DEFAULTS["storageFood"])
        out.setdefault("population", FIELD_DEFAULTS["population"])
        out.setdefault("foodProducedPerTick", FIELD_DEFAULTS["foodProducedPerTick"])
        out.setdefault("popFoodConsumedPerCapita", FIELD_DEFAULTS["popFoodConsumedPerCapita"])
        out.setdefault("appliedBirthRate", FIELD_DEFAULTS["appliedBirthRate"])

    elif subsection == "labor_wage_productivity":
        out.setdefault("wage", FIELD_DEFAULTS["wage"])
        out.setdefault("expectedWage", FIELD_DEFAULTS["expectedWage"])
        out.setdefault("cityMorale", FIELD_DEFAULTS["cityMorale"])
        out.setdefault("productivity", FIELD_DEFAULTS["productivity"])

    elif subsection == "upkeep_progressive_costs":
        out.setdefault("garrisonUnitCount", FIELD_DEFAULTS["garrisonUnitCount"])
        out.setdefault("upkeepPerUnit", FIELD_DEFAULTS["upkeepPerUnit"])
        out.setdefault("supplyDistance", FIELD_DEFAULTS["supplyDistance"])
        garrison = out.get("garrisonSize")
        if garrison is not None:
            out["garrisonUnitCount"] = min(out["garrisonUnitCount"], max(8.0, garrison))
        if out.get("garrisonUnitCount", FIELD_DEFAULTS["garrisonUnitCount"]) > 40.0:
            out["garrisonUnitCount"] = max(8.0, out["garrisonUnitCount"] * 0.2)

    elif subsection == "resource_projection_cache_integrity":
        out.setdefault("netGoldDelta", FIELD_DEFAULTS["netGoldDelta"])
        out.setdefault("resourceProjectionCacheDirty", 0.0)
        out.setdefault("resourceProjectionValid", FIELD_DEFAULTS["resourceProjectionValid"])

    elif subsection == "adversarial_multidomain_economy":
        out.setdefault("storageFood", FIELD_DEFAULTS["storageFood"])
        out.setdefault("population", FIELD_DEFAULTS["population"])
        out.setdefault("marketPriceDirty", FIELD_DEFAULTS["marketPriceDirty"])
        army_size = out.get("garrisonUnitCount", FIELD_DEFAULTS["garrisonUnitCount"])
        out.setdefault("upkeepTotal", FIELD_DEFAULTS["upkeepPerUnit"] * army_size)
        out.setdefault("cityMorale", FIELD_DEFAULTS["cityMorale"])
        out.setdefault("resourceProjectionValid", FIELD_DEFAULTS["resourceProjectionValid"])
        out.setdefault("storageCapFood", FIELD_DEFAULTS["storageCapFood"])
        out["storageFood"] = max(out.get("storageFood", 140.0), out["storageCapFood"] * 1.35)

    return out


def _goal_payload(task: dict[str, Any], goal_scores: dict[str, float], observed: dict[str, Any]) -> dict[str, Any]:
    goals = []
    for goal in task.get("simulation_spec", {}).get("goals") or []:
        if not isinstance(goal, dict):
            continue
        name = str(goal.get("name") or "")
        goals.append(
            {
                "name": name,
                "score": float(goal_scores.get(name, 0.0)),
                "weight": float(goal.get("weight") or 1.0),
                "observed": observed,
            }
        )
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    return {"ticks": tick_count, "goals": goals, "trace_summary": observed, "env": "economist_rl_toy_v2"}


def _reject_unbounded_language(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "unbounded",
            "no clamp",
            "without clamp",
            "random price",
            "ignore food",
            "always grow",
            "no stock read",
        )
    )


def _rollout_mechanics_signal(task: dict[str, Any], output: str) -> float:
    lower = output.lower()
    bounded_markers = (
        "clamp",
        "floor",
        "cap",
        "bounded",
        "smooth",
        "ratio",
        "threshold",
        "taper",
        "hysteresis",
        "invariant",
    )
    marker_hits = sum(1 for marker in bounded_markers if marker in lower)
    flex = task.get("static_code_mechanics") if isinstance(task.get("static_code_mechanics"), dict) else {}
    flex_hits = sum(1 for signal in flex.get("flexible_signals") or [] if str(signal).lower() in lower)
    expect = task.get("expect") if isinstance(task.get("expect"), dict) else {}
    req_hits = sum(1 for token in expect.get("all_contains") or [] if str(token).lower() in lower)
    goal_hits = 0
    for goal in (task.get("simulation_spec") or {}).get("goals") or []:
        if not isinstance(goal, dict):
            continue
        for token in re.findall(r"[a-z]{5,}", str(goal.get("name") or "").lower()):
            if token in lower:
                goal_hits += 1
                break
    sim_context = sum(1 for token in ("simulation", "20-tick", "tick", "deterministic") if token in lower)
    return marker_hits * 0.35 + flex_hits * 0.12 + req_hits * 0.18 + min(goal_hits, 4) * 0.12 + sim_context * 0.1


def _uses_bounded_mechanics(output: str, task: dict[str, Any] | None = None) -> bool:
    if _reject_unbounded_language(output):
        return False
    lower = output.lower()
    markers = ("clamp", "floor", "cap", "bounded", "smooth", "ratio", "threshold", "taper")
    marker_hits = sum(1 for marker in markers if marker in lower)
    if marker_hits >= 2:
        return True
    if task is not None and _rollout_mechanics_signal(task, output) >= 1.0:
        return True
    if marker_hits >= 1 and any(token in lower for token in ("simulation", "20-tick", "invariant", "threshold", "reserve")):
        return True
    return False


def _output_covers_goal(name: str, output: str) -> bool:
    tokens = [tok for tok in re.findall(r"[a-z]{4,}", name.lower()) if tok not in {"this", "that", "with", "from", "still", "does", "are"}]
    if not tokens:
        return False
    lower = output.lower()
    hits = sum(1 for tok in tokens if tok in lower)
    return hits >= max(1, len(tokens) // 2)


def _infer_goal_score(name: str, trace: dict[str, Any], *, bounded: bool, output: str) -> float:
    lower = name.lower()
    if not bounded:
        return 0.0

    if "birth" in lower and any(token in lower for token in ("taper", "diff", "safe", "unsafe", "buffer")):
        return float(trace.get("birth_rate_decreased") or trace.get("birth_rate_diff_safe_unsafe"))
    if "food" in lower and any(token in lower for token in ("negative", "silent")):
        return float(trace.get("min_food", -1.0) >= 0.0 or trace.get("starvation_handled"))
    if "population" in lower and any(token in lower for token in ("spike", "oscill", "smooth")):
        return float(trace.get("smooth_population", False))
    if "scarcity" in lower and "price" in lower:
        return float(trace.get("scarcity_price_up", False))
    if "surplus" in lower and "price" in lower:
        return float(trace.get("surplus_price_down", False))
    if "price" in lower and any(token in lower for token in ("bound", "cap", "floor")):
        return float(trace.get("price_bounded", False))
    if "smooth" in lower or "oscill" in lower:
        return float(trace.get("smooth_adjustment", trace.get("smooth_population", False)))
    if "wage" in lower and any(token in lower for token in ("gold", "save", "cost", "upkeep")):
        return float(trace.get("wage_cut_saves", False))
    if "morale" in lower:
        return float(trace.get("morale_reacts", False))
    if "productivity" in lower:
        return float(trace.get("productivity_penalty", False))
    if "recover" in lower or "gradual" in lower:
        return float(trace.get("gradual_recovery", False))
    if "decay" in lower or "spoil" in lower:
        if "preserv" in lower or "upgrade" in lower:
            return float(trace.get("preservation_reduces_decay", False))
        return float(trace.get("decay_applies", False))
    if "capacity" in lower or "preserv" in lower or "warehouse" in lower:
        if "preserv" in lower or "upgrade" in lower:
            return float(trace.get("preservation_reduces_decay", trace.get("decay_applies", False)))
        return float(trace.get("capacity_respected", False))
    if "delete" in lower and "food" in lower:
        return float(trace.get("no_total_wipe", trace.get("capacity_respected", False)))
    if "upkeep" in lower and any(token in lower for token in ("bound", "cap", "afford")):
        return float(trace.get("upkeep_bounded", trace.get("small_force_affordable", False)))
    if "supply" in lower:
        return float(trace.get("supply_matters", False))
    if "small" in lower and any(token in lower for token in ("force", "garrison", "afford", "army")):
        return float(trace.get("small_force_affordable", False))
    if "large" in lower and any(token in lower for token in ("army", "force", "pressure", "multiplier")):
        return float(trace.get("large_force_pressures", trace.get("end_gt_start", False)))
    if "gold" in lower and any(token in lower for token in ("spiral", "impossible", "tick")):
        return float(trace.get("upkeep_bounded", trace.get("no_total_wipe", bounded)))
    if "stable" in lower or "combined" in lower:
        return float(trace.get("multidomain_stable", bounded))
    if "unrelated" in lower or "generalizes" in lower or "unseen" in lower:
        return float(bounded and trace.get("multidomain_stable", True))
    if any(token in lower for token in ("invalidate", "invalid", "dirty", "recompute", "cache", "stale", "projection")):
        if "stale" in lower:
            return float(trace.get("stale_reads", 1) == 0)
        if "projection" in lower or "recompute" in lower or "invalid" in lower or "dirty" in lower or "dependent" in lower:
            return float(trace.get("recomputes", 0) > 0 or trace.get("cache_integrity", False))
        return float(trace.get("cache_integrity", False))
    if "test" in lower and "pass" in lower:
        return float(_output_covers_goal(name, output))
    if _output_covers_goal(name, output):
        return 1.0
    return float(bounded)


def _score_all_goals(task: dict[str, Any], trace: dict[str, Any], *, bounded: bool, output: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for goal in task.get("simulation_spec", {}).get("goals") or []:
        if not isinstance(goal, dict):
            continue
        name = str(goal.get("name") or "")
        if name:
            scores[name] = _infer_goal_score(name, trace, bounded=bounded, output=output)
    return scores


def run_food_population_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    if not state:
        return {"ticks": 0, "goals": [], "error": "missing_initial_state"}
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)

    def _trace(food_stock: float) -> dict[str, Any]:
        population = state["population"]
        food_produced = state.get("foodProducedPerTick", FIELD_DEFAULTS["foodProducedPerTick"])
        food_consumed_per_pop = state.get("popFoodConsumedPerCapita", FIELD_DEFAULTS["popFoodConsumedPerCapita"])
        base_birth_rate = state.get("appliedBirthRate", FIELD_DEFAULTS["appliedBirthRate"])
        reserve_floor = state.get("popReserveFoodFloor", FIELD_DEFAULTS["popReserveFoodFloor"])
        unsafe_buffer_span = state.get("popUnsafeFoodBufferSpan", FIELD_DEFAULTS["popUnsafeFoodBufferSpan"])
        stock = food_stock
        birth_rates: list[float] = []
        population_deltas: list[float] = []
        food_values: list[float] = []
        starvation_handled = False
        for _ in range(tick_count):
            stock += food_produced
            stock -= population * food_consumed_per_pop
            if bounded:
                buffer_ratio = (stock - reserve_floor) / max(1.0, unsafe_buffer_span)
                effective_birth_rate = base_birth_rate * _clamp01(buffer_ratio)
                planned_births = population * effective_birth_rate
                projected = stock - planned_births * food_consumed_per_pop
                if projected < 0:
                    planned_births = 0.0
                    effective_birth_rate = 0.0
            else:
                effective_birth_rate = base_birth_rate
                planned_births = population * base_birth_rate
            if stock < 0:
                starvation_handled = True
                stock = 0.0
            population += planned_births
            birth_rates.append(effective_birth_rate)
            population_deltas.append(planned_births)
            food_values.append(stock)
        median_delta = statistics.median(population_deltas) if population_deltas else 0.0
        max_delta = max(population_deltas) if population_deltas else 0.0
        return {
            "birth_rates": birth_rates,
            "min_food": min(food_values) if food_values else 0.0,
            "birth_rate_decreased": bool(birth_rates and birth_rates[-1] < birth_rates[0]),
            "starvation_handled": starvation_handled,
            "smooth_population": max_delta <= 2.0 * max(median_delta, 1e-6),
        }

    safe = _trace(state["storageFood"] * 1.4)
    unsafe = _trace(max(state.get("popReserveFoodFloor", 50.0), state["storageFood"] * 0.45))
    trace = {
        **safe,
        "birth_rate_diff_safe_unsafe": bool(
            safe["birth_rates"] and unsafe["birth_rates"] and safe["birth_rates"][-1] > unsafe["birth_rates"][-1]
        ),
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    for legacy_name, legacy_score in {
        "birth_rate_tapers_near_food_floor": float(trace["birth_rate_decreased"]),
        "food_not_silently_negative": float(trace["min_food"] >= 0.0 or trace["starvation_handled"]),
        "no_population_oscillation": float(trace["smooth_population"]),
    }.items():
        if legacy_name not in goal_scores:
            goal_scores[legacy_name] = legacy_score if bounded else 0.0
    observed = {
        "bounded_mechanics_detected": bounded,
        "tick1_birth_rate": round(safe["birth_rates"][0], 4) if safe["birth_rates"] else 0.0,
        "tick20_birth_rate": round(safe["birth_rates"][-1], 4) if safe["birth_rates"] else 0.0,
        "min_food": round(safe["min_food"], 4),
        "birth_rate_diff_safe_unsafe": trace["birth_rate_diff_safe_unsafe"],
    }
    return _goal_payload(task, goal_scores, observed)


def run_market_elasticity_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    if not state:
        return {"ticks": 0, "goals": [], "error": "missing_initial_state"}
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    desired_stock = state.get("storageCapGoods", FIELD_DEFAULTS["storageCapGoods"])
    last_price = state.get("marketPriceGold", FIELD_DEFAULTS["marketPriceGold"])
    demand = state.get("demand", FIELD_DEFAULTS["demand"])
    scarcity_stock = state.get("storageGoods", FIELD_DEFAULTS["storageGoods"])
    surplus_stock = state.get("surplusStorageGoods", desired_stock * 1.5)

    def _prices(stock: float) -> list[float]:
        prices = [last_price]
        price = last_price
        for _ in range(tick_count - 1):
            if bounded:
                pressure = desired_stock / max(stock, 1.0)
                multiplier = _clamp(1.0 + 0.18 * (pressure - 1.0), 0.75, 1.35)
                target = _clamp(price * multiplier * demand, 5.0, 25.0)
                price = _clamp(price + 0.22 * (target - price), 5.0, 25.0)
            else:
                price += 3.5
            prices.append(price)
        return prices

    scarcity_prices = _prices(scarcity_stock)
    surplus_prices = _prices(surplus_stock)
    scarcity_deltas = [abs(scarcity_prices[i] - scarcity_prices[i - 1]) for i in range(1, len(scarcity_prices))]
    trace = {
        "scarcity_price_up": scarcity_prices[-1] > scarcity_prices[0],
        "surplus_price_down": surplus_prices[-1] < scarcity_prices[-1],
        "price_bounded": all(5.0 <= p <= 25.0 for p in scarcity_prices + surplus_prices),
        "smooth_adjustment": bool(scarcity_deltas and max(scarcity_deltas) <= 1.25),
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    for legacy_name, legacy_score in {
        "scarcity_raises_price": float(trace["scarcity_price_up"]),
        "surplus_lowers_price": float(trace["surplus_price_down"]),
        "price_bounded": float(trace["price_bounded"]),
        "smooth_adjustment": float(trace["smooth_adjustment"]),
    }.items():
        if legacy_name not in goal_scores:
            goal_scores[legacy_name] = legacy_score if bounded else 0.0
    observed = {
        "bounded_mechanics_detected": bounded,
        "scarcity_end_price": round(scarcity_prices[-1], 4),
        "surplus_end_price": round(surplus_prices[-1], 4),
    }
    return _goal_payload(task, goal_scores, observed)


def run_labor_wage_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    wage = state.get("wage", FIELD_DEFAULTS["wage"])
    expected = state.get("expectedWage", FIELD_DEFAULTS["expectedWage"])
    morale = state.get("cityMorale", FIELD_DEFAULTS["cityMorale"])
    productivity = state.get("productivity", FIELD_DEFAULTS["productivity"])
    gold = state.get("playerGold", FIELD_DEFAULTS["playerGold"])
    upkeep_series: list[float] = []
    morale_series: list[float] = []
    productivity_series: list[float] = []
    for tick in range(tick_count):
        if bounded:
            cut_wage = max(expected * 0.7, wage * (1.0 - 0.02 * tick))
            upkeep = cut_wage * state.get("totalEmployed", 100.0) / 100.0
            morale_gap = max(0.0, expected - cut_wage)
            morale = _clamp(morale - 0.08 * morale_gap, 0.2, 1.0)
            productivity = _clamp(productivity - 0.12 * morale_gap if morale < 0.55 else productivity * 0.01, 0.35, 1.0)
        else:
            upkeep = wage * 2.0
        gold -= upkeep
        upkeep_series.append(upkeep)
        morale_series.append(morale)
        productivity_series.append(productivity)
    trace = {
        "wage_cut_saves": upkeep_series[-1] < upkeep_series[0],
        "morale_reacts": morale_series[-1] < morale_series[0],
        "productivity_penalty": productivity_series[-1] < 0.95,
        "gradual_recovery": max(abs(morale_series[i] - morale_series[i - 1]) for i in range(1, len(morale_series))) <= 0.2,
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    observed = {"bounded_mechanics_detected": bounded, "end_morale": round(morale_series[-1], 4)}
    return _goal_payload(task, goal_scores, observed)


def run_inventory_spoilage_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    capacity = state.get("storageCapFood", FIELD_DEFAULTS["storageCapFood"])
    stock = state.get("storageFood", FIELD_DEFAULTS["storageFood"])
    decay_rate = state.get("foodDecayPerTick", FIELD_DEFAULTS["foodDecayPerTick"])
    preservation = state.get("preservationModifier", FIELD_DEFAULTS["preservationModifier"])

    def _simulate(initial_stock: float, preservation_mod: float) -> list[float]:
        values = [initial_stock]
        for _ in range(tick_count - 1):
            surplus = max(0.0, values[-1] - capacity)
            if bounded:
                decay = surplus * decay_rate * (1.0 - preservation_mod * 0.5)
                next_stock = max(0.0, values[-1] - decay)
            else:
                next_stock = values[-1] * 0.5 if surplus > 0 else values[-1]
            values.append(next_stock)
        return values

    values = _simulate(stock, preservation)
    weak_preservation_values = _simulate(stock, max(0.15, preservation - 0.45))
    trace = {
        "decay_applies": values[-1] < values[0],
        "preservation_reduces_decay": values[-1] >= weak_preservation_values[-1],
        "capacity_respected": min(values) >= 0.0,
        "no_total_wipe": values[-1] > 0.0 and values[-1] >= values[0] * 0.2,
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    for legacy_name, legacy_score in {
        "surplus_decay_applies": float(trace["decay_applies"]),
        "within_capacity_preserved": float(trace["capacity_respected"]),
        "upgrade_reduces_decay": float(trace["preservation_reduces_decay"]),
        "decay_bounded": float(trace["no_total_wipe"]),
    }.items():
        if legacy_name not in goal_scores:
            goal_scores[legacy_name] = legacy_score if bounded else 0.0
    observed = {
        "bounded_mechanics_detected": bounded,
        "start_stock": round(values[0], 4),
        "end_stock": round(values[-1], 4),
        "weak_preservation_end_stock": round(weak_preservation_values[-1], 4),
    }
    return _goal_payload(task, goal_scores, observed)


def run_upkeep_scaling_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    small_army = max(1.0, state.get("garrisonUnitCount", FIELD_DEFAULTS["garrisonUnitCount"]))
    large_army = small_army * 4.0
    base = state.get("upkeepPerUnit", FIELD_DEFAULTS["upkeepPerUnit"])
    distance = state.get("supplyDistance", FIELD_DEFAULTS["supplyDistance"])

    def _upkeep(size: float) -> float:
        raw = base * size * (1.0 + 0.04 * size) * (1.0 + 0.05 * distance)
        if bounded:
            return _clamp(raw, base, base * size * 2.5)
        return raw * 3.0

    small = _upkeep(small_army)
    large = _upkeep(large_army)
    trace = {
        "small_force_affordable": bounded and (small < large and small_army <= 24.0),
        "large_force_pressures": large > small * 2.0,
        "supply_matters": bounded,
        "upkeep_bounded": large <= base * large_army * 2.5 and small <= large,
        "no_total_wipe": bounded,
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    observed = {"bounded_mechanics_detected": bounded, "small_upkeep": round(small, 4), "large_upkeep": round(large, 4)}
    return _goal_payload(task, goal_scores, observed)


def run_cache_projection_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    state = resolve_scenario_state(task)
    bounded = _uses_bounded_mechanics(output, task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    lower = output.lower()
    mentions_invalidation = any(token in lower for token in ("invalidate", "dirty", "recompute", "cache", "stale", "projection"))
    if not mentions_invalidation and str(task.get("subsection") or "") == "adversarial_multidomain_economy":
        mentions_invalidation = bounded and any(
            token in lower for token in ("simulation", "domain", "dependent", "deterministic", "bounded")
        )
    stale_reads = 0
    recomputes = 0
    for tick in range(tick_count):
        mutation = tick % 4 == 0
        if bounded and mentions_invalidation:
            if mutation:
                recomputes += 1
            stale_reads = 0
        elif mutation:
            stale_reads += 1
    trace = {
        "recomputes": recomputes,
        "stale_reads": stale_reads,
        "cache_integrity": bounded and mentions_invalidation and stale_reads == 0,
        "bounded_mechanics_detected": bounded,
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    observed = {
        "bounded_mechanics_detected": bounded,
        "recomputes": recomputes,
        "stale_reads": stale_reads,
        "seed_net_delta": round(state.get("netGoldDelta", FIELD_DEFAULTS["netGoldDelta"]), 4),
    }
    return _goal_payload(task, goal_scores, observed)


def run_bounded_scalar_sim(task: dict[str, Any], output: str, *, direction: str) -> dict[str, Any]:
    """Fallback monotone scalar env for economy subsections without a bespoke sim."""
    state = resolve_scenario_state(task)
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    bounded = _uses_bounded_mechanics(output, task)
    start = float(statistics.mean(state.values())) if state else 1.0
    values = [start]
    current = start
    for tick in range(tick_count - 1):
        pressure = 1.0 + (0.08 * tick if direction == "up" else -0.05 * tick)
        if bounded:
            target = _clamp(current * pressure, current * 0.7, current * 1.35)
            current = current + 0.2 * (target - current)
        else:
            current *= pressure
        values.append(current)
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    trace = {
        "smooth_adjustment": bool(deltas and max(deltas) <= 2.0 * statistics.median(deltas)),
        "bounded_mechanics_detected": bounded,
        "end_gt_start": values[-1] > values[0],
        "end_lt_start": values[-1] < values[0],
    }
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=output)
    observed = {"bounded_mechanics_detected": bounded, "start": round(start, 4), "end": round(values[-1], 4)}
    return _goal_payload(task, goal_scores, observed)


def run_adversarial_multidomain_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    """Couple food + market + cache traces for adversarial multidomain tasks."""
    bounded = _uses_bounded_mechanics(output, task)
    food = run_food_population_sim(task, output)
    market = run_market_elasticity_sim(task, output)
    cache = run_cache_projection_sim(task, output)
    merged_scores: dict[str, float] = {}
    domain_scores: list[float] = []
    for payload in (food, market, cache):
        goals = payload.get("goals") or []
        if goals:
            domain_scores.append(sum(float(g.get("score") or 0) for g in goals) / len(goals))
        for goal in goals:
            if isinstance(goal, dict):
                merged_scores[str(goal.get("name") or "")] = float(goal.get("score") or 0.0)
    cache_trace = cache.get("trace_summary") or {}
    goal_scores = _score_all_goals(
        task,
        {
            "bounded_mechanics_detected": bounded,
            "cache_integrity": bool(cache_trace.get("stale_reads", 1) == 0 and cache_trace.get("recomputes", 0) > 0),
            "recomputes": int(cache_trace.get("recomputes") or 0),
            "stale_reads": int(cache_trace.get("stale_reads") or 0),
            "multidomain_stable": bounded and bool(domain_scores) and min(domain_scores) >= 0.75,
            "scarcity_price_up": market.get("trace_summary", {}).get("scarcity_end_price", 0) > 0,
        },
        bounded=bounded,
        output=output,
    )
    for name, score in merged_scores.items():
        goal_scores.setdefault(name, score)
    observed = {
        "bounded_mechanics_detected": bounded,
        "domains": ["food", "market", "cache"],
        "domain_mean_scores": [round(score, 4) for score in domain_scores],
        "recomputes": cache_trace.get("recomputes", 0),
    }
    return _goal_payload(task, goal_scores, observed)


def run_generalist_rubric_sim(task: dict[str, Any], output: str) -> dict[str, Any]:
    """Generalist tasks: score goals from prompt-rubric alignment rather than mechanics."""
    expect = task.get("expect") if isinstance(task.get("expect"), dict) else {}
    lower = output.lower()
    goal_scores: dict[str, float] = {}
    for goal in task.get("simulation_spec", {}).get("goals") or []:
        name = str(goal.get("name") or "")
        required = [str(token).lower() for token in expect.get("all_contains") or []]
        optional = [str(token).lower() for token in expect.get("any_contains") or []]
        req_hits = sum(1 for token in required if token in lower)
        opt_hit = any(token in lower for token in optional) if optional else True
        score = 0.0
        if required:
            score = _clamp01(req_hits / len(required))
        if opt_hit:
            score = min(1.0, score + 0.15)
        goal_scores[name] = float(score >= 0.8)
    observed = {"output_chars": len(output), "rubric_mode": True}
    return _goal_payload(task, goal_scores, observed)


SUBSECTION_SIMULATORS: dict[str, SimFn] = {
    "food_population_feedback": run_food_population_sim,
    "market_elasticity_pricing": run_market_elasticity_sim,
    "labor_wage_productivity": run_labor_wage_sim,
    "inventory_storage_spoilage": run_inventory_spoilage_sim,
    "upkeep_progressive_costs": run_upkeep_scaling_sim,
    "resource_projection_cache_integrity": run_cache_projection_sim,
    "adversarial_multidomain_economy": run_adversarial_multidomain_sim,
    "instruction_following": run_generalist_rubric_sim,
    "structured_reasoning": run_generalist_rubric_sim,
    "code_patch_planning": run_generalist_rubric_sim,
    "test_design_and_invariants": run_generalist_rubric_sim,
    "debugging_and_root_cause": run_generalist_rubric_sim,
    "concise_explanation": run_generalist_rubric_sim,
    "safety_and_scope_control": run_generalist_rubric_sim,
}


def run_simulation_for_task(task: dict[str, Any], output: str) -> dict[str, Any]:
    subsection = str(task.get("subsection") or "").strip()
    sim_fn = SUBSECTION_SIMULATORS.get(subsection)
    if sim_fn is None:
        return {"ticks": 0, "goals": [], "error": f"no_simulator:{subsection or 'unknown'}"}
    try:
        return sim_fn(task, output)
    except Exception as exc:  # pragma: no cover - safety net for task-bank drift
        return {
            "ticks": 0,
            "goals": [],
            "error": f"simulator_error:{type(exc).__name__}:{exc}",
            "subsection": subsection,
        }
