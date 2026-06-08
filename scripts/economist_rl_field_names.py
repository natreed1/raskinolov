#!/usr/bin/env python3
"""Canonical economistRL sandbox field names aligned to Fallen Empire vocabulary.

Flat numeric state slices mirror production concepts without nested ``City`` objects:
``storageFood`` ↔ ``City.storage.food``, ``storageCapFood`` ↔ ``City.storageCap.food``,
``population`` ↔ ``City.population``, ``playerGold`` ↔ player treasury, etc.

Legacy lab keys from v1/v2 task banks are accepted via :data:`SCENARIO_FIELD_ALIASES`
and normalized at scenario read time.
"""

from __future__ import annotations

from typing import Any

# Legacy lab / task-bank key → canonical sandbox key (game-aligned).
SCENARIO_FIELD_ALIASES: dict[str, str] = {
    "foodStock": "storageFood",
    "surplusFood": "storageFood",
    "reserveFloor": "popReserveFoodFloor",
    "unsafeBufferSpan": "popUnsafeFoodBufferSpan",
    "birthRate": "appliedBirthRate",
    "foodConsumedPerPop": "popFoodConsumedPerCapita",
    "currentStock": "storageGoods",
    "desiredStock": "storageCapGoods",
    "surplusStock": "surplusStorageGoods",
    "lastPrice": "marketPriceGold",
    "gold": "playerGold",
    "workerOutput": "totalEmployed",
    "storageCapacity": "storageCapFood",
    "armySize": "garrisonUnitCount",
    "baseUpkeep": "upkeepPerUnit",
    "projectionCache": "resourceProjectionValid",
    "projectionCacheDirty": "resourceProjectionCacheDirty",
    "dirtyScopes": "invalidationScopeCount",
    "netDelta": "netGoldDelta",
    "buildingState": "buildingCompletionDirty",
    "marketPrices": "marketPriceDirty",
    "workerAssignments": "workerReassignmentDirty",
    "upkeepModifiers": "upkeepModifierDirty",
    "armyUpkeep": "upkeepTotal",
    "workerMorale": "cityMorale",
    "morale": "cityMorale",
    "foodDecayRate": "foodDecayPerTick",
    "upkeep": "upkeepTotal",
}

SUBSECTION_CANONICAL_DEFAULTS: dict[str, list[str]] = {
    "food_population_feedback": [
        "storageFood",
        "storageCapFood",
        "population",
        "appliedBirthRate",
        "popReserveFoodFloor",
        "popUnsafeFoodBufferSpan",
    ],
    "market_elasticity_pricing": [
        "storageGoods",
        "storageCapGoods",
        "marketPriceGold",
        "demand",
        "scarcityPressure",
        "surplusPressure",
    ],
    "labor_wage_productivity": [
        "playerGold",
        "wage",
        "expectedWage",
        "cityMorale",
        "productivity",
        "totalEmployed",
    ],
    "inventory_storage_spoilage": [
        "storageFood",
        "storageCapFood",
        "warehouseLevel",
        "preservationModifier",
        "foodDecayPerTick",
    ],
    "upkeep_progressive_costs": [
        "playerGold",
        "garrisonUnitCount",
        "upkeepPerUnit",
        "supplyDistance",
        "upkeepTotal",
        "netGoldDelta",
    ],
    "resource_projection_cache_integrity": [
        "resourceProjectionValid",
        "resourceProjectionCacheDirty",
        "invalidationScopeCount",
        "marketPriceDirty",
        "buildingCompletionDirty",
    ],
    "adversarial_multidomain_economy": [
        "storageFood",
        "population",
        "storageGoods",
        "storageCapGoods",
        "marketPriceGold",
        "playerGold",
        "upkeepTotal",
    ],
    "generalist_bounded": ["scalar", "pressure", "boundedSignal"],
}


def canonicalize_field_name(name: str) -> str:
    token = str(name or "").strip()
    if not token:
        return token
    return SCENARIO_FIELD_ALIASES.get(token, token)


def canonicalize_field_list(fields: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in fields:
        name = canonicalize_field_name(str(raw).strip())
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _coerce_scalar(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        nums = [_coerce_scalar(v) for v in raw.values()]
        nums = [n for n in nums if n is not None]
        return float(sum(nums)) if nums else float(len(raw))
    if isinstance(raw, list):
        nums = [_coerce_scalar(v) for v in raw]
        nums = [n for n in nums if n is not None]
        return float(sum(nums)) if nums else float(len(raw))
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None


def canonicalize_state(state: dict[str, Any]) -> dict[str, float]:
    """Map legacy scenario keys to canonical names; merge duplicate aliases."""
    out: dict[str, float] = {}
    for key, raw in state.items():
        canon = canonicalize_field_name(str(key))
        value = _coerce_scalar(raw)
        if value is None:
            continue
        if canon in out:
            out[canon] = max(out[canon], value)
        else:
            out[canon] = value
    return out
