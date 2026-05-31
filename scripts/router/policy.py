"""Adapter-first route policy helpers."""

from __future__ import annotations

import re
from typing import Iterable


FRONTIER_ADAPTERS = {
    "security_review",
    "architecture_design",
    "save_load_api_guard",
}

HYBRID_ADAPTERS = {
    "combat_risk",
    "ai_planning_explanation",
    "economistRL",
}

LOCAL_ADAPTERS = {
    "loading_screen",
    "hud_status",
    "economy_tooltip",
    "documentation",
    "general_fallback",
}

_COARSE_RULES = [
    {
        "bucket": "state_contract_guardrails",
        "keywords": (
            "state persistence",
            "state contract",
            "save/load",
            "save load",
            "savegame",
            "autosave",
            "load game",
            "serialization",
            "snapshot schema",
            "request validation",
            "identity integrity",
            "api guard",
            "auth guard",
            "endpoint",
        ),
        "candidates": ("save_load_api_guard", "ai_planning_explanation", "general_fallback"),
    },
    {
        "bucket": "crossdomain_state_patch",
        "keywords": (
            "crossdomain",
            "cross-system",
            "state logic",
            "schema update",
            "shared logic",
            "repair candidate",
            "general patch",
            "tradeoff",
            "review",
            "optimize",
            "plan",
        ),
        "candidates": ("ai_planning_explanation", "combat_risk", "economy_tooltip", "general_fallback"),
    },
    {
        "bucket": "army_ui_flow",
        "keywords": (
            "army ui",
            "formation flow",
            "order dispatch",
            "movement ui",
            "tactical controls",
            "selection flow",
            "combat risk",
            "combat preview",
            "battle risk",
            "enemy stats",
            "threat",
            "engagement",
        ),
        "candidates": ("combat_risk", "ai_planning_explanation", "general_fallback"),
    },
    {
        "bucket": "ui_surface_and_state_signals",
        "keywords": (
            "ui surface",
            "visual surface",
            "panel theme",
            "sprite overlay",
            "isometric display",
            "loading screen",
            "load screen",
            "splash screen",
            "title screen",
            "hud",
            "status panel",
            "status overlay",
            "ui overlay",
            "status summary",
            "status signal",
            "resource signal",
            "state exposure",
            "indicator logic",
        ),
        "candidates": ("loading_screen", "hud_status", "economy_tooltip", "general_fallback"),
    },
    {
        "bucket": "economy_simulation_rl",
        "keywords": (
            "economistrl",
            "economist rl",
            "economy simulation",
            "market dynamics",
            "market elasticity",
            "population dynamics",
            "food economy",
            "food buffer",
            "labor economy",
            "worker wage",
            "productivity feedback",
            "spoilage",
            "progressive upkeep",
            "feedback control",
        ),
        "candidates": ("economistRL", "economy_tooltip", "ai_planning_explanation", "general_fallback"),
    },
    {
        "bucket": "resource_ui_projection",
        "keywords": (
            "resource projection",
            "economy ui",
            "tooltip",
            "cost display",
            "production signal",
            "resource cache",
            "signed delta",
            "resource",
            "net income",
            "gold per turn",
        ),
        "candidates": ("economy_tooltip", "hud_status", "general_fallback"),
    },
    {
        "bucket": "docs_and_workflow",
        "keywords": (
            "documentation",
            "project_state",
            "session_log",
            "run_history",
            "workflow",
            "docs/",
            "benchmark script",
            "run log",
        ),
        "candidates": ("documentation", "general_fallback"),
    },
]


def _contains_keyword(prompt: str, keyword: str) -> bool:
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
    return re.search(pattern, prompt) is not None


def infer_coarse_adapter_candidates(
    *,
    prompt: str,
    available_adapters: set[str],
    max_candidates: int = 4,
) -> tuple[str, list[str], int, str]:
    """Return (coarse_bucket, candidate_adapters, keyword_hits, reason)."""
    best_rule = None
    best_hits = 0
    for rule in _COARSE_RULES:
        hits = sum(1 for kw in rule["keywords"] if _contains_keyword(prompt, kw))
        if hits > best_hits:
            best_hits = hits
            best_rule = rule

    if best_rule is None or best_hits <= 0:
        candidates = [aid for aid in ("general_fallback", *sorted(available_adapters)) if aid in available_adapters]
        if "general_fallback" not in candidates and "general_fallback" in available_adapters:
            candidates.insert(0, "general_fallback")
        deduped = list(dict.fromkeys(candidates))[: max(1, max_candidates)]
        return (
            "unclassified",
            deduped,
            0,
            "no coarse taxonomy keyword hits; fallback candidate pool",
        )

    ordered = [aid for aid in best_rule["candidates"] if aid in available_adapters]
    if "general_fallback" in available_adapters and "general_fallback" not in ordered:
        ordered.append("general_fallback")
    if not ordered:
        ordered = ["general_fallback"]
    deduped = list(dict.fromkeys(ordered))[: max(1, max_candidates)]
    return (
        str(best_rule["bucket"]),
        deduped,
        best_hits,
        f"coarse taxonomy bucket `{best_rule['bucket']}` ({best_hits} keyword hits)",
    )


def derive_route_from_adapter(
    *,
    adapter_id: str,
    input_tokens: int,
    force_route: str | None,
    long_prompt_tokens: int,
    frontier_keywords_matched: bool,
    hybrid_keywords_matched: bool,
) -> tuple[str, str]:
    if force_route in {"local", "frontier", "hybrid"}:
        return force_route, f"forced route: {force_route}"

    if frontier_keywords_matched:
        return "frontier", "high-risk keyword override"
    if input_tokens >= long_prompt_tokens:
        return "frontier", "long prompt token override"
    if hybrid_keywords_matched and adapter_id == "general_fallback":
        return "hybrid", "review/planning keyword override for general fallback"

    if adapter_id in FRONTIER_ADAPTERS:
        return "frontier", f"adapter policy escalates `{adapter_id}` to frontier"
    if adapter_id in HYBRID_ADAPTERS:
        return "hybrid", f"adapter policy routes `{adapter_id}` to hybrid"
    if adapter_id in LOCAL_ADAPTERS:
        return "local", f"adapter policy routes `{adapter_id}` local-first"
    return "local", "default local route for routine request"

