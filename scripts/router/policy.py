#!/usr/bin/env python3
"""Adapter-aware routing policy with council modes and escalation ladder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .classifier import ClassificationResult


@dataclass
class RoutingPlan:
    route: str
    adapter_id: str
    execution_tier: str
    fallback_reason: str
    retry_policy: str
    council_mode: str
    council_size: int
    judge_mode: str
    confidence: float
    ambiguity: float
    risk_class: str
    complexity: str
    policy_version: str = "router_policy_v1"
    lineage: Dict[str, Any] = field(default_factory=dict)


def build_plan(result: ClassificationResult, *, metadata: Optional[Dict[str, Any]] = None) -> RoutingPlan:
    c = result.confidence
    a = result.ambiguity
    r = result.risk_class
    k = result.complexity
    route = "local"
    execution_tier = "mac_pool"
    fallback_reason = "none"
    retry_policy = "repair_once_then_escalate"
    council_mode = "off"
    council_size = 1
    judge_mode = "none"
    adapter_id = result.adapter_id

    # Fast-path for proven specialist domains — route directly local by default.
    # loading_screen: omit when classifier marks literal high-risk (auth/security-heavy non-domain).
    # save_load_api_guard: in-scope prompts legitimately trigger auth/security/api signals; treat
    # classifier winner as authoritative so we avoid council/API escalations from those keywords alone.
    if adapter_id == "loading_screen" and r != "high":
        route = "local"
        execution_tier = "mac_pool"
        fallback_reason = "loading_screen_specialist_fastpath"
        council_mode = "off"
        council_size = 1
        judge_mode = "none"
    elif adapter_id == "save_load_api_guard":
        route = "local"
        execution_tier = "mac_pool"
        fallback_reason = "save_load_api_guard_specialist_fastpath"
        council_mode = "off"
        council_size = 1
        judge_mode = "none"
    elif adapter_id == "economy_tooltip":
        route = "local"
        execution_tier = "mac_pool"
        fallback_reason = "economy_tooltip_specialist_fastpath"
        council_mode = "off"
        council_size = 1
        judge_mode = "none"
    elif adapter_id == "documentation":
        route = "local"
        execution_tier = "mac_pool"
        fallback_reason = "documentation_specialist_fastpath"
        council_mode = "off"
        council_size = 1
        judge_mode = "none"
    elif r == "high" and c < 0.72:
        route = "api"
        execution_tier = "api_judge"
        fallback_reason = "high_risk_low_confidence_override"
        council_mode = "api_judge_first"
        council_size = 0
        judge_mode = "api_first"
    elif k == "high" or c < 0.58 or a > 0.58 or r == "high":
        route = "council"
        execution_tier = "high_compute_mac"
        council_mode = "debate_judge"
        council_size = 3
        judge_mode = "api_first"
        fallback_reason = "high_complexity_or_ambiguity"
    elif k == "medium" or (0.58 <= c < 0.78) or (0.32 < a <= 0.58):
        route = "council"
        execution_tier = "mac_pool"
        council_mode = "fast_vote"
        council_size = 4
        judge_mode = "route_plan_then_executor"
        fallback_reason = "medium_complexity_or_uncertain"
    elif k == "low" and c >= 0.78 and a <= 0.32 and r != "high":
        route = "local"
        execution_tier = "mac_pool"
        council_mode = "off"
        council_size = 1
        judge_mode = "none"
    else:
        route = "council"
        execution_tier = "mac_pool"
        council_mode = "fast_vote"
        council_size = 4
        judge_mode = "route_plan_then_executor"
        fallback_reason = "default_ambiguity_path"

    if route == "api":
        adapter_id = "general_fallback"
    return RoutingPlan(
        route=route,
        adapter_id=adapter_id,
        execution_tier=execution_tier,
        fallback_reason=fallback_reason,
        retry_policy=retry_policy,
        council_mode=council_mode,
        council_size=council_size,
        judge_mode=judge_mode,
        confidence=result.confidence,
        ambiguity=result.ambiguity,
        risk_class=result.risk_class,
        complexity=result.complexity,
        lineage={
            "policy_version": "router_policy_v1",
            "council_thresholds_version": "council_thresholds_v1",
            "metadata": metadata or {},
        },
    )


def plan_to_legacy_route(plan: RoutingPlan) -> str:
    if plan.route == "api":
        return "frontier"
    if plan.route == "council":
        return "hybrid"
    return "local"

