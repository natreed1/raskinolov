#!/usr/bin/env python3
"""Deterministic v1 task classifier with confidence/ambiguity/risk outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "documentation": (
        "session_log",
        "project_state",
        "run_history",
        "ml_workflow",
        "data_layout",
        "fallen-empire-lora",
        "mlx lab",
        "mlx-lab",
        "lora lab",
        "training run",
        "training runs",
        "run manifest",
        "manifest.json",
        "run.md",
        "training_trajectory",
        "trajectory.jsonl",
        "token usage",
        "cursor_usage",
        "lab_dashboard",
        "adapter registry",
        "adapter_registry_v1",
        "documentation_specialist",
        "documentation-dataset",
        "benchmarks/results/runs",
        "precise-ml-documentation",
        "chunked_game_text",
        "checkpoints/adapters/documentation",
    ),
    "loading_screen": ("loading", "start screen", "splash"),
    "hud_status": ("hud", "status", "morale", "supply"),
    "economy_tooltip": ("economy", "tooltip", "income", "resource"),
    "combat_risk": ("combat", "risk", "battle", "terrain", "morale"),
    "save_load_api_guard": ("save", "load", "serialization", "api", "auth", "security"),
    "ai_planning_explanation": ("ai", "planning", "strategy", "rationale", "architecture"),
}

HIGH_RISK_TERMS = ("security", "auth", "authentication", "privacy", "data loss", "migration")
HIGH_COMPLEXITY_TERMS = ("architecture", "cross-domain", "distributed", "refactor", "multi-step")


@dataclass
class ClassificationResult:
    adapter_id: str
    confidence: float
    ambiguity: float
    risk_class: str
    complexity: str
    secondary_adapter_id: str
    top_candidates: List[Dict[str, float]]
    policy_version: str = "router_policy_v1"


def classify_prompt(prompt: str) -> ClassificationResult:
    text = prompt.lower()
    words = text.split()
    scores: Dict[str, float] = {"general_fallback": 0.15}
    for adapter, terms in KEYWORDS.items():
        score = 0.0
        for term in terms:
            if term in text:
                score += 1.0
        if score:
            scores[adapter] = score
    ordered = sorted(scores.items(), key=lambda it: it[1], reverse=True)
    top_adapter, top_score = ordered[0]
    second_adapter, second_score = ordered[1] if len(ordered) > 1 else ("general_fallback", 0.0)
    total = sum(max(v, 0.0) for _, v in ordered) or 1.0
    confidence = min(1.0, max(0.2, top_score / total + 0.2))
    margin = top_score - second_score
    ambiguity = max(0.0, min(1.0, 1.0 - (margin / max(1.0, top_score + second_score + 0.5))))
    risk_class = "high" if any(term in text for term in HIGH_RISK_TERMS) else "medium" if "api" in text else "low"
    if any(term in text for term in HIGH_COMPLEXITY_TERMS):
        complexity = "high"
    elif len(words) > 220 or len(text) > 1500:
        complexity = "medium"
    else:
        complexity = "low"
    top_candidates = [{"adapter_id": aid, "score": round(score, 4)} for aid, score in ordered[:4]]
    return ClassificationResult(
        adapter_id=top_adapter,
        confidence=round(confidence, 4),
        ambiguity=round(ambiguity, 4),
        risk_class=risk_class,
        complexity=complexity,
        secondary_adapter_id=second_adapter,
        top_candidates=top_candidates,
    )
