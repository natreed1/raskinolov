"""Planner rollout document contract.

The planner document is the explicit action that planner policies will learn
to improve. Current routers may still provide only a ``council_plan``; in that
case we derive a compatible document so every rollout can be graded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PLANNER_ROLLOUT_PLAN_SCHEMA = "planner_rollout_plan_v1"

REQUIRED_PLAN_FIELDS = (
    "task_synopsis",
    "primary_goal",
    "success_criteria",
    "known_constraints",
    "risk_tags",
    "required_context",
    "selected_experts",
    "excluded_experts",
    "debate_plan",
)

REQUIRED_SELECTED_EXPERT_FIELDS = (
    "expert_id",
    "role",
    "why_selected",
    "assigned_question",
    "expected_contribution",
    "handoff_context",
)


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float(default)
    return max(0.0, min(1.0, raw))


def estimate_tokens(text: str) -> int:
    return max(1, int(round(len(str(text or "")) / 4.0)))


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_str(item) for item in value if _clean_str(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def infer_risk_tags(text: str, *, participants: list[dict[str, Any]] | None = None) -> list[str]:
    haystack = f"{text} " + " ".join(
        f"{row.get('participant_id', '')} {row.get('base_expert_id', '')} {row.get('strategy', '')}"
        for row in list(participants or [])
    )
    lowered = haystack.lower()
    rules = {
        "persistence": ("save", "load", "migration", "schema", "serialize", "storage"),
        "ui_state": ("hud", "ui", "status", "screen", "render", "component", "frontend"),
        "economy_balance": ("economy", "food", "market", "price", "morale", "population", "resource"),
        "test_coverage": ("test", "verify", "regression", "compile", "lint", "coverage"),
        "api_contract": ("api", "contract", "interface", "payload", "request", "response"),
        "context_overload": ("large", "context", "many files", "compress", "summary"),
        "security": ("auth", "permission", "token", "secret", "sanitize", "security"),
    }
    tags = [tag for tag, tokens in rules.items() if any(token in lowered for token in tokens)]
    return tags or ["general_correctness"]


def infer_task_type(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token in lowered for token in ("fix", "bug", "debug", "regression")):
        return "debugging"
    if any(token in lowered for token in ("implement", "add", "build", "create")):
        return "implementation"
    if any(token in lowered for token in ("review", "audit", "inspect")):
        return "review"
    if any(token in lowered for token in ("evaluate", "benchmark", "score", "test")):
        return "evaluation"
    if any(token in lowered for token in ("design", "plan", "architecture")):
        return "design"
    return "research"


@dataclass(frozen=True)
class PlannerPlanValidation:
    valid: bool
    diagnostics: list[str] = field(default_factory=list)
    completeness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "diagnostics": list(self.diagnostics),
            "completeness": round(clamp01(self.completeness), 6),
        }


@dataclass(frozen=True)
class PlannerRolloutPlan:
    task_synopsis: str
    primary_goal: str
    success_criteria: list[str]
    known_constraints: list[str]
    risk_tags: list[str]
    required_context: list[str]
    selected_experts: list[dict[str, Any]]
    excluded_experts: list[dict[str, Any]]
    debate_plan: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLANNER_ROLLOUT_PLAN_SCHEMA,
            "task_synopsis": self.task_synopsis,
            "primary_goal": self.primary_goal,
            "success_criteria": list(self.success_criteria),
            "known_constraints": list(self.known_constraints),
            "risk_tags": list(self.risk_tags),
            "required_context": list(self.required_context),
            "selected_experts": [dict(row) for row in self.selected_experts],
            "excluded_experts": [dict(row) for row in self.excluded_experts],
            "debate_plan": dict(self.debate_plan),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerRolloutPlan":
        return cls(
            task_synopsis=_clean_str(payload.get("task_synopsis")),
            primary_goal=_clean_str(payload.get("primary_goal")),
            success_criteria=_as_str_list(payload.get("success_criteria")),
            known_constraints=_as_str_list(payload.get("known_constraints")),
            risk_tags=_as_str_list(payload.get("risk_tags")),
            required_context=_as_str_list(payload.get("required_context")),
            selected_experts=[dict(row) for row in list(payload.get("selected_experts") or []) if isinstance(row, dict)],
            excluded_experts=[dict(row) for row in list(payload.get("excluded_experts") or []) if isinstance(row, dict)],
            debate_plan=dict(payload.get("debate_plan") or {}),
            raw=dict(payload),
        )


def derive_planner_document(*, prompt: str, council_plan: dict[str, Any]) -> PlannerRolloutPlan:
    participants = list((council_plan or {}).get("participants") or [])
    risk_tags = infer_risk_tags(prompt, participants=participants)
    task_type = infer_task_type(prompt)
    selected = []
    for row in participants:
        expert_id = _clean_str(row.get("base_expert_id") or row.get("participant_id"))
        role = _clean_str(row.get("role") or row.get("participant_type") or "expert")
        strategy = _clean_str(row.get("strategy") or row.get("variant") or "task_support")
        selected.append(
            {
                "expert_id": expert_id,
                "participant_id": _clean_str(row.get("participant_id") or expert_id),
                "role": role,
                "why_selected": f"Selected for {role} coverage using {strategy}.",
                "assigned_question": "Identify task-relevant risks, concrete edits, evidence, and verification steps.",
                "expected_contribution": "A bounded recommendation with evidence, risks, and test guidance.",
                "handoff_context": prompt[:700],
            }
        )
    doc = PlannerRolloutPlan(
        task_synopsis=prompt[:280],
        primary_goal=f"{task_type}: satisfy the user request with verified, low-risk changes.",
        success_criteria=[
            "Answer or implementation directly addresses the requested task.",
            "Important risks, files, and verification steps are surfaced.",
            "Final result is concise enough for the user to act on.",
        ],
        known_constraints=["Preserve existing behavior outside the requested scope."],
        risk_tags=risk_tags,
        required_context=["Relevant files, tests, logs, prior council outputs, and task constraints."],
        selected_experts=selected,
        excluded_experts=[],
        debate_plan={
            "max_rounds": int((council_plan or {}).get("debate_max_rounds") or 1),
            "reroute_conditions": ["Low confidence", "high disagreement", "missing domain expertise"],
            "stop_conditions": ["sufficient confidence", "risks and verification steps covered"],
        },
    )
    return doc


def planner_document_from_payload(
    *,
    prompt: str,
    council_plan: dict[str, Any],
    planner_document: dict[str, Any] | None = None,
) -> tuple[PlannerRolloutPlan, PlannerPlanValidation]:
    if isinstance(planner_document, dict) and planner_document:
        plan = PlannerRolloutPlan.from_dict(planner_document)
    else:
        plan = derive_planner_document(prompt=prompt, council_plan=council_plan)
    return plan, validate_planner_document(plan.to_dict())


def validate_planner_document(payload: dict[str, Any]) -> PlannerPlanValidation:
    diagnostics: list[str] = []
    present = 0
    for field_name in REQUIRED_PLAN_FIELDS:
        value = payload.get(field_name)
        has_value = bool(value) if not isinstance(value, list) else len(value) > 0
        if has_value:
            present += 1
        else:
            diagnostics.append(f"missing_or_empty:{field_name}")

    selected = payload.get("selected_experts")
    if not isinstance(selected, list) or not selected:
        diagnostics.append("selected_experts_empty")
    else:
        for idx, row in enumerate(selected):
            if not isinstance(row, dict):
                diagnostics.append(f"selected_experts[{idx}]:not_object")
                continue
            for field_name in REQUIRED_SELECTED_EXPERT_FIELDS:
                if not _clean_str(row.get(field_name)):
                    diagnostics.append(f"selected_experts[{idx}]:missing:{field_name}")

    debate = payload.get("debate_plan") if isinstance(payload.get("debate_plan"), dict) else {}
    if not _as_str_list(debate.get("reroute_conditions")):
        diagnostics.append("debate_plan:missing:reroute_conditions")
    if not _as_str_list(debate.get("stop_conditions")):
        diagnostics.append("debate_plan:missing:stop_conditions")
    if int(debate.get("max_rounds") or 0) < 1:
        diagnostics.append("debate_plan:invalid:max_rounds")

    completeness = present / len(REQUIRED_PLAN_FIELDS)
    if selected:
        required_total = len(selected) * len(REQUIRED_SELECTED_EXPERT_FIELDS)
        missing_selected = len([d for d in diagnostics if ":missing:" in d and d.startswith("selected_experts[")])
        completeness = (completeness + max(0.0, 1.0 - (missing_selected / max(1, required_total)))) / 2.0
    return PlannerPlanValidation(valid=not diagnostics, diagnostics=diagnostics, completeness=completeness)
