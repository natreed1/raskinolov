"""Multi-agent subtask planning and deterministic merge helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class AdapterSubtask:
    adapter_id: str
    role: str
    prompt: str
    priority: int = 1


_FOCUS_HINTS = {
    "loading_screen": "Focus on loading/splash visual framing and readability polish.",
    "hud_status": "Focus on HUD/status hierarchy, scanability, and compact tactical readability.",
    "economy_tooltip": "Focus on economy tooltip clarity, income/upkeep/net explanations, and wording precision.",
    "combat_risk": "Focus on combat preview risk communication and enemy-vs-player comparison clarity.",
    "save_load_api_guard": "Focus on save/load API guardrails, auth checks, and serialization safety.",
    "ai_planning_explanation": "Focus on AI rationale, planning tradeoffs, and decision explanation quality.",
    "documentation": "Focus on documentation accuracy, canonical paths, and process consistency.",
    "general_fallback": "Focus on general correctness, structure, and concise implementation details.",
}


def _focus_for_adapter(adapter_id: str) -> str:
    return _FOCUS_HINTS.get(adapter_id, _FOCUS_HINTS["general_fallback"])


def build_multi_agent_subtasks(
    *,
    prompt: str,
    primary_adapter_id: str,
    secondary_adapter_id: str | None,
    secondary_confidence: float,
    coarse_bucket: str,
    secondary_min_confidence: float = 0.2,
) -> List[AdapterSubtask]:
    """Build executable subtasks for a single prompt.

    V1 policy:
    - Always run primary adapter task.
    - Run secondary only when non-empty, distinct from primary, and confidence threshold is met.
    """
    trimmed = prompt.strip()
    if not trimmed:
        return []
    primary = primary_adapter_id or "general_fallback"
    subtasks: List[AdapterSubtask] = [
        AdapterSubtask(
            adapter_id=primary,
            role="primary",
            priority=1,
            prompt=(
                f"{trimmed}\n\n"
                "Subtask focus (primary specialist): "
                f"{_focus_for_adapter(primary)}"
            ),
        )
    ]

    secondary = (secondary_adapter_id or "").strip()
    if (
        secondary
        and secondary != primary
        and secondary != "general_fallback"
        and float(secondary_confidence) >= float(secondary_min_confidence)
    ):
        subtasks.append(
            AdapterSubtask(
                adapter_id=secondary,
                role="secondary",
                priority=2,
                prompt=(
                    f"{trimmed}\n\n"
                    f"Coarse bucket context: {coarse_bucket or 'unclassified'}.\n"
                    "Subtask focus (secondary specialist): "
                    f"{_focus_for_adapter(secondary)}"
                ),
            )
        )
    return subtasks


def merge_multi_agent_outputs(
    *,
    user_prompt: str,
    outputs: Sequence[dict],
) -> str:
    """Deterministically merge agent outputs into one final answer payload."""
    if not outputs:
        return ""
    ordered = sorted(outputs, key=lambda row: int(row.get("priority") or 999))
    lines: List[str] = [
        "## Combined Specialist Draft",
        "",
        f"Prompt: {user_prompt.strip()}",
        "",
    ]
    for row in ordered:
        role = str(row.get("role") or "agent")
        adapter = str(row.get("adapter_id") or "general_fallback")
        text = str(row.get("text") or "").strip()
        lines += [
            f"### {role.title()} agent (`{adapter}`)",
            text or "_No output_",
            "",
        ]
    if len(ordered) > 1:
        lines += [
            "### Integration notes",
            "- Preserve primary solution structure as the source of truth.",
            "- Apply secondary refinements where they do not conflict.",
            "- Keep final response coherent and non-duplicative.",
            "",
        ]
    return "\n".join(lines).rstrip()


def summarize_subtask_ids(subtasks: Iterable[AdapterSubtask]) -> list[str]:
    return [f"{task.role}:{task.adapter_id}" for task in subtasks]
