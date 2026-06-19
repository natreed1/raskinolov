"""Structured council trace contracts and JSONL emission helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .planner_plan import planner_document_from_payload

TRACE_ENVELOPE_SCHEMA = "council_trace_envelope_v1"
PLANNER_TRACE_SCHEMA = "planner_trace_v1"
GRADER_TRACE_SCHEMA = "grader_trace_v1"
CONTEXT_COMPRESSION_TRACE_SCHEMA = "context_compression_trace_v1"
SPECIALIST_EQ_TRACE_SCHEMA = "specialist_eq_trace_v1"

TRACE_SCHEMAS: dict[str, dict[str, Any]] = {
    PLANNER_TRACE_SCHEMA: {
        "purpose": "Planner routing document: task synopsis, goals, risks, expert choices, instructions, and debate plan.",
        "required_payload": ["planner_document", "planner_document_validation", "selected_participants", "debate_max_rounds"],
        "reward_metrics": [
            "task_synopsis_quality",
            "goal_alignment",
            "success_criteria_quality",
            "risk_identification",
            "expert_selection_justification",
            "instruction_quality",
            "context_request_quality",
            "reroute_logic",
            "expert_usefulness",
            "final_outcome",
            "context_efficiency",
            "irrelevant_expert_penalty",
            "unnecessary_complexity_penalty",
        ],
    },
    GRADER_TRACE_SCHEMA: {
        "purpose": "Grader adjudication, contribution credit, confidence, escalation, and reroute decision.",
        "required_payload": ["winner_ids", "confidence", "disagreement", "escalation_recommended"],
        "reward_metrics": ["final_outcome", "contribution_credit", "confidence_calibration", "routing_or_reroute_quality"],
    },
    CONTEXT_COMPRESSION_TRACE_SCHEMA: {
        "purpose": "Compressor input/output accounting and preservation of task constraints, paths, decisions, and debate state.",
        "required_payload": ["input_token_estimate", "output_token_estimate", "preserved_items"],
        "reward_metrics": ["context_efficiency", "evidence_quality"],
    },
    SPECIALIST_EQ_TRACE_SCHEMA: {
        "purpose": "Specialist collaboration behavior: contribution quality, evidence use, confidence, and peer interaction.",
        "required_payload": ["participant_id", "base_expert_id", "text"],
        "reward_metrics": ["contribution_credit", "evidence_quality", "confidence_calibration", "cost_efficiency"],
    },
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    # Cheap deterministic estimate good enough for routing/context-efficiency scoring.
    return max(1, int(round(len(str(text or "")) / 4.0)))


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float(default)
    return max(0.0, min(1.0, raw))


def _metric_defaults_from_text(text: str, confidence: float = 0.5) -> dict[str, float]:
    lowered = str(text or "").lower()
    evidence_terms = ("test", "verify", "file", "path", "because", "risk", "line", "trace", "metric")
    evidence_hits = sum(1 for term in evidence_terms if term in lowered)
    token_estimate = estimate_tokens(text)
    return {
        "contribution_credit": min(1.0, 0.25 + token_estimate / 500.0),
        "evidence_quality": min(1.0, evidence_hits / 5.0),
        "confidence_calibration": 1.0 - abs(_clamp01(confidence, 0.5) - 0.72),
        "cost_efficiency": max(0.05, min(1.0, 220.0 / max(60.0, float(token_estimate)))),
    }


@dataclass(frozen=True)
class CouncilTraceRecord:
    trace_schema: str
    event_type: str
    archetype_id: str
    payload: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    task_id: str = ""
    season_id: str = ""
    organism_id: str = ""
    round_idx: int = 0
    prompt_hash: str = ""
    parent_trace_ids: list[str] = field(default_factory=list)
    created_at_utc: str = ""

    @property
    def trace_id(self) -> str:
        return stable_hash(
            {
                "trace_schema": self.trace_schema,
                "event_type": self.event_type,
                "archetype_id": self.archetype_id,
                "task_id": self.task_id,
                "season_id": self.season_id,
                "organism_id": self.organism_id,
                "round_idx": self.round_idx,
                "prompt_hash": self.prompt_hash,
                "payload": self.payload,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_ENVELOPE_SCHEMA,
            "trace_id": self.trace_id,
            "trace_schema": self.trace_schema,
            "event_type": self.event_type,
            "archetype_id": self.archetype_id,
            "organism_id": self.organism_id,
            "task_id": self.task_id,
            "season_id": self.season_id,
            "round_idx": int(self.round_idx),
            "prompt_hash": self.prompt_hash,
            "parent_trace_ids": list(self.parent_trace_ids),
            "created_at_utc": self.created_at_utc or utc_iso(),
            "payload": dict(self.payload),
            "metrics": {k: _clamp01(v) for k, v in sorted(self.metrics.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CouncilTraceRecord":
        if payload.get("schema_version") != TRACE_ENVELOPE_SCHEMA:
            raise ValueError(f"Unsupported trace envelope schema: {payload.get('schema_version')!r}")
        return cls(
            trace_schema=str(payload.get("trace_schema") or ""),
            event_type=str(payload.get("event_type") or ""),
            archetype_id=str(payload.get("archetype_id") or ""),
            organism_id=str(payload.get("organism_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            season_id=str(payload.get("season_id") or ""),
            round_idx=int(payload.get("round_idx") or 0),
            prompt_hash=str(payload.get("prompt_hash") or ""),
            parent_trace_ids=[str(x) for x in list(payload.get("parent_trace_ids") or [])],
            created_at_utc=str(payload.get("created_at_utc") or ""),
            payload=dict(payload.get("payload") or {}),
            metrics={str(k): _clamp01(v) for k, v in dict(payload.get("metrics") or {}).items()},
        )


TraceSink = Callable[[CouncilTraceRecord], None]


class JsonlTraceWriter:
    """Append-only JSONL trace sink."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: CouncilTraceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def build_planner_trace(
    *,
    prompt: str,
    council_plan: dict[str, Any],
    trace_context: dict[str, Any] | None = None,
    planner_document: dict[str, Any] | None = None,
) -> CouncilTraceRecord:
    context = dict(trace_context or {})
    participants = list((council_plan or {}).get("participants") or [])
    plan_doc, validation = planner_document_from_payload(
        prompt=prompt,
        council_plan=council_plan,
        planner_document=planner_document or context.get("planner_document"),
    )
    payload = {
        "planner_document": plan_doc.to_dict(),
        "planner_document_validation": validation.to_dict(),
        "selected_participants": [
            {
                "participant_id": str(row.get("participant_id") or ""),
                "base_expert_id": str(row.get("base_expert_id") or row.get("participant_id") or ""),
                "participant_type": str(row.get("participant_type") or ""),
                "role": str(row.get("role") or ""),
                "strategy": str(row.get("strategy") or ""),
                "variant": str(row.get("variant") or "baseline"),
            }
            for row in participants
        ],
        "debate_max_rounds": int((council_plan or {}).get("debate_max_rounds") or 1),
        "low_confidence_threshold": float((council_plan or {}).get("low_confidence_threshold") or 0.0),
        "disagreement_threshold": float((council_plan or {}).get("disagreement_threshold") or 0.0),
        "escalation_rule": str((council_plan or {}).get("escalation_rule") or ""),
    }
    participant_count = max(1, len(participants))
    metrics = {
        "planner_document_completeness": validation.completeness,
        "routing_or_reroute_quality": _clamp01(context.get("routing_quality"), 0.5),
        "context_efficiency": max(0.05, min(1.0, 4.0 / float(participant_count + payload["debate_max_rounds"]))),
        "cost_efficiency": max(0.05, min(1.0, 3.0 / float(participant_count))),
    }
    return CouncilTraceRecord(
        trace_schema=PLANNER_TRACE_SCHEMA,
        event_type="planner_routing",
        archetype_id="planner",
        organism_id=str(context.get("planner_organism_id") or ""),
        task_id=str(context.get("task_id") or ""),
        season_id=str(context.get("season_id") or ""),
        prompt_hash=text_hash(prompt),
        payload=payload,
        metrics=metrics,
    )


def build_specialist_trace(
    *,
    prompt: str,
    participant_output: dict[str, Any],
    round_idx: int,
    trace_context: dict[str, Any] | None = None,
) -> CouncilTraceRecord:
    context = dict(trace_context or {})
    participant_id = str(participant_output.get("participant_id") or "")
    base_expert_id = str(participant_output.get("base_expert_id") or participant_id)
    confidence = _clamp01(participant_output.get("confidence"), 0.5)
    text = str(participant_output.get("text") or "")
    metrics = _metric_defaults_from_text(text, confidence)
    for key in ("contribution_credit", "evidence_quality", "confidence_calibration", "cost_efficiency"):
        if key in participant_output:
            metrics[key] = _clamp01(participant_output[key], metrics[key])
    return CouncilTraceRecord(
        trace_schema=SPECIALIST_EQ_TRACE_SCHEMA,
        event_type="specialist_generation",
        archetype_id=f"{base_expert_id}_eq" if base_expert_id else "specialist_eq",
        organism_id=str(participant_output.get("organism_id") or context.get("organism_id_by_participant", {}).get(participant_id, "")),
        task_id=str(context.get("task_id") or ""),
        season_id=str(context.get("season_id") or ""),
        round_idx=round_idx,
        prompt_hash=text_hash(prompt),
        payload={
            "participant_id": participant_id,
            "base_expert_id": base_expert_id,
            "participant_type": str(participant_output.get("participant_type") or ""),
            "role": str(participant_output.get("role") or ""),
            "strategy": str(participant_output.get("strategy") or ""),
            "variant": str(participant_output.get("variant") or "baseline"),
            "confidence": confidence,
            "token_estimate": estimate_tokens(text),
            "text": text,
        },
        metrics=metrics,
    )


def build_grader_trace(
    *,
    prompt: str,
    adjudication: dict[str, Any],
    round_idx: int,
    participant_outputs: list[dict[str, Any]],
    trace_context: dict[str, Any] | None = None,
    final: bool = False,
) -> CouncilTraceRecord:
    context = dict(trace_context or {})
    confidence = _clamp01(adjudication.get("confidence"), 0.5)
    disagreement = _clamp01(adjudication.get("disagreement"), 0.5)
    winner_ids = [str(x) for x in list(adjudication.get("winner_ids") or [])]
    contribution = 0.0
    if participant_outputs:
        contribution = sum(_clamp01(row.get("task_outcome_score"), 0.5) for row in participant_outputs) / len(participant_outputs)
    metrics = {
        "final_outcome": _clamp01(context.get("final_outcome"), confidence if final else 0.0),
        "contribution_credit": contribution,
        "confidence_calibration": max(0.0, 1.0 - abs(confidence - (1.0 - disagreement))),
        "routing_or_reroute_quality": 1.0 if not bool(adjudication.get("escalation_recommended")) else 0.45,
    }
    return CouncilTraceRecord(
        trace_schema=GRADER_TRACE_SCHEMA,
        event_type="grader_final" if final else "grader_round_preview",
        archetype_id="grader",
        organism_id=str(context.get("grader_organism_id") or ""),
        task_id=str(context.get("task_id") or ""),
        season_id=str(context.get("season_id") or ""),
        round_idx=round_idx,
        prompt_hash=text_hash(prompt),
        payload={
            "winner_ids": winner_ids,
            "confidence": confidence,
            "disagreement": disagreement,
            "escalation_recommended": bool(adjudication.get("escalation_recommended")),
            "participant_count": len(participant_outputs),
            "final": bool(final),
            "final_text": str(adjudication.get("final_text") or "") if final else "",
        },
        metrics=metrics,
    )
