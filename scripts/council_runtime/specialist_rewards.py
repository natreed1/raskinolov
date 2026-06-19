"""Deterministic specialist interaction rewards for council rollouts."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .planner_plan import clamp01, infer_risk_tags
from .rollouts import CouncilRollout

SPECIALIST_REWARD_SCHEMA = "specialist_interaction_reward_v1"
SPECIALIST_REWARD_SUMMARY_SCHEMA = "specialist_interaction_reward_summary_v1"

SPECIALIST_POSITIVE_WEIGHTS = {
    "assigned_role_fit": 0.14,
    "evidence_quality": 0.18,
    "unique_contribution": 0.14,
    "validated_contribution_influence": 0.14,
    "confidence_calibration": 0.14,
    "collaboration_quality": 0.12,
    "task_outcome": 0.14,
}

SPECIALIST_PENALTY_WEIGHTS = {
    "redundancy_penalty": 0.10,
    "misleading_penalty": 0.18,
    "off_role_penalty": 0.08,
    "bad_influence_penalty": 0.14,
}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _tokens(text: str) -> set[str]:
    cleaned = []
    for ch in str(text or "").lower():
        cleaned.append(ch if ch.isalnum() or ch == "_" else " ")
    return {token for token in "".join(cleaned).split() if len(token) >= 4}


def _overlap(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(1, min(len(left), len(right)))


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(term).lower() in lowered for term in terms)


def _planner_document(rollout: CouncilRollout) -> dict[str, Any]:
    if rollout.planner_trace is None:
        return {}
    return dict((rollout.planner_trace.payload or {}).get("planner_document") or {})


def _assignment_for(trace_payload: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    base_id = _as_text(trace_payload.get("base_expert_id")).lower()
    participant_id = _as_text(trace_payload.get("participant_id")).lower()
    for row in _as_list(document.get("selected_experts")):
        if not isinstance(row, dict):
            continue
        expert_id = _as_text(row.get("expert_id")).lower()
        row_participant = _as_text(row.get("participant_id")).lower()
        if expert_id and expert_id == base_id:
            return row
        if row_participant and row_participant == participant_id:
            return row
    return {}


def _final_outcome(rollout: CouncilRollout) -> float:
    if rollout.final_grader_trace is None:
        return 0.0
    metrics = rollout.final_grader_trace.metrics
    if "final_outcome" in metrics:
        return clamp01(metrics.get("final_outcome"), 0.0)
    confidence = clamp01(rollout.final_grader_trace.payload.get("confidence"), 0.5)
    disagreement = clamp01(rollout.final_grader_trace.payload.get("disagreement"), 0.5)
    return clamp01(confidence * (1.0 - disagreement))


@dataclass(frozen=True)
class SpecialistRewardResult:
    season_id: str
    task_id: str
    rollout_id: str
    archetype_id: str
    organism_id: str
    participant_id: str
    base_expert_id: str
    reward: float
    positive_score: float
    penalty_score: float
    components: dict[str, float]
    diagnostics: list[str] = field(default_factory=list)
    specialist_trace_id: str = ""
    training_prompt: str = ""
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPECIALIST_REWARD_SCHEMA,
            "season_id": self.season_id,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "archetype_id": self.archetype_id,
            "organism_id": self.organism_id,
            "participant_id": self.participant_id,
            "base_expert_id": self.base_expert_id,
            "specialist_trace_id": self.specialist_trace_id,
            "reward": round(clamp01(self.reward), 6),
            "positive_score": round(clamp01(self.positive_score), 6),
            "penalty_score": round(clamp01(self.penalty_score), 6),
            "components": {k: round(clamp01(v), 6) for k, v in sorted(self.components.items())},
            "diagnostics": list(self.diagnostics),
            "training_prompt": self.training_prompt,
            "output": self.output,
            "weights": {
                "positive": dict(SPECIALIST_POSITIVE_WEIGHTS),
                "penalties": dict(SPECIALIST_PENALTY_WEIGHTS),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpecialistRewardResult":
        if payload.get("schema_version") != SPECIALIST_REWARD_SCHEMA:
            raise ValueError(f"Unsupported specialist reward schema: {payload.get('schema_version')!r}")
        return cls(
            season_id=str(payload.get("season_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            rollout_id=str(payload.get("rollout_id") or ""),
            archetype_id=str(payload.get("archetype_id") or ""),
            organism_id=str(payload.get("organism_id") or ""),
            participant_id=str(payload.get("participant_id") or ""),
            base_expert_id=str(payload.get("base_expert_id") or ""),
            specialist_trace_id=str(payload.get("specialist_trace_id") or ""),
            reward=float(payload.get("reward") or 0.0),
            positive_score=float(payload.get("positive_score") or 0.0),
            penalty_score=float(payload.get("penalty_score") or 0.0),
            components={str(k): float(v) for k, v in dict(payload.get("components") or {}).items()},
            diagnostics=[str(x) for x in list(payload.get("diagnostics") or [])],
            training_prompt=str(payload.get("training_prompt") or ""),
            output=str(payload.get("output") or ""),
        )


@dataclass(frozen=True)
class SpecialistRewardSummary:
    row_count: int
    mean_reward: float
    by_archetype: dict[str, dict[str, float]]
    diagnostics: dict[str, int]
    output_path: str
    reward_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPECIALIST_REWARD_SUMMARY_SCHEMA,
            "row_count": int(self.row_count),
            "mean_reward": round(float(self.mean_reward), 6),
            "by_archetype": self.by_archetype,
            "diagnostics": {k: int(v) for k, v in sorted(self.diagnostics.items())},
            "output_path": self.output_path,
            "reward_path": self.reward_path,
        }


class SpecialistRewardFunction:
    def score_rollout(self, rollout: CouncilRollout) -> list[SpecialistRewardResult]:
        document = _planner_document(rollout)
        final_text = ""
        if rollout.final_grader_trace is not None:
            final_text = _as_text(rollout.final_grader_trace.payload.get("final_text"))
        final_outcome = _final_outcome(rollout)
        peer_texts = {
            trace.trace_id: _as_text(trace.payload.get("text"))
            for trace in rollout.specialist_traces
        }
        results = []
        for trace in rollout.specialist_traces:
            payload = trace.payload
            assignment = _assignment_for(payload, document)
            text = _as_text(payload.get("text"))
            base_id = _as_text(payload.get("base_expert_id") or payload.get("participant_id"))
            participant_id = _as_text(payload.get("participant_id") or base_id)
            assigned_role_fit = self._assigned_role_fit(text, payload, assignment, document)
            evidence_quality = clamp01(trace.metrics.get("evidence_quality"), 0.0)
            unique_contribution = self._unique_contribution(trace.trace_id, text, peer_texts)
            contribution_quality = clamp01(
                0.45 * assigned_role_fit
                + 0.35 * evidence_quality
                + 0.20 * clamp01(trace.metrics.get("contribution_credit"), 0.0)
            )
            final_overlap = _overlap(text, final_text)
            validated_influence = (
                clamp01(final_overlap * final_outcome * contribution_quality)
                if final_outcome >= 0.5
                else 0.0
            )
            domain_match = self._domain_match(text, payload, assignment, document)
            confidence = clamp01(payload.get("confidence"), clamp01(trace.metrics.get("confidence_calibration"), 0.5))
            confidence_calibration = self._confidence_calibration(confidence, assigned_role_fit, evidence_quality, domain_match, text)
            collaboration = self._collaboration_quality(text)
            task_outcome = final_outcome
            redundancy = 1.0 - unique_contribution
            misleading = self._misleading_penalty(text, confidence, final_outcome, rollout)
            off_role = clamp01(1.0 - assigned_role_fit)
            bad_influence = clamp01(final_overlap * (1.0 - final_outcome) * confidence)
            components = {
                "assigned_role_fit": assigned_role_fit,
                "evidence_quality": evidence_quality,
                "unique_contribution": unique_contribution,
                "validated_contribution_influence": validated_influence,
                "confidence_calibration": confidence_calibration,
                "collaboration_quality": collaboration,
                "task_outcome": task_outcome,
                "redundancy_penalty": redundancy,
                "misleading_penalty": misleading,
                "off_role_penalty": off_role,
                "bad_influence_penalty": bad_influence,
            }
            positive_total = sum(SPECIALIST_POSITIVE_WEIGHTS.values())
            positive = sum(SPECIALIST_POSITIVE_WEIGHTS[k] * components[k] for k in SPECIALIST_POSITIVE_WEIGHTS) / positive_total
            penalty = sum(SPECIALIST_PENALTY_WEIGHTS[k] * components[k] for k in SPECIALIST_PENALTY_WEIGHTS)
            diagnostics = self._diagnostics(components, assignment)
            results.append(
                SpecialistRewardResult(
                    season_id=rollout.season_id,
                    task_id=rollout.task_id,
                    rollout_id=rollout.rollout_id,
                    archetype_id=f"{base_id}_eq" if base_id else "specialist_eq",
                    organism_id=str(trace.organism_id or ""),
                    participant_id=participant_id,
                    base_expert_id=base_id,
                    specialist_trace_id=trace.trace_id,
                    reward=clamp01(positive - penalty),
                    positive_score=positive,
                    penalty_score=penalty,
                    components=components,
                    diagnostics=diagnostics,
                    training_prompt=self._training_prompt(assignment, document),
                    output=text,
                )
            )
        return results

    def _assigned_role_fit(self, text: str, payload: dict[str, Any], assignment: dict[str, Any], document: dict[str, Any]) -> float:
        if not assignment:
            return 0.15 if self._domain_match(text, payload, assignment, document) > 0.4 else 0.0
        assigned_question = _as_text(assignment.get("assigned_question"))
        expected = _as_text(assignment.get("expected_contribution"))
        handoff = _as_text(assignment.get("handoff_context"))
        role = _as_text(assignment.get("role"))
        role_terms = f"{role} {_as_text(assignment.get('why_selected'))}"
        domain_match = self._domain_match(text, payload, assignment, document)
        out_of_domain_flag = 1.0 if _contains_any(text, ("outside my domain", "not my domain", "defer", "reroute", "another expert")) else 0.0
        return clamp01(
            0.30 * _overlap(text, assigned_question)
            + 0.25 * _overlap(text, expected)
            + 0.20 * _overlap(text, handoff)
            + 0.15 * max(domain_match, _overlap(text, role_terms))
            + 0.10 * out_of_domain_flag
        )

    def _domain_match(self, text: str, payload: dict[str, Any], assignment: dict[str, Any], document: dict[str, Any]) -> float:
        base = _as_text(payload.get("base_expert_id"))
        risk_text = " ".join(str(x) for x in _as_list(document.get("risk_tags")))
        assignment_text = json.dumps(assignment, sort_keys=True) if assignment else ""
        inferred = infer_risk_tags(f"{base} {risk_text} {assignment_text}")
        if not inferred:
            return 0.0
        return max(_overlap(text, " ".join(inferred)), _overlap(text, f"{base} {assignment_text} {risk_text}"))

    def _unique_contribution(self, trace_id: str, text: str, peer_texts: dict[str, str]) -> float:
        overlaps = []
        for other_id, other in peer_texts.items():
            if other_id == trace_id:
                continue
            overlaps.append(_overlap(text, other))
        if not overlaps:
            return 1.0
        return clamp01(1.0 - max(overlaps))

    def _confidence_calibration(self, confidence: float, role_fit: float, evidence: float, domain_match: float, text: str) -> float:
        uncertainty = 1.0 if _contains_any(text, ("uncertain", "missing", "reroute", "defer", "need context")) else 0.0
        expected = clamp01(0.20 + 0.35 * role_fit + 0.25 * evidence + 0.20 * domain_match - 0.20 * uncertainty)
        return clamp01(1.0 - abs(confidence - expected))

    def _collaboration_quality(self, text: str) -> float:
        score = 0.20
        if _contains_any(text, ("risk", "verify", "test", "evidence", "because")):
            score += 0.30
        if _contains_any(text, ("uncertain", "missing", "reroute", "another expert", "defer")):
            score += 0.25
        if _contains_any(text, ("assumption", "contradict", "weak", "edge case")):
            score += 0.15
        if len(_tokens(text)) >= 8:
            score += 0.10
        return clamp01(score)

    def _misleading_penalty(self, text: str, confidence: float, final_outcome: float, rollout: CouncilRollout) -> float:
        penalty = 0.0
        if confidence > 0.7 and final_outcome < 0.4:
            penalty += 0.35
        if _contains_any(text, ("tests pass", "verified", "no risk", "safe")) and final_outcome < 0.5:
            penalty += 0.35
        if rollout.final_grader_trace is not None and rollout.final_grader_trace.payload.get("escalation_recommended") and _contains_any(text, ("no risk", "safe")):
            penalty += 0.30
        return clamp01(penalty)

    def _training_prompt(self, assignment: dict[str, Any], document: dict[str, Any]) -> str:
        if assignment:
            return (
                f"Role: {_as_text(assignment.get('role'))}\n"
                f"Question: {_as_text(assignment.get('assigned_question'))}\n"
                f"Expected contribution: {_as_text(assignment.get('expected_contribution'))}\n"
                f"Context: {_as_text(assignment.get('handoff_context'))}"
            ).strip()
        return json.dumps(document, sort_keys=True)[:1200]

    def _diagnostics(self, components: dict[str, float], assignment: dict[str, Any]) -> list[str]:
        diagnostics = []
        if not assignment:
            diagnostics.append("missing_planner_assignment")
        for key, value in components.items():
            if key.endswith("_penalty"):
                if value >= 0.5:
                    diagnostics.append(f"high_penalty:{key}")
            elif value <= 0.15:
                diagnostics.append(f"low_component:{key}")
        return sorted(set(diagnostics))


def grade_specialist_rollouts(rollouts: Iterable[CouncilRollout]) -> list[SpecialistRewardResult]:
    scorer = SpecialistRewardFunction()
    out: list[SpecialistRewardResult] = []
    for rollout in rollouts:
        out.extend(scorer.score_rollout(rollout))
    return out


def write_specialist_rewards(path: Path, rows: Iterable[SpecialistRewardResult]) -> None:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def summarize_specialist_rewards(
    *,
    rows: list[SpecialistRewardResult],
    summary_path: Path,
    reward_path: Path,
) -> SpecialistRewardSummary:
    rewards = [row.reward for row in rows]
    by_arch: dict[str, dict[str, float]] = {}
    for archetype in sorted({row.archetype_id for row in rows}):
        bucket = [row.reward for row in rows if row.archetype_id == archetype]
        by_arch[archetype] = {"count": float(len(bucket)), "mean_reward": round(sum(bucket) / max(1, len(bucket)), 6)}
    diagnostics = Counter()
    for row in rows:
        diagnostics.update(row.diagnostics)
    summary = SpecialistRewardSummary(
        row_count=len(rows),
        mean_reward=sum(rewards) / max(1, len(rewards)),
        by_archetype=by_arch,
        diagnostics=dict(diagnostics),
        output_path=str(summary_path.expanduser().resolve()),
        reward_path=str(reward_path.expanduser().resolve()),
    )
    summary_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    summary_path.expanduser().resolve().write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def read_specialist_reward_jsonl(path: Path) -> list[SpecialistRewardResult]:
    rows: list[SpecialistRewardResult] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(SpecialistRewardResult.from_dict(json.loads(stripped)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid specialist reward row {path}:{line_no}: {exc}") from exc
    return rows
