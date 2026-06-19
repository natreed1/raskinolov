"""Deterministic rollout-level reward functions for council policies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .planner_plan import clamp01, estimate_tokens, infer_risk_tags, infer_task_type, validate_planner_document
from .rollouts import CouncilRollout, group_traces_into_rollouts
from .trace_scoring import read_trace_jsonl

GRADED_ROLLOUT_SCHEMA = "council_graded_rollout_v1"
ROLLOUT_REWARD_SUMMARY_SCHEMA = "council_rollout_reward_summary_v1"

PLANNER_POSITIVE_WEIGHTS = {
    "task_synopsis_quality": 0.12,
    "goal_alignment": 0.12,
    "success_criteria_quality": 0.10,
    "risk_identification": 0.10,
    "expert_selection_justification": 0.14,
    "instruction_quality": 0.12,
    "context_request_quality": 0.08,
    "reroute_logic": 0.06,
    "expert_usefulness": 0.12,
    "final_outcome": 0.16,
    "context_efficiency": 0.08,
}

PLANNER_PENALTY_WEIGHTS = {
    "irrelevant_expert_penalty": 0.12,
    "unnecessary_complexity_penalty": 0.08,
}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(str(token).lower() in lowered for token in tokens)


def _token_score(text: str, *, min_tokens: int, max_tokens: int) -> float:
    count = estimate_tokens(text)
    if count < min_tokens:
        return clamp01(count / max(1, min_tokens))
    if count <= max_tokens:
        return 1.0
    return clamp01(max_tokens / max(1, count))


@dataclass(frozen=True)
class PlannerRewardResult:
    rollout_id: str
    task_id: str
    season_id: str
    reward: float
    positive_score: float
    penalty_score: float
    components: dict[str, float]
    diagnostics: list[str] = field(default_factory=list)
    planner_trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRADED_ROLLOUT_SCHEMA,
            "rollout_id": self.rollout_id,
            "task_id": self.task_id,
            "season_id": self.season_id,
            "archetype_id": "planner",
            "planner_trace_id": self.planner_trace_id,
            "reward": round(clamp01(self.reward), 6),
            "positive_score": round(clamp01(self.positive_score), 6),
            "penalty_score": round(clamp01(self.penalty_score), 6),
            "components": {k: round(clamp01(v), 6) for k, v in sorted(self.components.items())},
            "diagnostics": list(self.diagnostics),
            "weights": {
                "positive": dict(PLANNER_POSITIVE_WEIGHTS),
                "penalties": dict(PLANNER_PENALTY_WEIGHTS),
            },
        }


@dataclass(frozen=True)
class RolloutRewardSummary:
    rollout_count: int
    mean_reward: float
    min_reward: float
    max_reward: float
    output_path: str
    graded_rollout_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLLOUT_REWARD_SUMMARY_SCHEMA,
            "rollout_count": int(self.rollout_count),
            "mean_reward": round(float(self.mean_reward), 6),
            "min_reward": round(float(self.min_reward), 6),
            "max_reward": round(float(self.max_reward), 6),
            "output_path": self.output_path,
            "graded_rollout_path": self.graded_rollout_path,
        }


class PlannerRewardFunction:
    """Score the planner's explicit rollout document and downstream result."""

    def score_rollout(self, rollout: CouncilRollout) -> PlannerRewardResult:
        diagnostics: list[str] = []
        if rollout.planner_trace is None:
            components = {key: 0.0 for key in [*PLANNER_POSITIVE_WEIGHTS, *PLANNER_PENALTY_WEIGHTS]}
            diagnostics.append("missing_planner_trace")
            return PlannerRewardResult(
                rollout_id=rollout.rollout_id,
                task_id=rollout.task_id,
                season_id=rollout.season_id,
                reward=0.0,
                positive_score=0.0,
                penalty_score=1.0,
                components=components,
                diagnostics=diagnostics,
            )

        planner_payload = rollout.planner_trace.payload
        document = dict(planner_payload.get("planner_document") or {})
        validation_payload = planner_payload.get("planner_document_validation")
        validation = validate_planner_document(document)
        if isinstance(validation_payload, dict):
            diagnostics.extend(str(x) for x in list(validation_payload.get("diagnostics") or []))
        diagnostics.extend(validation.diagnostics)

        selected_experts = [dict(row) for row in _as_list(document.get("selected_experts")) if isinstance(row, dict)]
        selected_ids = [_as_text(row.get("expert_id") or row.get("base_expert_id") or row.get("participant_id")) for row in selected_experts]
        selected_ids = [sid for sid in selected_ids if sid]
        risk_tags = [_as_text(tag) for tag in _as_list(document.get("risk_tags")) if _as_text(tag)]

        components = {
            "task_synopsis_quality": self._task_synopsis_quality(document, rollout),
            "goal_alignment": self._goal_alignment(document, rollout),
            "success_criteria_quality": self._success_criteria_quality(document),
            "risk_identification": self._risk_identification(document, rollout),
            "expert_selection_justification": self._expert_selection_justification(selected_experts),
            "instruction_quality": self._instruction_quality(selected_experts),
            "context_request_quality": self._context_request_quality(document),
            "reroute_logic": self._reroute_logic(document),
            "expert_usefulness": self._expert_usefulness(rollout, selected_ids),
            "final_outcome": self._final_outcome(rollout),
            "context_efficiency": self._context_efficiency(rollout, selected_ids, risk_tags),
            "irrelevant_expert_penalty": self._irrelevant_expert_penalty(rollout, selected_ids),
            "unnecessary_complexity_penalty": self._unnecessary_complexity_penalty(document, selected_ids, risk_tags),
        }

        for key, value in components.items():
            if value <= 0.15:
                diagnostics.append(f"low_component:{key}")
        positive = sum(PLANNER_POSITIVE_WEIGHTS[key] * components[key] for key in PLANNER_POSITIVE_WEIGHTS)
        positive_total = sum(PLANNER_POSITIVE_WEIGHTS.values())
        positive_score = positive / max(1e-9, positive_total)
        penalty_score = sum(PLANNER_PENALTY_WEIGHTS[key] * components[key] for key in PLANNER_PENALTY_WEIGHTS)
        reward = clamp01(positive_score - penalty_score)
        return PlannerRewardResult(
            rollout_id=rollout.rollout_id,
            task_id=rollout.task_id,
            season_id=rollout.season_id,
            planner_trace_id=rollout.planner_trace.trace_id,
            reward=reward,
            positive_score=positive_score,
            penalty_score=penalty_score,
            components=components,
            diagnostics=sorted(set(diagnostics)),
        )

    def _task_synopsis_quality(self, document: dict[str, Any], rollout: CouncilRollout) -> float:
        synopsis = _as_text(document.get("task_synopsis"))
        if not synopsis:
            return 0.0
        score = 0.55 * _token_score(synopsis, min_tokens=6, max_tokens=90)
        final_text = ""
        if rollout.final_grader_trace is not None:
            final_text = _as_text((rollout.final_grader_trace.payload or {}).get("final_text"))
        if _contains_any(synopsis, ("fix", "implement", "review", "test", "verify", "risk", "debug", "plan")):
            score += 0.25
        if final_text and any(token in final_text.lower() for token in synopsis.lower().split()[:8]):
            score += 0.20
        else:
            score += 0.10
        return clamp01(score)

    def _goal_alignment(self, document: dict[str, Any], rollout: CouncilRollout) -> float:
        goal = _as_text(document.get("primary_goal"))
        if not goal:
            return 0.0
        task_type = infer_task_type(f"{document.get('task_synopsis', '')} {goal}")
        score = 0.45 * _token_score(goal, min_tokens=5, max_tokens=80)
        if task_type in goal.lower():
            score += 0.25
        if _contains_any(goal, ("user request", "requested", "verified", "low-risk", "directly")):
            score += 0.30
        return clamp01(score)

    def _success_criteria_quality(self, document: dict[str, Any]) -> float:
        rows = [_as_text(row) for row in _as_list(document.get("success_criteria")) if _as_text(row)]
        if not rows:
            return 0.0
        concrete = sum(1 for row in rows if _contains_any(row, ("test", "verify", "file", "risk", "final", "behavior", "pass", "run")))
        return clamp01(0.35 + 0.15 * min(len(rows), 3) + 0.20 * min(concrete, 2))

    def _risk_identification(self, document: dict[str, Any], rollout: CouncilRollout) -> float:
        tags = {tag.lower() for tag in _as_list(document.get("risk_tags")) if _as_text(tag)}
        text = json.dumps(document, sort_keys=True)
        for trace in rollout.specialist_traces:
            text += " " + json.dumps(trace.payload, sort_keys=True)
        expected = set(infer_risk_tags(text))
        if not expected:
            return 0.5 if tags else 0.0
        overlap = len(tags.intersection(expected)) / len(expected)
        return clamp01(0.25 if tags else 0.0) if overlap == 0 else clamp01(overlap)

    def _expert_selection_justification(self, selected_experts: list[dict[str, Any]]) -> float:
        if not selected_experts:
            return 0.0
        required = ("expert_id", "role", "why_selected", "assigned_question", "expected_contribution", "handoff_context")
        scores = []
        for row in selected_experts:
            filled = sum(1 for key in required if _as_text(row.get(key)))
            scores.append(filled / len(required))
        return clamp01(sum(scores) / len(scores))

    def _instruction_quality(self, selected_experts: list[dict[str, Any]]) -> float:
        if not selected_experts:
            return 0.0
        scores = []
        for row in selected_experts:
            instruction = " ".join(
                _as_text(row.get(key))
                for key in ("assigned_question", "expected_contribution", "handoff_context", "why_selected")
            )
            score = 0.35 * _token_score(instruction, min_tokens=10, max_tokens=160)
            if _contains_any(instruction, ("verify", "test", "risk", "file", "evidence", "specific", "bounded")):
                score += 0.35
            if _as_text(row.get("role")) and _as_text(row.get("expert_id")):
                score += 0.30
            scores.append(clamp01(score))
        return clamp01(sum(scores) / len(scores))

    def _context_request_quality(self, document: dict[str, Any]) -> float:
        rows = [_as_text(row) for row in _as_list(document.get("required_context")) if _as_text(row)]
        if not rows:
            return 0.0
        joined = " ".join(rows)
        score = 0.30 + 0.15 * min(len(rows), 3)
        if _contains_any(joined, ("file", "test", "log", "trace", "constraint", "prior", "docs", "diff")):
            score += 0.30
        if _contains_any(joined, ("everything", "entire repo", "all files", "all context")):
            score -= 0.35
        return clamp01(score)

    def _reroute_logic(self, document: dict[str, Any]) -> float:
        debate = document.get("debate_plan") if isinstance(document.get("debate_plan"), dict) else {}
        reroute = [_as_text(row) for row in _as_list(debate.get("reroute_conditions")) if _as_text(row)]
        stop = [_as_text(row) for row in _as_list(debate.get("stop_conditions")) if _as_text(row)]
        if not reroute and not stop:
            return 0.0
        joined = " ".join(reroute + stop)
        score = 0.25 + 0.20 * min(len(reroute), 2) + 0.15 * min(len(stop), 2)
        if _contains_any(joined, ("confidence", "disagreement", "missing", "failed", "risk", "evidence", "uncertain")):
            score += 0.25
        return clamp01(score)

    def _expert_usefulness(self, rollout: CouncilRollout, selected_ids: list[str]) -> float:
        if not rollout.specialist_traces:
            return 0.0
        selected = {sid.lower() for sid in selected_ids}
        final_winners = set()
        if rollout.final_grader_trace is not None:
            final_winners = {str(x).lower() for x in _as_list(rollout.final_grader_trace.payload.get("winner_ids"))}
        scores = []
        for trace in rollout.specialist_traces:
            base_id = _as_text(trace.payload.get("base_expert_id")).lower()
            participant_id = _as_text(trace.payload.get("participant_id")).lower()
            if selected and base_id not in selected and participant_id not in selected:
                continue
            metrics = trace.metrics
            winner_bonus = 1.0 if participant_id in final_winners or base_id in final_winners else 0.0
            scores.append(
                0.40 * clamp01(metrics.get("contribution_credit"), 0.0)
                + 0.25 * clamp01(metrics.get("evidence_quality"), 0.0)
                + 0.20 * winner_bonus
                + 0.15 * clamp01(metrics.get("confidence_calibration"), 0.0)
            )
        return clamp01(sum(scores) / max(1, len(scores)))

    def _final_outcome(self, rollout: CouncilRollout) -> float:
        if rollout.final_grader_trace is None:
            return 0.0
        metrics = rollout.final_grader_trace.metrics
        if "final_outcome" in metrics:
            return clamp01(metrics.get("final_outcome"), 0.0)
        confidence = clamp01(rollout.final_grader_trace.payload.get("confidence"), 0.5)
        disagreement = clamp01(rollout.final_grader_trace.payload.get("disagreement"), 0.5)
        return clamp01(confidence * (1.0 - disagreement))

    def _context_efficiency(self, rollout: CouncilRollout, selected_ids: list[str], risk_tags: list[str]) -> float:
        selected_count = max(1, len(selected_ids))
        risk_count = max(1, len(set(risk_tags)))
        ideal = 1 if risk_count <= 1 else 2 if risk_count <= 3 else 3
        rounds = max([trace.round_idx for trace in rollout.traces] or [1])
        expert_efficiency = min(1.0, ideal / selected_count)
        round_efficiency = min(1.0, 2.0 / max(1.0, float(rounds)))
        return clamp01(0.70 * expert_efficiency + 0.30 * round_efficiency)

    def _irrelevant_expert_penalty(self, rollout: CouncilRollout, selected_ids: list[str]) -> float:
        if not selected_ids:
            return 1.0
        useful_by_base: dict[str, float] = {}
        for trace in rollout.specialist_traces:
            base_id = _as_text(trace.payload.get("base_expert_id")).lower()
            if not base_id:
                continue
            useful_by_base[base_id] = max(
                useful_by_base.get(base_id, 0.0),
                0.60 * clamp01(trace.metrics.get("contribution_credit"), 0.0)
                + 0.40 * clamp01(trace.metrics.get("evidence_quality"), 0.0),
            )
        low = 0
        for sid in selected_ids:
            if useful_by_base.get(sid.lower(), 0.0) < 0.25:
                low += 1
        return clamp01(low / len(selected_ids))

    def _unnecessary_complexity_penalty(self, document: dict[str, Any], selected_ids: list[str], risk_tags: list[str]) -> float:
        if not selected_ids:
            return 1.0
        risk_count = max(1, len(set(risk_tags)))
        ideal = 1 if risk_count <= 1 else 2 if risk_count <= 3 else 3
        over = max(0, len(selected_ids) - ideal) / max(1, ideal)
        duplicate_count = len(selected_ids) - len(set(sid.lower() for sid in selected_ids))
        duplicate_fraction = duplicate_count / len(selected_ids)
        strategies = []
        for row in _as_list(document.get("selected_experts")):
            if isinstance(row, dict):
                strategies.append(_as_text(row.get("role")).lower())
        redundant_roles = 0.0
        if strategies:
            redundant_roles = (len(strategies) - len(set(strategies))) / len(strategies)
        return clamp01(0.55 * over + 0.25 * duplicate_fraction + 0.20 * redundant_roles)


def grade_rollouts(rollouts: Iterable[CouncilRollout]) -> list[PlannerRewardResult]:
    scorer = PlannerRewardFunction()
    return [scorer.score_rollout(rollout) for rollout in rollouts]


def write_graded_rollouts(path: Path, rows: Iterable[PlannerRewardResult]) -> None:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def summarize_graded_rollouts(
    *,
    rows: list[PlannerRewardResult],
    summary_path: Path,
    graded_rollout_path: Path,
) -> RolloutRewardSummary:
    rewards = [row.reward for row in rows]
    summary = RolloutRewardSummary(
        rollout_count=len(rows),
        mean_reward=sum(rewards) / max(1, len(rewards)),
        min_reward=min(rewards) if rewards else 0.0,
        max_reward=max(rewards) if rewards else 0.0,
        output_path=str(summary_path.expanduser().resolve()),
        graded_rollout_path=str(graded_rollout_path.expanduser().resolve()),
    )
    out = summary_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def grade_trace_file(*, trace_path: Path, graded_rollout_path: Path, summary_path: Path) -> RolloutRewardSummary:
    records = read_trace_jsonl(trace_path)
    rollouts = group_traces_into_rollouts(records)
    graded = grade_rollouts(rollouts)
    write_graded_rollouts(graded_rollout_path, graded)
    return summarize_graded_rollouts(rows=graded, summary_path=summary_path, graded_rollout_path=graded_rollout_path)
