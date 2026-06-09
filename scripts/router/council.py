"""Router V3 council planning and adjudication helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Iterable, Sequence

GENERALIST_PROFILES = (
    "wide_compressed",
    "precise_short",
    "sliding_window",
)

PROFILE_STRATEGIES = {
    "wide_compressed": "broad_summary_context",
    "precise_short": "strict_relevance_context",
    "sliding_window": "rolling_window_context",
}

_SPECIALIST_VARIANT_DELTAS = {
    "cautious": {
        "assertiveness": -0.18,
        "verbosity": -0.08,
        "risk_tolerance": -0.25,
        "creativity": -0.10,
        "skepticism": 0.20,
        "decisiveness": 0.08,
    },
    "balanced": {},
    "assertive": {
        "assertiveness": 0.20,
        "verbosity": 0.05,
        "risk_tolerance": 0.15,
        "creativity": 0.10,
        "skepticism": -0.10,
        "decisiveness": 0.12,
    },
}


def _sanitize_variant_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name or "").strip())
    return cleaned.strip("_") or "personality"


@dataclass(frozen=True)
class CouncilParticipant:
    participant_id: str
    base_expert_id: str
    participant_type: str
    role: str
    strategy: str
    variant: str
    assertiveness: float
    traits: dict[str, float]
    weight: float
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "base_expert_id": self.base_expert_id,
            "participant_type": self.participant_type,
            "role": self.role,
            "strategy": self.strategy,
            "variant": self.variant,
            "assertiveness": round(float(self.assertiveness), 4),
            "traits": {k: round(float(v), 4) for k, v in sorted(self.traits.items())},
            "weight": round(float(self.weight), 4),
            "priority": int(self.priority),
        }


@dataclass(frozen=True)
class CouncilPlan:
    participants: list[CouncilParticipant]
    selected_specialists: list[str]
    selected_specialist_variants: list[str]
    specialist_limit: int
    disagreement_threshold: float
    low_confidence_threshold: float
    debate_max_rounds: int
    escalation_rule: str
    transparency: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "router_council_plan_v1",
            "participants": [row.to_dict() for row in self.participants],
            "selected_specialists": list(self.selected_specialists),
            "selected_specialist_variants": list(self.selected_specialist_variants),
            "specialist_limit": int(self.specialist_limit),
            "disagreement_threshold": round(float(self.disagreement_threshold), 4),
            "low_confidence_threshold": round(float(self.low_confidence_threshold), 4),
            "debate_max_rounds": int(self.debate_max_rounds),
            "escalation_rule": self.escalation_rule,
            "transparency": self.transparency,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CouncilAdjudication:
    final_text: str
    confidence: float
    disagreement: float
    escalation_recommended: bool
    winner_ids: list[str] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    conflict_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "router_council_adjudication_v1",
            "final_text": self.final_text,
            "confidence": round(float(self.confidence), 4),
            "disagreement": round(float(self.disagreement), 4),
            "escalation_recommended": bool(self.escalation_recommended),
            "winner_ids": list(self.winner_ids),
            "contributions": list(self.contributions),
            "conflict_notes": list(self.conflict_notes),
        }


def shape_prompt_for_profile(*, prompt: str, profile: str, window_chars: int = 900) -> str:
    """Apply profile-specific context shaping without changing backend models."""
    text = (prompt or "").strip()
    if not text:
        return ""
    if profile == "wide_compressed":
        preview = text[: max(256, window_chars)]
        return (
            "Council profile: wide_compressed\n"
            "Goal: prioritize broad context coverage, summarize aggressively, keep major constraints.\n\n"
            f"Prompt:\n{preview}"
        )
    if profile == "precise_short":
        preview = text[: max(180, window_chars // 2)]
        return (
            "Council profile: precise_short\n"
            "Goal: answer with strict relevance, exact constraints, and minimal tangents.\n\n"
            f"Prompt:\n{preview}"
        )
    if profile == "sliding_window":
        chunks: list[str] = []
        step = max(200, window_chars // 2)
        for i in range(0, len(text), step):
            window = text[i : i + window_chars]
            if not window:
                continue
            chunks.append(window)
            if len(chunks) >= 3:
                break
        chunk_blob = "\n\n".join(f"[window_{idx + 1}]\n{chunk}" for idx, chunk in enumerate(chunks))
        return (
            "Council profile: sliding_window\n"
            "Goal: evaluate rolling windows and stitch consistent final reasoning.\n\n"
            f"{chunk_blob}"
        )
    return text


def _rank_specialist_candidates(
    *,
    primary_adapter_id: str,
    secondary_adapter_id: str | None,
    candidate_adapters: Iterable[str],
    active_experts: set[str],
) -> list[str]:
    preferred = [primary_adapter_id, secondary_adapter_id or ""] + list(candidate_adapters)
    out: list[str] = []
    for raw in preferred:
        aid = str(raw or "").strip()
        if not aid or aid == "general_fallback":
            continue
        if aid not in active_experts:
            continue
        if aid in out:
            continue
        out.append(aid)
    return out


def build_council_plan(
    *,
    primary_adapter_id: str,
    secondary_adapter_id: str | None,
    candidate_adapters: Sequence[str],
    active_experts: set[str],
    expert_assertiveness: dict[str, float] | None = None,
    expert_traits: dict[str, dict[str, float]] | None = None,
    expert_personalities: dict[str, list[dict[str, Any]]] | None = None,
    specialist_limit: int = 3,
    specialist_personality_variants: int = 1,
    personality_selection_policy: str = "bandit",
    personality_exploration_rate: float = 0.15,
    disagreement_threshold: float = 0.45,
    low_confidence_threshold: float = 0.58,
    debate_max_rounds: int = 2,
    escalation_rule: str = "either_trigger",
    transparency: str = "detailed",
) -> CouncilPlan:
    assertiveness_map = expert_assertiveness or {}
    traits_map = expert_traits or {}
    personalities_map = expert_personalities or {}

    def _assertiveness_for(expert_id: str, default: float) -> float:
        raw = float(assertiveness_map.get(expert_id, default))
        return max(0.0, min(1.0, raw))

    def _traits_for(expert_id: str, default_assertiveness: float) -> dict[str, float]:
        traits = dict((traits_map.get(expert_id) or {}))
        out = {
            "assertiveness": _assertiveness_for(
                expert_id,
                float(traits.get("assertiveness", default_assertiveness)),
            ),
            "verbosity": max(0.0, min(1.0, float(traits.get("verbosity", 0.5)))),
            "risk_tolerance": max(0.0, min(1.0, float(traits.get("risk_tolerance", 0.5)))),
            "creativity": max(0.0, min(1.0, float(traits.get("creativity", 0.45)))),
            "skepticism": max(0.0, min(1.0, float(traits.get("skepticism", 0.55)))),
            "decisiveness": max(0.0, min(1.0, float(traits.get("decisiveness", 0.6)))),
        }
        return out

    def _fallback_variant_names(count: int) -> list[str]:
        n = max(1, int(count))
        if n == 1:
            return ["balanced"]
        if n == 2:
            return ["cautious", "assertive"]
        names = ["cautious", "balanced", "assertive"]
        if n <= 3:
            return names[:n]
        for idx in range(4, n + 1):
            names.append(f"balanced_{idx}")
        return names

    def _apply_fallback_variant_traits(base: dict[str, float], variant: str) -> dict[str, float]:
        out = dict(base)
        deltas = dict(_SPECIALIST_VARIANT_DELTAS.get(variant, {}))
        for key, delta in deltas.items():
            out[key] = max(0.0, min(1.0, float(out.get(key, 0.5)) + float(delta)))
        return out

    def _personality_bandit_score(adapter_id: str, personality: dict[str, Any]) -> float:
        offline = float(personality.get("offline_score", 0.0) or 0.0)
        online = float(personality.get("online_task_outcome", 0.0) or 0.0)
        offline_n = max(0, int(personality.get("offline_sample_count", 0) or 0))
        online_n = max(0, int(personality.get("online_sample_count", 0) or 0))
        attempts = offline_n + online_n
        observed = []
        if offline_n > 0:
            observed.append(offline)
        if online_n > 0:
            observed.append(online)
        mean = (sum(observed) / len(observed)) if observed else 0.5
        state = str(personality.get("state") or "candidate")
        state_bias = {
            "active": 0.08,
            "candidate": 0.03,
            "cooldown": -0.08,
            "retired": -10.0,
        }.get(state, 0.0)
        exploration = math.sqrt(2.0 / float(attempts + 1))
        jitter_key = f"{adapter_id}::{personality.get('name') or ''}".encode("utf-8")
        jitter = int(hashlib.sha256(jitter_key).hexdigest()[:8], 16) / 0xFFFFFFFF
        return mean + state_bias + (float(personality_exploration_rate) * exploration) + (jitter * 0.0001)

    def _personality_specs_for(adapter_id: str, count: int) -> list[tuple[str, dict[str, float]]]:
        n = max(1, int(count))
        base_traits = _traits_for(adapter_id, 0.6)
        has_roster_personalities = adapter_id in personalities_map
        raw_personalities = list(personalities_map.get(adapter_id) or [])
        selection_policy = str(personality_selection_policy or "bandit").strip().lower()
        if selection_policy in {"bandit", "ucb", "explore_exploit"}:
            raw_personalities = sorted(
                raw_personalities,
                key=lambda row: _personality_bandit_score(adapter_id, row if isinstance(row, dict) else {}),
                reverse=True,
            )
        specs: list[tuple[str, dict[str, float]]] = []
        seen: set[str] = set()
        for idx, raw in enumerate(raw_personalities, start=1):
            if not isinstance(raw, dict):
                continue
            variant = _sanitize_variant_name(str(raw.get("name") or raw.get("id") or f"personality_{idx}"))
            traits = dict(base_traits)
            raw_traits = raw.get("traits")
            if isinstance(raw_traits, dict):
                for key in base_traits:
                    if key in raw_traits and raw_traits.get(key) is not None:
                        try:
                            traits[key] = max(0.0, min(1.0, float(raw_traits.get(key))))
                        except (TypeError, ValueError):
                            pass
            raw_deltas = raw.get("trait_deltas")
            if isinstance(raw_deltas, dict):
                for key in base_traits:
                    if key in raw_deltas and raw_deltas.get(key) is not None:
                        try:
                            traits[key] = max(0.0, min(1.0, float(traits.get(key, 0.5)) + float(raw_deltas.get(key))))
                        except (TypeError, ValueError):
                            pass
            unique_variant = variant
            suffix = 2
            while unique_variant in seen:
                unique_variant = f"{variant}_{suffix}"
                suffix += 1
            seen.add(unique_variant)
            specs.append((unique_variant, traits))
            if len(specs) >= n:
                break
        if specs:
            return specs
        if has_roster_personalities:
            return []
        return [
            (variant, _apply_fallback_variant_traits(base_traits, variant))
            for variant in _fallback_variant_names(n)
        ]

    participants: list[CouncilParticipant] = []
    selected_variant_ids: list[str] = []
    priority = 1
    for profile in GENERALIST_PROFILES:
        default_profile_assertiveness = 0.6 if profile == "wide_compressed" else 0.5
        if profile == "precise_short":
            default_profile_assertiveness = 0.45
        profile_traits = _traits_for(profile, default_profile_assertiveness)
        profile_assertiveness = profile_traits["assertiveness"]
        profile_weight = 1.0 + (profile_assertiveness - 0.5) * 0.4
        participants.append(
            CouncilParticipant(
                participant_id=profile,
                base_expert_id=profile,
                participant_type="generalist_profile",
                role="generalist",
                strategy=PROFILE_STRATEGIES[profile],
                variant="baseline",
                assertiveness=profile_assertiveness,
                traits=profile_traits,
                weight=max(0.75, min(1.25, profile_weight)),
                priority=priority,
            )
        )
        priority += 1

    ranked = _rank_specialist_candidates(
        primary_adapter_id=primary_adapter_id,
        secondary_adapter_id=secondary_adapter_id,
        candidate_adapters=candidate_adapters,
        active_experts=active_experts,
    )
    selected = ranked[: max(0, int(specialist_limit))]
    for adapter_id in selected:
        personality_specs = _personality_specs_for(adapter_id, specialist_personality_variants)
        for variant, specialist_traits in personality_specs:
            specialist_assertiveness = specialist_traits["assertiveness"]
            specialist_weight = 1.15 + (specialist_assertiveness - 0.5) * 0.5
            participant_id = (
                adapter_id if variant == "balanced" and len(personality_specs) == 1 else f"{adapter_id}::{variant}"
            )
            participants.append(
                CouncilParticipant(
                    participant_id=participant_id,
                    base_expert_id=adapter_id,
                    participant_type="specialist_adapter",
                    role="specialist",
                    strategy="adapter_specialist_focus",
                    variant=variant,
                    assertiveness=specialist_assertiveness,
                    traits=specialist_traits,
                    weight=max(0.85, min(1.45, specialist_weight)),
                    priority=priority,
                )
            )
            selected_variant_ids.append(participant_id)
            priority += 1

    rationale = (
        "parallel council with fixed 3 generalists; specialist slots allocated by ranked "
        "router adapters filtered through active roster"
    )
    return CouncilPlan(
        participants=participants,
        selected_specialists=selected,
        selected_specialist_variants=selected_variant_ids,
        specialist_limit=max(0, int(specialist_limit)),
        disagreement_threshold=float(disagreement_threshold),
        low_confidence_threshold=float(low_confidence_threshold),
        debate_max_rounds=max(1, int(debate_max_rounds)),
        escalation_rule=str(escalation_rule or "either_trigger"),
        transparency=str(transparency or "detailed"),
        rationale=rationale,
    )


def estimate_disagreement(*, confidence: float, ambiguity: float, participant_count: int) -> float:
    base = max(0.0, min(1.0, float(ambiguity)))
    conf_term = max(0.0, min(1.0, 1.0 - float(confidence)))
    spread = min(0.2, max(0, participant_count - 3) * 0.03)
    return max(0.0, min(1.0, (0.55 * base) + (0.35 * conf_term) + spread))


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/-]{2,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "your",
    "you",
    "are",
    "can",
    "will",
    "should",
    "need",
    "needs",
    "task",
    "prompt",
    "create",
    "update",
    "ensure",
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "") if m.group(0).lower() not in _STOPWORDS}


def _blind_id_for(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    idx = max(0, int(index))
    label = ""
    while True:
        label = alphabet[idx % len(alphabet)] + label
        idx = (idx // len(alphabet)) - 1
        if idx < 0:
            break
    return f"answer_{label}"


def _rubric_score(*, prompt: str, text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    prompt_terms = _tokens(prompt)
    text_terms = _tokens(stripped)
    overlap = len(prompt_terms & text_terms)
    coverage = overlap / max(1, min(len(prompt_terms), 24))
    lower = stripped.lower()
    components = {
        "answer_present": 1.0 if stripped else 0.0,
        "substantive_length": min(1.0, len(stripped) / 600.0) if stripped else 0.0,
        "prompt_keyword_coverage": max(0.0, min(1.0, coverage)),
        "concrete_reference": 1.0 if ("`" in stripped or "/" in stripped or "." in stripped) else 0.0,
        "risk_awareness": 1.0 if any(term in lower for term in ("risk", "regress", "edge case", "failure")) else 0.0,
        "verification_plan": 1.0 if any(term in lower for term in ("test", "verify", "typecheck", "lint", "ci", "rebuild")) else 0.0,
        "not_mock_or_refusal": 0.0 if any(term in lower for term in ("[mock]", "cannot help", "can't help")) else 1.0,
    }
    score = (
        0.20 * components["answer_present"]
        + 0.15 * components["substantive_length"]
        + 0.25 * components["prompt_keyword_coverage"]
        + 0.12 * components["concrete_reference"]
        + 0.10 * components["risk_awareness"]
        + 0.10 * components["verification_plan"]
        + 0.08 * components["not_mock_or_refusal"]
    )
    return {
        "schema_version": "council_content_rubric_v1",
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "components": {key: round(float(value), 4) for key, value in components.items()},
        "prompt_terms_considered": min(len(prompt_terms), 24),
        "prompt_terms_matched": int(overlap),
    }


def adjudicate_council_outputs(
    *,
    outputs: Sequence[dict[str, Any]],
    disagreement: float,
    low_confidence_threshold: float,
    disagreement_threshold: float,
    escalation_rule: str,
    prompt: str = "",
) -> CouncilAdjudication:
    if not outputs:
        return CouncilAdjudication(
            final_text="",
            confidence=0.0,
            disagreement=max(0.0, min(1.0, float(disagreement))),
            escalation_recommended=True,
            winner_ids=[],
            contributions=[],
            conflict_notes=["no council outputs available"],
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, row in enumerate(outputs):
        pid = str(row.get("participant_id") or "")
        text = str(row.get("text") or "").strip()
        blind_id = _blind_id_for(idx)
        rubric = _rubric_score(prompt=prompt, text=text)
        score = float(rubric["score"])
        scored.append(
            (
                score,
                {
                    "blind_id": blind_id,
                    "participant_id": pid,
                    "text": text,
                    "rubric_scores": rubric,
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)

    top_score, top = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else top_score
    final_conf = max(0.0, min(1.0, float(top_score) + min(0.12, max(0.0, top_score - second_score))))
    winner_ids = [str(top.get("participant_id") or "")]
    if len(scored) > 1 and (top_score - second_score) <= 0.06:
        winner_ids.append(str(scored[1][1].get("participant_id") or ""))

    disagree = max(0.0, min(1.0, float(disagreement)))
    if escalation_rule == "on_disagreement":
        escalate = disagree >= disagreement_threshold
    elif escalation_rule == "on_low_conf":
        escalate = final_conf <= low_confidence_threshold
    elif escalation_rule == "manual_only":
        escalate = False
    else:
        escalate = (disagree >= disagreement_threshold) or (final_conf <= low_confidence_threshold)

    contributions = []
    for rank, (score, row) in enumerate(scored, start=1):
        contributions.append(
            {
                "participant_id": row.get("participant_id"),
                "blind_id": row.get("blind_id"),
                "rank": rank,
                "score": round(float(score), 4),
                "confidence": round(float(score), 4),
                "task_outcome_score": 0.5,
                "rubric_scores": row.get("rubric_scores"),
                "summary": f"candidate_rank_{rank}",
            }
        )
    conflict_notes = []
    if disagree >= disagreement_threshold:
        conflict_notes.append("high_council_disagreement")
    if final_conf <= low_confidence_threshold:
        conflict_notes.append("low_adjudicator_confidence")

    return CouncilAdjudication(
        final_text=str(top.get("text") or ""),
        confidence=final_conf,
        disagreement=disagree,
        escalation_recommended=escalate,
        winner_ids=winner_ids,
        contributions=contributions,
        conflict_notes=conflict_notes,
    )
