"""Shared council round executor.

This module owns the repeatable mechanics of running a council: participant
turn order, peer summaries, prompt construction, adjudication previews, early
convergence, and trace shape. Callers still own UI, model loading, task loading,
and persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from router.council import adjudicate_council_outputs, shape_prompt_for_profile
from .traces import (
    TraceSink,
    build_grader_trace,
    build_planner_trace,
    build_specialist_trace,
)


@dataclass(frozen=True)
class CouncilTurn:
    prompt: str
    participant: dict[str, Any]
    participant_prompt: str
    round_idx: int
    debate_max_rounds: int
    peer_digest: str


@dataclass(frozen=True)
class CouncilGeneration:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


PromptBuilder = Callable[..., str]
GenerationFn = Callable[[CouncilTurn], CouncilGeneration]


def _clamp01(value: Any, default: float) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = float(default)
    return max(0.0, min(1.0, raw))


def _participant_field(participant: dict[str, Any], key: str, default: str = "") -> str:
    return str((participant or {}).get(key) or default).strip()


def _peer_digest(outputs: Sequence[dict[str, Any]]) -> str:
    rows: list[str] = []
    for prev in outputs:
        pid = str(prev.get("participant_id") or "").strip()
        text = str(prev.get("text") or "").strip()
        if pid and text:
            rows.append(f"- {pid}: {text[:260]}")
    return "\n".join(rows[:4])


def _common_participant_output(
    *,
    participant: dict[str, Any],
    round_idx: int,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    participant_id = _participant_field(participant, "participant_id")
    base_expert_id = _participant_field(participant, "base_expert_id", participant_id)
    participant_type = _participant_field(participant, "participant_type")
    traits = dict((participant or {}).get("traits") or {})
    out = {
        "round_idx": int(round_idx),
        "participant_id": participant_id,
        "base_expert_id": base_expert_id,
        "participant_type": participant_type,
        "role": _participant_field(participant, "role", "agent"),
        "strategy": _participant_field(participant, "strategy"),
        "variant": _participant_field(participant, "variant", "baseline") or "baseline",
        "assertiveness": _clamp01((participant or {}).get("assertiveness", 0.5), 0.5),
        "traits": {
            "verbosity": _clamp01(traits.get("verbosity", 0.5), 0.5),
            "risk_tolerance": _clamp01(traits.get("risk_tolerance", 0.5), 0.5),
            "creativity": _clamp01(traits.get("creativity", 0.45), 0.45),
            "skepticism": _clamp01(traits.get("skepticism", 0.55), 0.55),
            "decisiveness": _clamp01(traits.get("decisiveness", 0.6), 0.6),
        },
        "text": str(text or "").strip(),
        "confidence": _clamp01(metadata.get("confidence", 0.5), 0.5),
        "task_outcome_score": _clamp01(metadata.get("task_outcome_score", 0.5), 0.5),
    }
    for key, value in metadata.items():
        if key in {"confidence", "task_outcome_score"}:
            continue
        out[key] = value
    return out


def build_detailed_participant_prompt(
    *,
    prompt: str,
    participant: dict[str, Any],
    round_idx: int,
    debate_max_rounds: int,
    peer_digest: str,
) -> str:
    """Prompt format used by the interactive router chat council lane."""
    participant_id = _participant_field(participant, "participant_id")
    participant_type = _participant_field(participant, "participant_type")
    role = _participant_field(participant, "role", "agent")
    strategy = _participant_field(participant, "strategy")
    variant = _participant_field(participant, "variant", "baseline") or "baseline"
    traits = dict((participant or {}).get("traits") or {})
    assertiveness = _clamp01((participant or {}).get("assertiveness", 0.5), 0.5)
    verbosity = _clamp01(traits.get("verbosity", 0.5), 0.5)
    risk_tolerance = _clamp01(traits.get("risk_tolerance", 0.5), 0.5)
    creativity = _clamp01(traits.get("creativity", 0.45), 0.45)
    skepticism = _clamp01(traits.get("skepticism", 0.55), 0.55)
    decisiveness = _clamp01(traits.get("decisiveness", 0.6), 0.6)

    participant_prompt = prompt
    if participant_type == "generalist_profile":
        participant_prompt = shape_prompt_for_profile(prompt=prompt, profile=participant_id)
    debate_instructions = ""
    if round_idx > 1:
        debate_instructions = (
            f"Round {round_idx}/{debate_max_rounds} debate mode.\n"
            "Critique weak points in peer ideas, then revise your recommendation.\n"
            f"Peer summaries:\n{peer_digest or '- no peer drafts yet'}\n"
        )
    return (
        f"{participant_prompt}\n\n"
        f"{debate_instructions}"
        f"Council role: {role}\n"
        f"Council strategy: {strategy}\n"
        f"Council variant: {variant}\n"
        f"Assertiveness level: {assertiveness:.2f} (0.0 cautious, 1.0 strongly opinionated).\n"
        f"Verbosity level: {verbosity:.2f} (0.0 concise, 1.0 expansive).\n"
        f"Risk tolerance: {risk_tolerance:.2f} (0.0 conservative, 1.0 aggressive).\n"
        f"Creativity: {creativity:.2f} (0.0 conventional, 1.0 novel).\n"
        f"Skepticism: {skepticism:.2f} (0.0 trusting, 1.0 challenge assumptions).\n"
        f"Decisiveness: {decisiveness:.2f} (0.0 hedged, 1.0 direct decisions).\n"
        "Express your recommendation in proportion to assertiveness and decisiveness.\n"
        "Return a directly useful answer draft."
    )


def build_eval_participant_prompt(
    *,
    prompt: str,
    participant: dict[str, Any],
    round_idx: int,
    debate_max_rounds: int,
    peer_digest: str,
) -> str:
    """Prompt format used by offline council conversation evaluation."""
    participant_id = _participant_field(participant, "participant_id")
    participant_type = _participant_field(participant, "participant_type")
    role = _participant_field(participant, "role", "agent")
    strategy = _participant_field(participant, "strategy")
    variant = _participant_field(participant, "variant", "baseline") or "baseline"
    assertiveness = _clamp01((participant or {}).get("assertiveness", 0.5), 0.5)
    traits = dict((participant or {}).get("traits") or {})

    shaped_prompt = prompt
    if participant_type == "generalist_profile":
        shaped_prompt = shape_prompt_for_profile(prompt=prompt, profile=participant_id)

    debate_instructions = ""
    if round_idx > 1:
        debate_instructions = (
            f"Round {round_idx}/{debate_max_rounds} council debate.\n"
            "Critique weak points in peer ideas, then revise your recommendation.\n"
            f"Peer summaries:\n{peer_digest or '- no peer drafts yet'}\n\n"
        )

    return (
        f"{shaped_prompt}\n\n"
        f"{debate_instructions}"
        f"Council participant: {participant_id}\n"
        f"Role: {role}\n"
        f"Strategy: {strategy}\n"
        f"Variant: {variant}\n"
        f"Assertiveness: {assertiveness:.2f}\n"
        f"Traits: {json.dumps(traits, sort_keys=True)}\n\n"
        "Return a directly useful recommendation for solving the task. "
        "Be explicit about risks, needed files, and whether another expert should override you."
    )


def run_council(
    *,
    prompt: str,
    council_plan: dict[str, Any],
    disagreement: float,
    generate_participant: GenerationFn,
    prompt_builder: PromptBuilder = build_detailed_participant_prompt,
    max_rounds: int | None = None,
    stop_on_convergence: bool = True,
    trace_sink: TraceSink | None = None,
    trace_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run a council plan and return final text plus trace metadata."""
    participants = list((council_plan or {}).get("participants") or [])
    if not participants:
        return "", {"mode": "council", "error": "empty_council_participants"}
    emitted_trace_ids: list[str] = []
    if trace_sink is not None:
        planner_trace = build_planner_trace(
            prompt=prompt,
            council_plan=council_plan,
            trace_context=trace_context,
        )
        trace_sink(planner_trace)
        emitted_trace_ids.append(planner_trace.trace_id)

    policy_rounds = max(1, int((council_plan or {}).get("debate_max_rounds", 2) or 2))
    debate_max_rounds = max(1, min(policy_rounds, int(max_rounds or policy_rounds)))
    low_confidence_threshold = float((council_plan or {}).get("low_confidence_threshold", 0.58) or 0.58)
    disagreement_threshold = float((council_plan or {}).get("disagreement_threshold", 0.45) or 0.45)
    escalation_rule = str((council_plan or {}).get("escalation_rule", "either_trigger") or "either_trigger")

    round_traces: list[dict[str, Any]] = []
    current_outputs: list[dict[str, Any]] = []
    for round_idx in range(1, debate_max_rounds + 1):
        prior_outputs = current_outputs
        current_outputs = []
        peer_digest = _peer_digest(prior_outputs)
        for participant in participants:
            participant_id = _participant_field(participant, "participant_id")
            if not participant_id:
                continue
            participant_prompt = prompt_builder(
                prompt=prompt,
                participant=participant,
                round_idx=round_idx,
                debate_max_rounds=debate_max_rounds,
                peer_digest=peer_digest,
            )
            turn = CouncilTurn(
                prompt=prompt,
                participant=participant,
                participant_prompt=participant_prompt,
                round_idx=round_idx,
                debate_max_rounds=debate_max_rounds,
                peer_digest=peer_digest,
            )
            generation = generate_participant(turn)
            participant_output = _common_participant_output(
                participant=participant,
                round_idx=round_idx,
                text=generation.text,
                metadata=dict(generation.metadata or {}),
            )
            current_outputs.append(participant_output)
            if trace_sink is not None:
                specialist_trace = build_specialist_trace(
                    prompt=prompt,
                    participant_output=participant_output,
                    round_idx=round_idx,
                    trace_context=trace_context,
                )
                trace_sink(specialist_trace)
                emitted_trace_ids.append(specialist_trace.trace_id)

        provisional = adjudicate_council_outputs(
            outputs=current_outputs,
            prompt=prompt,
            disagreement=float(disagreement or 0.0),
            low_confidence_threshold=low_confidence_threshold,
            disagreement_threshold=disagreement_threshold,
            escalation_rule=escalation_rule,
        ).to_dict()
        round_traces.append(
            {
                "round_idx": round_idx,
                "participants": current_outputs,
                "adjudication_preview": {
                    "winner_ids": list(provisional.get("winner_ids") or []),
                    "confidence": provisional.get("confidence"),
                    "disagreement": provisional.get("disagreement"),
                    "escalation_recommended": provisional.get("escalation_recommended"),
                },
            }
        )
        if trace_sink is not None:
            grader_trace = build_grader_trace(
                prompt=prompt,
                adjudication=provisional,
                round_idx=round_idx,
                participant_outputs=current_outputs,
                trace_context=trace_context,
                final=False,
            )
            trace_sink(grader_trace)
            emitted_trace_ids.append(grader_trace.trace_id)
        if (
            stop_on_convergence
            and round_idx >= 2
            and float(provisional.get("confidence", 0.0) or 0.0) >= (low_confidence_threshold + 0.08)
            and float(provisional.get("disagreement", 1.0) or 1.0) <= (disagreement_threshold * 0.8)
        ):
            break

    adjudication = adjudicate_council_outputs(
        outputs=current_outputs,
        prompt=prompt,
        disagreement=float(disagreement or 0.0),
        low_confidence_threshold=low_confidence_threshold,
        disagreement_threshold=disagreement_threshold,
        escalation_rule=escalation_rule,
    ).to_dict()
    final_text = str(adjudication.get("final_text") or "").strip()
    if not final_text and current_outputs:
        final_text = str(current_outputs[0].get("text") or "").strip()
    if trace_sink is not None:
        grader_trace = build_grader_trace(
            prompt=prompt,
            adjudication=adjudication,
            round_idx=len(round_traces),
            participant_outputs=current_outputs,
            trace_context=trace_context,
            final=True,
        )
        trace_sink(grader_trace)
        emitted_trace_ids.append(grader_trace.trace_id)
    return final_text, {
        "mode": "council",
        "participants": current_outputs,
        "rounds": round_traces,
        "trace_ids": emitted_trace_ids,
        "debate_max_rounds": debate_max_rounds,
        "debate_rounds_run": len(round_traces),
        "adjudication": adjudication,
        "transparency": str((council_plan or {}).get("transparency", "detailed") or "detailed"),
    }
