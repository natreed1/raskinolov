#!/usr/bin/env python3
"""Collect Router V3 council conversation traces over the 121-task manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_router import ChatMessage, GenerationRequest, LocalMlxBackend, RoutingPolicy, messages_from_prompt
from router.council import adjudicate_council_outputs, shape_prompt_for_profile
from run_final_mass_testing_system import DEFAULT_MANIFEST, _compile_manifest_tasks, _load_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS_JSONL = ROOT / "benchmarks" / "results" / "council_conversation_eval_rows_v1.jsonl"
DEFAULT_SUMMARY_JSON = ROOT / "benchmarks" / "results" / "council_conversation_eval_summary_v1.json"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_run_artifact_checkpoint(completed_steps: int, last_checkpoint_at: float) -> float:
    command = os.environ.get("FE_ARTIFACT_CHECKPOINT_COMMAND", "").strip()
    if not command:
        return last_checkpoint_at
    every_steps = int(os.environ.get("FE_ARTIFACT_UPLOAD_EVERY_STEPS", "0") or "0")
    every_seconds = int(os.environ.get("FE_ARTIFACT_UPLOAD_EVERY_SECONDS", "0") or "0")
    now = time.time()
    due_by_steps = every_steps > 0 and completed_steps > 0 and completed_steps % every_steps == 0
    due_by_time = every_seconds > 0 and now - last_checkpoint_at >= every_seconds
    if not due_by_steps and not due_by_time:
        return last_checkpoint_at
    env = os.environ.copy()
    env["FE_ARTIFACT_COMPLETED_STEPS"] = str(completed_steps)
    env["FE_ARTIFACT_CHECKPOINT_KIND"] = "periodic"
    result = subprocess.run(command, shell=True, env=env, text=True, capture_output=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    if result.returncode != 0:
        print(f"artifact_checkpoint_failed exit={result.returncode}", flush=True)
    return now


def _participant_prompt(*, prompt: str, participant: dict[str, Any], round_idx: int, debate_max_rounds: int, peer_digest: str) -> str:
    participant_id = str(participant.get("participant_id") or "").strip()
    participant_type = str(participant.get("participant_type") or "").strip()
    role = str(participant.get("role") or "agent").strip()
    strategy = str(participant.get("strategy") or "").strip()
    variant = str(participant.get("variant") or "baseline").strip()
    assertiveness = float(participant.get("assertiveness", 0.5) or 0.5)
    traits = dict(participant.get("traits") or {})

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


def _mock_generate(*, participant_id: str, task_id: str, round_idx: int) -> tuple[str, dict[str, Any]]:
    return (
        f"[mock] {participant_id} round {round_idx} recommendation for {task_id}: "
        "identify the narrow edit scope, preserve existing contracts, and escalate if verification risk is high.",
        {"backend": "mock", "generation_tokens": 24, "prompt_tokens": 32},
    )


def _generate_once(
    *,
    backend_cache: dict[str, LocalMlxBackend],
    model_id: str,
    prompt: str,
    max_tokens: int,
    mock_generation: bool,
    participant_id: str,
    task_id: str,
    round_idx: int,
) -> tuple[str, dict[str, Any]]:
    if mock_generation:
        return _mock_generate(participant_id=participant_id, task_id=task_id, round_idx=round_idx)
    backend = backend_cache.get(model_id)
    if backend is None:
        backend = LocalMlxBackend(model_id=model_id)
        backend_cache[model_id] = backend
    text = backend.generate(
        GenerationRequest(
            messages=[
                ChatMessage("system", "You are one participant in a multi-agent engineering council."),
                ChatMessage("user", prompt),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    )
    return text.strip(), {"backend": "local", "generation_tokens": None, "prompt_tokens": None}


def _neutral_scoring_metadata() -> tuple[float, float]:
    return 0.5, 0.5


def _run_task(
    *,
    task: dict[str, Any],
    task_meta: dict[str, Any],
    policy: RoutingPolicy,
    backend_cache: dict[str, LocalMlxBackend],
    model_id: str,
    debate_max_rounds: int,
    participant_max_tokens: int,
    mock_generation: bool,
) -> dict[str, Any]:
    prompt = str(task.get("prompt") or "").strip()
    decision = policy.decide(GenerationRequest(messages=messages_from_prompt(prompt), max_tokens=4096))
    council_plan = dict(getattr(decision, "council_plan", {}) or {})
    participants = list(council_plan.get("participants") or [])
    policy_rounds = max(1, int(council_plan.get("debate_max_rounds", debate_max_rounds) or debate_max_rounds))
    rounds_to_run = max(1, min(policy_rounds, int(debate_max_rounds)))
    round_traces: list[dict[str, Any]] = []
    current_outputs: list[dict[str, Any]] = []

    for round_idx in range(1, rounds_to_run + 1):
        prior_outputs = current_outputs
        current_outputs = []
        peer_digest = "\n".join(
            f"- {row.get('participant_id')}: {str(row.get('text') or '')[:260]}" for row in prior_outputs[:4]
        )
        for participant in participants:
            participant_id = str(participant.get("participant_id") or "").strip()
            if not participant_id:
                continue
            participant_prompt = _participant_prompt(
                prompt=prompt,
                participant=participant,
                round_idx=round_idx,
                debate_max_rounds=rounds_to_run,
                peer_digest=peer_digest,
            )
            output_text, usage = _generate_once(
                backend_cache=backend_cache,
                model_id=model_id,
                prompt=participant_prompt,
                max_tokens=participant_max_tokens,
                mock_generation=mock_generation,
                participant_id=participant_id,
                task_id=str(task.get("id") or ""),
                round_idx=round_idx,
            )
            confidence, task_outcome_score = _neutral_scoring_metadata()
            base_expert_id = str(participant.get("base_expert_id") or participant_id)
            participant_type = str(participant.get("participant_type") or "")
            adapter_requested = base_expert_id if participant_type == "specialist_adapter" else None
            current_outputs.append(
                {
                    "round_idx": round_idx,
                    "participant_id": participant_id,
                    "base_expert_id": base_expert_id,
                    "participant_type": participant_type,
                    "role": str(participant.get("role") or ""),
                    "strategy": str(participant.get("strategy") or ""),
                    "variant": str(participant.get("variant") or "baseline"),
                    "assertiveness": participant.get("assertiveness"),
                    "traits": dict(participant.get("traits") or {}),
                    "text": output_text,
                    "confidence": confidence,
                    "task_outcome_score": task_outcome_score,
                    "adapter_requested": adapter_requested,
                    "resolved_mlx_adapter": None,
                    "adapter_loaded": False,
                    "adapter_load_note": (
                        "council_conversation_eval uses one shared base model; per-specialist adapters are not loaded"
                        if adapter_requested
                        else "generalist_profile_uses_base_model"
                    ),
                    "usage": usage,
                }
            )
        provisional = adjudicate_council_outputs(
            outputs=current_outputs,
            prompt=prompt,
            disagreement=float(getattr(decision, "council_disagreement", 0.0) or 0.0),
            low_confidence_threshold=float(council_plan.get("low_confidence_threshold", 0.58) or 0.58),
            disagreement_threshold=float(council_plan.get("disagreement_threshold", 0.45) or 0.45),
            escalation_rule=str(council_plan.get("escalation_rule", "either_trigger") or "either_trigger"),
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

    adjudication = adjudicate_council_outputs(
        outputs=current_outputs,
        prompt=prompt,
        disagreement=float(getattr(decision, "council_disagreement", 0.0) or 0.0),
        low_confidence_threshold=float(council_plan.get("low_confidence_threshold", 0.58) or 0.58),
        disagreement_threshold=float(council_plan.get("disagreement_threshold", 0.45) or 0.45),
        escalation_rule=str(council_plan.get("escalation_rule", "either_trigger") or "either_trigger"),
    ).to_dict()

    return {
        "schema_version": "council_conversation_eval_row_v1",
        "utc": _utc_iso(),
        "task_id": str(task.get("id") or ""),
        "source_file": str(task_meta.get("source_file") or ""),
        "domain_primary_normalized": str(task_meta.get("domain_primary_normalized") or ""),
        "subskill": str(task_meta.get("subskill") or ""),
        "prompt": prompt,
        "routing": {
            "route": decision.route,
            "adapter_id": decision.adapter_id,
            "confidence": round(float(decision.confidence), 4),
            "ambiguity": round(float(decision.ambiguity), 4),
            "risk_class": decision.risk_class,
            "complexity": decision.complexity,
            "secondary_adapter_id": decision.secondary_adapter_id,
            "secondary_confidence": round(float(decision.secondary_confidence), 4),
            "candidate_adapters": list(decision.candidate_adapters),
            "policy_version": decision.policy_version,
            "reason": decision.reason,
        },
        "council_plan": council_plan,
        "council_disagreement": round(float(getattr(decision, "council_disagreement", 0.0) or 0.0), 4),
        "council_escalation_candidate": bool(getattr(decision, "council_escalation_candidate", False)),
        "flow": {
            "mode": "council_conversation_eval",
            "model": model_id,
            "mock_generation": bool(mock_generation),
            "debate_rounds_run": len(round_traces),
            "participant_count": len(participants),
        },
        "rounds": round_traces,
        "adjudication": adjudication,
        "final_output": str(adjudication.get("final_text") or ""),
        "accepted_for_training": True,
    }


def _summarize(rows: list[dict[str, Any]], *, manifest_path: Path) -> dict[str, Any]:
    total = len(rows)
    escalations = sum(1 for row in rows if bool((row.get("adjudication") or {}).get("escalation_recommended")))
    participant_counts = [int((row.get("flow") or {}).get("participant_count") or 0) for row in rows]
    round_counts = [int((row.get("flow") or {}).get("debate_rounds_run") or 0) for row in rows]
    return {
        "schema_version": "council_conversation_eval_summary_v1",
        "generated_utc": _utc_iso(),
        "manifest": str(manifest_path),
        "rows": total,
        "escalations": escalations,
        "escalation_rate": round(escalations / total, 4) if total else 0.0,
        "avg_participants": round(sum(participant_counts) / len(participant_counts), 2) if participant_counts else 0.0,
        "avg_rounds": round(sum(round_counts) / len(round_counts), 2) if round_counts else 0.0,
        "route_counts": {
            route: sum(1 for row in rows if str((row.get("routing") or {}).get("route") or "unknown") == route)
            for route in sorted({str((row.get("routing") or {}).get("route") or "unknown") for row in rows})
        },
        "adapter_counts": {
            aid: sum(1 for row in rows if str((row.get("routing") or {}).get("adapter_id") or "unknown") == aid)
            for aid in sorted({str((row.get("routing") or {}).get("adapter_id") or "unknown") for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect council conversation traces over the final 121-task manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rows-jsonl", type=Path, default=DEFAULT_ROWS_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--adapter-registry", type=Path, default=ROOT / "training" / "adapter_registry_v1.json")
    parser.add_argument("--local-model", default=os.environ.get("MODEL", DEFAULT_LOCAL_MODEL))
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--debate-max-rounds", type=int, default=int(os.environ.get("ROUTER_COUNCIL_DEBATE_MAX_ROUNDS", "2")))
    parser.add_argument("--participant-max-tokens", type=int, default=512)
    parser.add_argument("--no-rows-reset", action="store_true")
    parser.add_argument("--mock-generation", action="store_true", help="Use deterministic mock participant outputs for smoke tests.")
    parser.add_argument(
        "--selective-generation",
        action="store_true",
        help="Only generate council conversations for high-value routing cases.",
    )
    parser.add_argument("--selective-min-ambiguity", type=float, default=0.35)
    parser.add_argument("--selective-max-confidence", type=float, default=0.72)
    parser.add_argument(
        "--selective-risk",
        action="append",
        default=["high"],
        help="Risk label to always include during selective generation; repeatable.",
    )
    args = parser.parse_args()

    os.environ["ROUTER_COUNCIL_ENABLED"] = "1"
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_json(manifest_path)
    runtime_tasks, task_meta_rows = _compile_manifest_tasks(manifest, manifest_path)
    policy = RoutingPolicy(adapter_registry_path=args.adapter_registry.expanduser().resolve())
    if args.selective_generation:
        selected_tasks: list[dict[str, Any]] = []
        selected_risks = {str(x).lower() for x in list(args.selective_risk or []) if str(x).strip()}
        for task in runtime_tasks:
            prompt = str(task.get("prompt") or "").strip()
            decision = policy.decide(GenerationRequest(messages=messages_from_prompt(prompt), max_tokens=4096))
            risk = str(task.get("risk") or "").lower()
            keep = (
                float(getattr(decision, "ambiguity", 0.0) or 0.0) >= float(args.selective_min_ambiguity)
                or float(getattr(decision, "confidence", 1.0) or 1.0) <= float(args.selective_max_confidence)
                or risk in selected_risks
            )
            if keep:
                selected_tasks.append(task)
        runtime_tasks = selected_tasks
    if args.max_tasks > 0:
        runtime_tasks = runtime_tasks[: args.max_tasks]
    wanted = {str(row.get("id") or "") for row in runtime_tasks}
    task_meta_rows = [row for row in task_meta_rows if str(row.get("id") or "") in wanted]
    task_meta = {str(row["id"]): row for row in task_meta_rows}

    rows_path = args.rows_jsonl.expanduser().resolve()
    if not args.no_rows_reset:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("", encoding="utf-8")

    backend_cache: dict[str, LocalMlxBackend] = {}
    rows: list[dict[str, Any]] = []
    artifact_checkpoint_at = time.time()
    for idx, task in enumerate(runtime_tasks, start=1):
        task_id = str(task.get("id") or "")
        row = _run_task(
            task=task,
            task_meta=task_meta.get(task_id, {}),
            policy=policy,
            backend_cache=backend_cache,
            model_id=str(args.local_model),
            debate_max_rounds=max(1, int(args.debate_max_rounds)),
            participant_max_tokens=max(32, int(args.participant_max_tokens)),
            mock_generation=bool(args.mock_generation),
        )
        rows.append(row)
        _append_jsonl(rows_path, row)
        artifact_checkpoint_at = _maybe_run_artifact_checkpoint(len(rows), artifact_checkpoint_at)
        print(
            f"[{idx}/{len(runtime_tasks)}] task={task_id} adapter={row['routing']['adapter_id']} "
            f"participants={row['flow']['participant_count']} rounds={row['flow']['debate_rounds_run']} "
            f"escalate={row['adjudication'].get('escalation_recommended')}",
            flush=True,
        )

    summary = _summarize(rows, manifest_path=manifest_path)
    summary["rows_jsonl"] = str(rows_path)
    summary["local_model"] = str(args.local_model)
    summary["mock_generation"] = bool(args.mock_generation)
    args.summary_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.expanduser().resolve().write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
