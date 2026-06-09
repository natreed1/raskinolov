#!/usr/bin/env python3
"""Promotion gate checks for Router council offline/online metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from router.roster import ExpertRoster


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "routing_benchmark_summary_v2",
        "routing_benchmark_summary_v3_council",
    }:
        raise SystemExit(f"Unsupported summary schema: {payload.get('schema_version')}")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_online(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"Missing online metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Online metrics must be a JSON object.")
    return payload


def _expert_offline_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    per_expert: dict[str, dict[str, int]] = {}
    for row in rows:
        expert = str(row.get("actual_adapter_id") or "").strip()
        if not expert or expert == "general_fallback":
            continue
        bucket = per_expert.setdefault(expert, {"total": 0, "passed": 0})
        if row.get("expected_adapter_id") is None:
            continue
        bucket["total"] += 1
        bucket["passed"] += int(bool(row.get("adapter_passed")))
    out: dict[str, float] = {}
    for expert, stats in per_expert.items():
        out[expert] = (stats["passed"] / stats["total"]) if stats["total"] else 0.0
    return out


def _personality_offline_credit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_personality: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        # Rich council-conversation rows carry per-participant outputs and final winners.
        if isinstance(row.get("adjudication"), dict) and isinstance(row.get("rounds"), list):
            winner_ids = {
                str(pid)
                for pid in list((row.get("adjudication") or {}).get("winner_ids") or [])
                if str(pid).strip()
            }
            rounds = list(row.get("rounds") or [])
            final_round = rounds[-1] if rounds else {}
            participants = list((final_round or {}).get("participants") or [])
            for participant in participants:
                if str((participant or {}).get("participant_type") or "") != "specialist_adapter":
                    continue
                expert = str((participant or {}).get("base_expert_id") or "").strip()
                variant = str((participant or {}).get("variant") or "").strip()
                participant_id = str((participant or {}).get("participant_id") or "").strip()
                if not expert or not variant:
                    continue
                bucket = per_personality.setdefault(expert, {}).setdefault(
                    variant,
                    {"attempts": 0, "wins": 0, "top2": 0, "escalation_helped": 0},
                )
                bucket["attempts"] += 1
                bucket["wins"] += int(participant_id in winner_ids)
                bucket["top2"] += int(participant_id in winner_ids)
                bucket["escalation_helped"] += int(
                    bool((row.get("adjudication") or {}).get("escalation_recommended"))
                    and participant_id in winner_ids
                )
            continue

        # Routing benchmark rows only know which personalities were selected by policy.
        expected = str(row.get("expected_adapter_id") or "").strip()
        if not expected or expected == "general_fallback":
            continue
        selected = list(row.get("council_selected_personalities") or [])
        passed = bool(row.get("council_quality_proxy_passed"))
        for participant in selected:
            expert = str((participant or {}).get("base_expert_id") or "").strip()
            variant = str((participant or {}).get("variant") or "").strip()
            if expert != expected or not variant:
                continue
            bucket = per_personality.setdefault(expert, {}).setdefault(
                variant,
                {"attempts": 0, "wins": 0, "top2": 0, "escalation_helped": 0},
            )
            bucket["attempts"] += 1
            bucket["wins"] += int(passed)
            bucket["top2"] += int(passed)
            bucket["escalation_helped"] += int(passed and bool(row.get("council_escalation_passed")))

    out: dict[str, dict[str, float]] = {}
    for expert, personality_rows in per_personality.items():
        out[expert] = {}
        for variant, stats in personality_rows.items():
            attempts = int(stats["attempts"])
            wins = int(stats["wins"])
            top2 = int(stats["top2"])
            out[expert][variant] = {
                "attempts": attempts,
                "wins": wins,
                "top2": top2,
                "correctness": (wins / attempts) if attempts else 0.0,
                "top2_rate": (top2 / attempts) if attempts else 0.0,
                "escalation_helped": int(stats["escalation_helped"]),
            }
    return out


def _personality_offline_scores(credit: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        expert: {
            variant: float((stats or {}).get("correctness") or 0.0)
            for variant, stats in personality_rows.items()
        }
        for expert, personality_rows in credit.items()
    }


def _personality_offline_counts(credit: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        expert: {
            variant: int((stats or {}).get("attempts") or 0)
            for variant, stats in personality_rows.items()
        }
        for expert, personality_rows in credit.items()
    }


def _personality_online_metrics(online: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    raw = online.get("personalities") if isinstance(online.get("personalities"), dict) else {}
    scores: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for expert_id, payload in raw.items():
        expert = str(expert_id)
        if isinstance(payload, dict) and "task_outcome_rate" in payload:
            # Also accept flat keys like "combat_risk::risk_auditor".
            if "::" not in expert:
                continue
            base, variant = expert.split("::", 1)
            scores.setdefault(base, {})[variant] = float((payload or {}).get("task_outcome_rate") or 0.0)
            counts.setdefault(base, {})[variant] = int((payload or {}).get("sample_count") or 0)
            continue
        if not isinstance(payload, dict):
            continue
        for variant_id, row in payload.items():
            variant = str(variant_id)
            if not isinstance(row, dict):
                continue
            scores.setdefault(expert, {})[variant] = float((row or {}).get("task_outcome_rate") or 0.0)
            counts.setdefault(expert, {})[variant] = int((row or {}).get("sample_count") or 0)
    return scores, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Router council promotion gate.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--min-overall-acc", type=float, default=0.90)
    parser.add_argument("--min-adapter-acc", type=float, default=0.88)
    parser.add_argument("--min-route-acc", type=float, default=0.95)
    parser.add_argument("--min-council-quality-acc", type=float, default=0.88)
    parser.add_argument("--min-council-selection-recall", type=float, default=0.80)
    parser.add_argument("--min-council-escalation-acc", type=float, default=0.85)
    parser.add_argument("--max-high-risk-route-miss-rate", type=float, default=0.05)
    parser.add_argument("--max-high-risk-misroutes", type=int, default=0)
    parser.add_argument("--online-json", type=Path, default=None)
    parser.add_argument("--min-online-task-outcome", type=float, default=0.80)
    parser.add_argument("--min-online-samples", type=int, default=20)
    parser.add_argument("--roster-json", type=Path, default=None)
    parser.add_argument("--write-roster", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = _load_summary(args.summary_json.expanduser().resolve())
    rows = _load_rows(args.rows_jsonl.expanduser().resolve())
    online = _load_online(args.online_json.expanduser().resolve() if args.online_json else None)

    overall_acc = float((summary.get("overall") or {}).get("accuracy") or 0.0)
    route_acc = float((summary.get("route") or {}).get("accuracy") or 0.0)
    adapter_acc = float((summary.get("adapter") or {}).get("accuracy") or 0.0)
    council_quality = float((summary.get("council") or {}).get("quality_proxy_accuracy") or 0.0)
    council_recall = float((summary.get("council") or {}).get("selection_recall") or 0.0)
    council_escalation = float((summary.get("council") or {}).get("escalation_accuracy") or 0.0)

    high_risk_rows = [r for r in rows if str(r.get("risk") or "").lower() == "high"]
    high_risk_route_fail = [
        r
        for r in high_risk_rows
        if r.get("route_passed") is False
    ]
    high_risk_misroute_rate = (
        (len(high_risk_route_fail) / len(high_risk_rows)) if high_risk_rows else 0.0
    )
    global_online_task_outcome = float((online.get("global") or {}).get("task_outcome_rate") or 0.0)
    global_online_samples = int((online.get("global") or {}).get("sample_count") or 0)
    online_gate_checked = bool(online)
    if online_gate_checked:
        online_gate_passed = (
            global_online_samples >= args.min_online_samples
            and global_online_task_outcome >= args.min_online_task_outcome
        )
    else:
        online_gate_passed = True

    checks = {
        "overall_accuracy": overall_acc >= args.min_overall_acc,
        "adapter_accuracy": adapter_acc >= args.min_adapter_acc,
        "route_accuracy": route_acc >= args.min_route_acc,
        "council_quality_accuracy": council_quality >= args.min_council_quality_acc,
        "council_selection_recall": council_recall >= args.min_council_selection_recall,
        "council_escalation_accuracy": council_escalation >= args.min_council_escalation_acc,
        "high_risk_misroute_rate": high_risk_misroute_rate <= args.max_high_risk_route_miss_rate,
        "high_risk_misroute_count": len(high_risk_route_fail) <= args.max_high_risk_misroutes,
        "online_task_outcome_gate": online_gate_passed,
    }
    passed = all(checks.values())
    roster_report: dict[str, Any] | None = None
    if args.roster_json:
        roster_path = args.roster_json.expanduser().resolve()
        specialist_ids = {
            str(row.get("actual_adapter_id") or "").strip()
            for row in rows
            if str(row.get("actual_adapter_id") or "").strip()
        }
        roster = ExpertRoster.load_or_bootstrap(path=roster_path, specialist_ids=specialist_ids)
        offline_scores = _expert_offline_scores(rows)
        expert_rows = online.get("experts") if isinstance(online.get("experts"), dict) else {}
        online_scores = {
            str(expert_id): float((payload or {}).get("task_outcome_rate") or 0.0)
            for expert_id, payload in expert_rows.items()
        }
        online_counts = {
            str(expert_id): int((payload or {}).get("sample_count") or 0)
            for expert_id, payload in expert_rows.items()
        }
        personality_online_scores, personality_online_counts = _personality_online_metrics(online)
        personality_credit = _personality_offline_credit(rows)
        roster_report = roster.apply_combined_gate(
            offline_scores=offline_scores,
            online_scores=online_scores,
            online_counts=online_counts,
            personality_offline_scores=_personality_offline_scores(personality_credit),
            personality_offline_counts=_personality_offline_counts(personality_credit),
            personality_online_scores=personality_online_scores,
            personality_online_counts=personality_online_counts,
            min_offline_score=args.min_adapter_acc,
            min_online_task_outcome=args.min_online_task_outcome,
            min_online_samples=args.min_online_samples,
        )
        if args.write_roster:
            roster.save(roster_path)
    report = {
        "schema_version": "router_promotion_gate_v2_combined",
        "passed": passed,
        "thresholds": {
            "min_overall_acc": args.min_overall_acc,
            "min_adapter_acc": args.min_adapter_acc,
            "min_route_acc": args.min_route_acc,
            "min_council_quality_acc": args.min_council_quality_acc,
            "min_council_selection_recall": args.min_council_selection_recall,
            "min_council_escalation_acc": args.min_council_escalation_acc,
            "max_high_risk_route_miss_rate": args.max_high_risk_route_miss_rate,
            "max_high_risk_misroutes": args.max_high_risk_misroutes,
            "min_online_task_outcome": args.min_online_task_outcome,
            "min_online_samples": args.min_online_samples,
        },
        "metrics": {
            "overall_accuracy": round(overall_acc, 4),
            "adapter_accuracy": round(adapter_acc, 4),
            "route_accuracy": round(route_acc, 4),
            "council_quality_accuracy": round(council_quality, 4),
            "council_selection_recall": round(council_recall, 4),
            "council_escalation_accuracy": round(council_escalation, 4),
            "high_risk_total": len(high_risk_rows),
            "high_risk_misroutes": len(high_risk_route_fail),
            "high_risk_misroute_rate": round(high_risk_misroute_rate, 4),
            "global_online_task_outcome": round(global_online_task_outcome, 4),
            "global_online_samples": global_online_samples,
        },
        "checks": checks,
        "high_risk_fail_ids": [str(r.get("id")) for r in high_risk_route_fail],
    }
    if roster_report is not None:
        report["roster"] = roster_report
        report["personality_credit"] = personality_credit

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.expanduser().resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

