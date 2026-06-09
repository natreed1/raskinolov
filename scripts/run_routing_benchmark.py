#!/usr/bin/env python3
"""Score adapter-first routing policy and council participant quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt
from router.council import build_council_plan, estimate_disagreement
from router.roster import ExpertRoster

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "task_routing_tasks.json"


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected array of routing tasks: {path}")
    return data


def _empty_summary() -> Dict[str, Any]:
    return {"total": 0, "passed": 0, "accuracy": 0.0}


def _safe_rate(passed: int, total: int) -> float:
    return round((passed / total), 4) if total else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return round((numerator / denominator), 4) if denominator else 0.0


def _empty_council_stats() -> Dict[str, Any]:
    return {
        "rows": 0,
        "with_expected_specialist": 0,
        "selection_hits": 0,
        "selection_precision_sum": 0.0,
        "selection_precision_count": 0,
        "selection_recall_sum": 0.0,
        "selection_recall_count": 0,
        "escalation_correct": 0,
        "escalation_total": 0,
        "quality_proxy_passed": 0,
        "quality_proxy_total": 0,
        "avg_selected_specialists_sum": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score task routing policy.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--mode", choices=["route", "adapter", "both", "council"], default="route")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--council-specialist-top-k", type=int, default=3)
    parser.add_argument(
        "--council-roster-json",
        type=Path,
        default=REPO / "data" / "routing" / "council_roster_v1.json",
    )
    parser.add_argument(
        "--compare-baseline-vs-council",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit baseline-vs-council comparison metrics in summary output.",
    )
    args = parser.parse_args()

    policy = RoutingPolicy()
    tasks = _load_tasks(args.tasks)
    sink = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None
    route_stats = _empty_summary()
    adapter_stats = _empty_summary()
    council_stats = _empty_council_stats()
    roster = ExpertRoster.load_or_bootstrap(
        path=args.council_roster_json.expanduser().resolve(),
        specialist_ids={aid for aid in policy.available_adapters if aid != "general_fallback"},
    )
    active_specialists = roster.active_specialists()
    if not active_specialists:
        active_specialists = {aid for aid in policy.available_adapters if aid != "general_fallback"}

    for task in tasks:
        expected_route = str(task.get("expected_route") or "").strip()
        expected_adapter = str(task.get("expected_adapter_id") or "").strip()
        prompt = task["prompt"]
        request = GenerationRequest(messages=messages_from_prompt(prompt))
        decision = policy.decide(request)
        route_ok = bool(expected_route) and (decision.route == expected_route)
        adapter_ok = bool(expected_adapter) and (decision.adapter_id == expected_adapter)

        if expected_route:
            route_stats["total"] += 1
            route_stats["passed"] += int(route_ok)
        if expected_adapter:
            adapter_stats["total"] += 1
            adapter_stats["passed"] += int(adapter_ok)

        plan_payload = dict(getattr(decision, "council_plan", {}) or {})
        if not plan_payload:
            plan_payload = build_council_plan(
                primary_adapter_id=str(getattr(decision, "adapter_id", "general_fallback") or "general_fallback"),
                secondary_adapter_id=str(getattr(decision, "secondary_adapter_id", "") or "").strip() or None,
                candidate_adapters=list(getattr(decision, "candidate_adapters", []) or []),
                active_experts=active_specialists,
                expert_assertiveness=roster.assertiveness_map(),
                expert_traits=roster.traits_map(),
                expert_personalities=roster.personalities_map(),
                specialist_limit=max(0, int(args.council_specialist_top_k)),
            ).to_dict()
        selected_specialists = list(plan_payload.get("selected_specialists") or [])
        selected_personalities = [
            {
                "participant_id": str(row.get("participant_id") or ""),
                "base_expert_id": str(row.get("base_expert_id") or ""),
                "variant": str(row.get("variant") or ""),
            }
            for row in list(plan_payload.get("participants") or [])
            if str(row.get("participant_type") or "") == "specialist_adapter"
        ]
        council_disagreement = estimate_disagreement(
            confidence=float(getattr(decision, "confidence", 0.5) or 0.5),
            ambiguity=float(getattr(decision, "ambiguity", 0.5) or 0.5),
            participant_count=len(list(plan_payload.get("participants") or [])),
        )
        disagreement_threshold = float(plan_payload.get("disagreement_threshold", 0.45) or 0.45)
        low_conf_threshold = float(plan_payload.get("low_confidence_threshold", 0.58) or 0.58)
        escalation_rule = str(plan_payload.get("escalation_rule", "either_trigger") or "either_trigger")
        if escalation_rule == "on_disagreement":
            escalation_pred = council_disagreement >= disagreement_threshold
        elif escalation_rule == "on_low_conf":
            escalation_pred = float(getattr(decision, "confidence", 0.0) or 0.0) <= low_conf_threshold
        elif escalation_rule == "manual_only":
            escalation_pred = False
        else:
            escalation_pred = (
                council_disagreement >= disagreement_threshold
                or float(getattr(decision, "confidence", 0.0) or 0.0) <= low_conf_threshold
            )
        escalation_expected = (str(task.get("risk") or "").lower() == "high") or (expected_route == "frontier")
        escalation_ok = escalation_pred == escalation_expected
        council_hit = bool(expected_adapter) and expected_adapter in selected_specialists
        precision = _safe_div(1.0 if council_hit else 0.0, float(len(selected_specialists)))
        recall = 1.0 if council_hit else 0.0
        quality_proxy_passed = int(adapter_ok or council_hit)
        council_stats["rows"] += 1
        council_stats["avg_selected_specialists_sum"] += float(len(selected_specialists))
        council_stats["escalation_total"] += 1
        council_stats["escalation_correct"] += int(escalation_ok)
        council_stats["quality_proxy_total"] += 1
        council_stats["quality_proxy_passed"] += quality_proxy_passed
        if expected_adapter and expected_adapter != "general_fallback":
            council_stats["with_expected_specialist"] += 1
            council_stats["selection_hits"] += int(council_hit)
            council_stats["selection_precision_sum"] += precision
            council_stats["selection_precision_count"] += 1
            council_stats["selection_recall_sum"] += recall
            council_stats["selection_recall_count"] += 1

        if args.mode == "route":
            judged = route_ok
            has_label = bool(expected_route)
        elif args.mode == "adapter":
            judged = adapter_ok
            has_label = bool(expected_adapter)
        elif args.mode == "council":
            judged = bool(quality_proxy_passed and escalation_ok)
            has_label = bool(expected_adapter or expected_route)
        else:
            judged = route_ok and adapter_ok
            has_label = bool(expected_route) and bool(expected_adapter)

        row = {
            "id": task["id"],
            "expected_route": expected_route or None,
            "actual_route": decision.route,
            "expected_adapter_id": expected_adapter or None,
            "actual_adapter_id": decision.adapter_id,
            "secondary_adapter_id": getattr(decision, "secondary_adapter_id", None),
            "secondary_confidence": round(float(getattr(decision, "secondary_confidence", 0.0)), 4),
            "coarse_bucket": str(getattr(decision, "coarse_bucket", "unclassified")),
            "candidate_adapters": list(getattr(decision, "candidate_adapters", []) or []),
            "hierarchy_stage": str(getattr(decision, "hierarchy_stage", "single_stage")),
            "route_passed": route_ok if expected_route else None,
            "adapter_passed": adapter_ok if expected_adapter else None,
            "passed": judged if has_label else None,
            "reason": decision.reason,
            "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
            "confidence": round(float(getattr(decision, "confidence", 0.0)), 4),
            "ambiguity": round(float(getattr(decision, "ambiguity", 0.0)), 4),
            "policy_version": str(getattr(decision, "policy_version", "router_policy_v2_adapter_first")),
            "risk": task.get("risk"),
            "council_selected_specialists": selected_specialists,
            "council_selected_personalities": selected_personalities,
            "council_participant_count": len(list(plan_payload.get("participants") or [])),
            "council_disagreement": round(float(council_disagreement), 4),
            "council_escalation_predicted": escalation_pred,
            "council_escalation_expected": escalation_expected,
            "council_escalation_passed": escalation_ok,
            "council_selection_hit": council_hit if expected_adapter else None,
            "council_quality_proxy_passed": bool(quality_proxy_passed),
        }
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not has_label:
            status = "SKIP"
        else:
            status = "PASS" if judged else "FAIL"
        print(
            f"[{status}] {task['id']} "
            f"route exp={expected_route or '—'} got={decision.route} "
            f"adapter exp={expected_adapter or '—'} got={decision.adapter_id} "
            f"reason={decision.reason}"
        )

    if sink:
        sink.close()

    route_stats["accuracy"] = _safe_rate(route_stats["passed"], route_stats["total"])
    adapter_stats["accuracy"] = _safe_rate(adapter_stats["passed"], adapter_stats["total"])
    if args.mode == "route":
        overall_passed = route_stats["passed"]
        overall_total = route_stats["total"]
    elif args.mode == "adapter":
        overall_passed = adapter_stats["passed"]
        overall_total = adapter_stats["total"]
    elif args.mode == "council":
        overall_passed = council_stats["quality_proxy_passed"]
        overall_total = council_stats["quality_proxy_total"]
    else:
        overall_total = min(route_stats["total"], adapter_stats["total"])
        overall_passed = 0
        for task in tasks:
            expected_route = str(task.get("expected_route") or "").strip()
            expected_adapter = str(task.get("expected_adapter_id") or "").strip()
            if not expected_route or not expected_adapter:
                continue
            decision = policy.decide(GenerationRequest(messages=messages_from_prompt(task["prompt"])))
            if decision.route == expected_route and decision.adapter_id == expected_adapter:
                overall_passed += 1

    overall_rate = _safe_rate(overall_passed, overall_total)
    council_summary = {
        "rows": int(council_stats["rows"]),
        "with_expected_specialist": int(council_stats["with_expected_specialist"]),
        "selection_hits": int(council_stats["selection_hits"]),
        "selection_precision": _safe_div(
            float(council_stats["selection_precision_sum"]),
            float(council_stats["selection_precision_count"]),
        ),
        "selection_recall": _safe_div(
            float(council_stats["selection_recall_sum"]),
            float(council_stats["selection_recall_count"]),
        ),
        "escalation_accuracy": _safe_rate(
            int(council_stats["escalation_correct"]),
            int(council_stats["escalation_total"]),
        ),
        "quality_proxy_accuracy": _safe_rate(
            int(council_stats["quality_proxy_passed"]),
            int(council_stats["quality_proxy_total"]),
        ),
        "avg_selected_specialists": _safe_div(
            float(council_stats["avg_selected_specialists_sum"]),
            float(council_stats["rows"]),
        ),
    }
    print(f"=== Route: {route_stats['passed']}/{route_stats['total']} ({route_stats['accuracy']:.0%}) ===")
    print(
        f"=== Adapter: {adapter_stats['passed']}/{adapter_stats['total']} ({adapter_stats['accuracy']:.0%}) ==="
    )
    print(
        "=== Council: "
        f"quality={council_summary['quality_proxy_accuracy']:.0%} "
        f"selection_recall={council_summary['selection_recall']:.0%} "
        f"escalation={council_summary['escalation_accuracy']:.0%} ==="
    )
    print(f"=== Overall: {overall_passed}/{overall_total} ({overall_rate:.0%}) ===")

    if args.summary_json:
        summary = {
            "schema_version": "routing_benchmark_summary_v3_council",
            "mode": args.mode,
            "tasks_path": str(args.tasks),
            "route": route_stats,
            "adapter": adapter_stats,
            "council": council_summary,
            "overall": {
                "passed": overall_passed,
                "total": overall_total,
                "accuracy": overall_rate,
            },
        }
        if args.compare_baseline_vs_council:
            summary["comparison"] = {
                "baseline_overall_accuracy": route_stats["accuracy"] if args.mode == "route" else overall_rate,
                "council_quality_proxy_accuracy": council_summary["quality_proxy_accuracy"],
                "council_selection_recall": council_summary["selection_recall"],
                "council_escalation_accuracy": council_summary["escalation_accuracy"],
                "quality_minus_adapter_delta": round(
                    float(council_summary["quality_proxy_accuracy"]) - float(adapter_stats["accuracy"]),
                    4,
                ),
            }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if overall_passed < overall_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
