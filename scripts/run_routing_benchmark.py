#!/usr/bin/env python3
"""
Score deterministic routing policy with optional adapter + route labels.

This runner does not call any model or paid API. It evaluates routing decisions
against one or both expected labels:
- `expected_adapter_id` (primary label)
- `expected_route` / `expected_legacy_route` (compatibility label)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "task_routing_tasks.json"


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected array of routing tasks: {path}")
    return data


def _expected_route(task: Dict[str, Any]) -> Optional[str]:
    return task.get("expected_legacy_route") or task.get("expected_route")


def _expected_adapter(task: Dict[str, Any]) -> Optional[str]:
    return task.get("expected_adapter_id")


def _empty_confusion() -> Dict[str, Counter[str]]:
    return {}


def _inc_confusion(matrix: Dict[str, Counter[str]], expected: str, predicted: str) -> None:
    if expected not in matrix:
        matrix[expected] = collections.Counter()
    matrix[expected][predicted] += 1


def _render_confusion(title: str, matrix: Dict[str, Counter[str]]) -> None:
    if not matrix:
        print(f"{title}: no labeled rows")
        return
    print(title)
    for expected in sorted(matrix.keys()):
        by_pred = matrix[expected]
        rendered = ", ".join(f"{pred}:{count}" for pred, count in by_pred.most_common())
        print(f"  expected={expected} -> {rendered}")


def _normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    allowed = {"route", "adapter", "both"}
    if mode not in allowed:
        raise SystemExit(f"--mode must be one of {sorted(allowed)}")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Score task routing policy.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--mode",
        default="route",
        help="Scoring target: route|adapter|both (default route for backward compatibility).",
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    mode = _normalize_mode(args.mode)
    policy = RoutingPolicy()
    tasks = _load_tasks(args.tasks)
    sink = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None
    overall_scored = 0
    overall_passed = 0
    route_scored = 0
    route_passed = 0
    adapter_scored = 0
    adapter_passed = 0
    route_confusion: Dict[str, Counter[str]] = _empty_confusion()
    adapter_confusion: Dict[str, Counter[str]] = _empty_confusion()

    for task in tasks:
        expected_route = _expected_route(task)
        expected_adapter = _expected_adapter(task)
        prompt = task["prompt"]
        request = GenerationRequest(messages=messages_from_prompt(prompt))
        decision = policy.decide(request)

        route_eval_enabled = mode in {"route", "both"}
        adapter_eval_enabled = mode in {"adapter", "both"}
        route_ok: Optional[bool] = None
        adapter_ok: Optional[bool] = None

        if route_eval_enabled and expected_route is not None:
            route_ok = decision.route == expected_route
            route_scored += 1
            route_passed += int(route_ok)
            _inc_confusion(route_confusion, expected_route, decision.route)

        if adapter_eval_enabled and expected_adapter is not None:
            adapter_ok = decision.adapter_id == expected_adapter
            adapter_scored += 1
            adapter_passed += int(adapter_ok)
            _inc_confusion(adapter_confusion, expected_adapter, decision.adapter_id)

        checks: List[bool] = []
        if route_ok is not None:
            checks.append(route_ok)
        if adapter_ok is not None:
            checks.append(adapter_ok)
        row_scored = bool(checks)
        row_passed = all(checks) if checks else None
        if row_scored:
            overall_scored += 1
            overall_passed += int(bool(row_passed))

        row = {
            "id": task["id"],
            "prompt": prompt,
            "expected_route": expected_route,
            "expected_legacy_route": expected_route,
            "expected_adapter_id": expected_adapter,
            "actual_route": decision.route,
            "actual_legacy_route": decision.route,
            "actual_adapter_id": decision.adapter_id,
            "route_passed": route_ok,
            "adapter_passed": adapter_ok,
            "passed": row_passed,
            "scored": row_scored,
            "mode": mode,
            "reason": decision.reason,
            "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
            "risk": task.get("risk"),
            "adapter_id": decision.adapter_id,
            "execution_tier": decision.execution_tier,
            "council_mode": decision.council_mode,
            "confidence": decision.confidence,
            "ambiguity": decision.ambiguity,
            "risk_class": decision.risk_class,
            "complexity": decision.complexity,
            "policy_version": decision.policy_version,
            "lineage": decision.lineage,
        }
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not row_scored:
            status = "SKIP"
        else:
            status = "PASS" if row_passed else "FAIL"
        print(
            f"[{status}] {task['id']} "
            f"route={decision.route} adapter={decision.adapter_id} "
            f"reason={decision.reason}"
        )

    if sink:
        sink.close()

    route_rate = route_passed / route_scored if route_scored else 0.0
    adapter_rate = adapter_passed / adapter_scored if adapter_scored else 0.0
    overall_rate = overall_passed / overall_scored if overall_scored else 0.0
    print(f"=== Route accuracy: {route_passed}/{route_scored} ({route_rate:.0%}) ===")
    print(f"=== Adapter accuracy: {adapter_passed}/{adapter_scored} ({adapter_rate:.0%}) ===")
    print(f"=== Overall: {overall_passed}/{overall_scored} ({overall_rate:.0%}) ===")
    _render_confusion("Route confusion", route_confusion)
    _render_confusion("Adapter confusion", adapter_confusion)

    summary = {
        "mode": mode,
        "tasks_total": len(tasks),
        "overall": {"passed": overall_passed, "scored": overall_scored, "rate": round(overall_rate, 4)},
        "route": {"passed": route_passed, "scored": route_scored, "rate": round(route_rate, 4)},
        "adapter": {"passed": adapter_passed, "scored": adapter_scored, "rate": round(adapter_rate, 4)},
        "route_confusion": {k: dict(v) for k, v in route_confusion.items()},
        "adapter_confusion": {k: dict(v) for k, v in adapter_confusion.items()},
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if overall_scored > 0 and overall_passed < overall_scored:
        sys.exit(1)


if __name__ == "__main__":
    main()
