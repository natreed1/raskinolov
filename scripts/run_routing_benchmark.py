#!/usr/bin/env python3
"""
Score the deterministic local/frontier/hybrid routing policy.

This runner does not call any model or paid API. It evaluates the routing
decision only and emits the same summary contract used by other benchmarks.
"""

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

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "task_routing_tasks.json"


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected array of routing tasks: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Score task routing policy.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    policy = RoutingPolicy()
    tasks = _load_tasks(args.tasks)
    sink = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None
    passed = 0

    for task in tasks:
        expected = task["expected_route"]
        prompt = task["prompt"]
        request = GenerationRequest(messages=messages_from_prompt(prompt))
        decision = policy.decide(request)
        ok = decision.route == expected
        passed += int(ok)

        row = {
            "id": task["id"],
            "expected_route": expected,
            "actual_route": decision.route,
            "passed": ok,
            "reason": decision.reason,
            "estimated_cost_usd": round(decision.estimated_cost_usd, 6),
            "risk": task.get("risk"),
        }
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task['id']} expected={expected} actual={decision.route} reason={decision.reason}")

    if sink:
        sink.close()

    total = len(tasks)
    rate = passed / total if total else 0.0
    print(f"=== Summary: {passed}/{total} ({rate:.0%}) ===")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
