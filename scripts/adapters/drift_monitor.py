#!/usr/bin/env python3
"""Adapter and router drift checks with automated action suggestions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DriftThresholds:
    warn_drop: float = 2.0
    freeze_drop: float = 4.0
    reduce_drop: float = 6.0
    rollback_drop: float = 8.0
    policy_version: str = "router_policy_v1"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(doc: Dict[str, Any]) -> float:
    if "aggregate" in doc:
        return float(doc["aggregate"].get("average_score", 0.0))
    return float(doc.get("arena_capability_index", 0.0))


def _action(drop: float, thresholds: DriftThresholds) -> str:
    if drop >= thresholds.rollback_drop:
        return "rollback"
    if drop >= thresholds.reduce_drop:
        return "reduce"
    if drop >= thresholds.freeze_drop:
        return "freeze"
    if drop >= thresholds.warn_drop:
        return "warn"
    return "ok"


def evaluate_drift(
    *,
    adapter_current: Dict[str, Any],
    adapter_baseline: Dict[str, Any],
    router_current: Dict[str, Any],
    router_baseline: Dict[str, Any],
    thresholds: DriftThresholds,
) -> Dict[str, Any]:
    adapter_drop = _score(adapter_baseline) - _score(adapter_current)
    router_drop = _score(router_baseline) - _score(router_current)
    adapter_action = _action(adapter_drop, thresholds)
    router_action = _action(router_drop, thresholds)
    global_action = "ok"
    for candidate in ("warn", "freeze", "reduce", "rollback"):
        if adapter_action == candidate or router_action == candidate:
            global_action = candidate
    return {
        "schema_version": "drift_report_v1",
        "policy_version": thresholds.policy_version,
        "adapter_drop": round(adapter_drop, 3),
        "router_drop": round(router_drop, 3),
        "adapter_action": adapter_action,
        "router_action": router_action,
        "global_action": global_action,
        "automated_actions": _automations(global_action),
    }


def _automations(action: str) -> List[str]:
    if action == "warn":
        return ["emit_alert"]
    if action == "freeze":
        return ["emit_alert", "freeze_promotions"]
    if action == "reduce":
        return ["emit_alert", "freeze_promotions", "reduce_canary_traffic"]
    if action == "rollback":
        return ["emit_alert", "freeze_promotions", "rollback_to_champion"]
    return ["none"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run adapter/router drift checks.")
    ap.add_argument("--adapter-current", type=Path, required=True)
    ap.add_argument("--adapter-baseline", type=Path, required=True)
    ap.add_argument("--router-current", type=Path, required=True)
    ap.add_argument("--router-baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    thresholds = DriftThresholds()
    report = evaluate_drift(
        adapter_current=_read(args.adapter_current),
        adapter_baseline=_read(args.adapter_baseline),
        router_current=_read(args.router_current),
        router_baseline=_read(args.router_baseline),
        thresholds=thresholds,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

