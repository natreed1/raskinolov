#!/usr/bin/env python3
"""Per-adapter promotion gates with avg-gain objective and rollback guards."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GateThresholds:
    min_average_gain: float = 1.5
    max_avg_regression_tolerance: float = -1.0
    min_apply_pass_rate: float = 0.80
    min_compile_export_pass_rate: float = 0.80
    max_latency_p95_s: float = 80.0
    min_coverage: int = 6
    rollback_apply_fail_pp: float = 8.0
    rollback_latency_p95_regression_pct: float = 40.0
    rollback_safety_events: int = 2
    policy_version: str = "router_policy_v1"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_metrics(path: Path) -> Dict[str, Any]:
    data = _read(path)
    if "aggregate" in data:
        return data
    task_scores = data.get("task_scores") or []
    accepted = float(data.get("accepted", 0))
    tasks = max(1.0, float(data.get("tasks", 1)))
    apply_rate = accepted / tasks
    compile_rate = accepted / tasks
    return {
        "aggregate": {
            "average_score": float(data.get("arena_capability_index", 0.0)),
            "apply_pass_rate": apply_rate,
            "compile_export_pass_rate": compile_rate,
            "latency_p95_s": float(data.get("p95_latency_s", 25.0)),
            "coverage": int(data.get("tasks", len(task_scores))),
            "safety_events": int(data.get("severe_safety_events", 0)),
        }
    }


def evaluate(candidate_doc: Dict[str, Any], champion_doc: Dict[str, Any], thresholds: GateThresholds) -> Dict[str, Any]:
    c = candidate_doc.get("aggregate", {})
    b = champion_doc.get("aggregate", {})
    avg_gain = float(c.get("average_score", 0.0)) - float(b.get("average_score", 0.0))
    apply_delta_pp = (float(c.get("apply_pass_rate", 0.0)) - float(b.get("apply_pass_rate", 0.0))) * 100.0
    base_latency = max(1e-6, float(b.get("latency_p95_s", 1.0)))
    latency_regression_pct = ((float(c.get("latency_p95_s", 0.0)) - base_latency) / base_latency) * 100.0
    safety_events = int(c.get("safety_events", 0))
    reasons: List[str] = []

    if avg_gain < thresholds.min_average_gain:
        reasons.append(f"avg_gain {avg_gain:.2f} < min_average_gain {thresholds.min_average_gain:.2f}")
    if avg_gain < thresholds.max_avg_regression_tolerance:
        reasons.append(
            f"avg_gain {avg_gain:.2f} < max_avg_regression_tolerance {thresholds.max_avg_regression_tolerance:.2f}"
        )
    if float(c.get("apply_pass_rate", 0.0)) < thresholds.min_apply_pass_rate:
        reasons.append("apply_pass_rate below floor")
    if float(c.get("compile_export_pass_rate", 0.0)) < thresholds.min_compile_export_pass_rate:
        reasons.append("compile_export_pass_rate below floor")
    if float(c.get("latency_p95_s", 0.0)) > thresholds.max_latency_p95_s:
        reasons.append("latency_p95_s above ceiling")
    if int(c.get("coverage", 0)) < thresholds.min_coverage:
        reasons.append("coverage below minimum")

    rollback_triggers: List[str] = []
    if apply_delta_pp <= -thresholds.rollback_apply_fail_pp:
        rollback_triggers.append("apply_fail_regression")
    if latency_regression_pct >= thresholds.rollback_latency_p95_regression_pct:
        rollback_triggers.append("latency_regression")
    if safety_events > thresholds.rollback_safety_events:
        rollback_triggers.append("safety_events")

    decision = "promote" if (not reasons and not rollback_triggers) else "reject"
    if rollback_triggers:
        decision = "rollback"

    return {
        "schema_version": "adapter_gate_result_v1",
        "policy_version": thresholds.policy_version,
        "decision": decision,
        "avg_gain": round(avg_gain, 3),
        "candidate": c,
        "champion": b,
        "reasons": reasons,
        "rollback_triggers": rollback_triggers,
        "signals": {
            "apply_delta_pp": round(apply_delta_pp, 3),
            "latency_regression_pct": round(latency_regression_pct, 3),
            "safety_events": safety_events,
        },
        "thresholds": asdict(thresholds),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate adapter promotion gate verdict.")
    ap.add_argument("--candidate", type=Path, required=True, help="Candidate quality JSON")
    ap.add_argument("--champion", type=Path, required=True, help="Champion baseline JSON")
    ap.add_argument("--output", type=Path, default=None, help="Write gate verdict JSON")
    args = ap.parse_args()

    thresholds = GateThresholds()
    verdict = evaluate(_fallback_metrics(args.candidate), _fallback_metrics(args.champion), thresholds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["decision"] == "promote" else 1)


if __name__ == "__main__":
    main()

