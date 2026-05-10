#!/usr/bin/env python3
"""Build routing/adapter/SLO dashboard artifacts for v1 architecture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_report(routing_rows: List[Dict[str, Any]], gate_doc: Dict[str, Any], drift_doc: Dict[str, Any]) -> Dict[str, Any]:
    route_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    confidence: List[float] = []
    for row in routing_rows:
        route = str(row.get("actual_route") or row.get("route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        tier = str(row.get("execution_tier") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if "confidence" in row:
            confidence.append(float(row["confidence"]))
    avg_conf = sum(confidence) / len(confidence) if confidence else 0.0
    return {
        "schema_version": "multi_adapter_dashboard_v1",
        "policy_version": "router_policy_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing": {
            "rows": len(routing_rows),
            "route_distribution": route_counts,
            "execution_tier_distribution": tier_counts,
            "avg_confidence": round(avg_conf, 4),
        },
        "adapter_quality": gate_doc,
        "slo": {
            "drift_action": drift_doc.get("global_action", "ok"),
            "adapter_drop": drift_doc.get("adapter_drop", 0.0),
            "router_drop": drift_doc.get("router_drop", 0.0),
        },
    }


def markdown(report: Dict[str, Any]) -> str:
    routing = report["routing"]
    lines = [
        "# Multi-Adapter V1 Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Policy version: `{report['policy_version']}`",
        f"- Routing rows: `{routing['rows']}`",
        f"- Average confidence: `{routing['avg_confidence']}`",
        f"- Route distribution: `{routing['route_distribution']}`",
        f"- Execution tier distribution: `{routing['execution_tier_distribution']}`",
        "",
        "## Adapter Quality",
        "",
        f"- Decision: `{report['adapter_quality'].get('decision', 'n/a')}`",
        f"- Avg gain: `{report['adapter_quality'].get('avg_gain', 'n/a')}`",
        f"- Rollback triggers: `{report['adapter_quality'].get('rollback_triggers', [])}`",
        "",
        "## SLO + Drift",
        "",
        f"- Global action: `{report['slo']['drift_action']}`",
        f"- Adapter drop: `{report['slo']['adapter_drop']}`",
        f"- Router drop: `{report['slo']['router_drop']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build multi-adapter dashboard/report JSON+MD.")
    ap.add_argument("--routing-jsonl", type=Path, required=True)
    ap.add_argument("--gate-json", type=Path, required=True)
    ap.add_argument("--drift-json", type=Path, required=True)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "benchmarks" / "results" / "multi_adapter_dashboard.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=REPO / "benchmarks" / "results" / "multi_adapter_report.md",
    )
    args = ap.parse_args()

    report = build_report(
        _read_jsonl(args.routing_jsonl),
        json.loads(args.gate_json.read_text(encoding="utf-8")),
        json.loads(args.drift_json.read_text(encoding="utf-8")),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.out_json), "md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()

