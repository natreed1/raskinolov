#!/usr/bin/env python3
"""
Build benchmarks/mixed_routing_eval_v1.json (shuffled) from:
  - benchmarks/loading_screen_eval_tasks_v1.json
  - benchmarks/save_load_api_guard_eval_tasks_v1.json
  - benchmarks/documentation_eval_tasks_v1.json
  - benchmarks/task_routing_tasks.json
  - four hand-picked specialist probe prompts (validated against router.classifier)
Default seed=42 so order is stable across regenerations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks" / "mixed_routing_eval_v1.json",
    )
    args = p.parse_args()

    loading = json.loads(
        (REPO / "benchmarks" / "loading_screen_eval_tasks_v1.json").read_text(encoding="utf-8")
    )
    save_sl = json.loads(
        (
            REPO / "benchmarks" / "save_load_api_guard_eval_tasks_v1.json"
        ).read_text(encoding="utf-8")
    )
    policy_rows = json.loads((REPO / "benchmarks" / "task_routing_tasks.json").read_text(encoding="utf-8"))

    mixed: list = []
    for row in loading:
        mixed.append(
            {
                "id": f"mixed-ls-{row['id']}",
                "prompt": row["prompt"],
                "expected_adapter_id": "loading_screen",
                "expected_legacy_route": "local",
                "risk": row.get("risk", "low"),
                "shard": "loading_screen_eval",
            }
        )
    for row in save_sl:
        mixed.append(
            {
                "id": f"mixed-sl-{row['id']}",
                "prompt": row["prompt"],
                "expected_adapter_id": "save_load_api_guard",
                "expected_legacy_route": "local",
                "risk": row.get("risk", "low"),
                "shard": "save_load_eval",
            }
        )

    docs_tasks = json.loads(
        (REPO / "benchmarks" / "documentation_eval_tasks_v1.json").read_text(encoding="utf-8")
    )
    for row in docs_tasks:
        mixed.append(
            {
                "id": f"mixed-doc-{row['id']}",
                "prompt": row["prompt"],
                "expected_adapter_id": "documentation",
                "expected_legacy_route": "local",
                "risk": row.get("risk", "low"),
                "shard": "documentation_eval",
            }
        )

    probes = [
        {
            "id": "mixed-probe-economy-tooltip",
            "prompt": (
                "Economy tooltip: breakdown of net income and resource deltas on hover."
            ),
            "expected_adapter_id": "economy_tooltip",
            "expected_legacy_route": "local",
            "risk": "low",
            "shard": "specialist_probe",
        },
        {
            "id": "mixed-probe-hud-status",
            "prompt": (
                "Tighten the HUD waypoint ribbon showing supply crates and ETA status glyphs."
            ),
            "expected_adapter_id": "hud_status",
            "expected_legacy_route": "local",
            "risk": "low",
            "shard": "specialist_probe",
        },
        {
            "id": "mixed-probe-combat-risk",
            "prompt": "Battle risk calculus over uneven ground and choke lanes.",
            "expected_adapter_id": "combat_risk",
            "expected_legacy_route": "local",
            "risk": "low",
            "shard": "specialist_probe",
        },
        {
            "id": "mixed-probe-ai-planning",
            "prompt": (
                "Faction strategy dossier summarizing foe empire rationale gleaned "
                "from countermoves."
            ),
            "expected_adapter_id": "ai_planning_explanation",
            "expected_legacy_route": "local",
            "risk": "low",
            "shard": "specialist_probe",
        },
    ]
    mixed.extend(probes)

    for row in policy_rows:
        pid = row.get("id") or ""
        merged = {
            "id": f"mixed-policy-{pid}",
            "prompt": row["prompt"],
            "expected_legacy_route": row["expected_route"],
            "risk": row.get("risk", "low"),
            "shard": "policy_fixture",
            "reason": row.get("reason"),
        }
        mixed.append(merged)

    random.seed(args.seed)
    random.shuffle(mixed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(mixed)} tasks to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
