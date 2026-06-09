#!/usr/bin/env python3
"""Build a large, auditable split/merge validation task suite.

This generator intentionally includes:
- Expected multi-agent prompts (dual + tri specialist intent)
- Expected single-agent prompts (single specialist and high-risk frontier prompts)
- Edge-case prompts (ambiguous/mechanical) to verify non-splitting behavior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "benchmarks" / "multi_agent_orchestration_mass_tasks_v1.json"

SPECIALISTS = [
    "loading_screen",
    "hud_status",
    "economy_tooltip",
    "combat_risk",
    "save_load_api_guard",
    "ai_planning_explanation",
]

DOMAINS_BY_SPECIALIST = {
    "loading_screen": ["ui", "loading", "readability"],
    "hud_status": ["ui", "hud", "status"],
    "economy_tooltip": ["economy", "ui", "tooltip"],
    "combat_risk": ["combat", "risk", "preview"],
    "save_load_api_guard": ["api", "auth", "serialization"],
    "ai_planning_explanation": ["ai", "planning", "rationale"],
}


def _category_name(kind: str, parts: list[str]) -> str:
    slug = "-".join(parts).replace("_", "-")
    return f"{kind}-{slug}"


def _dual_prompt(a: str, b: str, variant: int) -> str:
    return (
        f"Design one cohesive update spanning {a.replace('_', ' ')} and {b.replace('_', ' ')}. "
        f"Keep it implementable in existing surfaces, include split responsibilities, and maintain concise tactical wording. "
        f"Variant {variant}."
    )


def _tri_prompt(a: str, b: str, c: str, variant: int) -> str:
    return (
        f"Propose a coordinated response across {a.replace('_', ' ')}, {b.replace('_', ' ')}, and {c.replace('_', ' ')}. "
        f"Keep one primary lane and at least one supporting lane with clear merge notes. Variant {variant}."
    )


def _single_prompt(a: str, variant: int) -> str:
    return (
        f"Focus only on {a.replace('_', ' ')} and produce a direct, single-lane refinement. "
        f"Avoid cross-domain expansion. Variant {variant}."
    )


def _frontier_prompt(variant: int) -> str:
    prompts = [
        "Audit production authentication and cryptography risks before rollout.",
        "Design payment migration rollback strategy with strict privacy constraints.",
        "Contain a distributed data-loss incident with immediate recovery steps.",
        "Review legal/compliance implications of save telemetry retention policy.",
    ]
    return f"{prompts[variant % len(prompts)]} Include concise action sequencing."


def _edge_prompt(variant: int) -> str:
    prompts = [
        "Rename a helper and update references only.",
        "Generate a tiny boilerplate script and exit cleanly.",
        "Rewrite one sentence for clarity with no architecture changes.",
        "Compare two short phrasings and pick one with rationale.",
    ]
    return f"{prompts[variant % len(prompts)]} Variant {variant}."


def _combinations2(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def _combinations3(items: list[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for k in range(j + 1, len(items)):
                out.append((items[i], items[j], items[k]))
    return out


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 1

    # 1) Broad dual-specialist coverage.
    for a, b in _combinations2(SPECIALISTS):
        for variant in range(1, 9):
            row = {
                "id": f"ma-dual-{idx:04d}",
                "category": _category_name("dual", [a, b]),
                "specialists": [a, b],
                "domains": sorted(set(DOMAINS_BY_SPECIALIST[a] + DOMAINS_BY_SPECIALIST[b])),
                "prompt": _dual_prompt(a, b, variant),
                "metadata": {
                    "suite": "multi_agent_orchestration_mass_v1",
                    "expected_multi_agent": True,
                    "complexity": "medium",
                    "edge_case": "cross_specialist_dual",
                },
            }
            rows.append(row)
            idx += 1

    # 2) Tri-specialist edge complexity.
    for a, b, c in _combinations3(SPECIALISTS):
        for variant in range(1, 4):
            row = {
                "id": f"ma-tri-{idx:04d}",
                "category": _category_name("tri", [a, b, c]),
                "specialists": [a, b, c],
                "domains": sorted(
                    set(DOMAINS_BY_SPECIALIST[a] + DOMAINS_BY_SPECIALIST[b] + DOMAINS_BY_SPECIALIST[c])
                ),
                "prompt": _tri_prompt(a, b, c, variant),
                "metadata": {
                    "suite": "multi_agent_orchestration_mass_v1",
                    "expected_multi_agent": True,
                    "complexity": "high",
                    "edge_case": "cross_specialist_tri",
                },
            }
            rows.append(row)
            idx += 1

    # 3) Single-specialist controls (expected no split).
    for a in SPECIALISTS:
        for variant in range(1, 9):
            row = {
                "id": f"ma-single-{idx:04d}",
                "category": _category_name("single", [a]),
                "specialists": [a],
                "domains": list(DOMAINS_BY_SPECIALIST[a]),
                "prompt": _single_prompt(a, variant),
                "metadata": {
                    "suite": "multi_agent_orchestration_mass_v1",
                    "expected_multi_agent": False,
                    "complexity": "low",
                    "edge_case": "single_lane_control",
                },
            }
            rows.append(row)
            idx += 1

    # 4) Frontier-risk controls (expected no split, should route frontier).
    for variant in range(1, 25):
        row = {
            "id": f"ma-frontier-{idx:04d}",
            "category": "frontier_high_risk_control",
            "specialists": ["general_fallback"],
            "domains": ["security", "risk", "incident"],
            "prompt": _frontier_prompt(variant),
            "metadata": {
                "suite": "multi_agent_orchestration_mass_v1",
                "expected_multi_agent": False,
                "complexity": "high",
                "edge_case": "high_risk_frontier",
            },
        }
        rows.append(row)
        idx += 1

    # 5) Mechanical/ambiguous controls (expected no split).
    for variant in range(1, 33):
        row = {
            "id": f"ma-edge-{idx:04d}",
            "category": "edge_mechanical_control",
            "specialists": ["general_fallback"],
            "domains": ["mechanical", "editorial"],
            "prompt": _edge_prompt(variant),
            "metadata": {
                "suite": "multi_agent_orchestration_mass_v1",
                "expected_multi_agent": False,
                "complexity": "low",
                "edge_case": "mechanical_or_ambiguous",
            },
        }
        rows.append(row)
        idx += 1

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded multi-agent orchestration validation suite.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = build_rows()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    expected_multi = sum(1 for r in rows if bool((r.get("metadata") or {}).get("expected_multi_agent")))
    summary = {
        "output": str(output),
        "rows": len(rows),
        "expected_multi_agent_rows": expected_multi,
        "expected_single_agent_rows": len(rows) - expected_multi,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
