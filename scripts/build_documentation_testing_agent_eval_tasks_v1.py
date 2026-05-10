#!/usr/bin/env python3
"""
Emit curated documentation/testing-agent routing prompts + deterministic ids.

Writes:
  - data/routing/documentation_testing_agent_eval_prompts_v1.jsonl
  - benchmarks/documentation_testing_agent_eval_tasks_v1.json

This benchmark is separate from the docs-only fixture so the documentation
specialist can be checked against test/runbook work without weakening the
original documentation-routing regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT_SCRIPTS = REPO / "scripts"
if str(SCRIPT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPT_SCRIPTS))

from router.classifier import classify_prompt  # noqa: E402

SHARD = "documentation_testing_agent_eval"

PROMPTS: tuple[str, ...] = (
    "Update docs/WORKFLOW.md with the exact unittest command for tests/test_documentation_routing_eval.py and mention fallen-empire-lora venv assumptions.",
    "Write a docs/SESSION_LOG.md entry summarizing a failed documentation benchmark test; keep prior dated entries append-only.",
    "Describe how documentation_specialist docs/WORKFLOW.md should connect python scripts/run_routing_benchmark.py --tasks benchmarks/documentation_eval_tasks_v1.json --mode both with unittest coverage.",
    "Review a tests/test_fe_ml_lab_tools.py failure and document the likely lab_dashboard fixture mismatch without changing old docs/run_history.md rows.",
    "Add a docs/RUNS.md note that smoke runs should cite benchmarks/results/runs/<run_id>/manifest.json and the test command used.",
    "For a documentation_specialist training report, list the test artifacts to inspect: RUN.md, training_trajectory.jsonl, and unittest output.",
    "Make a fallen-empire-lora README troubleshooting note for python3 -m unittest discover -s tests when documentation benchmark fixtures are missing.",
    "Check that documentation_specialist data/routing/documentation_eval_prompts_v1.jsonl and benchmarks/documentation_eval_tasks_v1.json stay in sync after test regeneration.",
    "Summarize a failing smoke benchmark in docs/PROJECT_STATE.md using only manifest.json, docs/run_history.md, and test stdout.",
    "Draft test-facing copy for .cursor/rules/precise-ml-documentation.mdc that tells agents to cite files and commands exactly.",
    "Describe for a docs/testing agent why benchmarks/results/runs artifacts are generated outputs while *_eval_tasks_v1.json fixtures are committed.",
    "Create a low-risk checklist for rerunning python scripts/ml_workflow.py smoke and recording the unittest command in docs/SESSION_LOG.md.",
    "Document how tests/test_documentation_routing_eval.py guards the local documentation adapter route for fallen-empire-lora runbook prompts.",
    "Write a concise documentation_specialist failure note when run_routing_benchmark.py reports adapter accuracy below 100% on documentation_testing_agent_eval.",
)


def task_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=REPO / "data" / "routing" / "documentation_testing_agent_eval_prompts_v1.jsonl",
    )
    ap.add_argument(
        "--tasks-json",
        type=Path,
        default=REPO / "benchmarks" / "documentation_testing_agent_eval_tasks_v1.json",
    )
    args = ap.parse_args()

    for prompt in PROMPTS:
        cls = classify_prompt(prompt)
        if cls.adapter_id != "documentation":
            raise SystemExit(
                f"PROMPTS lost documentation label (got {cls.adapter_id}):\n"
                f"{prompt}\ncandidates={cls.top_candidates}"
            )

    prompts_by_id: dict[str, str] = {}
    task_rows: list[dict] = []
    jsonl_lines: list[str] = []

    for prompt in PROMPTS:
        tid = task_id(prompt)
        if tid in prompts_by_id and prompts_by_id[tid] != prompt:
            raise SystemExit(f"hash collision between prompts for id={tid}")
        prompts_by_id[tid] = prompt
        task_rows.append(
            {
                "id": tid,
                "prompt": prompt,
                "expected_adapter_id": "documentation",
                "expected_legacy_route": "local",
                "risk": "low",
                "shard": SHARD,
            }
        )
        jsonl_lines.append(
            json.dumps(
                {
                    "prompt": prompt,
                    "expected_adapter_id": "documentation",
                    "accepted_for_training": True,
                    "source": "curated",
                    "notes": SHARD,
                },
                ensure_ascii=False,
            )
        )

    args.prompts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_jsonl.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    args.tasks_json.write_text(json.dumps(task_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.prompts_jsonl} ({len(jsonl_lines)} rows)")
    print(f"Wrote {args.tasks_json} ({len(task_rows)} rows)")


if __name__ == "__main__":
    main()
