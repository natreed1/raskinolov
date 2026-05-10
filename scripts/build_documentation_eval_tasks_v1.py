#!/usr/bin/env python3
"""
Emit curated documentation-routing prompts + deterministic task ids.

Writes:
  - data/routing/documentation_eval_prompts_v1.jsonl
  - benchmarks/documentation_eval_tasks_v1.json

Regenerate whenever prompts change so keyword routing regressions surface in
``run_routing_benchmark.py`` / ``tests/test_documentation_routing_eval.py``.
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

# Keyword-rich prompts local to mlx-lab documentation (avoid game domains).
PROMPTS: tuple[str, ...] = (
    "Editor note: cite HF pins and mlx versions strictly from docs/PROJECT_STATE.md—not the README preamble.",
    "Append-only rule for docs/SESSION_LOG.md: add dated sections at top; treat prior entries like an audit trail.",
    "Where do manifests land? Mention benchmarks/results/runs/<run_id>/manifest.json beside the docs/run_history.md row.",
    "How do I regenerate the stewardship JSONL? Point them at python scripts/ml_workflow.py documentation-dataset defaults.",
    "Cross-link docs/DATA_LAYOUT.md when trainees confuse qwen25-coder-7b game_text lineage with adapters.",
    "Onboarding: fallen-empire-lora is the MLX lane; SOURCE_REPO differs from GAME_ARENA_ROOT worktrees.",
    ".cursor/rules/precise-ml-documentation.mdc is the enforced rule file name—mention docs/SESSION_LOG.md discipline too.",
    "Manifest lineage field documentation_specialist:v1 should accompany synthetic mentions of mlx-lora-docs-normalize.",
    "Promotion note: checkpoints/adapters/documentation/cycle1 is shadow-grade until gated; cite training/adapter_registry_v1.json.",
    "Readers skim README; consolidate stack facts inside docs/PROJECT_STATE.md and avoid duplicating run_history anecdotes.",
    "Note that python scripts/ml_workflow.py arena-dashboard rewires HTML artifacts but skips new benchmarks/results manifests.",
    "Runbook line: rerun python scripts/ml_workflow.py smoke after venv swaps; jot exit codes inside docs/run_history.md.",
    "Train config stub: checkpoints/adapters/documentation/cycle1 should pass --data for data/lora/adapters/documentation_specialist/.",
    "Folder symmetry: mirror other specialists so data/lora/adapters/documentation_specialist holds train.jsonl—see docs/DATA_LAYOUT.md.",
    "Coach writers using docs/CHUNKED_GAME_TEXT.md when chunked exports omit tail bytes on long typescript files.",
    "CI reminder: substantive ml_workflow trains append docs/run_history.md except the arena-dashboard subcommand.",
    "Link writers to SESSION_LOG etiquette when they attempt to revise earlier dated sections instead of appending.",
    "When renaming checkpoint dirs, synchronize training/adapter_registry_v1.json lines with docs mentioning checkpoints/adapters/documentation paths.",
    "Training run review: read benchmarks/results/runs/<run_id>/manifest.json, RUN.md, and training_trajectory.jsonl before summarizing adapter quality.",
    "Token usage routing: low-risk fallen-empire-lora docs and training-run questions should use the local documentation adapter.",
    "Scope guard: the documentation adapter is for fallen-empire-lora run docs, not Fallen Empire game implementation changes.",
    "Registry update note: adapter_registry_v1 should point documentation to the newest shadow checkpoint only after a clean ml_workflow train run.",
)


def task_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=REPO / "data" / "routing" / "documentation_eval_prompts_v1.jsonl",
    )
    ap.add_argument(
        "--tasks-json",
        type=Path,
        default=REPO / "benchmarks" / "documentation_eval_tasks_v1.json",
    )
    args = ap.parse_args()

    for p in PROMPTS:
        cls = classify_prompt(p)
        if cls.adapter_id != "documentation":
            raise SystemExit(
                f"PROMPTS lost documentation label (got {cls.adapter_id}):\n{p}\ncandidates={cls.top_candidates}"
            )

    prompts_by_id: dict[str, str] = {}
    rows_tasks: list[dict] = []
    jsonl_lines: list[str] = []

    for p in PROMPTS:
        tid = task_id(p)
        if tid in prompts_by_id and prompts_by_id[tid] != p:
            raise SystemExit(f"hash collision between prompts for id={tid}")
        prompts_by_id[tid] = p
        rows_tasks.append(
            {
                "id": tid,
                "prompt": p,
                "expected_adapter_id": "documentation",
                "expected_legacy_route": "local",
                "risk": "low",
            }
        )
        jsonl_lines.append(
            json.dumps(
                {
                    "prompt": p,
                    "expected_adapter_id": "documentation",
                    "accepted_for_training": True,
                    "source": "curated",
                    "notes": "keyword_search_routing_docs_v1",
                },
                ensure_ascii=False,
            )
        )

    args.prompts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_jsonl.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    args.tasks_json.write_text(json.dumps(rows_tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.prompts_jsonl} ({len(jsonl_lines)} rows)")
    print(f"Wrote {args.tasks_json} ({len(rows_tasks)} rows)")


if __name__ == "__main__":
    main()
