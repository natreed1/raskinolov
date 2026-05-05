#!/usr/bin/env python3
"""Build run-analysis RAG corpus manifest from docs + latest run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "rag" / "run_analysis_agent_corpus.json"
RUNS_DIR = REPO / "benchmarks" / "results" / "runs"


def _latest_run_docs(limit: int) -> list[str]:
    if not RUNS_DIR.is_dir():
        return []
    candidates = []
    for path in RUNS_DIR.glob("*/manifest.json"):
        candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[: max(0, limit)]
    rel_paths: list[str] = []
    for manifest in chosen:
        run_dir = manifest.parent
        run_md = run_dir / "RUN.md"
        for p in (manifest, run_md):
            if p.is_file():
                rel_paths.append(str(p.relative_to(REPO)))
    return rel_paths


def build_payload(latest_runs: int) -> dict:
    sources = [
        {"path": "docs/PROJECT_STATE.md", "kind": "current_state", "weight": 2.4},
        {"path": "docs/SESSION_LOG.md", "kind": "session_log", "weight": 2.3},
        {"path": "docs/run_history.md", "kind": "workflow_history", "weight": 2.2},
        {"path": "docs/SPECIALIZED_RUN_HISTORY.md", "kind": "specialized_history", "weight": 2.4},
        {"path": "docs/RUNS.md", "kind": "run_interpretation", "weight": 2.0},
        {"path": "docs/WORKFLOW.md", "kind": "runbook", "weight": 1.9},
    ]
    timeseries = REPO / "benchmarks" / "results" / "documentation_rag_timeseries.jsonl"
    if timeseries.is_file():
        sources.append(
            {"path": "benchmarks/results/documentation_rag_timeseries.jsonl", "kind": "timeseries", "weight": 1.8}
        )
    for rel in _latest_run_docs(latest_runs):
        kind = "run_manifest" if rel.endswith("manifest.json") else "run_md"
        weight = 1.9 if kind == "run_manifest" else 1.7
        sources.append({"path": rel, "kind": kind, "weight": weight})
    return {
        "schema_version": "run_analysis_agent_corpus_v1",
        "description": "Run-analysis retrieval corpus for documenting and analyzing workflow runs.",
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run-analysis RAG corpus manifest.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--latest-runs", type=int, default=12, help="Include latest N run manifests/RUN.md pairs.")
    args = parser.parse_args()

    payload = build_payload(args.latest_runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} with {len(payload['sources'])} sources")


if __name__ == "__main__":
    main()
