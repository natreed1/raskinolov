#!/usr/bin/env python3
"""
Small task runner tailored to this MLX lab repo.

Primary goal: delegate long **local** ML pipelines to ``scripts/ml_workflow.py`` instead of
holding an expensive interactive Cursor thread open.

The default ``learning`` task maps to the repo's quickest safe end-to-end check
(``ml_workflow.py smoke``). Use ``--sequence full`` for prepare→train→benchmark when
``SOURCE_REPO`` and data paths are wired (see docs/WORKFLOW.md).

Task ``documentation-rag-benchmark`` runs ``ml_workflow.py documentation-rag-benchmark``
(documentation-agent RAG eval + telemetry; see ``docs/SPECIALIZED_RUN_HISTORY.md``).

Completing a learning run appends ``lab_dashboard/agent_events.jsonl`` so
``scripts/build_lab_optimization_dashboard.py`` can show a coarse optimization timeline.

Env:
  FE_ML_LAB_SPARED_USD  Optional heuristic (float) appended to events as ``est_cursor_usd_spared``
                        for bookkeeping only — not Cursor billing telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
VENV_BIN = REPO / ".venv" / "bin"


def _ensure_repo_venv() -> None:
    """Re-exec through ``.venv/bin/python`` when it exists."""
    expected = VENV_BIN / "python"
    if not expected.is_file():
        return
    current = Path(sys.executable).resolve()
    if current == expected.resolve():
        return
    if os.environ.get("FE_ML_LAB_RUNNER_REEXEC") == str(expected.resolve()):
        raise SystemExit(f"Refusing repeated lab-runner re-exec to {expected}")
    env = {**os.environ, "FE_ML_LAB_RUNNER_REEXEC": str(expected.resolve())}
    print(f"Re-executing with repo venv: {expected}", file=sys.stderr)
    os.execve(str(expected), [str(expected), *sys.argv], env)


def _append_event(repo: Path, record: Dict[str, Any]) -> None:
    out_dir = repo / "lab_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "agent_events.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cmd_documentation_rag_benchmark(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Run ml_workflow.py documentation-rag-benchmark (RAG eval + optional baseline telemetry).",
    )
    p.add_argument(
        "--no-event",
        action="store_true",
        help="Do not append lab_dashboard/agent_events.jsonl.",
    )
    args, passthrough = p.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    spare_raw = os.environ.get("FE_ML_LAB_SPARED_USD", "0").strip()
    try:
        spared = float(spare_raw)
    except ValueError:
        spared = 0.0

    wf = sys.executable
    cmd: List[str] = [wf, str(REPO / "scripts" / "ml_workflow.py"), "documentation-rag-benchmark"]
    cmd.extend(passthrough)

    t0 = time.perf_counter()
    started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proc = subprocess.run(cmd, cwd=str(REPO))
    elapsed_s = round(time.perf_counter() - t0, 3)
    rc = proc.returncode

    if not args.no_event:
        _append_event(
            REPO,
            {
                "kind": "documentation_rag_benchmark",
                "started_utc_iso": started_iso,
                "exit_code": rc,
                "elapsed_s": elapsed_s,
                "argv": cmd,
                "est_cursor_usd_spared": spared,
                "notes": "See docs/SPECIALIZED_RUN_HISTORY.md and benchmarks/results/documentation_rag_timeseries.jsonl",
            },
        )
    print(
        f"[fe_ml_lab_runner] documentation-rag-benchmark exit={rc}, {elapsed_s}s",
        file=sys.stderr,
    )
    return rc


def cmd_learning(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Run a standard ML workflow learning sequence.")
    p.add_argument(
        "--sequence",
        choices=("smoke", "full"),
        default="smoke",
        help="Workflow subcommand passed to ml_workflow.py (default smoke = CI-fast).",
    )
    p.add_argument(
        "--no-event",
        action="store_true",
        help="Do not append lab_dashboard/agent_events.jsonl.",
    )
    args, passthrough = p.parse_known_args(argv)

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    spare_raw = os.environ.get("FE_ML_LAB_SPARED_USD", "0").strip()
    try:
        spared = float(spare_raw)
    except ValueError:
        spared = 0.0

    wf = sys.executable
    cmd: List[str] = [wf, str(REPO / "scripts" / "ml_workflow.py"), args.sequence]
    cmd.extend(passthrough)

    t0 = time.perf_counter()
    started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    merged = dict(os.environ)
    proc = subprocess.run(cmd, cwd=str(REPO), env=merged)
    elapsed_s = round(time.perf_counter() - t0, 3)
    rc = proc.returncode

    if not args.no_event:
        _append_event(
            REPO,
            {
                "kind": "ml_workflow_learning",
                "started_utc_iso": started_iso,
                "sequence": args.sequence,
                "exit_code": rc,
                "elapsed_s": elapsed_s,
                "argv": cmd,
                "est_cursor_usd_spared": spared,
                "notes": "",
            },
        )
    print(f"[fe_ml_lab_runner] learning (--sequence {args.sequence}) exit={rc}, {elapsed_s}s", file=sys.stderr)
    return rc


def main() -> None:
    _ensure_repo_venv()
    root = argparse.ArgumentParser(
        prog="fe_ml_lab_runner.py",
        description="Terminal-side runner for MLX lab workflows (reduce Cursor-chat burn).",
    )
    root.add_argument(
        "task",
        choices=("learning", "documentation-rag-benchmark"),
        nargs="?",
        default="learning",
        help='Task name (default "learning"). Use documentation-rag-benchmark for doc-agent RAG eval.',
    )
    args, forwarded = root.parse_known_args()

    if args.task == "learning":
        raise SystemExit(cmd_learning(forwarded))
    if args.task == "documentation-rag-benchmark":
        raise SystemExit(cmd_documentation_rag_benchmark(forwarded))

    raise SystemExit(2)


if __name__ == "__main__":
    main()
