#!/usr/bin/env python3
"""Trigger documentation training prep from codebase changes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
QUEUE_PATH = REPO / "data" / "training_triggers" / "documentation_training_queue.jsonl"
STATE_PATH = REPO / ".cursor" / "hooks" / ".doc_training_trigger_state.json"
WATCH_PREFIXES = ("docs/", "scripts/", "benchmarks/", "tests/", "training/")
TRAINING_PAGE_PREFIXES = (
    "docs/WORKFLOW.md",
    "training/README.md",
    "training/",
    "scripts/ml_workflow.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_watched(path: str) -> bool:
    return path.startswith(WATCH_PREFIXES)


def _is_training_page_change(path: str) -> bool:
    return path.startswith(TRAINING_PAGE_PREFIXES)


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    return int(proc.returncode)


def _append_queue(row: dict[str, Any]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"seen": {}, "last_ts": ""}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"seen": {}, "last_ts": ""}


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger documentation training prep on file changes.")
    parser.add_argument("--path", required=True, help="Changed file path relative to repo root.")
    parser.add_argument("--event", default="afterFileEdit")
    args = parser.parse_args()

    rel = args.path.strip().lstrip("./")
    if not rel or not _is_watched(rel):
        return

    abs_path = (REPO / rel).resolve()
    if not abs_path.exists():
        return

    mtime = abs_path.stat().st_mtime
    state = _load_state()
    prev = state.get("seen", {}).get(rel)
    if prev is not None and float(prev) == float(mtime):
        return
    state.setdefault("seen", {})[rel] = mtime
    now_epoch = int(time.time())
    min_interval_s = int(os.environ.get("FE_LAB_DOC_TRIGGER_MIN_SECONDS", "900") or "900")
    last_epoch = int(state.get("last_trigger_epoch", 0) or 0)
    if now_epoch - last_epoch < min_interval_s:
        state["last_ts"] = _now()
        _save_state(state)
        _append_queue(
            {
                "ts": _now(),
                "trigger": args.event,
                "changed_path": rel,
                "watched": True,
                "skipped_due_cooldown": True,
                "min_interval_seconds": min_interval_s,
                "seconds_until_next_trigger": max(0, min_interval_s - (now_epoch - last_epoch)),
            }
        )
        return
    state["last_ts"] = _now()
    state["last_trigger_epoch"] = now_epoch
    _save_state(state)

    build_run_analysis = [sys.executable, str(REPO / "scripts" / "build_run_analysis_rag_corpus.py")]
    rag_rc = _run(build_run_analysis)

    auto_dataset = os.environ.get("FE_LAB_AUTODOC_DATASET_ON_CHANGE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ds_rc = None
    if auto_dataset:
        ds_rc = _run([sys.executable, str(REPO / "scripts" / "ml_workflow.py"), "documentation-dataset"])

    dashboard_cache_rc = None
    if _is_training_page_change(rel):
        dashboard_cache_rc = _run([sys.executable, str(REPO / "scripts" / "build_training_data_dashboard_cache.py")])

    row = {
        "ts": _now(),
        "trigger": args.event,
        "changed_path": rel,
        "watched": True,
        "actions": {
            "build_run_analysis_rag_corpus": {"exit_code": rag_rc},
            "documentation_dataset": {"enabled": auto_dataset, "exit_code": ds_rc},
            "training_dashboard_cache": {
                "enabled_on_training_page_change": _is_training_page_change(rel),
                "exit_code": dashboard_cache_rc,
            },
        },
        "recommended_next": "python scripts/ml_workflow.py documentation-rag-benchmark --skip-no-rag-baseline",
    }
    _append_queue(row)


if __name__ == "__main__":
    main()
