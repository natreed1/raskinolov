#!/usr/bin/env python3
"""Hook: afterFileEdit -> trigger doc-training prep for watched codebase changes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
TRIGGER = REPO / "scripts" / "trigger_doc_training_on_changes.py"


def _extract_paths(payload: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower().endswith("path") and isinstance(value, str):
                paths.append(value)
            else:
                paths.extend(_extract_paths(value))
    elif isinstance(payload, list):
        for item in payload:
            paths.extend(_extract_paths(item))
    return paths


def _normalize_repo_rel(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_absolute():
        candidate = (REPO / p).resolve()
    else:
        candidate = p.resolve()
    try:
        rel = str(candidate.relative_to(REPO))
    except ValueError:
        return None
    return rel


def main() -> int:
    if os.environ.get("FE_LAB_FILEEDIT_TRAINING_TRIGGER", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return 0

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    rel_paths = set()
    for raw_path in _extract_paths(payload):
        rel = _normalize_repo_rel(raw_path)
        if rel:
            rel_paths.add(rel)

    if not rel_paths:
        return 0

    for rel in sorted(rel_paths):
        try:
            subprocess.run(
                [sys.executable, str(TRIGGER), "--path", rel, "--event", "afterFileEdit"],
                cwd=str(REPO),
                check=False,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
