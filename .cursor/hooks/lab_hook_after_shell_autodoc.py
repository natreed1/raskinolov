#!/usr/bin/env python3
"""
Auto-document test/benchmark shell runs from Cursor's afterShellExecution hook.

Opt-in via FE_LAB_AUTODOC_APPEND=1.

Behavior:
- Appends a JSONL event to lab_dashboard/shell_command_events.jsonl
- Appends a compact row to docs/SPECIALIZED_RUN_HISTORY.md
- Captures a lightweight git status snapshot (counts + file list)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS_PATH = REPO_ROOT / "lab_dashboard" / "shell_command_events.jsonl"
SPECIALIZED_HISTORY = REPO_ROOT / "docs" / "SPECIALIZED_RUN_HISTORY.md"
MAX_COMMAND_CHARS = 180
MAX_PATHS = 12

_TARGET_PATTERNS = [
    r"(^|\s)pytest(\s|$)",
    r"(^|\s)unittest(\s|$)",
    r"(^|\s)nose2(\s|$)",
    r"(^|\s)go test(\s|$)",
    r"(^|\s)cargo test(\s|$)",
    r"(^|\s)ctest(\s|$)",
    r"(^|\s)npm\s+test(\s|$)",
    r"(^|\s)pnpm\s+test(\s|$)",
    r"(^|\s)yarn\s+test(\s|$)",
    r"scripts/ml_workflow\.py",
    r"run_documentation_agent_benchmark\.py",
    r"run_routing_benchmark\.py",
    r"run_game_benchmark\.py",
]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_command(payload: Dict[str, Any]) -> str:
    for key in ("command", "shell_command", "input", "argv"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = " ".join(str(x) for x in value if x is not None).strip()
            if joined:
                return joined
    return ""


def _extract_exit_code(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("exitCode", "exit_code", "returncode", "return_code", "code"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _should_record(command: str) -> bool:
    if _truthy_env("FE_LAB_AUTODOC_INCLUDE_ALL"):
        return True
    for pattern in _TARGET_PATTERNS:
        if re.search(pattern, command):
            return True
    return False


def _git_change_snapshot() -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"tracked_modified": 0, "untracked": 0, "deleted": 0, "paths": []}

    tracked_modified = 0
    untracked = 0
    deleted = 0
    paths: List[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        code = line[:2]
        path = line[3:].strip() if len(line) > 3 else ""
        if path:
            paths.append(path)
        if code == "??":
            untracked += 1
            continue
        if "D" in code:
            deleted += 1
        if "M" in code or "A" in code or "R" in code or "C" in code:
            tracked_modified += 1

    return {
        "tracked_modified": tracked_modified,
        "untracked": untracked,
        "deleted": deleted,
        "paths": paths[:MAX_PATHS],
    }


def _append_specialized_row(
    utc_iso: str, command: str, exit_code: Optional[int], git_snap: Dict[str, Any]
) -> None:
    status = "ok" if exit_code == 0 else "failed"
    code_text = "—" if exit_code is None else str(exit_code)
    short_cmd = command if len(command) <= MAX_COMMAND_CHARS else command[:MAX_COMMAND_CHARS] + "..."
    summary = (
        f"exit {code_text}; "
        f"changes m/u/d={git_snap['tracked_modified']}/{git_snap['untracked']}/{git_snap['deleted']}"
    )
    artifacts = "`lab_dashboard/shell_command_events.jsonl`"
    notes = f"paths: {', '.join(git_snap['paths'])}" if git_snap["paths"] else "paths: (none)"
    row = (
        f"| {utc_iso} | cursor_shell_test | `{short_cmd}` | {code_text} | {status} | "
        f"{summary} | {artifacts} | {notes} |\n"
    )

    if not SPECIALIZED_HISTORY.is_file():
        SPECIALIZED_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        SPECIALIZED_HISTORY.write_text(
            "# Specialized run history (auxiliary benchmarks)\n\n"
            "| UTC ISO | Track | Entry point | Exit | Status | Summary | Artifacts | Notes |\n"
            "|---------|-------|-------------|-----:|--------|---------|-----------|-------|\n",
            encoding="utf-8",
        )
    with SPECIALIZED_HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(row)


def _post_remote_event(event: Dict[str, Any]) -> None:
    url = os.environ.get("FE_LAB_REMOTE_INGEST_URL", "").strip()
    token = os.environ.get("FE_LAB_REMOTE_INGEST_TOKEN", "").strip()
    if not url or not token:
        return
    payload = json.dumps(
        {
            "ts": event.get("ts"),
            "source": "cursor_hook",
            "kind": event.get("kind", "after_shell"),
            "command": event.get("command"),
            "exit_code": event.get("exit_code"),
            "payload": event,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with request.urlopen(req, timeout=4):  # noqa: S310
            pass
    except Exception:
        # Fail-open: local logging should never break shell execution.
        return


def main() -> int:
    if not _truthy_env("FE_LAB_AUTODOC_APPEND"):
        return 0

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {"_parse_error": "invalid_json", "_raw": raw[:1500]}

    command = _extract_command(payload)
    if not command or not _should_record(command):
        return 0

    git_snap = _git_change_snapshot()
    exit_code = _extract_exit_code(payload)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    event = {
        "ts": now_iso,
        "kind": "after_shell_execution",
        "hook": "afterShellExecution",
        "command": command,
        "exit_code": exit_code,
        "git": git_snap,
        "payload": payload,
    }
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    _append_specialized_row(now_iso, command, exit_code, git_snap)
    _post_remote_event(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
