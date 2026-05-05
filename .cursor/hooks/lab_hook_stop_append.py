#!/usr/bin/env python3
"""
Append a tiny JSON row when Cursor fires the workspace ``stop`` hook.

Opt-in via env **`FE_LAB_CURSOR_HOOK_APPEND=1`** so clones do not silently grow ledger files.

Never runs MLX; safe to wire on every agent stop. Optionally rebuild
``lab_dashboard/index.html`` when ``FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD=1`` (still cheap).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS = REPO_ROOT / "lab_dashboard" / "cursor_hook_events.jsonl"
MAX_EMBED_CHARS = 4000


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    if not _truthy_env("FE_LAB_CURSOR_HOOK_APPEND"):
        return 0

    raw = sys.stdin.read()
    payload: dict
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        snippet = raw[:MAX_EMBED_CHARS]
        payload = {"_parse_error": "invalid_json", "_snippet": snippet}

    trimmed = payload
    if raw and len(raw) > MAX_EMBED_CHARS:
        trimmed = dict(payload)
        trimmed["_truncated_stdio"] = True
        trimmed["_stdio_len"] = len(raw)

    row = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hook": "stop",
        "payload": trimmed,
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    remote_url = os.environ.get("FE_LAB_REMOTE_INGEST_URL", "").strip()
    remote_token = os.environ.get("FE_LAB_REMOTE_INGEST_TOKEN", "").strip()
    if remote_url and remote_token:
        payload = json.dumps(
            {
                "ts": row["ts"],
                "source": "cursor_hook",
                "kind": "stop_hook",
                "payload": row,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            url=remote_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {remote_token}",
            },
        )
        try:
            with request.urlopen(req, timeout=4):  # noqa: S310
                pass
        except Exception:
            # Fail-open on network errors.
            pass

    if _truthy_env("FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD"):
        try:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "build_lab_optimization_dashboard.py")],
                cwd=str(REPO_ROOT),
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
