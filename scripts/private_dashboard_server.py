#!/usr/bin/env python3
"""
Private telemetry dashboard for this repo (Railway-friendly, no extra deps).

Security model:
- Viewer endpoints (`/`, `/api/summary`, `/api/events`) require HTTP Basic auth.
- Ingestion endpoint (`/api/ingest`) requires Bearer token.

Environment:
  FE_DASHBOARD_HOST            Bind host (default 0.0.0.0)
  FE_DASHBOARD_PORT            Bind port (default 8787)
  FE_DASHBOARD_DB_PATH         SQLite file path (default data/private_dashboard.sqlite3)
  FE_DASHBOARD_USER            Basic-auth user (default admin)
  FE_DASHBOARD_PASSWORD        Basic-auth password (required for production)
  FE_DASHBOARD_INGEST_TOKEN    Bearer token for POST /api/ingest (required)
"""

from __future__ import annotations

import base64
import html
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def _db_path() -> Path:
    raw = _env("FE_DASHBOARD_DB_PATH", str(REPO_ROOT / "data" / "private_dashboard.sqlite3"))
    return Path(raw).expanduser().resolve()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            command TEXT,
            exit_code INTEGER,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _json(status: str, payload: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]], bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    return status, headers, body


def _html(status: str, html: str) -> Tuple[str, List[Tuple[str, str]], bytes]:
    body = html.encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    return status, headers, body


def _read_body(environ: Dict[str, Any]) -> bytes:
    try:
        size = int(environ.get("CONTENT_LENGTH", "0") or "0")
    except ValueError:
        size = 0
    return environ["wsgi.input"].read(size if size > 0 else 0)


def _basic_ok(environ: Dict[str, Any]) -> bool:
    expected_user = _env("FE_DASHBOARD_USER", "admin")
    expected_pass = _env("FE_DASHBOARD_PASSWORD")
    if not expected_pass:
        return False
    auth = environ.get("HTTP_AUTHORIZATION", "")
    if not auth.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
    except Exception:
        return False
    if ":" not in raw:
        return False
    user, password = raw.split(":", 1)
    return user == expected_user and password == expected_pass


def _bearer_ok(environ: Dict[str, Any]) -> bool:
    token = _env("FE_DASHBOARD_INGEST_TOKEN")
    if not token:
        return False
    auth = environ.get("HTTP_AUTHORIZATION", "")
    return auth == f"Bearer {token}"


def _insert_event(conn: sqlite3.Connection, row: Dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO events (ts, source, kind, command, exit_code, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("ts") or _utc_now(),
            row.get("source", "unknown"),
            row.get("kind", "unknown"),
            row.get("command"),
            row.get("exit_code"),
            json.dumps(row.get("payload", {}), ensure_ascii=False),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    total = conn.execute("SELECT COUNT(1) FROM events").fetchone()[0]
    today = _utc_now()[:10]
    today_count = conn.execute("SELECT COUNT(1) FROM events WHERE substr(ts,1,10)=?", (today,)).fetchone()[0]
    failed = conn.execute("SELECT COUNT(1) FROM events WHERE exit_code IS NOT NULL AND exit_code != 0").fetchone()[0]
    rows = conn.execute(
        """
        SELECT ts, source, kind, command, exit_code
        FROM events
        ORDER BY id DESC
        LIMIT 40
        """
    ).fetchall()
    recent = [
        {
            "ts": r[0],
            "source": r[1],
            "kind": r[2],
            "command": r[3],
            "exit_code": r[4],
        }
        for r in rows
    ]
    return {
        "generated_at": _utc_now(),
        "events_total": total,
        "events_today": today_count,
        "events_failed": failed,
        "recent_events": recent,
    }


def _latest_docs_snippets() -> Dict[str, str]:
    snippets: Dict[str, str] = {}
    targets = {
        "project_state": REPO_ROOT / "docs" / "PROJECT_STATE.md",
        "session_log": REPO_ROOT / "docs" / "SESSION_LOG.md",
        "specialized_history": REPO_ROOT / "docs" / "SPECIALIZED_RUN_HISTORY.md",
    }
    for key, path in targets.items():
        if not path.is_file():
            snippets[key] = "(missing)"
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        snippets[key] = text[:2500]
    return snippets


def _load_scoring_summary() -> Dict[str, Any]:
    generated_dir = REPO_ROOT / "docs" / "generated"
    if not generated_dir.is_dir():
        return {"available": False, "reason": "docs/generated is missing"}
    candidates = sorted(generated_dir.glob("*.comparison.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {"available": False, "reason": "No comparison JSON found"}
    path = candidates[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False, "reason": f"Invalid JSON in {path.name}"}
    tracks = {}
    for key in ("codex_authored", "opensource", "specialized"):
        if key in payload and isinstance(payload[key], dict):
            tracks[key] = {
                "word_count": int(payload[key].get("word_count", 0) or 0),
                "keyword_hits": int(payload[key].get("keyword_hits", 0) or 0),
                "keywords": [str(k) for k in payload[key].get("keywords", [])],
            }
    if not tracks:
        return {"available": False, "reason": f"No scoring tracks in {path.name}"}
    winner = max(tracks.items(), key=lambda kv: (kv[1]["keyword_hits"], -kv[1]["word_count"]))[0]
    return {
        "available": True,
        "file": str(path.relative_to(REPO_ROOT)),
        "generated_utc": payload.get("generated_utc", ""),
        "model": payload.get("model", ""),
        "specialized_adapter": payload.get("specialized_adapter", ""),
        "tracks": tracks,
        "winner": winner,
    }


def _render_dashboard(summary: Dict[str, Any], docs_snippets: Dict[str, str], scoring: Dict[str, Any]) -> str:
    rows = []
    for item in summary["recent_events"]:
        cmd = _h(item.get("command") or "")
        rows.append(
            "<tr>"
            f"<td>{_h(item.get('ts',''))}</td>"
            f"<td>{_h(item.get('source',''))}</td>"
            f"<td>{_h(item.get('kind',''))}</td>"
            f"<td><code>{cmd}</code></td>"
            f"<td>{_h(item.get('exit_code',''))}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows) if rows else "<tr><td colspan='5'>(no events)</td></tr>"

    scoring_html = ""
    if scoring.get("available"):
        tracks = scoring["tracks"]
        labels = {
            "codex_authored": "Cursor/Codex Authored",
            "opensource": "Open-source Base",
            "specialized": "Specialized Adapter",
        }
        score_rows = []
        for key in ("codex_authored", "opensource", "specialized"):
            if key not in tracks:
                continue
            row = tracks[key]
            score_rows.append(
                "<tr>"
                f"<td>{_h(labels.get(key,key))}</td>"
                f"<td>{_h(row['keyword_hits'])}</td>"
                f"<td>{_h(row['word_count'])}</td>"
                f"<td>{_h(', '.join(row['keywords']))}</td>"
                "</tr>"
            )
        scoring_html = (
            f"<p><strong>Latest comparison:</strong> <code>{_h(scoring.get('file',''))}</code></p>"
            f"<p><strong>Winner by keyword-hit score:</strong> {_h(labels.get(scoring.get('winner',''), scoring.get('winner','')))}</p>"
            "<table>"
            "<thead><tr><th>Track</th><th>Keyword hits</th><th>Word count</th><th>Matched keywords</th></tr></thead>"
            f"<tbody>{''.join(score_rows)}</tbody>"
            "</table>"
        )
    else:
        scoring_html = (
            f"<p>No scoring artifact yet: {_h(scoring.get('reason','unknown'))}</p>"
            "<p>Generate one with your duplicate-doc run and place the JSON under <code>docs/generated/*.comparison.json</code>.</p>"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FE Private Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: #0b1020; color: #e5e7eb; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ border: 1px solid #334155; border-radius: 10px; padding: 12px; background: #111827; }}
    .tabs {{ display:flex; gap:8px; margin: 12px 0 16px 0; }}
    .tab {{ padding: 8px 12px; border:1px solid #334155; border-radius: 8px; background:#0f172a; color:#e5e7eb; cursor:pointer; }}
    .tab.active {{ background:#1d4ed8; border-color:#1d4ed8; }}
    .panel {{ display:none; }}
    .panel.active {{ display:block; }}
    h1,h2 {{ margin: 0 0 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; }}
    th,td {{ border: 1px solid #334155; padding: 8px; text-align: left; vertical-align: top; }}
    pre {{ white-space: pre-wrap; background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 10px; max-height: 240px; overflow: auto; }}
    a {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <h1>Fallen Empire Private Dashboard</h1>
  <p>Generated: {_h(summary['generated_at'])} UTC</p>

  <div class="tabs">
    <button class="tab active" data-panel="overview">Overview</button>
    <button class="tab" data-panel="scoring">Scoring vs Cursor Work</button>
    <button class="tab" data-panel="docs">Docs Snapshot</button>
  </div>

  <section id="panel-overview" class="panel active">
    <div class="cards">
      <div class="card"><strong>Total events</strong><div>{_h(summary['events_total'])}</div></div>
      <div class="card"><strong>Events today</strong><div>{_h(summary['events_today'])}</div></div>
      <div class="card"><strong>Failed events</strong><div>{_h(summary['events_failed'])}</div></div>
    </div>

    <h2>Recent Events</h2>
    <table>
      <thead><tr><th>UTC</th><th>Source</th><th>Kind</th><th>Command</th><th>Exit</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </section>

  <section id="panel-scoring" class="panel">
    <h2>Scoring Against Cursor Work</h2>
    <p>This tab compares generated docs against Cursor/Codex-authored reference scoring artifacts.</p>
    {scoring_html}
  </section>

  <section id="panel-docs" class="panel">
    <h2>Project Documentation Snapshot</h2>
    <p>This section mirrors committed docs in this repo deployment.</p>
    <h3>PROJECT_STATE</h3>
    <pre>{_h(docs_snippets.get('project_state',''))}</pre>
    <h3>SESSION_LOG</h3>
    <pre>{_h(docs_snippets.get('session_log',''))}</pre>
    <h3>SPECIALIZED_RUN_HISTORY</h3>
    <pre>{_h(docs_snippets.get('specialized_history',''))}</pre>
  </section>

  <script>
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const panels = Array.from(document.querySelectorAll('.panel'));
    for (const tab of tabs) {{
      tab.addEventListener('click', () => {{
        const target = tab.getAttribute('data-panel');
        for (const t of tabs) t.classList.remove('active');
        for (const p of panels) p.classList.remove('active');
        tab.classList.add('active');
        const panel = document.getElementById('panel-' + target);
        if (panel) panel.classList.add('active');
      }});
    }}
  </script>
</body>
</html>
"""


def app(environ: Dict[str, Any], start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    # Public health endpoint.
    if path == "/healthz":
        status, headers, body = _json("200 OK", {"ok": True, "ts": _utc_now()})
        start_response(status, headers)
        return [body]

    # Ingestion endpoint (token-based, no basic auth).
    if path == "/api/ingest" and method == "POST":
        if not _bearer_ok(environ):
            status, headers, body = _json("401 Unauthorized", {"ok": False, "error": "invalid bearer token"})
            start_response(status, headers)
            return [body]
        raw = _read_body(environ)
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid json"})
            start_response(status, headers)
            return [body]
        conn = _connect()
        try:
            event_id = _insert_event(
                conn,
                {
                    "ts": payload.get("ts") or _utc_now(),
                    "source": payload.get("source", "cursor"),
                    "kind": payload.get("kind", "event"),
                    "command": payload.get("command"),
                    "exit_code": payload.get("exit_code"),
                    "payload": payload,
                },
            )
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "event_id": event_id})
        start_response(status, headers)
        return [body]

    # Viewer endpoints require basic auth.
    if not _basic_ok(environ):
        status = "401 Unauthorized"
        body = b"Unauthorized"
        headers = [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("WWW-Authenticate", 'Basic realm="fe-private-dashboard"'),
        ]
        start_response(status, headers)
        return [body]

    conn = _connect()
    try:
        summary = _summary(conn)
    finally:
        conn.close()

    if path == "/api/summary":
        status, headers, body = _json("200 OK", summary)
        start_response(status, headers)
        return [body]

    if path == "/api/scoring":
        status, headers, body = _json("200 OK", {"ok": True, "scoring": _load_scoring_summary()})
        start_response(status, headers)
        return [body]

    if path == "/api/events":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        limit = 100
        if "limit" in qs:
            try:
                limit = max(1, min(1000, int(qs["limit"][0])))
            except (ValueError, TypeError):
                limit = 100
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, ts, source, kind, command, exit_code, payload_json
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        payload = []
        for row in rows:
            try:
                parsed_payload = json.loads(row[6])
            except json.JSONDecodeError:
                parsed_payload = {"_invalid_payload_json": True}
            payload.append(
                {
                    "id": row[0],
                    "ts": row[1],
                    "source": row[2],
                    "kind": row[3],
                    "command": row[4],
                    "exit_code": row[5],
                    "payload": parsed_payload,
                }
            )
        status, headers, body = _json("200 OK", {"ok": True, "events": payload})
        start_response(status, headers)
        return [body]

    docs_snippets = _latest_docs_snippets()
    scoring = _load_scoring_summary()
    html = _render_dashboard(summary, docs_snippets, scoring)
    status, headers, body = _html("200 OK", html)
    start_response(status, headers)
    return [body]


def main() -> None:
    host = _env("FE_DASHBOARD_HOST", "0.0.0.0")
    port = int(_env("FE_DASHBOARD_PORT", _env("PORT", "8787")))

    if not _env("FE_DASHBOARD_PASSWORD"):
        print("WARNING: FE_DASHBOARD_PASSWORD is not set; viewer auth will always fail.", file=sys.stderr)
    if not _env("FE_DASHBOARD_INGEST_TOKEN"):
        print("WARNING: FE_DASHBOARD_INGEST_TOKEN is not set; ingestion will fail.", file=sys.stderr)

    print(f"[private_dashboard_server] serving on http://{host}:{port}", file=sys.stderr)
    with make_server(host, port, app) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
