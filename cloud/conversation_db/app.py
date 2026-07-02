"""Fallen Empire Conversation DB — always-on cloud home for booked model debates.

A tiny FastAPI service meant to run on Railway (or any container host):

- ``POST /api/conversations``  — ingest a debate record from the local arena
  (Bearer token auth via ``FE_CONVERSATION_DB_TOKEN``).
- ``GET  /api/conversations``  — JSON list (id, topic, speakers, winner, …).
- ``GET  /api/conversations/{id}`` — full JSON record.
- ``GET  /``                   — website: searchable conversation table.
- ``GET  /c/{id}``             — website: full transcript + judge verdicts.

Storage is SQLite at ``$DATA_DIR/conversations.db`` (mount a Railway volume at
``/data`` for persistence). The chat itself stays on your computer — this service
only stores and displays what the arena pushes up.

Env vars:
- ``FE_CONVERSATION_DB_TOKEN`` — shared secret required on ingest (recommended).
  If unset, ingest is open (fine for a private test deploy, not for real use).
- ``DATA_DIR`` — directory for the SQLite file (default ``./data``).
- ``PORT`` — listen port (Railway sets this automatically).

Run locally:  uvicorn app:app --port 8090
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "conversations.db"
INGEST_TOKEN = (os.environ.get("FE_CONVERSATION_DB_TOKEN") or "").strip()

app = FastAPI(title="Fallen Empire Conversation DB", docs_url="/api/docs")


def _db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            received_at REAL NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            speakers TEXT NOT NULL DEFAULT '[]',
            winner TEXT,
            tie INTEGER NOT NULL DEFAULT 0,
            turns INTEGER NOT NULL DEFAULT 0,
            record TEXT NOT NULL
        )
        """
    )
    return conn


def _derive_topic(record: Dict[str, Any]) -> str:
    explicit = str(record.get("topic") or "").strip()
    if explicit:
        return explicit[:200]
    for item in record.get("transcript") or []:
        name = str(item.get("speaker_name") or item.get("speaker") or "")
        if name.strip().lower() == "human":
            content = " ".join(str(item.get("content") or "").split())
            if content:
                return content[:200]
    return "(no topic)"


def _speaker_names(record: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for s in record.get("speakers") or []:
        name = str(s.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


@app.post("/api/conversations")
async def ingest(request: Request, authorization: Optional[str] = Header(default=None)) -> JSONResponse:
    if INGEST_TOKEN:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if supplied != INGEST_TOKEN:
            raise HTTPException(status_code=401, detail="Bad or missing bearer token")
    try:
        record = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be JSON")
    if not isinstance(record, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    conv_id = str(record.get("debate_id") or record.get("id") or f"conv_{uuid.uuid4().hex[:12]}")
    transcript = record.get("transcript") or []
    row = (
        conv_id,
        str(record.get("created_at") or ""),
        time.time(),
        _derive_topic(record),
        json.dumps(_speaker_names(record)),
        record.get("winner"),
        1 if record.get("tie") else 0,
        len(transcript),
        json.dumps(record),
    )
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conversations "
            "(id, created_at, received_at, topic, speakers, winner, tie, turns, record) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    return JSONResponse({"ok": True, "id": conv_id, "url": f"/c/{conv_id}"})


def _list_rows(q: str = "", limit: int = 200) -> List[sqlite3.Row]:
    with _db() as conn:
        if q:
            like = f"%{q}%"
            cur = conn.execute(
                "SELECT id, created_at, received_at, topic, speakers, winner, tie, turns "
                "FROM conversations WHERE topic LIKE ? OR speakers LIKE ? OR record LIKE ? "
                "ORDER BY received_at DESC LIMIT ?",
                (like, like, like, limit),
            )
        else:
            cur = conn.execute(
                "SELECT id, created_at, received_at, topic, speakers, winner, tie, turns "
                "FROM conversations ORDER BY received_at DESC LIMIT ?",
                (limit,),
            )
        return cur.fetchall()


@app.get("/api/conversations")
def list_conversations(q: str = Query(default=""), limit: int = Query(default=200, le=1000)) -> JSONResponse:
    rows = _list_rows(q, limit)
    return JSONResponse(
        [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "topic": r["topic"],
                "speakers": json.loads(r["speakers"]),
                "winner": r["winner"],
                "tie": bool(r["tie"]),
                "turns": r["turns"],
            }
            for r in rows
        ]
    )


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str) -> JSONResponse:
    with _db() as conn:
        row = conn.execute("SELECT record FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(json.loads(row["record"]))


_PAGE_CSS = """
:root { --bg:#050506; --panel:#15181d; --border:rgba(228,219,196,.18); --text:#f6f2e9;
        --muted:#b9b2a4; --accent:#d6ad4b; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font:15px/1.55 -apple-system,'Segoe UI',sans-serif;
       margin:0; padding:2rem 1.25rem; }
.wrap { max-width:1000px; margin:0 auto; }
h1 { font-size:1.5rem; } h1 a { color:var(--text); text-decoration:none; }
.sub { color:var(--muted); margin-bottom:1.5rem; }
table { width:100%; border-collapse:collapse; background:var(--panel); border-radius:10px; overflow:hidden; }
th, td { text-align:left; padding:.6rem .8rem; border-bottom:1px solid var(--border); vertical-align:top; }
th { color:var(--accent); font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
tr:hover td { background:rgba(214,173,75,.06); }
a { color:var(--accent); }
.badge { display:inline-block; padding:.1rem .5rem; border:1px solid var(--border); border-radius:99px;
         font-size:.78rem; color:var(--muted); margin-right:.3rem; }
.winner { color:#8fd98f; border-color:#8fd98f; }
input[type=search] { background:#0b0d10; border:1px solid var(--border); color:var(--text);
         padding:.55rem .8rem; border-radius:8px; width:320px; margin-bottom:1rem; }
.turn { background:var(--panel); border:1px solid var(--border); border-radius:10px;
        padding:.8rem 1rem; margin:.6rem 0; }
.turn .who { color:var(--accent); font-weight:700; margin-bottom:.25rem; }
.turn.human .who { color:#8fb8d9; }
.verdict { border-left:3px solid var(--accent); padding-left:.8rem; margin:.6rem 0; color:var(--muted); }
.empty { color:var(--muted); padding:2rem; text-align:center; }
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )


@app.get("/", response_class=HTMLResponse)
def index(q: str = Query(default="")) -> str:
    rows = _list_rows(q)
    cells = []
    for r in rows:
        speakers = ", ".join(json.loads(r["speakers"])) or "—"
        outcome = (
            "<span class='badge'>tie</span>" if r["tie"]
            else (f"<span class='badge winner'>{html.escape(str(r['winner']))}</span>" if r["winner"] else "—")
        )
        when = html.escape((r["created_at"] or "")[:19].replace("T", " "))
        cells.append(
            f"<tr><td><a href='/c/{html.escape(r['id'])}'>{html.escape(r['topic'] or '(no topic)')}</a></td>"
            f"<td>{html.escape(speakers)}</td><td>{outcome}</td>"
            f"<td>{r['turns']}</td><td>{when}</td></tr>"
        )
    table = (
        "<table><tr><th>Topic</th><th>Speakers</th><th>Winner</th><th>Turns</th><th>Date (UTC)</th></tr>"
        + "".join(cells) + "</table>"
        if cells
        else "<div class='empty'>No conversations booked yet. Run a debate in the arena and it will appear here.</div>"
    )
    search = (
        "<form method='get'><input type='search' name='q' placeholder='Search topic, speaker, content…' "
        f"value='{html.escape(q)}'></form>"
    )
    body = (
        "<h1><a href='/'>Fallen Empire — Conversation DB</a></h1>"
        "<div class='sub'>Debates booked from the local Model Chat arena. "
        f"{len(rows)} conversation(s).</div>" + search + table
    )
    return _page("Conversation DB", body)


@app.get("/c/{conv_id}", response_class=HTMLResponse)
def conversation_page(conv_id: str) -> str:
    with _db() as conn:
        row = conn.execute("SELECT record FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    record = json.loads(row["record"])
    topic = _derive_topic(record)
    turns = []
    for item in record.get("transcript") or []:
        name = str(item.get("speaker_name") or item.get("speaker") or "?")
        klass = "turn human" if name.strip().lower() == "human" else "turn"
        content = html.escape(str(item.get("content") or "")).replace("\n", "<br>")
        turns.append(f"<div class='{klass}'><div class='who'>{html.escape(name)}</div>{content}</div>")
    verdicts = []
    for v in record.get("verdicts") or []:
        verdicts.append(
            f"<div class='verdict'><b>{html.escape(str(v.get('judge') or 'judge'))}</b> → "
            f"<b>{html.escape(str(v.get('winner') or 'unparsed'))}</b><br>"
            f"{html.escape(str(v.get('reasoning') or ''))}</div>"
        )
    tally = record.get("vote_tally") or {}
    outcome = "Tie" if tally.get("tie") else (str(tally.get("winner") or "") or "No verdict")
    body = (
        f"<h1><a href='/'>← Conversation DB</a></h1>"
        f"<div class='sub'>{html.escape(topic)}</div>"
        f"<p><span class='badge'>outcome: {html.escape(outcome)}</span>"
        f"<span class='badge'>{len(record.get('transcript') or [])} turns</span>"
        f"<span class='badge'>{html.escape(str(record.get('created_at') or '')[:19])}</span></p>"
        "<h3>Transcript</h3>" + "".join(turns)
        + ("<h3>Judge verdicts</h3>" + "".join(verdicts) if verdicts else "")
    )
    return _page(topic, body)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}
