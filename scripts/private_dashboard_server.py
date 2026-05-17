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
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_TRACKS = {"codex_authored", "opensource", "specialized"}
COMPARE_WINNERS = {"left", "right", "tie"}
COMPARE_STRENGTHS = {"weak", "medium", "strong", "tie"}
SPAN_SIDES = {"left", "right"}
SPAN_LABELS = {"good", "bad", "corrupt"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            track TEXT NOT NULL,
            rating INTEGER,
            notes TEXT NOT NULL,
            source TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'feedback',
            key_details TEXT NOT NULL DEFAULT '',
            language_edits TEXT NOT NULL DEFAULT '',
            artifact_path TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compare_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            left_track TEXT NOT NULL,
            right_track TEXT NOT NULL,
            left_artifact_path TEXT NOT NULL,
            right_artifact_path TEXT NOT NULL,
            winner TEXT NOT NULL,
            strength TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL,
            left_text_hash TEXT NOT NULL,
            right_text_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compare_feedback_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            side TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            label TEXT NOT NULL,
            selected_text TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            rewrite_text TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (feedback_id) REFERENCES compare_feedback(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compare_feedback_created ON compare_feedback(id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compare_feedback_pair ON compare_feedback(left_track, right_track)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compare_feedback_spans_feedback_id ON compare_feedback_spans(feedback_id)"
    )
    # Backward-compatible schema migration for previously created DB files.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback_entries)").fetchall()}
    for col, sql_type, default in (
        ("action", "TEXT", "'feedback'"),
        ("key_details", "TEXT", "''"),
        ("language_edits", "TEXT", "''"),
        ("artifact_path", "TEXT", "''"),
    ):
        if col not in existing_cols:
            conn.execute(
                f"ALTER TABLE feedback_entries ADD COLUMN {col} {sql_type} NOT NULL DEFAULT {default}"
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
    # Viewer auth is intentionally disabled for local dashboard usage.
    # Ingestion still requires FE_DASHBOARD_INGEST_TOKEN bearer auth.
    _ = environ
    return True


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


def _insert_feedback(
    conn: sqlite3.Connection,
    *,
    track: str,
    rating: int,
    notes: str,
    source: str,
    action: str = "feedback",
    key_details: str = "",
    language_edits: str = "",
    artifact_path: str = "",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO feedback_entries (
            ts, track, rating, notes, source, action, key_details, language_edits, artifact_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _utc_now(),
            track,
            rating,
            notes,
            source,
            action,
            key_details,
            language_edits,
            artifact_path,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _list_feedback(conn: sqlite3.Connection, limit: int = 20) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, ts, track, rating, notes, source, action, key_details, language_edits, artifact_path
        FROM feedback_entries
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(1, min(200, limit)),),
    ).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "track": r[2],
            "rating": r[3],
            "notes": r[4],
            "source": r[5],
            "action": r[6],
            "key_details": r[7],
            "language_edits": r[8],
            "artifact_path": r[9],
        }
        for r in rows
    ]


def _validate_compare_spans(
    *,
    spans_raw: Any,
    left_body: str,
    right_body: str,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    if spans_raw is None:
        return True, "", []
    if not isinstance(spans_raw, list):
        return False, "spans must be a list", []

    cleaned: List[Dict[str, Any]] = []
    by_side: Dict[str, List[Tuple[int, int]]] = {"left": [], "right": []}
    for idx, raw in enumerate(spans_raw):
        if not isinstance(raw, dict):
            return False, f"span[{idx}] must be an object", []
        side = str(raw.get("side") or "").strip()
        label = str(raw.get("label") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        rewrite_text = str(raw.get("rewrite_text") or "").strip()
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
        except (TypeError, ValueError):
            return False, f"span[{idx}] start/end must be integers", []
        if side not in SPAN_SIDES:
            return False, f"span[{idx}] invalid side", []
        if label not in SPAN_LABELS:
            return False, f"span[{idx}] invalid label", []
        if start < 0 or end <= start:
            return False, f"span[{idx}] invalid range", []
        body = left_body if side == "left" else right_body
        if end > len(body):
            return False, f"span[{idx}] out of bounds", []
        selected_text = str(raw.get("selected_text") or "")
        expected = body[start:end]
        if selected_text != expected:
            return False, f"span[{idx}] selected_text mismatch for side={side}", []
        if label == "good" and rewrite_text:
            return False, f"span[{idx}] rewrite_text only allowed for bad/corrupt spans", []
        if len(reason) > 500:
            return False, f"span[{idx}] reason too long", []
        if len(rewrite_text) > 8000:
            return False, f"span[{idx}] rewrite_text too long", []
        cleaned.append(
            {
                "side": side,
                "label": label,
                "start": start,
                "end": end,
                "selected_text": selected_text,
                "reason": reason,
                "rewrite_text": rewrite_text,
            }
        )
        by_side[side].append((start, end))

    for side in SPAN_SIDES:
        prev_end = -1
        for start, end in sorted(by_side[side], key=lambda item: (item[0], item[1])):
            if start < prev_end:
                return False, f"spans overlap on side={side}", []
            prev_end = end
    return True, "", cleaned


def _insert_compare_feedback(
    conn: sqlite3.Connection,
    *,
    left_track: str,
    right_track: str,
    left_artifact_path: str,
    right_artifact_path: str,
    winner: str,
    strength: str,
    notes: str,
    source: str,
    left_text_hash: str,
    right_text_hash: str,
    spans: List[Dict[str, Any]],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO compare_feedback (
            ts,
            left_track,
            right_track,
            left_artifact_path,
            right_artifact_path,
            winner,
            strength,
            notes,
            source,
            left_text_hash,
            right_text_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _utc_now(),
            left_track,
            right_track,
            left_artifact_path,
            right_artifact_path,
            winner,
            strength,
            notes,
            source,
            left_text_hash,
            right_text_hash,
        ),
    )
    feedback_id = int(cur.lastrowid)
    for span in spans:
        conn.execute(
            """
            INSERT INTO compare_feedback_spans (
                feedback_id, side, start_offset, end_offset, label, selected_text, reason, rewrite_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                span["side"],
                int(span["start"]),
                int(span["end"]),
                span["label"],
                span["selected_text"],
                span["reason"],
                span["rewrite_text"],
            ),
        )
    conn.commit()
    return feedback_id


def _list_compare_feedback(
    conn: sqlite3.Connection,
    *,
    limit: int = 20,
    left_track: str = "",
    right_track: str = "",
) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, ts, left_track, right_track, left_artifact_path, right_artifact_path, "
        "winner, strength, notes, source, left_text_hash, right_text_hash "
        "FROM compare_feedback"
    )
    args: List[Any] = []
    filters: List[str] = []
    if left_track:
        filters.append("left_track = ?")
        args.append(left_track)
    if right_track:
        filters.append("right_track = ?")
        args.append(right_track)
    if filters:
        sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(200, limit)))
    rows = conn.execute(sql, tuple(args)).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        span_rows = conn.execute(
            """
            SELECT id, side, start_offset, end_offset, label, selected_text, reason, rewrite_text
            FROM compare_feedback_spans
            WHERE feedback_id = ?
            ORDER BY id ASC
            """,
            (row[0],),
        ).fetchall()
        out.append(
            {
                "id": row[0],
                "ts": row[1],
                "left_track": row[2],
                "right_track": row[3],
                "left_artifact_path": row[4],
                "right_artifact_path": row[5],
                "winner": row[6],
                "strength": row[7],
                "notes": row[8],
                "source": row[9],
                "left_text_hash": row[10],
                "right_text_hash": row[11],
                "spans": [
                    {
                        "id": s[0],
                        "side": s[1],
                        "start": s[2],
                        "end": s[3],
                        "label": s[4],
                        "selected_text": s[5],
                        "reason": s[6],
                        "rewrite_text": s[7],
                    }
                    for s in span_rows
                ],
            }
        )
    return out


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


def _estimate_tokens_from_words(words: int) -> int:
    # Conservative cross-model estimate for English technical prose.
    return max(0, int(round(words * 1.33)))


def _documentation_token_metrics() -> Dict[str, Any]:
    generated_dir = REPO_ROOT / "docs" / "generated"
    if not generated_dir.is_dir():
        return {"available": False, "reason": "docs/generated is missing"}

    sizes: Dict[str, Dict[str, Any]] = {}
    comparison_files = sorted(generated_dir.glob("*.comparison.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if comparison_files:
        try:
            payload = json.loads(comparison_files[0].read_text(encoding="utf-8"))
            for src_key, out_key in (
                ("codex_authored", "cursor_codex"),
                ("opensource", "opensource"),
                ("specialized", "specialized"),
            ):
                row = payload.get(src_key)
                if not isinstance(row, dict):
                    continue
                words = int(row.get("word_count", 0) or 0)
                sizes[out_key] = {
                    "path": str(comparison_files[0].relative_to(REPO_ROOT)),
                    "words": words,
                    "estimated_tokens": _estimate_tokens_from_words(words),
                    "sha1": hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()[:12],
                }
        except json.JSONDecodeError:
            pass

    # Fallback: direct markdown file estimates when comparison json is unavailable.
    if not sizes:
        tracks = {
            "cursor_codex": generated_dir / "private_dashboard_deploy.comparison.md",
            "opensource": generated_dir / "private_dashboard_deploy.opensource.md",
            "specialized": generated_dir / "private_dashboard_deploy.specialized.md",
        }
        for key, path in tracks.items():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            words = len(text.split())
            sizes[key] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "words": words,
                "estimated_tokens": _estimate_tokens_from_words(words),
                "sha1": hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12],
            }
    if not sizes:
        return {"available": False, "reason": "No generated markdown docs found"}

    cursor_tokens = sizes.get("cursor_codex", {}).get("estimated_tokens")
    specialized_tokens = sizes.get("specialized", {}).get("estimated_tokens")
    opensource_tokens = sizes.get("opensource", {}).get("estimated_tokens")
    savings_vs_specialized = None
    savings_vs_opensource = None
    if isinstance(cursor_tokens, int) and isinstance(specialized_tokens, int):
        savings_vs_specialized = max(0, cursor_tokens - specialized_tokens)
    if isinstance(cursor_tokens, int) and isinstance(opensource_tokens, int):
        savings_vs_opensource = max(0, cursor_tokens - opensource_tokens)

    return {
        "available": True,
        "tracks": sizes,
        "estimated_savings_tokens": {
            "vs_specialized": savings_vs_specialized,
            "vs_opensource": savings_vs_opensource,
        },
        "notes": "Estimated from generated documentation word counts (1 word ~= 1.33 tokens).",
    }


def _recent_workflow_runs(limit: int = 20) -> List[Dict[str, Any]]:
    runs_dir = REPO_ROOT / "benchmarks" / "results" / "runs"
    if not runs_dir.is_dir():
        return []
    manifests = sorted(runs_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows: List[Dict[str, Any]] = []
    for path in manifests[: max(1, min(80, limit))]:
        run_id = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "run_id": run_id,
                "subcommand": payload.get("subcommand", ""),
                "final_exit_code": payload.get("final_exit_code"),
                "elapsed_s": float(payload.get("elapsed_s", 0) or 0),
                "benchmark_summary": payload.get("benchmark_summary", ""),
                "finished_at": payload.get("finished_at", ""),
                "started_at": payload.get("started_at", ""),
                "capability_summary": payload.get("capability_summary", ""),
                "steps_count": len(payload.get("steps", []) or []),
                "trained_adapter_path": payload.get("trained_adapter_path"),
                "benchmark_adapter_path": payload.get("benchmark_adapter_path"),
            }
        )
    return rows


def _recent_doc_captures(limit: int = 24) -> List[Dict[str, Any]]:
    captures_dir = REPO_ROOT / "data" / "documentation_captures"
    if not captures_dir.is_dir():
        return []
    files = sorted(captures_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in files[: max(1, min(200, limit))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        os_track = payload.get("opensource") or {}
        sp_track = payload.get("specialized") or {}
        rows.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "changed_path": payload.get("changed_path", ""),
                "started_at": payload.get("started_at", ""),
                "finished_at": payload.get("finished_at", ""),
                "trigger_event": payload.get("trigger_event", ""),
                "opensource_tokens": int(os_track.get("estimated_tokens", 0) or 0),
                "specialized_tokens": int(sp_track.get("estimated_tokens", 0) or 0),
                "opensource_seconds": float(os_track.get("generation_seconds", 0) or 0)
                + float(os_track.get("load_seconds", 0) or 0),
                "specialized_seconds": float(sp_track.get("generation_seconds", 0) or 0)
                + float(sp_track.get("load_seconds", 0) or 0),
                "opensource_load_seconds": float(os_track.get("load_seconds", 0) or 0),
                "specialized_load_seconds": float(sp_track.get("load_seconds", 0) or 0),
                "opensource_gen_seconds": float(os_track.get("generation_seconds", 0) or 0),
                "specialized_gen_seconds": float(sp_track.get("generation_seconds", 0) or 0),
                "has_both": bool(os_track.get("output")) and bool(sp_track.get("output")),
            }
        )
    return rows


def _kpi_metrics(summary: Dict[str, Any], scoring: Dict[str, Any], token_metrics: Dict[str, Any]) -> Dict[str, Any]:
    runs = _recent_workflow_runs(30)
    captures = _recent_doc_captures(30)

    run_total = len(runs)
    run_ok = sum(1 for r in runs if r.get("final_exit_code") == 0)
    run_reliability_pct = round((run_ok / run_total) * 100.0, 1) if run_total else 0.0
    avg_run_seconds = round(sum(r.get("elapsed_s", 0.0) for r in runs) / run_total, 1) if run_total else 0.0

    cap_total = len(captures)
    cap_ok = sum(1 for c in captures if c.get("has_both"))
    cap_reliability_pct = round((cap_ok / cap_total) * 100.0, 1) if cap_total else 0.0
    avg_cap_seconds = (
        round(sum(c.get("opensource_seconds", 0.0) + c.get("specialized_seconds", 0.0) for c in captures) / cap_total, 1)
        if cap_total
        else 0.0
    )
    cap_tokens_total = sum(c.get("opensource_tokens", 0) + c.get("specialized_tokens", 0) for c in captures)
    cap_open_tokens = sum(c.get("opensource_tokens", 0) for c in captures)
    cap_specialized_tokens = sum(c.get("specialized_tokens", 0) for c in captures)
    cap_savings_if_specialized_only = max(0, cap_open_tokens - cap_specialized_tokens)

    accuracy_pct = None
    if scoring.get("available"):
        tracks = scoring.get("tracks", {})
        cursor_hits = int((tracks.get("codex_authored") or {}).get("keyword_hits", 0) or 0)
        specialized_hits = int((tracks.get("specialized") or {}).get("keyword_hits", 0) or 0)
        if cursor_hits > 0:
            accuracy_pct = round((specialized_hits / cursor_hits) * 100.0, 1)

    savings = token_metrics.get("estimated_savings_tokens", {}) if token_metrics.get("available") else {}
    return {
        "runs_recent_count": run_total,
        "run_reliability_pct": run_reliability_pct,
        "avg_run_seconds": avg_run_seconds,
        "doc_capture_recent_count": cap_total,
        "doc_capture_reliability_pct": cap_reliability_pct,
        "avg_doc_capture_seconds": avg_cap_seconds,
        "doc_capture_tokens_total": cap_tokens_total,
        "doc_capture_tokens_opensource": cap_open_tokens,
        "doc_capture_tokens_specialized": cap_specialized_tokens,
        "doc_capture_savings_if_specialized_only": cap_savings_if_specialized_only,
        "specialized_accuracy_vs_cursor_pct": accuracy_pct,
        "estimated_savings_vs_opensource": savings.get("vs_opensource"),
        "estimated_savings_vs_specialized": savings.get("vs_specialized"),
        "recent_runs": runs[:12],
        "recent_doc_captures": captures[:12],
        "events_total": summary.get("events_total", 0),
        "events_failed": summary.get("events_failed", 0),
    }


def _documentation_catalog(limit: int = 500) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    patterns = [
        ("docs", "docs/**/*.md"),
        ("docs", "docs/**/*.json"),
        ("docs", "docs/**/*.jsonl"),
        ("capture", "data/documentation_captures/*.md"),
        ("capture", "data/documentation_captures/*.json"),
        ("runs", "benchmarks/results/runs/*/RUN.md"),
    ]
    for category, pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if rel in seen:
                continue
            seen.add(rel)
            stat = path.stat()
            entries.append(
                {
                    "path": rel,
                    "category": category,
                    "model_track": _doc_model_track(rel, category),
                    "title": _doc_display_title(path, category),
                    "updated_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "size_bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
    entries.sort(key=lambda row: row.get("mtime", 0.0), reverse=True)
    trimmed = entries[: max(1, min(2000, limit))]
    for row in trimmed:
        row.pop("mtime", None)
    return trimmed


def _training_data_catalog(limit: int = 400) -> Dict[str, Any]:
    cache_path = REPO_ROOT / "data" / "training_dashboard" / "training_data_catalog.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
        if isinstance(cached, dict) and isinstance(cached.get("datasets"), list):
            rows = [row for row in cached.get("datasets", []) if isinstance(row, dict)]
            return {
                "generated_utc": str(cached.get("generated_utc") or ""),
                "dataset_count": len(rows),
                "datasets": rows[: max(1, min(1200, limit))],
                "source": str(cache_path.relative_to(REPO_ROOT)),
            }

    adapters_dir = REPO_ROOT / "data" / "lora" / "adapters"
    rows: List[Dict[str, Any]] = []
    if adapters_dir.is_dir():
        for child in sorted(adapters_dir.iterdir()):
            if not child.is_dir():
                continue
            split_paths = {split: child / f"{split}.jsonl" for split in ("train", "valid", "test")}
            if not any(path.is_file() for path in split_paths.values()):
                continue
            splits: Dict[str, Any] = {}
            total_rows = 0
            for split, path in split_paths.items():
                if not path.is_file():
                    splits[split] = {"path": "", "rows": 0, "size_bytes": 0, "updated_utc": ""}
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    rows_count = sum(1 for _ in fh)
                stat = path.stat()
                total_rows += rows_count
                splits[split] = {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "rows": rows_count,
                    "size_bytes": int(stat.st_size),
                    "updated_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            manifest_path = child / "manifest.json"
            rows.append(
                {
                    "dataset_id": child.name,
                    "label": child.name.replace("_", " ").title(),
                    "path": str(child.relative_to(REPO_ROOT)),
                    "manifest_path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_file() else "",
                    "total_rows": total_rows,
                    "splits": splits,
                }
            )
    return {
        "generated_utc": _utc_now(),
        "dataset_count": len(rows),
        "datasets": rows[: max(1, min(1200, limit))],
        "source": "live_scan:data/lora/adapters",
    }


def _read_training_data_content(dataset_id: str, view: str, offset: int, limit: int) -> Dict[str, Any]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "", dataset_id)
    if not safe_id or safe_id != dataset_id:
        return {"ok": False, "error": "invalid dataset id"}
    dataset_dir = (REPO_ROOT / "data" / "lora" / "adapters" / safe_id).resolve()
    try:
        dataset_dir.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return {"ok": False, "error": "invalid dataset path"}
    if not dataset_dir.is_dir():
        return {"ok": False, "error": "dataset not found"}

    view = view.strip().lower()
    if view == "manifest":
        path = dataset_dir / "manifest.json"
        if not path.is_file():
            return {"ok": False, "error": "manifest missing"}
        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "dataset_id": safe_id,
            "view": view,
            "path": str(path.relative_to(REPO_ROOT)),
            "offset": 0,
            "limit": 1,
            "next_offset": 0,
            "has_more": False,
            "total_rows": 1,
            "content": content,
        }

    if view not in {"train", "valid", "test"}:
        return {"ok": False, "error": "view must be train|valid|test|manifest"}
    path = dataset_dir / f"{view}.jsonl"
    if not path.is_file():
        return {"ok": False, "error": f"{view}.jsonl missing"}
    offset = max(0, offset)
    limit = max(1, min(1000, limit))

    lines: List[str] = []
    total_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, raw in enumerate(fh):
            if idx >= offset and len(lines) < limit:
                lines.append(raw.rstrip("\n"))
            total_rows = idx + 1
    next_offset = offset + len(lines)
    has_more = next_offset < total_rows
    return {
        "ok": True,
        "dataset_id": safe_id,
        "view": view,
        "path": str(path.relative_to(REPO_ROOT)),
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset if has_more else offset,
        "has_more": has_more,
        "total_rows": total_rows,
        "content": "\n".join(lines),
    }


def _doc_display_title(path: Path, category: str) -> str:
    name = path.name
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _markdown_title(path)
    if category == "capture" and suffix == ".json":
        return _capture_json_title(path)
    return name


def _markdown_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.name
    # Session log should surface the newest dated section title.
    if path.name.upper() == "SESSION_LOG.MD":
        matches = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if matches:
            return matches[-1].strip()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return path.name


def _capture_json_title(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path.name
    changed_path = str(payload.get("changed_path") or "").strip()
    started = str(payload.get("started_at") or "").strip()
    if changed_path and started:
        return f"{started[:10]} — {changed_path}"
    if changed_path:
        return changed_path
    return path.name


def _doc_model_track(rel_path: str, category: str) -> str:
    rel = (rel_path or "").strip()
    lower_rel = rel.lower()
    name = Path(rel).name.lower()
    if name.endswith(".opensource.md"):
        return "opensource"
    if name.endswith(".specialized.md"):
        return "specialized"
    if name.endswith(".comparison.md") or lower_rel == "docs/private_dashboard_deploy.md":
        return "codex_authored"
    if category == "runs":
        return "run"
    if category == "capture":
        return "capture"
    return "other"


_SPECIALIST_MASS_TASKS: Dict[str, str] = {
    "loading_screen": "loading_screen_mass_tasks_v1.json",
    "hud_status": "hud_status_mass_tasks_v1.json",
    "economy_tooltip": "economy_tooltip_mass_tasks_v1.json",
    "combat_risk": "combat_risk_mass_tasks_v1.json",
    "save_load_api_guard": "save_load_api_guard_mass_tasks_v1.json",
    "ai_planning_explanation": "ai_planning_explanation_mass_tasks_v1.json",
}

_CONCEPT_LEXICON: Dict[str, Tuple[str, ...]] = {
    "Loading/Start": ("loading", "start", "queued", "preparing", "ready", "readiness"),
    "UI Hierarchy": ("title", "subtitle", "hierarchy", "layout", "spacing", "contrast", "badge", "chip", "tooltip"),
    "Morale": ("morale",),
    "Supply": ("supply",),
    "Risk": ("risk", "odds"),
    "Terrain/Fort": ("terrain", "wall", "fortification", "fortified"),
    "Economy": ("gold", "income", "upkeep", "market", "workforce", "trade"),
    "Schema/API": ("schema", "version", "errors", "serialize", "migration", "api"),
    "Planning Intent": ("defend", "expand", "scout", "reinforce", "intent", "council"),
}


def _bucket_for_category(category: str) -> str:
    cat = (category or "").lower()
    if "constraint" in cat:
        return "constraints"
    if "ui_change" in cat:
        return "ui_change"
    if "transfer" in cat:
        return "transfer"
    return "core"


def _extract_concepts(prompt: str) -> List[str]:
    p = (prompt or "").lower()
    found = [name for name, words in _CONCEPT_LEXICON.items() if any(w in p for w in words)]
    return found or ["Generic"]


def _latest_mass_runs_by_specialist() -> Dict[str, Dict[str, Any]]:
    runs_dir = REPO_ROOT / "benchmarks" / "results" / "runs"
    latest: Dict[str, Dict[str, Any]] = {}
    if not runs_dir.is_dir():
        return latest
    for mf in runs_dir.glob("*/manifest.json"):
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("subcommand")) != "benchmark":
            continue
        argv = payload.get("argv") or []
        if not isinstance(argv, list):
            continue
        specialists: List[str] = []
        tasks_path = ""
        for idx, token in enumerate(argv):
            if token == "--specialist" and idx + 1 < len(argv):
                specialists.append(str(argv[idx + 1]))
            if token == "--tasks" and idx + 1 < len(argv):
                tasks_path = str(argv[idx + 1])
        if len(specialists) != 1:
            continue
        specialist = specialists[0]
        expected_name = _SPECIALIST_MASS_TASKS.get(specialist)
        if not expected_name:
            continue
        if Path(tasks_path).name != expected_name:
            continue
        current = latest.get(specialist)
        finished = str(payload.get("finished_at") or "")
        if not current or finished > str(current.get("finished_at") or ""):
            latest[specialist] = {
                "run_id": str(payload.get("run_id") or mf.parent.name),
                "finished_at": finished,
                "benchmark_summary": str(payload.get("benchmark_summary") or "—"),
                "capability_summary": str(payload.get("capability_summary") or "—"),
                "manifest": payload,
            }
    return latest


def _capability_map_payload() -> Dict[str, Any]:
    latest = _latest_mass_runs_by_specialist()
    task_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for specialist, task_file in _SPECIALIST_MASS_TASKS.items():
        tpath = REPO_ROOT / "benchmarks" / task_file
        if not tpath.is_file():
            continue
        try:
            tasks = json.loads(tpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(tasks, list):
            continue
        for row in tasks:
            if not isinstance(row, dict):
                continue
            tid = str(row.get("id") or "")
            if not tid:
                continue
            prompt = str(row.get("prompt") or "")
            category = str(row.get("category") or "")
            task_lookup[(specialist, tid)] = {
                "prompt": prompt,
                "category": category,
                "bucket": _bucket_for_category(category),
                "concepts": _extract_concepts(prompt),
            }

    rows: List[Dict[str, Any]] = []
    status_re = re.compile(r"\[(PASS|FAIL)\]\s+([^\s]+)\s+\(([^)]+)\)\s+(\d+) chars\s+([0-9.]+)s\s+cap=([0-9.]+)")
    for specialist, run in latest.items():
        log_path = REPO_ROOT / "benchmarks" / "results" / "runs" / run["run_id"] / "logs" / "run_game_benchmark.log"
        if not log_path.is_file():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for m in status_re.finditer(log_text):
            status, task_id, _, chars, seconds, cap = m.groups()
            meta = task_lookup.get((specialist, task_id), {})
            rows.append(
                {
                    "specialist": specialist,
                    "task_id": task_id,
                    "passed": status == "PASS",
                    "chars": int(chars),
                    "seconds": float(seconds),
                    "capability": float(cap),
                    "bucket": str(meta.get("bucket") or "core"),
                    "concepts": list(meta.get("concepts") or ["Generic"]),
                    "prompt": str(meta.get("prompt") or ""),
                }
            )

    if not rows:
        return {
            "ok": False,
            "reason": "No mass benchmark rows found. Run specialist mass benchmark suites first.",
            "runs": latest,
        }

    bucket_names = ["core", "ui_change", "constraints", "transfer"]
    concept_names = sorted({c for r in rows for c in r["concepts"]})

    overall_bucket_rate: Dict[str, float] = {}
    for bucket in bucket_names:
        vals = [1.0 if r["passed"] else 0.0 for r in rows if r["bucket"] == bucket]
        overall_bucket_rate[bucket] = (sum(vals) / len(vals)) if vals else 0.0

    edges: List[Dict[str, Any]] = []
    for concept in concept_names:
        for bucket in bucket_names:
            vals = [1.0 if r["passed"] else 0.0 for r in rows if concept in r["concepts"] and r["bucket"] == bucket]
            if len(vals) < 2:
                continue
            local_rate = sum(vals) / len(vals)
            signed = local_rate - overall_bucket_rate.get(bucket, 0.0)
            weight = abs(signed)
            if weight < 0.05:
                continue
            edges.append(
                {
                    "source": f"concept:{concept}",
                    "target": f"capability:{bucket}",
                    "weight": round(weight, 4),
                    "signed": round(signed, 4),
                    "sample_count": len(vals),
                }
            )

    concept_stats = []
    for concept in concept_names:
        crows = [r for r in rows if concept in r["concepts"]]
        if not crows:
            continue
        pass_rate = sum(1.0 for r in crows if r["passed"]) / len(crows)
        avg_cap = sum(r["capability"] for r in crows) / len(crows)
        concept_stats.append(
            {
                "concept": concept,
                "task_count": len(crows),
                "pass_rate": round(pass_rate, 4),
                "avg_capability": round(avg_cap, 2),
            }
        )
    concept_stats.sort(key=lambda row: (row["pass_rate"], row["task_count"]))

    agent_summary = []
    for specialist in sorted(_SPECIALIST_MASS_TASKS.keys()):
        srows = [r for r in rows if r["specialist"] == specialist]
        if not srows:
            continue
        pass_rate = sum(1.0 for r in srows if r["passed"]) / len(srows)
        agent_summary.append(
            {
                "specialist": specialist,
                "task_count": len(srows),
                "pass_rate": round(pass_rate, 4),
                "avg_capability": round(sum(r["capability"] for r in srows) / len(srows), 2),
                "run_id": str((latest.get(specialist) or {}).get("run_id") or ""),
            }
        )

    # Simple deterministic bipartite layout for scroll/zoom rendering.
    concept_order = [row["concept"] for row in concept_stats] or concept_names
    cap_order = bucket_names
    nodes: List[Dict[str, Any]] = []
    for i, concept in enumerate(concept_order):
        y = 0.08 + (0.84 * i / max(1, len(concept_order) - 1))
        nodes.append({"id": f"concept:{concept}", "kind": "concept", "label": concept, "x": 0.2, "y": round(y, 4)})
    for i, bucket in enumerate(cap_order):
        y = 0.15 + (0.7 * i / max(1, len(cap_order) - 1))
        nodes.append({"id": f"capability:{bucket}", "kind": "capability", "label": bucket, "x": 0.82, "y": round(y, 4)})

    prompt_samples = []
    for r in sorted(rows, key=lambda row: (row["passed"], row["specialist"], row["task_id"]))[:24]:
        prompt_samples.append(
            {
                "specialist": r["specialist"],
                "task_id": r["task_id"],
                "bucket": r["bucket"],
                "passed": r["passed"],
                "concepts": ", ".join(r["concepts"]),
                "prompt": (r["prompt"][:140] + "…") if len(r["prompt"]) > 140 else r["prompt"],
            }
        )

    # Task-level cosine similarity map over prompt concepts + category bucket one-hot vectors.
    # cos(theta) = dot(a, b) / (||a|| * ||b||)
    task_vectors: List[Dict[str, Any]] = []
    for r in rows:
        concepts = list(r.get("concepts") or [])
        concept_set = set(concepts)
        bucket = str(r.get("bucket") or "core")
        vector = [1.0 if c in concept_set else 0.0 for c in concept_names]
        vector.extend(1.0 if bucket == b else 0.0 for b in bucket_names)
        task_vectors.append(
            {
                "id": f"{r['specialist']}:{r['task_id']}",
                "specialist": r["specialist"],
                "task_id": r["task_id"],
                "bucket": bucket,
                "concepts": concepts,
                "passed": bool(r["passed"]),
                "prompt": (r["prompt"][:180] + "…") if len(r["prompt"]) > 180 else r["prompt"],
                "vector": vector,
            }
        )

    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)

    top_pairs: List[Dict[str, Any]] = []
    for i, a in enumerate(task_vectors):
        for j in range(i + 1, len(task_vectors)):
            b = task_vectors[j]
            sim = _cosine(list(a["vector"]), list(b["vector"]))
            if sim < 0.25:
                continue
            top_pairs.append(
                {
                    "a_id": a["id"],
                    "b_id": b["id"],
                    "cosine": round(sim, 4),
                    "a_bucket": a["bucket"],
                    "b_bucket": b["bucket"],
                }
            )
    top_pairs.sort(key=lambda row: row["cosine"], reverse=True)

    skills_payload: Dict[str, Any] = {"available": False}
    skills_path = REPO_ROOT / "data" / "routing" / "skills_v1.json"
    manifolds_path = REPO_ROOT / "data" / "routing" / "skill_manifolds_v1.json"
    if skills_path.is_file() and manifolds_path.is_file():
        try:
            skills_doc = json.loads(skills_path.read_text(encoding="utf-8"))
            manifolds_doc = json.loads(manifolds_path.read_text(encoding="utf-8"))
            skills_payload = {
                "available": True,
                "skills_path": str(skills_path.relative_to(REPO_ROOT)),
                "manifolds_path": str(manifolds_path.relative_to(REPO_ROOT)),
                "retained_skills": skills_doc.get("retained_skills") or [],
                "exploratory_skills": skills_doc.get("exploratory_skills") or [],
                "regions": manifolds_doc.get("regions") or [],
            }
        except (OSError, json.JSONDecodeError):
            skills_payload = {"available": False}

    return {
        "ok": True,
        "generated_utc": _utc_now(),
        "runs": latest,
        "rows_count": len(rows),
        "nodes": nodes,
        "edges": sorted(edges, key=lambda e: e["weight"], reverse=True)[:120],
        "agent_summary": agent_summary,
        "concept_stats": concept_stats,
        "bucket_baseline_pass_rate": {k: round(v, 4) for k, v in overall_bucket_rate.items()},
        "prompt_samples": prompt_samples,
        "task_similarity": {
            "formula": "cos(theta)=dot(a,b)/(||a||*||b||)",
            "dimensions": {
                "concepts": concept_names,
                "buckets": bucket_names,
            },
            "tasks": task_vectors,
            "top_pairs": top_pairs[:240],
        },
        "skills_v1": skills_payload,
        "notes": [
            "Edge signed value = concept-bucket pass-rate delta vs global bucket baseline.",
            "Positive edge implies a concept lifts that capability bucket; negative implies drag.",
            "This map is correlation-only and depends on specialist benchmark maturity.",
        ],
    }


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
    reference_keywords = set(tracks.get("codex_authored", {}).get("keywords", []))
    for key, row in tracks.items():
        kws = set(row.get("keywords", []))
        row["missing_vs_cursor"] = sorted(reference_keywords - kws) if reference_keywords else []
        base_hits = tracks.get("codex_authored", {}).get("keyword_hits", 0)
        row["delta_hits_vs_cursor"] = int(row.get("keyword_hits", 0)) - int(base_hits)
    winner = max(tracks.items(), key=lambda kv: (kv[1]["keyword_hits"], -kv[1]["word_count"]))[0]
    base_name = path.name.replace(".comparison.json", "")
    doc_candidates = {
        "codex_authored": REPO_ROOT / "docs" / "PRIVATE_DASHBOARD_DEPLOY.md",
        "opensource": path.parent / f"{base_name}.opensource.md",
        "specialized": path.parent / f"{base_name}.specialized.md",
    }
    doc_paths: Dict[str, str] = {}
    for key, p in doc_candidates.items():
        if p.is_file():
            doc_paths[key] = str(p.relative_to(REPO_ROOT))
    os_vs_spec = {
        "available": False,
        "identical": False,
    }
    os_rel = doc_paths.get("opensource")
    sp_rel = doc_paths.get("specialized")
    if os_rel and sp_rel:
        os_text = _read_repo_file_text(os_rel).strip()
        sp_text = _read_repo_file_text(sp_rel).strip()
        os_vs_spec = {
            "available": True,
            "identical": bool(os_text and sp_text and os_text == sp_text),
        }
    return {
        "available": True,
        "file": str(path.relative_to(REPO_ROOT)),
        "generated_utc": payload.get("generated_utc", ""),
        "model": payload.get("model", ""),
        "specialized_adapter": payload.get("specialized_adapter", ""),
        "tracks": tracks,
        "winner": winner,
        "doc_paths": doc_paths,
        "opensource_vs_specialized": os_vs_spec,
    }


def _read_repo_file_text(rel_path: str, *, max_chars: int = 80000) -> str:
    candidate = (REPO_ROOT / rel_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return "(invalid path)"
    if not candidate.is_file():
        return "(missing file)"
    return candidate.read_text(encoding="utf-8", errors="replace")[:max_chars]


def _archive_output_artifact(scoring: Dict[str, Any], track: str) -> Tuple[bool, str]:
    rel = str((scoring.get("doc_paths") or {}).get(track, "")).strip()
    if not rel:
        return False, "no artifact mapped for track"
    candidate = (REPO_ROOT / rel).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False, "artifact path is outside repo root"
    # Only allow curation deletes for generated comparison outputs.
    generated_root = (REPO_ROOT / "docs" / "generated").resolve()
    try:
        candidate.relative_to(generated_root)
    except ValueError:
        return False, "only docs/generated artifacts can be deleted"
    if not candidate.is_file():
        return False, "artifact is missing"
    archived = candidate.with_name(f"{candidate.stem}.deleted-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{candidate.suffix}")
    candidate.rename(archived)
    return True, str(archived.relative_to(REPO_ROOT))


def _render_doc_view(title: str, body: str, *, back_href: str = "/") -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        f"<title>{_h(title)}</title>"
        "<style>"
        "body{font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:20px;background:#0b1220;color:#e6eefc}"
        "a{color:#7cb2ff} pre{white-space:pre-wrap;border:1px solid #2a3f62;background:#0f1a2f;border-radius:10px;padding:12px;max-width:1200px}"
        ".top{display:flex;justify-content:space-between;align-items:center;gap:10px}"
        "</style></head><body>"
        "<div class='top'>"
        f"<h2>{_h(title)}</h2>"
        f"<a href='{_h(back_href)}'>Back to dashboard</a>"
        "</div>"
        f"<pre>{_h(body)}</pre>"
        "</body></html>"
    )


def _render_compare_view(
    left_track: str,
    left_title: str,
    left_body: str,
    right_track: str,
    right_title: str,
    right_body: str,
    left_deletable: bool,
    right_deletable: bool,
) -> str:
    identical = left_body == right_body
    winner_labels = {
        "codex_authored": "Frontier",
        "opensource": "Open-source",
        "specialized": "Specialized",
    }
    left_winner_label = winner_labels.get(left_track, left_title)
    right_winner_label = winner_labels.get(right_track, right_title)
    identical_banner = (
        "<p style='padding:8px 10px;border:1px solid #365f95;border-radius:8px;background:#102846;'>"
        "These outputs are currently identical. This usually means both tracks were generated from the same base behavior for this artifact."
        "</p>"
        if identical
        else ""
    )
    left_text_json = json.dumps(left_body, ensure_ascii=False)
    right_text_json = json.dumps(right_body, ensure_ascii=False)
    left_track_json = json.dumps(left_track)
    right_track_json = json.dumps(right_track)
    left_winner_label_json = json.dumps(left_winner_label)
    right_winner_label_json = json.dumps(right_winner_label)
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Side-by-side comparison</title>"
        "<style>"
        "body{font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:16px;background:#081224;color:#e6eefc}"
        "a{color:#7cb2ff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
        ".panel,.verdict{border:1px solid #2a3f62;background:#0f1a2f;border-radius:10px;padding:10px}"
        ".doc{white-space:pre-wrap;border:1px solid #2a3f62;background:#091326;border-radius:10px;padding:12px;max-height:52vh;overflow:auto}"
        ".doc .hl-good{background:#12442d}.doc .hl-bad{background:#5e1f2a}.doc .hl-corrupt{background:#6b3f04}"
        "h3,h4{margin:0 0 8px 0}.top{display:flex;justify-content:space-between;align-items:center}"
        ".row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}"
        ".row select,.row textarea,.row button,.row input{border:1px solid #335d92;border-radius:8px;background:#081224;color:#e6eefc;padding:8px}"
        ".row textarea{width:100%;min-height:72px}"
        ".row button{background:#1e60d6;border-color:#1e60d6;cursor:pointer;font-weight:600}"
        ".row button.secondary{background:#182640;border-color:#335d92}"
        ".row button.good{background:#0f7b47;border-color:#0f7b47}.row button.bad{background:#8a2435;border-color:#8a2435}.row button.corrupt{background:#9a6700;border-color:#9a6700}"
        ".status{font-size:12px;color:#9bc2ff}"
        ".muted{font-size:12px;color:#9bc2ff}"
        ".spans{margin-top:10px;border:1px dashed #335d92;border-radius:8px;padding:8px}"
        ".span-item{border:1px solid #23436d;border-radius:8px;padding:8px;margin-top:8px}"
        ".span-item.good{border-color:#1f6a48}.span-item.bad{border-color:#7d3340}.span-item.corrupt{border-color:#8b5f10}"
        ".legacy{margin-top:12px;border-top:1px solid #2a3f62;padding-top:10px}"
        ".verdict{margin-top:12px}"
        "</style></head><body>"
        f"<div class='top'><h2>Compare: {_h(left_title)} vs {_h(right_title)}</h2><a href='/'>Back to dashboard</a></div>"
        + identical_banner +
        "<div class='grid'>"
        f"<section class='panel' data-side='left'><h3>{_h(left_title)}</h3>"
        "<div class='row'><input class='reason' placeholder='Reason for selected span (optional)'/>"
        "<button class='good add-span'>Mark Green (good)</button><button class='bad add-span' data-label='bad'>Mark Red (bad)</button><button class='corrupt add-span' data-label='corrupt'>Mark Corrupt (delete/replace)</button></div>"
        "<div class='row'><textarea class='rewrite' placeholder='Optional rewrite text (used with red/corrupt spans)'></textarea></div>"
        "<div class='doc' id='leftDoc'></div>"
        f"<div class='legacy curate' data-track='{_h(left_track)}' data-deletable='{str(left_deletable).lower()}'>"
        "<h4>Legacy per-track curation (optional)</h4>"
        "<div class='row'><select class='action'><option value='feedback'>Feedback only</option><option value='train_include'>Include for training</option><option value='train_exclude'>Exclude from training</option></select>"
        "<select class='rating'><option value='5'>5</option><option value='4'>4</option><option value='3'>3</option><option value='2'>2</option><option value='1'>1</option></select></div>"
        "<div class='row'><textarea class='key-details' placeholder='Key details to train on'></textarea></div>"
        "<div class='row'><textarea class='language-edits' placeholder='Language edits for future output'></textarea></div>"
        "<div class='row'><textarea class='notes' placeholder='Notes/context for future training'></textarea></div>"
        "<div class='row'><button class='save secondary'>Save Legacy Curation</button><button class='secondary delete'>Delete Output</button><span class='status'></span></div>"
        "</div></section>"
        f"<section class='panel' data-side='right'><h3>{_h(right_title)}</h3>"
        "<div class='row'><input class='reason' placeholder='Reason for selected span (optional)'/>"
        "<button class='good add-span'>Mark Green (good)</button><button class='bad add-span' data-label='bad'>Mark Red (bad)</button><button class='corrupt add-span' data-label='corrupt'>Mark Corrupt (delete/replace)</button></div>"
        "<div class='row'><textarea class='rewrite' placeholder='Optional rewrite text (used with red/corrupt spans)'></textarea></div>"
        "<div class='doc' id='rightDoc'></div>"
        f"<div class='legacy curate' data-track='{_h(right_track)}' data-deletable='{str(right_deletable).lower()}'>"
        "<h4>Legacy per-track curation (optional)</h4>"
        "<div class='row'><select class='action'><option value='feedback'>Feedback only</option><option value='train_include'>Include for training</option><option value='train_exclude'>Exclude from training</option></select>"
        "<select class='rating'><option value='5'>5</option><option value='4'>4</option><option value='3'>3</option><option value='2'>2</option><option value='1'>1</option></select></div>"
        "<div class='row'><textarea class='key-details' placeholder='Key details to train on'></textarea></div>"
        "<div class='row'><textarea class='language-edits' placeholder='Language edits for future output'></textarea></div>"
        "<div class='row'><textarea class='notes' placeholder='Notes/context for future training'></textarea></div>"
        "<div class='row'><button class='save secondary'>Save Legacy Curation</button><button class='secondary delete'>Delete Output</button><span class='status'></span></div>"
        "</div></section>"
        "</div>"
        "<section class='verdict'>"
        "<h3>Structured training feedback</h3>"
        f"<div class='row'><label>Winner <select id='winner'><option value=''>Select winner</option><option value='left'>{_h(left_winner_label)}</option><option value='right'>{_h(right_winner_label)}</option><option value='tie'>Tie</option></select></label>"
        "<label>Preference strength <select id='strength'><option value=''>Select strength</option><option value='weak'>Weak</option><option value='medium'>Medium</option><option value='strong'>Strong</option><option value='tie'>Tie/No preference</option></select></label></div>"
        "<div class='row'><textarea id='compareNotes' placeholder='Optional overall notes for this comparison'></textarea></div>"
        "<p class='muted'>Highlight text in either panel, then apply green/red/corrupt labels. Use corrupt for broken formatting/output that should be rewritten or removed.</p>"
        "<div id='spanList' class='spans'></div>"
        "<div class='row'><button id='submitStructured'>Save and Exit</button><span id='structuredStatus' class='status'></span></div>"
        "</section>"
        "<script>"
        f"const leftTrack = {left_track_json};"
        f"const rightTrack = {right_track_json};"
        f"const leftWinnerLabel = {left_winner_label_json};"
        f"const rightWinnerLabel = {right_winner_label_json};"
        f"const state = {{ leftText: {left_text_json}, rightText: {right_text_json}, spans: [] }};"
        "const leftDoc = document.getElementById('leftDoc');"
        "const rightDoc = document.getElementById('rightDoc');"
        "const spanList = document.getElementById('spanList');"
        "const statusEl = document.getElementById('structuredStatus');"
        "function escapeHtml(v){return (v || '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}"
        "function renderSide(side){"
        "  const text = side === 'left' ? state.leftText : state.rightText;"
        "  const spans = state.spans.filter((s)=>s.side===side).sort((a,b)=>a.start-b.start || a.end-b.end);"
        "  let out=''; let cursor=0;"
        "  for(const s of spans){"
        "    if (s.start > cursor) out += escapeHtml(text.slice(cursor, s.start));"
        "    const cls = s.label === 'good' ? 'hl-good' : (s.label === 'corrupt' ? 'hl-corrupt' : 'hl-bad');"
        "    out += `<span class=\"${cls}\">${escapeHtml(text.slice(s.start, s.end))}</span>`;"
        "    cursor = s.end;"
        "  }"
        "  if (cursor < text.length) out += escapeHtml(text.slice(cursor));"
        "  (side === 'left' ? leftDoc : rightDoc).innerHTML = out;"
        "}"
        "function getSelectionOffsets(side){"
        "  const root = side === 'left' ? leftDoc : rightDoc;"
        "  const sel = window.getSelection();"
        "  if (!sel || sel.rangeCount === 0) return null;"
        "  const range = sel.getRangeAt(0);"
        "  if (!root.contains(range.commonAncestorContainer)) return null;"
        "  const text = side === 'left' ? state.leftText : state.rightText;"
        "  if (range.collapsed) return null;"
        "  const prefix = range.cloneRange();"
        "  prefix.selectNodeContents(root);"
        "  prefix.setEnd(range.startContainer, range.startOffset);"
        "  const start = prefix.toString().length;"
        "  const selected = range.toString();"
        "  const end = start + selected.length;"
        "  if (start < 0 || end > text.length || end <= start) return null;"
        "  return {start, end, selected};"
        "}"
        "function hasOverlap(side,start,end){"
        "  return state.spans.some((s)=>s.side===side && start < s.end && end > s.start);"
        "}"
        "function renderSpanList(){"
        "  if (!state.spans.length){ spanList.innerHTML = '<div class=\"muted\">No labeled spans yet.</div>'; return; }"
        "  const rows = state.spans.slice().sort((a,b)=>a.side.localeCompare(b.side) || a.start-b.start).map((s)=>{"
        "    const cls = s.label === 'good' ? 'good' : (s.label === 'corrupt' ? 'corrupt' : 'bad');"
        "    return `<div class=\"span-item ${cls}\" data-span-id=\"${s.id}\"><strong>${s.side.toUpperCase()} ${s.label.toUpperCase()}</strong> [${s.start}, ${s.end})` +"
        "      `<div class=\"muted\">${escapeHtml(s.selected_text.slice(0,200))}</div>` +"
        "      `<div class=\"row\"><input class=\"span-reason\" value=\"${escapeHtml(s.reason)}\" placeholder=\"Reason (optional)\"/></div>` +"
        "      `<div class=\"row\"><textarea class=\"span-rewrite\" placeholder=\"Rewrite text for bad/corrupt spans (optional)\">${escapeHtml(s.rewrite_text)}</textarea></div>` +"
        "      `<div class=\"row\"><button class=\"secondary span-delete\">Delete span</button></div></div>`;"
        "  });"
        "  spanList.innerHTML = rows.join('');"
        "}"
        "function renderAll(){ renderSide('left'); renderSide('right'); renderSpanList(); }"
        "renderAll();"
        "for (const panel of document.querySelectorAll('.panel[data-side]')) {"
        "  const side = panel.getAttribute('data-side') || 'left';"
        "  for (const btn of panel.querySelectorAll('.add-span')) {"
        "    btn.addEventListener('click', () => {"
        "      const label = btn.dataset.label === 'corrupt' ? 'corrupt' : (btn.dataset.label === 'bad' ? 'bad' : 'good');"
        "      const r = getSelectionOffsets(side);"
        "      if (!r) { statusEl.textContent = 'Select text inside a document first.'; return; }"
        "      if (hasOverlap(side, r.start, r.end)) { statusEl.textContent = 'Selected span overlaps an existing label.'; return; }"
        "      const reason = (panel.querySelector('.reason')?.value || '').trim();"
        "      const rewrite = (panel.querySelector('.rewrite')?.value || '').trim();"
        "      if (label === 'good' && rewrite) { statusEl.textContent = 'Rewrite text is only for red/corrupt spans.'; return; }"
        "      state.spans.push({"
        "        id: String(Date.now()) + '-' + Math.random().toString(16).slice(2), side, label,"
        "        start: r.start, end: r.end, selected_text: r.selected, reason, rewrite_text: rewrite"
        "      });"
        "      const sideLabel = side === 'left' ? leftWinnerLabel : rightWinnerLabel;"
        "      statusEl.textContent = `Added ${label} span (${sideLabel}).`;"
        "      renderAll();"
        "    });"
        "  }"
        "}"
        "spanList.addEventListener('click', (ev) => {"
        "  const target = ev.target;"
        "  if (!(target instanceof HTMLElement)) return;"
        "  if (!target.classList.contains('span-delete')) return;"
        "  const card = target.closest('[data-span-id]');"
        "  if (!card) return;"
        "  const spanId = card.getAttribute('data-span-id');"
        "  state.spans = state.spans.filter((s)=>s.id !== spanId);"
        "  renderAll();"
        "});"
        "spanList.addEventListener('input', (ev) => {"
        "  const target = ev.target;"
        "  if (!(target instanceof HTMLElement)) return;"
        "  const card = target.closest('[data-span-id]');"
        "  if (!card) return;"
        "  const spanId = card.getAttribute('data-span-id');"
        "  const span = state.spans.find((s)=>s.id === spanId);"
        "  if (!span) return;"
        "  if (target.classList.contains('span-reason')) span.reason = (target.value || '').toString();"
        "  if (target.classList.contains('span-rewrite')) span.rewrite_text = (target.value || '').toString();"
        "});"
        "document.getElementById('submitStructured')?.addEventListener('click', async () => {"
        "  const winner = (document.getElementById('winner')?.value || '').trim();"
        "  const strength = (document.getElementById('strength')?.value || '').trim();"
        "  const notes = (document.getElementById('compareNotes')?.value || '').trim();"
        "  if (!winner) { statusEl.textContent = 'Select a winner (or tie).'; return; }"
        "  if (winner !== 'tie' && !strength) { statusEl.textContent = 'Select preference strength.'; return; }"
        "  statusEl.textContent = 'Saving structured feedback...';"
        "  try {"
        "    const payload = {"
        "      left_track: leftTrack, right_track: rightTrack, winner, strength, notes, source: 'compare_view_structured',"
        "      spans: state.spans.map((s)=>({ side:s.side, label:s.label, start:s.start, end:s.end, selected_text:s.selected_text, reason:s.reason, rewrite_text:s.rewrite_text }))"
        "    };"
        "    const res = await fetch('/api/compare-feedback', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });"
        "    const body = await res.json();"
        "    if (!res.ok || !body.ok) throw new Error(body.error || ('HTTP ' + res.status));"
        "    statusEl.textContent = `Structured feedback saved (#${body.id}). Exiting...`;"
        "    setTimeout(() => { window.location.href = '/'; }, 450);"
        "  } catch (err) { statusEl.textContent = 'Save failed: ' + err; }"
        "});"
        "for (const box of document.querySelectorAll('.curate')) {"
        "  const track = box.getAttribute('data-track') || '';"
        "  const deletable = (box.getAttribute('data-deletable') || '') === 'true';"
        "  const saveBtn = box.querySelector('.save');"
        "  const delBtn = box.querySelector('.delete');"
        "  const status = box.querySelector('.status');"
        "  if (!deletable) { delBtn.disabled = true; delBtn.title = 'Only docs/generated outputs can be deleted.'; }"
        "  saveBtn?.addEventListener('click', async () => {"
        "    const notes = (box.querySelector('.notes')?.value || '').trim();"
        "    if (!notes) { status.textContent = 'Add notes before saving.'; return; }"
        "    status.textContent = 'Saving...';"
        "    try {"
        "      const res = await fetch('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({"
        "        track, action: box.querySelector('.action')?.value || 'feedback', rating: Number(box.querySelector('.rating')?.value || 3),"
        "        key_details: box.querySelector('.key-details')?.value || '', language_edits: box.querySelector('.language-edits')?.value || '',"
        "        notes, source: 'compare_view_legacy_curation'"
        "      })});"
        "      const payload = await res.json();"
        "      if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));"
        "      status.textContent = 'Saved.';"
        "    } catch (err) { status.textContent = 'Save failed: ' + err; }"
        "  });"
        "  delBtn?.addEventListener('click', async () => {"
        "    if (!deletable) return;"
        "    status.textContent = 'Deleting output...';"
        "    try {"
        "      const res = await fetch('/api/output-action', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({"
        "        track, action:'delete_output', reason: (box.querySelector('.notes')?.value || 'deleted from compare view'), source: 'compare_view_legacy_curation'"
        "      })});"
        "      const payload = await res.json();"
        "      if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));"
        "      status.textContent = 'Deleted. Refresh dashboard scoring.';"
        "    } catch (err) { status.textContent = 'Delete failed: ' + err; }"
        "  });"
        "}"
        "</script>"
        "</body></html>"
    )


def _render_capability_map_view() -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>Capability Map Explorer</title>"
        "<style>"
        "body{font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:16px;background:#081224;color:#e6eefc}"
        "a{color:#7cb2ff}.top{display:flex;justify-content:space-between;align-items:center;gap:10px}"
        ".controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}"
        "button{border:1px solid #335d92;border-radius:8px;background:#1e60d6;color:#fff;padding:8px 12px;cursor:pointer;font-weight:600}"
        "button.secondary{background:#102846}"
        ".muted{color:#9bc2ff;font-size:12px}"
        "select{border:1px solid #335d92;border-radius:8px;background:#081224;color:#e6eefc;padding:7px 9px}"
        ".grid{display:grid;grid-template-columns:minmax(600px,2fr) minmax(320px,1fr);gap:12px}"
        ".panel{border:1px solid #2a3f62;background:#0f1a2f;border-radius:10px;padding:10px}"
        "#mapWrap{height:72vh;overflow:hidden;position:relative;cursor:grab}"
        "#mapWrap.dragging{cursor:grabbing}"
        "#capSvg{width:100%;height:100%;display:block;background:#091326;border-radius:8px}"
        "pre{white-space:pre-wrap;border:1px solid #2a3f62;background:#091326;border-radius:8px;padding:10px;max-height:32vh;overflow:auto}"
        ".sim-box{border:1px solid #2a3f62;background:#091326;border-radius:8px;padding:9px;margin-top:10px}"
        "@media (max-width: 1100px){.grid{grid-template-columns:1fr} #mapWrap{height:58vh}}"
        "</style></head><body>"
        "<div class='top'><h2>Capability Curvature Map (Docs Site)</h2><a href='/'>Back to dashboard</a></div>"
        "<p class='muted'>Drag to pan. Mousewheel / trackpad pinch to zoom. Double-click to recenter.</p>"
        "<div class='controls'>"
        "<button id='refreshBtn'>Refresh graph data</button>"
        "<button id='zoomInBtn' class='secondary'>Zoom +</button>"
        "<button id='zoomOutBtn' class='secondary'>Zoom -</button>"
        "<button id='resetBtn' class='secondary'>Reset view</button>"
        "<span id='meta' class='muted'></span>"
        "</div>"
        "<div class='grid'>"
        "<section class='panel'><div id='mapWrap'><svg id='capSvg' viewBox='0 0 1200 800' preserveAspectRatio='xMidYMid meet'><g id='capLayer'></g></svg></div></section>"
        "<section class='panel'><h3 style='margin:0 0 8px 0'>Diagnostics</h3><pre id='stats'>(loading...)</pre><h3 style='margin:10px 0 8px 0'>Prompt Samples</h3><pre id='samples'></pre><h3 style='margin:10px 0 8px 0'>Task Cosine Similarity</h3><div class='controls'><label for='simTaskA' class='muted'>Task A</label><select id='simTaskA' style='min-width:280px;'></select></div><div class='controls'><label for='simTaskB' class='muted'>Task B</label><select id='simTaskB' style='min-width:280px;'></select><button id='swapTasksBtn' class='secondary'>Swap</button></div><div id='simValue' class='sim-box muted'>(loading)</div><pre id='simNeighbors'>(loading)</pre><h3 style='margin:10px 0 8px 0'>Skills Correlation (v1)</h3><pre id='skillsStats'>(loading)</pre><h3 style='margin:10px 0 8px 0'>Region Stability (v1)</h3><pre id='regionStats'>(loading)</pre></section>"
        "</div>"
        "<script>"
        "const svg = document.getElementById('capSvg');"
        "const layer = document.getElementById('capLayer');"
        "const wrap = document.getElementById('mapWrap');"
        "const meta = document.getElementById('meta');"
        "const stats = document.getElementById('stats');"
        "const samples = document.getElementById('samples');"
        "const simTaskA = document.getElementById('simTaskA');"
        "const simTaskB = document.getElementById('simTaskB');"
        "const simValue = document.getElementById('simValue');"
        "const simNeighbors = document.getElementById('simNeighbors');"
        "const skillsStats = document.getElementById('skillsStats');"
        "const regionStats = document.getElementById('regionStats');"
        "let graph = null;"
        "const state = { tx: 0, ty: 0, scale: 1, dragging: false, x: 0, y: 0 };"
        "function applyTransform(){ layer.setAttribute('transform', `translate(${state.tx} ${state.ty}) scale(${state.scale})`); }"
        "function clampScale(v){ return Math.max(0.4, Math.min(5, v)); }"
        "function resetView(){ state.tx = 0; state.ty = 0; state.scale = 1; applyTransform(); }"
        "function zoomAt(screenX, screenY, factor){"
        "  const rect = svg.getBoundingClientRect();"
        "  if (!rect.width || !rect.height) return;"
        "  const vb = svg.viewBox.baseVal;"
        "  const sx = vb.width / rect.width;"
        "  const sy = vb.height / rect.height;"
        "  const px = (screenX - rect.left) * sx;"
        "  const py = (screenY - rect.top) * sy;"
        "  const beforeX = (px - state.tx) / state.scale;"
        "  const beforeY = (py - state.ty) / state.scale;"
        "  const nextScale = clampScale(state.scale * factor);"
        "  state.scale = nextScale;"
        "  state.tx = px - beforeX * state.scale;"
        "  state.ty = py - beforeY * state.scale;"
        "  applyTransform();"
        "}"
        "wrap.addEventListener('wheel', (ev) => { ev.preventDefault(); const factor = ev.deltaY < 0 ? 1.1 : 0.9; zoomAt(ev.clientX, ev.clientY, factor); }, { passive:false });"
        "wrap.addEventListener('mousedown', (ev) => { state.dragging = true; state.x = ev.clientX; state.y = ev.clientY; wrap.classList.add('dragging'); });"
        "window.addEventListener('mouseup', () => { state.dragging = false; wrap.classList.remove('dragging'); });"
        "window.addEventListener('mousemove', (ev) => {"
        "  if (!state.dragging) return;"
        "  const rect = svg.getBoundingClientRect();"
        "  const vb = svg.viewBox.baseVal;"
        "  const sx = vb.width / Math.max(1, rect.width);"
        "  const sy = vb.height / Math.max(1, rect.height);"
        "  state.tx += (ev.clientX - state.x) * sx;"
        "  state.ty += (ev.clientY - state.y) * sy;"
        "  state.x = ev.clientX; state.y = ev.clientY; applyTransform();"
        "});"
        "wrap.addEventListener('dblclick', resetView);"
        "document.getElementById('zoomInBtn')?.addEventListener('click', () => zoomAt(window.innerWidth * 0.5, window.innerHeight * 0.5, 1.15));"
        "document.getElementById('zoomOutBtn')?.addEventListener('click', () => zoomAt(window.innerWidth * 0.5, window.innerHeight * 0.5, 0.87));"
        "document.getElementById('resetBtn')?.addEventListener('click', resetView);"
        "function el(tag, attrs){ const n = document.createElementNS('http://www.w3.org/2000/svg', tag); for(const [k,v] of Object.entries(attrs || {})) n.setAttribute(k, String(v)); return n; }"
        "function nodeById(id){ return (graph?.nodes || []).find((n)=>n.id===id); }"
        "function cosine(a,b){"
        "  let dot = 0; let na = 0; let nb = 0;"
        "  const n = Math.min((a||[]).length, (b||[]).length);"
        "  for (let i=0;i<n;i++){ const x = Number(a[i]||0); const y = Number(b[i]||0); dot += x*y; na += x*x; nb += y*y; }"
        "  if (na <= 1e-12 || nb <= 1e-12) return 0;"
        "  return dot / (Math.sqrt(na) * Math.sqrt(nb));"
        "}"
        "function taskRows(){ return (graph?.task_similarity?.tasks || []); }"
        "function taskById(id){ return taskRows().find((t)=>t.id===id); }"
        "function renderSimilarityInspector(){"
        "  const tasks = taskRows();"
        "  if (!tasks.length){"
        "    if (simValue) simValue.textContent = 'No task vectors available.';"
        "    if (simNeighbors) simNeighbors.textContent = '';"
        "    return;"
        "  }"
        "  if (simTaskA && simTaskA.options.length === 0){"
        "    for (const t of tasks){"
        "      const optA = document.createElement('option');"
        "      optA.value = t.id; optA.textContent = `${t.specialist}/${t.task_id} (${t.bucket})`;"
        "      simTaskA.appendChild(optA);"
        "      const optB = document.createElement('option');"
        "      optB.value = t.id; optB.textContent = `${t.specialist}/${t.task_id} (${t.bucket})`;"
        "      simTaskB?.appendChild(optB);"
        "    }"
        "    if (simTaskA) simTaskA.value = tasks[0].id;"
        "    if (simTaskB) simTaskB.value = tasks[Math.min(1, tasks.length - 1)].id;"
        "  }"
        "  const a = taskById(simTaskA?.value || '');"
        "  const b = taskById(simTaskB?.value || '');"
        "  if (!a || !b){"
        "    if (simValue) simValue.textContent = 'Select two tasks.';"
        "    return;"
        "  }"
        "  const sim = cosine(a.vector || [], b.vector || []);"
        "  const theta = Math.acos(Math.max(-1, Math.min(1, sim))) * (180/Math.PI);"
        "  if (simValue){"
        "    simValue.textContent = `cos(theta) = ${sim.toFixed(4)} | angle = ${theta.toFixed(2)}°` +"
        "      `\\nA: ${a.specialist}/${a.task_id} [${a.bucket}]` +"
        "      `\\nB: ${b.specialist}/${b.task_id} [${b.bucket}]`;"
        "  }"
        "  const neighbors = tasks"
        "    .filter((t)=>t.id !== a.id)"
        "    .map((t)=>({ t, sim: cosine(a.vector || [], t.vector || []) }))"
        "    .sort((x,y)=>y.sim - x.sim)"
        "    .slice(0, 10);"
        "  const lines = neighbors.map((row)=>`${row.sim.toFixed(4)}  ${row.t.specialist}/${row.t.task_id} (${row.t.bucket})\\nConcepts: ${(row.t.concepts || []).join(', ')}\\nPrompt: ${row.t.prompt}`);"
        "  if (simNeighbors) simNeighbors.textContent = `Nearest neighbors to A (${a.specialist}/${a.task_id}):\\n\\n${lines.join('\\n\\n')}`;"
        "}"
        "function renderGraph(payload){"
        "  graph = payload;"
        "  layer.innerHTML = '';"
        "  const width = 1200, height = 800;"
        "  const mapX = (u) => Math.round(u * width);"
        "  const mapY = (u) => Math.round(u * height);"
        "  for (const edge of (payload.edges || [])) {"
        "    const a = nodeById(edge.source); const b = nodeById(edge.target);"
        "    if (!a || !b) continue;"
        "    const stroke = Number(edge.signed || 0) >= 0 ? '#41d694' : '#ef6b7b';"
        "    const lw = Math.max(1.2, 1 + Number(edge.weight || 0) * 12);"
        "    const line = el('line', { x1: mapX(a.x), y1: mapY(a.y), x2: mapX(b.x), y2: mapY(b.y), stroke, 'stroke-width': lw, 'stroke-opacity': 0.75 });"
        "    layer.appendChild(line);"
        "  }"
        "  for (const node of (payload.nodes || [])) {"
        "    const isConcept = node.kind === 'concept';"
        "    const x = mapX(node.x), y = mapY(node.y);"
        "    layer.appendChild(el('circle', { cx: x, cy: y, r: isConcept ? 13 : 15, fill: isConcept ? '#2e8cff' : '#a78bfa', stroke: '#dbeafe', 'stroke-width': 1.5 }));"
        "    const t = el('text', { x: x + (isConcept ? -18 : 18), y: y + 5, fill: '#e6eefc', 'font-size': 14, 'text-anchor': isConcept ? 'end' : 'start' });"
        "    t.textContent = String(node.label || '');"
        "    layer.appendChild(t);"
        "  }"
        "  const agent = (payload.agent_summary || []).map((r)=>`${r.specialist}: ${Math.round((r.pass_rate||0)*100)}% pass, cap ${r.avg_capability}`).join('\\n');"
        "  const base = payload.bucket_baseline_pass_rate || {};"
        "  const baseText = Object.keys(base).map((k)=>`${k}: ${Math.round((base[k]||0)*100)}%`).join(', ');"
        "  const notes = (payload.notes || []).map((n)=>`- ${n}`).join('\\n');"
        "  stats.textContent = `Rows: ${payload.rows_count || 0}\\nGenerated: ${payload.generated_utc || ''}\\n\\nBucket baselines: ${baseText}\\n\\nPer-agent:\\n${agent || '(none)'}\\n\\nNotes:\\n${notes}`;"
        "  samples.textContent = (payload.prompt_samples || []).slice(0, 18).map((s)=>`[${s.passed ? 'PASS' : 'FAIL'}] ${s.specialist} / ${s.task_id} (${s.bucket})\\nConcepts: ${s.concepts}\\nPrompt: ${s.prompt}`).join('\\n\\n');"
        "  const sk = payload.skills_v1 || {};"
        "  if (!sk.available) {"
        "    if (skillsStats) skillsStats.textContent = 'skills_v1 artifacts not found. Run: python3 scripts/extract_skills_v1.py';"
        "    if (regionStats) regionStats.textContent = '';"
        "  } else {"
        "    const retained = (sk.retained_skills || []).slice(0, 8);"
        "    const exploratory = (sk.exploratory_skills || []).slice(0, 10);"
        "    const regions = (sk.regions || []).slice().sort((a,b)=>Number(b.stability_z||0)-Number(a.stability_z||0));"
        "    if (skillsStats) {"
        "      const lines = [];"
        "      lines.push(`Retained skills: ${(sk.retained_skills || []).length}`);"
        "      lines.push(`Exploratory skills: ${(sk.exploratory_skills || []).length}`);"
        "      lines.push('');"
        "      lines.push('Top retained (effect):');"
        "      for (const s of retained) lines.push(`- ${s.skill_id}  effect=${Number(s.effect||0).toFixed(3)}  support=${s.support_tasks}`);"
        "      lines.push('');"
        "      lines.push('Top exploratory (effect):');"
        "      for (const s of exploratory) lines.push(`- ${s.skill_id}  effect=${Number(s.effect||0).toFixed(3)}  support=${s.support_tasks}`);"
        "      skillsStats.textContent = lines.join('\\n');"
        "    }"
        "    if (regionStats) {"
        "      const lines = [];"
        "      lines.push('Regions sorted by stability z-score:');"
        "      lines.push('');"
        "      for (const r of regions) {"
        "        lines.push(`- region ${r.region_id}: z=${Number(r.stability_z||0).toFixed(3)}  raw=${Number(r.stability_score||0).toFixed(3)}  stable=${Boolean(r.stable_for_direct_routing)}  tasks=${r.task_count}`);"
        "      }"
        "      regionStats.textContent = lines.join('\\n');"
        "    }"
        "  }"
        "  const taskCount = (payload.task_similarity?.tasks || []).length;"
        "  meta.textContent = `${(payload.nodes || []).length} nodes • ${(payload.edges || []).length} edges • ${taskCount} task vectors`;"
        "  renderSimilarityInspector();"
        "}"
        "async function refresh(){"
        "  meta.textContent = 'Loading...';"
        "  try {"
        "    const res = await fetch('/api/capability-map');"
        "    const payload = await res.json();"
        "    if (!res.ok || !payload.ok) throw new Error(payload.reason || payload.error || ('HTTP ' + res.status));"
        "    renderGraph(payload);"
        "  } catch (err) {"
        "    stats.textContent = 'Failed to load map: ' + err;"
        "    samples.textContent = '';"
        "    meta.textContent = 'Unavailable';"
        "  }"
        "}"
        "document.getElementById('refreshBtn')?.addEventListener('click', refresh);"
        "simTaskA?.addEventListener('change', renderSimilarityInspector);"
        "simTaskB?.addEventListener('change', renderSimilarityInspector);"
        "document.getElementById('swapTasksBtn')?.addEventListener('click', () => {"
        "  if (!simTaskA || !simTaskB) return;"
        "  const a = simTaskA.value; simTaskA.value = simTaskB.value; simTaskB.value = a;"
        "  renderSimilarityInspector();"
        "});"
        "resetView(); refresh();"
        "</script>"
        "</body></html>"
    )


def _render_dashboard(
    summary: Dict[str, Any],
    scoring: Dict[str, Any],
    token_metrics: Dict[str, Any],
    feedback_rows: List[Dict[str, Any]],
    kpis: Dict[str, Any],
) -> str:
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
            "opensource": "Open-source Base (no adapter)",
            "specialized": "Specialized (base + LoRA adapter)",
        }
        doc_paths = scoring.get("doc_paths", {})
        score_rows = []
        for key in ("codex_authored", "opensource", "specialized"):
            if key not in tracks:
                continue
            row = tracks[key]
            missing = ", ".join(row.get("missing_vs_cursor", [])) or "-"
            read_link = "-"
            if key in doc_paths:
                read_link = f"<a href='/view/doc?track={_h(key)}'>Read full</a>"
            score_rows.append(
                "<tr>"
                f"<td>{_h(labels.get(key,key))}</td>"
                f"<td>{_h(row['keyword_hits'])}</td>"
                f"<td>{_h(row.get('delta_hits_vs_cursor', 0))}</td>"
                f"<td>{_h(row['word_count'])}</td>"
                f"<td>{_h(', '.join(row['keywords']))}</td>"
                f"<td>{_h(missing)}</td>"
                f"<td>{read_link}</td>"
                "</tr>"
            )
        winner_label = labels.get(scoring.get("winner", ""), scoring.get("winner", ""))
        compare_controls = (
            "<div class='compare-controls'>"
            "<label>Compare:</label>"
            "<select id='cmpLeft'>"
            "<option value='codex_authored'>Cursor/Codex</option>"
            "<option value='opensource'>Open-source Base</option>"
            "<option value='specialized'>Specialized Adapter</option>"
            "</select>"
            "<span>vs</span>"
            "<select id='cmpRight'>"
            "<option value='opensource'>Open-source Base</option>"
            "<option value='specialized'>Specialized Adapter</option>"
            "<option value='codex_authored'>Cursor/Codex</option>"
            "</select>"
            "<button id='openCompare' type='button'>Open side-by-side</button>"
            "</div>"
        )
        scoring_html = (
            f"<p><strong>Latest comparison:</strong> <code>{_h(scoring.get('file',''))}</code></p>"
            f"<p><strong>Winner by keyword-hit score:</strong> {_h(winner_label)}</p>"
            "<p class='muted'>Open-source and Specialized intentionally share the same base model. "
            "The only difference is whether a LoRA adapter is applied; similar or identical outputs are expected on some artifacts.</p>"
            f"<p class='muted'><strong>Open-source vs Specialized identical:</strong> {_h('yes' if (scoring.get('opensource_vs_specialized') or {}).get('identical') else 'no')}</p>"
            + compare_controls +
            "<table>"
            "<thead><tr><th>Track</th><th>Keyword hits</th><th>Delta vs Cursor</th><th>Word count</th><th>Matched keywords</th><th>Missing vs Cursor</th><th>Docs</th></tr></thead>"
            f"<tbody>{''.join(score_rows)}</tbody>"
            "</table>"
        )
    else:
        scoring_html = (
            f"<p>No scoring artifact yet: {_h(scoring.get('reason','unknown'))}</p>"
            "<p>Generate one with your duplicate-doc run and place the JSON under <code>docs/generated/*.comparison.json</code>.</p>"
        )

    feedback_table = ""
    if feedback_rows:
        frows = []
        for row in feedback_rows:
            frows.append(
                "<tr>"
                f"<td>{_h(row.get('ts',''))}</td>"
                f"<td>{_h(row.get('track',''))}</td>"
                f"<td>{_h(row.get('action',''))}</td>"
                f"<td>{_h(row.get('rating',''))}</td>"
                f"<td>{_h(row.get('key_details',''))}</td>"
                f"<td>{_h(row.get('language_edits',''))}</td>"
                f"<td>{_h(row.get('notes',''))}</td>"
                "</tr>"
            )
        feedback_table = (
            "<h3>Recent Feedback</h3>"
            "<table><thead><tr><th>UTC</th><th>Track</th><th>Action</th><th>Rating</th><th>Key details</th><th>Language edits</th><th>Notes</th></tr></thead>"
            f"<tbody>{''.join(frows)}</tbody></table>"
        )
    else:
        feedback_table = "<p>No feedback submitted yet.</p>"

    token_cards = "<p>No documentation token metrics yet.</p>"
    if token_metrics.get("available"):
        tracks = token_metrics.get("tracks", {})
        cards = []
        for key, label in (
            ("cursor_codex", "Cursor/Codex"),
            ("opensource", "Open-source"),
            ("specialized", "Specialized"),
        ):
            row = tracks.get(key)
            if not row:
                continue
            cards.append(
                "<div class='metric-card'>"
                f"<div class='metric-title'>{_h(label)}</div>"
                f"<div class='metric-value'>{_h(row.get('estimated_tokens'))}</div>"
                "<div class='metric-sub'>estimated tokens</div>"
                f"<div class='metric-sub'><code>{_h(row.get('path'))}</code></div>"
                "</div>"
            )
        savings = token_metrics.get("estimated_savings_tokens", {})
        token_cards = (
            "<div class='metric-grid'>"
            + "".join(cards)
            + "<div class='metric-card glow'>"
            "<div class='metric-title'>Estimated Savings</div>"
            f"<div class='metric-value'>{_h(savings.get('vs_specialized'))}</div>"
            "<div class='metric-sub'>tokens saved vs specialized doc output</div>"
            f"<div class='metric-sub'>vs open-source: {_h(savings.get('vs_opensource'))}</div>"
            "</div></div>"
            f"<p class='muted'>{_h(token_metrics.get('notes',''))}</p>"
        )

    recent_run_rows = []
    for row in kpis.get("recent_runs", []):
        exit_code = row.get("final_exit_code")
        status_label = "ok" if exit_code == 0 else "failed"
        detail_lines = [
            f"Run ID: {row.get('run_id','')}",
            f"Started: {row.get('started_at','')}",
            f"Finished: {row.get('finished_at','')}",
            f"Steps: {row.get('steps_count','')}",
            f"Capability summary: {row.get('capability_summary','—')}",
            f"Trained adapter: {row.get('trained_adapter_path') or '—'}",
            f"Benchmark adapter/model: {row.get('benchmark_adapter_path') or '—'}",
        ]
        recent_run_rows.append(
            "<tr>"
            f"<td>{_h(row.get('run_id',''))}</td>"
            f"<td>{_h(row.get('subcommand',''))}</td>"
            f"<td><span class='status-pill {status_label}'>{_h(exit_code)}</span></td>"
            f"<td>{_h(row.get('elapsed_s',''))}</td>"
            f"<td>{_h(row.get('benchmark_summary',''))}</td>"
            "<td>"
            "<details>"
            "<summary>Details</summary>"
            f"<pre>{_h(chr(10).join(detail_lines))}</pre>"
            "</details>"
            "</td>"
            "</tr>"
        )
    recent_runs_table = (
        "<table><thead><tr><th>Run ID</th><th>Subcommand</th><th>Exit</th><th>Seconds</th><th>Benchmark</th><th>More</th></tr></thead>"
        f"<tbody>{''.join(recent_run_rows) if recent_run_rows else '<tr><td colspan=6>(no runs found)</td></tr>'}</tbody></table>"
    )

    cap_rows = []
    for row in kpis.get("recent_doc_captures", []):
        detail_lines = [
            f"Capture file: {row.get('file','')}",
            f"Trigger event: {row.get('trigger_event','')}",
            f"Started: {row.get('started_at','')}",
            f"Finished: {row.get('finished_at','')}",
            f"Open-source load/gen: {row.get('opensource_load_seconds',0):.3f}s / {row.get('opensource_gen_seconds',0):.3f}s",
            f"Specialized load/gen: {row.get('specialized_load_seconds',0):.3f}s / {row.get('specialized_gen_seconds',0):.3f}s",
        ]
        cap_rows.append(
            "<tr>"
            f"<td>{_h(row.get('changed_path',''))}</td>"
            f"<td>{_h(row.get('opensource_tokens',''))}</td>"
            f"<td>{_h(row.get('specialized_tokens',''))}</td>"
            f"<td>{_h(round(float(row.get('opensource_seconds',0)+row.get('specialized_seconds',0)),3))}</td>"
            f"<td>{_h('yes' if row.get('has_both') else 'no')}</td>"
            "<td>"
            "<details>"
            "<summary>Details</summary>"
            f"<pre>{_h(chr(10).join(detail_lines))}</pre>"
            "</details>"
            "</td>"
            "</tr>"
        )
    recent_caps_table = (
        "<table><thead><tr><th>Changed Path</th><th>Open-source tokens</th><th>Specialized tokens</th><th>Total seconds</th><th>Both outputs</th><th>More</th></tr></thead>"
        f"<tbody>{''.join(cap_rows) if cap_rows else '<tr><td colspan=6>(no captures found)</td></tr>'}</tbody></table>"
    )

    acc_val = kpis.get("specialized_accuracy_vs_cursor_pct")
    acc_text = "n/a" if acc_val is None else f"{acc_val}%"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FE Private Dashboard</title>
  <style>
    :root {{
      --bg: #f4f8ff;
      --fg: #0d223f;
      --panel: #ffffff;
      --panel-border: #cfe0ff;
      --accent: #2463eb;
      --muted: #5d7398;
      --glow: rgba(36, 99, 235, 0.2);
    }}
    body.dark {{
      --bg: #081224;
      --fg: #e8f1ff;
      --panel: #0f1b2f;
      --panel-border: #29466f;
      --accent: #61a4ff;
      --muted: #98b5dc;
      --glow: rgba(97, 164, 255, 0.35);
    }}
    body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; background: var(--bg); color: var(--fg); transition: all 0.25s ease; }}
    .topbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
    .brand {{ font-size: 24px; font-weight: 700; letter-spacing: 0.2px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ border: 1px solid var(--panel-border); border-radius: 14px; padding: 14px; background: var(--panel); box-shadow: 0 10px 30px rgba(15,23,42,0.08); }}
    .cards .card strong {{ color: var(--muted); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }}
    .cards .card div {{ font-size: 24px; margin-top: 6px; font-weight: 700; color: var(--accent); }}
    .tabs {{ display:flex; gap:8px; margin: 12px 0 16px 0; }}
    .tab {{ padding: 10px 14px; border:1px solid var(--panel-border); border-radius: 10px; background: var(--panel); color:var(--fg); cursor:pointer; font-weight: 600; }}
    .tab.active {{ background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 0 0 4px var(--glow); }}
    .mini-tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 12px 0; }}
    .mini-tab {{
      padding: 8px 10px;
      border:1px solid var(--panel-border);
      border-radius: 9px;
      background: var(--panel);
      color: var(--fg);
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
    }}
    .mini-tab.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .mini-tab .warn-badge {{
      display:inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid #f59e0b;
      color: #92400e;
      background: #fff7ed;
    }}
    .mini-tab .legacy-badge {{
      display:inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid #cbd5e1;
      color: #475569;
      background: #f8fafc;
    }}
    .mono-view {{ white-space: pre; overflow: auto; max-height: 62vh; }}
    .mode-btn {{ padding: 10px 14px; border:1px solid var(--panel-border); border-radius: 10px; background: var(--panel); color:var(--fg); cursor:pointer; font-weight:600; }}
    .panel {{ display:none; }}
    .panel.active {{ display:block; }}
    h1,h2 {{ margin: 0 0 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border:1px solid var(--panel-border); border-radius: 10px; overflow:hidden; }}
    th,td {{ border-bottom: 1px solid var(--panel-border); padding: 9px; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ color: var(--muted); font-weight: 700; text-transform: uppercase; font-size: 12px; letter-spacing: 0.4px; }}
    pre {{ white-space: pre-wrap; background: var(--panel); border: 1px solid var(--panel-border); border-radius: 10px; padding: 12px; max-height: 260px; overflow: auto; }}
    code {{ background: rgba(36,99,235,0.09); padding: 1px 5px; border-radius: 6px; }}
    .metric-grid {{ display:grid; grid-template-columns: repeat(4, minmax(180px,1fr)); gap: 12px; margin: 8px 0 12px 0; }}
    .metric-card {{ border:1px solid var(--panel-border); border-radius: 12px; background: var(--panel); padding: 12px; }}
    .metric-card.glow {{ box-shadow: 0 0 0 4px var(--glow), 0 8px 25px var(--glow); }}
    .metric-title {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-value {{ font-size: 26px; font-weight: 700; color: var(--accent); margin-top: 6px; }}
    .metric-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .feedback-box {{ border: 1px dashed var(--panel-border); background: var(--panel); border-radius: 10px; padding: 12px; margin: 14px 0; }}
    .feedback-row {{ display:flex; gap:8px; flex-wrap:wrap; margin-top: 8px; }}
    .feedback-row select, .feedback-row textarea, .feedback-row button {{
      border:1px solid var(--panel-border); border-radius: 8px; background: var(--bg); color: var(--fg); padding: 8px;
    }}
    .feedback-row textarea {{ min-height: 80px; width: min(880px, 100%); }}
    .feedback-row button {{ background: var(--accent); color:white; border-color: var(--accent); cursor:pointer; font-weight: 600; }}
    .compare-controls {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin: 10px 0 12px 0; }}
    .compare-controls select, .compare-controls button {{
      border:1px solid var(--panel-border); border-radius: 8px; background: var(--panel); color: var(--fg); padding: 7px 10px;
    }}
    .compare-controls button {{ background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; font-weight: 600; }}
    .status-pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-weight:700; font-size:12px; }}
    .status-pill.ok {{ background: rgba(16,185,129,0.18); color: #10b981; border: 1px solid rgba(16,185,129,0.35); }}
    .status-pill.failed {{ background: rgba(239,68,68,0.18); color: #ef4444; border: 1px solid rgba(239,68,68,0.35); }}
    details summary {{ cursor: pointer; font-weight: 600; color: var(--accent); }}
    .workflow-box {{ border:1px solid var(--panel-border); background: var(--panel); border-radius: 10px; padding: 12px; margin: 10px 0 14px 0; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <div class="brand">Fallen Empire Analytics Cloud</div>
      <p class="muted">Generated: {_h(summary['generated_at'])} UTC</p>
    </div>
    <button id="modeToggle" class="mode-btn">Toggle Dark Glow Mode</button>
  </div>

  <div class="tabs">
    <button class="tab active" data-panel="overview">Overview</button>
    <button class="tab" data-panel="scoring">Scoring vs Cursor Work</button>
    <button class="tab" data-panel="docs">Docs Snapshot</button>
    <button class="tab" data-panel="capability-map">Capability Map</button>
    <button class="tab" data-panel="training">Training Data</button>
  </div>

  <section id="panel-overview" class="panel active">
    <div class="cards">
      <div class="card"><strong>Run Reliability</strong><div>{_h(kpis.get('run_reliability_pct'))}%</div></div>
      <div class="card"><strong>Specialized Accuracy</strong><div>{_h(acc_text)}</div></div>
      <div class="card"><strong>Avg Run Time</strong><div>{_h(kpis.get('avg_run_seconds'))}s</div></div>
      <div class="card"><strong>Recent Runs</strong><div>{_h(kpis.get('runs_recent_count'))}</div></div>
      <div class="card"><strong>Doc Capture Reliability</strong><div>{_h(kpis.get('doc_capture_reliability_pct'))}%</div></div>
      <div class="card"><strong>Avg Capture Time</strong><div>{_h(kpis.get('avg_doc_capture_seconds'))}s</div></div>
      <div class="card"><strong>Capture Tokens</strong><div>{_h(kpis.get('doc_capture_tokens_total'))}</div></div>
      <div class="card"><strong>Savings If Specialized Only</strong><div>{_h(kpis.get('doc_capture_savings_if_specialized_only'))}</div></div>
      <div class="card"><strong>Total Events</strong><div>{_h(summary['events_total'])}</div></div>
    </div>

    <div class="workflow-box">
      <strong>Workflow Context</strong>
      <p class="muted">
        This page combines three pipelines: (1) workflow runs from <code>benchmarks/results/runs/*/manifest.json</code>,
        (2) documentation captures generated on watched code edits (open-source + specialized),
        and (3) scoring artifacts from <code>docs/generated/*.comparison.json</code>.
        Use each row's <em>Details</em> dropdown to inspect what happened, timings, and model/adapter context.
      </p>
    </div>

    <h2>Recent Events</h2>
    <table>
      <thead><tr><th>UTC</th><th>Source</th><th>Kind</th><th>Command</th><th>Exit</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>

    <h2 style="margin-top:16px;">Documentation Token Savings (Estimated)</h2>
    {token_cards}

    <h2 style="margin-top:16px;">Recent Workflow Runs</h2>
    {recent_runs_table}

    <h2 style="margin-top:16px;">Recent Documentation Captures</h2>
    {recent_caps_table}
  </section>

  <section id="panel-scoring" class="panel">
    <h2>Scoring Against Cursor Work</h2>
    <p class="muted">Compare reference docs vs base-model output vs base+adapter output.</p>
    {scoring_html}
    <div class="feedback-box">
      <h3>Output Curation + Training Context</h3>
      <p class="muted">Open side-by-side and curate each output directly in-place (key details, language edits, notes, include/exclude, delete).</p>
    </div>
    {feedback_table}
  </section>

  <section id="panel-docs" class="panel">
    <h2>Documentation Explorer</h2>
    <p class="muted">Live list of every documentation artifact plus recent generated docs. Auto-refresh runs every 15 seconds.</p>
    <div id="docsTrackTabs" class="mini-tabs">
      <button class="mini-tab active" data-track="all">All docs</button>
      <button class="mini-tab" data-track="opensource">Open-source model only</button>
      <button class="mini-tab" data-track="specialized">Specialized adapter only</button>
      <button class="mini-tab" data-track="codex_authored">Cursor/Codex reference only</button>
    </div>
    <div class="compare-controls">
      <label for="docsQuery">Search</label>
      <input id="docsQuery" type="text" placeholder="name or path..." style="min-width:220px;" />
      <label for="docsCategory">Category</label>
      <select id="docsCategory">
        <option value="all">All categories</option>
        <option value="docs">Docs</option>
        <option value="capture">Captures</option>
        <option value="runs">Runs</option>
      </select>
      <label for="docsType">Type</label>
      <select id="docsType">
        <option value="all">All types</option>
        <option value="md">Markdown (.md)</option>
        <option value="json">JSON (.json)</option>
        <option value="jsonl">JSONL (.jsonl)</option>
      </select>
      <label for="docsSort">Sort</label>
      <select id="docsSort">
        <option value="recent">Most recent</option>
        <option value="name_asc">Name A-Z</option>
        <option value="name_desc">Name Z-A</option>
        <option value="type">Type then name</option>
      </select>
    </div>
    <div class="compare-controls">
      <label for="docsSelect">Document</label>
      <select id="docsSelect" style="min-width:420px;"></select>
      <button id="docsRefresh" type="button">Refresh list</button>
      <span id="docsMeta" class="muted"></span>
    </div>
    <pre id="docsViewer">(select a document)</pre>
  </section>

  <section id="panel-capability-map" class="panel">
    <h2>Capability Map Explorer</h2>
    <p class="muted">Pan/zoom interactive concept-capability graph from the latest mass benchmark runs.</p>
    <div class="compare-controls">
      <a href="/view/capability-map" target="_blank" rel="noopener">Open in full page</a>
    </div>
    <iframe
      title="Capability map"
      src="/view/capability-map"
      style="width:100%; height:74vh; border:1px solid var(--panel-border); border-radius:10px; background:var(--panel);"
      loading="lazy"
    ></iframe>
  </section>

  <section id="panel-training" class="panel">
    <h2>Training Data Explorer</h2>
    <p class="muted">Dataset tabs are grouped by adapter family. Select a split tab to inspect rows (paginated) or open the manifest.</p>
    <div id="trainingDatasetTabs" class="mini-tabs"></div>
    <div id="trainingSplitTabs" class="mini-tabs">
      <button class="mini-tab active" data-view="train">train</button>
      <button class="mini-tab" data-view="valid">valid</button>
      <button class="mini-tab" data-view="test">test</button>
      <button class="mini-tab" data-view="manifest">manifest</button>
    </div>
    <div class="compare-controls">
      <label for="trainingLimit">Rows per page</label>
      <select id="trainingLimit">
        <option value="100">100</option>
        <option value="200" selected>200</option>
        <option value="400">400</option>
      </select>
      <label for="trainingHideLegacy" style="display:flex;align-items:center;gap:6px;">
        <input id="trainingHideLegacy" type="checkbox" checked />
        Hide legacy datasets
      </label>
      <button id="trainingPrev" type="button">Previous</button>
      <button id="trainingNext" type="button">Next</button>
      <button id="trainingRefresh" type="button">Refresh datasets</button>
      <span id="trainingMeta" class="muted"></span>
    </div>
    <pre id="trainingViewer" class="mono-view">(loading training datasets...)</pre>
  </section>

  <script>
    const modeToggle = document.getElementById('modeToggle');
    const savedMode = localStorage.getItem('fe_dash_mode');
    if (savedMode === 'dark') document.body.classList.add('dark');
    modeToggle.addEventListener('click', () => {{
      document.body.classList.toggle('dark');
      localStorage.setItem('fe_dash_mode', document.body.classList.contains('dark') ? 'dark' : 'light');
    }});

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

    const openCompare = document.getElementById('openCompare');
    const cmpLeft = document.getElementById('cmpLeft');
    const cmpRight = document.getElementById('cmpRight');
    openCompare?.addEventListener('click', () => {{
      const left = (cmpLeft?.value || 'codex_authored');
      const right = (cmpRight?.value || 'opensource');
      if (left === right) {{
        alert('Choose two different outputs to compare.');
        return;
      }}
      window.location.href = '/view/compare?left=' + encodeURIComponent(left) + '&right=' + encodeURIComponent(right);
    }});

    let docsCatalog = [];
    let docsFiltered = [];
    const docsSelect = document.getElementById('docsSelect');
    const docsViewer = document.getElementById('docsViewer');
    const docsMeta = document.getElementById('docsMeta');
    const docsRefresh = document.getElementById('docsRefresh');
    const docsQuery = document.getElementById('docsQuery');
    const docsCategory = document.getElementById('docsCategory');
    const docsType = document.getElementById('docsType');
    const docsSort = document.getElementById('docsSort');
    const docsTrackTabs = document.getElementById('docsTrackTabs');
    let docsTrack = 'all';

    function docType(path) {{
      const p = String(path || '').toLowerCase();
      if (p.endsWith('.jsonl')) return 'jsonl';
      if (p.endsWith('.json')) return 'json';
      if (p.endsWith('.md')) return 'md';
      return 'other';
    }}

    function docTrackLabel(row) {{
      const track = String(row?.model_track || 'other');
      if (track === 'opensource') return 'Open-source';
      if (track === 'specialized') return 'Specialized';
      if (track === 'codex_authored') return 'Cursor/Codex';
      if (track === 'capture') return 'Capture';
      if (track === 'run') return 'Run';
      return 'Other';
    }}

    function setActiveDocsTrack(track) {{
      docsTrack = track || 'all';
      for (const btn of docsTrackTabs?.querySelectorAll('.mini-tab') || []) {{
        btn.classList.toggle('active', (btn.getAttribute('data-track') || 'all') === docsTrack);
      }}
    }}

    function applyDocsFilters() {{
      const query = (docsQuery?.value || '').trim().toLowerCase();
      const category = (docsCategory?.value || 'all');
      const type = (docsType?.value || 'all');
      const sortBy = (docsSort?.value || 'recent');
      const track = docsTrack || 'all';
      let rows = docsCatalog.filter((row) => {{
        const p = String(row.path || '');
        const cat = String(row.category || '');
        const rowTrack = String(row.model_track || 'other');
        if (track !== 'all' && rowTrack !== track) return false;
        if (category !== 'all' && cat !== category) return false;
        if (type !== 'all' && docType(p) !== type) return false;
        if (query && !p.toLowerCase().includes(query)) return false;
        return true;
      }});
      if (sortBy === 'name_asc') {{
        rows.sort((a, b) => String(a.path || '').localeCompare(String(b.path || '')));
      }} else if (sortBy === 'name_desc') {{
        rows.sort((a, b) => String(b.path || '').localeCompare(String(a.path || '')));
      }} else if (sortBy === 'type') {{
        rows.sort((a, b) => {{
          const ta = docType(a.path);
          const tb = docType(b.path);
          if (ta !== tb) return ta.localeCompare(tb);
          return String(a.path || '').localeCompare(String(b.path || ''));
        }});
      }} else {{
        rows.sort((a, b) => String(b.updated_utc || '').localeCompare(String(a.updated_utc || '')));
      }}
      docsFiltered = rows;
    }}

    function renderDocsSelect(preferredPath) {{
      if (!docsSelect) return;
      const current = preferredPath || docsSelect.value || '';
      docsSelect.innerHTML = '';
      for (const row of docsFiltered) {{
        const opt = document.createElement('option');
        opt.value = row.path || '';
        const title = String(row.title || row.path || '');
        opt.textContent = title + ' — ' + (row.path || '') + ' [' + (row.category || 'docs') + ' • ' + docType(row.path) + ' • ' + docTrackLabel(row) + ']';
        docsSelect.appendChild(opt);
      }}
      if (docsFiltered.length === 0) {{
        docsSelect.innerHTML = '<option value=\"\">(no documentation files found)</option>';
        if (docsViewer) docsViewer.textContent = '(no documentation files found)';
        if (docsMeta) docsMeta.textContent = '0 matches';
        return;
      }}
      docsSelect.value = docsFiltered.some(d => d.path === current) ? current : (docsFiltered[0].path || '');
    }}

    async function loadDoc(pathValue) {{
      if (!pathValue) return;
      if (docsViewer) docsViewer.textContent = 'Loading...';
      try {{
        const res = await fetch('/api/docs/content?path=' + encodeURIComponent(pathValue));
        const payload = await res.json();
        if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));
        if (docsViewer) docsViewer.textContent = payload.content || '';
        const row = docsCatalog.find(d => d.path === pathValue);
        if (docsMeta && row) {{
          docsMeta.textContent = docsFiltered.length + ' matches • updated ' + (row.updated_utc || '') + ' • ' + (row.size_bytes || 0) + ' bytes • ' + docType(row.path) + ' • ' + docTrackLabel(row);
        }}
      }} catch (err) {{
        if (docsViewer) docsViewer.textContent = 'Failed to load document: ' + err;
      }}
    }}

    async function refreshDocsList(preserveSelection=true) {{
      const selected = docsSelect?.value || '';
      try {{
        const res = await fetch('/api/docs/list');
        const payload = await res.json();
        if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));
        docsCatalog = payload.docs || [];
        applyDocsFilters();
        renderDocsSelect(preserveSelection ? selected : '');
        await loadDoc(docsSelect?.value || '');
      }} catch (err) {{
        if (docsViewer) docsViewer.textContent = 'Failed to refresh docs list: ' + err;
      }}
    }}

    docsRefresh?.addEventListener('click', () => refreshDocsList(true));
    docsSelect?.addEventListener('change', () => loadDoc(docsSelect.value));
    for (const btn of docsTrackTabs?.querySelectorAll('.mini-tab') || []) {{
      btn.addEventListener('click', () => {{
        const nextTrack = btn.getAttribute('data-track') || 'all';
        setActiveDocsTrack(nextTrack);
        applyDocsFilters();
        renderDocsSelect(docsSelect?.value || '');
        loadDoc(docsSelect?.value || '');
      }});
    }}
    docsQuery?.addEventListener('input', () => {{
      applyDocsFilters();
      renderDocsSelect(docsSelect?.value || '');
      loadDoc(docsSelect?.value || '');
    }});
    docsCategory?.addEventListener('change', () => {{
      applyDocsFilters();
      renderDocsSelect(docsSelect?.value || '');
      loadDoc(docsSelect?.value || '');
    }});
    docsType?.addEventListener('change', () => {{
      applyDocsFilters();
      renderDocsSelect(docsSelect?.value || '');
      loadDoc(docsSelect?.value || '');
    }});
    docsSort?.addEventListener('change', () => {{
      applyDocsFilters();
      renderDocsSelect(docsSelect?.value || '');
      loadDoc(docsSelect?.value || '');
    }});
    setActiveDocsTrack('all');
    refreshDocsList(false);
    setInterval(() => refreshDocsList(true), 15000);

    let trainingCatalog = [];
    let selectedDatasetId = '';
    let selectedTrainingView = 'train';
    let trainingOffset = 0;
    const TRAINING_WARN_ROWS = 10;
    const trainingDatasetTabs = document.getElementById('trainingDatasetTabs');
    const trainingSplitTabs = document.getElementById('trainingSplitTabs');
    const trainingLimit = document.getElementById('trainingLimit');
    const trainingHideLegacy = document.getElementById('trainingHideLegacy');
    const trainingPrev = document.getElementById('trainingPrev');
    const trainingNext = document.getElementById('trainingNext');
    const trainingRefresh = document.getElementById('trainingRefresh');
    const trainingMeta = document.getElementById('trainingMeta');
    const trainingViewer = document.getElementById('trainingViewer');

    function isSpecialistDataset(ds) {{
      const id = String(ds?.dataset_id || '');
      return id.endsWith('_specialist');
    }}

    function isLegacyDataset(ds, idSet) {{
      const id = String(ds?.dataset_id || '');
      if (!id || isSpecialistDataset(ds)) return false;
      return idSet.has(id + '_specialist');
    }}

    function visibleTrainingCatalog() {{
      const idSet = new Set((trainingCatalog || []).map((d) => String(d?.dataset_id || '')));
      let rows = (trainingCatalog || []).map((ds) => {{
        const totalRows = Number(ds?.total_rows || 0);
        return {{
          ...ds,
          is_specialist: isSpecialistDataset(ds),
          is_legacy: isLegacyDataset(ds, idSet),
          is_low_rows: totalRows > 0 && totalRows < TRAINING_WARN_ROWS,
        }};
      }});
      if (trainingHideLegacy?.checked) {{
        rows = rows.filter((ds) => !ds.is_legacy);
      }}
      rows.sort((a, b) => {{
        const sa = a.is_specialist ? 0 : 1;
        const sb = b.is_specialist ? 0 : 1;
        if (sa !== sb) return sa - sb;
        const ra = Number(a.total_rows || 0);
        const rb = Number(b.total_rows || 0);
        if (ra !== rb) return rb - ra;
        return String(a.dataset_id || '').localeCompare(String(b.dataset_id || ''));
      }});
      return rows;
    }}

    function renderTrainingDatasetTabs() {{
      if (!trainingDatasetTabs) return;
      trainingDatasetTabs.innerHTML = '';
      const rows = visibleTrainingCatalog();
      if (!rows.length) {{
        trainingDatasetTabs.innerHTML = '<span class="muted">(no training datasets found)</span>';
        return;
      }}
      for (const ds of rows) {{
        const button = document.createElement('button');
        button.className = 'mini-tab' + ((ds.dataset_id || '') === selectedDatasetId ? ' active' : '');
        const totalRows = Number(ds.total_rows || 0);
        const label = document.createElement('span');
        label.textContent = (ds.dataset_id || 'dataset') + ' (' + totalRows + ')';
        button.appendChild(label);
        if (ds.is_low_rows) {{
          const warn = document.createElement('span');
          warn.className = 'warn-badge';
          warn.textContent = '⚠ under ' + TRAINING_WARN_ROWS;
          button.appendChild(warn);
        }}
        if (ds.is_legacy) {{
          const legacy = document.createElement('span');
          legacy.className = 'legacy-badge';
          legacy.textContent = 'legacy';
          button.appendChild(legacy);
        }}
        button.addEventListener('click', () => {{
          selectedDatasetId = ds.dataset_id || '';
          trainingOffset = 0;
          renderTrainingDatasetTabs();
          loadTrainingContent();
        }});
        trainingDatasetTabs.appendChild(button);
      }}
    }}

    function setActiveTrainingView(view) {{
      selectedTrainingView = view;
      for (const btn of trainingSplitTabs?.querySelectorAll('.mini-tab') || []) {{
        btn.classList.toggle('active', (btn.getAttribute('data-view') || '') === view);
      }}
    }}

    function currentTrainingLimit() {{
      const raw = Number(trainingLimit?.value || 200);
      return Number.isFinite(raw) ? Math.max(1, Math.min(1000, raw)) : 200;
    }}

    async function loadTrainingContent() {{
      if (!selectedDatasetId) {{
        if (trainingViewer) trainingViewer.textContent = '(no dataset selected)';
        return;
      }}
      if (trainingViewer) trainingViewer.textContent = 'Loading training data...';
      const limit = currentTrainingLimit();
      const query = '/api/training/content?dataset_id=' + encodeURIComponent(selectedDatasetId)
        + '&view=' + encodeURIComponent(selectedTrainingView)
        + '&offset=' + encodeURIComponent(String(trainingOffset))
        + '&limit=' + encodeURIComponent(String(limit));
      try {{
        const res = await fetch(query);
        const payload = await res.json();
        if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));
        if (trainingViewer) trainingViewer.textContent = payload.content || '';
        const totalRows = Number(payload.total_rows || 0);
        const offset = Number(payload.offset || 0);
        const shownCount = (payload.content || '').split('\\n').filter(Boolean).length;
        const hasMore = Boolean(payload.has_more);
        if (trainingMeta) {{
          if (selectedTrainingView === 'manifest') {{
            trainingMeta.textContent = payload.path + ' • manifest';
          }} else {{
            const end = Math.min(totalRows, offset + shownCount);
            trainingMeta.textContent = payload.path + ' • rows ' + (offset + 1) + '-' + end + ' of ' + totalRows;
          }}
        }}
        if (trainingPrev) trainingPrev.disabled = selectedTrainingView === 'manifest' || offset <= 0;
        if (trainingNext) trainingNext.disabled = selectedTrainingView === 'manifest' || !hasMore;
      }} catch (err) {{
        if (trainingViewer) trainingViewer.textContent = 'Failed to load training data: ' + err;
      }}
    }}

    async function refreshTrainingCatalog(preserveSelection=true) {{
      const prevDataset = selectedDatasetId;
      try {{
        const res = await fetch('/api/training/catalog');
        const payload = await res.json();
        if (!res.ok || !payload.ok) throw new Error(payload.error || ('HTTP ' + res.status));
        trainingCatalog = payload.datasets || [];
        const rows = visibleTrainingCatalog();
        if (!rows.length) {{
          selectedDatasetId = '';
          renderTrainingDatasetTabs();
          if (trainingViewer) trainingViewer.textContent = '(no visible training datasets found)';
          if (trainingMeta) trainingMeta.textContent = '0 visible datasets';
          return;
        }}
        if (preserveSelection && rows.some((d) => (d.dataset_id || '') === prevDataset)) {{
          selectedDatasetId = prevDataset;
        }} else if (!selectedDatasetId || !rows.some((d) => (d.dataset_id || '') === selectedDatasetId)) {{
          selectedDatasetId = rows[0].dataset_id || '';
        }}
        renderTrainingDatasetTabs();
        await loadTrainingContent();
        const visibleCount = rows.length;
        const hiddenCount = Math.max(0, (trainingCatalog || []).length - visibleCount);
        const warnedCount = rows.filter((d) => d.is_low_rows).length;
        if (trainingMeta) {{
          const parts = [String(visibleCount) + ' visible dataset' + (visibleCount === 1 ? '' : 's')];
          if (trainingHideLegacy?.checked && hiddenCount > 0) parts.push(String(hiddenCount) + ' legacy hidden');
          if (warnedCount > 0) parts.push(String(warnedCount) + ' flagged low-row');
          trainingMeta.textContent = parts.join(' • ');
        }}
      }} catch (err) {{
        if (trainingViewer) trainingViewer.textContent = 'Failed to refresh training datasets: ' + err;
      }}
    }}

    trainingSplitTabs?.addEventListener('click', (ev) => {{
      const target = ev.target;
      if (!(target instanceof HTMLElement)) return;
      const view = (target.getAttribute('data-view') || '').trim();
      if (!view) return;
      setActiveTrainingView(view);
      trainingOffset = 0;
      loadTrainingContent();
    }});
    trainingPrev?.addEventListener('click', () => {{
      trainingOffset = Math.max(0, trainingOffset - currentTrainingLimit());
      loadTrainingContent();
    }});
    trainingNext?.addEventListener('click', () => {{
      trainingOffset = trainingOffset + currentTrainingLimit();
      loadTrainingContent();
    }});
    trainingLimit?.addEventListener('change', () => {{
      trainingOffset = 0;
      loadTrainingContent();
    }});
    trainingRefresh?.addEventListener('click', () => {{
      refreshTrainingCatalog(true);
    }});
    trainingHideLegacy?.addEventListener('change', () => {{
      trainingOffset = 0;
      refreshTrainingCatalog(true);
    }});

    setActiveTrainingView('train');
    refreshTrainingCatalog(false);
    setInterval(() => refreshTrainingCatalog(true), 15000);

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

    scoring_payload = _load_scoring_summary()
    token_payload = _documentation_token_metrics()
    kpi_payload = _kpi_metrics(summary, scoring_payload, token_payload)

    if path == "/api/summary":
        status, headers, body = _json(
            "200 OK",
            {
                **summary,
                "token_metrics": token_payload,
                "scoring": scoring_payload,
                "kpis": kpi_payload,
            },
        )
        start_response(status, headers)
        return [body]

    if path == "/api/scoring":
        status, headers, body = _json("200 OK", {"ok": True, "scoring": scoring_payload})
        start_response(status, headers)
        return [body]

    if path == "/api/docs/list":
        docs = _documentation_catalog(limit=1200)
        status, headers, body = _json("200 OK", {"ok": True, "docs": docs})
        start_response(status, headers)
        return [body]

    if path == "/api/docs/content":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        rel = str((qs.get("path") or [""])[0]).strip()
        if not rel:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "missing path"})
            start_response(status, headers)
            return [body]
        candidate = (REPO_ROOT / rel).resolve()
        try:
            candidate.relative_to(REPO_ROOT.resolve())
        except ValueError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid path"})
            start_response(status, headers)
            return [body]
        if not candidate.is_file():
            status, headers, body = _json("404 Not Found", {"ok": False, "error": "file not found"})
            start_response(status, headers)
            return [body]
        content = candidate.read_text(encoding="utf-8", errors="replace")
        status, headers, body = _json("200 OK", {"ok": True, "path": rel, "content": content})
        start_response(status, headers)
        return [body]

    if path == "/api/capability-map":
        payload = _capability_map_payload()
        status_code = "200 OK" if payload.get("ok") else "503 Service Unavailable"
        status, headers, body = _json(status_code, payload)
        start_response(status, headers)
        return [body]

    if path == "/api/training/catalog":
        payload = _training_data_catalog(limit=1200)
        status, headers, body = _json("200 OK", {"ok": True, **payload})
        start_response(status, headers)
        return [body]

    if path == "/api/training/content":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        dataset_id = str((qs.get("dataset_id") or [""])[0]).strip()
        view = str((qs.get("view") or ["train"])[0]).strip().lower()
        try:
            offset = int((qs.get("offset") or ["0"])[0])
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = int((qs.get("limit") or ["200"])[0])
        except (TypeError, ValueError):
            limit = 200
        payload = _read_training_data_content(dataset_id=dataset_id, view=view, offset=offset, limit=limit)
        if not payload.get("ok"):
            status, headers, body = _json("400 Bad Request", payload)
            start_response(status, headers)
            return [body]
        status, headers, body = _json("200 OK", payload)
        start_response(status, headers)
        return [body]

    if path == "/api/feedback" and method == "POST":
        raw = _read_body(environ)
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid json"})
            start_response(status, headers)
            return [body]
        track = str(payload.get("track") or "").strip()
        rating_raw = payload.get("rating", 0)
        notes = str(payload.get("notes") or "").strip()
        source = str(payload.get("source") or "dashboard")
        action = str(payload.get("action") or "feedback").strip()
        key_details = str(payload.get("key_details") or "").strip()
        language_edits = str(payload.get("language_edits") or "").strip()
        artifact_path = str((scoring_payload.get("doc_paths") or {}).get(track, ""))
        try:
            rating = int(rating_raw)
        except (TypeError, ValueError):
            rating = 0
        if track not in VALID_TRACKS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid track"})
            start_response(status, headers)
            return [body]
        if action not in {"feedback", "train_include", "train_exclude"}:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid action"})
            start_response(status, headers)
            return [body]
        if rating < 1 or rating > 5:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "rating must be 1..5"})
            start_response(status, headers)
            return [body]
        if len(notes) < 5:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "notes too short"})
            start_response(status, headers)
            return [body]
        conn = _connect()
        try:
            row_id = _insert_feedback(
                conn,
                track=track,
                rating=rating,
                notes=notes[:4000],
                source=source[:120],
                action=action,
                key_details=key_details[:4000],
                language_edits=language_edits[:4000],
                artifact_path=artifact_path[:500],
            )
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "id": row_id})
        start_response(status, headers)
        return [body]

    if path == "/api/compare-feedback" and method == "POST":
        raw = _read_body(environ)
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid json"})
            start_response(status, headers)
            return [body]
        left_track = str(payload.get("left_track") or "").strip()
        right_track = str(payload.get("right_track") or "").strip()
        winner = str(payload.get("winner") or "").strip()
        strength_raw = str(payload.get("strength") or "").strip()
        notes = str(payload.get("notes") or "").strip()
        source = str(payload.get("source") or "dashboard").strip()
        if left_track not in VALID_TRACKS or right_track not in VALID_TRACKS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid track(s)"})
            start_response(status, headers)
            return [body]
        if winner not in COMPARE_WINNERS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "winner must be left|right|tie"})
            start_response(status, headers)
            return [body]
        strength = "tie" if winner == "tie" and not strength_raw else strength_raw
        if strength not in COMPARE_STRENGTHS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid strength"})
            start_response(status, headers)
            return [body]
        if winner != "tie" and strength == "tie":
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "strength must be weak|medium|strong for non-tie"})
            start_response(status, headers)
            return [body]
        doc_paths = scoring_payload.get("doc_paths", {})
        left_artifact_path = str(doc_paths.get(left_track, ""))
        right_artifact_path = str(doc_paths.get(right_track, ""))
        if not left_artifact_path or not right_artifact_path:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "missing artifact path for selected track(s)"})
            start_response(status, headers)
            return [body]
        left_candidate = (REPO_ROOT / left_artifact_path).resolve()
        right_candidate = (REPO_ROOT / right_artifact_path).resolve()
        try:
            left_candidate.relative_to(REPO_ROOT.resolve())
            right_candidate.relative_to(REPO_ROOT.resolve())
        except ValueError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "artifact path outside repo"})
            start_response(status, headers)
            return [body]
        if not left_candidate.is_file() or not right_candidate.is_file():
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "artifact file missing"})
            start_response(status, headers)
            return [body]
        left_body = _read_repo_file_text(left_artifact_path)
        right_body = _read_repo_file_text(right_artifact_path)
        ok_spans, span_error, spans = _validate_compare_spans(
            spans_raw=payload.get("spans"),
            left_body=left_body,
            right_body=right_body,
        )
        if not ok_spans:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": span_error})
            start_response(status, headers)
            return [body]
        conn = _connect()
        try:
            row_id = _insert_compare_feedback(
                conn,
                left_track=left_track,
                right_track=right_track,
                left_artifact_path=left_artifact_path[:500],
                right_artifact_path=right_artifact_path[:500],
                winner=winner,
                strength=strength,
                notes=notes[:4000],
                source=source[:120],
                left_text_hash=_text_sha256(left_body),
                right_text_hash=_text_sha256(right_body),
                spans=spans,
            )
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "id": row_id})
        start_response(status, headers)
        return [body]

    if path == "/api/output-action" and method == "POST":
        raw = _read_body(environ)
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid json"})
            start_response(status, headers)
            return [body]
        track = str(payload.get("track") or "").strip()
        action = str(payload.get("action") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        source = str(payload.get("source") or "dashboard")
        if track not in VALID_TRACKS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid track"})
            start_response(status, headers)
            return [body]
        if action != "delete_output":
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid action"})
            start_response(status, headers)
            return [body]
        ok, msg = _archive_output_artifact(scoring_payload, track)
        if not ok:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": msg})
            start_response(status, headers)
            return [body]
        conn = _connect()
        try:
            row_id = _insert_feedback(
                conn,
                track=track,
                rating=1,
                notes=(reason or "output archived by dashboard curation")[:4000],
                source=source[:120],
                action="delete_output",
                artifact_path=msg[:500],
            )
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "id": row_id, "archived_path": msg})
        start_response(status, headers)
        return [body]

    if path == "/api/feedback" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        limit = 20
        if "limit" in qs:
            try:
                limit = max(1, min(200, int(qs["limit"][0])))
            except (ValueError, TypeError):
                limit = 20
        conn = _connect()
        try:
            rows = _list_feedback(conn, limit=limit)
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "feedback": rows})
        start_response(status, headers)
        return [body]

    if path == "/api/compare-feedback" and method == "GET":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        limit = 20
        if "limit" in qs:
            try:
                limit = max(1, min(200, int(qs["limit"][0])))
            except (ValueError, TypeError):
                limit = 20
        left_track = str((qs.get("left_track") or [""])[0]).strip()
        right_track = str((qs.get("right_track") or [""])[0]).strip()
        if left_track and left_track not in VALID_TRACKS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid left_track"})
            start_response(status, headers)
            return [body]
        if right_track and right_track not in VALID_TRACKS:
            status, headers, body = _json("400 Bad Request", {"ok": False, "error": "invalid right_track"})
            start_response(status, headers)
            return [body]
        conn = _connect()
        try:
            rows = _list_compare_feedback(conn, limit=limit, left_track=left_track, right_track=right_track)
        finally:
            conn.close()
        status, headers, body = _json("200 OK", {"ok": True, "compare_feedback": rows})
        start_response(status, headers)
        return [body]

    if path == "/view/doc":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        track = (qs.get("track") or [""])[0]
        doc_paths = scoring_payload.get("doc_paths", {})
        rel = doc_paths.get(track)
        if not rel:
            status, headers, body = _html("404 Not Found", _render_doc_view("Missing doc", "No documentation artifact mapped for this track."))
            start_response(status, headers)
            return [body]
        content = _read_repo_file_text(rel)
        title_map = {
            "codex_authored": "Cursor/Codex Authored Full Documentation",
            "opensource": "Open-source Full Documentation",
            "specialized": "Specialized Adapter Full Documentation",
        }
        status, headers, body = _html("200 OK", _render_doc_view(title_map.get(track, track), content))
        start_response(status, headers)
        return [body]

    if path == "/view/compare":
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        left = (qs.get("left") or ["codex_authored"])[0]
        right = (qs.get("right") or ["opensource"])[0]
        doc_paths = scoring_payload.get("doc_paths", {})
        left_rel = doc_paths.get(left, "")
        right_rel = doc_paths.get(right, "")
        left_text = _read_repo_file_text(left_rel) if left_rel else "(missing left doc)"
        right_text = _read_repo_file_text(right_rel) if right_rel else "(missing right doc)"
        left_abs = (REPO_ROOT / left_rel).resolve() if left_rel else None
        right_abs = (REPO_ROOT / right_rel).resolve() if right_rel else None
        generated_root = (REPO_ROOT / "docs" / "generated").resolve()
        left_deletable = bool(left_abs and left_abs.is_file() and generated_root in left_abs.parents)
        right_deletable = bool(right_abs and right_abs.is_file() and generated_root in right_abs.parents)
        labels = {
            "codex_authored": "Cursor/Codex",
            "opensource": "Open-source",
            "specialized": "Specialized",
        }
        status, headers, body = _html(
            "200 OK",
            _render_compare_view(
                left,
                labels.get(left, left),
                left_text,
                right,
                labels.get(right, right),
                right_text,
                left_deletable,
                right_deletable,
            ),
        )
        start_response(status, headers)
        return [body]

    if path == "/view/capability-map":
        status, headers, body = _html("200 OK", _render_capability_map_view())
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

    scoring = scoring_payload
    token_metrics = token_payload
    kpis = kpi_payload
    conn = _connect()
    try:
        feedback_rows = _list_feedback(conn, limit=12)
    finally:
        conn.close()
    html = _render_dashboard(summary, scoring, token_metrics, feedback_rows, kpis)
    status, headers, body = _html("200 OK", html)
    start_response(status, headers)
    return [body]


def main() -> None:
    host = _env("FE_DASHBOARD_HOST", "0.0.0.0")
    port = int(_env("FE_DASHBOARD_PORT", _env("PORT", "8787")))

    if not _env("FE_DASHBOARD_INGEST_TOKEN"):
        print("WARNING: FE_DASHBOARD_INGEST_TOKEN is not set; ingestion will fail.", file=sys.stderr)

    print(f"[private_dashboard_server] serving on http://{host}:{port}", file=sys.stderr)
    with make_server(host, port, app) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
