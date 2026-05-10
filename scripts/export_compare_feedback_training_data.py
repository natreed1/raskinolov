#!/usr/bin/env python3
"""
Export structured compare-view feedback into training-ready JSONL files.

Outputs:
- pairwise_feedback.jsonl: winner/loser records with hybrid weighting
- rewrite_feedback.jsonl: span-level rewrite supervision rows for bad spans
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "private_dashboard.sqlite3"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "lora" / "compare_feedback"

STRENGTH_WEIGHT = {
    "weak": 1.0,
    "medium": 1.5,
    "strong": 2.0,
    "tie": 0.5,
}


def _read_repo_text(rel_path: str) -> str:
    rel = str(rel_path or "").strip()
    if not rel:
        return ""
    candidate = (REPO_ROOT / rel).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return ""
    if not candidate.is_file():
        return ""
    return candidate.read_text(encoding="utf-8", errors="replace")


def _read_compare_feedback(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id,
            ts,
            left_track,
            right_track,
            left_artifact_path,
            right_artifact_path,
            winner,
            strength,
            notes,
            source
        FROM compare_feedback
        ORDER BY id ASC
        """
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        spans = conn.execute(
            """
            SELECT side, start_offset, end_offset, label, selected_text, reason, rewrite_text
            FROM compare_feedback_spans
            WHERE feedback_id = ?
            ORDER BY id ASC
            """,
            (row[0],),
        ).fetchall()
        out.append(
            {
                "id": int(row[0]),
                "ts": row[1],
                "left_track": row[2],
                "right_track": row[3],
                "left_artifact_path": row[4],
                "right_artifact_path": row[5],
                "winner": row[6],
                "strength": row[7],
                "notes": row[8] or "",
                "source": row[9] or "",
                "spans": [
                    {
                        "side": s[0],
                        "start": int(s[1]),
                        "end": int(s[2]),
                        "label": s[3],
                        "selected_text": s[4] or "",
                        "reason": s[5] or "",
                        "rewrite_text": s[6] or "",
                    }
                    for s in spans
                ],
            }
        )
    return out


def _hybrid_weight(rec: Dict[str, Any], winner_side: str, loser_side: str) -> float:
    strength = str(rec.get("strength") or "weak")
    base = STRENGTH_WEIGHT.get(strength, 1.0)
    winner_bad = sum(1 for s in rec["spans"] if s["side"] == winner_side and s["label"] == "bad")
    loser_bad = sum(1 for s in rec["spans"] if s["side"] == loser_side and s["label"] == "bad")
    loser_penalty = min(1.5, loser_bad * 0.15)
    winner_discount = 0.2 if winner_bad > loser_bad else 0.0
    return round(max(0.25, base * (1.0 + loser_penalty - winner_discount)), 4)


def _build_pairwise_records(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    skipped = 0
    for rec in rows:
        winner = rec.get("winner")
        if winner not in {"left", "right"}:
            skipped += 1
            continue
        winner_side = str(winner)
        loser_side = "right" if winner_side == "left" else "left"
        left_text = _read_repo_text(rec.get("left_artifact_path", ""))
        right_text = _read_repo_text(rec.get("right_artifact_path", ""))
        if not left_text or not right_text:
            skipped += 1
            continue
        winner_output = left_text if winner_side == "left" else right_text
        loser_output = right_text if winner_side == "left" else left_text
        out.append(
            {
                "kind": "compare_feedback_pairwise",
                "feedback_id": rec["id"],
                "ts": rec["ts"],
                "winner": winner_side,
                "loser": loser_side,
                "strength": rec.get("strength") or "weak",
                "weight": _hybrid_weight(rec, winner_side, loser_side),
                "left_track": rec["left_track"],
                "right_track": rec["right_track"],
                "left_artifact_path": rec["left_artifact_path"],
                "right_artifact_path": rec["right_artifact_path"],
                "notes": rec.get("notes") or "",
                "winner_output": winner_output,
                "loser_output": loser_output,
                "spans": rec["spans"],
            }
        )
    return out, skipped


def _build_rewrite_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in rows:
        for span in rec["spans"]:
            if span["label"] != "bad" or not span["rewrite_text"].strip():
                continue
            side = span["side"]
            artifact_path = rec["left_artifact_path"] if side == "left" else rec["right_artifact_path"]
            out.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You rewrite technical documentation snippets to be clear, correct, and concise.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Artifact: {artifact_path}\n"
                                f"Reason: {span['reason'] or 'improve this snippet'}\n\n"
                                f"Original snippet:\n{span['selected_text']}\n\n"
                                "Rewrite the snippet."
                            ),
                        },
                        {"role": "assistant", "content": span["rewrite_text"]},
                    ],
                    "metadata": {
                        "source": "compare_feedback_rewrite",
                        "feedback_id": rec["id"],
                        "artifact_path": artifact_path,
                        "side": side,
                        "span_start": span["start"],
                        "span_end": span["end"],
                    },
                }
            )
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structured compare feedback to training JSONL")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pairwise-file", default="pairwise_feedback.jsonl")
    parser.add_argument("--rewrite-file", default="rewrite_feedback.jsonl")
    args = parser.parse_args()

    if not args.db_path.is_file():
        raise SystemExit(f"DB file not found: {args.db_path}")
    conn = sqlite3.connect(str(args.db_path))
    try:
        feedback_rows = _read_compare_feedback(conn)
    finally:
        conn.close()

    pairwise_rows, skipped_pairwise = _build_pairwise_records(feedback_rows)
    rewrite_rows = _build_rewrite_rows(feedback_rows)
    out_dir = args.out_dir
    pairwise_path = out_dir / args.pairwise_file
    rewrite_path = out_dir / args.rewrite_file
    _write_jsonl(pairwise_path, pairwise_rows)
    _write_jsonl(rewrite_path, rewrite_rows)

    manifest = {
        "db_path": str(args.db_path),
        "out_dir": str(out_dir),
        "pairwise_path": str(pairwise_path),
        "rewrite_path": str(rewrite_path),
        "feedback_rows": len(feedback_rows),
        "pairwise_rows": len(pairwise_rows),
        "rewrite_rows": len(rewrite_rows),
        "skipped_pairwise_rows": skipped_pairwise,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
