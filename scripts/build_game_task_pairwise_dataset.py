#!/usr/bin/env python3
"""
Build a chat SFT dataset from Game Task Arena pairwise apply-preference records.

Input records come from `benchmarks/results/game_task_pairwise_training_data.jsonl`.
Each output row teaches the local model to produce the winning/applyable answer for
the same task context, with explicit reminders about output format and schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INPUT = Path("benchmarks/results/game_task_pairwise_training_data.jsonl")
DEFAULT_OUT = Path("data/lora/game_task_pairwise")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _clean_output(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text[:max_chars].rstrip()


def _messages_for_record(rec: Dict[str, Any], max_output_chars: int) -> Dict[str, Any]:
    task = rec.get("task", {})
    prompt = task.get("prompt") or ""
    preview_path = task.get("preview_path") or ""
    allowed = "\n".join(f"- `{p}`" for p in task.get("allowed_paths", []))
    winner = rec.get("winner", "winner")
    loser_error = rec.get("loser_error") or ""
    user = f"""You are editing a disposable Fallen Empire worktree.

Task: {task.get("title", task.get("id", "game task"))}
Preview route: `{preview_path}`

User request:
{prompt}

Allowed paths:
{allowed}

Return only one applyable output format. For small UI edits, prefer fenced full-file blocks with repo-relative paths. Preserve existing exports and TypeScript schemas from the provided code. Avoid malformed diffs, declaration stubs, summaries, and filler text.

The rejected attempt failed with:
{loser_error[-1200:]}
"""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a careful TypeScript game engineer working in the Fallen Empire codebase.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": _clean_output(rec.get("winner_output", ""), max_output_chars)},
        ],
        "metadata": {
            "source": "game_task_pairwise",
            "trial_id": rec.get("trial_id"),
            "winner": winner,
            "loser": rec.get("loser"),
            "task_id": task.get("id"),
        },
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows: List[Dict[str, Any]], seed: int) -> tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    if len(rows) < 3:
        # mlx-lm wants train/valid/test files; duplicate tiny corpora for a smoke-sized pass.
        rows = rows * max(3, 3 // max(1, len(rows)))
    n = len(rows)
    valid_n = max(1, min(2, n // 5))
    test_n = max(1, min(2, n // 5))
    train_n = max(1, n - valid_n - test_n)
    train = rows[:train_n]
    valid = rows[train_n : train_n + valid_n] or rows[:1]
    test = rows[train_n + valid_n : train_n + valid_n + test_n] or rows[-1:]
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LoRA chat dataset from Game Task Arena pairwise records")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat records to make tiny preference corpora usable for short training passes.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Pairwise training data not found: {args.input}")
    raw_rows = _read_jsonl(args.input)
    rows = [_messages_for_record(rec, args.max_output_chars) for rec in raw_rows if rec.get("winner_output")]
    if not rows:
        raise SystemExit(f"No usable winner outputs in {args.input}")
    rows = rows * max(1, args.repeat)
    train, valid, test = _split_rows(rows, args.seed)
    out = args.out_dir
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)
    _write_jsonl(out / "test.jsonl", test)
    manifest = {
        "input": str(args.input),
        "out_dir": str(out),
        "raw_records": len(raw_rows),
        "expanded_records": len(rows),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
