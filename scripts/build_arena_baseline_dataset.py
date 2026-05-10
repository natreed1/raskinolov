#!/usr/bin/env python3
"""
Build chat SFT JSONL (messages format) from curated arena task baselines.

Each task in ``benchmarks/game_task_arena_examples.json`` can have a matching
``<task_id>.assistant.txt`` under ``data/arena_task_baselines/`` containing the
exact assistant text you want the model to learn (same style as a winning
``model_output.md`` from the arena: fenced files and/or a unified diff).

This mirrors ``build_game_task_pairwise_dataset.py`` but uses human/reference
answers instead of pairwise winner/loser records.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "game_task_arena_examples.json"
DEFAULT_SOURCES = REPO / "data" / "arena_task_baselines"
DEFAULT_OUT = REPO / "data" / "lora" / "arena_task_baselines"
APPLY_CONTRACT_DOC = REPO / "docs" / "GAME_ARENA_APPLY_CONTRACT.md"
STANDARD_DEV_GUIDE_DOC = REPO / "docs" / "GAME_ARENA_STANDARD_DEV_PATCH_GUIDE.md"
STANDARD_DEV_TASK_IDS = frozenset({"hud-status-summary", "economy-tooltip"})


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("tasks") or [])


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _clean_output(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text[:max_chars].rstrip()


def _load_apply_contract(max_chars: int = 2600) -> str:
    if not APPLY_CONTRACT_DOC.is_file():
        return ""
    body = APPLY_CONTRACT_DOC.read_text(encoding="utf-8").strip()
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "\n\n[contract truncated]\n"


def _load_standard_dev_guide(max_chars: int = 2200) -> str:
    if not STANDARD_DEV_GUIDE_DOC.is_file():
        return ""
    body = STANDARD_DEV_GUIDE_DOC.read_text(encoding="utf-8").strip()
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "\n\n[guide truncated]\n"


def _user_message(task: Dict[str, Any], contract_text: str, standard_dev_guide_text: str) -> str:
    prompt = task.get("prompt") or ""
    preview_path = task.get("preview_path") or ""
    allowed = "\n".join(f"- `{p}`" for p in task.get("allowed_paths", []))
    contract_block = ""
    if contract_text:
        contract_block = f"\nArena apply contract:\n```markdown\n{contract_text}\n```\n"
    standard_dev_block = ""
    task_id = str(task.get("id") or "")
    if task_id in STANDARD_DEV_TASK_IDS and standard_dev_guide_text:
        standard_dev_block = (
            "\nStandard-dev patch guide:\n"
            f"```markdown\n{standard_dev_guide_text}\n```\n"
        )
    return f"""You are editing a disposable Fallen Empire worktree.

Task: {task.get("title", task.get("id", "game task"))}
Preview route: `{preview_path}`

User request:
{prompt}

Allowed paths:
{allowed}

Return only one applyable output format. For small UI edits, prefer fenced full-file blocks with repo-relative paths. Preserve existing exports and TypeScript schemas from the provided code. Avoid malformed diffs, declaration stubs, summaries, and filler text.
{contract_block}
{standard_dev_block}

Reference baseline: produce the same kind of applyable artifact as a passing arena attempt (no meta commentary)."""


def _messages_for_task(
    task: Dict[str, Any],
    assistant_text: str,
    max_output_chars: int,
    contract_text: str,
    standard_dev_guide_text: str,
) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a careful TypeScript game engineer working in the Fallen Empire codebase.",
            },
            {"role": "user", "content": _user_message(task, contract_text, standard_dev_guide_text)},
            {"role": "assistant", "content": _clean_output(assistant_text, max_output_chars)},
        ],
        "metadata": {
            "source": "arena_task_baseline",
            "task_id": task.get("id"),
        },
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    if len(rows) < 3:
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
    p = argparse.ArgumentParser(description="Build LoRA chat dataset from arena baseline assistant files")
    p.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-output-chars", type=int, default=32000)
    p.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="Repeat each baseline row to keep tiny corpora usable for short mlx_lm.lora passes.",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Exit with error if any task is missing a baseline file.",
    )
    p.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        default=[],
        help="Optional task id filter (repeatable). When omitted, include all tasks with baseline files.",
    )
    args = p.parse_args()

    tasks = _load_tasks(args.tasks_json)
    contract_text = _load_apply_contract()
    standard_dev_guide_text = _load_standard_dev_guide()
    selected_ids = {tid.strip() for tid in args.task_ids if tid.strip()}
    if selected_ids:
        tasks = [t for t in tasks if str(t.get("id") or "") in selected_ids]
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for task in tasks:
        tid = task.get("id")
        if not tid:
            continue
        src = args.sources_dir / f"{tid}.assistant.txt"
        if not src.is_file():
            missing.append(str(tid))
            continue
        assistant = _read_text(src).strip()
        if not assistant:
            missing.append(f"{tid} (empty)")
            continue
        rows.append(
            _messages_for_task(
                task,
                assistant,
                args.max_output_chars,
                contract_text,
                standard_dev_guide_text,
            )
        )

    baseline_n = len(rows)

    if args.require_all and (missing or not rows):
        raise SystemExit(
            "Missing or empty baselines for: "
            + ", ".join(missing)
            + (f"; also no usable rows ({missing})" if not rows else "")
        )

    if not rows:
        raise SystemExit(
            f"No baseline files found under {args.sources_dir}. "
            f"Add `<task_id>.assistant.txt` files. See {DEFAULT_SOURCES / 'README.md'}."
        )

    rows = rows * max(1, args.repeat)
    train, valid, test = _split_rows(rows, args.seed)
    out = args.out_dir
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)
    _write_jsonl(out / "test.jsonl", test)
    manifest = {
        "tasks_json": str(args.tasks_json),
        "task_filter_ids": sorted(selected_ids),
        "sources_dir": str(args.sources_dir),
        "out_dir": str(out),
        "task_count_in_specs": len(tasks),
        "baselines_found": baseline_n,
        "missing_task_ids": missing,
        "expanded_records": len(rows),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "apply_contract_doc": str(APPLY_CONTRACT_DOC),
        "standard_dev_guide_doc": str(STANDARD_DEV_GUIDE_DOC),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if missing and not args.require_all:
        print(f"NOTE: Missing baselines (skipped): {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()