#!/usr/bin/env python3
"""Build a seed SFT dataset for the economistRL adapter experiment.

The RL lane should ultimately train from scored rollouts, but a small SFT seed
keeps the adapter grounded in the economy task format before RL optimization.
Rows come from `benchmarks/economistRL_tasks_v1.json` reference answers.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = REPO / "benchmarks" / "economistRL_tasks_v1.json"
DEFAULT_OUT_DIR = REPO / "data" / "lora" / "adapters" / "economistRL_seed"

SYSTEM_PROMPT = (
    "You are economistRL, an experimental Fallen Empire economy systems specialist. "
    "Optimize for explicit resource accounting, feedback loops, bounded formulas, "
    "and testable invariants. Avoid shallow tooltip-only answers."
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_tasks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"adapter_id": "economistRL", "tasks": payload}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"Task bank missing `tasks`: {path}")
    return payload, [task for task in tasks if isinstance(task, dict)]


def _fallback_answer(task: dict[str, Any]) -> str:
    expect = task.get("expect") if isinstance(task.get("expect"), dict) else {}
    required = [str(x) for x in expect.get("all_contains") or []]
    mechanics = []
    for item in expect.get("mechanics") or []:
        if isinstance(item, dict) and item.get("name"):
            mechanics.append(str(item["name"]))
    parts = [
        f"Patch target: {task.get('title') or task.get('id')}.",
        "Track explicit inputs, derived deltas, and bounded feedback before mutating economy state.",
    ]
    if required:
        parts.append("Required concepts: " + ", ".join(required) + ".")
    if mechanics:
        parts.append("Mechanics to preserve: " + ", ".join(mechanics) + ".")
    parts.append("Add an invariant or clamp so one tick cannot create runaway growth, debt, or stale projections.")
    return " ".join(parts)


def _row(task: dict[str, Any], variant: str, assistant: str) -> dict[str, Any]:
    task_id = str(task.get("id") or "")
    prompt = str(task.get("prompt") or "")
    if variant == "rubric":
        user = (
            f"{prompt}\n\nReturn a concise implementation plan plus the economy invariant "
            "that should be rewarded by economistRL."
        )
    elif variant == "test":
        user = f"{prompt}\n\nAlso name one regression test that would catch a broken economy update."
    else:
        user = prompt
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task_id": task_id,
        "task_type": "economy_rl",
        "complexity": str(task.get("difficulty") or "standard"),
        "risk_class": "medium",
        "dataset_role": f"economistRL_seed_{variant}",
        "record_id": f"economistRL:{task_id}:{variant}",
        "lineage": "economistRL:v1:seed_sft",
        "policy_version": "economist_rl_policy_v1",
        "rl_focus": list(task.get("rl_focus") or []),
        "subskill": str(task.get("subskill") or ""),
        "curriculum_track": str(task.get("curriculum_track") or "economy"),
    }


def build_dataset(tasks_path: Path, out_dir: Path, seed: int, repeats: int) -> dict[str, Any]:
    payload, tasks = _load_tasks(tasks_path)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        answer = str(task.get("reference_answer") or "").strip() or _fallback_answer(task)
        rows.append(_row(task, "direct", answer))
        rows.append(_row(task, "rubric", answer))
        rows.append(_row(task, "test", answer))

    rng = random.Random(seed)
    train_pool = list(rows)
    while len(train_pool) < max(len(rows), repeats):
        train_pool.append(dict(rng.choice(rows)))
    rng.shuffle(train_pool)

    valid_n = min(12, max(1, len(rows) // 4))
    test_n = min(12, max(1, len(rows) // 4))
    valid = rows[:valid_n]
    test = rows[valid_n : valid_n + test_n] or rows[-valid_n:]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_pool)
    _write_jsonl(out_dir / "valid.jsonl", valid)
    _write_jsonl(out_dir / "test.jsonl", test)
    manifest = {
        "schema_version": "economist_rl_seed_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_id": "economistRL",
        "out_dir": str(out_dir),
        "lineage": "economistRL:v1:seed_sft",
        "policy_version": "economist_rl_policy_v1",
        "inputs": {
            "tasks": str(tasks_path),
            "task_bank_schema": str(payload.get("schema_version") or ""),
        },
        "split_counts": {
            "train": len(train_pool),
            "valid": len(valid),
            "test": len(test),
        },
        "task_count": len(tasks),
        "curriculum_track_counts": {
            "economy": sum(1 for task in tasks if str(task.get("curriculum_track") or "economy") == "economy"),
            "generalist": sum(1 for task in tasks if str(task.get("curriculum_track") or "economy") == "generalist"),
        },
        "rl_focus": sorted({str(tag) for task in tasks for tag in task.get("rl_focus") or []}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build economistRL seed SFT dataset.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=96, help="Minimum train rows after resampling seed tasks.")
    args = parser.parse_args()
    manifest = build_dataset(
        tasks_path=args.tasks.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        seed=int(args.seed),
        repeats=int(args.repeats),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
