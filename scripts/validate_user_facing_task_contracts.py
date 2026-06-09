#!/usr/bin/env python3
"""Validate metadata contracts for user-facing task entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO / "benchmarks" / "task_bank" / "organization" / "user_facing_task_requirements_v1.json"
DEFAULT_SOURCE = REPO / "benchmarks" / "task_bank" / "sources" / "swebench_verified_game_mapped_tasks_v1.json"


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return [x for x in payload["tasks"] if isinstance(x, dict)]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise SystemExit(f"Unsupported task file format: {path}")


def _is_user_facing(task: dict[str, Any], task_types: set[str], keywords: set[str]) -> bool:
    if str(task.get("task_type") or "") in task_types:
        return True
    text = " ".join(
        [
            str(task.get("prompt") or ""),
            " ".join(task.get("acceptance_signals") or []),
            str(task.get("subskill") or ""),
        ]
    ).lower()
    return any(f" {k} " in f" {text} " for k in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required user-facing task fields.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source", type=Path, action="append", default=[])
    args = parser.parse_args()

    policy = json.loads(args.policy.expanduser().resolve().read_text(encoding="utf-8"))
    detect = policy.get("user_facing_detection") or {}
    task_types = {str(x) for x in (detect.get("task_types") or [])}
    keywords = {str(x).lower() for x in (detect.get("prompt_keywords") or [])}
    required_fields = [str(x) for x in (policy.get("required_fields") or [])]

    violations: list[dict[str, str]] = []
    checked = 0
    sources = args.source or [DEFAULT_SOURCE]
    source_paths = sorted({src.expanduser().resolve() for src in sources})
    for source_path in source_paths:
        tasks = _load_tasks(source_path)
        for task in tasks:
            if not _is_user_facing(task, task_types, keywords):
                continue
            checked += 1
            for field in required_fields:
                value = task.get(field)
                missing = value is None or value == [] or value == {}
                if missing:
                    violations.append(
                        {
                            "source": str(source_path),
                            "task_id": str(task.get("id") or ""),
                            "field": field,
                        }
                    )

    print(json.dumps({"checked_user_facing_tasks": checked, "violations": violations}, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

