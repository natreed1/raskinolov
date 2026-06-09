#!/usr/bin/env python3
"""Tag economistRL tasks with execution modes, starters, context paths, and verify commands.

Reads v2 coding bank (or v1) and writes v3 with per-task ``execution`` metadata.

Example:
  ECONOMIST_RL_SOURCE_REPO=~/fallen-empire \\
    python scripts/adapters/tag_economist_rl_task_execution.py \\
    --in benchmarks/economistRL_tasks_v2_coding.json \\
    --out benchmarks/economistRL_tasks_v3_execution.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_coding_contract import convert_task_to_coding  # noqa: E402
from economist_rl_task_execution import (  # noqa: E402
    EXECUTION_MODE_ARENA,
    EXECUTION_MODE_INTEGRATION,
    INTEGRATION_KIND_SANDBOX,
    INTEGRATION_KIND_STANDARD,
    enrich_task_execution,
    merge_arena_task,
    load_arena_task_specs,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise SystemExit(f"Invalid task bank: {path}")
    return payload


def _resolve_source_repo(cli: Path | None) -> Path | None:
    raw = (os.environ.get("ECONOMIST_RL_SOURCE_REPO") or "").strip()
    if cli is not None:
        path = cli.expanduser().resolve()
        if path.is_dir():
            return path
        raise SystemExit(f"--source-repo is not a directory: {path}")
    if raw:
        path = Path(raw).expanduser().resolve()
        return path if path.is_dir() else None
    return None


def tag_bank(
    payload: dict[str, Any],
    *,
    source_repo: Path | None,
    reconvert_coding: bool,
    refresh_all: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arena_specs = load_arena_task_specs()
    arena_ids = set(arena_specs.keys())
    tasks_out: list[dict[str, Any]] = []
    for raw in payload["tasks"]:
        if not isinstance(raw, dict):
            continue
        if reconvert_coding or refresh_all:
            task = convert_task_to_coding(raw, source_repo=source_repo)
        else:
            task = dict(raw)
            if not task.get("reference_answer") or "```" not in str(task.get("reference_answer") or ""):
                task = convert_task_to_coding(task, source_repo=source_repo)
        task = enrich_task_execution(task, source_repo=source_repo, arena_task_ids=arena_ids)
        if str(task.get("arena_task_id") or "").strip():
            task = merge_arena_task(task, arena_specs)
            task = enrich_task_execution(task, source_repo=source_repo, arena_task_ids=arena_ids)
        starters = (task.get("execution") or {}).get("starter_files") or []
        if starters:
            ex = dict(task.get("execution") or {})
            ex["starter_files"] = starters
            task["execution"] = ex
        tasks_out.append(task)

    out = dict(payload)
    out["schema_version"] = "economist_rl_task_bank_v3_execution"
    out["tagged_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["execution_policy"] = {
        "modes": [EXECUTION_MODE_ARENA, EXECUTION_MODE_INTEGRATION],
        "integration_kinds": [INTEGRATION_KIND_SANDBOX, INTEGRATION_KIND_STANDARD],
        "default_generation_max_tokens": 4000,
        "default_context_max_chars": 12000,
        "bm25_context": True,
    }
    out["tasks"] = tasks_out
    summary = _summary(tasks_out)
    return out, summary


def _summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(t.get("execution_mode") or "?") for t in tasks)
    kinds = Counter(
        str(t.get("integration_kind") or "n/a")
        for t in tasks
        if str(t.get("execution_mode") or "") == EXECUTION_MODE_INTEGRATION
    )
    with_starters = sum(1 for t in tasks if (t.get("execution") or {}).get("starter_files"))
    return {
        "task_count": len(tasks),
        "execution_mode": dict(modes),
        "integration_kind": dict(kinds),
        "tasks_with_starter_files": with_starters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag economistRL tasks with execution metadata (v3).")
    parser.add_argument("--in", dest="in_path", type=Path, default=REPO / "benchmarks/economistRL_tasks_v2_coding.json")
    parser.add_argument("--out", dest="out_path", type=Path, default=REPO / "benchmarks/economistRL_tasks_v3_execution.json")
    parser.add_argument("--source-repo", type=Path, default=None)
    parser.add_argument(
        "--reconvert-coding",
        action="store_true",
        help="Re-run convert_task_to_coding before tagging (use when input is v1/v2).",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Always re-run convert_task_to_coding + enrich (regenerate v3 sandbox prompts/starters).",
    )
    args = parser.parse_args()

    source_repo = _resolve_source_repo(args.source_repo)
    payload = _load(args.in_path.expanduser().resolve())
    tagged, summary = tag_bank(
        payload,
        source_repo=source_repo,
        reconvert_coding=bool(args.reconvert_coding or args.refresh_all),
        refresh_all=bool(args.refresh_all),
    )
    args.out_path.expanduser().resolve().write_text(
        json.dumps(tagged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out_path), "summary": summary, "source_repo": str(source_repo or "")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
