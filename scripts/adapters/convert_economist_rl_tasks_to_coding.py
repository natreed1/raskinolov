#!/usr/bin/env python3
"""Convert economistRL task bank v1 plan stubs to arena-style coding tasks.

Uses LoRA eval / game-arena apply contracts (fenced TS or unified diff) for prompts
and generates fenced ``reference_answer`` rows for seed bootstrap SFT.

Example:
  python scripts/adapters/convert_economist_rl_tasks_to_coding.py \\
    --in benchmarks/economistRL_tasks_v1.json \\
    --out benchmarks/economistRL_tasks_v2_coding.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_coding_contract import convert_task_to_coding  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object at root: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"Missing tasks[] in {path}")
    return payload


def convert_bank(payload: dict[str, Any]) -> dict[str, Any]:
    tasks = [convert_task_to_coding(task) for task in payload["tasks"] if isinstance(task, dict)]
    out = dict(payload)
    out["schema_version"] = "economist_rl_task_bank_v2_coding"
    out["converted_from"] = payload.get("schema_version") or "economist_rl_task_bank_v1"
    out["converted_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["coding_contract"] = {
        "apply_contract": "game_arena_apply_v1",
        "output_format": "arena_fenced_files_or_unified_diff",
        "prompt_style": "game_task_arena_model_packet",
        "reference_style": "fenced_typescript_stubs",
    }
    out["tasks"] = tasks
    return out


def _summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    with_fences = sum(1 for t in tasks if "```" in str(t.get("reference_answer") or ""))
    with_allowed = sum(1 for t in tasks if t.get("allowed_paths"))
    return {
        "task_count": len(tasks),
        "reference_with_code_fences": with_fences,
        "tasks_with_allowed_paths": with_allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert economistRL tasks to coding/arena format.")
    parser.add_argument("--in", dest="in_path", type=Path, default=REPO / "benchmarks/economistRL_tasks_v1.json")
    parser.add_argument("--out", dest="out_path", type=Path, default=REPO / "benchmarks/economistRL_tasks_v2_coding.json")
    parser.add_argument("--backup", action="store_true", help="Copy input to <out>.v1_backup.json before writing.")
    args = parser.parse_args()

    in_path = args.in_path.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()
    payload = _load(in_path)
    converted = convert_bank(payload)
    if args.backup and out_path.is_file():
        backup = out_path.with_suffix(out_path.suffix + ".v1_backup.json")
        shutil.copy2(out_path, backup)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), **_summary(converted["tasks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
