#!/usr/bin/env python3
"""
Run broad, headless Game Task Arena acceptance tests for a local adapter.

This is the no-human-supervision gate: for each task, the local model must
generate/apply an edit, pass the arena's compile/export metrics, and start a
viewable preview route.

`--progressive-context` is forwarded to `run_arena_gate_benchmark.py` /
`game_task_arena.py generate` (`off` = baseline without the extra retrieval pass).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from arena_capability_index import compute_index, write_markdown

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "run_arena_gate_benchmark.py"
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
# Full suite (docs/ARENA_PROGRESSION.md): one task per complexity band so ACI
# tier_breakdown spans Smoke Test, Standard Dev, and High-Reasoning Architecture.
DEFAULT_TASKS = [
    "loading-screen-polish",
    "hud-status-summary",
    "economy-tooltip",
    "combat-risk-preview",
    "save-load-api-guard",
    "ai-planning-explanation",
]


def load_task_ids(tasks_path: Path) -> List[str]:
    data = _read_json(tasks_path)
    tasks = data.get("tasks", data)
    out: List[str] = []
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id") or "").strip()
            if tid:
                out.append(tid)
    return out


def _run(argv: List[str], log_path: Path, timeout_s: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(argv)}\n\n")
        log.flush()
        try:
            p = subprocess.run(
                argv,
                cwd=str(REPO),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            return int(p.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model(explicit_model: Optional[str], adapter_path: str) -> str:
    if explicit_model:
        return explicit_model
    cfg = Path(adapter_path) / "adapter_config.json"
    if cfg.is_file():
        try:
            model = _read_json(cfg).get("model")
            if model:
                return str(model)
        except (OSError, json.JSONDecodeError):
            pass
    return os.environ.get("MODEL", DEFAULT_MODEL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run arena acceptance tests requiring compile/export metrics and a "
            "ready preview page."
        )
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[])
    parser.add_argument(
        "--suite",
        choices=["core6", "all"],
        default="core6",
        help="Task selection when --task-id is omitted: fixed core six or all tasks from --tasks.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO / "benchmarks" / "game_task_arena_examples.json",
        help="Arena task spec JSON.",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))),
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena"))),
    )
    parser.add_argument("--context-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tsc-retries", type=int, default=1)
    parser.add_argument("--model", default=None, help="Base model id. Defaults to adapter_config.json model when available.")
    parser.add_argument("--preview-port", type=int, default=5174)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=14_400,
        help="Overall timeout for run_arena_gate_benchmark (default 4h for full six-task preview).",
    )
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument(
        "--progressive-context",
        choices=["auto", "on", "off"],
        default="auto",
        help="Forward to run_arena_gate_benchmark → game_task_arena generate.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON path. Default: benchmarks/results/arena_acceptance/<batch>/summary.json",
    )
    args = parser.parse_args()

    if args.task_ids:
        task_ids = args.task_ids
        suite_label = "custom"
    elif args.suite == "all":
        task_ids = load_task_ids(args.tasks)
        suite_label = "all"
    else:
        task_ids = DEFAULT_TASKS
        suite_label = "core6"
    if not task_ids:
        raise SystemExit(f"No task ids resolved from --tasks {args.tasks}")
    model = resolve_model(args.model, args.adapter_path)
    batch_id = f"acceptance-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    out_dir = (
        args.summary.parent
        if args.summary
        else REPO / "benchmarks" / "results" / "arena_acceptance" / batch_id
    )
    summary_path = args.summary or out_dir / "summary.json"
    gate_summary = out_dir / "gate_summary.json"
    log_path = out_dir / "arena_gate.log"

    argv = [
        sys.executable,
        str(GATE),
        "--adapter-path",
        args.adapter_path,
        "--source-repo",
        str(args.source_repo.expanduser()),
        "--worktree-root",
        str(args.worktree_root.expanduser()),
        "--tasks",
        str(args.tasks),
        "--gate",
        "preview",
        "--preview-port",
        str(args.preview_port),
        "--context-chars",
        str(args.context_chars),
        "--max-tokens",
        str(args.max_tokens),
        "--tsc-retries",
        str(args.tsc_retries),
        "--model",
        model,
        "--log-dir",
        str(out_dir / "task_logs"),
        "--write-summary",
        str(gate_summary),
    ]
    for task_id in task_ids:
        argv.extend(["--task-id", task_id])
    if args.no_cleanup:
        argv.append("--no-cleanup")
    argv.extend(["--progressive-context", args.progressive_context])

    exit_code = _run(argv, log_path, args.timeout_s)
    gate_data = _read_json(gate_summary) if gate_summary.is_file() else {}
    results = gate_data.get("results", [])
    summary = {
        "adapter_path": args.adapter_path,
        "task_ids": task_ids,
        "suite": suite_label,
        "model": model,
        "arena_context": {
            "progressive_context": args.progressive_context,
            "note": "BM25 pack ordering is still on unless game_task_arena is invoked with --no-context-bm25.",
        },
        "gate": "preview",
        "exit_code": exit_code,
        "passed_all": bool(gate_data.get("passed_all")) and exit_code == 0,
        "passed_count": gate_data.get("passed_count", 0),
        "tasks": gate_data.get("tasks", len(task_ids)),
        "results": results,
        "log": str(log_path),
        "gate_summary": str(gate_summary),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    arena_capability = compute_index(summary_path, args.tasks)
    summary["arena_capability"] = arena_capability
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    capability_json = summary_path.with_name("arena_capability.json")
    capability_md = summary_path.with_name("arena_capability.md")
    capability_json.write_text(json.dumps(arena_capability, indent=2) + "\n", encoding="utf-8")
    write_markdown(arena_capability, capability_md)
    print(json.dumps(summary, indent=2))
    print(
        "=== Arena Capability Index: "
        f"{arena_capability['arena_capability_index']:.1f}/100 "
        f"(accepted {arena_capability['accepted']}/{arena_capability['tasks']}, "
        f"completion {arena_capability['completion']:.1f}, "
        f"integration {arena_capability['integration']:.1f}, "
        f"efficiency {arena_capability['efficiency']:.1f}) ==="
    )
    sys.exit(0 if summary["passed_all"] else 1)


if __name__ == "__main__":
    main()
