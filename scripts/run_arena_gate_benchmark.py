#!/usr/bin/env python3
"""
Headless Game Task Arena gate: create → local generate (with tsc/export retries) → optional preview.

Exit 0 iff every requested task passes the selected gate:
  metrics — generation_metrics.json has round_final_ok true for the local attempt
  preview — additionally run `preview --start` and require preview_status == ready

Designed for CI / ml_workflow loops (not the Gradio UI).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))).expanduser()
DEFAULT_ARENA_ROOT = Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena"))).expanduser()
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
GA = REPO / "scripts" / "game_task_arena.py"
RESULTS = REPO / "benchmarks" / "results" / "game_task_trials"


def _run(
    argv: List[str],
    log_path: Path,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    timeout_s: int = 1200,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**os.environ, **(env or {})}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(argv)}\n\n")
        log.flush()
        try:
            p = subprocess.run(
                argv,
                cwd=str(cwd),
                env=merged,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            return int(p.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def _slug(s: str, max_len: int = 40) -> str:
    import re as _re

    v = _re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower()).strip("-")
    return (v or "task")[:max_len]


def _trial_id(task_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"gatebench-{ts}-{_slug(task_id, 24)}-{uuid.uuid4().hex[:6]}"


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


def metrics_gate(trial_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    mpath = RESULTS / trial_id / "attempts" / "local" / "generation_metrics.json"
    if not mpath.is_file():
        return False, f"missing_metrics:{mpath}", {}
    data = _read_json(mpath)
    ok = bool(data.get("round_final_ok"))
    return ok, "round_final_ok" if ok else "round_final_ok_false", data


def preview_gate(trial_id: str, port: int, log_root: Path, tasks_file: Path) -> Tuple[bool, str]:
    argv = [
        sys.executable,
        str(GA),
        "--tasks",
        str(tasks_file),
        "preview",
        "--trial-id",
        trial_id,
        "--attempt",
        "local",
        "--port",
        str(port),
        "--start",
    ]
    code = _run(argv, log_root / "preview.log", REPO, timeout_s=600)
    if code != 0:
        return False, f"preview_exit_{code}"
    pj = RESULTS / trial_id / "attempts" / "local" / "preview.json"
    if not pj.is_file():
        return False, "missing_preview_json"
    meta = _read_json(pj)
    st = str(meta.get("status", ""))
    if st == "ready":
        return True, "preview_ready"
    return False, f"preview_status:{st}"


def run_one_task(
    task_id: str,
    adapter_path: str,
    source_repo: Path,
    worktree_root: Path,
    tasks_file: Path,
    gate: str,
    preview_port: int,
    context_chars: int,
    max_tokens: int,
    tsc_retries: int,
    model: str,
    log_root: Path,
    cleanup: bool,
    *,
    progressive_context: str = "auto",
) -> Dict[str, Any]:
    tid = _trial_id(task_id)
    log_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODEL": model}

    create_argv = [
        sys.executable,
        str(GA),
        "--tasks",
        str(tasks_file),
        "create",
        "--task-id",
        task_id,
        "--source-repo",
        str(source_repo),
        "--worktree-root",
        str(worktree_root),
        "--attempts",
        "local",
        "--local-adapter",
        adapter_path,
        "--trial-id",
        tid,
    ]
    rc = _run(create_argv, log_root / "create.log", REPO, env=env, timeout_s=600)
    out: Dict[str, Any] = {"trial_id": tid, "task_id": task_id, "create_exit": rc}
    if rc != 0:
        out["passed"] = False
        out["reason"] = "create_failed"
        return out

    gen_argv = [
        sys.executable,
        str(GA),
        "--tasks",
        str(tasks_file),
        "generate",
        "--trial-id",
        tid,
        "--attempt",
        "local",
        "--backend",
        "local",
        "--adapter-path",
        adapter_path,
        "--context-chars",
        str(context_chars),
        "--max-tokens",
        str(max_tokens),
        "--tsc-retries",
        str(tsc_retries),
        "--progressive-context",
        progressive_context,
    ]
    rc = _run(gen_argv, log_root / "generate.log", REPO, env=env, timeout_s=1200)
    out["generate_exit"] = rc
    if rc != 0:
        out["passed"] = False
        out["reason"] = "generate_failed"
        if cleanup:
            _run(
                [
                    sys.executable,
                    str(GA),
                    "--tasks",
                    str(tasks_file),
                    "cleanup",
                    "--trial-id",
                    tid,
                    "--attempt",
                    "local",
                ],
                log_root / "cleanup.log",
                REPO,
                env=env,
                timeout_s=300,
            )
        return out

    ok_m, reason_m, metrics = metrics_gate(tid)
    out["metrics_ok"] = ok_m
    out["metrics_reason"] = reason_m
    out["generation_metrics"] = {
        "round_final_ok": metrics.get("round_final_ok"),
        "apply_final_ok": metrics.get("apply_final_ok"),
        "tsc_final_ok": metrics.get("tsc_final_ok"),
        "exports_final_ok": metrics.get("exports_final_ok"),
        "tsc_rounds": metrics.get("tsc_rounds"),
    }
    passed = ok_m
    out["preview_ok"] = None
    out["preview_reason"] = None
    if gate == "preview" and passed:
        ok_p, reason_p = preview_gate(tid, preview_port, log_root, tasks_file)
        out["preview_ok"] = ok_p
        out["preview_reason"] = reason_p
        passed = ok_p

    out["passed"] = passed
    out["reason"] = "ok" if passed else reason_m if gate == "metrics" else out.get("preview_reason", reason_m)

    if cleanup:
        _run(
            [
                sys.executable,
                str(GA),
                "--tasks",
                str(tasks_file),
                "cleanup",
                "--trial-id",
                tid,
                "--attempt",
                "local",
            ],
            log_root / "cleanup.log",
            REPO,
            env=env,
            timeout_s=300,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Headless arena gate benchmark (local lane)")
    p.add_argument("--task-id", action="append", dest="task_ids", default=[], help="Task id (repeatable)")
    p.add_argument(
        "--tasks",
        type=Path,
        default=REPO / "benchmarks" / "game_task_arena_examples.json",
        help="Task specs JSON",
    )
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--worktree-root", type=Path, default=DEFAULT_ARENA_ROOT)
    p.add_argument("--gate", choices=["metrics", "preview"], default="metrics")
    p.add_argument("--preview-port", type=int, default=5174)
    p.add_argument("--context-chars", type=int, default=9000)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--tsc-retries", type=int, default=1)
    p.add_argument("--model", default=None, help="Base model id. Defaults to adapter_config.json model when available.")
    p.add_argument("--log-dir", type=Path, default=None, help="Per-invocation logs (default: stderr-only summary)")
    p.add_argument("--no-cleanup", action="store_true", help="Keep disposable worktrees for inspection")
    p.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        help="Write the JSON summary to this path (for orchestrators that log stdout to a file).",
    )
    p.add_argument(
        "--progressive-context",
        choices=["auto", "on", "off"],
        default="auto",
        help="Forward to game_task_arena.py generate (extra retrieval pass before editing).",
    )
    args = p.parse_args()
    task_ids: List[str] = list(args.task_ids) or ["loading-screen-polish"]
    model = resolve_model(args.model, args.adapter_path)

    batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log_parent = args.log_dir or (REPO / "benchmarks" / "results" / "arena_gate_runs" / batch_id)

    results = []
    all_pass = True
    for i, tid in enumerate(task_ids):
        log_root = log_parent / f"task_{i}_{_slug(tid, 30)}"
        row = run_one_task(
            task_id=tid,
            adapter_path=args.adapter_path,
            source_repo=args.source_repo.expanduser().resolve(),
            worktree_root=args.worktree_root.expanduser().resolve(),
            tasks_file=args.tasks.resolve(),
            gate=args.gate,
            preview_port=args.preview_port,
            context_chars=args.context_chars,
            max_tokens=args.max_tokens,
            tsc_retries=args.tsc_retries,
            model=model,
            log_root=log_root,
            cleanup=not args.no_cleanup,
            progressive_context=args.progressive_context,
        )
        results.append(row)
        if not row.get("passed"):
            all_pass = False

    summary = {
        "passed_all": all_pass,
        "tasks": len(task_ids),
        "model": model,
        "passed_count": sum(1 for r in results if r.get("passed")),
        "results": results,
        "progressive_context": args.progressive_context,
    }
    print(json.dumps(summary, indent=2))
    if args.write_summary:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
