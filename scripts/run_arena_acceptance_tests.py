#!/usr/bin/env python3
"""Deterministic arena acceptance runner with Arena Capability Index output."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from game_task_arena import (
    apply_output,
    attempt_applied,
    cleanup,
    create_trial,
    generate_attempt,
    load_trial,
    load_task_specs,
    preview,
    verify,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _core6_default_ids(tasks: Dict[str, Any]) -> List[str]:
    preferred = [
        "loading-screen-polish",
        "hud-status-summary",
        "economy-tooltip",
        "combat-risk-preview",
        "save-load-api-guard",
        "ai-planning-explanation",
    ]
    out = [tid for tid in preferred if tid in tasks]
    if len(out) == 6:
        return out
    # Fallback: first six declared tasks.
    return list(tasks.keys())[:6]


def _safe_ratio(passed: int, total: int) -> float:
    return float(passed) / float(total) if total else 0.0


def _arena_capability_metrics(rows: List[Dict[str, Any]], rounds_allowed: int) -> Dict[str, Any]:
    total = len(rows)
    accepted = sum(1 for row in rows if row.get("accepted"))
    verify_passed = sum(1 for row in rows if row.get("verify_status") == "passed")
    if total:
        avg_round = sum(int(row.get("rounds_used", 1)) for row in rows) / float(total)
    else:
        avg_round = 1.0
    max_extra = max(1, rounds_allowed - 1)
    retry_penalty = max(0.0, (avg_round - 1.0) / float(max_extra))
    completion = round(_safe_ratio(accepted, total) * 100.0, 1)
    integration = round(_safe_ratio(verify_passed, total) * 100.0, 1)
    efficiency = round(max(0.0, (1.0 - retry_penalty)) * 100.0, 1)
    aci = round((0.5 * completion) + (0.3 * integration) + (0.2 * efficiency), 1)
    return {
        "arena_capability_index": aci,
        "accepted": accepted,
        "tasks": total,
        "completion": completion,
        "integration": integration,
        "efficiency": efficiency,
        "average_rounds_used": round(avg_round, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-human arena acceptance and emit ACI summary.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids", default=[])
    parser.add_argument("--suite", choices=["core6", "all"], default="core6")
    parser.add_argument("--context-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tsc-retries", type=int, default=1)
    parser.add_argument("--preview-port", type=int, default=5174)
    parser.add_argument("--timeout-s", type=int, default=14_400)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--progressive-context", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    tasks_json = args.tasks.expanduser().resolve()
    tasks = load_task_specs(tasks_json)
    if args.task_ids:
        selected_ids = list(dict.fromkeys(tid for tid in args.task_ids if tid in tasks))
        suite_label = "custom"
    else:
        if args.suite == "all":
            selected_ids = list(tasks.keys())
            suite_label = "all"
        else:
            selected_ids = _core6_default_ids(tasks)
            suite_label = "core6"
    if not selected_ids:
        raise SystemExit("No tasks selected for arena acceptance.")

    rows: List[Dict[str, Any]] = []
    rounds_allowed = max(1, int(args.tsc_retries) + 1)
    base_timeout = max(300, min(1800, int(args.timeout_s)))

    for idx, task_id in enumerate(selected_ids):
        trial_id = f"gatebench-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{task_id}-{uuid.uuid4().hex[:6]}"
        create_ns = argparse.Namespace(
            tasks=tasks_json,
            task_id=task_id,
            source_repo=str(args.source_repo.expanduser().resolve()),
            worktree_root=str(args.worktree_root.expanduser().resolve()),
            base_ref="HEAD",
            trial_id=trial_id,
            attempts=["local"],
            local_adapter=args.adapter_path,
            frontier_model="unused-frontier",
            copy=False,
        )
        create_trial(create_ns)

        row: Dict[str, Any] = {
            "task_id": task_id,
            "trial_id": trial_id,
            "rounds_used": 0,
            "apply_status": "not_run",
            "verify_status": "not_run",
            "preview_status": "not_run",
            "accepted": False,
            "error": "",
        }
        try:
            for round_idx in range(rounds_allowed):
                row["rounds_used"] = round_idx + 1
                gen_ns = argparse.Namespace(
                    trial_id=trial_id,
                    attempt="local",
                    backend="local",
                    adapter_path=args.adapter_path,
                    local_model=args.model,
                    model=args.model or "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                    max_tokens=int(args.max_tokens),
                    temp=0.0,
                    context_chars=int(args.context_chars),
                    bug_check_loop=True,
                    bug_check_rounds=1,
                    bug_check_max_tokens=min(2048, int(args.max_tokens)),
                    bug_check_system_prompt=(
                        "You are a strict bug-fix reviewer for code patches. "
                        "Return only a corrected final patch output."
                    ),
                    bug_check_rag=True,
                    bug_check_rag_corpus="data/rag/bug_fix_agent_corpus.json",
                    bug_check_rag_top_k=6,
                    bug_check_rag_max_chars=2200,
                    no_context_bm25=False,
                )
                output_path = generate_attempt(gen_ns)
                apply_ns = argparse.Namespace(trial_id=trial_id, attempt="local", input=str(output_path))
                applied_attempt = apply_output(apply_ns)
                row["apply_status"] = applied_attempt.apply_status
                verify_ns = argparse.Namespace(trial_id=trial_id, attempt="local", timeout=base_timeout)
                verified_attempt = verify(verify_ns)
                row["verify_status"] = verified_attempt.verify_status
                if attempt_applied(verified_attempt) and verified_attempt.verify_status == "passed":
                    break

            preview_ns = argparse.Namespace(
                trial_id=trial_id,
                attempt="local",
                port=int(args.preview_port) + idx,
                command=None,
                preview_command_override=None,
                start=True,
            )
            preview_attempt = preview(preview_ns)
            row["preview_status"] = preview_attempt.preview_status
            row["preview_url"] = preview_attempt.preview_url
            row["accepted"] = bool(
                attempt_applied(preview_attempt)
                and preview_attempt.verify_status == "passed"
                and preview_attempt.preview_status == "ready"
            )
            print(
                f"[{'PASS' if row['accepted'] else 'FAIL'}] {task_id} "
                f"apply={row['apply_status']} verify={row['verify_status']} "
                f"preview={row['preview_status']} rounds={row['rounds_used']}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {task_id} error={row['error']}")
        finally:
            if not args.no_cleanup:
                try:
                    cleanup(argparse.Namespace(trial_id=trial_id, attempt="local", copy=False, keep_branch=False))
                except Exception:
                    pass
        rows.append(row)

    cap = _arena_capability_metrics(rows, rounds_allowed=rounds_allowed)
    summary = {
        "schema_version": "arena_acceptance_summary_v2",
        "created_utc": _utc_now(),
        "suite": suite_label,
        "tasks": len(rows),
        "task_ids": selected_ids,
        "adapter_path": args.adapter_path,
        "source_repo": str(args.source_repo.expanduser().resolve()),
        "worktree_root": str(args.worktree_root.expanduser().resolve()),
        "max_tokens": int(args.max_tokens),
        "context_chars": int(args.context_chars),
        "tsc_retries": int(args.tsc_retries),
        "preview_port_base": int(args.preview_port),
        "progressive_context": args.progressive_context,
        "passed_count": cap["accepted"],
        "arena_capability": cap,
        "rows": rows,
    }
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "=== Arena Capability Index: "
        f"{cap['arena_capability_index']:.1f}/100 "
        f"(accepted {cap['accepted']}/{cap['tasks']}, completion {cap['completion']:.1f}, "
        f"integration {cap['integration']:.1f}, efficiency {cap['efficiency']:.1f}) ==="
    )
    if cap["accepted"] < len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
