#!/usr/bin/env python3
"""Step-1 oracle sanity: score reference_answer through evidence + Vitest + reward.

Picks N sandbox tasks from the task bank, converts to coding contract, runs full
execution evidence against ECONOMIST_RL_SOURCE_REPO, and prints per-task scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _pick_sandbox_tasks(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    from economist_rl_task_execution import (
        EXECUTION_MODE_INTEGRATION,
        INTEGRATION_KIND_SANDBOX,
        classify_execution_mode,
        classify_integration_kind,
    )

    seen_sub: set[str] = set()
    out: list[dict[str, Any]] = []
    for task in tasks:
        if classify_execution_mode(task) != EXECUTION_MODE_INTEGRATION:
            continue
        if classify_integration_kind(task, source_repo=None) != INTEGRATION_KIND_SANDBOX:
            continue
        sub = str(task.get("subsection") or "")
        if sub in seen_sub:
            continue
        seen_sub.add(sub)
        out.append(task)
        if len(out) >= limit:
            break
    return out


def _mechanic_only_reference(task: dict[str, Any]) -> str:
    """Reference output for apply: mechanic fence only (matches rollout policy)."""
    from economist_rl_coding_contract import format_reference_answer_coding
    from economist_rl_task_execution import sandbox_paths

    full = format_reference_answer_coding(task)
    lib, _test = sandbox_paths(task)
    marker = f"```ts path={lib}"
    if marker not in full:
        return full.strip()
    start = full.index(marker)
    rest = full[start:]
    end = rest.find("\n```", len(marker))
    if end < 0:
        return full.strip()
    return rest[: end + len("\n```")].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-db",
        type=Path,
        default=REPO / "benchmarks" / "economistRL_tasks_v3_execution.json",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execution-source-repo", type=Path, default=None)
    parser.add_argument("--cycle-id", type=int, default=9001)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable report only.")
    args = parser.parse_args()

    from economist_rl_coding_contract import convert_task_to_coding
    from economist_rl_evidence_runner import EvidenceRunnerConfig, attach_rollout_evidence
    from economist_rl_execution_evidence import (
        ExecutionWorktreePool,
        attach_execution_evidence,
        resolve_execution_config,
        resolve_source_repo,
    )
    from economist_rl_reward_engine import _compile_status, score_output

    payload = json.loads(args.task_db.expanduser().resolve().read_text(encoding="utf-8"))
    tasks = [t for t in payload.get("tasks") or [] if isinstance(t, dict)]
    picked = _pick_sandbox_tasks(tasks, max(1, int(args.limit)))

    source = resolve_source_repo(cli_path=args.execution_source_repo)
    if source is None:
        print("Set --execution-source-repo or ECONOMIST_RL_SOURCE_REPO", file=sys.stderr)
        return 2
    vitest_bin = source / "node_modules" / ".bin" / "vitest"
    if not vitest_bin.is_file():
        print(f"vitest missing: {vitest_bin}", file=sys.stderr)
        return 2

    config = resolve_execution_config(
        enabled=True,
        source_repo=source,
        worktree_root=REPO / "benchmarks" / "results" / "economistRL" / "worktrees",
        compile_commands=[],
        timeout_s=600,
    )
    pool = ExecutionWorktreePool(config, cycle_id=int(args.cycle_id))
    log_root = REPO / "benchmarks" / "results" / "economistRL" / "execution_logs" / "oracle_sanity"

    rows: list[dict[str, Any]] = []
    try:
        for raw in picked:
            task = convert_task_to_coding(raw, source_repo=source)
            output = _mechanic_only_reference(task)
            rollout = attach_rollout_evidence(
                task=task,
                rollout_row={"task_id": task["id"], "output": output, "prompt": task.get("prompt")},
                config=EvidenceRunnerConfig(),
                dry_run=False,
            )
            rollout = attach_execution_evidence(
                task=task,
                rollout_row=rollout,
                config=config,
                worktree=pool.worktree_path,
                log_root=log_root / str(task["id"]),
            )
            score = score_output(
                task,
                output,
                payload,
                rollout_row=rollout,
                compiled=_compile_status(rollout),
                rolling_compile_rate=0.5,
            )
            tt = rollout.get("targeted_tests") or {}
            vitest = rollout.get("vitest_report") or {}
            rows.append(
                {
                    "task_id": task["id"],
                    "subsection": task.get("subsection"),
                    "compiled": rollout.get("compiled"),
                    "vitest_ran": vitest.get("vitest_ran"),
                    "targeted_tests_source": tt.get("source"),
                    "targeted_tests_score": tt.get("score"),
                    "targeted_tests_component": (score.get("components") or {}).get("targeted_tests"),
                    "reward": score.get("reward"),
                    "per_it_passed": sum(1 for r in (vitest.get("per_it") or []) if r.get("passed")),
                    "per_it_total": len(vitest.get("per_it") or []),
                    "goals": len((task.get("simulation_spec") or {}).get("goals") or []),
                    "apply_status": rollout.get("apply_status"),
                    "changed_files": rollout.get("changed_files"),
                }
            )
    finally:
        pool.cleanup()

    report = {
        "tasks": len(rows),
        "source_repo": str(source),
        "task_db": str(args.task_db),
        "rows": rows,
        "pass_step1": all(
            (r.get("compiled") is True)
            and (r.get("targeted_tests_component") or 0) >= 50
            and (r.get("vitest_ran") is True)
            for r in rows
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Oracle sanity ({len(rows)} sandbox tasks) — source: {source}")
        for r in rows:
            ok = (
                r.get("compiled") is True
                and (r.get("targeted_tests_component") or 0) >= 50
                and r.get("vitest_ran")
            )
            flag = "PASS" if ok else "FAIL"
            print(
                f"  [{flag}] {r['task_id']}: reward={r.get('reward'):.3f} "
                f"targeted_tests={r.get('targeted_tests_component')}% "
                f"vitest {r.get('per_it_passed')}/{r.get('per_it_total')} "
                f"compiled={r.get('compiled')} apply={r.get('apply_status')}"
            )
        print(f"Step-1 gate (all refs compiled + targeted_tests>=50): {'PASS' if report['pass_step1'] else 'FAIL'}")
    return 0 if report["pass_step1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
