#!/usr/bin/env python3
"""Re-run text evidence, execution (Vitest), and reward scoring on existing rollouts.

Use when rollout JSONL is already saved but execution deps or reward weights changed.
Does not regenerate model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-file", type=Path, required=True)
    parser.add_argument("--cycle-id", type=int, required=True)
    parser.add_argument(
        "--task-db",
        type=Path,
        default=REPO / "benchmarks" / "economistRL_tasks_v3_execution.json",
    )
    parser.add_argument("--execution-source-repo", type=Path, default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "benchmarks" / "results" / "economistRL",
    )
    parser.add_argument("--execution-timeout-s", type=int, default=600)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(REPO / "scripts"))

    from economist_rl_evidence_runner import EvidenceRunnerConfig, attach_batch_evidence, write_evidence_jsonl
    from economist_rl_execution_evidence import (
        ExecutionWorktreePool,
        attach_batch_execution_evidence,
        resolve_execution_config,
        resolve_source_repo,
    )
    from economist_rl_reward_engine import _compile_status, score_output

    rollout_file = args.rollout_file.expanduser().resolve()
    if not rollout_file.is_file():
        raise SystemExit(f"rollout file not found: {rollout_file}")

    results_root = args.results_root.expanduser().resolve()
    cycle_id = int(args.cycle_id)
    payload = json.loads(args.task_db.expanduser().resolve().read_text(encoding="utf-8"))
    tasks = [t for t in payload.get("tasks") or [] if isinstance(t, dict)]
    task_by_id = {str(t.get("id") or ""): t for t in tasks}

    rollouts = _load_jsonl(rollout_file)
    missing = [str(r.get("task_id")) for r in rollouts if str(r.get("task_id") or "") not in task_by_id]
    if missing:
        raise SystemExit(f"task ids missing from bank: {missing[:5]}")

    evidenced = attach_batch_evidence(
        tasks_by_id=task_by_id,
        rollout_rows=rollouts,
        config=EvidenceRunnerConfig(),
        dry_run=False,
    )

    source = resolve_source_repo(cli_path=args.execution_source_repo)
    if source is None:
        raise SystemExit("Set --execution-source-repo or ECONOMIST_RL_SOURCE_REPO")
    vitest_bin = source / "node_modules" / ".bin" / "vitest"
    if not vitest_bin.is_file():
        raise SystemExit(f"vitest missing in source repo: {vitest_bin}")

    config = resolve_execution_config(
        enabled=True,
        source_repo=source,
        worktree_root=results_root / "worktrees",
        compile_commands=[],
        timeout_s=int(args.execution_timeout_s),
    )

    pool = ExecutionWorktreePool(config, cycle_id=cycle_id)
    log_root = results_root / "execution_logs" / f"cycle_{cycle_id:03d}"
    try:
        evidenced = attach_batch_execution_evidence(
            tasks_by_id=task_by_id,
            rollout_rows=evidenced,
            config=config,
            pool=pool,
            log_root=log_root,
        )
    finally:
        pool.cleanup()

    evidence_file = results_root / "evidence" / f"evidence_batch_{cycle_id:03d}.jsonl"
    write_evidence_jsonl(evidence_file, evidenced)

    scored_rows: list[dict[str, Any]] = []
    for row in evidenced:
        task_id = str(row.get("task_id") or "")
        task = task_by_id[task_id]
        score = score_output(
            task,
            str(row.get("output") or ""),
            payload,
            rollout_row=row,
            compiled=_compile_status(row),
            rolling_compile_rate=0.0,
        )
        scored_rows.append({"task_id": task_id, "rollout": row, "score": score})

    compile_flags = [_compile_status(item["rollout"]) for item in scored_rows]
    compile_known = [f for f in compile_flags if f is not None]
    compile_rate = sum(1 for f in compile_known if f) / max(1, len(compile_known))

    for item in scored_rows:
        row = item["rollout"]
        task = task_by_id[str(row.get("task_id") or "")]
        item["score"] = score_output(
            task,
            str(row.get("output") or ""),
            payload,
            rollout_row=row,
            compiled=_compile_status(row),
            rolling_compile_rate=compile_rate,
        )

    scored_file = results_root / "scores" / f"scored_batch_{cycle_id:03d}.jsonl"
    _write_jsonl(scored_file, scored_rows)

    rewards = [float(item["score"].get("reward") or 0.0) for item in scored_rows]
    vitest_ran = sum(
        1
        for item in scored_rows
        if (item["rollout"].get("vitest_report") or {}).get("vitest_ran")
    )
    goal_weighted = sum(
        1
        for item in scored_rows
        if (item["rollout"].get("targeted_tests") or {}).get("source") == "vitest_goal_weighted"
    )
    compiled_true = sum(1 for f in compile_known if f)

    manifest_path = results_root / "manifests" / f"cycle_{cycle_id:03d}_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["mean_rollout_reward"] = round(sum(rewards) / max(1, len(rewards)), 4)
        manifest["rescored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["rescore_note"] = "rescore_economist_rl_rollouts.py (execution + vitest + score)"
        manifest["rescore_vitest_ran"] = vitest_ran
        manifest["rescore_compiled_true"] = compiled_true
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"rescored {len(scored_rows)} rollouts -> {scored_file}")
    print(f"mean_reward={sum(rewards)/len(rewards):.4f} min={min(rewards):.4f} max={max(rewards):.4f}")
    print(f"compiled_true={compiled_true}/{len(compile_known)} vitest_ran={vitest_ran} goal_weighted={goal_weighted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
