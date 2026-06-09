#!/usr/bin/env python3
"""Audit economistRL PPO/reward pipeline semantics on scored rollout rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_evidence_runner import attach_rollout_evidence  # noqa: E402
from economist_rl_reward_engine import _compile_status, _load_json, score_output  # noqa: E402


def _load_scored_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score_from_rollout(payload: dict[str, Any], task: dict[str, Any], rollout: dict[str, Any], compile_rate: float) -> dict[str, Any]:
    return score_output(
        task,
        str(rollout.get("output") or ""),
        payload,
        rollout_row=rollout,
        compiled=_compile_status(rollout),
        rolling_compile_rate=compile_rate,
    )


def _legacy_would_exclude(score: dict[str, Any]) -> bool:
    """Heuristic for old logic that treated diagnostics or scorecard fail as bad PPO rows."""
    if not score.get("strict_scorecard_pass") and not score.get("high_reward"):
        return True
    if score.get("failures"):
        return True
    return False


def audit_rows(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compile_counts: Counter[str] = Counter()
    hard_cap = 0
    training_usable = 0
    strict_fail = 0
    with_diagnostics = 0
    legacy_excluded = 0
    reward_with_diag: list[float] = []
    reward_strict_fail: list[float] = []

    for row in scored_rows:
        score = row.get("score") if isinstance(row.get("score"), dict) else row
        cg = score.get("compile_gate") or {}
        compiled = cg.get("compiled")
        key = "true" if compiled is True else "false" if compiled is False else "none"
        compile_counts[key] += 1
        if score.get("training_usable"):
            training_usable += 1
        if score.get("hard_cap_applied"):
            hard_cap += 1
        if not score.get("strict_scorecard_pass"):
            strict_fail += 1
            reward_strict_fail.append(float(score.get("reward") or 0.0))
        if score.get("diagnostics"):
            with_diagnostics += 1
            reward_with_diag.append(float(score.get("reward") or 0.0))
        if _legacy_would_exclude(score) and score.get("training_usable"):
            legacy_excluded += 1

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    return {
        "rows": len(scored_rows),
        "compiled_counts": dict(compile_counts),
        "training_usable_count": training_usable,
        "hard_cap_applied_count": hard_cap,
        "strict_scorecard_fail_count": strict_fail,
        "rows_with_diagnostics_count": with_diagnostics,
        "legacy_would_exclude_but_training_usable": legacy_excluded,
        "reward_with_diagnostics": _stats(reward_with_diag),
        "reward_strict_scorecard_fail": _stats(reward_strict_fail),
    }


def audit_task_bank(tasks_path: Path, *, sample: int) -> dict[str, Any]:
    payload = _load_json(tasks_path)
    tasks = payload["tasks"][:sample] if sample > 0 else payload["tasks"]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        ref = str(task.get("reference_answer") or "")
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={"task_id": task["id"], "output": ref, "prompt": task.get("prompt")},
            dry_run=False,
        )
        score = _score_from_rollout(payload, task, rollout, compile_rate=0.5)
        rows.append({"task_id": task["id"], "rollout": rollout, "score": score})
    report = audit_rows(rows)
    report["source"] = "task_bank_oracle_sample"
    report["tasks_path"] = str(tasks_path)
    report["sample_size"] = len(tasks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit economistRL PPO/reward semantics.")
    parser.add_argument("--scored-jsonl", type=Path, default=None, help="Scored batch JSONL from lambda cycle.")
    parser.add_argument("--tasks", type=Path, default=REPO / "benchmarks" / "economistRL_tasks_v1.json")
    parser.add_argument("--sample", type=int, default=50, help="Task-bank oracle sample size when no scored JSONL.")
    args = parser.parse_args()

    if args.scored_jsonl is not None:
        rows = _load_scored_rows(args.scored_jsonl.expanduser().resolve())
        report = audit_rows(rows)
        report["source"] = str(args.scored_jsonl)
    else:
        report = audit_task_bank(args.tasks.expanduser().resolve(), sample=int(args.sample))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
