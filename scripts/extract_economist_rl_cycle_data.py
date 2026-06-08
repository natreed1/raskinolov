#!/usr/bin/env python3
"""Summarize economistRL cycle rollouts, scores, and PPO manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize_scored(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    rewards = [float(r["score"]["reward"]) for r in rows]
    tt = [float(r["score"]["components"]["targeted_tests"]) for r in rows]
    compiled = sum(1 for r in rows if (r["rollout"].get("compiled") is True))
    tests_apply = sum(
        1 for r in rows if any("tests/" in str(p) for p in (r["rollout"].get("changed_files") or []))
    )
    vitest_pass = 0
    vitest_total = 0
    for r in rows:
        for it in (r["rollout"].get("vitest_report") or {}).get("per_it") or []:
            vitest_total += 1
            if it.get("passed"):
                vitest_pass += 1
    return {
        "count": len(rows),
        "mean_reward": round(statistics.mean(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "stdev_reward": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "mean_targeted_tests_pct": round(statistics.mean(tt), 2),
        "targeted_tests_gt0": sum(1 for x in tt if x > 0),
        "targeted_tests_gte50": sum(1 for x in tt if x >= 50),
        "compiled_true": compiled,
        "vitest_goals_passed": vitest_pass,
        "vitest_goals_total": vitest_total,
        "changed_files_with_tests": tests_apply,
        "top_by_targeted_tests": sorted(
            [
                {
                    "task_id": r["task_id"],
                    "reward": round(float(r["score"]["reward"]), 4),
                    "targeted_tests_pct": float(r["score"]["components"]["targeted_tests"]),
                }
                for r in rows
            ],
            key=lambda x: (-x["targeted_tests_pct"], -x["reward"]),
        )[:8],
    }


def _summarize_ppo(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": data.get("status"),
        "train_backend": data.get("train_backend"),
        "samples": data.get("samples"),
        "ppo_epochs": data.get("ppo_epochs"),
        "source_adapter": data.get("source_adapter"),
        "candidate_adapter": data.get("candidate_adapter"),
        "sample_stats": data.get("sample_stats"),
        "steps": data.get("steps"),
        "saved_lora_tensors": data.get("saved_lora_tensors"),
    }


def summarize_cycle(cycle_id: int, results_root: Path) -> dict[str, Any]:
    suffix = f"{cycle_id:03d}"
    scored_path = results_root / "scores" / f"scored_batch_{suffix}.jsonl"
    rollout_path = results_root / "rollouts" / f"rollout_batch_{suffix}.jsonl"
    ppo_path = results_root / "ppo" / f"ppo_train_{suffix}.json"
    manifest_path = results_root / "manifests" / f"cycle_{suffix}_manifest.json"
    candidate = REPO / "checkpoints" / "adapters" / "economistRL" / f"rl_pass_{suffix}"

    scored = _load_jsonl(scored_path)
    rollouts = _load_jsonl(rollout_path)
    first_rollout = rollouts[0] if rollouts else {}
    return {
        "cycle_id": cycle_id,
        "artifacts": {
            "rollout_file": str(rollout_path) if rollout_path.is_file() else None,
            "scored_file": str(scored_path) if scored_path.is_file() else None,
            "ppo_manifest": str(ppo_path) if ppo_path.is_file() else None,
            "cycle_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "candidate_adapter": str(candidate) if candidate.is_dir() else None,
        },
        "rollout_adapter": first_rollout.get("adapter_path"),
        "scored_summary": _summarize_scored(scored),
        "ppo_summary": _summarize_ppo(ppo_path),
        "cycle_manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, nargs="+", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "benchmarks" / "results" / "economistRL",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    payload = {
        "schema_version": "economist_rl_cycle_extract_v1",
        "results_root": str(results_root),
        "cycles": [summarize_cycle(cid, results_root) for cid in args.cycles],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    for row in payload["cycles"]:
        s = row["scored_summary"]
        p = row.get("ppo_summary") or {}
        print(
            f"cycle {row['cycle_id']:03d}: n={s.get('count',0)} "
            f"mean_reward={s.get('mean_reward')} tt_mean={s.get('mean_targeted_tests_pct')}% "
            f"ppo={p.get('status')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
