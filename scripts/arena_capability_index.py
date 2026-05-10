#!/usr/bin/env python3
"""
Compute a deterministic Arena Capability Index from Game Task Arena acceptance artifacts.

This is intentionally not a judge. It scores objective task-completion signals:
applyability, TypeScript compile, export preservation, preview readiness, repair
rounds, runtime, and token pressure. String-substring benchmarks remain a separate
lexical smoke/regression signal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "game_task_arena_examples.json"
TRIALS_ROOT = REPO / "benchmarks" / "results" / "game_task_trials"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "summary.json"
        if candidate.is_file():
            return candidate
        raise SystemExit(f"Directory does not contain summary.json: {path}")
    return path


def load_tasks(tasks_path: Path) -> Dict[str, Dict[str, Any]]:
    data = _read_json(tasks_path)
    tasks = data.get("tasks", data)
    return {str(task["id"]): task for task in tasks}


def complexity_label(task: Dict[str, Any]) -> str:
    text = f"{task.get('complexity', '')} {task.get('notes', '')}".lower()
    match = re.search(r"complexity:\s*([a-z-]+)", text)
    if match:
        return match.group(1)
    if "medium-high" in text or "medium high" in text:
        return "medium-high"
    if "high" in text:
        return "high"
    if "low-medium" in text or "low medium" in text:
        return "low-medium"
    if "medium" in text:
        return "medium"
    if "low" in text:
        return "low"
    return "medium"


def complexity_weight(label: str) -> float:
    return {
        "low": 0.85,
        "low-medium": 0.95,
        "medium": 1.0,
        "medium-high": 1.15,
        "high": 1.25,
    }.get(label, 1.0)


def benchmark_tier(label: str) -> str:
    if label == "low":
        return "smoke_test"
    if label in {"low-medium", "medium"}:
        return "standard_dev_benchmark"
    return "high_reasoning_architecture_benchmark"


def tier_display_name(tier: str) -> str:
    return {
        "smoke_test": "Smoke Test",
        "standard_dev_benchmark": "Standard Dev Benchmark",
        "high_reasoning_architecture_benchmark": "High-Reasoning Architecture Benchmark",
    }.get(tier, tier)


def headline_cap(tiers_present: set[str]) -> float:
    """Avoid reporting a low-only pass as full arena capability."""
    if tiers_present == {"smoke_test"}:
        return 70.0
    if "high_reasoning_architecture_benchmark" not in tiers_present:
        return 85.0
    return 100.0


def _trial_attempt_dir(trial_id: str) -> Path:
    return TRIALS_ROOT / trial_id / "attempts" / "local"


def _load_generation_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    trial_id = str(result.get("trial_id") or "")
    if trial_id:
        path = _trial_attempt_dir(trial_id) / "generation_metrics.json"
        if path.is_file():
            try:
                return _read_json(path)
            except (OSError, json.JSONDecodeError):
                pass
    # Acceptance summaries already carry the most important booleans.
    return dict(result.get("generation_metrics") or {})


def _load_preview_status(result: Dict[str, Any]) -> Optional[bool]:
    if result.get("preview_ok") is not None:
        return bool(result.get("preview_ok"))
    trial_id = str(result.get("trial_id") or "")
    if not trial_id:
        return None
    path = _trial_attempt_dir(trial_id) / "preview.json"
    if not path.is_file():
        return None
    try:
        return _read_json(path).get("status") == "ready"
    except (OSError, json.JSONDecodeError):
        return None


def _last_round(metrics: Dict[str, Any]) -> Dict[str, Any]:
    rounds = metrics.get("tsc_rounds") or []
    if isinstance(rounds, list) and rounds:
        return dict(rounds[-1])
    return {}


def completion_score(metrics: Dict[str, Any], preview_ok: Optional[bool], gate: str) -> float:
    apply_ok = bool(metrics.get("apply_final_ok"))
    tsc_ok = bool(metrics.get("tsc_final_ok"))
    exports_ok = bool(metrics.get("exports_final_ok", True))
    preview_required = gate == "preview"
    if preview_required:
        preview_score = 1.0 if preview_ok else 0.0
        return (0.30 * apply_ok) + (0.30 * tsc_ok) + (0.20 * exports_ok) + (0.20 * preview_score)
    return (0.375 * apply_ok) + (0.375 * tsc_ok) + (0.25 * exports_ok)


def integration_score(metrics: Dict[str, Any]) -> float:
    rounds = metrics.get("tsc_rounds") or []
    retries = max(0, len(rounds) - 1) if isinstance(rounds, list) else 0
    last = _last_round(metrics)
    score = 1.0
    if not bool(metrics.get("apply_final_ok")):
        score -= 0.35
    if not bool(metrics.get("tsc_final_ok")):
        score -= 0.30
    if not bool(metrics.get("exports_final_ok", True)):
        score -= 0.25
    if str(last.get("apply_status", "")).startswith(("applied", "wrote")) is False:
        score -= 0.10
    score -= min(0.30, retries * 0.15)
    return _clamp01(score)


def efficiency_score(metrics: Dict[str, Any]) -> float:
    elapsed = float(metrics.get("elapsed_s") or 0.0)
    total_tokens = int(metrics.get("total_tokens") or 0)
    rounds = metrics.get("tsc_rounds") or []
    retries = max(0, len(rounds) - 1) if isinstance(rounds, list) else 0

    if elapsed <= 0:
        time_score = 0.5
    elif elapsed <= 120:
        time_score = 1.0
    elif elapsed <= 300:
        time_score = 0.75
    elif elapsed <= 600:
        time_score = 0.55
    else:
        time_score = 0.35

    if total_tokens <= 0:
        token_score = 0.5
    elif total_tokens <= 8_000:
        token_score = 1.0
    elif total_tokens <= 16_000:
        token_score = 0.8
    elif total_tokens <= 32_000:
        token_score = 0.6
    else:
        token_score = 0.4

    retry_score = _clamp01(1.0 - (0.20 * retries))
    return (0.45 * time_score) + (0.35 * token_score) + (0.20 * retry_score)


def score_task(result: Dict[str, Any], task: Dict[str, Any], gate: str) -> Dict[str, Any]:
    metrics = _load_generation_metrics(result)
    preview_ok = _load_preview_status(result)
    label = complexity_label(task)
    completion = completion_score(metrics, preview_ok, gate)
    integration = integration_score(metrics)
    efficiency = efficiency_score(metrics)
    total = (0.70 * completion) + (0.20 * integration) + (0.10 * efficiency)
    passed = bool(result.get("passed")) or (
        completion >= 0.999 and (gate != "preview" or preview_ok is True)
    )
    return {
        "task_id": result.get("task_id"),
        "trial_id": result.get("trial_id"),
        "passed": passed,
        "reason": result.get("reason") or result.get("metrics_reason") or result.get("preview_reason"),
        "complexity": label,
        "benchmark_tier": benchmark_tier(label),
        "complexity_weight": complexity_weight(label),
        "score": round(100.0 * total, 2),
        "completion": round(100.0 * completion, 2),
        "integration": round(100.0 * integration, 2),
        "efficiency": round(100.0 * efficiency, 2),
        "apply_ok": bool(metrics.get("apply_final_ok")),
        "tsc_ok": bool(metrics.get("tsc_final_ok")),
        "exports_ok": bool(metrics.get("exports_final_ok", True)),
        "preview_ok": preview_ok,
        "rounds": len(metrics.get("tsc_rounds") or []),
        "elapsed_s": metrics.get("elapsed_s"),
        "total_tokens": metrics.get("total_tokens"),
    }


def weighted_average(rows: Iterable[Dict[str, Any]], key: str) -> float:
    total_weight = 0.0
    total = 0.0
    for row in rows:
        weight = float(row.get("complexity_weight") or 1.0)
        total_weight += weight
        total += float(row.get(key) or 0.0) * weight
    return round(total / total_weight, 2) if total_weight else 0.0


def tier_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tier in ["smoke_test", "standard_dev_benchmark", "high_reasoning_architecture_benchmark"]:
        subset = [row for row in rows if row.get("benchmark_tier") == tier]
        if not subset:
            continue
        out[tier] = {
            "name": tier_display_name(tier),
            "tasks": len(subset),
            "accepted": sum(1 for row in subset if row.get("passed")),
            "index": weighted_average(subset, "score"),
            "completion": weighted_average(subset, "completion"),
            "integration": weighted_average(subset, "integration"),
            "efficiency": weighted_average(subset, "efficiency"),
        }
    return out


def task_type_breakdown(rows: List[Dict[str, Any]], tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        tid = str(row.get("task_id") or "")
        task = tasks.get(tid, {})
        task_type = str(task.get("task_type") or "unknown")
        by_type.setdefault(task_type, []).append(row)
    for task_type, subset in sorted(by_type.items()):
        out[task_type] = {
            "tasks": len(subset),
            "accepted": sum(1 for row in subset if row.get("passed")),
            "index": weighted_average(subset, "score"),
            "completion": weighted_average(subset, "completion"),
            "integration": weighted_average(subset, "integration"),
            "efficiency": weighted_average(subset, "efficiency"),
        }
    return out


def diversified_index(
    rows: List[Dict[str, Any]],
    raw_index: float,
    task_types: Dict[str, Dict[str, Any]],
    catalog_task_count: int,
) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "task_type_balanced_index": 0.0,
            "tail_robustness_index": 0.0,
            "coverage_ratio": 0.0,
            "coverage_multiplier": 0.7,
            "active_task_types": 0,
            "task_type_cap": 70.0,
            "raw_diversified_index": 0.0,
            "diversified_arena_capability_index": 0.0,
        }

    type_scores = [
        float(value.get("index") or 0.0)
        for _, value in sorted(task_types.items(), key=lambda item: item[0])
    ]
    task_type_balanced = round(sum(type_scores) / len(type_scores), 2) if type_scores else 0.0

    sorted_scores = sorted(float(row.get("score") or 0.0) for row in rows)
    tail_k = min(n, max(2, int(math.ceil(n * 0.25))))
    tail_mean = round(sum(sorted_scores[:tail_k]) / tail_k, 2)

    coverage_ratio = min(1.0, n / max(1, catalog_task_count))
    coverage_multiplier = 0.7 + (0.3 * coverage_ratio)
    active_types = len(type_scores)
    if active_types <= 1:
        task_type_cap = 70.0
    elif active_types == 2:
        task_type_cap = 82.0
    elif active_types == 3:
        task_type_cap = 92.0
    else:
        task_type_cap = 100.0

    raw_diversified = round(
        (0.55 * task_type_balanced) + (0.30 * tail_mean) + (0.15 * raw_index),
        2,
    )
    coverage_adjusted = round(raw_diversified * coverage_multiplier, 2)
    diversified = round(min(coverage_adjusted, task_type_cap), 2)

    return {
        "task_type_balanced_index": task_type_balanced,
        "tail_robustness_index": tail_mean,
        "coverage_ratio": round(coverage_ratio, 4),
        "coverage_multiplier": round(coverage_multiplier, 4),
        "active_task_types": active_types,
        "task_type_cap": task_type_cap,
        "raw_diversified_index": raw_diversified,
        "diversified_arena_capability_index": diversified,
    }


def compute_index(summary_path: Path, tasks_path: Path = DEFAULT_TASKS) -> Dict[str, Any]:
    summary_path = resolve_summary_path(summary_path)
    summary = _read_json(summary_path)
    tasks = load_tasks(tasks_path)
    gate = str(summary.get("gate") or "preview")
    rows: List[Dict[str, Any]] = []
    for result in summary.get("results", []):
        task = tasks.get(str(result.get("task_id")), {})
        rows.append(score_task(result, task, gate))

    accepted = sum(1 for row in rows if row.get("passed"))
    total_tasks = len(rows)
    raw_index = weighted_average(rows, "score")
    tiers_present = {str(row.get("benchmark_tier")) for row in rows if row.get("benchmark_tier")}
    cap = headline_cap(tiers_present)
    headline_index = round(min(raw_index, cap), 2)
    task_types = task_type_breakdown(rows, tasks)
    diversified = diversified_index(
        rows=rows,
        raw_index=raw_index,
        task_types=task_types,
        catalog_task_count=len(tasks),
    )
    headline_kind = (
        "smoke_test"
        if tiers_present == {"smoke_test"}
        else "standard_dev_benchmark"
        if "high_reasoning_architecture_benchmark" not in tiers_present
        else "high_reasoning_architecture_benchmark"
    )
    index = {
        "summary_path": str(summary_path),
        "gate": gate,
        "tasks": total_tasks,
        "accepted": accepted,
        "acceptance_rate": round(accepted / total_tasks, 4) if total_tasks else 0.0,
        "arena_capability_index": headline_index,
        "raw_arena_capability_index": raw_index,
        "headline_kind": headline_kind,
        "headline_cap": cap,
        "completion": weighted_average(rows, "completion"),
        "integration": weighted_average(rows, "integration"),
        "efficiency": weighted_average(rows, "efficiency"),
        "tier_breakdown": tier_breakdown(rows),
        "task_type_breakdown": task_types,
        "task_catalog_count": len(tasks),
        "task_coverage_ratio": diversified["coverage_ratio"],
        "task_type_balanced_index": diversified["task_type_balanced_index"],
        "tail_robustness_index": diversified["tail_robustness_index"],
        "raw_diversified_index": diversified["raw_diversified_index"],
        "diversified_arena_capability_index": diversified["diversified_arena_capability_index"],
        "diversified_details": diversified,
        "task_scores": rows,
    }
    return index


def write_markdown(index: Dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Arena Capability Index",
        "",
        f"- **Summary:** `{index['summary_path']}`",
        f"- **Gate:** `{index['gate']}`",
        f"- **Accepted:** `{index['accepted']}/{index['tasks']}`",
        f"- **Headline:** `{tier_display_name(str(index.get('headline_kind')))}`",
        f"- **Arena Capability Index:** `{index['arena_capability_index']}/100`",
        f"- **Raw index:** `{index.get('raw_arena_capability_index')}/100`",
        f"- **Diversified Arena Capability Index:** `{index.get('diversified_arena_capability_index')}/100`",
        f"- **Task-type balanced index:** `{index.get('task_type_balanced_index')}/100`",
        f"- **Tail robustness index:** `{index.get('tail_robustness_index')}/100`",
        f"- **Task coverage:** `{index.get('tasks')}/{index.get('task_catalog_count')}` "
        f"(`{round(100.0 * float(index.get('task_coverage_ratio') or 0.0), 1)}%`)",
        f"- **Completion:** `{index['completion']}/100`",
        f"- **Integration:** `{index['integration']}/100`",
        f"- **Efficiency:** `{index['efficiency']}/100`",
        "",
        "",
        "## Tier Breakdown",
        "",
        "| Tier | Accepted | Index | Completion | Integration | Efficiency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tier, row in index.get("tier_breakdown", {}).items():
        lines.append(
            f"| {row['name']} | {row['accepted']}/{row['tasks']} | {row['index']} | "
            f"{row['completion']} | {row['integration']} | {row['efficiency']} |"
        )
    lines.extend(
        [
            "",
            "## Task Type Breakdown",
            "",
            "| Task Type | Accepted | Index | Completion | Integration | Efficiency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task_type, row in index.get("task_type_breakdown", {}).items():
        lines.append(
            f"| `{task_type}` | {row['accepted']}/{row['tasks']} | {row['index']} | "
            f"{row['completion']} | {row['integration']} | {row['efficiency']} |"
        )
    lines.extend(
        [
            "",
            "## Tasks",
            "",
            "| Task | Pass | Score | Completion | Integration | Efficiency | Complexity | Tier | Reason |",
            "|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in index["task_scores"]:
        lines.append(
            f"| `{row.get('task_id')}` | {bool(row.get('passed'))} | "
            f"{row.get('score')} | {row.get('completion')} | {row.get('integration')} | "
            f"{row.get('efficiency')} | {row.get('complexity')} | {row.get('benchmark_tier')} | "
            f"`{row.get('reason') or 'ok'}` |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deterministic Arena Capability Index")
    parser.add_argument("summary", type=Path, help="Acceptance summary.json or directory containing it")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    index = compute_index(args.summary, args.tasks)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(index, args.output_md)

    print(json.dumps(index, indent=2))
    print(
        "=== Arena Capability Index: "
        f"{index['arena_capability_index']:.1f}/100 "
        f"(diversified {index.get('diversified_arena_capability_index', 0):.1f}/100) "
        f"[{tier_display_name(str(index.get('headline_kind')))}; raw {index.get('raw_arena_capability_index', 0):.1f}] "
        f"(accepted {index['accepted']}/{index['tasks']}, "
        f"completion {index['completion']:.1f}, integration {index['integration']:.1f}, "
        f"efficiency {index['efficiency']:.1f}) ==="
    )


if __name__ == "__main__":
    main()
