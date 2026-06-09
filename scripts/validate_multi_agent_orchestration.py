#!/usr/bin/env python3
"""Validate hierarchical multi-agent split/merge orchestration logic.

Expanded validation supports:
- Multiple task sources (repeat --tasks and optional --tasks-glob).
- Expected multi-agent coverage checks (from metadata/specialists).
- Additional robustness metrics (priority order, duplicate adapters, empty subtasks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt
from router.multi_agent import build_multi_agent_subtasks, merge_multi_agent_outputs

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO / "benchmarks" / "task_routing_mixed_tasks_v1.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON array at {path}")
    return [row for row in payload if isinstance(row, dict) and str(row.get("prompt") or "").strip()]


def _iter_task_paths(tasks: Iterable[Path], tasks_glob: str | None) -> list[Path]:
    resolved: list[Path] = []
    for p in tasks:
        rp = p.expanduser().resolve()
        if rp.is_file():
            resolved.append(rp)
    if tasks_glob:
        for p in sorted((REPO).glob(tasks_glob)):
            rp = p.expanduser().resolve()
            if rp.is_file() and rp not in resolved:
                resolved.append(rp)
    if not resolved:
        raise SystemExit("No task files found for validation.")
    return resolved


def _expected_multi_agent(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if isinstance(metadata, dict) and "expected_multi_agent" in metadata:
        return bool(metadata.get("expected_multi_agent"))
    specialists = row.get("specialists")
    if isinstance(specialists, list):
        names = [str(s).strip() for s in specialists if str(s).strip()]
        return len(set(names)) >= 2
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-agent split/merge orchestration.")
    parser.add_argument(
        "--tasks",
        type=Path,
        action="append",
        default=None,
        help="Task JSON file path (repeatable). Defaults to benchmark routing mixed tasks.",
    )
    parser.add_argument(
        "--tasks-glob",
        type=str,
        default=None,
        help="Optional repo-relative glob to include additional task files (e.g. 'benchmarks/*dual*json').",
    )
    parser.add_argument("--secondary-min-confidence", type=float, default=0.2)
    parser.add_argument(
        "--expected-multi-agent-min-rate",
        type=float,
        default=0.0,
        help="Optional minimum hit rate across rows expected to split into multi-agent (0 disables gate).",
    )
    parser.add_argument(
        "--max-unexpected-multi-agent-rate",
        type=float,
        default=1.0,
        help="Optional upper bound for multi-agent rate on rows not expected to split (1 disables gate).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "benchmarks" / "results" / "multi_agent_orchestration_validation.json",
    )
    args = parser.parse_args()

    policy = RoutingPolicy()
    task_paths = _iter_task_paths(args.tasks or [DEFAULT_TASKS], args.tasks_glob)
    rows: list[dict[str, Any]] = []
    row_sources: list[str] = []
    for path in task_paths:
        loaded = _load_rows(path)
        rows.extend(loaded)
        row_sources.extend([str(path)] * len(loaded))
    details: list[dict[str, Any]] = []
    split_ok = 0
    merge_ok = 0
    multi_agent_rows = 0
    expected_multi_agent_rows = 0
    expected_multi_agent_hit_rows = 0
    unexpected_multi_agent_rows = 0
    priority_ok_rows = 0
    unique_adapter_rows = 0
    empty_subtasks_rows = 0
    bucket_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    category_multi_counter: Counter[str] = Counter()
    category_expected_counter: Counter[str] = Counter()
    category_expected_hit_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    validation_errors_total = 0

    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt") or "").strip()
        source_path = row_sources[idx] if idx < len(row_sources) else "(unknown)"
        req = GenerationRequest(messages=messages_from_prompt(prompt))
        decision = policy.decide(req)
        subtasks = build_multi_agent_subtasks(
            prompt=prompt,
            primary_adapter_id=decision.adapter_id,
            secondary_adapter_id=getattr(decision, "secondary_adapter_id", None),
            secondary_confidence=float(getattr(decision, "secondary_confidence", 0.0) or 0.0),
            coarse_bucket=str(getattr(decision, "coarse_bucket", "unclassified") or "unclassified"),
            secondary_min_confidence=args.secondary_min_confidence,
        )
        split_valid = bool(subtasks and subtasks[0].adapter_id == decision.adapter_id)
        split_ok += int(split_valid)
        if len(subtasks) > 1:
            multi_agent_rows += 1

        # Simulate deterministic per-agent outputs (stands in for adapter generation).
        simulated_outputs = []
        for task in subtasks:
            simulated_outputs.append(
                {
                    "adapter_id": task.adapter_id,
                    "role": task.role,
                    "priority": task.priority,
                    "text": f"[{task.role}:{task.adapter_id}] simulated contribution for prompt-{idx}",
                }
            )
        merged = merge_multi_agent_outputs(user_prompt=prompt, outputs=simulated_outputs)
        merge_valid = all(
            marker in merged
            for marker in [f"[{task.role}:{task.adapter_id}] simulated contribution for prompt-{idx}" for task in subtasks]
        )
        merge_ok += int(merge_valid)
        expected_multi = _expected_multi_agent(row)
        if expected_multi:
            expected_multi_agent_rows += 1
        if expected_multi and len(subtasks) > 1:
            expected_multi_agent_hit_rows += 1
        if (not expected_multi) and len(subtasks) > 1:
            unexpected_multi_agent_rows += 1

        priorities = [int(task.priority) for task in subtasks]
        priority_valid = priorities == sorted(priorities) and (priorities[0] == 1 if priorities else False)
        priority_ok_rows += int(priority_valid)
        adapters = [task.adapter_id for task in subtasks]
        unique_adapters_valid = len(adapters) == len(set(adapters))
        unique_adapter_rows += int(unique_adapters_valid)
        if not subtasks:
            empty_subtasks_rows += 1

        bucket = str(getattr(decision, "coarse_bucket", "unclassified") or "unclassified")
        category = str(row.get("category") or "uncategorized")
        bucket_counter[bucket] += 1
        category_counter[category] += 1
        category_multi_counter[category] += int(len(subtasks) > 1)
        category_expected_counter[category] += int(expected_multi)
        category_expected_hit_counter[category] += int(expected_multi and len(subtasks) > 1)
        source_counter[source_path] += 1

        validation_errors: list[str] = []
        if not split_valid:
            validation_errors.append("split_primary_mismatch")
        if not merge_valid:
            validation_errors.append("merge_missing_subtask_output")
        if not priority_valid:
            validation_errors.append("priority_sequence_invalid")
        if not unique_adapters_valid:
            validation_errors.append("duplicate_adapter_in_subtasks")
        if expected_multi and len(subtasks) <= 1:
            validation_errors.append("expected_multi_agent_not_triggered")
        if (not expected_multi) and len(subtasks) > 1:
            validation_errors.append("unexpected_multi_agent_triggered")
        validation_errors_total += len(validation_errors)
        details.append(
            {
                "id": str(row.get("id") or f"row-{idx}"),
                "source_path": source_path,
                "category": category,
                "adapter_id": decision.adapter_id,
                "secondary_adapter_id": getattr(decision, "secondary_adapter_id", None),
                "secondary_confidence": float(getattr(decision, "secondary_confidence", 0.0) or 0.0),
                "coarse_bucket": bucket,
                "expected_multi_agent": expected_multi,
                "subtasks": [
                    {"role": task.role, "adapter_id": task.adapter_id, "priority": task.priority}
                    for task in subtasks
                ],
                "split_valid": split_valid,
                "merge_valid": merge_valid,
                "priority_valid": priority_valid,
                "unique_adapters_valid": unique_adapters_valid,
                "validation_errors": validation_errors,
            }
        )

    rows_total = len(rows)
    expected_multi_agent_rate = (
        expected_multi_agent_hit_rows / expected_multi_agent_rows if expected_multi_agent_rows else 0.0
    )
    non_expected_rows = max(0, rows_total - expected_multi_agent_rows)
    unexpected_multi_agent_rate = (unexpected_multi_agent_rows / non_expected_rows) if non_expected_rows else 0.0
    category_metrics: dict[str, Any] = {}
    for cat in sorted(category_counter):
        total = category_counter[cat]
        multi = category_multi_counter[cat]
        expected = category_expected_counter[cat]
        expected_hit = category_expected_hit_counter[cat]
        category_metrics[cat] = {
            "rows": total,
            "multi_agent_rows": multi,
            "multi_agent_rate": round(multi / total, 4) if total else 0.0,
            "expected_multi_agent_rows": expected,
            "expected_multi_agent_hit_rows": expected_hit,
            "expected_multi_agent_hit_rate": round(expected_hit / expected, 4) if expected else 0.0,
        }

    summary = {
        "schema_version": "multi_agent_orchestration_validation_v2",
        "tasks_paths": [str(p) for p in task_paths],
        "tasks_path": str(task_paths[0]) if task_paths else "",  # backward compatibility
        "rows": rows_total,
        "split_valid_rows": split_ok,
        "merge_valid_rows": merge_ok,
        "multi_agent_rows": multi_agent_rows,
        "split_valid_rate": round(split_ok / rows_total, 4) if rows_total else 0.0,
        "merge_valid_rate": round(merge_ok / rows_total, 4) if rows_total else 0.0,
        "expected_multi_agent_rows": expected_multi_agent_rows,
        "expected_multi_agent_hit_rows": expected_multi_agent_hit_rows,
        "expected_multi_agent_hit_rate": round(expected_multi_agent_rate, 4),
        "unexpected_multi_agent_rows": unexpected_multi_agent_rows,
        "unexpected_multi_agent_rate": round(unexpected_multi_agent_rate, 4),
        "priority_valid_rows": priority_ok_rows,
        "priority_valid_rate": round(priority_ok_rows / rows_total, 4) if rows_total else 0.0,
        "unique_adapter_rows": unique_adapter_rows,
        "unique_adapter_rate": round(unique_adapter_rows / rows_total, 4) if rows_total else 0.0,
        "empty_subtasks_rows": empty_subtasks_rows,
        "coarse_bucket_counts": dict(sorted(bucket_counter.items())),
        "source_counts": dict(sorted(source_counter.items())),
        "category_metrics": category_metrics,
        "validation_errors_total": validation_errors_total,
        "details": details,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    gate_failed = False
    if split_ok < rows_total or merge_ok < rows_total or multi_agent_rows == 0:
        gate_failed = True
    if args.expected_multi_agent_min_rate > 0 and expected_multi_agent_rate < args.expected_multi_agent_min_rate:
        gate_failed = True
    if unexpected_multi_agent_rate > args.max_unexpected_multi_agent_rate:
        gate_failed = True
    if gate_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
