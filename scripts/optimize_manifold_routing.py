#!/usr/bin/env python3
"""Grid-search manifold/similarity router thresholds and emit best config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = [
    REPO / "benchmarks" / "task_routing_tasks.json",
    REPO / "benchmarks" / "task_routing_mixed_tasks_v1.json",
]


def _parse_float_grid(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


def _safe_rate(passed: int, total: int) -> float:
    return round((passed / total), 4) if total else 0.0


def _name_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


@dataclass
class BenchmarkCase:
    name: str
    score: float
    margin: float
    unknown_conf: float
    hierarchy_width: int
    hierarchy_min_hits: int
    weighted_overall: float
    weighted_route: float
    weighted_adapter: float
    total_rows: int
    rows_path: Path
    summary_path: Path
    gate_output_path: Path
    gate_passed: bool | None


def _run_single_benchmark(
    *,
    task_path: Path,
    score: float,
    margin: float,
    unknown_conf: float,
    hierarchy_width: int,
    hierarchy_min_hits: int,
    out_rows: Path,
    out_summary: Path,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["ROUTER_ADAPTER_SELECTION_MODE"] = "similarity"
    env["ROUTER_SIMILARITY_MIN_SCORE"] = str(score)
    env["ROUTER_SIMILARITY_MIN_MARGIN"] = str(margin)
    env["ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD"] = str(unknown_conf)
    env["ROUTER_HIERARCHY_CANDIDATE_WIDTH"] = str(hierarchy_width)
    env["ROUTER_HIERARCHY_MIN_COARSE_HITS"] = str(hierarchy_min_hits)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_routing_benchmark.py"),
        "--tasks",
        str(task_path),
        "--mode",
        "both",
        "--output-jsonl",
        str(out_rows),
        "--summary-json",
        str(out_summary),
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    summary = json.loads(out_summary.read_text(encoding="utf-8"))
    return {
        "returncode": proc.returncode,
        "task_path": str(task_path.relative_to(REPO)),
        "rows_path": str(out_rows.relative_to(REPO)),
        "summary_path": str(out_summary.relative_to(REPO)),
        "summary": summary,
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-8:]),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-8:]),
    }


def _merge_rows(paths: list[Path], dest: Path) -> int:
    count = 0
    with dest.open("w", encoding="utf-8") as sink:
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                sink.write(line + "\n")
                count += 1
    return count


def _merge_summaries(task_results: list[dict[str, Any]], dest: Path) -> dict[str, Any]:
    route_total = route_passed = 0
    adapter_total = adapter_passed = 0
    overall_total = overall_passed = 0
    for result in task_results:
        summary = result["summary"]
        route = summary.get("route") or {}
        adapter = summary.get("adapter") or {}
        overall = summary.get("overall") or {}
        route_total += int(route.get("total") or 0)
        route_passed += int(route.get("passed") or 0)
        adapter_total += int(adapter.get("total") or 0)
        adapter_passed += int(adapter.get("passed") or 0)
        overall_total += int(overall.get("total") or 0)
        overall_passed += int(overall.get("passed") or 0)
    merged = {
        "schema_version": "routing_benchmark_summary_v2",
        "mode": "both",
        "tasks_path": "merged",
        "route": {
            "total": route_total,
            "passed": route_passed,
            "accuracy": _safe_rate(route_passed, route_total),
        },
        "adapter": {
            "total": adapter_total,
            "passed": adapter_passed,
            "accuracy": _safe_rate(adapter_passed, adapter_total),
        },
        "overall": {
            "total": overall_total,
            "passed": overall_passed,
            "accuracy": _safe_rate(overall_passed, overall_total),
        },
    }
    dest.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _run_gate(summary_path: Path, rows_path: Path, output_path: Path) -> bool:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "router_promotion_gate.py"),
        "--summary-json",
        str(summary_path),
        "--rows-jsonl",
        str(rows_path),
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return bool(payload.get("passed"))
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize manifold routing thresholds.")
    parser.add_argument("--tasks", type=Path, action="append", default=[])
    parser.add_argument("--score-grid", default="0.08,0.10,0.12,0.14")
    parser.add_argument("--margin-grid", default="0.02,0.03,0.05")
    parser.add_argument("--unknown-conf-grid", default="0.58,0.62,0.66")
    parser.add_argument("--hierarchy-width-grid", default="3,4")
    parser.add_argument("--hierarchy-min-hits-grid", default="1,2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "benchmarks" / "results" / "routing_manifold_optimization",
    )
    args = parser.parse_args()

    tasks = [p.expanduser().resolve() for p in (args.tasks or DEFAULT_TASKS)]
    scores = _parse_float_grid(args.score_grid)
    margins = _parse_float_grid(args.margin_grid)
    unknown_thresholds = _parse_float_grid(args.unknown_conf_grid)
    hierarchy_widths = [int(v) for v in _parse_float_grid(args.hierarchy_width_grid)]
    hierarchy_min_hits = [int(v) for v in _parse_float_grid(args.hierarchy_min_hits_grid)]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[BenchmarkCase] = []
    details: list[dict[str, Any]] = []
    for score in scores:
        for margin in margins:
            for unknown_conf in unknown_thresholds:
                for hierarchy_width in hierarchy_widths:
                    for hierarchy_hits in hierarchy_min_hits:
                        case_name = (
                            f"s{_name_token(score)}_m{_name_token(margin)}_u{_name_token(unknown_conf)}"
                            f"_w{hierarchy_width}_h{hierarchy_hits}"
                        )
                        task_results: list[dict[str, Any]] = []
                        row_paths: list[Path] = []
                        for idx, task in enumerate(tasks):
                            task_tag = task.stem.replace("-", "_")
                            rows_path = out_dir / f"{case_name}_{idx}_{task_tag}_rows.jsonl"
                            summary_path = out_dir / f"{case_name}_{idx}_{task_tag}_summary.json"
                            result = _run_single_benchmark(
                                task_path=task,
                                score=score,
                                margin=margin,
                                unknown_conf=unknown_conf,
                                hierarchy_width=hierarchy_width,
                                hierarchy_min_hits=hierarchy_hits,
                                out_rows=rows_path,
                                out_summary=summary_path,
                            )
                            row_paths.append(rows_path)
                            task_results.append(result)

                        merged_rows = out_dir / f"{case_name}_merged_rows.jsonl"
                        merged_summary = out_dir / f"{case_name}_merged_summary.json"
                        gate_output = out_dir / f"{case_name}_gate.json"
                        merged_total = _merge_rows(row_paths, merged_rows)
                        merged_summary_payload = _merge_summaries(task_results, merged_summary)
                        gate_passed = _run_gate(merged_summary, merged_rows, gate_output)

                        case = BenchmarkCase(
                            name=case_name,
                            score=score,
                            margin=margin,
                            unknown_conf=unknown_conf,
                            hierarchy_width=hierarchy_width,
                            hierarchy_min_hits=hierarchy_hits,
                            weighted_overall=float((merged_summary_payload.get("overall") or {}).get("accuracy") or 0.0),
                            weighted_route=float((merged_summary_payload.get("route") or {}).get("accuracy") or 0.0),
                            weighted_adapter=float((merged_summary_payload.get("adapter") or {}).get("accuracy") or 0.0),
                            total_rows=merged_total,
                            rows_path=merged_rows,
                            summary_path=merged_summary,
                            gate_output_path=gate_output,
                            gate_passed=gate_passed,
                        )
                        cases.append(case)
                        details.append(
                            {
                                "case": case_name,
                                "score": score,
                                "margin": margin,
                                "unknown_conf_threshold": unknown_conf,
                                "hierarchy_candidate_width": hierarchy_width,
                                "hierarchy_min_coarse_hits": hierarchy_hits,
                                "task_results": task_results,
                                "merged_rows": str(merged_rows.relative_to(REPO)),
                                "merged_summary": str(merged_summary.relative_to(REPO)),
                                "gate_output": str(gate_output.relative_to(REPO)),
                                "weighted_overall": case.weighted_overall,
                                "weighted_route": case.weighted_route,
                                "weighted_adapter": case.weighted_adapter,
                                "gate_passed": gate_passed,
                            }
                        )
                        print(
                            f"{case_name}: overall={case.weighted_overall:.4f} "
                            f"route={case.weighted_route:.4f} adapter={case.weighted_adapter:.4f} "
                            f"gate={'PASS' if gate_passed else 'FAIL'}"
                        )

    if not cases:
        raise SystemExit("No optimization cases were executed.")

    best = sorted(
        cases,
        key=lambda c: (c.weighted_overall, c.weighted_adapter, c.weighted_route),
        reverse=True,
    )[0]
    report = {
        "schema_version": "routing_manifold_optimization_v1",
        "tasks": [str(p.relative_to(REPO)) for p in tasks],
        "score_grid": scores,
        "margin_grid": margins,
        "unknown_conf_grid": unknown_thresholds,
        "hierarchy_width_grid": hierarchy_widths,
        "hierarchy_min_hits_grid": hierarchy_min_hits,
        "cases": details,
        "best_case": {
            "case": best.name,
            "score": best.score,
            "margin": best.margin,
            "unknown_conf_threshold": best.unknown_conf,
            "hierarchy_candidate_width": best.hierarchy_width,
            "hierarchy_min_coarse_hits": best.hierarchy_min_hits,
            "weighted_overall": best.weighted_overall,
            "weighted_route": best.weighted_route,
            "weighted_adapter": best.weighted_adapter,
            "gate_passed": best.gate_passed,
            "summary_path": str(best.summary_path.relative_to(REPO)),
            "rows_path": str(best.rows_path.relative_to(REPO)),
            "gate_output_path": str(best.gate_output_path.relative_to(REPO)),
        },
    }
    report_path = out_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best case: {best.name}")
    print(f"Optimization report: {report_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
