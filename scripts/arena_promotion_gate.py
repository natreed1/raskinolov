#!/usr/bin/env python3
"""Check arena capability JSON against worst-task promotion thresholds.

Reads ``task_scores[]`` from ``arena_capability.json`` (see ``arena_capability_index.py``)
and exits **0** when distribution gates pass.

Use after ``ml_workflow.py arena-acceptance`` to avoid promoting adapters on headline ACI
alone when a single catastrophic task hides in the worst-of-six.

Examples::

    python scripts/arena_promotion_gate.py \\
      benchmarks/results/runs/latest-run-by-hand/arena_capability.json

    python scripts/arena_promotion_gate.py \\
      benchmarks/results/runs/20260429-025429_9793ff/arena_capability.json \\
      --min-worst-score 42 --min-mean-two-worst 45
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def worst_stats(task_scores: List[Dict[str, Any]]) -> Tuple[float, float, List[Tuple[str, float]]]:
    scores: List[float] = []
    by_id: List[Tuple[str, float]] = []
    for row in task_scores:
        s = row.get("score")
        tid = str(row.get("task_id") or "")
        if isinstance(s, (int, float)):
            scores.append(float(s))
            by_id.append((tid, float(s)))
        else:
            scores.append(0.0)
            by_id.append((tid, 0.0))
    scores.sort()
    worst = scores[0] if scores else 0.0
    worst_two_mean = sum(scores[:2]) / min(2, len(scores)) if scores else 0.0
    by_id.sort(key=lambda x: x[1])
    return worst, worst_two_mean, by_id


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gate on arena worst-of-six task scores")
    ap.add_argument(
        "arena_capability_json",
        type=Path,
        help="Path to arena_capability.json (often under benchmarks/results/runs/<id>/).",
    )
    ap.add_argument(
        "--min-worst-score",
        type=float,
        default=40.0,
        metavar="N",
        help="Minimum acceptable single worst task score (default 40).",
    )
    ap.add_argument(
        "--min-mean-two-worst",
        type=float,
        default=None,
        metavar="MEAN",
        help="Optional: mean of the two lowest task scores must be >= this.",
    )
    ap.add_argument(
        "--min-tasks",
        type=int,
        default=6,
        metavar="N",
        help="Minimum number of evaluated tasks required for promotion (default 6).",
    )
    ap.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=1.0,
        metavar="R",
        help="Minimum task coverage ratio required (default 1.0 = full catalog coverage for the suite).",
    )
    ap.add_argument(
        "--require-suite",
        choices=["core6", "all", "custom"],
        default=None,
        help="Optional: require matching suite label from acceptance summary metadata.",
    )
    ns = ap.parse_args(argv)

    path = ns.arena_capability_json.expanduser().resolve()
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 2
    doc = _read(path)
    rows = list(doc.get("task_scores") or [])
    worst, wm2, ordered = worst_stats(rows)
    aci = doc.get("arena_capability_index")
    tasks_evaluated = int(doc.get("tasks") or len(rows))
    total_catalog_tasks = int(doc.get("total_catalog_tasks") or 0)
    coverage_raw = doc.get("task_coverage_ratio")
    if coverage_raw is not None:
        coverage_ratio = float(coverage_raw)
    elif total_catalog_tasks > 0:
        coverage_ratio = tasks_evaluated / float(total_catalog_tasks)
    elif rows:
        # Older capability docs may omit coverage metadata; full-row fall back.
        coverage_ratio = 1.0
    else:
        coverage_ratio = 0.0
    suite_label = "unknown"
    summary_path_raw = doc.get("summary_path")
    if isinstance(summary_path_raw, str) and summary_path_raw.strip():
        summary_path = Path(summary_path_raw).expanduser()
        if summary_path.is_file():
            try:
                suite_label = str(_read(summary_path).get("suite") or "unknown")
            except Exception:
                suite_label = "unknown"

    lines = [
        f"arena_capability_index={aci}",
        f"tasks_evaluated={tasks_evaluated}",
        f"task_coverage_ratio={coverage_ratio:.4f}",
        f"suite={suite_label}",
        f"worst_task_score={worst:.2f}",
        f"mean_two_worst={wm2:.2f}",
        f"worst_tasks={ordered[:3]}",
    ]

    fails: List[str] = []
    if tasks_evaluated < ns.min_tasks:
        fails.append(f"tasks {tasks_evaluated} < --min-tasks {ns.min_tasks}")
    if coverage_ratio < ns.min_coverage_ratio:
        fails.append(
            f"coverage_ratio {coverage_ratio:.4f} < --min-coverage-ratio {ns.min_coverage_ratio:.4f}"
        )
    if ns.require_suite is not None and suite_label != ns.require_suite:
        fails.append(f"suite {suite_label!r} != --require-suite {ns.require_suite!r}")
    if worst < ns.min_worst_score:
        fails.append(f"worst {worst:.2f} < --min-worst-score {ns.min_worst_score}")
    if ns.min_mean_two_worst is not None and wm2 < ns.min_mean_two_worst:
        fails.append(f"mean_two_worst {wm2:.2f} < --min-mean-two-worst {ns.min_mean_two_worst}")

    ok = not fails
    print("\n".join(lines))
    if not ok:
        print("FAILED: " + "; ".join(fails), file=sys.stderr)
        return 1
    print("PASSED promotion gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
