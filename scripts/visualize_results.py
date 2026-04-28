#!/usr/bin/env python3
"""
Generate a static HTML dashboard for `ml_workflow.py` run artifacts.

Reads:
  - benchmarks/results/runs/*/manifest.json
  - docs/run_history.md (optional metadata enrichment)

Writes:
  - benchmarks/results/run_dashboard.html (default)

Usage:
  source .venv/bin/activate
  python scripts/visualize_results.py
  python scripts/visualize_results.py --out benchmarks/results/custom_dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO / "benchmarks" / "results" / "runs"
DEFAULT_HISTORY = REPO / "docs" / "run_history.md"
DEFAULT_OUT = REPO / "benchmarks" / "results" / "run_dashboard.html"


@dataclass
class RunRow:
    run_id: str
    subcommand: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    elapsed_s: float
    final_exit_code: Optional[int]
    step_count: int
    failed_steps: int
    benchmark_summary: str
    benchmark_rate: Optional[float]
    adapter_path: str
    git_short: str
    history_iso: str
    history_train_iters: str

    @property
    def status(self) -> str:
        if self.final_exit_code is not None:
            return "failed" if self.final_exit_code != 0 else "ok"
        return "failed" if self.failed_steps > 0 else "ok"


def _parse_iso_dt(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_benchmark_rate(summary: str) -> Optional[float]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", summary or "")
    if not m:
        return None
    den = int(m.group(2))
    if den <= 0:
        return None
    return int(m.group(1)) / den


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_history_table(history_path: Path) -> Dict[str, Dict[str, str]]:
    if not history_path.is_file():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    for raw in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("|") or "benchmarks/results/runs/" not in line:
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        if parts[0] == "UTC ISO" or set(parts[0]) == {"-"}:
            continue

        # Current history rows include Exit/Status plus separate trained and
        # benchmarked adapters. Older rows had only six columns; keep both
        # readable so old generated dashboards do not become orphaned.
        if len(parts) >= 9:
            artifacts = parts[8].strip("` ")
            train_iters = parts[4].strip("` ")
            benchmark = parts[5].strip("` ")
            trained_adapter = parts[6].strip("` ")
            benchmark_adapter = parts[7].strip("` ")
        else:
            artifacts = parts[5].strip("` ")
            train_iters = parts[2].strip("` ")
            benchmark = parts[3].strip("` ")
            trained_adapter = parts[4].strip("` ")
            benchmark_adapter = parts[4].strip("` ")
        m = re.search(r"benchmarks/results/runs/([^/]+)/?$", artifacts)
        if not m:
            continue
        run_id = m.group(1)
        rows[run_id] = {
            "utc_iso": parts[0].strip("` "),
            "subcommand": parts[1].strip("` "),
            "train_iters": train_iters,
            "benchmark": benchmark,
            "adapter": trained_adapter if trained_adapter != "—" else benchmark_adapter,
            "trained_adapter": trained_adapter,
            "benchmark_adapter": benchmark_adapter,
        }
    return rows


def _load_runs(runs_dir: Path, history_rows: Dict[str, Dict[str, str]]) -> List[RunRow]:
    rows: List[RunRow] = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_id = str(payload.get("run_id") or manifest_path.parent.name)
        steps = payload.get("steps")
        if not isinstance(steps, list):
            steps = []

        step_elapsed_s = 0.0
        failed_steps = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            try:
                step_elapsed_s += float(step.get("elapsed_s") or 0.0)
            except (TypeError, ValueError):
                pass
            step_exit = _parse_optional_int(step.get("exit_code"))
            if step_exit is not None and step_exit != 0:
                failed_steps += 1

        started_at = _parse_iso_dt(payload.get("started_at"))
        finished_at = _parse_iso_dt(payload.get("finished_at"))
        elapsed_s = step_elapsed_s
        try:
            elapsed_s = float(payload.get("elapsed_s") or step_elapsed_s)
        except (TypeError, ValueError):
            elapsed_s = step_elapsed_s
        if finished_at is None and started_at is not None:
            finished_at = started_at + timedelta_seconds(elapsed_s)
        final_exit_code = _parse_optional_int(payload.get("final_exit_code"))

        benchmark_summary = str(payload.get("benchmark_summary") or "—")
        history = history_rows.get(run_id, {})
        history_benchmark = history.get("benchmark", "")
        if benchmark_summary == "—" and history_benchmark and history_benchmark != "—":
            benchmark_summary = history_benchmark

        rows.append(
            RunRow(
                run_id=run_id,
                subcommand=str(payload.get("subcommand") or history.get("subcommand") or "unknown"),
                started_at=started_at,
                finished_at=finished_at,
                elapsed_s=elapsed_s,
                final_exit_code=final_exit_code,
                step_count=len(steps),
                failed_steps=failed_steps,
                benchmark_summary=benchmark_summary,
                benchmark_rate=_parse_benchmark_rate(benchmark_summary),
                adapter_path=str(
                    payload.get("benchmark_adapter_path")
                    or payload.get("adapter_path")
                    or history.get("adapter")
                    or "—"
                ),
                git_short=str(payload.get("git_short") or "unknown"),
                history_iso=history.get("utc_iso", ""),
                history_train_iters=history.get("train_iters", ""),
            )
        )

    rows.sort(
        key=lambda r: (
            r.started_at or _parse_iso_dt(r.history_iso) or datetime.min.replace(tzinfo=timezone.utc),
            r.run_id,
        ),
        reverse=True,
    )
    return rows


def timedelta_seconds(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=max(0.0, seconds))


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def _bucket_by_day(rows: List[RunRow]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for r in rows:
        dt = r.started_at or _parse_iso_dt(r.history_iso)
        if dt is None:
            continue
        day = dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
        counts[day] = counts.get(day, 0) + 1
    return sorted(counts.items())


def _render_html(rows: List[RunRow], generated_at: datetime) -> str:
    total = len(rows)
    passed = sum(1 for r in rows if r.status == "ok")
    failed = total - passed
    avg_elapsed = statistics.mean([r.elapsed_s for r in rows]) if rows else 0.0

    bench_rows = [r for r in rows if r.benchmark_rate is not None]
    avg_bench = statistics.mean([r.benchmark_rate for r in bench_rows]) if bench_rows else None

    max_elapsed = max((r.elapsed_s for r in rows), default=1.0)
    sub_counts: Dict[str, int] = {}
    for r in rows:
        sub_counts[r.subcommand] = sub_counts.get(r.subcommand, 0) + 1

    day_counts = _bucket_by_day(rows)
    day_max = max((count for _, count in day_counts), default=1)

    summary_items = [
        ("Total runs", str(total)),
        ("Successful", f"{passed} ({(passed / total * 100):.0f}%)" if total else "0"),
        ("Failed", str(failed)),
        ("Avg duration", _fmt_elapsed(avg_elapsed) if total else "—"),
        ("Benchmarks with score", str(len(bench_rows))),
        ("Avg benchmark pass rate", f"{(avg_bench * 100):.1f}%" if avg_bench is not None else "—"),
    ]
    summary_cards = "\n".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'
        for k, v in summary_items
    )

    sub_bars = "\n".join(
        (
            '<div class="bar-row">'
            f'<span class="bar-label">{html.escape(sub)}</span>'
            f'<div class="bar"><span style="width:{(count / total * 100) if total else 0:.1f}%"></span></div>'
            f'<span class="bar-val">{count}</span>'
            "</div>"
        )
        for sub, count in sorted(sub_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    day_bars = "\n".join(
        (
            '<div class="bar-row">'
            f'<span class="bar-label">{html.escape(day)}</span>'
            f'<div class="bar"><span style="width:{(count / day_max * 100) if day_max else 0:.1f}%"></span></div>'
            f'<span class="bar-val">{count}</span>'
            "</div>"
        )
        for day, count in day_counts
    )

    table_rows = []
    for r in rows:
        status_class = "ok" if r.status == "ok" else "failed"
        rate_label = f"{r.benchmark_rate * 100:.0f}%" if r.benchmark_rate is not None else "—"
        elapsed_pct = (r.elapsed_s / max_elapsed * 100) if max_elapsed else 0
        table_rows.append(
            "<tr>"
            f'<td><code>{html.escape(r.run_id)}</code></td>'
            f"<td>{html.escape(r.subcommand)}</td>"
            f'<td><span class="status {status_class}">{html.escape(r.status)}</span></td>'
            f"<td>{r.final_exit_code if r.final_exit_code is not None else '—'}</td>"
            f"<td>{_fmt_dt(r.started_at)}</td>"
            f"<td>{_fmt_elapsed(r.elapsed_s)}</td>"
            f"<td>{html.escape(r.benchmark_summary)} ({rate_label})</td>"
            f'<td><code>{html.escape(r.adapter_path)}</code></td>'
            f"<td>{r.step_count}</td>"
            f"<td>{r.failed_steps}</td>"
            f"<td><div class='mini'><span style='width:{elapsed_pct:.1f}%'></span></div></td>"
            "</tr>"
        )

    if not table_rows:
        table_rows.append("<tr><td colspan='11'>No runs found.</td></tr>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML Workflow Results Dashboard</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    p.meta {{ margin: 4px 0 16px; color: #475569; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 20px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
    }}
    .card .k {{ color: #475569; font-size: 12px; }}
    .card .v {{ font-weight: 700; font-size: 20px; margin-top: 4px; }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 140px 1fr 40px;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      font-size: 13px;
    }}
    .bar {{
      height: 10px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: #2563eb;
    }}
    .bar-label {{ color: #334155; }}
    .bar-val {{ text-align: right; color: #334155; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 8px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 600;
      color: #334155;
      position: sticky;
      top: 0;
    }}
    .status {{
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .status.ok {{
      background: #dcfce7;
      color: #166534;
    }}
    .status.failed {{
      background: #fee2e2;
      color: #991b1b;
    }}
    .mini {{
      height: 8px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
      min-width: 80px;
    }}
    .mini > span {{
      display: block;
      height: 100%;
      background: #0ea5e9;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>ML Workflow Results Dashboard</h1>
  <p class="meta">Generated UTC: {generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")} &middot; Source: <code>benchmarks/results/runs/*/manifest.json</code> + <code>docs/run_history.md</code></p>

  <section class="grid">
    {summary_cards}
  </section>

  <section class="panel">
    <h2>Runs by Subcommand</h2>
    {sub_bars or "<p>No run data.</p>"}
  </section>

  <section class="panel">
    <h2>Runs by UTC Day</h2>
    {day_bars or "<p>No dated runs yet.</p>"}
  </section>

  <section class="panel">
    <h2>Run Table</h2>
    <table>
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Subcommand</th>
          <th>Status</th>
          <th>Exit</th>
          <th>Started (UTC)</th>
          <th>Duration</th>
          <th>Benchmark</th>
          <th>Adapter</th>
          <th>Steps</th>
          <th>Failed Steps</th>
          <th>Duration Bar</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </section>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a static HTML dashboard from ml_workflow run manifests.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Directory containing <run_id>/manifest.json files.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="Path to docs/run_history.md (optional enrichment).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output HTML path.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser after writing.")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    history = args.history.resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    history_rows = _parse_history_table(history)
    runs = _load_runs(runs_dir, history_rows)
    html_text = _render_html(runs, datetime.now(timezone.utc))
    out.write_text(html_text, encoding="utf-8")

    print(f"Wrote dashboard: {out}")
    print(f"Runs included: {len(runs)}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
