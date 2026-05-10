#!/usr/bin/env python3
"""
Build a static **lab optimization** dashboard committed under ``lab_dashboard/``.

Reads:
  - ``docs/run_history.md`` (always committed; works without local manifest trees)
  - ``lab_dashboard/cursor_usage.jsonl`` (optional; manual billing / usage estimates per line)
  - ``lab_dashboard/agent_events.jsonl`` (written by ``scripts/fe_ml_lab_runner.py``)
  - ``lab_dashboard/cursor_hook_events.jsonl`` (optional; appended by Cursor ``stop`` hook when enabled)

Writes:
  - ``lab_dashboard/index.html``

Deploy ``lab_dashboard/`` to any static host (GitHub Pages, Netlify, S3).

Usage:
  python scripts/build_lab_optimization_dashboard.py
  python scripts/build_lab_optimization_dashboard.py --cursor-usage /tmp/ledger.jsonl --agent-events /tmp/agent.jsonl --cursor-hooks /tmp/stop.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = REPO / "docs" / "run_history.md"
DEFAULT_OUT_DIR = REPO / "lab_dashboard"
CURSOR_USAGE = DEFAULT_OUT_DIR / "cursor_usage.jsonl"
AGENT_EVENTS = DEFAULT_OUT_DIR / "agent_events.jsonl"
CURSOR_HOOK_EVENTS = DEFAULT_OUT_DIR / "cursor_hook_events.jsonl"


@dataclass
class HistoryRow:
    utc_iso: str
    subcommand: str
    exit_code: str
    status: str
    train_iters: str
    benchmark: str
    artifacts: str


def _parse_iso_dt(value: str) -> Optional[datetime]:
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_run_history(history_path: Path) -> List[HistoryRow]:
    if not history_path.is_file():
        return []
    rows: List[HistoryRow] = []
    for raw in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("|") or "`benchmarks/results/runs/" not in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 9:
            continue
        if parts[0] == "UTC ISO" or set(parts[0]) == {"-"}:
            continue
        utc = parts[0].strip("` ")
        subcmd = parts[1].strip("` ")
        exit_c = parts[2].strip("` ")
        status = parts[3].strip("` ")
        train_iters = parts[4].strip("` ")
        bench = parts[5].strip("` ")
        artifacts = parts[8].strip("` ")
        rows.append(
            HistoryRow(
                utc_iso=utc,
                subcommand=subcmd,
                exit_code=exit_c,
                status=status,
                train_iters=train_iters,
                benchmark=bench,
                artifacts=artifacts,
            )
        )
    rows.sort(
        key=lambda r: (_parse_iso_dt(r.utc_iso) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    # Chronological append; newest at bottom → reverse for display.
    out.reverse()
    return out


def _sum_cursor_usd(lines: Iterable[Dict[str, Any]]) -> Tuple[float, int]:
    total = 0.0
    n = 0
    for row in lines:
        v = row.get("usd")
        try:
            if v is None:
                continue
            total += float(v)
            n += 1
        except (TypeError, ValueError):
            continue
    return total, n


def _render_table(rows: List[HistoryRow], max_rows: int) -> str:
    trimmed = rows[:max_rows]
    cells = []
    for r in trimmed:
        st = r.status.lower()
        cls = "ok" if "ok" in st and "fail" not in st else "failed"
        cells.append(
            "<tr>"
            f"<td>{html.escape(r.utc_iso)}</td>"
            f"<td><code>{html.escape(r.subcommand)}</code></td>"
            f"<td>{html.escape(r.exit_code)}</td>"
            f'<td><span class="pill {cls}">{html.escape(r.status)}</span></td>'
            f"<td>{html.escape(r.train_iters)}</td>"
            f"<td>{html.escape(r.benchmark)}</td>"
            f"<td><code>{html.escape(r.artifacts)}</code></td>"
            "</tr>"
        )
    if not cells:
        return "<tbody><tr><td colspan='7'>No parsed rows yet — run <code>ml_workflow.py</code> once.</td></tr></tbody>"
    return (
        "<thead><tr>"
        "<th>UTC</th><th>Subcommand</th><th>Exit</th><th>Status</th>"
        "<th>Train iters</th><th>Benchmark</th><th>Artifacts</th>"
        "</tr></thead><tbody>"
        + "".join(cells)
        + "</tbody>"
    )


def _render_jsonl_kv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No events yet.</p>"
    chunks = []
    for row in rows:
        pretty = json.dumps(row, ensure_ascii=False, indent=2)
        chunks.append(f"<pre>{html.escape(pretty)}</pre>")
    return "\n".join(chunks[:40])


def _render_html(
    args: argparse.Namespace,
    *,
    cursor_usage_path: Optional[Path] = None,
    agent_events_path: Optional[Path] = None,
    cursor_hooks_path: Optional[Path] = None,
) -> str:
    cursor_path = CURSOR_USAGE if cursor_usage_path is None else cursor_usage_path
    agent_path = AGENT_EVENTS if agent_events_path is None else agent_events_path
    hook_path = CURSOR_HOOK_EVENTS if cursor_hooks_path is None else cursor_hooks_path
    history_rows = _parse_run_history(args.history.resolve())
    cursor_lines = _read_jsonl(cursor_path)
    spared_total, spared_n = _sum_cursor_usd(cursor_lines)

    agents = _read_jsonl(agent_path)
    hooks = _read_jsonl(hook_path)
    spared_runbook = round(sum(float(x.get("est_cursor_usd_spared") or 0.0) for x in agents if x.get("kind") == "ml_workflow_learning"), 4)

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FE MLX Lab · optimization</title>
  <style>
    body {{
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:28px;color:#0f172a;background:#f8fafc;line-height:1.45;
    }}
    header {{ margin-bottom:18px }}
    .grid {{ display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }}
    .card {{ background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px; }}
    .k {{ font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin:0 0 4px }}
    .v {{ font-size:26px;margin:0;font-weight:780 }}
    h2 {{ font-size:16px;margin:22px 0 8px;color:#334155 }}
    table {{ border-collapse:collapse;width:100%;font-size:13px;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0; }}
    th,td {{ border-bottom:1px solid #e2e8f0;padding:8px;text-align:left;vertical-align:top }}
    thead th {{ background:#f8fafc;position:sticky;top:0;font-weight:600;color:#475569 }}
    code,pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px }}
    pre {{ overflow:auto;background:#f1f5f9;padding:12px;border-radius:8px;border:1px solid #e2e8f0 }}
    .pill {{ padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700;text-transform:uppercase }}
    .pill.ok {{ background:#dcfce7;color:#15803d }}
    .pill.failed {{ background:#fee2e2;color:#b91c1c }}
    .meta {{ color:#64748b;font-size:14px;margin:8px 0 0 }}
  </style>
</head>
<body>
  <header>
    <h1 style="margin:0 0 6px;font-size:22px;">Fallen Empire MLX Lab</h1>
    <div class="meta">Generated UTC <strong>{html.escape(gen)}</strong>.
      Local manifests live under <code>benchmarks/results/runs/</code> (often gitignored);
      committed <code>docs/run_history.md</code> is the portable backbone for Pages deploys.</div>
  </header>

  <section class="grid">
    <div class="card">
      <div class="k">Workflow rows (committed)</div>
      <div class="v">{len(history_rows)}</div>
      <div class="meta">Rows parsed from docs/run_history.md</div>
    </div>
    <div class="card">
      <div class="k">Cursor usage ledger (usd)</div>
      <div class="v">${spared_total:.2f}</div>
      <div class="meta">{spared_n} entries in cursor_usage.jsonl</div>
    </div>
    <div class="card">
      <div class="k">Heuristic lab-runner savings</div>
      <div class="v">${spared_runbook:.2f}</div>
      <div class="meta">Sum est_cursor_usd_spared on ml_workflow_learning events</div>
    </div>
    <div class="card">
      <div class="k">Cursor stop-hook pings</div>
      <div class="v">{len(hooks)}</div>
      <div class="meta">Requires FE_LAB_CURSOR_HOOK_APPEND=1</div>
    </div>
  </section>

  <section>
    <h2>What this dashboard is</h2>
    <p>This is <strong>not</strong> a live Cursor telemetry feed — it merges <strong>committed workflow history</strong>
    with optional <strong>manual</strong> Cursor spend entries (<code>lab_dashboard/cursor_usage.jsonl</code>) plus
    <strong>local runner events</strong> (<code>lab_dashboard/agent_events.jsonl</code>) and optional
    <strong>Cursor <code>stop</code> hook rows</strong> (<code>lab_dashboard/cursor_hook_events.jsonl</code>) so you have a lightweight
    static page to watch trends after each push or deploy bundle.</p>
    <ul>
      <li>Run sequences through the terminal:<br/><code>.venv/bin/python scripts/fe_ml_lab_runner.py learning</code> (defaults to smoke) —
      avoids parking a long Composer session.</li>
      <li>Add a Cursor bill line:<br/><code>{{"ts":"2026-05-04T01:02:03Z","usd":12.34,"note":"Monthly invoice slice"}}</code></li>
      <li>Set <code>FE_ML_LAB_SPARED_USD</code> when kicking off learner runs if you want a heuristic “USD not spent in-chat” tally.</li>
    </ul>
  </section>

  <section>
    <h2>Recent Cursor usage ledger</h2>
    {_render_jsonl_kv(cursor_lines[:12])}
  </section>

  <section>
    <h2>Recent lab-runner events</h2>
    {_render_jsonl_kv(agents[:12])}
  </section>

  <section>
    <h2>Recent Cursor stop-hook payloads</h2>
    {_render_jsonl_kv(hooks[:12])}
  </section>

  <section style="overflow:auto;">
    <h2>Committed ml_workflow rows (latest {args.max_history_rows})</h2>
    <table>{_render_table(history_rows, args.max_history_rows)}</table>
  </section>

  <footer class="meta" style="margin-top:22px;">
    Rebuild:<br/><code>python scripts/build_lab_optimization_dashboard.py</code> · Deploy folder <code>lab_dashboard/</code>
  </footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Static optimization dashboard under lab_dashboard/.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Target directory (writes index.html).")
    parser.add_argument("--max-history-rows", type=int, default=80)
    parser.add_argument(
        "--cursor-usage",
        type=Path,
        default=None,
        help=f"Optional override for Cursor ledger JSONL (default: {CURSOR_USAGE})",
    )
    parser.add_argument(
        "--agent-events",
        type=Path,
        default=None,
        help=f"Optional override for runner events JSONL (default: {AGENT_EVENTS})",
    )
    parser.add_argument(
        "--cursor-hooks",
        type=Path,
        default=None,
        help=f"Optional override for Cursor hook JSONL (default: {CURSOR_HOOK_EVENTS})",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "index.html"
    payload = _render_html(
        args,
        cursor_usage_path=args.cursor_usage,
        agent_events_path=args.agent_events,
        cursor_hooks_path=args.cursor_hooks,
    )
    out_html.write_text(payload, encoding="utf-8")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
