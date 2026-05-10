#!/usr/bin/env python3
"""
Build a single-page HTML arena dashboard from locally materialized run artifacts:

  benchmarks/results/runs/<run_id>/arena_capability.json (+ optional manifest.json)

The ``benchmarks/results/`` tree is gitignored; regenerate after arena-acceptance runs.

Usage::
    python scripts/build_arena_dashboard.py --last 30
    python scripts/ml_workflow.py arena-dashboard --last 40
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "benchmarks" / "results" / "runs"

DEFAULT_TASK_ORDER = [
    "loading-screen-polish",
    "hud-status-summary",
    "economy-tooltip",
    "combat-risk-preview",
    "save-load-api-guard",
    "ai-planning-explanation",
]

APPLY_KPI_TASKS = ("combat-risk-preview", "ai-planning-explanation")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _short_adapter(p: str) -> str:
    p = p.strip()
    if "checkpoints/" in p:
        p = p.split("checkpoints/")[-1]
    return p if len(p) <= 52 else p[:24] + "…" + p[-24:]


def _failure_tag(ts: Dict[str, Any]) -> str:
    if ts.get("passed"):
        return "ok"
    if not ts.get("apply_ok"):
        return "apply"
    if not ts.get("tsc_ok"):
        return "tsc"
    if ts.get("exports_ok") is False:
        return "export"
    po = ts.get("preview_ok")
    if po is False:
        return "preview"
    if po is None:
        return "preview?"
    return "gate"


def _discover_arena_run_dirs(limit: Optional[int]) -> List[Path]:
    if not RUNS.is_dir():
        return []
    rows: List[Tuple[str, Path]] = []
    for child in RUNS.iterdir():
        if not child.is_dir():
            continue
        cap = child / "arena_capability.json"
        if not cap.is_file():
            continue
        man = child / "manifest.json"
        ts = ""
        if man.is_file():
            try:
                m = _read_json(man)
                ts = str(m.get("started_at") or "")
            except (OSError, json.JSONDecodeError):
                ts = ""
        # Sort key: ISO timestamp strings sort lexically with chronological order here.
        key = ts or child.name
        rows.append((key, child))
    rows.sort(key=lambda x: x[0], reverse=True)
    dirs = [r[1] for r in rows]
    if limit is not None and limit > 0:
        dirs = dirs[:limit]
    return dirs


def _load_run_bundle(run_dir: Path) -> Optional[Dict[str, Any]]:
    cap_path = run_dir / "arena_capability.json"
    if not cap_path.is_file():
        return None
    cap = _read_json(cap_path)
    man_path = run_dir / "manifest.json"
    progressive = ""
    adapter = ""
    started = ""
    wall_s = None
    if man_path.is_file():
        try:
            man = _read_json(man_path)
            ac = man.get("arena_context") or {}
            progressive = str(ac.get("progressive_context") or "")
            legacy = man.get("adapter_path") or man.get("benchmark_adapter_path") or ""
            adapter = legacy or ""
            started = str(man.get("started_at") or "")
            wall_s = man.get("elapsed_s")
            if adapter == "—":
                adapter = ""
        except (OSError, json.JSONDecodeError):
            pass
    if not adapter:
        summ = cap.get("summary_path") or ""
        adapter = ""

    summary_ref = run_dir / "arena_acceptance_summary.json"
    if summary_ref.is_file():
        try:
            s = _read_json(summary_ref)
            if not progressive:
                ac2 = s.get("arena_context") or {}
                progressive = str(ac2.get("progressive_context") or "")
            adapter = adapter or _short_adapter(str(s.get("adapter_path") or ""))
        except (OSError, json.JSONDecodeError):
            pass

    by_task = {str(row.get("task_id")): row for row in cap.get("task_scores") or []}
    return {
        "run_id": run_dir.name,
        "started": started,
        "adapter": adapter or "—",
        "progressive": progressive or "?",
        "wall_s": wall_s,
        "aci": cap.get("arena_capability_index"),
        "accepted": cap.get("accepted"),
        "tasks_n": cap.get("tasks"),
        "by_task": by_task,
    }


def _kpi_cells(by_task: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    parts = []
    for tid in APPLY_KPI_TASKS:
        r = by_task.get(tid) or {}
        sc = r.get("score")
        ok = bool(r.get("passed"))
        flag = _failure_tag(r) if sc is not None else "—"
        scs = f"{float(sc):.0f}" if isinstance(sc, (int, float)) else "?"
        parts.append(f"{scs}{'✓' if ok else ''} {flag}")
    return parts[0], parts[1]


def build_html(last_n: Optional[int]) -> Tuple[str, int]:
    run_dirs = _discover_arena_run_dirs(last_n)
    rows_data: List[Dict[str, Any]] = []
    for rd in run_dirs:
        bundle = _load_run_bundle(rd)
        if bundle:
            rows_data.append(bundle)

    headers = (
        ["run", "started (UTC)", "adapter", "prog", "pass", "ACI", "combat KPI", "ai KPI", "wall s"]
        + DEFAULT_TASK_ORDER
    )
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    ncol = len(headers)

    body_rows = []
    for bundle in rows_data:
        bt = bundle["by_task"]
        aci = bundle["aci"]
        acs = bundle["accepted"]
        tn = bundle["tasks_n"]
        pass_cell = (
            f"{acs}/{tn}" if acs is not None and tn is not None else "—"
        )
        kp1, kp2 = _kpi_cells(bt)
        ws = bundle["wall_s"]
        ws_cell = f"{ws:.0f}" if isinstance(ws, (int, float)) else ("—")

        task_cells = []
        for tid in DEFAULT_TASK_ORDER:
            r = bt.get(tid) or {}
            sc = r.get("score")
            ok = r.get("passed")
            fg = _failure_tag(r) if sc is not None else "—"
            if sc is None:
                task_cells.append("<td>—</td>")
            else:
                mark = "✓" if ok else ""
                bg = "#d9f7d9" if ok else "#f9f9f9"
                title = html.escape(
                    f"{tid} score={sc} apply={r.get('apply_ok')} "
                    f"tsc={r.get('tsc_ok')} reason={r.get('reason') or ''}"[:380]
                )
                task_cells.append(
                    f"<td title='{title}' style='background:{bg}'>{sc:.1f}<small>{mark}</small>"
                    f"<br/><small style='opacity:0.8'>{fg}</small></td>"
                )

        rid = html.escape(bundle["run_id"])
        body_rows.append(
            "<tr>"
            f"<td><code>{rid}</code></td>"
            f"<td>{html.escape(bundle['started'][:19] if bundle['started'] else '')}</td>"
            f"<td><abbr title=\"{html.escape(bundle['adapter'])}\">{html.escape(_short_adapter(bundle['adapter']))}</abbr></td>"
            f"<td>{html.escape(bundle['progressive'])}</td>"
            f"<td>{pass_cell}</td>"
            f"<td>{aci if isinstance(aci, (int, float)) else '—'}</td>"
            f"<td style='font-size:75%'><code>{html.escape(kp1)}</code></td>"
            f"<td style='font-size:75%'><code>{html.escape(kp2)}</code></td>"
            f"<td>{ws_cell}</td>"
            + "".join(task_cells)
            + "</tr>"
        )

    body = "".join(body_rows) or (
        f"<tr><td colspan='{ncol}'>No <code>arena_capability.json</code> files under "
        f"<code>{html.escape(str(RUNS))}</code>. Run <code>python scripts/ml_workflow.py "
        "arena-acceptance …</code> first.</td></tr>"
    )
    css = """
    body { font-family: system-ui, sans-serif; margin: 1rem 1.25rem 3rem; color: #1a1a1a; }
    table { border-collapse: collapse; width: 100%; overflow: auto; display: block; max-width: 100%; }
    thead th { position: sticky; top: 0; background: #eaeaea; z-index: 1; padding: 0.35rem 0.4rem; border: 1px solid #ccc; font-weight: 600; font-size: 0.75rem; text-align: left; }
    td { padding: 0.25rem 0.35rem; border: 1px solid #ddd; font-size: 0.76rem; vertical-align: top; }
    code { font-size: 0.72rem; }
    small { font-size: 0.68rem; }
    h1 { font-size: 1.05rem; }
    .meta { color: #444; margin-bottom: 0.75rem; max-width: 52rem; }
    """
    out_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Arena dashboard</title><style>{css}</style></head>
<body>
<h1>Arena acceptance dashboard</h1>
<p class="meta">Generated from <code>benchmarks/results/runs/*/arena_capability.json</code>.
Rows are newest-first. Task cells show score plus pass mark and failure tag
(<strong>apply</strong> first when patch did not apply; <strong>tsc</strong>/<strong>export</strong>/<strong>preview</strong> afterward).
<code>combat KPI</code> / <code>ai KPI</code> show score plus failure tag for the two primary apply-tracking tasks.</p>
<table>
<thead><tr>{thead}</tr></thead>
<tbody>{body}</tbody>
</table>
</body>
</html>
"""
    return out_html, len(rows_data)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build benchmarks/results/arena_dashboard.html")
    ap.add_argument("--last", type=int, default=35, metavar="N", help="Maximum runs to include (default 35)")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks" / "results" / "arena_dashboard.html",
        help="Output HTML path",
    )
    ns = ap.parse_args(argv)

    html_out, nrows = build_html(ns.last if ns.last > 0 else None)
    out = ns.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({nrows} arena runs in table; --last={ns.last})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
