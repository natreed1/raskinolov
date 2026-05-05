#!/usr/bin/env python3
"""
Built-in ML pipeline orchestrator: export → LoRA JSONL → train → benchmark,
with **per-run documentation** under `benchmarks/results/runs/<run_id>/` plus an
append-only row in **committed** `docs/run_history.md`.

Subcommands
  prepare   SOURCE_REPO export + build_lora_dataset (no training).
  train     mlx_lm.lora --train (expects default lineage game-text JSONL; see docs/DATA_LAYOUT.md).
  benchmark Run run_game_benchmark.py (--adapter-path optional; --profile game or general).
  evalplus  Run execution-based EvalPlus benchmark (--suite humaneval|mbpp).
  full      prepare + train + benchmark using the new adapter path.
  smoke     Synthetic data, a few train iters, benchmark base model (CI-friendly).
  arena-acceptance  Run deterministic apply/compile/export/preview arena acceptance.
  arena-dashboard  Regenerate benchmarks/results/arena_dashboard.html from local arena run artifacts.
  arena-gate-train  Train on arena pairwise chat JSONL, run scripts/run_arena_gate_benchmark.py, repeat until gate passes or max cycles.
  adapter-registry  Write/inspect adapter registry v1.
  adapter-datasets  Build per-adapter datasets with shared anti-overfit anchor rows.
  loading-screen-dataset  Build richer loading-screen specialist dataset (core + UI transfer).
  economy-tooltip-dataset Build richer economy-tooltip specialist dataset (TSC-focused curator shards).
  combat-risk-dataset Build combat-risk specialist dataset (mirrors economy TSC recipe).
  hud-status-dataset    Build HUD specialist dataset (baseline shards, guardrails, filtered pairwise; v2 alignment).
  documentation-dataset Build documentation-steward specialist JSONL (canonical paths, SESSION_LOG/run_history discipline).
  documentation-rag-benchmark Run documentation-agent string tasks with corpus RAG (+ optional no-RAG baseline); appends aggregate telemetry JSONL.
  adapter-gate      Evaluate per-adapter promotion gates (avg-gain objective + rollback guards).
  drift-check       Run drift checks and emit automated action suggestions.
  control-plane-schedule  Run multi-Mac worker heartbeat/capability scheduler scaffolding.
  multi-adapter-report    Publish routing + adapter quality + SLO report artifacts.
  routing-prompt-lab      Capture live/batch prompts with predicted adapter + legacy route labels.
  routing-benchmark       Evaluate routing on route-only or dual adapter+route labels.
  routing-dataset         Build deterministic train/valid/test routing classifier dataset.

Every invocation except **arena-dashboard** creates:
  benchmarks/results/runs/<run_id>/manifest.json
  benchmarks/results/runs/<run_id>/RUN.md
  benchmarks/results/runs/<run_id>/logs/*.log

**And appends `docs/run_history.md`** for every invocation **except `arena-dashboard`**, which skips new run artifacts entirely (see `docs/ARENA_ROADMAP.md`). The **`documentation-rag-benchmark`** subcommand also appends **`docs/SPECIALIZED_RUN_HISTORY.md`** and a machine line to **`benchmarks/results/documentation_rag_timeseries.jsonl`**.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fe_lineage as _fe

REPO = Path(__file__).resolve().parent.parent
RUNS_PARENT = REPO / "benchmarks" / "results" / "runs"
HISTORY_PATH = REPO / "docs" / "run_history.md"
SPECIALIZED_HISTORY_PATH = REPO / "docs" / "SPECIALIZED_RUN_HISTORY.md"
DOC_RAG_TIMESERIES_PATH = REPO / "benchmarks" / "results" / "documentation_rag_timeseries.jsonl"
_DEFAULT_DOC_AGENT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
VENV_BIN = REPO / ".venv" / "bin"


def _ensure_repo_venv() -> None:
    """Run the workflow with the repo venv when it exists.

    Most child steps are Python modules that need mlx-lm/evalplus installed. If a
    user launches this script with system Python, re-exec into `.venv/bin/python`
    so the entire workflow uses one toolchain.
    """
    expected = VENV_BIN / "python"
    if not expected.is_file():
        return
    current = Path(sys.executable).resolve()
    target = expected.resolve()
    if current == target:
        return
    if os.environ.get("FE_LORA_WORKFLOW_REEXEC") == str(target):
        raise SystemExit(
            f"Refusing repeated workflow re-exec; expected venv interpreter at {target}"
        )
    env = {**os.environ, "FE_LORA_WORKFLOW_REEXEC": str(target)}
    print(f"Re-executing workflow with repo venv: {target}", file=sys.stderr)
    os.execve(str(target), [str(target), *sys.argv], env)


def _venv_exe(name: str) -> str:
    p = VENV_BIN / name
    return str(p) if p.is_file() else name


def _utc_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _git_short() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pkg_ver(dist: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(dist)
    except Exception:
        return "unknown"


def _new_run_dir(run_id: str) -> Path:
    d = RUNS_PARENT / run_id
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def _run_cmd(
    run_dir: Path,
    step: str,
    argv: List[str],
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    log_path = run_dir / "logs" / f"{step}.log"
    t0 = time.perf_counter()
    merged = {**os.environ, **(env or {})}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(argv)}\n\n")
        log.flush()
        p = subprocess.run(argv, cwd=str(REPO), env=merged, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    return p.returncode, elapsed


def _parse_benchmark_summary(log_text: str) -> str:
    m = re.search(r"=== Summary: (\d+)/(\d+)", log_text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "—"


def _parse_routing_overall_summary(log_text: str) -> str:
    m = re.search(r"=== Overall: (\d+)/(\d+)", log_text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "—"


def _parse_capability_summary(log_text: str) -> str:
    m = re.search(r"=== Capability Index: ([0-9.]+/100.*?) ===", log_text)
    return m.group(1) if m else "—"


def _parse_arena_capability_summary(log_text: str) -> str:
    m = re.search(r"=== Arena Capability Index: ([0-9.]+/100.*?) ===", log_text)
    return f"arena {m.group(1)}" if m else "—"


def _extract_training_trajectory(run_dir: Path, log_path: Path) -> Dict[str, Any]:
    """Parse mlx-lm LoRA logs into a structured trajectory for later controllers."""
    if not log_path.is_file():
        return {}

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    by_iter: Dict[int, Dict[str, Any]] = {}

    def point(iteration: int) -> Dict[str, Any]:
        if iteration not in by_iter:
            by_iter[iteration] = {"iter": iteration}
        return by_iter[iteration]

    for m in re.finditer(r"Starting training\.\.\., iters: (\d+)", log_text):
        total_iters = int(m.group(1))
        point(0)["planned_iters"] = total_iters

    for m in re.finditer(
        r"Iter (\d+): Val loss ([0-9.]+), Val took ([0-9.]+)s",
        log_text,
    ):
        p = point(int(m.group(1)))
        p["val_loss"] = float(m.group(2))
        p["val_elapsed_s"] = float(m.group(3))

    for m in re.finditer(
        r"Iter (\d+): Train loss ([0-9.]+), Learning Rate ([0-9.eE+-]+), "
        r"It/sec ([0-9.]+), Tokens/sec ([0-9.]+), Trained Tokens (\d+), "
        r"Peak mem ([0-9.]+) GB",
        log_text,
    ):
        p = point(int(m.group(1)))
        p["train_loss"] = float(m.group(2))
        p["learning_rate"] = float(m.group(3))
        p["iters_per_sec"] = float(m.group(4))
        p["tokens_per_sec"] = float(m.group(5))
        p["trained_tokens"] = int(m.group(6))
        p["peak_mem_gb"] = float(m.group(7))

    for m in re.finditer(r"Iter (\d+): Saved adapter weights", log_text):
        point(int(m.group(1)))["saved_checkpoint"] = True

    rows = [by_iter[i] for i in sorted(by_iter) if i != 0 or len(by_iter[i]) > 1]
    if rows:
        trajectory_path = run_dir / "training_trajectory.jsonl"
        with trajectory_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    train_rows = [r for r in rows if "train_loss" in r]
    val_rows = [r for r in rows if "val_loss" in r]
    lr_values = sorted({r["learning_rate"] for r in train_rows if "learning_rate" in r})
    summary: Dict[str, Any] = {
        "trajectory_path": str(run_dir / "training_trajectory.jsonl") if rows else None,
        "points": len(rows),
        "train_points": len(train_rows),
        "val_points": len(val_rows),
        "warning_truncated_sequences": log_text.count("[WARNING] Some sequences are longer"),
    }
    if train_rows:
        summary["final_train_loss"] = train_rows[-1]["train_loss"]
        summary["final_train_iter"] = train_rows[-1]["iter"]
        summary["final_learning_rate"] = train_rows[-1].get("learning_rate")
        summary["peak_mem_gb"] = max(r.get("peak_mem_gb", 0.0) for r in train_rows)
        summary["trained_tokens"] = train_rows[-1].get("trained_tokens")
    if val_rows:
        best_val = min(val_rows, key=lambda r: r["val_loss"])
        summary["final_val_loss"] = val_rows[-1]["val_loss"]
        summary["final_val_iter"] = val_rows[-1]["iter"]
        summary["best_val_loss"] = best_val["val_loss"]
        summary["best_val_iter"] = best_val["iter"]
    if lr_values:
        summary["learning_rates"] = lr_values
    return summary


def _write_run_md(
    run_dir: Path,
    run_id: str,
    command: str,
    steps: List[Dict[str, Any]],
    benchmark_summary: str,
    capability_summary: str,
    trained_adapter_path: str,
    benchmark_adapter_path: str,
    final_exit_code: int,
    started_at: datetime,
    finished_at: datetime,
    elapsed_s: float,
) -> None:
    lines = [
        f"# ML workflow run `{run_id}`",
        "",
        f"- **Command:** `{command}`",
        f"- **Final exit code:** `{final_exit_code}`",
        f"- **Started UTC:** `{started_at.isoformat()}`",
        f"- **Finished UTC:** `{finished_at.isoformat()}`",
        f"- **Duration:** `{elapsed_s:.2f}s`",
        f"- **Git:** `{_git_short()}`",
        f"- **Python:** `{sys.version.split()[0]}`",
        f"- **mlx-lm:** `{_pkg_ver('mlx-lm')}`",
        f"- **mlx:** `{_pkg_ver('mlx')}`",
        f"- **Benchmark summary:** `{benchmark_summary}`",
        f"- **Capability index:** `{capability_summary}`",
        f"- **Trained adapter (if any):** `{trained_adapter_path or '—'}`",
        f"- **Benchmarked adapter/model:** `{benchmark_adapter_path or '—'}`",
        "",
        "## Steps",
        "",
        "| Step | Exit | Seconds | Log |",
        "|------|------|---------|-----|",
    ]
    for s in steps:
        rel = s["log"].replace(str(REPO) + "/", "")
        lines.append(
            f"| {s['name']} | {s['exit_code']} | {s['elapsed_s']:.2f} | `{rel}` |"
        )
    trajectory_steps = [s for s in steps if s.get("training_trajectory", {}).get("points")]
    if trajectory_steps:
        lines += ["", "## Training Trajectory", ""]
        for s in trajectory_steps:
            t = s["training_trajectory"]
            rel = (t.get("trajectory_path") or "").replace(str(REPO) + "/", "")
            final_train = t.get("final_train_loss", "—")
            final_val = t.get("final_val_loss", "—")
            best_val = t.get("best_val_loss", "—")
            best_iter = t.get("best_val_iter", "—")
            lines.append(
                f"- `{rel}`: {t['points']} points; final train `{final_train}`, "
                f"final val `{final_val}`, best val `{best_val}` at iter `{best_iter}`."
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("Artifacts are under `benchmarks/results/` (gitignored). Copy `manifest.json` elsewhere if you need to archive a run.")
    (run_dir / "RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_history_row(
    run_id: str,
    sub: str,
    final_exit_code: int,
    train_iters: str,
    bench: str,
    trained_adapter: str,
    benchmark_adapter: str,
) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "ok" if final_exit_code == 0 else "failed"
    row = (
        f"| {iso} | `{sub}` | {final_exit_code} | {status} | {train_iters} | {bench} | "
        f"`{trained_adapter or '—'}` | `{benchmark_adapter or '—'}` | "
        f"`benchmarks/results/runs/{run_id}/` |\n"
    )
    if not HISTORY_PATH.is_file():
        HISTORY_PATH.write_text(
            "# ML workflow run history\n\n"
            "One row per `python scripts/ml_workflow.py …` run. **Full logs**, `manifest.json`, and `RUN.md` for each run live on disk under the **Artifacts** column path (that tree is gitignored).\n\n"
            "| UTC ISO | Subcommand | Exit | Status | Train iters | Benchmark | Trained adapter | Benchmarked adapter/model | Artifacts |\n"
            "|---------|------------|-----:|--------|-------------|-----------|-----------------|---------------------------|-----------|\n",
            encoding="utf-8",
        )
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(row)


def _count_doc_benchmark_jsonl(path: Path) -> Tuple[int, int]:
    """Return (passed, total) task counts from documentation-agent benchmark JSONL."""
    if not path.is_file():
        return 0, 0
    passed = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            if json.loads(line).get("passed"):
                passed += 1
        except json.JSONDecodeError:
            continue
    return passed, total


def _append_specialized_history_row(
    utc_iso: str,
    track: str,
    entry_point: str,
    exit_code: int,
    summary: str,
    artifacts: str,
    notes: str,
) -> None:
    """Append one row to docs/SPECIALIZED_RUN_HISTORY.md (auxiliary / cross-system runs)."""
    SPECIALIZED_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    status = "ok" if exit_code == 0 else "failed"
    row = (
        f"| {utc_iso} | {track} | `{entry_point}` | {exit_code} | {status} | {summary} | "
        f"{artifacts} | {notes} |\n"
    )
    if not SPECIALIZED_HISTORY_PATH.is_file():
        SPECIALIZED_HISTORY_PATH.write_text(
            "# Specialized run history (auxiliary benchmarks)\n\n"
            "Append-only index for **cross-cutting** or **standalone** evaluators that are **not** the primary "
            "game/LoRA benchmark loop. Complements **`docs/run_history.md`** (every `ml_workflow.py` run): "
            "when an orchestrated subcommand also produces domain-specific artifacts, link both tables via "
            "**`ml_workflow_run_id`** in the Notes column.\n\n"
            "Typical entries: documentation-agent RAG string tasks, future standalone retrieval evals, "
            "external CI jobs mirrored here for traceability.\n\n"
            "| UTC ISO | Track | Entry point | Exit | Status | Summary | Artifacts | Notes |\n"
            "|---------|-------|-------------|-----:|--------|---------|-----------|-------|\n",
            encoding="utf-8",
        )
    with SPECIALIZED_HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(row)


def _append_documentation_rag_timeseries(record: Dict[str, Any]) -> None:
    """Append one machine-readable summary line for trend analysis (gitignored parent)."""
    DOC_RAG_TIMESERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with DOC_RAG_TIMESERIES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _cmd_prepare(run_dir: Path, steps: List[Dict[str, Any]]) -> int:
    py = sys.executable
    argv_exp = [py, str(REPO / "scripts" / "export_repo_for_training.py")]
    code, elapsed = _run_cmd(run_dir, "export", argv_exp)
    steps.append(
        {
            "name": "export",
            "argv": argv_exp,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "export.log"),
        }
    )
    if code != 0:
        return code
    argv_b = [
        py,
        str(REPO / "scripts" / "build_lora_dataset.py"),
        "--out-dir",
        _fe.GAME_TEXT_DIR_RELPATH,
    ]
    code, elapsed = _run_cmd(run_dir, "build_lora_dataset", argv_b)
    steps.append(
        {
            "name": "build_lora_dataset",
            "argv": argv_b,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "build_lora_dataset.log"),
        }
    )
    return code


def _cmd_train(run_dir: Path, steps: List[Dict[str, Any]], extra: List[str], adapter_path: str) -> int:
    exe = _venv_exe("mlx_lm.lora")
    argv = [
        exe,
        "--train",
        "-c",
        _fe.LORA_CONFIG_RELPATH,
        "--adapter-path",
        adapter_path,
        *extra,
    ]
    code, elapsed = _run_cmd(run_dir, "mlx_lm_lora_train", argv)
    log_path = run_dir / "logs" / "mlx_lm_lora_train.log"
    trajectory = _extract_training_trajectory(run_dir, log_path)
    steps.append(
        {
            "name": "mlx_lm_lora_train",
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(log_path),
            "training_trajectory": trajectory,
        }
    )
    return code


def _cmd_benchmark(
    run_dir: Path, steps: List[Dict[str, Any]], adapter: Optional[str], profile: str = "game"
) -> Tuple[int, str]:
    py = sys.executable
    argv = [py, str(REPO / "scripts" / "run_game_benchmark.py"), "--profile", profile]
    if adapter:
        argv += ["--adapter-path", adapter]
    code, elapsed = _run_cmd(run_dir, "run_game_benchmark", argv)
    log_text = (run_dir / "logs" / "run_game_benchmark.log").read_text(encoding="utf-8", errors="replace")
    summary = _parse_benchmark_summary(log_text)
    capability = _parse_capability_summary(log_text)
    steps.append(
        {
            "name": "run_game_benchmark",
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "run_game_benchmark.log"),
            "benchmark_summary": summary,
            "capability_summary": capability,
        }
    )
    return code, summary


def _cmd_evalplus(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    adapter: Optional[str],
    suite: str,
    limit: Optional[int],
    full: bool,
    max_tokens: int,
    temp: float,
    base_only: bool,
) -> Tuple[int, str]:
    py = sys.executable
    output_jsonl = run_dir / "evalplus_results.jsonl"
    argv = [
        py,
        str(REPO / "scripts" / "run_evalplus_benchmark.py"),
        "--suite",
        suite,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temp),
        "--output-jsonl",
        str(output_jsonl),
    ]
    if adapter:
        argv += ["--adapter-path", adapter]
    if full:
        argv.append("--full")
    elif limit is not None:
        argv += ["--limit", str(limit)]
    if base_only:
        argv.append("--base-only")

    code, elapsed = _run_cmd(run_dir, "run_evalplus_benchmark", argv)
    log_text = (run_dir / "logs" / "run_evalplus_benchmark.log").read_text(
        encoding="utf-8", errors="replace"
    )
    summary = _parse_benchmark_summary(log_text)
    steps.append(
        {
            "name": "run_evalplus_benchmark",
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "run_evalplus_benchmark.log"),
            "benchmark_summary": summary,
            "results": str(output_jsonl),
        }
    )
    return code, summary


def _cmd_tool(run_dir: Path, steps: List[Dict[str, Any]], step_name: str, argv: List[str]) -> int:
    code, elapsed = _run_cmd(run_dir, step_name, argv)
    steps.append(
        {
            "name": step_name,
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / f"{step_name}.log"),
        }
    )
    return code


def _cmd_build_game_task_pairwise(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    input_path: Path,
    out_dir: Path,
    repeat: int,
    focus_apply_failures: bool = False,
) -> int:
    py = sys.executable
    argv = [
        py,
        str(REPO / "scripts" / "build_game_task_pairwise_dataset.py"),
        "--input",
        str(input_path),
        "--out-dir",
        str(out_dir),
        "--repeat",
        str(repeat),
    ]
    if focus_apply_failures:
        argv.append("--focus-apply-failures")
    code, elapsed = _run_cmd(run_dir, "build_game_task_pairwise_dataset", argv)
    steps.append(
        {
            "name": "build_game_task_pairwise_dataset",
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "build_game_task_pairwise_dataset.log"),
        }
    )
    return code


def _cmd_arena_gate_benchmark(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    step_key: str,
    adapter_path: str,
    gate: str,
    task_ids: List[str],
    tasks_json: Path,
    source_repo: Path,
    worktree_root: Path,
) -> Tuple[int, Dict[str, Any]]:
    py = sys.executable
    summary_path = run_dir / f"{step_key}_summary.json"
    argv = [
        py,
        str(REPO / "scripts" / "run_arena_gate_benchmark.py"),
        "--adapter-path",
        adapter_path,
        "--gate",
        gate,
        "--source-repo",
        str(source_repo),
        "--worktree-root",
        str(worktree_root),
        "--tasks",
        str(tasks_json),
        "--write-summary",
        str(summary_path),
        "--max-tokens",
        "8192",
    ]
    for tid in task_ids:
        argv.extend(["--task-id", tid])
    code, elapsed = _run_cmd(run_dir, step_key, argv)
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    steps.append(
        {
            "name": step_key,
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / f"{step_key}.log"),
            "arena_gate_summary": summary,
        }
    )
    return code, summary


def _cmd_arena_acceptance(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    adapter_path: str,
    task_ids: List[str],
    tasks_json: Path,
    source_repo: Path,
    worktree_root: Path,
    context_chars: int,
    max_tokens: int,
    tsc_retries: int,
    preview_port: int,
    timeout_s: int,
    model: Optional[str],
    no_cleanup: bool,
    progressive_context: str,
    suite: str,
) -> Tuple[int, str, str]:
    py = sys.executable
    summary_path = run_dir / "arena_acceptance_summary.json"
    argv = [
        py,
        str(REPO / "scripts" / "run_arena_acceptance_tests.py"),
        "--adapter-path",
        adapter_path,
        "--source-repo",
        str(source_repo),
        "--worktree-root",
        str(worktree_root),
        "--tasks",
        str(tasks_json),
        "--context-chars",
        str(context_chars),
        "--max-tokens",
        str(max_tokens),
        "--tsc-retries",
        str(tsc_retries),
        "--preview-port",
        str(preview_port),
        "--timeout-s",
        str(timeout_s),
        "--summary",
        str(summary_path),
    ]
    if model:
        argv += ["--model", model]
    argv += ["--suite", suite]
    if no_cleanup:
        argv.append("--no-cleanup")
    for tid in task_ids:
        argv.extend(["--task-id", tid])
    argv.extend(["--progressive-context", progressive_context])

    code, elapsed = _run_cmd(run_dir, "run_arena_acceptance_tests", argv)
    log_text = (run_dir / "logs" / "run_arena_acceptance_tests.log").read_text(
        encoding="utf-8", errors="replace"
    )
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    accepted = summary.get("passed_count", 0)
    tasks = summary.get("tasks", len(task_ids))
    bench = f"arena {accepted}/{tasks}"
    capability = _parse_arena_capability_summary(log_text)
    if capability == "—":
        cap_data = summary.get("arena_capability") or {}
        if cap_data:
            capability = (
                f"arena {cap_data.get('arena_capability_index', 0):.1f}/100 "
                f"(accepted {cap_data.get('accepted', 0)}/{cap_data.get('tasks', tasks)}, "
                f"completion {cap_data.get('completion', 0):.1f}, "
                f"integration {cap_data.get('integration', 0):.1f}, "
                f"efficiency {cap_data.get('efficiency', 0):.1f})"
            )
    steps.append(
        {
            "name": "run_arena_acceptance_tests",
            "argv": argv,
            "exit_code": code,
            "elapsed_s": elapsed,
            "log": str(run_dir / "logs" / "run_arena_acceptance_tests.log"),
            "arena_acceptance_summary": summary,
            "arena_capability_summary": capability,
        }
    )
    return code, bench, capability


def main() -> None:
    _ensure_repo_venv()

    parser = argparse.ArgumentParser(description="Orchestrated ML workflow with per-run docs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Export + build LoRA JSONL")
    p_train = sub.add_parser("train", help="Run mlx_lm.lora --train (no benchmark unless --evaluate)")
    p_train.add_argument("--adapter-path", default=_fe.DEFAULT_ADAPTER_LATEST_RELPATH)
    p_train.add_argument(
        "--evaluate",
        action="store_true",
        help="After a successful train, run run_game_benchmark.py on the new adapter",
    )
    p_train.add_argument(
        "--bench-profile",
        choices=["game", "general"],
        default="game",
        help="With --evaluate: profile passed to run_game_benchmark.py",
    )
    p_train.add_argument(
        "lora_args",
        nargs=argparse.REMAINDER,
        help="Extra args for mlx_lm.lora (use `-- --iters 200` if your shell needs `--`)",
    )

    p_bench = sub.add_parser("benchmark", help="Run heuristic benchmark suite")
    p_bench.add_argument("--adapter-path", default=None)
    p_bench.add_argument(
        "--profile",
        choices=["game", "general"],
        default="game",
        help="Forward to run_game_benchmark.py (default: game = Fallen Empire tasks).",
    )

    p_evalplus = sub.add_parser("evalplus", help="Run execution-based EvalPlus benchmark")
    p_evalplus.add_argument("--adapter-path", default=None)
    p_evalplus.add_argument("--suite", choices=["humaneval", "mbpp"], default="humaneval")
    p_evalplus.add_argument("--limit", type=int, default=5)
    p_evalplus.add_argument("--full", action="store_true", help="Run full suite instead of --limit")
    p_evalplus.add_argument("--max-tokens", type=int, default=512)
    p_evalplus.add_argument("--temp", type=float, default=0.0)
    p_evalplus.add_argument("--base-only", action="store_true")

    p_accept = sub.add_parser(
        "arena-acceptance",
        help="Run no-human arena acceptance with deterministic Arena Capability Index",
    )
    p_accept.add_argument("--adapter-path", required=True)
    p_accept.add_argument("--task-id", action="append", dest="task_ids", default=[])
    p_accept.add_argument(
        "--suite",
        choices=["core6", "all"],
        default="core6",
        help="When --task-id is omitted, run fixed core six or all tasks from --tasks-json.",
    )
    p_accept.add_argument(
        "--tasks-json",
        type=Path,
        default=REPO / "benchmarks" / "game_task_arena_examples.json",
    )
    p_accept.add_argument("--source-repo", type=Path, default=None)
    p_accept.add_argument("--worktree-root", type=Path, default=None)
    p_accept.add_argument("--context-chars", type=int, default=9000)
    p_accept.add_argument("--max-tokens", type=int, default=8192)
    p_accept.add_argument("--tsc-retries", type=int, default=1)
    p_accept.add_argument("--preview-port", type=int, default=5174)
    p_accept.add_argument(
        "--timeout-s",
        type=int,
        default=14_400,
        help="Subprocess timeout for the full gate (default 4h; six-task preview runs often need this).",
    )
    p_accept.add_argument("--model", default=None)
    p_accept.add_argument("--no-cleanup", action="store_true")
    p_accept.add_argument(
        "--progressive-context",
        choices=["auto", "on", "off"],
        default="auto",
        help="Arena generate: extra progressive file-retrieval pass (default auto). Use off for baseline A/B.",
    )

    p_full = sub.add_parser("full", help="prepare + train + benchmark")
    p_full.add_argument("--adapter-path", default=_fe.DEFAULT_ADAPTER_LATEST_RELPATH)
    p_full.add_argument(
        "--bench-profile",
        choices=["game", "general"],
        default="game",
        help="run_game_benchmark.py profile when the benchmark step runs",
    )
    p_full.add_argument("lora_args", nargs=argparse.REMAINDER, help="Extra args after -- for mlx_lm.lora")

    p_agt = sub.add_parser(
        "arena-gate-train",
        help="Train on arena pairwise chat JSONL, then headless arena gate; repeat until gate passes or max cycles",
    )
    p_agt.add_argument("--adapter-path", required=True)
    p_agt.add_argument("--data-dir", type=Path, default=REPO / "data" / "lora" / "game_task_pairwise")
    p_agt.add_argument("--max-cycles", type=int, default=20)
    p_agt.add_argument("--iters-per-cycle", type=int, default=100)
    p_agt.add_argument("--gate", choices=["metrics", "preview"], default="metrics")
    p_agt.add_argument("--task-id", action="append", dest="arena_task_ids", default=[])
    p_agt.add_argument(
        "--tasks-json",
        type=Path,
        default=REPO / "benchmarks" / "game_task_arena_examples.json",
    )
    p_agt.add_argument("--source-repo", type=Path, default=None)
    p_agt.add_argument("--worktree-root", type=Path, default=None)
    p_agt.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Each cycle: run build_game_task_pairwise_dataset.py before train (needs --pairwise-input).",
    )
    p_agt.add_argument(
        "--pairwise-input",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    p_agt.add_argument("--pairwise-repeat", type=int, default=4)
    p_agt.add_argument(
        "--focus-apply-failures",
        action="store_true",
        help="With --rebuild-dataset, keep only pairwise rows where loser_error indicates no_applyable_changes/apply_check_failed patterns.",
    )
    p_agt.add_argument(
        "lora_args",
        nargs=argparse.REMAINDER,
        help="Extra mlx_lm.lora flags after `--` (merged with per-cycle --data and --iters)",
    )

    p_registry = sub.add_parser("adapter-registry", help="Write or inspect adapter registry v1")
    p_registry.add_argument("--write-default", action="store_true")
    p_registry.add_argument("--path", type=Path, default=REPO / "training" / "adapter_registry_v1.json")

    p_datasets = sub.add_parser(
        "adapter-datasets",
        help="Build per-adapter datasets with 85/10/5 family/shared/hard-negative mix",
    )
    p_datasets.add_argument("--sources-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    p_datasets.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "shared_general_anchor")
    p_datasets.add_argument("--out-dir", type=Path, default=REPO / "data" / "lora" / "adapters")
    p_datasets.add_argument("--seed", type=int, default=42)

    p_loading_ds = sub.add_parser(
        "loading-screen-dataset",
        help="Build loading-screen specialist dataset with UI transfer rows.",
    )
    p_loading_ds.add_argument(
        "--pairwise-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    p_loading_ds.add_argument("--baselines-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    p_loading_ds.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    p_loading_ds.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "loading_screen_specialist",
    )
    p_loading_ds.add_argument("--seed", type=int, default=42)
    p_loading_ds.add_argument("--core-ratio", type=float, default=0.70)
    p_loading_ds.add_argument("--transfer-ratio", type=float, default=0.20)
    p_loading_ds.add_argument("--shared-ratio", type=float, default=0.10)
    p_loading_ds.add_argument("--max-core-rows", type=int, default=120)
    p_loading_ds.add_argument("--max-transfer-rows", type=int, default=80)
    p_loading_ds.add_argument("--min-train-core-rows", type=int, default=120)
    p_loading_ds.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=["hud-status-summary"],
    )

    p_economy_ds = sub.add_parser(
        "economy-tooltip-dataset",
        help="Build economy-tooltip specialist dataset with UI transfer rows.",
    )
    p_economy_ds.add_argument(
        "--pairwise-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    p_economy_ds.add_argument("--baselines-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    p_economy_ds.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    p_economy_ds.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "economy_tooltip_specialist",
    )
    p_economy_ds.add_argument("--seed", type=int, default=42)
    p_economy_ds.add_argument("--core-ratio", type=float, default=0.70)
    p_economy_ds.add_argument("--transfer-ratio", type=float, default=0.20)
    p_economy_ds.add_argument("--shared-ratio", type=float, default=0.10)
    p_economy_ds.add_argument(
        "--max-core-rows",
        type=int,
        default=48,
        help="Pairwise cap (curator TSC shards dominate). Lower is safer.",
    )
    p_economy_ds.add_argument(
        "--baseline-shards",
        type=int,
        default=18,
        help="Duplicate curator baseline shards (economy TSC density).",
    )
    p_economy_ds.add_argument(
        "--tsc-guard-rows",
        type=int,
        default=8,
        help="Repeat identical assistant with strict TSC-focused user prompts.",
    )
    p_economy_ds.add_argument("--max-transfer-rows", type=int, default=80)
    p_economy_ds.add_argument("--min-train-core-rows", type=int, default=100)
    p_economy_ds.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=["loading-screen-polish", "hud-status-summary"],
    )

    p_combat_ds = sub.add_parser(
        "combat-risk-dataset",
        help="Build combat-risk-preview specialist dataset (curator TSC gold + pairwise transfer).",
    )
    p_combat_ds.add_argument(
        "--pairwise-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    p_combat_ds.add_argument("--baselines-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    p_combat_ds.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    p_combat_ds.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "combat_risk_specialist",
    )
    p_combat_ds.add_argument("--seed", type=int, default=42)
    p_combat_ds.add_argument("--core-ratio", type=float, default=0.70)
    p_combat_ds.add_argument("--transfer-ratio", type=float, default=0.20)
    p_combat_ds.add_argument("--shared-ratio", type=float, default=0.10)
    p_combat_ds.add_argument(
        "--max-core-rows",
        type=int,
        default=40,
        help="Pairwise cap for combat task (baseline shards dominate).",
    )
    p_combat_ds.add_argument("--max-transfer-rows", type=int, default=80)
    p_combat_ds.add_argument("--min-train-core-rows", type=int, default=110)
    p_combat_ds.add_argument(
        "--baseline-shards",
        type=int,
        default=18,
        help="Duplicate curator baseline rows for combat TSC density.",
    )
    p_combat_ds.add_argument(
        "--tsc-guard-rows",
        type=int,
        default=8,
        help="Strict combat TSC prompts with identical curator assistant.",
    )
    p_combat_ds.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=["loading-screen-polish", "economy-tooltip", "hud-status-summary"],
    )

    p_hud_ds = sub.add_parser(
        "hud-status-dataset",
        help="Build HUD-status specialist dataset with UI transfer rows.",
    )
    p_hud_ds.add_argument(
        "--pairwise-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    p_hud_ds.add_argument("--baselines-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    p_hud_ds.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    p_hud_ds.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "hud_status_specialist",
    )
    p_hud_ds.add_argument("--seed", type=int, default=42)
    p_hud_ds.add_argument("--core-ratio", type=float, default=0.70)
    p_hud_ds.add_argument("--transfer-ratio", type=float, default=0.20)
    p_hud_ds.add_argument("--shared-ratio", type=float, default=0.10)
    p_hud_ds.add_argument(
        "--max-core-rows",
        type=int,
        default=0,
        help="HUD pairwise core cap (default 0: baseline shards + guardrails dominate).",
    )
    p_hud_ds.add_argument("--max-transfer-rows", type=int, default=80)
    p_hud_ds.add_argument("--min-train-core-rows", type=int, default=100)
    p_hud_ds.add_argument(
        "--baseline-shards",
        type=int,
        default=12,
        help="Duplicate curator baseline N times in core (see build_hud_status_specialist_dataset.py).",
    )
    p_hud_ds.add_argument(
        "--no-hud-guardrails",
        action="store_true",
        help="Pass --no-guardrails to the HUD dataset builder.",
    )
    p_hud_ds.add_argument(
        "--no-filter-pairwise-hud",
        action="store_true",
        help="Pass --no-filter-pairwise-hud to keep all HUD pairwise winners.",
    )
    p_hud_ds.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=["loading-screen-polish", "economy-tooltip"],
    )

    p_docs_ds = sub.add_parser(
        "documentation-dataset",
        help="Build documentation-specialist dataset (canonical mlx-lab prose).",
    )
    p_docs_ds.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "documentation_specialist",
    )
    p_docs_ds.add_argument("--seed", type=int, default=11)
    p_docs_ds.add_argument("--min-train-core-rows", type=int, default=120)

    p_doc_rag = sub.add_parser(
        "documentation-rag-benchmark",
        help="Documentation-agent string tasks with optional RAG + no-RAG baseline telemetry.",
    )
    p_doc_rag.add_argument(
        "--tasks",
        type=Path,
        default=REPO / "benchmarks" / "documentation_agent_rag_tasks_v1.json",
    )
    p_doc_rag.add_argument("--corpus", type=Path, default=REPO / "data" / "rag" / "documentation_agent_corpus.json")
    p_doc_rag.add_argument("--model", default=None, help=f"Default: $MODEL or {_DEFAULT_DOC_AGENT_MODEL}")
    p_doc_rag.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    p_doc_rag.add_argument(
        "--skip-no-rag-baseline",
        action="store_true",
        help="Do not run the second pass without retrieval (faster; less telemetry).",
    )
    p_doc_rag.add_argument(
        "--no-timeseries-append",
        action="store_true",
        help="Skip appending benchmarks/results/documentation_rag_timeseries.jsonl.",
    )

    p_agate = sub.add_parser("adapter-gate", help="Evaluate per-adapter promotion gate.")
    p_agate.add_argument("--candidate", type=Path, required=True)
    p_agate.add_argument("--champion", type=Path, required=True)
    p_agate.add_argument("--output", type=Path, default=None)

    p_drift = sub.add_parser("drift-check", help="Run adapter/router drift checks.")
    p_drift.add_argument("--adapter-current", type=Path, required=True)
    p_drift.add_argument("--adapter-baseline", type=Path, required=True)
    p_drift.add_argument("--router-current", type=Path, required=True)
    p_drift.add_argument("--router-baseline", type=Path, required=True)
    p_drift.add_argument("--output", type=Path, default=None)

    p_sched = sub.add_parser(
        "control-plane-schedule",
        help="Schedule jobs from worker heartbeat/capability snapshots.",
    )
    p_sched.add_argument("--workers-json", type=Path, required=True)
    p_sched.add_argument("--jobs-json", type=Path, required=True)

    p_report = sub.add_parser(
        "multi-adapter-report",
        help="Publish routing + adapter quality + SLO report artifacts.",
    )
    p_report.add_argument("--routing-jsonl", type=Path, required=True)
    p_report.add_argument("--gate-json", type=Path, required=True)
    p_report.add_argument("--drift-json", type=Path, required=True)
    p_report.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "benchmarks" / "results" / "multi_adapter_dashboard.json",
    )
    p_report.add_argument(
        "--out-md",
        type=Path,
        default=REPO / "benchmarks" / "results" / "multi_adapter_report.md",
    )

    p_route_lab = sub.add_parser(
        "routing-prompt-lab",
        help="Capture live/batch prompts with adapter + legacy-route predictions.",
    )
    p_route_lab.add_argument("--prompt", action="append", default=[])
    p_route_lab.add_argument("--prompts-file", type=Path, default=None)
    p_route_lab.add_argument("--interactive", action="store_true")
    p_route_lab.add_argument("--source", default="live")
    p_route_lab.add_argument("--expected-adapter-id", default=None)
    p_route_lab.add_argument(
        "--expected-legacy-route",
        default=None,
        choices=["local", "hybrid", "frontier"],
    )
    p_route_lab.add_argument("--accepted-for-training", action="store_true")
    p_route_lab.add_argument("--reviewer", default=None)
    p_route_lab.add_argument("--notes", default=None)
    p_route_lab.add_argument("--output-jsonl", type=Path, default=None)

    p_route_bench = sub.add_parser(
        "routing-benchmark",
        help="Run route-only or adapter+route routing benchmark.",
    )
    p_route_bench.add_argument("--tasks", type=Path, default=REPO / "benchmarks" / "task_routing_tasks.json")
    p_route_bench.add_argument("--mode", choices=["route", "adapter", "both"], default="route")
    p_route_bench.add_argument("--output-jsonl", type=Path, default=None)
    p_route_bench.add_argument("--summary-json", type=Path, default=None)

    p_route_ds = sub.add_parser(
        "routing-dataset",
        help="Build deterministic routing train/valid/test dataset from benchmark/live/curated labels.",
    )
    p_route_ds.add_argument("--benchmark-tasks", type=Path, default=REPO / "benchmarks" / "task_routing_tasks.json")
    p_route_ds.add_argument("--live-jsonl", type=Path, action="append", default=[])
    p_route_ds.add_argument("--curated-jsonl", type=Path, action="append", default=[])
    p_route_ds.add_argument("--seed", type=int, default=42)
    p_route_ds.add_argument("--dataset-version", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    p_route_ds.add_argument("--out-root", type=Path, default=REPO / "data" / "lora" / "routing_classifier")
    p_route_ds.add_argument("--low-confidence-threshold", type=float, default=0.62)

    p_smoke = sub.add_parser("smoke", help="Synthetic data + tiny train + benchmark (no export)")

    p_dash = sub.add_parser(
        "arena-dashboard",
        help="Build benchmarks/results/arena_dashboard.html from runs/*/arena_capability.json (no new run artifacts)",
    )
    p_dash.add_argument("--last", type=int, default=35, help="Latest N arena runs (default 35)")
    p_dash.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks" / "results" / "arena_dashboard.html",
        help="Output HTML path",
    )

    args = parser.parse_args()
    if args.command == "arena-dashboard":
        dash_argv = [
            sys.executable,
            str(REPO / "scripts" / "build_arena_dashboard.py"),
            "--last",
            str(args.last),
            "--out",
            str(args.out.expanduser().resolve()),
        ]
        proc = subprocess.run(dash_argv, cwd=str(REPO))
        sys.exit(proc.returncode)

    workflow_started = datetime.now(timezone.utc)
    run_id = _utc_run_id()
    run_dir = _new_run_dir(run_id)
    steps: List[Dict[str, Any]] = []
    cmdline = shlex.join(sys.argv)
    bench_summary = "—"
    capability_summary = "—"
    adapter_note = ""
    benchmark_adapter_note = "—"
    final_code = 0
    doc_rag_detail: Dict[str, Any] = {}

    if args.command == "prepare":
        final_code = _cmd_prepare(run_dir, steps)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "train":
        extra = [x for x in args.lora_args if x != "--"]
        final_code = _cmd_train(run_dir, steps, extra, args.adapter_path)
        adapter_note = args.adapter_path
        if final_code == 0 and args.evaluate:
            c, bench_summary = _cmd_benchmark(
                run_dir, steps, args.adapter_path, args.bench_profile
            )
            benchmark_adapter_note = args.adapter_path
            final_code = c
        elif not args.evaluate:
            benchmark_adapter_note = "—"

    elif args.command == "benchmark":
        c, bench_summary = _cmd_benchmark(
            run_dir, steps, args.adapter_path, args.profile
        )
        final_code = c
        adapter_note = "—"
        benchmark_adapter_note = args.adapter_path or "base"

    elif args.command == "evalplus":
        c, bench_summary = _cmd_evalplus(
            run_dir,
            steps,
            args.adapter_path,
            args.suite,
            args.limit,
            args.full,
            args.max_tokens,
            args.temp,
            args.base_only,
        )
        final_code = c
        adapter_note = "—"
        benchmark_adapter_note = args.adapter_path or "base"

    elif args.command == "arena-acceptance":
        source_repo = (
            args.source_repo
            or Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire")))
        ).expanduser().resolve()
        worktree_root = (
            args.worktree_root
            or Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena")))
        ).expanduser().resolve()
        task_ids = list(args.task_ids)
        c, bench_summary, capability_summary = _cmd_arena_acceptance(
            run_dir,
            steps,
            args.adapter_path,
            task_ids,
            args.tasks_json.expanduser().resolve(),
            source_repo,
            worktree_root,
            args.context_chars,
            args.max_tokens,
            args.tsc_retries,
            args.preview_port,
            args.timeout_s,
            args.model,
            args.no_cleanup,
            args.progressive_context,
            args.suite,
        )
        final_code = c
        adapter_note = "—"
        benchmark_adapter_note = args.adapter_path

    elif args.command == "full":
        extra = [x for x in args.lora_args if x != "--"]
        final_code = _cmd_prepare(run_dir, steps)
        if final_code == 0:
            final_code = _cmd_train(run_dir, steps, extra, args.adapter_path)
        adapter_note = args.adapter_path
        if final_code == 0:
            c, bench_summary = _cmd_benchmark(
                run_dir, steps, args.adapter_path, args.bench_profile
            )
            benchmark_adapter_note = args.adapter_path
            final_code = c

    elif args.command == "arena-gate-train":
        ag = args
        extra = [x for x in ag.lora_args if x != "--"]
        data_dir = ag.data_dir.expanduser().resolve()
        source_repo = (
            ag.source_repo
            or Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire")))
        ).expanduser().resolve()
        worktree_root = (
            ag.worktree_root
            or Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena")))
        ).expanduser().resolve()
        task_ids = list(ag.arena_task_ids) or ["loading-screen-polish"]
        tasks_json = ag.tasks_json.expanduser().resolve()
        adapter_note = ag.adapter_path
        benchmark_adapter_note = ag.adapter_path
        if ag.max_cycles < 1:
            print("--max-cycles must be >= 1", file=sys.stderr)
            final_code = 2
        elif not ag.rebuild_dataset and not (data_dir / "train.jsonl").is_file():
            print(
                f"Missing training data {data_dir / 'train.jsonl'}. "
                "Run: python scripts/build_game_task_pairwise_dataset.py "
                "or pass --rebuild-dataset (requires --pairwise-input).",
                file=sys.stderr,
            )
            final_code = 2
        else:
            won = False
            last_summary: Dict[str, Any] = {}
            for cycle in range(ag.max_cycles):
                if ag.rebuild_dataset:
                    if ag.pairwise_input.is_file():
                        br = _cmd_build_game_task_pairwise(
                            run_dir,
                            steps,
                            ag.pairwise_input,
                            data_dir,
                            ag.pairwise_repeat,
                            focus_apply_failures=ag.focus_apply_failures,
                        )
                        if br != 0:
                            final_code = br
                            break
                    else:
                        print(
                            f"--rebuild-dataset set but missing {ag.pairwise_input}",
                            file=sys.stderr,
                        )
                if not (data_dir / "train.jsonl").is_file():
                    print(
                        f"Missing {data_dir / 'train.jsonl'} after optional rebuild.",
                        file=sys.stderr,
                    )
                    final_code = 2
                    break
                train_extra = ["--data", str(data_dir), "--iters", str(ag.iters_per_cycle), *extra]
                tr = _cmd_train(run_dir, steps, train_extra, ag.adapter_path)
                if tr != 0:
                    final_code = tr
                    break
                gc, last_summary = _cmd_arena_gate_benchmark(
                    run_dir,
                    steps,
                    f"arena_gate_cycle_{cycle}",
                    ag.adapter_path,
                    ag.gate,
                    task_ids,
                    tasks_json,
                    source_repo,
                    worktree_root,
                )
                bench_summary = (
                    f"arena {last_summary.get('passed_count', 0)}/{last_summary.get('tasks', 0)} c{cycle}"
                )
                if last_summary.get("passed_all"):
                    won = True
                    bench_summary = f"arena PASS c{cycle}"
                    final_code = 0
                    break
                final_code = gc
            if not won and final_code == 0:
                final_code = 1

    elif args.command == "adapter-registry":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "taxonomy.py"),
            "--path",
            str(args.path.expanduser().resolve()),
        ]
        if args.write_default:
            argv.append("--write-default")
        final_code = _cmd_tool(run_dir, steps, "adapter_registry", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "adapter-datasets":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "dataset_builder.py"),
            "--sources-dir",
            str(args.sources_dir.expanduser().resolve()),
            "--shared-anchor-dir",
            str(args.shared_anchor_dir.expanduser().resolve()),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
        ]
        final_code = _cmd_tool(run_dir, steps, "adapter_datasets", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "loading-screen-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "build_loading_screen_specialist_dataset.py"),
            "--pairwise-jsonl",
            str(args.pairwise_jsonl.expanduser().resolve()),
            "--baselines-dir",
            str(args.baselines_dir.expanduser().resolve()),
            "--shared-anchor-dir",
            str(args.shared_anchor_dir.expanduser().resolve()),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--core-ratio",
            str(args.core_ratio),
            "--transfer-ratio",
            str(args.transfer_ratio),
            "--shared-ratio",
            str(args.shared_ratio),
            "--max-core-rows",
            str(args.max_core_rows),
            "--max-transfer-rows",
            str(args.max_transfer_rows),
            "--min-train-core-rows",
            str(args.min_train_core_rows),
        ]
        for tid in args.transfer_task_ids:
            argv.extend(["--transfer-task-id", str(tid)])
        final_code = _cmd_tool(run_dir, steps, "loading_screen_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "economy-tooltip-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "build_economy_tooltip_specialist_dataset.py"),
            "--pairwise-jsonl",
            str(args.pairwise_jsonl.expanduser().resolve()),
            "--baselines-dir",
            str(args.baselines_dir.expanduser().resolve()),
            "--shared-anchor-dir",
            str(args.shared_anchor_dir.expanduser().resolve()),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--core-ratio",
            str(args.core_ratio),
            "--transfer-ratio",
            str(args.transfer_ratio),
            "--shared-ratio",
            str(args.shared_ratio),
            "--max-core-rows",
            str(args.max_core_rows),
            "--max-transfer-rows",
            str(args.max_transfer_rows),
            "--min-train-core-rows",
            str(args.min_train_core_rows),
            "--baseline-shards",
            str(args.baseline_shards),
            "--tsc-guard-rows",
            str(args.tsc_guard_rows),
        ]
        for tid in args.transfer_task_ids:
            argv.extend(["--transfer-task-id", str(tid)])
        final_code = _cmd_tool(run_dir, steps, "economy_tooltip_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "combat-risk-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "build_combat_risk_specialist_dataset.py"),
            "--pairwise-jsonl",
            str(args.pairwise_jsonl.expanduser().resolve()),
            "--baselines-dir",
            str(args.baselines_dir.expanduser().resolve()),
            "--shared-anchor-dir",
            str(args.shared_anchor_dir.expanduser().resolve()),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--core-ratio",
            str(args.core_ratio),
            "--transfer-ratio",
            str(args.transfer_ratio),
            "--shared-ratio",
            str(args.shared_ratio),
            "--max-core-rows",
            str(args.max_core_rows),
            "--max-transfer-rows",
            str(args.max_transfer_rows),
            "--min-train-core-rows",
            str(args.min_train_core_rows),
            "--baseline-shards",
            str(args.baseline_shards),
            "--tsc-guard-rows",
            str(args.tsc_guard_rows),
        ]
        for tid in args.transfer_task_ids:
            argv.extend(["--transfer-task-id", str(tid)])
        final_code = _cmd_tool(run_dir, steps, "combat_risk_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "hud-status-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "build_hud_status_specialist_dataset.py"),
            "--pairwise-jsonl",
            str(args.pairwise_jsonl.expanduser().resolve()),
            "--baselines-dir",
            str(args.baselines_dir.expanduser().resolve()),
            "--shared-anchor-dir",
            str(args.shared_anchor_dir.expanduser().resolve()),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--core-ratio",
            str(args.core_ratio),
            "--transfer-ratio",
            str(args.transfer_ratio),
            "--shared-ratio",
            str(args.shared_ratio),
            "--max-core-rows",
            str(args.max_core_rows),
            "--max-transfer-rows",
            str(args.max_transfer_rows),
            "--min-train-core-rows",
            str(args.min_train_core_rows),
        ]
        for tid in args.transfer_task_ids:
            argv.extend(["--transfer-task-id", str(tid)])
        argv.extend(["--baseline-shards", str(args.baseline_shards)])
        if args.no_hud_guardrails:
            argv.append("--no-guardrails")
        if args.no_filter_pairwise_hud:
            argv.append("--no-filter-pairwise-hud")
        final_code = _cmd_tool(run_dir, steps, "hud_status_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "documentation-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "build_documentation_specialist_dataset.py"),
            "--out-dir",
            str(args.out_dir.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--min-train-core-rows",
            str(args.min_train_core_rows),
        ]
        final_code = _cmd_tool(run_dir, steps, "documentation_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "documentation-rag-benchmark":
        model_resolved = args.model or os.environ.get("MODEL", _DEFAULT_DOC_AGENT_MODEL)
        adapter_note = args.adapter_path or "—"
        benchmark_adapter_note = args.adapter_path if args.adapter_path else "base"
        rag_out = run_dir / "documentation_agent_rag.jsonl"
        norag_out = run_dir / "documentation_agent_no_rag.jsonl"
        argv_rag = [
            sys.executable,
            str(REPO / "scripts" / "run_documentation_agent_benchmark.py"),
            "--tasks",
            str(args.tasks.expanduser().resolve()),
            "--corpus",
            str(args.corpus.expanduser().resolve()),
            "--model",
            model_resolved,
            "--use-rag",
            "--output-jsonl",
            str(rag_out),
        ]
        if args.adapter_path:
            argv_rag.extend(
                ["--adapter-path", str(Path(args.adapter_path).expanduser().resolve())]
            )
        final_code = _cmd_tool(run_dir, steps, "documentation_agent_benchmark_rag", argv_rag)
        rp, rt = _count_doc_benchmark_jsonl(rag_out)
        np, nt = 0, 0
        if final_code == 0 and not args.skip_no_rag_baseline:
            argv_norag = [
                sys.executable,
                str(REPO / "scripts" / "run_documentation_agent_benchmark.py"),
                "--tasks",
                str(args.tasks.expanduser().resolve()),
                "--corpus",
                str(args.corpus.expanduser().resolve()),
                "--model",
                model_resolved,
                "--output-jsonl",
                str(norag_out),
                "--no-fail",
            ]
            if args.adapter_path:
                argv_norag.extend(
                    ["--adapter-path", str(Path(args.adapter_path).expanduser().resolve())]
                )
            _cmd_tool(run_dir, steps, "documentation_agent_benchmark_no_rag", argv_norag)
            np, nt = _count_doc_benchmark_jsonl(norag_out)
        bench_summary = f"doc-rag {rp}/{rt}"
        if not args.skip_no_rag_baseline:
            bench_summary += f"; no-rag {np}/{nt}"
        doc_rag_detail = {
            "model": model_resolved,
            "adapter_path": None if adapter_note == "—" else adapter_note,
            "tasks_path": str(args.tasks.expanduser().resolve()),
            "corpus_path": str(args.corpus.expanduser().resolve()),
            "rag_passed": rp,
            "rag_total": rt,
            "no_rag_passed": np if not args.skip_no_rag_baseline else None,
            "no_rag_total": nt if not args.skip_no_rag_baseline else None,
            "rag_jsonl": str(rag_out.relative_to(REPO)),
            "no_rag_jsonl": str(norag_out.relative_to(REPO)) if not args.skip_no_rag_baseline else None,
        }

    elif args.command == "adapter-gate":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "gates.py"),
            "--candidate",
            str(args.candidate.expanduser().resolve()),
            "--champion",
            str(args.champion.expanduser().resolve()),
        ]
        if args.output:
            argv += ["--output", str(args.output.expanduser().resolve())]
        final_code = _cmd_tool(run_dir, steps, "adapter_gate", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "drift-check":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "adapters" / "drift_monitor.py"),
            "--adapter-current",
            str(args.adapter_current.expanduser().resolve()),
            "--adapter-baseline",
            str(args.adapter_baseline.expanduser().resolve()),
            "--router-current",
            str(args.router_current.expanduser().resolve()),
            "--router-baseline",
            str(args.router_baseline.expanduser().resolve()),
        ]
        if args.output:
            argv += ["--output", str(args.output.expanduser().resolve())]
        final_code = _cmd_tool(run_dir, steps, "drift_check", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "control-plane-schedule":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "control_plane" / "scheduler.py"),
            "--workers-json",
            str(args.workers_json.expanduser().resolve()),
            "--jobs-json",
            str(args.jobs_json.expanduser().resolve()),
        ]
        final_code = _cmd_tool(run_dir, steps, "control_plane_schedule", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "multi-adapter-report":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "build_multi_adapter_report.py"),
            "--routing-jsonl",
            str(args.routing_jsonl.expanduser().resolve()),
            "--gate-json",
            str(args.gate_json.expanduser().resolve()),
            "--drift-json",
            str(args.drift_json.expanduser().resolve()),
            "--out-json",
            str(args.out_json.expanduser().resolve()),
            "--out-md",
            str(args.out_md.expanduser().resolve()),
        ]
        final_code = _cmd_tool(run_dir, steps, "multi_adapter_report", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "routing-prompt-lab":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "routing_prompt_lab.py"),
            "--source",
            str(args.source),
        ]
        for prompt in args.prompt:
            argv.extend(["--prompt", str(prompt)])
        if args.prompts_file:
            argv.extend(["--prompts-file", str(args.prompts_file.expanduser().resolve())])
        if args.interactive:
            argv.append("--interactive")
        if args.expected_adapter_id:
            argv.extend(["--expected-adapter-id", str(args.expected_adapter_id)])
        if args.expected_legacy_route:
            argv.extend(["--expected-legacy-route", str(args.expected_legacy_route)])
        if args.accepted_for_training:
            argv.append("--accepted-for-training")
        if args.reviewer:
            argv.extend(["--reviewer", str(args.reviewer)])
        if args.notes:
            argv.extend(["--notes", str(args.notes)])
        if args.output_jsonl:
            argv.extend(["--output-jsonl", str(args.output_jsonl.expanduser().resolve())])
        final_code = _cmd_tool(run_dir, steps, "routing_prompt_lab", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "routing-benchmark":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "run_routing_benchmark.py"),
            "--tasks",
            str(args.tasks.expanduser().resolve()),
            "--mode",
            str(args.mode),
        ]
        if args.output_jsonl:
            argv.extend(["--output-jsonl", str(args.output_jsonl.expanduser().resolve())])
        if args.summary_json:
            argv.extend(["--summary-json", str(args.summary_json.expanduser().resolve())])
        final_code = _cmd_tool(run_dir, steps, "routing_benchmark", argv)
        log_text = (run_dir / "logs" / "routing_benchmark.log").read_text(
            encoding="utf-8", errors="replace"
        )
        bench_summary = _parse_routing_overall_summary(log_text)
        adapter_note = "—"
        benchmark_adapter_note = "router-policy"

    elif args.command == "routing-dataset":
        argv = [
            sys.executable,
            str(REPO / "scripts" / "build_routing_training_dataset.py"),
            "--benchmark-tasks",
            str(args.benchmark_tasks.expanduser().resolve()),
            "--seed",
            str(args.seed),
            "--dataset-version",
            str(args.dataset_version),
            "--out-root",
            str(args.out_root.expanduser().resolve()),
            "--low-confidence-threshold",
            str(args.low_confidence_threshold),
        ]
        for p in args.live_jsonl:
            argv.extend(["--live-jsonl", str(p.expanduser().resolve())])
        for p in args.curated_jsonl:
            argv.extend(["--curated-jsonl", str(p.expanduser().resolve())])
        final_code = _cmd_tool(run_dir, steps, "routing_dataset", argv)
        adapter_note = "—"
        benchmark_adapter_note = "—"

    elif args.command == "smoke":
        py = sys.executable
        argv_b = [
            py,
            str(REPO / "scripts" / "build_lora_dataset.py"),
            "--synthetic-smoke",
            "--out-dir",
            _fe.GAME_TEXT_DIR_RELPATH,
        ]
        code, elapsed = _run_cmd(run_dir, "build_synthetic", argv_b)
        steps.append(
            {
                "name": "build_lora_dataset_synthetic",
                "argv": argv_b,
                "exit_code": code,
                "elapsed_s": elapsed,
                "log": str(run_dir / "logs" / "build_synthetic.log"),
            }
        )
        adapter_note = "checkpoints/_workflow_smoke"
        final_code = code
        if code == 0:
            final_code = _cmd_train(
                run_dir,
                steps,
                [
                    "--iters",
                    "4",
                    "--batch-size",
                    "1",
                    "--val-batches",
                    "1",
                    "--steps-per-eval",
                    "2",
                    "--steps-per-report",
                    "1",
                    "--max-seq-length",
                    "1024",
                    "--save-every",
                    "1000",
                ],
                adapter_note,
            )
        if final_code == 0:
            c, bench_summary = _cmd_benchmark(run_dir, steps, None, "game")
            benchmark_adapter_note = "base"
            final_code = c

    train_iters = "—"
    for s in steps:
        if s["name"] == "mlx_lm_lora_train" and s["exit_code"] == 0:
            log = Path(s["log"]).read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Starting training.*iters: (\d+)", log)
            if m:
                train_iters = m.group(1)
        if s["name"] == "run_game_benchmark":
            capability_summary = s.get("capability_summary", "—")
        if s["name"] == "run_arena_acceptance_tests":
            capability_summary = s.get("arena_capability_summary", "—")

    if args.command == "arena-gate-train":
        train_iters = f"{args.iters_per_cycle} iters/cycle (max {args.max_cycles} cycles)"

    workflow_finished = datetime.now(timezone.utc)
    workflow_elapsed_s = (workflow_finished - workflow_started).total_seconds()
    legacy_adapter_path = adapter_note
    if (
        (not legacy_adapter_path or legacy_adapter_path == "—")
        and benchmark_adapter_note not in {"base", "—"}
    ):
        legacy_adapter_path = benchmark_adapter_note

    manifest = {
        "run_id": run_id,
        "subcommand": args.command,
        "started_at": workflow_started.isoformat(),
        "finished_at": workflow_finished.isoformat(),
        "elapsed_s": workflow_elapsed_s,
        "final_exit_code": final_code,
        "repo": str(REPO),
        "git_short": _git_short(),
        "argv": sys.argv,
        "steps": steps,
        "benchmark_summary": bench_summary,
        "capability_summary": capability_summary,
        "adapter_path": None if legacy_adapter_path == "—" else legacy_adapter_path or None,
        "trained_adapter_path": adapter_note or None,
        "benchmark_adapter_path": benchmark_adapter_note or None,
    }
    if args.command == "arena-acceptance":
        manifest["arena_context"] = {
            "progressive_context": args.progressive_context,
            "suite": args.suite,
        }
    if doc_rag_detail:
        manifest["documentation_agent_rag"] = doc_rag_detail
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_run_md(
        run_dir,
        run_id,
        cmdline,
        steps,
        bench_summary,
        capability_summary,
        adapter_note,
        benchmark_adapter_note,
        final_code,
        workflow_started,
        workflow_finished,
        workflow_elapsed_s,
    )

    train_iters_cell = train_iters
    if args.command == "prepare":
        train_iters_cell = "0 (prepare only)"
    _append_history_row(
        run_id,
        args.command,
        final_code,
        train_iters_cell,
        bench_summary,
        adapter_note,
        benchmark_adapter_note,
    )

    if args.command == "documentation-rag-benchmark" and not args.no_timeseries_append and doc_rag_detail:
        ts_record = {
            "run_id": run_id,
            "final_exit_code": final_code,
            "utc_iso": workflow_finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_short": _git_short(),
            "workflow_run_dir": f"benchmarks/results/runs/{run_id}/",
            **doc_rag_detail,
        }
        _append_documentation_rag_timeseries(ts_record)
    if doc_rag_detail:
        _append_specialized_history_row(
            utc_iso=workflow_finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
            track="documentation_agent_rag",
            entry_point="ml_workflow.py documentation-rag-benchmark",
            exit_code=final_code,
            summary=bench_summary,
            artifacts=f"`benchmarks/results/runs/{run_id}/`, `benchmarks/results/documentation_rag_timeseries.jsonl`",
            notes=(
                f"ml_workflow_run_id `{run_id}`; process exit reflects RAG pass only; "
                "no-RAG baseline uses `run_documentation_agent_benchmark.py --no-fail` when run."
            ),
        )

    print(f"Run documentation: {run_dir}", file=sys.stderr)
    print(f"Committed index append: {HISTORY_PATH}", file=sys.stderr)
    if doc_rag_detail:
        print(f"Specialized index append: {SPECIALIZED_HISTORY_PATH}", file=sys.stderr)

    sys.exit(final_code)


if __name__ == "__main__":
    main()
