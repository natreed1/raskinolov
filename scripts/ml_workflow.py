#!/usr/bin/env python3
"""
Built-in ML pipeline orchestrator: export → LoRA JSONL → train → benchmark,
with **per-run documentation** under `benchmarks/results/runs/<run_id>/` plus an
append-only row in **committed** `docs/run_history.md`.

Subcommands
  prepare   SOURCE_REPO export + build_lora_dataset (no training).
  train     mlx_lm.lora --train (expects data/lora/game_text already).
  benchmark Run run_game_benchmark.py (--adapter-path optional; --profile game or general).
  evalplus  Run execution-based EvalPlus benchmark (--suite humaneval|mbpp).
  full      prepare + train + benchmark using the new adapter path.
  smoke     Synthetic data, a few train iters, benchmark base model (CI-friendly).

Every invocation creates:
  benchmarks/results/runs/<run_id>/manifest.json
  benchmarks/results/runs/<run_id>/RUN.md
  benchmarks/results/runs/<run_id>/logs/*.log

And appends one line to docs/run_history.md (repo root, committed), including
exit/status fields so failed audit rows are visible without opening artifacts.
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

REPO = Path(__file__).resolve().parent.parent
RUNS_PARENT = REPO / "benchmarks" / "results" / "runs"
HISTORY_PATH = REPO / "docs" / "run_history.md"
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


def _parse_capability_summary(log_text: str) -> str:
    m = re.search(r"=== Capability Index: ([0-9.]+/100.*?) ===", log_text)
    return m.group(1) if m else "—"


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
    argv_b = [py, str(REPO / "scripts" / "build_lora_dataset.py"), "--out-dir", "data/lora/game_text"]
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
        "training/lora_qwen_coder.yaml",
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


def main() -> None:
    _ensure_repo_venv()

    parser = argparse.ArgumentParser(description="Orchestrated ML workflow with per-run docs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Export + build LoRA JSONL")
    p_train = sub.add_parser("train", help="Run mlx_lm.lora --train (no benchmark unless --evaluate)")
    p_train.add_argument("--adapter-path", default="checkpoints/fe-lora-latest")
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

    p_full = sub.add_parser("full", help="prepare + train + benchmark")
    p_full.add_argument("--adapter-path", default="checkpoints/fe-lora-latest")
    p_full.add_argument(
        "--bench-profile",
        choices=["game", "general"],
        default="game",
        help="run_game_benchmark.py profile when the benchmark step runs",
    )
    p_full.add_argument("lora_args", nargs=argparse.REMAINDER, help="Extra args after -- for mlx_lm.lora")

    p_smoke = sub.add_parser("smoke", help="Synthetic data + tiny train + benchmark (no export)")

    args = parser.parse_args()
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

    elif args.command == "smoke":
        py = sys.executable
        argv_b = [
            py,
            str(REPO / "scripts" / "build_lora_dataset.py"),
            "--synthetic-smoke",
            "--out-dir",
            "data/lora/game_text",
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

    print(f"Run documentation: {run_dir}", file=sys.stderr)
    print(f"Committed index append: {HISTORY_PATH}", file=sys.stderr)

    sys.exit(final_code)


if __name__ == "__main__":
    main()
