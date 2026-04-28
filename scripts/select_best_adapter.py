#!/usr/bin/env python3
"""
Select the best currently trained adapter from workflow artifacts.

This script does not run training. It scans `benchmarks/results/runs/*` for
training logs and benchmark manifests, then writes a small Markdown report with
a conservative recommendation that penalizes overfitting.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "benchmarks" / "results" / "runs"
DEFAULT_OUT = REPO / "benchmarks" / "results" / "adapter_selection.md"


@dataclass
class Candidate:
    adapter_path: str
    train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_val_iter: Optional[int] = None
    peak_mem_gb: Optional[float] = None
    train_run: Optional[str] = None
    benchmark_runs: List[str] = field(default_factory=list)
    benchmark_scores: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    overfit_flags: List[str] = field(default_factory=list)


def _parse_rate(summary: str) -> Optional[float]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", summary or "")
    if not m:
        return None
    den = int(m.group(2))
    return int(m.group(1)) / den if den else None


def _profile_from_step(step: dict) -> str:
    argv = step.get("argv") or []
    if "run_evalplus_benchmark.py" in " ".join(argv):
        suite = "evalplus"
        if "--suite" in argv:
            try:
                suite = f"evalplus-{argv[argv.index('--suite') + 1]}"
            except Exception:
                pass
        return suite
    if "--profile" in argv:
        try:
            return argv[argv.index("--profile") + 1]
        except Exception:
            return "benchmark"
    return "benchmark"


def _read_manifest(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _update_training_metrics(candidate: Candidate, log_path: Path) -> None:
    if not log_path.is_file():
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    val_points: List[Tuple[int, float]] = [
        (int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", text)
    ]
    train_points = [
        (int(i), float(loss), float(mem))
        for i, loss, mem in re.findall(
            r"Iter (\d+): Train loss ([0-9.]+).*?Peak mem ([0-9.]+) GB", text
        )
    ]
    if val_points:
        candidate.final_val_loss = val_points[-1][1]
        best_iter, best_val = min(val_points, key=lambda p: p[1])
        candidate.best_val_iter = best_iter
        candidate.best_val_loss = best_val
    if train_points:
        _, train_loss, peak_mem = train_points[-1]
        candidate.train_loss = train_loss
        candidate.peak_mem_gb = peak_mem


def _collect_candidates(runs_dir: Path) -> Dict[str, Candidate]:
    candidates: Dict[str, Candidate] = {}
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = _read_manifest(manifest_path)
        if not manifest:
            continue
        run_id = manifest.get("run_id") or manifest_path.parent.name
        adapter = manifest.get("adapter_path")
        if not adapter or adapter == "—":
            continue
        if "_workflow_smoke" in adapter:
            continue
        candidate = candidates.setdefault(adapter, Candidate(adapter_path=adapter))

        if manifest.get("subcommand") == "train":
            candidate.train_run = str(run_id)
            for step in manifest.get("steps", []):
                if step.get("name") == "mlx_lm_lora_train":
                    _update_training_metrics(candidate, Path(step.get("log", "")))

        for step in manifest.get("steps", []):
            summary = step.get("benchmark_summary")
            if not summary or summary == "—":
                continue
            profile = _profile_from_step(step)
            candidate.benchmark_scores[profile] = summary
            candidate.benchmark_runs.append(str(run_id))
    return candidates


def _score(candidate: Candidate, best_seen_val: Optional[float]) -> None:
    score = 0.0
    weights = {
        "game": 35.0,
        "general": 15.0,
        "evalplus-humaneval": 35.0,
        "evalplus-mbpp": 25.0,
    }
    for profile, weight in weights.items():
        rate = _parse_rate(candidate.benchmark_scores.get(profile, ""))
        if rate is not None:
            score += rate * weight

    if candidate.final_val_loss is not None:
        score += max(0.0, 15.0 - candidate.final_val_loss * 5.0)
    if candidate.best_val_loss is not None and candidate.final_val_loss is not None:
        if candidate.final_val_loss > candidate.best_val_loss * 1.25:
            candidate.overfit_flags.append(
                f"final validation loss {candidate.final_val_loss:.3f} is materially worse than best {candidate.best_val_loss:.3f}"
            )
            score -= 20.0
    if best_seen_val is not None and candidate.final_val_loss is not None:
        if candidate.final_val_loss > best_seen_val * 1.5:
            candidate.overfit_flags.append(
                f"validation loss is >1.5x the best candidate ({best_seen_val:.3f})"
            )
            score -= 15.0

    game_rate = _parse_rate(candidate.benchmark_scores.get("game", ""))
    if game_rate is not None and game_rate < 1.0:
        candidate.overfit_flags.append("game benchmark regressed below 100%")
        score -= (1.0 - game_rate) * 20.0

    candidate.score = score


def _render_report(candidates: List[Candidate]) -> str:
    recommended = candidates[0] if candidates else None
    lines = [
        "# Adapter Selection Report",
        "",
        "Generated from `benchmarks/results/runs/*/manifest.json` and training logs.",
        "",
    ]
    if recommended:
        lines.extend(
            [
                f"**Recommended adapter:** `{recommended.adapter_path}`",
                "",
                "Reason: highest conservative score after penalizing validation-loss drift and benchmark regressions.",
                "",
            ]
        )
    lines.extend(
        [
            "| Adapter | Score | Train loss | Final val | Best val | Benchmarks | Overfit flags |",
            "|---------|------:|-----------:|----------:|---------:|------------|---------------|",
        ]
    )
    for c in candidates:
        bench = ", ".join(f"{k}: {v}" for k, v in sorted(c.benchmark_scores.items())) or "—"
        flags = "; ".join(c.overfit_flags) or "—"
        lines.append(
            "| `{adapter}` | {score:.1f} | {train} | {final_val} | {best_val} | {bench} | {flags} |".format(
                adapter=c.adapter_path,
                score=c.score,
                train=f"{c.train_loss:.3f}" if c.train_loss is not None else "—",
                final_val=f"{c.final_val_loss:.3f}" if c.final_val_loss is not None else "—",
                best_val=(
                    f"{c.best_val_loss:.3f} @ {c.best_val_iter}"
                    if c.best_val_loss is not None
                    else "—"
                ),
                bench=bench,
                flags=flags,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the best non-overfit LoRA adapter from run artifacts.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    candidates = list(_collect_candidates(args.runs_dir).values())
    vals = [c.final_val_loss for c in candidates if c.final_val_loss is not None]
    best_seen_val = min(vals) if vals else None
    for c in candidates:
        _score(c, best_seen_val)
    candidates.sort(key=lambda c: c.score, reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_report(candidates), encoding="utf-8")

    if not candidates:
        print("No adapter candidates found.")
        return
    print(f"Recommended adapter: {candidates[0].adapter_path}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
