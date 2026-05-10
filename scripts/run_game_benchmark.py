#!/usr/bin/env python3
"""
Run a fixed JSON task suite against an mlx-lm model and score outputs with simple string rules.

**Game profile** (default) uses Fallen Empire–flavored tasks and a matching system prompt.
**General profile** uses `benchmarks/general_coding_tasks.json` and a neutral coding-assistant
system prompt (portable “how well does the base vs adapter do on generic Q&A?”).

Usage:
  source .venv/bin/activate
  python scripts/run_game_benchmark.py
  python scripts/run_game_benchmark.py --profile general
  python scripts/run_game_benchmark.py --tier A --tasks benchmarks/fallen_empire_tasks.json
  python scripts/run_game_benchmark.py --tasks benchmarks/custom_tasks.json --output-jsonl benchmarks/results/run.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_evolution_lib as bel

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_GAME = REPO_ROOT / "benchmarks" / "fallen_empire_tasks.json"
DEFAULT_TASKS_GENERAL = REPO_ROOT / "benchmarks" / "general_coding_tasks.json"
SYSTEM_GAME = (
    "You are Albert, assisting with the Fallen Empire strategy game (TypeScript / React / Zustand). "
    "Follow each user instruction literally. Prefer correct game terminology when relevant."
)
SYSTEM_GENERAL = (
    "You are a careful programming assistant. Follow each user instruction literally. "
    "Answer clearly; prefer exact terminology for standards (HTTP, SQL, regex, encodings, complexity)."
)


@dataclass
class TaskResult:
    task_id: str
    category: str
    passed: bool
    failures: List[str]
    chars: int
    seconds: float
    output_preview: str
    capability_score: float
    correctness_score: float
    instruction_score: float
    concision_score: float
    speed_score: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sentence_count(text: str) -> int:
    return max(1, sum(text.count(p) for p in ".!?") or 1)


def correctness_score(passed: bool, failures: List[str], expect: Dict[str, Any]) -> float:
    if passed:
        return 1.0
    required = len(expect.get("all_contains") or [])
    optional = 1 if expect.get("any_contains") else 0
    forbidden = len(expect.get("none_contains") or [])
    min_chars = 1 if expect.get("min_chars") is not None else 0
    total_checks = max(1, required + optional + forbidden + min_chars)
    return _clamp01(1.0 - (len(failures) / total_checks))


def instruction_score(output: str, prompt: str) -> float:
    prompt_l = prompt.lower()
    score = 1.0
    line_count = len([line for line in output.splitlines() if line.strip()])
    chars = len(output)
    if any(k in prompt_l for k in ["only", "exactly one", "one-line", "one line", "no extra", "no explanation"]):
        if line_count > 4:
            score -= 0.35
        if chars > 400:
            score -= 0.35
    if "comma-separated" in prompt_l and "\n" in output.strip():
        score -= 0.25
    if "no code" in prompt_l and any(tok in output for tok in ["```", "function ", "=>", "const "]):
        score -= 0.3
    if "code only" in prompt_l and any(tok in output.lower() for tok in ["explanation", "here is", "this function"]):
        score -= 0.25
    return _clamp01(score)


def concision_score(output: str, prompt: str, expect: Dict[str, Any]) -> float:
    chars = len(output.strip())
    if chars == 0:
        return 0.0
    prompt_l = prompt.lower()
    if any(k in prompt_l for k in ["only", "exactly one", "one-line", "one line", "no extra", "no explanation"]):
        target_max = 220
    elif "1–2 sentences" in prompt or "one or two sentences" in prompt_l:
        target_max = 420
    elif "3–5 sentences" in prompt or "3-5 sentences" in prompt:
        target_max = 900
    elif "4–6 sentences" in prompt or "4-6 sentences" in prompt:
        target_max = 1200
    else:
        min_chars = int(expect.get("min_chars") or 80)
        target_max = max(320, min(1400, min_chars * 5))
    if chars <= target_max:
        return 1.0
    # Softly penalize runaway answers without making one long but correct answer zero.
    return _clamp01(target_max / chars)


def speed_score(seconds: float) -> float:
    if seconds <= 1.0:
        return 1.0
    if seconds <= 3.0:
        return 0.8
    if seconds <= 6.0:
        return 0.55
    return 0.3


def capability_scores(
    output: str,
    prompt: str,
    expect: Dict[str, Any],
    passed: bool,
    failures: List[str],
    seconds: float,
) -> Dict[str, float]:
    c = correctness_score(passed, failures, expect)
    i = instruction_score(output, prompt)
    k = concision_score(output, prompt, expect)
    s = speed_score(seconds)
    # Correctness dominates, but instruction discipline matters enough to catch
    # overfit adapters that pass substring checks with huge irrelevant outputs.
    total = (0.62 * c) + (0.20 * i) + (0.13 * k) + (0.05 * s)
    return {
        "capability": round(100.0 * total, 2),
        "correctness": round(100.0 * c, 2),
        "instruction": round(100.0 * i, 2),
        "concision": round(100.0 * k, 2),
        "speed": round(100.0 * s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX benchmark runner (game or general string-rubric tasks)"
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--profile",
        choices=["game", "general"],
        default="game",
        help="Default tasks + system: game=Fallen Empire suite; general=generic coding rubrics.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Task JSON (array). Default depends on --profile if omitted.",
    )
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    parser.add_argument("--max-tokens", type=int, default=None, help="Override decode cap (default: tier or 512)")
    parser.add_argument(
        "--tier",
        choices=["C", "B", "A"],
        default=None,
        help="Sim-style tier: scales min_chars rubric and default max_tokens (C/B/A).",
    )
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Append one JSON line per task")
    args = parser.parse_args()

    if args.tasks is None:
        tasks_path = (
            DEFAULT_TASKS_GENERAL if args.profile == "general" else DEFAULT_TASKS_GAME
        )
    else:
        tasks_path = args.tasks
    system = SYSTEM_GENERAL if args.profile == "general" else SYSTEM_GAME

    tasks_path = tasks_path.resolve()
    if not tasks_path.is_file():
        raise SystemExit(f"Tasks file not found: {tasks_path}")

    tasks = bel.load_tasks(tasks_path)
    load_kw: dict = {}
    if args.adapter_path:
        load_kw["adapter_path"] = args.adapter_path

    print(
        f"Loading {args.model!r} (profile={args.profile!r}, tasks={tasks_path.name}) …",
        file=sys.stderr,
    )
    t_load = time.perf_counter()
    model, tokenizer = load(args.model, **load_kw)
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s", file=sys.stderr)

    if args.max_tokens is not None:
        cap = args.max_tokens
    elif args.tier:
        cap = bel.tier_max_tokens(args.tier)
    else:
        cap = 512
    gen_kwargs: dict = {"max_tokens": cap}
    if args.temp > 0:
        gen_kwargs["sampler"] = make_sampler(temp=args.temp, top_p=1.0)

    results: List[TaskResult] = []
    sink = args.output_jsonl.open("a", encoding="utf-8") if args.output_jsonl else None

    for task in tasks:
        tid = task["id"]
        cat = task.get("category", "")
        user = task["prompt"]
        expect_raw = task["expect"]
        expect_use = bel.scale_expect_for_tier(expect_raw, args.tier) if args.tier else expect_raw
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=prompt, verbose=False, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        ok, fails = bel.check_expect(out, expect_use)
        scores = capability_scores(out, user, expect_use, ok, fails, elapsed)
        preview = (out[:400] + "…") if len(out) > 400 else out
        tr = TaskResult(
            tid,
            cat,
            ok,
            fails,
            len(out),
            elapsed,
            preview,
            scores["capability"],
            scores["correctness"],
            scores["instruction"],
            scores["concision"],
            scores["speed"],
        )
        results.append(tr)

        row = {
            "task_id": tid,
            "category": cat,
            "passed": ok,
            "failures": fails,
            "chars": len(out),
            "seconds": round(elapsed, 3),
            "capability_score": scores["capability"],
            "correctness_score": scores["correctness"],
            "instruction_score": scores["instruction"],
            "concision_score": scores["concision"],
            "speed_score": scores["speed"],
            "model": args.model,
            "profile": args.profile,
            "tasks_file": str(tasks_path),
            "tier": args.tier,
        }
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

        tier_note = f" [{args.tier}]" if args.tier else ""
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {tid} ({cat}){tier_note}  {len(out)} chars  {elapsed:.2f}s"
            f"  cap={scores['capability']:.1f}"
        )
        if not ok:
            for f in fails:
                print(f"         - {f}")
        print()

    if sink:
        sink.close()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = passed / total if total else 0.0
    avg_cap = sum(r.capability_score for r in results) / total if total else 0.0
    avg_correct = sum(r.correctness_score for r in results) / total if total else 0.0
    avg_instruction = sum(r.instruction_score for r in results) / total if total else 0.0
    avg_concision = sum(r.concision_score for r in results) / total if total else 0.0
    avg_speed = sum(r.speed_score for r in results) / total if total else 0.0
    print(f"=== Summary: {passed}/{total} ({rate:.0%}) ===")
    print(
        "=== Capability Index: "
        f"{avg_cap:.1f}/100 "
        f"(correctness {avg_correct:.1f}, instruction {avg_instruction:.1f}, "
        f"concision {avg_concision:.1f}, speed {avg_speed:.1f}) ==="
    )
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
