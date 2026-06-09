#!/usr/bin/env python3
"""
Run specialist-focused benchmark tasks against an mlx-lm model and score outputs
with simple string rules.

Usage:
  source .venv/bin/activate
  python scripts/run_game_benchmark.py
  python scripts/run_game_benchmark.py --specialist combat_risk --specialist economy_tooltip
  python scripts/run_game_benchmark.py --tier A --tasks benchmarks/specialist_benchmark_tasks.json
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
import fe_lineage as _fe

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops

DEFAULT_MODEL = _fe.HF_MODEL_ID
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_SPECIALIST = REPO_ROOT / "benchmarks" / "specialist_benchmark_tasks.json"
SYSTEM_SPECIALIST = (
    "You are Albert, assisting with the Fallen Empire strategy game (TypeScript / React / Zustand). "
    "Follow each user instruction literally. Prefer correct game terminology when relevant."
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
    task_weight: float
    domains: List[str]
    multi_domain: bool


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


def infer_domains(task: Dict[str, Any]) -> List[str]:
    declared = [str(d).strip().lower() for d in (task.get("domains") or []) if str(d).strip()]
    if declared:
        return list(dict.fromkeys(declared))
    category = str(task.get("category") or "").lower()
    domains: List[str] = []
    mapping = [
        ("hud", "hud"),
        ("eco", "economy"),
        ("economy", "economy"),
        ("combat", "combat"),
        ("save", "save_load"),
        ("load", "save_load"),
        ("ai", "planning"),
        ("planning", "planning"),
        ("loading", "loading_screen"),
        ("ui", "ui"),
    ]
    for token, domain in mapping:
        if token in category:
            domains.append(domain)
    if not domains:
        domains.append("general")
    return list(dict.fromkeys(domains))


def task_weight(task: Dict[str, Any], domains: List[str]) -> float:
    expect = task.get("expect") or {}
    category = str(task.get("category") or "").lower()
    # Difficulty proxy from rubric complexity.
    min_chars = int(expect.get("min_chars") or 0)
    n_all = len(expect.get("all_contains") or [])
    n_any = len(expect.get("any_contains") or [])
    n_none = len(expect.get("none_contains") or [])
    complexity = 1.0 + min(0.35, (min_chars / 1200.0) + (n_all * 0.03) + (n_any * 0.01) + (n_none * 0.02))

    category_boost = 1.0
    if "multidomain" in category or "multi_domain" in category:
        category_boost += 0.20
    elif "transfer" in category:
        category_boost += 0.12
    elif "ui_change" in category:
        category_boost += 0.05
    elif "constraints" in category:
        category_boost -= 0.04

    # Reward tasks that truly span multiple domains.
    multi_domain_bonus = 1.0 + (0.08 * max(0, len(domains) - 1))
    return round(complexity * category_boost * multi_domain_bonus, 4)


def advanced_aci(results: List[TaskResult]) -> Dict[str, float]:
    if not results:
        return {
            "weighted_capability": 0.0,
            "domain_balance": 0.0,
            "multi_domain_mastery": 0.0,
            "advanced_aci": 0.0,
        }

    total_weight = sum(max(0.0001, r.task_weight) for r in results)
    weighted_cap = sum(r.capability_score * max(0.0001, r.task_weight) for r in results) / total_weight

    domain_scores: Dict[str, List[float]] = {}
    for r in results:
        score = 100.0 if r.passed else 0.0
        for d in r.domains:
            domain_scores.setdefault(d, []).append(score)
    domain_balance = (
        sum(sum(vals) / len(vals) for vals in domain_scores.values()) / len(domain_scores)
        if domain_scores
        else 0.0
    )

    md = [r for r in results if r.multi_domain]
    multi_domain_mastery = (
        100.0 * (sum(1 for r in md if r.passed) / len(md))
        if md
        else (100.0 * sum(1 for r in results if r.passed) / len(results))
    )

    # Advanced ACI: weighted capability + domain spread + multi-domain success.
    aci = (0.65 * weighted_cap) + (0.20 * domain_balance) + (0.15 * multi_domain_mastery)
    return {
        "weighted_capability": round(weighted_cap, 2),
        "domain_balance": round(domain_balance, 2),
        "multi_domain_mastery": round(multi_domain_mastery, 2),
        "advanced_aci": round(aci, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX benchmark runner for specialist string-rubric tasks"
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS_SPECIALIST,
        help="Task JSON (array). Default: specialist benchmark task set.",
    )
    parser.add_argument(
        "--specialist",
        action="append",
        default=[],
        help=(
            "Filter to benchmark tasks mapped to one or more specialists. "
            "Repeat for multiple specialists, e.g. --specialist combat_risk --specialist economy_tooltip."
        ),
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

    tasks_path = args.tasks.resolve()
    system = SYSTEM_SPECIALIST
    if not tasks_path.is_file():
        raise SystemExit(f"Tasks file not found: {tasks_path}")

    tasks = bel.load_tasks(tasks_path)
    requested_specialists = [s.strip() for s in args.specialist if s.strip()]
    if requested_specialists:
        requested_set = set(requested_specialists)
        filtered = []
        for task in tasks:
            task_specialists = set(task.get("specialists") or [])
            if task_specialists & requested_set:
                filtered.append(task)
        tasks = filtered
        if not tasks:
            raise SystemExit(
                "No benchmark tasks matched requested specialists: "
                + ", ".join(sorted(requested_set))
            )

    load_kw: dict = {}
    if args.adapter_path:
        load_kw["adapter_path"] = args.adapter_path

    specialist_label = ",".join(requested_specialists) if requested_specialists else "all"
    print(
        f"Loading {args.model!r} (specialists={specialist_label!r}, tasks={tasks_path.name}) …",
        file=sys.stderr,
    )
    t_load = time.perf_counter()
    model, tokenizer = load(args.model, **load_kw)
    register_qwen_coder_instruct_extra_stops(tokenizer)
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
        domains = infer_domains(task)
        weight = task_weight(task, domains)
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
            weight,
            domains,
            len(domains) > 1,
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
            "task_weight": weight,
            "domains": domains,
            "multi_domain": len(domains) > 1,
            "model": args.model,
            "specialists": requested_specialists,
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
    adv = advanced_aci(results)
    print(f"=== Summary: {passed}/{total} ({rate:.0%}) ===")
    print(
        "=== Capability Index: "
        f"{avg_cap:.1f}/100 "
        f"(correctness {avg_correct:.1f}, instruction {avg_instruction:.1f}, "
        f"concision {avg_concision:.1f}, speed {avg_speed:.1f}) ==="
    )
    print(
        "=== Advanced ACI: "
        f"{adv['advanced_aci']:.1f}/100 "
        f"(weighted {adv['weighted_capability']:.1f}, "
        f"domain-balance {adv['domain_balance']:.1f}, "
        f"multi-domain {adv['multi_domain_mastery']:.1f}) ==="
    )
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
