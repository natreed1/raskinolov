#!/usr/bin/env python3
"""Run run-analysis-agent string benchmarks against an MLX model with optional RAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_evolution_lib as bel
from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops
from run_analysis_rag import DEFAULT_CORPUS, build_context, load_corpus, retrieve

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO / "benchmarks" / "run_analysis_rag_tasks_v1.json"
SYSTEM = (
    "You are a careful run-analysis and documentation agent for fallen-empire-lora. "
    "Use retrieved context when provided. Cite exact repository paths and commands. "
    "Separate evidence from inference and do not claim runs/tests passed without command output."
)


def _messages(prompt: str, context: str) -> list[dict[str, str]]:
    if context:
        user = (
            "Retrieved run-analysis context:\n"
            f"{context}\n\n"
            "Answer using retrieved context first, cite exact paths/commands, and end with a `Sources:` line.\n\n"
            f"Question: {prompt}"
        )
    else:
        user = prompt
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _score_retrieval(tasks: list[dict[str, Any]], corpus_path: Path, top_k: int) -> tuple[int, int]:
    corpus = load_corpus(corpus_path)
    covered = 0
    for task in tasks:
        hits = retrieve(task["prompt"], corpus, top_k=top_k)
        context = build_context(hits, max_chars=5000).lower()
        terms = [str(t).lower() for t in task.get("expect", {}).get("all_contains", [])]
        if terms and all(term in context for term in terms):
            covered += 1
    return covered, len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run-analysis-agent MLX benchmark.")
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--dry-run-retrieval", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--context-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even when tasks fail.")
    args = parser.parse_args()

    tasks = bel.load_tasks(args.tasks)
    if args.dry_run_retrieval:
        covered, total = _score_retrieval(tasks, args.corpus, args.top_k)
        rate = covered / total if total else 0.0
        print(f"=== Retrieval coverage: {covered}/{total} ({rate:.0%}) ===")
        if covered < total:
            sys.exit(1)
        return

    from mlx_lm import generate, load

    load_kw: dict[str, str] = {}
    if args.adapter_path:
        load_kw["adapter_path"] = args.adapter_path
    print(f"Loading {args.model!r} for run-analysis-agent benchmark...", file=sys.stderr)
    t_load = time.perf_counter()
    model, tokenizer = load(args.model, **load_kw)
    register_qwen_coder_instruct_extra_stops(tokenizer)
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s", file=sys.stderr)

    corpus = load_corpus(args.corpus) if args.use_rag else []
    sink = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None
    passed = 0
    rows = []
    for task in tasks:
        hits = retrieve(task["prompt"], corpus, top_k=args.top_k) if args.use_rag else []
        context = build_context(hits, max_chars=args.context_chars) if hits else ""
        prompt = tokenizer.apply_chat_template(
            _messages(task["prompt"], context), tokenize=False, add_generation_prompt=True
        )
        t0 = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        elapsed = time.perf_counter() - t0
        ok, failures = bel.check_expect(output, task["expect"])
        passed += int(ok)
        row = {
            "task_id": task["id"],
            "category": task.get("category", ""),
            "passed": ok,
            "failures": failures,
            "seconds": round(elapsed, 3),
            "chars": len(output),
            "used_rag": args.use_rag,
            "retrieved_sources": [f"{hit.path}#{hit.index}" for hit in hits],
            "output_preview": output[:700],
        }
        rows.append(row)
        if sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{'PASS' if ok else 'FAIL'}] {task['id']} {len(output)} chars {elapsed:.2f}s")
    if sink:
        sink.close()

    total = len(rows)
    rate = passed / total if total else 0.0
    print(f"=== Summary: {passed}/{total} ({rate:.0%}) ===")
    if passed < total and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
