#!/usr/bin/env python3
"""
Run an execution-based EvalPlus benchmark against an MLX model.

This is intentionally separate from `run_game_benchmark.py`: the game/general
runner uses cheap substring rubrics, while EvalPlus executes generated Python
code in a guarded subprocess. Start with a small `--limit` before full runs.

Examples:
  source .venv/bin/activate
  python scripts/run_evalplus_benchmark.py --suite humaneval --limit 5
  python scripts/run_evalplus_benchmark.py --adapter-path checkpoints/fe-lora-30m --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import PASS
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

SYSTEM_PROMPT = (
    "You are a precise Python coding assistant. Complete the requested function. "
    "Return only valid Python code; do not include prose or Markdown."
)


def _sorted_task_items(problems: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def key(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, int]:
        tid = item[0]
        m = re.search(r"(\d+)$", tid)
        return (tid.rsplit("/", 1)[0], int(m.group(1)) if m else 0)

    return sorted(problems.items(), key=key)


def _select_tasks(
    problems: Dict[str, Dict[str, Any]], limit: Optional[int], task_ids: Optional[List[str]]
) -> Dict[str, Dict[str, Any]]:
    if task_ids:
        missing = [tid for tid in task_ids if tid not in problems]
        if missing:
            raise SystemExit(f"Unknown EvalPlus task id(s): {', '.join(missing)}")
        return {tid: problems[tid] for tid in task_ids}

    items = _sorted_task_items(problems)
    if limit is not None:
        items = items[:limit]
    return dict(items)


def _extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    text = text.strip()

    # Drop common assistant prefixes before the first plausible code line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("def ", "from ", "import ", "class ", "@")):
            return "\n".join(lines[i:]).strip()
    return text


def _build_solution(problem: Dict[str, Any], raw_output: str) -> str:
    code = _extract_python_code(raw_output)
    entry_point = problem["entry_point"]
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, flags=re.MULTILINE):
        return code
    return problem["prompt"] + "\n" + code


def _default_output_path(suite: str, adapter_path: Optional[str]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    adapter_slug = "base"
    if adapter_path:
        adapter_slug = Path(adapter_path).name or "adapter"
    return DEFAULT_RESULTS_DIR / f"evalplus_{suite}_{adapter_slug}_{ts}.jsonl"


def _resolve_adapter_path(adapter_path: Optional[str]) -> Tuple[Optional[str], Optional[tempfile.TemporaryDirectory]]:
    """Allow either adapter directories or snapshot `.safetensors` files."""
    if not adapter_path:
        return None, None

    path = Path(adapter_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if path.is_dir():
        return str(path), None
    if path.is_file() and path.name.endswith(".safetensors"):
        parent = path.parent
        config = parent / "adapter_config.json"
        if not config.is_file():
            raise SystemExit(f"Snapshot adapter file needs sibling adapter_config.json: {path}")
        tmp = tempfile.TemporaryDirectory(prefix="fe-lora-snapshot-")
        tmp_path = Path(tmp.name)
        shutil.copy2(config, tmp_path / "adapter_config.json")
        shutil.copy2(path, tmp_path / "adapters.safetensors")
        return str(tmp_path), tmp
    raise SystemExit(f"Adapter path not found or unsupported: {adapter_path}")


def _load_problems(suite: str, mini: bool, noextreme: bool) -> Dict[str, Dict[str, Any]]:
    try:
        if suite == "humaneval":
            return get_human_eval_plus(mini=mini, noextreme=noextreme)
        if suite == "mbpp":
            return get_mbpp_plus(mini=mini, noextreme=noextreme)
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Failed to download EvalPlus dataset. Retry when GitHub/network is healthy, "
            "or set HUMANEVAL_OVERRIDE_PATH / MBPP_OVERRIDE_PATH to a local EvalPlus JSONL."
        ) from exc
    raise SystemExit(f"Unsupported suite: {suite}")


def _tasks_only_output_not_none(suite: str) -> Iterable[str]:
    return MBPP_OUTPUT_NOT_NONE_TASKS if suite == "mbpp" else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvalPlus HumanEval/MBPP with an MLX model.")
    parser.add_argument("--suite", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    parser.add_argument("--limit", type=int, default=5, help="Number of tasks to run; omit with --full.")
    parser.add_argument("--full", action="store_true", help="Run the full selected suite.")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Specific task id to run; repeatable.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--base-only", action="store_true", help="Only require original HumanEval/MBPP tests.")
    parser.add_argument("--mini", action="store_true", help="Use EvalPlus mini suite when available.")
    parser.add_argument("--noextreme", action="store_true", help="Use EvalPlus no-extreme variant.")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    limit = None if args.full else args.limit
    problems_all = _load_problems(args.suite, args.mini, args.noextreme)
    problems = _select_tasks(problems_all, limit, args.task_ids)
    if not problems:
        raise SystemExit("No EvalPlus tasks selected.")

    output_path = args.output_jsonl or _default_output_path(args.suite, args.adapter_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_resolved, tmp_adapter = _resolve_adapter_path(args.adapter_path)
    load_kw: Dict[str, Any] = {}
    if adapter_resolved:
        load_kw["adapter_path"] = adapter_resolved

    print(
        f"Loading {args.model!r} (suite={args.suite}, tasks={len(problems)}, adapter={args.adapter_path or '—'}) …",
        file=sys.stderr,
    )
    t_load = time.perf_counter()
    model, tokenizer = load(args.model, **load_kw)
    print(f"Loaded in {time.perf_counter() - t_load:.1f}s", file=sys.stderr)

    gen_kwargs: Dict[str, Any] = {"max_tokens": args.max_tokens}
    if args.temp > 0:
        gen_kwargs["sampler"] = make_sampler(temp=args.temp, top_p=1.0)

    # EvalPlus ground truth generation executes canonical solutions. Keep the
    # cache key subset-specific so small runs do not pretend to cover the suite.
    subset_key = f"{args.suite}-subset-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(problems))
    expected = get_groundtruth(problems, subset_key, _tasks_only_output_not_none(args.suite))

    passed = 0
    rows: List[Dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as sink:
        for index, (task_id, problem) in enumerate(_sorted_task_items(problems), start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Complete this Python function. Return only the implementation code.\n\n"
                        f"{problem['prompt']}"
                    ),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            t0 = time.perf_counter()
            raw = generate(model, tokenizer, prompt=prompt, verbose=False, **gen_kwargs)
            gen_s = time.perf_counter() - t0
            solution = _build_solution(problem, raw)

            result = check_correctness(
                args.suite,
                0,
                problem,
                solution,
                expected[task_id],
                base_only=args.base_only,
                fast_check=True,
                identifier=task_id,
            )
            base_status = result["base"][0]
            plus_status = None if args.base_only else result["plus"][0]
            ok = base_status == PASS and (args.base_only or plus_status == PASS)
            passed += int(ok)

            row = {
                "task_id": task_id,
                "entry_point": problem["entry_point"],
                "passed": ok,
                "base_status": base_status,
                "plus_status": plus_status,
                "seconds": round(gen_s, 3),
                "model": args.model,
                "adapter_path": args.adapter_path,
                "suite": args.suite,
                "solution_preview": solution[:500],
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {task_id} ({problem['entry_point']}) {gen_s:.2f}s")

    total = len(rows)
    rate = passed / total if total else 0.0
    print(f"Results JSONL: {output_path}")
    print(f"=== Summary: {passed}/{total} ({rate:.0%}) ===")

    if tmp_adapter is not None:
        tmp_adapter.cleanup()
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
