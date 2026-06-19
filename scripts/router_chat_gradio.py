#!/usr/bin/env python3
"""
Gradio router prompt compare lab: specialist-router lane vs frontier lane.

Each prompt runs:
1) A routed specialist MLX lane (registry LoRA + routing metadata),
2) A frontier OpenAI-compatible lane,
3) Optional bug-check review loops on both outputs,
4) Human grading saved to JSONL for feedback/training.

Backbone controls:
- Auto: normal policy decisions (still local specialist execution),
- Codebase OSS: forces `force_route=local` in specialist lane.

Examples:
  source .venv/bin/activate
  FRONTIER_MODEL=gpt-4o-mini python scripts/router_chat_gradio.py --port 7864
  python scripts/router_chat_gradio.py --compare-log-jsonl benchmarks/results/router_compare_feedback.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

REPO = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import (
    ChatMessage,
    GenerationRequest,
    OpenAICompatibleBackend,
    RoutingPolicy,
    messages_from_prompt,
)
from council_runtime.executor import (
    CouncilGeneration,
    CouncilTurn,
    build_detailed_participant_prompt,
    run_council,
)
from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops
from router.multi_agent import (
    AdapterSubtask,
    build_multi_agent_subtasks,
    merge_multi_agent_outputs,
    summarize_subtask_ids,
)


DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DEFAULT_SYSTEM = (
    "You are a concise assistant. Follow the user's instruction to completion where possible "
    "(use structured Markdown/code fences for UI/game tasks). Avoid stopping mid-section. "
    "For MLX-lab prose-only questions, cite repo paths accurately."
)
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BUG_CHECK_SYSTEM = (
    "You are a strict bug-fix reviewer for code patches. Inspect the candidate output for likely "
    "compile/runtime/import/export/path issues. If you find issues, return a corrected final answer "
    "using the same output shape as the candidate (diff or fenced files). If no issues are found, "
    "return the candidate output unchanged. Do not add commentary."
)
DEFAULT_FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "gpt-4o-mini")
DEFAULT_ARENA_TASKS_PATH = REPO / "benchmarks" / "game_task_arena_examples.json"
DEFAULT_COMPARE_ARTIFACTS_ROOT = REPO / "benchmarks" / "results" / "router_compare_trials"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_registry_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid adapter registry schema: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        aid = str(row.get("adapter_id") or "")
        if aid:
            out[aid] = row
    return out


def _load_arena_prompt_presets(path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    choices: List[str] = ["custom"]
    prompt_map: Dict[str, str] = {"custom": ""}
    title_map: Dict[str, str] = {"custom": "Custom prompt"}
    if not path.is_file():
        return choices, prompt_map, title_map
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return choices, prompt_map, title_map
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return choices, prompt_map, title_map
    for row in tasks:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        if not task_id or not prompt:
            continue
        title_map[task_id] = title or task_id
        choices.append(task_id)
        prompt_map[task_id] = prompt
    return choices, prompt_map, title_map


def _safe_slug(text: str, max_len: int = 48) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_"}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "prompt")[:max_len]


def _render_compare_html(prompt: str, specialist_text: str, frontier_text: str) -> str:
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Router Compare View</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
    }}
    .prompt {{
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 16px;
      white-space: pre-wrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .panel {{
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      min-height: 60vh;
      display: flex;
      flex-direction: column;
    }}
    .title {{
      padding: 10px 12px;
      border-bottom: 1px solid #334155;
      font-weight: 600;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      color: #e2e8f0;
      overflow: auto;
      flex: 1;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Router Compare Output</h2>
    <div class="prompt">{esc(prompt)}</div>
    <div class="grid">
      <section class="panel">
        <div class="title">Specialist output</div>
        <pre>{esc(specialist_text)}</pre>
      </section>
      <section class="panel">
        <div class="title">Frontier output</div>
        <pre>{esc(frontier_text)}</pre>
      </section>
    </div>
  </div>
</body>
</html>
"""


def _write_compare_artifacts(
    *,
    artifacts_root: Path,
    prompt: str,
    specialist_text: str,
    frontier_text: str,
) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_{_safe_slug(prompt)}"
    out_dir = artifacts_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = out_dir / "prompt.txt"
    specialist_path = out_dir / "specialist_output.md"
    frontier_path = out_dir / "frontier_output.md"
    compare_html_path = out_dir / "compare_view.html"
    specialist_html_path = out_dir / "specialist_view.html"
    frontier_html_path = out_dir / "frontier_view.html"

    prompt_path.write_text(prompt, encoding="utf-8")
    specialist_path.write_text(specialist_text, encoding="utf-8")
    frontier_path.write_text(frontier_text, encoding="utf-8")
    compare_html_path.write_text(
        _render_compare_html(prompt, specialist_text, frontier_text),
        encoding="utf-8",
    )
    specialist_html_path.write_text(
        _render_compare_html(prompt, specialist_text, ""),
        encoding="utf-8",
    )
    frontier_html_path.write_text(
        _render_compare_html(prompt, "", frontier_text),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "dir": str(out_dir),
        "compare_html": compare_html_path.as_uri(),
        "specialist_html": specialist_html_path.as_uri(),
        "frontier_html": frontier_html_path.as_uri(),
        "specialist_md": specialist_path.as_uri(),
        "frontier_md": frontier_path.as_uri(),
    }


def _resolve_adapter_abs(repo: Path, adapter_id: str, reg: Dict[str, Dict[str, Any]]) -> Optional[str]:
    entry = reg.get(adapter_id) or {}
    rel = entry.get("adapter_path")
    if not rel:
        return None
    ap = Path(str(rel))
    if not ap.is_absolute():
        ap = (repo / ap).resolve()
    weights = ap / "adapters.safetensors"
    if weights.is_file():
        return str(ap)
    return None


def main() -> None:
    _load_dotenv(REPO / ".env")
    parser = argparse.ArgumentParser(description="Router-aware Gradio chat (MLX adapters)")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO / "training" / "adapter_registry_v1.json",
        help="Adapter registry defining adapter_path per adapter_id",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="Base MLX model id (must match adapters)",
    )
    parser.add_argument("--system-prompt", default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))
    parser.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.0")))
    parser.add_argument(
        "--bug-check-loop",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ROUTER_CHAT_BUG_CHECK_LOOP", "1") not in {"0", "false", "False"},
        help="Run post-route bug-check review loop on generated patch-style outputs.",
    )
    parser.add_argument(
        "--bug-check-rounds",
        type=int,
        default=max(1, int(os.environ.get("ROUTER_CHAT_BUG_CHECK_ROUNDS", "1"))),
        help="Max review rounds for bug-check loop (default 1).",
    )
    parser.add_argument(
        "--bug-check-max-tokens",
        type=int,
        default=max(256, int(os.environ.get("ROUTER_CHAT_BUG_CHECK_MAX_TOKENS", "2048"))),
        help="Decode budget for each bug-check round.",
    )
    parser.add_argument(
        "--bug-check-system-prompt",
        default=os.environ.get("ROUTER_CHAT_BUG_CHECK_SYSTEM_PROMPT", DEFAULT_BUG_CHECK_SYSTEM),
        help="System prompt used by the bug-check reviewer stage.",
    )
    parser.add_argument(
        "--multi-agent-specialist-lane",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ROUTER_CHAT_MULTI_AGENT_LANE", "1") not in {"0", "false", "False"},
        help="Enable multi-agent subtask execution in specialist lane (primary + optional secondary adapter).",
    )
    parser.add_argument(
        "--multi-agent-secondary-min-confidence",
        type=float,
        default=float(os.environ.get("ROUTER_CHAT_MULTI_AGENT_SECONDARY_MIN_CONFIDENCE", "0.2")),
        help="Minimum secondary-adapter confidence required to execute a secondary subtask.",
    )
    parser.add_argument(
        "--multi-agent-merge-max-tokens",
        type=int,
        default=max(512, int(os.environ.get("ROUTER_CHAT_MULTI_AGENT_MERGE_MAX_TOKENS", "2048"))),
        help="Decode budget for final merge pass when multi-agent lane is enabled.",
    )
    parser.add_argument(
        "--council-specialist-lane",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ROUTER_CHAT_COUNCIL_LANE", "1") not in {"0", "false", "False"},
        help="Enable council lane (3 generalist profiles + selected specialists + adjudication).",
    )
    parser.add_argument(
        "--council-debate-max-rounds",
        type=int,
        default=max(1, int(os.environ.get("ROUTER_CHAT_COUNCIL_DEBATE_MAX_ROUNDS", "2"))),
        help="Hard cap for council debate rounds (iterative back-and-forth).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--default-backbone",
        choices=("auto", "codebase_oss"),
        default=os.environ.get("ROUTER_CHAT_DEFAULT_BACKBONE", "auto"),
        help="UI default for Auto vs Codebase OSS (env ROUTER_CHAT_DEFAULT_BACKBONE).",
    )
    parser.add_argument(
        "--default-adapter-lock",
        default=os.environ.get("ROUTER_CHAT_DEFAULT_ADAPTER_LOCK", "auto"),
        help=(
            '`auto` keeps classifier/registry routing; otherwise a registry '
            '`adapter_id` pinned at startup (env ROUTER_CHAT_DEFAULT_ADAPTER_LOCK).'
        ),
    )
    parser.add_argument(
        "--interaction-log-jsonl",
        type=Path,
        default=None,
        help=(
            "Append one JSON line per assistant completion (routing + generation). "
            "If unset, set env ROUTER_CHAT_LOG_JSONL to a path instead."
        ),
    )
    parser.add_argument(
        "--frontier-model",
        default=os.environ.get("FRONTIER_MODEL", DEFAULT_FRONTIER_MODEL),
        help="Frontier model id used for side-by-side comparison lane.",
    )
    parser.add_argument(
        "--frontier-base-url",
        default=os.environ.get("FRONTIER_API_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL for the frontier lane.",
    )
    parser.add_argument(
        "--compare-log-jsonl",
        type=Path,
        default=None,
        help=(
            "Append one JSON line per compare run + saved grade. "
            "If unset, defaults to benchmarks/results/router_compare_feedback.jsonl."
        ),
    )
    parser.add_argument(
        "--compare-artifacts-dir",
        type=Path,
        default=DEFAULT_COMPARE_ARTIFACTS_ROOT,
        help="Directory where per-run compare output artifacts + viewer HTML pages are written.",
    )
    args = parser.parse_args()

    reg_path = args.registry.expanduser().resolve()
    if not reg_path.is_file():
        raise SystemExit(f"Registry missing: {reg_path}")
    reg = _load_registry_entries(reg_path)
    arena_prompt_choices, arena_prompt_map, arena_title_map = _load_arena_prompt_presets(DEFAULT_ARENA_TASKS_PATH)
    arena_quick_choices = [c for c in arena_prompt_choices if c != "custom"]
    print(
        f"[router_chat] Loaded {len(arena_quick_choices)} arena prompt presets from {DEFAULT_ARENA_TASKS_PATH}",
        file=sys.stderr,
    )
    policy = RoutingPolicy()

    adapter_lock_choices = ["auto"] + sorted(reg.keys())
    default_adapter = (args.default_adapter_lock or "").strip() or "auto"
    if default_adapter not in adapter_lock_choices:
        raise SystemExit(
            f"--default-adapter-lock {default_adapter!r} not in registry keys: {sorted(reg.keys())}"
        )

    interaction_log_dest: Optional[Path] = None
    env_log = os.environ.get("ROUTER_CHAT_LOG_JSONL") or ""
    if env_log.strip():
        interaction_log_dest = Path(env_log.strip()).expanduser().resolve()
    elif args.interaction_log_jsonl is not None:
        interaction_log_dest = args.interaction_log_jsonl.expanduser().resolve()
    else:
        # Default-on logging so router prompts are captured as training/eval data.
        interaction_log_dest = (REPO / "benchmarks" / "results" / "router_chat_interactions.jsonl").resolve()
    compare_log_dest = (
        args.compare_log_jsonl.expanduser().resolve()
        if args.compare_log_jsonl is not None
        else (REPO / "benchmarks" / "results" / "router_compare_feedback.jsonl").resolve()
    )
    compare_artifacts_root = args.compare_artifacts_dir.expanduser().resolve()

    gen_kwargs_base: Dict[str, Any] = {"max_tokens": args.max_tokens}
    if args.temp > 0:
        gen_kwargs_base["sampler"] = make_sampler(temp=args.temp, top_p=1.0)

    cache: Dict[str, Any] = {"key": None, "model": None, "tokenizer": None}

    def ensure_mlx_loaded(adapter_abs: Optional[str]) -> Tuple[Any, Any]:
        key = f"{args.model}::{adapter_abs or 'base'}"
        if cache["key"] == key and cache["model"] is not None and cache["tokenizer"] is not None:
            return cache["model"], cache["tokenizer"]
        load_kw: Dict[str, Any] = {}
        if adapter_abs:
            load_kw["adapter_path"] = adapter_abs
        t0 = time.perf_counter()
        model, tokenizer = load(args.model, **load_kw)
        elapsed = time.perf_counter() - t0
        register_qwen_coder_instruct_extra_stops(tokenizer)
        print(f"[router_chat] Loaded {key!r} in {elapsed:.1f}s", file=sys.stderr)
        cache.update(key=key, model=model, tokenizer=tokenizer)
        return model, tokenizer

    def _generate_local_once(
        *,
        model: Any,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        max_tokens_override: Optional[int] = None,
    ) -> Tuple[str, Any]:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        acc = ""
        last_resp: Any = None
        kwargs = dict(gen_kwargs_base)
        if max_tokens_override is not None:
            kwargs["max_tokens"] = int(max_tokens_override)
        for resp in stream_generate(model, tokenizer, rendered, **kwargs):
            last_resp = resp
            acc += resp.text
        return acc, last_resp

    def _run_multi_agent_specialist_lane(
        *,
        prompt: str,
        subtasks: List[AdapterSubtask],
    ) -> Tuple[str, Dict[str, Any]]:
        subtask_outputs: List[Dict[str, Any]] = []
        for subtask in sorted(subtasks, key=lambda item: item.priority):
            adapter_abs = _resolve_adapter_abs(REPO, subtask.adapter_id, reg)
            model, tokenizer = ensure_mlx_loaded(adapter_abs)
            output_text, output_last = _generate_local_once(
                model=model,
                tokenizer=tokenizer,
                messages=_history_to_messages([], subtask.prompt),
            )
            subtask_outputs.append(
                {
                    "adapter_id": subtask.adapter_id,
                    "role": subtask.role,
                    "priority": subtask.priority,
                    "resolved_mlx_adapter": adapter_abs,
                    "text": output_text.strip(),
                    "finish_reason": getattr(output_last, "finish_reason", None) if output_last is not None else None,
                    "generation_tokens": getattr(output_last, "generation_tokens", None) if output_last is not None else None,
                    "prompt_tokens": getattr(output_last, "prompt_tokens", None) if output_last is not None else None,
                }
            )

        merged_seed = merge_multi_agent_outputs(user_prompt=prompt, outputs=subtask_outputs)
        primary = subtask_outputs[0] if subtask_outputs else {}
        merge_adapter_abs = str(primary.get("resolved_mlx_adapter") or "") or None
        merge_model, merge_tokenizer = ensure_mlx_loaded(merge_adapter_abs)
        merge_messages = [
            {"role": "system", "content": args.system_prompt},
            {
                "role": "user",
                "content": (
                    "You are integrating multiple specialist drafts for one user request.\n"
                    "Return a single coherent final answer that combines both specialist contributions.\n"
                    "Do not mention internal orchestration.\n\n"
                    f"Original prompt:\n{prompt}\n\n"
                    f"Specialist drafts:\n{merged_seed}"
                ),
            },
        ]
        merged_text, merged_last = _generate_local_once(
            model=merge_model,
            tokenizer=merge_tokenizer,
            messages=merge_messages,
            max_tokens_override=args.multi_agent_merge_max_tokens,
        )
        merge_meta = {
            "mode": "multi_agent_subtasks",
            "subtasks": subtask_outputs,
            "subtask_ids": summarize_subtask_ids(subtasks),
            "merge_adapter_path": merge_adapter_abs,
            "merge_finish_reason": getattr(merged_last, "finish_reason", None) if merged_last is not None else None,
            "merge_generation_tokens": getattr(merged_last, "generation_tokens", None) if merged_last is not None else None,
            "merge_prompt_tokens": getattr(merged_last, "prompt_tokens", None) if merged_last is not None else None,
            "merged_seed_preview": merged_seed[:6000],
        }
        return merged_text.strip(), merge_meta

    def _run_council_specialist_lane(
        *,
        prompt: str,
        decision: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        council_plan = dict(getattr(decision, "council_plan", {}) or {})
        participants = list(council_plan.get("participants") or [])
        if not participants:
            return "", {"mode": "council", "error": "empty_council_participants"}
        policy_rounds = max(1, int(council_plan.get("debate_max_rounds", 2) or 2))
        debate_max_rounds = max(1, min(policy_rounds, int(args.council_debate_max_rounds)))

        def generate_participant(turn: CouncilTurn) -> CouncilGeneration:
            row = turn.participant
            participant_id = str((row or {}).get("participant_id") or "").strip()
            base_expert_id = str((row or {}).get("base_expert_id") or participant_id).strip()
            participant_type = str((row or {}).get("participant_type") or "").strip()
            adapter_abs: Optional[str] = None
            if participant_type == "specialist_adapter":
                adapter_abs = _resolve_adapter_abs(REPO, base_expert_id, reg)
            model, tokenizer = ensure_mlx_loaded(adapter_abs)
            participant_messages = _history_to_messages([], turn.participant_prompt)
            output_text, output_last = _generate_local_once(
                model=model,
                tokenizer=tokenizer,
                messages=participant_messages,
            )
            adapter_requested = base_expert_id if participant_type == "specialist_adapter" else None
            return CouncilGeneration(
                text=output_text,
                metadata={
                    "confidence": 0.5,
                    "task_outcome_score": 0.5,
                    "adapter_requested": adapter_requested,
                    "resolved_mlx_adapter": adapter_abs,
                    "adapter_loaded": bool(adapter_abs) if participant_type == "specialist_adapter" else False,
                    "finish_reason": getattr(output_last, "finish_reason", None) if output_last is not None else None,
                    "generation_tokens": getattr(output_last, "generation_tokens", None) if output_last is not None else None,
                    "prompt_tokens": getattr(output_last, "prompt_tokens", None) if output_last is not None else None,
                },
            )

        return run_council(
            prompt=prompt,
            council_plan=council_plan,
            disagreement=float(getattr(decision, "council_disagreement", 0.0) or 0.0),
            generate_participant=generate_participant,
            prompt_builder=build_detailed_participant_prompt,
            max_rounds=debate_max_rounds,
            stop_on_convergence=True,
        )

    def _history_to_messages(
        history: List[Tuple[Optional[str], Optional[str]]], user_message: str
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": args.system_prompt}]
        for turn in history:
            u, a = turn[0] or "", turn[1] or ""
            if u.strip():
                msgs.append({"role": "user", "content": u})
            if a.strip():
                msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _run_local_bug_check_loop(
        *,
        model: Any,
        tokenizer: Any,
        original_prompt: str,
        seed_text: str,
    ) -> Tuple[str, bool, int]:
        changed = False
        rounds_run = 0
        review_seed = seed_text.strip()
        if not args.bug_check_loop or not review_seed:
            return seed_text, changed, rounds_run
        for _ in range(max(1, args.bug_check_rounds)):
            rounds_run += 1
            review_messages = [
                {"role": "system", "content": args.bug_check_system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Original user prompt:\n"
                        f"{original_prompt}\n\n"
                        "Candidate output to review:\n"
                        f"{review_seed}\n\n"
                        "Return only the final candidate output."
                    ),
                },
            ]
            review_prompt = tokenizer.apply_chat_template(
                review_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            review_text = ""
            for review_resp in stream_generate(
                model,
                tokenizer,
                review_prompt,
                max_tokens=args.bug_check_max_tokens,
            ):
                review_text += review_resp.text
            review_text = review_text.strip()
            if not review_text or review_text == review_seed:
                break
            changed = True
            review_seed = review_text
        return review_seed, changed, rounds_run

    def _run_frontier_bug_check_loop(
        *,
        backend: OpenAICompatibleBackend,
        original_prompt: str,
        seed_text: str,
    ) -> Tuple[str, bool, int, Dict[str, Any]]:
        changed = False
        rounds_run = 0
        total_usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        current = seed_text.strip()
        if not args.bug_check_loop or not current:
            return seed_text, changed, rounds_run, total_usage
        for _ in range(max(1, args.bug_check_rounds)):
            rounds_run += 1
            review_req = GenerationRequest(
                messages=[
                    ChatMessage("system", args.bug_check_system_prompt),
                    ChatMessage(
                        "user",
                        "Original user prompt:\n"
                        f"{original_prompt}\n\n"
                        "Candidate output to review:\n"
                        f"{current}\n\n"
                        "Return only the final candidate output.",
                    ),
                ],
                max_tokens=args.bug_check_max_tokens,
                temperature=0.0,
            )
            review_text, usage = backend.generate(review_req)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] = int(total_usage.get(key, 0) or 0) + int(usage.get(key, 0) or 0)
            reviewed = review_text.strip()
            if not reviewed or reviewed == current:
                break
            changed = True
            current = reviewed
        return current, changed, rounds_run, total_usage

    def _run_specialist_lane(
        prompt: str,
        backbone_mode: str,
        adapter_lock: str,
    ) -> Dict[str, Any]:
        trimmed = prompt.strip()
        if not trimmed:
            return {"error": "Empty prompt."}
        forced = "local" if backbone_mode == "codebase_oss" else None
        req = GenerationRequest(
            messages=messages_from_prompt(trimmed, args.system_prompt),
            force_route=forced,
        )
        decision = policy.decide(req)
        adapter_id = str(getattr(decision, "adapter_id", "") or "general_fallback")
        secondary_adapter_id = str(getattr(decision, "secondary_adapter_id", "") or "").strip() or None
        secondary_confidence = float(getattr(decision, "secondary_confidence", 0.0) or 0.0)
        coarse_bucket = str(getattr(decision, "coarse_bucket", "unclassified") or "unclassified")
        hierarchy_stage = str(getattr(decision, "hierarchy_stage", "single_stage") or "single_stage")
        candidate_adapters = list(getattr(decision, "candidate_adapters", []) or [])
        decision_reason = str(getattr(decision, "reason", "") or "")
        if adapter_lock and adapter_lock.strip() != "auto":
            lock = adapter_lock.strip()
            adapter_id = lock
            decision_reason = f"{decision_reason} · adapter_lock={lock}" if decision_reason else f"adapter_lock={lock}"

        resolved = _resolve_adapter_abs(REPO, adapter_id, reg)
        if adapter_id != "general_fallback" and not resolved:
            decision_reason = (
                f"{decision_reason} · adapter_weights_missing_for_{adapter_id}_using_base_model"
                if decision_reason
                else f"adapter_weights_missing_for_{adapter_id}_using_base_model"
            )

        mode_tag = ""
        if backbone_mode == "codebase_oss":
            mode_tag = "**Backbone · Codebase OSS** (local MLX forced) · "
        elif adapter_lock and adapter_lock.strip() != "auto":
            mode_tag = f"**Adapter lock** `{adapter_lock.strip()}` · "

        confidence = getattr(decision, "confidence", "n/a")
        hdr = (
            f"{mode_tag}"
            f"**Routing** · route `{decision.route}` · adapter **`{adapter_id}`** · "
            f"conf `{confidence}`\n*{decision_reason}*\n---\n"
        )
        if bool(getattr(decision, "council_enabled", False)):
            council_plan = dict(getattr(decision, "council_plan", {}) or {})
            selected_specialists = list(council_plan.get("selected_specialists") or [])
            hdr += (
                f"**Council** · specialists `{selected_specialists}` · "
                f"disagreement `{getattr(decision, 'council_disagreement', 0.0)}` · "
                f"escalation_candidate `{bool(getattr(decision, 'council_escalation_candidate', False))}`\n---\n"
            )
        if decision.route != "local":
            hdr += (
                "(Policy requests frontier/hybrid; running local MLX preview with adapter/base below.)\n"
            )

        mlx_adapter: Optional[str] = resolved if resolved else None
        model, tokenizer = ensure_mlx_loaded(mlx_adapter)
        acc = ""
        last_resp: Any = None
        trunc_banner = ""
        multi_agent_meta: Dict[str, Any] = {"mode": "single_agent"}
        routing_meta = {
            "adapter_id": adapter_id,
            "secondary_adapter_id": secondary_adapter_id,
            "secondary_confidence": secondary_confidence,
            "coarse_bucket": coarse_bucket,
            "candidate_adapters": candidate_adapters,
            "hierarchy_stage": hierarchy_stage,
            "policy_route": decision.route,
            "confidence": getattr(decision, "confidence", None),
            "ambiguity": getattr(decision, "ambiguity", None),
            "risk_class": getattr(decision, "risk_class", None),
            "complexity": getattr(decision, "complexity", None),
            "reason": decision_reason,
            "resolved_mlx_adapter": mlx_adapter,
            "ui_backbone_mode": backbone_mode,
            "ui_adapter_lock": adapter_lock,
            "multi_agent_lane_enabled": bool(args.multi_agent_specialist_lane),
            "council_lane_enabled": bool(args.council_specialist_lane),
            "council_enabled": bool(getattr(decision, "council_enabled", False)),
            "council_plan": dict(getattr(decision, "council_plan", {}) or {}),
            "council_disagreement": float(getattr(decision, "council_disagreement", 0.0) or 0.0),
            "council_escalation_candidate": bool(
                getattr(decision, "council_escalation_candidate", False)
            ),
        }
        subtasks = build_multi_agent_subtasks(
            prompt=trimmed,
            primary_adapter_id=adapter_id,
            secondary_adapter_id=secondary_adapter_id,
            secondary_confidence=secondary_confidence,
            coarse_bucket=coarse_bucket,
            secondary_min_confidence=args.multi_agent_secondary_min_confidence,
        )
        run_council = bool(
            args.council_specialist_lane
            and bool(getattr(decision, "council_enabled", False))
            and len(list((routing_meta.get("council_plan") or {}).get("participants") or [])) > 0
        )
        run_multi_agent = bool(args.multi_agent_specialist_lane and len(subtasks) > 1)
        if run_council:
            acc, multi_agent_meta = _run_council_specialist_lane(prompt=trimmed, decision=decision)
            routing_meta["council_participant_ids"] = [
                str((row or {}).get("participant_id") or "")
                for row in list((routing_meta.get("council_plan") or {}).get("participants") or [])
                if str((row or {}).get("participant_id") or "")
            ]
            last_resp = None
        elif run_multi_agent:
            acc, multi_agent_meta = _run_multi_agent_specialist_lane(prompt=trimmed, subtasks=subtasks)
            routing_meta["multi_agent_subtask_ids"] = summarize_subtask_ids(subtasks)
            last_resp = None
        else:
            full_messages = _history_to_messages([], trimmed)
            acc, last_resp = _generate_local_once(
                model=model,
                tokenizer=tokenizer,
                messages=full_messages,
            )
        if last_resp is not None and getattr(last_resp, "finish_reason", None) == "length":
            trunc_banner = (
                f"\n\n---\n*Stopped at decoding budget (`max_tokens={args.max_tokens}`). "
                "Output may be **incomplete for SFT**. Increase `MAX_TOKENS` or `--max-tokens` and regenerate.*"
            )
        acc_reviewed, bug_check_changed, bug_check_rounds_run = _run_local_bug_check_loop(
            model=model,
            tokenizer=tokenizer,
            original_prompt=trimmed,
            seed_text=acc,
        )
        acc = acc_reviewed
        log_generation_plain = acc
        if run_multi_agent:
            finish = multi_agent_meta.get("merge_finish_reason")
            gen_toks = multi_agent_meta.get("merge_generation_tokens")
            prompt_toks = multi_agent_meta.get("merge_prompt_tokens")
        else:
            finish = getattr(last_resp, "finish_reason", None) if last_resp is not None else None
            gen_toks = getattr(last_resp, "generation_tokens", None) if last_resp is not None else None
            prompt_toks = getattr(last_resp, "prompt_tokens", None) if last_resp is not None else None
        likely_truncated_length = finish == "length" or (
            gen_toks is not None and gen_toks >= args.max_tokens
        )

        def _unbalanced_code_fence(body: str) -> bool:
            return body.count("```") % 2 == 1

        sloppy_fence = _unbalanced_code_fence(log_generation_plain)
        likely_incomplete = bool(likely_truncated_length or sloppy_fence)
        assistant_eligible_for_sft = (
            finish == "stop"
            and bool(log_generation_plain.strip())
            and not sloppy_fence
        )

        if interaction_log_dest is not None:
            log_row = {
                "schema_version": "router_chat_supervisor_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tool": "router_chat_gradio",
                "user_message": trimmed,
                "system_prompt": args.system_prompt,
                "routing": routing_meta,
                "multi_agent": multi_agent_meta,
                "routing_banner_markdown": hdr,
                "generation_text_only": log_generation_plain,
                "ui_truncation_notice_markdown": trunc_banner.strip() or None,
                "combined_markdown_chat_bubble_approx": hdr + acc + trunc_banner,
                "mlx_model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temp,
                "mlx_finish_reason": finish,
                "mlx_generation_tokens": gen_toks,
                "mlx_prompt_tokens": prompt_toks,
                "generation_budget_hit": finish == "length",
                "unbalanced_markdown_fence": sloppy_fence,
                "likely_incomplete_generation": likely_incomplete,
                "recommended_for_sft_assistant_turn": assistant_eligible_for_sft,
                "bug_check_loop_enabled": bool(args.bug_check_loop),
                "bug_check_rounds_run": bug_check_rounds_run,
                "bug_check_changed_output": bug_check_changed,
                "lane": "specialist_router",
            }
            try:
                interaction_log_dest.parent.mkdir(parents=True, exist_ok=True)
                with interaction_log_dest.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"[router_chat] interaction log append failed: {exc}", file=sys.stderr)
        return {
            "ok": True,
            "text": acc + trunc_banner,
            "header": hdr,
            "routing": routing_meta,
            "multi_agent": multi_agent_meta,
            "bug_check_rounds_run": bug_check_rounds_run,
            "bug_check_changed_output": bug_check_changed,
            "finish_reason": finish,
            "generation_tokens": gen_toks,
            "prompt_tokens": prompt_toks,
        }

    def _run_frontier_lane(prompt: str, frontier_model: str) -> Dict[str, Any]:
        trimmed = prompt.strip()
        if not trimmed:
            return {"error": "Empty prompt."}
        model_name = frontier_model.strip() or args.frontier_model
        backend = OpenAICompatibleBackend(base_url=args.frontier_base_url, model=model_name)
        req = GenerationRequest(
            messages=messages_from_prompt(trimmed, args.system_prompt),
            max_tokens=args.max_tokens,
            temperature=args.temp,
        )
        try:
            text, usage = backend.generate(req)
        except Exception as exc:
            return {
                "ok": False,
                "text": "",
                "error": f"{type(exc).__name__}: {exc}",
                "model": model_name,
                "usage": {},
                "bug_check_rounds_run": 0,
                "bug_check_changed_output": False,
            }
        reviewed_text, bug_changed, bug_rounds, bug_usage = _run_frontier_bug_check_loop(
            backend=backend,
            original_prompt=trimmed,
            seed_text=text,
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(usage.get(key, 0) or 0) + int(bug_usage.get(key, 0) or 0)
        return {
            "ok": True,
            "text": reviewed_text,
            "error": "",
            "model": model_name,
            "usage": usage,
            "bug_check_rounds_run": bug_rounds,
            "bug_check_changed_output": bug_changed,
        }

    def run_compare(
        prompt: str,
        backbone_mode: str,
        adapter_lock: str,
        frontier_model: str,
    ) -> Tuple[str, str, str, str, Dict[str, Any]]:
        trimmed = prompt.strip()
        if not trimmed:
            empty_row: Dict[str, Any] = {}
            return "", "", "Enter a prompt.", "", empty_row
        specialist = _run_specialist_lane(trimmed, backbone_mode, adapter_lock)
        frontier = _run_frontier_lane(trimmed, frontier_model)
        routing_md = specialist.get("header", "_Router decision unavailable._")
        specialist_text = specialist.get("text", "")
        frontier_text = frontier.get("text", "")
        status_lines = [
            f"- Specialist bug-check: rounds `{specialist.get('bug_check_rounds_run', 0)}` · changed `{specialist.get('bug_check_changed_output', False)}`",
            f"- Frontier bug-check: rounds `{frontier.get('bug_check_rounds_run', 0)}` · changed `{frontier.get('bug_check_changed_output', False)}`",
        ]
        if frontier.get("ok") is False:
            status_lines.append(
                f"- Frontier error: `{frontier.get('error', 'unknown frontier failure')}`"
            )
        compare_row = {
            "schema_version": "router_compare_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": trimmed,
            "system_prompt": args.system_prompt,
            "specialist": specialist,
            "frontier": frontier,
            "ui_backbone_mode": backbone_mode,
            "ui_adapter_lock": adapter_lock,
            "frontier_model": frontier.get("model", frontier_model.strip() or args.frontier_model),
        }
        artifacts = _write_compare_artifacts(
            artifacts_root=compare_artifacts_root,
            prompt=trimmed,
            specialist_text=specialist_text,
            frontier_text=frontier_text,
        )
        compare_row["artifacts"] = artifacts
        status_lines.extend(
            [
                f"- Compare view: [open side-by-side]({artifacts['compare_html']})",
                f"- Specialist view: [open]({artifacts['specialist_html']})",
                f"- Frontier view: [open]({artifacts['frontier_html']})",
            ]
        )
        try:
            compare_log_dest.parent.mkdir(parents=True, exist_ok=True)
            with compare_log_dest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**compare_row, "event": "compare_run"}, ensure_ascii=False) + "\n")
        except OSError as exc:
            status_lines.append(f"- Compare log append failed: `{exc}`")
        return routing_md, specialist_text, frontier_text, "\n".join(status_lines), compare_row

    def save_grade(
        compare_row: Dict[str, Any],
        winner: str,
        specialist_score: float,
        frontier_score: float,
        notes: str,
    ) -> str:
        if not compare_row:
            return "Run a compare prompt first."
        grade_row = {
            **compare_row,
            "event": "grade",
            "graded_at": datetime.now(timezone.utc).isoformat(),
            "winner": winner,
            "scores": {
                "specialist": int(specialist_score),
                "frontier": int(frontier_score),
            },
            "notes": (notes or "").strip(),
        }
        try:
            compare_log_dest.parent.mkdir(parents=True, exist_ok=True)
            with compare_log_dest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(grade_row, ensure_ascii=False) + "\n")
        except OSError as exc:
            return f"Grade save failed: {exc}"
        return f"Saved grade to `{compare_log_dest}`"

    details_md = (
        f"Each prompt compares **specialist router lane** and **frontier lane** in one place.\n\n"
        f"- **Specialist registry:** `{reg_path.relative_to(REPO)}`\n"
        f"- **Backbone → Codebase OSS:** forces specialist routing to local MLX.\n"
        f"- **LoRA adapter lock:** pin a specialist instead of classifier pick.\n"
        f"- **Frontier lane model:** UI textbox default `{args.frontier_model}` via `{args.frontier_base_url}`.\n"
        f"- **Bug-check loop:** runs post-generation review on both lanes (default on).\n"
        f"- **Specialist lane log:** `{interaction_log_dest}`\n"
        f"- **Compare + grading log:** `{compare_log_dest}`\n\n"
        f"**Decode budget:** `{args.max_tokens}` new tokens (`MAX_TOKENS` / `--max-tokens`).\n\n"
        f"**Local specialist base model:** `{args.model}`"
    )
    examples_md = (
        "### Example prompts\n\n"
        "- Document SESSION_LOG append-only etiquette for fallen-empire-lora maintainers.\n"
        "- Polish the loading splash typography on the game's start screen mock.\n"
        "- Economy tooltip: net village income after upkeep on the HUD resource ribbon.\n"
    )
    theme_css = """
    .gradio-container {
      background: radial-gradient(circle at top, #f8fbff 0%, #f3f6fb 48%, #eef2f8 100%) !important;
      color: #0f172a !important;
    }
    .gradio-container .block,
    .gradio-container .panel,
    .gradio-container .wrap {
      border-radius: 18px !important;
      border: 1px solid rgba(15, 23, 42, 0.08) !important;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08) !important;
      background: rgba(255, 255, 255, 0.92) !important;
      backdrop-filter: blur(8px);
    }
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
      letter-spacing: -0.02em;
      color: #0f172a !important;
    }
    /* Keep the top hero copy dark without overriding chat/output markdown colors. */
    #router-hero,
    #router-hero p,
    #router-hero h1,
    #router-hero h2,
    #router-hero h3,
    #router-hero span {
      color: #0f172a !important;
      opacity: 1 !important;
      -webkit-text-fill-color: #0f172a !important;
    }
    .gradio-container .accordion button,
    .gradio-container .accordion .label-wrap,
    .gradio-container button.label-wrap,
    .gradio-container .label-wrap,
    .gradio-container .accordion button span,
    .gradio-container .accordion button .label,
    .gradio-container .accordion button .label-wrap span,
    .gradio-container .accordion .icon,
    .gradio-container .accordion .accordion-header button,
    .gradio-container .accordion .accordion-header button span {
      color: #0f172a !important;
      opacity: 1 !important;
      -webkit-text-fill-color: #0f172a !important;
    }
    .gradio-container .block label,
    .gradio-container .block .label-wrap,
    .gradio-container .block .label-wrap span {
      color: #0f172a !important;
      opacity: 1 !important;
      -webkit-text-fill-color: #0f172a !important;
    }
    /* Keep radio pill text readable against dark pill backgrounds. */
    .gradio-container label[data-testid$="-radio-label"],
    .gradio-container label[data-testid$="-radio-label"] span {
      color: #f8fafc !important;
      opacity: 1 !important;
      -webkit-text-fill-color: #f8fafc !important;
    }
    /* Ensure chat text remains visible in the output panel. */
    .gradio-container .message,
    .gradio-container .message .md,
    .gradio-container .message .md p,
    .gradio-container .message .md li,
    .gradio-container .message .md span,
    .gradio-container .message .md strong,
    .gradio-container .message .md em {
      color: #0f172a !important;
      opacity: 1 !important;
      -webkit-text-fill-color: #0f172a !important;
    }
    .gradio-container .message *,
    .gradio-container .message.user *,
    .gradio-container .message.bot * {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      opacity: 1 !important;
    }
    .gradio-container .message.user,
    .gradio-container .message.bot {
      background: rgba(255, 255, 255, 0.98) !important;
    }
    /* Preserve readable contrast inside assistant output/code blocks. */
    .gradio-container .message pre,
    .gradio-container .message code,
    .gradio-container .message .md pre,
    .gradio-container .message .md code {
      color: #e2e8f0 !important;
      -webkit-text-fill-color: #e2e8f0 !important;
    }
    .gradio-container .message pre *,
    .gradio-container .message code * {
      color: #e2e8f0 !important;
      -webkit-text-fill-color: #e2e8f0 !important;
    }
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      caret-color: #0f172a !important;
      background: rgba(255, 255, 255, 0.96) !important;
    }
    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder {
      color: #64748b !important;
    }
    /* Keep dropdown option text visible on light panels. */
    .gradio-container [role="listbox"],
    .gradio-container [role="listbox"] *,
    .gradio-container [role="option"],
    .gradio-container [role="option"] * {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
      opacity: 1 !important;
    }
    .gradio-container button.primary,
    .gradio-container button.primary:hover {
      background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%) !important;
      border: 1px solid #1d4ed8 !important;
      color: #ffffff !important;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24) !important;
    }
    #compare-prompt textarea {
      color: #0f172a !important;
      -webkit-text-fill-color: #0f172a !important;
    }
    """
    with gr.Blocks(title="Router Prompt Lab", css=theme_css) as demo:
        gr.Markdown(
            "## Router Prompt Lab\n"
            "One prompt runs specialist router vs frontier, with side-by-side grading.",
            elem_id="router-hero",
        )
        with gr.Row():
            prompt_tb = gr.Textbox(
                label="Prompt",
                lines=6,
                placeholder="Enter one prompt to compare specialist router vs frontier model.",
                elem_id="compare-prompt",
                scale=7,
            )
            arena_prompt_dd = gr.Dropdown(
                choices=arena_prompt_choices,
                value="custom",
                label="Arena task prompt (task id)",
                info="Pulled from game_task_arena task set; selecting one prefills Prompt.",
                scale=3,
                allow_custom_value=False,
                filterable=True,
            )

        arena_quick_pick = gr.Radio(
            choices=arena_quick_choices,
            value=None,
            label="Arena prompt quick pick",
            info="Visible fallback selector: choose a task id to preview and load its prompt.",
        )

        preset_preview = gr.Textbox(
            label="Selected arena prompt preview",
            lines=5,
            interactive=False,
            value="Select an arena task id to preview its prompt.",
        )
        load_preset_btn = gr.Button("Load selected preset into Prompt")

        def _render_preset_preview(task_id: str) -> str:
            picked = str(task_id or "").strip()
            if not picked or picked == "custom":
                return "Custom prompt selected. Keep typing your own prompt above."
            title = arena_title_map.get(picked, picked)
            prompt = arena_prompt_map.get(picked, "")
            return f"[{picked}] {title}\n\n{prompt}" if prompt else f"[{picked}] {title}\n\n(no prompt found)"

        def _on_arena_prompt_selected(task_id: str) -> str:
            return _render_preset_preview(task_id)

        def _on_arena_quick_selected(task_id: str) -> Tuple[gr.Dropdown, str]:
            picked = str(task_id or "").strip()
            if not picked:
                return gr.Dropdown(value="custom"), _render_preset_preview("custom")
            return gr.Dropdown(value=picked), _render_preset_preview(picked)

        def _load_selected_preset(task_id: str, current_prompt: str) -> str:
            picked = str(task_id or "").strip()
            if not picked or picked == "custom":
                return current_prompt or ""
            return arena_prompt_map.get(picked, current_prompt or "")

        arena_prompt_dd.change(
            _on_arena_prompt_selected,
            inputs=[arena_prompt_dd],
            outputs=[preset_preview],
        )
        arena_quick_pick.change(
            _on_arena_quick_selected,
            inputs=[arena_quick_pick],
            outputs=[arena_prompt_dd, preset_preview],
        )
        load_preset_btn.click(
            _load_selected_preset,
            inputs=[arena_prompt_dd, prompt_tb],
            outputs=[prompt_tb],
        )
        with gr.Accordion("Routing and model controls", open=False):
            backbone = gr.Radio(
                choices=[
                    ("Auto — cost-aware policy (may tag frontier/long prompts)", "auto"),
                    ("Codebase OSS — always force local MLX + registry LoRA", "codebase_oss"),
                ],
                value=args.default_backbone,
                label="Specialist backbone",
                info="Codebase OSS forces local route for specialist lane.",
            )
            adapter_dropdown = gr.Dropdown(
                choices=adapter_lock_choices,
                value=default_adapter,
                label="Specialist LoRA adapter lock",
                info='"auto" uses classifier/registry; otherwise pin a checkpoint from the registry.',
            )
            frontier_model_tb = gr.Textbox(
                label="Frontier model",
                value=args.frontier_model,
            )
        run_btn = gr.Button("Run compare", variant="primary")
        routing_md_out = gr.Markdown(label="Specialist routing decision")
        compare_state = gr.State({})
        with gr.Row():
            specialist_out = gr.Textbox(label="Specialist output", lines=18)
            frontier_out = gr.Textbox(label="Frontier output", lines=18)
        status_md = gr.Markdown()
        run_btn.click(
            run_compare,
            inputs=[prompt_tb, backbone, adapter_dropdown, frontier_model_tb],
            outputs=[routing_md_out, specialist_out, frontier_out, status_md, compare_state],
        )
        with gr.Accordion("Grade this prompt pair", open=True):
            winner = gr.Radio(
                choices=["specialist", "frontier", "tie", "neither"],
                value="tie",
                label="Winner",
            )
            with gr.Row():
                specialist_score = gr.Slider(1, 5, value=3, step=1, label="Specialist score")
                frontier_score = gr.Slider(1, 5, value=3, step=1, label="Frontier score")
            grade_notes = gr.Textbox(label="Notes", lines=3)
            save_grade_btn = gr.Button("Save grade", variant="primary")
            grade_status = gr.Markdown()
            save_grade_btn.click(
                save_grade,
                inputs=[compare_state, winner, specialist_score, frontier_score, grade_notes],
                outputs=[grade_status],
            )
        with gr.Accordion("Details and examples", open=False):
            gr.Markdown(details_md)
            gr.Markdown(examples_md)
    demo.queue()
    print(f"Router supervisor chat: http://{args.host}:{args.port}", file=sys.stderr)
    if interaction_log_dest is not None:
        print(f"[router_chat] Appending supervisor turns → {interaction_log_dest}", file=sys.stderr)
    print(f"[router_chat] Appending compare rows → {compare_log_dest}", file=sys.stderr)
    print(f"[router_chat] Writing compare artifacts → {compare_artifacts_root}", file=sys.stderr)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
