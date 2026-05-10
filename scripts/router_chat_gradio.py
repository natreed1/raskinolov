#!/usr/bin/env python3
"""
Gradio supervisor chat: deterministic router preview + MLX generate with routed LoRA.

For each turn, reads ``training/adapter_registry_v1.json`` to resolve ``adapter_path``,
calls ``RoutingPolicy().decide`` on your prompt, optionally reloads MLX weights when the
chosen adapter differs from the prior turn, and streams assistant text prefixed with the
routing header (route / tier / adapter / confidence).

Non-local routes (frontier/hybrid) do not invoke paid APIs here; generation still runs
locally using the routed adapter hint (or base MLX if checkpoint weights are missing).

**Backbone** control: choose **Auto** for cost-aware routing, or **Codebase OSS** to
``force_route=local`` on every turn (always Qwen MLX + registry LoRA, never downgrade to
policy ``frontier`` labels for long/high-risk prompts). Optional **LoRA adapter lock**
overrides classifier adapter selection — use ``auto`` unless you intentionally pin a specialist.

Logged rows (optional JSONL) include ``mlx_finish_reason``, ``recommended_for_sft_assistant_turn``,
``generation_budget_hit``, fenced-Markdown balance — default assistant decode budget **2048** tokens
(``MAX_TOKENS`` env / ``--max-tokens``).

Examples:
  source .venv/bin/activate
  python scripts/router_chat_gradio.py --port 7864
  ROUTER_CHAT_LOG_JSONL=benchmarks/results/router_chat_interactions.jsonl python scripts/router_chat_gradio.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

REPO = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt
from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops


DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DEFAULT_SYSTEM = (
    "You are a concise assistant. Follow the user's instruction to completion where possible "
    "(use structured Markdown/code fences for UI/game tasks). Avoid stopping mid-section. "
    "For MLX-lab prose-only questions, cite repo paths accurately."
)
DEFAULT_MAX_TOKENS = 2048


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
    args = parser.parse_args()

    reg_path = args.registry.expanduser().resolve()
    if not reg_path.is_file():
        raise SystemExit(f"Registry missing: {reg_path}")
    reg = _load_registry_entries(reg_path)
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

    def respond(
        message: str,
        history: List[Tuple[Optional[str], Optional[str]]],
        backbone_mode: str,
        adapter_lock: str,
    ) -> Generator[str, None, None]:
        trimmed = message.strip()
        if not trimmed:
            yield ""
            return
        forced = "local" if backbone_mode == "codebase_oss" else None
        req = GenerationRequest(
            messages=messages_from_prompt(trimmed, args.system_prompt),
            force_route=forced,
        )
        decision = policy.decide(req)
        if adapter_lock and adapter_lock.strip() != "auto":
            lock = adapter_lock.strip()
            decision = replace(
                decision,
                adapter_id=lock,
                reason=f"{decision.reason} · adapter_lock={lock}",
            )
        resolved = _resolve_adapter_abs(REPO, decision.adapter_id, reg)

        mode_tag = ""
        if backbone_mode == "codebase_oss":
            mode_tag = "**Backbone · Codebase OSS** (local MLX forced) · "
        elif adapter_lock and adapter_lock.strip() != "auto":
            mode_tag = f"**Adapter lock** `{adapter_lock.strip()}` · "

        hdr = (
            f"{mode_tag}"
            f"**Routing** · route `{decision.route}` · adapter **`{decision.adapter_id}`** · "
            f"conf `{decision.confidence}`\n*{decision.reason}*\n---\n"
        )
        if decision.route != "local":
            hdr += (
                "(Policy requests frontier/hybrid; running local MLX preview with adapter/base below.)\n"
            )

        mlx_adapter: Optional[str] = resolved if resolved else None

        yield hdr + "(loading weights…)"

        model, tokenizer = ensure_mlx_loaded(mlx_adapter)
        full_messages = _history_to_messages(history, trimmed)

        rendered = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        acc = ""
        last_resp: Any = None
        trunc_banner = ""
        routing_meta = {
            "adapter_id": decision.adapter_id,
            "policy_route": decision.route,
            "confidence": decision.confidence,
            "ambiguity": decision.ambiguity,
            "risk_class": decision.risk_class,
            "complexity": decision.complexity,
            "reason": decision.reason,
            "resolved_mlx_adapter": mlx_adapter,
            "ui_backbone_mode": backbone_mode,
            "ui_adapter_lock": adapter_lock,
        }
        for resp in stream_generate(model, tokenizer, rendered, **gen_kwargs_base):
            last_resp = resp
            acc += resp.text
            yield hdr + acc
        if last_resp is not None and last_resp.finish_reason == "length":
            trunc_banner = (
                f"\n\n---\n*Stopped at decoding budget (`max_tokens={args.max_tokens}`). "
                "Output may be **incomplete for SFT**. Increase `MAX_TOKENS` or `--max-tokens` and regenerate.*"
            )
            yield hdr + acc + trunc_banner
        log_generation_plain = acc
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
            }
            try:
                interaction_log_dest.parent.mkdir(parents=True, exist_ok=True)
                with interaction_log_dest.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"[router_chat] interaction log append failed: {exc}", file=sys.stderr)

    desc = (
        f"Each message runs **offline** `RoutingPolicy` + MLX: specialist weights come from **`{reg_path.relative_to(REPO)}`**.\n\n"
        f"- **Backbone → Codebase OSS:** forces **local** routing every turn (open Qwen + your LoRAs) even when the "
        f"policy would label a prompt `frontier`.\n"
        f"- **LoRA adapter lock:** pin a registry specialist instead of classifier pick.\n"
        f"- **Documentation** / **economy_tooltip** / **loading_screen**: LoRA loads when `adapters.safetensors` exists.\n"
        f"- **general_fallback**: base model unless you add checkpoint weights.\n"
        f"- **frontier/hybrid** policy hints (Auto mode): UI still generates **locally** (no API calls).\n"
        + (
            f"- **Supervisor log:** `ROUTER_CHAT_LOG_JSONL` or `--interaction-log-jsonl` appends labeled JSON rows per assistant reply.\n"
            if interaction_log_dest is not None
            else ""
        )
        + "\n"
        f'**Decode budget (assistant):** `{args.max_tokens}` new tokens — raise `MAX_TOKENS` or `--max-tokens` for '
        'longer economy/HUD/UI completions. Supervisor JSONL sets **`recommended_for_sft_assistant_turn`** when '
        "`mlx_finish_reason` is **`stop`**, prose is non-empty, and fenced Markdown closes (even ``` fence count).\n\n"
        f"**Base model:** `{args.model}`"
    )
    backbone = gr.Radio(
        choices=[
            ("Auto — cost-aware policy (may tag frontier/long prompts)", "auto"),
            ("Codebase OSS — always force local MLX + registry LoRA", "codebase_oss"),
        ],
        value=args.default_backbone,
        label="Backbone",
        info="OSS mode matches GenerationRequest(force_route='local'): open Qwen on Apple Silicon plus your adapters.",
    )
    adapter_dropdown = gr.Dropdown(
        choices=adapter_lock_choices,
        value=default_adapter,
        label="LoRA adapter lock",
        info='"auto" uses classifier/registry; otherwise pin a checkpoint from the registry.',
    )
    demo = gr.ChatInterface(
        fn=respond,
        title="Router supervisor chat (mlx-lora lab)",
        description=desc,
        additional_inputs=[backbone, adapter_dropdown],
        additional_inputs_accordion_name="OSS / routing controls",
        examples=[
            "Document SESSION_LOG append-only etiquette for fallen-empire-lora maintainers.",
            "Polish the loading splash typography on the game's start screen mock.",
            "Economy tooltip: net village income after upkeep on the HUD resource ribbon.",
        ],
        concurrency_limit=1,
    )
    demo.queue()
    print(f"Router supervisor chat: http://{args.host}:{args.port}", file=sys.stderr)
    if interaction_log_dest is not None:
        print(f"[router_chat] Appending supervisor turns → {interaction_log_dest}", file=sys.stderr)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
