#!/usr/bin/env python3
"""
Local web UI to chat with an MLX base model (and optional LoRA adapters).

Loads the model once at startup, then streams tokens into a Gradio chat.

Environment:
  MODEL          HF repo id or local path (default: Qwen2.5-Coder-1.5B 4-bit).

Examples:
  python scripts/chat_gradio.py
  python scripts/chat_gradio.py --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit --port 7860
  ADAPTER_PATH=./checkpoints/my-lora python scripts/chat_gradio.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
DEFAULT_SYSTEM = (
    "You are Albert, a concise coding assistant for Fallen Empire, a strategy game written in TypeScript. "
    "Prefer short answers with correct code when asked for implementation."
)
LEGACY_UI_DEPRECATION = (
    "scripts/chat_gradio.py is a legacy UI and is no longer a supported site.\n"
    "Supported sites are:\n"
    "  1) Documentation site: python scripts/private_dashboard_server.py\n"
    "  2) Arena training supervision site: python scripts/game_task_arena.py ui\n"
    "  3) Router prompt site: python scripts/router_chat_gradio.py\n"
    "To run this legacy UI anyway, pass --allow-legacy-ui."
)

if __name__ == "__main__" and "--allow-legacy-ui" not in sys.argv and not any(
    flag in sys.argv for flag in ("-h", "--help")
):
    raise SystemExit(LEGACY_UI_DEPRECATION)

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def _history_to_messages(
    system_prompt: str,
    history: List[Tuple[Optional[str], Optional[str]]],
    user_message: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in history:
        u, a = turn[0] or "", turn[1] or ""
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_message})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio chat UI for mlx-lm models")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="HF repo id or local MLX weights directory",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("ADAPTER_PATH"),
        help="Optional LoRA adapter directory (mlx_lm.load adapter_path)",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM),
        help="System message passed into the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", "1024")),
        help="Max new tokens per reply",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=float(os.environ.get("TEMP", "0.0")),
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=7860, help="Listen port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary Gradio public link (optional)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to tokenizer load if needed",
    )
    parser.add_argument(
        "--allow-legacy-ui",
        action="store_true",
        help="Run this deprecated UI anyway (not a supported site).",
    )
    args = parser.parse_args()
    if not args.allow_legacy_ui:
        raise SystemExit(LEGACY_UI_DEPRECATION)
    print("[deprecated] Running legacy UI: scripts/chat_gradio.py", file=sys.stderr)

    repo_root = Path(__file__).resolve().parent.parent
    adapter_resolved: Optional[str] = None
    if args.adapter_path:
        ap = Path(args.adapter_path)
        if not ap.is_absolute():
            ap = (repo_root / ap).resolve()
        weights = ap / "adapters.safetensors"
        if not weights.is_file():
            raise SystemExit(
                f"Adapter weights not found: {weights}\n"
                "Training may have failed (see benchmarks/results/runs/*/logs/mlx_lm_lora_train.log) "
                "or the path is wrong. Re-run: python scripts/build_lora_dataset.py … && "
                "python scripts/ml_workflow.py train --evaluate …"
            )
        adapter_resolved = str(ap)

    print(f"Loading model {args.model!r} …")
    t0 = time.perf_counter()
    load_kw: dict = {}
    if adapter_resolved:
        load_kw["adapter_path"] = adapter_resolved
    if args.trust_remote_code:
        load_kw["tokenizer_config"] = {"trust_remote_code": True}
    model, tokenizer = load(args.model, **load_kw)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    gen_kwargs: dict = {"max_tokens": args.max_tokens}
    if args.temp > 0:
        gen_kwargs["sampler"] = make_sampler(temp=args.temp, top_p=1.0)

    def respond(
        message: str,
        history: List[Tuple[Optional[str], Optional[str]]],
    ) -> Generator[str, None, None]:
        if not message.strip():
            yield ""
            return
        messages = _history_to_messages(args.system_prompt, history, message)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        accumulated = ""
        for resp in stream_generate(model, tokenizer, prompt, **gen_kwargs):
            accumulated += resp.text
            yield accumulated

    title = "Albert — local MLX chat"
    desc = (
        f"**Model:** `{args.model}`"
        + (f"  \n**Adapter:** `{args.adapter_path}`" if args.adapter_path else "")
        + f"  \n**Max tokens:** {args.max_tokens} · **Temperature:** {args.temp}"
    )

    demo = gr.ChatInterface(
        fn=respond,
        title=title,
        description=desc,
        examples=[
            "Explain how you'd structure a small combat resolver in TypeScript.",
            "Write a function that merges two partial unit stat objects, with later keys winning.",
        ],
        concurrency_limit=1,
    )
    demo.queue()
    print(f"Open http://{args.host}:{args.port} in your browser (Ctrl+C to stop).")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
