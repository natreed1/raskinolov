#!/usr/bin/env python3
"""
Download (if needed) an MLX Community code model and run one short generation.
Use this to confirm HF access, disk space, and Apple Silicon MLX before LoRA.

Default is 4-bit **7B** (~4 GB weights on disk); allow headroom on unified memory for load, activations, and concurrent MLX/Gradio processes.

Environment:
  MODEL   Hugging Face repo id or local path (default: Qwen2.5-Coder-7B 4-bit).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test MLX base model load + generate")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help="HF repo id or local directory with MLX weights",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    args = parser.parse_args()

    user_msg = (
        "In TypeScript, write a tiny function `clamp(n: number, lo: number, hi: number): number` "
        "that returns n bounded to [lo, hi]. Reply with code only."
    )

    print(f"Loading {args.model!r} …")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    register_qwen_coder_instruct_extra_stops(tokenizer)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    messages = [
        {
            "role": "system",
            "content": "You are Albert, a concise coding assistant for a strategy game codebase (TypeScript).",
        },
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Generating …")
    t1 = time.perf_counter()
    gen_kwargs = {"max_tokens": args.max_tokens}
    if args.temp > 0:
        gen_kwargs["sampler"] = make_sampler(temp=args.temp, top_p=1.0)
    out = generate(model, tokenizer, prompt=prompt, verbose=False, **gen_kwargs)
    elapsed = time.perf_counter() - t1
    print(f"\n--- ({elapsed:.1f}s, max_tokens={args.max_tokens}) ---\n")
    print(out)


if __name__ == "__main__":
    main()
