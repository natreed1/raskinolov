#!/usr/bin/env python3
"""MLX + Qwen2.5-Coder-Instruct: register extra EOS IDs for streaming."""

from __future__ import annotations

from typing import Any

# Qwen chat assistant terminator token.
QWEN25_CODER_INSTRUCT_EXTRA_EOS_IDS: tuple[int, ...] = (151645,)


def register_qwen_coder_instruct_extra_stops(tokenizer: Any) -> None:
    """Best-effort: append `<|im_end|>` to tokenizer EOS set."""
    if tokenizer is None:
        return

    add = getattr(tokenizer, "add_eos_token", None)
    if callable(add):
        for tid in QWEN25_CODER_INSTRUCT_EXTRA_EOS_IDS:
            try:
                add(str(int(tid)))
            except (TypeError, ValueError):
                continue
        return

    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids, list):
        for tid in QWEN25_CODER_INSTRUCT_EXTRA_EOS_IDS:
            if tid not in eos_ids:
                eos_ids.append(tid)
