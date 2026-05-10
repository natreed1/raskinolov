"""MLX + Qwen2.5-Coder-Instruct: extra EOS IDs for mlx_lm streaming.

mlx_lm wraps Hugging Face tokenizers in ``TokenizerWrapper`` whose default
``eos_token_ids`` matches ``<|endoftext|>`` only (id **151643**). Qwen chat
completion uses ``<|im_end|>`` (**151645**) as an assistant terminator;
unless that id is registered with ``tokenizer.add_eos_token``, ``stream_generate``
never stops on it and the UI shows repeated ``…<|…im_end…>…`` gibberish.
"""

from __future__ import annotations

from typing import Any

# Verified on mlx-community/Qwen2.5-Coder-7B-Instruct-4bit + mlx_lm tokenizer.
QWEN25_CODER_INSTRUCT_EXTRA_EOS_IDS: tuple[int, ...] = (
    151645,  # <|im_end|>
)


def register_qwen_coder_instruct_extra_stops(tokenizer: Any) -> None:
    add = getattr(tokenizer, "add_eos_token", None)
    if not callable(add):
        return
    for tid in QWEN25_CODER_INSTRUCT_EXTRA_EOS_IDS:
        try:
            add(str(int(tid)))
        except (TypeError, ValueError):
            continue
