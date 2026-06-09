"""Canonical paths and Hugging Face id for the default Qwen2.5-Coder **7B** MLX lineage.

Import from other `scripts/` modules (`python scripts/...` puts this package on sys.path).
See **docs/DATA_LAYOUT.md** for how exports, JSONL splits, and checkpoints relate.
"""

from __future__ import annotations

from typing import Dict

HF_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# Used in directory names (hyphenated for readability in paths).
LINEAGE_SLUG = "qwen25-coder-7b"

LORA_CONFIG_RELPATH = "training/lora_qwen25_coder_7b.yaml"
GAME_TEXT_DIR_RELPATH = f"data/lora/{LINEAGE_SLUG}/game_text"
DEFAULT_ADAPTER_LATEST_RELPATH = f"checkpoints/fe-lora-{LINEAGE_SLUG}-latest"
TRAIN_UI_ADAPTER_DEFAULT_RELPATH = f"checkpoints/fe-lora-{LINEAGE_SLUG}-ui"

# Current best local arena/eval adapter for the 7B lineage. Keep this 7B-only
# so UI defaults cannot accidentally load a 1.5B adapter against the 7B base.
DEFAULT_ARENA_ADAPTER_RELPATH = f"checkpoints/fe-lora-{LINEAGE_SLUG}-chunk6k-20260428"

# Repeatable arena-acceptance A/B defaults (April 2026). Use the same pairing when
# comparing checkpoints; `--progressive-context off` is for deltas on identical weights only.
DEFAULT_ARENA_PROGRESSIVE_CONTEXT = "auto"
ARENA_ADAPTER_PROGRESSIVE_POLICY: Dict[str, str] = {
    DEFAULT_ARENA_ADAPTER_RELPATH: DEFAULT_ARENA_PROGRESSIVE_CONTEXT,
    f"checkpoints/fe-lora-{LINEAGE_SLUG}-best-val300": DEFAULT_ARENA_PROGRESSIVE_CONTEXT,
}
