#!/usr/bin/env python3
"""Hugging Face PEFT LoRA helpers for economistRL Transformers/CUDA paths.

Rollout, eval, and PPO on Lambda use ``PeftModel`` with separate low-rank weights
(``adapter_model.safetensors`` + PEFT ``adapter_config.json``), saved via
``PeftModel.save_pretrained()`` — not MLX-style merge into base ``Linear.weight``.

Legacy MLX checkpoints (``adapters.safetensors`` with ``*.lora_a`` / ``*.lora_b``) are
converted on first load and written beside the MLX files for reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PEFT_ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
MLX_ADAPTER_WEIGHT_NAME = "adapters.safetensors"
MLX_ADAPTER_CONFIG_NAME = "adapter_config.json"


def is_peft_adapter_dir(adapter_dir: Path) -> bool:
    adapter_dir = adapter_dir.expanduser().resolve()
    cfg_path = adapter_dir / MLX_ADAPTER_CONFIG_NAME
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(cfg.get("peft_type") or "").upper() == "LORA" and peft_adapter_weights_file(adapter_dir) is not None


def is_mlx_lora_adapter_dir(adapter_dir: Path) -> bool:
    adapter_dir = adapter_dir.expanduser().resolve()
    return (adapter_dir / MLX_ADAPTER_WEIGHT_NAME).is_file() and not is_peft_adapter_dir(adapter_dir)


def peft_adapter_weights_file(adapter_dir: Path) -> Path | None:
    adapter_dir = adapter_dir.expanduser().resolve()
    for name in PEFT_ADAPTER_WEIGHT_NAMES:
        path = adapter_dir / name
        if path.is_file():
            return path
    return None


def adapter_weights_file(adapter_dir: Path) -> Path | None:
    """Return on-disk LoRA weights path (PEFT preferred, then MLX ``adapters.safetensors``)."""
    adapter_dir = adapter_dir.expanduser().resolve()
    peft = peft_adapter_weights_file(adapter_dir)
    if peft is not None:
        return peft
    mlx = adapter_dir / MLX_ADAPTER_WEIGHT_NAME
    if mlx.is_file():
        return mlx
    return None


def adapter_dir_has_weights(adapter_dir: Path) -> bool:
    return adapter_weights_file(adapter_dir) is not None


def read_mlx_lora_hyperparameters(adapter_dir: Path) -> tuple[float, int, float]:
    """Return (scale, rank, dropout) from MLX-style adapter_config.json."""
    cfg_path = adapter_dir / MLX_ADAPTER_CONFIG_NAME
    if not cfg_path.is_file():
        return 20.0, 16, 0.05
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return 20.0, 16, 0.05
    lora = cfg.get("lora_parameters") if isinstance(cfg, dict) else None
    if not isinstance(lora, dict):
        return 20.0, 16, 0.05
    scale = float(lora.get("scale") or 20.0)
    rank = max(1, int(lora.get("rank") or 16))
    dropout = float(lora.get("dropout") or 0.05)
    return scale, rank, dropout


def infer_target_modules_from_mlx_keys(keys: list[str]) -> list[str]:
    modules: set[str] = set()
    for key in keys:
        if not key.endswith(".lora_a"):
            continue
        stem = key[: -len(".lora_a")]
        module_name = stem.rsplit(".", 1)[-1]
        if module_name:
            modules.add(module_name)
    return sorted(modules)


def mlx_lora_key_to_peft_key(mlx_key: str) -> str:
    key = str(mlx_key or "").strip()
    if key.endswith(".lora_a"):
        suffix = ".lora_A.weight"
        stem = key[: -len(".lora_a")]
    elif key.endswith(".lora_b"):
        suffix = ".lora_B.weight"
        stem = key[: -len(".lora_b")]
    else:
        raise ValueError(f"Not an MLX LoRA key: {mlx_key!r}")
    if stem.startswith("base_model."):
        return f"{stem}{suffix}"
    return f"base_model.model.{stem}{suffix}"


def mlx_lora_state_dict_to_peft(mlx_weights_path: Path) -> dict[str, Any]:
    """Convert MLX ``adapters.safetensors`` tensors to a PEFT state dict."""
    from safetensors import safe_open

    out: dict[str, Any] = {}
    with safe_open(str(mlx_weights_path), framework="pt") as st:
        for key in st.keys():
            if not (key.endswith(".lora_a") or key.endswith(".lora_b")):
                continue
            tensor = st.get_tensor(key)
            peft_key = mlx_lora_key_to_peft_key(key)
            out[peft_key] = tensor.T.contiguous()
    if not out:
        raise RuntimeError(f"No MLX LoRA tensors found in {mlx_weights_path}")
    return out


def build_peft_adapter_config(
    *,
    base_model_name_or_path: str,
    target_modules: list[str],
    rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> dict[str, Any]:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_name_or_path,
        "r": int(rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": list(target_modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "modules_to_save": None,
        "alpha_pattern": {},
        "rank_pattern": {},
    }


def ensure_peft_adapter_dir(adapter_dir: Path, base_model_name_or_path: str) -> Path:
    """Ensure ``adapter_dir`` contains a PEFT layout; convert MLX once if needed."""
    adapter_dir = adapter_dir.expanduser().resolve()
    if is_peft_adapter_dir(adapter_dir):
        return adapter_dir

    mlx_weights = adapter_dir / MLX_ADAPTER_WEIGHT_NAME
    if not mlx_weights.is_file():
        raise RuntimeError(
            f"Adapter directory has neither PEFT weights nor {MLX_ADAPTER_WEIGHT_NAME}: {adapter_dir}"
        )

    from safetensors.torch import save_file

    from safetensors import safe_open

    with safe_open(str(mlx_weights), framework="pt") as st:
        mlx_keys = list(st.keys())
    peft_state = mlx_lora_state_dict_to_peft(mlx_weights)
    target_modules = infer_target_modules_from_mlx_keys(mlx_keys)

    scale, rank, dropout = read_mlx_lora_hyperparameters(adapter_dir)
    peft_cfg = build_peft_adapter_config(
        base_model_name_or_path=base_model_name_or_path,
        target_modules=target_modules,
        rank=rank,
        lora_alpha=scale,
        lora_dropout=dropout,
    )

    out_weights = adapter_dir / "adapter_model.safetensors"
    save_file(peft_state, str(out_weights))

    # Preserve MLX metadata but record PEFT fields for loaders.
    cfg_path = adapter_dir / MLX_ADAPTER_CONFIG_NAME
    merged: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged.update(existing)
        except Exception:
            pass
    merged.update(peft_cfg)
    cfg_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return adapter_dir


def load_peft_causal_lm(
    model_id: str,
    adapter_dir: Path | None,
    *,
    device: Any,
    dtype: Any,
    trainable: bool = False,
) -> tuple[Any, Any]:
    """Load base causal LM + optional PEFT adapter (converting MLX layout if needed)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(model_id or "").strip()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    base.to(device)

    if adapter_dir is None:
        if trainable:
            raise RuntimeError("trainable=True requires an adapter_dir")
        return base, tokenizer

    resolved = ensure_peft_adapter_dir(adapter_dir.expanduser().resolve(), model_id)
    model = PeftModel.from_pretrained(
        base,
        str(resolved),
        is_trainable=bool(trainable),
    )
    model.to(device)
    if trainable:
        model.train()
    else:
        model.eval()
    return model, tokenizer


def peft_trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def save_peft_adapter(model: Any, output_dir: Path) -> None:
    """Save LoRA adapter using Hugging Face ``save_pretrained()``."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise RuntimeError("Model does not implement save_pretrained (expected PeftModel)")
    model.save_pretrained(str(output_dir), safe_serialization=True)
