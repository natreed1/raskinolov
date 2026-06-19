#!/usr/bin/env python3
"""PPO trainer for economistRL LoRA adapter updates.

Consumes scored rollout batches where each row includes prompt/completion text,
old log-probability summaries, and reward-engine outputs. Updates only LoRA adapter
weights using a clipped surrogate objective.

Log-probability scale: mean log-prob per completion token, computed with the same
encoding path for rollout attach, pre-train old_logprob refresh, and PPO new_logprob.

Training backends:
- MLX (`train_ppo_batch_mlx`): local Apple Silicon default.
- Transformers/CUDA (`train_ppo_batch_transformers`): Lambda / NVIDIA default when
  `LOCAL_BACKEND=transformers` or `PPO_TRAIN_BACKEND=transformers`. Uses Hugging Face
  PEFT (`PeftModel`, `save_pretrained()`); legacy MLX `adapters.safetensors` is converted
  on first load via `economist_rl_peft.ensure_peft_adapter_dir`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
PPO_TRAINED_MARKER = ".economist_rl_ppo_trained"


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2
    ppo_epochs: int = 2
    mini_batch_size: int = 8
    learning_rate: float = 5e-6
    kl_coef: float = 0.02
    max_grad_norm: float = 1.0
    min_reward: float = 0.0
    max_reward: float = 1.0
    min_samples: int = 4
    max_samples: int | None = 16
    max_logprob_window_tokens: int = 2048
    completion_outlier_cap_ratio: float = 2.5
    logprob_window_strategy: str = "grouped"
    target_logprob_chunk_tokens: int = 512
    # Rollouts attach windowed ``old_logprob`` via ``max_logprob_window_tokens`` so PPO
    # old/new ratios share the same scale. Recomputing at train time is off by default.
    refresh_old_logprobs: bool = False


@dataclass
class PPOSample:
    task_id: str
    prompt: str
    completion: str
    reward: float
    advantage: float
    old_logprob: float
    system_prompt: str = ""
    original_completion_chars: int = 0
    completion_capped: bool = False


@dataclass
class TrainableLoRALayer:
    name: str
    lora_a: Any
    lora_b: Any
    factor: float
    restore_forward: Callable[..., Any]


@dataclass(frozen=True)
class CompletionLogprobWindow:
    input_ids: list[int]
    target_pos: int
    target_token_id: int


@dataclass(frozen=True)
class GroupedCompletionLogprobWindow:
    input_ids: list[int]
    target_positions: list[int]
    target_token_ids: list[int]
    completion_start: int


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def proxy_old_logprob(completion: str) -> float:
    """Fallback mean log-prob per token (same scale as `_mean_completion_logprob_*`)."""
    _ = completion
    return -0.35


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int((ordered[mid - 1] + ordered[mid]) / 2)


def _completion_outlier_char_cap(completions: list[str], cfg: PPOConfig) -> int | None:
    ratio = float(cfg.completion_outlier_cap_ratio or 0.0)
    if ratio <= 0.0 or len(completions) < 3:
        return None
    lengths = [len(completion) for completion in completions if completion]
    if len(lengths) < 3:
        return None
    median_len = _median_int(lengths)
    if median_len <= 0:
        return None
    return max(median_len, int(math.ceil(median_len * ratio)))


def candidate_staging_dir(candidate_adapter: Path) -> Path:
    return candidate_adapter.with_name(f"{candidate_adapter.name}.staging")


def discard_candidate_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def promote_candidate_staging(staging: Path, candidate_adapter: Path) -> None:
    discard_candidate_staging(candidate_adapter)
    if not staging.exists():
        raise RuntimeError(f"PPO staging adapter missing: {staging}")
    staging.rename(candidate_adapter)


def mark_candidate_ppo_trained(candidate_adapter: Path, *, manifest: dict[str, Any] | None = None) -> None:
    payload = {"status": "trained", "candidate_adapter": str(candidate_adapter.resolve())}
    if manifest:
        payload["train_backend"] = manifest.get("train_backend")
        payload["source_adapter"] = manifest.get("source_adapter")
    marker = candidate_adapter / PPO_TRAINED_MARKER
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def candidate_has_ppo_training_marker(adapter_dir: Path) -> bool:
    return (adapter_dir / PPO_TRAINED_MARKER).is_file()


def collect_old_logprob_proxy_errors(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for row in rows:
        if row.get("old_logprob_source") != "proxy_error":
            continue
        errors.append(
            {
                "task_id": str(row.get("task_id") or ""),
                "error": str(row.get("old_logprob_error") or "unknown"),
            }
        )
    return errors


def _encode_prompt_and_completion(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
) -> tuple[list[int], list[int]]:
    rendered = _render_prompt(tokenizer, prompt, system_prompt)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    return prompt_ids, completion_ids


def _build_completion_logprob_windows(
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    max_window_tokens: int,
) -> list[CompletionLogprobWindow]:
    """Build bounded causal-LM windows for completion-token logprobs.

    Each window contains the preceding context plus the target token. The logit at
    ``target_pos`` predicts ``target_token_id`` while keeping the model forward
    length at or below ``max_window_tokens``.
    """
    if max_window_tokens < 2:
        raise ValueError("max_window_tokens must be >= 2 for causal logprob windows")
    windows: list[CompletionLogprobWindow] = []
    max_context_tokens = max_window_tokens - 1
    for idx, token_id in enumerate(completion_ids):
        context_ids = prompt_ids + completion_ids[:idx]
        if not context_ids:
            continue
        context_tail = context_ids[-max_context_tokens:]
        input_ids = context_tail + [token_id]
        windows.append(
            CompletionLogprobWindow(
                input_ids=input_ids,
                target_pos=len(context_tail) - 1,
                target_token_id=token_id,
            )
        )
    return windows


def _normalize_logprob_window_strategy(strategy: str | None) -> str:
    value = str(strategy or "grouped").strip().lower().replace("-", "_")
    if value not in {"grouped", "per_token"}:
        raise ValueError(f"Unsupported logprob_window_strategy: {strategy!r}")
    return value


def _build_grouped_completion_logprob_windows(
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    max_window_tokens: int,
    target_chunk_tokens: int,
) -> list[GroupedCompletionLogprobWindow]:
    """Build bounded windows that score contiguous completion spans.

    This is an optimized approximation when the full sequence exceeds the
    window. It uses normal causal masking inside each grouped forward pass.
    """
    if max_window_tokens < 2:
        raise ValueError("max_window_tokens must be >= 2 for grouped causal logprob windows")
    chunk_size = max(1, min(int(target_chunk_tokens or 1), max_window_tokens - 1))
    windows: list[GroupedCompletionLogprobWindow] = []
    start = 0
    while start < len(completion_ids):
        context_ids = prompt_ids + completion_ids[:start]
        if not context_ids:
            start += 1
            continue
        remaining = len(completion_ids) - start
        span_len = min(chunk_size, remaining)
        max_context_tokens = max_window_tokens - span_len
        while max_context_tokens < 1 and span_len > 1:
            span_len -= 1
            max_context_tokens = max_window_tokens - span_len
        if max_context_tokens < 1:
            break
        context_tail = context_ids[-max_context_tokens:]
        target_ids = completion_ids[start : start + span_len]
        input_ids = context_tail + target_ids
        first_target_pos = len(context_tail) - 1
        target_positions = [first_target_pos + idx for idx in range(len(target_ids))]
        windows.append(
            GroupedCompletionLogprobWindow(
                input_ids=input_ids,
                target_positions=target_positions,
                target_token_ids=target_ids,
                completion_start=start,
            )
        )
        start += span_len
    return windows


def _is_mlx_lora_param_key(key: str) -> bool:
    return key.endswith(".lora_a") or key.endswith(".lora_b")


def _mlx_flat_trainable(model: Any, tree_flatten: Any) -> dict[str, Any]:
    return dict(tree_flatten(model.trainable_parameters()))


def _mlx_extract_lora_flat(model: Any, tree_flatten: Any) -> dict[str, Any]:
    flat = _mlx_flat_trainable(model, tree_flatten)
    return {key: value for key, value in flat.items() if _is_mlx_lora_param_key(key)}


def _mlx_apply_flat_trainable(
    model: Any,
    flat: dict[str, Any],
    *,
    tree_flatten: Any,
    tree_unflatten: Any,
) -> None:
    merged = _mlx_flat_trainable(model, tree_flatten)
    unknown = [key for key in flat if key not in merged]
    if unknown:
        raise KeyError(f"Unknown trainable parameter keys: {unknown[:3]}")
    merged.update(flat)
    model.update(tree_unflatten(list(merged.items())))


def _mlx_save_lora_adapter_weights(
    adapter_dir: Path,
    model: Any,
    *,
    tree_flatten: Any,
    mx_mod: Any,
) -> int:
    lora_weights = _mlx_extract_lora_flat(model, tree_flatten)
    if not lora_weights:
        raise RuntimeError("No LoRA tensors found in model.trainable_parameters()")
    adapter_out = adapter_dir / "adapters.safetensors"
    mx_mod.save_safetensors(str(adapter_out), lora_weights)
    return len(lora_weights)


def _mean_completion_logprob_mlx(
    logits: Any,
    completion_ids: list[int],
    prompt_len: int,
    mx_mod: Any,
) -> Any:
    if not completion_ids:
        return mx_mod.array(proxy_old_logprob(""))
    values = []
    for i, token_id in enumerate(completion_ids):
        pos = prompt_len - 1 + i
        row = logits[pos]
        # log_softmax via logsumexp — stable for MLX autograd (log(softmax) yields NaN grads).
        values.append(row[token_id] - mx_mod.logsumexp(row))
    return mx_mod.sum(mx_mod.stack(values)) / len(values)


def _mean_completion_logprob_torch(
    logits: Any,
    completion_ids: list[int],
    prompt_len: int,
    torch_mod: Any,
) -> Any:
    if not completion_ids:
        device = logits.device if hasattr(logits, "device") else None
        return torch_mod.tensor(proxy_old_logprob(""), device=device)
    values = []
    for i, token_id in enumerate(completion_ids):
        pos = prompt_len - 1 + i
        log_probs = torch_mod.nn.functional.log_softmax(logits[pos], dim=-1)
        values.append(log_probs[token_id])
    return torch_mod.stack(values).mean()


def _mean_grouped_completion_logprob_mlx(
    *,
    model: Any,
    windows: list[GroupedCompletionLogprobWindow],
    mx_mod: Any,
) -> Any:
    values = []
    for window in windows:
        logits = model(mx_mod.array(window.input_ids)[None])[0]
        for pos, token_id in zip(window.target_positions, window.target_token_ids):
            row = logits[pos]
            values.append(row[token_id] - mx_mod.logsumexp(row))
    if values:
        return mx_mod.sum(mx_mod.stack(values)) / len(values)
    return mx_mod.array(proxy_old_logprob(""))


def _mean_grouped_completion_logprob_torch(
    *,
    model: Any,
    windows: list[GroupedCompletionLogprobWindow],
    torch_mod: Any,
    device: Any,
) -> Any:
    values = []
    for window in windows:
        input_ids = torch_mod.tensor([window.input_ids], device=device)
        logits = model(input_ids).logits[0]
        positions = torch_mod.tensor(window.target_positions, device=device)
        target_ids = torch_mod.tensor(window.target_token_ids, device=device).unsqueeze(-1)
        target_logits = logits.index_select(0, positions)
        log_probs = torch_mod.nn.functional.log_softmax(target_logits, dim=-1)
        values.append(log_probs.gather(-1, target_ids).squeeze(-1))
    if values:
        return torch_mod.cat(values).mean()
    return torch_mod.tensor(proxy_old_logprob(""), device=device)


def _mean_completion_logprob_mlx_from_model(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    mx_mod: Any,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> Any:
    prompt_ids, completion_ids = _encode_prompt_and_completion(
        tokenizer,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
    )
    if not completion_ids:
        return mx_mod.array(proxy_old_logprob(completion))
    all_ids = prompt_ids + completion_ids
    if max_window_tokens and len(all_ids) > max_window_tokens:
        strategy = _normalize_logprob_window_strategy(logprob_window_strategy)
        if strategy == "grouped":
            grouped = _build_grouped_completion_logprob_windows(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                max_window_tokens=max_window_tokens,
                target_chunk_tokens=target_logprob_chunk_tokens,
            )
            try:
                return _mean_grouped_completion_logprob_mlx(model=model, windows=grouped, mx_mod=mx_mod)
            except Exception:
                pass
        windows = _build_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=max_window_tokens,
        )
        values = []
        for window in windows:
            logits = model(mx_mod.array(window.input_ids)[None])[0]
            row = logits[window.target_pos]
            values.append(row[window.target_token_id] - mx_mod.logsumexp(row))
        return mx_mod.sum(mx_mod.stack(values)) / len(values) if values else mx_mod.array(proxy_old_logprob(completion))
    logits = model(mx_mod.array(all_ids)[None])[0]
    return _mean_completion_logprob_mlx(logits, completion_ids, len(prompt_ids), mx_mod)


def _mean_completion_logprob_torch_from_model(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    torch_mod: Any,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> Any:
    prompt_ids, completion_ids = _encode_prompt_and_completion(
        tokenizer,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
    )
    if not completion_ids:
        device = next(model.parameters()).device
        return torch_mod.tensor(proxy_old_logprob(completion), device=device)
    all_ids = prompt_ids + completion_ids
    device = next(model.parameters()).device
    if max_window_tokens and len(all_ids) > max_window_tokens:
        strategy = _normalize_logprob_window_strategy(logprob_window_strategy)
        if strategy == "grouped":
            grouped = _build_grouped_completion_logprob_windows(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                max_window_tokens=max_window_tokens,
                target_chunk_tokens=target_logprob_chunk_tokens,
            )
            try:
                return _mean_grouped_completion_logprob_torch(
                    model=model,
                    windows=grouped,
                    torch_mod=torch_mod,
                    device=device,
                )
            except Exception:
                pass
        values = []
        for window in _build_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=max_window_tokens,
        ):
            input_ids = torch_mod.tensor([window.input_ids], device=device)
            logits = model(input_ids).logits[0]
            log_probs = torch_mod.nn.functional.log_softmax(logits[window.target_pos], dim=-1)
            values.append(log_probs[window.target_token_id])
        return torch_mod.stack(values).mean() if values else torch_mod.tensor(proxy_old_logprob(completion), device=device)
    input_ids = torch_mod.tensor([all_ids], device=device)
    logits = model(input_ids).logits[0]
    return _mean_completion_logprob_torch(logits, completion_ids, len(prompt_ids), torch_mod)


def refresh_ppo_old_logprobs_mlx(
    *,
    samples: list[PPOSample],
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> list[PPOSample]:
    import mlx.core as mx  # type: ignore

    refreshed: list[PPOSample] = []
    for sample in samples:
        lp = _mean_completion_logprob_mlx_from_model(
            model,
            tokenizer,
            prompt=sample.prompt,
            completion=sample.completion,
            system_prompt=system_prompt,
            mx_mod=mx,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
        refreshed.append(
            PPOSample(
                task_id=sample.task_id,
                prompt=sample.prompt,
                completion=sample.completion,
                reward=sample.reward,
                advantage=sample.advantage,
                old_logprob=float(lp),
                system_prompt=sample.system_prompt,
                original_completion_chars=sample.original_completion_chars,
                completion_capped=sample.completion_capped,
            )
        )
    return refreshed


def refresh_ppo_old_logprobs_torch(
    *,
    samples: list[PPOSample],
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    torch_mod: Any,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> list[PPOSample]:
    refreshed: list[PPOSample] = []
    with torch_mod.no_grad():
        for sample in samples:
            lp = _mean_completion_logprob_torch_from_model(
                model,
                tokenizer,
                prompt=sample.prompt,
                completion=sample.completion,
                system_prompt=system_prompt,
                torch_mod=torch_mod,
                max_window_tokens=max_window_tokens,
                logprob_window_strategy=logprob_window_strategy,
                target_logprob_chunk_tokens=target_logprob_chunk_tokens,
            )
            refreshed.append(
                PPOSample(
                    task_id=sample.task_id,
                    prompt=sample.prompt,
                    completion=sample.completion,
                    reward=sample.reward,
                    advantage=sample.advantage,
                    old_logprob=float(lp),
                    system_prompt=sample.system_prompt,
                    original_completion_chars=sample.original_completion_chars,
                    completion_capped=sample.completion_capped,
                )
            )
    return refreshed


def ppo_clip_loss_mlx(
    *,
    new_logprob: Any,
    old_logprob: float,
    advantage: float,
    clip_epsilon: float,
    mx_mod: Any,
) -> Any:
    ratio = mx_mod.exp(mx_mod.clip(new_logprob - mx_mod.array(old_logprob), -20.0, 20.0))
    clipped_ratio = mx_mod.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    adv = mx_mod.array(advantage)
    return -mx_mod.minimum(ratio * adv, clipped_ratio * adv)


def normalize_rewards(rewards: list[float], cfg: PPOConfig) -> list[float]:
    if not rewards:
        return []
    clipped = [_clamp(r, cfg.min_reward, cfg.max_reward) for r in rewards]
    mean = sum(clipped) / len(clipped)
    var = sum((r - mean) ** 2 for r in clipped) / max(1, len(clipped))
    std = math.sqrt(var) if var > 1e-9 else 1.0
    return [(r - mean) / std for r in clipped]


def build_ppo_samples(
    *,
    scored_rows: list[dict[str, Any]],
    system_prompt: str,
    cfg: PPOConfig | None = None,
) -> list[PPOSample]:
    """Build PPO samples from scored rollout rows.

    Uses continuous ``score.reward`` only. ``failures[]``, ``diagnostics``,
    ``high_reward`` / ``strict_scorecard_pass`` are ignored for sample selection.
    Rows with ``training_usable=False`` (empty output) are skipped.
    """
    cfg = cfg or PPOConfig()
    rewards: list[float] = []
    raw: list[tuple[str, str, str, float, float]] = []
    for row in scored_rows:
        rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else {}
        score_obj = row.get("score") if isinstance(row.get("score"), dict) else {}
        if score_obj.get("training_usable") is False:
            continue
        task_id = str(row.get("task_id") or rollout.get("task_id") or "")
        prompt = str(rollout.get("generation_prompt") or rollout.get("prompt") or "")
        completion = str(rollout.get("output") or "").strip()
        if not prompt or not completion:
            continue
        reward = float(
            score_obj.get("reward")
            if score_obj.get("reward") is not None
            else float(score_obj.get("score") or 0.0) / 100.0
        )
        old_logprob = float(
            rollout.get("old_logprob") if rollout.get("old_logprob") is not None else proxy_old_logprob(completion)
        )
        rewards.append(reward)
        raw.append((task_id, prompt, completion, reward, old_logprob))
    completion_cap = _completion_outlier_char_cap([item[2] for item in raw], cfg)
    advantages = normalize_rewards(rewards, cfg)
    samples = []
    for (task_id, prompt, completion, reward, old_logprob), advantage in zip(raw, advantages):
        original_completion_chars = len(completion)
        capped_completion = completion
        completion_capped = False
        if completion_cap is not None and original_completion_chars > completion_cap:
            capped_completion = completion[:completion_cap].rstrip()
            completion_capped = capped_completion != completion
        samples.append(
            PPOSample(
                task_id=task_id,
                prompt=prompt,
                completion=capped_completion,
                reward=reward,
                advantage=advantage,
                old_logprob=old_logprob,
                system_prompt=system_prompt,
                original_completion_chars=original_completion_chars,
                completion_capped=completion_capped,
            )
        )
    cap = cfg.max_samples
    if cap is not None and len(samples) > int(cap):
        # Prefer reward spread: keep top/bottom halves for advantage signal.
        ranked = sorted(samples, key=lambda s: s.reward)
        half = max(1, int(cap) // 2)
        keep = ranked[:half] + ranked[-half:]
        if len(keep) < int(cap):
            mid = ranked[len(ranked) // 2 : len(ranked) // 2 + (int(cap) - len(keep))]
            keep.extend(mid)
        samples = keep[: int(cap)]
    return samples


def ppo_clip_loss_scalar(*, new_logprob: float, old_logprob: float, advantage: float, clip_epsilon: float) -> float:
    ratio = math.exp(_clamp(new_logprob - old_logprob, -20.0, 20.0))
    clipped_ratio = _clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -min(ratio * advantage, clipped_ratio * advantage)


def select_ppo_train_backend() -> str:
    forced = (os.environ.get("PPO_TRAIN_BACKEND") or os.environ.get("LOCAL_BACKEND") or "").strip().lower()
    if forced in {"transformers", "torch", "cuda"}:
        return "transformers"
    if forced == "mlx":
        return "mlx"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "transformers"
    except ImportError:
        pass
    try:
        import mlx.core  # type: ignore  # noqa: F401

        return "mlx"
    except ImportError:
        return "transformers"


def _build_manifest_prefix(
    *,
    cfg: PPOConfig,
    samples: list[PPOSample],
    dry_run: bool,
    train_backend: str,
) -> dict[str, Any]:
    rewards = [s.reward for s in samples]
    return {
        "schema_version": "economist_rl_ppo_train_v1",
        "train_mode": "ppo",
        "train_backend": train_backend,
        "samples": len(samples),
        "clip_epsilon": cfg.clip_epsilon,
        "ppo_epochs": cfg.ppo_epochs,
        "learning_rate": cfg.learning_rate,
        "max_logprob_window_tokens": cfg.max_logprob_window_tokens,
        "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
        "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
        "dry_run": dry_run,
        "sample_stats": {
            "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
            "min_reward": round(min(rewards), 4) if rewards else 0.0,
            "max_reward": round(max(rewards), 4) if rewards else 0.0,
            "completion_outlier_cap_ratio": cfg.completion_outlier_cap_ratio,
            "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
            "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
            "completion_capped_samples": sum(1 for sample in samples if sample.completion_capped),
            "max_completion_chars": max((len(sample.completion) for sample in samples), default=0),
            "max_original_completion_chars": max(
                (sample.original_completion_chars for sample in samples),
                default=0,
            ),
        },
        "steps": [],
    }


def _validate_train_inputs(
    *,
    samples: list[PPOSample],
    candidate_adapter: Path,
    cfg: PPOConfig,
    dry_run: bool,
) -> None:
    if len(samples) < cfg.min_samples:
        raise SystemExit(f"PPO requires at least {cfg.min_samples} scored rollout samples; got {len(samples)}.")
    if dry_run:
        return
    if candidate_adapter.name == "seed_bootstrap":
        raise SystemExit("Refusing to overwrite seed_bootstrap.")
    if candidate_adapter.exists() and any(candidate_adapter.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing non-empty candidate adapter: {candidate_adapter}")


def _infer_backend_kind(*, backend_kind: str, inference_backend: Any | None) -> str:
    kind = str(backend_kind or "").strip().lower()
    if kind in {"mlx", "transformers"}:
        return kind
    if inference_backend is not None:
        loaded = str(getattr(inference_backend, "_backend_kind", "") or "").strip().lower()
        if loaded in {"mlx", "transformers"}:
            return loaded
    force = (os.environ.get("LOCAL_BACKEND") or "").strip().lower()
    if force == "transformers":
        return "transformers"
    if force == "mlx":
        return "mlx"
    return "transformers" if force and force not in {"", "mlx"} else "mlx"


def _sequence_logprob_from_loaded(
    *,
    model: Any,
    tokenizer: Any,
    backend_kind: str,
    prompt: str,
    completion: str,
    system_prompt: str,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> float:
    if backend_kind == "mlx":
        import mlx.core as mx  # type: ignore

        lp = _mean_completion_logprob_mlx_from_model(
            model,
            tokenizer,
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            mx_mod=mx,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
        return float(lp)

    import torch

    with torch.no_grad():
        lp = _mean_completion_logprob_torch_from_model(
            model,
            tokenizer,
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            torch_mod=torch,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
    return float(lp)


def _sequence_logprob_from_backend(
    inference_backend: Any,
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> float:
    model, tokenizer = inference_backend._ensure_loaded()
    kind = _infer_backend_kind(backend_kind="", inference_backend=inference_backend)
    return _sequence_logprob_from_loaded(
        model=model,
        tokenizer=tokenizer,
        backend_kind=kind,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
        max_window_tokens=max_window_tokens,
        logprob_window_strategy=logprob_window_strategy,
        target_logprob_chunk_tokens=target_logprob_chunk_tokens,
    )


def attach_old_logprob_to_rollout(
    *,
    rollout_row: dict[str, Any],
    system_prompt: str,
    base_model: str,
    adapter_path: Path | None,
    dry_run: bool = False,
    inference_backend: Any | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    backend_kind: str = "",
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
    strict: bool = False,
) -> dict[str, Any]:
    row = dict(rollout_row)
    output = str(row.get("output") or "")
    prompt = str(row.get("generation_prompt") or row.get("prompt") or "")
    if dry_run or not output.strip():
        row["old_logprob"] = proxy_old_logprob(output)
        row["old_logprob_source"] = "proxy_dry_run" if dry_run else "proxy_empty"
        row["old_logprob_window_strategy"] = _normalize_logprob_window_strategy(logprob_window_strategy)
        row["old_logprob_target_chunk_tokens"] = int(target_logprob_chunk_tokens)
        return row
    try:
        row["old_logprob"] = compute_sequence_logprob(
            prompt=prompt,
            completion=output,
            system_prompt=system_prompt,
            base_model=base_model,
            adapter_path=adapter_path,
            inference_backend=inference_backend,
            model=model,
            tokenizer=tokenizer,
            backend_kind=backend_kind,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
        row["old_logprob_source"] = "computed"
        row["old_logprob_window_strategy"] = _normalize_logprob_window_strategy(logprob_window_strategy)
        row["old_logprob_target_chunk_tokens"] = int(target_logprob_chunk_tokens)
    except Exception as exc:
        row["old_logprob"] = proxy_old_logprob(output)
        row["old_logprob_source"] = "proxy_error"
        row["old_logprob_error"] = f"{type(exc).__name__}: {exc}"
        row["old_logprob_window_strategy"] = _normalize_logprob_window_strategy(logprob_window_strategy)
        row["old_logprob_target_chunk_tokens"] = int(target_logprob_chunk_tokens)
        if strict:
            raise RuntimeError(
                f"old_logprob attach failed for task {row.get('task_id')!r}: {row['old_logprob_error']}"
            ) from exc
    return row


def compute_sequence_logprob(
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    base_model: str,
    adapter_path: Path | None,
    inference_backend: Any | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    backend_kind: str = "",
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> float:
    if model is not None and tokenizer is not None:
        kind = _infer_backend_kind(backend_kind=backend_kind, inference_backend=inference_backend)
        return _sequence_logprob_from_loaded(
            model=model,
            tokenizer=tokenizer,
            backend_kind=kind,
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
    if inference_backend is not None:
        return _sequence_logprob_from_backend(
            inference_backend,
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )

    force = (os.environ.get("LOCAL_BACKEND") or "").strip().lower()
    if force == "transformers" or force not in {"", "mlx"}:
        return _sequence_logprob_transformers(
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            base_model=base_model,
            adapter_path=adapter_path,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
    try:
        return _sequence_logprob_mlx(
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            base_model=base_model,
            adapter_path=adapter_path,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
    except Exception:
        return _sequence_logprob_transformers(
            prompt=prompt,
            completion=completion,
            system_prompt=system_prompt,
            base_model=base_model,
            adapter_path=adapter_path,
            max_window_tokens=max_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )


def _render_prompt(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _transformers_model_id(model_id: str) -> str:
    model = (model_id or "").strip()
    if model.startswith("mlx-community/") and "Qwen2.5-Coder-" in model:
        suffix = model.split("/", 1)[1]
        if suffix.endswith("-4bit"):
            suffix = suffix[: -len("-4bit")]
        return f"Qwen/{suffix}"
    return model


def _should_load_transformers_4bit(model_id: str, device: Any) -> bool:
    model = (model_id or "").strip()
    return str(device).startswith("cuda") and model.startswith("mlx-community/") and model.endswith("-4bit")


def _read_mlx_adapter_scale_rank(adapter_dir: Path) -> tuple[float, int]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return 20.0, 16
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return 20.0, 16
    lora = cfg.get("lora_parameters") if isinstance(cfg, dict) else None
    if not isinstance(lora, dict):
        return 20.0, 16
    scale = float(lora.get("scale") or 20.0)
    rank = int(lora.get("rank") or 16)
    return scale, max(1, rank)


def _module_name_candidates(base_key: str) -> list[str]:
    names = [base_key]
    if base_key.startswith("model.model."):
        names.append("model." + base_key[len("model.model.") :])
    elif base_key.startswith("model."):
        names.append("model.model." + base_key[len("model.") :])
    return names


def _resolve_linear_module(model: Any, base_key: str) -> tuple[str, Any] | tuple[None, None]:
    import torch

    for name in _module_name_candidates(base_key):
        mod: Any = model
        try:
            for part in name.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if isinstance(mod, torch.nn.Linear):
            return name, mod
    return None, None


def _finalize_trained_candidate(
    *,
    staging_adapter: Path,
    candidate_adapter: Path,
    manifest: dict[str, Any],
) -> None:
    promote_candidate_staging(staging_adapter, candidate_adapter)
    mark_candidate_ppo_trained(candidate_adapter, manifest=manifest)


def _attach_trainable_mlx_lora(model: Any, adapter_dir: Path, torch_mod: Any) -> list[TrainableLoRALayer]:
    from safetensors import safe_open

    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.is_file():
        raise RuntimeError(f"Adapter weights not found: {adapter_file}")

    scale, rank = _read_mlx_adapter_scale_rank(adapter_dir)
    factor = float(scale) / float(rank)
    layers: list[TrainableLoRALayer] = []

    with safe_open(str(adapter_file), framework="pt") as st:
        keys = list(st.keys())
        for key in keys:
            if not key.endswith(".lora_a"):
                continue
            base_key = key[: -len(".lora_a")]
            key_b = f"{base_key}.lora_b"
            if key_b not in keys:
                continue
            _, module = _resolve_linear_module(model, base_key)
            if module is None:
                continue

            lora_a = torch_mod.nn.Parameter(st.get_tensor(key).to(dtype=module.weight.dtype, device=module.weight.device))
            lora_b = torch_mod.nn.Parameter(st.get_tensor(key_b).to(dtype=module.weight.dtype, device=module.weight.device))
            original_forward = module.forward

            def forward_with_lora(x, *args, _a=lora_a, _b=lora_b, _factor=factor, _orig=original_forward, **kwargs):
                out = _orig(x, *args, **kwargs)
                if x.ndim == 3:
                    delta = (x @ _a @ _b) * _factor
                else:
                    delta = (x @ _a @ _b) * _factor
                return out + delta

            module.forward = forward_with_lora  # type: ignore[method-assign]
            layers.append(
                TrainableLoRALayer(
                    name=base_key,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    factor=factor,
                    restore_forward=original_forward,
                )
            )

    if not layers:
        raise RuntimeError(f"No trainable LoRA layers attached from adapter: {adapter_file}")
    return layers


def _save_trainable_mlx_lora(adapter_dir: Path, layers: list[TrainableLoRALayer], scale: float, rank: int) -> None:
    from safetensors.torch import save_file

    tensors = {}
    for layer in layers:
        tensors[f"{layer.name}.lora_a"] = layer.lora_a.detach().cpu()
        tensors[f"{layer.name}.lora_b"] = layer.lora_b.detach().cpu()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(adapter_dir / "adapters.safetensors"))
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        cfg_path.write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "lora_parameters": {"rank": rank, "dropout": 0.05, "scale": scale},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _completion_logprob_tensor(
    *,
    model: Any,
    tokenizer: Any,
    sample: PPOSample,
    system_prompt: str,
    torch_mod: Any,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> Any:
    return _mean_completion_logprob_torch_from_model(
        model,
        tokenizer,
        prompt=sample.prompt,
        completion=sample.completion,
        system_prompt=system_prompt,
        torch_mod=torch_mod,
        max_window_tokens=max_window_tokens,
        logprob_window_strategy=logprob_window_strategy,
        target_logprob_chunk_tokens=target_logprob_chunk_tokens,
    )


def _windowed_completion_logprob_windows_for_sample(
    tokenizer: Any,
    *,
    sample: PPOSample,
    system_prompt: str,
    max_window_tokens: int | None,
) -> list[CompletionLogprobWindow]:
    if not max_window_tokens:
        return []
    prompt_ids, completion_ids = _encode_prompt_and_completion(
        tokenizer,
        prompt=sample.prompt,
        completion=sample.completion,
        system_prompt=system_prompt,
    )
    if not completion_ids or len(prompt_ids) + len(completion_ids) <= max_window_tokens:
        return []
    return _build_completion_logprob_windows(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        max_window_tokens=max_window_tokens,
    )


def _grouped_completion_logprob_windows_for_sample(
    tokenizer: Any,
    *,
    sample: PPOSample,
    system_prompt: str,
    max_window_tokens: int | None,
    target_chunk_tokens: int,
) -> list[GroupedCompletionLogprobWindow]:
    if not max_window_tokens:
        return []
    prompt_ids, completion_ids = _encode_prompt_and_completion(
        tokenizer,
        prompt=sample.prompt,
        completion=sample.completion,
        system_prompt=system_prompt,
    )
    if not completion_ids or len(prompt_ids) + len(completion_ids) <= max_window_tokens:
        return []
    return _build_grouped_completion_logprob_windows(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        max_window_tokens=max_window_tokens,
        target_chunk_tokens=target_chunk_tokens,
    )


def _iter_windowed_completion_logprob_tensors(
    *,
    model: Any,
    tokenizer: Any,
    sample: PPOSample,
    system_prompt: str,
    torch_mod: Any,
    max_window_tokens: int,
):
    windows = _windowed_completion_logprob_windows_for_sample(
        tokenizer,
        sample=sample,
        system_prompt=system_prompt,
        max_window_tokens=max_window_tokens,
    )
    device = next(model.parameters()).device
    for window in windows:
        input_ids = torch_mod.tensor([window.input_ids], device=device)
        logits = model(input_ids).logits[0]
        log_probs = torch_mod.nn.functional.log_softmax(logits[window.target_pos], dim=-1)
        yield log_probs[window.target_token_id]


def _ppo_loss_tensor_torch(
    *,
    new_logprob: Any,
    old_logprob: float,
    advantage: float,
    clip_epsilon: float,
    torch_mod: Any,
    device: Any,
) -> Any:
    old_lp = torch_mod.tensor(old_logprob, device=device, dtype=new_logprob.dtype)
    adv = torch_mod.tensor(advantage, device=device, dtype=new_logprob.dtype)
    ratio = torch_mod.exp(torch_mod.clamp(new_logprob - old_lp, -20.0, 20.0))
    clipped = torch_mod.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch_mod.min(ratio * adv, clipped * adv)


def _backward_ppo_sample_transformers(
    *,
    model: Any,
    tokenizer: Any,
    sample: PPOSample,
    system_prompt: str,
    torch_mod: Any,
    cfg: PPOConfig,
    device: Any,
    use_windowed_backward: bool | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> float:
    strategy = _normalize_logprob_window_strategy(cfg.logprob_window_strategy)
    windows = (
        _windowed_completion_logprob_windows_for_sample(
            tokenizer,
            sample=sample,
            system_prompt=system_prompt,
            max_window_tokens=cfg.max_logprob_window_tokens,
        )
        if strategy == "per_token"
        else _grouped_completion_logprob_windows_for_sample(
            tokenizer,
            sample=sample,
            system_prompt=system_prompt,
            max_window_tokens=cfg.max_logprob_window_tokens,
            target_chunk_tokens=cfg.target_logprob_chunk_tokens,
        )
    )
    if use_windowed_backward is False or (use_windowed_backward is None and not windows):
        new_logprob = _completion_logprob_tensor(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            system_prompt=system_prompt,
            torch_mod=torch_mod,
            max_window_tokens=cfg.max_logprob_window_tokens,
            logprob_window_strategy=cfg.logprob_window_strategy,
            target_logprob_chunk_tokens=cfg.target_logprob_chunk_tokens,
        )
        loss = _ppo_loss_tensor_torch(
            new_logprob=new_logprob,
            old_logprob=sample.old_logprob,
            advantage=sample.advantage,
            clip_epsilon=cfg.clip_epsilon,
            torch_mod=torch_mod,
            device=device,
        )
        loss.backward()
        return float(loss.detach().cpu())

    window_count = len(windows)
    if window_count == 0:
        return 0.0
    if strategy == "grouped":
        try:
            new_logprob = _mean_grouped_completion_logprob_torch(
                model=model,
                windows=windows,
                torch_mod=torch_mod,
                device=device,
            )
            loss = _ppo_loss_tensor_torch(
                new_logprob=new_logprob,
                old_logprob=sample.old_logprob,
                advantage=sample.advantage,
                clip_epsilon=cfg.clip_epsilon,
                torch_mod=torch_mod,
                device=device,
            )
            loss.backward()
            return float(loss.detach().cpu())
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "task_id": sample.task_id,
                        "requested_strategy": "grouped",
                        "fallback_strategy": "per_token",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            windows = _windowed_completion_logprob_windows_for_sample(
                tokenizer,
                sample=sample,
                system_prompt=system_prompt,
                max_window_tokens=cfg.max_logprob_window_tokens,
            )
            window_count = len(windows)
            if window_count == 0:
                return 0.0
    loss_total = 0.0
    for new_logprob in _iter_windowed_completion_logprob_tensors(
        model=model,
        tokenizer=tokenizer,
        sample=sample,
        system_prompt=system_prompt,
        torch_mod=torch_mod,
        max_window_tokens=cfg.max_logprob_window_tokens,
    ):
        loss = _ppo_loss_tensor_torch(
            new_logprob=new_logprob,
            old_logprob=sample.old_logprob,
            advantage=sample.advantage,
            clip_epsilon=cfg.clip_epsilon,
            torch_mod=torch_mod,
            device=device,
        )
        scaled_loss = loss * (1.0 / window_count)
        scaled_loss.backward()
        loss_total += float(loss.detach().cpu()) / window_count
    return loss_total


def _sequence_logprob_mlx(
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    base_model: str,
    adapter_path: Path | None,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> float:
    from mlx_lm import load  # type: ignore
    import mlx.core as mx  # type: ignore

    model, tokenizer = load(base_model, adapter_path=str(adapter_path) if adapter_path else None)
    lp = _mean_completion_logprob_mlx_from_model(
        model,
        tokenizer,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
        mx_mod=mx,
        max_window_tokens=max_window_tokens,
        logprob_window_strategy=logprob_window_strategy,
        target_logprob_chunk_tokens=target_logprob_chunk_tokens,
    )
    return float(lp)


def _sequence_logprob_transformers(
    *,
    prompt: str,
    completion: str,
    system_prompt: str,
    base_model: str,
    adapter_path: Path | None,
    max_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> float:
    from model_router import LocalMlxBackend

    backend = LocalMlxBackend(model_id=base_model, adapter_path=str(adapter_path) if adapter_path else None)
    return _sequence_logprob_from_backend(
        backend,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
        max_window_tokens=max_window_tokens,
        logprob_window_strategy=logprob_window_strategy,
        target_logprob_chunk_tokens=target_logprob_chunk_tokens,
    )


def train_ppo_batch(
    *,
    scored_rows: list[dict[str, Any]],
    system_prompt: str,
    base_model: str,
    source_adapter: Path,
    candidate_adapter: Path,
    cfg: PPOConfig | None = None,
    dry_run: bool = False,
    train_backend: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    backend = (train_backend or select_ppo_train_backend()).strip().lower()
    if backend == "transformers":
        return train_ppo_batch_transformers(
            scored_rows=scored_rows,
            system_prompt=system_prompt,
            base_model=base_model,
            source_adapter=source_adapter,
            candidate_adapter=candidate_adapter,
            cfg=cfg,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )
    return train_ppo_batch_mlx(
        scored_rows=scored_rows,
        system_prompt=system_prompt,
        base_model=base_model,
        source_adapter=source_adapter,
        candidate_adapter=candidate_adapter,
        cfg=cfg,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )


def train_ppo_batch_transformers(
    *,
    scored_rows: list[dict[str, Any]],
    system_prompt: str,
    base_model: str,
    source_adapter: Path,
    candidate_adapter: Path,
    cfg: PPOConfig | None = None,
    dry_run: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = cfg or PPOConfig()
    samples = build_ppo_samples(scored_rows=scored_rows, system_prompt=system_prompt, cfg=cfg)
    manifest = _build_manifest_prefix(cfg=cfg, samples=samples, dry_run=dry_run, train_backend="transformers")
    total_batches = math.ceil(len(samples) / max(1, cfg.mini_batch_size)) * max(1, cfg.ppo_epochs)
    if progress_callback:
        progress_callback(
            {
                "phase": "start",
                "status": "running",
                "train_backend": "transformers",
                "samples": len(samples),
                "ppo_epochs": cfg.ppo_epochs,
                "mini_batch_size": cfg.mini_batch_size,
                "total_minibatches": total_batches,
                "max_logprob_window_tokens": cfg.max_logprob_window_tokens,
                "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
                "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
            }
        )
    _validate_train_inputs(samples=samples, candidate_adapter=candidate_adapter, cfg=cfg, dry_run=dry_run)
    if dry_run:
        manifest["status"] = "dry_run_no_weight_update"
        if progress_callback:
            progress_callback({"phase": "complete", "status": manifest["status"], "completed_minibatches": 0})
        return manifest

    import torch
    from torch.optim import AdamW

    from economist_rl_peft import (
        load_peft_causal_lm,
        peft_adapter_weights_file,
        peft_trainable_parameters,
        save_peft_adapter,
    )

    has_weights = (source_adapter / "adapters.safetensors").is_file() or peft_adapter_weights_file(source_adapter)
    if not source_adapter.exists() or not has_weights:
        raise SystemExit(
            f"PPO transformers training requires LoRA weights at {source_adapter} "
            f"(adapters.safetensors or adapter_model.safetensors)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    staging_adapter = candidate_staging_dir(candidate_adapter)
    discard_candidate_staging(staging_adapter)
    staging_adapter.mkdir(parents=True, exist_ok=True)

    model_id = _transformers_model_id(base_model)
    load_in_4bit = _should_load_transformers_4bit(base_model, device)
    try:
        model, tokenizer = load_peft_causal_lm(
            model_id,
            source_adapter,
            device=device,
            dtype=dtype,
            trainable=True,
            load_in_4bit=load_in_4bit,
            require_local_files=(os.environ.get("FE_REQUIRE_LOCAL_HF_CACHE", "") == "1"),
        )
        train_params = peft_trainable_parameters(model)
        if not train_params:
            raise SystemExit(f"No trainable PEFT parameters found for adapter: {source_adapter}")
        optimizer = AdamW(train_params, lr=cfg.learning_rate)
        model.eval()
        if cfg.refresh_old_logprobs:
            samples = refresh_ppo_old_logprobs_torch(
                samples=samples,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                torch_mod=torch,
                max_window_tokens=cfg.max_logprob_window_tokens,
                logprob_window_strategy=cfg.logprob_window_strategy,
                target_logprob_chunk_tokens=cfg.target_logprob_chunk_tokens,
            )

        step_rows: list[dict[str, Any]] = []
        logprob_fallbacks: list[dict[str, Any]] = []
        model.train()
        completed_batches = 0
        for epoch in range(cfg.ppo_epochs):
            for start in range(0, len(samples), cfg.mini_batch_size):
                batch = samples[start : start + cfg.mini_batch_size]
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "minibatch_start",
                            "status": "running",
                            "epoch": epoch + 1,
                            "ppo_epochs": cfg.ppo_epochs,
                            "batch_start": start,
                            "batch_size": len(batch),
                            "completed_minibatches": completed_batches,
                            "total_minibatches": total_batches,
                        }
                    )
                batch_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for sample in batch:
                    batch_loss += _backward_ppo_sample_transformers(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        system_prompt=system_prompt,
                        torch_mod=torch,
                        cfg=cfg,
                        device=device,
                        diagnostics=logprob_fallbacks,
                    )
                torch.nn.utils.clip_grad_norm_(train_params, cfg.max_grad_norm)
                optimizer.step()
                completed_batches += 1
                mean_loss = round(batch_loss / max(1, len(batch)), 6)
                step_rows.append(
                    {
                        "epoch": epoch + 1,
                        "batch_start": start,
                        "batch_size": len(batch),
                        "loss": mean_loss,
                        "device": str(device),
                        "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
                        "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
                    }
                )
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "minibatch_done",
                            "status": "running",
                            "epoch": epoch + 1,
                            "ppo_epochs": cfg.ppo_epochs,
                            "batch_start": start,
                            "batch_size": len(batch),
                            "completed_minibatches": completed_batches,
                            "total_minibatches": total_batches,
                            "loss": mean_loss,
                        }
                    )

        if progress_callback:
            progress_callback(
                {
                    "phase": "save_start",
                    "status": "saving",
                    "completed_minibatches": completed_batches,
                    "total_minibatches": total_batches,
                    "staging_adapter": str(staging_adapter),
                    "candidate_adapter": str(candidate_adapter),
                }
            )
        save_peft_adapter(model, staging_adapter)
        manifest["status"] = "trained"
        manifest["adapter_format"] = "peft"
        manifest["steps"] = step_rows
        manifest["source_adapter"] = str(source_adapter)
        manifest["candidate_adapter"] = str(candidate_adapter)
        manifest["staging_adapter"] = str(staging_adapter)
        manifest["device"] = str(device)
        manifest["load_in_4bit"] = bool(load_in_4bit)
        if logprob_fallbacks:
            manifest["logprob_strategy_fallbacks"] = logprob_fallbacks
        _finalize_trained_candidate(
            staging_adapter=staging_adapter,
            candidate_adapter=candidate_adapter,
            manifest=manifest,
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "complete",
                    "status": "trained",
                    "completed_minibatches": completed_batches,
                    "total_minibatches": total_batches,
                    "candidate_adapter": str(candidate_adapter),
                }
            )
        return manifest
    except Exception:
        discard_candidate_staging(staging_adapter)
        discard_candidate_staging(candidate_adapter)
        raise


def train_ppo_batch_mlx(
    *,
    scored_rows: list[dict[str, Any]],
    system_prompt: str,
    base_model: str,
    source_adapter: Path,
    candidate_adapter: Path,
    cfg: PPOConfig | None = None,
    dry_run: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = cfg or PPOConfig()
    samples = build_ppo_samples(scored_rows=scored_rows, system_prompt=system_prompt, cfg=cfg)
    manifest = _build_manifest_prefix(cfg=cfg, samples=samples, dry_run=dry_run, train_backend="mlx")
    total_batches = math.ceil(len(samples) / max(1, cfg.mini_batch_size)) * max(1, cfg.ppo_epochs)
    if progress_callback:
        progress_callback(
            {
                "phase": "start",
                "status": "running",
                "train_backend": "mlx",
                "samples": len(samples),
                "ppo_epochs": cfg.ppo_epochs,
                "mini_batch_size": cfg.mini_batch_size,
                "total_minibatches": total_batches,
                "max_logprob_window_tokens": cfg.max_logprob_window_tokens,
                "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
                "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
            }
        )
    _validate_train_inputs(samples=samples, candidate_adapter=candidate_adapter, cfg=cfg, dry_run=dry_run)
    if dry_run:
        manifest["status"] = "dry_run_no_weight_update"
        if progress_callback:
            progress_callback({"phase": "complete", "status": manifest["status"], "completed_minibatches": 0})
        return manifest

    try:
        from mlx_lm import load  # type: ignore
        import mlx.core as mx  # type: ignore
        import mlx.optimizers as optim  # type: ignore
        from mlx.utils import tree_flatten, tree_unflatten  # type: ignore
    except ImportError as exc:
        raise SystemExit("PPO MLX training requires mlx and mlx_lm to be installed.") from exc

    staging_adapter = candidate_staging_dir(candidate_adapter)
    discard_candidate_staging(staging_adapter)
    staging_adapter.mkdir(parents=True, exist_ok=True)
    train_adapter = source_adapter if source_adapter.exists() else candidate_adapter
    try:
        model, tokenizer = load(base_model, adapter_path=str(train_adapter if train_adapter.exists() else None))
        lora_param_count = len(_mlx_extract_lora_flat(model, tree_flatten))
        if lora_param_count == 0:
            raise SystemExit("PPO MLX training found no LoRA parameters in model.trainable_parameters().")
        optimizer = optim.AdamW(learning_rate=cfg.learning_rate)
        if cfg.refresh_old_logprobs:
            samples = refresh_ppo_old_logprobs_mlx(
                samples=samples,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_window_tokens=cfg.max_logprob_window_tokens,
                logprob_window_strategy=cfg.logprob_window_strategy,
                target_logprob_chunk_tokens=cfg.target_logprob_chunk_tokens,
            )

        step_rows: list[dict[str, Any]] = []
        completed_batches = 0
        for epoch in range(cfg.ppo_epochs):
            for start in range(0, len(samples), cfg.mini_batch_size):
                batch = samples[start : start + cfg.mini_batch_size]
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "minibatch_start",
                            "status": "running",
                            "epoch": epoch + 1,
                            "ppo_epochs": cfg.ppo_epochs,
                            "batch_start": start,
                            "batch_size": len(batch),
                            "completed_minibatches": completed_batches,
                            "total_minibatches": total_batches,
                        }
                    )
                batch_loss = 0.0
                for sample in batch:
                    lora_params = _mlx_extract_lora_flat(model, tree_flatten)

                    def ppo_loss_fn(lora_params_flat: dict[str, Any]) -> Any:
                        _mlx_apply_flat_trainable(
                            model,
                            lora_params_flat,
                            tree_flatten=tree_flatten,
                            tree_unflatten=tree_unflatten,
                        )
                        new_logprob = _mean_completion_logprob_mlx_from_model(
                            model,
                            tokenizer,
                            prompt=sample.prompt,
                            completion=sample.completion,
                            system_prompt=system_prompt,
                            mx_mod=mx,
                            max_window_tokens=cfg.max_logprob_window_tokens,
                            logprob_window_strategy=cfg.logprob_window_strategy,
                            target_logprob_chunk_tokens=cfg.target_logprob_chunk_tokens,
                        )
                        return ppo_clip_loss_mlx(
                            new_logprob=new_logprob,
                            old_logprob=sample.old_logprob,
                            advantage=sample.advantage,
                            clip_epsilon=cfg.clip_epsilon,
                            mx_mod=mx,
                        )

                    loss, lora_grads = mx.value_and_grad(ppo_loss_fn)(lora_params)
                    # Update LoRA tensors only; zero-grad updates on layernorms/biases corrupt MLX state.
                    optimizer.update(lora_params, lora_grads)
                    _mlx_apply_flat_trainable(
                        model,
                        lora_params,
                        tree_flatten=tree_flatten,
                        tree_unflatten=tree_unflatten,
                    )
                    mx.eval(model.parameters(), optimizer.state)
                    batch_loss += float(loss)
                completed_batches += 1
                mean_loss = round(batch_loss / max(1, len(batch)), 6)
                step_rows.append(
                    {
                        "epoch": epoch + 1,
                        "batch_start": start,
                        "batch_size": len(batch),
                        "loss": mean_loss,
                        "logprob_window_strategy": _normalize_logprob_window_strategy(cfg.logprob_window_strategy),
                        "target_logprob_chunk_tokens": cfg.target_logprob_chunk_tokens,
                    }
                )
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "minibatch_done",
                            "status": "running",
                            "epoch": epoch + 1,
                            "ppo_epochs": cfg.ppo_epochs,
                            "batch_start": start,
                            "batch_size": len(batch),
                            "completed_minibatches": completed_batches,
                            "total_minibatches": total_batches,
                            "loss": mean_loss,
                        }
                    )

        if progress_callback:
            progress_callback(
                {
                    "phase": "save_start",
                    "status": "saving",
                    "completed_minibatches": completed_batches,
                    "total_minibatches": total_batches,
                    "staging_adapter": str(staging_adapter),
                    "candidate_adapter": str(candidate_adapter),
                }
            )
        adapter_out = staging_adapter / "adapters.safetensors"
        saved_tensors = _mlx_save_lora_adapter_weights(
            staging_adapter,
            model,
            tree_flatten=tree_flatten,
            mx_mod=mx,
        )
        manifest["status"] = "trained"
        manifest["steps"] = step_rows
        manifest["source_adapter"] = str(source_adapter)
        manifest["ppo_load_adapter"] = str(train_adapter)
        manifest["candidate_adapter"] = str(candidate_adapter)
        manifest["staging_adapter"] = str(staging_adapter)
        manifest["adapter_file"] = str(adapter_out)
        manifest["saved_lora_tensors"] = saved_tensors
        _finalize_trained_candidate(
            staging_adapter=staging_adapter,
            candidate_adapter=candidate_adapter,
            manifest=manifest,
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "complete",
                    "status": "trained",
                    "completed_minibatches": completed_batches,
                    "total_minibatches": total_batches,
                    "candidate_adapter": str(candidate_adapter),
                }
            )
        return manifest
    except Exception:
        discard_candidate_staging(staging_adapter)
        discard_candidate_staging(candidate_adapter)
        raise


def write_ppo_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
