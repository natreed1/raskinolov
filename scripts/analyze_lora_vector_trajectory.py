#!/usr/bin/env python3
"""
Analyze LoRA adapter checkpoint movement as vectors over training time.

This is the vector-level companion to `ml_workflow.py` training trajectories:
it reads saved `*_adapters.safetensors` checkpoints, treats all LoRA tensors as
one ordered vector, and writes norm / delta / cosine trend lines that can feed
dashboards, canvases, or future training controllers.

Example:
  python scripts/analyze_lora_vector_trajectory.py \
    --adapter-path checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m \
    --reference-adapter-file checkpoints/fe-lora-30m/adapters.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PARENT = REPO / "benchmarks" / "results" / "lora_vector_trajectories"
LAYER_RE = re.compile(r"\bmodel\.layers\.(\d+)\.")
ITER_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    label: str
    iteration: Optional[int]


@dataclass
class VectorStats:
    tensors: int = 0
    parameters: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    dot_reference: Optional[float] = None
    dot_first: Optional[float] = None
    dot_prev: Optional[float] = None
    delta_reference_sq: Optional[float] = None
    delta_first_sq: Optional[float] = None
    delta_prev_sq: Optional[float] = None

    @property
    def l2_norm(self) -> float:
        return math.sqrt(max(self.sum_sq, 0.0))

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.parameters if self.parameters else 0.0

    @property
    def rms(self) -> float:
        return math.sqrt(self.sum_sq / self.parameters) if self.parameters else 0.0


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return _repo_rel(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    m = ITER_RE.match(path.name)
    if m:
        return int(m.group(1)), path.name
    return 10**12, path.name


def _discover_checkpoints(adapter_path: Path, include_final: bool) -> List[Checkpoint]:
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path is not a directory: {adapter_path}")

    files = sorted(adapter_path.glob("*_adapters.safetensors"), key=_checkpoint_sort_key)
    numbered = [p for p in files if ITER_RE.match(p.name)]
    selected = numbered or []
    final_path = adapter_path / "adapters.safetensors"
    if include_final and final_path.is_file() and final_path not in selected:
        selected.append(final_path)
    if not selected and final_path.is_file():
        selected = [final_path]
    if not selected:
        raise FileNotFoundError(f"No adapter safetensors found under {adapter_path}")

    checkpoints: List[Checkpoint] = []
    for p in selected:
        m = ITER_RE.match(p.name)
        iteration = int(m.group(1)) if m else None
        label = f"iter_{iteration}" if iteration is not None else p.stem
        checkpoints.append(Checkpoint(path=p, label=label, iteration=iteration))
    return checkpoints


def _tensor_keys(path: Path) -> List[str]:
    with safe_open(path, framework="numpy") as fh:
        return list(fh.keys())


def _norm_lookup(path: Optional[Path], keys: Sequence[str]) -> Optional[float]:
    if path is None:
        return None
    stats = _compare_checkpoint(path, keys, reference=None, first=None, previous=None)[0]
    return stats.l2_norm


def _safe_cos(dot: Optional[float], left_norm: float, right_norm: Optional[float]) -> Optional[float]:
    if dot is None or right_norm is None or left_norm <= 0 or right_norm <= 0:
        return None
    # Tiny float error can push just outside [-1, 1].
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _empty_layer_stats() -> VectorStats:
    return VectorStats(
        dot_reference=0.0,
        dot_first=0.0,
        dot_prev=0.0,
        delta_reference_sq=0.0,
        delta_first_sq=0.0,
        delta_prev_sq=0.0,
    )


def _accumulate_delta(
    stats: VectorStats,
    values: np.ndarray,
    other: Optional[np.ndarray],
    dot_attr: str,
    delta_attr: str,
) -> None:
    if other is None:
        setattr(stats, dot_attr, None)
        setattr(stats, delta_attr, None)
        return
    current_dot = getattr(stats, dot_attr)
    current_delta = getattr(stats, delta_attr)
    if current_dot is not None:
        setattr(stats, dot_attr, current_dot + float(np.sum(values * other, dtype=np.float64)))
    if current_delta is not None:
        diff = values - other
        setattr(stats, delta_attr, current_delta + float(np.sum(diff * diff, dtype=np.float64)))


def _get_optional_tensor(handle: Any, key: str) -> Optional[np.ndarray]:
    if handle is None:
        return None
    return np.asarray(handle.get_tensor(key), dtype=np.float64)


def _compare_checkpoint(
    checkpoint: Path,
    keys: Sequence[str],
    reference: Optional[Path],
    first: Optional[Path],
    previous: Optional[Path],
) -> Tuple[VectorStats, Dict[int, VectorStats]]:
    stats = VectorStats(
        dot_reference=0.0 if reference else None,
        dot_first=0.0 if first else None,
        dot_prev=0.0 if previous else None,
        delta_reference_sq=0.0 if reference else None,
        delta_first_sq=0.0 if first else None,
        delta_prev_sq=0.0 if previous else None,
    )
    layer_stats: Dict[int, VectorStats] = {}

    with safe_open(checkpoint, framework="numpy") as current_fh:
        with safe_open(reference, framework="numpy") if reference else _null_open() as ref_fh:
            with safe_open(first, framework="numpy") if first else _null_open() as first_fh:
                with safe_open(previous, framework="numpy") if previous else _null_open() as prev_fh:
                    for key in keys:
                        values = np.asarray(current_fh.get_tensor(key), dtype=np.float64)
                        stats.tensors += 1
                        stats.parameters += values.size
                        stats.sum_sq += float(np.sum(values * values, dtype=np.float64))
                        stats.sum_abs += float(np.sum(np.abs(values), dtype=np.float64))

                        layer_match = LAYER_RE.search(key)
                        layer_id = int(layer_match.group(1)) if layer_match else -1
                        layer = layer_stats.setdefault(layer_id, _empty_layer_stats())
                        layer.tensors += 1
                        layer.parameters += values.size
                        layer.sum_sq += float(np.sum(values * values, dtype=np.float64))
                        layer.sum_abs += float(np.sum(np.abs(values), dtype=np.float64))

                        ref_values = _get_optional_tensor(ref_fh, key)
                        first_values = _get_optional_tensor(first_fh, key)
                        prev_values = _get_optional_tensor(prev_fh, key)

                        _accumulate_delta(
                            stats,
                            values,
                            ref_values,
                            "dot_reference",
                            "delta_reference_sq",
                        )
                        _accumulate_delta(stats, values, first_values, "dot_first", "delta_first_sq")
                        _accumulate_delta(stats, values, prev_values, "dot_prev", "delta_prev_sq")
                        _accumulate_delta(
                            layer,
                            values,
                            ref_values,
                            "dot_reference",
                            "delta_reference_sq",
                        )
                        _accumulate_delta(layer, values, first_values, "dot_first", "delta_first_sq")
                        _accumulate_delta(layer, values, prev_values, "dot_prev", "delta_prev_sq")

    return stats, layer_stats


class _null_open:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def _stats_row(
    checkpoint: Checkpoint,
    stats: VectorStats,
    reference_norm: Optional[float],
    first_norm: Optional[float],
    prev_norm: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "checkpoint": _repo_rel(checkpoint.path),
        "label": checkpoint.label,
        "iteration": checkpoint.iteration,
        "tensors": stats.tensors,
        "parameters": stats.parameters,
        "l2_norm": stats.l2_norm,
        "mean_abs": stats.mean_abs,
        "rms": stats.rms,
        "delta_from_reference_l2": (
            math.sqrt(max(stats.delta_reference_sq, 0.0))
            if stats.delta_reference_sq is not None
            else None
        ),
        "delta_from_first_l2": (
            math.sqrt(max(stats.delta_first_sq, 0.0)) if stats.delta_first_sq is not None else None
        ),
        "delta_from_prev_l2": (
            math.sqrt(max(stats.delta_prev_sq, 0.0)) if stats.delta_prev_sq is not None else None
        ),
        "cosine_to_reference": _safe_cos(stats.dot_reference, stats.l2_norm, reference_norm),
        "cosine_to_first": _safe_cos(stats.dot_first, stats.l2_norm, first_norm),
        "cosine_to_prev": _safe_cos(stats.dot_prev, stats.l2_norm, prev_norm),
    }
    if extra:
        row.update(extra)
    return row


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_summary(
    out_path: Path,
    adapter_path: Path,
    reference: Optional[Path],
    vector_rows: Sequence[Dict[str, Any]],
    layer_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        f"# LoRA vector trajectory: `{_repo_rel(adapter_path)}`",
        "",
        f"- **Reference adapter:** `{_repo_rel(reference) if reference else 'first checkpoint only'}`",
        f"- **Checkpoints analyzed:** `{len(vector_rows)}`",
        f"- **Layer rows:** `{len(layer_rows)}`",
        "",
        "## Checkpoint Movement",
        "",
        "| Label | Iter | L2 norm | Delta from ref | Delta from prev | Cosine to ref | Cosine to prev |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in vector_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    _fmt(row["iteration"]),
                    _fmt(row["l2_norm"]),
                    _fmt(row["delta_from_reference_l2"]),
                    _fmt(row["delta_from_prev_l2"]),
                    _fmt(row["cosine_to_reference"]),
                    _fmt(row["cosine_to_prev"]),
                ]
            )
            + " |"
        )

    if vector_rows:
        final = vector_rows[-1]
        peak_delta = max(
            vector_rows,
            key=lambda r: r["delta_from_reference_l2"] or r["delta_from_first_l2"] or 0.0,
        )
        lines += [
            "",
            "## Reading",
            "",
            f"- Final checkpoint `{final['label']}` has vector norm `{_fmt(final['l2_norm'])}` and moved `{_fmt(final['delta_from_reference_l2'])}` from the reference adapter.",
            f"- Largest observed movement is `{_fmt(peak_delta['delta_from_reference_l2'] or peak_delta['delta_from_first_l2'])}` at `{peak_delta['label']}`.",
            "- Use `vector_trajectory.jsonl` for whole-adapter trend lines and `layer_trajectory.jsonl` for per-layer movement heatmaps.",
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> Path:
    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_absolute():
        adapter_path = REPO / adapter_path
    reference = Path(args.reference_adapter_file) if args.reference_adapter_file else None
    if reference is not None and not reference.is_absolute():
        reference = REPO / reference

    checkpoints = _discover_checkpoints(adapter_path, include_final=args.include_final)
    keys = _tensor_keys(checkpoints[0].path)
    for checkpoint in checkpoints[1:]:
        other_keys = _tensor_keys(checkpoint.path)
        if other_keys != keys:
            raise ValueError(
                f"Tensor keys differ between {checkpoints[0].path} and {checkpoint.path}; "
                "cannot compare as one stable vector."
            )
    if reference is not None:
        reference_keys = _tensor_keys(reference)
        if reference_keys != keys:
            raise ValueError(
                f"Reference adapter keys differ from checkpoint keys: {reference}"
            )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_PARENT / adapter_path.name
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_norm = _norm_lookup(reference, keys)
    first_norm = _norm_lookup(checkpoints[0].path, keys)
    vector_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    previous_path: Optional[Path] = None
    previous_norm: Optional[float] = None

    for checkpoint in checkpoints:
        stats, layers = _compare_checkpoint(
            checkpoint.path,
            keys,
            reference=reference,
            first=checkpoints[0].path,
            previous=previous_path,
        )
        vector_rows.append(
            _stats_row(
                checkpoint,
                stats,
                reference_norm=reference_norm,
                first_norm=first_norm,
                prev_norm=previous_norm,
            )
        )
        for layer_id, layer_stats in sorted(layers.items()):
            layer_rows.append(
                _stats_row(
                    checkpoint,
                    layer_stats,
                    reference_norm=None,
                    first_norm=None,
                    prev_norm=None,
                    extra={"layer": layer_id if layer_id >= 0 else None},
                )
            )
        previous_path = checkpoint.path
        previous_norm = stats.l2_norm

    _write_jsonl(out_dir / "vector_trajectory.jsonl", vector_rows)
    _write_jsonl(out_dir / "layer_trajectory.jsonl", layer_rows)
    manifest = {
        "adapter_path": _repo_rel(adapter_path),
        "reference_adapter_file": _repo_rel(reference) if reference else None,
        "checkpoints": [_repo_rel(c.path) for c in checkpoints],
        "tensor_count": len(keys),
        "parameter_count": vector_rows[0]["parameters"] if vector_rows else 0,
        "outputs": {
            "vector_trajectory": _repo_rel(out_dir / "vector_trajectory.jsonl"),
            "layer_trajectory": _repo_rel(out_dir / "layer_trajectory.jsonl"),
            "summary": _repo_rel(out_dir / "SUMMARY.md"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_summary(out_dir / "SUMMARY.md", adapter_path, reference, vector_rows, layer_rows)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LoRA checkpoint vector movement over training time."
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Directory containing numbered *_adapters.safetensors checkpoints.",
    )
    parser.add_argument(
        "--reference-adapter-file",
        default=None,
        help="Optional adapter file to measure deltas from, usually the resume source.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUT_PARENT}/<adapter-name>).",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also include adapters.safetensors after numbered checkpoints.",
    )
    args = parser.parse_args()

    out_dir = analyze(args)
    print(f"Wrote LoRA vector trajectory: {_repo_rel(out_dir)}")


if __name__ == "__main__":
    main()
