#!/usr/bin/env python3
"""Build per-adapter datasets with shared anti-overfit anchor rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DATA_ROOT = REPO / "data" / "lora"
ARENA_BASELINES = REPO / "data" / "arena_task_baselines"
DEFAULT_SHARED = DATA_ROOT / "shared_general_anchor"
DEFAULT_OUT = DATA_ROOT / "adapters"

TASK_TO_ADAPTER = {
    "loading-screen-polish": "loading_screen",
    "hud-status-summary": "hud_status",
    "economy-tooltip": "economy_tooltip",
    "combat-risk-preview": "combat_risk",
    "save-load-api-guard": "save_load_api_guard",
    "ai-planning-explanation": "ai_planning_explanation",
}


def _stable_unit(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _split_rows(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get("task_id") or row.get("record_id") or row.get("id") or json.dumps(row, sort_keys=True))
        u = _stable_unit(seed, key)
        if u < 0.80:
            train.append(row)
        elif u < 0.90:
            valid.append(row)
        else:
            test.append(row)
    if not train and rows:
        train.append(rows[0])
    if not valid and len(rows) > 2:
        valid.append(rows[1])
    if not test and len(rows) > 3:
        test.append(rows[2])
    # mlx_lm local JSONL requires non-empty valid/test subsets; tiny families can land
    # every row in train only → IndexError loading empty splits.
    if rows and not valid:
        valid.append(dict(rows[0]))
    if rows and not test:
        test.append(dict(rows[-1]))
    return train, valid, test


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_shared_anchor(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        src = path / f"{split}.jsonl"
        if not src.is_file():
            continue
        with src.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["dataset_role"] = "shared_anchor"
                out.append(rec)
    return out


def _load_family_rows(sources_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_adapter: Dict[str, List[Dict[str, Any]]] = {}
    for task_id, adapter_id in TASK_TO_ADAPTER.items():
        baseline_path = sources_dir / f"{task_id}.assistant.txt"
        if not baseline_path.is_file():
            continue
        text = baseline_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        row = {
            "messages": [
                {"role": "system", "content": "You are a careful coding assistant."},
                {"role": "user", "content": f"Task `{task_id}`: produce an applyable solution."},
                {"role": "assistant", "content": text},
            ],
            "task_id": task_id,
            "task_type": adapter_id,
            "complexity": "low" if task_id == "loading-screen-polish" else "medium",
            "risk_class": "high" if adapter_id == "save_load_api_guard" else "medium",
            "dataset_role": "family_specific",
            "policy_version": "router_policy_v1",
            "lineage": f"{adapter_id}:baseline",
        }
        by_adapter.setdefault(adapter_id, []).append(row)
    return by_adapter


def _sample_repeated(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    if not rows or target <= 0:
        return []
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < target:
        out.append(dict(rows[i % len(rows)]))
        i += 1
    return out


def _build_adapter_split(
    family_rows: List[Dict[str, Any]],
    shared_rows: List[Dict[str, Any]],
    *,
    seed: int,
    family_ratio: float,
    shared_ratio: float,
    hard_negative_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    train_family, valid_family, test_family = _split_rows(family_rows, seed)
    hard_negative = [
        {**row, "dataset_role": "hard_negative", "risk_class": "high"}
        for row in train_family[: max(1, len(train_family) // 10)]
    ]
    if not train_family:
        train_family = list(family_rows)

    target_family = max(1, len(train_family))
    target_shared = max(1, int(round(target_family * (shared_ratio / max(1e-6, family_ratio)))))
    target_hard = max(1, int(round(target_family * (hard_negative_ratio / max(1e-6, family_ratio)))))
    train_rows = (
        _sample_repeated(train_family, target_family)
        + _sample_repeated(shared_rows, target_shared)
        + _sample_repeated(hard_negative, target_hard)
    )
    return {
        "train": train_rows,
        "valid": valid_family + _sample_repeated(shared_rows, max(1, len(valid_family) // 4)),
        "test": test_family + _sample_repeated(shared_rows, max(1, len(test_family) // 4)),
    }


def build_datasets(
    *,
    sources_dir: Path,
    shared_anchor_dir: Path,
    out_dir: Path,
    seed: int,
    family_ratio: float,
    shared_ratio: float,
    hard_negative_ratio: float,
) -> Dict[str, Any]:
    shared_rows = _load_shared_anchor(shared_anchor_dir)
    family_rows = _load_family_rows(sources_dir)
    written: Dict[str, Any] = {}
    for adapter_id, rows in sorted(family_rows.items()):
        adapter_out = out_dir / adapter_id
        splits = _build_adapter_split(
            rows,
            shared_rows,
            seed=seed,
            family_ratio=family_ratio,
            shared_ratio=shared_ratio,
            hard_negative_ratio=hard_negative_ratio,
        )
        for split, split_rows in splits.items():
            _write_jsonl(adapter_out / f"{split}.jsonl", split_rows)
        manifest = {
            "schema_version": "adapter_dataset_manifest_v1",
            "adapter_id": adapter_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(sources_dir),
            "shared_anchor_dir": str(shared_anchor_dir),
            "policy_version": "router_policy_v1",
            "mix": {
                "family_specific": family_ratio,
                "shared_anchor": shared_ratio,
                "hard_negative": hard_negative_ratio,
            },
            "counts": {name: len(items) for name, items in splits.items()},
            "lineage": f"{adapter_id}:dataset:v1",
        }
        (adapter_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written[adapter_id] = manifest
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-adapter datasets with shared anchor rows.")
    ap.add_argument("--sources-dir", type=Path, default=ARENA_BASELINES)
    ap.add_argument("--shared-anchor-dir", type=Path, default=DEFAULT_SHARED)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-ratio", type=float, default=0.85)
    ap.add_argument("--shared-ratio", type=float, default=0.10)
    ap.add_argument("--hard-negative-ratio", type=float, default=0.05)
    args = ap.parse_args()

    total = args.family_ratio + args.shared_ratio + args.hard_negative_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("Ratios must sum to 1.0")
    written = build_datasets(
        sources_dir=args.sources_dir.expanduser().resolve(),
        shared_anchor_dir=args.shared_anchor_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        seed=args.seed,
        family_ratio=args.family_ratio,
        shared_ratio=args.shared_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    print(json.dumps({"adapters": len(written), "written": written}, indent=2))


if __name__ == "__main__":
    main()

