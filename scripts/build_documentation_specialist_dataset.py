#!/usr/bin/env python3
"""Build the documentation specialist dataset for the MLX lab repo.

This script is intentionally lightweight and deterministic. It rebuilds
`documentation_specialist` splits from the existing dataset rows when present,
and falls back to a compact in-script seed set when no rows exist yet.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You maintain Markdown technical notes and training-run analysis for "
    "fallen-empire-lora, the MLX/LoRA lab adjacent to Fallen Empire. This "
    "specialist is for the LoRA lab repo, not game feature implementation. "
    "Prefer plain language and exact identifiers from `docs/PROJECT_STATE.md`: "
    "default models, script names, canonical paths (`docs/run_history.md` "
    "append-only, `docs/SESSION_LOG.md` append-only, `benchmarks/results/runs/`, "
    "`training_trajectory.jsonl`, `SOURCE_REPO`, `GAME_ARENA_ROOT`), and "
    "`python scripts/ml_workflow.py` subcommands."
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_rows() -> List[Dict[str, Any]]:
    seed_pairs = [
        (
            "Train with ml workflow full on fe-lora latest 400 iterations chunk 6k.",
            "**Train:** `python scripts/ml_workflow.py full --adapter-path "
            "checkpoints/fe-lora-qwen25-coder-7b-latest -- --iters 400`.",
        ),
        (
            "Run history commits when you iterate ml workflow except dashboard.",
            "Every `python scripts/ml_workflow.py` invocation except "
            "`arena-dashboard` should append `docs/run_history.md` and write "
            "artifacts to `benchmarks/results/runs/<run_id>/`.",
        ),
        (
            "training config filename for qwen 25 coder 7b",
            "LoRA config: `training/lora_qwen25_coder_7b.yaml`.",
        ),
        (
            "where do workflow manifests go",
            "Each run writes `manifest.json` and `RUN.md` under "
            "`benchmarks/results/runs/<run_id>/`.",
        ),
        (
            "what docs are append only",
            "`docs/SESSION_LOG.md` and `docs/run_history.md` are append-only.",
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (prompt, answer) in enumerate(seed_pairs):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "task_id": "mlx-lora-docs-normalize",
                "task_type": "documentation",
                "complexity": "low",
                "risk_class": "low",
                "dataset_role": "core_docs_style",
                "record_id": f"seed-docs:{idx}",
                "lineage": "documentation_specialist:v1",
                "policy_version": "router_policy_v1",
            }
        )
    return rows


def _canonicalize_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return {}
    norm = dict(row)
    norm["messages"] = messages
    norm["task_id"] = "mlx-lora-docs-normalize"
    norm["task_type"] = "documentation"
    norm["complexity"] = "low"
    norm["risk_class"] = "low"
    norm["dataset_role"] = "core_docs_style"
    norm["record_id"] = str(row.get("record_id") or f"gold-docs:{idx}")
    norm["lineage"] = "documentation_specialist:v1"
    norm["policy_version"] = "router_policy_v1"
    return norm


def _build_splits(
    core_rows: List[Dict[str, Any]],
    *,
    seed: int,
    min_train_core_rows: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(core_rows)
    rng.shuffle(rows)
    if not rows:
        rows = _seed_rows()
        rng.shuffle(rows)

    test_n = 1 if len(rows) >= 1 else 0
    valid_n = 3 if len(rows) >= 4 else max(0, min(1, len(rows) - test_n))
    test_rows = rows[:test_n]
    valid_rows = rows[test_n : test_n + valid_n]
    train_base = rows[test_n + valid_n :]
    if not train_base:
        train_base = rows

    target_train = max(min_train_core_rows, len(train_base))
    train_rows: List[Dict[str, Any]] = []
    i = 0
    while len(train_rows) < target_train:
        train_rows.append(dict(train_base[i % len(train_base)]))
        i += 1

    return {"train": train_rows, "valid": valid_rows, "test": test_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build documentation specialist dataset.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "documentation_specialist",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-train-core-rows", type=int, default=120)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_rows: List[Dict[str, Any]] = []
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        existing_rows.extend(_load_jsonl(out_dir / name))
    if not existing_rows:
        existing_rows = _seed_rows()

    canonical_rows: List[Dict[str, Any]] = []
    seen_ids = set()
    for idx, row in enumerate(existing_rows):
        norm = _canonicalize_row(row, idx)
        if not norm:
            continue
        rid = norm["record_id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        canonical_rows.append(norm)

    if not canonical_rows:
        canonical_rows = _seed_rows()

    splits = _build_splits(
        canonical_rows,
        seed=args.seed,
        min_train_core_rows=args.min_train_core_rows,
    )

    _write_jsonl(out_dir / "train.jsonl", splits["train"])
    _write_jsonl(out_dir / "valid.jsonl", splits["valid"])
    _write_jsonl(out_dir / "test.jsonl", splits["test"])

    manifest = {
        "schema_version": "documentation_specialist_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "policy_version": "router_policy_v1",
        "lineage": "documentation_specialist:v1",
        "task_id_alias": "mlx-lora-docs-normalize",
        "raw_counts": {"core_docs_rows": len(canonical_rows)},
        "split_counts": {
            "train": len(splits["train"]),
            "valid": len(splits["valid"]),
            "test": len(splits["test"]),
        },
        "min_train_core_rows": int(args.min_train_core_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

