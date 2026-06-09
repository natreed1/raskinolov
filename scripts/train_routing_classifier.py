#!/usr/bin/env python3
"""Train and evaluate Router V2 OSS linear classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from router.classifier import evaluate_classifier, save_classifier, train_classifier


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router adapter classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/valid/test jsonl.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Artifact output directory.")
    parser.add_argument("--max-vocab", type=int, default=8192)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    train_rows = _read_jsonl(data_dir / "train.jsonl")
    valid_rows = _read_jsonl(data_dir / "valid.jsonl") if (data_dir / "valid.jsonl").is_file() else []
    test_rows = _read_jsonl(data_dir / "test.jsonl") if (data_dir / "test.jsonl").is_file() else []
    dataset_manifest = {}
    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    clf, train_summary = train_classifier(
        train_rows=train_rows,
        valid_rows=valid_rows,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
    )
    test_metrics = evaluate_classifier(clf, test_rows) if test_rows else {"rows": 0, "accuracy": None}
    train_summary["test_accuracy"] = test_metrics.get("accuracy")
    train_summary["test_rows"] = test_metrics.get("rows")
    artifact_dir = save_classifier(
        clf,
        out_dir=out_dir,
        train_summary=train_summary,
        dataset_manifest=dataset_manifest,
    )
    summary = {
        "artifact_dir": str(artifact_dir),
        "train_summary": train_summary,
        "test_metrics": test_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

