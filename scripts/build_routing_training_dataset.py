#!/usr/bin/env python3
"""Build deterministic routing classifier train/valid/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

REPO = Path(__file__).resolve().parent.parent


def _stable_key(prompt: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{prompt}".encode("utf-8")).hexdigest()


def _as_row(
    *,
    prompt: str,
    expected_adapter_id: str,
    source: str,
    case_id: str,
    expected_keywords: list[str] | None = None,
    expected_legacy_route: str | None = None,
    accepted_for_training: bool = True,
    label_source: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "routing_classifier_row_v1",
        "prompt": prompt.strip(),
        "expected_adapter_id": expected_adapter_id.strip(),
        "source": source,
        "case_id": case_id,
        "accepted_for_training": bool(accepted_for_training),
    }
    if expected_keywords:
        row["expected_keywords"] = [str(k) for k in expected_keywords if str(k).strip()]
    if expected_legacy_route:
        row["expected_legacy_route"] = expected_legacy_route
    if label_source:
        row["label_source"] = label_source
    return row


def _from_benchmark(tasks_path: Path, policy: RoutingPolicy) -> list[dict[str, Any]]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit(f"Expected array in benchmark tasks: {tasks_path}")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        prompt = str(task.get("prompt") or "").strip()
        if not prompt:
            continue
        expected_adapter = str(task.get("expected_adapter_id") or "").strip()
        label_source = "human_label"
        if not expected_adapter:
            # Bootstrap for legacy route-only files.
            decision = policy.decide(GenerationRequest(messages=messages_from_prompt(prompt)))
            expected_adapter = decision.adapter_id
            label_source = "bootstrap_policy_v2"
        rows.append(
            _as_row(
                prompt=prompt,
                expected_adapter_id=expected_adapter,
                expected_legacy_route=str(task.get("expected_route") or "").strip() or None,
                source="benchmark_tasks",
                case_id=str(task.get("id") or f"bench-{len(rows)}"),
                label_source=label_source,
            )
        )
    return rows


def _from_jsonl(path: Path, *, source_name: str, low_confidence_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            continue
        expected_adapter = str(
            payload.get("expected_adapter_id")
            or payload.get("adapter_id")
            or payload.get("predicted_adapter_id")
            or ""
        ).strip()
        if not expected_adapter:
            continue
        accepted = payload.get("accepted_for_training")
        if accepted is False:
            continue
        conf = payload.get("confidence")
        if conf is not None:
            try:
                if float(conf) < low_confidence_threshold and source_name == "live":
                    continue
            except (TypeError, ValueError):
                pass
        rows.append(
            _as_row(
                prompt=prompt,
                expected_adapter_id=expected_adapter,
                expected_legacy_route=str(payload.get("expected_legacy_route") or payload.get("expected_route") or "").strip() or None,
                source=source_name,
                case_id=str(payload.get("case_id") or payload.get("id") or f"{path.name}:{line_no}"),
                expected_keywords=payload.get("expected_keywords") if isinstance(payload.get("expected_keywords"), list) else None,
                accepted_for_training=True,
                label_source=str(payload.get("label_source") or "human_or_lab"),
            )
        )
    return rows


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["prompt"].strip().lower()
        existing = out.get(key)
        if not existing:
            out[key] = row
            continue
        # Prefer curated/human labels over bootstrap.
        rank = {"curated": 3, "live": 2, "benchmark_tasks": 1}
        src_new = rank.get(str(row.get("source")), 0)
        src_old = rank.get(str(existing.get("source")), 0)
        if src_new >= src_old:
            out[key] = row
    return list(out.values())


def _split(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = sorted(rows, key=lambda r: _stable_key(r["prompt"], seed))
    n = len(keyed)
    n_train = max(1, int(n * 0.85))
    n_valid = max(1, int(n * 0.10)) if n >= 10 else 0
    n_test = n - n_train - n_valid
    if n_test <= 0 and n >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_valid > 1:
            n_valid -= 1
    train = keyed[:n_train]
    valid = keyed[n_train : n_train + n_valid]
    test = keyed[n_train + n_valid :]
    train_labels = {str(r["expected_adapter_id"]) for r in train}
    for bucket in (valid, test):
        i = 0
        while i < len(bucket):
            label = str(bucket[i]["expected_adapter_id"])
            if label not in train_labels:
                train.append(bucket.pop(i))
                train_labels.add(label)
                continue
            i += 1
    return train, valid, test


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build routing classifier dataset.")
    parser.add_argument("--benchmark-tasks", type=Path, required=True)
    parser.add_argument("--live-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--curated-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.62)
    args = parser.parse_args()

    policy = RoutingPolicy()
    all_rows = _from_benchmark(args.benchmark_tasks.expanduser().resolve(), policy)
    for p in args.live_jsonl:
        all_rows.extend(
            _from_jsonl(
                p.expanduser().resolve(),
                source_name="live",
                low_confidence_threshold=args.low_confidence_threshold,
            )
        )
    for p in args.curated_jsonl:
        all_rows.extend(
            _from_jsonl(
                p.expanduser().resolve(),
                source_name="curated",
                low_confidence_threshold=0.0,
            )
        )

    rows = _dedupe(all_rows)
    if not rows:
        raise SystemExit("No rows after filtering.")
    train, valid, test = _split(rows, seed=args.seed)

    out_dir = args.out_root.expanduser().resolve() / args.dataset_version
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "valid.jsonl", valid)
    _write_jsonl(out_dir / "test.jsonl", test)

    label_counts = Counter(str(row["expected_adapter_id"]) for row in rows)
    source_counts = Counter(str(row["source"]) for row in rows)
    manifest = {
        "schema_version": "routing_classifier_dataset_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_version": args.dataset_version,
        "seed": args.seed,
        "rows_total": len(rows),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "rows_test": len(test),
        "low_confidence_threshold": args.low_confidence_threshold,
        "labels": dict(sorted(label_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "inputs": {
            "benchmark_tasks": str(args.benchmark_tasks.expanduser().resolve()),
            "live_jsonl": [str(p.expanduser().resolve()) for p in args.live_jsonl],
            "curated_jsonl": [str(p.expanduser().resolve()) for p in args.curated_jsonl],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

