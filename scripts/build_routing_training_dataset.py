#!/usr/bin/env python3
"""
Build a supervised routing dataset from benchmark + live + curated prompts.

Output split files use deterministic hash-based partitioning and include lineage
metadata so routing improvements can be tracked over time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from router.classifier import KEYWORDS, classify_prompt

DEFAULT_BENCHMARK_TASKS = REPO / "benchmarks" / "task_routing_tasks.json"
DEFAULT_OUT_ROOT = REPO / "data" / "lora" / "routing_classifier"
VALID_ROUTES = {"local", "hybrid", "frontier"}
VALID_ADAPTERS = set(KEYWORDS.keys()) | {"general_fallback"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm_route(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    route = str(value).strip().lower()
    return route if route in VALID_ROUTES else None


def _norm_adapter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    adapter = str(value).strip()
    return adapter if adapter in VALID_ADAPTERS else None


def _stable_split(record_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    if unit < 0.85:
        return "train"
    if unit < 0.95:
        return "valid"
    return "test"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            yield parsed


def _read_benchmark(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Benchmark tasks not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array in {path}")
    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        expected_route = _norm_route(item.get("expected_legacy_route") or item.get("expected_route"))
        expected_adapter = _norm_adapter(item.get("expected_adapter_id"))
        if expected_adapter is None:
            expected_adapter = classify_prompt(prompt).adapter_id
            label_origin = "auto_classifier_v1"
        else:
            label_origin = "explicit"
        rows.append(
            {
                "record_id": item.get("id") or f"benchmark-{_hash16(prompt)}",
                "prompt": prompt,
                "expected_adapter_id": expected_adapter,
                "expected_legacy_route": expected_route,
                "source": "benchmark",
                "created_at": _utc_now(),
                "policy_version": "router_policy_v1",
                "lineage": {
                    "origin": "benchmarks/task_routing_tasks.json",
                    "label_origin": label_origin,
                },
                "notes": item.get("reason"),
            }
        )
    return rows


def _read_label_jsonl(path: Path, source_name: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Input JSONL not found: {path}")
    out: List[Dict[str, Any]] = []
    for row in _read_jsonl(path):
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        expected_adapter = _norm_adapter(row.get("expected_adapter_id"))
        expected_route = _norm_route(row.get("expected_legacy_route") or row.get("expected_route"))
        if expected_adapter is None or expected_route is None:
            continue
        out.append(
            {
                "record_id": row.get("record_id") or f"{source_name}-{_hash16(prompt)}",
                "prompt": prompt,
                "expected_adapter_id": expected_adapter,
                "expected_legacy_route": expected_route,
                "source": row.get("source") or source_name,
                "created_at": row.get("created_at") or _utc_now(),
                "policy_version": row.get("policy_version") or "router_policy_v1",
                "lineage": row.get("lineage") or {"origin": str(path)},
                "reviewer": row.get("reviewer"),
                "notes": row.get("notes"),
                "predicted_adapter_id": row.get("predicted_adapter_id"),
                "predicted_legacy_route": row.get("predicted_legacy_route"),
                "confidence": row.get("confidence"),
                "ambiguity": row.get("ambiguity"),
                "risk_class": row.get("risk_class"),
                "complexity": row.get("complexity"),
                "accepted_for_training": bool(row.get("accepted_for_training", False)),
            }
        )
    return out


def _priority_flag(row: Dict[str, Any], low_conf_threshold: float) -> bool:
    if bool(row.get("accepted_for_training")):
        return True
    pred_adapter = row.get("predicted_adapter_id")
    pred_route = row.get("predicted_legacy_route")
    if pred_adapter and pred_adapter != row.get("expected_adapter_id"):
        return True
    if pred_route and pred_route != row.get("expected_legacy_route"):
        return True
    conf = row.get("confidence")
    if isinstance(conf, (float, int)) and float(conf) < low_conf_threshold:
        return True
    return False


def _to_training_row(row: Dict[str, Any], low_conf_threshold: float) -> Dict[str, Any]:
    record_id = str(row.get("record_id") or f"routing-{_hash16(str(row['prompt']))}")
    priority = _priority_flag(row, low_conf_threshold)
    return {
        "record_id": record_id,
        "text": row["prompt"],
        "labels": {
            "adapter_id": row["expected_adapter_id"],
            "legacy_route": row["expected_legacy_route"],
        },
        "source": row.get("source") or "unknown",
        "policy_version": row.get("policy_version") or "router_policy_v1",
        "lineage": row.get("lineage") or {},
        "created_at": row.get("created_at") or _utc_now(),
        "priority_hard_example": priority,
        "training_weight": 2.0 if priority else 1.0,
        "metadata": {
            "reviewer": row.get("reviewer"),
            "notes": row.get("notes"),
            "predicted_adapter_id": row.get("predicted_adapter_id"),
            "predicted_legacy_route": row.get("predicted_legacy_route"),
            "confidence": row.get("confidence"),
            "ambiguity": row.get("ambiguity"),
            "risk_class": row.get("risk_class"),
            "complexity": row.get("complexity"),
        },
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/valid/test routing classifier dataset")
    ap.add_argument("--benchmark-tasks", type=Path, default=DEFAULT_BENCHMARK_TASKS)
    ap.add_argument("--live-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--curated-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-version", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--low-confidence-threshold", type=float, default=0.62)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    rows.extend(_read_benchmark(args.benchmark_tasks.expanduser().resolve()))
    for path in args.live_jsonl:
        rows.extend(_read_label_jsonl(path.expanduser().resolve(), "live"))
    for path in args.curated_jsonl:
        rows.extend(_read_label_jsonl(path.expanduser().resolve(), "curated"))

    if not rows:
        raise SystemExit("No labeled routing rows collected.")

    # Last write wins for duplicate record_id.
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("record_id") or f"routing-{_hash16(str(row.get('prompt') or ''))}")
        row["record_id"] = rid
        dedup[rid] = row
    normalized = list(dedup.values())
    dataset_rows = [_to_training_row(row, args.low_confidence_threshold) for row in normalized]

    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in dataset_rows:
        split = _stable_split(row["record_id"], args.seed)
        if split == "train":
            train.append(row)
        elif split == "valid":
            valid.append(row)
        else:
            test.append(row)

    if not train or not valid or not test:
        raise SystemExit("Split produced an empty partition; add more rows.")

    out_dir = args.out_root.expanduser().resolve() / args.dataset_version
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "valid.jsonl", valid)
    _write_jsonl(out_dir / "test.jsonl", test)

    manifest = {
        "schema": "routing_dataset_v1",
        "dataset_version": args.dataset_version,
        "built_at": _utc_now(),
        "seed": args.seed,
        "low_confidence_threshold": args.low_confidence_threshold,
        "sources": {
            "benchmark_tasks": str(args.benchmark_tasks.expanduser().resolve()),
            "live_jsonl": [str(p.expanduser().resolve()) for p in args.live_jsonl],
            "curated_jsonl": [str(p.expanduser().resolve()) for p in args.curated_jsonl],
        },
        "rows_before_dedup": len(rows),
        "rows_after_dedup": len(normalized),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "priority_hard_examples": sum(1 for row in dataset_rows if row["priority_hard_example"]),
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
