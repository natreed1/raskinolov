#!/usr/bin/env python3
"""Build a cached training-data index for the private dashboard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTERS_DIR = REPO / "data" / "lora" / "adapters"
DEFAULT_OUT_PATH = REPO / "data" / "training_dashboard" / "training_data_catalog.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            total += 1
    return total


def _build_dataset_row(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path)
    files: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        split_path = dataset_dir / f"{split}.jsonl"
        stat = split_path.stat() if split_path.is_file() else None
        files[split] = {
            "path": str(split_path.relative_to(REPO)) if split_path.is_file() else "",
            "rows": _count_lines(split_path),
            "size_bytes": int(stat.st_size) if stat else 0,
            "updated_utc": (
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if stat
                else ""
            ),
        }
    total_rows = sum(files[split]["rows"] for split in ("train", "valid", "test"))
    return {
        "dataset_id": dataset_dir.name,
        "label": dataset_dir.name.replace("_", " ").title(),
        "path": str(dataset_dir.relative_to(REPO)),
        "manifest_path": str(manifest_path.relative_to(REPO)) if manifest_path.is_file() else "",
        "task_family": str(manifest.get("task_family") or ""),
        "source": str(manifest.get("source") or ""),
        "total_rows": total_rows,
        "splits": files,
    }


def build_catalog(adapters_dir: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    if adapters_dir.is_dir():
        for child in sorted(adapters_dir.iterdir()):
            if not child.is_dir():
                continue
            has_any_split = any((child / f"{split}.jsonl").is_file() for split in ("train", "valid", "test"))
            if not has_any_split:
                continue
            datasets.append(_build_dataset_row(child))
    return {
        "generated_utc": _utc_now(),
        "adapters_root": str(adapters_dir.relative_to(REPO)) if adapters_dir.is_absolute() else str(adapters_dir),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard cache for training data sets.")
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    adapters_dir = args.adapters_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    payload = build_catalog(adapters_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
