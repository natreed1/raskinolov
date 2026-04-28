#!/usr/bin/env python3
"""
Build `train.jsonl`, `valid.jsonl`, and `test.jsonl` for `mlx_lm.lora --data <dir>`.

mlx-lm expects each line to be JSON in one of these shapes (auto-detected from first row):
  - `{"text": "..."}`  — causal LM on text (used here for whole-file code)
  - `{"messages": [...]}` — chat SFT

This script emits **`text`** samples from `export_repo_for_training.py` output:
each record is `# <path>\\n\\n<file contents>` (truncated to `--max-chars`).

If the export file is missing, use `--synthetic-smoke` to emit a tiny dataset so you can
verify `mlx_lm.lora --train` without the game repo.

Environment:
  SOURCE_REPO — only mentioned in docs; export script uses it separately.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, List, Tuple

SYNTHETIC_SAMPLES = [
    (
        "src/example/AiStub.ts",
        """export type AiMood = 'patient' | 'hungry' | 'vengeful';

export function pickRecruitCap(mood: AiMood, gold: number): number {
  if (mood === 'patient') return Math.min(6, Math.floor(gold / 120));
  if (mood === 'hungry') return Math.min(10, Math.floor(gold / 80));
  return Math.min(14, Math.floor(gold / 60));
}
""",
    ),
    (
        "src/example/hex.ts",
        """export type Axial = { q: number; r: number };

export function axialDistance(a: Axial, b: Axial): number {
  const dq = b.q - a.q;
  const dr = b.r - a.r;
  return (Math.abs(dq + dr) + Math.abs(dq) + Math.abs(dr)) / 2;
}
""",
    ),
    (
        "src/example/store.ts",
        """import { create } from 'zustand';

type Slice = { ticks: number; bump: () => void };

export const useTickStore = create<Slice>((set) => ({
  ticks: 0,
  bump: () => set((s) => ({ ticks: s.ticks + 1 })),
}));
""",
    ),
]


def _iter_export(path: Path) -> Iterator[Tuple[str, str]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("path") or rec.get("file") or "unknown"
            t = rec.get("text") or rec.get("content") or ""
            yield (str(p), str(t))


def _to_text_sample(path: str, body: str, max_chars: int) -> str:
    header = f"# {path}\n\n"
    room = max(0, max_chars - len(header))
    body_use = body if len(body) <= room else body[:room]
    return header + body_use


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mlx_lm.lora JSONL dataset directory")
    parser.add_argument(
        "--from-export",
        type=Path,
        default=Path("data/raw/repo_text.jsonl"),
        help="JSONL from export_repo_for_training.py (path + text)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lora/game_text"),
        help="Directory that will contain train.jsonl, valid.jsonl, test.jsonl",
    )
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters per text sample")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Ignore export; write a tiny synthetic dataset for pipeline smoke tests",
    )
    args = parser.parse_args()

    r_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(r_sum - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {r_sum}")

    rows: List[dict] = []
    if args.synthetic_smoke:
        for path, body in SYNTHETIC_SAMPLES * 12:
            rows.append({"text": _to_text_sample(path, body, args.max_chars)})
    else:
        src = args.from_export.resolve()
        if not src.is_file():
            raise SystemExit(
                f"Export not found: {src}\n"
                "Run: export SOURCE_REPO=/path/to/fallen-empire && python scripts/export_repo_for_training.py\n"
                "Or pass --synthetic-smoke for a tiny test corpus."
            )
        for path, text in _iter_export(src):
            if not text.strip():
                continue
            rows.append({"text": _to_text_sample(path, text, args.max_chars)})

    if len(rows) < 3:
        raise SystemExit(f"Need at least 3 samples, got {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)
    # mlx-lm evaluation uses batches of size `batch_size` (often 2); a tiny valid set
    # (e.g. 1 row after smoke tests) crashes with "Dataset must have at least batch_size=2".
    # Prefer ≥8 valid rows when corpus allows; never below 2 when n ≥ 4.
    min_valid = 8 if n >= 20 else max(2, min(4, n // 3))
    min_test = 2 if n >= 12 else 1
    n_valid = max(min_valid, int(n * args.valid_ratio))
    n_test = max(min_test, int(n * args.test_ratio))
    if n_valid + n_test >= n:
        n_valid = max(min_valid, min(n - 2, n // 5))
        n_test = max(min_test, min(n - n_valid - 1, n // 6))
    n_train = n - n_valid - n_test
    if n_train < 1:
        n_test = max(min_test, n_test - 1) if n_test > min_test else n_test
        n_train = n - n_valid - n_test
    if n_train < 1:
        n_valid = max(2, n_valid - 1)
        n_train = n - n_valid - n_test
    if n_train < 1:
        raise SystemExit(
            f"Split ratios leave no training rows (n={n}). Lower valid/test ratios or add more files."
        )

    train = rows[:n_train]
    valid = rows[n_train : n_train + n_valid]
    test = rows[n_train + n_valid : n_train + n_valid + n_test]

    out = args.out_dir.resolve()
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)
    _write_jsonl(out / "test.jsonl", test)

    print(f"Wrote {out}/train.jsonl ({len(train)}), valid.jsonl ({len(valid)}), test.jsonl ({len(test)})")


if __name__ == "__main__":
    main()
