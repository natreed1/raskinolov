#!/usr/bin/env python3
"""
Build `train.jsonl`, `valid.jsonl`, and `test.jsonl` for `mlx_lm.lora --data <dir>`.

mlx-lm expects each line to be JSON in one of these shapes (auto-detected from first row):
  - `{"text": "..."}`  — causal LM on text (used here for whole-file code)
  - `{"messages": [...]}` — chat SFT

This script emits **`text`** samples from `export_repo_for_training.py` output.

**Default (chunked mode):** long files are split into multiple rows so **tail content is not
lost**—each chunk has a header with path, part index, and character span. Train/valid/test
are chosen **per source file** (hash of seed + path) so chunks from the same file never
straddle splits.

**Legacy:** pass `--no-chunk` for the old behavior (one truncated row per file, `--max-chars`).

Default `--out-dir` matches the **Qwen2.5-Coder-7B** lineage (`scripts/fe_lineage.py`);
see **`docs/DATA_LAYOUT.md`** and **`docs/CHUNKED_GAME_TEXT.md`**.

Environment:
  SOURCE_REPO — only mentioned in docs; export script uses it separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import fe_lineage as _fe

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


def _to_text_sample_legacy(path: str, body: str, max_chars: int) -> str:
    header = f"# {path}\n\n"
    room = max(0, max_chars - len(header))
    body_use = body if len(body) <= room else body[:room]
    return header + body_use


def _path_unit(path: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _iter_body_chunks(body: str, chunk_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Return (char_start, char_end_exclusive, chunk_text) for each chunk."""
    n = len(body)
    if n == 0:
        return []
    if n <= chunk_chars:
        return [(0, n, body)]
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_chars")
    out: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        out.append((start, end, body[start:end]))
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        start = nxt
    return out


def _format_chunk_text(path: str, part: int, total: int, c0: int, c1_exclusive: int, body: str) -> str:
    header = f"# path: {path}\n# part: {part}/{total}\n# chars: {c0}-{c1_exclusive - 1}\n\n"
    return header + body


def _rows_for_file_chunked(path: str, body: str, chunk_chars: int, overlap: int) -> List[dict]:
    spans = _iter_body_chunks(body, chunk_chars, overlap)
    if not spans:
        return []
    total = len(spans)
    rows: List[dict] = []
    for i, (c0, c1, chunk) in enumerate(spans, start=1):
        rows.append({"text": _format_chunk_text(path, i, total, c0, c1, chunk)})
    return rows


def _assign_splits_by_file(
    path_to_rows: Dict[str, List[dict]],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    min_valid: int,
    min_test: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    paths = list(path_to_rows.keys())
    train_p: set[str] = set()
    valid_p: set[str] = set()
    test_p: set[str] = set()
    for p in paths:
        u = _path_unit(p, seed)
        if u < train_ratio:
            train_p.add(p)
        elif u < train_ratio + valid_ratio:
            valid_p.add(p)
        else:
            test_p.add(p)
    # Ensure every path assigned (hash always hits one branch — but empty sets if no paths)
    if not paths:
        return [], [], []

    def collect(ps: set[str]) -> List[dict]:
        out: List[dict] = []
        for p in sorted(ps):
            out.extend(path_to_rows[p])
        return out

    def n_rows(ps: set[str]) -> int:
        return sum(len(path_to_rows[p]) for p in ps)

    def move_path(p: str, src: set[str], dst: set[str]) -> None:
        src.remove(p)
        dst.add(p)

    # Rebalance so valid/test have enough rows for small batch sizes in mlx-lm.
    while n_rows(valid_p) < min_valid and train_p:
        p = max(train_p, key=lambda x: len(path_to_rows[x]))
        move_path(p, train_p, valid_p)
    while n_rows(test_p) < min_test and train_p:
        p = max(train_p, key=lambda x: len(path_to_rows[x]))
        move_path(p, train_p, test_p)
    while n_rows(train_p) < 1 and valid_p:
        p = next(iter(sorted(valid_p)))
        move_path(p, valid_p, train_p)
    while n_rows(train_p) < 1 and test_p:
        p = next(iter(sorted(test_p)))
        move_path(p, test_p, train_p)

    return collect(train_p), collect(valid_p), collect(test_p)


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
        default=Path(_fe.GAME_TEXT_DIR_RELPATH),
        help="Directory that will contain train.jsonl, valid.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Legacy: one row per file, truncate body to --max-chars (no multi-part headers)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="With --no-chunk: max characters for the full text sample (header counts). Ignored when chunking.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=8000,
        help="Chunked mode: max body characters per chunk (header is not counted against this)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=512,
        help="Chunked mode: characters repeated between consecutive chunks (0 allowed)",
    )
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

    if not args.no_chunk:
        if args.chunk_chars < 256:
            raise SystemExit("--chunk-chars must be at least 256")
        if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
            raise SystemExit("--chunk-overlap must satisfy 0 <= overlap < --chunk-chars")

    rng = random.Random(args.seed)

    def _shuffle_split_flat(all_rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
        n = len(all_rows)
        if n < 3:
            raise SystemExit(f"Need at least 3 samples, got {n}")
        rows_copy = list(all_rows)
        rng.shuffle(rows_copy)
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
        tr = rows_copy[:n_train]
        va = rows_copy[n_train : n_train + n_valid]
        te = rows_copy[n_train + n_valid : n_train + n_valid + n_test]
        return tr, va, te

    train: List[dict]
    valid: List[dict]
    test: List[dict]
    n_files: int
    n: int
    mode: str

    if args.synthetic_smoke:
        flat: List[dict] = []
        for path, body in SYNTHETIC_SAMPLES * 12:
            if args.no_chunk:
                flat.append({"text": _to_text_sample_legacy(path, body, args.max_chars)})
            else:
                flat.extend(_rows_for_file_chunked(path, body, args.chunk_chars, args.chunk_overlap))
        n = len(flat)
        n_files = len({p for p, _ in SYNTHETIC_SAMPLES})
        train, valid, test = _shuffle_split_flat(flat)
        mode = "synthetic " + ("legacy" if args.no_chunk else f"chunked c={args.chunk_chars} o={args.chunk_overlap}")
    elif args.no_chunk:
        flat: List[dict] = []
        src = args.from_export.resolve()
        if not src.is_file():
            raise SystemExit(
                f"Export not found: {src}\n"
                "Run: export SOURCE_REPO=/path/to/fallen-empire && python scripts/export_repo_for_training.py\n"
                "Or pass --synthetic-smoke for a tiny test corpus."
            )
        paths_seen: set[str] = set()
        for path, text in _iter_export(src):
            if not text.strip():
                continue
            paths_seen.add(path)
            flat.append({"text": _to_text_sample_legacy(path, text, args.max_chars)})
        n = len(flat)
        n_files = len(paths_seen)
        train, valid, test = _shuffle_split_flat(flat)
        mode = "legacy no-chunk"
    else:
        path_to_rows: Dict[str, List[dict]] = {}
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
            rows = _rows_for_file_chunked(path, text, args.chunk_chars, args.chunk_overlap)
            if rows:
                path_to_rows.setdefault(path, []).extend(rows)
        n_files = len(path_to_rows)
        all_rows = [r for rows in path_to_rows.values() for r in rows]
        n = len(all_rows)
        if n < 3:
            raise SystemExit(f"Need at least 3 samples, got {n} (from {n_files} paths)")
        # Few distinct paths: per-file split + row-count floors can deadlock (e.g. one huge file).
        if n_files <= 2:
            train, valid, test = _shuffle_split_flat(all_rows)
            mode = f"chunked c={args.chunk_chars} o={args.chunk_overlap} (flat split: {n_files} paths)"
        else:
            min_valid = 8 if n >= 20 else max(2, min(4, n // 3))
            min_test = 2 if n >= 12 else 1
            train, valid, test = _assign_splits_by_file(
                path_to_rows,
                args.seed,
                args.train_ratio,
                args.valid_ratio,
                args.test_ratio,
                min_valid,
                min_test,
            )
            rng.shuffle(train)
            rng.shuffle(valid)
            rng.shuffle(test)
            mode = f"chunked c={args.chunk_chars} o={args.chunk_overlap}"

    out = args.out_dir.resolve()
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)
    _write_jsonl(out / "test.jsonl", test)

    print(
        f"Wrote {out}/train.jsonl ({len(train)}), valid.jsonl ({len(valid)}), test.jsonl ({len(test)}) "
        f"[{n_files} source paths, {n} rows, {mode}]"
    )


if __name__ == "__main__":
    main()
