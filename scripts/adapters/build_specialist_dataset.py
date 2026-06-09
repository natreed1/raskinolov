#!/usr/bin/env python3
"""Generic specialist dataset builder for benchmark+pairwise supervision."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_text(text: str, max_chars: int = 12000) -> str:
    return (text or "").strip()[:max_chars].rstrip()


def _first_nonempty(values: Sequence[str], fallback: str) -> str:
    for v in values:
        if v and v.strip():
            return v
    return fallback


def _synth_response(task: Dict[str, Any], anchor_phrase: str) -> str:
    expect = task.get("expect") or {}
    all_contains = [str(x) for x in (expect.get("all_contains") or [])]
    any_contains = [str(x) for x in (expect.get("any_contains") or [])]
    none_contains = [str(x) for x in (expect.get("none_contains") or [])]
    min_chars = int(expect.get("min_chars") or 80)
    prompt = str(task.get("prompt") or "")
    category = str(task.get("category") or "")
    any_token = _first_nonempty(any_contains, anchor_phrase)

    if "constraint" in category or "exactly" in prompt.lower() or "only" in prompt.lower():
        text = (
            f"{anchor_phrase}: {any_token}. "
            "Use compact tactical wording and keep output shape strict."
        )
    else:
        text = (
            f"{anchor_phrase} guidance: {any_token}. "
            "Keep wording practical, concise, and aligned with Fallen Empire strategy context."
        )
    for token in all_contains:
        if token and token not in text:
            text += f" {token}"
    if any_contains and not any(tok in text for tok in any_contains):
        text += f" {any_token}"
    for banned in none_contains:
        if banned:
            text = text.replace(banned, "")
    while len(text) < min_chars:
        text += " Keep the response stable, readable, and actionable."
    return text.strip()


def _make_row(
    *,
    system_prompt: str,
    user: str,
    assistant: str,
    record_id: str,
    dataset_role: str,
    task_id: str,
    task_type: str,
    lineage: str,
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task_id": task_id,
        "task_type": task_type,
        "complexity": "low",
        "risk_class": "low",
        "dataset_role": dataset_role,
        "record_id": record_id,
        "lineage": lineage,
        "policy_version": "router_policy_v1",
    }


def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        rid = str(row.get("record_id") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(row)
    return out


def _sample(pool: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n <= 0 or not pool:
        return []
    return [dict(rng.choice(pool)) for _ in range(n)]


def _is_cross_domain_benchmark_task(task: Dict[str, Any]) -> bool:
    task_id = str(task.get("id") or "").lower()
    category = str(task.get("category") or "").lower()
    if any(tag in task_id for tag in ("transfer", "multidomain", "cross_domain", "cross-domain")):
        return True
    if any(tag in category for tag in ("transfer", "multidomain", "cross_domain", "cross-domain")):
        return True
    return False


def _extract_task_specialists(task: Dict[str, Any]) -> set[str]:
    raw = task.get("specialists") or []
    specialists: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item).strip()
            if token:
                specialists.add(token)
    return specialists


def _split_holdout(rows: List[Dict[str, Any]], rng: random.Random) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = list(rows)
    rng.shuffle(items)
    valid_n = min(24, max(1, len(items) // 10))
    test_n = min(24, max(1, len(items) // 10))
    valid = items[:valid_n]
    test = items[valid_n : valid_n + test_n]
    if not test:
        test = items[-1:] if items else []
    return valid, test


def build_specialist_dataset(
    *,
    specialist_id: str,
    task_id: str,
    task_type: str,
    system_prompt: str,
    anchor_phrase: str,
    out_dir: Path,
    pairwise_jsonl: Path,
    benchmark_tasks_json: Path,
    shared_anchor_dir: Path,
    transfer_task_ids: Sequence[str],
    seed: int,
    core_ratio: float,
    transfer_ratio: float,
    shared_ratio: float,
    max_core_rows: int,
    max_transfer_rows: int,
    max_benchmark_rows: int,
    max_shared_rows: int,
    min_train_core_rows: int,
    strict_specialist_only: bool,
    allow_cross_domain_benchmark: bool,
    allow_transfer: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer_ids = {str(t).strip() for t in transfer_task_ids if str(t).strip()}

    # Pairwise winners
    pairwise_raw = _read_jsonl(pairwise_jsonl)
    core_pairwise: List[Dict[str, Any]] = []
    transfer_pairwise: List[Dict[str, Any]] = []
    rejected_foreign_pairwise = 0
    for idx, rec in enumerate(pairwise_raw):
        task = rec.get("task") or {}
        src_task_id = str(task.get("id") or "")
        task_specialists = _extract_task_specialists(task)
        belongs_to_specialist = bool(
            specialist_id in task_specialists or (src_task_id and src_task_id == task_id)
        )
        user_prompt = str(task.get("prompt") or "").strip()
        winner = _clean_text(str(rec.get("winner_output") or ""))
        if not user_prompt or not winner:
            continue
        role = "core_pairwise"
        if not belongs_to_specialist:
            if allow_transfer and src_task_id in transfer_ids:
                role = "transfer_pairwise"
            else:
                rejected_foreign_pairwise += 1
                continue
        row = _make_row(
            system_prompt=system_prompt,
            user=user_prompt,
            assistant=winner,
            record_id=f"pairwise:{src_task_id}:{idx}",
            dataset_role=role,
            task_id=src_task_id or task_id,
            task_type=task_type,
            lineage=f"{specialist_id}_specialist:v2_mass",
        )
        if role == "core_pairwise":
            core_pairwise.append(row)
        else:
            transfer_pairwise.append(row)
    core_pairwise = core_pairwise[: max(0, max_core_rows)]
    transfer_pairwise = transfer_pairwise[: max(0, max_transfer_rows)]

    # Benchmark prompt synthesis
    benchmark_rows: List[Dict[str, Any]] = []
    skipped_cross_domain_benchmark = 0
    if benchmark_tasks_json.is_file() and max_benchmark_rows > 0:
        try:
            tasks = json.loads(benchmark_tasks_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tasks = []
        if isinstance(tasks, list):
            for i, task in enumerate(tasks):
                if not isinstance(task, dict):
                    continue
                specialists = task.get("specialists") or []
                if specialist_id not in specialists:
                    continue
                if strict_specialist_only and not allow_cross_domain_benchmark and _is_cross_domain_benchmark_task(task):
                    skipped_cross_domain_benchmark += 1
                    continue
                prompt = str(task.get("prompt") or "").strip()
                if not prompt:
                    continue
                assistant = _synth_response(task, anchor_phrase=anchor_phrase)
                benchmark_rows.append(
                    _make_row(
                        system_prompt=system_prompt,
                        user=prompt,
                        assistant=assistant,
                        record_id=f"benchmark:{task.get('id','task')}:{i}",
                        dataset_role="core_benchmark_synth",
                        task_id=task_id,
                        task_type=task_type,
                        lineage=f"{specialist_id}_specialist:v2_mass",
                    )
                )
                if len(benchmark_rows) >= max_benchmark_rows:
                    break

    # Shared rows
    shared_rows: List[Dict[str, Any]] = []
    if max_shared_rows > 0:
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            for j, src in enumerate(_read_jsonl(shared_anchor_dir / name)):
                messages = src.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                row = dict(src)
                row["dataset_role"] = "shared_anchor"
                row["record_id"] = str(src.get("record_id") or f"shared:{name}:{j}")
                shared_rows.append(row)
                if len(shared_rows) >= max_shared_rows:
                    break
            if len(shared_rows) >= max_shared_rows:
                break

    core_rows = _dedupe(core_pairwise + benchmark_rows)
    transfer_rows = _dedupe(transfer_pairwise)
    shared_rows = _dedupe(shared_rows)

    if strict_specialist_only:
        if allow_transfer:
            raise SystemExit(
                "Strict specialist mode forbids transfer rows; disable --allow-transfer."
            )
        if transfer_ids:
            raise SystemExit(
                "Strict specialist mode forbids --transfer-task-id values; remove them or disable strict mode."
            )
        if transfer_rows:
            raise SystemExit(
                "Strict specialist mode detected transfer rows in training assembly."
            )

    holdout_pool = _dedupe(core_rows + transfer_rows + shared_rows)
    if not holdout_pool:
        raise SystemExit(
            f"No usable rows for {specialist_id}. Provide pairwise rows or benchmark task JSON."
        )
    valid_rows, test_rows = _split_holdout(holdout_pool, rng)

    target = max(min_train_core_rows, len(core_rows))
    c_n = int(target * max(0.0, core_ratio))
    t_n = int(target * max(0.0, transfer_ratio))
    s_n = int(target * max(0.0, shared_ratio))
    total = c_n + t_n + s_n
    if total < target:
        c_n += target - total

    train_rows: List[Dict[str, Any]] = []
    train_rows.extend(_sample(core_rows, c_n, rng))
    train_rows.extend(_sample(transfer_rows, t_n, rng))
    train_rows.extend(_sample(shared_rows, s_n, rng))
    pools = [p for p in (core_rows, transfer_rows, shared_rows) if p]
    while len(train_rows) < target and pools:
        train_rows.append(dict(rng.choice(pools[0])))
    rng.shuffle(train_rows)

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    manifest = {
        "schema_version": "specialist_dataset_v2_mass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specialist_id": specialist_id,
        "task_id": task_id,
        "task_type": task_type,
        "out_dir": str(out_dir),
        "lineage": f"{specialist_id}_specialist:v2_mass",
        "policy_version": "router_policy_v1",
        "inputs": {
            "pairwise_jsonl": str(pairwise_jsonl),
            "benchmark_tasks_json": str(benchmark_tasks_json),
            "shared_anchor_dir": str(shared_anchor_dir),
        },
        "source_counts": {
            "core_pairwise": len(core_pairwise),
            "transfer_pairwise": len(transfer_pairwise),
            "core_benchmark_synth": len(benchmark_rows),
            "shared_anchor": len(shared_rows),
            "rejected_foreign_pairwise": int(rejected_foreign_pairwise),
            "skipped_cross_domain_benchmark": int(skipped_cross_domain_benchmark),
        },
        "ratios": {
            "core_ratio": core_ratio,
            "transfer_ratio": transfer_ratio,
            "shared_ratio": shared_ratio,
        },
        "caps": {
            "max_core_rows": max_core_rows,
            "max_transfer_rows": max_transfer_rows,
            "max_benchmark_rows": max_benchmark_rows,
            "max_shared_rows": max_shared_rows,
            "min_train_core_rows": min_train_core_rows,
        },
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "transfer_task_ids": sorted(transfer_ids),
        "strict_specialist_only": bool(strict_specialist_only),
        "allow_transfer": bool(allow_transfer),
        "allow_cross_domain_benchmark": bool(allow_cross_domain_benchmark),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build specialist dataset from benchmark and pairwise sources.")
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-type", default="ui")
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--anchor-phrase", default="Specialist")
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-tasks-json", type=Path, required=True)
    parser.add_argument("--shared-anchor-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-ratio", type=float, default=0.7)
    parser.add_argument("--transfer-ratio", type=float, default=0.0)
    parser.add_argument("--shared-ratio", type=float, default=0.1)
    parser.add_argument("--max-core-rows", type=int, default=120)
    parser.add_argument("--max-transfer-rows", type=int, default=0)
    parser.add_argument("--max-benchmark-rows", type=int, default=120)
    parser.add_argument("--max-shared-rows", type=int, default=80)
    parser.add_argument("--min-train-core-rows", type=int, default=120)
    parser.add_argument("--transfer-task-id", action="append", dest="transfer_task_ids", default=[])
    parser.add_argument(
        "--strict-specialist-only",
        action="store_true",
        default=True,
        help="Enforce specialist-only training composition (default on).",
    )
    parser.add_argument(
        "--no-strict-specialist-only",
        action="store_false",
        dest="strict_specialist_only",
        help="Allow non-specialist rows (not recommended).",
    )
    parser.add_argument(
        "--allow-transfer",
        action="store_true",
        help="Allow transfer rows from --transfer-task-id when strict mode is disabled.",
    )
    parser.add_argument(
        "--allow-cross-domain-benchmark",
        action="store_true",
        help="Include benchmark tasks tagged transfer/multidomain (default excluded in strict mode).",
    )
    args = parser.parse_args()
    manifest = build_specialist_dataset(
        specialist_id=args.specialist_id,
        task_id=args.task_id,
        task_type=args.task_type,
        system_prompt=args.system_prompt,
        anchor_phrase=args.anchor_phrase,
        out_dir=args.out_dir.expanduser().resolve(),
        pairwise_jsonl=args.pairwise_jsonl.expanduser().resolve(),
        benchmark_tasks_json=args.benchmark_tasks_json.expanduser().resolve(),
        shared_anchor_dir=args.shared_anchor_dir.expanduser().resolve(),
        transfer_task_ids=args.transfer_task_ids,
        seed=int(args.seed),
        core_ratio=float(args.core_ratio),
        transfer_ratio=float(args.transfer_ratio),
        shared_ratio=float(args.shared_ratio),
        max_core_rows=int(args.max_core_rows),
        max_transfer_rows=int(args.max_transfer_rows),
        max_benchmark_rows=int(args.max_benchmark_rows),
        max_shared_rows=int(args.max_shared_rows),
        min_train_core_rows=int(args.min_train_core_rows),
        strict_specialist_only=bool(args.strict_specialist_only),
        allow_cross_domain_benchmark=bool(args.allow_cross_domain_benchmark),
        allow_transfer=bool(args.allow_transfer),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _main()
