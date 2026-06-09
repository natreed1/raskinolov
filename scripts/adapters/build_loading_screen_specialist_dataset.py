#!/usr/bin/env python3
"""Build the loading-screen specialist dataset.

Sources:
- Game-task pairwise winners (core + transfer task ids)
- Loading benchmark tasks (prompt -> synthesized target answers)
- Shared anchor rows (optional)

Output:
- messages JSONL splits: train/valid/test
- manifest.json with source counts and split counts
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You are Albert, a UI writing specialist for Fallen Empire loading screens. "
    "Use compact medieval-strategy language, clear status framing, and practical "
    "readability. Follow output-shape constraints literally when requested."
)


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


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_text(text: str, max_chars: int = 12000) -> str:
    return (text or "").strip()[:max_chars].rstrip()


def _pick_any(any_contains: List[str]) -> str:
    if not any_contains:
        return "ready"
    for token in any_contains:
        if token and token.strip():
            return token
    return "ready"


def _extract_task_specialists(task: Dict[str, Any]) -> set[str]:
    raw = task.get("specialists") or []
    specialists: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            token = str(item).strip()
            if token:
                specialists.add(token)
    return specialists


def _is_cross_domain_benchmark_task(task: Dict[str, Any]) -> bool:
    task_id = str(task.get("id") or "").lower()
    category = str(task.get("category") or "").lower()
    if any(tag in task_id for tag in ("transfer", "multidomain", "cross_domain", "cross-domain")):
        return True
    if any(tag in category for tag in ("transfer", "multidomain", "cross_domain", "cross-domain")):
        return True
    return False


def _synthesize_from_task(task: Dict[str, Any]) -> str:
    expect = task.get("expect") or {}
    all_contains = [str(x) for x in (expect.get("all_contains") or [])]
    any_contains = [str(x) for x in (expect.get("any_contains") or [])]
    none_contains = [str(x) for x in (expect.get("none_contains") or [])]
    min_chars = int(expect.get("min_chars") or 80)
    prompt = str(task.get("prompt") or "")
    category = str(task.get("category") or "")

    anchor = _pick_any(any_contains)
    if category.endswith("constraints"):
        base = (
            f"Loading status: {anchor}. "
            "Morale and supply reports are prepared; command interface turns active when checks are complete."
        )
    elif "ui_change" in category or "hierarchy" in prompt.lower():
        base = (
            "Title stays dominant, subtitle gives tactical context, and the progress bar advances in clear phases. "
            f"Primary status cue uses {anchor} wording so readiness is readable at a glance."
        )
    else:
        base = (
            f"Empire logistics are aligning and {anchor} checks are underway. "
            "Council dispatches update in compact, readable phases before the next command."
        )

    for token in all_contains:
        if token and token not in base:
            base += f" {token}"
    if any_contains and not any(tok in base for tok in any_contains):
        base += f" {anchor}"

    for banned in none_contains:
        if banned:
            base = base.replace(banned, "")

    while len(base) < min_chars:
        base += " Readiness remains clear, stable, and strategically framed."
    return base.strip()


def _row(
    *,
    user: str,
    assistant: str,
    record_id: str,
    dataset_role: str,
    task_id: str = "loading-screen-polish",
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task_id": task_id,
        "task_type": "ui",
        "complexity": "low",
        "risk_class": "low",
        "dataset_role": dataset_role,
        "record_id": record_id,
        "lineage": "loading_screen_specialist:v3_mass_benchmark",
        "policy_version": "router_policy_v1",
    }


def _build_pairwise_rows(
    pairwise_rows: List[Dict[str, Any]],
    *,
    transfer_task_ids: set[str],
    allow_transfer: bool,
    strict_specialist_only: bool,
    max_core_rows: int,
    max_transfer_rows: int,
) -> tuple[Dict[str, List[Dict[str, Any]]], int]:
    core: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []
    rejected_foreign = 0
    for idx, rec in enumerate(pairwise_rows):
        task = rec.get("task") or {}
        task_id = str(task.get("id") or "")
        task_specialists = _extract_task_specialists(task)
        belongs_to_specialist = bool(
            "loading_screen" in task_specialists or (task_id and task_id == "loading-screen-polish")
        )
        winner_output = _clean_text(str(rec.get("winner_output") or ""))
        if not winner_output:
            continue
        prompt = str(task.get("prompt") or "").strip()
        if not prompt:
            continue
        role = "core_pairwise"
        if not belongs_to_specialist:
            if allow_transfer and task_id in transfer_task_ids:
                role = "transfer_pairwise"
            else:
                rejected_foreign += 1
                continue
        row = _row(
            user=prompt,
            assistant=winner_output,
            record_id=f"pairwise:{task_id}:{idx}",
            dataset_role=role,
            task_id=task_id or "loading-screen-polish",
        )
        if role == "core_pairwise":
            core.append(row)
        else:
            transfer.append(row)

    if strict_specialist_only and transfer:
        raise SystemExit("Strict specialist mode detected transfer rows in loading-screen dataset assembly.")

    return (
        {
            "core_pairwise": core[: max(0, max_core_rows)],
            "transfer_pairwise": transfer[: max(0, max_transfer_rows)],
        },
        rejected_foreign,
    )


def _build_benchmark_rows(
    tasks_path: Path, max_rows: int, strict_specialist_only: bool, allow_cross_domain_benchmark: bool
) -> tuple[List[Dict[str, Any]], int]:
    if not tasks_path.is_file() or max_rows <= 0:
        return [], 0
    try:
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0
    if not isinstance(tasks, list):
        return [], 0
    rows: List[Dict[str, Any]] = []
    skipped_cross_domain = 0
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        specialists = task.get("specialists") or []
        if "loading_screen" not in specialists:
            continue
        if strict_specialist_only and not allow_cross_domain_benchmark and _is_cross_domain_benchmark_task(task):
            skipped_cross_domain += 1
            continue
        prompt = str(task.get("prompt") or "").strip()
        if not prompt:
            continue
        assistant = _synthesize_from_task(task)
        rows.append(
            _row(
                user=prompt,
                assistant=assistant,
                record_id=f"benchmark:{task.get('id','task')}:{idx}",
                dataset_role="core_benchmark_synth",
                task_id="loading-screen-polish",
            )
        )
        if len(rows) >= max_rows:
            break
    return rows, skipped_cross_domain


def _build_shared_rows(shared_anchor_dir: Path, max_rows: int) -> List[Dict[str, Any]]:
    if max_rows <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        for src in _read_jsonl(shared_anchor_dir / name):
            messages = src.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            rows.append(
                {
                    **src,
                    "dataset_role": "shared_anchor",
                    "record_id": str(src.get("record_id") or f"shared:{name}:{len(rows)}"),
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


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


def _sample_with_replacement(pool: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    if count <= 0 or not pool:
        return []
    return [dict(rng.choice(pool)) for _ in range(count)]


def _build_train_rows(
    *,
    core_rows: List[Dict[str, Any]],
    transfer_rows: List[Dict[str, Any]],
    shared_rows: List[Dict[str, Any]],
    core_ratio: float,
    transfer_ratio: float,
    shared_ratio: float,
    min_train_core_rows: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    target = max(min_train_core_rows, len(core_rows))
    c_n = int(target * max(0.0, core_ratio))
    t_n = int(target * max(0.0, transfer_ratio))
    s_n = int(target * max(0.0, shared_ratio))
    total = c_n + t_n + s_n
    if total < target:
        c_n += target - total

    train = []
    train.extend(_sample_with_replacement(core_rows, c_n, rng))
    train.extend(_sample_with_replacement(transfer_rows, t_n, rng))
    train.extend(_sample_with_replacement(shared_rows, s_n, rng))

    # Refill from any non-empty pool if one bucket is empty.
    pools = [p for p in (core_rows, transfer_rows, shared_rows) if p]
    while len(train) < target and pools:
        train.append(dict(rng.choice(pools[0])))
    rng.shuffle(train)
    return train


def _split_holdout(
    all_rows: List[Dict[str, Any]],
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(all_rows)
    rng.shuffle(rows)
    valid_n = min(24, max(1, len(rows) // 10))
    test_n = min(24, max(1, len(rows) // 10))
    valid = rows[:valid_n]
    test = rows[valid_n : valid_n + test_n]
    if not test:
        test = rows[-1:] if rows else []
    return valid, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build loading-screen specialist dataset.")
    parser.add_argument(
        "--pairwise-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl",
    )
    parser.add_argument("--baselines-dir", type=Path, default=REPO / "data" / "arena_task_baselines")
    parser.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    parser.add_argument(
        "--benchmark-tasks-json",
        type=Path,
        default=REPO / "benchmarks" / "loading_screen_mass_tasks_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data" / "lora" / "adapters" / "loading_screen_specialist",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-ratio", type=float, default=0.70)
    parser.add_argument("--transfer-ratio", type=float, default=0.00)
    parser.add_argument("--shared-ratio", type=float, default=0.10)
    parser.add_argument("--max-core-rows", type=int, default=120)
    parser.add_argument("--max-transfer-rows", type=int, default=0)
    parser.add_argument("--max-benchmark-rows", type=int, default=120)
    parser.add_argument("--max-shared-rows", type=int, default=80)
    parser.add_argument("--min-train-core-rows", type=int, default=120)
    parser.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=[],
    )
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

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    transfer_task_ids = {str(t).strip() for t in args.transfer_task_ids if str(t).strip()}
    if args.strict_specialist_only and transfer_task_ids:
        raise SystemExit(
            "Strict specialist mode forbids --transfer-task-id values; remove them or disable strict mode."
        )
    if args.strict_specialist_only and args.allow_transfer:
        raise SystemExit(
            "Strict specialist mode forbids transfer rows; disable --allow-transfer."
        )
    pairwise_rows = _read_jsonl(args.pairwise_jsonl.expanduser().resolve())
    pairwise, rejected_foreign_pairwise = _build_pairwise_rows(
        pairwise_rows,
        transfer_task_ids=transfer_task_ids,
        allow_transfer=bool(args.allow_transfer),
        strict_specialist_only=bool(args.strict_specialist_only),
        max_core_rows=max(0, args.max_core_rows),
        max_transfer_rows=max(0, args.max_transfer_rows),
    )
    benchmark_rows, skipped_cross_domain_benchmark = _build_benchmark_rows(
        args.benchmark_tasks_json.expanduser().resolve(),
        max_rows=max(0, args.max_benchmark_rows),
        strict_specialist_only=bool(args.strict_specialist_only),
        allow_cross_domain_benchmark=bool(args.allow_cross_domain_benchmark),
    )
    shared_rows = _build_shared_rows(
        args.shared_anchor_dir.expanduser().resolve(),
        max_rows=max(0, args.max_shared_rows),
    )

    core_rows = _dedupe(pairwise["core_pairwise"] + benchmark_rows)
    transfer_rows = _dedupe(pairwise["transfer_pairwise"])
    shared_rows = _dedupe(shared_rows)

    combined_holdout_pool = _dedupe(core_rows + transfer_rows + shared_rows)
    if not combined_holdout_pool:
        raise SystemExit(
            "No usable rows found. Provide pairwise data or benchmark tasks JSON."
        )

    valid_rows, test_rows = _split_holdout(combined_holdout_pool, rng)
    train_rows = _build_train_rows(
        core_rows=core_rows,
        transfer_rows=transfer_rows,
        shared_rows=shared_rows,
        core_ratio=float(args.core_ratio),
        transfer_ratio=float(args.transfer_ratio),
        shared_ratio=float(args.shared_ratio),
        min_train_core_rows=int(args.min_train_core_rows),
        rng=rng,
    )

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    manifest = {
        "schema_version": "loading_screen_specialist_dataset_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "lineage": "loading_screen_specialist:v3_mass_benchmark",
        "policy_version": "router_policy_v1",
        "inputs": {
            "pairwise_jsonl": str(args.pairwise_jsonl),
            "baselines_dir": str(args.baselines_dir),
            "shared_anchor_dir": str(args.shared_anchor_dir),
            "benchmark_tasks_json": str(args.benchmark_tasks_json),
        },
        "source_counts": {
            "core_pairwise": len(pairwise["core_pairwise"]),
            "transfer_pairwise": len(pairwise["transfer_pairwise"]),
            "core_benchmark_synth": len(benchmark_rows),
            "shared_anchor": len(shared_rows),
            "rejected_foreign_pairwise": int(rejected_foreign_pairwise),
            "skipped_cross_domain_benchmark": int(skipped_cross_domain_benchmark),
        },
        "ratios": {
            "core_ratio": float(args.core_ratio),
            "transfer_ratio": float(args.transfer_ratio),
            "shared_ratio": float(args.shared_ratio),
        },
        "caps": {
            "max_core_rows": int(args.max_core_rows),
            "max_transfer_rows": int(args.max_transfer_rows),
            "max_benchmark_rows": int(args.max_benchmark_rows),
            "max_shared_rows": int(args.max_shared_rows),
            "min_train_core_rows": int(args.min_train_core_rows),
        },
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "transfer_task_ids": sorted(transfer_task_ids),
        "strict_specialist_only": bool(args.strict_specialist_only),
        "allow_transfer": bool(args.allow_transfer),
        "allow_cross_domain_benchmark": bool(args.allow_cross_domain_benchmark),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
