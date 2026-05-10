#!/usr/bin/env python3
"""Build richer HUD-status specialist dataset (core + shallow UI/economy transfer rows).

v2 defaults bias toward the curator baseline and real `useGameStore` slices: duplicate baseline
shards, optional guardrail instruction rows, filtered HUD pairwise (drop winners that mention
known hallucinated store APIs), and ``--max-core-rows 0`` so pairwise does not drown gold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
PAIRWISE_DEFAULT = REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl"
BASELINES_DEFAULT = REPO / "data" / "arena_task_baselines"
SHARED_ANCHOR_DEFAULT = REPO / "data" / "lora" / "game_text"
OUT_DEFAULT = REPO / "data" / "lora" / "adapters" / "hud_status_specialist"

TARGET_TASK_ID = "hud-status-summary"
DEFAULT_TRANSFER_TASKS = ("loading-screen-polish", "economy-tooltip")
LINEAGE = "hud_status_specialist:v2"

# Arena-aligned user stem (keep in sync with `benchmarks/game_task_arena_examples.json` intent).
HUD_TASK_USER = (
    "Task `hud-status-summary` (preview `/test-env/hud-status-summary`): add a compact empire status "
    "summary (morale, supply pressure, territory risk, collapse/match pressure, holdings). "
    "Implement as a client overlay under `src/components/test/overlays/`, import it in "
    "`src/components/test/TestEnvironmentShell.tsx`, and render "
    "`{environment.id === 'hud-status-summary' && <YourOverlay />}` immediately before `<GameScene />`. "
    "Preserve existing spectate bootstrap: `generateWorld({ width: 38, height: 38, "
    "seed: environment.seed ?? 910_000, ensureCornerLand: true, mapTerrain: 'continents' })` then "
    "`startSpectateMatch({ opponentCount: environment.opponentCount ?? 1 })`, then defer "
    "`setRealTimePaused(true)`. Do not replace `<main>` or invent store APIs. "
    "Return applyable fenced files only."
)

# If a pairwise winner for this task mentions these, it likely failed TSC in the wild — drop it.
_PAIRWISE_REJECT_SUBSTRINGS: Tuple[str, ...] = (
    "matchstate",
    "myid",
    "goldpile",
    "territoryriskstate",
    "joinmatch",
    "(arena as any)",
    "</1>",
    "startnewmatch",
    "getmatch",
)


def _stable_unit(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _split_rows(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get("record_id") or row.get("task_id") or row.get("trial_id") or id(row))
        u = _stable_unit(seed, key)
        if u < 0.80:
            train.append(row)
        elif u < 0.90:
            valid.append(row)
        else:
            test.append(row)
    if rows and not train:
        train.append(rows[0])
    if len(rows) > 1 and not valid:
        valid.append(rows[1])
    if len(rows) > 2 and not test:
        test.append(rows[2])
    return train, valid, test


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _hud_baseline_rows(baselines_dir: Path, *, baseline_shards: int) -> List[Dict[str, Any]]:
    baseline_path = baselines_dir / f"{TARGET_TASK_ID}.assistant.txt"
    if not baseline_path.is_file():
        return []
    text = baseline_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    base_row: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a careful coding assistant."},
            {"role": "user", "content": HUD_TASK_USER},
            {"role": "assistant", "content": text},
        ],
        "task_id": TARGET_TASK_ID,
        "task_type": "hud_status",
        "complexity": "medium",
        "risk_class": "medium",
        "dataset_role": "core_hud",
        "lineage": LINEAGE,
        "policy_version": "router_policy_v1",
    }
    n = max(1, baseline_shards)
    return [copy.deepcopy({**base_row, "record_id": f"baseline:hud-status-summary:{i}"}) for i in range(n)]


def _guardrail_rows() -> List[Dict[str, Any]]:
    """Short instruction→reply rows that anchor real store usage (no full gold dump)."""
    common = {
        "task_id": TARGET_TASK_ID,
        "task_type": "hud_status",
        "complexity": "low",
        "risk_class": "low",
        "dataset_role": "core_hud_guard",
        "lineage": LINEAGE,
        "policy_version": "router_policy_v1",
    }
    a1 = (
        "For `hud-status-summary` I will only read Fallen Empire state via `useGameStore` selectors "
        "that exist in this repo, e.g. `useGameStore(s => s.players)`, `s.cities`, `s.units`, "
        "`s.tiles`, `s.territory`, `s.contestedZoneHexKeys`, `s.cycle`. "
        "I will not invent `matchState`, `myId`, `goldPile`, or types not exported from `@/types/game`. "
        "The overlay lives under `src/components/test/overlays/` and mounts from `TestEnvironmentShell` "
        "before `<GameScene />` for `environment.id === 'hud-status-summary'`."
    )
    a2 = (
        "`startSpectateMatch` only accepts `{ opponentCount }`. I will call `generateWorld` with "
        "`environment.seed` first, then `startSpectateMatch`, then `setRealTimePaused(true)` in a "
        "timeout—matching the shipped test shell."
    )
    a3 = (
        "I will double-check JSX: every opening tag has a valid closing tag (e.g. `</h1>` not `</1>`) "
        "and run mental lint against `Player` / `GameState` fields from `src/types/game.ts` and "
        "`useGameStore` before emitting code."
    )
    users = [
        "Restate store/API constraints for the HUD test-env overlay before you write files.",
        "How must spectate bootstrap stay compatible with the real zustand store?",
        "What JSX and typing pitfalls should you avoid for this task?",
    ]
    replies = [a1, a2, a3]
    out: List[Dict[str, Any]] = []
    for i, (u, r) in enumerate(zip(users, replies)):
        out.append(
            {
                **common,
                "messages": [
                    {"role": "system", "content": "You are a careful coding assistant."},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": r},
                ],
                "record_id": f"guardrail:hud-status-summary:{i}",
            }
        )
    return out


def _pairwise_winner_rejected(winner_text: str, reject: Sequence[str]) -> bool:
    lower = winner_text.lower()
    return any(s in lower for s in reject)


def _pairwise_rows(
    pairwise_jsonl: Path,
    *,
    target_task_id: str,
    transfer_task_ids: Tuple[str, ...],
    max_core_rows: int,
    max_transfer_rows: int,
    filter_hud_core: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = _read_jsonl(pairwise_jsonl)
    core: List[Dict[str, Any]] = []
    transfer: List[Dict[str, Any]] = []
    transfer_set = set(transfer_task_ids)
    for rec in rows:
        task = rec.get("task") or {}
        task_id = str(task.get("id") or "")
        winner_text = (rec.get("winner_output") or "").strip()
        if not winner_text:
            continue
        if task_id == target_task_id and filter_hud_core:
            if _pairwise_winner_rejected(winner_text, _PAIRWISE_REJECT_SUBSTRINGS):
                continue
        prompt = (task.get("prompt") or "").strip()
        if not prompt:
            continue
        out = {
            "messages": [
                {"role": "system", "content": "You are a careful coding assistant."},
                {"role": "user", "content": f"{prompt}\n\nReturn applyable output only."},
                {"role": "assistant", "content": winner_text},
            ],
            "task_id": task_id,
            "task_type": str(task.get("task_type") or "ui"),
            "complexity": "medium",
            "risk_class": "medium",
            "record_id": f"pairwise:{rec.get('trial_id','unknown')}",
            "trial_id": rec.get("trial_id"),
            "policy_version": "router_policy_v1",
            "lineage": LINEAGE,
        }
        if task_id == target_task_id:
            out["dataset_role"] = "core_hud"
            core.append(out)
        elif task_id in transfer_set:
            out["dataset_role"] = "transfer_ui"
            transfer.append(out)
    return core[:max_core_rows], transfer[:max_transfer_rows]


def _shared_rows(shared_anchor_dir: Path, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        src = shared_anchor_dir / f"{split}.jsonl"
        for row in _read_jsonl(src):
            if not isinstance(row.get("messages"), list):
                continue
            out.append(
                {
                    **row,
                    "dataset_role": "shared_anchor",
                    "lineage": LINEAGE,
                    "policy_version": "router_policy_v1",
                    "record_id": str(row.get("record_id") or row.get("task_id") or f"shared:{len(out)}"),
                }
            )
            if len(out) >= limit:
                return out
    return out


def _sample_repeated(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    if not rows or target <= 0:
        return []
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < target:
        out.append(dict(rows[i % len(rows)]))
        i += 1
    return out


def build_dataset(
    *,
    pairwise_jsonl: Path,
    baselines_dir: Path,
    shared_anchor_dir: Path,
    out_dir: Path,
    seed: int,
    core_ratio: float,
    transfer_ratio: float,
    shared_ratio: float,
    max_core_rows: int,
    max_transfer_rows: int,
    min_train_core_rows: int,
    transfer_task_ids: Tuple[str, ...],
    baseline_shards: int,
    include_guardrails: bool,
    filter_pairwise_hud: bool,
) -> Dict[str, Any]:
    baseline_rows = _hud_baseline_rows(baselines_dir, baseline_shards=baseline_shards)
    guard_rows = _guardrail_rows() if include_guardrails else []
    core_pairwise, transfer_rows = _pairwise_rows(
        pairwise_jsonl,
        target_task_id=TARGET_TASK_ID,
        transfer_task_ids=transfer_task_ids,
        max_core_rows=max_core_rows,
        max_transfer_rows=max_transfer_rows,
        filter_hud_core=filter_pairwise_hud,
    )
    core_rows = baseline_rows + guard_rows + core_pairwise
    if not core_rows:
        raise SystemExit(
            "No hud-status-summary core rows found (baseline + pairwise). Cannot build specialist dataset."
        )
    shared_rows = _shared_rows(shared_anchor_dir, limit=max(20, len(core_rows)))

    train_core, valid_core, test_core = _split_rows(core_rows, seed)
    train_transfer, valid_transfer, test_transfer = _split_rows(transfer_rows, seed + 1)
    train_shared, valid_shared, test_shared = _split_rows(shared_rows, seed + 2)

    target_core = max(1, len(train_core), min_train_core_rows)
    target_transfer = int(round(target_core * (transfer_ratio / max(1e-6, core_ratio))))
    target_shared = int(round(target_core * (shared_ratio / max(1e-6, core_ratio))))

    train_rows = (
        _sample_repeated(train_core, target_core)
        + _sample_repeated(train_transfer, target_transfer)
        + _sample_repeated(train_shared, target_shared)
    )
    valid_rows = (
        valid_core
        + _sample_repeated(valid_transfer, max(1, len(valid_core) // 2))
        + _sample_repeated(valid_shared, max(1, len(valid_core) // 4))
    )
    test_rows = (
        test_core
        + _sample_repeated(test_transfer, max(1, len(test_core) // 2))
        + _sample_repeated(test_shared, max(1, len(test_core) // 4))
    )

    if not valid_rows:
        valid_rows = _sample_repeated(train_rows, 1)
    if not test_rows:
        test_rows = _sample_repeated(train_rows, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    manifest = {
        "schema_version": "hud_status_specialist_dataset_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "policy_version": "router_policy_v1",
        "lineage": LINEAGE,
        "target_task_id": TARGET_TASK_ID,
        "transfer_task_ids": list(transfer_task_ids),
        "alignment": {
            "baseline_shards": baseline_shards,
            "guardrail_rows": len(guard_rows),
            "pairwise_hud_max": max_core_rows,
            "filter_pairwise_hud": filter_pairwise_hud,
        },
        "sources": {
            "pairwise_jsonl": str(pairwise_jsonl),
            "baselines_dir": str(baselines_dir),
            "shared_anchor_dir": str(shared_anchor_dir),
        },
        "mix": {
            "core_hud": core_ratio,
            "transfer_ui": transfer_ratio,
            "shared_anchor": shared_ratio,
        },
        "min_train_core_rows": min_train_core_rows,
        "raw_counts": {
            "core_rows": len(core_rows),
            "baseline_rows": len(baseline_rows),
            "guardrail_rows": len(guard_rows),
            "pairwise_hud_rows": len(core_pairwise),
            "transfer_rows": len(transfer_rows),
            "shared_rows": len(shared_rows),
        },
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HUD-status specialist dataset with transfers.")
    ap.add_argument("--pairwise-jsonl", type=Path, default=PAIRWISE_DEFAULT)
    ap.add_argument("--baselines-dir", type=Path, default=BASELINES_DEFAULT)
    ap.add_argument("--shared-anchor-dir", type=Path, default=SHARED_ANCHOR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--core-ratio", type=float, default=0.70)
    ap.add_argument("--transfer-ratio", type=float, default=0.20)
    ap.add_argument("--shared-ratio", type=float, default=0.10)
    ap.add_argument(
        "--max-core-rows",
        type=int,
        default=0,
        help="Max HUD pairwise core rows (0 = baseline+guardrails only unless you raise this).",
    )
    ap.add_argument("--max-transfer-rows", type=int, default=80)
    ap.add_argument("--min-train-core-rows", type=int, default=100)
    ap.add_argument(
        "--baseline-shards",
        type=int,
        default=12,
        help="Duplicate curator baseline this many times with distinct record_id (strong gold signal).",
    )
    ap.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Omit store/API/JSX guardrail instruction rows.",
    )
    ap.add_argument(
        "--no-filter-pairwise-hud",
        action="store_true",
        help="Keep all HUD pairwise winners (default: drop rows matching known bad substrings).",
    )
    ap.add_argument(
        "--transfer-task-id",
        action="append",
        dest="transfer_task_ids",
        default=list(DEFAULT_TRANSFER_TASKS),
    )
    args = ap.parse_args()

    total = args.core_ratio + args.transfer_ratio + args.shared_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("core/transfer/shared ratios must sum to 1.0")

    transfer_ids = tuple(dict.fromkeys(args.transfer_task_ids))
    manifest = build_dataset(
        pairwise_jsonl=args.pairwise_jsonl.expanduser().resolve(),
        baselines_dir=args.baselines_dir.expanduser().resolve(),
        shared_anchor_dir=args.shared_anchor_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        seed=args.seed,
        core_ratio=args.core_ratio,
        transfer_ratio=args.transfer_ratio,
        shared_ratio=args.shared_ratio,
        max_core_rows=args.max_core_rows,
        max_transfer_rows=args.max_transfer_rows,
        min_train_core_rows=args.min_train_core_rows,
        transfer_task_ids=transfer_ids,
        baseline_shards=args.baseline_shards,
        include_guardrails=not args.no_guardrails,
        filter_pairwise_hud=not args.no_filter_pairwise_hud,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
