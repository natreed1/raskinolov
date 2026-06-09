#!/usr/bin/env python3
"""Build train/valid/test council-orchestration dataset from interaction logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _stable_key(prompt: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{prompt}".encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _from_router_chat(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("lane") or "") != "specialist_router":
            continue
        prompt = str(row.get("user_message") or "").strip()
        if not prompt:
            continue
        routing = dict(row.get("routing") or {})
        council_plan = dict(routing.get("council_plan") or {})
        multi_agent = dict(row.get("multi_agent") or {})
        adjudication = dict(multi_agent.get("adjudication") or {})
        participants = list(multi_agent.get("participants") or [])
        rounds = list(multi_agent.get("rounds") or [])
        if not council_plan and not participants:
            continue
        out.append(
            {
                "schema_version": "council_training_row_v1",
                "source": "router_chat_interactions",
                "prompt": prompt,
                "policy_version": str(routing.get("policy_version") or row.get("policy_version") or ""),
                "adapter_id": str(routing.get("adapter_id") or ""),
                "council_plan": council_plan,
                "participants": participants,
                "rounds": rounds,
                "adjudication": adjudication,
                "final_output": str(row.get("generation_text_only") or ""),
                "accepted_for_training": bool(row.get("recommended_for_sft_assistant_turn", True)),
            }
        )
    return out


def _from_prompt_lab(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        council_plan = dict(row.get("council_plan") or {})
        if not council_plan:
            continue
        out.append(
            {
                "schema_version": "council_training_row_v1",
                "source": "routing_prompt_lab",
                "prompt": prompt,
                "policy_version": str(row.get("policy_version") or ""),
                "adapter_id": str(row.get("predicted_adapter_id") or ""),
                "council_plan": council_plan,
                "participants": [],
                "adjudication": {},
                "final_output": "",
                "accepted_for_training": bool(row.get("accepted_for_training", False)),
            }
        )
    return out


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep: dict[str, dict[str, Any]] = {}
    rank = {"router_chat_interactions": 2, "routing_prompt_lab": 1}
    for row in rows:
        prompt = str(row.get("prompt") or "").strip().lower()
        if not prompt:
            continue
        existing = keep.get(prompt)
        if not existing:
            keep[prompt] = row
            continue
        src_new = rank.get(str(row.get("source") or ""), 0)
        src_old = rank.get(str(existing.get("source") or ""), 0)
        if src_new >= src_old:
            keep[prompt] = row
    return list(keep.values())


def _split(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = sorted(rows, key=lambda r: _stable_key(str(r.get("prompt") or ""), seed))
    n = len(keyed)
    n_train = max(1, int(n * 0.85))
    n_valid = max(1, int(n * 0.10)) if n >= 10 else 0
    n_test = max(0, n - n_train - n_valid)
    train = keyed[:n_train]
    valid = keyed[n_train : n_train + n_valid]
    test = keyed[n_train + n_valid : n_train + n_valid + n_test]
    return train, valid, test


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build council training dataset from logs.")
    parser.add_argument(
        "--router-chat-jsonl",
        type=Path,
        action="append",
        default=[],
        help="JSONL logs from router_chat_gradio specialist lane",
    )
    parser.add_argument(
        "--prompt-lab-jsonl",
        type=Path,
        action="append",
        default=[],
        help="JSONL logs from routing_prompt_lab",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.router_chat_jsonl:
        rows.extend(_from_router_chat(_load_jsonl(path.expanduser().resolve())))
    for path in args.prompt_lab_jsonl:
        rows.extend(_from_prompt_lab(_load_jsonl(path.expanduser().resolve())))

    rows = [r for r in rows if bool(r.get("accepted_for_training", False))]
    rows = _dedupe(rows)
    if not rows:
        raise SystemExit("No council training rows found.")
    train, valid, test = _split(rows, seed=args.seed)

    out_dir = args.out_root.expanduser().resolve() / args.dataset_version
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "valid.jsonl", valid)
    _write_jsonl(out_dir / "test.jsonl", test)

    manifest = {
        "schema_version": "council_training_dataset_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_version": args.dataset_version,
        "seed": int(args.seed),
        "rows_total": len(rows),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "rows_test": len(test),
        "inputs": {
            "router_chat_jsonl": [str(p.expanduser().resolve()) for p in args.router_chat_jsonl],
            "prompt_lab_jsonl": [str(p.expanduser().resolve()) for p in args.prompt_lab_jsonl],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
