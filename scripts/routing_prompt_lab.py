#!/usr/bin/env python3
"""
Interactive routing prompt lab.

Use this tool to:
- type live prompts (vibe-coding style),
- batch score prompts from text/jsonl files,
- capture lineage-rich JSONL rows for downstream routing dataset builds.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO / "benchmarks" / "results" / "routing_prompt_lab"
ALLOWED_SOURCES = {"live", "benchmark", "curated"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUT_DIR / f"{stamp}.jsonl"


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _iter_prompts_from_text(path: Path) -> Iterable[str]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            yield text


def _iter_prompts_from_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            continue
        yield row


def _load_prompts(
    direct_prompts: List[str],
    prompts_file: Optional[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prompt in direct_prompts:
        text = prompt.strip()
        if text:
            rows.append({"prompt": text})

    if not prompts_file:
        return rows

    if not prompts_file.is_file():
        raise SystemExit(f"Prompts file not found: {prompts_file}")

    if prompts_file.suffix.lower() == ".jsonl":
        for row in _iter_prompts_from_jsonl(prompts_file):
            prompt = str(row.get("prompt") or "").strip()
            if prompt:
                rows.append(row)
        return rows

    for prompt in _iter_prompts_from_text(prompts_file):
        rows.append({"prompt": prompt})
    return rows


def _normalize_source(source: str) -> str:
    source = source.strip().lower()
    if source not in ALLOWED_SOURCES:
        raise SystemExit(f"--source must be one of {sorted(ALLOWED_SOURCES)}")
    return source


def _build_row(
    prompt: str,
    policy: RoutingPolicy,
    *,
    source: str,
    expected_adapter_id: Optional[str],
    expected_legacy_route: Optional[str],
    reviewer: Optional[str],
    notes: Optional[str],
    accepted_for_training: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request = GenerationRequest(messages=messages_from_prompt(prompt))
    decision = policy.decide(request)
    row: Dict[str, Any] = {
        "record_id": _prompt_id(prompt),
        "prompt": prompt,
        "source": source,
        "created_at": _utc_now(),
        "policy_version": decision.policy_version,
        "lineage": {
            "tool": "routing_prompt_lab",
            "policy_lineage": decision.lineage,
        },
        "predicted_adapter_id": decision.adapter_id,
        "predicted_legacy_route": decision.route,
        "confidence": decision.confidence,
        "ambiguity": decision.ambiguity,
        "risk_class": decision.risk_class,
        "complexity": decision.complexity,
        "reason": decision.reason,
        "accepted_for_training": accepted_for_training,
    }
    if expected_adapter_id:
        row["expected_adapter_id"] = expected_adapter_id
        row["adapter_match"] = decision.adapter_id == expected_adapter_id
    if expected_legacy_route:
        row["expected_legacy_route"] = expected_legacy_route
        row["route_match"] = decision.route == expected_legacy_route
    if reviewer:
        row["reviewer"] = reviewer
    if notes:
        row["notes"] = notes
    if extra:
        for key, value in extra.items():
            if key in {"prompt"}:
                continue
            row.setdefault(key, value)
    return row


def _interactive_prompts() -> Iterable[str]:
    print("Routing prompt lab interactive mode. Submit empty line to stop.")
    while True:
        try:
            text = input("prompt> ").strip()
        except EOFError:
            print("")
            return
        if not text:
            return
        yield text


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive routing prompt lab")
    ap.add_argument("--prompt", action="append", default=[], help="Inline prompt (repeatable)")
    ap.add_argument("--prompts-file", type=Path, default=None, help="Text or JSONL prompt source")
    ap.add_argument("--interactive", action="store_true", help="Read prompts from stdin interactively")
    ap.add_argument("--source", default="live", help="Row source label: live|benchmark|curated")
    ap.add_argument("--expected-adapter-id", default=None)
    ap.add_argument("--expected-legacy-route", default=None, choices=["local", "hybrid", "frontier"])
    ap.add_argument("--accepted-for-training", action="store_true")
    ap.add_argument("--reviewer", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--output-jsonl", type=Path, default=None)
    args = ap.parse_args()

    source = _normalize_source(args.source)
    policy = RoutingPolicy()
    prompt_rows = _load_prompts(args.prompt, args.prompts_file)
    if args.interactive:
        prompt_rows.extend({"prompt": p} for p in _interactive_prompts())
    if not prompt_rows:
        raise SystemExit("No prompts provided. Use --prompt, --prompts-file, or --interactive.")

    rows: List[Dict[str, Any]] = []
    for row in prompt_rows:
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        built = _build_row(
            prompt,
            policy,
            source=source,
            expected_adapter_id=args.expected_adapter_id or row.get("expected_adapter_id"),
            expected_legacy_route=args.expected_legacy_route or row.get("expected_legacy_route"),
            reviewer=args.reviewer or row.get("reviewer"),
            notes=args.notes or row.get("notes"),
            accepted_for_training=bool(args.accepted_for_training or row.get("accepted_for_training")),
            extra=row,
        )
        rows.append(built)

    if not rows:
        raise SystemExit("No usable prompts after parsing input.")

    output = (args.output_jsonl or _default_output_path()).expanduser().resolve()
    _write_jsonl(output, rows)

    print(f"Wrote {len(rows)} routing rows to {output}")
    for row in rows:
        rid = row["record_id"]
        adapter = row["predicted_adapter_id"]
        route = row["predicted_legacy_route"]
        conf = row["confidence"]
        print(f"- {rid}: adapter={adapter} route={route} conf={conf:.3f}")


if __name__ == "__main__":
    main()
