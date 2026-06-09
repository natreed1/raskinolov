#!/usr/bin/env python3
"""Initialize/refresh council roster file from adapter registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from router.roster import ExpertRoster

REPO = Path(__file__).resolve().parent.parent


def _load_specialist_ids(registry_path: Path) -> set[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        aid = str((row or {}).get("adapter_id") or "").strip()
        if not aid or aid == "general_fallback":
            continue
        out.add(aid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize council roster with default traits and personalities.")
    parser.add_argument(
        "--adapter-registry",
        type=Path,
        default=REPO / "training" / "adapter_registry_v1.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "data" / "routing" / "council_roster_v1.json",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge existing roster values where present instead of rewriting defaults.",
    )
    args = parser.parse_args()

    registry_path = args.adapter_registry.expanduser().resolve()
    out_path = args.out_json.expanduser().resolve()
    specialist_ids = _load_specialist_ids(registry_path)
    if args.merge_existing:
        roster = ExpertRoster.load_or_bootstrap(path=out_path, specialist_ids=specialist_ids)
    else:
        roster = ExpertRoster.bootstrap(specialist_ids=specialist_ids)
    roster.save(out_path)
    print(f"Wrote council roster -> {out_path}")
    print(f"Specialists: {len(specialist_ids)}")


if __name__ == "__main__":
    main()
