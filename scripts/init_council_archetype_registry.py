#!/usr/bin/env python3
"""Initialize the council archetype registry contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from council_runtime.archetypes import CouncilArchetypeRegistry, DEFAULT_REGISTRY_PATH

REPO = Path(__file__).resolve().parent.parent


def _load_specialist_ids(registry_path: Path) -> set[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for row in rows:
        aid = str((row or {}).get("adapter_id") or "").strip()
        if aid and aid != "general_fallback" and bool((row or {}).get("is_specialized", False)):
            out.add(aid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize council archetype registry.")
    parser.add_argument(
        "--adapter-registry",
        type=Path,
        default=REPO / "training" / "adapter_registry_v1.json",
        help="Specialist adapter registry used to seed specialist EQ archetype lanes.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Output council archetype registry path.",
    )
    args = parser.parse_args()

    specialist_ids = _load_specialist_ids(args.adapter_registry.expanduser().resolve())
    registry = CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids=specialist_ids)
    registry.save(args.out_json)
    print(f"Wrote council archetype registry -> {args.out_json.expanduser().resolve()}")
    print(f"Archetypes: {len(registry.entries)}")
    print(f"Specialist EQ lanes: {sum(1 for entry in registry.entries if entry.archetype == 'specialist_eq')}")


if __name__ == "__main__":
    main()
