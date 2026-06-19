#!/usr/bin/env python3
"""Run the council co-evolution framework.

V1 is intentionally a dry-run scaffold: it bootstraps state, writes a season
manifest, and records all repeatable phase contracts without training adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from council_runtime.archetypes import CouncilArchetypeRegistry, DEFAULT_REGISTRY_PATH
from council_runtime.coevolution import DEFAULT_RESULTS_ROOT, DEFAULT_STATE_PATH
from council_runtime.coevolution_runner import run_dry_season


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetype-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--season-id", default="")
    parser.add_argument(
        "--bootstrap-state",
        action="store_true",
        help="Rewrite/bootstrap coevolution state from the archetype registry before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Only dry-run is implemented in v1.",
    )
    args = parser.parse_args()

    registry = CouncilArchetypeRegistry.load(args.archetype_registry.expanduser().resolve())
    manifest = run_dry_season(
        archetype_registry=registry,
        state_path=args.state_path,
        results_root=args.results_root,
        season_id=args.season_id.strip() or None,
        bootstrap=bool(args.bootstrap_state),
    )
    print(json.dumps({"status": manifest.status, "season_id": manifest.season_id, "run_dir": manifest.run_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
