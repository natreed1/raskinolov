#!/usr/bin/env python3
"""Run a council coevolution season from trace JSONL to population fitness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from council_runtime.coevolution import CoevolutionState, DEFAULT_RESULTS_ROOT, DEFAULT_STATE_PATH, next_season_id
from council_runtime.population_update import VALID_POPULATION_UPDATE_MODES
from council_runtime.season_ops import CouncilSeasonOperator, SeasonContext


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--traces-jsonl", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--season-id", default="")
    parser.add_argument("--mode", choices=sorted(VALID_POPULATION_UPDATE_MODES), default="score_only")
    args = parser.parse_args()

    state = CoevolutionState.load(args.state_path.expanduser().resolve())
    season_id = args.season_id.strip() or next_season_id(args.results_root)
    run_dir = args.results_root.expanduser().resolve() / "seasons" / season_id
    result = CouncilSeasonOperator().run(
        context=SeasonContext(
            season_id=season_id,
            traces_jsonl=args.traces_jsonl,
            run_dir=run_dir,
            state_path=args.state_path,
            mode=args.mode,
        ),
        state=state,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
