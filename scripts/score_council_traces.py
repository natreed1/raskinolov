#!/usr/bin/env python3
"""Score council trace JSONL with a trainable reward matrix contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from council_runtime.coevolution import CoevolutionState, DEFAULT_STATE_PATH, RewardMatrix, default_reward_population
from council_runtime.trace_scoring import score_trace_file


def _load_reward_matrix(args: argparse.Namespace) -> RewardMatrix:
    if args.reward_matrix_json:
        payload = json.loads(args.reward_matrix_json.expanduser().resolve().read_text(encoding="utf-8"))
        return RewardMatrix.from_dict(payload)
    if args.state_path.expanduser().resolve().is_file():
        state = CoevolutionState.load(args.state_path)
        if args.reward_matrix_id:
            return state.reward_population.by_id()[args.reward_matrix_id]
        return state.reward_population.active()
    population = default_reward_population()
    if args.reward_matrix_id:
        return population.by_id()[args.reward_matrix_id]
    return population.active()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-jsonl", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--reward-matrix-id", default="")
    parser.add_argument("--reward-matrix-json", type=Path)
    args = parser.parse_args()

    reward_matrix = _load_reward_matrix(args)
    summary = score_trace_file(
        trace_path=args.traces_jsonl,
        reward_matrix=reward_matrix,
        scored_trace_path=args.scored_jsonl,
        summary_path=args.summary_json,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
