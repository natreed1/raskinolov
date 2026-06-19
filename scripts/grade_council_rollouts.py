#!/usr/bin/env python3
"""Grade council rollouts with deterministic planner reward functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from council_runtime.reward_functions import grade_trace_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-jsonl", type=Path, required=True)
    parser.add_argument("--graded-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    summary = grade_trace_file(
        trace_path=args.traces_jsonl,
        graded_rollout_path=args.graded_jsonl,
        summary_path=args.summary_json,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
