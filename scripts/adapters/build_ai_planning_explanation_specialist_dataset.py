#!/usr/bin/env python3
"""Build AI-planning explanation specialist dataset with benchmark prompt ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_specialist_dataset import build_specialist_dataset

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI-planning explanation specialist dataset.")
    parser.add_argument("--pairwise-jsonl", type=Path, default=REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl")
    parser.add_argument("--shared-anchor-dir", type=Path, default=REPO / "data" / "lora" / "game_text")
    parser.add_argument("--benchmark-tasks-json", type=Path, default=REPO / "benchmarks" / "ai_planning_explanation_mass_tasks_v1.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "lora" / "adapters" / "ai_planning_explanation_specialist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-ratio", type=float, default=0.7)
    parser.add_argument("--transfer-ratio", type=float, default=0.0)
    parser.add_argument("--shared-ratio", type=float, default=0.1)
    parser.add_argument("--max-core-rows", type=int, default=120)
    parser.add_argument("--max-transfer-rows", type=int, default=0)
    parser.add_argument("--max-benchmark-rows", type=int, default=120)
    parser.add_argument("--max-shared-rows", type=int, default=80)
    parser.add_argument("--min-train-core-rows", type=int, default=120)
    parser.add_argument("--transfer-task-id", action="append", dest="transfer_task_ids", default=[])
    args = parser.parse_args()

    manifest = build_specialist_dataset(
        specialist_id="ai_planning_explanation",
        task_id="ai-planning-explanation",
        task_type="planning",
        system_prompt=(
            "You are Albert, an AI planning explanation specialist for Fallen Empire. "
            "Produce concise rationale using real strategic factors and actionable follow-ups."
        ),
        anchor_phrase="AI planning",
        out_dir=args.out_dir.expanduser().resolve(),
        pairwise_jsonl=args.pairwise_jsonl.expanduser().resolve(),
        benchmark_tasks_json=args.benchmark_tasks_json.expanduser().resolve(),
        shared_anchor_dir=args.shared_anchor_dir.expanduser().resolve(),
        transfer_task_ids=args.transfer_task_ids,
        seed=int(args.seed),
        core_ratio=float(args.core_ratio),
        transfer_ratio=float(args.transfer_ratio),
        shared_ratio=float(args.shared_ratio),
        max_core_rows=int(args.max_core_rows),
        max_transfer_rows=int(args.max_transfer_rows),
        max_benchmark_rows=int(args.max_benchmark_rows),
        max_shared_rows=int(args.max_shared_rows),
        min_train_core_rows=int(args.min_train_core_rows),
        strict_specialist_only=True,
        allow_cross_domain_benchmark=False,
        allow_transfer=False,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
