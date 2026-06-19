"""Tests for trace contracts and reward scoring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CouncilTraceScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_trace_writer_and_reward_scoring_outputs_fitness_summary(self) -> None:
        from council_runtime.coevolution import default_reward_population
        from council_runtime.trace_scoring import SCORED_TRACE_SCHEMA, TRACE_SCORE_SUMMARY_SCHEMA, score_trace_file
        from council_runtime.traces import JsonlTraceWriter, build_grader_trace, build_planner_trace, build_specialist_trace

        plan = {
            "participants": [
                {
                    "participant_id": "economistRL::risk",
                    "base_expert_id": "economistRL",
                    "participant_type": "specialist_adapter",
                    "role": "specialist",
                    "strategy": "risk_review",
                }
            ],
            "debate_max_rounds": 1,
            "low_confidence_threshold": 0.58,
            "disagreement_threshold": 0.45,
        }
        participant_output = {
            "participant_id": "economistRL::risk",
            "base_expert_id": "economistRL",
            "participant_type": "specialist_adapter",
            "role": "specialist",
            "strategy": "risk_review",
            "text": "Verify the economy file, run tests, and check risk around save/load migration.",
            "confidence": 0.72,
            "task_outcome_score": 0.8,
        }
        adjudication = {
            "winner_ids": ["economistRL::risk"],
            "confidence": 0.74,
            "disagreement": 0.12,
            "escalation_recommended": False,
            "final_text": "Use the risk review and run tests.",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces_path = root / "traces.jsonl"
            writer = JsonlTraceWriter(traces_path)
            context = {"task_id": "task_1", "season_id": "season_1", "final_outcome": 0.8}
            for record in (
                build_planner_trace(prompt="Patch economy logic", council_plan=plan, trace_context=context),
                build_specialist_trace(
                    prompt="Patch economy logic",
                    participant_output=participant_output,
                    round_idx=1,
                    trace_context=context,
                ),
                build_grader_trace(
                    prompt="Patch economy logic",
                    adjudication=adjudication,
                    round_idx=1,
                    participant_outputs=[participant_output],
                    trace_context=context,
                    final=True,
                ),
            ):
                writer(record)

            summary = score_trace_file(
                trace_path=traces_path,
                reward_matrix=default_reward_population().active(),
                scored_trace_path=root / "scored.jsonl",
                summary_path=root / "summary.json",
            )
            scored_rows = [
                json.loads(line)
                for line in (root / "scored.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary_payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(len(scored_rows), 3)
            self.assertEqual(scored_rows[0]["schema_version"], SCORED_TRACE_SCHEMA)
            self.assertEqual(summary_payload["schema_version"], TRACE_SCORE_SUMMARY_SCHEMA)
            self.assertEqual(summary.trace_count, 3)
            self.assertIn("planner", summary.by_archetype)
            self.assertIn("economistRL_eq", summary.by_archetype)
            self.assertGreater(summary.overall_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
