"""Tests for the reusable council executor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CouncilRuntimeExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_executor_runs_rounds_and_preserves_participant_metadata(self) -> None:
        from council_runtime.executor import CouncilGeneration, build_eval_participant_prompt, run_council

        prompts_seen: list[str] = []
        plan = {
            "participants": [
                {
                    "participant_id": "wide_compressed",
                    "base_expert_id": "wide_compressed",
                    "participant_type": "generalist_profile",
                    "role": "generalist",
                    "strategy": "broad_summary_context",
                    "variant": "baseline",
                    "assertiveness": 0.6,
                    "traits": {"verbosity": 0.62},
                },
                {
                    "participant_id": "economistRL::risk_auditor",
                    "base_expert_id": "economistRL",
                    "participant_type": "specialist_adapter",
                    "role": "specialist",
                    "strategy": "adapter_specialist_focus",
                    "variant": "risk_auditor",
                    "assertiveness": 0.4,
                    "traits": {"skepticism": 0.82},
                },
            ],
            "debate_max_rounds": 2,
            "low_confidence_threshold": 0.58,
            "disagreement_threshold": 0.45,
            "escalation_rule": "manual_only",
        }

        def generate(turn):
            prompts_seen.append(turn.participant_prompt)
            pid = turn.participant["participant_id"]
            return CouncilGeneration(
                text=f"{pid} says verify `src/economy.ts` and run tests for risk.",
                metadata={
                    "adapter_requested": turn.participant.get("base_expert_id"),
                    "adapter_loaded": False,
                    "usage": {"backend": "mock"},
                },
            )

        final_text, meta = run_council(
            prompt="Patch food population economy logic and verify tests.",
            council_plan=plan,
            disagreement=0.2,
            generate_participant=generate,
            prompt_builder=build_eval_participant_prompt,
            max_rounds=2,
            stop_on_convergence=False,
        )

        self.assertIn("verify", final_text.lower())
        self.assertEqual(meta["debate_rounds_run"], 2)
        self.assertEqual(len(meta["rounds"]), 2)
        self.assertEqual(len(meta["participants"]), 2)
        self.assertIn("Peer summaries", prompts_seen[-1])
        specialist = [p for p in meta["participants"] if p["participant_type"] == "specialist_adapter"][0]
        self.assertEqual(specialist["adapter_requested"], "economistRL")
        self.assertFalse(specialist["adapter_loaded"])

    def test_executor_reports_empty_plan(self) -> None:
        from council_runtime.executor import run_council

        final_text, meta = run_council(
            prompt="Task",
            council_plan={"participants": []},
            disagreement=0.0,
            generate_participant=lambda turn: None,  # type: ignore[arg-type]
        )

        self.assertEqual(final_text, "")
        self.assertEqual(meta["error"], "empty_council_participants")

    def test_executor_emits_structured_traces_when_sink_is_supplied(self) -> None:
        from council_runtime.executor import CouncilGeneration, build_eval_participant_prompt, run_council
        from council_runtime.traces import GRADER_TRACE_SCHEMA, PLANNER_TRACE_SCHEMA, SPECIALIST_EQ_TRACE_SCHEMA

        traces = []
        plan = {
            "participants": [
                {
                    "participant_id": "hud_status::critic",
                    "base_expert_id": "hud_status",
                    "participant_type": "specialist_adapter",
                    "role": "critic",
                    "strategy": "ui_state_review",
                    "variant": "critic",
                }
            ],
            "debate_max_rounds": 1,
            "low_confidence_threshold": 0.58,
            "disagreement_threshold": 0.45,
        }

        def generate(turn):
            return CouncilGeneration(
                text="Verify the HUD file path and run the UI state tests because the risk is stale state.",
                metadata={"confidence": 0.7, "task_outcome_score": 0.8},
            )

        _, meta = run_council(
            prompt="Fix HUD state refresh.",
            council_plan=plan,
            disagreement=0.1,
            generate_participant=generate,
            prompt_builder=build_eval_participant_prompt,
            max_rounds=1,
            trace_sink=traces.append,
            trace_context={"task_id": "task_hud_1", "season_id": "season_test"},
        )

        schemas = [record.trace_schema for record in traces]
        self.assertEqual(meta["trace_ids"], [record.trace_id for record in traces])
        self.assertIn(PLANNER_TRACE_SCHEMA, schemas)
        self.assertIn(SPECIALIST_EQ_TRACE_SCHEMA, schemas)
        self.assertEqual(schemas.count(GRADER_TRACE_SCHEMA), 2)
        specialist = [record for record in traces if record.trace_schema == SPECIALIST_EQ_TRACE_SCHEMA][0]
        self.assertEqual(specialist.archetype_id, "hud_status_eq")
        self.assertGreater(specialist.metrics["evidence_quality"], 0.0)


if __name__ == "__main__":
    unittest.main()
