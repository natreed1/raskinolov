"""Tests for planner rollout documents and deterministic rewards."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _good_planner_document() -> dict:
    return {
        "schema_version": "planner_rollout_plan_v1",
        "task_synopsis": "Implement a safe economy persistence fix and verify the migration risk with targeted tests.",
        "primary_goal": "implementation: satisfy the user request with verified, low-risk changes.",
        "success_criteria": [
            "Relevant economy and save/load files are identified.",
            "Targeted tests or verification steps are run.",
            "Final answer explains remaining risk.",
        ],
        "known_constraints": ["Do not change unrelated economy behavior."],
        "risk_tags": ["economy_balance", "persistence", "test_coverage"],
        "required_context": ["economy files", "save/load migration tests", "recent traces"],
        "selected_experts": [
            {
                "expert_id": "economistRL",
                "role": "economy specialist",
                "why_selected": "Economy balance risk is central to the task.",
                "assigned_question": "Identify economy mechanics and verification requirements.",
                "expected_contribution": "Concrete risk review with files and tests.",
                "handoff_context": "Focus on food, morale, market, and persistence interactions.",
            },
            {
                "expert_id": "save_load",
                "role": "persistence guard",
                "why_selected": "Save/load migration risk can break persisted games.",
                "assigned_question": "Check schema and migration implications.",
                "expected_contribution": "API and migration risk guidance.",
                "handoff_context": "Focus on saved state compatibility and tests.",
            },
        ],
        "excluded_experts": [{"expert_id": "hud_status", "why_excluded": "No UI state risk in this task."}],
        "debate_plan": {
            "max_rounds": 1,
            "reroute_conditions": ["low confidence", "missing persistence evidence"],
            "stop_conditions": ["tests and risks covered"],
        },
    }


class CouncilPlannerRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_planner_document_validation_grades_malformed_without_crashing(self) -> None:
        from council_runtime.planner_plan import validate_planner_document

        validation = validate_planner_document({"task_synopsis": "too small", "selected_experts": [{}]})

        self.assertFalse(validation.valid)
        self.assertIn("missing_or_empty:primary_goal", validation.diagnostics)
        self.assertIn("selected_experts[0]:missing:expert_id", validation.diagnostics)
        self.assertLess(validation.completeness, 0.5)

    def test_planner_reward_scores_good_rollout_above_bad_rollout(self) -> None:
        from council_runtime.reward_functions import PlannerRewardFunction
        from council_runtime.rollouts import group_traces_into_rollouts
        from council_runtime.traces import build_grader_trace, build_planner_trace, build_specialist_trace

        council_plan = {
            "participants": [
                {"participant_id": "economistRL::risk", "base_expert_id": "economistRL", "participant_type": "specialist_adapter", "role": "specialist", "strategy": "economy_risk"},
                {"participant_id": "save_load::guard", "base_expert_id": "save_load", "participant_type": "specialist_adapter", "role": "guard", "strategy": "persistence_guard"},
            ],
            "debate_max_rounds": 1,
        }
        context = {"task_id": "task_good", "season_id": "season_1", "final_outcome": 0.85}
        good_records = [
            build_planner_trace(
                prompt="Implement economy save/load fix and verify tests.",
                council_plan=council_plan,
                planner_document=_good_planner_document(),
                trace_context=context,
            ),
            build_specialist_trace(
                prompt="Implement economy save/load fix and verify tests.",
                participant_output={
                    "participant_id": "economistRL::risk",
                    "base_expert_id": "economistRL",
                    "participant_type": "specialist_adapter",
                    "text": "Verify economy files, tests, and risk around food and morale.",
                    "confidence": 0.72,
                    "task_outcome_score": 0.85,
                    "contribution_credit": 0.9,
                    "evidence_quality": 0.8,
                },
                round_idx=1,
                trace_context=context,
            ),
            build_specialist_trace(
                prompt="Implement economy save/load fix and verify tests.",
                participant_output={
                    "participant_id": "save_load::guard",
                    "base_expert_id": "save_load",
                    "participant_type": "specialist_adapter",
                    "text": "Check save/load migration schema and run persistence tests.",
                    "confidence": 0.72,
                    "task_outcome_score": 0.8,
                    "contribution_credit": 0.82,
                    "evidence_quality": 0.78,
                },
                round_idx=1,
                trace_context=context,
            ),
            build_grader_trace(
                prompt="Implement economy save/load fix and verify tests.",
                adjudication={
                    "winner_ids": ["economistRL::risk", "save_load::guard"],
                    "confidence": 0.82,
                    "disagreement": 0.1,
                    "escalation_recommended": False,
                    "final_text": "Use both expert reviews and run targeted tests.",
                },
                round_idx=1,
                participant_outputs=[],
                trace_context=context,
                final=True,
            ),
        ]
        bad_context = {"task_id": "task_bad", "season_id": "season_1", "final_outcome": 0.25}
        bad_records = [
            build_planner_trace(
                prompt="Implement economy save/load fix and verify tests.",
                council_plan={
                    "participants": [
                        {"participant_id": "hud_status::critic", "base_expert_id": "hud_status", "participant_type": "specialist_adapter", "role": "critic"}
                    ],
                    "debate_max_rounds": 4,
                },
                planner_document={"task_synopsis": "do thing", "selected_experts": [{}]},
                trace_context=bad_context,
            ),
            build_specialist_trace(
                prompt="Implement economy save/load fix and verify tests.",
                participant_output={
                    "participant_id": "hud_status::critic",
                    "base_expert_id": "hud_status",
                    "participant_type": "specialist_adapter",
                    "text": "Maybe inspect UI.",
                    "confidence": 0.5,
                    "task_outcome_score": 0.2,
                    "contribution_credit": 0.1,
                    "evidence_quality": 0.0,
                },
                round_idx=1,
                trace_context=bad_context,
            ),
            build_grader_trace(
                prompt="Implement economy save/load fix and verify tests.",
                adjudication={"winner_ids": [], "confidence": 0.3, "disagreement": 0.6, "escalation_recommended": True},
                round_idx=1,
                participant_outputs=[],
                trace_context=bad_context,
                final=True,
            ),
        ]

        rollouts = group_traces_into_rollouts([*good_records, *bad_records])
        results = {row.task_id: PlannerRewardFunction().score_rollout(row) for row in rollouts}

        self.assertGreater(results["task_good"].reward, results["task_bad"].reward)
        self.assertGreater(results["task_good"].components["expert_usefulness"], 0.5)
        self.assertGreater(results["task_bad"].components["irrelevant_expert_penalty"], 0.0)
        self.assertIn("missing_or_empty:primary_goal", results["task_bad"].diagnostics)

    def test_grade_trace_file_writes_graded_rollouts_and_summary(self) -> None:
        from council_runtime.reward_functions import GRADED_ROLLOUT_SCHEMA, ROLLOUT_REWARD_SUMMARY_SCHEMA, grade_trace_file
        from council_runtime.traces import JsonlTraceWriter, build_grader_trace, build_planner_trace, build_specialist_trace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = JsonlTraceWriter(root / "traces.jsonl")
            context = {"task_id": "task_file", "season_id": "season_file", "final_outcome": 0.8}
            plan = {"participants": [{"participant_id": "economistRL::risk", "base_expert_id": "economistRL"}], "debate_max_rounds": 1}
            writer(build_planner_trace(prompt="Fix economy tests.", council_plan=plan, planner_document=_good_planner_document(), trace_context=context))
            writer(
                build_specialist_trace(
                    prompt="Fix economy tests.",
                    participant_output={
                        "participant_id": "economistRL::risk",
                        "base_expert_id": "economistRL",
                        "text": "Run tests and verify economy file risk.",
                        "contribution_credit": 0.8,
                        "evidence_quality": 0.8,
                    },
                    round_idx=1,
                    trace_context=context,
                )
            )
            writer(
                build_grader_trace(
                    prompt="Fix economy tests.",
                    adjudication={"winner_ids": ["economistRL::risk"], "confidence": 0.8, "disagreement": 0.1, "escalation_recommended": False},
                    round_idx=1,
                    participant_outputs=[],
                    trace_context=context,
                    final=True,
                )
            )
            summary = grade_trace_file(
                trace_path=root / "traces.jsonl",
                graded_rollout_path=root / "graded.jsonl",
                summary_path=root / "summary.json",
            )
            graded = json.loads((root / "graded.jsonl").read_text(encoding="utf-8").splitlines()[0])
            summary_payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(graded["schema_version"], GRADED_ROLLOUT_SCHEMA)
            self.assertEqual(summary_payload["schema_version"], ROLLOUT_REWARD_SUMMARY_SCHEMA)
            self.assertEqual(summary.rollout_count, 1)
            self.assertGreater(graded["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
