"""Tests for specialist interaction reward scoring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _doc(expert_id: str = "save_load") -> dict:
    return {
        "schema_version": "planner_rollout_plan_v1",
        "task_synopsis": "Fix save/load migration risk and verify persistence tests.",
        "primary_goal": "debugging: satisfy the requested persistence fix with verified behavior.",
        "success_criteria": ["Save/load tests run", "Migration risk explained"],
        "known_constraints": ["Do not change unrelated UI behavior"],
        "risk_tags": ["persistence", "test_coverage"],
        "required_context": ["save schema", "migration tests"],
        "selected_experts": [
            {
                "expert_id": expert_id,
                "participant_id": f"{expert_id}::guard",
                "role": "persistence guard",
                "why_selected": "Save/load migration risk can break persisted games.",
                "assigned_question": "Check schema and migration implications.",
                "expected_contribution": "API and migration risk guidance with tests.",
                "handoff_context": "Focus on saved state compatibility and persistence tests.",
            }
        ],
        "excluded_experts": [],
        "debate_plan": {"max_rounds": 1, "reroute_conditions": ["missing evidence"], "stop_conditions": ["tests covered"]},
    }


class CouncilSpecialistRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def _rollout(self, *, output: str, confidence: float, final_outcome: float, expert_id: str = "save_load", final_text: str = ""):
        from council_runtime.rollouts import group_traces_into_rollouts
        from council_runtime.traces import build_grader_trace, build_planner_trace, build_specialist_trace

        context = {"task_id": "task_specialist", "season_id": "season_s", "final_outcome": final_outcome}
        plan = {"participants": [{"participant_id": f"{expert_id}::guard", "base_expert_id": expert_id, "participant_type": "specialist_adapter"}], "debate_max_rounds": 1}
        records = [
            build_planner_trace(prompt="Fix save/load migration risk.", council_plan=plan, planner_document=_doc(expert_id), trace_context=context),
            build_specialist_trace(
                prompt="Fix save/load migration risk.",
                participant_output={
                    "participant_id": f"{expert_id}::guard",
                    "base_expert_id": expert_id,
                    "participant_type": "specialist_adapter",
                    "text": output,
                    "confidence": confidence,
                    "task_outcome_score": final_outcome,
                    "contribution_credit": 0.8 if final_outcome > 0.5 else 0.2,
                    "evidence_quality": 0.8 if "test" in output.lower() or "schema" in output.lower() else 0.1,
                },
                round_idx=1,
                trace_context=context,
            ),
            build_grader_trace(
                prompt="Fix save/load migration risk.",
                adjudication={
                    "winner_ids": [f"{expert_id}::guard"] if final_outcome > 0.5 else [],
                    "confidence": final_outcome,
                    "disagreement": 0.1 if final_outcome > 0.5 else 0.6,
                    "escalation_recommended": final_outcome < 0.5,
                    "final_text": final_text,
                },
                round_idx=1,
                participant_outputs=[],
                trace_context=context,
                final=True,
            ),
        ]
        return group_traces_into_rollouts(records)[0]

    def test_in_domain_specialist_with_evidence_scores_high(self) -> None:
        from council_runtime.specialist_rewards import SpecialistRewardFunction

        rollout = self._rollout(
            output="Check the save schema migration, verify persistence tests, and flag compatibility risk.",
            confidence=0.72,
            final_outcome=0.85,
            final_text="Check the save schema migration and run persistence tests.",
        )
        result = SpecialistRewardFunction().score_rollout(rollout)[0]

        self.assertGreater(result.reward, 0.45)
        self.assertGreater(result.components["assigned_role_fit"], 0.2)
        self.assertGreater(result.components["validated_contribution_influence"], 0.0)

    def test_final_answer_overlap_does_not_help_when_outcome_is_bad(self) -> None:
        from council_runtime.specialist_rewards import SpecialistRewardFunction

        rollout = self._rollout(
            output="No risk, save schema is safe and tests pass.",
            confidence=0.9,
            final_outcome=0.2,
            final_text="No risk, save schema is safe and tests pass.",
        )
        result = SpecialistRewardFunction().score_rollout(rollout)[0]

        self.assertEqual(result.components["validated_contribution_influence"], 0.0)
        self.assertGreater(result.components["bad_influence_penalty"], 0.5)
        self.assertGreater(result.components["misleading_penalty"], 0.0)

    def test_low_confidence_out_of_domain_reroute_is_not_punished_as_misleading(self) -> None:
        from council_runtime.specialist_rewards import SpecialistRewardFunction

        rollout = self._rollout(
            output="This is outside my domain; reroute to save_load and ask for migration test evidence.",
            confidence=0.25,
            final_outcome=0.6,
            expert_id="hud_status",
            final_text="Reroute to save_load for migration test evidence.",
        )
        result = SpecialistRewardFunction().score_rollout(rollout)[0]

        self.assertGreater(result.components["confidence_calibration"], 0.5)
        self.assertLess(result.components["misleading_penalty"], 0.1)


if __name__ == "__main__":
    unittest.main()
