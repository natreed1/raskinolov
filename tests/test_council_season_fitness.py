"""Tests for feeding graded rollouts into population fitness/state."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _planner_document() -> dict:
    return {
        "schema_version": "planner_rollout_plan_v1",
        "task_synopsis": "Implement a persistence-safe economy fix and verify targeted tests.",
        "primary_goal": "implementation: satisfy the requested change with verified low-risk behavior.",
        "success_criteria": ["Files identified", "Tests run", "Risk explained"],
        "known_constraints": ["Avoid unrelated changes"],
        "risk_tags": ["persistence", "economy_balance", "test_coverage"],
        "required_context": ["economy file", "save/load tests", "recent trace"],
        "selected_experts": [
            {
                "expert_id": "economistRL",
                "role": "economy specialist",
                "why_selected": "Economy behavior is central.",
                "assigned_question": "Find mechanic risks and tests.",
                "expected_contribution": "Risk review with evidence.",
                "handoff_context": "Food, morale, market, and save/load interaction.",
            }
        ],
        "excluded_experts": [],
        "debate_plan": {
            "max_rounds": 1,
            "reroute_conditions": ["low confidence"],
            "stop_conditions": ["tests covered"],
        },
    }


class CouncilSeasonFitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def _write_trace_file(self, path: Path) -> None:
        from council_runtime.traces import JsonlTraceWriter, build_grader_trace, build_planner_trace, build_specialist_trace

        writer = JsonlTraceWriter(path)
        context = {"task_id": "task_season", "season_id": "season_test", "final_outcome": 0.8}
        plan = {
            "participants": [
                {"participant_id": "economistRL::risk", "base_expert_id": "economistRL", "participant_type": "specialist_adapter"}
            ],
            "debate_max_rounds": 1,
        }
        writer(
            build_planner_trace(
                prompt="Implement economy persistence fix.",
                council_plan=plan,
                planner_document=_planner_document(),
                trace_context=context,
            )
        )
        writer(
            build_specialist_trace(
                prompt="Implement economy persistence fix.",
                participant_output={
                    "participant_id": "economistRL::risk",
                    "base_expert_id": "economistRL",
                    "participant_type": "specialist_adapter",
                    "text": "Verify economy file, risk, and tests.",
                    "confidence": 0.72,
                    "task_outcome_score": 0.8,
                    "contribution_credit": 0.85,
                    "evidence_quality": 0.8,
                },
                round_idx=1,
                trace_context=context,
            )
        )
        writer(
            build_grader_trace(
                prompt="Implement economy persistence fix.",
                adjudication={
                    "winner_ids": ["economistRL::risk"],
                    "confidence": 0.8,
                    "disagreement": 0.1,
                    "escalation_recommended": False,
                    "final_text": "Use the economy risk review.",
                },
                round_idx=1,
                participant_outputs=[],
                trace_context=context,
                final=True,
            )
        )

    def test_season_operator_updates_active_planner_fitness_score_only(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry
        from council_runtime.coevolution import bootstrap_state
        from council_runtime.fitness import POPULATION_FITNESS_SCHEMA
        from council_runtime.season_ops import CouncilSeasonOperator, SeasonContext

        registry = CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"economistRL"})
        state = bootstrap_state(registry)
        before = state.archetypes_by_id()["planner"].active()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / "traces.jsonl"
            self._write_trace_file(traces)
            result = CouncilSeasonOperator().run(
                context=SeasonContext(
                    season_id="season_test",
                    traces_jsonl=traces,
                    run_dir=root / "season",
                    state_path=root / "state.json",
                    mode="score_only",
                ),
                state=state,
            )
            updated_payload = json.loads(Path(result.updated_state_path).read_text(encoding="utf-8"))
            fitness_payload = json.loads(Path(result.fitness_path).read_text(encoding="utf-8"))

            self.assertEqual(fitness_payload["schema_version"], POPULATION_FITNESS_SCHEMA)
            self.assertIn(before.organism_id, fitness_payload["by_organism"])
            updated_state = state.from_dict(updated_payload)
            after = updated_state.archetypes_by_id()["planner"].active()
            self.assertEqual(after.organism_id, before.organism_id)
            self.assertEqual(after.adapter_path, before.adapter_path)
            self.assertGreater(after.fitness["rollout_reward"], 0.0)
            self.assertEqual(after.state, before.state)

    def test_shadow_select_marks_scored_planner_elite_without_changing_active_id(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry
        from council_runtime.coevolution import bootstrap_state
        from council_runtime.season_ops import CouncilSeasonOperator, SeasonContext

        state = bootstrap_state(CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"economistRL"}))
        active_id = state.archetypes_by_id()["planner"].active_organism_id

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / "traces.jsonl"
            self._write_trace_file(traces)
            result = CouncilSeasonOperator().run(
                context=SeasonContext(
                    season_id="season_test",
                    traces_jsonl=traces,
                    run_dir=root / "season",
                    state_path=root / "state.json",
                    mode="shadow_select",
                ),
                state=state,
            )
            updated_state = state.from_dict(json.loads(Path(result.updated_state_path).read_text(encoding="utf-8")))
            planner_population = updated_state.archetypes_by_id()["planner"]

            self.assertEqual(planner_population.active_organism_id, active_id)
            self.assertEqual(planner_population.active().state, "elite")


if __name__ == "__main__":
    unittest.main()
