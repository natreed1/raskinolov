#!/usr/bin/env python3
"""Tests for simulation_spec.goals → Vitest it() mapping."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_vitest_goals import (  # noqa: E402
    goal_it_title,
    render_vitest_stub,
    simulation_goals,
)
from economist_rl_vitest_scoring import merge_outcome_checks_with_vitest  # noqa: E402


class EconomistRLVitestGoalsTests(unittest.TestCase):
    def test_food_goals_emit_goal_it_blocks(self) -> None:
        task = {
            "id": "economistRL-food-steady-state-01",
            "subsection": "food_population_feedback",
            "simulation_spec": {
                "tick_count": 20,
                "goals": [
                    {"name": "birth_rate_tapers_near_food_floor", "weight": 0.4},
                    {"name": "food_not_silently_negative", "weight": 0.35},
                    {"name": "no_population_oscillation", "weight": 0.25},
                ],
            },
        }
        body = render_vitest_stub(
            task,
            lib_file="src/lib/economistRl/economistrl_food_steady_state_01/mechanic.ts",
            test_file="tests/economistRl/economistrl_food_steady_state_01.test.ts",
        )
        self.assertIn("Goals from simulation_spec: 3 loaded into vitest", body)
        for goal in simulation_goals(task):
            self.assertIn(f"it('{goal_it_title(goal['name'])}'", body)
        self.assertIn("cities[0].storage.food", body)

    def test_merge_goal_weighted_vitest_partial_credit(self) -> None:
        task = {
            "simulation_spec": {
                "goals": [
                    {"name": "scarcity_raises_price", "weight": 0.6},
                    {"name": "surplus_lowers_price", "weight": 0.4},
                ],
            },
        }
        vitest = {
            "score": 0.5,
            "vitest_ran": True,
            "per_it": [
                {"title": goal_it_title("scarcity_raises_price"), "passed": True, "score": 1.0},
                {"title": goal_it_title("surplus_lowers_price"), "passed": False, "score": 0.0},
            ],
        }
        merged = merge_outcome_checks_with_vitest(task, vitest)
        self.assertEqual(merged["source"], "vitest_goal_weighted")
        self.assertAlmostEqual(merged["score"], 0.6, places=3)

    def test_merge_matches_truncated_vitest_it_title(self) -> None:
        from economist_rl_vitest_scoring import merge_outcome_checks_with_vitest

        task = {
            "simulation_spec": {
                "goals": [
                    {"name": "birth_rate_differs_between_safe_and_unsafe_food_buffers", "weight": 1.0},
                ],
            },
        }
        vitest = {
            "vitest_ran": True,
            "per_it": [
                {
                    "title": "birth_rate_differs_between_safe_and_unsafe_food_",
                    "lookup_key": "goal_birth_rate_differs_between_safe_and_unsafe_food_",
                    "passed": True,
                    "score": 1.0,
                },
            ],
        }
        merged = merge_outcome_checks_with_vitest(task, vitest)
        self.assertEqual(merged["source"], "vitest_goal_weighted")
        self.assertAlmostEqual(merged["score"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
