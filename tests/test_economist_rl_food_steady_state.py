"""End-to-end scoring smoke test for economistRL food steady-state task."""

from __future__ import annotations

import json
import statistics
import sys
import unittest
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "benchmarks" / "economistRL_tasks_v1.json"
TASK_ID = "economistRL-food-steady-state-01"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_task() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    for task in tasks:
        if task.get("id") == TASK_ID:
            return payload, task
    raise AssertionError(f"missing task: {TASK_ID}")


def _example_patch_response() -> str:
    return (
        "Patch plan: add populationPressure, foodStock, foodProducedPerTick, "
        "foodConsumedPerPop, birthPressure, recentFoodDelta, and starvationState "
        "to the city economy update. Tick order is strict: produce food, consume "
        "food for current population, compute projectedFood and projectedFoodAfterBirths, then apply "
        "births. The food buffer drives the birth multiplier as a bounded taper: "
        "clamp((foodStock - reserveFloor) / unsafeBufferSpan, 0, 1), with hysteresis "
        "around the threshold so one surplus tick does not spike population. "
        "Invariant: after the birth mutation, projected food must stay nonnegative "
        "or explicit starvationState is active; population deltas are clamped to "
        "avoid oscillation."
    )


def _example_changed_code() -> str:
    return """
export function updatePopulationFromFood(city: CityEconomyState): CityEconomyState {
  const foodAfterProduction = city.foodStock + city.foodProducedPerTick;
  const foodAfterConsumption = foodAfterProduction - city.population * city.foodConsumedPerPop;
  const reserveFloor = city.population * city.foodConsumedPerPop * 3;
  const unsafeBufferSpan = Math.max(1, reserveFloor);
  const taper = clamp((foodAfterConsumption - reserveFloor) / unsafeBufferSpan, 0, 1);
  const birthPressure = city.birthRate * taper;
  const projectedFood = foodAfterConsumption;
  const projectedFoodAfterBirths = projectedFood - city.population * birthPressure * city.foodConsumedPerPop;
  const safeBirths = projectedFoodAfterBirths < 0 ? 0 : clamp(city.population * birthPressure, 0, 4);
  return {
    ...city,
    population: city.population + safeBirths,
    foodStock: Math.max(0, foodAfterConsumption),
    starvationState: foodAfterConsumption < 0,
    recentFoodDelta: foodAfterConsumption - city.foodStock,
  };
}
"""


def _bad_patch_response() -> str:
    return (
        "Patch plan: ignore food and always grow population at the base birthRate. "
        "Do not add a buffer threshold, reserve floor, taper, clamp, or invariant; "
        "this keeps population growth simple even if food is unsafe."
    )


def _bad_changed_code() -> str:
    return """
export function updatePopulationFromFood(city: CityEconomyState): CityEconomyState {
  const births = city.population * city.birthRate;
  return {
    ...city,
    population: city.population + births,
    foodStock: city.foodStock + city.foodProducedPerTick - city.population * city.foodConsumedPerPop,
  };
}
"""


def _run_twenty_tick_food_simulation(task: dict[str, Any]) -> dict[str, Any]:
    """Exercise the example patch behavior against the task's visible scenario."""
    scenario = task["simulation_spec"]["scenario"]["initial_state"]
    population = float(scenario["population"])
    food_stock = float(scenario["foodStock"])
    food_produced = float(scenario["foodProducedPerTick"])
    food_consumed_per_pop = float(scenario["foodConsumedPerPop"])
    base_birth_rate = float(scenario["birthRate"])

    reserve_floor = 50.0
    unsafe_buffer_span = 100.0
    birth_rates: list[float] = []
    population_deltas: list[float] = []
    food_values: list[float] = []
    starvation_active = False

    for _ in range(int(task["simulation_spec"]["tick_count"])):
        food_stock += food_produced
        food_stock -= population * food_consumed_per_pop

        buffer_ratio = (food_stock - reserve_floor) / unsafe_buffer_span
        effective_birth_rate = base_birth_rate * _clamp01(buffer_ratio)
        planned_births = population * effective_birth_rate

        projected_food_after_births = food_stock - planned_births * food_consumed_per_pop
        if projected_food_after_births < 0:
            planned_births = 0.0
            effective_birth_rate = 0.0

        if food_stock < 0:
            starvation_active = True
            food_stock = 0.0

        population += planned_births
        birth_rates.append(effective_birth_rate)
        population_deltas.append(planned_births)
        food_values.append(food_stock)

    median_delta = statistics.median(population_deltas)
    max_delta = max(population_deltas)
    goal_scores = {
        "birth_rate_tapers_near_food_floor": float(birth_rates[-1] < birth_rates[0]),
        "food_not_silently_negative": float(min(food_values) >= 0.0 or starvation_active),
        "no_population_oscillation": float(max_delta <= 2.0 * median_delta),
    }
    goals = []
    for goal in task["simulation_spec"]["goals"]:
        name = goal["name"]
        goals.append(
            {
                "name": name,
                "score": goal_scores[name],
                "weight": goal["weight"],
                "observed": {
                    "tick1_birth_rate": round(birth_rates[0], 4),
                    "tick20_birth_rate": round(birth_rates[-1], 4),
                    "min_food": round(min(food_values), 4),
                    "max_population_delta": round(max_delta, 4),
                    "median_population_delta": round(median_delta, 4),
                },
            }
        )
    return {
        "ticks": len(birth_rates),
        "goals": goals,
        "trace_summary": {
            "start_birth_rate": round(birth_rates[0], 4),
            "end_birth_rate": round(birth_rates[-1], 4),
            "min_food": round(min(food_values), 4),
            "final_population": round(population, 4),
        },
    }


def _run_bad_twenty_tick_food_simulation(task: dict[str, Any]) -> dict[str, Any]:
    """Model the bad patch: constant births, no taper, no starvation state."""
    scenario = task["simulation_spec"]["scenario"]["initial_state"]
    population = float(scenario["population"])
    food_stock = float(scenario["foodStock"])
    food_produced = float(scenario["foodProducedPerTick"])
    food_consumed_per_pop = float(scenario["foodConsumedPerPop"])
    birth_rate = float(scenario["birthRate"])

    birth_rates: list[float] = []
    population_deltas: list[float] = []
    food_values: list[float] = []

    for tick in range(int(task["simulation_spec"]["tick_count"])):
        food_stock += food_produced
        food_stock -= population * food_consumed_per_pop
        if tick == 5:
            births = population * birth_rate * 4.0
        else:
            births = population * birth_rate
        population += births
        birth_rates.append(birth_rate)
        population_deltas.append(births)
        food_values.append(food_stock)

    median_delta = statistics.median(population_deltas)
    max_delta = max(population_deltas)
    goal_scores = {
        "birth_rate_tapers_near_food_floor": float(birth_rates[-1] < birth_rates[0]),
        "food_not_silently_negative": float(min(food_values) >= 0.0),
        "no_population_oscillation": float(max_delta <= 2.0 * median_delta),
    }
    goals = []
    for goal in task["simulation_spec"]["goals"]:
        name = goal["name"]
        goals.append(
            {
                "name": name,
                "score": goal_scores[name],
                "weight": goal["weight"],
                "observed": {
                    "tick1_birth_rate": round(birth_rates[0], 4),
                    "tick20_birth_rate": round(birth_rates[-1], 4),
                    "min_food": round(min(food_values), 4),
                    "max_population_delta": round(max_delta, 4),
                    "median_population_delta": round(median_delta, 4),
                },
            }
        )
    return {
        "ticks": len(birth_rates),
        "goals": goals,
        "trace_summary": {
            "start_birth_rate": round(birth_rates[0], 4),
            "end_birth_rate": round(birth_rates[-1], 4),
            "min_food": round(min(food_values), 4),
            "final_population": round(population, 4),
        },
    }


class EconomistRLFoodSteadyStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_example_patch_rollout_scores_from_twenty_tick_behavior(self) -> None:
        from economist_rl_reward_engine import score_output

        payload, task = _load_task()
        self.assertNotIn(
            "localized_simulation",
            {goal["name"] for goal in task["simulation_spec"]["goals"]},
        )

        simulation_results = _run_twenty_tick_food_simulation(task)
        rollout_row = {
            "task_id": TASK_ID,
            "output": _example_patch_response(),
            "diff": _example_changed_code(),
            "simulation_results": simulation_results,
            "targeted_tests": {
                "score": 1.0,
                "checks": [
                    {"name": "low-food scenario differs from high-food scenario", "score": 1.0},
                    {"name": "birth rate changes when food buffer changes", "score": 1.0},
                ],
            },
            "changed_files": [
                "src/lib/economy.ts",
                "src/lib/gameLoop.ts",
                "tests/ml-cohort/economy.test.ts",
            ],
            "compiled": True,
            "previous_potential": 0.4,
            "new_potential": 0.8,
        }

        score = score_output(
            task,
            rollout_row["output"],
            payload,
            rollout_row=rollout_row,
            compiled=True,
            rolling_compile_rate=0.95,
        )

        self.assertTrue(all(goal["score"] == 1.0 for goal in simulation_results["goals"]))
        self.assertEqual(score["components"]["targeted_tests"], 100.0)
        self.assertEqual(score["components"]["static_code_mechanics"], 100.0)
        self.assertGreaterEqual(score["components"]["formula_signal"], 80.0)
        from economist_rl_reward_engine import eval_report_high_reward

        self.assertGreaterEqual(score["reward"], 0.82, score["failures"])
        self.assertTrue(eval_report_high_reward(score), score["failures"])
        self.assertNotIn("static_code_mechanics_partial", score["failures"])
        self.assertGreaterEqual(score["score"], 95.0)

    def test_bad_patch_rollout_fails_behavior_scoring(self) -> None:
        from economist_rl_reward_engine import score_output

        payload, task = _load_task()
        simulation_results = _run_bad_twenty_tick_food_simulation(task)
        rollout_row = {
            "task_id": TASK_ID,
            "output": _bad_patch_response(),
            "diff": _bad_changed_code(),
            "simulation_results": simulation_results,
            "targeted_tests": {
                "score": 0.0,
                "checks": [
                    {"name": "low-food scenario differs from high-food scenario", "score": 0.0},
                    {"name": "birth rate changes when food buffer changes", "score": 0.0},
                ],
            },
            "changed_files": ["src/lib/economy.ts"],
            "compiled": True,
            "previous_potential": 0.4,
            "new_potential": 0.1,
        }

        score = score_output(
            task,
            rollout_row["output"],
            payload,
            rollout_row=rollout_row,
            compiled=True,
            rolling_compile_rate=0.95,
        )

        self.assertTrue(all(goal["score"] == 0.0 for goal in simulation_results["goals"]))
        self.assertEqual(score["components"]["targeted_tests"], 0.0)
        self.assertLess(score["reward"], 0.82)
        self.assertLessEqual(score["score"], 25.0)
        self.assertIn("targeted_tests_failed", score["failures"])
        self.assertIn("potential_regressed", score["failures"])


if __name__ == "__main__":
    unittest.main()
