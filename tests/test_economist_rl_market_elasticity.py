"""End-to-end scoring smoke test for economistRL market elasticity task."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "benchmarks" / "economistRL_tasks_v1.json"
TASK_ID = "economistRL-market-elasticity-02"

PRICE_FLOOR = 5.0
PRICE_CAP = 25.0
MULT_FLOOR = 0.75
MULT_CAP = 1.35
MAX_STEP = 1.25
SMOOTH = 0.22


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_task() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    for task in tasks:
        if task.get("id") == TASK_ID:
            return payload, task
    raise AssertionError(f"missing task: {TASK_ID}")


def _example_patch_response() -> str:
    return (
        "Market tick reads currentStock and desiredStock, caches lastPrice and demand per tick. "
        "Stock pressure is desiredStock / currentStock; scarcity raises pressure above 1 and "
        "surplus lowers it below 1. Map pressure through a bounded elasticity curve: "
        "multiplier = clamp(1 + 0.18 * (pressure - 1), floor, cap). Target price is "
        "lastPrice * multiplier * demand, clamped to price floor/cap, then smoothed toward "
        "target each tick so one scarcity spike cannot jump price unbounded."
    )


def _example_changed_code() -> str:
    return """
export function updateMarketPrice(market: MarketState): MarketState {
  const pressure = market.desiredStock / Math.max(1, market.currentStock);
  const multiplier = clamp(1 + 0.18 * (pressure - 1), 0.75, 1.35);
  const target = clamp(market.lastPrice * multiplier * market.demand, 5, 25);
  const price = clamp(market.lastPrice + 0.22 * (target - market.lastPrice), 5, 25);
  return {
    ...market,
    lastPrice: price,
    scarcityPressure: Math.max(0, pressure - 1),
    surplusPressure: Math.max(0, 1 - pressure),
  };
}
"""


def _bad_patch_response() -> str:
    return (
        "Use a random price each market tick with no stock read and no clamp. "
        "Price is unbounded so scarcity and surplus do not matter."
    )


def _bad_changed_code() -> str:
    return """
export function updateMarketPrice(market: MarketState): MarketState {
  const price = market.lastPrice + Math.random() * 10;
  return { ...market, lastPrice: price };
}
"""


def _run_price_ticks(
    *,
    current_stock: float,
    desired_stock: float,
    last_price: float,
    demand: float,
    tick_count: int,
    use_bounded_elasticity: bool,
) -> list[float]:
    prices = [last_price]
    price = last_price
    for _ in range(tick_count - 1):
        if use_bounded_elasticity:
            pressure = desired_stock / max(current_stock, 1.0)
            multiplier = _clamp(1.0 + 0.18 * (pressure - 1.0), MULT_FLOOR, MULT_CAP)
            target = _clamp(price * multiplier * demand, PRICE_FLOOR, PRICE_CAP)
            next_price = price + SMOOTH * (target - price)
            price = _clamp(next_price, PRICE_FLOOR, PRICE_CAP)
        else:
            price = price + 3.5
        prices.append(price)
    return prices


def _run_twenty_tick_market_simulation(task: dict[str, Any]) -> dict[str, Any]:
    tick_count = int(task["simulation_spec"]["tick_count"])
    scenario = task["simulation_spec"]["scenario"]["initial_state"]
    desired_stock = float(scenario["desiredStock"])
    last_price = float(scenario["lastPrice"])
    demand = float(scenario["demand"])

    scarcity_stock = float(scenario["currentStock"])
    surplus_stock = desired_stock * 1.5

    scarcity_prices = _run_price_ticks(
        current_stock=scarcity_stock,
        desired_stock=desired_stock,
        last_price=last_price,
        demand=demand,
        tick_count=tick_count,
        use_bounded_elasticity=True,
    )
    surplus_prices = _run_price_ticks(
        current_stock=surplus_stock,
        desired_stock=desired_stock,
        last_price=last_price,
        demand=demand,
        tick_count=tick_count,
        use_bounded_elasticity=True,
    )

    scarcity_deltas = [
        abs(scarcity_prices[i] - scarcity_prices[i - 1]) for i in range(1, len(scarcity_prices))
    ]
    all_prices = scarcity_prices + surplus_prices

    goal_scores = {
        "scarcity_raises_price": float(scarcity_prices[-1] > scarcity_prices[0]),
        "surplus_lowers_price": float(surplus_prices[-1] < scarcity_prices[-1]),
        "price_bounded": float(all(p >= PRICE_FLOOR and p <= PRICE_CAP for p in all_prices)),
        "smooth_adjustment": float(max(scarcity_deltas) <= MAX_STEP),
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
                    "scarcity_tick1_price": round(scarcity_prices[0], 4),
                    "scarcity_tick20_price": round(scarcity_prices[-1], 4),
                    "surplus_tick20_price": round(surplus_prices[-1], 4),
                    "max_scarcity_delta": round(max(scarcity_deltas), 4),
                    "price_floor": PRICE_FLOOR,
                    "price_cap": PRICE_CAP,
                },
            }
        )
    return {
        "ticks": tick_count,
        "goals": goals,
        "trace_summary": {
            "scarcity_start_price": round(scarcity_prices[0], 4),
            "scarcity_end_price": round(scarcity_prices[-1], 4),
            "surplus_end_price": round(surplus_prices[-1], 4),
        },
    }


def _run_bad_twenty_tick_market_simulation(task: dict[str, Any]) -> dict[str, Any]:
    tick_count = int(task["simulation_spec"]["tick_count"])
    scenario = task["simulation_spec"]["scenario"]["initial_state"]
    desired_stock = float(scenario["desiredStock"])
    last_price = float(scenario["lastPrice"])
    demand = float(scenario["demand"])
    scarcity_stock = float(scenario["currentStock"])
    surplus_stock = desired_stock * 1.5

    scarcity_prices = _run_price_ticks(
        current_stock=scarcity_stock,
        desired_stock=desired_stock,
        last_price=last_price,
        demand=demand,
        tick_count=tick_count,
        use_bounded_elasticity=False,
    )
    surplus_prices = _run_price_ticks(
        current_stock=surplus_stock,
        desired_stock=desired_stock,
        last_price=last_price,
        demand=demand,
        tick_count=tick_count,
        use_bounded_elasticity=False,
    )
    scarcity_deltas = [
        abs(scarcity_prices[i] - scarcity_prices[i - 1]) for i in range(1, len(scarcity_prices))
    ]
    all_prices = scarcity_prices + surplus_prices

    goal_scores = {
        "scarcity_raises_price": float(scarcity_prices[-1] > scarcity_prices[0]),
        "surplus_lowers_price": float(surplus_prices[-1] < scarcity_prices[-1]),
        "price_bounded": float(all(p >= PRICE_FLOOR and p <= PRICE_CAP for p in all_prices)),
        "smooth_adjustment": float(max(scarcity_deltas) <= MAX_STEP),
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
                    "scarcity_tick1_price": round(scarcity_prices[0], 4),
                    "scarcity_tick20_price": round(scarcity_prices[-1], 4),
                    "surplus_tick20_price": round(surplus_prices[-1], 4),
                    "max_scarcity_delta": round(max(scarcity_deltas), 4),
                },
            }
        )
    return {"ticks": tick_count, "goals": goals}


class EconomistRLMarketElasticityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_example_patch_rollout_scores_from_twenty_tick_behavior(self) -> None:
        from economist_rl_reward_engine import score_output

        payload, task = _load_task()
        simulation_results = _run_twenty_tick_market_simulation(task)
        rollout_row = {
            "task_id": TASK_ID,
            "output": _example_patch_response(),
            "diff": _example_changed_code(),
            "simulation_results": simulation_results,
            "targeted_tests": {
                "score": 1.0,
                "checks": [
                    {"name": "scarcity and surplus cases diverge", "score": 1.0},
                    {"name": "price floor/cap is respected", "score": 1.0},
                ],
            },
            "changed_files": [
                "src/lib/market.ts",
                "src/lib/economy.ts",
                "tests/ml-cohort/market.test.ts",
            ],
            "compiled": True,
            "previous_potential": 0.4,
            "new_potential": 0.82,
        }

        score = score_output(
            task,
            rollout_row["output"],
            payload,
            rollout_row=rollout_row,
            compiled=True,
            rolling_compile_rate=0.95,
        )

        self.assertTrue(all(goal["score"] == 1.0 for goal in simulation_results["goals"]), simulation_results)
        self.assertEqual(score["components"]["targeted_tests"], 100.0)
        self.assertGreaterEqual(score["components"]["formula_signal"], 80.0)
        from economist_rl_reward_engine import eval_report_high_reward

        self.assertGreaterEqual(score["reward"], 0.82, score["failures"])
        self.assertTrue(eval_report_high_reward(score), score["failures"])
        self.assertGreaterEqual(score["score"], 95.0)

    def test_bad_patch_rollout_fails_behavior_scoring(self) -> None:
        from economist_rl_reward_engine import score_output

        payload, task = _load_task()
        simulation_results = _run_bad_twenty_tick_market_simulation(task)
        rollout_row = {
            "task_id": TASK_ID,
            "output": _bad_patch_response(),
            "diff": _bad_changed_code(),
            "simulation_results": simulation_results,
            "targeted_tests": {"score": 0.0},
            "changed_files": ["src/lib/market.ts"],
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

        self.assertFalse(all(goal["score"] == 1.0 for goal in simulation_results["goals"]))
        self.assertEqual(score["components"]["targeted_tests"], 0.0)
        self.assertLess(score["reward"], 0.82)
        self.assertIn("potential_regressed", score["failures"])


if __name__ == "__main__":
    unittest.main()
