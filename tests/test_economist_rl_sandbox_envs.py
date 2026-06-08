#!/usr/bin/env python3
"""Tests for economistRL per-subsection vitest sandbox environments."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FE_REPO = Path("/Users/natreed/fallen-empire")
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_coding_contract import convert_task_to_coding, sandbox_coding_user_prompt  # noqa: E402
from economist_rl_sandbox_envs import (  # noqa: E402
    mechanic_stub_body,
    resolve_subsection,
    sandbox_starter_bodies,
    test_stub_body,
)
from economist_rl_task_execution import materialize_starter_files, sandbox_paths  # noqa: E402
from game_task_arena import apply_fenced_files  # noqa: E402

V3_BANK = REPO / "benchmarks/economistRL_tasks_v3_execution.json"

# Cycle 015 false-positive: game-shaped loops on numeric sandbox state (no state change).
CYCLE_015_CACHE_NOOP_OUTPUT = """```ts path=src/lib/economistRl/economistrl_production_cache_invalid_06/mechanic.ts
export function applyResourceProjectionCacheIntegrityTick(state: EconomistRlState): EconomistRlState {
  for (const cid in state.workers) {
    const w = state.workers[cid];
    if (w.playerId !== state.activePlayerId) continue;
    w.resourceCacheValid = false;
  }

  for (const cid in state.buildings) {
    const b = state.buildings[cid];
    if (b.owner !== state.activePlayerId) continue;
    if (b.type !== 'market') continue;
    b.cacheValid = false;
  }

  for (const cid in state.cities) {
    const c = state.cities[cid];
    if (c.owner !== state.activePlayerId) continue;
    c.productionCacheValid = false;
  }

  for (const cid in state.units) {
    const u = state.units[cid];
    if (u.owner !== state.activePlayerId) continue;
    u.morale = Math.min(99, u.morale);
  }

  return state;
}
```"""

MARKET_TASK = {
    "id": "economistRL-market-elasticity-02",
    "title": "Market Price Elasticity From Stock Pressure",
    "subsection": "market_elasticity_pricing",
    "curriculum_track": "economy",
    "prompt": "Specify market-pricing where surplus lowers price and scarcity raises it with clamps.",
    "simulation_spec": {
        "tick_count": 20,
        "relevant_state": [
            "storageGoods",
            "storageCapGoods",
            "marketPriceGold",
            "demand",
            "scarcityPressure",
            "surplusPressure",
        ],
        "scenario": {
            "initial_state": {
                "storageGoods": 50,
                "storageCapGoods": 100,
                "marketPriceGold": 10,
                "demand": 1.0,
            }
        },
        "goals": [
            {"name": "scarcity_raises_price", "weight": 0.3},
            {"name": "surplus_lowers_price", "weight": 0.25},
            {"name": "price_bounded", "weight": 0.3},
            {"name": "smooth_adjustment", "weight": 0.15},
        ],
    },
    "targeted_tests": {
        "outcome_checks": [
            "scarcity and surplus cases diverge",
            "price floor/cap is respected",
        ]
    },
}


def _vitest_on_task(task: dict[str, Any], *, extra_output: str | None = None) -> int:
    if not (FE_REPO / "package.json").is_file():
        raise unittest.SkipTest("Fallen Empire repo not available for vitest")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wt"
        shutil.copytree(
            FE_REPO,
            root,
            ignore=shutil.ignore_patterns("node_modules", ".next", ".git"),
        )
        materialize_starter_files(root, task)
        if extra_output:
            allowed = task["execution"]["allowed_paths"]
            apply_fenced_files(root, extra_output, allowed, root / "apply.log")
        _lib, test_path = sandbox_paths(task)
        proc = subprocess.run(
            ["sh", "-lc", f"npx vitest run {test_path}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"vitest failed ({proc.returncode})\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )
        return int(proc.returncode)


def _load_v3_task(task_id: str) -> dict[str, Any]:
    bank = json.loads(V3_BANK.read_text(encoding="utf-8"))
    for raw in bank.get("tasks") or []:
        if isinstance(raw, dict) and raw.get("id") == task_id:
            return raw
    raise unittest.SkipTest(f"task not in v3 bank: {task_id}")


class EconomistRLSandboxEnvsTests(unittest.TestCase):
    def test_run_ticks_clones_state_each_tick(self) -> None:
        body = test_stub_body(MARKET_TASK, lib_file="src/lib/economistRl/x/mechanic.ts", test_file="tests/x.test.ts")
        env_ticks = resolve_subsection(MARKET_TASK)
        from economist_rl_sandbox_envs import _env_run_ticks_ts

        run_ticks = _env_run_ticks_ts(env_ticks)
        self.assertTrue(
            "state = fn({ ...state })" in run_ticks or "cloneGameTurnState(state)" in run_ticks,
            run_ticks[:200],
        )
        self.assertIn("let state = { ...initial }", run_ticks)

    def test_market_test_asserts_delta_not_static_seed_compare(self) -> None:
        from economist_rl_vitest_goals import goal_it_title

        lib, test = sandbox_paths(MARKET_TASK)
        body = test_stub_body(MARKET_TASK, lib_file=lib, test_file=test)
        self.assertIn(".not.toBe(", body)
        self.assertIn("taskExt.marketPriceGold", body)
        self.assertIn(goal_it_title("scarcity_raises_price"), body)
        self.assertNotIn("toBeGreaterThanOrEqual(dirty[3]", body)

    def test_cache_test_asserts_projection_cache_delta(self) -> None:
        task = _load_v3_task("economistRL-production-cache-invalid-06")
        lib, test = sandbox_paths(task)
        body = test_stub_body(task, lib_file=lib, test_file=test)
        self.assertIn("goal_worker_change_invalidates_cache", body)
        self.assertIn("dirty[8].taskExt.resourceProjectionValid).not.toBe(dirty[0].taskExt.resourceProjectionValid)", body)

    def test_market_mechanic_uses_price_not_food(self) -> None:
        lib, _test = sandbox_paths(MARKET_TASK)
        body = mechanic_stub_body(MARKET_TASK, lib)
        self.assertIn("taskExt.marketPriceGold", body)
        self.assertIn("city.storageCap.goods", body)
        self.assertNotIn("taskExt.storageFood", body)

    def test_market_test_uses_scarcity_surplus_seeds(self) -> None:
        lib, test = sandbox_paths(MARKET_TASK)
        body = test_stub_body(MARKET_TASK, lib_file=lib, test_file=test)
        self.assertIn("SCARCITY_SEED", body)
        self.assertIn("SURPLUS_SEED", body)
        self.assertNotIn("storageFood: 200", body)

    def test_materialize_includes_env_package(self) -> None:
        converted = convert_task_to_coding(dict(MARKET_TASK))
        with tempfile.TemporaryDirectory() as tmp:
            written = materialize_starter_files(Path(tmp), converted)
            self.assertGreaterEqual(len(written), 8)
            self.assertTrue(any("envs/shared/fixture.ts" in p for p in written))
            self.assertTrue(any("envs/market_elasticity_pricing/types.ts" in p for p in written))

    def test_sandbox_prompt_no_fictional_production_path(self) -> None:
        converted = convert_task_to_coding(dict(MARKET_TASK))
        prompt = sandbox_coding_user_prompt(converted)
        self.assertNotIn("src/lib/economy.ts", prompt)
        self.assertIn("path=", prompt)
        lib, test = sandbox_paths(MARKET_TASK)
        self.assertIn(lib, prompt)
        self.assertIn(test, prompt)

    def test_starter_bodies_count(self) -> None:
        lib, test = sandbox_paths(MARKET_TASK)
        bodies = sandbox_starter_bodies(MARKET_TASK, lib_file=lib, test_file=test)
        paths = [b["path"] for b in bodies]
        self.assertGreaterEqual(len(paths), 8)
        self.assertTrue(any("envs/shared/fixture.ts" in p for p in paths))
        self.assertEqual(resolve_subsection(MARKET_TASK), "market_elasticity_pricing")

    def test_reference_market_stub_passes_vitest(self) -> None:
        converted = convert_task_to_coding(dict(MARKET_TASK))
        _vitest_on_task(converted)

    def test_reference_cache_stub_passes_vitest(self) -> None:
        task = _load_v3_task("economistRL-production-cache-invalid-06")
        _vitest_on_task(task)

    def test_cycle_015_cache_noop_patch_fails_vitest(self) -> None:
        task = _load_v3_task("economistRL-production-cache-invalid-06")
        if not (FE_REPO / "package.json").is_file():
            self.skipTest("Fallen Empire repo not available for vitest")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "wt"
            shutil.copytree(
                FE_REPO,
                root,
                ignore=shutil.ignore_patterns("node_modules", ".next", ".git"),
            )
            materialize_starter_files(root, task)
            allowed = task["execution"]["allowed_paths"]
            apply_fenced_files(root, CYCLE_015_CACHE_NOOP_OUTPUT, allowed, root / "apply.log")
            _lib, test_path = sandbox_paths(task)
            proc = subprocess.run(
                ["sh", "-lc", f"npx vitest run {test_path}"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(proc.returncode, 0, "cycle-015 no-op patch must fail stricter vitest")


if __name__ == "__main__":
    unittest.main()
