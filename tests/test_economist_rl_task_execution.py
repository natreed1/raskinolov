#!/usr/bin/env python3
"""Tests for economistRL execution tagging, verify commands, and prompt assembly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_task_execution import (  # noqa: E402
    EXECUTION_MODE_ARENA,
    EXECUTION_MODE_INTEGRATION,
    INTEGRATION_KIND_SANDBOX,
    enrich_task_execution,
    materialize_starter_files,
    resolve_eval_tasks,
    sandbox_paths,
    verify_commands_for_task,
)


class EconomistRLTaskExecutionTests(unittest.TestCase):
    def test_sandbox_paths_stable(self) -> None:
        task = {"id": "economistRL-food-steady-state-01"}
        lib, test = sandbox_paths(task)
        self.assertEqual(lib, "src/lib/economistRl/economistrl_food_steady_state_01/mechanic.ts")
        self.assertEqual(test, "tests/economistRl/economistrl_food_steady_state_01.test.ts")

    def test_sandbox_verify_command(self) -> None:
        task = enrich_task_execution(
            {
                "id": "economistRL-food-steady-state-01",
                "curriculum_track": "economy",
                "codebase_requirements": {"relevant_files": ["src/lib/economy.ts"]},
            }
        )
        cmds = verify_commands_for_task(task)
        self.assertEqual(len(cmds), 1)
        self.assertTrue(cmds[0].startswith("node_modules/.bin/vitest run "))
        self.assertEqual(task["execution_mode"], EXECUTION_MODE_INTEGRATION)
        self.assertEqual(task["integration_kind"], INTEGRATION_KIND_SANDBOX)
        self.assertTrue(task["execution"]["starter_files"])

    def test_materialize_starters(self) -> None:
        import tempfile

        task = enrich_task_execution(
            {
                "id": "economistRL-food-steady-state-01",
                "curriculum_track": "economy",
                "subsection": "food_population_feedback",
                "subskill": "food_supported_population_steady_state_dynamics",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = materialize_starter_files(root, task)
            self.assertGreaterEqual(len(written), 5)
            for rel in written:
                self.assertTrue((root / rel).is_file())
            self.assertTrue(any("envs/" in p for p in written))

    def test_eval_manifest_includes_arena_tier(self) -> None:
        manifest = {
            "tiers": {
                "arena": ["economy-tooltip"],
                "sandbox": ["economistRL-food-steady-state-01"],
                "generalist": [],
            }
        }
        bank = [
            enrich_task_execution(
                {
                    "id": "economistRL-food-steady-state-01",
                    "curriculum_track": "economy",
                    "prompt": "fix food",
                }
            )
        ]
        selected = resolve_eval_tasks(bank, manifest, limit=10)
        ids = [t["id"] for t in selected]
        self.assertTrue(any(i.startswith("arena-eval-") for i in ids))
        self.assertIn("economistRL-food-steady-state-01", ids)

    def test_arena_mode_verify_uses_cohort(self) -> None:
        task = {
            "id": "arena-eval-economy-tooltip",
            "execution_mode": EXECUTION_MODE_ARENA,
            "verify_commands": ["npm run test:ml-cohort"],
        }
        self.assertEqual(verify_commands_for_task(task), ["npm run test:ml-cohort"])


if __name__ == "__main__":
    unittest.main()
