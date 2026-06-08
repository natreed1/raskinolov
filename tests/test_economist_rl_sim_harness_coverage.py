"""Verify toy sim environments cover the full economistRL task bank."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class EconomistRLSimHarnessCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from economist_rl_reward_engine import _load_json
        from economist_rl_sim_harnesses import SUBSECTION_SIMULATORS

        cls.payload = _load_json(REPO / "benchmarks" / "economistRL_tasks_v1.json")
        cls.tasks = cls.payload["tasks"]
        cls.simulators = SUBSECTION_SIMULATORS

    def test_all_subsections_have_simulators(self) -> None:
        subs = {str(task.get("subsection") or "") for task in self.tasks}
        missing = sorted(sub for sub in subs if sub and sub not in self.simulators)
        self.assertEqual(missing, [], f"missing simulators for: {missing}")

    def test_all_tasks_attach_evidence_without_error(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence

        errors: list[str] = []
        for task in self.tasks:
            ref = str(task.get("reference_answer") or task.get("prompt") or "bounded clamp smooth threshold")
            row = attach_rollout_evidence(
                task=task,
                rollout_row={"task_id": task["id"], "output": ref},
                dry_run=False,
            )
            if not row.get("targeted_tests"):
                errors.append(f"{task['id']}: missing targeted_tests")
        self.assertEqual(errors, [], f"first errors: {errors[:5]}")

    def test_v3_tasks_emit_goal_vitest_blocks(self) -> None:
        from economist_rl_reward_engine import _load_json
        from economist_rl_sandbox_envs import test_stub_body
        from economist_rl_task_execution import sandbox_paths
        from economist_rl_vitest_goals import goal_it_title, simulation_goals

        payload = _load_json(REPO / "benchmarks" / "economistRL_tasks_v3_execution.json")
        missing = 0
        for task in payload.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            goals = simulation_goals(task)
            if not goals:
                missing += 1
                continue
            lib, test = sandbox_paths(task)
            body = test_stub_body(task, lib_file=lib, test_file=test)
            for goal in goals:
                if goal_it_title(goal["name"]) not in body:
                    missing += 1
                    break
        self.assertEqual(missing, 0, f"{missing} tasks missing goal_* vitest blocks")


if __name__ == "__main__":
    unittest.main()
