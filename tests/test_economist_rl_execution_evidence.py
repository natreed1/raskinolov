#!/usr/bin/env python3
"""Tests for economist RL execution evidence (apply + compile gate inputs)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class EconomistRLExecutionEvidenceTests(unittest.TestCase):
    def _init_git_repo(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        cwd = str(root)
        subprocess.run(["git", "init"], cwd=cwd, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=cwd, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=cwd, check=True, capture_output=True)
        (root / "src" / "lib").mkdir(parents=True)
        (root / "src" / "lib" / "economy.ts").write_text("export const foodStock = 1;\n", encoding="utf-8")
        (root / "src" / "lib" / "empireEconomy.ts").write_text("export const foodStock = 1;\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=cwd, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=cwd, check=True, capture_output=True)

    def test_apply_fenced_file_sets_compiled_true(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence
        from economist_rl_execution_evidence import (
            ExecutionEvidenceConfig,
            ExecutionWorktreePool,
            attach_execution_evidence,
        )
        from economist_rl_reward_engine import score_output

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "game"
            self._init_git_repo(source)
            config = ExecutionEvidenceConfig(
                enabled=True,
                source_repo=source,
                worktree_root=Path(tmp) / "worktrees",
                compile_commands=("true",),
                timeout_s=30,
            )
            pool = ExecutionWorktreePool(config, cycle_id=99)
            try:
                from economist_rl_task_execution import enrich_task_execution

                task = enrich_task_execution(
                    {
                        "id": "economistRL-exec-test-01",
                        "curriculum_track": "economy",
                        "codebase_requirements": {"relevant_files": ["src/lib/empireEconomy.ts"]},
                        "targeted_tests": {"outcome_checks": ["food stock updates"]},
                        "subsection": "food_population_feedback",
                        "simulation_spec": {"tick_count": 20, "goals": []},
                    },
                    source_repo=source,
                )
                task["execution"]["allowed_paths"] = ["src/lib/empireEconomy.ts", "src/**/*.ts"]
                task["execution"]["starter_files"] = []
                task["execution"]["verify_commands"] = ["true"]
                task["allowed_paths"] = ["src/lib/empireEconomy.ts", "src/**/*.ts"]
                task["verify_commands"] = ["true"]
                task["compile_commands"] = ["true"]
                output = """```ts path=src/lib/empireEconomy.ts
export const foodStock = 2;
export function clampFood(n: number) { return Math.max(0, Math.min(n, 999)); }
```
"""
                row = attach_rollout_evidence(
                    task=task,
                    rollout_row={"task_id": task["id"], "output": output, "prompt": "patch"},
                )
                row = attach_execution_evidence(
                    task=task,
                    rollout_row=row,
                    config=config,
                    worktree=pool.worktree_path,
                    log_root=Path(tmp) / "logs",
                )
                self.assertTrue(row.get("compiled"))
                self.assertTrue(row.get("compile_checked"))
                payload = json.loads((REPO / "benchmarks" / "economistRL_tasks_v1.json").read_text())
                score = score_output(
                    task,
                    output,
                    payload,
                    rollout_row=row,
                    compiled=row.get("compiled"),
                    rolling_compile_rate=0.5,
                )
                self.assertTrue(score["compile_gate"]["applied"])
                self.assertEqual(score["compile_gate"]["compiled"], True)
            finally:
                pool.cleanup()

    def test_no_applyable_output_sets_compiled_false(self) -> None:
        from economist_rl_execution_evidence import (
            ExecutionEvidenceConfig,
            ExecutionWorktreePool,
            attach_execution_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "game"
            self._init_git_repo(source)
            config = ExecutionEvidenceConfig(
                enabled=True,
                source_repo=source,
                worktree_root=Path(tmp) / "worktrees",
                compile_commands=("true",),
            )
            pool = ExecutionWorktreePool(config, cycle_id=98)
            try:
                task = {
                    "id": "economistRL-exec-test-02",
                    "curriculum_track": "economy",
                    "codebase_requirements": {"relevant_files": ["src/lib/economy.ts"]},
                }
                row = attach_execution_evidence(
                    task=task,
                    rollout_row={"task_id": task["id"], "output": "plan only, no fences"},
                    config=config,
                    worktree=pool.worktree_path,
                    log_root=Path(tmp) / "logs",
                )
                self.assertFalse(row.get("compiled"))
                self.assertEqual(row.get("apply_status"), "no_applyable_changes")
            finally:
                pool.cleanup()


if __name__ == "__main__":
    unittest.main()
