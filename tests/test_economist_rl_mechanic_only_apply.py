#!/usr/bin/env python3
"""Sandbox apply must not accept model-written tests/ fences."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class EconomistRLMechanicOnlyApplyTests(unittest.TestCase):
    def test_sandbox_apply_rejects_tests_fence(self) -> None:
        from economist_rl_execution_evidence import (
            ExecutionEvidenceConfig,
            ExecutionWorktreePool,
            attach_execution_evidence,
        )
        from economist_rl_task_execution import enrich_task_execution, sandbox_paths

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "game"
            source.mkdir(parents=True)
            (source / "package.json").write_text('{"name":"fe"}\n', encoding="utf-8")
            subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "t@t.com"],
                cwd=source,
                check=True,
                capture_output=True,
            )
            subprocess.run(["git", "config", "user.name", "t"], cwd=source, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "i"], cwd=source, check=True, capture_output=True)

            task = enrich_task_execution(
                {
                    "id": "economistRL-mechanic-only-01",
                    "subsection": "market_elasticity_pricing",
                    "curriculum_track": "economy",
                    "simulation_spec": {
                        "tick_count": 20,
                        "goals": [{"name": "scarcity_raises_price", "weight": 1.0}],
                    },
                }
            )
            lib, test = sandbox_paths(task)
            output = f"""```ts path={lib}
export function applyMarketElasticityPricingTick(state: any): any {{ return state; }}
```
```ts path={test}
import {{ test }} from '@jest/globals';
test('bad', () => {{}});
```
"""
            config = ExecutionEvidenceConfig(
                enabled=True,
                source_repo=source,
                worktree_root=Path(tmp) / "wt",
                compile_commands=("true",),
                timeout_s=30,
            )
            pool = ExecutionWorktreePool(config, cycle_id=77)
            try:
                row = attach_execution_evidence(
                    task=task,
                    rollout_row={"task_id": task["id"], "output": output},
                    config=config,
                    worktree=pool.worktree_path,
                    log_root=Path(tmp) / "logs",
                )
                self.assertNotIn(test, str(row.get("changed_files") or ""))
                self.assertFalse((pool.worktree_path / test).exists())
            finally:
                pool.cleanup()

    def test_sandbox_prompt_forbids_tests_in_apply_list(self) -> None:
        from economist_rl_coding_contract import sandbox_coding_user_prompt
        from economist_rl_task_execution import apply_allowed_paths_for_task, enrich_task_execution

        task = enrich_task_execution(
            {
                "id": "economistRL-market-elasticity-02",
                "subsection": "market_elasticity_pricing",
                "simulation_spec": {"goals": [{"name": "scarcity_raises_price", "weight": 1.0}]},
            }
        )
        prompt = sandbox_coding_user_prompt(task)
        self.assertIn("Do **not** output any path under `tests/`", prompt)
        apply_allowed = apply_allowed_paths_for_task(task)
        self.assertFalse(any("tests/" in p for p in apply_allowed))


if __name__ == "__main__":
    unittest.main()
