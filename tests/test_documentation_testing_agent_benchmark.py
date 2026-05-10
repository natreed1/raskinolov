"""Regression coverage for the documentation/testing-agent benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS_JSON = REPO / "benchmarks" / "documentation_testing_agent_eval_tasks_v1.json"
PROMPTS_JSONL = REPO / "data" / "routing" / "documentation_testing_agent_eval_prompts_v1.jsonl"
BUILDER = REPO / "scripts" / "build_documentation_testing_agent_eval_tasks_v1.py"


class DocumentationTestingAgentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

        cls._GenerationRequest = GenerationRequest
        cls._RoutingPolicy = RoutingPolicy
        cls._messages_from_prompt = staticmethod(messages_from_prompt)

    def test_benchmark_assets_exist(self) -> None:
        self.assertTrue(BUILDER.is_file(), "missing benchmark builder")
        self.assertTrue(TASKS_JSON.is_file(), "run the benchmark builder")
        self.assertTrue(PROMPTS_JSONL.is_file(), "run the benchmark builder")

    def test_fixture_has_stable_docs_testing_shape(self) -> None:
        rows = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 12)
        ids = {row["id"] for row in rows}
        self.assertEqual(len(ids), len(rows), "task ids must be unique")
        for row in rows:
            prompt = row["prompt"].lower()
            self.assertEqual(row["expected_adapter_id"], "documentation")
            self.assertEqual(row["expected_legacy_route"], "local")
            self.assertEqual(row["risk"], "low")
            self.assertEqual(row["shard"], "documentation_testing_agent_eval")
            self.assertTrue(
                any(term in prompt for term in ("test", "unittest", "pytest", "benchmark", "smoke")),
                f"{row['id']} does not exercise testing context",
            )

    def test_each_row_routes_to_documentation_local(self) -> None:
        policy = self._RoutingPolicy()
        rows = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
        failures: list[str] = []
        for row in rows:
            req = self._GenerationRequest(messages=self._messages_from_prompt(row["prompt"]))
            decision = policy.decide(req)
            if decision.adapter_id != row["expected_adapter_id"]:
                failures.append(f"{row['id']}: adapter wanted documentation got {decision.adapter_id}")
            if decision.route != row["expected_legacy_route"]:
                failures.append(f"{row['id']}: route wanted local got {decision.route}")
        self.assertFalse(failures, ";\n".join(failures))

    def test_cli_benchmark_matches_fixture(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "run_routing_benchmark.py"),
                "--tasks",
                str(TASKS_JSON),
                "--mode",
                "both",
            ],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO / "scripts")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")


if __name__ == "__main__":
    unittest.main()
