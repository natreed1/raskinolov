"""Regression coverage for keyword-routed documentation specialist prompts."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS_JSON = REPO / "benchmarks" / "documentation_eval_tasks_v1.json"
REGISTRY_JSON = REPO / "training" / "adapter_registry_v1.json"


def _documentation_adapter_weights() -> Path:
    data = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    for row in data.get("entries", []):
        if row.get("adapter_id") == "documentation":
            adapter_path = Path(str(row["adapter_path"]))
            if not adapter_path.is_absolute():
                adapter_path = REPO / adapter_path
            return adapter_path / "adapters.safetensors"
    raise AssertionError("documentation adapter missing from training/adapter_registry_v1.json")


class DocumentationRoutingEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

        cls._GenerationRequest = GenerationRequest
        cls._RoutingPolicy = RoutingPolicy
        cls._messages_from_prompt = staticmethod(messages_from_prompt)

        raw = TASKS_JSON.read_text(encoding="utf-8")
        cls.tasks = json.loads(raw)

    def test_benchmark_file_exists(self) -> None:
        self.assertTrue(TASKS_JSON.is_file(), "Run scripts/build_documentation_eval_tasks_v1.py")

    def test_each_row_routes_to_documentation_local(self) -> None:
        policy = self._RoutingPolicy()
        failures: list[str] = []
        for row in self.tasks:
            prompt = row["prompt"]
            expected_adapter = row.get("expected_adapter_id")
            expected_route = row.get("expected_legacy_route") or row.get("expected_route")
            req = self._GenerationRequest(messages=self._messages_from_prompt(prompt))
            d = policy.decide(req)
            if expected_adapter is not None and d.adapter_id != expected_adapter:
                failures.append(f"{row['id']}: adapter wanted {expected_adapter} got {d.adapter_id}")
            if expected_route is not None and d.route != expected_route:
                failures.append(f"{row['id']}: route wanted {expected_route} got {d.route}")
        self.assertFalse(failures, ";\n".join(failures))

    def test_cli_benchmark_matches_fixture(self) -> None:
        """Keeps parity with automation that runs ``run_routing_benchmark.py`` in CI-ish shells."""
        import os

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
        self.assertEqual(
            proc.returncode,
            0,
            f"routing benchmark stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
        )


@unittest.skipUnless(
    _documentation_adapter_weights().is_file(),
    "No trained adapters.safetensors for registry documentation adapter",
)
class DocumentationAdapterCheckpointTests(unittest.TestCase):
    def test_checkpoint_non_empty(self) -> None:
        doc_adapter_weights = _documentation_adapter_weights()
        self.assertGreater(
            doc_adapter_weights.stat().st_size,
            1_000_000,
            "adapter weights unexpectedly small — training may have failed",
        )


if __name__ == "__main__":
    unittest.main()
