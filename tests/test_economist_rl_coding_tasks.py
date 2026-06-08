#!/usr/bin/env python3
"""Tests for economistRL coding task conversion."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/adapters"))


class EconomistRLCodingTasksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2 = REPO / "benchmarks/economistRL_tasks_v2_coding.json"
        if not cls.v2.is_file():
            from convert_economist_rl_tasks_to_coding import _load, convert_bank

            payload = _load(REPO / "benchmarks/economistRL_tasks_v1.json")
            cls.v2.write_text(json.dumps(convert_bank(payload), indent=2) + "\n", encoding="utf-8")
        cls.payload = json.loads(cls.v2.read_text(encoding="utf-8"))

    def test_all_tasks_have_fenced_reference_answers(self) -> None:
        tasks = self.payload["tasks"]
        missing = [t["id"] for t in tasks if "```" not in str(t.get("reference_answer") or "")]
        self.assertEqual(missing, [], f"missing fences: {missing[:5]}")

    def test_prompts_require_applyable_output(self) -> None:
        task = self.payload["tasks"][0]
        prompt = str(task.get("prompt") or "")
        self.assertIn("Fenced full-file blocks", prompt)
        self.assertIn("allowed paths", prompt.lower())

    def test_allowed_paths_and_compile_commands_present(self) -> None:
        task = self.payload["tasks"][0]
        self.assertTrue(task.get("allowed_paths"))
        self.assertTrue(task.get("compile_commands"))


if __name__ == "__main__":
    unittest.main()
