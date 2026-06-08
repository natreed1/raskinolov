#!/usr/bin/env python3
"""Tests for Vitest JSON partial credit parsing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_vitest_scoring import merge_outcome_checks_with_vitest, parse_vitest_json_report  # noqa: E402


class EconomistRLVitestScoringTests(unittest.TestCase):
    def test_partial_credit_from_assertions(self) -> None:
        payload = {
            "numTotalTests": 1,
            "numPassedTests": 0,
            "testResults": [
                {
                    "assertionResults": [
                        {"title": "a", "status": "passed"},
                        {"title": "b", "status": "failed"},
                        {"title": "c", "status": "passed"},
                    ]
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vitest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = parse_vitest_json_report(path)
        self.assertAlmostEqual(parsed["score"], 2 / 3, places=3)
        self.assertEqual(parsed["assertions_total"], 3)
        merged = merge_outcome_checks_with_vitest(
            {"targeted_tests": {"outcome_checks": ["goal_a", "goal_b"]}},
            parsed,
        )
        self.assertAlmostEqual(merged["score"], 2 / 3, places=3)
        self.assertEqual(len(merged["checks"]), 2)


if __name__ == "__main__":
    unittest.main()
