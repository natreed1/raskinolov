#!/usr/bin/env python3
"""Tests for sandbox ↔ game field name normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_field_names import canonicalize_field_list, canonicalize_state  # noqa: E402


class EconomistRLFieldNamesTests(unittest.TestCase):
    def test_legacy_scenario_keys_normalize(self) -> None:
        state = canonicalize_state(
            {
                "foodStock": 40,
                "currentStock": 50,
                "projectionCache": 0,
                "gold": 100,
            }
        )
        self.assertEqual(state["storageFood"], 40.0)
        self.assertEqual(state["storageGoods"], 50.0)
        self.assertEqual(state["resourceProjectionValid"], 0.0)
        self.assertEqual(state["playerGold"], 100.0)

    def test_relevant_state_list_dedupes_aliases(self) -> None:
        fields = canonicalize_field_list(["foodStock", "storageFood", "population"])
        self.assertEqual(fields, ["storageFood", "population"])


if __name__ == "__main__":
    unittest.main()
