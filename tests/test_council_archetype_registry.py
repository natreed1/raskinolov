"""Tests for council archetype registry contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CouncilArchetypeRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_bootstrap_creates_global_and_specialist_eq_lanes(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry

        registry = CouncilArchetypeRegistry.bootstrap(
            specialist_adapter_ids={"economistRL", "hud_status", "general_fallback"}
        )
        by_id = registry.by_id()

        self.assertIn("planner", by_id)
        self.assertIn("grader", by_id)
        self.assertIn("context_compressor", by_id)
        self.assertIn("economistRL_eq", by_id)
        self.assertIn("hud_status_eq", by_id)
        self.assertNotIn("general_fallback_eq", by_id)
        self.assertEqual(by_id["planner"].default_rank, 16)
        self.assertEqual(by_id["grader"].lane.trace_schema, "grader_trace_v1")
        self.assertTrue(by_id["economistRL_eq"].depends_on_adapter_registry)
        self.assertEqual(by_id["economistRL_eq"].source_adapter_id, "economistRL")

    def test_save_and_load_round_trip_validates_schema(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry, SCHEMA_VERSION

        registry = CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"save_load_api_guard"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            registry.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)

            loaded = CouncilArchetypeRegistry.load(path)
            self.assertEqual(set(loaded.by_id()), set(registry.by_id()))

    def test_invalid_archetype_is_rejected(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry

        payload = CouncilArchetypeRegistry.bootstrap().to_dict()
        payload["entries"][0]["archetype"] = "not_a_real_archetype"

        with self.assertRaises(ValueError):
            CouncilArchetypeRegistry.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
