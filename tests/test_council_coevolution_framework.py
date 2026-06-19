"""Tests for council co-evolution framework contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CouncilCoevolutionFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_bootstrap_state_creates_reward_and_lora_populations(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry
        from council_runtime.coevolution import COEVOLUTION_STATE_SCHEMA, bootstrap_state

        registry = CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"economistRL"})
        state = bootstrap_state(registry)
        payload = state.to_dict()

        self.assertEqual(payload["schema_version"], COEVOLUTION_STATE_SCHEMA)
        self.assertEqual(state.reward_population.active_matrix_id, "reward_balanced_v1")
        self.assertIn("planner", state.archetypes_by_id())
        self.assertIn("economistRL_eq", state.archetypes_by_id())
        planner = state.archetypes_by_id()["planner"].active()
        self.assertEqual(planner.rank, 16)
        self.assertEqual(planner.reward_matrix_id, "reward_balanced_v1")

    def test_reward_matrix_scores_and_mutates(self) -> None:
        from council_runtime.coevolution import default_reward_population

        reward = default_reward_population().active()
        score = reward.score(
            {
                "final_outcome": 1.0,
                "contribution_credit": 0.5,
                "evidence_quality": 0.5,
                "routing_or_reroute_quality": 0.0,
            }
        )
        self.assertGreater(score, 0.3)
        self.assertLessEqual(score, 1.0)

        child = reward.mutate(child_id="reward_child", focus_signal="evidence_quality", delta=0.1)
        self.assertEqual(child.parent_id, reward.matrix_id)
        self.assertGreater(child.weights["evidence_quality"], reward.weights["evidence_quality"])

    def test_dry_season_writes_manifest_and_phase_artifacts(self) -> None:
        from council_runtime.archetypes import CouncilArchetypeRegistry
        from council_runtime.coevolution import SEASON_MANIFEST_SCHEMA
        from council_runtime.coevolution_runner import run_dry_season

        registry = CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"hud_status"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = run_dry_season(
                archetype_registry=registry,
                state_path=root / "state.json",
                results_root=root / "results",
                season_id="season_0001_test",
                bootstrap=True,
            )
            manifest_path = Path(manifest.run_dir) / "SEASON_MANIFEST.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], SEASON_MANIFEST_SCHEMA)
            self.assertEqual(payload["status"], "dry_run_complete")
            self.assertEqual(len(payload["phases"]), 9)
            for phase in payload["phases"]:
                self.assertTrue(Path(phase["outputs"]["phase_artifact"]).is_file())
            self.assertTrue((Path(manifest.run_dir) / "RUN.md").is_file())


if __name__ == "__main__":
    unittest.main()
