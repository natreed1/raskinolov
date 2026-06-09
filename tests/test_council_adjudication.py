"""Tests for equal-prior, blind council adjudication."""

from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class CouncilAdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_selected_specialist_prior_does_not_decide_winner(self) -> None:
        from router.council import adjudicate_council_outputs

        prompt = "Patch Docker runtime configuration so bare imports resolve under /app/src and add verification."
        outputs = [
            {
                "participant_id": "loading_screen::assertive",
                "text": "Use a nice loading hint.",
                "confidence": 0.99,
                "task_outcome_score": 0.99,
            },
            {
                "participant_id": "precise_short",
                "text": (
                    "Update the Dockerfile runtime configuration to set `NODE_PATH=/app/src:/app/node_modules`, "
                    "verify the import path in `game-server/src/index.ts`, rebuild the container, and run CI tests. "
                    "Risk: changing module resolution can mask a missing dependency."
                ),
                "confidence": 0.5,
                "task_outcome_score": 0.5,
            },
        ]

        result = adjudicate_council_outputs(
            outputs=outputs,
            prompt=prompt,
            disagreement=0.0,
            low_confidence_threshold=0.2,
            disagreement_threshold=0.9,
            escalation_rule="manual_only",
        ).to_dict()

        self.assertEqual(result["winner_ids"][0], "precise_short")
        self.assertGreater(result["contributions"][0]["score"], result["contributions"][1]["score"])

    def test_contributions_include_blind_ids_and_rubric_scores(self) -> None:
        from router.council import adjudicate_council_outputs

        result = adjudicate_council_outputs(
            outputs=[
                {"participant_id": "alpha", "text": "Test and verify `src/file.ts` for risk."},
                {"participant_id": "beta", "text": "Test and verify `src/file.ts` for risk."},
            ],
            prompt="Verify src file risk",
            disagreement=0.0,
            low_confidence_threshold=0.2,
            disagreement_threshold=0.9,
            escalation_rule="manual_only",
        ).to_dict()

        contribution = result["contributions"][0]
        self.assertIn("blind_id", contribution)
        self.assertIn("rubric_scores", contribution)
        self.assertEqual(contribution["task_outcome_score"], 0.5)

    def test_conversation_eval_marks_specialist_adapters_unloaded(self) -> None:
        from model_router import RoutingPolicy
        from run_council_conversation_eval import DEFAULT_LOCAL_MODEL, _neutral_scoring_metadata, _run_task

        confidence, task_outcome = _neutral_scoring_metadata()

        self.assertEqual(confidence, 0.5)
        self.assertEqual(task_outcome, 0.5)

        old_enabled = os.environ.get("ROUTER_COUNCIL_ENABLED")
        os.environ["ROUTER_COUNCIL_ENABLED"] = "1"
        try:
            row = _run_task(
                task={
                    "id": "smoke-loading",
                    "prompt": "Write loading screen copy and verify the UI text stays concise.",
                },
                task_meta={},
                policy=RoutingPolicy(),
                backend_cache={},
                model_id=DEFAULT_LOCAL_MODEL,
                debate_max_rounds=1,
                participant_max_tokens=32,
                mock_generation=True,
            )
        finally:
            if old_enabled is None:
                os.environ.pop("ROUTER_COUNCIL_ENABLED", None)
            else:
                os.environ["ROUTER_COUNCIL_ENABLED"] = old_enabled
        specialist_outputs = [
            participant
            for round_row in row["rounds"]
            for participant in round_row["participants"]
            if participant["participant_type"] == "specialist_adapter"
        ]

        self.assertTrue(specialist_outputs)
        self.assertTrue(any(participant["adapter_requested"] for participant in specialist_outputs))
        self.assertTrue(all(participant["adapter_loaded"] is False for participant in specialist_outputs))


if __name__ == "__main__":
    unittest.main()
