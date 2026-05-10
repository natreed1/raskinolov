"""Unit tests for policy ``force_route=local`` (Codebase OSS mode in router_chat_gradio UI)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class RouterBackboneModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

        cls._GenerationRequest = GenerationRequest
        cls._RoutingPolicy = RoutingPolicy
        cls._messages_from_prompt = staticmethod(messages_from_prompt)

    def test_force_local_beats_security_keywords(self) -> None:
        policy = self._RoutingPolicy()
        prompt = "Full production security cryptography audit rewrite — payment tokens"
        auto = policy.decide(self._GenerationRequest(messages=self._messages_from_prompt(prompt)))
        local = policy.decide(
            self._GenerationRequest(
                messages=self._messages_from_prompt(prompt),
                force_route="local",
            )
        )
        self.assertEqual(local.route, "local")
        self.assertNotEqual(
            auto.route,
            local.route,
            "fixture should probe something policy normally escalates to frontier",
        )

    def test_force_local_keeps_classifier_adapter_choice(self) -> None:
        policy = self._RoutingPolicy()
        prompt = (
            "Update docs/PROJECT_STATE mlx pin table for maintainers drafting SESSION_LOG etiquette."
        )
        unlocked = policy.decide(self._GenerationRequest(messages=self._messages_from_prompt(prompt)))
        forced = policy.decide(
            self._GenerationRequest(messages=self._messages_from_prompt(prompt), force_route="local")
        )
        self.assertEqual(forced.route, "local")
        self.assertEqual(
            unlocked.adapter_id,
            forced.adapter_id,
            "force_route local should reuse classifier.adapter_id per RoutingPolicy.decide",
        )

    def test_force_local_keeps_loading_screen_when_router_would_stay_specialist(self) -> None:
        policy = self._RoutingPolicy()
        prompt = "Polish the loading screen serif headings and pacing on empire crest reveal."
        unlocked = policy.decide(self._GenerationRequest(messages=self._messages_from_prompt(prompt)))
        forced = policy.decide(
            self._GenerationRequest(messages=self._messages_from_prompt(prompt), force_route="local")
        )
        self.assertEqual(forced.adapter_id, "loading_screen")
        self.assertGreaterEqual(forced.confidence, 0.0)
        self.assertEqual(unlocked.adapter_id, forced.adapter_id)

