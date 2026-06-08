"""Tests for economistRL PPO vs scorecard vs compile evidence semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class EconomistRLPPORewardSemanticsTests(unittest.TestCase):
    def _task_and_payload(self):
        from economist_rl_reward_engine import _load_json

        payload = _load_json(REPO / "benchmarks" / "economistRL_tasks_v1.json")
        task = next(t for t in payload["tasks"] if t["id"] == "economistRL-food-steady-state-01")
        return payload, task

    def _score(self, task, payload, rollout, *, compiled=None, compile_rate=0.5):
        from economist_rl_reward_engine import score_output

        return score_output(
            task,
            str(rollout.get("output") or ""),
            payload,
            rollout_row=rollout,
            compiled=compiled,
            rolling_compile_rate=compile_rate,
        )

    def test_diagnostic_failures_keep_training_usable_and_high_reward(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence
        from economist_rl_ppo_trainer import build_ppo_samples

        payload, task = self._task_and_payload()
        ref = task["reference_answer"]
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={"task_id": task["id"], "output": ref, "prompt": task["prompt"]},
            dry_run=False,
        )
        score = self._score(task, payload, rollout, compiled=None)
        self.assertGreaterEqual(score["reward"], 0.70)
        self.assertTrue(score["training_usable"])
        self.assertTrue(score["diagnostics"])
        self.assertFalse(score["hard_cap_applied"])
        samples = build_ppo_samples(
            scored_rows=[{"task_id": task["id"], "rollout": rollout, "score": score}],
            system_prompt="system",
        )
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].reward, score["reward"], places=4)

    def test_strict_scorecard_fail_still_training_usable(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence
        from economist_rl_ppo_trainer import build_ppo_samples

        payload, task = self._task_and_payload()
        ref = task["reference_answer"]
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={"task_id": task["id"], "output": ref, "prompt": task["prompt"]},
            dry_run=False,
        )
        score = self._score(task, payload, rollout, compiled=None)
        score["strict_scorecard_pass"] = False
        score["high_reward"] = False
        score["reward"] = 0.79
        self.assertTrue(score["training_usable"])
        samples = build_ppo_samples(
            scored_rows=[{"task_id": task["id"], "rollout": rollout, "score": score}],
            system_prompt="system",
        )
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].reward, 0.79, places=4)

    def test_targeted_tests_weight_dominates_reward(self) -> None:
        from economist_rl_reward_engine import DEFAULT_BASE_REWARD_WEIGHTS

        self.assertGreater(DEFAULT_BASE_REWARD_WEIGHTS["targeted_tests"], 0.5)
        self.assertNotIn("simulation_behavior", DEFAULT_BASE_REWARD_WEIGHTS)

    def test_compiled_false_hard_cap_without_compile_bonus(self) -> None:
        payload, task = self._task_and_payload()
        rollout = {"task_id": task["id"], "output": task["reference_answer"], "prompt": task["prompt"]}
        score = self._score(task, payload, rollout, compiled=False, compile_rate=0.5)
        self.assertTrue(score["hard_cap_applied"])
        self.assertIn("compile_failed", score["cap_reason"])
        self.assertLessEqual(score["reward"], 0.10)
        self.assertEqual((score.get("compile_gate") or {}).get("compiled"), False)
        self.assertTrue(score["training_usable"])

    def test_compiled_none_no_bonus_no_hard_cap(self) -> None:
        payload, task = self._task_and_payload()
        rollout = {"task_id": task["id"], "output": task["reference_answer"], "prompt": task["prompt"]}
        score_none = self._score(task, payload, rollout, compiled=None, compile_rate=0.5)
        score_true = self._score(task, payload, rollout, compiled=True, compile_rate=0.5)
        self.assertFalse(score_none["hard_cap_applied"])
        self.assertEqual((score_none.get("compile_gate") or {}).get("compiled"), None)
        self.assertFalse((score_none.get("compile_gate") or {}).get("applied"))
        self.assertGreater(score_true["reward"], score_none["reward"])

    def test_failures_alone_do_not_zero_reward(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence

        payload, task = self._task_and_payload()
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={"task_id": task["id"], "output": task["reference_answer"], "prompt": task["prompt"]},
            dry_run=False,
        )
        score = self._score(task, payload, rollout, compiled=None)
        self.assertTrue(score["failures"])
        self.assertGreater(score["reward"], 0.5)

    def test_evidence_runner_does_not_set_compiled_from_code_fences(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence

        payload, task = self._task_and_payload()
        fenced = "```ts\nexport function foo() { return 1; }\n```"
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={"task_id": task["id"], "output": fenced, "prompt": task["prompt"]},
            dry_run=False,
        )
        self.assertTrue(rollout.get("has_code_fence"))
        self.assertTrue(rollout.get("looks_code_like"))
        self.assertNotIn("compiled", rollout)
        self.assertFalse(rollout.get("compile_checked"))

    def test_real_compile_evidence_sets_compiled(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence

        payload, task = self._task_and_payload()
        rollout = attach_rollout_evidence(
            task=task,
            rollout_row={
                "task_id": task["id"],
                "output": task["reference_answer"],
                "prompt": task["prompt"],
                "compile_evidence": {"passed": True, "command": "tsc --noEmit"},
            },
            dry_run=False,
        )
        self.assertTrue(rollout.get("compiled"))
        self.assertTrue(rollout.get("compile_checked"))


if __name__ == "__main__":
    unittest.main()
