"""Tests for the economistRL Lambda split-worker launcher wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class LaunchEconomistRlLambdaSplitWorkersTests(unittest.TestCase):
    def test_ensure_split_worker_argv_injects_policy_defaults(self) -> None:
        from launch_economist_rl_lambda_split_workers import _ensure_split_worker_argv

        argv = _ensure_split_worker_argv(["--", "--max-ppo-updates", "2"])

        self.assertIn("--lambda-mode", argv)
        self.assertIn("--specialization", argv)
        self.assertIn("economist_rl", argv)
        self.assertIn("--rollout-batch-size", argv)
        self.assertEqual(argv[argv.index("--rollout-batch-size") + 1], "25")
        self.assertIn("--bootstrap-rollout-batch-size", argv)
        self.assertEqual(argv[argv.index("--bootstrap-rollout-batch-size") + 1], "50")
        self.assertIn("--ppo-min-samples", argv)
        self.assertEqual(argv[argv.index("--ppo-min-samples") + 1], "25")
        self.assertIn("--ppo-max-samples", argv)
        self.assertEqual(argv[argv.index("--ppo-max-samples") + 1], "64")
        self.assertIn("--ppo-mini-batch-size", argv)
        self.assertEqual(argv[argv.index("--ppo-mini-batch-size") + 1], "8")
        self.assertIn("--acceptance-mode", argv)
        self.assertEqual(argv[argv.index("--acceptance-mode") + 1], "trained_marker")

    def test_ensure_split_worker_argv_preserves_explicit_knobs(self) -> None:
        from launch_economist_rl_lambda_split_workers import _ensure_split_worker_argv

        argv = _ensure_split_worker_argv(
            [
                "--",
                "--rollout-batch-size",
                "40",
                "--bootstrap-rollout-batch-size",
                "80",
                "--ppo-min-samples",
                "30",
                "--ppo-max-samples",
                "96",
                "--ppo-mini-batch-size",
                "16",
                "--acceptance-mode",
                "none",
            ]
        )

        self.assertEqual(argv[argv.index("--rollout-batch-size") + 1], "40")
        self.assertEqual(argv[argv.index("--bootstrap-rollout-batch-size") + 1], "80")
        self.assertEqual(argv[argv.index("--ppo-min-samples") + 1], "30")
        self.assertEqual(argv[argv.index("--ppo-max-samples") + 1], "96")
        self.assertEqual(argv[argv.index("--ppo-mini-batch-size") + 1], "16")
        self.assertEqual(argv[argv.index("--acceptance-mode") + 1], "none")

    def test_build_split_worker_command_targets_split_orchestrator(self) -> None:
        from launch_economist_rl_lambda_split_workers import _build_split_worker_command

        cmd = _build_split_worker_command(["--lambda-mode", "--max-ppo-updates", "1"])

        self.assertIn("scripts/lambda/run_economist_rl_split_workers.py", cmd)
        self.assertIn("--lambda-mode", cmd)
        self.assertIn("--max-ppo-updates", cmd)

    def test_effective_watchdog_idle_minutes_has_five_hour_floor(self) -> None:
        from launch_economist_rl_lambda_split_workers import _effective_watchdog_idle_minutes

        self.assertEqual(_effective_watchdog_idle_minutes(180), 300.0)
        self.assertEqual(_effective_watchdog_idle_minutes(360), 360.0)

    def test_custom_init_adapter_path_is_selected_for_sync(self) -> None:
        from launch_economist_rl_lambda_split_workers import _custom_init_adapter_sync_path

        self.assertEqual(
            _custom_init_adapter_sync_path(
                [
                    "--lambda-mode",
                    "--init-adapter-path",
                    "checkpoints/adapters/economistRL/rl_pass_split_001",
                ],
                default_adapter_id="economistRL",
            ),
            "checkpoints/adapters/economistRL/rl_pass_split_001",
        )
        self.assertEqual(
            _custom_init_adapter_sync_path(
                ["--lambda-mode", "--init-adapter-path", "checkpoints/fe-lora-arena-apply-sft"],
                default_adapter_id="economistRL",
            ),
            "",
        )

    def test_hf_cache_model_id_maps_mlx_qwen_to_transformers_qwen(self) -> None:
        from launch_economist_rl_lambda_split_workers import _hf_cache_model_id

        self.assertEqual(
            _hf_cache_model_id(["--base-model", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]),
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        )
        self.assertEqual(
            _hf_cache_model_id(["--base-model", "Qwen/Qwen2.5-Coder-7B-Instruct"]),
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        )


if __name__ == "__main__":
    unittest.main()
