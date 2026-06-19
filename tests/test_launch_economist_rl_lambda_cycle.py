"""Tests for deprecated economistRL Lambda cycle launcher CLI wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class LaunchEconomistRlLambdaCycleTests(unittest.TestCase):
    def test_load_repo_dotenv_maps_lambda_url_to_cloud_base(self) -> None:
        import os
        from launch_economist_rl_lambda_cycle import _load_repo_dotenv

        prior = {
            k: os.environ.get(k)
            for k in ("LAMBDA_URL", "LAMBDA_CLOUD_BASE_URL", "LAMBDA_API_BASE", "LAMBDA_API_KEY")
        }
        try:
            for key in prior:
                os.environ.pop(key, None)
            env_file = REPO / ".env"
            if not env_file.is_file():
                self.skipTest(".env not present")
            _load_repo_dotenv(env_file)
            self.assertTrue((os.environ.get("LAMBDA_API_KEY") or "").strip())
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_ensure_cycle_argv_injects_lambda_mode(self) -> None:
        from launch_economist_rl_lambda_cycle import _ensure_cycle_argv

        argv = _ensure_cycle_argv(["--", "--cycles", "2"])
        self.assertIn("--lambda-mode", argv)
        self.assertIn("--specialization", argv)
        self.assertIn("economist_rl", argv)
        self.assertIn("--rollouts-per-cycle", argv)
        self.assertEqual(argv[argv.index("--rollouts-per-cycle") + 1], "25")
        self.assertIn("--bootstrap-rollouts-per-cycle", argv)
        self.assertEqual(argv[argv.index("--bootstrap-rollouts-per-cycle") + 1], "50")
        self.assertIn("--ppo-min-samples", argv)
        self.assertEqual(argv[argv.index("--ppo-min-samples") + 1], "25")
        self.assertIn("--ppo-max-samples", argv)
        self.assertEqual(argv[argv.index("--ppo-max-samples") + 1], "64")

    def test_ensure_cycle_argv_preserves_explicit_worker_knobs(self) -> None:
        from launch_economist_rl_lambda_cycle import _ensure_cycle_argv

        argv = _ensure_cycle_argv(
            [
                "--",
                "--rollouts-per-cycle",
                "40",
                "--bootstrap-rollouts-per-cycle",
                "80",
                "--ppo-min-samples",
                "30",
                "--ppo-max-samples",
                "90",
            ]
        )
        self.assertEqual(argv[argv.index("--rollouts-per-cycle") + 1], "40")
        self.assertEqual(argv[argv.index("--bootstrap-rollouts-per-cycle") + 1], "80")
        self.assertEqual(argv[argv.index("--ppo-min-samples") + 1], "30")
        self.assertEqual(argv[argv.index("--ppo-max-samples") + 1], "90")

    def test_build_cycle_command_quotes_args(self) -> None:
        from launch_economist_rl_lambda_cycle import _build_cycle_command

        cmd = _build_cycle_command(["--lambda-mode", "--cycles", "1"])
        self.assertIn("scripts/lambda/run_economist_rl_lambda_cycle.py", cmd)
        self.assertIn("--lambda-mode", cmd)

    def test_deprecation_notice_points_to_split_worker_launcher(self) -> None:
        from launch_economist_rl_lambda_cycle import DEPRECATION_NOTICE

        self.assertIn("DEPRECATED", DEPRECATION_NOTICE)
        self.assertIn("launch_economist_rl_lambda_split_workers.py", DEPRECATION_NOTICE)


if __name__ == "__main__":
    unittest.main()
