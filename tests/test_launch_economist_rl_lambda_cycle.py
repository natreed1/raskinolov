"""Tests for economistRL Lambda cycle launcher CLI wiring."""

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

        argv = _ensure_cycle_argv(["--", "--cycles", "2", "--rollouts-per-cycle", "50"])
        self.assertIn("--lambda-mode", argv)
        self.assertIn("--specialization", argv)
        self.assertIn("economist_rl", argv)

    def test_build_cycle_command_quotes_args(self) -> None:
        from launch_economist_rl_lambda_cycle import _build_cycle_command

        cmd = _build_cycle_command(["--lambda-mode", "--cycles", "1"])
        self.assertIn("scripts/lambda/run_economist_rl_lambda_cycle.py", cmd)
        self.assertIn("--lambda-mode", cmd)


if __name__ == "__main__":
    unittest.main()
