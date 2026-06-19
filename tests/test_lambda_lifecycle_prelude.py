"""Tests for Lambda lifecycle guard script generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class LambdaLifecyclePreludeTests(unittest.TestCase):
    def test_lifecycle_prelude_includes_hard_runtime_and_verified_termination(self) -> None:
        from launch_lambda_parallel_ablation import _remote_lifecycle_prelude

        script = "\n".join(
            _remote_lifecycle_prelude(
                api_base="https://cloud.lambdalabs.com/api/v1",
                api_key="secret",
                instance_id="i-123",
                auto_terminate=True,
                watchdog_enabled=True,
                watchdog_idle_minutes=5,
                watchdog_check_minutes=1,
                watchdog_log_path="/tmp/run.log",
                gpu_monitor_log_path="/tmp/gpu.csv",
                artifact_export_command="",
                artifact_upload_every_steps=0,
                artifact_upload_every_minutes=15,
                require_artifact_export_before_terminate=True,
                artifact_staging_dir="/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts",
                artifact_staging_is_durable=True,
                watchdog_max_runtime_minutes=30,
            )
        )

        self.assertIn("export FE_WATCHDOG_MAX_RUNTIME_SECONDS=1800", script)
        self.assertIn("runtime=$((now - started_at))", script)
        self.assertIn("max runtime", script)
        self.assertIn("verify_lambda_terminated", script)
        self.assertIn("[lambda-terminate] verified termination", script)


if __name__ == "__main__":
    unittest.main()
