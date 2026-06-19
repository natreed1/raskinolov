"""Tests for reusable Lambda run extraction helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lambda"))


class LambdaRunExtractionTests(unittest.TestCase):
    def test_write_extract_manifest_records_readme_index_and_launch_log(self) -> None:
        from lambda_run_extraction import LambdaRunExtractor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launch_log = root / "logs" / "launch.log"
            launch_log.parent.mkdir(parents=True)
            launch_log.write_text("launch output\n", encoding="utf-8")

            extractor = LambdaRunExtractor(repo_root=root, extracts_root=root / "extracts")
            dest = extractor.extract_dir(attempt=2, instance_id="abc123456789", label="training_failed")
            extractor.write_extract_manifest(
                dest,
                run_id="run_0002_20260613T220000Z",
                run_number=2,
                run_date_utc="2026-06-13",
                attempt=2,
                instance_id="abc123456789",
                host="1.2.3.4",
                failure_class="training_failed",
                status="failed",
                progress={"last_rollout": "7/50", "last_cycle": "1"},
                remote_tail="tail text\n",
                launch_log=launch_log,
                rsync_paths=["cloud-eval-logs"],
            )

            manifest = json.loads((dest / "EXTRACT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "economist_rl_extract_manifest_v2")
            self.assertEqual(manifest["run_id"], "run_0002_20260613T220000Z")
            self.assertEqual(manifest["run_number"], 2)
            self.assertEqual(manifest["run_date_utc"], "2026-06-13")
            self.assertEqual(manifest["attempt"], 2)
            self.assertEqual(manifest["instance_id"], "abc123456789")
            self.assertEqual(manifest["rsync_paths"], ["cloud-eval-logs"])
            self.assertEqual(manifest["launch_log"], "logs/launch.log")
            self.assertIn("training_failed", (dest / "README.md").read_text(encoding="utf-8"))
            self.assertEqual((dest / "cycle_log_tail.txt").read_text(encoding="utf-8"), "tail text\n")
            self.assertEqual((dest / "launch_log.txt").read_text(encoding="utf-8"), "launch output\n")

            index_rows = (root / "extracts" / "index.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(index_rows), 1)
            index_row = json.loads(index_rows[0])
            self.assertEqual(index_row["run_id"], "run_0002_20260613T220000Z")
            self.assertEqual(index_row["last_rollout"], "7/50")

    def test_rsync_artifacts_uses_configured_specs_and_records_successes(self) -> None:
        from lambda_run_extraction import LambdaRunExtractor, RemoteRsyncSpec

        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extractor = LambdaRunExtractor(
                repo_root=root,
                extracts_root=root / "extracts",
                ssh_key_path=root / "id_rsa",
                command_runner=fake_run,
            )
            dest = extractor.rsync_artifacts(
                "5.6.7.8",
                attempt=1,
                instance_id="feedface0000",
                failure_class="completed",
                status="completed",
                progress={},
                remote_tail="",
                launch_log=None,
                specs=[
                    RemoteRsyncSpec("~/cloud-eval-logs/", "cloud-eval-logs"),
                    RemoteRsyncSpec("~/fallen-empire-lora/benchmarks/results/economistRL/", "economistRL"),
                ],
            )

            self.assertEqual(len(calls), 2)
            self.assertIn("ubuntu@5.6.7.8:~/cloud-eval-logs/", calls[0])
            manifest = json.loads((dest / "EXTRACT_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rsync_paths"], ["cloud-eval-logs", "economistRL"])

    def test_live_snapshot_writes_progress_tail_and_timestamp(self) -> None:
        from lambda_run_extraction import LambdaRunExtractor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extractor = LambdaRunExtractor(repo_root=root, extracts_root=root / "extracts")
            extractor.write_live_snapshot(
                instance_id="abc123456789",
                progress={"last_rollout": "3/50"},
                remote_log="log tail",
            )

            live = root / "extracts" / "live_abc123456789"
            self.assertEqual(json.loads((live / "progress.json").read_text(encoding="utf-8"))["last_rollout"], "3/50")
            self.assertEqual((live / "cycle_log_tail.txt").read_text(encoding="utf-8"), "log tail")
            self.assertTrue((live / "LAST_UPDATED_UTC.txt").read_text(encoding="utf-8").strip())

    def test_watcher_injects_run_identity_args_once(self) -> None:
        from watch_economist_rl_lambda_overnight import _ensure_run_identity_args

        argv = _ensure_run_identity_args(["--lambda-mode"], run_id="run_0007_20260613T220000Z", run_number=7)
        self.assertEqual(argv[-4:], ["--run-id", "run_0007_20260613T220000Z", "--run-number", "7"])

        preserved = _ensure_run_identity_args(
            ["--run-id", "custom", "--run-number", "12"],
            run_id="run_0007_20260613T220000Z",
            run_number=7,
        )
        self.assertEqual(preserved, ["--run-id", "custom", "--run-number", "12"])


if __name__ == "__main__":
    unittest.main()
