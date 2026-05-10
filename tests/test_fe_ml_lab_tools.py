"""Unit tests for the MLX lab runner + optimization dashboard scaffolding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def _load_dashboard_module():
    path = REPO / "scripts" / "build_lab_optimization_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_lab_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def _load_runner_module():
    path = REPO / "scripts" / "fe_ml_lab_runner.py"
    spec = importlib.util.spec_from_file_location("fe_ml_lab_runner_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


_SAMPLE_HISTORY = """# ML workflow run history

| UTC ISO              | Subcommand  | Exit | Status  | Train iters | Benchmark | X | X | Artifacts                                         |
| -------------------- | ----------- | ---- | ------- | ----------- | --------- | --------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| 2026-04-26T22:09:08Z | `evalplus`  | 1    | failed  | —           | —         | — | — | `benchmarks/results/runs/20260426-220907_986ba0/` |
| 2026-05-03T09:05:41Z | `smoke`     | 0    | ok      | 8           | 15/15     | — | — | `benchmarks/results/runs/20260503-090519_f5d4c9/` |

"""


class BuildLabOptimizationDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dashboard = _load_dashboard_module()

    def test_parse_run_history_sorts_desc(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as fh:
            fh.write(_SAMPLE_HISTORY)
            fh.flush()
            hist_path = Path(fh.name)
        try:
            rows = self.dashboard._parse_run_history(hist_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].subcommand, "smoke")
            self.assertEqual(rows[0].utc_iso, "2026-05-03T09:05:41Z")
            self.assertEqual(rows[1].utc_iso, "2026-04-26T22:09:08Z")
        finally:
            hist_path.unlink()

    def test_parse_run_history_escapes_separator_rows(self) -> None:
        text = "| UTC ISO | dummy |\n|--|--|\n"
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as fh:
            fh.write(text)
            hist_path = Path(fh.name)
        try:
            rows = self.dashboard._parse_run_history(hist_path)
            self.assertEqual(rows, [])
        finally:
            hist_path.unlink()

    def test_read_jsonl_newest_first(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"slot": "older"}) + "\n")
            fh.write(json.dumps({"slot": "newer"}) + "\n")
            fh.flush()
            path = Path(fh.name)
        try:
            rows = self.dashboard._read_jsonl(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["slot"], "newer")
            self.assertEqual(rows[1]["slot"], "older")
        finally:
            path.unlink()

    def test_read_jsonl_skips_invalid_rows(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
            fh.write("{not-json\n")
            fh.write(json.dumps({"ok": True}) + "\n")
            fh.flush()
            path = Path(fh.name)
        try:
            rows = self.dashboard._read_jsonl(path)
            self.assertEqual(rows, [{"ok": True}])
        finally:
            path.unlink()

    def test_sum_cursor_usd(self) -> None:
        payload = [{"usd": 1.25}, {"note": "no usd"}, {"usd": "2"}]
        total, cnt = self.dashboard._sum_cursor_usd(payload)
        self.assertAlmostEqual(total, 3.25)
        self.assertEqual(cnt, 2)

    def test_render_html_reflects_fixture_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp) / "run_history.md"
            curs = Path(tmp) / "cursor_usage.jsonl"
            ag = Path(tmp) / "agent_events.jsonl"
            hist.write_text(_SAMPLE_HISTORY, encoding="utf-8")
            curs.write_text(
                json.dumps({"ts": "2026-05-03T09:06:59Z", "usd": 1.5, "note": "fixture"})
                + "\n"
                + json.dumps({"ts": "2026-05-04T10:06:59Z", "usd": -0.5, "category": "refund"})
                + "\n",
                encoding="utf-8",
            )
            ag.write_text(
                json.dumps(
                    {
                        "kind": "ml_workflow_learning",
                        "exit_code": 0,
                        "est_cursor_usd_spared": 0.25,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ns = argparse.Namespace(history=hist, max_history_rows=10)
            html_text = self.dashboard._render_html(
                ns,
                cursor_usage_path=curs,
                agent_events_path=ag,
            )

        compact = "".join(html_text.split())
        self.assertRegex(html_text.lower(), r"<code>\s*smoke\s*</code>")
        self.assertIn("$1.00", compact)
        self.assertIn("$0.25", compact)
        self.assertIn("Heuristic lab-runner savings", html_text)


class FeMlLabRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _load_runner_module()

    def test_cmd_learning_invokes_ml_workflow(self) -> None:
        tmp_repo = tempfile.TemporaryDirectory()
        root = Path(tmp_repo.name)
        (root / "scripts").mkdir(parents=True)
        fake_ml = root / "scripts" / "ml_workflow.py"
        fake_ml.write_text("# stub\n", encoding="utf-8")
        try:
            with mock.patch.object(self.runner, "REPO", root):
                with mock.patch.object(self.runner, "_append_event") as mocked_event:
                    with mock.patch.object(
                        self.runner.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=3),
                    ) as mocked_run:
                        rc = self.runner.cmd_learning(["--sequence", "smoke"])
            self.assertEqual(rc, 3)
            mocked_run.assert_called_once()
            argv = mocked_run.call_args.args[0]
            self.assertTrue(str(fake_ml.resolve()) == argv[-2] or argv[-2].endswith("ml_workflow.py"))
            self.assertEqual(argv[-1], "smoke")
            mocked_event.assert_called_once()
            event = mocked_event.call_args.args[1]
            self.assertEqual(event["sequence"], "smoke")
            self.assertEqual(event["exit_code"], 3)

        finally:
            tmp_repo.cleanup()

    def test_cmd_learning_skip_event_when_flag(self) -> None:
        tmp_repo = tempfile.TemporaryDirectory()
        root = Path(tmp_repo.name)
        (root / "scripts").mkdir(parents=True)
        fake_ml = root / "scripts" / "ml_workflow.py"
        fake_ml.write_text("# stub\n", encoding="utf-8")
        try:
            with mock.patch.object(self.runner, "REPO", root):
                with mock.patch.object(self.runner, "_append_event") as mocked_event:
                    with mock.patch.object(
                        self.runner.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=0),
                    ):
                        rc = self.runner.cmd_learning(["--sequence", "full", "--no-event"])
            self.assertEqual(rc, 0)
            mocked_event.assert_not_called()
        finally:
            tmp_repo.cleanup()

    def test_cmd_documentation_rag_benchmark_invokes_ml_workflow(self) -> None:
        tmp_repo = tempfile.TemporaryDirectory()
        root = Path(tmp_repo.name)
        (root / "scripts").mkdir(parents=True)
        fake_ml = root / "scripts" / "ml_workflow.py"
        fake_ml.write_text("# stub\n", encoding="utf-8")
        try:
            with mock.patch.object(self.runner, "REPO", root):
                with mock.patch.object(self.runner, "_append_event") as mocked_event:
                    with mock.patch.object(
                        self.runner.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=0),
                    ) as mocked_run:
                        rc = self.runner.cmd_documentation_rag_benchmark(
                            ["--skip-no-rag-baseline"]
                        )
            self.assertEqual(rc, 0)
            mocked_run.assert_called_once()
            argv = mocked_run.call_args.args[0]
            self.assertIn("documentation-rag-benchmark", argv)
            self.assertEqual(mocked_event.call_args.args[1]["kind"], "documentation_rag_benchmark")
        finally:
            tmp_repo.cleanup()
