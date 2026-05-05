"""Tests for run-analysis RAG corpus and retrieval fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "rag" / "run_analysis_agent_corpus.json"
TASKS = REPO / "benchmarks" / "run_analysis_rag_tasks_v1.json"
RUNNER = REPO / "scripts" / "run_run_analysis_agent_benchmark.py"


class RunAnalysisRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_corpus_manifest_points_to_existing_sources(self) -> None:
        self.assertTrue(CORPUS.is_file())
        payload = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "run_analysis_agent_corpus_v1")
        paths = [row["path"] for row in payload["sources"]]
        self.assertIn("docs/run_history.md", paths)
        self.assertIn("docs/SPECIALIZED_RUN_HISTORY.md", paths)
        for path in paths:
            self.assertTrue((REPO / path).is_file(), path)

    def test_retriever_returns_manifest_and_history_context(self) -> None:
        from run_analysis_rag import build_context, load_corpus, retrieve

        corpus = load_corpus(CORPUS)
        hits = retrieve(
            "Where should I inspect run_id machine details and append-only history?",
            corpus,
            top_k=6,
        )
        self.assertGreaterEqual(len(hits), 2)
        paths = {hit.path for hit in hits}
        self.assertTrue(
            any(p.endswith("manifest.json") for p in paths)
            or "docs/run_history.md" in paths
            or "docs/SPECIALIZED_RUN_HISTORY.md" in paths
        )
        context = build_context(hits, max_chars=2000)
        self.assertLessEqual(len(context), 2000)
        self.assertIn("[source:", context)

    def test_benchmark_fixture_shape(self) -> None:
        self.assertTrue(TASKS.is_file())
        rows = json.loads(TASKS.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertIn("id", row)
            self.assertIn("prompt", row)
            self.assertIn("expect", row)
            self.assertEqual(row.get("category"), "run_analysis_rag")

    def test_runner_dry_run_retrieval(self) -> None:
        self.assertTrue(RUNNER.is_file())
        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--tasks",
                str(TASKS),
                "--use-rag",
                "--dry-run-retrieval",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("=== Retrieval coverage:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
