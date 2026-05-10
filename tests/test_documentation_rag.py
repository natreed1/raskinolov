"""Tests for documentation-agent RAG corpus, retrieval, and benchmark fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "rag" / "documentation_agent_corpus.json"
PRACTICES = REPO / "docs" / "DOCUMENTATION_AGENT_PRACTICES.md"
TASKS = REPO / "benchmarks" / "documentation_agent_rag_tasks_v1.json"
RUNNER = REPO / "scripts" / "run_documentation_agent_benchmark.py"


class DocumentationRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def test_corpus_manifest_points_to_existing_sources(self) -> None:
        self.assertTrue(PRACTICES.is_file())
        self.assertTrue(CORPUS.is_file())
        payload = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "documentation_agent_corpus_v1")
        paths = [row["path"] for row in payload["sources"]]
        self.assertIn("docs/DOCUMENTATION_AGENT_PRACTICES.md", paths)
        self.assertIn("docs/PROJECT_STATE.md", paths)
        self.assertIn("docs/WORKFLOW.md", paths)
        for path in paths:
            self.assertTrue((REPO / path).is_file(), path)

    def test_retriever_returns_relevant_cited_context(self) -> None:
        from documentation_rag import build_context, load_corpus, retrieve

        corpus = load_corpus(CORPUS)
        hits = retrieve(
            "Where do run manifests live and how should docs/testing agents cite benchmark evidence?",
            corpus,
            top_k=4,
        )
        self.assertGreaterEqual(len(hits), 2)
        paths = {hit.path for hit in hits}
        self.assertIn("docs/DOCUMENTATION_AGENT_PRACTICES.md", paths)
        self.assertTrue(any("manifest" in hit.text.lower() for hit in hits))
        context = build_context(hits, max_chars=1800)
        self.assertLessEqual(len(context), 1800)
        self.assertIn("[source: docs/DOCUMENTATION_AGENT_PRACTICES.md", context)
        self.assertIn("manifest", context.lower())

    def test_benchmark_fixture_shape(self) -> None:
        self.assertTrue(TASKS.is_file())
        rows = json.loads(TASKS.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            self.assertIn("id", row)
            self.assertIn("prompt", row)
            self.assertIn("expect", row)
            self.assertTrue(row["expect"].get("all_contains"), row["id"])
            self.assertEqual(row.get("category"), "documentation_rag")

    def test_runner_dry_run_scores_retrieval_without_loading_model(self) -> None:
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
