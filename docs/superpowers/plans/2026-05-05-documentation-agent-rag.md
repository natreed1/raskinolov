# Documentation Agent RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable, repo-local retrieval layer for the 7B documentation/testing agent and benchmark it with and without retrieved context.

**Architecture:** Create one canonical practices document, one corpus manifest, a lightweight lexical retriever, and a benchmark runner that can inject retrieved snippets before model generation. Keep retrieval deterministic and file-based so it works before any vector database exists.

**Tech Stack:** Python `unittest`, standard-library JSON/path handling, existing `mlx_lm` model runner patterns, existing benchmark substring rubric style.

---

### Task 1: RAG Corpus And Retriever

**Files:**
- Create: `docs/DOCUMENTATION_AGENT_PRACTICES.md`
- Create: `data/rag/documentation_agent_corpus.json`
- Create: `scripts/documentation_rag.py`
- Test: `tests/test_documentation_rag.py`

- [x] Write tests that require corpus loading, retrieval hits, and bounded citation context.
- [x] Verify tests fail before implementation.
- [x] Add the practices doc and corpus manifest.
- [x] Implement deterministic lexical retrieval.
- [x] Verify tests pass.

### Task 2: Documentation-Agent Model Benchmark

**Files:**
- Create: `benchmarks/documentation_agent_rag_tasks_v1.json`
- Create: `scripts/run_documentation_agent_benchmark.py`
- Modify: `benchmarks/README.md`
- Test: `tests/test_documentation_rag.py`

- [x] Write tests that require the benchmark fixture and runner.
- [x] Verify tests fail before implementation.
- [x] Add benchmark tasks covering run manifests, append-only docs, committed fixtures, generated artifacts, and verification discipline.
- [x] Implement a benchmark runner that loads the 7B model, optionally injects RAG context, and scores string expectations.
- [x] Verify unit tests pass.

### Task 3: Local 7B Verification

**Commands:**
- `python3 -m unittest tests/test_documentation_rag.py -v`
- `PYTHONPATH=scripts python3 scripts/run_documentation_agent_benchmark.py --tasks benchmarks/documentation_agent_rag_tasks_v1.json --use-rag --output-jsonl benchmarks/results/documentation_agent_rag_7b.jsonl`
- `python3 -m unittest discover -s tests -v`

- [x] Run the focused unit tests.
- [x] Run the local 7B documentation-agent benchmark.
- [x] Run full Python test discovery.
