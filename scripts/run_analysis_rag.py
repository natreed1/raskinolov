#!/usr/bin/env python3
"""Deterministic lexical retrieval for run-analysis context."""

from __future__ import annotations

import argparse
from pathlib import Path

from documentation_rag import build_context, load_corpus as _load_corpus, retrieve

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "data" / "rag" / "run_analysis_agent_corpus.json"


def load_corpus(path: Path = DEFAULT_CORPUS):
    return _load_corpus(path, allowed_schemas=("run_analysis_agent_corpus_v1",))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve run-analysis RAG context.")
    parser.add_argument("query")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()

    hits = retrieve(args.query, load_corpus(args.corpus), top_k=args.top_k)
    print(build_context(hits, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
