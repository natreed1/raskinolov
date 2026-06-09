#!/usr/bin/env python3
"""Router-focused RAG retrieval using router corpus sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from documentation_rag import build_context, load_corpus, retrieve

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO / "data" / "rag" / "router_agent_corpus.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve router RAG context.")
    parser.add_argument("query")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()
    corpus = load_corpus(
        args.corpus,
        allowed_schemas=("router_agent_corpus_v1", "documentation_agent_corpus_v1"),
    )
    hits = retrieve(args.query, corpus, top_k=args.top_k)
    print(build_context(hits, max_chars=args.max_chars))


if __name__ == "__main__":
    main()

