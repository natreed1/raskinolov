#!/usr/bin/env python3
"""Deterministic lexical retrieval for documentation-agent context."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "data" / "rag" / "documentation_agent_corpus.json"
TOKEN_RE = re.compile(r"[a-zA-Z0-9_./*-]+")


@dataclass(frozen=True)
class Source:
    path: str
    kind: str
    weight: float
    text: str


@dataclass(frozen=True)
class Chunk:
    path: str
    kind: str
    weight: float
    index: int
    text: str
    score: float = 0.0


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _resolve_repo_path(path: str) -> Path:
    candidate = (REPO / path).resolve()
    try:
        candidate.relative_to(REPO.resolve())
    except ValueError as exc:
        raise ValueError(f"Corpus path escapes repo: {path}") from exc
    return candidate


def load_corpus(
    path: Path = DEFAULT_CORPUS, *, allowed_schemas: tuple[str, ...] = ("documentation_agent_corpus_v1",)
) -> list[Source]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in allowed_schemas:
        raise ValueError(f"Unsupported corpus schema: {payload.get('schema_version')}")
    sources: list[Source] = []
    for row in payload.get("sources", []):
        rel = str(row["path"])
        source_path = _resolve_repo_path(rel)
        sources.append(
            Source(
                path=rel,
                kind=str(row.get("kind") or "document"),
                weight=float(row.get("weight") or 1.0),
                text=source_path.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return sources


def chunk_source(source: Source, *, chunk_chars: int = 1400, overlap_chars: int = 180) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source.text) if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        next_text = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(next_text) <= chunk_chars:
            current = next_text
            continue
        if current:
            chunks.append(Chunk(source.path, source.kind, source.weight, len(chunks), current))
        if len(paragraph) <= chunk_chars:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                part = paragraph[start : start + chunk_chars]
                chunks.append(Chunk(source.path, source.kind, source.weight, len(chunks), part.strip()))
                start += max(1, chunk_chars - overlap_chars)
            current = ""
    if current:
        chunks.append(Chunk(source.path, source.kind, source.weight, len(chunks), current))
    return chunks


def _all_chunks(corpus: Iterable[Source]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in corpus:
        chunks.extend(chunk_source(source))
    return chunks


def retrieve(query: str, corpus: list[Source], *, top_k: int = 6) -> list[Chunk]:
    chunks = _all_chunks(corpus)
    query_terms = _tokens(query)
    if not query_terms:
        return []
    query_counts = {term: query_terms.count(term) for term in set(query_terms)}
    doc_freq: dict[str, int] = {}
    chunk_tokens = []
    for chunk in chunks:
        terms = set(_tokens(f"{chunk.path} {chunk.kind} {chunk.text}"))
        chunk_tokens.append(terms)
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scored: list[Chunk] = []
    total_chunks = max(1, len(chunks))
    for chunk, terms in zip(chunks, chunk_tokens):
        score = 0.0
        weighted_text = f"{chunk.path} {chunk.kind} {chunk.text}"
        term_list = _tokens(weighted_text)
        for term, q_count in query_counts.items():
            tf = term_list.count(term)
            if tf == 0:
                continue
            idf = math.log((1 + total_chunks) / (1 + doc_freq.get(term, 0))) + 1.0
            score += (1.0 + math.log(tf)) * idf * q_count
        if score > 0:
            scored.append(
                Chunk(
                    path=chunk.path,
                    kind=chunk.kind,
                    weight=chunk.weight,
                    index=chunk.index,
                    text=chunk.text,
                    score=round(score * chunk.weight, 4),
                )
            )
    scored.sort(key=lambda hit: (hit.score, hit.weight, hit.path), reverse=True)
    return scored[:top_k]


def build_context(hits: list[Chunk], *, max_chars: int = 6000) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        header = f"[source: {hit.path}#{hit.index} kind={hit.kind} score={hit.score}]"
        body = hit.text.strip()
        block = f"{header}\n{body}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining < len(header) + 80:
                break
            block = f"{header}\n{body[: remaining - len(header) - 6].rstrip()}..."
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve documentation-agent context.")
    parser.add_argument("query")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()

    hits = retrieve(args.query, load_corpus(args.corpus), top_k=args.top_k)
    print(build_context(hits, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
