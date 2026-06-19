"""Reward scoring operator for council trace JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .coevolution import RewardMatrix, write_json
from .traces import CouncilTraceRecord

SCORED_TRACE_SCHEMA = "council_scored_trace_v1"
TRACE_SCORE_SUMMARY_SCHEMA = "council_trace_score_summary_v1"


@dataclass(frozen=True)
class ScoredTrace:
    trace: CouncilTraceRecord
    reward_matrix_id: str
    reward_score: float
    metric_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCORED_TRACE_SCHEMA,
            "trace_id": self.trace.trace_id,
            "trace_schema": self.trace.trace_schema,
            "event_type": self.trace.event_type,
            "archetype_id": self.trace.archetype_id,
            "organism_id": self.trace.organism_id,
            "task_id": self.trace.task_id,
            "season_id": self.trace.season_id,
            "round_idx": self.trace.round_idx,
            "reward_matrix_id": self.reward_matrix_id,
            "reward_score": float(self.reward_score),
            "metric_scores": {k: float(v) for k, v in sorted(self.metric_scores.items())},
            "trace": self.trace.to_dict(),
        }


@dataclass
class TraceScoreSummary:
    reward_matrix_id: str
    trace_count: int
    overall_mean: float
    by_archetype: dict[str, dict[str, float]] = field(default_factory=dict)
    by_organism: dict[str, dict[str, float]] = field(default_factory=dict)
    output_path: str = ""
    scored_trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCORE_SUMMARY_SCHEMA,
            "reward_matrix_id": self.reward_matrix_id,
            "trace_count": int(self.trace_count),
            "overall_mean": float(self.overall_mean),
            "by_archetype": self.by_archetype,
            "by_organism": self.by_organism,
            "output_path": self.output_path,
            "scored_trace_path": self.scored_trace_path,
        }


def read_trace_jsonl(path: Path) -> list[CouncilTraceRecord]:
    records: list[CouncilTraceRecord] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                records.append(CouncilTraceRecord.from_dict(payload))
            except Exception as exc:  # noqa: BLE001 - include file/line context for batch jobs.
                raise ValueError(f"Invalid trace JSONL row {path}:{line_no}: {exc}") from exc
    return records


def write_scored_jsonl(path: Path, scored: Iterable[ScoredTrace]) -> None:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def score_trace_records(records: Iterable[CouncilTraceRecord], reward_matrix: RewardMatrix) -> list[ScoredTrace]:
    scored: list[ScoredTrace] = []
    for record in records:
        metric_scores = dict(record.metrics)
        reward_score = reward_matrix.score(metric_scores)
        scored.append(
            ScoredTrace(
                trace=record,
                reward_matrix_id=reward_matrix.matrix_id,
                reward_score=reward_score,
                metric_scores=metric_scores,
            )
        )
    return scored


def _aggregate(rows: Iterable[ScoredTrace], key_name: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[ScoredTrace]] = {}
    for row in rows:
        key = str(getattr(row.trace, key_name) or "unassigned")
        buckets.setdefault(key, []).append(row)

    out: dict[str, dict[str, float]] = {}
    for key, bucket in sorted(buckets.items()):
        scores = [row.reward_score for row in bucket]
        out[key] = {
            "trace_count": float(len(bucket)),
            "fitness": round(sum(scores) / max(1, len(scores)), 6),
            "min_score": round(min(scores), 6),
            "max_score": round(max(scores), 6),
        }
    return out


def summarize_scores(
    *,
    scored: list[ScoredTrace],
    reward_matrix: RewardMatrix,
    summary_path: Path | None = None,
    scored_trace_path: Path | None = None,
) -> TraceScoreSummary:
    scores = [row.reward_score for row in scored]
    return TraceScoreSummary(
        reward_matrix_id=reward_matrix.matrix_id,
        trace_count=len(scored),
        overall_mean=round(sum(scores) / max(1, len(scores)), 6),
        by_archetype=_aggregate(scored, "archetype_id"),
        by_organism=_aggregate(scored, "organism_id"),
        output_path=str(summary_path.expanduser().resolve()) if summary_path else "",
        scored_trace_path=str(scored_trace_path.expanduser().resolve()) if scored_trace_path else "",
    )


def score_trace_file(
    *,
    trace_path: Path,
    reward_matrix: RewardMatrix,
    scored_trace_path: Path,
    summary_path: Path,
) -> TraceScoreSummary:
    records = read_trace_jsonl(trace_path)
    scored = score_trace_records(records, reward_matrix)
    write_scored_jsonl(scored_trace_path, scored)
    summary = summarize_scores(
        scored=scored,
        reward_matrix=reward_matrix,
        summary_path=summary_path,
        scored_trace_path=scored_trace_path,
    )
    write_json(summary_path, summary.to_dict())
    return summary
