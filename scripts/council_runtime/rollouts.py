"""Group council trace rows into task-level rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .traces import (
    GRADER_TRACE_SCHEMA,
    PLANNER_TRACE_SCHEMA,
    SPECIALIST_EQ_TRACE_SCHEMA,
    CouncilTraceRecord,
)


@dataclass(frozen=True)
class CouncilRollout:
    rollout_id: str
    season_id: str
    task_id: str
    planner_trace: CouncilTraceRecord | None = None
    specialist_traces: list[CouncilTraceRecord] = field(default_factory=list)
    grader_preview_traces: list[CouncilTraceRecord] = field(default_factory=list)
    final_grader_trace: CouncilTraceRecord | None = None
    other_traces: list[CouncilTraceRecord] = field(default_factory=list)

    @property
    def traces(self) -> list[CouncilTraceRecord]:
        rows: list[CouncilTraceRecord] = []
        if self.planner_trace is not None:
            rows.append(self.planner_trace)
        rows.extend(self.specialist_traces)
        rows.extend(self.grader_preview_traces)
        if self.final_grader_trace is not None:
            rows.append(self.final_grader_trace)
        rows.extend(self.other_traces)
        return rows


def group_traces_into_rollouts(records: Iterable[CouncilTraceRecord]) -> list[CouncilRollout]:
    buckets: dict[tuple[str, str], list[CouncilTraceRecord]] = {}
    for record in records:
        key = (record.season_id or "unassigned_season", record.task_id or "unassigned_task")
        buckets.setdefault(key, []).append(record)

    rollouts: list[CouncilRollout] = []
    for (season_id, task_id), rows in sorted(buckets.items()):
        planner = None
        specialists: list[CouncilTraceRecord] = []
        grader_previews: list[CouncilTraceRecord] = []
        final_grader = None
        other: list[CouncilTraceRecord] = []
        for record in rows:
            if record.trace_schema == PLANNER_TRACE_SCHEMA:
                if planner is None:
                    planner = record
                else:
                    other.append(record)
            elif record.trace_schema == SPECIALIST_EQ_TRACE_SCHEMA:
                specialists.append(record)
            elif record.trace_schema == GRADER_TRACE_SCHEMA and record.event_type == "grader_final":
                final_grader = record
            elif record.trace_schema == GRADER_TRACE_SCHEMA:
                grader_previews.append(record)
            else:
                other.append(record)
        rid = f"{season_id}:{task_id}"
        rollouts.append(
            CouncilRollout(
                rollout_id=rid,
                season_id=season_id,
                task_id=task_id,
                planner_trace=planner,
                specialist_traces=sorted(specialists, key=lambda row: (row.round_idx, row.trace_id)),
                grader_preview_traces=sorted(grader_previews, key=lambda row: (row.round_idx, row.trace_id)),
                final_grader_trace=final_grader,
                other_traces=other,
            )
        )
    return rollouts
