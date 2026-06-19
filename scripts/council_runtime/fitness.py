"""Population fitness aggregation for graded council rollouts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .coevolution import CoevolutionState, write_json
from .reward_functions import GRADED_ROLLOUT_SCHEMA, PlannerRewardResult
from .specialist_rewards import SPECIALIST_REWARD_SCHEMA, SpecialistRewardResult

POPULATION_FITNESS_SCHEMA = "council_population_fitness_v1"


@dataclass(frozen=True)
class OrganismFitness:
    archetype_id: str
    organism_id: str
    rollout_count: int
    mean_reward: float
    min_reward: float
    max_reward: float
    diagnostics: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_id": self.archetype_id,
            "organism_id": self.organism_id,
            "rollout_count": int(self.rollout_count),
            "mean_reward": round(float(self.mean_reward), 6),
            "min_reward": round(float(self.min_reward), 6),
            "max_reward": round(float(self.max_reward), 6),
            "diagnostics": {k: int(v) for k, v in sorted(self.diagnostics.items())},
        }


@dataclass(frozen=True)
class PopulationFitnessReport:
    season_id: str
    by_organism: dict[str, OrganismFitness]
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POPULATION_FITNESS_SCHEMA,
            "season_id": self.season_id,
            "source_path": self.source_path,
            "by_organism": {key: value.to_dict() for key, value in sorted(self.by_organism.items())},
        }


def read_graded_rollout_jsonl(path: Path) -> list[PlannerRewardResult]:
    rows: list[PlannerRewardResult] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload.get("schema_version") != GRADED_ROLLOUT_SCHEMA:
                raise ValueError(f"Unsupported graded rollout schema at {path}:{line_no}: {payload.get('schema_version')!r}")
            rows.append(
                PlannerRewardResult(
                    rollout_id=str(payload.get("rollout_id") or ""),
                    task_id=str(payload.get("task_id") or ""),
                    season_id=str(payload.get("season_id") or ""),
                    reward=float(payload.get("reward") or 0.0),
                    positive_score=float(payload.get("positive_score") or 0.0),
                    penalty_score=float(payload.get("penalty_score") or 0.0),
                    components={str(k): float(v) for k, v in dict(payload.get("components") or {}).items()},
                    diagnostics=[str(x) for x in list(payload.get("diagnostics") or [])],
                    planner_trace_id=str(payload.get("planner_trace_id") or ""),
                )
            )
    return rows


def read_specialist_reward_jsonl(path: Path) -> list[SpecialistRewardResult]:
    rows: list[SpecialistRewardResult] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload.get("schema_version") != SPECIALIST_REWARD_SCHEMA:
                raise ValueError(f"Unsupported specialist reward schema at {path}:{line_no}: {payload.get('schema_version')!r}")
            rows.append(SpecialistRewardResult.from_dict(payload))
    return rows


def _active_organism_id(state: CoevolutionState, archetype_id: str) -> str:
    population = state.archetypes_by_id().get(archetype_id)
    return population.active_organism_id if population else f"{archetype_id}.unassigned"


class FitnessAggregator:
    """Map graded rollout rewards onto population organisms."""

    def aggregate_planner(
        self,
        *,
        rows: Iterable[PlannerRewardResult],
        state: CoevolutionState,
        source_path: Path | None = None,
    ) -> PopulationFitnessReport:
        buckets: dict[str, list[PlannerRewardResult]] = {}
        rows_list = list(rows)
        season_id = ""
        for row in rows_list:
            season_id = season_id or row.season_id
            organism_id = _active_organism_id(state, "planner")
            buckets.setdefault(organism_id, []).append(row)

        by_organism: dict[str, OrganismFitness] = {}
        for organism_id, bucket in sorted(buckets.items()):
            rewards = [float(row.reward) for row in bucket]
            diagnostics = Counter()
            for row in bucket:
                diagnostics.update(row.diagnostics)
            by_organism[organism_id] = OrganismFitness(
                archetype_id="planner",
                organism_id=organism_id,
                rollout_count=len(bucket),
                mean_reward=sum(rewards) / max(1, len(rewards)),
                min_reward=min(rewards) if rewards else 0.0,
                max_reward=max(rewards) if rewards else 0.0,
                diagnostics=dict(diagnostics),
            )

        return PopulationFitnessReport(
            season_id=season_id or "unassigned_season",
            by_organism=by_organism,
            source_path=str(source_path.expanduser().resolve()) if source_path else "",
        )

    def aggregate_specialist(
        self,
        *,
        rows: Iterable[SpecialistRewardResult],
        state: CoevolutionState,
        source_path: Path | None = None,
    ) -> PopulationFitnessReport:
        buckets: dict[str, list[SpecialistRewardResult]] = {}
        archetype_by_organism: dict[str, str] = {}
        rows_list = list(rows)
        season_id = ""
        for row in rows_list:
            season_id = season_id or row.season_id
            archetype_id = row.archetype_id or (f"{row.base_expert_id}_eq" if row.base_expert_id else "specialist_eq")
            organism_id = row.organism_id or _active_organism_id(state, archetype_id)
            buckets.setdefault(organism_id, []).append(row)
            archetype_by_organism[organism_id] = archetype_id

        by_organism: dict[str, OrganismFitness] = {}
        for organism_id, bucket in sorted(buckets.items()):
            rewards = [float(row.reward) for row in bucket]
            diagnostics = Counter()
            for row in bucket:
                diagnostics.update(row.diagnostics)
            by_organism[organism_id] = OrganismFitness(
                archetype_id=archetype_by_organism.get(organism_id, "specialist_eq"),
                organism_id=organism_id,
                rollout_count=len(bucket),
                mean_reward=sum(rewards) / max(1, len(rewards)),
                min_reward=min(rewards) if rewards else 0.0,
                max_reward=max(rewards) if rewards else 0.0,
                diagnostics=dict(diagnostics),
            )

        return PopulationFitnessReport(
            season_id=season_id or "unassigned_season",
            by_organism=by_organism,
            source_path=str(source_path.expanduser().resolve()) if source_path else "",
        )

    def merge_reports(
        self,
        *,
        reports: Iterable[PopulationFitnessReport],
        source_path: Path | None = None,
    ) -> PopulationFitnessReport:
        merged: dict[str, OrganismFitness] = {}
        season_id = ""
        for report in reports:
            season_id = season_id or report.season_id
            for organism_id, fitness in report.by_organism.items():
                if organism_id not in merged:
                    merged[organism_id] = fitness
                    continue
                prev = merged[organism_id]
                total = prev.rollout_count + fitness.rollout_count
                mean = (
                    (prev.mean_reward * prev.rollout_count) + (fitness.mean_reward * fitness.rollout_count)
                ) / max(1, total)
                diagnostics = Counter(prev.diagnostics)
                diagnostics.update(fitness.diagnostics)
                merged[organism_id] = OrganismFitness(
                    archetype_id=prev.archetype_id,
                    organism_id=organism_id,
                    rollout_count=total,
                    mean_reward=mean,
                    min_reward=min(prev.min_reward, fitness.min_reward),
                    max_reward=max(prev.max_reward, fitness.max_reward),
                    diagnostics=dict(diagnostics),
                )
        return PopulationFitnessReport(
            season_id=season_id or "unassigned_season",
            by_organism=merged,
            source_path=str(source_path.expanduser().resolve()) if source_path else "",
        )


def write_fitness_report(path: Path, report: PopulationFitnessReport) -> None:
    write_json(path.expanduser().resolve(), report.to_dict())
