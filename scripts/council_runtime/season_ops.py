"""Operator-backed council coevolution season execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapter_training import DryRunAdapterTrainingBackend
from .coevolution import CoevolutionState, write_json
from .fitness import FitnessAggregator, PopulationFitnessReport, write_fitness_report
from .population_update import PopulationUpdater
from .pbt_selection import select_all_populations_for_pbt
from .reward_functions import PlannerRewardResult, grade_rollouts, summarize_graded_rollouts, write_graded_rollouts
from .rollouts import CouncilRollout, group_traces_into_rollouts
from .specialist_rewards import (
    SpecialistRewardResult,
    grade_specialist_rollouts,
    summarize_specialist_rewards,
    write_specialist_rewards,
)
from .trace_scoring import read_trace_jsonl
from .train_requests import build_train_requests, specialist_rewards_to_ppo_scored_rows, write_ppo_scored_rows


@dataclass(frozen=True)
class SeasonContext:
    season_id: str
    traces_jsonl: Path
    run_dir: Path
    state_path: Path
    mode: str = "score_only"


@dataclass(frozen=True)
class RolloutBatch:
    season_id: str
    source_path: str
    rollouts: list[CouncilRollout]


@dataclass(frozen=True)
class GradedRolloutBatch:
    season_id: str
    rows: list[PlannerRewardResult]
    graded_rollout_path: str
    summary_path: str


@dataclass(frozen=True)
class SpecialistRewardBatch:
    season_id: str
    rows: list[SpecialistRewardResult]
    reward_path: str
    summary_path: str


@dataclass(frozen=True)
class SeasonRunResult:
    season_id: str
    run_dir: str
    graded_rollout_path: str
    rollout_reward_summary_path: str
    fitness_path: str
    updated_state_path: str
    manifest_path: str
    mode: str
    specialist_reward_path: str = ""
    specialist_reward_summary_path: str = ""
    train_request_root: str = ""
    ppo_scored_rows_root: str = ""
    train_manifest_root: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "season_id": self.season_id,
            "run_dir": self.run_dir,
            "graded_rollout_path": self.graded_rollout_path,
            "rollout_reward_summary_path": self.rollout_reward_summary_path,
            "fitness_path": self.fitness_path,
            "updated_state_path": self.updated_state_path,
            "manifest_path": self.manifest_path,
            "mode": self.mode,
            "specialist_reward_path": self.specialist_reward_path,
            "specialist_reward_summary_path": self.specialist_reward_summary_path,
            "train_request_root": self.train_request_root,
            "ppo_scored_rows_root": self.ppo_scored_rows_root,
            "train_manifest_root": self.train_manifest_root,
        }


class JsonlTraceRolloutSource:
    def collect(self, context: SeasonContext) -> RolloutBatch:
        records = read_trace_jsonl(context.traces_jsonl)
        rollouts = group_traces_into_rollouts(records)
        return RolloutBatch(
            season_id=context.season_id,
            source_path=str(context.traces_jsonl.expanduser().resolve()),
            rollouts=rollouts,
        )


class PlannerRolloutGrader:
    def grade(self, *, batch: RolloutBatch, graded_path: Path, summary_path: Path) -> GradedRolloutBatch:
        rows = grade_rollouts(batch.rollouts)
        write_graded_rollouts(graded_path, rows)
        summarize_graded_rollouts(rows=rows, summary_path=summary_path, graded_rollout_path=graded_path)
        return GradedRolloutBatch(
            season_id=batch.season_id,
            rows=rows,
            graded_rollout_path=str(graded_path.expanduser().resolve()),
            summary_path=str(summary_path.expanduser().resolve()),
        )


class SpecialistRolloutGrader:
    def grade(self, *, batch: RolloutBatch, reward_path: Path, summary_path: Path) -> SpecialistRewardBatch:
        rows = grade_specialist_rollouts(batch.rollouts)
        write_specialist_rewards(reward_path, rows)
        summarize_specialist_rewards(rows=rows, summary_path=summary_path, reward_path=reward_path)
        return SpecialistRewardBatch(
            season_id=batch.season_id,
            rows=rows,
            reward_path=str(reward_path.expanduser().resolve()),
            summary_path=str(summary_path.expanduser().resolve()),
        )


class CouncilSeasonOperator:
    def __init__(
        self,
        *,
        rollout_source: JsonlTraceRolloutSource | None = None,
        rollout_grader: PlannerRolloutGrader | None = None,
        specialist_grader: SpecialistRolloutGrader | None = None,
        fitness_aggregator: FitnessAggregator | None = None,
        population_updater: PopulationUpdater | None = None,
        training_backend: DryRunAdapterTrainingBackend | None = None,
    ) -> None:
        self.rollout_source = rollout_source or JsonlTraceRolloutSource()
        self.rollout_grader = rollout_grader or PlannerRolloutGrader()
        self.specialist_grader = specialist_grader or SpecialistRolloutGrader()
        self.fitness_aggregator = fitness_aggregator or FitnessAggregator()
        self.population_updater = population_updater or PopulationUpdater()
        self.training_backend = training_backend or DryRunAdapterTrainingBackend()

    def run(self, *, context: SeasonContext, state: CoevolutionState) -> SeasonRunResult:
        run_dir = context.run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        graded_path = run_dir / "graded_rollouts.jsonl"
        reward_summary_path = run_dir / "rollout_reward_summary.json"
        specialist_reward_path = run_dir / "specialist_rewards.jsonl"
        specialist_summary_path = run_dir / "specialist_reward_summary.json"
        fitness_path = run_dir / "population_fitness.json"
        updated_state_path = run_dir / "updated_coevolution_state.json"
        manifest_path = run_dir / "SEASON_OPERATOR_MANIFEST.json"
        train_request_root = run_dir / "train_requests"
        ppo_rows_root = run_dir / "ppo_scored_rows"
        train_manifest_root = run_dir / "train_manifests"

        rollout_batch = self.rollout_source.collect(context)
        graded_batch = self.rollout_grader.grade(
            batch=rollout_batch,
            graded_path=graded_path,
            summary_path=reward_summary_path,
        )
        specialist_batch = self.specialist_grader.grade(
            batch=rollout_batch,
            reward_path=specialist_reward_path,
            summary_path=specialist_summary_path,
        )
        planner_fitness = self.fitness_aggregator.aggregate_planner(
            rows=graded_batch.rows,
            state=state,
            source_path=graded_path,
        )
        specialist_fitness = self.fitness_aggregator.aggregate_specialist(
            rows=specialist_batch.rows,
            state=state,
            source_path=specialist_reward_path,
        )
        fitness_report: PopulationFitnessReport = self.fitness_aggregator.merge_reports(
            reports=[planner_fitness, specialist_fitness],
            source_path=fitness_path,
        )
        write_fitness_report(fitness_path, fitness_report)
        train_requests_by_parent = {}
        train_results = []
        pbt_selections = {}
        if context.mode == "pbt_mutate":
            pbt_selections = select_all_populations_for_pbt(
                populations=state.archetype_populations,
                fitness_report=fitness_report,
            )
            parent_organism_ids = {
                organism_id
                for selection in pbt_selections.values()
                for organism_id in selection.parent_ids
            }
            train_requests_by_parent = build_train_requests(
                state=state,
                fitness_report=fitness_report,
                season_id=context.season_id,
                request_root=train_request_root,
                reward_source={
                    "graded_rollouts": str(graded_path),
                    "specialist_rewards": str(specialist_reward_path),
                    "population_fitness": str(fitness_path),
                },
                parent_organism_ids=parent_organism_ids,
            )
            for request in train_requests_by_parent.values():
                source_rows = [
                    row for row in specialist_batch.rows
                    if row.archetype_id == request.archetype_id or row.organism_id == request.parent_organism_id
                ]
                if not source_rows:
                    continue
                ppo_rows = specialist_rewards_to_ppo_scored_rows(source_rows)
                ppo_path = ppo_rows_root / request.archetype_id / f"{request.candidate_organism_id}.jsonl"
                write_ppo_scored_rows(ppo_path, ppo_rows)
                train_result = self.training_backend.train(
                    request=request,
                    ppo_scored_rows_path=ppo_path,
                    manifest_path=train_manifest_root / request.archetype_id / f"{request.candidate_organism_id}.json",
                )
                train_results.append(train_result.to_dict())
        updated_state = self.population_updater.update(
            state=state,
            fitness_report=fitness_report,
            mode=context.mode,
            train_requests_by_parent=train_requests_by_parent,
            pbt_selections=pbt_selections,
        )
        updated_state.save(updated_state_path)

        result = SeasonRunResult(
            season_id=context.season_id,
            run_dir=str(run_dir),
            graded_rollout_path=str(graded_path),
            rollout_reward_summary_path=str(reward_summary_path),
            fitness_path=str(fitness_path),
            updated_state_path=str(updated_state_path),
            manifest_path=str(manifest_path),
            mode=context.mode,
            specialist_reward_path=str(specialist_reward_path),
            specialist_reward_summary_path=str(specialist_summary_path),
            train_request_root=str(train_request_root),
            ppo_scored_rows_root=str(ppo_rows_root),
            train_manifest_root=str(train_manifest_root),
        )
        write_json(
            manifest_path,
            {
                "schema_version": "council_season_operator_manifest_v1",
                **result.to_dict(),
                "rollout_count": len(rollout_batch.rollouts),
                "graded_rollout_count": len(graded_batch.rows),
                "specialist_reward_count": len(specialist_batch.rows),
                "fitness_organism_count": len(fitness_report.by_organism),
                "train_request_count": len(train_requests_by_parent),
                "train_results": train_results,
                "pbt_selection": {
                    key: selection.to_dict()
                    for key, selection in sorted(pbt_selections.items())
                },
            },
        )
        return result
