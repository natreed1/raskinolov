"""Population-based season selection contracts for council organisms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .coevolution import ArchetypePopulation
from .fitness import OrganismFitness, PopulationFitnessReport


@dataclass(frozen=True)
class PBTSelectionConfig:
    top_fraction: float = 0.10
    bottom_fraction: float = 0.10
    min_competitors_for_elimination: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.top_fraction) <= 1.0:
            raise ValueError("top_fraction must be between 0.0 and 1.0")
        if not 0.0 <= float(self.bottom_fraction) <= 1.0:
            raise ValueError("bottom_fraction must be between 0.0 and 1.0")
        if int(self.min_competitors_for_elimination) < 1:
            raise ValueError("min_competitors_for_elimination must be >= 1")

    def top_count(self, competitor_count: int) -> int:
        if competitor_count <= 0 or self.top_fraction <= 0:
            return 0
        return max(1, int(math.ceil(competitor_count * float(self.top_fraction))))

    def bottom_count(self, competitor_count: int, *, reserved_top_count: int = 0) -> int:
        if (
            competitor_count < int(self.min_competitors_for_elimination)
            or self.bottom_fraction <= 0
        ):
            return 0
        count = max(1, int(math.ceil(competitor_count * float(self.bottom_fraction))))
        return min(count, max(0, competitor_count - int(reserved_top_count)))


@dataclass(frozen=True)
class PBTOrganismRank:
    organism_id: str
    archetype_id: str
    reward: float
    rollout_count: int
    rank: int

    @classmethod
    def from_fitness(cls, *, fitness: OrganismFitness, rank: int) -> "PBTOrganismRank":
        return cls(
            organism_id=fitness.organism_id,
            archetype_id=fitness.archetype_id,
            reward=float(fitness.mean_reward),
            rollout_count=int(fitness.rollout_count),
            rank=int(rank),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "organism_id": self.organism_id,
            "archetype_id": self.archetype_id,
            "reward": round(float(self.reward), 6),
            "rollout_count": int(self.rollout_count),
            "rank": int(self.rank),
        }


@dataclass(frozen=True)
class PBTSelection:
    archetype_id: str
    ranked: list[PBTOrganismRank] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    retired_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "archetype_id": self.archetype_id,
            "ranked": [row.to_dict() for row in self.ranked],
            "parent_ids": list(self.parent_ids),
            "retired_ids": list(self.retired_ids),
        }


def select_population_for_pbt(
    *,
    population: ArchetypePopulation,
    fitness_report: PopulationFitnessReport,
    config: PBTSelectionConfig | None = None,
) -> PBTSelection:
    cfg = config or PBTSelectionConfig()
    by_id = population.by_id()
    eligible = [
        fitness
        for organism_id, fitness in fitness_report.by_organism.items()
        if organism_id in by_id and fitness.archetype_id == population.archetype_id and fitness.rollout_count > 0
    ]
    eligible.sort(key=lambda row: (float(row.mean_reward), int(row.rollout_count), row.organism_id), reverse=True)
    ranked = [
        PBTOrganismRank.from_fitness(fitness=fitness, rank=index)
        for index, fitness in enumerate(eligible, start=1)
    ]
    top_count = cfg.top_count(len(ranked))
    parent_ids = [row.organism_id for row in ranked[:top_count]]
    bottom_count = cfg.bottom_count(len(ranked), reserved_top_count=len(parent_ids))
    parent_set = set(parent_ids)
    retired_ids: list[str] = []
    for row in reversed(ranked):
        if len(retired_ids) >= bottom_count:
            break
        if row.organism_id in parent_set:
            continue
        retired_ids.append(row.organism_id)
    return PBTSelection(
        archetype_id=population.archetype_id,
        ranked=ranked,
        parent_ids=parent_ids,
        retired_ids=retired_ids,
    )


def select_all_populations_for_pbt(
    *,
    populations: list[ArchetypePopulation],
    fitness_report: PopulationFitnessReport,
    config: PBTSelectionConfig | None = None,
) -> dict[str, PBTSelection]:
    return {
        population.archetype_id: select_population_for_pbt(
            population=population,
            fitness_report=fitness_report,
            config=config,
        )
        for population in populations
    }
