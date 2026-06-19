"""Population state update operators for coevolution seasons."""

from __future__ import annotations

from dataclasses import replace

from .coevolution import ArchetypePopulation, CoevolutionState, LoRAOrganism, utc_iso
from .fitness import PopulationFitnessReport
from .pbt_selection import PBTSelection, PBTSelectionConfig, select_all_populations_for_pbt
from .train_requests import CouncilLoRATrainRequest

VALID_POPULATION_UPDATE_MODES = {"score_only", "shadow_select", "pbt_mutate"}


class PopulationUpdater:
    """Apply fitness reports to coevolution state without training adapters."""

    def update(
        self,
        *,
        state: CoevolutionState,
        fitness_report: PopulationFitnessReport,
        mode: str = "score_only",
        train_requests_by_parent: dict[str, CouncilLoRATrainRequest] | None = None,
        pbt_selection_config: PBTSelectionConfig | None = None,
        pbt_selections: dict[str, PBTSelection] | None = None,
    ) -> CoevolutionState:
        if mode not in VALID_POPULATION_UPDATE_MODES:
            raise ValueError(f"Unsupported population update mode: {mode}")

        fitness_by_id = fitness_report.by_organism
        train_requests_by_parent = dict(train_requests_by_parent or {})
        if pbt_selections is None and mode == "pbt_mutate":
            pbt_selections = select_all_populations_for_pbt(
                populations=state.archetype_populations,
                fitness_report=fitness_report,
                config=pbt_selection_config,
            )
        pbt_selections = dict(pbt_selections or {})
        populations: list[ArchetypePopulation] = []
        for population in state.archetype_populations:
            updated_organisms: list[LoRAOrganism] = []
            existing_ids = {organism.organism_id for organism in population.organisms}
            selection = pbt_selections.get(population.archetype_id)
            parent_ids = set(selection.parent_ids if selection else [])
            retired_ids = set(selection.retired_ids if selection else [])
            for organism in population.organisms:
                fitness = fitness_by_id.get(organism.organism_id)
                if fitness is None and organism.organism_id not in retired_ids:
                    updated_organisms.append(organism)
                    continue
                merged = dict(organism.fitness)
                if fitness is not None:
                    merged.update(
                        {
                            "overall": fitness.mean_reward,
                            "rollout_reward": fitness.mean_reward,
                            "rollout_count": float(fitness.rollout_count),
                            "min_reward": fitness.min_reward,
                            "max_reward": fitness.max_reward,
                        }
                    )
                state_name = organism.state
                if mode == "shadow_select" and fitness is not None and fitness.rollout_count > 0:
                    state_name = "elite"
                if mode == "pbt_mutate":
                    if organism.organism_id in retired_ids:
                        state_name = "retired"
                    elif organism.organism_id in parent_ids:
                        state_name = "elite"
                updated_organisms.append(replace(organism, fitness=merged, state=state_name))
                request = train_requests_by_parent.get(organism.organism_id)
                if (
                    mode == "pbt_mutate"
                    and organism.organism_id in parent_ids
                    and request is not None
                    and request.candidate_organism_id not in existing_ids
                ):
                    child = organism.child(
                        organism_id=request.candidate_organism_id,
                        adapter_path=request.candidate_adapter_path,
                        reward_matrix_id=organism.reward_matrix_id,
                        training_recipe={
                            "kind": "pbt_mutation_train_request",
                            "train_request_path": request.request_path,
                            "parent_organism_id": organism.organism_id,
                            "season_id": request.season_id,
                            "training_objective": request.training_objective,
                        },
                    )
                    updated_organisms.append(child)
                    existing_ids.add(child.organism_id)
            populations.append(
                ArchetypePopulation(
                    archetype_id=population.archetype_id,
                    organisms=updated_organisms,
                    active_organism_id=population.active_organism_id,
                    generation=population.generation,
                    elite_fraction=population.elite_fraction,
                    mutation_count=population.mutation_count,
                )
            )

        return CoevolutionState(
            reward_population=state.reward_population,
            archetype_populations=populations,
            created_or_updated_utc=utc_iso(),
        )
