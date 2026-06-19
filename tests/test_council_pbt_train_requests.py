"""Tests for council PBT train requests and adapter backend wrappers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


class CouncilPBTTrainRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO / "scripts"))

    def _state_and_fitness(self):
        from council_runtime.archetypes import CouncilArchetypeRegistry
        from council_runtime.coevolution import bootstrap_state
        from council_runtime.fitness import FitnessAggregator
        from council_runtime.specialist_rewards import SpecialistRewardResult

        state = bootstrap_state(CouncilArchetypeRegistry.bootstrap(specialist_adapter_ids={"save_load"}))
        row = SpecialistRewardResult(
            season_id="season_pbt",
            task_id="task_1",
            rollout_id="season_pbt:task_1",
            archetype_id="save_load_eq",
            organism_id="",
            participant_id="save_load::guard",
            base_expert_id="save_load",
            reward=0.7,
            positive_score=0.8,
            penalty_score=0.1,
            components={"assigned_role_fit": 0.8},
            diagnostics=[],
            training_prompt="Check save/load migration risk.",
            output="Verify save schema and run persistence tests.",
        )
        report = FitnessAggregator().aggregate_specialist(rows=[row], state=state)
        return state, report, row

    def test_train_request_contains_candidate_and_ppo_rows(self) -> None:
        from council_runtime.train_requests import build_train_requests, specialist_rewards_to_ppo_scored_rows

        state, report, row = self._state_and_fitness()
        parent_id = state.archetypes_by_id()["save_load_eq"].active_organism_id

        with tempfile.TemporaryDirectory() as tmp:
            requests = build_train_requests(
                state=state,
                fitness_report=report,
                season_id="season_pbt",
                request_root=Path(tmp) / "requests",
                reward_source={"specialist_rewards": "/tmp/specialist_rewards.jsonl"},
            )
            request = requests[parent_id]
            ppo_rows = specialist_rewards_to_ppo_scored_rows([row])

            self.assertEqual(request.parent_organism_id, parent_id)
            self.assertIn("candidates/season_pbt", request.candidate_adapter_path)
            self.assertTrue(Path(request.request_path).is_file())
            self.assertEqual(ppo_rows[0]["score"]["reward"], row.reward)
            self.assertEqual(ppo_rows[0]["rollout"]["output"], row.output)

    def test_pbt_mutate_adds_candidate_child_without_changing_active(self) -> None:
        from council_runtime.population_update import PopulationUpdater
        from council_runtime.train_requests import build_train_requests

        state, report, _ = self._state_and_fitness()
        parent_id = state.archetypes_by_id()["save_load_eq"].active_organism_id

        with tempfile.TemporaryDirectory() as tmp:
            requests = build_train_requests(
                state=state,
                fitness_report=report,
                season_id="season_pbt",
                request_root=Path(tmp) / "requests",
                reward_source={},
            )
            updated = PopulationUpdater().update(
                state=state,
                fitness_report=report,
                mode="pbt_mutate",
                train_requests_by_parent=requests,
            )
            population = updated.archetypes_by_id()["save_load_eq"]

            self.assertEqual(population.active_organism_id, parent_id)
            self.assertGreater(len(population.organisms), 1)
            child = [organism for organism in population.organisms if organism.organism_id != parent_id][0]
            self.assertEqual(child.state, "candidate")
            self.assertIn("train_request_path", child.training_recipe)
            self.assertIn("candidates/season_pbt", child.adapter_path)

    def test_pbt_selection_rounds_top_and_bottom_percentages_up(self) -> None:
        from council_runtime.coevolution import (
            ArchetypePopulation,
            CoevolutionState,
            LoRAOrganism,
            RewardMatrix,
            RewardPopulation,
            normalize_weights,
        )
        from council_runtime.fitness import OrganismFitness, PopulationFitnessReport
        from council_runtime.pbt_selection import select_population_for_pbt
        from council_runtime.population_update import PopulationUpdater
        from council_runtime.train_requests import build_train_requests

        organisms = [
            LoRAOrganism(
                organism_id=f"planner.g000.lineage_{idx:02d}",
                archetype_id="planner",
                adapter_path=f"checkpoints/adapters/planner/lineage_{idx:02d}",
                state="candidate",
            )
            for idx in range(12)
        ]
        population = ArchetypePopulation(
            archetype_id="planner",
            organisms=organisms,
            active_organism_id=organisms[0].organism_id,
        )
        reward_population = RewardPopulation(
            matrices=[
                RewardMatrix(
                    matrix_id="reward_test",
                    description="test",
                    weights=normalize_weights({"final_outcome": 1.0}),
                    state="champion",
                )
            ],
            active_matrix_id="reward_test",
        )
        state = CoevolutionState(
            reward_population=reward_population,
            archetype_populations=[population],
        )
        report = PopulationFitnessReport(
            season_id="season_pbt",
            by_organism={
                organism.organism_id: OrganismFitness(
                    archetype_id="planner",
                    organism_id=organism.organism_id,
                    rollout_count=3,
                    mean_reward=idx / 11,
                    min_reward=idx / 11,
                    max_reward=idx / 11,
                )
                for idx, organism in enumerate(organisms)
            },
        )

        selection = select_population_for_pbt(population=population, fitness_report=report)

        self.assertEqual(selection.parent_ids, [organisms[11].organism_id, organisms[10].organism_id])
        self.assertEqual(selection.retired_ids, [organisms[0].organism_id, organisms[1].organism_id])

        with tempfile.TemporaryDirectory() as tmp:
            requests = build_train_requests(
                state=state,
                fitness_report=report,
                season_id="season_pbt",
                request_root=Path(tmp) / "requests",
                reward_source={},
                parent_organism_ids=set(selection.parent_ids),
            )
            updated = PopulationUpdater().update(
                state=state,
                fitness_report=report,
                mode="pbt_mutate",
                train_requests_by_parent=requests,
                pbt_selections={"planner": selection},
            )

        updated_population = updated.archetypes_by_id()["planner"]
        by_id = updated_population.by_id()
        self.assertEqual(by_id[organisms[11].organism_id].state, "elite")
        self.assertEqual(by_id[organisms[10].organism_id].state, "elite")
        self.assertEqual(by_id[organisms[0].organism_id].state, "retired")
        self.assertEqual(by_id[organisms[1].organism_id].state, "retired")
        children = [
            organism
            for organism in updated_population.organisms
            if organism.parent_ids and organism.parent_ids[0] in selection.parent_ids
        ]
        self.assertEqual(len(children), 2)

    def test_dry_run_backend_writes_manifest_without_weights(self) -> None:
        from council_runtime.adapter_training import DryRunAdapterTrainingBackend
        from council_runtime.train_requests import build_train_requests, specialist_rewards_to_ppo_scored_rows, write_ppo_scored_rows

        state, report, row = self._state_and_fitness()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = next(iter(build_train_requests(
                state=state,
                fitness_report=report,
                season_id="season_pbt",
                request_root=root / "requests",
                reward_source={},
            ).values()))
            ppo_path = root / "ppo.jsonl"
            write_ppo_scored_rows(ppo_path, specialist_rewards_to_ppo_scored_rows([row]))
            result = DryRunAdapterTrainingBackend().train(
                request=request,
                ppo_scored_rows_path=ppo_path,
                manifest_path=root / "train_manifest.json",
            )

            self.assertEqual(result.status, "dry_run_no_weight_update")
            self.assertTrue(Path(result.manifest_path).is_file())
            self.assertFalse(Path(request.candidate_adapter_path).exists())

    def test_ppo_backend_delegates_to_existing_trainer(self) -> None:
        from council_runtime.adapter_training import PPOAdapterTrainingBackend
        from council_runtime.train_requests import build_train_requests, specialist_rewards_to_ppo_scored_rows, write_ppo_scored_rows

        state, report, row = self._state_and_fitness()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = next(iter(build_train_requests(
                state=state,
                fitness_report=report,
                season_id="season_pbt",
                request_root=root / "requests",
                reward_source={},
            ).values()))
            ppo_path = root / "ppo.jsonl"
            write_ppo_scored_rows(ppo_path, specialist_rewards_to_ppo_scored_rows([row]))
            with mock.patch("economist_rl_ppo_trainer.train_ppo_batch", return_value={"status": "dry_run_no_weight_update", "candidate_adapter": request.candidate_adapter_path}) as train:
                with mock.patch("economist_rl_ppo_trainer.write_ppo_manifest") as write_manifest:
                    result = PPOAdapterTrainingBackend(dry_run=True).train(
                        request=request,
                        ppo_scored_rows_path=ppo_path,
                        manifest_path=root / "ppo_manifest.json",
                    )

            self.assertEqual(result.backend, "ppo")
            self.assertTrue(train.called)
            self.assertTrue(write_manifest.called)
            kwargs = train.call_args.kwargs
            self.assertEqual(str(kwargs["candidate_adapter"]), request.candidate_adapter_path)
            self.assertEqual(kwargs["base_model"], request.base_model)


if __name__ == "__main__":
    unittest.main()
