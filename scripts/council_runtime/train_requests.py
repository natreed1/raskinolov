"""LoRA train request contracts for council population updates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .coevolution import CoevolutionState, LoRAOrganism, clean_id, write_json
from .fitness import PopulationFitnessReport
from .specialist_rewards import SpecialistRewardResult

COUNCIL_LORA_TRAIN_REQUEST_SCHEMA = "council_lora_train_request_v1"


@dataclass(frozen=True)
class CouncilLoRATrainRequest:
    season_id: str
    archetype_id: str
    parent_organism_id: str
    candidate_organism_id: str
    base_model: str
    parent_adapter_path: str
    candidate_adapter_path: str
    rank: int
    scale: float
    reward_source: dict[str, str]
    training_objective: dict[str, Any]
    mutation: dict[str, Any]
    request_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COUNCIL_LORA_TRAIN_REQUEST_SCHEMA,
            "season_id": self.season_id,
            "archetype_id": self.archetype_id,
            "parent_organism_id": self.parent_organism_id,
            "candidate_organism_id": self.candidate_organism_id,
            "base_model": self.base_model,
            "parent_adapter_path": self.parent_adapter_path,
            "candidate_adapter_path": self.candidate_adapter_path,
            "lora_config": {"rank": int(self.rank), "scale": float(self.scale)},
            "reward_source": dict(self.reward_source),
            "training_objective": dict(self.training_objective),
            "mutation": dict(self.mutation),
            "request_path": self.request_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CouncilLoRATrainRequest":
        if payload.get("schema_version") != COUNCIL_LORA_TRAIN_REQUEST_SCHEMA:
            raise ValueError(f"Unsupported train request schema: {payload.get('schema_version')!r}")
        lora = dict(payload.get("lora_config") or {})
        return cls(
            season_id=str(payload.get("season_id") or ""),
            archetype_id=str(payload.get("archetype_id") or ""),
            parent_organism_id=str(payload.get("parent_organism_id") or ""),
            candidate_organism_id=str(payload.get("candidate_organism_id") or ""),
            base_model=str(payload.get("base_model") or ""),
            parent_adapter_path=str(payload.get("parent_adapter_path") or ""),
            candidate_adapter_path=str(payload.get("candidate_adapter_path") or ""),
            rank=int(lora.get("rank") or 16),
            scale=float(lora.get("scale") or 20.0),
            reward_source=dict(payload.get("reward_source") or {}),
            training_objective=dict(payload.get("training_objective") or {}),
            mutation=dict(payload.get("mutation") or {}),
            request_path=str(payload.get("request_path") or ""),
        )


def candidate_adapter_path(*, archetype_id: str, season_id: str, candidate_organism_id: str) -> str:
    return f"checkpoints/adapters/{clean_id(archetype_id)}/candidates/{clean_id(season_id)}/{clean_id(candidate_organism_id)}"


def candidate_organism_id(*, parent: LoRAOrganism, season_id: str, mutation_index: int = 1) -> str:
    return f"{parent.archetype_id}.g{parent.generation + 1:03d}.{clean_id(season_id)}.mut{mutation_index:03d}"


class TrainRequestWriter:
    def __init__(self, request_root: Path) -> None:
        self.request_root = request_root.expanduser().resolve()

    def write(self, request: CouncilLoRATrainRequest) -> CouncilLoRATrainRequest:
        path = self.request_root / request.archetype_id / f"{request.candidate_organism_id}.json"
        with_path = CouncilLoRATrainRequest(**{**request.__dict__, "request_path": str(path)})
        write_json(path, with_path.to_dict())
        return with_path


def build_train_requests(
    *,
    state: CoevolutionState,
    fitness_report: PopulationFitnessReport,
    season_id: str,
    request_root: Path,
    reward_source: dict[str, str],
    parent_organism_ids: set[str] | None = None,
) -> dict[str, CouncilLoRATrainRequest]:
    by_archetype = state.archetypes_by_id()
    writer = TrainRequestWriter(request_root)
    requests: dict[str, CouncilLoRATrainRequest] = {}
    mutation_index_by_archetype: dict[str, int] = {}
    for organism_id, fitness in sorted(fitness_report.by_organism.items()):
        if parent_organism_ids is not None and organism_id not in parent_organism_ids:
            continue
        population = by_archetype.get(fitness.archetype_id)
        if population is None:
            continue
        parent = population.by_id().get(organism_id)
        if parent is None or fitness.rollout_count <= 0:
            continue
        mutation_index_by_archetype[parent.archetype_id] = mutation_index_by_archetype.get(parent.archetype_id, 0) + 1
        child_id = candidate_organism_id(
            parent=parent,
            season_id=season_id,
            mutation_index=mutation_index_by_archetype[parent.archetype_id],
        )
        request = CouncilLoRATrainRequest(
            season_id=season_id,
            archetype_id=parent.archetype_id,
            parent_organism_id=parent.organism_id,
            candidate_organism_id=child_id,
            base_model=parent.base_model,
            parent_adapter_path=parent.adapter_path,
            candidate_adapter_path=candidate_adapter_path(
                archetype_id=parent.archetype_id,
                season_id=season_id,
                candidate_organism_id=child_id,
            ),
            rank=parent.rank,
            scale=parent.scale,
            reward_source=dict(reward_source),
            training_objective={
                "type": "rl_policy_update",
                "algorithm": "ppo",
                "reward_field": "score.reward",
                "training_usable_field": "score.training_usable",
            },
            mutation={
                "kind": "pbt_child",
                "source": "elite_parent",
                "exploration": {
                    "rank_delta": 0,
                    "scale_multiplier": 1.0,
                    "learning_rate_multiplier": 1.0,
                },
            },
        )
        requests[parent.organism_id] = writer.write(request)
    return requests


def specialist_rewards_to_ppo_scored_rows(rows: Iterable[SpecialistRewardResult]) -> list[dict[str, Any]]:
    try:
        from economist_rl_ppo_trainer import proxy_old_logprob
    except Exception:  # noqa: BLE001
        proxy_old_logprob = lambda completion: -0.35  # type: ignore
    scored = []
    for row in rows:
        output = row.output.strip()
        prompt = row.training_prompt.strip()
        if not output or not prompt:
            continue
        scored.append(
            {
                "task_id": row.task_id,
                "rollout": {
                    "task_id": row.task_id,
                    "prompt": prompt,
                    "generation_prompt": prompt,
                    "output": output,
                    "old_logprob": float(proxy_old_logprob(output)),
                    "old_logprob_source": "proxy",
                    "archetype_id": row.archetype_id,
                    "organism_id": row.organism_id,
                },
                "score": {
                    "reward": row.reward,
                    "score": 100.0 * row.reward,
                    "training_usable": bool(output),
                    "diagnostics": list(row.diagnostics),
                },
            }
        )
    return scored


def write_ppo_scored_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
