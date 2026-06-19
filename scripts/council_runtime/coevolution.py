"""Co-evolution contracts for council archetype training.

This layer models the repeatable outer loop:

- a reward/scorer population learns what to value,
- LoRA organism populations learn behavior selected by those rewards,
- seasons evaluate both against fixed artifacts before promotion.

The module is intentionally training-backend agnostic. It writes explicit state
and manifest objects that future SFT/DPO/PPO operators can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .archetypes import CouncilArchetypeRegistry, DEFAULT_BASE_MODEL

COEVOLUTION_STATE_SCHEMA = "council_coevolution_state_v1"
SEASON_MANIFEST_SCHEMA = "council_coevolution_season_manifest_v1"
REWARD_MATRIX_SCHEMA = "council_reward_matrix_v1"
ORGANISM_SCHEMA = "council_lora_organism_v1"

DEFAULT_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "routing" / "council_coevolution_state_v1.json"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "council_eq"

VALID_ORGANISM_STATES = {"seed", "candidate", "elite", "shadow", "champion", "retired"}
VALID_SCORER_STATES = {"candidate", "elite", "champion", "retired"}


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_id(value: str) -> str:
    out = []
    for ch in str(value or "").strip():
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive_total = sum(max(0.0, float(v)) for v in weights.values())
    if positive_total <= 0:
        raise ValueError("Reward matrix must include at least one positive weight")
    return {str(k): round(float(v) / positive_total, 6) for k, v in sorted(weights.items())}


@dataclass(frozen=True)
class RewardMatrix:
    matrix_id: str
    description: str
    weights: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: str = ""
    state: str = "candidate"

    def __post_init__(self) -> None:
        if not clean_id(self.matrix_id):
            raise ValueError("Reward matrix missing matrix_id")
        if self.state not in VALID_SCORER_STATES:
            raise ValueError(f"Invalid reward matrix state: {self.state}")

    def score(self, metrics: dict[str, Any]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            try:
                value = float(metrics.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            total += float(weight) * max(0.0, min(1.0, value))
        return round(max(0.0, min(1.0, total)), 6)

    def mutate(self, *, child_id: str, focus_signal: str, delta: float = 0.05) -> "RewardMatrix":
        weights = dict(self.weights)
        if focus_signal not in weights:
            weights[focus_signal] = max(0.01, float(delta))
        else:
            weights[focus_signal] = max(0.0, float(weights[focus_signal]) + float(delta))
        weights = normalize_weights(weights)
        return RewardMatrix(
            matrix_id=clean_id(child_id),
            description=f"Mutation of {self.matrix_id}; boosted {focus_signal}",
            weights=weights,
            thresholds=dict(self.thresholds),
            generation=int(self.generation) + 1,
            parent_id=self.matrix_id,
            state="candidate",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_MATRIX_SCHEMA,
            "matrix_id": self.matrix_id,
            "description": self.description,
            "weights": normalize_weights(self.weights),
            "thresholds": {k: float(v) for k, v in sorted(self.thresholds.items())},
            "generation": int(self.generation),
            "parent_id": self.parent_id,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardMatrix":
        if payload.get("schema_version") != REWARD_MATRIX_SCHEMA:
            raise ValueError(f"Unsupported reward matrix schema: {payload.get('schema_version')!r}")
        weights = payload.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("Reward matrix missing weights")
        return cls(
            matrix_id=clean_id(str(payload.get("matrix_id") or "")),
            description=str(payload.get("description") or ""),
            weights=normalize_weights({str(k): float(v) for k, v in weights.items()}),
            thresholds={str(k): float(v) for k, v in dict(payload.get("thresholds") or {}).items()},
            generation=int(payload.get("generation") or 0),
            parent_id=str(payload.get("parent_id") or ""),
            state=str(payload.get("state") or "candidate"),
        )


@dataclass
class RewardPopulation:
    matrices: list[RewardMatrix]
    active_matrix_id: str
    generation: int = 0

    def __post_init__(self) -> None:
        ids = {matrix.matrix_id for matrix in self.matrices}
        if self.active_matrix_id not in ids:
            raise ValueError(f"Active reward matrix not found: {self.active_matrix_id}")

    def active(self) -> RewardMatrix:
        return self.by_id()[self.active_matrix_id]

    def by_id(self) -> dict[str, RewardMatrix]:
        return {matrix.matrix_id: matrix for matrix in self.matrices}

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_matrix_id": self.active_matrix_id,
            "generation": int(self.generation),
            "matrices": [matrix.to_dict() for matrix in self.matrices],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardPopulation":
        rows = payload.get("matrices")
        if not isinstance(rows, list):
            raise ValueError("Reward population missing matrices")
        return cls(
            matrices=[RewardMatrix.from_dict(row) for row in rows if isinstance(row, dict)],
            active_matrix_id=str(payload.get("active_matrix_id") or ""),
            generation=int(payload.get("generation") or 0),
        )


@dataclass(frozen=True)
class LoRAOrganism:
    organism_id: str
    archetype_id: str
    adapter_path: str
    base_model: str = DEFAULT_BASE_MODEL
    rank: int = 16
    scale: float = 20.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    state: str = "seed"
    reward_matrix_id: str = ""
    fitness: dict[str, float] = field(default_factory=dict)
    training_recipe: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_id(self.organism_id):
            raise ValueError("LoRA organism missing organism_id")
        if not clean_id(self.archetype_id):
            raise ValueError("LoRA organism missing archetype_id")
        if self.state not in VALID_ORGANISM_STATES:
            raise ValueError(f"Invalid organism state: {self.state}")

    def child(
        self,
        *,
        organism_id: str,
        adapter_path: str,
        reward_matrix_id: str,
        training_recipe: dict[str, Any],
    ) -> "LoRAOrganism":
        return LoRAOrganism(
            organism_id=clean_id(organism_id),
            archetype_id=self.archetype_id,
            adapter_path=adapter_path,
            base_model=self.base_model,
            rank=self.rank,
            scale=self.scale,
            generation=self.generation + 1,
            parent_ids=[self.organism_id],
            state="candidate",
            reward_matrix_id=reward_matrix_id,
            fitness={},
            training_recipe=dict(training_recipe),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ORGANISM_SCHEMA,
            "organism_id": self.organism_id,
            "archetype_id": self.archetype_id,
            "adapter_path": self.adapter_path,
            "base_model": self.base_model,
            "rank": int(self.rank),
            "scale": float(self.scale),
            "generation": int(self.generation),
            "parent_ids": list(self.parent_ids),
            "state": self.state,
            "reward_matrix_id": self.reward_matrix_id,
            "fitness": {k: float(v) for k, v in sorted(self.fitness.items())},
            "training_recipe": dict(self.training_recipe),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoRAOrganism":
        if payload.get("schema_version") != ORGANISM_SCHEMA:
            raise ValueError(f"Unsupported organism schema: {payload.get('schema_version')!r}")
        return cls(
            organism_id=clean_id(str(payload.get("organism_id") or "")),
            archetype_id=clean_id(str(payload.get("archetype_id") or "")),
            adapter_path=str(payload.get("adapter_path") or ""),
            base_model=str(payload.get("base_model") or DEFAULT_BASE_MODEL),
            rank=max(1, int(payload.get("rank") or 16)),
            scale=float(payload.get("scale") or 20.0),
            generation=int(payload.get("generation") or 0),
            parent_ids=[str(x) for x in list(payload.get("parent_ids") or [])],
            state=str(payload.get("state") or "seed"),
            reward_matrix_id=str(payload.get("reward_matrix_id") or ""),
            fitness={str(k): float(v) for k, v in dict(payload.get("fitness") or {}).items()},
            training_recipe=dict(payload.get("training_recipe") or {}),
        )


@dataclass
class ArchetypePopulation:
    archetype_id: str
    organisms: list[LoRAOrganism]
    active_organism_id: str
    generation: int = 0
    elite_fraction: float = 0.2
    mutation_count: int = 2

    def __post_init__(self) -> None:
        ids = {organism.organism_id for organism in self.organisms}
        if self.active_organism_id not in ids:
            raise ValueError(f"Active organism not found for {self.archetype_id}: {self.active_organism_id}")

    def by_id(self) -> dict[str, LoRAOrganism]:
        return {organism.organism_id: organism for organism in self.organisms}

    def active(self) -> LoRAOrganism:
        return self.by_id()[self.active_organism_id]

    def elites(self, *, metric: str = "fitness") -> list[LoRAOrganism]:
        ordered = sorted(
            self.organisms,
            key=lambda organism: float(organism.fitness.get(metric, organism.fitness.get("overall", 0.0))),
            reverse=True,
        )
        keep = max(1, int(round(len(ordered) * float(self.elite_fraction))))
        return ordered[:keep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_id": self.archetype_id,
            "active_organism_id": self.active_organism_id,
            "generation": int(self.generation),
            "elite_fraction": float(self.elite_fraction),
            "mutation_count": int(self.mutation_count),
            "organisms": [organism.to_dict() for organism in self.organisms],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchetypePopulation":
        rows = payload.get("organisms")
        if not isinstance(rows, list):
            raise ValueError("Archetype population missing organisms")
        return cls(
            archetype_id=clean_id(str(payload.get("archetype_id") or "")),
            organisms=[LoRAOrganism.from_dict(row) for row in rows if isinstance(row, dict)],
            active_organism_id=str(payload.get("active_organism_id") or ""),
            generation=int(payload.get("generation") or 0),
            elite_fraction=float(payload.get("elite_fraction") or 0.2),
            mutation_count=int(payload.get("mutation_count") or 2),
        )


@dataclass
class CoevolutionState:
    reward_population: RewardPopulation
    archetype_populations: list[ArchetypePopulation]
    created_or_updated_utc: str = ""

    def archetypes_by_id(self) -> dict[str, ArchetypePopulation]:
        return {population.archetype_id: population for population in self.archetype_populations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COEVOLUTION_STATE_SCHEMA,
            "created_or_updated_utc": self.created_or_updated_utc or utc_iso(),
            "reward_population": self.reward_population.to_dict(),
            "archetype_populations": [population.to_dict() for population in self.archetype_populations],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CoevolutionState":
        if payload.get("schema_version") != COEVOLUTION_STATE_SCHEMA:
            raise ValueError(f"Unsupported coevolution state schema: {payload.get('schema_version')!r}")
        reward_payload = payload.get("reward_population")
        if not isinstance(reward_payload, dict):
            raise ValueError("Coevolution state missing reward_population")
        rows = payload.get("archetype_populations")
        if not isinstance(rows, list):
            raise ValueError("Coevolution state missing archetype_populations")
        return cls(
            reward_population=RewardPopulation.from_dict(reward_payload),
            archetype_populations=[ArchetypePopulation.from_dict(row) for row in rows if isinstance(row, dict)],
            created_or_updated_utc=str(payload.get("created_or_updated_utc") or ""),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> "CoevolutionState":
        return cls.from_dict(json.loads(path.expanduser().resolve().read_text(encoding="utf-8")))

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        out = path.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_reward_population() -> RewardPopulation:
    balanced = RewardMatrix(
        matrix_id="reward_balanced_v1",
        description="Balanced seed scorer for outcome, contribution, evidence, calibration, and efficiency.",
        weights=normalize_weights(
            {
                "final_outcome": 0.30,
                "contribution_credit": 0.20,
                "evidence_quality": 0.15,
                "routing_or_reroute_quality": 0.12,
                "confidence_calibration": 0.10,
                "context_efficiency": 0.08,
                "cost_efficiency": 0.05,
            }
        ),
        thresholds={"promote_min_holdout": 0.62, "retire_below": 0.25},
        generation=0,
        parent_id="",
        state="champion",
    )
    contribution = balanced.mutate(
        child_id="reward_contribution_hawk_v1",
        focus_signal="contribution_credit",
        delta=0.08,
    )
    efficiency = balanced.mutate(
        child_id="reward_efficiency_hawk_v1",
        focus_signal="context_efficiency",
        delta=0.08,
    )
    return RewardPopulation(
        matrices=[balanced, contribution, efficiency],
        active_matrix_id=balanced.matrix_id,
        generation=0,
    )


def bootstrap_state(registry: CouncilArchetypeRegistry) -> CoevolutionState:
    reward_population = default_reward_population()
    populations: list[ArchetypePopulation] = []
    for entry in registry.entries:
        seed_id = f"{entry.archetype_id}.seed.g000"
        organism = LoRAOrganism(
            organism_id=seed_id,
            archetype_id=entry.archetype_id,
            adapter_path=entry.seed_adapter_path or entry.active_adapter_path,
            base_model=entry.base_model,
            rank=entry.default_rank,
            scale=entry.default_scale,
            generation=0,
            parent_ids=[],
            state="seed",
            reward_matrix_id=reward_population.active_matrix_id,
            training_recipe={
                "kind": "seed_from_archetype_registry",
                "dataset_schema": entry.lane.dataset_schema,
                "trace_schema": entry.lane.trace_schema,
            },
        )
        populations.append(
            ArchetypePopulation(
                archetype_id=entry.archetype_id,
                organisms=[organism],
                active_organism_id=seed_id,
                generation=0,
                elite_fraction=0.2,
                mutation_count=2,
            )
        )
    return CoevolutionState(
        reward_population=reward_population,
        archetype_populations=populations,
        created_or_updated_utc=utc_iso(),
    )


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "summary": dict(self.summary),
            "outputs": dict(self.outputs),
            "metrics": {k: float(v) for k, v in sorted(self.metrics.items())},
        }


@dataclass
class SeasonManifest:
    season_id: str
    run_dir: str
    state_path: str
    archetype_ids: list[str]
    reward_matrix_ids: list[str]
    dry_run: bool
    phases: list[PhaseResult]
    started_at_utc: str
    finished_at_utc: str = ""
    status: str = "initialized"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEASON_MANIFEST_SCHEMA,
            "season_id": self.season_id,
            "run_dir": self.run_dir,
            "state_path": self.state_path,
            "archetype_ids": list(self.archetype_ids),
            "reward_matrix_ids": list(self.reward_matrix_ids),
            "dry_run": bool(self.dry_run),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "status": self.status,
            "phases": [phase.to_dict() for phase in self.phases],
        }


def next_season_id(results_root: Path = DEFAULT_RESULTS_ROOT) -> str:
    seasons_root = results_root.expanduser().resolve() / "seasons"
    highest = 0
    if seasons_root.is_dir():
        for child in seasons_root.iterdir():
            if not child.is_dir() or not child.name.startswith("season_"):
                continue
            parts = child.name.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                highest = max(highest, int(parts[1]))
    return f"season_{highest + 1:04d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
