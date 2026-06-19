"""Council archetype registry contract.

The adapter registry tracks domain checkpoints. This registry tracks trainable
council roles such as planner, grader, compressor, and specialist EQ lanes.
Each entry declares where traces, datasets, train requests, candidates, and
promotion artifacts are expected to live, so future runners can share one
repeatable contract instead of inventing paths per archetype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
SCHEMA_VERSION = "council_archetype_registry_v1"
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "training" / "council_archetype_registry_v1.json"

VALID_ARCHETYPES = {"planner", "grader", "compressor", "specialist_eq"}
VALID_PROMOTION_STATES = {"uninitialized", "candidate", "shadow", "champion", "retired"}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_id(value: str) -> str:
    out = []
    for ch in str(value or "").strip():
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


@dataclass(frozen=True)
class TrainingLaneContract:
    trace_schema: str
    dataset_schema: str
    dataset_root: str
    request_root: str
    candidate_root: str
    eval_root: str
    train_entrypoint: str
    eval_entrypoint: str
    promotion_gate: str
    default_lora_config: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema": self.trace_schema,
            "dataset_schema": self.dataset_schema,
            "dataset_root": self.dataset_root,
            "request_root": self.request_root,
            "candidate_root": self.candidate_root,
            "eval_root": self.eval_root,
            "train_entrypoint": self.train_entrypoint,
            "eval_entrypoint": self.eval_entrypoint,
            "promotion_gate": self.promotion_gate,
            "default_lora_config": self.default_lora_config,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingLaneContract":
        required = {
            "trace_schema",
            "dataset_schema",
            "dataset_root",
            "request_root",
            "candidate_root",
            "eval_root",
            "train_entrypoint",
            "eval_entrypoint",
            "promotion_gate",
            "default_lora_config",
        }
        missing = sorted(key for key in required if not str(payload.get(key) or "").strip())
        if missing:
            raise ValueError(f"Training lane contract missing required fields: {missing}")
        return cls(**{key: str(payload[key]) for key in sorted(required)})


@dataclass
class CouncilArchetypeEntry:
    archetype_id: str
    archetype: str
    display_name: str
    description: str
    base_model: str
    active_adapter_path: str
    seed_adapter_path: str
    promotion_state: str
    default_rank: int
    default_scale: float
    trainable: bool
    lane: TrainingLaneContract
    depends_on_adapter_registry: bool = False
    source_adapter_id: str = ""
    tags: list[str] = field(default_factory=list)
    updated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        state = self.promotion_state if self.promotion_state in VALID_PROMOTION_STATES else "uninitialized"
        return {
            "archetype_id": self.archetype_id,
            "archetype": self.archetype,
            "display_name": self.display_name,
            "description": self.description,
            "base_model": self.base_model,
            "active_adapter_path": self.active_adapter_path,
            "seed_adapter_path": self.seed_adapter_path,
            "promotion_state": state,
            "default_rank": int(self.default_rank),
            "default_scale": float(self.default_scale),
            "trainable": bool(self.trainable),
            "depends_on_adapter_registry": bool(self.depends_on_adapter_registry),
            "source_adapter_id": self.source_adapter_id,
            "tags": list(self.tags),
            "lane": self.lane.to_dict(),
            "updated_at_utc": self.updated_at_utc or _utc_iso(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CouncilArchetypeEntry":
        archetype_id = _clean_id(str(payload.get("archetype_id") or ""))
        archetype = _clean_id(str(payload.get("archetype") or ""))
        if not archetype_id:
            raise ValueError("Council archetype entry missing archetype_id")
        if archetype not in VALID_ARCHETYPES:
            raise ValueError(f"Invalid archetype {archetype!r} for {archetype_id}")
        lane_payload = payload.get("lane")
        if not isinstance(lane_payload, dict):
            raise ValueError(f"Council archetype {archetype_id} missing lane contract")
        return cls(
            archetype_id=archetype_id,
            archetype=archetype,
            display_name=str(payload.get("display_name") or archetype_id),
            description=str(payload.get("description") or ""),
            base_model=str(payload.get("base_model") or DEFAULT_BASE_MODEL),
            active_adapter_path=str(payload.get("active_adapter_path") or ""),
            seed_adapter_path=str(payload.get("seed_adapter_path") or ""),
            promotion_state=str(payload.get("promotion_state") or "uninitialized"),
            default_rank=max(1, int(payload.get("default_rank") or 16)),
            default_scale=float(payload.get("default_scale") or 20.0),
            trainable=bool(payload.get("trainable", True)),
            lane=TrainingLaneContract.from_dict(lane_payload),
            depends_on_adapter_registry=bool(payload.get("depends_on_adapter_registry", False)),
            source_adapter_id=str(payload.get("source_adapter_id") or ""),
            tags=[str(tag) for tag in list(payload.get("tags") or [])],
            updated_at_utc=str(payload.get("updated_at_utc") or ""),
        )


def _lane(archetype_id: str, trace_schema: str, dataset_schema: str) -> TrainingLaneContract:
    return TrainingLaneContract(
        trace_schema=trace_schema,
        dataset_schema=dataset_schema,
        dataset_root=f"data/lora/council_eq/{archetype_id}",
        request_root=f"benchmarks/results/council_eq/train_requests/{archetype_id}",
        candidate_root=f"checkpoints/adapters/{archetype_id}",
        eval_root=f"benchmarks/results/council_eq/evals/{archetype_id}",
        train_entrypoint=f"scripts/train_{archetype_id}_adapter.py",
        eval_entrypoint=f"scripts/eval_{archetype_id}_adapter.py",
        promotion_gate=f"scripts/promote_{archetype_id}_adapter.py",
        default_lora_config=f"training/{archetype_id}_lora_qwen25_coder_7b.yaml",
    )


def default_entries(*, specialist_adapter_ids: Iterable[str] = ()) -> list[CouncilArchetypeEntry]:
    entries = [
        CouncilArchetypeEntry(
            archetype_id="planner",
            archetype="planner",
            display_name="Council Planner",
            description="Trainable planner that reads the task first, selects experts, requests context, and builds expert packets.",
            base_model=DEFAULT_BASE_MODEL,
            active_adapter_path="checkpoints/adapters/planner/champion",
            seed_adapter_path="",
            promotion_state="uninitialized",
            default_rank=16,
            default_scale=20.0,
            trainable=True,
            tags=["routing", "context_requests", "expert_selection", "packet_building"],
            lane=_lane("planner", "planner_trace_v1", "planner_sft_or_preference_row_v1"),
        ),
        CouncilArchetypeEntry(
            archetype_id="grader",
            archetype="grader",
            display_name="Council Grader",
            description="Trainable adjudicator that scores answers, credits contributions, detects missing expertise, and reroutes when needed.",
            base_model=DEFAULT_BASE_MODEL,
            active_adapter_path="checkpoints/adapters/grader/champion",
            seed_adapter_path="",
            promotion_state="uninitialized",
            default_rank=16,
            default_scale=20.0,
            trainable=True,
            tags=["adjudication", "contribution_credit", "reroute", "confidence_calibration"],
            lane=_lane("grader", "grader_trace_v1", "grader_sft_or_preference_row_v1"),
        ),
        CouncilArchetypeEntry(
            archetype_id="context_compressor",
            archetype="compressor",
            display_name="Context Compressor",
            description="Trainable compressor that turns raw prompt/repo/debate material into compact council working state.",
            base_model=DEFAULT_BASE_MODEL,
            active_adapter_path="checkpoints/adapters/context_compressor/champion",
            seed_adapter_path="",
            promotion_state="uninitialized",
            default_rank=16,
            default_scale=20.0,
            trainable=True,
            tags=["compression", "working_state", "path_preservation", "constraint_preservation"],
            lane=_lane("context_compressor", "context_compression_trace_v1", "context_compression_sft_row_v1"),
        ),
    ]
    for adapter_id in sorted({_clean_id(aid) for aid in specialist_adapter_ids if _clean_id(aid)}):
        if adapter_id == "general_fallback":
            continue
        archetype_id = f"{adapter_id}_eq"
        entries.append(
            CouncilArchetypeEntry(
                archetype_id=archetype_id,
                archetype="specialist_eq",
                display_name=f"{adapter_id} EQ",
                description=(
                    f"Trainable council-behavior lane attached to specialist adapter {adapter_id}; "
                    "optimizes collaboration, critique, confidence, and contribution quality."
                ),
                base_model=DEFAULT_BASE_MODEL,
                active_adapter_path=f"checkpoints/adapters/{archetype_id}/champion",
                seed_adapter_path="",
                promotion_state="uninitialized",
                default_rank=16,
                default_scale=20.0,
                trainable=True,
                depends_on_adapter_registry=True,
                source_adapter_id=adapter_id,
                tags=["specialist_eq", "council_behavior", adapter_id],
                lane=_lane(archetype_id, "specialist_eq_trace_v1", "specialist_eq_sft_or_preference_row_v1"),
            )
        )
    return entries


class CouncilArchetypeRegistry:
    def __init__(self, entries: Iterable[CouncilArchetypeEntry]) -> None:
        self.entries = list(entries)
        seen: set[str] = set()
        for entry in self.entries:
            if entry.archetype_id in seen:
                raise ValueError(f"Duplicate council archetype_id: {entry.archetype_id}")
            seen.add(entry.archetype_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_or_updated_utc": _utc_iso(),
            "base_model_default": DEFAULT_BASE_MODEL,
            "description": (
                "Trainable council archetype contract for planner, grader, compressor, "
                "and specialist EQ LoRA lanes."
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CouncilArchetypeRegistry":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported council archetype registry schema: {payload.get('schema_version')!r}")
        rows = payload.get("entries")
        if not isinstance(rows, list):
            raise ValueError("Council archetype registry missing entries list")
        return cls(CouncilArchetypeEntry.from_dict(row) for row in rows if isinstance(row, dict))

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "CouncilArchetypeRegistry":
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def bootstrap(cls, *, specialist_adapter_ids: Iterable[str] = ()) -> "CouncilArchetypeRegistry":
        return cls(default_entries(specialist_adapter_ids=specialist_adapter_ids))

    def save(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        out = path.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def by_id(self) -> dict[str, CouncilArchetypeEntry]:
        return {entry.archetype_id: entry for entry in self.entries}
