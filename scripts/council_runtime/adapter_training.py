"""Adapter training backend abstractions for council LoRA requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .train_requests import CouncilLoRATrainRequest

ADAPTER_TRAIN_RESULT_SCHEMA = "council_adapter_train_result_v1"


@dataclass(frozen=True)
class AdapterTrainResult:
    request_path: str
    candidate_adapter_path: str
    status: str
    backend: str
    manifest_path: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTER_TRAIN_RESULT_SCHEMA,
            "request_path": self.request_path,
            "candidate_adapter_path": self.candidate_adapter_path,
            "status": self.status,
            "backend": self.backend,
            "manifest_path": self.manifest_path,
            "details": dict(self.details),
        }


class AdapterTrainingBackend(Protocol):
    def train(
        self,
        *,
        request: CouncilLoRATrainRequest,
        ppo_scored_rows_path: Path,
        manifest_path: Path,
    ) -> AdapterTrainResult:
        ...


class DryRunAdapterTrainingBackend:
    def train(
        self,
        *,
        request: CouncilLoRATrainRequest,
        ppo_scored_rows_path: Path,
        manifest_path: Path,
    ) -> AdapterTrainResult:
        result = AdapterTrainResult(
            request_path=request.request_path,
            candidate_adapter_path=request.candidate_adapter_path,
            status="dry_run_no_weight_update",
            backend="dry_run",
            manifest_path=str(manifest_path.expanduser().resolve()),
            details={
                "request": request.to_dict(),
                "ppo_scored_rows_path": str(ppo_scored_rows_path.expanduser().resolve()),
                "note": "No LoRA weights were modified.",
            },
        )
        manifest_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        manifest_path.expanduser().resolve().write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


class PPOAdapterTrainingBackend:
    def __init__(self, *, dry_run: bool = False, train_backend: str | None = None, ppo_config: Any | None = None) -> None:
        self.dry_run = bool(dry_run)
        self.train_backend = train_backend
        self.ppo_config = ppo_config

    def train(
        self,
        *,
        request: CouncilLoRATrainRequest,
        ppo_scored_rows_path: Path,
        manifest_path: Path,
    ) -> AdapterTrainResult:
        from economist_rl_ppo_trainer import PPOConfig, train_ppo_batch, write_ppo_manifest

        scored_rows = [
            json.loads(line)
            for line in ppo_scored_rows_path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = train_ppo_batch(
            scored_rows=scored_rows,
            system_prompt="You are a council specialist adapter optimizing collaborative expert behavior.",
            base_model=request.base_model,
            source_adapter=Path(request.parent_adapter_path),
            candidate_adapter=Path(request.candidate_adapter_path),
            cfg=self.ppo_config or PPOConfig(),
            dry_run=self.dry_run,
            train_backend=self.train_backend,
        )
        write_ppo_manifest(manifest_path.expanduser().resolve(), manifest)
        status = str(manifest.get("status") or "unknown")
        result = AdapterTrainResult(
            request_path=request.request_path,
            candidate_adapter_path=request.candidate_adapter_path,
            status=status,
            backend="ppo",
            manifest_path=str(manifest_path.expanduser().resolve()),
            details=manifest,
        )
        return result
