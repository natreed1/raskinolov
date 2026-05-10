#!/usr/bin/env python3
"""Data models for worker heartbeat/capability and job scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkerCapabilities:
    adapters: List[str]
    max_parallel_jobs: int = 1
    has_high_compute: bool = False
    supports_api_judge: bool = False


@dataclass
class WorkerHeartbeat:
    worker_id: str
    host: str
    queue_depth: int
    cpu_load: float
    memory_free_gb: float
    capabilities: WorkerCapabilities
    auth_token_hash: str
    timestamp: str = field(default_factory=_utc_now)


@dataclass
class JobSpec:
    job_id: str
    task_id: str
    adapter_id: str
    execution_tier: str
    payload_path: str
    retry_count: int = 0
    max_retries: int = 2
    policy_version: str = "router_policy_v1"

