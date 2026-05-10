#!/usr/bin/env python3
"""In-memory control-plane scheduler with retries and auth checks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT = SCRIPT_DIR.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from control_plane.models import JobSpec, WorkerCapabilities, WorkerHeartbeat


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: Dict[str, WorkerHeartbeat] = {}

    def upsert_heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        if not heartbeat.auth_token_hash:
            raise ValueError("missing auth token hash")
        self._workers[heartbeat.worker_id] = heartbeat

    def list_workers(self) -> List[WorkerHeartbeat]:
        return sorted(self._workers.values(), key=lambda w: (w.queue_depth, -w.memory_free_gb))

    def pick_worker(self, job: JobSpec) -> Optional[WorkerHeartbeat]:
        for worker in self.list_workers():
            if job.adapter_id not in worker.capabilities.adapters and "general_fallback" not in worker.capabilities.adapters:
                continue
            if job.execution_tier == "high_compute_mac" and not worker.capabilities.has_high_compute:
                continue
            if worker.queue_depth >= worker.capabilities.max_parallel_jobs:
                continue
            return worker
        return None


def schedule_jobs(registry: WorkerRegistry, jobs: List[JobSpec]) -> Dict[str, Dict[str, str]]:
    assignments: Dict[str, Dict[str, str]] = {}
    for job in jobs:
        worker = registry.pick_worker(job)
        if worker is None:
            assignments[job.job_id] = {"status": "queued", "reason": "no_eligible_worker"}
            continue
        assignments[job.job_id] = {"status": "assigned", "worker_id": worker.worker_id}
    return assignments


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Schedule jobs across multi-Mac workers.")
    ap.add_argument("--workers-json", type=Path, required=True, help="JSON with {workers:[...heartbeats...]}")
    ap.add_argument("--jobs-json", type=Path, required=True, help="JSON with {jobs:[...specs...]}")
    args = ap.parse_args()

    workers_doc = _read_json(args.workers_json)
    jobs_doc = _read_json(args.jobs_json)
    reg = WorkerRegistry()
    for raw in workers_doc.get("workers", []):
        caps = raw.get("capabilities", {})
        hb = WorkerHeartbeat(
            worker_id=str(raw["worker_id"]),
            host=str(raw.get("host") or "unknown"),
            queue_depth=int(raw.get("queue_depth", 0)),
            cpu_load=float(raw.get("cpu_load", 0.0)),
            memory_free_gb=float(raw.get("memory_free_gb", 0.0)),
            auth_token_hash=str(raw.get("auth_token_hash") or ""),
            capabilities=WorkerCapabilities(
                adapters=list(caps.get("adapters") or []),
                max_parallel_jobs=int(caps.get("max_parallel_jobs", 1)),
                has_high_compute=bool(caps.get("has_high_compute", False)),
                supports_api_judge=bool(caps.get("supports_api_judge", False)),
            ),
        )
        reg.upsert_heartbeat(hb)

    jobs: List[JobSpec] = []
    for raw in jobs_doc.get("jobs", []):
        jobs.append(
            JobSpec(
                job_id=str(raw["job_id"]),
                task_id=str(raw.get("task_id") or ""),
                adapter_id=str(raw.get("adapter_id") or "general_fallback"),
                execution_tier=str(raw.get("execution_tier") or "mac_pool"),
                payload_path=str(raw.get("payload_path") or ""),
                retry_count=int(raw.get("retry_count", 0)),
                max_retries=int(raw.get("max_retries", 2)),
            )
        )
    result = {
        "schema_version": "control_plane_schedule_v1",
        "workers": [asdict(w) for w in reg.list_workers()],
        "assignments": schedule_jobs(reg, jobs),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

