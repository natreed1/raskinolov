#!/usr/bin/env python3
"""
Launch and dispatch prompt-ablation cells across Lambda Cloud workers in parallel.

Default matrix cells (one worker per cell):
  - rag_current
  - context_max_potential
  - rag_current_plus_max_potential

Use `--include-baseline` when a fresh control run is needed.

Each worker runs one `run_final_mass_testing_system.py` invocation with isolated
worktree root + output artifacts.

Example:
  python scripts/launch_lambda_parallel_ablation.py \
    --launch-instances \
    --instance-type gpu_1x_a10 \
    --region us-east-1 \
    --fallback-region us-west-1 \
    --fallback-instance-type gpu_1x_a100 \
    --ssh-key-name lambda-cloud-cursor
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMBDA_CLOUD_BASE_URL = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_LAMBDA_API_BASE = DEFAULT_LAMBDA_CLOUD_BASE_URL
DEFAULT_LAMBDA_FILE_SYSTEM_ID = "5795235886dd4e45a5adcdb5637de9d6"
DEFAULT_RAG_CURRENT_CORPUS = "data/rag/current_rag_dataset.json"
DEFAULT_RSYNC_CONNECT_TIMEOUT_SECONDS = 30
DEFAULT_RSYNC_IDLE_TIMEOUT_SECONDS = 120


@dataclass
class MatrixCell:
    name: str
    extra_args: List[str]
    adapter_id: str = ""
    task_domains: List[str] | None = None


@dataclass(frozen=True)
class LaunchCandidate:
    instance_type: str
    region: str


class LambdaLaunchCandidateError(RuntimeError):
    """A launch candidate failed after cleaning up any partial instances."""


BASELINE_CELL = MatrixCell("baseline", [])

DEFAULT_CELLS: List[MatrixCell] = [
    MatrixCell(
        "rag_current",
        [
            "--prompt-rag-current-corpus",
            DEFAULT_RAG_CURRENT_CORPUS,
            "--prompt-rag-current-top-k",
            "3",
            "--prompt-rag-current-max-chars",
            "1600",
        ],
    ),
    MatrixCell("context_max_potential", ["--prompt-context-engineering", "max_potential"]),
    MatrixCell(
        "rag_current_plus_max_potential",
        [
            "--prompt-rag-current-corpus",
            DEFAULT_RAG_CURRENT_CORPUS,
            "--prompt-rag-current-top-k",
            "3",
            "--prompt-rag-current-max-chars",
            "1600",
            "--prompt-context-engineering",
            "max_potential",
        ],
    ),
]

SPECIALIST_DOMAIN_MAP: Dict[str, List[str]] = {
    "loading_screen": ["hud_status"],
    "hud_status": ["hud_status"],
    "economy_tooltip": ["economy"],
    "economistRL": ["economy"],
    "combat_risk": ["army_operations"],
    "save_load_api_guard": ["state_perstitence_integrity"],
    "ai_planning_explanation": ["ai_strategy_and_planning"],
}


def _api_call(api_base: str, api_key: str, method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("utf-8")
    normalized_base = api_base.strip().rstrip("/")
    if normalized_base.endswith("/instances"):
        normalized_base = normalized_base[: -len("/instances")]
    url = f"{normalized_base}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "fe-lambda-parallel-ablation/1.0",
    }
    transient_codes = {429, 500, 502, 503, 504}
    last_detail = ""
    for attempt in range(6):
        req = request.Request(url, data=body, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            last_detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in transient_codes or attempt == 5:
                raise RuntimeError(f"Lambda API {method} {path} HTTP {exc.code}: {last_detail[:500]}") from exc
            sleep_s = min(60, 5 * (attempt + 1))
            print(f"lambda_api_retry method={method} path={path} http={exc.code} sleep_s={sleep_s}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"Lambda API {method} {path} failed after retries: {last_detail[:500]}")


def _lambda_cloud_base_url_from_env() -> str:
    raw = (
        os.environ.get("LAMBDA_CLOUD_BASE_URL")
        or os.environ.get("LAMBDA_API_BASE")
        or DEFAULT_LAMBDA_CLOUD_BASE_URL
    )
    base = raw.strip().rstrip("/")
    if base.endswith("/instances"):
        base = base[: -len("/instances")]
    return base


def _shell(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _rsync_ssh_command(ssh_key_path: Path) -> str:
    return (
        f"ssh -i {ssh_key_path} "
        f"-o ConnectTimeout={DEFAULT_RSYNC_CONNECT_TIMEOUT_SECONDS} "
        "-o ServerAliveInterval=30 "
        "-o ServerAliveCountMax=4"
    )


def _rsync_timeout_args() -> List[str]:
    return [
        f"--contimeout={DEFAULT_RSYNC_CONNECT_TIMEOUT_SECONDS}",
        f"--timeout={DEFAULT_RSYNC_IDLE_TIMEOUT_SECONDS}",
    ]


def _resolve_file_system(api_base: str, api_key: str, identifier: str) -> Dict[str, Any]:
    wanted = str(identifier or "").strip()
    if not wanted:
        return {}
    payload = _api_call(api_base, api_key, "GET", "/file-systems")
    for row in payload.get("data") or []:
        if wanted in {str(row.get("id") or ""), str(row.get("name") or "")}:
            return dict(row)
    raise RuntimeError(f"Lambda file system not found by id/name: {wanted}")


def _split_csv_cli_values(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def _unique_preserving_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _region_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "region_name", "region"):
            raw = value.get(key)
            if isinstance(raw, dict):
                nested = _region_name(raw)
                if nested:
                    return nested
            if raw:
                return str(raw).strip()
    return ""


def _lambda_instance_type_capacity(api_base: str, api_key: str) -> Dict[str, set[str]]:
    payload = _api_call(api_base, api_key, "GET", "/instance-types")
    data = payload.get("data") or {}
    rows: List[tuple[str, Any]] = []
    if isinstance(data, dict):
        rows = [(str(instance_type), row) for instance_type, row in data.items()]
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            instance_type = str(row.get("name") or row.get("instance_type_name") or row.get("type") or "").strip()
            if instance_type:
                rows.append((instance_type, row))

    capacity: Dict[str, set[str]] = {}
    for instance_type, row in rows:
        if not isinstance(row, dict):
            continue
        raw_regions = row.get("regions_with_capacity_available")
        if raw_regions is None:
            raw_regions = row.get("regions_with_capacity")
        if raw_regions is None:
            continue
        if not isinstance(raw_regions, list):
            raw_regions = [raw_regions]
        regions = {_region_name(region) for region in raw_regions}
        capacity[instance_type] = {region for region in regions if region}
    return capacity


def _candidate_allowed(capacity: Dict[str, set[str]], candidate: LaunchCandidate) -> bool:
    if not capacity:
        return True
    return candidate.region in capacity.get(candidate.instance_type, set())


def _append_candidate(
    candidates: List[LaunchCandidate],
    seen: set[tuple[str, str]],
    capacity: Dict[str, set[str]],
    instance_type: str,
    region: str,
    *,
    reason: str,
) -> None:
    candidate = LaunchCandidate(instance_type=instance_type, region=region)
    key = (candidate.instance_type, candidate.region)
    if key in seen:
        return
    seen.add(key)
    if _candidate_allowed(capacity, candidate):
        print(
            f"lambda_capacity_candidate_added instance_type={candidate.instance_type} "
            f"region={candidate.region} reason={reason}",
            flush=True,
        )
        candidates.append(candidate)
        return
    print(
        f"lambda_capacity_candidate_skipped instance_type={candidate.instance_type} "
        f"region={candidate.region} reason=no_reported_capacity source={reason}",
        flush=True,
    )


def _select_launch_candidates(
    api_base: str,
    api_key: str,
    *,
    requested_instance_type: str,
    requested_region: str,
    fallback_instance_types: List[str],
    fallback_regions: List[str],
) -> List[LaunchCandidate]:
    requested_instance_type = str(requested_instance_type or "").strip()
    requested_region = str(requested_region or "").strip()
    if not requested_instance_type:
        raise RuntimeError("--instance-type must not be empty.")
    if not requested_region:
        raise RuntimeError("--region/effective region must not be empty.")

    try:
        capacity = _lambda_instance_type_capacity(api_base, api_key)
        if capacity:
            print(
                "lambda_capacity_preflight_loaded "
                + " ".join(f"{itype}={','.join(sorted(regions)) or '-'}" for itype, regions in sorted(capacity.items())),
                flush=True,
            )
        else:
            print("lambda_capacity_preflight_empty using_requested_and_fallback_pairs", flush=True)
    except Exception as exc:
        capacity = {}
        print(f"lambda_capacity_preflight_failed using_requested_and_fallback_pairs error={exc}", flush=True)

    fallback_instance_types = _unique_preserving_order(fallback_instance_types)
    fallback_regions = _unique_preserving_order(fallback_regions)
    preferred_regions = _unique_preserving_order([requested_region, *fallback_regions])
    candidates: List[LaunchCandidate] = []
    seen: set[tuple[str, str]] = set()

    _append_candidate(
        candidates,
        seen,
        capacity,
        requested_instance_type,
        requested_region,
        reason="requested_pair",
    )
    for region in fallback_regions:
        _append_candidate(candidates, seen, capacity, requested_instance_type, region, reason="requested_type_fallback_region")
    for region in sorted(capacity.get(requested_instance_type, set())):
        _append_candidate(candidates, seen, capacity, requested_instance_type, region, reason="requested_type_available_region")
    for instance_type in fallback_instance_types:
        for region in preferred_regions:
            _append_candidate(candidates, seen, capacity, instance_type, region, reason="fallback_type_preferred_region")

    if not candidates:
        raise RuntimeError(
            "No Lambda launch candidates have reported capacity. "
            f"requested_instance_type={requested_instance_type} requested_region={requested_region} "
            f"fallback_instance_types={','.join(fallback_instance_types)} fallback_regions={','.join(fallback_regions)}"
        )
    print(
        "lambda_capacity_candidate_order "
        + " ".join(f"{idx + 1}:{candidate.instance_type}@{candidate.region}" for idx, candidate in enumerate(candidates)),
        flush=True,
    )
    return candidates


def _is_insufficient_capacity_error(message: str) -> bool:
    normalized = str(message or "").lower()
    signals = [
        "insufficient-capacity",
        "insufficient capacity",
        "capacity is not available",
        "not enough capacity",
        "no capacity",
        "no available",
        "currently unavailable",
    ]
    return any(signal in normalized for signal in signals)


def _launch_instances(
    api_base: str,
    api_key: str,
    *,
    region: str,
    instance_type: str,
    ssh_key_name: str,
    quantity: int,
    name_prefix: str,
    file_system_names: List[str],
) -> List[str]:
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "quantity": quantity,
        "name": f"{name_prefix}-{int(time.time())}",
    }
    if file_system_names:
        payload["file_system_names"] = file_system_names
    try:
        resp = _api_call(api_base, api_key, "POST", "/instance-operations/launch", payload)
        ids = list((resp.get("data") or {}).get("instance_ids") or [])
        if len(ids) != quantity:
            _terminate_instances_best_effort(api_base, api_key, ids, reason="incomplete_bulk_launch")
            raise LambdaLaunchCandidateError(f"Expected {quantity} launched instances, got {len(ids)}: {ids}")
        return ids
    except RuntimeError as exc:
        if quantity <= 1 or "Specify at most 1 instances" not in str(exc):
            raise

    # Some Lambda accounts only allow launching one instance per API request.
    ids: List[str] = []
    try:
        for idx in range(quantity):
            single_payload = dict(payload)
            single_payload["quantity"] = 1
            single_payload["name"] = f"{name_prefix}-{idx + 1}-{int(time.time())}"
            resp = _api_call(api_base, api_key, "POST", "/instance-operations/launch", single_payload)
            launched = list((resp.get("data") or {}).get("instance_ids") or [])
            if len(launched) != 1:
                _terminate_instances_best_effort(
                    api_base,
                    api_key,
                    [*ids, *launched],
                    reason="partial_single_launch_wrong_quantity",
                )
                raise LambdaLaunchCandidateError(f"Expected one launched instance, got {launched}")
            ids.extend(launched)
    except Exception as exc:
        _terminate_instances_best_effort(api_base, api_key, ids, reason="partial_launch_failure")
        if isinstance(exc, LambdaLaunchCandidateError) or _is_insufficient_capacity_error(str(exc)):
            raise LambdaLaunchCandidateError(str(exc)) from exc
        raise
    return ids


def _launch_instances_with_fallback(
    api_base: str,
    api_key: str,
    *,
    requested_region: str,
    requested_instance_type: str,
    fallback_regions: List[str],
    fallback_instance_types: List[str],
    ssh_key_name: str,
    quantity: int,
    name_prefix: str,
    file_system_names: List[str],
) -> tuple[List[str], LaunchCandidate]:
    candidates = _select_launch_candidates(
        api_base,
        api_key,
        requested_instance_type=requested_instance_type,
        requested_region=requested_region,
        fallback_instance_types=fallback_instance_types,
        fallback_regions=fallback_regions,
    )
    failures: List[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"lambda_launch_candidate_try index={idx}/{len(candidates)} "
            f"instance_type={candidate.instance_type} region={candidate.region} quantity={quantity}",
            flush=True,
        )
        try:
            ids = _launch_instances(
                api_base,
                api_key,
                region=candidate.region,
                instance_type=candidate.instance_type,
                ssh_key_name=ssh_key_name,
                quantity=quantity,
                name_prefix=name_prefix,
                file_system_names=file_system_names,
            )
            print(
                f"lambda_launch_candidate_selected instance_type={candidate.instance_type} "
                f"region={candidate.region} instance_count={len(ids)}",
                flush=True,
            )
            return ids, candidate
        except Exception as exc:
            retryable = isinstance(exc, LambdaLaunchCandidateError) or _is_insufficient_capacity_error(str(exc))
            print(
                f"lambda_launch_candidate_failed instance_type={candidate.instance_type} "
                f"region={candidate.region} retryable={int(bool(retryable))} error={exc}",
                flush=True,
            )
            if not retryable:
                raise
            failures.append(f"{candidate.instance_type}@{candidate.region}: {exc}")
    raise RuntimeError("All Lambda launch candidates failed: " + " | ".join(failures))


def _terminate_instances_best_effort(api_base: str, api_key: str, instance_ids: List[str], *, reason: str) -> None:
    ids = [str(instance_id).strip() for instance_id in instance_ids if str(instance_id).strip()]
    if not ids:
        return
    print(f"setup_cleanup_terminate_start reason={reason} instance_count={len(ids)} ids={','.join(ids)}", flush=True)
    try:
        resp = _api_call(api_base, api_key, "POST", "/instance-operations/terminate", {"instance_ids": ids})
    except Exception as exc:
        print(f"setup_cleanup_terminate_failed reason={reason} error={exc}", flush=True)
        return
    terminated = list((resp.get("data") or {}).get("terminated_instances") or [])
    print(f"setup_cleanup_terminate_done reason={reason} terminated_count={len(terminated)}", flush=True)
    for row in terminated:
        print(
            "setup_cleanup_terminated "
            f"id={row.get('id', '')} "
            f"name={row.get('name', '')} "
            f"status={row.get('status', '')}",
            flush=True,
        )


def _wait_for_instance_ips(api_base: str, api_key: str, instance_ids: List[str], timeout_s: int = 900) -> Dict[str, str]:
    wanted = set(instance_ids)
    deadline = time.time() + timeout_s
    out: Dict[str, str] = {}
    while time.time() < deadline:
        payload = _api_call(api_base, api_key, "GET", "/instances")
        for row in payload.get("data") or []:
            iid = str(row.get("id") or "")
            if iid not in wanted:
                continue
            ip = str(row.get("ip") or row.get("public_ip") or "").strip()
            status = str(row.get("status") or "")
            if status == "active" and ip:
                out[iid] = ip
        if len(out) == len(wanted):
            return out
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for instance IPs: {sorted(wanted - set(out))}")


def _ssh(ssh_key_path: Path, host: str, command: str) -> None:
    _shell(
        [
            "ssh",
            "-i",
            str(ssh_key_path),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
            f"ubuntu@{host}",
            command,
        ]
    )


def _scp(ssh_key_path: Path, local_path: Path, host: str, remote_path: str) -> None:
    _shell(["scp", "-i", str(ssh_key_path), str(local_path), f"ubuntu@{host}:{remote_path}"])


def _remote_gpu_telemetry_summary_lines() -> List[str]:
    return [
        "summarize_gpu_telemetry() {",
        "  mkdir -p \"${HOME}/cloud-eval-logs\"",
        "  python3 - <<'PY'",
        "import csv, glob, json, math, os, statistics",
        "from datetime import datetime, timezone",
        "",
        "log_dir = os.path.expanduser('~/cloud-eval-logs')",
        "csv_paths = sorted(glob.glob(os.path.join(log_dir, 'gpu-smi*.csv')))",
        "summary_json = os.path.join(log_dir, 'gpu-telemetry-summary.json')",
        "summary_md = os.path.join(log_dir, 'gpu-telemetry-summary.md')",
        "",
        "def as_float(value):",
        "    text = str(value or '').strip()",
        "    if text in {'', 'N/A', 'Not Supported', '[Not Supported]'}:",
        "        return None",
        "    try:",
        "        return float(text)",
        "    except ValueError:",
        "        return None",
        "",
        "def percentile(values, p):",
        "    vals = sorted(v for v in values if v is not None)",
        "    if not vals:",
        "        return None",
        "    if len(vals) == 1:",
        "        return vals[0]",
        "    k = (len(vals) - 1) * p",
        "    lo = math.floor(k)",
        "    hi = math.ceil(k)",
        "    if lo == hi:",
        "        return vals[int(k)]",
        "    return vals[lo] * (hi - k) + vals[hi] * (k - lo)",
        "",
        "def rounded(value):",
        "    return None if value is None else round(value, 2)",
        "",
        "def summarize_rows(rows):",
        "    gpu = [row['gpu_util'] for row in rows if row['gpu_util'] is not None]",
        "    mem_util = [row['mem_util'] for row in rows if row['mem_util'] is not None]",
        "    mem_used = [row['mem_used'] for row in rows if row['mem_used'] is not None]",
        "    mem_total = [row['mem_total'] for row in rows if row['mem_total'] is not None]",
        "    power = [row['power'] for row in rows if row['power'] is not None]",
        "    temp = [row['temp'] for row in rows if row['temp'] is not None]",
        "    active_gpu = [v for v in gpu if v > 0]",
        "    return {",
        "        'samples': len(rows),",
        "        'first_timestamp': rows[0]['timestamp'] if rows else None,",
        "        'last_timestamp': rows[-1]['timestamp'] if rows else None,",
        "        'avg_gpu_util_pct': rounded(statistics.fmean(gpu)) if gpu else None,",
        "        'median_gpu_util_pct': rounded(statistics.median(gpu)) if gpu else None,",
        "        'p95_gpu_util_pct': rounded(percentile(gpu, 0.95)),",
        "        'max_gpu_util_pct': rounded(max(gpu)) if gpu else None,",
        "        'active_gpu_samples_pct': rounded(100 * len(active_gpu) / len(gpu)) if gpu else None,",
        "        'avg_mem_util_pct': rounded(statistics.fmean(mem_util)) if mem_util else None,",
        "        'max_mem_util_pct': rounded(max(mem_util)) if mem_util else None,",
        "        'avg_memory_used_mib': rounded(statistics.fmean(mem_used)) if mem_used else None,",
        "        'max_memory_used_mib': rounded(max(mem_used)) if mem_used else None,",
        "        'memory_total_mib': rounded(max(mem_total)) if mem_total else None,",
        "        'avg_power_w': rounded(statistics.fmean(power)) if power else None,",
        "        'max_power_w': rounded(max(power)) if power else None,",
        "        'max_temperature_c': rounded(max(temp)) if temp else None,",
        "    }",
        "",
        "workers = []",
        "for path in csv_paths:",
        "    rows = []",
        "    try:",
        "        with open(path, newline='', encoding='utf-8', errors='replace') as handle:",
        "            for row in csv.DictReader(handle):",
        "                if not row or str(row.get('timestamp', '')).startswith('nvidia-smi'):",
        "                    continue",
        "                parsed = {",
        "                    'timestamp': row.get('timestamp'),",
        "                    'gpu_util': as_float(row.get('utilization_gpu_pct')),",
        "                    'mem_util': as_float(row.get('utilization_memory_pct')),",
        "                    'mem_used': as_float(row.get('memory_used_mib')),",
        "                    'mem_total': as_float(row.get('memory_total_mib')),",
        "                    'power': as_float(row.get('power_draw_w')),",
        "                    'temp': as_float(row.get('temperature_gpu_c')),",
        "                }",
        "                if any(parsed[key] is not None for key in ('gpu_util', 'mem_util', 'mem_used', 'power', 'temp')):",
        "                    rows.append(parsed)",
        "    except OSError as exc:",
        "        workers.append({'csv': os.path.relpath(path, os.path.expanduser('~')), 'error': repr(exc)})",
        "        continue",
        "    workers.append({'csv': os.path.relpath(path, os.path.expanduser('~')), **summarize_rows(rows)})",
        "",
        "valid = [worker for worker in workers if worker.get('samples')]",
        "aggregate = {'workers_with_gpu_csv': len(valid), 'gpu_csv_files': len(csv_paths)}",
        "for key in ('avg_gpu_util_pct', 'max_gpu_util_pct', 'avg_memory_used_mib', 'max_memory_used_mib', 'avg_power_w', 'max_power_w'):",
        "    vals = [worker[key] for worker in valid if worker.get(key) is not None]",
        "    if vals:",
        "        aggregate[f'{key}_mean_across_workers'] = rounded(statistics.fmean(vals))",
        "        aggregate[f'{key}_max_across_workers'] = rounded(max(vals))",
        "",
        "payload = {",
        "    'generated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),",
        "    'artifact_label': os.environ.get('FE_ARTIFACT_LABEL', ''),",
        "    'instance_id': os.environ.get('FE_LAMBDA_INSTANCE_ID', ''),",
        "    'aggregate': aggregate,",
        "    'workers': workers,",
        "}",
        "with open(summary_json, 'w', encoding='utf-8') as handle:",
        "    json.dump(payload, handle, indent=2, sort_keys=True)",
        "    handle.write('\\n')",
        "",
        "lines = ['# GPU telemetry summary', '', f\"Generated: `{payload['generated_at_utc']}`\", f\"Artifact label: `{payload['artifact_label']}`\", '']",
        "if valid:",
        "    lines.extend(['| CSV | Samples | Avg GPU % | P95 GPU % | Max GPU % | Avg MiB | Max MiB | Avg W | Max W |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])",
        "    keys = ['csv', 'samples', 'avg_gpu_util_pct', 'p95_gpu_util_pct', 'max_gpu_util_pct', 'avg_memory_used_mib', 'max_memory_used_mib', 'avg_power_w', 'max_power_w']",
        "    for worker in workers:",
        "        values = {key: worker.get(key, '') for key in keys}",
        "        lines.append('| {csv} | {samples} | {avg_gpu_util_pct} | {p95_gpu_util_pct} | {max_gpu_util_pct} | {avg_memory_used_mib} | {max_memory_used_mib} | {avg_power_w} | {max_power_w} |'.format(**values))",
        "else:",
        "    lines.append('No parseable GPU telemetry samples were found.')",
        "with open(summary_md, 'w', encoding='utf-8') as handle:",
        "    handle.write('\\n'.join(str(line) for line in lines) + '\\n')",
        "print(f'[gpu-telemetry] summarized {len(valid)}/{len(csv_paths)} csv files to {summary_json}')",
        "PY",
        "}",
    ]


def _remote_lifecycle_prelude(
    *,
    api_base: str,
    api_key: str,
    instance_id: str,
    auto_terminate: bool,
    watchdog_enabled: bool,
    watchdog_idle_minutes: float,
    watchdog_check_minutes: float,
    watchdog_log_path: str,
    gpu_monitor_log_path: str,
    artifact_export_command: str,
    artifact_upload_every_steps: int,
    artifact_upload_every_minutes: float,
    require_artifact_export_before_terminate: bool,
    artifact_staging_dir: str,
    artifact_staging_is_durable: bool,
) -> List[str]:
    lines = [
        f"export FE_AUTO_TERMINATE={'1' if auto_terminate else '0'}",
        f"export FE_WATCHDOG_ENABLED={'1' if watchdog_enabled else '0'}",
        f"export FE_WATCHDOG_IDLE_SECONDS={max(1, int(float(watchdog_idle_minutes) * 60))}",
        f"export FE_WATCHDOG_CHECK_SECONDS={max(1, int(float(watchdog_check_minutes) * 60))}",
        f"export FE_WATCHDOG_LOG_PATH={shlex.quote(watchdog_log_path)}",
        f"export FE_GPU_MONITOR_LOG_PATH={shlex.quote(gpu_monitor_log_path)}",
        f"export FE_ARTIFACT_EXPORT_COMMAND={shlex.quote(artifact_export_command)}",
        f"export FE_ARTIFACT_UPLOAD_EVERY_STEPS={max(0, int(artifact_upload_every_steps))}",
        f"export FE_ARTIFACT_UPLOAD_EVERY_SECONDS={max(0, int(float(artifact_upload_every_minutes) * 60))}",
        f"export FE_REQUIRE_ARTIFACT_EXPORT_BEFORE_TERMINATE={'1' if require_artifact_export_before_terminate else '0'}",
        f"export FE_ARTIFACT_STAGING_DIR={shlex.quote(artifact_staging_dir)}",
        f"export FE_ARTIFACT_STAGING_IS_DURABLE={'1' if artifact_staging_is_durable else '0'}",
        "export FE_ARTIFACT_CHECKPOINT_COMMAND='/tmp/fe_collect_lambda_artifacts.sh checkpoint_${FE_ARTIFACT_COMPLETED_STEPS:-0}'",
        f"export FE_LAMBDA_CLOUD_BASE_URL={shlex.quote(api_base)}",
        f"export FE_LAMBDA_API_KEY={shlex.quote(api_key)}",
        f"export FE_LAMBDA_INSTANCE_ID={shlex.quote(instance_id)}",
        "write_artifact_collector() {",
        "  cat > /tmp/fe_collect_lambda_artifacts.sh <<'SH'",
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        *_remote_gpu_telemetry_summary_lines(),
        "label=${1:-checkpoint}",
        "safe_label=$(printf '%s' \"${label}\" | tr -c 'A-Za-z0-9_.-' '_')",
        "ts=$(date -u +%Y%m%dT%H%M%SZ)",
        "dest_dir=${FE_ARTIFACT_STAGING_DIR:-${HOME}/cloud-eval-artifacts}",
        "mkdir -p \"${dest_dir}\"",
        "tarball=\"${dest_dir}/${safe_label}-${FE_LAMBDA_INSTANCE_ID:-unknown}-${ts}.tar.gz\"",
        "paths=()",
        "add_path() {",
        "  if [ -e \"${HOME}/$1\" ]; then",
        "    paths+=(\"$1\")",
        "  fi",
        "}",
        "add_path cloud-eval-logs",
        "add_path fallen-empire-lora/benchmarks/results",
        "add_path fallen-empire-lora/training/adapter_registry_v1.json",
        "add_path fallen-empire-lora/scripts/launch_lambda_parallel_ablation.py",
        "add_path fallen-empire-lora/scripts/launch_lambda_council_eval.py",
        "add_path fallen-empire-lora/scripts/run_final_mass_testing_system.py",
        "add_path fallen-empire-lora/scripts/run_council_conversation_eval.py",
        "add_path fallen-empire-lora/scripts/run_game_benchmark.py",
        "add_path fallen-empire-lora/scripts/run_routing_benchmark.py",
        "add_path fallen-empire-lora/scripts/model_router.py",
        "add_path fallen-empire-lora/scripts/router",
        "add_path fallen-empire-lora/benchmarks/task_bank/compiled/final_mass_testing_system_v1.json",
        "export FE_ARTIFACT_LABEL=\"${label}\"",
        "summarize_gpu_telemetry || true",
        "if [ \"${#paths[@]}\" -eq 0 ]; then",
        "  echo \"[lambda-artifacts] no artifact paths found\"",
        "  exit 1",
        "fi",
        "tar --warning=no-file-changed --ignore-failed-read -czf \"${tarball}\" -C \"${HOME}\" \"${paths[@]}\"",
        "echo \"[lambda-artifacts] collected ${tarball}\"",
        "if [ -n \"${FE_ARTIFACT_EXPORT_COMMAND:-}\" ]; then",
        "  FE_ARTIFACT_TARBALL=\"${tarball}\" FE_ARTIFACT_LABEL=\"${label}\" bash -lc \"${FE_ARTIFACT_EXPORT_COMMAND}\"",
        "  echo \"[lambda-artifacts] exported ${tarball}\"",
        "else",
        "  echo \"[lambda-artifacts] no FE_ARTIFACT_EXPORT_COMMAND configured; artifact remains on instance\"",
        "  if [ \"${FE_ARTIFACT_STAGING_IS_DURABLE:-0}\" = \"1\" ]; then",
        "    echo \"[lambda-artifacts] staging directory is durable; treating local artifact as exported\"",
        "    exit 0",
        "  fi",
        "  case \"${label}\" in",
        "    cleanup_*|watchdog_*|final*)",
        "      if [ \"${FE_REQUIRE_ARTIFACT_EXPORT_BEFORE_TERMINATE:-1}\" = \"1\" ]; then",
        "        exit 2",
        "      fi",
        "      ;;",
        "  esac",
        "fi",
        "SH",
        "  chmod 700 /tmp/fe_collect_lambda_artifacts.sh",
        "}",
        "collect_lambda_artifacts() {",
        "  /tmp/fe_collect_lambda_artifacts.sh \"${1:-checkpoint}\"",
        "}",
        "terminate_lambda_instance() {",
        "  reason=${1:-cleanup}",
        "  if collect_lambda_artifacts \"${reason}\"; then",
        "    artifact_status=0",
        "  else",
        "    artifact_status=$?",
        "    echo \"[lambda-terminate] artifact export failed status=${artifact_status} reason=${reason}\"",
        "    if [ \"${FE_REQUIRE_ARTIFACT_EXPORT_BEFORE_TERMINATE}\" = \"1\" ]; then",
        "      echo \"[lambda-terminate] skipping termination to preserve instance artifacts\"",
        "      return \"${artifact_status}\"",
        "    fi",
        "  fi",
        "  echo \"[lambda-terminate] ${reason}: terminating ${FE_LAMBDA_INSTANCE_ID}\"",
        "  python3 - <<'PY' || true",
        "import base64, json, os, urllib.request",
        "api_base = os.environ['FE_LAMBDA_CLOUD_BASE_URL'].rstrip('/')",
        "api_key = os.environ['FE_LAMBDA_API_KEY']",
        "instance_id = os.environ['FE_LAMBDA_INSTANCE_ID']",
        "body = json.dumps({'instance_ids': [instance_id]}).encode('utf-8')",
        "token = base64.b64encode(f'{api_key}:'.encode('utf-8')).decode('utf-8')",
        "req = urllib.request.Request(",
        "    api_base + '/instance-operations/terminate',",
        "    data=body,",
        "    method='POST',",
        "    headers={",
        "        'Authorization': f'Basic {token}',",
        "        'Accept': 'application/json',",
        "        'Content-Type': 'application/json',",
        "        'User-Agent': 'fe-lambda-remote-cleanup/1.0',",
        "    },",
        ")",
        "with urllib.request.urlopen(req, timeout=30) as resp:",
        "    print(resp.read().decode('utf-8', errors='replace'))",
        "PY",
        "}",
        "start_lambda_watchdog() {",
        "  if [ \"${FE_WATCHDOG_ENABLED}\" != \"1\" ]; then",
        "    return 0",
        "  fi",
        "  echo \"[lambda-watchdog] monitoring ${FE_WATCHDOG_LOG_PATH}; idle_limit=${FE_WATCHDOG_IDLE_SECONDS}s check=${FE_WATCHDOG_CHECK_SECONDS}s\"",
        "  (",
        "    started_at=$(date +%s)",
        "    while true; do",
        "      sleep \"${FE_WATCHDOG_CHECK_SECONDS}\"",
        "      now=$(date +%s)",
        "      last_change=$(stat -c %Y \"${FE_WATCHDOG_LOG_PATH}\" 2>/dev/null || echo \"${started_at}\")",
        "      idle=$((now - last_change))",
        "      if [ \"${idle}\" -ge \"${FE_WATCHDOG_IDLE_SECONDS}\" ]; then",
        "        echo \"[lambda-watchdog] log idle for ${idle}s; terminating ${FE_LAMBDA_INSTANCE_ID}\"",
        "        if terminate_lambda_instance \"watchdog_idle_${idle}s\"; then",
        "          exit 0",
        "        fi",
        "      fi",
        "    done",
        "  ) &",
        "  FE_WATCHDOG_PID=$!",
        "  export FE_WATCHDOG_PID",
        "}",
        "start_gpu_monitor() {",
        "  mkdir -p \"$(dirname \"${FE_GPU_MONITOR_LOG_PATH}\")\"",
        "  echo \"timestamp,index,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c\" > \"${FE_GPU_MONITOR_LOG_PATH}\"",
        "  if ! command -v nvidia-smi >/dev/null 2>&1; then",
        "    echo \"nvidia-smi unavailable\" >> \"${FE_GPU_MONITOR_LOG_PATH}\"",
        "    return 0",
        "  fi",
        "  (",
        "    while true; do",
        "      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits >> \"${FE_GPU_MONITOR_LOG_PATH}\" 2>&1 || true",
        "      sleep 5",
        "    done",
        "  ) &",
        "  FE_GPU_MONITOR_PID=$!",
        "  export FE_GPU_MONITOR_PID",
        "  echo \"[gpu-monitor] writing ${FE_GPU_MONITOR_LOG_PATH} pid=${FE_GPU_MONITOR_PID}\"",
        "}",
        "cleanup() {",
        "  status=$?",
        "  if [ -n \"${FE_WATCHDOG_PID:-}\" ]; then",
        "    kill \"${FE_WATCHDOG_PID}\" >/dev/null 2>&1 || true",
        "  fi",
        "  if [ -n \"${FE_GPU_MONITOR_PID:-}\" ]; then",
        "    kill \"${FE_GPU_MONITOR_PID}\" >/dev/null 2>&1 || true",
        "  fi",
        "  if [ \"${FE_AUTO_TERMINATE}\" = \"1\" ]; then",
        "    echo \"[lambda-cleanup] terminating ${FE_LAMBDA_INSTANCE_ID} after exit status ${status}\"",
        "    terminate_lambda_instance \"cleanup_exit_${status}\"",
        "  fi",
        "  exit \"$status\"",
        "}",
        "trap cleanup EXIT",
        "write_artifact_collector",
        "start_gpu_monitor",
        "start_lambda_watchdog",
    ]
    return lines


def _bootstrap_worker(ssh_key_path: Path, host: str, file_system_mount_point: str = "") -> None:
    fs_check = ""
    if file_system_mount_point:
        fs_check = (
            f"test -d {shlex.quote(file_system_mount_point)}; "
            f"mountpoint -q {shlex.quote(file_system_mount_point)}; "
        )
    cmd = (
        "set -euo pipefail; "
        + fs_check
        +
        "mkdir -p ~/fallen-empire-lora ~/fallen-empire ~/cloud-eval-logs; "
        "sudo apt-get update -y; "
        "sudo apt-get install -y curl ca-certificates gnupg tmux rsync; "
        "if ! command -v node >/dev/null 2>&1 || [ \"$(node -v | tr -d v | cut -d. -f1)\" -lt 20 ]; then "
        "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -; "
        "  sudo apt-get install -y nodejs; "
        "fi"
    )
    _ssh(ssh_key_path, host, cmd)


def _sync_repos(ssh_key_path: Path, host: str, ml_repo: Path, game_repo: Path) -> None:
    _shell(
        [
            "rsync",
            "-az",
            "--delete",
            *_rsync_timeout_args(),
            "-e",
            _rsync_ssh_command(ssh_key_path),
            "--exclude",
            ".venv",
            "--exclude",
            "benchmarks/results",
            "--exclude",
            "checkpoints",
            "--exclude",
            "models",
            "--exclude",
            "data/raw",
            "--exclude",
            "data/lora",
            "--exclude",
            ".git",
            str(ml_repo) + "/",
            f"ubuntu@{host}:~/fallen-empire-lora/",
        ]
    )
    _shell(
        [
            "rsync",
            "-az",
            "--delete",
            *_rsync_timeout_args(),
            "-e",
            _rsync_ssh_command(ssh_key_path),
            "--exclude",
            ".venv",
            "--exclude",
            "node_modules",
            "--exclude",
            ".next",
            str(game_repo) + "/",
            f"ubuntu@{host}:~/fallen-empire/",
        ]
    )
    _ssh(
        ssh_key_path,
        host,
        "set -euo pipefail; cd ~/fallen-empire; npm ci; "
        "sudo npm install -g ts-node tsconfig-paths typescript; "
        "cd ~/fallen-empire-lora; python3 -m venv .venv-linux-port; "
        "source .venv-linux-port/bin/activate; "
        "pip install -U pip; "
        "pip install torch --index-url https://download.pytorch.org/whl/cu121; "
        "pip install transformers accelerate sentencepiece numpy",
    )


def _adapter_path_for_id(registry_path: Path, adapter_id: str) -> Path | None:
    wanted = str(adapter_id or "").strip()
    if not wanted:
        return None
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    for row in data.get("entries") or []:
        if str(row.get("adapter_id") or "") == wanted:
            raw = str(row.get("adapter_path") or "").strip()
            if raw:
                return Path(raw)
    return None


def _registry_adapter_ids(registry_path: Path) -> List[str]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    out: List[str] = []
    for row in data.get("entries") or []:
        adapter_id = str(row.get("adapter_id") or "").strip()
        if adapter_id and bool(row.get("is_specialized")):
            out.append(adapter_id)
    return out


def _sync_adapter_checkpoint(ssh_key_path: Path, host: str, ml_repo: Path, adapter_id: str) -> None:
    rel_path = _adapter_path_for_id(ml_repo / "training" / "adapter_registry_v1.json", adapter_id)
    if rel_path is None:
        raise RuntimeError(f"No adapter_path found for adapter_id={adapter_id!r}")
    local_path = (ml_repo / rel_path).resolve()
    source_path = local_path
    remote_dest = f"ubuntu@{host}:/home/ubuntu/fallen-empire-lora/{rel_path.parent.as_posix()}/"
    if not local_path.exists():
        candidates_root = (ml_repo / rel_path.parent).resolve()
        candidates = sorted([child for child in candidates_root.iterdir() if child.is_dir()]) if candidates_root.exists() else []
        if not candidates:
            raise RuntimeError(f"Adapter checkpoint path does not exist: {local_path}")
        source_path = candidates[-1]
        print(
            f"adapter_sync_fallback adapter_id={adapter_id} expected={rel_path} source={source_path.relative_to(ml_repo)}",
            flush=True,
        )
        remote_parent = f"/home/ubuntu/fallen-empire-lora/{rel_path.as_posix()}"
        _ssh(ssh_key_path, host, f"mkdir -p {shlex.quote(remote_parent)}")
        remote_dest = f"ubuntu@{host}:/home/ubuntu/fallen-empire-lora/{rel_path.as_posix()}/"
        local_source = str(source_path) + "/"
    else:
        remote_parent = f"/home/ubuntu/fallen-empire-lora/{rel_path.parent.as_posix()}"
        _ssh(ssh_key_path, host, f"mkdir -p {shlex.quote(remote_parent)}")
        local_source = str(source_path)
    _shell(
        [
            "rsync",
            "-az",
            *_rsync_timeout_args(),
            "-e",
            _rsync_ssh_command(ssh_key_path),
            local_source,
            remote_dest,
        ]
    )


def _write_and_start_cell_runner(
    ssh_key_path: Path,
    host: str,
    instance_id: str,
    cell: MatrixCell,
    *,
    api_base: str,
    api_key: str,
    auto_terminate: bool,
    watchdog_enabled: bool,
    watchdog_idle_minutes: float,
    watchdog_check_minutes: float,
    artifact_export_command: str,
    artifact_upload_every_steps: int,
    artifact_upload_every_minutes: float,
    require_artifact_export_before_terminate: bool,
    artifact_staging_dir: str,
    artifact_staging_is_durable: bool,
    source_repo: str,
    worktree_root_base: str,
    variant: str,
    max_tasks: int,
    task_domains: List[str],
    single_specialist_adapter_id: str,
    rag_current_corpus: str,
) -> None:
    row_path = f"benchmarks/results/cloud_ablation_rows_{cell.name}.jsonl"
    summary_json = f"benchmarks/results/cloud_ablation_summary_{cell.name}.json"
    summary_md = f"benchmarks/results/cloud_ablation_summary_{cell.name}.md"
    tasks_json = f"benchmarks/results/cloud_ablation_runtime_tasks_{cell.name}.json"
    worktree_root = f"{worktree_root_base}-{cell.name}"
    log_path = f"/home/ubuntu/cloud-eval-logs/fe-ablation-{cell.name}.log"
    gpu_monitor_log_path = f"/home/ubuntu/cloud-eval-logs/gpu-smi-{cell.name}.csv"

    cmd_parts = [
        "python scripts/run_final_mass_testing_system.py",
        f"--source-repo {source_repo}",
        f"--worktree-root {worktree_root}",
        f"--variants {variant}",
        f"--rows-jsonl {row_path}",
        f"--summary-json {summary_json}",
        f"--summary-md {summary_md}",
        f"--runtime-tasks-json {tasks_json}",
    ]
    if max_tasks > 0:
        cmd_parts.append(f"--max-tasks {int(max_tasks)}")
    for domain in task_domains:
        cmd_parts.append(f"--task-domain {shlex.quote(domain)}")
    if single_specialist_adapter_id:
        cmd_parts.append(f"--single-specialist-adapter-id {shlex.quote(single_specialist_adapter_id)}")
    extra = []
    i = 0
    while i < len(cell.extra_args):
        token = cell.extra_args[i]
        if token == DEFAULT_RAG_CURRENT_CORPUS:
            extra.append(rag_current_corpus)
        else:
            extra.append(token)
        i += 1
    if extra:
        cmd_parts.extend(extra)
    command = " ".join(cmd_parts)

    remote_script = f"/tmp/run_{cell.name}_ablation.sh"
    local_script = Path(f"/tmp/run_{cell.name}_ablation.sh")
    local_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *_remote_lifecycle_prelude(
                    api_base=api_base,
                    api_key=api_key,
                    instance_id=instance_id,
                    auto_terminate=auto_terminate,
                    watchdog_enabled=watchdog_enabled,
                    watchdog_idle_minutes=watchdog_idle_minutes,
                    watchdog_check_minutes=watchdog_check_minutes,
                    watchdog_log_path=log_path,
                    gpu_monitor_log_path=gpu_monitor_log_path,
                    artifact_export_command=artifact_export_command,
                    artifact_upload_every_steps=artifact_upload_every_steps,
                    artifact_upload_every_minutes=artifact_upload_every_minutes,
                    require_artifact_export_before_terminate=require_artifact_export_before_terminate,
                    artifact_staging_dir=artifact_staging_dir,
                    artifact_staging_is_durable=artifact_staging_is_durable,
                ),
                "cd ~/fallen-empire-lora",
                "source .venv-linux-port/bin/activate",
                "export LOCAL_BACKEND=transformers",
                f"bash -lc {shlex.quote(command)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        _scp(ssh_key_path, local_script, host, remote_script)
    finally:
        local_script.unlink(missing_ok=True)
    _ssh(
        ssh_key_path,
        host,
        (
            f"chmod 700 {remote_script}; "
            f"tmux kill-session -t fe-ablation-{cell.name} >/dev/null 2>&1 || true; "
            f"tmux new-session -d -s fe-ablation-{cell.name} '{remote_script} > {log_path} 2>&1'"
        ),
    )


def _run_workers_parallel(
    label: str,
    items: List[tuple[str, Any]],
    fn: Any,
    *,
    retries: int = 0,
    retry_sleep_s: float = 10.0,
) -> None:
    def run_item_with_retries(item: tuple[str, Any]) -> None:
        attempts = max(1, int(retries) + 1)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                fn(item)
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                print(
                    f"{label}_retry host={item[0]} attempt={attempt + 1}/{attempts} "
                    f"sleep_s={retry_sleep_s} error={exc}",
                    flush=True,
                )
                time.sleep(max(0.0, float(retry_sleep_s)))
        assert last_exc is not None
        raise last_exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(items))) as executor:
        future_to_item = {executor.submit(run_item_with_retries, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{label} failed for {item[0]}: {exc}") from exc
            print(f"{label}_ok host={item[0]}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch prompt ablation cells across Lambda workers in parallel.")
    parser.add_argument("--ml-repo", type=Path, default=ROOT)
    parser.add_argument("--game-repo", type=Path, default=Path.home() / "fallen-empire")
    parser.add_argument("--ssh-key-path", type=Path, default=Path.home() / ".ssh" / "lambda_cloud_cursor")
    parser.add_argument("--ssh-key-name", default="lambda-cloud-cursor")
    parser.add_argument(
        "--api-base",
        default=_lambda_cloud_base_url_from_env(),
        help="Lambda Cloud API base URL. Defaults to LAMBDA_CLOUD_BASE_URL, then legacy LAMBDA_API_BASE.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="", help="Lambda region. Defaults to attached file-system region, then us-east-1.")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument(
        "--fallback-region",
        action="append",
        default=[],
        help="Fallback Lambda region(s) for launch capacity. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--fallback-instance-type",
        action="append",
        default=[],
        help="Fallback Lambda instance type(s) for launch capacity. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--file-system-id",
        default=os.environ.get("LAMBDA_FILE_SYSTEM_ID", DEFAULT_LAMBDA_FILE_SYSTEM_ID),
        help="Lambda file system id/name to attach to launched instances. Empty disables attachment.",
    )
    parser.add_argument(
        "--file-system-name",
        default=os.environ.get("LAMBDA_FILE_SYSTEM_NAME", ""),
        help="Optional Lambda file system name override; resolved from --file-system-id when omitted.",
    )
    parser.add_argument(
        "--no-file-system",
        action="store_true",
        help="Disable default Lambda file system attachment.",
    )
    parser.add_argument("--name-prefix", default="fe-ablation")
    parser.add_argument("--launch-instances", action="store_true", help="Launch new Lambda instances for each cell.")
    parser.add_argument("--instance-ids", default="", help="Comma-separated existing instance ids to use.")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None, help="Terminate workers when their remote eval script exits.")
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false", help="Leave workers running after remote eval script exit.")
    parser.add_argument(
        "--cleanup-on-setup-failure",
        dest="cleanup_on_setup_failure",
        action="store_true",
        default=True,
        help="Terminate newly launched instances if setup fails before their remote eval runner starts.",
    )
    parser.add_argument(
        "--no-cleanup-on-setup-failure",
        dest="cleanup_on_setup_failure",
        action="store_false",
        help="Leave newly launched instances running when bootstrap/sync/adapter-sync/start setup fails.",
    )
    parser.add_argument("--watchdog", dest="watchdog_enabled", action="store_true", default=None, help="Terminate workers when their run log is idle for --watchdog-idle-minutes.")
    parser.add_argument("--no-watchdog", dest="watchdog_enabled", action="store_false", help="Disable idle-log watchdog termination.")
    parser.add_argument("--watchdog-idle-minutes", type=float, default=60.0, help="Idle log minutes before watchdog termination.")
    parser.add_argument("--watchdog-check-minutes", type=float, default=5.0, help="Minutes between watchdog log-idle checks.")
    parser.add_argument("--artifact-export-command", default=os.environ.get("FE_ARTIFACT_EXPORT_COMMAND", ""), help="Shell command that durably exports FE_ARTIFACT_TARBALL.")
    parser.add_argument(
        "--artifact-staging-dir",
        default=os.environ.get("FE_ARTIFACT_STAGING_DIR", ""),
        help="Remote artifact staging directory. Defaults to the Lambda file system mount when attached.",
    )
    parser.add_argument("--artifact-upload-every-steps", type=int, default=5, help="Run artifact checkpoint command every N completed rows; 0 disables step checkpoints.")
    parser.add_argument("--artifact-upload-every-minutes", type=float, default=10.0, help="Run artifact checkpoint command after N minutes since last checkpoint; 0 disables time checkpoints.")
    parser.add_argument("--require-artifact-export-before-terminate", dest="require_artifact_export_before_terminate", action="store_true", default=True, help="Skip auto-termination if final artifact export fails.")
    parser.add_argument("--allow-terminate-without-artifact-export", dest="require_artifact_export_before_terminate", action="store_false", help="Terminate even if final artifact export is missing or fails.")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-adapter-sync", action="store_true", help="Skip targeted adapter checkpoint rsync; use only when adapters are already present remotely.")
    parser.add_argument(
        "--setup-command-retries",
        type=int,
        default=1,
        help="Retry each failed bootstrap/sync/adapter-sync/start command this many times before setup cleanup terminates launched instances.",
    )
    parser.add_argument(
        "--setup-retry-sleep-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait before the one-shot setup command retry.",
    )
    parser.add_argument("--include-baseline", action="store_true", help="Also run the baseline/off prompt cell.")
    parser.add_argument(
        "--only-cell",
        choices=["baseline", "rag_current", "context_max_potential", "rag_current_plus_max_potential"],
        default="",
        help="Run exactly one matrix cell instead of the default ablation set.",
    )
    parser.add_argument(
        "--specialist-suite",
        action="store_true",
        help="Run one worker per game specialist, forcing each adapter against its mapped task domain.",
    )
    parser.add_argument(
        "--specialist-suite-all-tasks",
        action="store_true",
        help="With --specialist-suite, run each fixed specialist adapter over the full manifest instead of its mapped domain slice.",
    )
    parser.add_argument(
        "--specialist-id",
        action="append",
        default=[],
        help="Restrict --specialist-suite to one or more adapter ids. Repeatable.",
    )
    parser.add_argument("--variant", default="qwen_7_5b_only")
    parser.add_argument("--max-tasks", type=int, default=0, help="Pass through to run_final_mass_testing_system.py.")
    parser.add_argument(
        "--task-domain",
        action="append",
        default=[],
        help="Pass normalized task-domain filter to run_final_mass_testing_system.py. Repeatable.",
    )
    parser.add_argument(
        "--single-specialist-adapter-id",
        default="",
        help="Pass forced specialist id when --variant single_specialist_local is used.",
    )
    parser.add_argument("--source-repo-remote", default="~/fallen-empire")
    parser.add_argument("--worktree-root-base", default="~/fallen-empire-arena-ablation")
    parser.add_argument("--rag-current-corpus-path", default=DEFAULT_RAG_CURRENT_CORPUS)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("LAMBDA_API_KEY or --api-key is required.")

    cell_by_name = {cell.name: cell for cell in [BASELINE_CELL, *DEFAULT_CELLS]}
    if args.specialist_suite:
        if args.only_cell:
            raise SystemExit("--only-cell cannot be combined with --specialist-suite.")
        if args.variant != "qwen_7_5b_only":
            raise SystemExit("--specialist-suite controls the variant; omit --variant.")
        if args.single_specialist_adapter_id:
            raise SystemExit("--single-specialist-adapter-id cannot be combined with --specialist-suite.")
        if args.task_domain:
            raise SystemExit("--task-domain cannot be combined with --specialist-suite.")
        requested_specialists = [str(item).strip() for item in args.specialist_id if str(item).strip()]
        if not requested_specialists:
            requested_specialists = [
                adapter_id
                for adapter_id in _registry_adapter_ids(args.ml_repo / "training" / "adapter_registry_v1.json")
                if adapter_id in SPECIALIST_DOMAIN_MAP
            ]
        unknown = [adapter_id for adapter_id in requested_specialists if adapter_id not in SPECIALIST_DOMAIN_MAP]
        if unknown:
            raise SystemExit(
                "No game task-domain mapping for specialist ids: "
                + ", ".join(unknown)
                + ". Supported game specialists: "
                + ", ".join(sorted(SPECIALIST_DOMAIN_MAP))
            )
        cells = [
            MatrixCell(
                name=f"specialist_{adapter_id}",
                extra_args=[],
                adapter_id=adapter_id,
                task_domains=[] if args.specialist_suite_all_tasks else SPECIALIST_DOMAIN_MAP[adapter_id],
            )
            for adapter_id in requested_specialists
        ]
        run_variant = "single_specialist_local"
    elif args.only_cell:
        cells = [cell_by_name[args.only_cell]]
        run_variant = args.variant
    else:
        cells = ([BASELINE_CELL] if args.include_baseline else []) + list(DEFAULT_CELLS)
        run_variant = args.variant
    if args.max_tasks < 0:
        raise SystemExit("--max-tasks must be >= 0.")
    if run_variant == "single_specialist_local" and not args.specialist_suite and not str(args.single_specialist_adapter_id).strip():
        raise SystemExit("--single-specialist-adapter-id is required with --variant single_specialist_local.")
    file_system = {}
    file_system_names: List[str] = []
    file_system_mount_point = ""
    if not args.no_file_system:
        file_system_identifier = str(args.file_system_name or args.file_system_id or "").strip()
        if file_system_identifier:
            file_system = _resolve_file_system(args.api_base, args.api_key, file_system_identifier)
            file_system_names = [str(file_system.get("name") or file_system_identifier)]
            file_system_mount_point = str(file_system.get("mount_point") or "").strip()
    effective_region = str(args.region or "").strip()
    if not effective_region:
        effective_region = str(((file_system.get("region") or {}).get("name") if file_system else "") or "us-east-1")
    artifact_staging_dir = str(args.artifact_staging_dir or "").strip()
    if not artifact_staging_dir:
        if file_system_mount_point:
            artifact_staging_dir = f"{file_system_mount_point.rstrip('/')}/fallen-empire-lora-artifacts"
        else:
            artifact_staging_dir = "${HOME}/cloud-eval-artifacts"
    artifact_staging_is_durable = bool(file_system_mount_point and artifact_staging_dir.startswith(file_system_mount_point.rstrip("/")))
    instance_ids: List[str] = []
    ips: List[str] = []
    launched_instances = False
    selected_launch_candidate = LaunchCandidate(instance_type=str(args.instance_type).strip(), region=effective_region)
    started_instance_ids: set[str] = set()
    if args.instance_ids.strip():
        instance_ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]
        if len(instance_ids) != len(cells):
            raise SystemExit(f"--instance-ids must match cell count ({len(cells)}).")
    elif args.launch_instances:
        instance_ids, selected_launch_candidate = _launch_instances_with_fallback(
            args.api_base,
            args.api_key,
            requested_region=effective_region,
            requested_instance_type=args.instance_type,
            fallback_regions=_split_csv_cli_values(args.fallback_region),
            fallback_instance_types=_split_csv_cli_values(args.fallback_instance_type),
            ssh_key_name=args.ssh_key_name,
            quantity=len(cells),
            name_prefix=args.name_prefix,
            file_system_names=file_system_names,
        )
        launched_instances = True
    else:
        raise SystemExit("Provide --instance-ids or --launch-instances.")
    auto_terminate = args.auto_terminate
    if auto_terminate is None:
        auto_terminate = launched_instances
    watchdog_enabled = args.watchdog_enabled
    if watchdog_enabled is None:
        watchdog_enabled = auto_terminate
    if args.watchdog_idle_minutes <= 0:
        raise SystemExit("--watchdog-idle-minutes must be > 0.")
    if args.watchdog_check_minutes <= 0:
        raise SystemExit("--watchdog-check-minutes must be > 0.")
    if args.artifact_upload_every_steps < 0:
        raise SystemExit("--artifact-upload-every-steps must be >= 0.")
    if args.artifact_upload_every_minutes < 0:
        raise SystemExit("--artifact-upload-every-minutes must be >= 0.")
    if args.setup_command_retries < 0:
        raise SystemExit("--setup-command-retries must be >= 0.")
    if args.setup_retry_sleep_seconds < 0:
        raise SystemExit("--setup-retry-sleep-seconds must be >= 0.")

    try:
        id_to_ip = _wait_for_instance_ips(args.api_base, args.api_key, instance_ids)
        ips = [id_to_ip[iid] for iid in instance_ids]

        _run_workers_parallel(
            "bootstrap",
            [(host, None) for host in ips],
            lambda item: _bootstrap_worker(args.ssh_key_path, item[0], file_system_mount_point),
            retries=args.setup_command_retries,
            retry_sleep_s=args.setup_retry_sleep_seconds,
        )
        if not args.skip_sync:
            _run_workers_parallel(
                "sync",
                [(host, None) for host in ips],
                lambda item: _sync_repos(args.ssh_key_path, item[0], args.ml_repo, args.game_repo),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )
        adapter_sync_items = []
        if args.specialist_suite:
            adapter_sync_items = [
                (host, str(cell.adapter_id or ""))
                for host, cell in zip(ips, cells)
                if str(cell.adapter_id or "").strip()
            ]
        elif run_variant == "single_specialist_local" and str(args.single_specialist_adapter_id).strip():
            adapter_sync_items = [(host, str(args.single_specialist_adapter_id).strip()) for host in ips]
        if adapter_sync_items and not args.skip_adapter_sync:
            _run_workers_parallel(
                "adapter_sync",
                adapter_sync_items,
                lambda item: _sync_adapter_checkpoint(
                    args.ssh_key_path,
                    item[0],
                    args.ml_repo,
                    item[1],
                ),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )

        def start_worker(item: tuple[str, str, MatrixCell]) -> None:
            _write_and_start_cell_runner(
                args.ssh_key_path,
                item[0],
                item[1],
                item[2],
                api_base=args.api_base,
                api_key=args.api_key,
                auto_terminate=auto_terminate,
                watchdog_enabled=watchdog_enabled,
                watchdog_idle_minutes=args.watchdog_idle_minutes,
                watchdog_check_minutes=args.watchdog_check_minutes,
                artifact_export_command=args.artifact_export_command,
                artifact_upload_every_steps=args.artifact_upload_every_steps,
                artifact_upload_every_minutes=args.artifact_upload_every_minutes,
                require_artifact_export_before_terminate=args.require_artifact_export_before_terminate,
                artifact_staging_dir=artifact_staging_dir,
                artifact_staging_is_durable=artifact_staging_is_durable,
                source_repo=args.source_repo_remote,
                worktree_root_base=args.worktree_root_base,
                variant=run_variant,
                max_tasks=args.max_tasks,
                task_domains=list(item[2].task_domains or args.task_domain),
                single_specialist_adapter_id=str(item[2].adapter_id or args.single_specialist_adapter_id or "").strip(),
                rag_current_corpus=args.rag_current_corpus_path,
            )
            started_instance_ids.add(item[1])

        _run_workers_parallel(
            "start",
            [(host, iid, cell) for cell, iid, host in zip(cells, instance_ids, ips)],
            start_worker,
            retries=args.setup_command_retries,
            retry_sleep_s=args.setup_retry_sleep_seconds,
        )
    except Exception:
        if launched_instances and args.cleanup_on_setup_failure:
            cleanup_ids = [iid for iid in instance_ids if iid not in started_instance_ids]
            _terminate_instances_best_effort(
                args.api_base,
                args.api_key,
                cleanup_ids,
                reason="setup_failure_before_remote_start",
            )
        raise

    print("parallel_ablation_started")
    print(f"auto_terminate={int(bool(auto_terminate))}")
    print(f"watchdog_enabled={int(bool(watchdog_enabled))}")
    print(f"watchdog_idle_minutes={args.watchdog_idle_minutes}")
    print(f"watchdog_check_minutes={args.watchdog_check_minutes}")
    print(f"artifact_export_configured={int(bool(args.artifact_export_command.strip()))}")
    print(f"artifact_staging_dir={artifact_staging_dir}")
    print(f"artifact_staging_is_durable={int(bool(artifact_staging_is_durable))}")
    print(f"artifact_upload_every_steps={args.artifact_upload_every_steps}")
    print(f"artifact_upload_every_minutes={args.artifact_upload_every_minutes}")
    print(f"require_artifact_export_before_terminate={int(bool(args.require_artifact_export_before_terminate))}")
    print(f"cleanup_on_setup_failure={int(bool(args.cleanup_on_setup_failure))}")
    print(f"setup_command_retries={args.setup_command_retries}")
    print(f"setup_retry_sleep_seconds={args.setup_retry_sleep_seconds}")
    print(f"region={effective_region}")
    print(f"selected_launch_region={selected_launch_candidate.region}")
    print(f"instance_type={args.instance_type}")
    print(f"selected_launch_instance_type={selected_launch_candidate.instance_type}")
    print(f"fallback_regions={','.join(_split_csv_cli_values(args.fallback_region))}")
    print(f"fallback_instance_types={','.join(_split_csv_cli_values(args.fallback_instance_type))}")
    print(f"file_system_names={','.join(file_system_names)}")
    print(f"file_system_mount_point={file_system_mount_point}")
    print(f"specialist_suite={int(bool(args.specialist_suite))}")
    print(f"variant={run_variant}")
    print(f"max_tasks={args.max_tasks}")
    print(f"task_domains={','.join(args.task_domain)}")
    print(f"single_specialist_adapter_id={args.single_specialist_adapter_id}")
    for cell, iid, host in zip(cells, instance_ids, ips):
        details = ""
        if cell.adapter_id:
            details = f"\tadapter_id={cell.adapter_id}\ttask_domains={','.join(cell.task_domains or [])}"
        print(f"{cell.name}\tinstance_id={iid}\thost={host}\tsession=fe-ablation-{cell.name}{details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

