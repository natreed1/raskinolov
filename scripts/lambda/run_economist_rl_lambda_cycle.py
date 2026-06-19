#!/usr/bin/env python3
"""DEPRECATED standalone serial-cycle runner for economistRL.

Do not use this module as the top-level Lambda RL entry point for new runs.
Use ``scripts/lambda/run_economist_rl_split_workers.py`` through
``scripts/launch_economist_rl_lambda_split_workers.py`` instead.

This file remains as an internal rollout/evidence/scoring primitive and as a
debug/historical serial PPO runner.

Cycle shape (PPO):

1. Rollout all tasks and capture old log-prob summaries.
2. Evidence runner attaches simulation/test/compile evidence per task.
3. Execution evidence applies output to a disposable repo and runs compile/tests (Design A).
4. Reward engine scores each rollout.
5. PPO updates LoRA weights in an isolated subprocess (reads ``scored_batch_*.jsonl``).
6. Eval telemetry (registry is never auto-updated).

Training never replays static reference answers in PPO mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from economist_rl_coding_contract import CODING_SYSTEM_PROMPT  # noqa: E402
from economist_rl_task_execution import (  # noqa: E402
    DEFAULT_ECONOMIST_RL_INIT_ADAPTER,
    DEFAULT_GENERATION_MAX_TOKENS,
    build_rollout_user_prompt,
    resolve_eval_tasks,
)
from economist_rl_evidence_runner import (  # noqa: E402
    EvidenceRunnerConfig,
    attach_batch_evidence,
    write_evidence_jsonl,
)
from economist_rl_execution_evidence import (  # noqa: E402
    ExecutionEvidenceConfig,
    ExecutionWorktreePool,
    attach_batch_execution_evidence,
    execution_config_summary,
    resolve_execution_config,
    resolve_source_repo,
)
from economist_rl_ppo_trainer import (  # noqa: E402
    PPOConfig,
    attach_old_logprob_to_rollout,
    candidate_has_ppo_training_marker,
    candidate_staging_dir,
    collect_old_logprob_proxy_errors,
    discard_candidate_staging,
    train_ppo_batch,
    write_ppo_manifest,
)
from economist_rl_peft import adapter_dir_has_weights  # noqa: E402
from economist_rl_reward_engine import _compile_status, score_output  # noqa: E402
from model_router import ChatMessage, GenerationRequest, LocalMlxBackend  # noqa: E402

DEFAULT_REGISTRY = REPO / "training" / "adapter_registry_v1.json"
DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
PPO_TRAIN_SCRIPT = REPO / "scripts" / "lambda" / "run_economist_rl_ppo_train.py"
RUNNER_TAG = "rl_lambda_runner"
RUNNER_DISPLAY_NAME = "Deprecated Serial RL Lambda Runner"
DEPRECATION_NOTICE = (
    "DEPRECATED: run_economist_rl_lambda_cycle.py is a standalone serial-cycle runner. "
    "Use run_economist_rl_split_workers.py / launch_economist_rl_lambda_split_workers.py "
    "for Lambda RL runs."
)
DEFAULT_STEADY_ROLLOUTS_PER_CYCLE = 25
DEFAULT_BOOTSTRAP_ROLLOUTS_PER_CYCLE = 50
DEFAULT_WORKER_PPO_MIN_SAMPLES = 25
DEFAULT_WORKER_PPO_MAX_SAMPLES = 64


@dataclass(frozen=True)
class SpecializationConfig:
    tag: str
    adapter_id: str
    display_name: str
    task_db: Path
    eval_set: Path
    output_root: Path
    results_root: Path
    data_root: Path
    task_bank_schema: str
    scorer: str
    system_prompt: str
    train_mode: str
    evidence_runner: str
    ppo_trainer: str
    evidence_runner_config: EvidenceRunnerConfig
    ppo_config: PPOConfig


_V3_TASK_DB = REPO / "benchmarks" / "economistRL_tasks_v3_execution.json"
_V2_TASK_DB = REPO / "benchmarks" / "economistRL_tasks_v2_coding.json"
ECONOMIST_RL_TASK_DB = _V3_TASK_DB if _V3_TASK_DB.is_file() else _V2_TASK_DB
DEFAULT_EVAL_MANIFEST = REPO / "benchmarks" / "economistRL_eval_manifest_v1.json"
SPECIALIZATIONS: dict[str, SpecializationConfig] = {
    "economist_rl": SpecializationConfig(
        tag="economist_rl",
        adapter_id="economistRL",
        display_name="economistRL economy reward specialization",
        task_db=ECONOMIST_RL_TASK_DB,
        eval_set=ECONOMIST_RL_TASK_DB,
        output_root=REPO / "checkpoints" / "adapters" / "economistRL",
        results_root=REPO / "benchmarks" / "results" / "economistRL",
        data_root=REPO / "data" / "economistRL",
        task_bank_schema="economist_rl_task_bank_v1",
        scorer="economist_rl_reward_engine.score_output",
        system_prompt=CODING_SYSTEM_PROMPT,
        train_mode="ppo",
        evidence_runner="economist_rl_evidence_runner.attach_batch_evidence",
        ppo_trainer="economist_rl_ppo_trainer.train_ppo_batch",
        evidence_runner_config=EvidenceRunnerConfig(),
        ppo_config=PPOConfig(),
    )
}


@dataclass(frozen=True)
class RegistryAdapter:
    adapter_id: str
    adapter_path: str
    resolved_path: Path
    exists: bool
    entry: dict[str, Any]


@dataclass(frozen=True)
class CyclePaths:
    cycle_id: int
    candidate_adapter: Path
    rollout_file: Path
    evidence_file: Path
    scored_file: Path
    ppo_manifest_file: Path
    eval_file: Path
    manifest_file: Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_task_payload(path: Path, spec: SpecializationConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        payload = {"schema_version": spec.task_bank_schema, "adapter_id": spec.adapter_id, "tasks": payload}
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise SystemExit(f"Task database missing `tasks` list: {path}")
    return payload, [task for task in tasks if isinstance(task, dict) and str(task.get("id") or "").strip()]


def _release_accelerator_memory() -> None:
    """Best-effort VRAM release in the parent before spawning PPO subprocess."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def _discard_failed_candidate_adapter(candidate_adapter: Path) -> None:
    """Remove aborted PPO outputs so failed cycles do not look trainable."""
    discard_candidate_staging(candidate_staging_dir(candidate_adapter))
    if candidate_adapter.exists():
        shutil.rmtree(candidate_adapter, ignore_errors=True)


def _ppo_subprocess_log_paths(ppo_manifest_file: Path) -> tuple[Path, Path]:
    stem = ppo_manifest_file.with_suffix("")
    return stem.with_name(f"{stem.name}.stdout.log"), stem.with_name(f"{stem.name}.stderr.log")


def _ppo_subprocess_failure_manifest(
    *,
    scored_file: Path,
    request_path: Path,
    ppo_manifest_file: Path,
    exit_code: int,
    reason: str,
) -> dict[str, Any]:
    stdout_log, stderr_log = _ppo_subprocess_log_paths(ppo_manifest_file)
    payload: dict[str, Any] = {
        "status": "failed",
        "reason": reason,
        "exit_code": int(exit_code),
        "scored_file": str(scored_file),
        "ppo_request_file": str(request_path),
        "subprocess_isolated": True,
        "ppo_stdout_log": str(stdout_log),
        "ppo_stderr_log": str(stderr_log),
    }
    if stderr_log.is_file():
        tail = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        if tail:
            payload["stderr_tail"] = tail
    if stdout_log.is_file():
        tail = stdout_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        if tail:
            payload["stdout_tail"] = tail
    return payload


def _ppo_config_to_dict(cfg: PPOConfig) -> dict[str, Any]:
    return asdict(cfg)


def run_ppo_train_subprocess(
    *,
    scored_file: Path,
    system_prompt: str,
    base_model: str,
    source_adapter: Path,
    candidate_adapter: Path,
    ppo_manifest_file: Path,
    ppo_config: PPOConfig,
    train_backend: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    scored_file = scored_file.expanduser().resolve()
    if not scored_file.is_file():
        raise SystemExit(f"Cannot spawn PPO subprocess; scored file missing: {scored_file}")

    request_path = ppo_manifest_file.with_name(f"{ppo_manifest_file.stem}.request.json")
    request = {
        "scored_file": str(scored_file),
        "ppo_manifest_file": str(ppo_manifest_file.expanduser().resolve()),
        "system_prompt": system_prompt,
        "base_model": base_model,
        "source_adapter": str(source_adapter.expanduser().resolve()),
        "candidate_adapter": str(candidate_adapter.expanduser().resolve()),
        "ppo_config": _ppo_config_to_dict(ppo_config),
        "train_backend": train_backend,
        "dry_run": dry_run,
    }
    _write_json(request_path, request)
    _release_accelerator_memory()

    ppo_manifest_file = ppo_manifest_file.expanduser().resolve()
    ppo_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    stdout_log, stderr_log = _ppo_subprocess_log_paths(ppo_manifest_file)

    cmd = [sys.executable, str(PPO_TRAIN_SCRIPT), "--request", str(request_path)]
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=os.environ.copy(),
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    if completed.returncode != 0:
        _discard_failed_candidate_adapter(candidate_adapter.expanduser().resolve())
        return _ppo_subprocess_failure_manifest(
            scored_file=scored_file,
            request_path=request_path,
            ppo_manifest_file=ppo_manifest_file,
            exit_code=int(completed.returncode),
            reason="subprocess_exit",
        )
    if not ppo_manifest_file.is_file():
        _discard_failed_candidate_adapter(candidate_adapter.expanduser().resolve())
        return _ppo_subprocess_failure_manifest(
            scored_file=scored_file,
            request_path=request_path,
            ppo_manifest_file=ppo_manifest_file,
            exit_code=int(completed.returncode),
            reason="missing_ppo_manifest",
        )
    manifest = _read_json(ppo_manifest_file)
    if manifest.get("status") != "trained":
        _discard_failed_candidate_adapter(candidate_adapter.expanduser().resolve())
        failure = _ppo_subprocess_failure_manifest(
            scored_file=scored_file,
            request_path=request_path,
            ppo_manifest_file=ppo_manifest_file,
            exit_code=int(completed.returncode),
            reason="ppo_manifest_not_trained",
        )
        failure["ppo_status"] = manifest.get("status")
        return failure
    manifest["subprocess_isolated"] = True
    manifest["ppo_request_file"] = str(request_path)
    manifest["ppo_stdout_log"] = str(stdout_log)
    manifest["ppo_stderr_log"] = str(stderr_log)
    return manifest


def _resolve_ppo_config(spec: SpecializationConfig, args: argparse.Namespace) -> PPOConfig:
    overrides: dict[str, Any] = {}
    if args.ppo_epochs is not None:
        overrides["ppo_epochs"] = int(args.ppo_epochs)
    if args.ppo_min_samples is not None:
        overrides["min_samples"] = int(args.ppo_min_samples)
    elif not args.dry_run:
        overrides["min_samples"] = DEFAULT_WORKER_PPO_MIN_SAMPLES
    if getattr(args, "ppo_max_samples", None) is not None:
        overrides["max_samples"] = int(args.ppo_max_samples)
    elif args.dry_run:
        overrides["min_samples"] = 1
    else:
        overrides["max_samples"] = DEFAULT_WORKER_PPO_MAX_SAMPLES
    if getattr(args, "ppo_logprob_window_tokens", None) is not None:
        overrides["max_logprob_window_tokens"] = int(args.ppo_logprob_window_tokens)
    if getattr(args, "ppo_logprob_window_strategy", None) is not None:
        overrides["logprob_window_strategy"] = str(args.ppo_logprob_window_strategy)
    if getattr(args, "ppo_target_logprob_chunk_tokens", None) is not None:
        overrides["target_logprob_chunk_tokens"] = int(args.ppo_target_logprob_chunk_tokens)
    if getattr(args, "ppo_mini_batch_size", None) is not None:
        overrides["mini_batch_size"] = int(args.ppo_mini_batch_size)
    if not overrides:
        return spec.ppo_config
    return PPOConfig(**{**spec.ppo_config.__dict__, **overrides})


def _rollout_limit_for_cycle(args: argparse.Namespace, *, bootstrap_cycle: bool) -> int:
    if bootstrap_cycle:
        configured = getattr(args, "bootstrap_rollouts_per_cycle", None)
        if configured is not None:
            return int(configured)
    return int(args.rollouts_per_cycle)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (REPO / path).resolve()


def resolve_registry_adapter(registry_path: Path, adapter_name: str) -> RegistryAdapter:
    payload = _read_json(registry_path)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise SystemExit(f"Registry missing entries list: {registry_path}")
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("adapter_id") or "") == adapter_name:
            adapter_path = str(entry.get("adapter_path") or "").strip()
            resolved = _resolve_path(adapter_path) if adapter_path else Path("")
            return RegistryAdapter(
                adapter_id=adapter_name,
                adapter_path=adapter_path,
                resolved_path=resolved,
                exists=bool(adapter_path and resolved.exists()),
                entry=dict(entry),
            )
    raise SystemExit(f"Adapter {adapter_name!r} not found in registry: {registry_path}")


def adapter_from_resolved_path(adapter_id: str, resolved: Path) -> RegistryAdapter:
    """Build a registry adapter view from an on-disk LoRA directory."""
    resolved = resolved.expanduser().resolve()
    try:
        rel = str(resolved.relative_to(REPO))
    except ValueError:
        rel = str(resolved)
    return RegistryAdapter(
        adapter_id=adapter_id,
        adapter_path=rel,
        resolved_path=resolved,
        exists=adapter_dir_has_weights(resolved),
        entry={"adapter_id": adapter_id, "adapter_path": rel},
    )


def resolve_cycle_current_adapter(
    *,
    registry_path: Path,
    adapter_name: str,
    chain_from_previous: Path | None,
    init_adapter_path: Path | None = None,
) -> tuple[RegistryAdapter, RegistryAdapter, str]:
    """Return (registry_current, effective_current, load_reason) for this cycle.

    When ``--cycles > 1`` in one invocation, cycle 2+ continues from the prior
    cycle's ``rl_pass_*`` candidate. The adapter registry is read-only in this runner.
    """
    registry_current = resolve_registry_adapter(registry_path, adapter_name)
    if chain_from_previous is not None:
        chain = chain_from_previous.expanduser().resolve()
        if adapter_dir_has_weights(chain) and candidate_has_ppo_training_marker(chain):
            chained = adapter_from_resolved_path(adapter_name, chain)
            if registry_current.resolved_path.resolve() == chain:
                return registry_current, registry_current, "registry"
            return registry_current, chained, "previous_cycle_candidate"
        if adapter_dir_has_weights(chain):
            print(
                f"[adapter-chain] ignoring untrained candidate copy at {chain} "
                "(missing .economist_rl_ppo_trained marker)",
                flush=True,
            )
    if init_adapter_path is not None:
        init_path = init_adapter_path.expanduser().resolve()
        if not adapter_dir_has_weights(init_path):
            raise SystemExit(f"--init-adapter-path has no LoRA weights: {init_path}")
        init_adapter = adapter_from_resolved_path(adapter_name, init_path)
        return registry_current, init_adapter, "init_adapter_path"
    if chain_from_previous is not None:
        return registry_current, registry_current, "registry_missing_chain_weights"
    return registry_current, registry_current, "registry"


def next_cycle_id(results_root: Path, output_root: Path) -> int:
    seen: set[int] = set()
    for root, pattern in ((results_root / "manifests", "cycle_*_manifest.json"), (output_root, "rl_pass_*")):
        if not root.exists():
            continue
        for path in root.glob(pattern):
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            if digits:
                seen.add(int(digits))
    cycle = 1
    while cycle in seen:
        cycle += 1
    return cycle


def paths_for_cycle(cycle_id: int, output_root: Path, results_root: Path, data_root: Path) -> CyclePaths:
    suffix = f"{cycle_id:03d}"
    return CyclePaths(
        cycle_id=cycle_id,
        candidate_adapter=output_root / f"rl_pass_{suffix}",
        rollout_file=results_root / "rollouts" / f"rollout_batch_{suffix}.jsonl",
        evidence_file=results_root / "evidence" / f"evidence_batch_{suffix}.jsonl",
        scored_file=results_root / "scores" / f"scored_batch_{suffix}.jsonl",
        ppo_manifest_file=results_root / "ppo" / f"ppo_train_{suffix}.json",
        eval_file=results_root / "evals" / f"eval_rl_pass_{suffix}.json",
        manifest_file=results_root / "manifests" / f"cycle_{suffix}_manifest.json",
    )


class CachedAdapterGenerator:
    """Small model cache: one backend instance, many task generations."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Path | None,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self.backend = LocalMlxBackend(model_id=base_model, adapter_path=self.adapter_path)
        self.loaded = False

    def load_once(self) -> None:
        if not self.loaded:
            self.backend._ensure_loaded()
            self.loaded = True

    def inference_backend(self) -> LocalMlxBackend:
        """Return the loaded backend shared by generation and logprob attach."""
        self.load_once()
        return self.backend

    def generate(self, prompt: str) -> str:
        self.load_once()
        request = GenerationRequest(
            messages=[ChatMessage("system", self.system_prompt), ChatMessage("user", prompt)],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self.backend.generate(request)

    def release(self) -> None:
        """Drop cached model weights so PPO subprocess can use the GPU."""
        if self.loaded:
            self.backend.release_gpu()
            self.loaded = False


def select_rollout_tasks(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return tasks[: max(0, min(int(limit), len(tasks)))]


def select_eval_tasks(
    tasks: list[dict[str, Any]],
    limit: int,
    *,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    path = manifest_path or DEFAULT_EVAL_MANIFEST
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return resolve_eval_tasks(tasks, manifest, limit)
    n = max(1, min(int(limit), len(tasks)))
    return tasks[-n:]


def run_rollouts(
    *,
    tasks: list[dict[str, Any]],
    adapter: RegistryAdapter,
    base_model: str,
    rollout_file: Path,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    dry_run: bool,
    source_repo: Path | None = None,
    context_log_root: Path | None = None,
    max_logprob_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
    strict_old_logprob: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    adapter_path = adapter.resolved_path if adapter.exists else None
    if dry_run:
        for task in tasks:
            row = {
                "task_id": str(task.get("id") or ""),
                "prompt": str(task.get("prompt") or ""),
                "output": "[dry-run] rollout generation skipped",
                "model": base_model,
                "adapter_id": adapter.adapter_id,
                "adapter_path": str(adapter_path or ""),
                "dry_run": True,
            }
            rows.append(attach_old_logprob_to_rollout(
                rollout_row=row,
                system_prompt=system_prompt,
                base_model=base_model,
                adapter_path=adapter_path,
                dry_run=True,
                max_window_tokens=max_logprob_window_tokens,
                logprob_window_strategy=logprob_window_strategy,
                target_logprob_chunk_tokens=target_logprob_chunk_tokens,
            ))
    else:
        generator = CachedAdapterGenerator(
            base_model=base_model,
            adapter_path=adapter_path,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        try:
            generator.load_once()
            for idx, task in enumerate(tasks, start=1):
                started = time.perf_counter()
                task_id = str(task.get("id") or "")
                log_dir = (context_log_root / task_id) if context_log_root else None
                user_prompt = build_rollout_user_prompt(
                    task,
                    source_repo=source_repo,
                    log_dir=log_dir,
                    include_starter_bodies_in_prompt=False,
                )
                output = generator.generate(user_prompt)
                row = {
                    "task_id": task_id,
                    "prompt": str(task.get("prompt") or ""),
                    "generation_prompt": user_prompt,
                    "output": output,
                    "model": base_model,
                    "adapter_id": adapter.adapter_id,
                    "adapter_path": str(adapter_path or ""),
                    "seconds": round(time.perf_counter() - started, 3),
                    "rollout_index": idx,
                }
                rows.append(
                    attach_old_logprob_to_rollout(
                        rollout_row=row,
                        system_prompt=system_prompt,
                        base_model=base_model,
                        adapter_path=adapter_path,
                        dry_run=False,
                        inference_backend=generator.inference_backend(),
                        max_window_tokens=max_logprob_window_tokens,
                        logprob_window_strategy=logprob_window_strategy,
                        target_logprob_chunk_tokens=target_logprob_chunk_tokens,
                        strict=strict_old_logprob,
                    )
                )
                print(f"[rollout] {idx}/{len(tasks)} {task.get('id')}", flush=True)
        finally:
            generator.release()
            _release_accelerator_memory()
    _append_jsonl(rollout_file, rows)
    return rows


def run_evidence(
    *,
    spec: SpecializationConfig,
    task_by_id: dict[str, dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    evidence_file: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    evidenced = attach_batch_evidence(
        tasks_by_id=task_by_id,
        rollout_rows=rollout_rows,
        config=spec.evidence_runner_config,
        dry_run=dry_run,
    )
    write_evidence_jsonl(evidence_file, evidenced)
    return evidenced


def run_execution_evidence(
    *,
    task_by_id: dict[str, dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    execution_config: ExecutionEvidenceConfig,
    pool: ExecutionWorktreePool,
    cycle_id: int,
    results_root: Path,
) -> list[dict[str, Any]]:
    log_root = results_root / "execution_logs" / f"cycle_{cycle_id:03d}"
    rows = attach_batch_execution_evidence(
        tasks_by_id=task_by_id,
        rollout_rows=rollout_rows,
        config=execution_config,
        pool=pool,
        log_root=log_root,
    )
    return rows


def _resolve_execution_from_args(args: argparse.Namespace, results_root: Path) -> ExecutionEvidenceConfig:
    if bool(args.skip_execution) or bool(args.dry_run):
        return resolve_execution_config(
            enabled=False,
            source_repo=None,
            worktree_root=results_root / "worktrees",
            compile_commands=[],
            timeout_s=int(args.execution_timeout_s),
        )
    source = resolve_source_repo(cli_path=getattr(args, "execution_source_repo", None))
    if source is None and not bool(args.no_require_execution_source):
        raise SystemExit(
            "Execution evidence is required (Design A). Set ECONOMIST_RL_SOURCE_REPO to the Fallen Empire "
            "game checkout, pass --execution-source-repo, or use --skip-execution for toy-only scoring."
        )
    return resolve_execution_config(
        enabled=source is not None,
        source_repo=source,
        worktree_root=results_root / "worktrees",
        compile_commands=list(args.execution_compile_command or []),
        timeout_s=int(args.execution_timeout_s),
    )


def score_rollouts(
    *,
    payload: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    scored_file: Path,
    rolling_compile_rate: float,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row in rollout_rows:
        task_id = str(row.get("task_id") or "")
        task = task_by_id.get(task_id)
        if not task:
            continue
        score = score_output(
            task,
            str(row.get("output") or ""),
            payload,
            rollout_row=row,
            compiled=_compile_status(row),
            rolling_compile_rate=rolling_compile_rate,
        )
        scored_rows.append({"task_id": task_id, "rollout": row, "score": score})
    _append_jsonl(scored_file, scored_rows)
    return scored_rows


def _rolling_compile_rate(scored_rows: list[dict[str, Any]]) -> float:
    flags: list[bool] = []
    for row in scored_rows:
        score_obj = row.get("score") if isinstance(row.get("score"), dict) else row
        compile_gate = score_obj.get("compile_gate") if isinstance(score_obj, dict) else {}
        compiled = compile_gate.get("compiled") if isinstance(compile_gate, dict) else None
        if compiled is None:
            rollout = row.get("rollout") if isinstance(row.get("rollout"), dict) else row
            compiled = _compile_status(rollout)
        if compiled is not None:
            flags.append(bool(compiled))
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def train_candidate_ppo(
    *,
    spec: SpecializationConfig,
    scored_rows: list[dict[str, Any]],
    scored_file: Path,
    source_adapter: Path,
    candidate_adapter: Path,
    base_model: str,
    ppo_manifest_file: Path,
    ppo_config: PPOConfig,
    train_backend: str | None = None,
    dry_run: bool,
    ppo_in_process: bool = False,
) -> dict[str, Any]:
    if dry_run or ppo_in_process:
        scored_rows_from_disk = [
            json.loads(line)
            for line in scored_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = train_ppo_batch(
            scored_rows=scored_rows_from_disk,
            system_prompt=spec.system_prompt,
            base_model=base_model,
            source_adapter=source_adapter,
            candidate_adapter=candidate_adapter,
            cfg=ppo_config,
            dry_run=dry_run,
            train_backend=train_backend,
        )
        manifest["subprocess_isolated"] = False
        write_ppo_manifest(ppo_manifest_file, manifest)
        return manifest

    _release_accelerator_memory()
    return run_ppo_train_subprocess(
        scored_file=scored_file,
        system_prompt=spec.system_prompt,
        base_model=base_model,
        source_adapter=source_adapter,
        candidate_adapter=candidate_adapter,
        ppo_manifest_file=ppo_manifest_file,
        ppo_config=ppo_config,
        train_backend=train_backend,
        dry_run=False,
    )


def eval_adapter(
    *,
    label: str,
    adapter_path: Path | None,
    tasks: list[dict[str, Any]],
    payload: dict[str, Any],
    spec: SpecializationConfig,
    base_model: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    execution_config: ExecutionEvidenceConfig,
    execution_pool: ExecutionWorktreePool | None = None,
    execution_log_root: Path | None = None,
    source_repo: Path | None = None,
    max_logprob_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if dry_run:
        return {"label": label, "tasks": len(tasks), "mean_score": 0.0, "mean_reward": 0.0, "high_reward_count": 0, "dry_run": True, "rows": []}
    generator = CachedAdapterGenerator(
        base_model=base_model,
        adapter_path=adapter_path if adapter_path and adapter_path.exists() else None,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=spec.system_prompt,
    )
    generator.load_once()
    task_by_id = {str(task.get("id") or ""): task for task in tasks}
    for task in tasks:
        task_id = str(task.get("id") or "")
        log_dir = (execution_log_root / label / task_id) if execution_log_root else None
        prompt = build_rollout_user_prompt(
            task,
            source_repo=source_repo,
            log_dir=log_dir,
            include_starter_bodies_in_prompt=False,
        )
        output = generator.generate(prompt)
        rollout = attach_old_logprob_to_rollout(
            rollout_row={
                "task_id": task.get("id"),
                "prompt": str(task.get("prompt") or ""),
                "generation_prompt": prompt,
                "output": output,
                "adapter_path": str(adapter_path or ""),
            },
            system_prompt=spec.system_prompt,
            base_model=base_model,
            adapter_path=adapter_path,
            dry_run=False,
            inference_backend=generator.inference_backend(),
            max_window_tokens=max_logprob_window_tokens,
            logprob_window_strategy=logprob_window_strategy,
            target_logprob_chunk_tokens=target_logprob_chunk_tokens,
        )
        rollout = attach_batch_evidence(
            tasks_by_id=task_by_id,
            rollout_rows=[rollout],
            config=spec.evidence_runner_config,
            dry_run=False,
        )[0]
        if execution_config.enabled and execution_pool is not None and execution_log_root is not None:
            rollout = attach_batch_execution_evidence(
                tasks_by_id=task_by_id,
                rollout_rows=[rollout],
                config=execution_config,
                pool=execution_pool,
                log_root=execution_log_root / label,
            )[0]
        score = score_output(
            task,
            output,
            payload,
            rollout_row=rollout,
            compiled=_compile_status(rollout),
            rolling_compile_rate=0.0,
        )
        rows.append(
            {
                "task_id": task.get("id"),
                "score": score["score"],
                "reward": score["reward"],
                "base_reward": score.get("base_reward"),
                "training_usable": score.get("training_usable"),
                "hard_cap_applied": score.get("hard_cap_applied"),
                "cap_reason": list(score.get("cap_reason") or []),
                "diagnostics": list(score.get("diagnostics") or []),
                "strict_scorecard_pass": score.get("strict_scorecard_pass"),
                "high_reward": score.get("high_reward"),
                "failures": list(score.get("failures") or []),
                "policy_caps": list(score.get("policy_caps") or []),
            }
        )
    mean_score = sum(float(row["score"]) for row in rows) / max(1, len(rows))
    mean_reward = sum(float(row["reward"]) for row in rows) / max(1, len(rows))
    return {
        "label": label,
        "tasks": len(tasks),
        "mean_score": round(mean_score, 4),
        "mean_reward": round(mean_reward, 4),
        "high_reward_count": sum(1 for row in rows if row.get("strict_scorecard_pass")),
        "strict_scorecard_pass_count": sum(1 for row in rows if row.get("strict_scorecard_pass")),
        "rows": rows,
    }


def _eval_comparison_decision(*, baseline_eval: dict[str, Any], candidate_eval: dict[str, Any]) -> dict[str, Any]:
    major_regression_flags = [
        row
        for row in candidate_eval.get("rows", [])
        if row.get("hard_cap_applied") or row.get("policy_caps")
    ]
    baseline_reward = float(baseline_eval["mean_reward"])
    candidate_reward = float(candidate_eval["mean_reward"])
    return {
        "candidate_mean_reward_delta": round(candidate_reward - baseline_reward, 4),
        "candidate_beats_baseline": candidate_reward > baseline_reward,
        "major_regression_count": len(major_regression_flags),
        "no_major_regression_flags": len(major_regression_flags) == 0,
        "baseline_mean_score": baseline_eval["mean_score"],
        "candidate_mean_score": candidate_eval["mean_score"],
        "baseline_mean_reward": baseline_eval["mean_reward"],
        "candidate_mean_reward": candidate_eval["mean_reward"],
        "registry_auto_update": False,
    }


def _eval_comparison_block(
    *,
    comparison_id: str,
    baseline_label: str,
    baseline_adapter: RegistryAdapter,
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "baseline_label": baseline_label,
        "baseline_adapter": baseline_adapter.adapter_path,
        "baseline_adapter_resolved": str(baseline_adapter.resolved_path),
        "baseline": baseline_eval,
        "candidate": candidate_eval,
        "decision": _eval_comparison_decision(baseline_eval=baseline_eval, candidate_eval=candidate_eval),
    }


def run_eval_and_compare(
    *,
    registry_baseline: RegistryAdapter,
    working_source: RegistryAdapter,
    candidate_adapter: Path,
    eval_tasks: list[dict[str, Any]],
    eval_payload: dict[str, Any],
    eval_file: Path,
    spec: SpecializationConfig,
    base_model: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
    execution_config: ExecutionEvidenceConfig,
    execution_pool: ExecutionWorktreePool | None = None,
    execution_log_root: Path | None = None,
    source_repo: Path | None = None,
    max_logprob_window_tokens: int | None = None,
    logprob_window_strategy: str = "grouped",
    target_logprob_chunk_tokens: int = 512,
) -> dict[str, Any]:
    eval_kwargs = {
        "tasks": eval_tasks,
        "payload": eval_payload,
        "spec": spec,
        "base_model": base_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "dry_run": dry_run,
        "execution_config": execution_config,
        "execution_pool": execution_pool,
        "execution_log_root": execution_log_root,
        "source_repo": source_repo,
        "max_logprob_window_tokens": max_logprob_window_tokens,
        "logprob_window_strategy": logprob_window_strategy,
        "target_logprob_chunk_tokens": target_logprob_chunk_tokens,
    }
    candidate_eval = eval_adapter(label="candidate", adapter_path=candidate_adapter, **eval_kwargs)
    registry_baseline_eval = eval_adapter(
        label="registry_baseline",
        adapter_path=registry_baseline.resolved_path if registry_baseline.exists else None,
        **eval_kwargs,
    )
    same_working_source = (
        registry_baseline.resolved_path.resolve() == working_source.resolved_path.resolve()
    )
    if same_working_source:
        working_source_eval = registry_baseline_eval
    else:
        working_source_eval = eval_adapter(
            label="working_source",
            adapter_path=working_source.resolved_path if working_source.exists else None,
            **eval_kwargs,
        )

    payload = {
        "created_utc": _utc_iso(),
        "candidate_adapter": str(candidate_adapter),
        "candidate": candidate_eval,
        "comparisons": {
            "vs_registry_baseline": _eval_comparison_block(
                comparison_id="vs_registry_baseline",
                baseline_label="registry_baseline",
                baseline_adapter=registry_baseline,
                baseline_eval=registry_baseline_eval,
                candidate_eval=candidate_eval,
            ),
            "vs_working_source": _eval_comparison_block(
                comparison_id="vs_working_source",
                baseline_label="working_source",
                baseline_adapter=working_source,
                baseline_eval=working_source_eval,
                candidate_eval=candidate_eval,
            ),
        },
        "working_source_same_as_registry_baseline": same_working_source,
        "registry_auto_update": False,
    }
    _write_json(eval_file, payload)
    return payload


def lambda_metadata(lambda_mode: bool) -> dict[str, Any]:
    return {
        "lambda_mode": bool(lambda_mode),
        "instance_id": os.environ.get("FE_LAMBDA_INSTANCE_ID") or os.environ.get("LAMBDA_INSTANCE_ID") or "",
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "local_backend": os.environ.get("LOCAL_BACKEND", ""),
    }


def specialization_from_args(args: argparse.Namespace) -> SpecializationConfig:
    try:
        return SPECIALIZATIONS[str(args.specialization)]
    except KeyError as exc:
        raise SystemExit(
            f"Unknown specialization {args.specialization!r}. Available: {', '.join(sorted(SPECIALIZATIONS))}"
        ) from exc


def apply_specialization_defaults(args: argparse.Namespace) -> argparse.Namespace:
    spec = specialization_from_args(args)
    if args.adapter_name is None:
        args.adapter_name = spec.adapter_id
    if args.task_db is None:
        args.task_db = spec.task_db
    if args.eval_set is None:
        args.eval_set = spec.eval_set
    if args.output_root is None:
        args.output_root = spec.output_root
    if args.results_root is None:
        args.results_root = spec.results_root
    if args.data_root is None:
        args.data_root = spec.data_root
    return args


def run_cycle(
    args: argparse.Namespace,
    cycle_id: int,
    *,
    chain_from_previous: Path | None = None,
    bootstrap_cycle: bool = False,
) -> dict[str, Any]:
    spec = specialization_from_args(args)
    output_root = args.output_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    paths = paths_for_cycle(cycle_id, output_root, results_root, data_root)
    registry = args.registry.expanduser().resolve()
    init_for_resolve: Path | None = None
    if chain_from_previous is None:
        init_path = getattr(args, "init_adapter_path", None)
        if init_path is None:
            default_init = DEFAULT_ECONOMIST_RL_INIT_ADAPTER.expanduser().resolve()
            if default_init.is_dir() and adapter_dir_has_weights(default_init):
                init_path = default_init
        if init_path is not None:
            init_for_resolve = Path(init_path)
    registry_current, current, load_reason = resolve_cycle_current_adapter(
        registry_path=registry,
        adapter_name=args.adapter_name,
        chain_from_previous=chain_from_previous,
        init_adapter_path=init_for_resolve,
    )
    task_payload, tasks = _load_task_payload(args.task_db.expanduser().resolve(), spec)
    eval_payload, eval_tasks_all = _load_task_payload(args.eval_set.expanduser().resolve(), spec)
    rollout_limit = _rollout_limit_for_cycle(args, bootstrap_cycle=bootstrap_cycle)
    rollout_tasks = select_rollout_tasks(tasks, rollout_limit)
    eval_manifest = args.eval_manifest.expanduser().resolve() if getattr(args, "eval_manifest", None) else DEFAULT_EVAL_MANIFEST
    eval_tasks = select_eval_tasks(
        eval_tasks_all,
        args.eval_limit,
        manifest_path=eval_manifest if eval_manifest.is_file() else None,
    )
    task_by_id = {str(t.get("id") or ""): t for t in tasks}
    for eval_task in eval_tasks:
        task_by_id[str(eval_task.get("id") or "")] = eval_task
    ppo_config = _resolve_ppo_config(spec, args)

    print(f"[cycle {cycle_id:03d}] base_model={args.base_model}")
    print(f"[cycle {cycle_id:03d}] registry_adapter={registry_current.adapter_path} exists={registry_current.exists}")
    print(f"[cycle {cycle_id:03d}] effective_adapter={current.adapter_path} load_reason={load_reason} exists={current.exists}")
    print(f"[cycle {cycle_id:03d}] candidate_adapter={paths.candidate_adapter}")
    if not current.exists and not args.dry_run:
        raise SystemExit(
            f"Effective adapter is missing: {current.adapter_path} (load_reason={load_reason}). "
            "Refusing to run live rollouts against an accidental base-model fallback."
        )

    manifest: dict[str, Any] = {
        "schema_version": "rl_lambda_runner_cycle_manifest_v2",
        "runner_tag": RUNNER_TAG,
        "runner_display_name": RUNNER_DISPLAY_NAME,
        "specialization_tag": spec.tag,
        "specialization_display_name": spec.display_name,
        "specialization_scorer": spec.scorer,
        "specialization_evidence_runner": spec.evidence_runner,
        "specialization_ppo_trainer": spec.ppo_trainer,
        "train_mode": spec.train_mode,
        "cycle_id": cycle_id,
        "timestamp": _utc_iso(),
        "adapter_name": args.adapter_name,
        "base_model": args.base_model,
        "registry_adapter": registry_current.adapter_path,
        "registry_adapter_exists": registry_current.exists,
        "effective_current_adapter": current.adapter_path,
        "effective_current_adapter_exists": current.exists,
        "adapter_load_reason": load_reason,
        "current_adapter": current.adapter_path,
        "current_adapter_exists": current.exists,
        "candidate_adapter": str(paths.candidate_adapter),
        "task_batch_source": str(args.task_db.expanduser().resolve()),
        "eval_set": str(args.eval_set.expanduser().resolve()),
        "eval_manifest": str(eval_manifest),
        "max_tokens": int(args.max_tokens),
        "context_max_chars": int(getattr(args, "context_max_chars", 12_000)),
        "rollout_worker": {
            "mode": "bootstrap" if bootstrap_cycle else "steady",
            "requested_rollouts": int(rollout_limit),
            "usable_filter": "score.training_usable != false",
        },
        "ppo_worker": {
            "mode": "isolated_subprocess",
            "min_samples": int(ppo_config.min_samples),
            "max_samples": ppo_config.max_samples,
            "max_logprob_window_tokens": int(ppo_config.max_logprob_window_tokens),
            "logprob_window_strategy": str(ppo_config.logprob_window_strategy),
            "target_logprob_chunk_tokens": int(ppo_config.target_logprob_chunk_tokens),
        },
        "rollout_file": str(paths.rollout_file),
        "evidence_file": str(paths.evidence_file),
        "scored_file": str(paths.scored_file),
        "ppo_manifest_file": str(paths.ppo_manifest_file),
        "eval_file": str(paths.eval_file),
        "manifest_file": str(paths.manifest_file),
        "cycle_status": "not_evaluated",
        "registry_auto_update": False,
        "lambda": lambda_metadata(bool(args.lambda_mode)),
        "dry_run": bool(args.dry_run),
    }

    execution_config = _resolve_execution_from_args(args, results_root)
    source_repo = execution_config.source_repo
    context_log_root = results_root / "context_logs" / f"cycle_{cycle_id:03d}"

    rollouts = run_rollouts(
        tasks=rollout_tasks,
        adapter=current,
        base_model=args.base_model,
        rollout_file=paths.rollout_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=spec.system_prompt,
        dry_run=bool(args.dry_run),
        source_repo=source_repo,
        context_log_root=context_log_root if source_repo else None,
        max_logprob_window_tokens=ppo_config.max_logprob_window_tokens,
        logprob_window_strategy=ppo_config.logprob_window_strategy,
        target_logprob_chunk_tokens=ppo_config.target_logprob_chunk_tokens,
        strict_old_logprob=bool(getattr(args, "strict_old_logprob", False)),
    )
    proxy_errors = collect_old_logprob_proxy_errors(rollouts)
    if proxy_errors:
        manifest["old_logprob_errors"] = proxy_errors
        print(
            f"[cycle {cycle_id:03d}] old_logprob attach used proxy fallback for "
            f"{len(proxy_errors)} rollout(s)",
            flush=True,
        )
        if getattr(args, "strict_old_logprob", False):
            manifest["cycle_status"] = "old_logprob_attach_failed"
            _write_json(paths.manifest_file, manifest)
            return manifest
    evidenced = run_evidence(
        spec=spec,
        task_by_id=task_by_id,
        rollout_rows=rollouts,
        evidence_file=paths.evidence_file,
        dry_run=bool(args.dry_run),
    )
    execution_pool: ExecutionWorktreePool | None = None
    try:
        if execution_config.enabled and execution_config.source_repo and not args.dry_run:
            execution_pool = ExecutionWorktreePool(execution_config, cycle_id=cycle_id)
            evidenced = run_execution_evidence(
                task_by_id=task_by_id,
                rollout_rows=evidenced,
                execution_config=execution_config,
                pool=execution_pool,
                cycle_id=cycle_id,
                results_root=results_root,
            )
            write_evidence_jsonl(paths.evidence_file, evidenced)
    finally:
        if execution_pool is not None:
            execution_pool.cleanup()

    manifest["execution_evidence"] = execution_config_summary(execution_config)
    compile_rate = _rolling_compile_rate([{"rollout": row} for row in evidenced])
    scored = score_rollouts(
        payload=task_payload,
        task_by_id=task_by_id,
        rollout_rows=evidenced,
        scored_file=paths.scored_file,
        rolling_compile_rate=compile_rate,
    )
    ppo_source = current.resolved_path if current.exists else paths.candidate_adapter
    train_result: dict[str, Any] = {"status": "skipped", "reason": "rollouts_only"}
    eval_result: dict[str, Any] = {"status": "skipped", "reason": "rollouts_only"}
    if not getattr(args, "skip_ppo", False) and not args.dry_run:
        _release_accelerator_memory()
        train_result = train_candidate_ppo(
            spec=spec,
            scored_rows=scored,
            scored_file=paths.scored_file,
            source_adapter=ppo_source,
            candidate_adapter=paths.candidate_adapter,
            base_model=args.base_model,
            ppo_manifest_file=paths.ppo_manifest_file,
            ppo_config=ppo_config,
            train_backend="transformers" if bool(args.lambda_mode) else None,
            dry_run=bool(args.dry_run),
            ppo_in_process=bool(getattr(args, "ppo_in_process", False)),
        )
    if (
        not getattr(args, "skip_ppo", False)
        and not args.dry_run
        and train_result.get("status") not in {"trained", "dry_run_no_weight_update", "skipped"}
    ):
        manifest["cycle_status"] = "training_failed"
        manifest["train_result"] = train_result
        _write_json(paths.manifest_file, manifest)
        return manifest
    if (
        not getattr(args, "skip_ppo", False)
        and not args.dry_run
        and not paths.candidate_adapter.exists()
    ):
        manifest["cycle_status"] = "candidate_missing_after_training"
        manifest["train_result"] = train_result
        _write_json(paths.manifest_file, manifest)
        return manifest
    eval_execution_pool: ExecutionWorktreePool | None = None
    eval_execution_log = results_root / "execution_logs" / f"cycle_{cycle_id:03d}"
    try:
        if (
            not getattr(args, "skip_eval", False)
            and execution_config.enabled
            and execution_config.source_repo
            and not args.dry_run
        ):
            eval_execution_pool = ExecutionWorktreePool(execution_config, cycle_id=cycle_id)
        if not getattr(args, "skip_eval", False) and not args.dry_run:
            eval_result = run_eval_and_compare(
                registry_baseline=registry_current,
                working_source=current,
                candidate_adapter=paths.candidate_adapter,
                eval_tasks=eval_tasks,
                eval_payload=eval_payload,
                eval_file=paths.eval_file,
                spec=spec,
                base_model=args.base_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                dry_run=bool(args.dry_run),
                execution_config=execution_config,
                execution_pool=eval_execution_pool,
                execution_log_root=eval_execution_log,
                source_repo=source_repo,
                max_logprob_window_tokens=ppo_config.max_logprob_window_tokens,
                logprob_window_strategy=ppo_config.logprob_window_strategy,
                target_logprob_chunk_tokens=ppo_config.target_logprob_chunk_tokens,
            )
    finally:
        if eval_execution_pool is not None:
            eval_execution_pool.cleanup()
    if getattr(args, "skip_ppo", False) or getattr(args, "skip_eval", False):
        manifest["cycle_status"] = "rollouts_only"
    else:
        manifest["cycle_status"] = "dry_run" if args.dry_run else "completed"
    manifest["train_result"] = train_result
    manifest["ppo_source_adapter"] = str(ppo_source)
    manifest["mean_rollout_reward"] = round(
        sum(float((row.get("score") or {}).get("reward") or 0.0) for row in scored) / max(1, len(scored)),
        4,
    )
    manifest["eval_result"] = eval_result
    _write_json(paths.manifest_file, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEPRECATED: run serial rollout/train/eval research cycles.")
    parser.add_argument("--specialization", choices=sorted(SPECIALIZATIONS), default="economist_rl")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--rollouts-per-cycle",
        type=int,
        default=DEFAULT_STEADY_ROLLOUTS_PER_CYCLE,
        help="Steady-state Rollout Worker target after the bootstrap cycle.",
    )
    parser.add_argument(
        "--bootstrap-rollouts-per-cycle",
        type=int,
        default=DEFAULT_BOOTSTRAP_ROLLOUTS_PER_CYCLE,
        help="First-cycle Rollout Worker target before the initial PPO Worker update.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=None, help="Override specialization PPO epochs.")
    parser.add_argument(
        "--ppo-min-samples",
        type=int,
        default=None,
        help="Minimum training_usable scored rollouts required for the PPO Worker.",
    )
    parser.add_argument(
        "--ppo-max-samples",
        type=int,
        default=None,
        help="Cap PPO Worker training samples per cycle (worker default 64; override for legacy/smoke runs).",
    )
    parser.add_argument(
        "--ppo-logprob-window-tokens",
        type=int,
        default=None,
        help="Max tokens per PPO logprob forward window.",
    )
    parser.add_argument(
        "--ppo-logprob-window-strategy",
        choices=("grouped", "per_token"),
        default=None,
        help="PPO logprob window strategy; grouped is optimized, per_token is the exact fallback/reference.",
    )
    parser.add_argument(
        "--ppo-target-logprob-chunk-tokens",
        type=int,
        default=None,
        help="Target completion tokens scored per grouped logprob window.",
    )
    parser.add_argument(
        "--ppo-mini-batch-size",
        type=int,
        default=None,
        help="PPO optimizer mini-batch size; increase when VRAM headroom exists.",
    )
    parser.add_argument("--adapter-name", default=None)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--task-db", type=Path, default=None)
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=None, help="Deprecated; ignored in PPO mode.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_GENERATION_MAX_TOKENS)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=DEFAULT_EVAL_MANIFEST,
        help="Stratified eval task manifest (arena + sandbox + generalist tiers).",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=12_000,
        help="BM25 context pack budget per rollout (requires execution source repo).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lambda-mode", action="store_true")
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip apply/compile execution evidence (toy sim + text rubric only).",
    )
    parser.add_argument(
        "--execution-source-repo",
        type=Path,
        default=None,
        help="Fallen Empire checkout for apply+compile (or set ECONOMIST_RL_SOURCE_REPO).",
    )
    parser.add_argument(
        "--execution-compile-command",
        action="append",
        default=[],
        help="Shell command for compile/test after apply (repeatable). Default: task compile_commands or npm run test:ml-cohort.",
    )
    parser.add_argument("--execution-timeout-s", type=int, default=600)
    parser.add_argument(
        "--no-require-execution-source",
        action="store_true",
        help="Allow live cycles without ECONOMIST_RL_SOURCE_REPO (compile gate stays inactive).",
    )
    parser.add_argument(
        "--init-adapter-path",
        type=Path,
        default=None,
        help="LoRA init for rollouts/PPO (default: checkpoints/fe-lora-arena-apply-sft).",
    )
    parser.add_argument(
        "--strict-old-logprob",
        action="store_true",
        help="Fail the cycle when rollout old_logprob attach falls back to proxy values.",
    )
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Rollouts + scoring only; do not run the PPO Worker.",
    )
    parser.add_argument(
        "--ppo-in-process",
        action="store_true",
        help="Deprecated debug path: run PPO in the Rollout Worker parent process instead of the PPO Worker.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-cycle eval comparison.",
    )
    return apply_specialization_defaults(parser.parse_args())


def main() -> int:
    args = parse_args()
    if os.environ.get("ECONOMIST_RL_CYCLE_RUNNER_INTERNAL") != "1":
        print(DEPRECATION_NOTICE, flush=True)
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    if args.rollouts_per_cycle < 1:
        raise SystemExit("--rollouts-per-cycle must be >= 1")
    if getattr(args, "bootstrap_rollouts_per_cycle", None) is not None and args.bootstrap_rollouts_per_cycle < 1:
        raise SystemExit("--bootstrap-rollouts-per-cycle must be >= 1")
    if args.lambda_mode:
        if not os.environ.get("LOCAL_BACKEND"):
            os.environ["LOCAL_BACKEND"] = "transformers"
        os.environ.setdefault("PPO_TRAIN_BACKEND", "transformers")
    start = next_cycle_id(args.results_root.expanduser().resolve(), args.output_root.expanduser().resolve())
    manifests = []
    previous_candidate: Path | None = None
    for offset, cycle_id in enumerate(range(start, start + int(args.cycles))):
        chain_from = previous_candidate if offset > 0 else None
        manifest = run_cycle(args, cycle_id, chain_from_previous=chain_from, bootstrap_cycle=(offset == 0))
        manifests.append(manifest)
        if manifest.get("cycle_status") == "training_failed":
            print(
                f"[cycle {cycle_id:03d}] stopping multi-cycle run after training_failed",
                flush=True,
            )
            break
        if manifest.get("cycle_status") == "old_logprob_attach_failed":
            print(
                f"[cycle {cycle_id:03d}] stopping multi-cycle run after old_logprob_attach_failed",
                flush=True,
            )
            break
        candidate_text = str(manifest.get("candidate_adapter") or "")
        if candidate_text and manifest.get("train_result", {}).get("status") == "trained":
            previous_candidate = Path(candidate_text).expanduser().resolve()
    print(json.dumps({"cycles": len(manifests), "manifests": [m.get("manifest_file", "") for m in manifests]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
