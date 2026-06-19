#!/usr/bin/env python3
"""Run economistRL split-worker orchestration on Lambda.

This is the preferred RL topology for economistRL:

- Rollout Worker: continuously generates, executes, scores, and queues
  training-usable rollouts using the latest approved adapter.
- PPO Optimisation Worker: consumes queued good samples, trains an isolated PPO
  subprocess, then promotes a trained candidate adapter back to the Rollout Worker.

The orchestration deliberately reuses ``run_economist_rl_lambda_cycle.py`` for
rollout/evidence/scoring so the quantized model policy, execution evidence, and
artifact extraction paths stay aligned with the proven Lambda runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
LAMBDA_SCRIPTS = SCRIPTS / "lambda"
for path in (SCRIPTS, LAMBDA_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from economist_rl_ppo_trainer import (  # noqa: E402
    PPOConfig,
    candidate_has_ppo_training_marker,
)
from run_economist_rl_lambda_cycle import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_EVAL_MANIFEST,
    DEFAULT_REGISTRY,
    ECONOMIST_RL_TASK_DB,
    SPECIALIZATIONS,
)

CYCLE_RUNNER = LAMBDA_SCRIPTS / "run_economist_rl_lambda_cycle.py"
PPO_TRAIN_SCRIPT = LAMBDA_SCRIPTS / "run_economist_rl_ppo_train.py"
DEFAULT_RESULTS_ROOT = REPO / "benchmarks" / "results" / "economistRL"
DEFAULT_OUTPUT_ROOT = REPO / "checkpoints" / "adapters" / "economistRL"
DEFAULT_QUEUE_FILE = DEFAULT_RESULTS_ROOT / "split_worker" / "training_queue.jsonl"
DEFAULT_STATE_FILE = DEFAULT_RESULTS_ROOT / "split_worker" / "state.json"
DEFAULT_STEADY_ROLLOUT_BATCH_SIZE = 25
DEFAULT_BOOTSTRAP_ROLLOUT_BATCH_SIZE = 50
DEFAULT_PPO_MIN_SAMPLES = 25
DEFAULT_PPO_MAX_SAMPLES = 64


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_stamp(utc_iso: str) -> str:
    return utc_iso.replace("-", "").replace(":", "").replace("+00:00", "Z").replace("Z", "Z")


def _date_from_utc(utc_iso: str) -> str:
    return utc_iso.split("T", 1)[0]


def _next_run_number(results_root: Path) -> int:
    runs_root = results_root / "runs"
    highest = 0
    if runs_root.is_dir():
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            match = re.match(r"run_(\d{4})_", child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _write_run_markdown(descriptor: dict[str, Any]) -> None:
    lines = [
        f"# economistRL Run {int(descriptor['run_number']):04d}",
        "",
        f"- Date: {descriptor['run_date_utc']}",
        f"- Run ID: `{descriptor['run_id']}`",
        f"- Status: `{descriptor['status']}`",
        f"- Entrypoint: `{descriptor['entrypoint']}`",
        f"- Task DB: `{descriptor['task_db']}`",
        f"- Init adapter: `{descriptor['init_adapter']}`",
        "",
        "## Success Definition",
        "",
        descriptor["success_definition"],
        "",
        "## Usable Outputs",
        "",
    ]
    for key, value in sorted((descriptor.get("usable_outputs") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Obsolete Paths", ""])
    for value in descriptor.get("obsolete_paths") or []:
        lines.append(f"- `{value}`")
    phases = descriptor.get("phases") or []
    if phases:
        lines.extend(["", "## Phases", ""])
        for phase in phases:
            label = str(phase.get("phase") or "unknown")
            status = str(phase.get("status") or "")
            lines.append(f"- `{label}`: `{status}`")
    Path(str(descriptor["run_md"])).write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_run_descriptor(
    *,
    results_root: Path,
    output_root: Path,
    queue_file: Path,
    state_file: Path,
    task_db: Path,
    init_adapter_path: Path,
    entrypoint: str,
    argv: list[str],
    started_at_utc: str | None = None,
    run_id: str | None = None,
    run_number: int | None = None,
) -> dict[str, Any]:
    started = started_at_utc or _utc_iso()
    number = int(run_number or _next_run_number(results_root))
    stamp = _run_stamp(started)
    rid = run_id or f"run_{number:04d}_{stamp}"
    run_dir = results_root / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = run_dir / "RUN_MANIFEST.json"
    run_md = run_dir / "RUN.md"
    descriptor: dict[str, Any] = {
        "schema_version": "economist_rl_pipeline_run_v1",
        "run_id": rid,
        "run_number": number,
        "run_date_utc": _date_from_utc(started),
        "started_at_utc": started,
        "finished_at_utc": "",
        "entrypoint": entrypoint,
        "argv": list(argv),
        "pipeline_lane": "economistRL_execution_rl",
        "status": "running",
        "success_definition": (
            "PPO success requires at least one candidate adapter with status `trained` "
            "and a PPO training marker. Rollouts-only completion is usable data, not PPO success."
        ),
        "obsolete_paths": [
            "python scripts/ml_workflow.py economist-rl-dataset",
            "scripts/adapters/build_economist_rl_dataset.py",
            "data/lora/adapters/_deprecated_economistRL_stub_sft/",
            "training/economistRL_lora_qwen25_coder_7b.yaml",
        ],
        "usable_outputs": {
            "rollouts_scores_queue": str(queue_file),
            "ppo_candidates": str(output_root / "rl_pass_*"),
            "split_worker_state": str(state_file),
        },
        "artifact_roots": {
            "results_root": str(results_root),
            "run_dir": str(run_dir),
            "output_root": str(output_root),
        },
        "extraction": {
            "extracts_root": str(results_root / "extracts"),
            "extract_index": str(results_root / "extracts" / "index.jsonl"),
            "live_snapshots": str(results_root / "extracts" / "live_<instance_id>/"),
        },
        "task_db": str(task_db),
        "init_adapter": str(init_adapter_path),
        "queue_file": str(queue_file),
        "state_file": str(state_file),
        "phases": [],
        "manifest_file": str(manifest_file),
        "run_md": str(run_md),
    }
    _write_json(manifest_file, descriptor)
    _write_run_markdown(descriptor)
    return descriptor


def _classify_run_status(state: dict[str, Any], phases: list[dict[str, Any]]) -> str:
    if any(str(phase.get("status") or "").startswith("failed") for phase in phases):
        if any(str(phase.get("phase")) == "ppo" for phase in phases):
            return "failed_ppo"
        return "failed_rollout"
    if int(state.get("ppo_updates") or 0) > 0 or str(state.get("last_train_status") or "") == "trained":
        return "completed_trained"
    if int(state.get("rollout_batches") or 0) > 0:
        return "completed_no_ppo"
    return "completed_no_data"


def finalize_run_descriptor(
    descriptor: dict[str, Any],
    *,
    state: dict[str, Any],
    phases: list[dict[str, Any]],
    finished_at_utc: str | None = None,
) -> dict[str, Any]:
    final = {
        **descriptor,
        "finished_at_utc": finished_at_utc or _utc_iso(),
        "status": _classify_run_status(state, phases),
        "state": dict(state),
        "phases": list(phases),
    }
    _write_json(Path(str(final["manifest_file"])), final)
    _write_run_markdown(final)
    return final


@dataclass(frozen=True)
class SplitWorkerState:
    latest_approved_adapter: str
    latest_adapter_version: str
    queue_read_offset: int = 0
    ppo_updates: int = 0
    rollout_batches: int = 0
    last_rollout_cycle_id: int = 0
    last_candidate_adapter: str = ""
    last_train_status: str = ""
    last_updated_utc: str = ""


def load_state(state_file: Path, *, init_adapter_path: Path) -> SplitWorkerState:
    if state_file.is_file():
        payload = _read_json(state_file)
        return SplitWorkerState(**{k: payload[k] for k in SplitWorkerState.__dataclass_fields__ if k in payload})
    init_adapter = str(init_adapter_path.expanduser().resolve())
    return SplitWorkerState(
        latest_approved_adapter=init_adapter,
        latest_adapter_version=init_adapter_path.name,
        last_updated_utc=_utc_iso(),
    )


def save_state(state_file: Path, state: SplitWorkerState) -> None:
    _write_json(state_file, {"schema_version": "economist_rl_split_worker_state_v1", **asdict(state)})


def enqueue_training_usable_samples(
    *,
    scored_file: Path,
    queue_file: Path,
    adapter_version: str,
    rollout_cycle_id: int,
) -> int:
    rows = _read_jsonl(scored_file)
    queued: list[dict[str, Any]] = []
    for row in rows:
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        if score.get("training_usable") is False:
            continue
        queued.append(
            {
                **row,
                "adapter_version": adapter_version,
                "rollout_cycle_id": int(rollout_cycle_id),
                "queued_utc": _utc_iso(),
            }
        )
    if queued:
        _append_jsonl(queue_file, queued)
    return len(queued)


def select_unconsumed_samples(
    *,
    queue_file: Path,
    offset: int,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], int]:
    rows = _read_jsonl(queue_file)
    start = max(0, int(offset))
    end = len(rows) if max_samples is None else min(len(rows), start + int(max_samples))
    return rows[start:end], end


def promote_candidate(
    state: SplitWorkerState,
    train_manifest: dict[str, Any],
    *,
    acceptance_mode: str,
) -> SplitWorkerState:
    status = str(train_manifest.get("status") or "")
    candidate = str(train_manifest.get("candidate_adapter") or "")
    accepted = status == "trained"
    if acceptance_mode == "none":
        accepted = accepted or status == "dry_run_no_weight_update"
    if not accepted or not candidate:
        return SplitWorkerState(
            **{
                **asdict(state),
                "last_candidate_adapter": candidate,
                "last_train_status": status or "not_promoted",
                "last_updated_utc": _utc_iso(),
            }
        )
    return SplitWorkerState(
        **{
            **asdict(state),
            "latest_approved_adapter": candidate,
            "latest_adapter_version": Path(candidate).name,
            "ppo_updates": state.ppo_updates + 1,
            "last_candidate_adapter": candidate,
            "last_train_status": status,
            "last_updated_utc": _utc_iso(),
        }
    )


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _cycle_manifest_paths(results_root: Path) -> set[Path]:
    root = results_root / "manifests"
    return set(root.glob("cycle_*_manifest.json")) if root.exists() else set()


def _new_cycle_manifest(before: set[Path], results_root: Path) -> Path:
    after = _cycle_manifest_paths(results_root)
    new_paths = sorted(after - before)
    if new_paths:
        return new_paths[-1]
    all_paths = sorted(after)
    if not all_paths:
        raise RuntimeError("Rollout Worker did not produce a cycle manifest.")
    return all_paths[-1]


def _rollout_cycle_args(args: argparse.Namespace, state: SplitWorkerState, *, batch_size: int) -> list[str]:
    out = [
        str(sys.executable),
        str(CYCLE_RUNNER),
        "--cycles",
        "1",
        "--rollouts-per-cycle",
        str(batch_size),
        "--bootstrap-rollouts-per-cycle",
        str(batch_size),
        "--skip-ppo",
        "--skip-eval",
        "--specialization",
        str(args.specialization),
        "--adapter-name",
        str(args.adapter_name),
        "--registry",
        str(args.registry),
        "--task-db",
        str(args.task_db),
        "--eval-set",
        str(args.eval_set),
        "--eval-manifest",
        str(args.eval_manifest),
        "--output-root",
        str(args.output_root),
        "--results-root",
        str(args.results_root),
        "--data-root",
        str(args.data_root),
        "--base-model",
        str(args.base_model),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--ppo-logprob-window-tokens",
        str(args.ppo_logprob_window_tokens),
        "--ppo-logprob-window-strategy",
        str(args.ppo_logprob_window_strategy),
        "--ppo-target-logprob-chunk-tokens",
        str(args.ppo_target_logprob_chunk_tokens),
        "--ppo-mini-batch-size",
        str(args.ppo_mini_batch_size),
        "--init-adapter-path",
        state.latest_approved_adapter,
    ]
    if args.lambda_mode:
        out.append("--lambda-mode")
    if args.execution_source_repo:
        out.extend(["--execution-source-repo", str(args.execution_source_repo)])
    if args.no_require_execution_source:
        out.append("--no-require-execution-source")
    if args.skip_execution:
        out.append("--skip-execution")
    for command in args.execution_compile_command or []:
        out.extend(["--execution-compile-command", str(command)])
    return out


def run_rollout_worker_batch(args: argparse.Namespace, state: SplitWorkerState, *, batch_size: int) -> dict[str, Any]:
    before = _cycle_manifest_paths(args.results_root)
    env = os.environ.copy()
    env["ECONOMIST_RL_CYCLE_RUNNER_INTERNAL"] = "1"
    _run_subprocess(_rollout_cycle_args(args, state, batch_size=batch_size), cwd=REPO, env=env)
    manifest_path = _new_cycle_manifest(before, args.results_root)
    manifest = _read_json(manifest_path)
    scored_file = Path(str(manifest.get("scored_file") or "")).expanduser()
    if not scored_file.is_absolute():
        scored_file = (REPO / scored_file).resolve()
    appended = enqueue_training_usable_samples(
        scored_file=scored_file,
        queue_file=args.queue_file,
        adapter_version=state.latest_adapter_version,
        rollout_cycle_id=int(manifest.get("cycle_id") or 0),
    )
    manifest["split_worker_queue_appended"] = appended
    return manifest


def _ppo_config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        ppo_epochs=int(args.ppo_epochs),
        min_samples=int(args.ppo_min_samples),
        max_samples=int(args.ppo_max_samples) if args.ppo_max_samples is not None else None,
        max_logprob_window_tokens=int(args.ppo_logprob_window_tokens),
        logprob_window_strategy=str(args.ppo_logprob_window_strategy),
        target_logprob_chunk_tokens=int(args.ppo_target_logprob_chunk_tokens),
        mini_batch_size=int(args.ppo_mini_batch_size),
    )


def _split_adapter_suffix(adapter_name: str) -> int | None:
    match = re.match(r"rl_pass_split_(\d+)$", adapter_name)
    return int(match.group(1)) if match else None


def next_candidate_adapter(output_root: Path, state: SplitWorkerState) -> Path:
    approved_name = Path(state.latest_approved_adapter).name or state.latest_adapter_version
    approved_suffix = _split_adapter_suffix(approved_name)
    next_id = state.ppo_updates + 1
    if approved_suffix is not None:
        next_id = max(next_id, approved_suffix + 1)

    source_adapter = Path(state.latest_approved_adapter).expanduser().resolve()
    while True:
        candidate = (output_root / f"rl_pass_split_{next_id:03d}").expanduser().resolve()
        if candidate != source_adapter and not (candidate.exists() and any(candidate.iterdir())):
            return candidate
        next_id += 1


def run_ppo_optimisation_worker(args: argparse.Namespace, state: SplitWorkerState) -> tuple[SplitWorkerState, bool]:
    samples, next_offset = select_unconsumed_samples(
        queue_file=args.queue_file,
        offset=state.queue_read_offset,
        max_samples=args.ppo_max_samples,
    )
    if len(samples) < int(args.ppo_min_samples):
        return state, False

    update_id = state.ppo_updates + 1
    split_root = args.results_root / "split_worker"
    sampled_file = split_root / f"ppo_samples_{update_id:03d}.jsonl"
    ppo_manifest_file = args.results_root / "ppo" / f"split_ppo_train_{update_id:03d}.json"
    candidate_adapter = next_candidate_adapter(args.output_root, state)
    _write_jsonl(sampled_file, samples)

    request = {
        "scored_file": str(sampled_file.resolve()),
        "ppo_manifest_file": str(ppo_manifest_file.resolve()),
        "system_prompt": SPECIALIZATIONS[str(args.specialization)].system_prompt,
        "base_model": str(args.base_model),
        "source_adapter": state.latest_approved_adapter,
        "candidate_adapter": str(candidate_adapter.resolve()),
        "ppo_config": asdict(_ppo_config_from_args(args)),
        "train_backend": "transformers" if args.lambda_mode else None,
        "dry_run": bool(args.dry_run_ppo),
    }
    request_path = ppo_manifest_file.with_name(f"{ppo_manifest_file.stem}.request.json")
    _write_json(request_path, request)
    _run_subprocess([str(sys.executable), str(PPO_TRAIN_SCRIPT), "--request", str(request_path)], cwd=REPO)
    train_manifest = _read_json(ppo_manifest_file)
    train_manifest["queue_start_offset"] = state.queue_read_offset
    train_manifest["queue_next_offset"] = next_offset
    _write_json(ppo_manifest_file, train_manifest)

    if args.acceptance_mode == "trained_marker" and train_manifest.get("status") == "trained":
        if not candidate_has_ppo_training_marker(candidate_adapter):
            train_manifest["status"] = "not_promoted"
            train_manifest["reason"] = "missing_ppo_training_marker"

    advanced = SplitWorkerState(
        **{
            **asdict(state),
            "queue_read_offset": next_offset,
            "last_updated_utc": _utc_iso(),
        }
    )
    return promote_candidate(advanced, train_manifest, acceptance_mode=args.acceptance_mode), True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run economistRL constant-rollout split-worker PPO orchestration.")
    parser.add_argument("--lambda-mode", action="store_true")
    parser.add_argument("--specialization", choices=sorted(SPECIALIZATIONS), default="economist_rl")
    parser.add_argument("--adapter-name", default="economistRL")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--task-db", type=Path, default=ECONOMIST_RL_TASK_DB)
    parser.add_argument("--eval-set", type=Path, default=ECONOMIST_RL_TASK_DB)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "economistRL")
    parser.add_argument("--queue-file", type=Path, default=DEFAULT_QUEUE_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--init-adapter-path", type=Path, default=REPO / "checkpoints" / "fe-lora-arena-apply-sft")
    parser.add_argument("--rollout-batch-size", type=int, default=DEFAULT_STEADY_ROLLOUT_BATCH_SIZE)
    parser.add_argument("--bootstrap-rollout-batch-size", type=int, default=DEFAULT_BOOTSTRAP_ROLLOUT_BATCH_SIZE)
    parser.add_argument("--ppo-min-samples", type=int, default=DEFAULT_PPO_MIN_SAMPLES)
    parser.add_argument("--ppo-max-samples", type=int, default=DEFAULT_PPO_MAX_SAMPLES)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-logprob-window-tokens", type=int, default=1536)
    parser.add_argument("--ppo-logprob-window-strategy", choices=("grouped", "per_token"), default="grouped")
    parser.add_argument("--ppo-target-logprob-chunk-tokens", type=int, default=512)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=8)
    parser.add_argument("--max-ppo-updates", type=int, default=1)
    parser.add_argument("--max-rollout-batches", type=int, default=20)
    parser.add_argument("--idle-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--acceptance-mode", choices=("trained_marker", "none"), default="trained_marker")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--execution-source-repo", type=Path, default=None)
    parser.add_argument("--execution-compile-command", action="append", default=[])
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--no-require-execution-source", action="store_true")
    parser.add_argument("--dry-run-ppo", action="store_true")
    parser.add_argument("--run-id", default="", help="Optional stable economistRL pipeline run id.")
    parser.add_argument("--run-number", type=int, default=None, help="Optional human run number for manifests.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.lambda_mode:
        os.environ.setdefault("LOCAL_BACKEND", "transformers")
        os.environ.setdefault("PPO_TRAIN_BACKEND", "transformers")

    args.registry = args.registry.expanduser().resolve()
    args.task_db = args.task_db.expanduser().resolve()
    args.eval_set = args.eval_set.expanduser().resolve()
    args.eval_manifest = args.eval_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.queue_file = args.queue_file.expanduser().resolve()
    args.state_file = args.state_file.expanduser().resolve()
    if args.execution_source_repo:
        args.execution_source_repo = args.execution_source_repo.expanduser().resolve()

    state = load_state(args.state_file, init_adapter_path=args.init_adapter_path)
    save_state(args.state_file, state)
    run_descriptor = create_run_descriptor(
        results_root=args.results_root,
        output_root=args.output_root,
        queue_file=args.queue_file,
        state_file=args.state_file,
        task_db=args.task_db,
        init_adapter_path=args.init_adapter_path,
        entrypoint="scripts/lambda/run_economist_rl_split_workers.py",
        argv=list(sys.argv[1:] if argv is None else argv),
        run_id=args.run_id or None,
        run_number=args.run_number,
    )
    phases: list[dict[str, Any]] = []
    current_phase = "setup"

    try:
        while state.ppo_updates < int(args.max_ppo_updates) and state.rollout_batches < int(args.max_rollout_batches):
            batch_size = (
                int(args.bootstrap_rollout_batch_size)
                if state.rollout_batches == 0 and state.queue_read_offset == 0
                else int(args.rollout_batch_size)
            )
            current_phase = "rollout"
            rollout_manifest = run_rollout_worker_batch(args, state, batch_size=batch_size)
            phases.append(
                {
                    "phase": "rollout",
                    "status": str(rollout_manifest.get("cycle_status") or "completed"),
                    "cycle_id": int(rollout_manifest.get("cycle_id") or 0),
                    "scored_file": str(rollout_manifest.get("scored_file") or ""),
                    "queue_appended": int(rollout_manifest.get("split_worker_queue_appended") or 0),
                }
            )
            state = SplitWorkerState(
                **{
                    **asdict(state),
                    "rollout_batches": state.rollout_batches + 1,
                    "last_rollout_cycle_id": int(rollout_manifest.get("cycle_id") or 0),
                    "last_updated_utc": _utc_iso(),
                }
            )
            before_updates = state.ppo_updates
            current_phase = "ppo"
            state, trained = run_ppo_optimisation_worker(args, state)
            if trained or state.ppo_updates != before_updates:
                phases.append(
                    {
                        "phase": "ppo",
                        "status": state.last_train_status or ("trained" if trained else "not_trained"),
                        "ppo_updates": state.ppo_updates,
                        "candidate_adapter": state.last_candidate_adapter,
                    }
                )
            save_state(args.state_file, state)
            if not trained:
                time.sleep(max(0.0, float(args.idle_sleep_seconds)))
    except Exception as exc:  # pylint: disable=broad-except
        phases.append({"phase": current_phase, "status": "failed_exception", "error": str(exc)})
        final = finalize_run_descriptor(
            run_descriptor,
            state=asdict(state),
            phases=phases,
        )
        print(json.dumps({"status": final["status"], "run_manifest": final["manifest_file"], "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    final = finalize_run_descriptor(
        run_descriptor,
        state=asdict(state),
        phases=phases,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": final["run_id"],
                "run_number": final["run_number"],
                "run_manifest": final["manifest_file"],
                "state": asdict(state),
                "queue_file": str(args.queue_file),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
