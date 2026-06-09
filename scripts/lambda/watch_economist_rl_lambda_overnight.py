#!/usr/bin/env python3
"""Local overnight watcher for economistRL Lambda PPO cycles.

Launches a fresh gpu_1x_a10 worker (or attaches to one), polls remote progress,
rsyncs failure artifacts, adapts PPO memory knobs after OOM (-9), and relaunches
until cycles complete or ``--max-restarts`` is exhausted.

State: ``logs/economist_rl_overnight_watch.json``
Log:   ``logs/economist_rl_overnight_watch.log``
Extracts: ``benchmarks/results/economistRL/extracts/`` (index + per-attempt manifests)

Example:

  .venv/bin/python scripts/lambda/watch_economist_rl_lambda_overnight.py \\
    --reset-state --launch-initial --poll-minutes 10 --max-restarts 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_lambda_parallel_ablation as lab  # noqa: E402

DEFAULT_SSH_KEY = Path.home() / ".ssh" / "lambda_cloud_cursor"
STATE_PATH = ROOT / "logs" / "economist_rl_overnight_watch.json"
LOG_PATH = ROOT / "logs" / "economist_rl_overnight_watch.log"
LAUNCH_LOG_DIR = ROOT / "logs"
EXTRACTS_ROOT = ROOT / "benchmarks" / "results" / "economistRL" / "extracts"
EXTRACT_INDEX_PATH = EXTRACTS_ROOT / "index.jsonl"
REMOTE_CYCLE_LOG = "/home/ubuntu/cloud-eval-logs/fe-economist-rl-cycle.log"
NAME_PREFIX = "fe-economist-rl"

DEFAULT_CYCLE_ARGV = [
    "--lambda-mode",
    "--cycles",
    "4",
    "--rollouts-per-cycle",
    "10",
    "--skip-eval",
    "--ppo-min-samples",
    "4",
    "--ppo-max-samples",
    "8",
    "--ppo-epochs",
    "1",
    "--ppo-logprob-window-tokens",
    "1536",
    "--temperature",
    "0.2",
    "--max-tokens",
    "4000",
    "--init-adapter-path",
    "checkpoints/fe-lora-arena-apply-sft",
    "--task-db",
    "benchmarks/economistRL_tasks_v3_execution.json",
    "--execution-source-repo",
    "/home/ubuntu/fallen-empire",
    "--specialization",
    "economist_rl",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_dotenv() -> None:
    dotenv = ROOT / ".env"
    if not dotenv.is_file():
        return
    for raw in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _append_extract_index(record: dict[str, Any]) -> None:
    EXTRACTS_ROOT.mkdir(parents=True, exist_ok=True)
    with EXTRACT_INDEX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _classify_failure(
    *,
    status: str,
    remote_log: str,
    progress: dict[str, Any],
) -> str:
    blob = (remote_log or "").lower()
    if status == "completed":
        return "completed"
    if status == "launch_failed" or status == "relaunch_failed":
        return "launch_failed"
    if status == "instance_gone":
        return "instance_gone"
    if status == "ssh_failed" or status == "lost":
        return "ssh_failed"
    if "-9" in blob or "outofmemory" in blob or "cuda out of memory" in blob:
        return "training_failed_oom"
    hits = progress.get("failure_hits") or []
    if any("training_failed" in str(h).lower() for h in hits):
        return "training_failed"
    if status == "failed":
        return "training_failed"
    return status or "unknown"


def _extract_dir(*, attempt: int, instance_id: str, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = instance_id[:8] if instance_id else "unknown"
    dest = EXTRACTS_ROOT / f"attempt_{attempt:03d}_{short}_{label}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _write_extract_manifest(
    dest: Path,
    *,
    attempt: int,
    instance_id: str,
    host: str,
    failure_class: str,
    status: str,
    progress: dict[str, Any],
    remote_tail: str,
    launch_log: Path | None,
    rsync_paths: list[str],
) -> None:
    manifest = {
        "extracted_at_utc": _utc_now(),
        "attempt": attempt,
        "instance_id": instance_id,
        "host": host,
        "status": status,
        "failure_class": failure_class,
        "progress": progress,
        "dest": str(dest.relative_to(ROOT)),
        "rsync_paths": rsync_paths,
        "launch_log": str(launch_log.relative_to(ROOT)) if launch_log and launch_log.is_file() else "",
        "docs": "benchmarks/results/economistRL/extracts/README.md",
    }
    (dest / "EXTRACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = [
        f"# economistRL extract — attempt {attempt:03d}",
        "",
        f"- **Instance:** `{instance_id}` @ `{host}`",
        f"- **Status:** `{status}`",
        f"- **Failure class:** `{failure_class}`",
        f"- **Last rollout:** `{progress.get('last_rollout', '')}`",
        f"- **Last cycle:** `{progress.get('last_cycle', '')}`",
        "",
        "## What to read",
        "",
        "1. `cycle_log_tail.txt` — remote runner log tail",
        "2. `cloud-eval-logs/fe-economist-rl-cycle.log` — full remote log if rsync succeeded",
        "3. `economistRL/ppo/` — PPO stderr/stdout/manifest if present",
        "4. `launch_log.txt` — local Lambda bootstrap log",
        "",
        "## Likely cause",
        "",
    ]
    if failure_class == "instance_gone":
        readme.append(
            "Worker disappeared from Lambda API mid-run (no clean `training_failed` in log). "
            "Check Lambda console / billing; compare GPU telemetry CSV if extracted."
        )
    elif failure_class == "training_failed_oom":
        readme.append("PPO subprocess OOM (often exit -9). Lower `--ppo-max-samples` and window tokens.")
    elif failure_class == "launch_failed":
        readme.append("Local launcher failed before remote cycle started (bootstrap rsync/SSH).")
    else:
        readme.append(f"See failure_class `{failure_class}` and cycle log tail.")
    (dest / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    if remote_tail.strip():
        (dest / "cycle_log_tail.txt").write_text(remote_tail, encoding="utf-8")
    if launch_log and launch_log.is_file():
        (dest / "launch_log.txt").write_text(
            launch_log.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )
    _append_extract_index(
        {
            "extracted_at_utc": manifest["extracted_at_utc"],
            "attempt": attempt,
            "instance_id": instance_id,
            "failure_class": failure_class,
            "status": status,
            "dest": manifest["dest"],
            "last_rollout": progress.get("last_rollout", ""),
        }
    )


def _rsync_artifacts(
    host: str,
    *,
    attempt: int,
    instance_id: str,
    failure_class: str,
    status: str,
    progress: dict[str, Any],
    remote_tail: str,
    launch_log: Path | None,
) -> Path:
    dest = _extract_dir(attempt=attempt, instance_id=instance_id, label=failure_class)
    ssh_cmd = f"ssh -i {shlex.quote(str(DEFAULT_SSH_KEY))} -o StrictHostKeyChecking=no -o ConnectTimeout=20"
    specs = [
        (f"ubuntu@{host}:~/cloud-eval-logs/", dest / "cloud-eval-logs"),
        (
            f"ubuntu@{host}:~/fallen-empire-lora/benchmarks/results/economistRL/",
            dest / "economistRL",
        ),
    ]
    rsync_ok: list[str] = []
    for src, target in specs:
        target.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["rsync", "-az", "--timeout=120", "-e", ssh_cmd, src, str(target) + "/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            rsync_ok.append(str(target.relative_to(dest)))
        else:
            _log(
                f"rsync_partial attempt={attempt} dest={dest.name} src={src} "
                f"rc={proc.returncode} err={proc.stderr.strip()[:160]}"
            )
    _write_extract_manifest(
        dest,
        attempt=attempt,
        instance_id=instance_id,
        host=host,
        failure_class=failure_class,
        status=status,
        progress=progress,
        remote_tail=remote_tail,
        launch_log=launch_log,
        rsync_paths=rsync_ok,
    )
    _log(f"extract_written dest={dest.relative_to(ROOT)} failure_class={failure_class}")
    return dest


def _live_snapshot(
    *,
    host: str,
    instance_id: str,
    progress: dict[str, Any],
    remote_log: str,
) -> None:
    live = EXTRACTS_ROOT / f"live_{instance_id[:12]}"
    live.mkdir(parents=True, exist_ok=True)
    (live / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    (live / "cycle_log_tail.txt").write_text(remote_log, encoding="utf-8")
    (live / "LAST_UPDATED_UTC.txt").write_text(_utc_now() + "\n", encoding="utf-8")


def _fetch_remote_log_tail(host: str, *, lines: int = 500) -> str:
    proc = _ssh(
        host,
        f"test -f {REMOTE_CYCLE_LOG} && tail -{lines} {REMOTE_CYCLE_LOG} || echo log_missing",
        timeout_s=25,
    )
    return proc.stdout if proc.returncode == 0 else ""


def _api() -> tuple[str, str]:
    _load_dotenv()
    api_base = lab._lambda_cloud_base_url_from_env()
    api_key = os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMBDA_CLOUD_API_KEY") or ""
    if not api_key:
        raise SystemExit("LAMBDA_API_KEY is required")
    return api_base, api_key


def _list_active_workers() -> list[dict[str, Any]]:
    api_base, api_key = _api()
    payload = lab._api_call(api_base, api_key, "GET", "/instances")
    rows = []
    for row in payload.get("data") or []:
        if row.get("status") != "active":
            continue
        name = str(row.get("name") or "")
        if NAME_PREFIX in name:
            rows.append(row)
    return rows


def _ssh(host: str, remote_cmd: str, *, timeout_s: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        "-i",
        str(DEFAULT_SSH_KEY),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={timeout_s}",
        f"ubuntu@{host}",
        remote_cmd,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _parse_remote_progress(text: str) -> dict[str, Any]:
    rollouts = re.findall(r"^\[rollout\] (\d+/\d+)", text, re.MULTILINE)
    cycles = re.findall(r"^\[cycle (\d+)\]", text, re.MULTILINE)
    failures = re.findall(
        r"training_failed|subprocess_exit|OutOfMemory|CUDA out of memory|exit_code.?-9|"
        r"old_logprob_attach_failed|stopping multi-cycle",
        text,
        re.IGNORECASE,
    )
    trained = "status\": \"trained\"" in text or "'status': 'trained'" in text
    ppo_lines = [line for line in text.splitlines() if "PPO" in line or "ppo_train" in line]
    return {
        "last_rollout": rollouts[-1] if rollouts else "",
        "last_cycle": cycles[-1] if cycles else "",
        "failure_hits": failures[-5:],
        "trained_hint": trained,
        "ppo_tail": ppo_lines[-3:],
        "log_lines": len(text.splitlines()),
    }


def _launch_worker(cycle_argv: list[str], *, attempt: int) -> dict[str, Any]:
    launch_log = LAUNCH_LOG_DIR / f"launch_economist_rl_overnight_attempt_{attempt:02d}.log"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "launch_economist_rl_lambda_cycle.py"),
        "--launch-instances",
        "--region",
        "us-west-1",
        "--watchdog-idle-minutes",
        "360",
        "--game-repo",
        str(Path.home() / "fallen-empire"),
        "--",
        *cycle_argv,
    ]
    _log(f"launch_start attempt={attempt} log={launch_log}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    launch_log.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"launch failed rc={proc.returncode}; see {launch_log}")
    meta: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line and not line.startswith(" "):
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    if "instance_id" not in meta or "host" not in meta:
        raise RuntimeError(f"launch missing instance metadata; see {launch_log}")
    _log(
        f"launch_ok attempt={attempt} instance_id={meta['instance_id']} host={meta['host']}"
    )
    return meta


def _terminate_instance(instance_id: str, *, reason: str) -> None:
    api_base, api_key = _api()
    _log(f"terminate_start id={instance_id} reason={reason}")
    lab._terminate_instances_best_effort(api_base, api_key, [instance_id], reason=reason)


def _adapt_cycle_argv_after_failure(
    cycle_argv: list[str],
    *,
    progress: dict[str, Any],
    remote_log: str,
) -> list[str]:
    """Tighten PPO memory after OOM/SIGKILL; otherwise keep args."""
    blob = remote_log.lower()
    oom = (
        "-9" in blob
        or "outofmemory" in blob
        or "cuda out of memory" in blob
        or any("training_failed" in hit.lower() for hit in progress.get("failure_hits") or [])
    )
    if not oom:
        return list(cycle_argv)

    argv = list(cycle_argv)

    def _set_flag(flag: str, value: str) -> None:
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv.extend([flag, value])

    max_samples = 8
    window = 1536
    if "--ppo-max-samples" in argv:
        max_samples = int(argv[argv.index("--ppo-max-samples") + 1])
    if "--ppo-logprob-window-tokens" in argv:
        window = int(argv[argv.index("--ppo-logprob-window-tokens") + 1])
    new_samples = max(4, max_samples - 2)
    new_window = max(1024, window - 256)
    _set_flag("--ppo-max-samples", str(new_samples))
    _set_flag("--ppo-logprob-window-tokens", str(new_window))
    _log(f"adapt_args_after_oom ppo_max_samples={new_samples} window={new_window}")
    return argv


def _cycle_complete(remote_log: str) -> bool:
    text = remote_log
    if "cycle_status" in text and "completed" in text:
        return True
    if re.search(r"stopping multi-cycle run after training_failed", text):
        return False
    # Heuristic: launcher child prints JSON summary with all manifests at end.
    if re.search(r'"cycles":\s*4', text) and "manifests" in text:
        return True
    return False


def _poll_once(state: dict[str, Any]) -> dict[str, Any]:
    workers = _list_active_workers()
    instance_id = str(state.get("instance_id") or "")
    host = str(state.get("host") or "")

    if not workers:
        state["status"] = "instance_gone"
        state["last_poll_utc"] = _utc_now()
        state["tmux"] = "gone"
        if instance_id:
            state["failure_class"] = "instance_gone"
        return state

    worker = workers[0]
    worker_id = str(worker.get("id") or "")
    worker_host = str(worker.get("ip") or "")
    if instance_id and worker_id and worker_id != instance_id:
        _log(f"worker_replaced old={instance_id} new={worker_id}")
    host = worker_host or host
    instance_id = worker_id or instance_id
    state["host"] = host
    state["instance_id"] = instance_id

    remote_log = _fetch_remote_log_tail(host, lines=500)
    progress = _parse_remote_progress(remote_log)
    ssh_ok = "log_missing" not in remote_log and bool(remote_log.strip())
    state.update(
        {
            "status": "running" if ssh_ok else "ssh_failed",
            "last_poll_utc": _utc_now(),
            "ssh_rc": 0 if ssh_ok else 1,
            "progress": progress,
            "failure_class": "running" if ssh_ok else "ssh_failed",
        }
    )

    tmux = _ssh(host, "tmux has-session -t fe-economist-rl 2>/dev/null && echo up || echo down")
    state["tmux"] = (tmux.stdout or "").strip()

    if ssh_ok:
        _live_snapshot(host=host, instance_id=instance_id, progress=progress, remote_log=remote_log)

    if _cycle_complete(remote_log):
        state["status"] = "completed"
        state["failure_class"] = "completed"
    elif progress.get("failure_hits") and state.get("tmux") == "down":
        state["status"] = "failed"
        state["failure_class"] = _classify_failure(
            status="failed", remote_log=remote_log, progress=progress
        )
    return state


def _run_watch(args: argparse.Namespace) -> int:
    if args.reset_state and STATE_PATH.is_file():
        STATE_PATH.unlink()
        _log("watch_state_reset")

    state = _read_state()
    cycle_argv = list(args.cycle_argv or DEFAULT_CYCLE_ARGV)
    attempt = int(state.get("attempt") or 0)
    restarts = int(state.get("restarts") or 0)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    poll_s = max(60, int(float(args.poll_minutes) * 60))
    deadline = time.time() + float(args.max_hours) * 3600.0

    if args.launch_initial or not state.get("instance_id"):
        launched = False
        while not launched and time.time() < deadline and restarts <= int(args.max_restarts):
            attempt += 1
            try:
                meta = _launch_worker(cycle_argv, attempt=attempt)
            except RuntimeError as exc:
                launch_log = LAUNCH_LOG_DIR / f"launch_economist_rl_overnight_attempt_{attempt:02d}.log"
                try:
                    dest = _extract_dir(attempt=attempt, instance_id="launch_failed", label="launch_failed")
                    _write_extract_manifest(
                        dest,
                        attempt=attempt,
                        instance_id="",
                        host="",
                        failure_class="launch_failed",
                        status="launch_failed",
                        progress={},
                        remote_tail="",
                        launch_log=launch_log if launch_log.is_file() else None,
                        rsync_paths=[],
                    )
                except Exception as extract_exc:
                    _log(f"extract_write_failed attempt={attempt} error={extract_exc}")
                    dest = EXTRACTS_ROOT
                _log(f"launch_failed attempt={attempt} error={exc}")
                restarts += 1
                if restarts > int(args.max_restarts):
                    state = {
                        "run_id": run_id,
                        "status": "gave_up",
                        "attempt": attempt,
                        "restarts": restarts,
                        "last_extract": str(dest.relative_to(ROOT)),
                        "finished_utc": _utc_now(),
                    }
                    _write_state(state)
                    return 1
                time.sleep(min(120, max(30, int(float(args.poll_minutes) * 30))))
                continue
            state = {
                "run_id": run_id,
                "attempt": attempt,
                "restarts": restarts,
                "instance_id": meta.get("instance_id"),
                "host": meta.get("host"),
                "cycle_argv": cycle_argv,
                "status": "launched",
                "started_utc": _utc_now(),
                "extracts_root": str(EXTRACTS_ROOT.relative_to(ROOT)),
            }
            _write_state(state)
            launched = True
        if not launched:
            return 1

    _log(
        f"watch_start poll_minutes={args.poll_minutes} max_restarts={args.max_restarts} "
        f"max_hours={args.max_hours}"
    )

    while time.time() < deadline:
        state = _poll_once(state)
        _write_state(state)
        _log(
            f"poll status={state.get('status')} tmux={state.get('tmux')} "
            f"progress={json.dumps(state.get('progress') or {}, sort_keys=True)}"
        )

        if state.get("status") == "completed":
            host = str(state.get("host") or "")
            instance_id = str(state.get("instance_id") or "")
            remote_tail = _fetch_remote_log_tail(host) if host else ""
            if host and instance_id:
                _rsync_artifacts(
                    host,
                    attempt=int(state.get("attempt") or 0),
                    instance_id=instance_id,
                    failure_class="completed",
                    status="completed",
                    progress=state.get("progress") or {},
                    remote_tail=remote_tail,
                    launch_log=LAUNCH_LOG_DIR / f"launch_economist_rl_overnight_attempt_{int(state.get('attempt') or 0):02d}.log",
                )
            _log("watch_done success")
            state["finished_utc"] = _utc_now()
            _write_state(state)
            return 0

        if state.get("status") in {"failed", "instance_gone", "lost", "ssh_failed"}:
            host = str(state.get("host") or "")
            instance_id = str(state.get("instance_id") or "")
            remote_tail = _fetch_remote_log_tail(host, lines=500) if host else ""
            progress = state.get("progress") or {}
            failure_class = _classify_failure(
                status=str(state.get("status") or ""),
                remote_log=remote_tail,
                progress=progress,
            )
            launch_log = LAUNCH_LOG_DIR / f"launch_economist_rl_overnight_attempt_{int(state.get('attempt') or 0):02d}.log"

            if host and instance_id:
                _rsync_artifacts(
                    host,
                    attempt=int(state.get("attempt") or 0),
                    instance_id=instance_id,
                    failure_class=failure_class,
                    status=str(state.get("status") or ""),
                    progress=progress,
                    remote_tail=remote_tail,
                    launch_log=launch_log if launch_log.is_file() else None,
                )
            else:
                dest = _extract_dir(
                    attempt=int(state.get("attempt") or 0),
                    instance_id=instance_id or "unknown",
                    label=failure_class,
                )
                _write_extract_manifest(
                    dest,
                    attempt=int(state.get("attempt") or 0),
                    instance_id=instance_id,
                    host=host,
                    failure_class=failure_class,
                    status=str(state.get("status") or ""),
                    progress=progress,
                    remote_tail=remote_tail,
                    launch_log=launch_log if launch_log.is_file() else None,
                    rsync_paths=[],
                )

            if instance_id and _list_active_workers():
                _terminate_instance(instance_id, reason=f"watch_{state.get('status')}")

            restarts += 1
            if restarts > int(args.max_restarts):
                _log(f"watch_give_up restarts={restarts}")
                state["status"] = "gave_up"
                state["finished_utc"] = _utc_now()
                _write_state(state)
                return 1

            cycle_argv = _adapt_cycle_argv_after_failure(
                cycle_argv,
                progress=state.get("progress") or {},
                remote_log=remote_tail,
            )
            attempt += 1
            try:
                meta = _launch_worker(cycle_argv, attempt=attempt)
            except RuntimeError as exc:
                _log(f"relaunch_failed attempt={attempt} error={exc}")
                launch_log = LAUNCH_LOG_DIR / f"launch_economist_rl_overnight_attempt_{attempt:02d}.log"
                dest = _extract_dir(attempt=attempt, instance_id="launch_failed", label="launch_failed")
                _write_extract_manifest(
                    dest,
                    attempt=attempt,
                    instance_id="",
                    host="",
                    failure_class="launch_failed",
                    status="relaunch_failed",
                    progress=state.get("progress") or {},
                    remote_tail=remote_tail,
                    launch_log=launch_log if launch_log.is_file() else None,
                    rsync_paths=[],
                )
                state = {
                    "run_id": run_id,
                    "status": "relaunch_failed",
                    "last_failure_utc": _utc_now(),
                    "restarts": restarts,
                    "attempt": attempt,
                    "cycle_argv": cycle_argv,
                    "last_extract": str(dest.relative_to(ROOT)),
                }
                _write_state(state)
                time.sleep(min(poll_s, 300))
                continue
            state = {
                "run_id": run_id,
                "attempt": attempt,
                "restarts": restarts,
                "instance_id": meta.get("instance_id"),
                "host": meta.get("host"),
                "cycle_argv": cycle_argv,
                "status": "relaunched",
                "last_failure_utc": _utc_now(),
                "last_failure_class": failure_class,
                "extracts_root": str(EXTRACTS_ROOT.relative_to(ROOT)),
            }
            _write_state(state)
            time.sleep(30)
            continue

        if args.once:
            return 0

        if args.agent_notify:
            print(
                'AGENT_LOOP_TICK_economist_rl '
                + json.dumps(
                    {
                        "prompt": (
                            "Read logs/economist_rl_overnight_watch.json and "
                            "logs/economist_rl_overnight_watch.log. If the economistRL "
                            "Lambda overnight watch failed, is stuck with no log progress "
                            "for 2+ polls, or needs a code fix, diagnose and relaunch or "
                            "patch the watcher."
                        )
                    }
                ),
                flush=True,
            )

        time.sleep(poll_s)

    _log("watch_deadline_exceeded")
    state["status"] = "deadline"
    _write_state(state)
    return 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight economistRL Lambda watcher")
    parser.add_argument("--launch-initial", action="store_true", help="Launch a worker immediately.")
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete logs/economist_rl_overnight_watch.json before starting.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Label for this watch session (default run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--poll-minutes", type=float, default=10.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=14.0)
    parser.add_argument("--once", action="store_true", help="Single poll then exit.")
    parser.add_argument(
        "--agent-notify",
        action="store_true",
        help="Emit AGENT_LOOP_TICK_economist_rl JSON each poll for Cursor agent wake.",
    )
    parser.add_argument(
        "cycle_argv",
        nargs=argparse.REMAINDER,
        help="Optional cycle args after -- (override defaults).",
    )
    args = parser.parse_args(argv)
    if args.cycle_argv and args.cycle_argv[0] == "--":
        args.cycle_argv = args.cycle_argv[1:]
    if not args.cycle_argv:
        args.cycle_argv = list(DEFAULT_CYCLE_ARGV)
    return args


def main(argv: list[str] | None = None) -> int:
    return _run_watch(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
