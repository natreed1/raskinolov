#!/usr/bin/env python3
"""DEPRECATED: launch legacy economistRL serial PPO cycles on Lambda Cloud.

Do not use this launcher for new Lambda RL runs. Use
``scripts/launch_economist_rl_lambda_split_workers.py`` instead.

Provisions (or reuses) one GPU worker, syncs this repo + game repo, rsyncs the
registry economistRL adapter checkpoint, and runs:

  scripts/lambda/run_economist_rl_lambda_cycle.py --lambda-mode ...

That sets ``LOCAL_BACKEND=transformers`` for the Rollout Worker path and
``PPO_TRAIN_BACKEND=transformers`` for the isolated CUDA PPO Worker.

Deprecated example (debug/historical only):

  python scripts/launch_economist_rl_lambda_cycle.py --launch-instances \\
    -- --cycles 2 --eval-limit 20 --temperature 0.2

Deprecated example (reuse an existing instance for debug/historical runs):

  python scripts/launch_economist_rl_lambda_cycle.py \\
    --instance-ids <instance_id> \\
    -- --cycles 1 --rollouts-per-cycle 10 --eval-limit 5
"""

from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path
from typing import Any, List

import launch_lambda_parallel_ablation as lab

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOTENV = ROOT / ".env"


def _load_repo_dotenv(path: Path = DEFAULT_DOTENV) -> None:
    """Load ``KEY=VALUE`` lines from repo ``.env`` without overwriting exported vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    def _normalize_lambda_api_base(url: str) -> str:
        base = str(url or "").strip().rstrip("/")
        if base.endswith("/instances"):
            base = base[: -len("/instances")]
        # Repo .env sometimes uses marketing/docs hostnames that 403 on the instance API.
        if base and ("cloud.lambda.ai" in base or "api.lambda.ai" in base):
            return "https://cloud.lambdalabs.com/api/v1"
        return base

    for key in ("LAMBDA_CLOUD_BASE_URL", "LAMBDA_API_BASE", "LAMBDA_API_BASE_URL", "LAMBDA_URL"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        normalized = _normalize_lambda_api_base(raw)
        if normalized:
            os.environ[key] = normalized
    if not (os.environ.get("LAMBDA_CLOUD_BASE_URL") or os.environ.get("LAMBDA_API_BASE")):
        fallback = _normalize_lambda_api_base(os.environ.get("LAMBDA_URL") or "")
        if fallback:
            os.environ.setdefault("LAMBDA_CLOUD_BASE_URL", fallback)


_load_repo_dotenv()

DEFAULT_ADAPTER_ID = "economistRL"
RUNNER_SCRIPT = "scripts/lambda/run_economist_rl_lambda_cycle.py"
SESSION_NAME = "fe-economist-rl"
LOG_PATH = "/home/ubuntu/cloud-eval-logs/fe-economist-rl-cycle.log"
GPU_MONITOR_LOG_PATH = "/home/ubuntu/cloud-eval-logs/gpu-smi-economist-rl.csv"
DEPRECATION_NOTICE = (
    "DEPRECATED: scripts/launch_economist_rl_lambda_cycle.py is the legacy serial-cycle "
    "launcher. For Lambda RL runs, use scripts/launch_economist_rl_lambda_split_workers.py."
)
WORKER_DEFAULT_FLAGS = {
    "--rollouts-per-cycle": "25",
    "--bootstrap-rollouts-per-cycle": "50",
    "--ppo-min-samples": "25",
    "--ppo-max-samples": "64",
}


def _ensure_cycle_argv(argv: List[str]) -> List[str]:
    out = [str(item) for item in argv if str(item).strip()]
    if out and out[0] == "--":
        out = out[1:]
    if "--lambda-mode" not in out:
        out = ["--lambda-mode", *out]
    for flag, value in WORKER_DEFAULT_FLAGS.items():
        if flag not in out:
            out.extend([flag, value])
    if "--specialization" not in out:
        out.extend(["--specialization", "economist_rl"])
    return out


def _build_cycle_command(cycle_argv: List[str]) -> str:
    parts = [
        "python",
        RUNNER_SCRIPT,
        *cycle_argv,
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _write_and_start_economist_rl_runner(
    ssh_key_path: Path,
    host: str,
    instance_id: str,
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
    cycle_argv: List[str],
) -> None:
    command = _build_cycle_command(cycle_argv)
    remote_script = "/tmp/run_economist_rl_cycle.sh"
    local_script = Path(f"/tmp/run_economist_rl_cycle_{int(time.time())}.sh")
    local_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *lab._remote_lifecycle_prelude(
                    api_base=api_base,
                    api_key=api_key,
                    instance_id=instance_id,
                    auto_terminate=auto_terminate,
                    watchdog_enabled=watchdog_enabled,
                    watchdog_idle_minutes=watchdog_idle_minutes,
                    watchdog_check_minutes=watchdog_check_minutes,
                    watchdog_log_path=LOG_PATH,
                    gpu_monitor_log_path=GPU_MONITOR_LOG_PATH,
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
                "export PPO_TRAIN_BACKEND=transformers",
                "export PYTHONPATH=scripts",
                "export ECONOMIST_RL_SOURCE_REPO=${HOME}/fallen-empire",
                f"bash -lc {shlex.quote(command)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        lab._scp(ssh_key_path, local_script, host, remote_script)
    finally:
        local_script.unlink(missing_ok=True)
    lab._ssh(
        ssh_key_path,
        host,
        (
            f"chmod 700 {remote_script}; "
            f"tmux kill-session -t {SESSION_NAME} >/dev/null 2>&1 || true; "
            f"tmux new-session -d -s {SESSION_NAME} '{remote_script} > {LOG_PATH} 2>&1'"
        ),
    )


def _ensure_remote_python_deps(ssh_key_path: Path, host: str) -> None:
    lab._ssh(
        ssh_key_path,
        host,
        "set -euo pipefail; "
        "cd ~/fallen-empire-lora; "
        "source .venv-linux-port/bin/activate; "
        "pip install -U pip; "
        "pip install 'transformers==4.57.6' 'peft>=0.13.0,<0.20' bitsandbytes safetensors huggingface_hub accelerate sentencepiece",
    )


def _ensure_remote_game_deps(ssh_key_path: Path, host: str) -> None:
    """Vitest sandbox compile requires ``npm ci`` in the synced game checkout."""
    lab._ssh(
        ssh_key_path,
        host,
        "set -euo pipefail; cd ~/fallen-empire; if [ -f package-lock.json ]; then npm ci; else npm install; fi",
    )


def _parse_launch_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="DEPRECATED: launch legacy economistRL serial PPO cycles on Lambda.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "New Lambda RL runs should use:\n"
            "  python scripts/launch_economist_rl_lambda_split_workers.py --launch-instances -- ...\n\n"
            "Pass cycle-runner flags after `--`, e.g.\n"
            "  python scripts/launch_economist_rl_lambda_cycle.py --launch-instances \\\n"
            "    -- --cycles 2 --rollouts-per-cycle 50 --eval-limit 20"
        ),
    )
    parser.add_argument("--ml-repo", type=Path, default=ROOT)
    parser.add_argument("--game-repo", type=Path, default=Path.home() / "fallen-empire")
    parser.add_argument("--ssh-key-path", type=Path, default=Path.home() / ".ssh" / "lambda_cloud_cursor")
    parser.add_argument("--ssh-key-name", default="lambda-cloud-cursor")
    parser.add_argument(
        "--api-base",
        default=lab._lambda_cloud_base_url_from_env(),
        help="Lambda Cloud API base URL.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument("--fallback-region", action="append", default=[])
    parser.add_argument("--fallback-instance-type", action="append", default=[])
    parser.add_argument(
        "--file-system-id",
        default=os.environ.get("LAMBDA_FILE_SYSTEM_ID", lab.DEFAULT_LAMBDA_FILE_SYSTEM_ID),
    )
    parser.add_argument("--file-system-name", default=os.environ.get("LAMBDA_FILE_SYSTEM_NAME", ""))
    parser.add_argument("--no-file-system", action="store_true")
    parser.add_argument("--name-prefix", default="fe-economist-rl")
    parser.add_argument("--launch-instances", action="store_true")
    parser.add_argument("--instance-ids", default="")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None)
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false")
    parser.add_argument("--cleanup-on-setup-failure", dest="cleanup_on_setup_failure", action="store_true", default=True)
    parser.add_argument("--no-cleanup-on-setup-failure", dest="cleanup_on_setup_failure", action="store_false")
    parser.add_argument("--watchdog", dest="watchdog_enabled", action="store_true", default=None)
    parser.add_argument("--no-watchdog", dest="watchdog_enabled", action="store_false")
    parser.add_argument("--watchdog-idle-minutes", type=float, default=180.0)
    parser.add_argument("--watchdog-check-minutes", type=float, default=10.0)
    parser.add_argument("--artifact-export-command", default=os.environ.get("FE_ARTIFACT_EXPORT_COMMAND", ""))
    parser.add_argument("--artifact-staging-dir", default=os.environ.get("FE_ARTIFACT_STAGING_DIR", ""))
    parser.add_argument("--artifact-upload-every-steps", type=int, default=0)
    parser.add_argument("--artifact-upload-every-minutes", type=float, default=15.0)
    parser.add_argument(
        "--require-artifact-export-before-terminate",
        dest="require_artifact_export_before_terminate",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-terminate-without-artifact-export",
        dest="require_artifact_export_before_terminate",
        action="store_false",
    )
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-adapter-sync", action="store_true")
    parser.add_argument("--setup-command-retries", type=int, default=1)
    parser.add_argument("--setup-retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument(
        "--adapter-id",
        default=DEFAULT_ADAPTER_ID,
        help="Registry adapter id to rsync before the cycle (default economistRL / seed_bootstrap).",
    )
    parser.add_argument(
        "cycle_argv",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_economist_rl_lambda_cycle.py (prefix with --).",
    )
    args = parser.parse_args()
    return args, _ensure_cycle_argv(args.cycle_argv)


def main() -> int:
    args, cycle_argv = _parse_launch_args()
    print(DEPRECATION_NOTICE, flush=True)
    if not args.api_key:
        raise SystemExit("LAMBDA_API_KEY or --api-key is required.")

    file_system: dict[str, Any] = {}
    file_system_names: List[str] = []
    file_system_mount_point = ""
    if not args.no_file_system:
        file_system_identifier = str(args.file_system_name or args.file_system_id or "").strip()
        if file_system_identifier:
            file_system = lab._resolve_file_system(args.api_base, args.api_key, file_system_identifier)
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
    artifact_staging_is_durable = bool(
        file_system_mount_point and artifact_staging_dir.startswith(file_system_mount_point.rstrip("/"))
    )

    instance_ids: List[str] = []
    launched_instances = False
    selected_launch_candidate = lab.LaunchCandidate(
        instance_type=str(args.instance_type).strip(),
        region=effective_region,
    )
    started_instance_ids: set[str] = set()

    if args.instance_ids.strip():
        instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]
        if len(instance_ids) != 1:
            raise SystemExit("Provide exactly one --instance-ids value for economistRL cycle runs.")
    elif args.launch_instances:
        instance_ids, selected_launch_candidate = lab._launch_instances_with_fallback(
            args.api_base,
            args.api_key,
            requested_region=effective_region,
            requested_instance_type=args.instance_type,
            fallback_regions=lab._split_csv_cli_values(args.fallback_region),
            fallback_instance_types=lab._split_csv_cli_values(args.fallback_instance_type),
            ssh_key_name=args.ssh_key_name,
            quantity=1,
            name_prefix=args.name_prefix,
            file_system_names=file_system_names,
        )
        launched_instances = True
    else:
        raise SystemExit("Provide --launch-instances or --instance-ids.")

    auto_terminate = args.auto_terminate
    if auto_terminate is None:
        auto_terminate = launched_instances
    watchdog_enabled = args.watchdog_enabled
    if watchdog_enabled is None:
        watchdog_enabled = auto_terminate

    instance_id = instance_ids[0]
    try:
        id_to_ip = lab._wait_for_instance_ips(args.api_base, args.api_key, instance_ids)
        host = id_to_ip[instance_id]

        lab._run_workers_parallel(
            "bootstrap",
            [(host, None)],
            lambda item: lab._bootstrap_worker(args.ssh_key_path, item[0], file_system_mount_point),
            retries=args.setup_command_retries,
            retry_sleep_s=args.setup_retry_sleep_seconds,
        )
        if not args.skip_sync:
            lab._run_workers_parallel(
                "sync",
                [(host, None)],
                lambda item: lab._sync_repos(args.ssh_key_path, item[0], args.ml_repo, args.game_repo),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )
            lab._run_workers_parallel(
                "deps",
                [(host, None)],
                lambda item: _ensure_remote_python_deps(args.ssh_key_path, item[0]),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )
            lab._run_workers_parallel(
                "game_deps",
                [(host, None)],
                lambda item: _ensure_remote_game_deps(args.ssh_key_path, item[0]),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )
        if not args.skip_adapter_sync:
            lab._run_workers_parallel(
                "adapter_sync",
                [(host, str(args.adapter_id).strip())],
                lambda item: lab._sync_adapter_checkpoint(
                    args.ssh_key_path,
                    item[0],
                    args.ml_repo,
                    item[1],
                ),
                retries=args.setup_command_retries,
                retry_sleep_s=args.setup_retry_sleep_seconds,
            )

        _write_and_start_economist_rl_runner(
            args.ssh_key_path,
            host,
            instance_id,
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
            cycle_argv=cycle_argv,
        )
        started_instance_ids.add(instance_id)
    except Exception:
        if launched_instances and args.cleanup_on_setup_failure:
            cleanup_ids = [iid for iid in instance_ids if iid not in started_instance_ids]
            lab._terminate_instances_best_effort(
                args.api_base,
                args.api_key,
                cleanup_ids,
                reason="setup_failure_before_remote_start",
            )
        raise

    print("economist_rl_lambda_cycle_started")
    print(f"instance_id={instance_id}")
    print(f"host={host}")
    print(f"tmux_session={SESSION_NAME}")
    print(f"remote_log={LOG_PATH}")
    print(f"auto_terminate={int(bool(auto_terminate))}")
    print(f"watchdog_enabled={int(bool(watchdog_enabled))}")
    print(f"watchdog_idle_minutes={args.watchdog_idle_minutes}")
    print(f"selected_launch_region={selected_launch_candidate.region}")
    print(f"selected_launch_instance_type={selected_launch_candidate.instance_type}")
    print(f"cycle_command={_build_cycle_command(cycle_argv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
