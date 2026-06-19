#!/usr/bin/env python3
"""Launch the preferred economistRL split-worker PPO system on Lambda Cloud.

This launcher is intentionally separate from the legacy cycle launcher. It starts
``scripts/lambda/run_economist_rl_split_workers.py``, which keeps a constant
Rollout Worker queue and lets the PPO Optimisation Worker promote trained
adapters back into subsequent rollout batches.
"""

from __future__ import annotations

import argparse
import os
import shlex
import time
from pathlib import Path
from typing import Any, List

import launch_lambda_parallel_ablation as lab
from launch_economist_rl_lambda_cycle import (
    DEFAULT_ADAPTER_ID,
    _ensure_remote_game_deps,
    _ensure_remote_python_deps,
    _load_repo_dotenv,
)

ROOT = Path(__file__).resolve().parents[1]
_load_repo_dotenv()

RUNNER_SCRIPT = "scripts/lambda/run_economist_rl_split_workers.py"
SESSION_NAME = "fe-economist-rl-split"
LOG_PATH = "/home/ubuntu/cloud-eval-logs/fe-economist-rl-split-workers.log"
GPU_MONITOR_LOG_PATH = "/home/ubuntu/cloud-eval-logs/gpu-smi-economist-rl-split.csv"
MIN_SPLIT_WORKER_WATCHDOG_IDLE_MINUTES = 300.0
SPLIT_WORKER_DEFAULT_FLAGS = {
    "--rollout-batch-size": "25",
    "--bootstrap-rollout-batch-size": "50",
    "--ppo-min-samples": "25",
    "--ppo-max-samples": "64",
    "--ppo-mini-batch-size": "8",
    "--acceptance-mode": "trained_marker",
}


def _flag_value(argv: List[str], flag: str) -> str:
    if flag not in argv:
        return ""
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return ""
    return str(argv[idx + 1]).strip()


def _custom_init_adapter_sync_path(split_worker_argv: List[str], *, default_adapter_id: str) -> str:
    raw = _flag_value(split_worker_argv, "--init-adapter-path")
    if not raw:
        return ""
    normalized = raw.strip().rstrip("/")
    default_paths = {
        "checkpoints/fe-lora-arena-apply-sft",
        f"checkpoints/adapters/{default_adapter_id}",
    }
    if normalized in default_paths:
        return ""
    return raw


def _hf_cache_model_id(split_worker_argv: List[str]) -> str:
    model = _flag_value(split_worker_argv, "--base-model") or "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    model = model.strip()
    if model.startswith("mlx-community/") and "Qwen2.5-Coder-" in model:
        suffix = model.split("/", 1)[1]
        if suffix.endswith("-4bit"):
            suffix = suffix[: -len("-4bit")]
        return f"Qwen/{suffix}"
    return model


def _ensure_split_worker_argv(argv: List[str]) -> List[str]:
    out = [str(item) for item in argv if str(item).strip()]
    if out and out[0] == "--":
        out = out[1:]
    if "--lambda-mode" not in out:
        out = ["--lambda-mode", *out]
    for flag, value in SPLIT_WORKER_DEFAULT_FLAGS.items():
        if flag not in out:
            out.extend([flag, value])
    if "--specialization" not in out:
        out.extend(["--specialization", "economist_rl"])
    return out


def _build_split_worker_command(split_worker_argv: List[str]) -> str:
    parts = ["python", RUNNER_SCRIPT, *split_worker_argv]
    return " ".join(shlex.quote(part) for part in parts)


def _effective_watchdog_idle_minutes(requested_minutes: float) -> float:
    return max(MIN_SPLIT_WORKER_WATCHDOG_IDLE_MINUTES, float(requested_minutes))


def _write_and_start_split_worker_runner(
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
    watchdog_max_runtime_minutes: float,
    artifact_export_command: str,
    artifact_upload_every_steps: int,
    artifact_upload_every_minutes: float,
    require_artifact_export_before_terminate: bool,
    artifact_staging_dir: str,
    artifact_staging_is_durable: bool,
    split_worker_argv: List[str],
) -> None:
    command = _build_split_worker_command(split_worker_argv)
    hf_cache_model = _hf_cache_model_id(split_worker_argv)
    remote_script = "/tmp/run_economist_rl_split_workers.sh"
    local_script = Path(f"/tmp/run_economist_rl_split_workers_{int(time.time())}.sh")
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
                    watchdog_max_runtime_minutes=watchdog_max_runtime_minutes,
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
                "python - <<'PY'",
                "from huggingface_hub import snapshot_download",
                f"model_id = {hf_cache_model!r}",
                "print(f'[hf-cache] verifying {model_id}', flush=True)",
                "snapshot_download(model_id, resume_download=True)",
                "print(f'[hf-cache] ready {model_id}', flush=True)",
                "PY",
                "export LOCAL_BACKEND=transformers",
                "export PPO_TRAIN_BACKEND=transformers",
                "export FE_REQUIRE_LOCAL_HF_CACHE=1",
                "export HF_HUB_OFFLINE=1",
                "export TRANSFORMERS_OFFLINE=1",
                "export HF_HUB_DISABLE_TELEMETRY=1",
                "export PYTHONPATH=scripts:scripts/lambda",
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


def _parse_launch_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Launch economistRL split-worker PPO orchestration on Lambda.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ml-repo", type=Path, default=ROOT)
    parser.add_argument("--game-repo", type=Path, default=Path.home() / "fallen-empire")
    parser.add_argument("--ssh-key-path", type=Path, default=Path.home() / ".ssh" / "lambda_cloud_cursor")
    parser.add_argument("--ssh-key-name", default="lambda-cloud-cursor")
    parser.add_argument("--api-base", default=lab._lambda_cloud_base_url_from_env())
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument("--fallback-region", action="append", default=[])
    parser.add_argument("--fallback-instance-type", action="append", default=[])
    parser.add_argument("--file-system-id", default=os.environ.get("LAMBDA_FILE_SYSTEM_ID", lab.DEFAULT_LAMBDA_FILE_SYSTEM_ID))
    parser.add_argument("--file-system-name", default=os.environ.get("LAMBDA_FILE_SYSTEM_NAME", ""))
    parser.add_argument("--no-file-system", action="store_true")
    parser.add_argument("--name-prefix", default="fe-economist-rl-split")
    parser.add_argument("--launch-instances", action="store_true")
    parser.add_argument("--instance-ids", default="")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None)
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false")
    parser.add_argument("--cleanup-on-setup-failure", dest="cleanup_on_setup_failure", action="store_true", default=True)
    parser.add_argument("--no-cleanup-on-setup-failure", dest="cleanup_on_setup_failure", action="store_false")
    parser.add_argument("--watchdog", dest="watchdog_enabled", action="store_true", default=None)
    parser.add_argument("--no-watchdog", dest="watchdog_enabled", action="store_false")
    parser.add_argument("--watchdog-idle-minutes", type=float, default=MIN_SPLIT_WORKER_WATCHDOG_IDLE_MINUTES)
    parser.add_argument("--watchdog-check-minutes", type=float, default=10.0)
    parser.add_argument("--watchdog-max-runtime-minutes", type=float, default=1080.0)
    parser.add_argument("--artifact-export-command", default=os.environ.get("FE_ARTIFACT_EXPORT_COMMAND", ""))
    parser.add_argument("--artifact-staging-dir", default=os.environ.get("FE_ARTIFACT_STAGING_DIR", ""))
    parser.add_argument("--artifact-upload-every-steps", type=int, default=0)
    parser.add_argument("--artifact-upload-every-minutes", type=float, default=15.0)
    parser.add_argument("--require-artifact-export-before-terminate", dest="require_artifact_export_before_terminate", action="store_true", default=True)
    parser.add_argument("--allow-terminate-without-artifact-export", dest="require_artifact_export_before_terminate", action="store_false")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-adapter-sync", action="store_true")
    parser.add_argument(
        "--skip-registry-adapter-sync",
        action="store_true",
        help="Skip registry adapter sync but still sync a custom --init-adapter-path when provided.",
    )
    parser.add_argument("--setup-command-retries", type=int, default=1)
    parser.add_argument("--setup-retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("split_worker_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args, _ensure_split_worker_argv(args.split_worker_argv)


def main() -> int:
    args, split_worker_argv = _parse_launch_args()
    if not args.api_key:
        raise SystemExit("LAMBDA_API_KEY or --api-key is required.")

    file_system: dict[str, Any] = {}
    file_system_names: List[str] = []
    file_system_mount_point = ""
    if not args.no_file_system:
        identifier = str(args.file_system_name or args.file_system_id or "").strip()
        if identifier:
            file_system = lab._resolve_file_system(args.api_base, args.api_key, identifier)
            file_system_names = [str(file_system.get("name") or identifier)]
            file_system_mount_point = str(file_system.get("mount_point") or "").strip()

    effective_region = str(args.region or "").strip()
    if not effective_region:
        effective_region = str(((file_system.get("region") or {}).get("name") if file_system else "") or "us-east-1")

    artifact_staging_dir = str(args.artifact_staging_dir or "").strip()
    if not artifact_staging_dir:
        artifact_staging_dir = (
            f"{file_system_mount_point.rstrip('/')}/fallen-empire-lora-artifacts"
            if file_system_mount_point
            else "${HOME}/cloud-eval-artifacts"
        )
    artifact_staging_is_durable = bool(
        file_system_mount_point and artifact_staging_dir.startswith(file_system_mount_point.rstrip("/"))
    )

    launched_instances = False
    if args.instance_ids.strip():
        instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]
        if len(instance_ids) != 1:
            raise SystemExit("Provide exactly one --instance-ids value for split-worker runs.")
        selected_launch_candidate = lab.LaunchCandidate(instance_type=str(args.instance_type), region=effective_region)
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

    auto_terminate = launched_instances if args.auto_terminate is None else args.auto_terminate
    watchdog_enabled = auto_terminate if args.watchdog_enabled is None else args.watchdog_enabled
    if args.watchdog_idle_minutes <= 0:
        raise SystemExit("--watchdog-idle-minutes must be > 0.")
    effective_watchdog_idle_minutes = _effective_watchdog_idle_minutes(args.watchdog_idle_minutes)
    if args.watchdog_max_runtime_minutes <= 0:
        raise SystemExit("--watchdog-max-runtime-minutes must be > 0.")
    instance_id = instance_ids[0]
    started_instance_ids: set[str] = set()

    try:
        host = lab._wait_for_instance_ips(args.api_base, args.api_key, instance_ids)[instance_id]
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
            custom_init_adapter = _custom_init_adapter_sync_path(
                split_worker_argv,
                default_adapter_id=str(args.adapter_id).strip() or DEFAULT_ADAPTER_ID,
            )
            if not args.skip_registry_adapter_sync:
                lab._run_workers_parallel(
                    "adapter_sync",
                    [(host, str(args.adapter_id).strip())],
                    lambda item: lab._sync_adapter_checkpoint(args.ssh_key_path, item[0], args.ml_repo, item[1]),
                    retries=args.setup_command_retries,
                    retry_sleep_s=args.setup_retry_sleep_seconds,
                )
            elif not custom_init_adapter:
                print("adapter_sync_skipped reason=skip_registry_adapter_sync no_custom_init_adapter", flush=True)
            if custom_init_adapter:
                lab._run_workers_parallel(
                    "custom_init_adapter_sync",
                    [(host, custom_init_adapter)],
                    lambda item: lab._sync_custom_adapter_path(args.ssh_key_path, item[0], args.ml_repo, item[1]),
                    retries=args.setup_command_retries,
                    retry_sleep_s=args.setup_retry_sleep_seconds,
                )

        _write_and_start_split_worker_runner(
            args.ssh_key_path,
            host,
            instance_id,
            api_base=args.api_base,
            api_key=args.api_key,
            auto_terminate=auto_terminate,
            watchdog_enabled=watchdog_enabled,
            watchdog_idle_minutes=effective_watchdog_idle_minutes,
            watchdog_check_minutes=args.watchdog_check_minutes,
            watchdog_max_runtime_minutes=args.watchdog_max_runtime_minutes,
            artifact_export_command=args.artifact_export_command,
            artifact_upload_every_steps=args.artifact_upload_every_steps,
            artifact_upload_every_minutes=args.artifact_upload_every_minutes,
            require_artifact_export_before_terminate=args.require_artifact_export_before_terminate,
            artifact_staging_dir=artifact_staging_dir,
            artifact_staging_is_durable=artifact_staging_is_durable,
            split_worker_argv=split_worker_argv,
        )
        started_instance_ids.add(instance_id)
        print("economist_rl_split_worker_started")
        print(f"instance_id={instance_id}")
        print(f"host={host}")
        print(f"tmux_session={SESSION_NAME}")
        print(f"remote_log={LOG_PATH}")
        print(f"selected_launch_region={selected_launch_candidate.region}")
        print(f"selected_launch_instance_type={selected_launch_candidate.instance_type}")
        print(f"watchdog_idle_minutes={effective_watchdog_idle_minutes}")
        print(f"split_worker_command={_build_split_worker_command(split_worker_argv)}")
    except Exception:
        if launched_instances and args.cleanup_on_setup_failure:
            cleanup_ids = [iid for iid in instance_ids if iid not in started_instance_ids]
            lab._terminate_instances_best_effort(
                args.api_base,
                args.api_key,
                cleanup_ids,
                reason="split_worker_setup_failure_before_remote_start",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
