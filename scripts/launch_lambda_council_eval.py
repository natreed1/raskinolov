#!/usr/bin/env python3
"""Launch one Lambda worker for the 121-task council conversation collection run."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from launch_lambda_parallel_ablation import (
    _lambda_cloud_base_url_from_env,
    _bootstrap_worker,
    _launch_instances,
    _remote_lifecycle_prelude,
    _run_workers_parallel,
    _shell,
    _ssh,
    _wait_for_instance_ips,
)

ROOT = Path(__file__).resolve().parents[1]


def _sync_ml_repo(ssh_key_path: Path, host: str, ml_repo: Path) -> None:
    _shell(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -i {ssh_key_path}",
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
    _ssh(
        ssh_key_path,
        host,
        "set -euo pipefail; cd ~/fallen-empire-lora; "
        "python3 -m venv .venv-linux-port; "
        "source .venv-linux-port/bin/activate; "
        "pip install -U pip; "
        "pip install torch --index-url https://download.pytorch.org/whl/cu121; "
        "pip install transformers accelerate sentencepiece numpy safetensors",
    )


def _start_council_eval(
    *,
    ssh_key_path: Path,
    host: str,
    instance_id: str,
    api_base: str,
    api_key: str,
    label: str,
    debate_max_rounds: int,
    participant_max_tokens: int,
    local_model: str,
    remote_smoke: bool,
    auto_terminate: bool,
    watchdog_enabled: bool,
    watchdog_idle_minutes: float,
    watchdog_check_minutes: float,
    artifact_export_command: str,
    artifact_upload_every_steps: int,
    artifact_upload_every_minutes: float,
    require_artifact_export_before_terminate: bool,
) -> None:
    smoke_cmd = ""
    if remote_smoke:
        smoke_cmd = (
            "python scripts/run_council_conversation_eval.py "
            "--max-tasks 1 "
            "--mock-generation "
            "--participant-max-tokens 64 "
            "--rows-jsonl benchmarks/results/council_conversation_remote_smoke_rows.jsonl "
            "--summary-json benchmarks/results/council_conversation_remote_smoke_summary.json; "
        )
    full_cmd = (
        "python scripts/run_council_conversation_eval.py "
        f"--debate-max-rounds {int(debate_max_rounds)} "
        f"--participant-max-tokens {int(participant_max_tokens)} "
        f"--local-model {local_model} "
        f"--rows-jsonl benchmarks/results/council_conversation_eval_rows_{label}.jsonl "
        f"--summary-json benchmarks/results/council_conversation_eval_summary_{label}.json"
    )
    eval_cmd = smoke_cmd + full_cmd
    remote_script = f"/tmp/run_council_eval_{label}.sh"
    log_path = f"/home/ubuntu/cloud-eval-logs/fe-council-eval-{label}.log"
    gpu_monitor_log_path = f"/home/ubuntu/cloud-eval-logs/gpu-smi-council-{label}.csv"
    prelude = "\n".join(
        _remote_lifecycle_prelude(
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
            artifact_staging_dir="${HOME}/cloud-eval-artifacts",
            artifact_staging_is_durable=False,
        )
    )
    _ssh(
        ssh_key_path,
        host,
        "cat > "
        + remote_script
        + " <<'EOF'\n"
        + "#!/usr/bin/env bash\n"
        + "set -euo pipefail\n"
        + prelude
        + "\n"
        + "cd ~/fallen-empire-lora\n"
        + "source .venv-linux-port/bin/activate\n"
        + "export LOCAL_BACKEND=transformers\n"
        + "export ROUTER_COUNCIL_ENABLED=1\n"
        + "export ROUTER_COUNCIL_DEBATE_MAX_ROUNDS="
        + str(int(debate_max_rounds))
        + "\n"
        + f"bash -lc {shlex.quote(eval_cmd)}"
        + "\nEOF\n"
        + f"chmod 700 {remote_script}; "
        + f"tmux kill-session -t fe-council-eval-{label} >/dev/null 2>&1 || true; "
        + f"tmux new-session -d -s fe-council-eval-{label} '{remote_script} > {log_path} 2>&1'",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a Lambda worker for council conversation eval collection.")
    parser.add_argument("--ml-repo", type=Path, default=ROOT)
    parser.add_argument("--ssh-key-path", type=Path, default=Path.home() / ".ssh" / "lambda_cloud_cursor")
    parser.add_argument("--ssh-key-name", default="lambda-cloud-cursor")
    parser.add_argument(
        "--api-base",
        default=_lambda_cloud_base_url_from_env(),
        help="Lambda Cloud API base URL. Defaults to LAMBDA_CLOUD_BASE_URL, then legacy LAMBDA_API_BASE.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument("--name-prefix", default="fe-council-eval")
    parser.add_argument("--instance-id", default="", help="Use an existing Lambda instance id instead of launching.")
    parser.add_argument("--launch-instance", action="store_true")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None, help="Terminate the worker when the remote eval script exits.")
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false", help="Leave the worker running after remote eval script exit.")
    parser.add_argument("--watchdog", dest="watchdog_enabled", action="store_true", default=None, help="Terminate the worker when its run log is idle for --watchdog-idle-minutes.")
    parser.add_argument("--no-watchdog", dest="watchdog_enabled", action="store_false", help="Disable idle-log watchdog termination.")
    parser.add_argument("--watchdog-idle-minutes", type=float, default=60.0, help="Idle log minutes before watchdog termination.")
    parser.add_argument("--watchdog-check-minutes", type=float, default=5.0, help="Minutes between watchdog log-idle checks.")
    parser.add_argument("--artifact-export-command", default=os.environ.get("FE_ARTIFACT_EXPORT_COMMAND", ""), help="Shell command that durably exports FE_ARTIFACT_TARBALL.")
    parser.add_argument("--artifact-upload-every-steps", type=int, default=5, help="Run artifact checkpoint command every N completed rows; 0 disables step checkpoints.")
    parser.add_argument("--artifact-upload-every-minutes", type=float, default=10.0, help="Run artifact checkpoint command after N minutes since last checkpoint; 0 disables time checkpoints.")
    parser.add_argument("--require-artifact-export-before-terminate", dest="require_artifact_export_before_terminate", action="store_true", default=True, help="Skip auto-termination if final artifact export fails.")
    parser.add_argument("--allow-terminate-without-artifact-export", dest="require_artifact_export_before_terminate", action="store_false", help="Terminate even if final artifact export is missing or fails.")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--label", default="v1")
    parser.add_argument("--debate-max-rounds", type=int, default=2)
    parser.add_argument("--participant-max-tokens", type=int, default=256)
    parser.add_argument("--local-model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    parser.add_argument("--no-remote-smoke", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("LAMBDA_API_KEY or --api-key is required.")
    if args.instance_id.strip():
        instance_ids = [args.instance_id.strip()]
        launched_instance = False
    elif args.launch_instance:
        instance_ids = _launch_instances(
            args.api_base,
            args.api_key,
            region=args.region,
            instance_type=args.instance_type,
            ssh_key_name=args.ssh_key_name,
            quantity=1,
            name_prefix=args.name_prefix,
            file_system_names=[],
        )
        launched_instance = True
    else:
        raise SystemExit("Provide --instance-id or --launch-instance.")
    auto_terminate = args.auto_terminate
    if auto_terminate is None:
        auto_terminate = launched_instance
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

    id_to_ip = _wait_for_instance_ips(args.api_base, args.api_key, instance_ids)
    host = id_to_ip[instance_ids[0]]
    _run_workers_parallel("bootstrap", [(host, None)], lambda item: _bootstrap_worker(args.ssh_key_path, item[0]))
    if not args.skip_sync:
        _run_workers_parallel("sync", [(host, None)], lambda item: _sync_ml_repo(args.ssh_key_path, item[0], args.ml_repo))
    _start_council_eval(
        ssh_key_path=args.ssh_key_path,
        host=host,
        instance_id=instance_ids[0],
        api_base=args.api_base,
        api_key=args.api_key,
        label=args.label,
        debate_max_rounds=args.debate_max_rounds,
        participant_max_tokens=args.participant_max_tokens,
        local_model=args.local_model,
        remote_smoke=not bool(args.no_remote_smoke),
        auto_terminate=auto_terminate,
        watchdog_enabled=watchdog_enabled,
        watchdog_idle_minutes=args.watchdog_idle_minutes,
        watchdog_check_minutes=args.watchdog_check_minutes,
        artifact_export_command=args.artifact_export_command,
        artifact_upload_every_steps=args.artifact_upload_every_steps,
        artifact_upload_every_minutes=args.artifact_upload_every_minutes,
        require_artifact_export_before_terminate=args.require_artifact_export_before_terminate,
    )
    print("council_eval_started")
    print(f"auto_terminate={int(bool(auto_terminate))}")
    print(f"watchdog_enabled={int(bool(watchdog_enabled))}")
    print(f"watchdog_idle_minutes={args.watchdog_idle_minutes}")
    print(f"watchdog_check_minutes={args.watchdog_check_minutes}")
    print(f"artifact_export_configured={int(bool(args.artifact_export_command.strip()))}")
    print(f"artifact_upload_every_steps={args.artifact_upload_every_steps}")
    print(f"artifact_upload_every_minutes={args.artifact_upload_every_minutes}")
    print(f"require_artifact_export_before_terminate={int(bool(args.require_artifact_export_before_terminate))}")
    print(f"instance_id={instance_ids[0]}")
    print(f"host={host}")
    print(f"session=fe-council-eval-{args.label}")
    print(f"log=~/cloud-eval-logs/fe-council-eval-{args.label}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
