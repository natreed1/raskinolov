#!/usr/bin/env python3
"""Launch one Lambda worker for the 121-task council conversation collection run."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from launch_lambda_parallel_ablation import (
    DEFAULT_LAMBDA_API_BASE,
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
    max_runtime_hours: float,
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
    prelude = "\n".join(
        _remote_lifecycle_prelude(
            api_base=api_base,
            api_key=api_key,
            instance_id=instance_id,
            auto_terminate=auto_terminate,
            max_runtime_hours=max_runtime_hours,
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
        + f"timeout --foreground \"${{FE_MAX_RUNTIME_SECONDS}}s\" bash -lc {shlex.quote(eval_cmd)}"
        + "\nEOF\n"
        + f"chmod 700 {remote_script}; "
        + f"tmux kill-session -t fe-council-eval-{label} >/dev/null 2>&1 || true; "
        + f"tmux new-session -d -s fe-council-eval-{label} '{remote_script} > ~/cloud-eval-logs/fe-council-eval-{label}.log 2>&1'",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a Lambda worker for council conversation eval collection.")
    parser.add_argument("--ml-repo", type=Path, default=ROOT)
    parser.add_argument("--ssh-key-path", type=Path, default=Path.home() / ".ssh" / "lambda_cloud_cursor")
    parser.add_argument("--ssh-key-name", default="lambda-cloud-cursor")
    parser.add_argument("--api-base", default=os.environ.get("LAMBDA_API_BASE", DEFAULT_LAMBDA_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument("--name-prefix", default="fe-council-eval")
    parser.add_argument("--instance-id", default="", help="Use an existing Lambda instance id instead of launching.")
    parser.add_argument("--launch-instance", action="store_true")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None, help="Terminate the worker when the remote eval script exits.")
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false", help="Leave the worker running after remote eval script exit.")
    parser.add_argument("--max-runtime-hours", type=float, default=12.0, help="Remote timeout before cleanup/termination runs.")
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
        )
        launched_instance = True
    else:
        raise SystemExit("Provide --instance-id or --launch-instance.")
    auto_terminate = args.auto_terminate
    if auto_terminate is None:
        auto_terminate = launched_instance

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
        max_runtime_hours=args.max_runtime_hours,
    )
    print("council_eval_started")
    print(f"auto_terminate={int(bool(auto_terminate))}")
    print(f"max_runtime_hours={args.max_runtime_hours}")
    print(f"instance_id={instance_ids[0]}")
    print(f"host={host}")
    print(f"session=fe-council-eval-{args.label}")
    print(f"log=~/cloud-eval-logs/fe-council-eval-{args.label}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
