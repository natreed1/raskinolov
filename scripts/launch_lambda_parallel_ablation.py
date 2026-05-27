#!/usr/bin/env python3
"""
Launch and dispatch prompt-ablation cells across Lambda Cloud workers in parallel.

Default matrix cells (one worker per cell):
  - style_rag
  - context_max_potential
  - style_plus_max_potential

Use `--include-baseline` when a fresh control run is needed.

Each worker runs one `run_final_mass_testing_system.py` invocation with isolated
worktree root + output artifacts.

Example:
  python scripts/launch_lambda_parallel_ablation.py \
    --launch-instances \
    --instance-type gpu_1x_a10 \
    --region us-east-1 \
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
DEFAULT_LAMBDA_API_BASE = "https://cloud.lambda.ai/api/v1"
DEFAULT_STYLE_CORPUS = "data/rag/style_prompt_rag_v2_failure_guided.json"


@dataclass
class MatrixCell:
    name: str
    extra_args: List[str]


BASELINE_CELL = MatrixCell("baseline", [])

DEFAULT_CELLS: List[MatrixCell] = [
    MatrixCell(
        "style_rag",
        [
            "--prompt-style-rag-corpus",
            DEFAULT_STYLE_CORPUS,
            "--prompt-style-rag-top-k",
            "3",
            "--prompt-style-rag-max-chars",
            "1600",
        ],
    ),
    MatrixCell("context_max_potential", ["--prompt-context-engineering", "max_potential"]),
    MatrixCell(
        "style_plus_max_potential",
        [
            "--prompt-style-rag-corpus",
            DEFAULT_STYLE_CORPUS,
            "--prompt-style-rag-top-k",
            "3",
            "--prompt-style-rag-max-chars",
            "1600",
            "--prompt-context-engineering",
            "max_potential",
        ],
    ),
]


def _api_call(api_base: str, api_key: str, method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("utf-8")
    req = request.Request(
        f"{api_base.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "fe-lambda-parallel-ablation/1.0",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lambda API {method} {path} HTTP {exc.code}: {detail[:500]}") from exc


def _shell(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _launch_instances(api_base: str, api_key: str, *, region: str, instance_type: str, ssh_key_name: str, quantity: int, name_prefix: str) -> List[str]:
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "quantity": quantity,
        "name": f"{name_prefix}-{int(time.time())}",
    }
    try:
        resp = _api_call(api_base, api_key, "POST", "/instance-operations/launch", payload)
        ids = list((resp.get("data") or {}).get("instance_ids") or [])
        if len(ids) != quantity:
            raise RuntimeError(f"Expected {quantity} launched instances, got {len(ids)}: {ids}")
        return ids
    except RuntimeError as exc:
        if quantity <= 1 or "Specify at most 1 instances" not in str(exc):
            raise

    # Some Lambda accounts only allow launching one instance per API request.
    ids: List[str] = []
    for idx in range(quantity):
        single_payload = dict(payload)
        single_payload["quantity"] = 1
        single_payload["name"] = f"{name_prefix}-{idx + 1}-{int(time.time())}"
        resp = _api_call(api_base, api_key, "POST", "/instance-operations/launch", single_payload)
        launched = list((resp.get("data") or {}).get("instance_ids") or [])
        if len(launched) != 1:
            raise RuntimeError(f"Expected one launched instance, got {launched}")
        ids.extend(launched)
    return ids


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


def _remote_lifecycle_prelude(
    *,
    api_base: str,
    api_key: str,
    instance_id: str,
    auto_terminate: bool,
    max_runtime_hours: float,
) -> List[str]:
    lines = [
        f"export FE_MAX_RUNTIME_SECONDS={max(1, int(float(max_runtime_hours) * 3600))}",
        f"export FE_AUTO_TERMINATE={'1' if auto_terminate else '0'}",
        f"export FE_LAMBDA_API_BASE={shlex.quote(api_base)}",
        f"export FE_LAMBDA_API_KEY={shlex.quote(api_key)}",
        f"export FE_LAMBDA_INSTANCE_ID={shlex.quote(instance_id)}",
        "cleanup() {",
        "  status=$?",
        "  if [ \"${FE_AUTO_TERMINATE}\" = \"1\" ]; then",
        "    echo \"[lambda-cleanup] terminating ${FE_LAMBDA_INSTANCE_ID} after exit status ${status}\"",
        "    python3 - <<'PY' || true",
        "import base64, json, os, urllib.request",
        "api_base = os.environ['FE_LAMBDA_API_BASE'].rstrip('/')",
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
        "    },",
        ")",
        "with urllib.request.urlopen(req, timeout=30) as resp:",
        "    print(resp.read().decode('utf-8', errors='replace'))",
        "PY",
        "  fi",
        "  exit \"$status\"",
        "}",
        "trap cleanup EXIT",
    ]
    return lines


def _bootstrap_worker(ssh_key_path: Path, host: str) -> None:
    cmd = (
        "set -euo pipefail; "
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


def _write_and_start_cell_runner(
    ssh_key_path: Path,
    host: str,
    instance_id: str,
    cell: MatrixCell,
    *,
    api_base: str,
    api_key: str,
    auto_terminate: bool,
    max_runtime_hours: float,
    source_repo: str,
    worktree_root_base: str,
    variant: str,
    style_corpus: str,
) -> None:
    row_path = f"benchmarks/results/cloud_ablation_rows_{cell.name}.jsonl"
    summary_json = f"benchmarks/results/cloud_ablation_summary_{cell.name}.json"
    summary_md = f"benchmarks/results/cloud_ablation_summary_{cell.name}.md"
    tasks_json = f"benchmarks/results/cloud_ablation_runtime_tasks_{cell.name}.json"
    worktree_root = f"{worktree_root_base}-{cell.name}"

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
    extra = []
    i = 0
    while i < len(cell.extra_args):
        token = cell.extra_args[i]
        if token == DEFAULT_STYLE_CORPUS:
            extra.append(style_corpus)
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
                    max_runtime_hours=max_runtime_hours,
                ),
                "cd ~/fallen-empire-lora",
                "source .venv-linux-port/bin/activate",
                "export LOCAL_BACKEND=transformers",
                f"timeout --foreground \"${{FE_MAX_RUNTIME_SECONDS}}s\" bash -lc {shlex.quote(command)}",
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
            f"tmux new-session -d -s fe-ablation-{cell.name} '{remote_script} > ~/cloud-eval-logs/fe-ablation-{cell.name}.log 2>&1'"
        ),
    )


def _run_workers_parallel(label: str, items: List[tuple[str, Any]], fn: Any) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(items))) as executor:
        future_to_item = {executor.submit(fn, item): item for item in items}
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
    parser.add_argument("--api-base", default=os.environ.get("LAMBDA_API_BASE", DEFAULT_LAMBDA_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("LAMBDA_API_KEY", ""))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="gpu_1x_a10")
    parser.add_argument("--name-prefix", default="fe-ablation")
    parser.add_argument("--launch-instances", action="store_true", help="Launch new Lambda instances for each cell.")
    parser.add_argument("--instance-ids", default="", help="Comma-separated existing instance ids to use.")
    parser.add_argument("--auto-terminate", dest="auto_terminate", action="store_true", default=None, help="Terminate workers when their remote eval script exits.")
    parser.add_argument("--no-auto-terminate", dest="auto_terminate", action="store_false", help="Leave workers running after remote eval script exit.")
    parser.add_argument("--max-runtime-hours", type=float, default=12.0, help="Remote timeout before cleanup/termination runs.")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--include-baseline", action="store_true", help="Also run the baseline/off prompt cell.")
    parser.add_argument("--variant", default="qwen_7_5b_only")
    parser.add_argument("--source-repo-remote", default="~/fallen-empire")
    parser.add_argument("--worktree-root-base", default="~/fallen-empire-arena-ablation")
    parser.add_argument("--style-corpus-path", default=DEFAULT_STYLE_CORPUS)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("LAMBDA_API_KEY or --api-key is required.")

    cells = ([BASELINE_CELL] if args.include_baseline else []) + list(DEFAULT_CELLS)
    if args.instance_ids.strip():
        instance_ids = [x.strip() for x in args.instance_ids.split(",") if x.strip()]
        if len(instance_ids) != len(cells):
            raise SystemExit(f"--instance-ids must match cell count ({len(cells)}).")
        launched_instances = False
    elif args.launch_instances:
        instance_ids = _launch_instances(
            args.api_base,
            args.api_key,
            region=args.region,
            instance_type=args.instance_type,
            ssh_key_name=args.ssh_key_name,
            quantity=len(cells),
            name_prefix=args.name_prefix,
        )
        launched_instances = True
    else:
        raise SystemExit("Provide --instance-ids or --launch-instances.")
    auto_terminate = args.auto_terminate
    if auto_terminate is None:
        auto_terminate = launched_instances

    id_to_ip = _wait_for_instance_ips(args.api_base, args.api_key, instance_ids)
    ips = [id_to_ip[iid] for iid in instance_ids]

    _run_workers_parallel(
        "bootstrap",
        [(host, None) for host in ips],
        lambda item: _bootstrap_worker(args.ssh_key_path, item[0]),
    )
    if not args.skip_sync:
        _run_workers_parallel(
            "sync",
            [(host, None) for host in ips],
            lambda item: _sync_repos(args.ssh_key_path, item[0], args.ml_repo, args.game_repo),
        )

    _run_workers_parallel(
        "start",
        [(host, iid, cell) for cell, iid, host in zip(cells, instance_ids, ips)],
        lambda item: _write_and_start_cell_runner(
            args.ssh_key_path,
            item[0],
            item[1],
            item[2],
            api_base=args.api_base,
            api_key=args.api_key,
            auto_terminate=auto_terminate,
            max_runtime_hours=args.max_runtime_hours,
            source_repo=args.source_repo_remote,
            worktree_root_base=args.worktree_root_base,
            variant=args.variant,
            style_corpus=args.style_corpus_path,
        ),
    )

    print("parallel_ablation_started")
    print(f"auto_terminate={int(bool(auto_terminate))}")
    print(f"max_runtime_hours={args.max_runtime_hours}")
    for cell, iid, host in zip(cells, instance_ids, ips):
        print(f"{cell.name}\tinstance_id={iid}\thost={host}\tsession=fe-ablation-{cell.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

