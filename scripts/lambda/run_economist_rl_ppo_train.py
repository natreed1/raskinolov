#!/usr/bin/env python3
"""Isolated PPO training entry point for economistRL Lambda cycles.

Runs in a fresh process so rollout inference can release GPU memory before PPO
loads the model for gradient steps. Reads ``scored_batch_*.jsonl`` from disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from economist_rl_ppo_trainer import PPOConfig, train_ppo_batch, write_ppo_manifest  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ppo_config_from_dict(data: dict[str, Any] | None) -> PPOConfig:
    if not data:
        return PPOConfig()
    allowed = set(PPOConfig.__dataclass_fields__)
    return PPOConfig(**{k: v for k, v in data.items() if k in allowed})


def run_ppo_train_from_request(request: dict[str, Any]) -> dict[str, Any]:
    scored_file = Path(str(request["scored_file"])).expanduser().resolve()
    ppo_manifest_file = Path(str(request["ppo_manifest_file"])).expanduser().resolve()
    source_adapter = Path(str(request["source_adapter"])).expanduser().resolve()
    candidate_adapter = Path(str(request["candidate_adapter"])).expanduser().resolve()
    if not scored_file.is_file():
        raise SystemExit(f"scored_file not found: {scored_file}")

    scored_rows = _load_jsonl(scored_file)
    cfg = _ppo_config_from_dict(request.get("ppo_config"))
    train_backend = request.get("train_backend")
    dry_run = bool(request.get("dry_run", False))

    manifest = train_ppo_batch(
        scored_rows=scored_rows,
        system_prompt=str(request.get("system_prompt") or ""),
        base_model=str(request["base_model"]),
        source_adapter=source_adapter,
        candidate_adapter=candidate_adapter,
        cfg=cfg,
        dry_run=dry_run,
        train_backend=str(train_backend) if train_backend else None,
    )
    manifest["scored_file"] = str(scored_file)
    manifest["subprocess_isolated"] = True
    write_ppo_manifest(ppo_manifest_file, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run economistRL PPO in an isolated subprocess.")
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON request file written by run_economist_rl_lambda_cycle.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = args.request.expanduser().resolve()
    if not request_path.is_file():
        raise SystemExit(f"request file not found: {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    manifest = run_ppo_train_from_request(request)
    print(json.dumps({"status": manifest.get("status"), "ppo_manifest": manifest}, indent=2))
    status = str(manifest.get("status") or "")
    if status in {"trained", "dry_run_no_weight_update"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
