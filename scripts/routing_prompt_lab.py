#!/usr/bin/env python3
"""Capture live prompts with adapter-first routing predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO / "benchmarks" / "results" / "routing_prompt_lab"


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = [str(p).strip() for p in args.prompt if str(p).strip()]
    if args.prompts_file:
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    if args.interactive:
        print("Enter prompts (blank line to finish):")
        while True:
            text = input("> ").strip()
            if not text:
                break
            prompts.append(text)
    return prompts


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUT_DIR / f"routing_prompt_lab_{ts}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture routing prompt labels from live prompts.")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--source", default="live")
    parser.add_argument("--expected-adapter-id", default=None)
    parser.add_argument("--expected-legacy-route", default=None, choices=["local", "hybrid", "frontier"])
    parser.add_argument("--accepted-for-training", action="store_true")
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    prompts = _load_prompts(args)
    if not prompts:
        raise SystemExit("No prompts provided. Use --prompt, --prompts-file, or --interactive.")

    out = args.output_jsonl.expanduser().resolve() if args.output_jsonl else _default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    policy = RoutingPolicy()

    with out.open("a", encoding="utf-8") as fh:
        for idx, prompt in enumerate(prompts):
            req = GenerationRequest(messages=messages_from_prompt(prompt))
            decision = policy.decide(req)
            row: dict[str, Any] = {
                "schema_version": "routing_prompt_lab_v1",
                "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": args.source,
                "prompt": prompt,
                "predicted_adapter_id": decision.adapter_id,
                "predicted_legacy_route": decision.route,
                "confidence": round(float(decision.confidence), 4),
                "ambiguity": round(float(decision.ambiguity), 4),
                "risk_class": decision.risk_class,
                "complexity": decision.complexity,
                "policy_version": decision.policy_version,
                "reason": decision.reason,
                "accepted_for_training": bool(args.accepted_for_training),
                "council_enabled": bool(getattr(decision, "council_enabled", False)),
                "council_plan": dict(getattr(decision, "council_plan", {}) or {}),
                "council_disagreement": round(float(getattr(decision, "council_disagreement", 0.0) or 0.0), 4),
                "council_escalation_candidate": bool(
                    getattr(decision, "council_escalation_candidate", False)
                ),
            }
            if args.expected_adapter_id:
                row["expected_adapter_id"] = args.expected_adapter_id
            if args.expected_legacy_route:
                row["expected_legacy_route"] = args.expected_legacy_route
            if args.reviewer:
                row["reviewer"] = args.reviewer
            if args.notes:
                row["notes"] = args.notes
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{idx + 1}/{len(prompts)}] adapter={decision.adapter_id} "
                f"route={decision.route} conf={decision.confidence:.2f}"
            )
    print(f"Wrote {len(prompts)} rows -> {out}")


if __name__ == "__main__":
    main()

