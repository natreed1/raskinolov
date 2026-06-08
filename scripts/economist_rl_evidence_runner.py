#!/usr/bin/env python3
"""Attach executable rollout evidence before economistRL reward scoring.

The evidence runner fills targeted_tests (text heuristics until execution Vitest),
compile status, and changed-file hints on rollout rows so score_output can produce
real RL rewards.

Compile evidence semantics:
- ``compiled=True/False`` only when a real compile/test command result is present
  (``compile_evidence``, ``compile_passed``, ``verify_status``, etc.).
- ``has_code_fence`` / ``has_export_function`` / ``looks_code_like`` are formatting
  hints only; they do not set ``compiled`` or trigger compile bonus.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class EvidenceRunnerConfig:
    runner_id: str = "economist_rl_evidence_runner_v1"
    require_simulation: bool = False
    static_test_score: float = 0.75


def _extract_code_blocks(text: str) -> list[str]:
    return [block.strip() for block in CODE_FENCE_RE.findall(text) if block.strip()]


def _infer_changed_files(task: dict[str, Any], text: str) -> list[str]:
    files: list[str] = []
    req = task.get("codebase_requirements") if isinstance(task.get("codebase_requirements"), dict) else {}
    for path in req.get("relevant_files") or []:
        token = str(path).strip()
        if token and token in text:
            files.append(token)
    if not files:
        for match in re.findall(r"(src/[\w./-]+\.(?:ts|tsx)|tests/[\w./-]+\.(?:ts|tsx))", text):
            if match not in files:
                files.append(match)
    return files[:8]


def _static_targeted_test_score(task: dict[str, Any], text: str, code_blocks: list[str]) -> dict[str, Any]:
    haystack = "\n".join([text, *code_blocks]).lower()
    spec = task.get("targeted_tests") if isinstance(task.get("targeted_tests"), dict) else {}
    checks = [str(item) for item in spec.get("outcome_checks") or [] if str(item).strip()]
    if not checks:
        checks = ["behavior changed", "bounded", "deterministic"]
    scored_checks = []
    passed = 0
    for check in checks:
        tokens = [tok for tok in re.findall(r"[a-zA-Z_]{4,}", check.lower()) if tok not in {"with", "from", "that", "this", "without"}]
        hit = bool(tokens) and sum(1 for tok in tokens if tok in haystack) >= max(1, len(tokens) // 2)
        scored_checks.append({"name": check, "score": 1.0 if hit else 0.0})
        passed += int(hit)
    score = passed / max(1, len(scored_checks))
    return {"score": score, "checks": scored_checks}


def _code_format_signals(text: str, code_blocks: list[str]) -> dict[str, bool]:
    joined = "\n".join(code_blocks) if code_blocks else text
    has_code_fence = bool(code_blocks) or "```" in text
    has_export_function = "export function" in joined or "export const" in joined
    looks_code_like = has_code_fence or has_export_function or "export class" in joined
    return {
        "has_code_fence": has_code_fence,
        "has_export_function": has_export_function,
        "looks_code_like": looks_code_like,
    }


def _compile_evidence(row: dict[str, Any]) -> bool | None:
    """Return compile pass/fail only when an actual compile/test command was executed."""
    for key in ("compiled", "compile_passed", "typecheck_passed", "tsc_passed"):
        if key in row and row.get(key) is not None:
            parsed = _boolish_compile(row.get(key))
            if parsed is not None:
                return parsed
    for key in ("compile_status", "typecheck_status", "tsc_status"):
        if key in row and row.get(key) is not None:
            parsed = _boolish_compile(row.get(key))
            if parsed is not None:
                return parsed
    verify_status = str(row.get("verify_status") or "").strip().lower()
    if verify_status == "passed":
        return True
    if verify_status in {"failed", "infra_precheck_failed"}:
        return False
    evidence = row.get("compile_evidence")
    if isinstance(evidence, dict):
        parsed = _boolish_compile(evidence.get("passed"))
        if parsed is not None:
            return parsed
    return None


def _boolish_compile(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "passed", "pass", "success", "ok"}:
        return True
    if token in {"0", "false", "no", "n", "failed", "fail", "error", "errored", "timeout"}:
        return False
    return None


def attach_rollout_evidence(
    *,
    task: dict[str, Any],
    rollout_row: dict[str, Any],
    config: EvidenceRunnerConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Return a copy of rollout_row with evidence fields attached."""
    cfg = config or EvidenceRunnerConfig()
    row = dict(rollout_row)
    output = str(row.get("output") or "")
    code_blocks = _extract_code_blocks(output)
    diff = "\n\n".join(code_blocks) if code_blocks else output

    row["simulation_source"] = "vitest_goals"
    row["targeted_tests"] = _static_targeted_test_score(task, output, code_blocks)
    row["diff"] = diff
    if code_blocks:
        row["code"] = code_blocks[0]
    row["changed_files"] = _infer_changed_files(task, output)
    row.update(_code_format_signals(output, code_blocks))
    compiled = _compile_evidence(row)
    if compiled is not None:
        row["compiled"] = compiled
    row["compile_checked"] = compiled is not None
    row["evidence_runner"] = cfg.runner_id
    row["evidence_attached"] = True
    return row


def attach_batch_evidence(
    *,
    tasks_by_id: dict[str, dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    config: EvidenceRunnerConfig | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rollout_rows:
        task_id = str(row.get("task_id") or "")
        task = tasks_by_id.get(task_id)
        if not task:
            out.append(dict(row))
            continue
        out.append(attach_rollout_evidence(task=task, rollout_row=row, config=config, dry_run=dry_run))
    return out


def write_evidence_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
