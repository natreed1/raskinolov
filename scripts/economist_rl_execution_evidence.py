#!/usr/bin/env python3
"""Apply rollout outputs to a disposable repo and attach real compile/test evidence.

Design A: toy/rubric evidence stays in ``economist_rl_evidence_runner``; this module
sets ``compiled`` / ``compile_evidence`` (and optional ``targeted_tests`` from command
exit codes) so ``compile_gate_reward`` in the reward engine can apply.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

DEFAULT_ENV_SOURCE_REPO = "ECONOMIST_RL_SOURCE_REPO"

VITEST_CONFIG_BODY = """\
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      '@': path.join(root, 'src'),
    },
  },
  test: {
    include: ['tests/economistRl/**/*.test.ts'],
  },
});
"""


def ensure_worktree_vitest_config(worktree: Path) -> Path:
    """Vitest in disposable worktrees needs ``@/*`` alias (tsconfig paths are not auto-loaded)."""
    dest = worktree / "vitest.config.mts"
    dest.write_text(VITEST_CONFIG_BODY, encoding="utf-8")
    return dest


@dataclass(frozen=True)
class ExecutionEvidenceConfig:
    enabled: bool = True
    source_repo: Path | None = None
    worktree_root: Path | None = None
    compile_commands: tuple[str, ...] = ()
    default_compile_commands: tuple[str, ...] = ("npm run test:ml-cohort",)
    timeout_s: int = 600
    require_applyable_for_economy: bool = True


@dataclass
class ApplyResult:
    ok: bool
    status: str
    written_files: list[str]
    log_path: str


@dataclass
class CompileResult:
    ok: bool
    commands: list[dict[str, Any]]


def resolve_source_repo(
    *,
    cli_path: Path | None = None,
    env_var: str = DEFAULT_ENV_SOURCE_REPO,
) -> Path | None:
    if cli_path is not None:
        path = cli_path.expanduser().resolve()
        if path.is_dir():
            return path
        raise SystemExit(f"--execution-source-repo is not a directory: {path}")
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_var} is not a directory: {path}")
    return path


def resolve_execution_config(
    *,
    enabled: bool,
    source_repo: Path | None,
    worktree_root: Path,
    compile_commands: list[str] | None,
    timeout_s: int,
) -> ExecutionEvidenceConfig:
    commands = tuple(cmd.strip() for cmd in (compile_commands or []) if str(cmd).strip())
    return ExecutionEvidenceConfig(
        enabled=bool(enabled),
        source_repo=source_repo,
        worktree_root=worktree_root.expanduser().resolve(),
        compile_commands=commands,
        timeout_s=int(timeout_s),
    )


def allowed_paths_for_task(task: dict[str, Any]) -> list[str]:
    from economist_rl_task_execution import allowed_paths_for_execution

    paths = allowed_paths_for_execution(task)
    if paths:
        return paths
    return ["src/**/*.ts", "src/**/*.tsx", "tests/**/*.ts", "tests/**/*.tsx"]


def _compile_commands_for_task(
    task: dict[str, Any],
    config: ExecutionEvidenceConfig,
    *,
    log_dir: Path | None = None,
) -> list[str]:
    if config.compile_commands:
        return list(config.compile_commands)
    from economist_rl_task_execution import (
        EXECUTION_MODE_INTEGRATION,
        INTEGRATION_KIND_SANDBOX,
        classify_execution_mode,
        classify_integration_kind,
        sandbox_paths,
        verify_commands_for_task,
    )

    per_task = verify_commands_for_task(task)
    if per_task:
        if (
            classify_execution_mode(task) == EXECUTION_MODE_INTEGRATION
            and classify_integration_kind(task, source_repo=config.source_repo) == INTEGRATION_KIND_SANDBOX
            and log_dir is not None
        ):
            _lib, test_path = sandbox_paths(task)
            from economist_rl_vitest_scoring import vitest_json_report_command

            report = log_dir / "vitest_report.json"
            return [vitest_json_report_command(test_path, str(report))]
        return per_task
    return list(config.default_compile_commands)


def _run_shell(cmd: str, *, cwd: Path, log_path: Path, timeout_s: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("CI", "true")
    env.setdefault("NPM_CONFIG_YES", "true")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {cmd}\n\n")
        log.flush()
        proc = subprocess.Popen(
            ["sh", "-lc", cmd],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
            code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"\n[timeout after {timeout_s}s]\n")
            log.flush()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            proc.wait(timeout=5)
            code = 124
    return {
        "command": cmd,
        "exit_code": code,
        "ok": code == 0,
        "seconds": round(time.perf_counter() - started, 3),
        "log_path": str(log_path),
    }


def _apply_output(worktree: Path, output: str, allowed: list[str], log_path: Path) -> ApplyResult:
    from game_task_arena import apply_fenced_files, extract_diff, run_cmd

    log_path.parent.mkdir(parents=True, exist_ok=True)
    diff = extract_diff(output)
    if diff:
        patch_path = log_path.parent / "model.patch"
        patch_path.write_text(diff, encoding="utf-8")
        check_code, _ = run_cmd(["git", "apply", "--check", str(patch_path)], cwd=worktree, log_path=log_path, timeout_s=120)
        if check_code == 0:
            code, _ = run_cmd(["git", "apply", str(patch_path)], cwd=worktree, log_path=log_path, timeout_s=120)
            if code == 0:
                return ApplyResult(True, "applied_diff", [], str(log_path))
        ok, written = apply_fenced_files(worktree, output, allowed, log_path.parent / "apply_fenced_fallback.log")
        if ok:
            return ApplyResult(True, f"wrote_files:{','.join(written)}", written, str(log_path))
        return ApplyResult(False, "apply_check_failed", written, str(log_path))

    ok, written = apply_fenced_files(worktree, output, allowed, log_path)
    if ok:
        return ApplyResult(True, f"wrote_files:{','.join(written)}", written, str(log_path))
    return ApplyResult(False, "no_applyable_changes", [], str(log_path))


def _run_compile_checks(
    *,
    worktree: Path,
    task: dict[str, Any],
    config: ExecutionEvidenceConfig,
    log_dir: Path,
) -> CompileResult:
    commands = _compile_commands_for_task(task, config, log_dir=log_dir)
    rows: list[dict[str, Any]] = []
    for idx, cmd in enumerate(commands):
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", cmd)[:48].strip("_") or f"cmd_{idx}"
        rows.append(_run_shell(cmd, cwd=worktree, log_path=log_dir / f"compile_{slug}.log", timeout_s=config.timeout_s))
    ok = bool(rows) and all(row.get("ok") for row in rows)
    return CompileResult(ok=ok, commands=rows)


def _economy_requires_apply(task: dict[str, Any], config: ExecutionEvidenceConfig) -> bool:
    if not config.require_applyable_for_economy:
        return False
    track = str(task.get("curriculum_track") or "economy").strip().lower()
    return track == "economy"


def attach_execution_evidence(
    *,
    task: dict[str, Any],
    rollout_row: dict[str, Any],
    config: ExecutionEvidenceConfig,
    worktree: Path,
    log_root: Path,
) -> dict[str, Any]:
    """Return rollout row copy with compile fields set from real apply + commands."""
    ensure_worktree_vitest_config(worktree)
    row = dict(rollout_row)
    task_id = str(row.get("task_id") or task.get("id") or "task")
    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", task_id)[:80]
    task_log = log_root / safe_id
    output = str(row.get("output") or "")
    from economist_rl_task_execution import apply_allowed_paths_for_task

    allowed = apply_allowed_paths_for_task(task)

    from economist_rl_task_execution import materialize_starter_files

    seeded = materialize_starter_files(worktree, task)
    if seeded:
        row["starter_files_seeded"] = seeded

    apply_result = _apply_output(worktree, output, allowed, task_log / "apply.log")
    row["apply_status"] = apply_result.status
    row["apply_log"] = apply_result.log_path
    if apply_result.written_files:
        row["changed_files"] = apply_result.written_files

    requires_apply = _economy_requires_apply(task, config)
    if not apply_result.ok:
        row["compile_evidence"] = {
            "passed": False,
            "reason": "no_applyable_changes" if requires_apply else "apply_skipped_or_failed",
            "apply_status": apply_result.status,
        }
        row["compiled"] = False
        row["compile_checked"] = True
        row["execution_evidence"] = "apply_failed"
        row["targeted_tests"] = {
            "score": 0.0,
            "checks": [{"name": "apply_failed", "score": 0.0}],
            "source": "execution_apply_failed",
        }
        return row

    compile_result = _run_compile_checks(worktree=worktree, task=task, config=config, log_dir=task_log)
    row["compile_evidence"] = {
        "passed": compile_result.ok,
        "commands": compile_result.commands,
        "apply_status": apply_result.status,
    }
    vitest_ran = any("vitest" in str(cmd.get("command") or "") for cmd in compile_result.commands)
    row["compile_checked"] = True
    row["execution_evidence"] = "ran_compile"

    if compile_result.commands:
        from economist_rl_vitest_scoring import merge_outcome_checks_with_vitest, parse_vitest_json_report

        report_file = task_log / "vitest_report.json"
        if report_file.is_file():
            parsed = parse_vitest_json_report(report_file)
            row["targeted_tests"] = merge_outcome_checks_with_vitest(task, parsed)
            row["vitest_report"] = parsed
            row["compiled"] = bool(parsed.get("vitest_ran"))
            if float(parsed.get("score") or 0.0) >= 0.999:
                row["compile_evidence"]["passed"] = True
        else:
            row["compiled"] = bool(vitest_ran and compile_result.ok)
            spec = task.get("targeted_tests") if isinstance(task.get("targeted_tests"), dict) else {}
            checks = [str(item) for item in spec.get("outcome_checks") or [] if str(item).strip()]
            cmd_ok = compile_result.ok
            score = 1.0 if cmd_ok else 0.0
            row["targeted_tests"] = {
                "score": score,
                "checks": [{"name": check, "score": score} for check in checks] if checks else [{"name": "compile_commands", "score": score}],
                "source": "execution_compile_commands",
            }
    else:
        row["compiled"] = bool(vitest_ran and compile_result.ok)

    row["simulation_source"] = "vitest_goals"
    return row


class ExecutionWorktreePool:
    """One disposable worktree per cycle; reset between rollouts."""

    def __init__(self, config: ExecutionEvidenceConfig, *, cycle_id: int) -> None:
        if not config.source_repo:
            raise ValueError("ExecutionWorktreePool requires source_repo")
        self.config = config
        self.cycle_id = cycle_id
        self.source_repo = config.source_repo
        root = (config.worktree_root or (REPO / "benchmarks/results/economistRL/worktrees")).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.worktree_path = root / f"cycle_{cycle_id:03d}_{uuid.uuid4().hex[:8]}"
        self._uses_git = (self.source_repo / ".git").exists()
        self._baseline_snapshot: Path | None = None
        self._create_worktree()

    def _create_worktree(self) -> None:
        if self.worktree_path.exists():
            shutil.rmtree(self.worktree_path)
        if self._uses_git:
            branch = f"economist-rl/exec-{self.cycle_id:03d}-{uuid.uuid4().hex[:6]}"
            code = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.source_repo),
                    "worktree",
                    "add",
                    "-b",
                    branch,
                    str(self.worktree_path),
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            if code != 0:
                self._uses_git = False
        if not self._uses_git:
            ignore = shutil.ignore_patterns(".git", "node_modules", ".next", "dist", "build")
            shutil.copytree(self.source_repo, self.worktree_path, ignore=ignore)
            self._baseline_snapshot = self.worktree_path.parent / f".baseline_{self.worktree_path.name}"
            if self._baseline_snapshot.exists():
                shutil.rmtree(self._baseline_snapshot)
            shutil.copytree(self.worktree_path, self._baseline_snapshot, ignore=ignore)
        self._ensure_worktree_node_modules()

    def _ensure_worktree_node_modules(self) -> None:
        src = self.source_repo / "node_modules"
        dst = self.worktree_path / "node_modules"
        if not src.is_dir():
            return
        if dst.exists() or dst.is_symlink():
            return
        dst.symlink_to(src, target_is_directory=True)

    def reset(self) -> None:
        if self._uses_git:
            subprocess.run(
                ["git", "-C", str(self.worktree_path), "reset", "--hard"],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(self.worktree_path), "clean", "-fd"],
                capture_output=True,
                check=False,
            )
            return
        if self._baseline_snapshot and self._baseline_snapshot.is_dir():
            shutil.rmtree(self.worktree_path)
            shutil.copytree(self._baseline_snapshot, self.worktree_path)

    def cleanup(self) -> None:
        if self._uses_git:
            subprocess.run(
                ["git", "-C", str(self.source_repo), "worktree", "remove", "--force", str(self.worktree_path)],
                capture_output=True,
                check=False,
            )
        elif self.worktree_path.exists():
            shutil.rmtree(self.worktree_path)
        if self._baseline_snapshot and self._baseline_snapshot.exists():
            shutil.rmtree(self._baseline_snapshot)


def attach_batch_execution_evidence(
    *,
    tasks_by_id: dict[str, dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    config: ExecutionEvidenceConfig,
    pool: ExecutionWorktreePool,
    log_root: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    log_root.mkdir(parents=True, exist_ok=True)
    for row in rollout_rows:
        task_id = str(row.get("task_id") or "")
        task = tasks_by_id.get(task_id)
        if not task:
            out.append(dict(row))
            continue
        pool.reset()
        out.append(
            attach_execution_evidence(
                task=task,
                rollout_row=row,
                config=config,
                worktree=pool.worktree_path,
                log_root=log_root,
            )
        )
    return out


def execution_config_summary(config: ExecutionEvidenceConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "source_repo": str(config.source_repo) if config.source_repo else None,
        "worktree_root": str(config.worktree_root) if config.worktree_root else None,
        "compile_commands": list(config.compile_commands) or list(config.default_compile_commands),
        "timeout_s": config.timeout_s,
        "require_applyable_for_economy": config.require_applyable_for_economy,
    }
