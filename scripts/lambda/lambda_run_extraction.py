#!/usr/bin/env python3
"""Reusable local extraction helpers for Lambda run artifacts.

This module deliberately does not launch, terminate, or query Lambda instances.
Callers provide the host, metadata, and artifact locations they already know;
the extractor only copies/unpacks files and writes local manifests.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RemoteRsyncSpec:
    """One remote path to copy into an extraction directory."""

    remote_path: str
    local_subdir: str


class LambdaRunExtractor:
    """Write local extraction artifacts for Lambda-backed runs."""

    def __init__(
        self,
        *,
        repo_root: Path,
        extracts_root: Path,
        ssh_key_path: Path | None = None,
        index_path: Path | None = None,
        command_runner: CommandRunner | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.extracts_root = extracts_root
        self.index_path = index_path or extracts_root / "index.jsonl"
        self.ssh_key_path = ssh_key_path or Path.home() / ".ssh" / "lambda_cloud_cursor"
        self.command_runner = command_runner or subprocess.run
        self.log = log or (lambda _msg: None)

    def append_index(self, record: dict[str, Any]) -> None:
        self.extracts_root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def extract_dir(self, *, attempt: int, instance_id: str, label: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = instance_id[:8] if instance_id else "unknown"
        safe_label = _safe_label(label or "unknown")
        dest = self.extracts_root / f"attempt_{attempt:03d}_{short}_{safe_label}_{stamp}"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def write_extract_manifest(
        self,
        dest: Path,
        *,
        run_id: str = "",
        run_number: int | None = None,
        run_date_utc: str = "",
        attempt: int,
        instance_id: str,
        host: str,
        failure_class: str,
        status: str,
        progress: dict[str, Any],
        remote_tail: str,
        launch_log: Path | None,
        rsync_paths: list[str],
        docs_path: str = "benchmarks/results/economistRL/extracts/README.md",
    ) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        extracted_at = utc_now()
        manifest = {
            "schema_version": "economist_rl_extract_manifest_v2",
            "run_id": run_id,
            "run_number": run_number,
            "run_date_utc": run_date_utc,
            "extracted_at_utc": extracted_at,
            "attempt": attempt,
            "instance_id": instance_id,
            "host": host,
            "status": status,
            "failure_class": failure_class,
            "progress": progress,
            "dest": _display_path(dest, self.repo_root),
            "rsync_paths": rsync_paths,
            "launch_log": _display_path(launch_log, self.repo_root) if launch_log and launch_log.is_file() else "",
            "docs": docs_path,
        }
        (dest / "EXTRACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (dest / "README.md").write_text(
            self._render_readme(
                attempt=attempt,
                instance_id=instance_id,
                host=host,
                status=status,
                failure_class=failure_class,
                progress=progress,
            ),
            encoding="utf-8",
        )
        if remote_tail.strip():
            (dest / "cycle_log_tail.txt").write_text(remote_tail, encoding="utf-8")
        if launch_log and launch_log.is_file():
            (dest / "launch_log.txt").write_text(
                launch_log.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
        self.append_index(
            {
                "schema_version": "economist_rl_extract_index_row_v2",
                "run_id": run_id,
                "run_number": run_number,
                "run_date_utc": run_date_utc,
                "extracted_at_utc": extracted_at,
                "attempt": attempt,
                "instance_id": instance_id,
                "failure_class": failure_class,
                "status": status,
                "dest": manifest["dest"],
                "last_rollout": progress.get("last_rollout", ""),
            }
        )

    def rsync_artifacts(
        self,
        host: str,
        *,
        run_id: str = "",
        run_number: int | None = None,
        run_date_utc: str = "",
        attempt: int,
        instance_id: str,
        failure_class: str,
        status: str,
        progress: dict[str, Any],
        remote_tail: str,
        launch_log: Path | None,
        specs: Sequence[RemoteRsyncSpec],
    ) -> Path:
        dest = self.extract_dir(attempt=attempt, instance_id=instance_id, label=failure_class)
        ssh_cmd = (
            f"ssh -i {shlex.quote(str(self.ssh_key_path))} "
            "-o StrictHostKeyChecking=no -o ConnectTimeout=20"
        )
        rsync_ok: list[str] = []
        for spec in specs:
            target = dest / spec.local_subdir
            target.mkdir(parents=True, exist_ok=True)
            src = f"ubuntu@{host}:{spec.remote_path}"
            proc = self.command_runner(
                ["rsync", "-az", "--timeout=120", "-e", ssh_cmd, src, str(target) + "/"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                rsync_ok.append(spec.local_subdir)
            else:
                self.log(
                    f"rsync_partial attempt={attempt} dest={dest.name} src={src} "
                    f"rc={proc.returncode} err={(proc.stderr or '').strip()[:160]}"
                )
        self.write_extract_manifest(
            dest,
            run_id=run_id,
            run_number=run_number,
            run_date_utc=run_date_utc,
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
        self.log(f"extract_written dest={_display_path(dest, self.repo_root)} failure_class={failure_class}")
        return dest

    def unpack_local_tarball(
        self,
        tarball: Path,
        *,
        run_id: str = "",
        run_number: int | None = None,
        run_date_utc: str = "",
        attempt: int,
        instance_id: str,
        label: str,
        status: str,
        failure_class: str,
        progress: dict[str, Any] | None = None,
        launch_log: Path | None = None,
    ) -> Path:
        """Unpack a local Lambda artifact tarball into an indexed extract dir."""
        if not tarball.is_file():
            raise FileNotFoundError(tarball)
        dest = self.extract_dir(attempt=attempt, instance_id=instance_id, label=label)
        with tarfile.open(tarball, "r:gz") as archive:
            archive.extractall(dest)
        self.write_extract_manifest(
            dest,
            run_id=run_id,
            run_number=run_number,
            run_date_utc=run_date_utc,
            attempt=attempt,
            instance_id=instance_id,
            host="",
            failure_class=failure_class,
            status=status,
            progress=progress or {},
            remote_tail=_read_first_existing(
                [
                    dest / "cloud-eval-logs" / "fe-economist-rl-split-workers.log",
                    dest / "cloud-eval-logs" / "fe-economist-rl-cycle.log",
                ],
                tail_lines=500,
            ),
            launch_log=launch_log,
            rsync_paths=[tarball.name],
        )
        self.log(f"extract_unpacked dest={_display_path(dest, self.repo_root)} tarball={tarball}")
        return dest

    def write_live_snapshot(
        self,
        *,
        instance_id: str,
        progress: dict[str, Any],
        remote_log: str,
    ) -> Path:
        live = self.extracts_root / f"live_{instance_id[:12]}"
        live.mkdir(parents=True, exist_ok=True)
        (live / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        (live / "cycle_log_tail.txt").write_text(remote_log, encoding="utf-8")
        (live / "LAST_UPDATED_UTC.txt").write_text(utc_now() + "\n", encoding="utf-8")
        return live

    def _render_readme(
        self,
        *,
        attempt: int,
        instance_id: str,
        host: str,
        status: str,
        failure_class: str,
        progress: dict[str, Any],
    ) -> str:
        readme = [
            f"# economistRL extract - attempt {attempt:03d}",
            "",
            f"- **Instance:** `{instance_id}` @ `{host}`",
            f"- **Status:** `{status}`",
            f"- **Failure class:** `{failure_class}`",
            f"- **Last rollout:** `{progress.get('last_rollout', '')}`",
            f"- **Last cycle:** `{progress.get('last_cycle', '')}`",
            "",
            "## What to read",
            "",
            "1. `cycle_log_tail.txt` - remote runner log tail",
            "2. `cloud-eval-logs/fe-economist-rl-split-workers.log` - full remote log if rsync succeeded",
            "3. `economistRL/ppo/` - PPO stderr/stdout/manifest if present",
            "4. `launch_log.txt` - local Lambda bootstrap log",
            "",
            "## Likely cause",
            "",
        ]
        if failure_class == "instance_gone":
            readme.append(
                "Worker disappeared from Lambda API mid-run. Check Lambda console or billing; "
                "compare GPU telemetry CSV if extracted."
            )
        elif failure_class == "training_failed_oom":
            readme.append("PPO subprocess OOM, often exit -9. Lower PPO max samples and window tokens.")
        elif failure_class == "launch_failed":
            readme.append("Local launcher failed before remote cycle started, usually during bootstrap, rsync, or SSH.")
        else:
            readme.append(f"See failure_class `{failure_class}` and cycle log tail.")
        return "\n".join(readme) + "\n"


def _safe_label(label: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
    return out[:80] or "unknown"


def _display_path(path: Path | None, repo_root: Path) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _read_first_existing(paths: Sequence[Path], *, tail_lines: int) -> str:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-tail_lines:]) + ("\n" if lines else "")
    return ""


def copy_local_artifact_tree(source: Path, dest: Path) -> None:
    """Copy a local artifact directory into an extraction directory."""
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
