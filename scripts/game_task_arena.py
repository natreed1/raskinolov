#!/usr/bin/env python3
"""
Disposable Fallen Empire game-task arena.

This is an evaluation harness, not a merge bot. It creates isolated game repo
worktrees/copies for local/frontier attempts, applies only allowlisted patches or
file writes, runs fixed verification commands from task specs, records previews
and human grades, then optionally removes disposable worktrees while preserving
all result artifacts.

Context for `generate` / `packet` is built by `build_context_pack()` (reserved prefix, literal paths
before globs, BM25 reorder among globs, head/tail truncation). Each attempt logs
`attempts/<slug>/logs/context_pack.json` when a log directory is available.

Examples:
  python scripts/game_task_arena.py create --task-id ui-hud-copy
  python scripts/game_task_arena.py packet --trial-id <id> --attempt local
  python scripts/game_task_arena.py apply --trial-id <id> --attempt local --input patch.md
  python scripts/game_task_arena.py verify --trial-id <id> --attempt local
  python scripts/game_task_arena.py preview --trial-id <id> --attempt local --port 5174
  python scripts/game_task_arena.py grade --trial-id <id> --attempt local --correctness 4
  python scripts/game_task_arena.py report
  python scripts/game_task_arena.py cleanup --trial-id <id> --attempt local

Preview starts Next.js directly from disposable worktrees, reusing the source
checkout's node_modules when available, and records readiness only after the
task URL responds. Generation artifacts include elapsed time and token counts.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
DEFAULT_SOURCE_REPO = Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))).expanduser()
DEFAULT_WORKTREE_ROOT = Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena"))).expanduser()
RESULTS_ROOT = REPO / "benchmarks" / "results" / "game_task_trials"
INDEX_PATH = REPO / "benchmarks" / "results" / "game_task_index.jsonl"
REPORTS_DIR = REPO / "benchmarks" / "results" / "game_task_reports"
DEFAULT_TASKS = REPO / "benchmarks" / "game_task_arena_examples.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import ChatMessage, GenerationRequest, LocalMlxBackend, OpenAICompatibleBackend, estimate_tokens


@dataclass
class TaskSpec:
    id: str
    task_type: str
    title: str
    prompt: str
    allowed_paths: List[str]
    context_paths: List[str] = field(default_factory=list)
    verify_commands: List[str] = field(default_factory=list)
    preview_command: str = "npm run dev -- --host 127.0.0.1 --port {port}"
    preview_path: str = ""
    max_tokens: int = 4096
    notes: str = ""
    grading: List[str] = field(default_factory=list)


@dataclass
class AttemptManifest:
    attempt: str
    model_label: str
    backend: str
    branch: str
    worktree_path: str
    created_at: str
    apply_status: str = "not_applied"
    verify_status: str = "not_run"
    preview_url: str = ""
    preview_pid: Optional[int] = None
    preview_status: str = "not_started"
    generation_elapsed_s: Optional[float] = None
    generation_input_tokens: Optional[int] = None
    generation_output_tokens: Optional[int] = None
    generation_total_tokens: Optional[int] = None
    generation_cost_usd: Optional[float] = None
    cleanup_state: str = "active"
    last_error: str = ""


@dataclass
class TrialManifest:
    trial_id: str
    task: TaskSpec
    source_repo: str
    base_ref: str
    base_git_ref: str
    worktree_root: str
    created_at: str
    attempts: Dict[str, AttemptManifest]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return (value or "task")[:max_len]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_preview_path(raw_path: str) -> str:
    path = (raw_path or "").strip()
    if not path:
        return ""
    if "://" in path or path.startswith("//"):
        raise SystemExit(f"preview_path must be a local path, got: {raw_path}")
    return path if path.startswith("/") else f"/{path}"


def normalize_preview_command(command: str) -> str:
    # Older specs used the package script, but disposable worktrees do not have
    # their own node_modules. Run Next directly with PATH pointed at source deps.
    if command.strip() == "npm run dev -- --host 127.0.0.1 --port {port}":
        return "next dev -H 127.0.0.1 -p {port}"
    return command


def preview_env(trial: TrialManifest, port: int) -> Dict[str, str]:
    source_repo = Path(trial.source_repo)
    node_bin = source_repo / "node_modules" / ".bin"
    path_parts = [str(node_bin)] if node_bin.is_dir() else []
    path_parts.append(os.environ.get("PATH", ""))
    env = {**os.environ, "PORT": str(port), "PATH": os.pathsep.join(path_parts), "WATCHPACK_POLLING": "true"}
    source_node_modules = source_repo / "node_modules"
    if source_node_modules.is_dir():
        env["NODE_PATH"] = str(source_node_modules)
    return env


def ensure_preview_node_modules(trial: TrialManifest, attempt: AttemptManifest) -> None:
    source_node_modules = Path(trial.source_repo) / "node_modules"
    worktree_node_modules = Path(attempt.worktree_path) / "node_modules"
    if worktree_node_modules.exists() or not source_node_modules.is_dir():
        return
    try:
        worktree_node_modules.symlink_to(source_node_modules, target_is_directory=True)
    except OSError:
        # PATH/NODE_PATH still give Next a chance to use the source checkout deps.
        pass


def wait_for_preview_url(url: str, proc: subprocess.Popen, timeout_s: float = 20.0) -> Tuple[str, float]:
    started = time.perf_counter()
    last_error = ""
    while time.perf_counter() - started < timeout_s:
        code = proc.poll()
        if code is not None:
            return f"failed_exit_{code}", time.perf_counter() - started
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 400:
                    return "ready", time.perf_counter() - started
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return f"http_{exc.code}", time.perf_counter() - started
            return f"http_{exc.code}", time.perf_counter() - started
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    return f"starting: {last_error}" if last_error else "starting", time.perf_counter() - started


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def choose_preview_port(requested_port: int, host: str = "127.0.0.1", max_tries: int = 50) -> int:
    for port in range(requested_port, requested_port + max_tries):
        if port_is_available(host, port):
            return port
    raise SystemExit(f"No free preview port found from {requested_port} through {requested_port + max_tries - 1}")


def preview_preflight(trial: TrialManifest, attempt: AttemptManifest, port: int) -> Tuple[bool, str]:
    worktree = Path(attempt.worktree_path)
    if not (worktree / "tsconfig.json").is_file():
        return True, "skipped_no_tsconfig"
    ensure_preview_node_modules(trial, attempt)
    code, _ = run_cmd(
        ["npx", "tsc", "--noEmit"],
        cwd=worktree,
        log_path=attempt_dir(trial.trial_id, attempt.attempt) / "logs" / "preview_preflight_tsc.log",
        timeout_s=120,
        env=preview_env(trial, port),
    )
    return code == 0, f"tsc_exit_{code}"


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(
    argv: List[str],
    cwd: Path,
    log_path: Path,
    timeout_s: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    merged_env = {**os.environ, **(env or {})}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(argv)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                env=merged_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
            code = 124
    return code, time.perf_counter() - started


def load_task_specs(path: Path = DEFAULT_TASKS) -> Dict[str, TaskSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", payload)
    return {t["id"]: TaskSpec(**t) for t in tasks}


def git_output(repo: Path, args: List[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def trial_dir(trial_id: str) -> Path:
    return RESULTS_ROOT / trial_id


def attempt_dir(trial_id: str, attempt: str) -> Path:
    return trial_dir(trial_id) / "attempts" / slug(attempt)


def manifest_path(trial_id: str) -> Path:
    return trial_dir(trial_id) / "trial_manifest.json"


def load_trial(trial_id: str) -> TrialManifest:
    path = manifest_path(trial_id)
    if not path.is_file():
        raise SystemExit(f"Trial manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = TaskSpec(**payload["task"])
    attempts = {k: AttemptManifest(**v) for k, v in payload["attempts"].items()}
    return TrialManifest(
        trial_id=payload["trial_id"],
        task=task,
        source_repo=payload["source_repo"],
        base_ref=payload["base_ref"],
        base_git_ref=payload["base_git_ref"],
        worktree_root=payload["worktree_root"],
        created_at=payload["created_at"],
        attempts=attempts,
    )


def save_trial(trial: TrialManifest) -> None:
    write_text(manifest_path(trial.trial_id), json.dumps(asdict(trial), indent=2) + "\n")


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ensure_clean_enough_source(source_repo: Path) -> None:
    if not (source_repo / ".git").exists():
        raise SystemExit(f"Source repo is not a git checkout: {source_repo}")


def overlay_game_test_environment(source_repo: Path, worktree: Path) -> None:
    """Carry the local sandbox route into disposable worktrees before it is committed."""
    for rel in [
        Path("src/app/test-env"),
        Path("src/components/test"),
        Path("src/lib/testEnvironments.ts"),
    ]:
        src = source_repo / rel
        dst = worktree / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def make_worktree(
    source_repo: Path,
    worktree_root: Path,
    trial_id: str,
    attempt: str,
    base_ref: str,
    copy_mode: bool = False,
) -> Tuple[str, Path]:
    worktree_root.mkdir(parents=True, exist_ok=True)
    branch = f"arena/{trial_id}-{slug(attempt, 20)}-{uuid.uuid4().hex[:6]}"
    path = worktree_root / f"{trial_id}-{slug(attempt, 20)}"
    if path.exists():
        raise SystemExit(f"Worktree path already exists: {path}")
    if copy_mode:
        ignore = shutil.ignore_patterns(".git", "node_modules", ".next", "dist", "build")
        shutil.copytree(source_repo, path, ignore=ignore)
        overlay_game_test_environment(source_repo, path)
        return branch, path
    code, _ = run_cmd(
        ["git", "-C", str(source_repo), "worktree", "add", "-b", branch, str(path), base_ref],
        cwd=source_repo,
        log_path=trial_dir(trial_id) / "logs" / f"worktree_{slug(attempt)}.log",
        timeout_s=300,
    )
    if code != 0:
        raise SystemExit(f"git worktree add failed; see {trial_dir(trial_id) / 'logs'}")
    overlay_game_test_environment(source_repo, path)
    return branch, path


def append_index(trial: TrialManifest, attempt: Optional[AttemptManifest] = None, event: str = "updated") -> None:
    row = {
        "event": event,
        "time": utc_now(),
        "trial_id": trial.trial_id,
        "task_id": trial.task.id,
        "task_type": trial.task.task_type,
        "title": trial.task.title,
        "source_repo": trial.source_repo,
        "base_git_ref": trial.base_git_ref,
    }
    if attempt:
        row.update(
            {
                "attempt": attempt.attempt,
                "model_label": attempt.model_label,
                "backend": attempt.backend,
                "apply_status": attempt.apply_status,
                "verify_status": attempt.verify_status,
                "cleanup_state": attempt.cleanup_state,
                "preview_url": attempt.preview_url,
            }
        )
    append_jsonl(INDEX_PATH, row)


def create_trial(args: argparse.Namespace) -> TrialManifest:
    tasks = load_task_specs(args.tasks)
    if args.task_id not in tasks:
        raise SystemExit(f"Unknown task id {args.task_id!r}. Available: {', '.join(tasks)}")
    source_repo = Path(args.source_repo).expanduser().resolve()
    ensure_clean_enough_source(source_repo)
    base_ref = args.base_ref
    base_git_ref = git_output(source_repo, ["rev-parse", base_ref])
    trial_id = args.trial_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{slug(args.task_id)}"
    tdir = trial_dir(trial_id)
    tdir.mkdir(parents=True, exist_ok=False)
    task = tasks[args.task_id]
    attempts: Dict[str, AttemptManifest] = {}
    for attempt in args.attempts:
        backend = "frontier_packet" if "frontier" in attempt else "local_mlx"
        model_label = args.frontier_model if backend == "frontier_packet" else args.local_adapter
        branch, path = make_worktree(
            source_repo=source_repo,
            worktree_root=Path(args.worktree_root).expanduser().resolve(),
            trial_id=trial_id,
            attempt=attempt,
            base_ref=base_ref,
            copy_mode=args.copy,
        )
        attempts[attempt] = AttemptManifest(
            attempt=attempt,
            model_label=model_label,
            backend=backend,
            branch=branch,
            worktree_path=str(path),
            created_at=utc_now(),
        )
    trial = TrialManifest(
        trial_id=trial_id,
        task=task,
        source_repo=str(source_repo),
        base_ref=base_ref,
        base_git_ref=base_git_ref,
        worktree_root=str(Path(args.worktree_root).expanduser().resolve()),
        created_at=utc_now(),
        attempts=attempts,
    )
    save_trial(trial)
    write_text(tdir / "task.md", render_task_markdown(trial))
    for attempt in attempts.values():
        append_index(trial, attempt, event="created")
        write_attempt_manifest(trial.trial_id, attempt)
    return trial


def write_attempt_manifest(trial_id: str, attempt: AttemptManifest) -> None:
    write_text(
        attempt_dir(trial_id, attempt.attempt) / "attempt_manifest.json",
        json.dumps(asdict(attempt), indent=2) + "\n",
    )


def render_task_markdown(trial: TrialManifest) -> str:
    task = trial.task
    return f"""# {task.title}

- **Task id:** `{task.id}`
- **Type:** `{task.task_type}`
- **Base ref:** `{trial.base_ref}` / `{trial.base_git_ref}`
- **Preview path:** `{task.preview_path or "/"}`
- **Generation cap:** `{task.max_tokens}` tokens

## Prompt

{task.prompt}

## Allowed Paths

{chr(10).join(f"- `{p}`" for p in task.allowed_paths)}

## Fixed Verification Commands

{chr(10).join(f"- `{c}`" for c in task.verify_commands) or "- None"}

## Notes

{task.notes}
"""


_CONTEXT_CODE_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
_CONTEXT_PACKAGE_JSON_CAP = 2500
_CONTEXT_BODY_RESERVE_MIN = 2600
_CONTEXT_HEAD_LINES = 220
_CONTEXT_TAIL_LINES = 120
_CONTEXT_BM25_CHUNK_LINES = 72


def _tokenize_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _read_text_for_bm25(path: Path, max_chars: int = 56000) -> str:
    raw = read_text(path)
    if len(raw) <= max_chars:
        return raw
    head = (max_chars * 2) // 3
    tail = max_chars - head
    return raw[:head] + "\n" + raw[-tail:]


def _line_chunks(text: str, lines_per_chunk: int) -> List[str]:
    lines = text.splitlines()
    if not lines:
        return [text] if text else []
    chunks: List[str] = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i : i + lines_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks or [""]


def _truncate_file_body(path: Path, text: str, max_chars: int) -> str:
    if max_chars <= 80 or len(text) <= max_chars:
        return text
    suf = path.suffix.lower()
    sep = "\n\n/* … context truncated … */\n\n"
    if suf in _CONTEXT_CODE_SUFFIXES:
        lines = text.splitlines(keepends=True)
        if len(lines) <= _CONTEXT_HEAD_LINES + _CONTEXT_TAIL_LINES + 2:
            return text[:max_chars]
        head = "".join(lines[:_CONTEXT_HEAD_LINES])
        tail = "".join(lines[-_CONTEXT_TAIL_LINES:])
        merged = head + sep + tail
        if len(merged) <= max_chars:
            return merged
    half = (max_chars - len(sep)) // 2
    if half < 40:
        return text[:max_chars]
    return text[:half] + sep + text[-half:]


def _context_pattern_sort_key(pattern: str, prompt_lower: str) -> Tuple[int, int, int, int, str]:
    if pattern in {"README.md", "package.json"}:
        tier = 2
    elif any(ch in pattern for ch in "*?[]"):
        tier = 1
    else:
        tier = 0
    toks = re.findall(r"[A-Za-z0-9_]{4,}", pattern)
    overlap = sum(1 for t in toks if t.lower() in prompt_lower)
    breadth = pattern.count("**") * 3 + pattern.count("*")
    return (tier, breadth, -overlap, len(pattern), pattern)


def _collect_context_files(repo: Path, task: TaskSpec) -> Tuple[List[Path], List[Path]]:
    """Literal-pattern files first (stable), then glob-discovered files (BM25-reordered later)."""
    prompt_lower = task.prompt.lower()
    ordered_patterns = sorted(
        task.context_paths,
        key=lambda p: _context_pattern_sort_key(p, prompt_lower),
    )
    literal_paths: List[Path] = []
    glob_paths: List[Path] = []
    seen: set[str] = set()
    for pattern in ordered_patterns:
        if pattern == "package.json":
            continue
        literal_path = repo / pattern
        is_glob_pattern = any(ch in pattern for ch in "*?[]")
        if literal_path.is_file() and not is_glob_pattern:
            paths_iter: List[Path] = [literal_path]
        else:
            paths_iter = sorted(repo.glob(pattern))
        bucket = glob_paths if is_glob_pattern else literal_paths
        for path in paths_iter:
            if not path.is_file():
                continue
            rel = path.relative_to(repo).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            bucket.append(path)
    return literal_paths, glob_paths


def _bm25_order_paths(paths: List[Path], task: TaskSpec) -> List[Path]:
    if len(paths) <= 1:
        return paths
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return paths
    query = f"{task.prompt}\n" + "\n".join(task.allowed_paths)
    q_tokens = _tokenize_bm25(query)
    if not q_tokens:
        return paths
    flat_chunks: List[Path] = []
    corpus_tokens: List[List[str]] = []
    for path in paths:
        sample = _read_text_for_bm25(path)
        for chunk in _line_chunks(sample, _CONTEXT_BM25_CHUNK_LINES):
            toks = _tokenize_bm25(chunk)
            if not toks:
                continue
            flat_chunks.append(path)
            corpus_tokens.append(toks)
    if not corpus_tokens:
        return paths
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(q_tokens)
    best: Dict[Path, float] = {}
    for i, path in enumerate(flat_chunks):
        s = float(scores[i])
        if s > best.get(path, 0.0):
            best[path] = s
    return sorted(paths, key=lambda p: best.get(p, 0.0), reverse=True)


@dataclass
class ContextPackResult:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


def build_context_pack(
    trial: TrialManifest,
    max_chars: int = 16000,
    *,
    log_dir: Optional[Path] = None,
    use_bm25: bool = True,
) -> ContextPackResult:
    """Assemble repo context: reserved prefix, literal vs glob files, BM25 order for globs, head/tail truncation."""
    repo = Path(trial.source_repo)
    task = trial.task
    meta: Dict[str, Any] = {
        "max_chars": max_chars,
        "use_bm25": use_bm25,
        "literal_paths": [],
        "glob_paths_considered": [],
        "included_files": [],
        "skipped_due_to_budget": [],
    }

    status = git_output(repo, ["status", "--short"])
    task_md = render_task_markdown(trial)
    git_chunk = f"## Git Status\n\n```text\n{status}\n```"
    prefix_parts = [task_md, git_chunk]
    package_json = repo / "package.json"
    if package_json.is_file():
        pkg_body = read_text(package_json)[:_CONTEXT_PACKAGE_JSON_CAP]
        prefix_parts.append(f"## package.json\n\n```json\n{pkg_body}\n```")
    prefix = "\n\n".join(prefix_parts)
    max_prefix = max(500, max_chars - _CONTEXT_BODY_RESERVE_MIN)
    if len(prefix) > max_prefix:
        prefix = prefix[:max_prefix].rstrip() + "\n\n[arena: prefix truncated for body budget]\n"
    body_budget = max(1200, max_chars - len(prefix) - 40)

    literal_paths, glob_paths = _collect_context_files(repo, task)
    meta["literal_paths"] = [p.relative_to(repo).as_posix() for p in literal_paths]
    meta["glob_paths_considered"] = [p.relative_to(repo).as_posix() for p in glob_paths]

    ordered_glob = _bm25_order_paths(glob_paths, task) if use_bm25 else glob_paths
    final_paths = [*literal_paths, *ordered_glob]
    literal_resolved = {p.resolve() for p in literal_paths}

    body_sections: List[str] = []
    remaining = body_budget
    n = len(final_paths)
    for idx, path in enumerate(final_paths):
        raw = read_text(path)
        rel = path.relative_to(repo)
        rel_text = rel.as_posix()
        per_cap = min(6200, max(900, remaining // max(1, min(n - idx, 8))))
        body = _truncate_file_body(path, raw, per_cap)
        section = f"## `{rel}`\n\n```\n{body}\n```"
        overhead = len(section) - len(body)
        if len(section) > remaining:
            tighter = _truncate_file_body(path, raw, max(200, remaining - overhead - 20))
            section = f"## `{rel}`\n\n```\n{tighter}\n```"
        if len(section) > remaining:
            meta["skipped_due_to_budget"].append(rel_text)
            continue
        body_sections.append(section)
        remaining -= len(section)
        meta["included_files"].append(
            {
                "path": rel_text,
                "source": "literal_pattern" if path.resolve() in literal_resolved else "glob_pattern",
                "raw_bytes": len(raw.encode("utf-8")),
                "packed_chars": len(body),
                "est_tokens": estimate_tokens(body),
            }
        )

    body_blob = "\n\n".join(body_sections)
    full = prefix + "\n\n" + body_blob if body_blob else prefix
    if len(full) > max_chars:
        full = full[:max_chars].rstrip() + "\n\n[arena: total context clipped]\n"
    meta["packed_total_chars"] = len(full)
    meta["packed_est_tokens"] = estimate_tokens(full)
    meta["body_budget_chars"] = body_budget
    meta["prefix_chars"] = len(prefix)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        write_text(log_dir / "context_pack.json", json.dumps(meta, indent=2) + "\n")

    return ContextPackResult(text=full, meta=meta)


def selected_context(
    trial: TrialManifest,
    max_chars: int = 16000,
    *,
    log_dir: Optional[Path] = None,
    use_bm25: bool = True,
) -> str:
    return build_context_pack(trial, max_chars, log_dir=log_dir, use_bm25=use_bm25).text


def packet(
    trial_id: str,
    attempt: str,
    *,
    context: Optional[str] = None,
    max_chars: int = 16000,
    use_bm25: bool = True,
) -> Path:
    trial = load_trial(trial_id)
    if attempt not in trial.attempts:
        raise SystemExit(f"Attempt not found: {attempt}")
    adir = attempt_dir(trial_id, attempt)
    if context is None:
        context = build_context_pack(
            trial,
            max_chars=max_chars,
            log_dir=adir / "logs",
            use_bm25=use_bm25,
        ).text
    prompt = f"""# Game Arena Model Packet

You are working inside a disposable Fallen Empire worktree.

Worktree path:
`{trial.attempts[attempt].worktree_path}`

Before editing, infer the app's schema and visual language from the provided files. This is a Next.js App Router game under `src/app` with React components under `src/components`, Zustand game state in `src/store/useGameStore.ts`, and typed systems under `src/types/game.ts` / `src/lib`. Keep the Fallen Empire vibe: dark medieval strategy UI, parchment text, empire-gold accents, restrained red/amber danger states, compact tactical language, and existing Tailwind classes/components. Do not introduce generic sci-fi dashboards, neon cyberpunk UI, unrelated routes, or invented data shapes when the existing code gives a schema.

Do not edit the main game checkout. For small UI edits, prefer fenced full-file blocks because they are less fragile than partial diffs. Return exactly one of:

1. A unified diff that applies with `git apply`, or
2. Fenced file blocks with repo-relative paths:

```tsx path=src/example.tsx
...
```

Only modify allowed paths. Do not include shell commands, commentary, summaries, or both output formats.

{context}
"""
    out = adir / "model_packet.md"
    write_text(out, prompt)
    return out


DIFF_RE = re.compile(r"```(?:diff|patch)?\s*\n(?P<body>diff --git .*?)```", re.DOTALL)
FENCE_RE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r"(?:<\|[^|]+?\|>|</s>|<s>)+")


def sanitize_model_output(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub("", text)
    return text.rstrip() + "\n"


def extract_diff(text: str) -> str:
    match = DIFF_RE.search(text)
    if match:
        return match.group("body").strip() + "\n"
    idx = text.find("diff --git ")
    if idx >= 0:
        return text[idx:].strip() + "\n"
    return ""


PATH_TEXT_RE = r"[A-Za-z0-9_./\[\]-]+\.(?:tsx|ts|jsx|js|json|css|md|html)"


def _clean_candidate_path(raw_path: str) -> str:
    path = raw_path.strip().strip('"').strip("'").strip("`")
    path = re.sub(r"^\s*(?://|#)\s*", "", path).strip()
    path = re.sub(r"^\s*(?:path|file|filename)\s*[:=]\s*", "", path).strip()
    return path.strip().strip('"').strip("'").strip("`")


def candidate_path(info: str, body: str = "") -> Optional[str]:
    patterns = [
        rf"(?:path|file|filename)\s*[:=]\s*[\"']?(?P<path>{PATH_TEXT_RE})",
        rf"(?P<path>{PATH_TEXT_RE})",
    ]
    for source in [info, "\n".join(body.splitlines()[:3])]:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return _clean_candidate_path(match.group("path"))
    return None


def resolve_candidate_repo_path(worktree: Path, raw_path: str) -> Path:
    rel = Path(raw_path.strip().strip('"').strip("'").strip("`").lstrip("./"))
    if (worktree / rel).is_file():
        return rel
    if rel.suffix == ".ts":
        tsx_rel = rel.with_suffix(".tsx")
        if (worktree / tsx_rel).is_file():
            return tsx_rel
    if rel.suffix == ".js":
        jsx_rel = rel.with_suffix(".jsx")
        if (worktree / jsx_rel).is_file():
            return jsx_rel
    return rel


def strip_fenced_path_comment(body: str, rel: Path) -> str:
    lines = body.strip().splitlines()
    if not lines:
        return ""
    first = lines[0].strip()
    rel_text = rel.as_posix()
    first_clean = _clean_candidate_path(first.removesuffix("*/").strip())
    first_rel = Path(first_clean)
    equivalent_ts_to_tsx = first_rel.suffix == ".ts" and rel.suffix == ".tsx" and first_rel.with_suffix(".tsx") == rel
    equivalent_js_to_jsx = first_rel.suffix == ".js" and rel.suffix == ".jsx" and first_rel.with_suffix(".jsx") == rel
    if first_clean == rel_text or equivalent_ts_to_tsx or equivalent_js_to_jsx:
        return "\n".join(lines[1:]).strip() + "\n"
    return body.strip() + "\n"


def body_is_only_path(body: str) -> Optional[str]:
    text = _clean_candidate_path(body)
    if "\n" in text:
        return None
    match = re.fullmatch(PATH_TEXT_RE, text)
    return match.group(0) if match else None


def is_safe_repo_path(path: Path | str, allowed: Iterable[str]) -> bool:
    raw = path.as_posix() if isinstance(path, Path) else path
    rel = Path(raw.strip().lstrip("./"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return False
    text = rel.as_posix()
    return any(fnmatch.fnmatch(text, glob) for glob in allowed)


def apply_fenced_files(worktree: Path, text: str, allowed: List[str], log_path: Path) -> Tuple[bool, List[str]]:
    written: List[str] = []
    log_lines: List[str] = []
    pending_rel: Optional[str] = None
    for match in FENCE_RE.finditer(text):
        body = match.group("body")
        before = text[:match.start()]
        heading_lines = [line.strip() for line in before.splitlines()[-3:]]
        heading_rel = None
        for line in reversed(heading_lines):
            line = line.lstrip("#").strip()
            maybe = body_is_only_path(line)
            if maybe and is_safe_repo_path(resolve_candidate_repo_path(worktree, maybe), allowed):
                heading_rel = maybe
                break
        path_only = body_is_only_path(body)
        if path_only and is_safe_repo_path(resolve_candidate_repo_path(worktree, path_only), allowed):
            pending_rel = path_only
            log_lines.append(f"found path marker {path_only}")
            continue
        raw_rel = pending_rel or candidate_path(match.group("info"), body) or heading_rel
        pending_rel = None
        if not raw_rel:
            log_lines.append(f"skip fence {match.start()}: no repo-relative path found")
            continue
        rel = resolve_candidate_repo_path(worktree, raw_rel)
        if not is_safe_repo_path(rel, allowed):
            log_lines.append(f"reject {rel}: outside allowed paths")
            continue
        target = worktree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(strip_fenced_path_comment(body, rel), encoding="utf-8")
        written.append(rel.as_posix())
        log_lines.append(f"wrote {rel}")
    write_text(log_path, "\n".join(log_lines) + "\n")
    return bool(written), written


def apply_output(args: argparse.Namespace) -> AttemptManifest:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    model_text = read_text(Path(args.input))
    write_text(adir / "model_output.md", model_text)
    worktree = Path(attempt.worktree_path)
    diff = extract_diff(model_text)
    log = adir / "logs" / "apply.log"
    if diff:
        patch_path = adir / "model.patch"
        write_text(patch_path, diff)
        check_code, _ = run_cmd(["git", "apply", "--check", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
        if check_code == 0:
            code, _ = run_cmd(["git", "apply", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
            attempt.apply_status = "applied_diff" if code == 0 else "apply_failed"
        else:
            ok, written = apply_fenced_files(worktree, model_text, trial.task.allowed_paths, adir / "logs" / "apply_fenced_fallback.log")
            attempt.apply_status = f"wrote_files:{','.join(written)}" if ok else "apply_check_failed"
    else:
        ok, written = apply_fenced_files(worktree, model_text, trial.task.allowed_paths, log)
        attempt.apply_status = f"wrote_files:{','.join(written)}" if ok else "no_applyable_changes"
    attempt.last_error = "" if attempt.apply_status.startswith(("applied", "wrote")) else read_text(log)[-1000:]
    save_after_attempt_update(trial, attempt, "applied")
    write_diff_artifacts(trial, attempt)
    return attempt


def save_after_attempt_update(trial: TrialManifest, attempt: AttemptManifest, event: str) -> None:
    trial.attempts[attempt.attempt] = attempt
    save_trial(trial)
    write_attempt_manifest(trial.trial_id, attempt)
    append_index(trial, attempt, event=event)


def write_diff_artifacts(trial: TrialManifest, attempt: AttemptManifest) -> None:
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    worktree = Path(attempt.worktree_path)
    diff_proc = subprocess.run(["git", "diff"], cwd=str(worktree), capture_output=True, text=True)
    stat_proc = subprocess.run(["git", "diff", "--stat"], cwd=str(worktree), capture_output=True, text=True)
    write_text(adir / "diff.patch", diff_proc.stdout)
    write_text(adir / "diff_stat.txt", stat_proc.stdout)


def attempt_applied(attempt: AttemptManifest) -> bool:
    return attempt.apply_status.startswith(("applied", "wrote"))


def write_pairwise_training_signal(trial: TrialManifest) -> Optional[Path]:
    if "local" not in trial.attempts or "frontier" not in trial.attempts:
        return None
    local = trial.attempts["local"]
    frontier = trial.attempts["frontier"]
    local_ok = attempt_applied(local)
    frontier_ok = attempt_applied(frontier)
    if local_ok == frontier_ok:
        return None

    winner_name = "local" if local_ok else "frontier"
    loser_name = "frontier" if local_ok else "local"
    winner = trial.attempts[winner_name]
    loser = trial.attempts[loser_name]
    winner_dir = attempt_dir(trial.trial_id, winner_name)
    loser_dir = attempt_dir(trial.trial_id, loser_name)
    record = {
        "kind": "pairwise_apply_preference",
        "trial_id": trial.trial_id,
        "task": asdict(trial.task),
        "created_at": utc_now(),
        "winner": winner_name,
        "loser": loser_name,
        "reason": "One attempt produced applyable changes and the other did not.",
        "winner_attempt": asdict(winner),
        "loser_attempt": asdict(loser),
        "winner_output": read_text(winner_dir / "model_output.md"),
        "loser_output": read_text(loser_dir / "model_output.md"),
        "winner_diff": read_text(winner_dir / "diff.patch"),
        "winner_diff_stat": read_text(winner_dir / "diff_stat.txt"),
        "loser_error": loser.last_error,
    }
    out = trial_dir(trial.trial_id) / "pairwise_training_signal.json"
    write_text(out, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    append_jsonl(REPO / "benchmarks" / "results" / "game_task_pairwise_training_data.jsonl", record)
    return out


def verify(args: argparse.Namespace) -> AttemptManifest:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    worktree = Path(attempt.worktree_path)
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    results = []
    ok = True
    for idx, command in enumerate(trial.task.verify_commands):
        argv = shlex.split(command)
        code, elapsed = run_cmd(
            argv,
            cwd=worktree,
            log_path=adir / "logs" / f"verify_{idx + 1}_{slug(command, 24)}.log",
            timeout_s=args.timeout,
            env={"CI": "1"},
        )
        results.append({"command": command, "exit_code": code, "elapsed_s": elapsed})
        if code != 0:
            ok = False
    attempt.verify_status = "passed" if ok else "failed"
    write_text(adir / "verify_results.json", json.dumps(results, indent=2) + "\n")
    save_after_attempt_update(trial, attempt, "verified")
    return attempt


def preview(args: argparse.Namespace) -> AttemptManifest:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    command_override = getattr(args, "preview_command_override", None) or getattr(args, "command", None)
    command = normalize_preview_command(command_override or trial.task.preview_command)
    requested_port = int(args.port)
    actual_port = choose_preview_port(requested_port) if args.start else requested_port
    command = command.format(port=actual_port)
    preview_path = normalize_preview_path(trial.task.preview_path)
    preview_url = f"http://127.0.0.1:{actual_port}{preview_path}"
    started = time.perf_counter()
    metadata = {
        "command": command,
        "requested_port": requested_port,
        "port": actual_port,
        "path": preview_path,
        "url": preview_url,
        "started_at": utc_now(),
        "status": "recorded",
    }
    if args.start:
        if not attempt_applied(attempt):
            metadata["status"] = f"skipped_apply_status:{attempt.apply_status}"
            attempt.last_error = attempt.last_error or f"Preview skipped because apply_status is {attempt.apply_status}."
            metadata["elapsed_s"] = time.perf_counter() - started
            attempt.preview_url = preview_url
            attempt.preview_status = str(metadata["status"])
            write_text(adir / "preview.json", json.dumps(metadata, indent=2) + "\n")
            save_after_attempt_update(trial, attempt, "previewed")
            return attempt
        preflight_ok, preflight_status = preview_preflight(trial, attempt, actual_port)
        metadata["preflight_status"] = preflight_status
        if not preflight_ok:
            metadata["status"] = "preflight_failed"
            attempt.last_error = read_text(adir / "logs" / "preview_preflight_tsc.log")[-1000:]
            metadata["elapsed_s"] = time.perf_counter() - started
            attempt.preview_url = preview_url
            attempt.preview_status = str(metadata["status"])
            write_text(adir / "preview.json", json.dumps(metadata, indent=2) + "\n")
            save_after_attempt_update(trial, attempt, "previewed")
            return attempt
        preview_log = adir / "logs" / "preview.log"
        preview_log.parent.mkdir(parents=True, exist_ok=True)
        log = preview_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            shlex.split(command),
            cwd=attempt.worktree_path,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=preview_env(trial, actual_port),
        )
        metadata["pid"] = proc.pid
        attempt.preview_pid = proc.pid
        status, ready_elapsed = wait_for_preview_url(preview_url, proc)
        metadata["status"] = status
        metadata["ready_elapsed_s"] = ready_elapsed
        if status != "ready":
            attempt.last_error = read_text(adir / "logs" / "preview.log")[-1000:]
    metadata["elapsed_s"] = time.perf_counter() - started
    attempt.preview_url = preview_url
    attempt.preview_status = str(metadata["status"])
    write_text(adir / "preview.json", json.dumps(metadata, indent=2) + "\n")
    save_after_attempt_update(trial, attempt, "previewed")
    return attempt


def grade(args: argparse.Namespace) -> Path:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    viability = {
        "applied": attempt_applied(attempt),
        "verified": attempt.verify_status == "passed",
        "preview_ready": attempt.preview_status == "ready",
    }
    if getattr(args, "manual_typecheck", False):
        viability["manual_typecheck"] = True
    if getattr(args, "manual_visible_change", False):
        viability["manual_visible_change"] = True
    rubric_labels = trial.task.grading or [
        "Correctness",
        "Gameplay feel",
        "UI quality",
        "Test confidence",
        "Mergeability",
    ]
    rubric_scores = dict(
        zip(
            rubric_labels,
            [
                args.correctness,
                args.gameplay_feel,
                args.ui_quality,
                args.test_confidence,
                args.mergeability,
            ],
        )
    )
    rating = {
        "trial_id": trial.trial_id,
        "attempt": attempt.attempt,
        "rated_at": utc_now(),
        "rubric_labels": rubric_labels,
        "rubric_scores": rubric_scores,
        "correctness": args.correctness,
        "gameplay_feel": args.gameplay_feel,
        "ui_quality": args.ui_quality,
        "test_confidence": args.test_confidence,
        "mergeability": args.mergeability,
        "cleanup_minutes": args.cleanup_minutes,
        "winner": args.winner,
        "preference_strength": getattr(args, "preference_strength", ""),
        "failure_modes": getattr(args, "failure_modes", []),
        "viability": viability,
        "notes": args.notes,
    }
    out = adir / "rating.json"
    write_text(out, json.dumps(rating, indent=2) + "\n")
    write_diff_artifacts(trial, attempt)
    training = {
        "trial": asdict(trial),
        "attempt": asdict(attempt),
        "rating": rating,
        "diff": read_text(adir / "diff.patch"),
        "diff_stat": read_text(adir / "diff_stat.txt"),
        "model_output": read_text(adir / "model_output.md"),
    }
    write_text(adir / "training_record.json", json.dumps(training, indent=2, ensure_ascii=False) + "\n")
    append_jsonl(REPO / "benchmarks" / "results" / "game_task_training_data.jsonl", training)
    append_index(trial, attempt, event="graded")
    return out


def generate_attempt(args: argparse.Namespace) -> Path:
    load_dotenv()
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    log_root = adir / "logs"
    use_bm25_ctx = not getattr(args, "no_context_bm25", False)
    context = build_context_pack(
        trial,
        max_chars=args.context_chars,
        log_dir=log_root,
        use_bm25=use_bm25_ctx,
    ).text
    local_small_model = args.backend == "local"
    format_instruction = (
        "Return only fenced full-file blocks with repo-relative paths. Do not return a diff. "
        "For this small UI task, rewrite the complete target file(s) exactly as they should exist. "
        "Preserve existing exported component/function names unless the task explicitly asks to rename them. "
        if local_small_model
        else "Return only one output format: either a unified diff that applies with git apply, "
        "or fenced full-file blocks with repo-relative paths. For small UI edits, fenced "
        "full-file blocks are preferred because they avoid fragile hunk headers. "
    )
    prompt = (
        format_instruction
        + "Do not "
        "include shell commands, commentary, summaries, or both output formats. Stay within "
        "allowed paths.\n\n"
        f"{context}"
    )
    requested_max_tokens = int(args.max_tokens or trial.task.max_tokens or 4096)
    input_tokens = estimate_tokens("You are a careful TypeScript game engineer.\n" + prompt)
    ctx_log = log_root / "context_pack.json"
    if ctx_log.is_file():
        try:
            pack_log = json.loads(read_text(ctx_log))
            pack_log["full_user_prompt_est_tokens"] = input_tokens
            write_text(ctx_log, json.dumps(pack_log, indent=2) + "\n")
        except (json.JSONDecodeError, OSError):
            pass
    started = time.perf_counter()
    usage: Dict[str, Any] = {}
    estimated_cost_usd = 0.0
    if args.backend == "local":
        backend = LocalMlxBackend(adapter_path=args.adapter_path)
        text = backend.generate(
            GenerationRequest(
                messages=[ChatMessage("system", "You are a careful TypeScript game engineer."), ChatMessage("user", prompt)],
                max_tokens=requested_max_tokens,
                temperature=args.temp,
            )
        )
    elif args.backend == "frontier":
        backend = OpenAICompatibleBackend(model=args.model)
        text, usage = backend.generate(
            GenerationRequest(
                messages=[ChatMessage("system", "You are a careful TypeScript game engineer."), ChatMessage("user", prompt)],
                max_tokens=requested_max_tokens,
                temperature=args.temp,
            )
        )
        estimated_cost_usd = (
            (usage.get("prompt_tokens", input_tokens) * 5.0)
            + (usage.get("completion_tokens", 0) * 15.0)
        ) / 1_000_000
        write_text(adir / "frontier_usage.json", json.dumps(usage, indent=2) + "\n")
    else:
        return packet(
            args.trial_id,
            args.attempt,
            context=context,
            max_chars=args.context_chars,
            use_bm25=use_bm25_ctx,
        )
    text = sanitize_model_output(text)
    elapsed = time.perf_counter() - started
    output_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    metrics = {
        "trial_id": trial.trial_id,
        "attempt": attempt.attempt,
        "backend": args.backend,
        "model_label": attempt.model_label,
        "started_at": utc_now(),
        "elapsed_s": elapsed,
        "max_tokens": requested_max_tokens,
        "input_tokens": int(usage.get("prompt_tokens") or input_tokens),
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "usage": usage,
    }
    attempt.generation_elapsed_s = elapsed
    attempt.generation_input_tokens = metrics["input_tokens"]
    attempt.generation_output_tokens = output_tokens
    attempt.generation_total_tokens = total_tokens
    attempt.generation_cost_usd = estimated_cost_usd
    trial.attempts[attempt.attempt] = attempt
    save_trial(trial)
    write_attempt_manifest(trial.trial_id, attempt)
    write_text(adir / "generation_metrics.json", json.dumps(metrics, indent=2) + "\n")
    append_index(trial, attempt, event="generated")
    out = adir / "model_output.md"
    write_text(out, text)
    return out


def cleanup(args: argparse.Namespace) -> AttemptManifest:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    if attempt.cleanup_state == "removed":
        return attempt
    worktree = Path(attempt.worktree_path)
    source = Path(trial.source_repo)
    if worktree.exists():
        if (source / ".git").exists() and not args.copy:
            run_cmd(
                ["git", "-C", str(source), "worktree", "remove", "--force", str(worktree)],
                cwd=source,
                log_path=attempt_dir(trial.trial_id, attempt.attempt) / "logs" / "cleanup.log",
                timeout_s=180,
            )
        else:
            shutil.rmtree(worktree)
    if attempt.branch.startswith("arena/") and (source / ".git").exists() and not args.keep_branch:
        run_cmd(
            ["git", "-C", str(source), "branch", "-D", attempt.branch],
            cwd=source,
            log_path=attempt_dir(trial.trial_id, attempt.attempt) / "logs" / "cleanup_branch.log",
            timeout_s=60,
        )
    attempt.cleanup_state = "removed"
    save_after_attempt_update(trial, attempt, "cleaned")
    return attempt


def list_trials(_: argparse.Namespace) -> None:
    if not RESULTS_ROOT.is_dir():
        print("No game task trials yet.")
        return
    for path in sorted(RESULTS_ROOT.glob("*/trial_manifest.json"), reverse=True):
        trial = load_trial(path.parent.name)
        attempts = ", ".join(f"{a.attempt}:{a.verify_status}/{a.cleanup_state}" for a in trial.attempts.values())
        print(f"{trial.trial_id}  {trial.task.id}  {attempts}")


def report(_: argparse.Namespace) -> Path:
    rows = []
    if INDEX_PATH.is_file():
        for raw in INDEX_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
    latest_by_attempt: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if "attempt" in row:
            latest_by_attempt[(row["trial_id"], row["attempt"])] = row
    summary: Dict[str, Dict[str, Any]] = {}
    for row in latest_by_attempt.values():
        model = row.get("model_label") or row.get("backend") or "unknown"
        bucket = summary.setdefault(model, {"attempts": 0, "passed": 0, "failed": 0, "wins": 0})
        bucket["attempts"] += 1
        if row.get("verify_status") == "passed":
            bucket["passed"] += 1
        if row.get("verify_status") == "failed":
            bucket["failed"] += 1
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "game_task_summary.md"
    lines = [
        "# Game Task Arena Summary",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "| Model / Adapter | Attempts | Verify Passed | Verify Failed |",
        "|---|---:|---:|---:|",
    ]
    for model, stats in sorted(summary.items()):
        lines.append(f"| `{model}` | {stats['attempts']} | {stats['passed']} | {stats['failed']} |")
    lines.extend(["", "## Latest Attempts", ""])
    for row in sorted(latest_by_attempt.values(), key=lambda r: r.get("time", ""), reverse=True)[:50]:
        lines.append(
            f"- `{row.get('trial_id')}` / `{row.get('attempt')}`: "
            f"{row.get('task_type')} `{row.get('model_label')}` "
            f"apply={row.get('apply_status')} verify={row.get('verify_status')} cleanup={row.get('cleanup_state')}"
        )
    write_text(out, "\n".join(lines) + "\n")
    print(out)
    return out


def build_app():
    import gradio as gr
    theme_css = """
:root {
  --arena-bg: #000000;
  --arena-panel: #0b0f19;
  --arena-panel-solid: #101624;
  --arena-panel-muted: #151d2e;
  --arena-border: rgba(148, 163, 184, 0.22);
  --arena-text: #f8fafc;
  --arena-muted: #cbd5e1;
  --arena-input: #050914;
  --arena-code: #172554;
  --arena-blue: #2563eb;
  --arena-blue-light: #93c5fd;
}
.gradio-container,
body {
  background: #000000 !important;
}
.gradio-container {
  background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.20), transparent 34rem), #000000 !important;
  color: var(--arena-text) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .input-container,
.gradio-container .wrap,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .accordion {
  border-radius: 18px !important;
  border-color: var(--arena-border) !important;
  background: var(--arena-panel) !important;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.36) !important;
}
.gradio-container,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container label,
.gradio-container p,
.gradio-container li,
.gradio-container span,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  color: var(--arena-text) !important;
}
.gradio-container div,
.gradio-container section,
.gradio-container article,
.gradio-container .contain,
.gradio-container .gap,
.gradio-container .row,
.gradio-container .column,
.gradio-container .block,
.gradio-container .form,
.gradio-container .prose,
.gradio-container .markdown {
  color: var(--arena-text) !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .info,
.gradio-container .secondary-text {
  color: var(--arena-muted) !important;
}
.gradio-container button.primary, .gradio-container button.primary:hover {
  background: linear-gradient(180deg, #3b82f6 0%, var(--arena-blue) 100%) !important;
  border: 0 !important;
  color: #ffffff !important;
  border-radius: 14px !important;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24) !important;
}
.gradio-container button.secondary, .gradio-container button {
  border-radius: 14px !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: var(--arena-input) !important;
  border: 1px solid var(--arena-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  color: var(--arena-text) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--arena-muted) !important;
  opacity: 0.85 !important;
}
.gradio-container a {
  color: var(--arena-blue-light) !important;
}
.gradio-container table,
.gradio-container th,
.gradio-container td {
  color: var(--arena-text) !important;
  border-color: var(--arena-border) !important;
}
.gradio-container h1 {
  letter-spacing: -0.04em;
}
.gradio-container h2, .gradio-container h3 {
  letter-spacing: -0.025em;
}
.gradio-container code {
  background: var(--arena-code) !important;
  color: var(--arena-text) !important;
  border-radius: 7px;
  padding: 0.08rem 0.3rem;
}
.gradio-container pre {
  background: var(--arena-panel-muted) !important;
  color: var(--arena-text) !important;
  border: 1px solid var(--arena-border) !important;
}
.gradio-container button.secondary,
.gradio-container button:not(.primary) {
  background: var(--arena-panel-solid) !important;
  border: 1px solid var(--arena-border) !important;
  color: var(--arena-text) !important;
}
"""

    def create_ui(task_id, source_repo, worktree_root, base_ref):
        ns = argparse.Namespace(
            task_id=task_id,
            tasks=DEFAULT_TASKS,
            source_repo=source_repo or str(DEFAULT_SOURCE_REPO),
            worktree_root=worktree_root or str(DEFAULT_WORKTREE_ROOT),
            base_ref=base_ref or "HEAD",
            trial_id=None,
            attempts=["local", "frontier"],
            local_adapter="checkpoints/fe-lora-30m",
            frontier_model=os.environ.get("FRONTIER_MODEL", "frontier"),
            copy=False,
        )
        trial = create_trial(ns)
        summary = (
            f"Created trial `{trial.trial_id}` from standardized task `{trial.task.id}`.\n\n"
            "Next: generate/apply model attempts, then start preview links."
        )
        return trial.trial_id, summary

    def run_full_trial_ui(
        task_id,
        source_repo,
        worktree_root,
        base_ref,
        local_adapter,
        frontier_model,
        max_tokens,
        local_port,
        frontier_port,
        start_servers,
    ):
        trial_id, create_summary = create_ui(task_id, source_repo, worktree_root, base_ref)
        generate_status = generate_apply_both_ui(trial_id, local_adapter, frontier_model, max_tokens)
        preview_status, local_link, frontier_link = preview_both_ui(
            trial_id,
            local_port,
            frontier_port,
            start_servers,
        )
        status = (
            f"{create_summary}\n\n"
            f"### Generation\n{generate_status}\n\n"
            f"### Preview\n{preview_status}"
        )
        return trial_id, status, local_link, frontier_link

    def generate_apply_both_ui(trial_id, local_adapter, frontier_model, max_tokens):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first."
        statuses = []
        for attempt_name, backend in [("local", "local"), ("frontier", "frontier")]:
            try:
                context_chars = 9000 if backend == "local" else 16000
                output_path = generate_attempt(
                    argparse.Namespace(
                        trial_id=trial_id,
                        attempt=attempt_name,
                        backend=backend,
                        adapter_path=local_adapter or "checkpoints/fe-lora-30m",
                        model=frontier_model or os.environ.get("FRONTIER_MODEL"),
                        max_tokens=int(max_tokens or 4096),
                        temp=0.0,
                        context_chars=context_chars,
                        no_context_bm25=False,
                    )
                )
                manifest = apply_output(
                    argparse.Namespace(trial_id=trial_id, attempt=attempt_name, input=str(output_path))
                )
                elapsed = manifest.generation_elapsed_s or 0.0
                tokens = manifest.generation_total_tokens or 0
                statuses.append(
                    f"- `{attempt_name}` generated in {elapsed:.1f}s using ~{tokens} tokens; apply status is `{manifest.apply_status}`"
                )
            except Exception as exc:
                statuses.append(f"- `{attempt_name}` failed: `{type(exc).__name__}: {exc}`")
        try:
            signal_path = write_pairwise_training_signal(load_trial(trial_id))
            if signal_path:
                statuses.append(f"- pairwise training signal saved: `{signal_path}`")
        except Exception as exc:
            statuses.append(f"- pairwise training signal failed: `{type(exc).__name__}: {exc}`")
        return "\n".join(statuses)

    def preview_both_ui(trial_id, local_port, frontier_port, start_servers):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first.", "", ""
        local = preview(
            argparse.Namespace(
                trial_id=trial_id,
                attempt="local",
                port=int(local_port or 5174),
                command=None,
                start=bool(start_servers),
            )
        )
        frontier = preview(
            argparse.Namespace(
                trial_id=trial_id,
                attempt="frontier",
                port=int(frontier_port or 5175),
                command=None,
                start=bool(start_servers),
            )
        )
        status = "\n".join(
            [
                "Preview metadata recorded.",
                f"- `local`: `{local.preview_status}`",
                f"- `frontier`: `{frontier.preview_status}`",
            ]
        )
        if start_servers:
            status += "\n\nLinks are marked ready only after the dev server answers the sandbox URL."
        local_link = f"[Open Local Preview]({local.preview_url})" if local.preview_status == "ready" else f"Local preview not ready: `{local.preview_status}`"
        frontier_link = f"[Open Frontier Preview]({frontier.preview_url})" if frontier.preview_status == "ready" else f"Frontier preview not ready: `{frontier.preview_status}`"
        return status, local_link, frontier_link

    def complete_ui(
        trial_id,
        winner,
        local_correctness,
        local_feel,
        local_ui,
        local_tests,
        local_merge,
        frontier_correctness,
        frontier_feel,
        frontier_ui,
        frontier_tests,
        frontier_merge,
        preference_strength,
        failure_modes,
        manual_local_typecheck,
        manual_local_visible_change,
        manual_frontier_typecheck,
        manual_frontier_visible_change,
        notes,
        cleanup_after,
    ):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first."
        outputs = []
        for attempt_name, scores in [
            (
                "local",
                {
                    "correctness": local_correctness,
                    "gameplay_feel": local_feel,
                    "ui_quality": local_ui,
                    "test_confidence": local_tests,
                    "mergeability": local_merge,
                },
            ),
            (
                "frontier",
                {
                    "correctness": frontier_correctness,
                    "gameplay_feel": frontier_feel,
                    "ui_quality": frontier_ui,
                    "test_confidence": frontier_tests,
                    "mergeability": frontier_merge,
                },
            ),
        ]:
            grade_path = grade(
                argparse.Namespace(
                    trial_id=trial_id,
                    attempt=attempt_name,
                    correctness=int(scores["correctness"]),
                    gameplay_feel=int(scores["gameplay_feel"]),
                    ui_quality=int(scores["ui_quality"]),
                    test_confidence=int(scores["test_confidence"]),
                    mergeability=int(scores["mergeability"]),
                    cleanup_minutes=0,
                    winner=winner if winner == attempt_name else "no",
                    preference_strength=preference_strength,
                    failure_modes=failure_modes or [],
                    manual_typecheck=bool(
                        manual_local_typecheck if attempt_name == "local" else manual_frontier_typecheck
                    ),
                    manual_visible_change=bool(
                        manual_local_visible_change if attempt_name == "local" else manual_frontier_visible_change
                    ),
                    notes=notes or "",
                )
            )
            outputs.append(f"- graded `{attempt_name}`: `{grade_path}`")
            if cleanup_after:
                cleaned = cleanup(
                    argparse.Namespace(
                        trial_id=trial_id,
                        attempt=attempt_name,
                        copy=False,
                        keep_branch=False,
                    )
                )
                outputs.append(f"- cleaned `{attempt_name}`: `{cleaned.cleanup_state}`")
        report(argparse.Namespace())
        outputs.append("- refreshed summary report")
        return "\n".join(outputs)

    def packet_ui(trial_id, attempt):
        if not trial_id.strip():
            return "", "Create a trial first.", ""
        out = packet(trial_id.strip(), attempt)
        trial = load_trial(trial_id.strip())
        manifest = trial.attempts[attempt]
        return str(out), read_text(out), json.dumps(asdict(manifest), indent=2)

    def verify_ui(trial_id, attempt):
        if not trial_id.strip():
            return "Create a trial first."
        att = verify(argparse.Namespace(trial_id=trial_id.strip(), attempt=attempt, timeout=600))
        return json.dumps(asdict(att), indent=2)

    def report_ui():
        return read_text(report(argparse.Namespace()))

    def task_details_ui(task_id):
        task = load_task_specs()[task_id]
        allowed = "\n".join(f"- `{p}`" for p in task.allowed_paths)
        verify_lines = "\n".join(f"- `{c}`" for c in task.verify_commands) or "- None"
        grading = "\n".join(f"- {g}" for g in (task.grading or [])) or "- Generic grading"
        return f"""## {task.title}

**Type:** `{task.task_type}`

{task.prompt}

### Allowed Paths

{allowed}

### Fixed Verification

{verify_lines}

### Task-Specific Grading

{grading}

### Notes

{task.notes}
"""

    def grading_labels_ui(task_id):
        task = load_task_specs()[task_id]
        labels = (task.grading or [
            "Correctness",
            "Gameplay feel",
            "UI quality",
            "Test confidence",
            "Mergeability",
        ])[:5]
        while len(labels) < 5:
            labels.append(f"Criterion {len(labels) + 1}")
        return [gr.update(label=label) for label in labels]

    def _worktree_link(attempt: AttemptManifest) -> str:
        path = Path(attempt.worktree_path)
        file_uri = path.as_uri()
        cursor_uri = f"cursor://file{path}"
        return (
            f"### `{attempt.attempt}`\n\n"
            f"[Open in Cursor]({cursor_uri})  \n"
            f"[Open as file URL]({file_uri})\n\n"
            f"Path: `{path}`\n\n"
            f"Branch: `{attempt.branch}`"
        )

    with gr.Blocks(title="Fallen Empire Game Task Arena", css=theme_css) as demo:
        gr.Markdown(
            "# Fallen Empire Game Task Arena\n"
            "Choose a standardized task, generate Local and Frontier attempts in disposable worktrees, open playable previews, then grade and clean up."
        )
        task_id = gr.Dropdown(
            choices=list(load_task_specs().keys()),
            label="Standardized Test",
            value=next(iter(load_task_specs())),
        )
        task_details = gr.Markdown()
        task_id.change(task_details_ui, inputs=[task_id], outputs=[task_details])
        demo.load(task_details_ui, inputs=[task_id], outputs=[task_details])

        with gr.Accordion("Advanced paths (usually leave these alone)", open=False):
            source_repo = gr.Textbox(label="Source repo", value=str(DEFAULT_SOURCE_REPO))
            worktree_root = gr.Textbox(label="Worktree root", value=str(DEFAULT_WORKTREE_ROOT))
            base_ref = gr.Textbox(label="Base ref", value="HEAD")

        with gr.Accordion("Model and preview settings", open=False):
            with gr.Row():
                gen_local_adapter = gr.Textbox(label="Local adapter", value="checkpoints/fe-lora-30m")
                gen_frontier_model = gr.Textbox(label="Frontier model", value=os.environ.get("FRONTIER_MODEL", ""))
                gen_tokens = gr.Number(label="Max tokens", value=4096, precision=0)
            with gr.Row():
                local_port = gr.Number(label="Local port", value=5174, precision=0)
                frontier_port = gr.Number(label="Frontier port", value=5175, precision=0)
                start_servers = gr.Checkbox(label="Start dev servers", value=True)

        run_trial_btn = gr.Button("Run Full Trial: Generate Both + Open Preview Links", variant="primary")
        trial_id = gr.Textbox(label="Trial ID", interactive=False)
        run_status = gr.Markdown()
        with gr.Row():
            local_preview_link = gr.Markdown()
            frontier_preview_link = gr.Markdown()
        run_trial_btn.click(
            run_full_trial_ui,
            inputs=[
                task_id,
                source_repo,
                worktree_root,
                base_ref,
                gen_local_adapter,
                gen_frontier_model,
                gen_tokens,
                local_port,
                frontier_port,
                start_servers,
            ],
            outputs=[trial_id, run_status, local_preview_link, frontier_preview_link],
        )

        gr.Markdown("## Grade And Complete")
        winner = gr.Radio(["local", "frontier", "tie", "neither"], value="tie", label="Winner")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Local")
                local_correctness = gr.Slider(1, 5, value=3, step=1, label="Correctness")
                local_feel = gr.Slider(1, 5, value=3, step=1, label="Gameplay feel")
                local_ui = gr.Slider(1, 5, value=3, step=1, label="UI quality")
                local_tests = gr.Slider(1, 5, value=3, step=1, label="Test confidence")
                local_merge = gr.Slider(1, 5, value=3, step=1, label="Mergeability")
            with gr.Column():
                gr.Markdown("### Frontier")
                frontier_correctness = gr.Slider(1, 5, value=3, step=1, label="Correctness")
                frontier_feel = gr.Slider(1, 5, value=3, step=1, label="Gameplay feel")
                frontier_ui = gr.Slider(1, 5, value=3, step=1, label="UI quality")
                frontier_tests = gr.Slider(1, 5, value=3, step=1, label="Test confidence")
                frontier_merge = gr.Slider(1, 5, value=3, step=1, label="Mergeability")
        rubric_outputs = [
            local_correctness,
            local_feel,
            local_ui,
            local_tests,
            local_merge,
            frontier_correctness,
            frontier_feel,
            frontier_ui,
            frontier_tests,
            frontier_merge,
        ]
        task_id.change(
            lambda task: grading_labels_ui(task) * 2,
            inputs=[task_id],
            outputs=rubric_outputs,
        )
        demo.load(
            lambda task: grading_labels_ui(task) * 2,
            inputs=[task_id],
            outputs=rubric_outputs,
        )
        notes = gr.Textbox(label="Notes", lines=4)
        preference_strength = gr.Radio(
            ["weak", "medium", "strong", "invalid/no winner"],
            value="medium",
            label="Preference strength",
            info="How clear was the winner after considering apply/typecheck/preview and quality?",
        )
        failure_modes = gr.CheckboxGroup(
            [
                "no output",
                "parse/apply failed",
                "typecheck/preflight failed",
                "preview failed",
                "no visible change",
                "wrong file/schema",
                "generic/off-theme",
                "regressed existing exports",
                "too broad/risky",
            ],
            label="Observed failure modes",
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Local viability")
                manual_local_typecheck = gr.Checkbox(label="Typecheck verified", value=False)
                manual_local_visible_change = gr.Checkbox(label="Visible change confirmed", value=False)
            with gr.Column():
                gr.Markdown("#### Frontier viability")
                manual_frontier_typecheck = gr.Checkbox(label="Typecheck verified", value=False)
                manual_frontier_visible_change = gr.Checkbox(label="Visible change confirmed", value=False)
        cleanup_after = gr.Checkbox(label="Delete disposable worktrees after saving", value=True)
        complete_btn = gr.Button("Complete Trial: Save Results + Cleanup", variant="primary")
        complete_status = gr.Markdown()
        complete_btn.click(
            complete_ui,
            inputs=[
                trial_id,
                winner,
                local_correctness,
                local_feel,
                local_ui,
                local_tests,
                local_merge,
                frontier_correctness,
                frontier_feel,
                frontier_ui,
                frontier_tests,
                frontier_merge,
                preference_strength,
                failure_modes,
                manual_local_typecheck,
                manual_local_visible_change,
                manual_frontier_typecheck,
                manual_frontier_visible_change,
                notes,
                cleanup_after,
            ],
            outputs=[complete_status],
        )

        with gr.Accordion("Details: manual packet / verification / reports", open=False):
            attempt = gr.Radio(["local", "frontier"], value="local", label="Attempt")
            with gr.Row():
                create_btn = gr.Button("Create Worktrees Only")
                generate_btn = gr.Button("Generate + Apply Both")
                preview_btn = gr.Button("Start Preview Links")
                packet_btn = gr.Button("Generate Model Packet")
                verify_btn = gr.Button("Run Fixed Verification")
                report_btn = gr.Button("Refresh Summary Report")
            detail_status = gr.Markdown()
            packet_path = gr.Textbox(label="Packet path", interactive=False)
            packet_text = gr.Textbox(label="Packet text", lines=12)
            attempt_manifest = gr.Textbox(label="Attempt manifest / verify output", lines=10)
            report_md = gr.Markdown()
            create_btn.click(create_ui, inputs=[task_id, source_repo, worktree_root, base_ref], outputs=[trial_id, detail_status])
            generate_btn.click(generate_apply_both_ui, inputs=[trial_id, gen_local_adapter, gen_frontier_model, gen_tokens], outputs=[detail_status])
            preview_btn.click(preview_both_ui, inputs=[trial_id, local_port, frontier_port, start_servers], outputs=[detail_status, local_preview_link, frontier_preview_link])
            packet_btn.click(packet_ui, inputs=[trial_id, attempt], outputs=[packet_path, packet_text, attempt_manifest])
            verify_btn.click(verify_ui, inputs=[trial_id, attempt], outputs=[attempt_manifest])
            report_btn.click(report_ui, outputs=[report_md])
    return demo


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Disposable game-task arena for Fallen Empire model evals")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("--task-id", required=True)
    p_create.add_argument("--source-repo", default=str(DEFAULT_SOURCE_REPO))
    p_create.add_argument("--worktree-root", default=str(DEFAULT_WORKTREE_ROOT))
    p_create.add_argument("--base-ref", default="HEAD")
    p_create.add_argument("--trial-id")
    p_create.add_argument("--attempts", nargs="+", default=["local", "frontier"])
    p_create.add_argument("--local-adapter", default="checkpoints/fe-lora-30m")
    p_create.add_argument("--frontier-model", default=os.environ.get("FRONTIER_MODEL", "frontier"))
    p_create.add_argument("--copy", action="store_true", help="Copy source repo instead of git worktree.")

    p_packet = sub.add_parser("packet")
    p_packet.add_argument("--trial-id", required=True)
    p_packet.add_argument("--attempt", required=True)
    p_packet.add_argument("--context-chars", type=int, default=16000)
    p_packet.add_argument(
        "--no-context-bm25",
        action="store_true",
        help="Disable BM25 reordering among glob-matched context files.",
    )

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("--trial-id", required=True)
    p_generate.add_argument("--attempt", required=True)
    p_generate.add_argument("--backend", choices=["packet", "local", "frontier"], default="packet")
    p_generate.add_argument("--adapter-path", default="checkpoints/fe-lora-30m")
    p_generate.add_argument("--model", default=os.environ.get("FRONTIER_MODEL"))
    p_generate.add_argument("--max-tokens", type=int, default=4096)
    p_generate.add_argument("--temp", type=float, default=0.0)
    p_generate.add_argument("--context-chars", type=int, default=16000)
    p_generate.add_argument(
        "--no-context-bm25",
        action="store_true",
        help="Disable BM25 reordering among glob-matched context files (deterministic order only).",
    )

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--trial-id", required=True)
    p_apply.add_argument("--attempt", required=True)
    p_apply.add_argument("--input", required=True)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--trial-id", required=True)
    p_verify.add_argument("--attempt", required=True)
    p_verify.add_argument("--timeout", type=int, default=600)

    p_preview = sub.add_parser("preview")
    p_preview.add_argument("--trial-id", required=True)
    p_preview.add_argument("--attempt", required=True)
    p_preview.add_argument("--port", type=int, default=5174)
    p_preview.add_argument("--command", dest="preview_command_override")
    p_preview.add_argument("--start", action="store_true")

    p_grade = sub.add_parser("grade")
    p_grade.add_argument("--trial-id", required=True)
    p_grade.add_argument("--attempt", required=True)
    p_grade.add_argument("--correctness", type=int, default=3)
    p_grade.add_argument("--gameplay-feel", type=int, default=3)
    p_grade.add_argument("--ui-quality", type=int, default=3)
    p_grade.add_argument("--test-confidence", type=int, default=3)
    p_grade.add_argument("--mergeability", type=int, default=3)
    p_grade.add_argument("--cleanup-minutes", type=int, default=0)
    p_grade.add_argument("--winner", default="no")
    p_grade.add_argument("--notes", default="")

    p_cleanup = sub.add_parser("cleanup")
    p_cleanup.add_argument("--trial-id", required=True)
    p_cleanup.add_argument("--attempt", required=True)
    p_cleanup.add_argument("--copy", action="store_true")
    p_cleanup.add_argument("--keep-branch", action="store_true")

    sub.add_parser("list")
    sub.add_parser("report")
    p_ui = sub.add_parser("ui")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7868)
    p_ui.add_argument("--share", action="store_true")

    args = parser.parse_args()
    if args.subcommand == "create":
        trial = create_trial(args)
        print(trial.trial_id)
    elif args.subcommand == "packet":
        print(
            packet(
                args.trial_id,
                args.attempt,
                max_chars=args.context_chars,
                use_bm25=not args.no_context_bm25,
            )
        )
    elif args.subcommand == "generate":
        print(generate_attempt(args))
    elif args.subcommand == "apply":
        print(json.dumps(asdict(apply_output(args)), indent=2))
    elif args.subcommand == "verify":
        print(json.dumps(asdict(verify(args)), indent=2))
    elif args.subcommand == "preview":
        print(json.dumps(asdict(preview(args)), indent=2))
    elif args.subcommand == "grade":
        print(grade(args))
    elif args.subcommand == "cleanup":
        print(json.dumps(asdict(cleanup(args)), indent=2))
    elif args.subcommand == "list":
        list_trials(args)
    elif args.subcommand == "report":
        report(args)
    elif args.subcommand == "ui":
        build_app().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
