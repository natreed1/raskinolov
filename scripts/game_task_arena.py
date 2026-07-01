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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
DEFAULT_SOURCE_REPO = Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))).expanduser()
DEFAULT_WORKTREE_ROOT = Path(os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena"))).expanduser()
RESULTS_ROOT = REPO / "benchmarks" / "results" / "game_task_trials"
INDEX_PATH = REPO / "benchmarks" / "results" / "game_task_index.jsonl"
REPORTS_DIR = REPO / "benchmarks" / "results" / "game_task_reports"
DEFAULT_TASKS = REPO / "benchmarks" / "game_task_arena_examples.json"
MODEL_CHAT_DEBATES_PATH = REPO / "benchmarks" / "results" / "model_chat_debates" / "debates.jsonl"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_router import ChatMessage, GenerationRequest, LocalMlxBackend, OpenAICompatibleBackend, estimate_tokens
from documentation_rag import build_context as rag_build_context, load_corpus as rag_load_corpus, retrieve as rag_retrieve

DEFAULT_BUG_CHECK_SYSTEM_PROMPT = (
    "You are a strict patch reviewer. Inspect the candidate patch for likely TypeScript/import/export/path "
    "bugs and task-shape violations. If needed, rewrite the candidate to fix issues. Return only the final "
    "patch output in the same format (unified diff or fenced files), with no commentary."
)
DEFAULT_BUG_FIX_RAG_CORPUS = REPO / "data" / "rag" / "bug_fix_agent_corpus.json"
_BUG_FIX_CORPUS_CACHE: Dict[str, List[Any]] = {}
_LOCAL_BACKEND_CACHE: Dict[Tuple[str, str], LocalMlxBackend] = {}
_FRONTIER_BACKEND_CACHE: Dict[Tuple[str, str], OpenAICompatibleBackend] = {}


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


def verify_env(trial: TrialManifest) -> Dict[str, str]:
    source_repo = Path(trial.source_repo)
    node_bin = source_repo / "node_modules" / ".bin"
    path_parts = [str(node_bin)] if node_bin.is_dir() else []
    path_parts.append(os.environ.get("PATH", ""))
    env = {**os.environ, "PATH": os.pathsep.join(path_parts)}
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
        if "council" in attempt:
            backend = "council_mlx"
            model_label = f"council({args.local_adapter})"
        elif "frontier" in attempt:
            backend = "frontier_packet"
            model_label = args.frontier_model
        else:
            backend = "local_mlx"
            model_label = args.local_adapter
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


def build_context_pack_for_task_spec(
    repo: Path,
    task: TaskSpec,
    max_chars: int = 16000,
    *,
    log_dir: Optional[Path] = None,
    use_bm25: bool = True,
    header_markdown: Optional[str] = None,
    include_git_status: bool = True,
) -> ContextPackResult:
    """Assemble repo context for a TaskSpec without a full trial manifest."""
    meta: Dict[str, Any] = {
        "max_chars": max_chars,
        "use_bm25": use_bm25,
        "literal_paths": [],
        "glob_paths_considered": [],
        "included_files": [],
        "skipped_due_to_budget": [],
    }
    if header_markdown is not None:
        task_md = header_markdown
    else:
        task_md = f"""# {task.title}

- **Task id:** `{task.id}`
- **Type:** `{task.task_type}`
- **Preview path:** `{task.preview_path or "/"}`

## Prompt

{task.prompt}

## Allowed Paths

{chr(10).join(f"- `{p}`" for p in task.allowed_paths)}

## Fixed Verification Commands

{chr(10).join(f"- `{c}`" for c in task.verify_commands) or "- None"}
"""
    prefix_parts = [task_md]
    if include_git_status:
        status = git_output(repo, ["status", "--short"])
        prefix_parts.append(f"## Git Status\n\n```text\n{status}\n```")
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
    return build_context_pack_for_task_spec(
        repo,
        task,
        max_chars,
        log_dir=log_dir,
        use_bm25=use_bm25,
        header_markdown=render_task_markdown(trial),
        include_git_status=True,
    )


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


def _resolve_corpus_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO / path).resolve()
    return path


def _load_bug_fix_corpus_cached(path: Path) -> List[Any]:
    key = str(path.resolve())
    cached = _BUG_FIX_CORPUS_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        _BUG_FIX_CORPUS_CACHE[key] = []
        return []
    try:
        corpus = rag_load_corpus(path)
    except Exception:
        corpus = []
    _BUG_FIX_CORPUS_CACHE[key] = corpus
    return corpus


def build_bug_fix_rag_context(
    *,
    original_prompt: str,
    candidate_text: str,
    corpus_path: Path,
    top_k: int,
    max_chars: int,
) -> Tuple[str, int]:
    corpus = _load_bug_fix_corpus_cached(corpus_path)
    if not corpus:
        return "", 0
    query = (
        f"{original_prompt}\n\n"
        "Patch candidate excerpt:\n"
        f"{candidate_text[:1600]}\n\n"
        "Find likely apply/tsc/export/path/format failures."
    )
    hits = rag_retrieve(query, corpus, top_k=max(1, top_k))
    if not hits:
        return "", 0
    return rag_build_context(hits, max_chars=max(400, max_chars)), len(hits)


def run_bug_check_loop(
    *,
    backend_kind: str,
    candidate_text: str,
    original_prompt: str,
    rounds: int,
    max_tokens: int,
    system_prompt: str,
    temp: float,
    rag_enabled: bool,
    rag_corpus_path: Path,
    rag_top_k: int,
    rag_max_chars: int,
    local_backend: Optional[LocalMlxBackend] = None,
    frontier_backend: Optional[OpenAICompatibleBackend] = None,
) -> Tuple[str, bool, int, int]:
    if rounds <= 0 or not candidate_text.strip():
        return candidate_text, False, 0, 0
    current = candidate_text.strip()
    changed = False
    rounds_run = 0
    rag_hits_total = 0
    for _ in range(rounds):
        rounds_run += 1
        rag_context = ""
        if rag_enabled:
            rag_context, hits = build_bug_fix_rag_context(
                original_prompt=original_prompt,
                candidate_text=current,
                corpus_path=rag_corpus_path,
                top_k=rag_top_k,
                max_chars=rag_max_chars,
            )
            rag_hits_total += hits
        review_user = (
            "Original task prompt:\n"
            f"{original_prompt}\n\n"
            + (
                "Reference bug-fix context:\n"
                f"{rag_context}\n\n"
                if rag_context
                else ""
            )
            +
            "Candidate patch output:\n"
            f"{current}\n\n"
            "Return only the final patch output."
        )
        req = GenerationRequest(
            messages=[
                ChatMessage("system", system_prompt),
                ChatMessage("user", review_user),
            ],
            max_tokens=max_tokens,
            temperature=temp,
        )
        if backend_kind == "frontier":
            assert frontier_backend is not None
            reviewed, _ = frontier_backend.generate(req)
        else:
            assert local_backend is not None
            reviewed = local_backend.generate(req)
        reviewed_clean = sanitize_model_output(reviewed).strip()
        if not reviewed_clean:
            break
        if reviewed_clean == current:
            break
        current = reviewed_clean
        changed = True
    return current + "\n", changed, rounds_run, rag_hits_total


def resolve_local_model_id(adapter_path: str, explicit_model: Optional[str]) -> str:
    if explicit_model:
        return explicit_model
    cfg = Path(adapter_path) / "adapter_config.json"
    if cfg.is_file():
        try:
            model = json.loads(cfg.read_text(encoding="utf-8")).get("model")
            if model:
                return str(model)
        except (OSError, json.JSONDecodeError):
            pass
    return os.environ.get("MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")


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
    path = path.strip().strip('"').strip("'").strip("`")
    # Drop git diff path prefixes (``a/src/x.tsx`` / ``b/src/x.tsx``) so a path
    # salvaged from diff markers still matches the repo-relative allowed globs.
    path = re.sub(r"^[ab]/", "", path)
    return path


def _looks_like_unified_diff(text: str) -> bool:
    """True when a block body is a unified diff rather than a full file.

    Guards the fenced-file fallback: a model (or the council's diff-first lane)
    may emit a ```diff block that ``git apply`` rejects; without this check the
    fallback would write the diff text *as a file*, producing broken source full
    of ``@@``/``+``/``-`` markers. Detecting it lets us skip instead.
    """
    if text.lstrip().startswith("diff --git "):
        return True
    return bool(re.search(r"(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", text)) and bool(
        re.search(r"(?m)^\+\+\+ ", text)
    )


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


def apply_fenced_files(
    worktree: Path,
    text: str,
    allowed: List[str],
    log_path: Path,
    overwrite_guard: Optional[Callable[[str, str, str], Optional[str]]] = None,
) -> Tuple[bool, List[str]]:
    """Apply fenced full-file blocks to a worktree.

    ``overwrite_guard`` (opt-in) is called as ``guard(rel_posix, old_text, new_text)``
    before overwriting an *existing* file. If it returns a non-empty reason string,
    the write is skipped (logged as ``reject ... overwrite_guard:<reason>``). This is
    the mass-deletion / dropped-export protection used by the integration curriculum;
    when ``None`` the historical behavior (unconditional overwrite) is preserved.
    """
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
        new_text = strip_fenced_path_comment(body, rel)
        if _looks_like_unified_diff(new_text):
            log_lines.append(f"skip {rel}: body is a unified diff, not a full file")
            continue
        if overwrite_guard is not None and target.is_file():
            try:
                old_text = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                old_text = ""
            reason = overwrite_guard(rel.as_posix(), old_text, new_text)
            if reason:
                log_lines.append(f"reject {rel}: overwrite_guard:{reason}")
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text, encoding="utf-8")
        written.append(rel.as_posix())
        log_lines.append(f"wrote {rel}")
    write_text(log_path, "\n".join(log_lines) + "\n")
    return bool(written), written


def _default_overwrite_guard() -> Optional[Callable[[str, str, str], Optional[str]]]:
    """Brownfield mass-deletion / dropped-export guard for full-file overwrites.

    Active by default so a truncated rewrite of a large existing file (the
    split-016 ``apply_output`` exploit) is rejected instead of silently deleting
    thousands of lines. Small seeded stubs (< ~40 non-blank lines) are unaffected,
    so the legacy economist sandbox pipeline keeps working. Set
    ``FE_DISABLE_OVERWRITE_GUARD=1`` to restore the historical unconditional
    overwrite behavior.
    """
    if os.environ.get("FE_DISABLE_OVERWRITE_GUARD") == "1":
        return None
    try:
        from integration_apply import OverwriteGuard, make_overwrite_guard
    except Exception:
        return None
    return make_overwrite_guard(OverwriteGuard())


def apply_output(args: argparse.Namespace) -> AttemptManifest:
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    model_text = read_text(Path(args.input))
    write_text(adir / "model_output.md", model_text)
    worktree = Path(attempt.worktree_path)
    diff = extract_diff(model_text)
    log = adir / "logs" / "apply.log"
    guard_cb = _default_overwrite_guard()
    if diff:
        patch_path = adir / "model.patch"
        write_text(patch_path, diff)
        check_code, _ = run_cmd(["git", "apply", "--check", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
        if check_code == 0:
            code, _ = run_cmd(["git", "apply", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
            attempt.apply_status = "applied_diff" if code == 0 else "apply_failed"
        else:
            ok, written = apply_fenced_files(worktree, model_text, trial.task.allowed_paths, adir / "logs" / "apply_fenced_fallback.log", overwrite_guard=guard_cb)
            attempt.apply_status = f"wrote_files:{','.join(written)}" if ok else "apply_check_failed"
    else:
        ok, written = apply_fenced_files(worktree, model_text, trial.task.allowed_paths, log, overwrite_guard=guard_cb)
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
    ensure_preview_node_modules(trial, attempt)
    verify_runtime_env = {**verify_env(trial), "CI": "1"}
    results = []
    ok = True
    for idx, command in enumerate(trial.task.verify_commands):
        argv = shlex.split(command)
        code, elapsed = run_cmd(
            argv,
            cwd=worktree,
            log_path=adir / "logs" / f"verify_{idx + 1}_{slug(command, 24)}.log",
            timeout_s=args.timeout,
            env=verify_runtime_env,
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
    bug_loop_enabled = bool(getattr(args, "bug_check_loop", True))
    bug_loop_rounds = max(1, int(getattr(args, "bug_check_rounds", 1)))
    bug_loop_max_tokens = max(256, int(getattr(args, "bug_check_max_tokens", requested_max_tokens)))
    bug_loop_system = str(getattr(args, "bug_check_system_prompt", DEFAULT_BUG_CHECK_SYSTEM_PROMPT))
    bug_rag_enabled = bool(getattr(args, "bug_check_rag", True))
    bug_rag_corpus = _resolve_corpus_path(
        str(getattr(args, "bug_check_rag_corpus", str(DEFAULT_BUG_FIX_RAG_CORPUS)))
    )
    bug_rag_top_k = max(1, int(getattr(args, "bug_check_rag_top_k", 6)))
    bug_rag_max_chars = max(400, int(getattr(args, "bug_check_rag_max_chars", 2200)))
    bug_loop_changed = False
    bug_loop_rounds_run = 0
    bug_rag_hits_total = 0
    if args.backend == "local":
        local_model_id = resolve_local_model_id(args.adapter_path, getattr(args, "local_model", None))
        adapter_key = str(args.adapter_path or "").strip()
        local_cache_key = (str(local_model_id), adapter_key)
        backend = _LOCAL_BACKEND_CACHE.get(local_cache_key)
        if backend is None:
            backend = LocalMlxBackend(model_id=local_model_id, adapter_path=args.adapter_path)
            _LOCAL_BACKEND_CACHE[local_cache_key] = backend
        text = backend.generate(
            GenerationRequest(
                messages=[ChatMessage("system", "You are a careful TypeScript game engineer."), ChatMessage("user", prompt)],
                max_tokens=requested_max_tokens,
                temperature=args.temp,
            )
        )
        if bug_loop_enabled:
            text, bug_loop_changed, bug_loop_rounds_run, bug_rag_hits_total = run_bug_check_loop(
                backend_kind="local",
                candidate_text=text,
                original_prompt=prompt,
                rounds=bug_loop_rounds,
                max_tokens=bug_loop_max_tokens,
                system_prompt=bug_loop_system,
                temp=args.temp,
                rag_enabled=bug_rag_enabled,
                rag_corpus_path=bug_rag_corpus,
                rag_top_k=bug_rag_top_k,
                rag_max_chars=bug_rag_max_chars,
                local_backend=backend,
            )
    elif args.backend == "frontier":
        frontier_model_key = str(args.model or "")
        frontier_base_url_key = str(
            os.environ.get("FRONTIER_API_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        frontier_cache_key = (frontier_model_key, frontier_base_url_key)
        backend = _FRONTIER_BACKEND_CACHE.get(frontier_cache_key)
        if backend is None:
            backend = OpenAICompatibleBackend(model=args.model)
            _FRONTIER_BACKEND_CACHE[frontier_cache_key] = backend
        text, usage = backend.generate(
            GenerationRequest(
                messages=[ChatMessage("system", "You are a careful TypeScript game engineer."), ChatMessage("user", prompt)],
                max_tokens=requested_max_tokens,
                temperature=args.temp,
            )
        )
        if bug_loop_enabled:
            text, bug_loop_changed, bug_loop_rounds_run, bug_rag_hits_total = run_bug_check_loop(
                backend_kind="frontier",
                candidate_text=text,
                original_prompt=prompt,
                rounds=bug_loop_rounds,
                max_tokens=bug_loop_max_tokens,
                system_prompt=bug_loop_system,
                temp=args.temp,
                rag_enabled=bug_rag_enabled,
                rag_corpus_path=bug_rag_corpus,
                rag_top_k=bug_rag_top_k,
                rag_max_chars=bug_rag_max_chars,
                frontier_backend=backend,
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
        "bug_check_loop_enabled": bug_loop_enabled,
        "bug_check_rounds_requested": bug_loop_rounds if bug_loop_enabled else 0,
        "bug_check_rounds_run": bug_loop_rounds_run,
        "bug_check_changed_output": bug_loop_changed,
        "bug_check_rag_enabled": bool(bug_loop_enabled and bug_rag_enabled),
        "bug_check_rag_corpus": str(bug_rag_corpus),
        "bug_check_rag_top_k": bug_rag_top_k if bug_rag_enabled else 0,
        "bug_check_rag_max_chars": bug_rag_max_chars if bug_rag_enabled else 0,
        "bug_check_rag_hits_total": bug_rag_hits_total,
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


def _diff_changed_paths(diff_text: str) -> List[str]:
    """Best-effort list of repo-relative paths touched by a unified diff."""
    paths: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[len("+++ b/"):].strip())
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            paths.append(line[len("+++ "):].strip())
    return [p for p in paths if p and p != "/dev/null"]


def _apply_text_to_worktree(
    worktree: Path,
    model_text: str,
    allowed: List[str],
    log: Path,
) -> Tuple[str, List[str]]:
    """Apply one model output (diff-first, fenced fallback) into a worktree.

    Mirrors ``apply_output`` but works on raw text + an explicit allowed list so
    the council orchestrator can stitch each subtask's change onto the shared
    worktree using the exact same brownfield guards as the local/frontier lanes.
    Returns ``(status, changed_files)``.
    """
    diff = extract_diff(model_text)
    guard_cb = _default_overwrite_guard()
    if diff:
        patch_path = log.parent / f"{log.stem}.patch"
        write_text(patch_path, diff)
        check_code, _ = run_cmd(["git", "apply", "--check", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
        if check_code == 0:
            code, _ = run_cmd(["git", "apply", str(patch_path)], cwd=worktree, log_path=log, timeout_s=120)
            if code == 0:
                return "applied_diff", _diff_changed_paths(diff)
        ok, written = apply_fenced_files(
            worktree, model_text, allowed, log.parent / f"{log.stem}_fenced.log", overwrite_guard=guard_cb
        )
        return ("wrote_files" if ok else "apply_failed"), written
    ok, written = apply_fenced_files(worktree, model_text, allowed, log, overwrite_guard=guard_cb)
    return ("wrote_files" if ok else "no_applyable_changes"), written


def _refit_plan_to_task(plan: Any, allowed_paths: List[str], *, max_experts: int = 2) -> Any:
    """Re-fence a planner-emitted plan onto the arena task's allowed paths.

    The planner decides *whether/how* to decompose and *which* specialists to call,
    but the arena owns the write fence (``allowed_paths``) and defers compile/test
    gating to its own preview/verify step, so each subtask's ``verify_commands`` is
    cleared (the council verify callback passes once a change applies).
    """
    from council_runtime.decomposition_plan import DecompositionPlan, SubtaskNode

    fence = list(allowed_paths) or ["**/*"]
    new_subtasks = []
    for node in plan.subtasks:
        new_subtasks.append(
            SubtaskNode(
                subtask_id=node.subtask_id,
                domain=node.domain,
                goal=node.goal,
                assigned_question=node.assigned_question,
                allowed_paths=fence,
                target_files=list(node.target_files),
                depends_on=list(node.depends_on),
                handoff_context=node.handoff_context,
                verify_commands=[],
                selected_experts=list(node.selected_experts)[:max_experts],
            )
        )
    return DecompositionPlan(
        task_synopsis=plan.task_synopsis,
        primary_goal=plan.primary_goal,
        success_criteria=list(plan.success_criteria),
        subtasks=new_subtasks,
        integration_plan=dict(plan.integration_plan),
        loop_policy=dict(plan.loop_policy),
    )


def generate_council_attempt(args: argparse.Namespace) -> Path:
    """Council lane: the planner decomposes the task and dispatches each subtask to
    its own specialist council; every subtask's change is stitched onto the shared
    arena worktree with the same diff/fenced apply + overwrite guards as the
    local/frontier lanes. Generation and apply happen together here (the loop
    applies as it goes), so the UI skips the separate ``apply_output`` step for
    council and relies on the arena's own preview/verify for the real gate.
    """
    load_dotenv()
    trial = load_trial(args.trial_id)
    attempt = trial.attempts[args.attempt]
    adir = attempt_dir(trial.trial_id, attempt.attempt)
    log_root = adir / "logs"
    worktree = Path(attempt.worktree_path)

    from council_runtime.executor import CouncilGeneration, run_council
    from council_runtime.orchestrator import (
        SubtaskApply,
        SubtaskDispatch,
        SubtaskVerify,
        run_decomposed_task,
    )
    from council_runtime.planner_decomposition_policy import (
        generate_decomposition_plan,
        make_council_plan_for_subtask,
    )
    from council_runtime.decomposition_plan import derive_decomposition_plan, validate_decomposition_plan

    use_bm25_ctx = not getattr(args, "no_context_bm25", False)
    context = build_context_pack(
        trial,
        max_chars=int(getattr(args, "context_chars", 9000)),
        log_dir=log_root,
        use_bm25=use_bm25_ctx,
    ).text
    requested_max_tokens = int(args.max_tokens or trial.task.max_tokens or 4096)
    temp = float(getattr(args, "temp", 0.0))
    council_mode = str(getattr(args, "council_mode", "planner-model"))
    council_rounds = max(1, int(getattr(args, "council_rounds", 1)))
    council_max_subtasks = max(1, int(getattr(args, "council_max_subtasks", 4)))

    local_model_id = resolve_local_model_id(args.adapter_path, getattr(args, "local_model", None))
    adapter_key = str(args.adapter_path or "").strip()
    local_cache_key = (str(local_model_id), adapter_key)
    backend = _LOCAL_BACKEND_CACHE.get(local_cache_key)
    if backend is None:
        backend = LocalMlxBackend(model_id=local_model_id, adapter_path=args.adapter_path)
        _LOCAL_BACKEND_CACHE[local_cache_key] = backend

    format_instruction = (
        "Return only fenced full-file blocks with repo-relative paths, or one unified diff "
        "that applies with git apply. Do not include shell commands, commentary, or summaries. "
        "Preserve existing exported component/function names unless the task explicitly asks to "
        "rename them. Stay within the allowed paths."
    )

    def generate_participant(turn: Any) -> Any:
        user = f"{format_instruction}\n\n{context}\n\n{turn.participant_prompt}"
        text = backend.generate(
            GenerationRequest(
                messages=[
                    ChatMessage("system", "You are a careful TypeScript game engineer on a focused subtask."),
                    ChatMessage("user", user),
                ],
                max_tokens=requested_max_tokens,
                temperature=temp,
            )
        )
        return CouncilGeneration(text=sanitize_model_output(text), metadata={"backend": "local"})

    council_plan_for_subtask = make_council_plan_for_subtask(debate_max_rounds=council_rounds)
    council_dispatches: List[dict] = []

    def dispatch(ctx: Any) -> Any:
        subtask_prompt = (
            f"{ctx.subtask.goal}\n\n{ctx.subtask.assigned_question}\n\n"
            f"Upstream changes already applied to the shared worktree (build on these, "
            f"do not redo them):\n{ctx.upstream_digest() or '- none'}\n\n"
            f"Edit only these paths: {', '.join(ctx.subtask.allowed_paths) or '(unspecified)'}.\n"
            f"{format_instruction}"
        )
        final_text, meta = run_council(
            prompt=subtask_prompt,
            council_plan=council_plan_for_subtask(ctx),
            disagreement=0.0,
            generate_participant=generate_participant,
            max_rounds=council_rounds,
        )
        council_dispatches.append(
            {
                "subtask_id": ctx.subtask.subtask_id,
                "domain": ctx.subtask.domain,
                "iteration": int(ctx.iteration),
                "attempt": int(ctx.attempt),
                "selected_experts": [str(e.get("expert_id")) for e in ctx.subtask.selected_experts],
                "subtask_prompt": subtask_prompt,
                "rounds": meta.get("rounds") or [],
                "final_text": final_text,
            }
        )
        return SubtaskDispatch(
            output=final_text,
            metadata={k: v for k, v in meta.items() if k != "participants"},
            trace_ids=list(meta.get("trace_ids") or []),
        )

    def apply_cb(node: Any, output: str, wt: Any) -> Any:
        log = log_root / f"council_apply_{slug(node.subtask_id, 24)}.log"
        status, changed = _apply_text_to_worktree(Path(wt), output, list(node.allowed_paths), log)
        ok = status.startswith(("applied", "wrote"))
        applied_via = "diff" if status.startswith("applied") else ("overwrite" if ok else "none")
        return SubtaskApply(ok=ok, applied_via=applied_via, changed_files=changed, detail=status)

    def verify_cb(node: Any, wt: Any) -> Any:
        # The arena owns the real gate (preview + fixed verification). A subtask is
        # "done" for the loop once its change applies, so dependents can build on it.
        return SubtaskVerify(compiled=True, tests_score=1.0, detail="arena_deferred_verify")

    started = time.perf_counter()
    plan_raw_text = ""
    if council_mode == "single":
        plan = derive_decomposition_plan(prompt=trial.task.prompt, allowed_paths=list(trial.task.allowed_paths))
        plan = _refit_plan_to_task(plan, trial.task.allowed_paths)
        plan_source = "single"
        plan_parse_ok = True
    else:
        emission = generate_decomposition_plan(
            trial.task.prompt,
            backend=(None if council_mode == "heuristic" else backend),
            mock=(council_mode == "heuristic"),
            max_subtasks=council_max_subtasks,
        )
        plan = _refit_plan_to_task(emission.plan, trial.task.allowed_paths)
        plan_source = emission.source
        plan_parse_ok = emission.parse_ok
        # Preserve the planner model's *actual* completion (esp. when it failed to
        # parse and we fell back to the heuristic) so the trace shows what it said.
        if emission.source in ("model", "model_fallback"):
            plan_raw_text = emission.raw_text

    validation = validate_decomposition_plan(plan.to_dict())
    write_text(adir / "council_plan.json", json.dumps(plan.to_dict(), indent=2) + "\n")

    result = run_decomposed_task(
        prompt=trial.task.prompt,
        plan=plan,
        dispatch_subtask=dispatch,
        apply_subtask=apply_cb,
        verify_subtask=verify_cb,
        worktree=worktree,
        # Retry a flaky subtask once so a single bad apply doesn't zero out the
        # whole council; widen the iteration budget so a retry can still
        # propagate down a dependency chain (worst case ~2 passes per subtask).
        max_attempts_per_subtask=2,
        max_iterations=max(2, len(plan.subtasks) * 2),
    )
    elapsed = time.perf_counter() - started

    applied_records = [rec for rec in result.history if rec.apply.ok]
    changed_files: List[str] = []
    for rec in applied_records:
        for path in rec.apply.changed_files:
            if path not in changed_files:
                changed_files.append(path)

    report_lines = [
        f"# Council attempt — {len(plan.subtasks)} subtask(s) ({plan_source})",
        "",
        f"- Decomposition source: `{plan_source}` (parse_ok={plan_parse_ok}, valid={validation.valid})",
        f"- Subtasks applied: {len(applied_records)}/{len(plan.subtasks)}",
        f"- Changed files: {', '.join(f'`{p}`' for p in changed_files) or 'none'}",
        "",
    ]
    for rec in result.history:
        node = plan.subtask_by_id(rec.subtask_id)
        experts = ", ".join(str(e.get('expert_id')) for e in (node.selected_experts if node else [])) or "generalist"
        report_lines.append(
            f"## subtask `{rec.subtask_id}` — domain `{rec.domain}` — experts: {experts}"
        )
        report_lines.append(f"_apply: {rec.apply.applied_via} ({rec.apply.detail})_")
        report_lines.append("")
        report_lines.append(rec.dispatch.output or "_(no output)_")
        report_lines.append("")
    out_text = "\n".join(report_lines)

    write_text(adir / "council_result.json", json.dumps(result.to_dict(), indent=2) + "\n")
    trace_doc = {
        "trial_id": trial.trial_id,
        "task_id": trial.task.id,
        "task_title": trial.task.title,
        "prompt": trial.task.prompt,
        "council_mode": council_mode,
        "council_rounds": council_rounds,
        "elapsed_s": elapsed,
        "apply_status": (
            f"applied_council:{len(applied_records)}/{len(plan.subtasks)}"
            if applied_records
            else "no_applyable_changes"
        ),
        "changed_files": changed_files,
        "decomposition": {
            "source": plan_source,
            "parse_ok": plan_parse_ok,
            "valid": validation.valid,
            "plan": plan.to_dict(),
            "raw_planner_output": plan_raw_text,
        },
        "dispatches": council_dispatches,
        "result": result.to_dict(),
    }
    write_text(adir / "council_trace.json", json.dumps(trace_doc, indent=2) + "\n")
    if applied_records:
        attempt.apply_status = f"applied_council:{len(applied_records)}/{len(plan.subtasks)}"
    else:
        attempt.apply_status = "no_applyable_changes"
    attempt.last_error = "" if applied_records else "council produced no applyable changes"
    attempt.generation_elapsed_s = elapsed
    metrics = {
        "trial_id": trial.trial_id,
        "attempt": attempt.attempt,
        "backend": "council_mlx",
        "model_label": attempt.model_label,
        "started_at": utc_now(),
        "elapsed_s": elapsed,
        "council_mode": council_mode,
        "council_rounds": council_rounds,
        "decomposition_source": plan_source,
        "decomposition_parse_ok": plan_parse_ok,
        "subtask_count": len(plan.subtasks),
        "subtasks_applied": len(applied_records),
        "changed_files": changed_files,
        "subtask_pass_rate": result.subtask_pass_rate,
    }
    trial.attempts[attempt.attempt] = attempt
    save_trial(trial)
    write_attempt_manifest(trial.trial_id, attempt)
    write_text(adir / "generation_metrics.json", json.dumps(metrics, indent=2) + "\n")
    append_index(trial, attempt, event="generated")
    out = adir / "model_output.md"
    write_text(out, out_text)
    write_diff_artifacts(trial, attempt)
    return out


def _indent_block(text: str, pad: str = "        ") -> str:
    text = str(text or "").strip()
    if not text:
        return f"{pad}(empty)"
    return "\n".join(pad + line for line in text.splitlines())


def render_council_trace_text(trial_id: str) -> str:
    """Human-readable transcript of a council run: planner decomposition, each
    subtask's specialist debate (per-round participant outputs + adjudication),
    the chosen change, and the apply outcome. Returned as plain text so model
    output containing ``` fences renders verbatim (no markdown nesting issues).
    """
    trial_id = (trial_id or "").strip()
    if not trial_id:
        return "Create/run a trial first."
    path = attempt_dir(trial_id, "council") / "council_trace.json"
    if not path.is_file():
        return (
            f"No council trace found for trial `{trial_id}`.\n"
            "Run a trial with the Council lane enabled, then click this button."
        )
    try:
        doc = json.loads(read_text(path))
    except (json.JSONDecodeError, OSError) as exc:
        return f"Could not read council trace: {type(exc).__name__}: {exc}"

    dec = doc.get("decomposition", {})
    plan = dec.get("plan", {})
    subtasks = plan.get("subtasks", [])
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"COUNCIL PROCESS — task: {doc.get('task_id')}  ({doc.get('task_title','')})")
    lines.append("=" * 78)
    lines.append("")
    lines.append("PROMPT:")
    lines.append(_indent_block(doc.get("prompt", ""), "    "))
    lines.append("")
    lines.append(
        f"DECOMPOSITION: source={dec.get('source')}  parse_ok={dec.get('parse_ok')}  "
        f"valid={dec.get('valid')}  subtasks={len(subtasks)}  mode={doc.get('council_mode')}  "
        f"rounds={doc.get('council_rounds')}"
    )
    lines.append(
        f"OUTCOME: apply_status={doc.get('apply_status')}  "
        f"changed_files={doc.get('changed_files') or 'none'}  elapsed={doc.get('elapsed_s', 0):.1f}s"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("PLAN (what the planner decided)")
    lines.append("-" * 78)
    for s in subtasks:
        experts = ", ".join(str(e.get("expert_id")) for e in s.get("selected_experts", [])) or "generalist"
        deps = ", ".join(s.get("depends_on", [])) or "none"
        lines.append(f"[{s.get('subtask_id')}] domain={s.get('domain')}  experts=[{experts}]  depends_on=[{deps}]")
        lines.append(f"    goal: {s.get('goal','')}")
        lines.append(f"    edit fence: {', '.join(s.get('allowed_paths', [])) or '(unspecified)'}")
    lines.append("")

    raw_planner = dec.get("raw_planner_output", "")
    if dec.get("source") == "model_fallback":
        lines.append("-" * 78)
        lines.append("PLANNER FALLBACK — the model plan above is the HEURISTIC, not the model")
        lines.append("-" * 78)
        lines.append(
            "The planner model was invoked but its output did not parse as a valid plan,"
            " so the deterministic heuristic was substituted. Raw model completion below:"
        )
        lines.append("")
        lines.append(_indent_block(raw_planner or "(empty)", "    "))
        lines.append("")
    elif dec.get("source") == "model" and raw_planner:
        lines.append("-" * 78)
        lines.append("PLANNER RAW OUTPUT (model completion that produced the plan)")
        lines.append("-" * 78)
        lines.append(_indent_block(raw_planner, "    "))
        lines.append("")

    history = {
        (h.get("subtask_id"), h.get("attempt")): h
        for h in doc.get("result", {}).get("history", [])
    }
    lines.append("-" * 78)
    lines.append("SUBTASK COUNCILS (the debate)")
    lines.append("-" * 78)
    dispatches = doc.get("dispatches", [])
    if not dispatches:
        lines.append("(no subtask councils ran)")
    for d in dispatches:
        sid, att = d.get("subtask_id"), d.get("attempt")
        h = history.get((sid, att), {})
        applied = (
            f"applied_via={h.get('applied_via')} ok={h.get('apply_ok')} changed={h.get('changed_files')}"
            if h
            else "n/a"
        )
        experts = ", ".join(d.get("selected_experts", [])) or "generalist"
        lines.append("")
        lines.append(
            f"### Subtask `{sid}`  domain={d.get('domain')}  (iteration {d.get('iteration')}, attempt {att})"
        )
        lines.append(f"    experts convened: {experts}")
        lines.append(f"    apply result: {applied}")
        for rnd in d.get("rounds", []):
            lines.append(f"  -- Round {rnd.get('round_idx')} --")
            for p in rnd.get("participants", []):
                who = p.get("base_expert_id") or p.get("participant_id")
                lines.append(f"    >> {who}  (role={p.get('role')}, confidence={p.get('confidence')})")
                lines.append(_indent_block(p.get("text", "")))
                lines.append("")
            adj = rnd.get("adjudication_preview", {})
            lines.append(
                f"    [adjudication] winners={adj.get('winner_ids')}  "
                f"confidence={adj.get('confidence')}  disagreement={adj.get('disagreement')}"
            )
        lines.append("    === CHOSEN OUTPUT for this subtask ===")
        lines.append(_indent_block(d.get("final_text", "")))
    lines.append("")
    return "\n".join(lines)


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


def validation_artifact_paths(limit: int = 40) -> List[Path]:
    return sorted(
        (REPO / "benchmarks" / "results").glob("multi_agent_orchestration_validation*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]


def summarize_validation_artifact(path: Path) -> Tuple[str, str]:
    if not path.is_file():
        return f"Validation artifact not found: `{path}`", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in `{path}`: `{exc}`", ""
    rel = path.relative_to(REPO).as_posix()
    lines = [
        f"### Validation: `{rel}`",
        "",
        f"- Rows: `{payload.get('rows', 0)}`",
        f"- Split valid: `{payload.get('split_valid_rows', 0)}` (`{payload.get('split_valid_rate', 0):.3f}`)",
        f"- Merge valid: `{payload.get('merge_valid_rows', 0)}` (`{payload.get('merge_valid_rate', 0):.3f}`)",
        f"- Multi-agent rows: `{payload.get('multi_agent_rows', 0)}`",
    ]
    if "expected_multi_agent_rows" in payload:
        lines.append(f"- Expected multi-agent rows: `{payload.get('expected_multi_agent_rows', 0)}`")
    if "expected_multi_agent_hit_rate" in payload:
        lines.append(f"- Expected multi-agent hit rate: `{payload.get('expected_multi_agent_hit_rate', 0):.3f}`")
    if "unexpected_multi_agent_rate" in payload:
        lines.append(f"- Unexpected multi-agent rate: `{payload.get('unexpected_multi_agent_rate', 0):.3f}`")
    if "split_unique_adapter_rows" in payload:
        lines.append(f"- Split unique-adapter rows: `{payload.get('split_unique_adapter_rows', 0)}`")
    if "split_nonempty_rows" in payload:
        lines.append(f"- Split non-empty rows: `{payload.get('split_nonempty_rows', 0)}`")
    if "split_priority_sorted_rows" in payload:
        lines.append(f"- Split priority-sorted rows: `{payload.get('split_priority_sorted_rows', 0)}`")
    return "\n".join(lines), json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Model Chat: agentic loop + judge voting
#
# Pure/testable helpers live at module scope (no Gradio dependency) so the
# stop-signal detection, verdict parsing, and vote tally can be unit tested
# without spinning up a model backend. Gradio-facing wiring stays nested in
# `build_app()` alongside the rest of the Model Chat callbacks.
# --------------------------------------------------------------------------- #
DEBATE_STOP_TOKEN = "[[DEBATE_CONCLUDED]]"

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge for a debate between AI models in the Fallen Empire arena. "
    "Read the full transcript and decide which named participant argued their position most "
    "convincingly overall, considering reasoning quality, evidence, and how well they addressed "
    "the other side's points. Do not favor a participant merely for speaking last. "
    "Respond in exactly this format:\n"
    "WINNER: <one of the candidate names, or TIE>\n"
    "REASONING: <2-4 sentences justifying the verdict>"
)

_JUDGE_WINNER_RE = re.compile(r"WINNER\s*:\s*(.+)", re.IGNORECASE)
_JUDGE_REASONING_RE = re.compile(r"REASONING\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _is_judge_speaker(speaker: Dict[str, Any]) -> bool:
    return bool(speaker.get("is_judge"))


def _debater_names(room_state: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [s.get("name") for s in room_state or [] if s.get("name") and not _is_judge_speaker(s)]


def _judge_names(room_state: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [s.get("name") for s in room_state or [] if s.get("name") and _is_judge_speaker(s)]


def _debate_signaled_stop(text: str) -> bool:
    return DEBATE_STOP_TOKEN.lower() in (text or "").lower()


def _strip_stop_token(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(re.escape(DEBATE_STOP_TOKEN), re.IGNORECASE)
    return pattern.sub("", text).strip()


def _transcript_speaker_names(chat_state: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Distinct non-human speaker names that have actually spoken, in first-seen order."""
    seen: List[str] = []
    for item in chat_state or []:
        name = item.get("speaker_name") or item.get("speaker")
        if name and name != "Human" and name not in seen:
            seen.append(name)
    return seen


def _chat_transcript_by_name(chat_state: Optional[List[Dict[str, Any]]], max_chars: int = 12000) -> str:
    """Like `_chat_transcript` but keyed by bare speaker name (legible for judges)."""
    lines = []
    for item in chat_state or []:
        name = item.get("speaker_name") or item.get("speaker", "speaker")
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"{name}: {content}")
    text = "\n\n".join(lines)
    if len(text) <= max_chars:
        return text
    return "[earlier transcript clipped]\n\n" + text[-max_chars:]


def _generate_with_speaker_backend(speaker: Dict[str, Any], req: "GenerationRequest") -> Tuple[str, Optional[int]]:
    """Route a GenerationRequest to a speaker's configured backend (Local or Frontier).

    Backends are cached at module scope (`_LOCAL_BACKEND_CACHE` / `_FRONTIER_BACKEND_CACHE`)
    so repeated turns/judgments reuse the same loaded model or client. Returns
    (sanitized_text, total_tokens_or_None).
    """
    if str(speaker.get("backend") or "Local") == "Frontier":
        model_key = str(speaker.get("frontier_model") or os.environ.get("FRONTIER_MODEL") or "")
        base_url_key = str(
            os.environ.get("FRONTIER_API_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        cache_key = (model_key, base_url_key)
        backend = _FRONTIER_BACKEND_CACHE.get(cache_key)
        if backend is None:
            backend = OpenAICompatibleBackend(model=model_key or None)
            _FRONTIER_BACKEND_CACHE[cache_key] = backend
        text, usage = backend.generate(req)
        return sanitize_model_output(text).strip(), usage.get("total_tokens")

    adapter = str(speaker.get("local_adapter") or "checkpoints/fe-lora-30m").strip()
    model_id = resolve_local_model_id(adapter, os.environ.get("MODEL"))
    cache_key = (str(model_id), adapter)
    backend = _LOCAL_BACKEND_CACHE.get(cache_key)
    if backend is None:
        backend = LocalMlxBackend(model_id=model_id, adapter_path=adapter)
        _LOCAL_BACKEND_CACHE[cache_key] = backend
    text = backend.generate(req)
    return sanitize_model_output(text).strip(), None


def _judge_generate(
    judge: Dict[str, Any],
    transcript: str,
    candidate_names: List[str],
    max_tokens: int,
    temp: float,
) -> str:
    user = (
        "Candidates: " + ", ".join(candidate_names) + "\n\n"
        "Transcript:\n" + (transcript or "(empty)") + "\n\n"
        "Give your verdict now in the required format."
    )
    profile = str(judge.get("profile") or "").strip()
    system = JUDGE_SYSTEM_PROMPT + (f"\n\nJudge profile:\n{profile}" if profile else "")
    req = GenerationRequest(
        messages=[ChatMessage("system", system), ChatMessage("user", user)],
        max_tokens=max(64, int(max_tokens or 512)),
        temperature=float(temp or 0.0),
    )
    text, _total_tokens = _generate_with_speaker_backend(judge, req)
    return text


def _parse_judge_verdict(raw_text: str, candidate_names: List[str]) -> Dict[str, Any]:
    """Parse a judge's free-text verdict into a structured winner + reasoning.

    Matches `candidate_names` case-insensitively and tolerates the judge repeating
    extra text on the winner line (e.g. "WINNER: Alice (Local)").
    """
    text = raw_text or ""
    winner_match = _JUDGE_WINNER_RE.search(text)
    reasoning_match = _JUDGE_REASONING_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    if not winner_match:
        return {"winner": None, "reasoning": reasoning, "raw": text, "parse_ok": False}
    lowered = winner_match.group(1).strip().lower()
    winner = None
    for name in candidate_names:
        if name.lower() in lowered or lowered in name.lower():
            winner = name
            break
    if winner is None and ("tie" in lowered or "both" in lowered or "draw" in lowered):
        return {"winner": "tie", "reasoning": reasoning, "raw": text, "parse_ok": True}
    return {"winner": winner, "reasoning": reasoning, "raw": text, "parse_ok": winner is not None}


def _tally_judge_votes(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    unparsed = 0
    for verdict in verdicts:
        winner = verdict.get("winner")
        if not winner:
            unparsed += 1
            continue
        counts[winner] = counts.get(winner, 0) + 1
    if not counts:
        return {"counts": {}, "winner": None, "tie": False, "leaders": [], "unparsed": unparsed}
    top = max(counts.values())
    leaders = [name for name, count in counts.items() if count == top]
    is_tie = len(leaders) > 1
    return {
        "counts": counts,
        "winner": None if is_tie else leaders[0],
        "tie": is_tie,
        "leaders": leaders,
        "unparsed": unparsed,
    }


def _render_judge_scoreboard(verdicts: List[Dict[str, Any]], tally: Dict[str, Any], candidates: List[str]) -> str:
    lines = ["## Judge scoreboard", ""]
    counts = tally.get("counts") or {}
    if candidates:
        lines.append(" · ".join(f"**{name}**: {counts.get(name, 0)} vote(s)" for name in candidates))
    if tally.get("tie"):
        lines.append(f"\n**Result: TIE** between {', '.join(tally.get('leaders') or [])}.")
    elif tally.get("winner"):
        lines.append(f"\n**Result: {tally['winner']} wins** ({counts.get(tally['winner'], 0)}/{len(verdicts)} votes).")
    else:
        lines.append("\n**Result: no clear verdict** (judges disagreed or could not be parsed).")
    if tally.get("unparsed"):
        lines.append(f"_{tally['unparsed']} judge verdict(s) could not be parsed cleanly — shown below as raw text._")
    lines.append("")
    for verdict in verdicts:
        pick = verdict.get("winner") or "(unparsed)"
        detail = verdict.get("reasoning") or verdict.get("raw") or ""
        lines.append(f"- **{verdict.get('judge_label') or verdict.get('judge')}** → `{pick}` — {detail}")
    return "\n".join(lines)


def _build_debate_record(
    *,
    room_state: List[Dict[str, Any]],
    chat_state: List[Dict[str, Any]],
    judge_names: List[str],
    verdicts: List[Dict[str, Any]],
    tally: Dict[str, Any],
) -> Dict[str, Any]:
    debaters = _transcript_speaker_names(chat_state)[:2]
    label = "_vs_".join(debaters) if debaters else "debate"
    return {
        "debate_id": f"model_chat_{slug(label)}_{uuid.uuid4().hex[:8]}",
        "created_at": utc_now(),
        "speakers": [
            {"name": s.get("name"), "backend": s.get("backend"), "is_judge": bool(s.get("is_judge"))}
            for s in room_state or []
        ],
        "transcript": [
            {
                "speaker": item.get("speaker"),
                "speaker_name": item.get("speaker_name") or item.get("speaker"),
                "content": item.get("content"),
            }
            for item in chat_state or []
        ],
        "judges": judge_names,
        "verdicts": verdicts,
        "vote_tally": tally,
        "winner": tally.get("winner"),
        "tie": bool(tally.get("tie")),
    }


def build_app():
    import gradio as gr
    theme_css = """
:root {
  --arena-bg: #050506;
  --arena-panel: rgba(16, 18, 22, 0.94);
  --arena-panel-solid: #15181d;
  --arena-panel-muted: #1c2027;
  --arena-border: rgba(228, 219, 196, 0.16);
  --arena-border-strong: rgba(228, 219, 196, 0.30);
  --arena-text: #f6f2e9;
  --arena-muted: #b9b2a4;
  --arena-input: #0b0d10;
  --arena-code: #212018;
  --arena-accent: #d6ad4b;
  --arena-accent-strong: #f0c96a;
  --arena-cyan: #5ec4bd;
}
.gradio-container,
body {
  background: var(--arena-bg) !important;
}
.gradio-container {
  background:
    radial-gradient(circle at 18% 0%, rgba(214, 173, 75, 0.13), transparent 28rem),
    linear-gradient(180deg, #090a0c 0%, #050506 52%, #030304 100%) !important;
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
  border-radius: 10px !important;
  border-color: var(--arena-border) !important;
  background: var(--arena-panel) !important;
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28) !important;
}
.gradio-container .tabs {
  padding: 0.35rem !important;
}
.gradio-container .tab-nav button,
.gradio-container [role="tab"] {
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}
.gradio-container [role="tab"][aria-selected="true"] {
  background: #f6f2e9 !important;
  color: #111111 !important;
  border-color: #f6f2e9 !important;
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
  color: var(--arena-cyan) !important;
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
  border-radius: 6px;
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
  border: 1px solid var(--arena-border-strong) !important;
  color: var(--arena-text) !important;
}
.gradio-container button {
  min-height: 42px !important;
  padding: 0.55rem 1rem !important;
  border-radius: 8px !important;
  font-weight: 800 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  transition: border-color 140ms ease, background 140ms ease, transform 140ms ease, box-shadow 140ms ease !important;
}
.gradio-container button:hover {
  border-color: var(--arena-accent) !important;
  transform: translateY(-1px);
}
.gradio-container button.primary,
.gradio-container button.primary:hover {
  background: linear-gradient(180deg, var(--arena-accent-strong), var(--arena-accent)) !important;
  border: 1px solid rgba(255, 236, 178, 0.72) !important;
  color: #17130a !important;
  box-shadow: 0 12px 28px rgba(214, 173, 75, 0.22) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  min-height: 42px !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--arena-accent) !important;
  box-shadow: 0 0 0 3px rgba(214, 173, 75, 0.12) !important;
}
.gradio-container .message,
.gradio-container .chatbot {
  border-radius: 10px !important;
  border-color: var(--arena-border) !important;
  background: rgba(10, 11, 13, 0.96) !important;
}
"""

    def create_ui(task_id, source_repo, worktree_root, base_ref, include_council=True):
        attempts = ["local", "frontier"]
        if include_council:
            attempts.append("council")
        ns = argparse.Namespace(
            task_id=task_id,
            tasks=DEFAULT_TASKS,
            source_repo=source_repo or str(DEFAULT_SOURCE_REPO),
            worktree_root=worktree_root or str(DEFAULT_WORKTREE_ROOT),
            base_ref=base_ref or "HEAD",
            trial_id=None,
            attempts=attempts,
            local_adapter="checkpoints/fe-lora-30m",
            frontier_model=os.environ.get("FRONTIER_MODEL", "frontier"),
            copy=False,
        )
        trial = create_trial(ns)
        lanes = ", ".join(f"`{a}`" for a in attempts)
        summary = (
            f"Created trial `{trial.trial_id}` from standardized task `{trial.task.id}` "
            f"with lanes: {lanes}.\n\n"
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
        council_port,
        start_servers,
        include_council,
        council_mode,
        council_rounds,
    ):
        trial_id, create_summary = create_ui(task_id, source_repo, worktree_root, base_ref, include_council)
        generate_status = generate_apply_both_ui(
            trial_id, local_adapter, frontier_model, max_tokens, council_mode, council_rounds
        )
        preview_status, local_link, frontier_link, council_link = preview_both_ui(
            trial_id,
            local_port,
            frontier_port,
            council_port,
            start_servers,
        )
        status = (
            f"{create_summary}\n\n"
            f"### Generation\n{generate_status}\n\n"
            f"### Preview\n{preview_status}"
        )
        return trial_id, status, local_link, frontier_link, council_link

    def generate_apply_both_ui(
        trial_id, local_adapter, frontier_model, max_tokens, council_mode="planner-model", council_rounds=1
    ):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first."
        statuses = []
        present = set(load_trial(trial_id).attempts.keys())
        for attempt_name, backend in [("local", "local"), ("frontier", "frontier")]:
            if attempt_name not in present:
                continue
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
                        bug_check_loop=True,
                        bug_check_rounds=1,
                        bug_check_max_tokens=int(max_tokens or 4096),
                        bug_check_system_prompt=DEFAULT_BUG_CHECK_SYSTEM_PROMPT,
                        bug_check_rag=True,
                        bug_check_rag_corpus=str(DEFAULT_BUG_FIX_RAG_CORPUS),
                        bug_check_rag_top_k=6,
                        bug_check_rag_max_chars=2200,
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
        if "council" in present:
            try:
                generate_council_attempt(
                    argparse.Namespace(
                        trial_id=trial_id,
                        attempt="council",
                        adapter_path=local_adapter or "checkpoints/fe-lora-30m",
                        local_model=os.environ.get("MODEL"),
                        max_tokens=int(max_tokens or 4096),
                        temp=0.0,
                        context_chars=9000,
                        no_context_bm25=False,
                        council_mode=str(council_mode or "planner-model"),
                        council_rounds=int(council_rounds or 1),
                        council_max_subtasks=4,
                    )
                )
                manifest = load_trial(trial_id).attempts["council"]
                elapsed = manifest.generation_elapsed_s or 0.0
                statuses.append(
                    f"- `council` decomposed + generated in {elapsed:.1f}s; apply status is `{manifest.apply_status}`"
                )
            except Exception as exc:
                statuses.append(f"- `council` failed: `{type(exc).__name__}: {exc}`")
        try:
            signal_path = write_pairwise_training_signal(load_trial(trial_id))
            if signal_path:
                statuses.append(f"- pairwise training signal saved: `{signal_path}`")
        except Exception as exc:
            statuses.append(f"- pairwise training signal failed: `{type(exc).__name__}: {exc}`")
        return "\n".join(statuses)

    def preview_both_ui(trial_id, local_port, frontier_port, council_port, start_servers):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first.", "", "", ""
        present = set(load_trial(trial_id).attempts.keys())

        def _preview_lane(name, port):
            if name not in present:
                return None
            return preview(
                argparse.Namespace(
                    trial_id=trial_id,
                    attempt=name,
                    port=int(port),
                    command=None,
                    start=bool(start_servers),
                )
            )

        local = _preview_lane("local", local_port or 5174)
        frontier = _preview_lane("frontier", frontier_port or 5175)
        council = _preview_lane("council", council_port or 5176)
        rows = ["Preview metadata recorded."]
        for name, att in [("local", local), ("frontier", frontier), ("council", council)]:
            if att is not None:
                rows.append(f"- `{name}`: `{att.preview_status}`")
        status = "\n".join(rows)
        if start_servers:
            status += "\n\nLinks are marked ready only after the dev server answers the sandbox URL."

        def _link(label, att):
            if att is None:
                return f"{label} lane not in this trial."
            return f"[Open {label} Preview]({att.preview_url})" if att.preview_status == "ready" else f"{label} preview not ready: `{att.preview_status}`"

        return status, _link("Local", local), _link("Frontier", frontier), _link("Council", council)

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
        council_correctness,
        council_feel,
        council_ui,
        council_tests,
        council_merge,
        preference_strength,
        failure_modes,
        manual_local_typecheck,
        manual_local_visible_change,
        manual_frontier_typecheck,
        manual_frontier_visible_change,
        manual_council_typecheck,
        manual_council_visible_change,
        notes,
        cleanup_after,
    ):
        trial_id = trial_id.strip()
        if not trial_id:
            return "Create a trial first."
        present = set(load_trial(trial_id).attempts.keys())
        outputs = []
        lane_scores = [
            (
                "local",
                {
                    "correctness": local_correctness,
                    "gameplay_feel": local_feel,
                    "ui_quality": local_ui,
                    "test_confidence": local_tests,
                    "mergeability": local_merge,
                },
                bool(manual_local_typecheck),
                bool(manual_local_visible_change),
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
                bool(manual_frontier_typecheck),
                bool(manual_frontier_visible_change),
            ),
            (
                "council",
                {
                    "correctness": council_correctness,
                    "gameplay_feel": council_feel,
                    "ui_quality": council_ui,
                    "test_confidence": council_tests,
                    "mergeability": council_merge,
                },
                bool(manual_council_typecheck),
                bool(manual_council_visible_change),
            ),
        ]
        for attempt_name, scores, manual_typecheck, manual_visible in lane_scores:
            if attempt_name not in present:
                continue
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
                    manual_typecheck=manual_typecheck,
                    manual_visible_change=manual_visible,
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

    def council_trace_ui(trial_id):
        return render_council_trace_text(trial_id)

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

    def validation_choices() -> List[str]:
        return [p.relative_to(REPO).as_posix() for p in validation_artifact_paths()]

    def refresh_validation_ui(selected_rel: str = ""):
        choices = validation_choices()
        selected = selected_rel if selected_rel in choices else (choices[0] if choices else "")
        summary, raw = ("No validation artifacts found yet.", "")
        if selected:
            summary, raw = summarize_validation_artifact(REPO / selected)
        return gr.update(choices=choices, value=selected), summary, raw

    def view_validation_ui(selected_rel: str):
        if not selected_rel:
            return "Select a validation artifact.", ""
        return summarize_validation_artifact(REPO / selected_rel)

    def run_expanded_validation_ui():
        out = REPO / "benchmarks" / "results" / "multi_agent_orchestration_validation_arena_latest.json"
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "validate_multi_agent_orchestration.py"),
            "--tasks",
            str(REPO / "benchmarks" / "hud_combat_field_flow_low_tasks_v1.json"),
            "--tasks",
            str(REPO / "benchmarks" / "multi_agent_orchestration_mass_tasks_v1.json"),
            "--tasks",
            str(REPO / "benchmarks" / "task_routing_mixed_tasks_v1.json"),
            "--expected-multi-agent-min-rate",
            "0.6",
            "--max-unexpected-multi-agent-rate",
            "0.25",
            "--output-json",
            str(out),
        ]
        log = REPORTS_DIR / "arena_validation_run.log"
        code, elapsed = run_cmd(cmd, cwd=REPO, log_path=log, timeout_s=600)
        if code != 0:
            return (
                f"Validation command failed (`exit={code}`) in {elapsed:.1f}s. "
                f"See `{log.relative_to(REPO).as_posix()}`.",
                gr.update(),
                "",
                "",
            )
        choices_update, summary, raw = refresh_validation_ui(out.relative_to(REPO).as_posix())
        status = (
            f"Expanded split/merge validation completed in {elapsed:.1f}s.\n\n"
            f"Output: `{out.relative_to(REPO).as_posix()}`\n"
            f"Log: `{log.relative_to(REPO).as_posix()}`"
        )
        return status, choices_update, summary, raw

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

    def _chat_pairs(chat_state):
        pairs: list[tuple[str, str]] = []
        pending_user = ""
        for item in chat_state or []:
            speaker = item.get("speaker", item.get("role", "speaker"))
            content = item.get("content", "")
            if speaker == "Human":
                if pending_user:
                    pairs.append((pending_user, ""))
                pending_user = content
            else:
                label = f"**{speaker}**\n\n{content}"
                if pending_user:
                    pairs.append((pending_user, label))
                    pending_user = ""
                else:
                    pairs.append(("", label))
        if pending_user:
            pairs.append((pending_user, ""))
        return pairs

    def _chat_transcript(chat_state, max_chars: int = 12000) -> str:
        lines = []
        for item in chat_state or []:
            speaker = item.get("speaker", item.get("role", "speaker"))
            content = (item.get("content") or "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        text = "\n\n".join(lines)
        if len(text) <= max_chars:
            return text
        return "[earlier transcript clipped]\n\n" + text[-max_chars:]

    def _default_chat_speakers():
        return []

    def _speaker_choices(room_state):
        names = [str(s.get("name", "")).strip() for s in room_state or [] if str(s.get("name", "")).strip()]
        return names

    def _find_speaker(room_state, speaker_name):
        speakers = list(room_state or [])
        for speaker in speakers:
            if speaker.get("name") == speaker_name:
                return speaker
        return speakers[0] if speakers else None

    def _speaker_label(speaker) -> str:
        if not speaker:
            return "No speaker"
        name = str(speaker.get("name") or "Speaker").strip() or "Speaker"
        backend = str(speaker.get("backend") or "Local").strip()
        if backend == "Frontier":
            model = str(speaker.get("frontier_model") or os.environ.get("FRONTIER_MODEL") or "default").strip()
            return f"{name} [{backend}: {model}]"
        adapter = str(speaker.get("local_adapter") or "checkpoints/fe-lora-30m").strip()
        model_id = resolve_local_model_id(adapter, os.environ.get("MODEL"))
        return f"{name} [{backend}: {adapter} on {model_id}]"

    def _chat_generate(
        speaker_name,
        room_state,
        chat_state,
        max_tokens,
        temp,
    ):
        speaker = _find_speaker(room_state, speaker_name)
        if not speaker:
            raise RuntimeError("Add at least one speaker in Setup Conversation Room, then enter the room.")
        label = _speaker_label(speaker)
        profile = str(speaker.get("profile") or "").strip()
        transcript = _chat_transcript(chat_state)
        system = (
            "You are participating in the Fallen Empire arena model chat. "
            "Respond as the selected model speaker, stay grounded in the shared transcript, "
            "and address the previous turn directly. Keep answers concise unless code or diagnosis requires detail. "
            "If you and the other speaker(s) have reached genuine agreement, or you have nothing new to contribute, "
            f"end your reply with the exact token {DEBATE_STOP_TOKEN} on its own final line to signal the "
            "conversation can conclude. Only use that token when the discussion has truly resolved — do not use "
            "it prematurely."
            + (f"\n\nSpeaker profile:\n{profile}" if profile else "")
        )
        user = (
            f"Current speaker: {label}\n\n"
            "Shared transcript:\n"
            f"{transcript or '(empty)'}\n\n"
            "Write the next contribution from the current speaker only."
        )
        req = GenerationRequest(
            messages=[ChatMessage("system", system), ChatMessage("user", user)],
            max_tokens=max(64, int(max_tokens or 768)),
            temperature=float(temp or 0.0),
        )
        text, total_tokens = _generate_with_speaker_backend(speaker, req)
        status = f"{label} responded" + (f" using {total_tokens} tokens." if total_tokens else ".")
        return text, status

    def chat_send_ui(
        user_text,
        speaker_name,
        room_state,
        chat_state,
        max_tokens,
        temp,
    ):
        state = list(chat_state or [])
        text = (user_text or "").strip()
        if not text:
            return _chat_pairs(state), state, "", "Enter a message or use Continue Selected Model."
        state.append({"speaker": "Human", "speaker_name": "Human", "content": text})
        try:
            reply, status = _chat_generate(speaker_name, room_state, state, int(max_tokens or 768), float(temp or 0.0))
            ended = _debate_signaled_stop(reply)
            state.append({
                "speaker": _speaker_label(_find_speaker(room_state, speaker_name)),
                "speaker_name": speaker_name,
                "content": _strip_stop_token(reply),
                "ended_debate": ended,
            })
            if ended:
                status = f"{status} (signaled debate conclusion)"
        except Exception as exc:
            status = f"{speaker_name} failed: `{type(exc).__name__}: {exc}`"
        return _chat_pairs(state), state, "", status

    def chat_continue_ui(
        speaker_name,
        room_state,
        chat_state,
        max_tokens,
        temp,
    ):
        state = list(chat_state or [])
        if not state:
            return _chat_pairs(state), state, "Start with a human message first."
        try:
            reply, status = _chat_generate(speaker_name, room_state, state, int(max_tokens or 768), float(temp or 0.0))
            ended = _debate_signaled_stop(reply)
            state.append({
                "speaker": _speaker_label(_find_speaker(room_state, speaker_name)),
                "speaker_name": speaker_name,
                "content": _strip_stop_token(reply),
                "ended_debate": ended,
            })
            if ended:
                status = f"{status} (signaled debate conclusion)"
        except Exception as exc:
            status = f"{speaker_name} failed: `{type(exc).__name__}: {exc}`"
        return _chat_pairs(state), state, status

    def chat_clear_ui():
        return [], [], ""

    def run_agentic_loop_ui(
        opening_message,
        participant_names,
        room_state,
        chat_state,
        max_tokens,
        temp,
        max_rounds,
    ):
        """Broadcast one message to several speakers and let them keep responding to
        each other automatically (round-robin, each seeing prior turns) until a
        speaker signals the debate is concluded or `max_rounds` is hit (safety cap)."""
        state = list(chat_state or [])
        names = [n for n in (participant_names or []) if n]
        opening = (opening_message or "").strip()
        if not names:
            yield _chat_pairs(state), state, opening_message, "Select at least one participant to broadcast to."
            return
        if not opening and not state:
            yield _chat_pairs(state), state, "", "Enter an opening message to broadcast to the selected participants."
            return
        if opening:
            state.append({"speaker": "Human", "speaker_name": "Human", "content": opening})
            yield _chat_pairs(state), state, "", f"Broadcasting to {', '.join(names)}…"

        rounds_cap = max(1, int(max_rounds or 1))
        stop_reason = None
        for round_idx in range(1, rounds_cap + 1):
            for name in names:
                speaker = _find_speaker(room_state, name)
                if not speaker or speaker.get("name") != name:
                    continue
                label = _speaker_label(speaker)
                try:
                    reply, _status = _chat_generate(name, room_state, state, int(max_tokens or 768), float(temp or 0.0))
                    ended = _debate_signaled_stop(reply)
                    state.append({
                        "speaker": label,
                        "speaker_name": name,
                        "content": _strip_stop_token(reply),
                        "ended_debate": ended,
                    })
                    status = f"Round {round_idx}/{rounds_cap} — {label} responded."
                    if ended:
                        stop_reason = f"{name} signaled the debate is concluded (round {round_idx})."
                        status += " Debate concluded."
                except Exception as exc:
                    status = f"Round {round_idx}/{rounds_cap} — {name} failed: `{type(exc).__name__}: {exc}`"
                yield _chat_pairs(state), state, "", status
                if stop_reason:
                    break
            if stop_reason:
                break

        final_status = stop_reason or f"Reached max rounds ({rounds_cap}) without a concession — stopping (safety cap)."
        yield _chat_pairs(state), state, "", final_status

    def judge_the_debate_ui(judge_names_selected, room_state, chat_state, max_tokens, temp):
        speakers = list(room_state or [])
        selected = set(judge_names_selected or [])
        judges = [s for s in speakers if s.get("name") in selected and _is_judge_speaker(s)]
        if not judges:
            return "", "Select at least one judge (add a speaker with 'This speaker is a judge' checked, then Enter Room)."
        candidates = _transcript_speaker_names(chat_state)
        if len(candidates) < 2:
            return "", "Need at least two participants with turns in the transcript before judging."
        transcript = _chat_transcript_by_name(chat_state)
        verdicts: list[Dict[str, Any]] = []
        for judge in judges:
            judge_label = _speaker_label(judge)
            try:
                raw = _judge_generate(judge, transcript, candidates, int(max_tokens or 768), float(temp or 0.0))
                parsed = _parse_judge_verdict(raw, candidates)
            except Exception as exc:
                parsed = {"winner": None, "reasoning": f"`{type(exc).__name__}: {exc}`", "raw": "", "parse_ok": False}
            verdicts.append({"judge": judge.get("name"), "judge_label": judge_label, **parsed})
        tally = _tally_judge_votes(verdicts)
        scoreboard = _render_judge_scoreboard(verdicts, tally, candidates)
        record = _build_debate_record(
            room_state=speakers,
            chat_state=chat_state,
            judge_names=[j.get("name") for j in judges],
            verdicts=verdicts,
            tally=tally,
        )
        try:
            append_jsonl(MODEL_CHAT_DEBATES_PATH, record)
            persist_note = f" Logged to `{MODEL_CHAT_DEBATES_PATH.relative_to(REPO)}` for training data."
        except Exception as exc:
            persist_note = f" (failed to log debate record: `{type(exc).__name__}: {exc}`)"
        status = f"{len(judges)} judge(s) voted." + persist_note
        return scoreboard, status

    def room_summary(room_state) -> str:
        speakers = list(room_state or [])
        if not speakers:
            return "No speakers yet. Add a speaker profile, then enter the room."
        rows = []
        for idx, speaker in enumerate(speakers, start=1):
            tag = " `[JUDGE]`" if _is_judge_speaker(speaker) else ""
            rows.append(f"{idx}. `{speaker.get('name')}`{tag} — {speaker.get('backend')} — {speaker.get('profile') or 'no profile'}")
        return "\n".join(rows)

    def room_reset_ui():
        speakers = []
        return (
            speakers,
            room_summary(speakers),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            "Room cleared.",
        )

    def room_add_speaker_ui(room_state, name, backend, profile, local_adapter, frontier_model, is_judge):
        speakers = list(room_state or [])
        clean_name = (name or "").strip()
        if not clean_name:
            return (
                speakers,
                room_summary(speakers),
                gr.update(choices=_speaker_choices(speakers)),
                gr.update(choices=_debater_names(speakers)),
                gr.update(choices=_judge_names(speakers)),
                "Speaker name is required.",
            )
        speaker = {
            "name": clean_name,
            "backend": backend or "Local",
            "profile": (profile or "").strip(),
            "local_adapter": (local_adapter or "checkpoints/fe-lora-30m").strip(),
            "frontier_model": (frontier_model or os.environ.get("FRONTIER_MODEL", "")).strip(),
            "is_judge": bool(is_judge),
        }
        replaced = False
        for idx, existing in enumerate(speakers):
            if existing.get("name") == clean_name:
                speakers[idx] = speaker
                replaced = True
                break
        if not replaced:
            speakers.append(speaker)
        choices = _speaker_choices(speakers)
        return (
            speakers,
            room_summary(speakers),
            gr.update(choices=choices, value=clean_name),
            gr.update(choices=_debater_names(speakers)),
            gr.update(choices=_judge_names(speakers)),
            f"Updated `{clean_name}`." if replaced else f"Added `{clean_name}`.",
        )

    def room_enter_ui(room_state):
        speakers = list(room_state or [])
        if not speakers:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
                "Add at least one speaker before entering the room.",
            )
        choices = _speaker_choices(speakers)
        return (
            gr.update(choices=choices, value=choices[0]),
            gr.update(choices=_debater_names(speakers), value=[]),
            gr.update(choices=_judge_names(speakers), value=[]),
            f"Entered room with {len(speakers)} speaker(s).",
        )

    with gr.Blocks(title="Fallen Empire Game Task Arena", css=theme_css) as demo:
        gr.Markdown(
            "# Fallen Empire Game Task Arena\n"
            "Choose a standardized task, then generate **Local LoRA**, **Frontier**, and a multi-agent "
            "**Council** attempt side by side in disposable worktrees. Each attempt applies real code into "
            "its own copy of the game's `/test-env` sandbox, opens a playable preview, and is graded head to head."
        )
        with gr.Tabs():
            with gr.Tab("Arena Trials"):
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
                        council_port = gr.Number(label="Council port", value=5176, precision=0)
                        start_servers = gr.Checkbox(label="Start dev servers", value=True)
                    with gr.Row():
                        include_council = gr.Checkbox(
                            label="Include Council lane (planner decomposes + specialist councils write code)",
                            value=True,
                        )
                        council_mode = gr.Dropdown(
                            choices=["planner-model", "heuristic", "single"],
                            value="planner-model",
                            label="Council decomposition",
                            info="planner-model: the LoRA planner decides how to split & which specialists to call (heuristic fallback). heuristic: deterministic split. single: no decomposition.",
                        )
                        council_rounds = gr.Number(label="Council debate rounds", value=1, precision=0)

                run_trial_btn = gr.Button("Run Full Trial: Generate All + Open Preview Links", variant="primary")
                trial_id = gr.Textbox(label="Trial ID", interactive=False)
                run_status = gr.Markdown()
                with gr.Row():
                    local_preview_link = gr.Markdown()
                    frontier_preview_link = gr.Markdown()
                    council_preview_link = gr.Markdown()
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
                        council_port,
                        start_servers,
                        include_council,
                        council_mode,
                        council_rounds,
                    ],
                    outputs=[trial_id, run_status, local_preview_link, frontier_preview_link, council_preview_link],
                )

            with gr.Tab("Model Chat"):
                room_state = gr.State([])
                chat_state = gr.State([])
                gr.Markdown("## Setup Conversation Room")
                with gr.Accordion("Create or update speakers", open=True):
                    room_speaker_summary = gr.Markdown(room_summary([]))
                    with gr.Row():
                        room_speaker_name = gr.Textbox(label="Speaker name", placeholder="e.g. Local implementer")
                        room_backend = gr.Dropdown(["Local", "Frontier"], value="Local", label="Backend")
                    room_profile = gr.Textbox(
                        label="Profile",
                        lines=3,
                        placeholder="Describe this speaker's role, style, constraints, and what it should focus on.",
                    )
                    with gr.Row():
                        room_local_adapter = gr.Textbox(label="Local adapter", value="checkpoints/fe-lora-30m")
                        room_frontier_model = gr.Textbox(label="Frontier model", value=os.environ.get("FRONTIER_MODEL", ""))
                    room_is_judge = gr.Checkbox(
                        label="This speaker is a judge (votes on debate winners; excluded from the debate loop)",
                        value=False,
                    )
                    with gr.Row():
                        room_add_btn = gr.Button("Add Or Update Speaker", variant="primary")
                        room_reset_btn = gr.Button("Clear Room")
                        room_enter_btn = gr.Button("Enter Room")
                    room_status = gr.Markdown()

                gr.Markdown("## Conversation Room")
                with gr.Row():
                    chat_speaker = gr.Dropdown(choices=[], value=None, label="Speak to")
                    chat_max_tokens = gr.Number(label="Max response tokens", value=768, precision=0)
                    chat_temp = gr.Slider(0, 1, value=0, step=0.05, label="Temperature")
                chat_box = gr.Chatbot(label="Shared conversation", height=520, type="tuples")
                chat_input = gr.Textbox(
                    label="Message",
                    lines=3,
                    placeholder="Enter the room, choose a speaker, then send a message. Switch speakers to have them respond in the same conversation.",
                )
                with gr.Row():
                    chat_send_btn = gr.Button("Send To Speaker", variant="primary")
                    chat_continue_btn = gr.Button("Continue Speaker")
                    chat_clear_btn = gr.Button("Clear Chat")
                chat_status = gr.Markdown()

                gr.Markdown("## Agentic Loop — Send To Both")
                gr.Markdown(
                    "Broadcast the message above to every selected participant at once, then let them "
                    "keep responding to each other automatically. A participant can end the debate by "
                    "signaling genuine agreement or that they have nothing new to add; otherwise the loop "
                    "stops at **Max rounds** as a safety cap."
                )
                with gr.Row():
                    loop_participants = gr.CheckboxGroup(choices=[], label="Broadcast to / loop participants")
                    loop_max_rounds = gr.Slider(1, 20, value=6, step=1, label="Max rounds (safety cap)")
                with gr.Row():
                    loop_run_btn = gr.Button("Send To Both & Run Loop", variant="primary")
                    loop_stop_btn = gr.Button("Stop Loop")
                loop_status = gr.Markdown()

                with gr.Accordion("Judge the debate", open=False):
                    gr.Markdown(
                        "Each selected judge independently reads the full transcript and votes for who "
                        "argued the case best. Votes are tallied into a scoreboard, and the transcript + "
                        "verdicts are appended to a JSONL log for later training/preference-data use."
                    )
                    judge_checkboxes = gr.CheckboxGroup(choices=[], label="Judges to consult")
                    judge_run_btn = gr.Button("Judge The Debate", variant="primary")
                    judge_status = gr.Markdown()
                    judge_scoreboard = gr.Markdown()

                room_add_outputs = [room_state, room_speaker_summary, chat_speaker, loop_participants, judge_checkboxes, room_status]
                room_add_btn.click(
                    room_add_speaker_ui,
                    inputs=[
                        room_state,
                        room_speaker_name,
                        room_backend,
                        room_profile,
                        room_local_adapter,
                        room_frontier_model,
                        room_is_judge,
                    ],
                    outputs=room_add_outputs,
                    show_api=False,
                )
                room_reset_btn.click(
                    room_reset_ui,
                    outputs=room_add_outputs,
                    show_api=False,
                )
                room_enter_btn.click(
                    room_enter_ui,
                    inputs=[room_state],
                    outputs=[chat_speaker, loop_participants, judge_checkboxes, room_status],
                    show_api=False,
                )
                chat_send_btn.click(
                    chat_send_ui,
                    inputs=[
                        chat_input,
                        chat_speaker,
                        room_state,
                        chat_state,
                        chat_max_tokens,
                        chat_temp,
                    ],
                    outputs=[chat_box, chat_state, chat_input, chat_status],
                    show_api=False,
                )
                chat_input.submit(
                    chat_send_ui,
                    inputs=[
                        chat_input,
                        chat_speaker,
                        room_state,
                        chat_state,
                        chat_max_tokens,
                        chat_temp,
                    ],
                    outputs=[chat_box, chat_state, chat_input, chat_status],
                    show_api=False,
                )
                chat_continue_btn.click(
                    chat_continue_ui,
                    inputs=[
                        chat_speaker,
                        room_state,
                        chat_state,
                        chat_max_tokens,
                        chat_temp,
                    ],
                    outputs=[chat_box, chat_state, chat_status],
                    show_api=False,
                )
                chat_clear_btn.click(chat_clear_ui, outputs=[chat_box, chat_state, chat_status], show_api=False)

                loop_event = loop_run_btn.click(
                    run_agentic_loop_ui,
                    inputs=[
                        chat_input,
                        loop_participants,
                        room_state,
                        chat_state,
                        chat_max_tokens,
                        chat_temp,
                        loop_max_rounds,
                    ],
                    outputs=[chat_box, chat_state, chat_input, loop_status],
                    show_api=False,
                )
                loop_stop_btn.click(
                    lambda: "Loop stop requested — finishing the current turn, then halting.",
                    inputs=None,
                    outputs=[loop_status],
                    cancels=[loop_event],
                    show_api=False,
                )
                judge_run_btn.click(
                    judge_the_debate_ui,
                    inputs=[judge_checkboxes, room_state, chat_state, chat_max_tokens, chat_temp],
                    outputs=[judge_scoreboard, judge_status],
                    show_api=False,
                )

        with gr.Accordion("Council process trace — read the full debate", open=False):
            gr.Markdown(
                "The planner's decomposition, each subtask's specialist council (per-round "
                "participant drafts + adjudication), the chosen change, and the apply result."
            )
            council_trace_btn = gr.Button("Show Full Council Trace")
            council_trace_box = gr.Textbox(label="Council trace", lines=30, max_lines=2000, interactive=False)
            council_trace_btn.click(council_trace_ui, inputs=[trial_id], outputs=[council_trace_box])

        gr.Markdown("## Grade And Complete")
        winner = gr.Radio(["local", "frontier", "council", "tie", "neither"], value="tie", label="Winner")
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
            with gr.Column():
                gr.Markdown("### Council")
                council_correctness = gr.Slider(1, 5, value=3, step=1, label="Correctness")
                council_feel = gr.Slider(1, 5, value=3, step=1, label="Gameplay feel")
                council_ui = gr.Slider(1, 5, value=3, step=1, label="UI quality")
                council_tests = gr.Slider(1, 5, value=3, step=1, label="Test confidence")
                council_merge = gr.Slider(1, 5, value=3, step=1, label="Mergeability")
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
            council_correctness,
            council_feel,
            council_ui,
            council_tests,
            council_merge,
        ]
        task_id.change(
            lambda task: grading_labels_ui(task) * 3,
            inputs=[task_id],
            outputs=rubric_outputs,
        )
        demo.load(
            lambda task: grading_labels_ui(task) * 3,
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
            with gr.Column():
                gr.Markdown("#### Council viability")
                manual_council_typecheck = gr.Checkbox(label="Typecheck verified", value=False)
                manual_council_visible_change = gr.Checkbox(label="Visible change confirmed", value=False)
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
                council_correctness,
                council_feel,
                council_ui,
                council_tests,
                council_merge,
                preference_strength,
                failure_modes,
                manual_local_typecheck,
                manual_local_visible_change,
                manual_frontier_typecheck,
                manual_frontier_visible_change,
                manual_council_typecheck,
                manual_council_visible_change,
                notes,
                cleanup_after,
            ],
            outputs=[complete_status],
        )

        with gr.Accordion("Split/Merge Validity Testing", open=False):
            gr.Markdown(
                "Review expanded multi-agent split/merge validation artifacts and run a fresh expanded pass "
                "directly from the arena UI."
            )
            validation_picker = gr.Dropdown(label="Validation artifact", choices=validation_choices(), value=(validation_choices()[0] if validation_choices() else ""))
            with gr.Row():
                validation_refresh_btn = gr.Button("Refresh Artifact List")
                validation_run_btn = gr.Button("Run Expanded Validation", variant="primary")
            validation_status = gr.Markdown()
            validation_summary = gr.Markdown()
            validation_raw = gr.Textbox(label="Validation JSON", lines=14)
            validation_refresh_btn.click(
                refresh_validation_ui,
                inputs=[validation_picker],
                outputs=[validation_picker, validation_summary, validation_raw],
            )
            validation_picker.change(
                view_validation_ui,
                inputs=[validation_picker],
                outputs=[validation_summary, validation_raw],
            )
            validation_run_btn.click(
                run_expanded_validation_ui,
                outputs=[validation_status, validation_picker, validation_summary, validation_raw],
            )
            demo.load(
                refresh_validation_ui,
                inputs=[validation_picker],
                outputs=[validation_picker, validation_summary, validation_raw],
            )

        with gr.Accordion("Details: manual packet / verification / reports", open=False):
            attempt = gr.Radio(["local", "frontier", "council"], value="local", label="Attempt")
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
            create_btn.click(create_ui, inputs=[task_id, source_repo, worktree_root, base_ref, include_council], outputs=[trial_id, detail_status])
            generate_btn.click(generate_apply_both_ui, inputs=[trial_id, gen_local_adapter, gen_frontier_model, gen_tokens, council_mode, council_rounds], outputs=[detail_status])
            preview_btn.click(preview_both_ui, inputs=[trial_id, local_port, frontier_port, council_port, start_servers], outputs=[detail_status, local_preview_link, frontier_preview_link, council_preview_link])
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
    p_generate.add_argument(
        "--local-model",
        default=os.environ.get("MODEL"),
        help="Local MLX base model id (defaults to adapter_config.json model, then $MODEL).",
    )
    p_generate.add_argument("--model", default=os.environ.get("FRONTIER_MODEL"))
    p_generate.add_argument("--max-tokens", type=int, default=4096)
    p_generate.add_argument("--temp", type=float, default=0.0)
    p_generate.add_argument("--context-chars", type=int, default=16000)
    p_generate.add_argument(
        "--bug-check-loop",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GAME_TASK_ARENA_BUG_CHECK_LOOP", "1") not in {"0", "false", "False"},
        help="Run a post-generation patch bug-check loop (default on).",
    )
    p_generate.add_argument(
        "--bug-check-rounds",
        type=int,
        default=max(1, int(os.environ.get("GAME_TASK_ARENA_BUG_CHECK_ROUNDS", "1"))),
        help="Maximum bug-check rounds (default 1).",
    )
    p_generate.add_argument(
        "--bug-check-max-tokens",
        type=int,
        default=max(256, int(os.environ.get("GAME_TASK_ARENA_BUG_CHECK_MAX_TOKENS", "4096"))),
        help="Decode budget per bug-check round.",
    )
    p_generate.add_argument(
        "--bug-check-system-prompt",
        default=os.environ.get("GAME_TASK_ARENA_BUG_CHECK_SYSTEM_PROMPT", DEFAULT_BUG_CHECK_SYSTEM_PROMPT),
        help="System prompt for the bug-check reviewer stage.",
    )
    p_generate.add_argument(
        "--bug-check-rag",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GAME_TASK_ARENA_BUG_CHECK_RAG", "1") not in {"0", "false", "False"},
        help="Enable bug-fix RAG context retrieval for bug-check rounds.",
    )
    p_generate.add_argument(
        "--bug-check-rag-corpus",
        default=os.environ.get("GAME_TASK_ARENA_BUG_CHECK_RAG_CORPUS", str(DEFAULT_BUG_FIX_RAG_CORPUS)),
        help="Bug-fix RAG corpus path.",
    )
    p_generate.add_argument(
        "--bug-check-rag-top-k",
        type=int,
        default=max(1, int(os.environ.get("GAME_TASK_ARENA_BUG_CHECK_RAG_TOP_K", "6"))),
        help="Top-k retrieval hits for each bug-check round.",
    )
    p_generate.add_argument(
        "--bug-check-rag-max-chars",
        type=int,
        default=max(400, int(os.environ.get("GAME_TASK_ARENA_BUG_CHECK_RAG_MAX_CHARS", "2200"))),
        help="Character cap for bug-check RAG context block.",
    )
    p_generate.add_argument(
        "--no-context-bm25",
        action="store_true",
        help="Disable BM25 reordering among glob-matched context files (deterministic order only).",
    )

    p_council = sub.add_parser("council", help="Council lane: planner decomposes + specialist councils write code into the worktree.")
    p_council.add_argument("--trial-id", required=True)
    p_council.add_argument("--attempt", default="council")
    p_council.add_argument("--adapter-path", default="checkpoints/fe-lora-30m")
    p_council.add_argument("--local-model", default=os.environ.get("MODEL"))
    p_council.add_argument("--max-tokens", type=int, default=4096)
    p_council.add_argument("--temp", type=float, default=0.0)
    p_council.add_argument("--context-chars", type=int, default=9000)
    p_council.add_argument("--no-context-bm25", action="store_true")
    p_council.add_argument("--council-mode", choices=["planner-model", "heuristic", "single"], default="planner-model")
    p_council.add_argument("--council-rounds", type=int, default=1)
    p_council.add_argument("--council-max-subtasks", type=int, default=4)

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
        build_app().launch(server_name=args.host, server_port=args.port, share=args.share, show_api=False)


if __name__ == "__main__":
    main()
