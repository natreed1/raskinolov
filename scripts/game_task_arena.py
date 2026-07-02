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
MODEL_CHAT_CONVERSATIONS_DIR = REPO / "benchmarks" / "results" / "model_chat_conversations"
MODEL_CHAT_CONVERSATIONS_INDEX = MODEL_CHAT_CONVERSATIONS_DIR / "index.json"
EARLY_STOP_MODES = ("first_signal", "unanimous", "majority")
DEFAULT_EARLY_STOP_MODE = "unanimous"

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
DEBATE_PASS_TOKEN = "[[PASS]]"

# Structured debate: proposal → develop (×N−2) → conclusion (debaters), then adjudicator.
# Legacy phase names (opening/rebuttal/final) remain aliases for prompts and tests.
DEBATE_TURN_MODES: Tuple[str, ...] = ("proposal", "develop", "conclusion")
DEBATE_TURN_MODE_LABELS: Dict[str, str] = {
    "proposal": "Proposal",
    "develop": "Deepen",
    "conclusion": "Conclusion",
}
DEBATE_PHASES: Tuple[str, ...] = ("opening", "rebuttal", "final")
DEBATE_PHASE_LABELS: Dict[str, str] = {
    "opening": "Opening statements",
    "rebuttal": "Rebuttal",
    "final": "Final statement",
    "proposal": "Proposal",
    "develop": "Deepen",
    "conclusion": "Conclusion",
}
DEFAULT_DEBATE_PHASES = 3

# Curated reference facts for Russian-author debates (offline fallback + search backing).
# Sourced from Britannica, Wikipedia, Encyclopedia.com summaries (2026-07-01).
DEBATE_FACT_SNIPPETS: Dict[str, List[str]] = {
    "tolstoy": [
        "Leo Tolstoy (1828–1910) wrote War and Peace (1869), an epic of Napoleon's 1812 invasion weaving five aristocratic families through history and private conscience.",
        "Anna Karenina (1878) is Tolstoy's other masterwork — a realist novel of love, adultery, and social hypocrisy; Fyodor Dostoevsky called it flawless as art.",
        "Tolstoy is widely ranked among the greatest novelists; Virginia Woolf called him 'the greatest of all novelists.'",
        "Tolstoy's realism penetrates inner consciousness through concrete observation — War and Peace is often cited as among the greatest novels ever written.",
    ],
    "dostoevsky": [
        "Fyodor Dostoevsky (1821–1881) wrote Crime and Punishment (1866), a psychological novel of guilt and moral torment through Raskolnikov in St. Petersburg.",
        "The Brothers Karamazov (1880) is Dostoevsky's final novel — a philosophical drama of faith, doubt, and patricide with the 'Grand Inquisitor' chapter.",
        "Dostoevsky pioneered polyphonic fiction: characters hold autonomous moral voices rather than serving one authorial thesis.",
        "Virginia Woolf said Dostoevsky alone among writers could reconstruct the swiftest, most complicated states of mind.",
    ],
    "chekhov": [
        "Anton Chekhov (1860–1904) is considered the father of the modern short story and a founder of modern drama alongside Ibsen.",
        "Major plays: The Seagull, Uncle Vanya, Three Sisters, The Cherry Orchard — 'theatre of mood' with subtext over plot machinery.",
        "Chekhov was a physician; his laconic style probes ordinary life without didactic moralizing — stories like 'The Lady with the Little Dog.'",
        "Chekhov reinvented the short story through mood, understatement, and psychological precision; Raymond Carver called him the greatest short-story writer.",
    ],
    "russian literature": [
        "The 'greatest Russian author' debate usually centers Tolstoy (epic social realism), Dostoevsky (psychological and spiritual depth), and Chekhov (modern short fiction and drama).",
        "War and Peace and Anna Karenina are Tolstoy's; Crime and Punishment and Brothers Karamazov are Dostoevsky's; Chekhov's masterpieces are plays and short stories, not epic novels.",
    ],
}

# Distinct fallback turns when the local model fails quality checks (probe + UI).
DEBATE_FALLBACK_TURNS: Dict[str, List[str]] = {
    "Tolstoy Advocate": [
        (
            "REBUTTAL: A single psyche in crisis cannot stand for all of literature. "
            "NEW EVIDENCE: War and Peace (1869) tracks five families through Napoleon's 1812 invasion — "
            "history and private conscience at a scale no one-case thriller can match. "
            "CLAIM: Tolstoy's panoramic realism makes him Russia's greatest novelist."
        ),
        (
            "REBUTTAL: Intensity of guilt is not the only measure of greatness. "
            "NEW EVIDENCE: Anna Karenina (1878) ties intimate betrayal to social hypocrisy; "
            "even Dostoevsky praised its artistic perfection. "
            "CLAIM: Tolstoy unites moral philosophy and society-wide drama like no other Russian writer."
        ),
        (
            "REBUTTAL: Modern drama's brevity does not erase the novel's reach. "
            "NEW EVIDENCE: Tolstoy's The Death of Ivan Ilyich compresses mortality and self-deception "
            "into a novella as piercing as any confession scene. "
            "CLAIM: Tolstoy owns both the epic and the intimate — that range crowns him."
        ),
    ],
    "Dostoevsky Advocate": [
        (
            "REBUTTAL: Sweeping battle scenes are not the same as moral depth. "
            "NEW EVIDENCE: Crime and Punishment (1866) maps Raskolnikov's guilt as an inner force "
            "that destroys him before any court does — psychology as drama. "
            "CLAIM: Dostoevsky's conscience-tragedies make him Russia's greatest author."
        ),
        (
            "REBUTTAL: Social panoramas can flatten the soul's contradictions. "
            "NEW EVIDENCE: The Brothers Karamazov (1880) pits faith against doubt in the 'Grand Inquisitor' "
            "and Dmitri's trial — ideas with blood on them. "
            "CLAIM: No Russian writer pushes further into the mystery of free will than Dostoevsky."
        ),
        (
            "REBUTTAL: Elegant society portraits are not the final test of genius. "
            "NEW EVIDENCE: Notes from Underground anticipates modern alienation — a voice that argues "
            "with itself and the reader at once. "
            "CLAIM: Dostoevsky's polyphonic minds remain unmatched in Russian literature."
        ),
    ],
    "Chekhov Advocate": [
        (
            "REBUTTAL: Epic length is not the same as lasting influence. "
            "NEW EVIDENCE: Chekhov's The Cherry Orchard (1904) distills a class in decline through mood "
            "and subtext rather than sermon — modern drama starts here. "
            "CLAIM: Chekhov's precision changed how all writers see ordinary life."
        ),
        (
            "REBUTTAL: Thunderous philosophy can miss the quiet truth of character. "
            "NEW EVIDENCE: 'The Lady with the Little Dog' proves love and regret in a few pages "
            "what thousand-page novels only gesture at. "
            "CLAIM: Chekhov is the greatest because he reinvented the short story and modern theatre."
        ),
    ],
}

_BRACKET_SPEAKER_HALLUCINATION_RE = re.compile(
    r"(?:^|\n)\s*\[(?:Turn\s+\d+\s*:\s*)?([^:\]\n]{1,48}):",
    re.IGNORECASE,
)

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


# Base speaker archetypes for the Model Chat room. These are *prompt-injection* personas
# (dropped into a speaker's `profile`), not trained adapters — chosen for fast iteration.
# Personas + trait numbers are lifted from the council stack so debates echo the trained
# roster: Council Studio voices (`scripts/council_studio.py` PERSONALITIES), roster
# personality variants (`scripts/router/roster.py` / `data/routing/council_roster_v1.json`),
# EQ bootstrap variants (`scripts/bootstrap_council_eq_traces.py`), and the planner
# (`scripts/council_runtime/planner_decomposition_policy.py`). Trait scale is 0.0–1.0 per
# `scripts/router/roster.py::TRAIT_KEYS`. `default_temperature` is a suggested pace only
# (the Model Chat temperature slider is global); pacing is also baked into the profile text.
CHAT_ARCHETYPE_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Researcher",
        "summary": "Slow, calculated, evidence-first; weighs multiple hypotheses before asserting.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.2,
        "profile": (
            "Role: researcher. Be slow, calculated, and evidence-driven — never rush to a verdict. "
            "Before asserting anything, gather and weigh the evidence, consider at least two competing "
            "hypotheses, and cite specific facts, mechanisms, files, or earlier turns. "
            "Traits (0-1): skepticism 0.80, risk_tolerance 0.30, creativity 0.50, decisiveness 0.40, "
            "assertiveness 0.45, verbosity 0.60. "
            "Prefer \"here is what the evidence shows\" over quick opinions. Explicitly flag claims that "
            "lack support and say what would verify them. When you would normally search for a source, "
            "state exactly what you'd look up and what result would change your mind."
        ),
    },
    {
        "name": "Risk Auditor",
        "summary": "Skeptical; challenges assumptions, names failure modes, demands evidence.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.2,
        "profile": (
            "Role: risk auditor. Traits (0-1): skepticism 0.85, risk_tolerance 0.20, assertiveness 0.60, "
            "creativity 0.30, decisiveness 0.50, verbosity 0.50. "
            "Challenge assumptions, name concrete failure modes and edge cases, and demand evidence before "
            "accepting any change. Push back hard on anything risky, unproven, or hand-wavy."
        ),
    },
    {
        "name": "Implementer",
        "summary": "Decisive builder; smallest correct concrete change, names files + verification.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.3,
        "profile": (
            "Role: implementer. Traits (0-1): decisiveness 0.85, assertiveness 0.70, skepticism 0.40, "
            "creativity 0.40, risk_tolerance 0.50, verbosity 0.45. "
            "Prefer the smallest correct concrete change that actually ships. Name the specific systems and "
            "files involved and how you'd verify it. Drive the debate toward a decision rather than circling."
        ),
    },
    {
        "name": "Explorer",
        "summary": "Creative; proposes bold alternatives and trade-offs before committing.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.7,
        "profile": (
            "Role: explorer. Traits (0-1): creativity 0.85, risk_tolerance 0.70, verbosity 0.60, "
            "skepticism 0.50, assertiveness 0.55, decisiveness 0.45. "
            "Propose bold alternatives and unexpected options, surface trade-offs others miss, and widen the "
            "option set before anyone commits to a single path."
        ),
    },
    {
        "name": "Verifier",
        "summary": "Conservative; fixates on tests, type checks, regressions, reproducible evidence.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.1,
        "profile": (
            "Role: verifier. Traits (0-1): skepticism 0.70, risk_tolerance 0.30, decisiveness 0.60, "
            "creativity 0.35, assertiveness 0.50, verbosity 0.50. "
            "Focus on tests, compile/type checks, regressions, and reproducible evidence. Keep asking "
            "\"how do we know this works?\" and insist on concrete verification steps."
        ),
    },
    {
        "name": "Direct Builder",
        "summary": "High-assertiveness; takes a firm stance fast and defends the most direct path.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.4,
        "profile": (
            "Role: direct builder. Traits (0-1): assertiveness 0.78, decisiveness 0.82, risk_tolerance 0.64, "
            "skepticism 0.48, creativity 0.45, verbosity 0.50. "
            "Take a firm position quickly and defend the most direct implementation path. Avoid hedging and "
            "cut through indecision, but concede cleanly if genuinely out-argued."
        ),
    },
    {
        "name": "Creative Synthesizer",
        "summary": "Combines competing proposals into a broader synthesis before choosing.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.7,
        "profile": (
            "Role: creative synthesizer. Traits (0-1): creativity 0.86, verbosity 0.72, risk_tolerance 0.55, "
            "skepticism 0.45, assertiveness 0.55, decisiveness 0.50. "
            "Combine competing proposals into a broader, better option set, then recommend a synthesis rather "
            "than picking a side prematurely. Find the version that captures the strengths of each argument."
        ),
    },
    {
        "name": "Calibrated Mediator",
        "summary": "Balanced; finds the real crux and drives a well-justified handoff.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.3,
        "profile": (
            "Role: calibrated mediator. Balanced traits (0-1): assertiveness 0.60, verbosity 0.50, "
            "risk_tolerance 0.50, creativity 0.45, skepticism 0.60, decisiveness 0.65. "
            "Weigh disagreement and uncertainty fairly, identify the real crux of the debate, and drive "
            "toward a clear, well-justified conclusion and handoff."
        ),
    },
    {
        "name": "Planner",
        "summary": "Big-picture; decomposes into a minimal, dependency-ordered plan.",
        "backend": "Local",
        "is_judge": False,
        "default_temperature": 0.6,
        "profile": (
            "Role: planner. Think in terms of decomposition and sequencing: break the problem into a minimal, "
            "dependency-ordered set of steps, say which kind of specialist should own each, and describe the "
            "integration plan that survives compile and test. Keep the big picture and call out ordering risks "
            "and hidden dependencies the others miss."
        ),
    },
    {
        "name": "Impartial Judge",
        "summary": "Judge preset (excluded from debate phases; adjudicates after phase 3).",
        "backend": "Local",
        "is_judge": True,
        "default_temperature": 0.0,
        "profile": (
            "Role: impartial judge. Read the whole debate and decide who argued most convincingly based on "
            "reasoning quality, evidence, and how well they answered the other side — not who spoke last or "
            "most. Be fair and specific about why."
        ),
    },
]


def _archetype_names() -> List[str]:
    return [str(p.get("name")) for p in CHAT_ARCHETYPE_PRESETS if p.get("name")]


def _find_archetype(name: Optional[str]) -> Optional[Dict[str, Any]]:
    target = (name or "").strip().lower()
    for preset in CHAT_ARCHETYPE_PRESETS:
        if str(preset.get("name", "")).strip().lower() == target:
            return preset
    return None


CUSTOM_ARCHETYPE_LABEL = "New custom agent"


def _archetype_picker_choices() -> List[str]:
    return _archetype_names() + [CUSTOM_ARCHETYPE_LABEL]


def _is_custom_archetype_pick(name: Optional[str]) -> bool:
    return (name or "").strip() == CUSTOM_ARCHETYPE_LABEL


def _lobby_custom_fields_visible(name: Optional[str]) -> bool:
    return _is_custom_archetype_pick(name)


def _archetype_preview_text(name: Optional[str]) -> str:
    if _is_custom_archetype_pick(name):
        return (
            "**New custom agent** — set name, backend, and profile below, "
            "then click **Add Or Update Speaker**."
        )
    preset = _find_archetype(name)
    if not preset:
        return "Pick a base archetype to preview its injected persona."
    tag = " · **judge** (adjudicates after debate phases; optional panel vote)" if preset.get("is_judge") else ""
    temp = preset.get("default_temperature")
    temp_note = f" · suggested temperature ≈ {temp}" if temp is not None else ""
    return (
        f"**{preset['name']}**{tag} — {preset.get('summary', '')}{temp_note}\n\n"
        f"> {preset['profile']}"
    )


def _archetype_load_into_form_values(name: Optional[str]) -> Optional[tuple[str, str, str, str, bool]]:
    """Preset fields for **Load Into Form**; switches the picker to the custom-agent option."""
    preset = _find_archetype(name)
    if not preset:
        return None
    return (
        CUSTOM_ARCHETYPE_LABEL,
        str(preset["name"]),
        str(preset.get("backend", "Local")),
        str(preset.get("profile", "")),
        bool(preset.get("is_judge")),
    )


def _unique_speaker_name(existing_names: Optional[List[str]], base: str) -> str:
    """Return `base`, or `base 2`, `base 3`, … so repeated archetype adds don't collide."""
    taken = {str(n).strip() for n in (existing_names or []) if str(n).strip()}
    clean = (base or "Speaker").strip() or "Speaker"
    if clean not in taken:
        return clean
    idx = 2
    while f"{clean} {idx}" in taken:
        idx += 1
    return f"{clean} {idx}"


def _is_judge_speaker(speaker: Dict[str, Any]) -> bool:
    return bool(speaker.get("is_judge"))


def _debater_names(room_state: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [s.get("name") for s in room_state or [] if s.get("name") and not _is_judge_speaker(s)]


def _judge_names(room_state: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [s.get("name") for s in room_state or [] if s.get("name") and _is_judge_speaker(s)]


def _judge_checkbox_state(room_state: Optional[List[Dict[str, Any]]]) -> tuple[List[str], List[str]]:
    """Return (choices, selected) for the judge CheckboxGroup — all judges pre-selected."""
    judges = _judge_names(room_state)
    return judges, judges


def _debate_turn_mode(round_idx: int, max_rounds: int) -> str:
    """Map a 1-based debate round to proposal, develop, or conclusion."""
    cap = max(1, int(max_rounds or 1))
    ri = max(1, int(round_idx or 1))
    if cap <= 1:
        return "conclusion"
    if ri <= 1:
        return "proposal"
    if ri >= cap:
        return "conclusion"
    return "develop"


_TURN_MODE_TO_LEGACY_PHASE: Dict[str, str] = {
    "proposal": "opening",
    "develop": "rebuttal",
    "conclusion": "final",
}


def _normalize_turn_mode(phase: Optional[str]) -> str:
    """Normalize legacy phase names to proposal/develop/conclusion."""
    key = str(phase or "").strip().lower()
    if key in ("opening", "proposal"):
        return "proposal"
    if key in ("rebuttal", "develop"):
        return "develop"
    if key in ("final", "conclusion"):
        return "conclusion"
    return "develop"


def _debate_phase_for_round(round_idx: int, max_rounds: int) -> str:
    """Legacy phase name (opening/rebuttal/final) for a 1-based debate round."""
    return _TURN_MODE_TO_LEGACY_PHASE[_debate_turn_mode(round_idx, max_rounds)]


def _debate_phase_label(phase: str) -> str:
    mode = _normalize_turn_mode(phase)
    if mode in DEBATE_TURN_MODE_LABELS:
        return DEBATE_TURN_MODE_LABELS[mode]
    return DEBATE_PHASE_LABELS.get(str(phase or "").strip(), str(phase or "debate").replace("_", " ").title())


def _debate_pacing_context(
    round_idx: int,
    max_rounds: int,
    speaker_name: str = "",
    chat_state: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Pacing hints so models self-schedule across N debate phases."""
    cap = max(1, int(max_rounds or 1))
    ri = max(1, int(round_idx or 1))
    mode = _debate_turn_mode(ri, cap)
    rounds_remaining = max(0, cap - ri)
    label = DEBATE_TURN_MODE_LABELS.get(mode, mode.title())
    if mode == "proposal":
        pacing_line = (
            f"You are in phase {ri} of {cap}. Establish your thesis clearly; "
            f"you have {rounds_remaining} phase(s) after this to deepen — hold depth in reserve."
        )
    elif mode == "develop":
        pacing_line = (
            f"You are in phase {ri} of {cap} ({rounds_remaining} remain after this). "
            "Mid-debate: read opponents carefully — rebut or respond if you see a gap or provocative claim, "
            "otherwise add new evidence. Push deeper than the prior phase."
        )
    elif cap == 1:
        pacing_line = "Single-phase debate — deliver your strongest closing argument."
    else:
        pacing_line = (
            f"Final phase ({ri} of {cap}) — synthesize your strongest threads; "
            "do not recycle prior wording verbatim."
        )
    speaker_phases = 0
    target = str(speaker_name or "").strip()
    if target:
        speaker_phases = sum(
            1 for item in _completed_chat_turns(chat_state) if _turn_speaker_name(item) == target
        )
    return {
        "round_idx": ri,
        "max_rounds": cap,
        "turn_mode": mode,
        "phase_label": label,
        "pacing_line": pacing_line,
        "rounds_remaining": rounds_remaining,
        "speaker_phases_spoken": speaker_phases,
    }


def _format_debate_phase_status(round_idx: int, max_rounds: int, *, detail: str = "") -> str:
    cap = max(1, int(max_rounds or 1))
    ri = max(1, int(round_idx or 1))
    ctx = _debate_pacing_context(ri, cap)
    base = f"Phase {ri}/{cap}: {ctx['phase_label']}"
    return f"{base} — {detail}" if detail else base


def _format_adjudicator_status(*, detail: str = "") -> str:
    base = "Adjudicator: Final write-up"
    return f"{base} — {detail}" if detail else base


def _resolve_adjudicator(room_state: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """First judge in the room (e.g. Impartial Judge) adjudicates after debate phases."""
    for speaker in room_state or []:
        if speaker.get("name") and _is_judge_speaker(speaker):
            return speaker
    return None


def _debate_signaled_stop(text: str) -> bool:
    return DEBATE_STOP_TOKEN.lower() in (text or "").lower()


def _strip_stop_token(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(re.escape(DEBATE_STOP_TOKEN), re.IGNORECASE)
    return pattern.sub("", text).strip()


def _debate_signaled_pass(text: str) -> bool:
    return DEBATE_PASS_TOKEN.lower() in (text or "").lower()


def _strip_pass_token(text: str) -> str:
    if not text:
        return text
    pattern = re.compile(re.escape(DEBATE_PASS_TOKEN), re.IGNORECASE)
    return pattern.sub("", text).strip()


def _is_debate_pass_turn(text: str, raw: str = "") -> bool:
    """True when a speaker explicitly passes (no new argument to add)."""
    combined = f"{raw or ''} {text or ''}"
    if _debate_signaled_pass(combined):
        return True
    cleaned = (text or "").strip()
    if not cleaned and not (raw or "").strip():
        return True
    lowered = cleaned.lower()
    if lowered in {"pass", "no new argument", "nothing new to add", "no new evidence", "no comment"}:
        return True
    return any(pattern.match(cleaned) for pattern in _DEBATE_PASS_PHRASE_RES)


def _format_pass_turn_status(speaker_name: str, *, debug: str = "") -> str:
    base = f"Round note — **{speaker_name}** passed (nothing new to add)."
    snippet = " ".join(str(debug or "").split()).strip()
    if snippet:
        if len(snippet) > 160:
            snippet = snippet[:159].rstrip() + "…"
        return f"{base} _({snippet})_"
    return base


def _format_debate_summary_panel(summary: str) -> str:
    clean = (summary or "").strip()
    if not clean:
        return ""
    return f"### Conversation summary\n\n{clean}"


# Small local models tend to (a) echo the "Name:" transcript format at the start of
# their reply and (b) emit garbled bracketed end-markers (e.g. "[[DEBATED]]",
# "[[DEBATE_CONCLUDED]"). Both pollute the debate; strip them so replies read cleanly.
_TRAILING_BRACKET_MARKER_RE = re.compile(r"\s*\[\[?[A-Z0-9][A-Z0-9 _]*\]?\]?\s*$")


def _strip_leading_speaker_prefix(text: str, names: Optional[List[str]]) -> str:
    """Remove one or more leading ``<known speaker name>:`` prefixes a model may echo.

    Only strips prefixes matching a name actually in the room (case-insensitive), so
    legitimate lead-ins like ``Note:`` or ``Step 1:`` are preserved. Also strips a
    leading ``Human:`` echo.
    """
    if not text:
        return text
    known = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    known.add("human")
    result = text.lstrip()
    while True:
        match = re.match(r"^([^\n:]{1,48}):\s*", result)
        if not match or match.group(1).strip().lower() not in known:
            break
        result = result[match.end():].lstrip()
    return result


def _debate_search_enabled() -> bool:
    """Whether to inject web/curated reference facts into debate prompts."""
    return os.environ.get("FE_DEBATE_SEARCH", "1").strip().lower() not in ("0", "false", "no", "off")


def _debate_search_queries(speaker: Dict[str, Any], topic: str) -> List[str]:
    """Build 1–2 targeted search queries from speaker stance and debate topic."""
    blob = f"{speaker.get('name', '')} {speaker.get('profile', '')}".lower()
    queries: List[str] = []
    if "tolstoy" in blob:
        queries.append("Leo Tolstoy War and Peace Anna Karenina literary significance")
    elif "dostoevsky" in blob:
        queries.append("Fyodor Dostoevsky Crime and Punishment Brothers Karamazov psychological depth")
    elif "chekhov" in blob:
        queries.append("Anton Chekhov modern short story drama literary influence")
    topic_clean = (topic or "").strip()
    if topic_clean and len(queries) < 2:
        queries.append(topic_clean[:140])
    return queries[:2]


def _curated_debate_facts_for_query(query: str, *, max_snippets: int = 3) -> List[str]:
    """Return curated fact lines whose keys match the query (offline-safe)."""
    q = (query or "").lower()
    hits: List[str] = []
    for key, facts in DEBATE_FACT_SNIPPETS.items():
        if key in q or any(part in q for part in key.split()):
            for fact in facts:
                if fact not in hits:
                    hits.append(fact)
                if len(hits) >= max_snippets:
                    return hits
    if not hits and "russian" in q:
        hits.extend(DEBATE_FACT_SNIPPETS.get("russian literature", [])[:max_snippets])
    return hits[:max_snippets]


def _debate_search_snippets(query: str, *, max_snippets: int = 3) -> str:
    """Fetch short reference snippets for a debate query.

    Tries DuckDuckGo (`ddgs` / `duckduckgo_search`) when installed; otherwise uses
    curated ``DEBATE_FACT_SNIPPETS``. Always offline-safe via the curated fallback.
    """
    query = (query or "").strip()
    if not query:
        return ""
    snippets: List[str] = []
    try:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max_snippets):
                body = str(row.get("body") or row.get("snippet") or "").strip()
                if body:
                    snippets.append(body[:320])
                if len(snippets) >= max_snippets:
                    break
    except Exception:
        snippets = []
    if not snippets:
        snippets = _curated_debate_facts_for_query(query, max_snippets=max_snippets)
    if not snippets:
        return ""
    return "\n".join(f"- {s}" for s in snippets[:max_snippets])


def _debate_reference_facts(speaker: Dict[str, Any], topic: str) -> str:
    """Combine search/curated snippets for a speaker's debate turn."""
    parts: List[str] = []
    seen: set[str] = set()
    for query in _debate_search_queries(speaker, topic):
        block = _debate_search_snippets(query, max_snippets=2)
        if block and block not in seen:
            seen.add(block)
            parts.append(block)
    return "\n".join(parts)


DEBATE_TYPING_CURSOR = " ▌"
_SPEAKER_LABEL_SUFFIX_RE = re.compile(r"\s*\[(?:Local|Frontier):[^\]]*\]\s*$", re.IGNORECASE)
_DEBATE_PLACEHOLDER_TURN_RE = re.compile(
    r"^(?:sure,?\s*)?(?:here(?:'s| is) my (?:next )?turn|my turn|let me (?:respond|reply|answer))\b",
    re.IGNORECASE,
)
_DEBATE_META_FILLER_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bstand(?:s|ing)? by my position\b", re.IGNORECASE),
    re.compile(r"\bmy position still stands\b", re.IGNORECASE),
    re.compile(
        r"\b(?:from|with|using|grounded in) (?:the )?reference facts above\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bevidence from the reference facts above\b", re.IGNORECASE),
    re.compile(r"\bnothing (?:new|more|substantive) to add\b", re.IGNORECASE),
    re.compile(r"\bno (?:new )?(?:argument|evidence|points?|contribution) to add\b", re.IGNORECASE),
    re.compile(r"\b(?:i have|i've got) nothing (?:new|more) to (?:add|say|contribute)\b", re.IGNORECASE),
    re.compile(r"\b(?:passing: pass)\s*$", re.IGNORECASE),
    re.compile(r"\bwill (?:hold|maintain) (?:my )?position\b", re.IGNORECASE),
)
_DEBATE_PASS_PHRASE_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*pass\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*nothing new to add\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*no new argument\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*i (?:have|'ve) nothing (?:new|more) to add\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*no (?:new )?(?:evidence|points?) to add\s*[.!]?\s*$", re.IGNORECASE),
)

# Opening fallbacks for generic personas on their first turn (no stance-specific advocate key).
GENERIC_DEBATE_OPENING_FALLBACKS: List[str] = [
    (
        "NEW EVIDENCE: The usual contenders are Tolstoy for epic realism (War and Peace), "
        "Dostoevsky for psychological depth (Crime and Punishment), and Chekhov for modern short fiction. "
        "CLAIM: Tolstoy's panoramic scope makes him the strongest default pick for greatest Russian author."
    ),
    (
        "NEW EVIDENCE: Dostoevsky's polyphonic novels put conscience and free will on trial "
        "in ways Tolstoy's social panoramas do not — The Brothers Karamazov is the clearest example. "
        "CLAIM: Dostoevsky is Russia's greatest author on moral and psychological intensity."
    ),
]


def _turn_speaker_name(item: Dict[str, Any]) -> str:
    """Bare speaker name for transcript/prompting (strips backend label suffixes)."""
    name = str(item.get("speaker_name") or "").strip()
    if name:
        return name
    raw = str(item.get("speaker") or "").strip()
    if not raw or raw == "Human":
        return raw or "Human"
    bare = _SPEAKER_LABEL_SUFFIX_RE.sub("", raw).strip()
    return bare or raw


def _normalize_turn_content(content: str) -> str:
    text = str(content or "").strip()
    if text.endswith(DEBATE_TYPING_CURSOR):
        text = text[: -len(DEBATE_TYPING_CURSOR)].strip()
    return text


def _is_completed_turn(item: Dict[str, Any]) -> bool:
    raw = str(item.get("content") or "")
    if raw.endswith(DEBATE_TYPING_CURSOR):
        return False
    content = _normalize_turn_content(raw)
    if not content:
        return False
    if content.startswith("_(failed:") or content == "_(no output)_":
        return False
    return True


def _completed_chat_turns(chat_state: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Completed turns only — excludes in-progress/empty rows from prompt context."""
    rows: List[Dict[str, Any]] = []
    for item in chat_state or []:
        if not _is_completed_turn(item):
            continue
        rows.append(
            {
                **item,
                "speaker_name": _turn_speaker_name(item),
                "content": _normalize_turn_content(str(item.get("content") or "")),
            }
        )
    return rows


def _hydrate_loop_chat_state(
    conversations_store: Optional[Dict[str, Any]],
    chat_state: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Prefer the longer transcript between Gradio State and the persisted active conversation."""
    incoming = list(chat_state or [])
    conv = _get_active_conversation(conversations_store or {})
    stored = list((conv or {}).get("chat_state") or [])
    return stored if len(stored) > len(incoming) else incoming


def _speaker_turn_index(chat_state: Optional[List[Dict[str, Any]]], speaker_name: str) -> int:
    """How many prior turns this speaker has already contributed."""
    count = 0
    target = str(speaker_name or "").strip()
    for item in _completed_chat_turns(chat_state):
        if _turn_speaker_name(item) == target:
            count += 1
    return count


def _fallback_turn_options(speaker: Dict[str, Any]) -> List[str]:
    """Curated fallback lines keyed by speaker name or inferred debate stance."""
    name = str(speaker.get("name") or "").strip()
    if name in DEBATE_FALLBACK_TURNS:
        return DEBATE_FALLBACK_TURNS[name]
    blob = f"{name} {speaker.get('profile', '')}".lower()
    if "tolstoy" in blob and "dostoevsky advocate" not in blob:
        return DEBATE_FALLBACK_TURNS.get("Tolstoy Advocate", [])
    if "dostoevsky" in blob:
        return DEBATE_FALLBACK_TURNS.get("Dostoevsky Advocate", [])
    if "chekhov" in blob:
        return DEBATE_FALLBACK_TURNS.get("Chekhov Advocate", [])
    return []


def _is_speaker_first_turn(chat_state: Optional[List[Dict[str, Any]]], speaker_name: str) -> bool:
    """True when this debater has not yet contributed a completed turn."""
    return _speaker_turn_index(chat_state, speaker_name) == 0


def _is_debate_opening_turn(chat_state: Optional[List[Dict[str, Any]]], speaker_name: str) -> bool:
    """True when a speaker should open (not rebut): first contribution and no opposing debater yet."""
    if not _is_speaker_first_turn(chat_state, speaker_name):
        return False
    opp_name, opp_content = _last_opponent_turn(chat_state, speaker_name)
    return not bool(opp_name and opp_content)


def _is_meta_filler_turn(text: str) -> bool:
    """Reject meta rebuttal filler that cites no concrete evidence (common small-model failure)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if any(pattern.search(cleaned) for pattern in _DEBATE_META_FILLER_RES):
        return True
    # Short turns that only gesture at "reference facts" without naming a work or author.
    if "reference facts" in lowered and len(cleaned) < 180:
        if not re.search(
            r"\b(?:tolstoy|dostoevsky|chekhov|war and peace|anna karenina|"
            r"crime and punishment|karamazov|seagull|cherry orchard)\b",
            lowered,
        ):
            return True
    return False


def _generic_fallback_turn(speaker: Dict[str, Any], chat_state: Optional[List[Dict[str, Any]]]) -> str:
    """Opening fallback on first turn; pass on later turns for generic personas."""
    name = str(speaker.get("name") or "").strip()
    if _is_speaker_first_turn(chat_state, name):
        completed = _completed_chat_turns(chat_state)
        if _is_debate_opening_turn(chat_state, name):
            for candidate in GENERIC_DEBATE_OPENING_FALLBACKS:
                if not _is_repetitive_turn(candidate, completed, name):
                    return candidate
    return DEBATE_PASS_TOKEN


def _fallback_debate_turn(speaker: Dict[str, Any], chat_state: Optional[List[Dict[str, Any]]]) -> str:
    """Curated on-topic turn when the model fails quality checks after retries."""
    name = str(speaker.get("name") or "").strip()
    options = list(_fallback_turn_options(speaker))
    if not options:
        return _generic_fallback_turn(speaker, chat_state)
    start = _speaker_turn_index(chat_state, name)
    completed = _completed_chat_turns(chat_state)
    for offset in range(len(options)):
        candidate = options[(start + offset) % len(options)]
        if not _is_repetitive_turn(candidate, completed, name):
            return candidate
    return DEBATE_PASS_TOKEN


def _speaker_last_turn_content(
    chat_state: Optional[List[Dict[str, Any]]],
    speaker_name: str,
) -> str:
    target = str(speaker_name or "").strip()
    for item in reversed(_completed_chat_turns(chat_state)):
        if _turn_speaker_name(item) == target:
            return str(item.get("content") or "").strip()
    return ""


def _is_substantive_less_turn(
    text: str,
    raw: str,
    speaker: Dict[str, Any],
    chat_state: List[Dict[str, Any]],
    *,
    phase: Optional[str] = None,
) -> bool:
    """True when a model turn should be omitted from the transcript (pass/skip)."""
    if _is_debate_pass_turn(text, raw):
        return True
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if len(cleaned) < 24:
        return True
    if _DEBATE_PLACEHOLDER_TURN_RE.match(cleaned):
        return True
    if _is_meta_filler_turn(cleaned):
        return True
    name = str(speaker.get("name") or "")
    prior_content = _speaker_last_turn_content(chat_state, name)
    if prior_content:
        if _text_similarity(cleaned, prior_content) >= 0.62:
            return True
        if _is_near_copy(cleaned, prior_content, min_len=40):
            return True
    phase_key = str(phase or "").strip().lower()
    turn_mode = _normalize_turn_mode(phase_key)
    later_phase = turn_mode in {"develop", "conclusion"} or _speaker_turn_index(chat_state, name) > 0
    if later_phase and not _turn_quality_ok(speaker, cleaned, chat_state, phase=phase):
        return True
    return False


def _stream_signals_pass(raw: str) -> bool:
    """Detect an explicit pass token early during streaming (before turn commit)."""
    return _debate_signaled_pass(raw or "")


def _parse_debate_model_output(
    speaker: Dict[str, Any],
    raw_output: str,
    prior_state: List[Dict[str, Any]],
    opponents: List[str],
    *,
    phase: Optional[str] = None,
) -> Tuple[Optional[str], bool, bool]:
    """Parse one model attempt without applying curated fallbacks.

    Returns ``(content, ended_debate, explicit_pass)``. ``content`` is ``None`` when
    the attempt should be retried or replaced by a fallback.
    """
    names = list(opponents or [])
    ended = _debate_signaled_stop(raw_output)
    final_text = _clean_debate_reply(sanitize_model_output(raw_output), names)
    if _is_substantive_less_turn(final_text, raw_output, speaker, prior_state, phase=phase):
        return None, ended, True
    if final_text and _turn_quality_ok(speaker, final_text, prior_state, phase=phase):
        return final_text, ended, False
    return None, ended, False


def _debate_stance_lock(speaker: Dict[str, Any]) -> str:
    """Infer a locked debate position from speaker name/profile (e.g. Tolstoy vs Dostoevsky)."""
    blob = f"{speaker.get('name', '')} {speaker.get('profile', '')}".lower()
    if "tolstoy" in blob and "dostoevsky advocate" not in blob:
        return (
            "LOCKED POSITION: Leo Tolstoy is the greatest Russian author. "
            "Never argue that Dostoevsky or Chekhov is greater. "
            "Anna Karenina and War and Peace are Tolstoy — not Dostoevsky."
        )
    if "dostoevsky" in blob:
        return (
            "LOCKED POSITION: Fyodor Dostoevsky is the greatest Russian author. "
            "Never argue that Tolstoy or Chekhov is greater. "
            "Crime and Punishment and The Brothers Karamazov are Dostoevsky — not Tolstoy."
        )
    if "chekhov" in blob:
        return (
            "LOCKED POSITION: Anton Chekhov is the greatest Russian author. "
            "Never argue that Tolstoy or Dostoevsky is greater."
        )
    return ""


def _has_author_mixups(text: str) -> bool:
    """Detect common wrong author↔work pairings (small models do this often)."""
    t = (text or "").lower()
    mixups = (
        ("dostoevsky", "anna karenina"),
        ("dostoevsky", "war and peace"),
        ("tolstoy", "crime and punishment"),
        ("tolstoy", "brothers karamazov"),
        ("tolstoy", "karamazov"),
        ("chekhov", "war and peace"),
        ("chekhov", "crime and punishment"),
    )
    return any(a in t and w in t for a, w in mixups)


def _violates_stance_lock(speaker: Dict[str, Any], text: str) -> bool:
    """True if a speaker argues for the wrong side (e.g. Tolstoy advocate crowns Dostoevsky)."""
    blob = f"{speaker.get('name', '')} {speaker.get('profile', '')}".lower()
    t = (text or "").lower()
    if _has_author_mixups(text):
        return True
    wrong_greatest = r".{0,120}\b(?:greatest|greatest russian|positions him as the greatest)\b"
    if "tolstoy" in blob and "dostoevsky advocate" not in blob:
        if re.search(rf"\bdostoevsky\b{wrong_greatest}|\bgreatest\b.{0,120}\bdostoevsky\b", t):
            return True
        if re.search(r"\bdostoevsky\b.{0,80}\b(?:greater|better|superior)\b", t):
            return True
    if "dostoevsky" in blob:
        if re.search(rf"\btolstoy\b{wrong_greatest}|\bgreatest\b.{0,120}\btolstoy\b", t):
            return True
        if re.search(r"\btolstoy\b.{0,80}\b(?:greater|better|superior)\b", t):
            return True
        if re.search(r"\btolstoy(?:'s)?\b.{0,100}\b(?:strongest case|makes him|crowns him)\b", t):
            return True
    if "chekhov" in blob:
        if re.search(rf"\b(?:tolstoy|dostoevsky)\b{wrong_greatest}", t):
            return True
    return False


def _looks_like_rebuttal_only(text: str) -> bool:
    """True when a first-turn reply is pure rebuttal/meta with no opening claim."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    has_claim = bool(re.search(r"\bclaim\s*:", lowered))
    has_evidence = bool(
        re.search(
            r"\b(?:war and peace|anna karenina|crime and punishment|brothers karamazov|"
            r"karamazov|tolstoy|dostoevsky|chekhov)\b",
            lowered,
        )
    )
    rebuttal_lead = lowered.startswith("rebuttal:") or bool(
        re.match(r"^(?:that (?:last|prior) point|you have not|repeating the same)\b", lowered)
    )
    if rebuttal_lead and not has_evidence and not has_claim:
        return True
    return _is_meta_filler_turn(text)


def _turn_quality_ok(
    speaker: Dict[str, Any],
    text: str,
    chat_state: List[Dict[str, Any]],
    *,
    phase: Optional[str] = None,
) -> bool:
    name = str(speaker.get("name") or "")
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) < 24:
        return False
    lowered = cleaned.lower()
    if lowered.startswith("transcript so far") or lowered == "_(no output)_":
        return False
    if _DEBATE_PLACEHOLDER_TURN_RE.match(cleaned):
        return False
    if _is_meta_filler_turn(cleaned):
        return False
    phase_key = str(phase or "").strip().lower()
    turn_mode = _normalize_turn_mode(phase_key) if phase_key else None
    opening_turn = turn_mode == "proposal" if turn_mode else _is_debate_opening_turn(chat_state, name)
    profile = str(speaker.get("profile") or "")
    opening_challenge = _opening_phase_challenge_turn(
        chat_state, name, profile, phase=phase_key or "opening",
    )
    if opening_turn and not opening_challenge and _looks_like_rebuttal_only(cleaned):
        return False
    if _violates_stance_lock(speaker, cleaned):
        return False
    prior = _completed_chat_turns(chat_state)
    if _is_repetitive_turn(cleaned, prior, name):
        return False
    # Reject obvious prompt-echo / template debris from small models.
    if re.search(r"\bnew evidence:\s*new evidence:\b", lowered):
        return False
    return True


def _is_near_copy(new_text: str, prior_text: str, *, min_len: int = 70) -> bool:
    prior = (prior_text or "").strip()
    new = (new_text or "").strip()
    if len(prior) < min_len or len(new) < min_len:
        return False
    probe = prior[: min(220, len(prior))].lower()
    return probe in new.lower()


def _dedupe_structured_sections(text: str) -> str:
    """If the model repeats REBUTTAL/NEW EVIDENCE/CLAIM blocks, keep only the first set."""
    if not text:
        return text
    markers = list(re.finditer(r"\b(REBUTTAL|NEW EVIDENCE|CLAIM)\s*:", text, re.I))
    if len(markers) <= 3:
        return text.strip()
    # Cut at the second REBUTTAL (model started the template over).
    second_rebuttal = [m for m in markers if m.group(1).upper() == "REBUTTAL"]
    if len(second_rebuttal) >= 2:
        return text[: second_rebuttal[1].start()].strip()
    return text.strip()


def _strip_trailing_bracket_marker(text: str) -> str:
    """Drop trailing garbled bracket end-markers (mangled stop tokens) from a reply."""
    if not text:
        return text
    result = text.rstrip()
    for _ in range(3):
        stripped = _TRAILING_BRACKET_MARKER_RE.sub("", result).rstrip()
        if stripped == result:
            break
        result = stripped
    return result


def _truncate_at_next_speaker(text: str, names: Optional[List[str]]) -> str:
    """Cut a reply at the first point the model starts another speaker's turn.

    Small local models often continue the "Name: text" chat pattern and hallucinate a
    whole multi-turn dialogue (including fake ``Human:`` turns) in a single generation.
    We keep only the current speaker's own contribution — everything up to the first
    line that begins with a known speaker name (or ``Human``) followed by a colon — and
    let the loop drive the actual back-and-forth.
    """
    if not text:
        return text
    markers = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    markers.add("human")
    lines = text.split("\n")
    kept: List[str] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*([^:\n]{1,48}):\s", line)
        if idx > 0 and match and match.group(1).strip().lower() in markers:
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_bracket_speaker_hallucinations(text: str, names: Optional[List[str]] = None) -> str:
    """Remove bracket-prefixed fake transcript chunks like ``[Dostoevsky Advocate: ...]``."""
    if not text:
        return text
    known = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    result = text.strip()
    # Peel leading [Speaker: ...] wrappers the model echoes.
    while True:
        match = re.match(r"^\s*\[([^\]]+)\]\s*", result)
        if not match:
            break
        result = result[match.end() :].strip()
    # Cut at the first embedded bracket turn (another speaker or Turn N:).
    for match in _BRACKET_SPEAKER_HALLUCINATION_RE.finditer(result):
        if match.start() <= 0:
            continue
        label = match.group(1).strip().lower()
        if label in known or "advocate" in label or label.startswith("turn"):
            return result[: match.start()].strip()
    return result


def _clean_debate_reply(text: str, names: Optional[List[str]]) -> str:
    """Full post-processing for one debate turn: strip the stop token, drop an echoed
    leading ``Name:`` prefix, truncate any hallucinated later turns, dedupe repeated
    structured sections, strip bracket hallucinations, and remove garbled trailing bracket markers."""
    cleaned = _strip_pass_token(_strip_stop_token(text or ""))
    cleaned = _strip_leading_speaker_prefix(cleaned, names)
    cleaned = _strip_bracket_speaker_hallucinations(cleaned, names)
    cleaned = _truncate_at_next_speaker(cleaned, names)
    cleaned = _dedupe_structured_sections(cleaned)
    cleaned = _strip_trailing_bracket_marker(cleaned)
    return cleaned.strip()


def _normalize_for_similarity(text: str) -> set[str]:
  words = re.findall(r"[a-z0-9']+", (text or "").lower())
  return {w for w in words if len(w) > 2 and w not in {"the", "and", "that", "this", "with", "your", "you", "are", "for", "not"}}


def _text_similarity(a: str, b: str) -> float:
    """Jaccard similarity on normalized word sets (0 = unrelated, 1 = identical)."""
    sa, sb = _normalize_for_similarity(a), _normalize_for_similarity(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _points_already_made(chat_state: Optional[List[Dict[str, Any]]], max_points: int = 10) -> List[str]:
    """Short bullets of prior non-empty turns for anti-repetition prompting."""
    points: List[str] = []
    for item in _completed_chat_turns(chat_state):
        content = " ".join(str(item.get("content") or "").split())
        speaker = _turn_speaker_name(item) or "?"
        snippet = content[:140] + ("…" if len(content) > 140 else "")
        points.append(f"- {speaker}: {snippet}")
    return points[-max_points:]


_DEBATE_KNOWN_WORKS: Tuple[str, ...] = (
    "War and Peace",
    "Anna Karenina",
    "Crime and Punishment",
    "The Brothers Karamazov",
    "Brothers Karamazov",
    "The Cherry Orchard",
    "Cherry Orchard",
    "The Seagull",
    "Uncle Vanya",
    "Three Sisters",
    "Notes from Underground",
    "The Death of Ivan Ilyich",
    "Grand Inquisitor",
    "Hadji Murad",
    "Ward No. 6",
    "The Lady with the Little Dog",
)
_DEBATE_KNOWN_AUTHORS: Tuple[str, ...] = (
    "Leo Tolstoy",
    "Tolstoy",
    "Fyodor Dostoevsky",
    "Dostoevsky",
    "Anton Chekhov",
    "Chekhov",
    "Virginia Woolf",
    "Raskolnikov",
)
_EVIDENCE_EXAMPLE_RE = re.compile(
    r"NEW EVIDENCE[:\s—-]+([^\n.]+(?:\.|$))",
    re.IGNORECASE,
)


def _evidence_already_used(
    chat_state: Optional[List[Dict[str, Any]]],
    *,
    max_items: int = 20,
) -> List[str]:
    """Extract book titles, author names, and concrete examples cited in prior turns."""
    found: List[str] = []
    seen_lower: set[str] = set()

    def _add(item: str) -> None:
        clean = " ".join((item or "").split()).strip(" .,;:")
        if not clean or len(clean) < 3:
            return
        key = clean.lower()
        if key in seen_lower:
            return
        seen_lower.add(key)
        found.append(clean)

    for item in _completed_chat_turns(chat_state):
        content = str(item.get("content") or "")
        content_lower = content.lower()
        for work in _DEBATE_KNOWN_WORKS:
            if work.lower() in content_lower:
                _add(work)
        for author in _DEBATE_KNOWN_AUTHORS:
            if re.search(r"\b" + re.escape(author) + r"\b", content, re.I):
                _add(author)
        for match in _EVIDENCE_EXAMPLE_RE.finditer(content):
            _add(match.group(1)[:120])
        for match in re.finditer(r'"([^"]{4,80})"', content):
            _add(match.group(1))
    return found[:max_items]


def _debate_alternate_search_query(
    speaker: Dict[str, Any],
    topic: str,
    used_evidence: Optional[List[str]] = None,
) -> str:
    """Build a search query skewed toward evidence not yet cited in the debate."""
    blob = f"{speaker.get('name', '')} {speaker.get('profile', '')}".lower()
    used_blob = " ".join(used_evidence or []).lower()
    alternates: List[str] = []
    if "tolstoy" in blob:
        alternates = [
            "Tolstoy Hadji Murad Sevastopol Stories lesser known works literary significance",
            "Tolstoy moral philosophy pacifism influence on Gandhi literary criticism",
            "Tolstoy realism inner consciousness War and Peace battle scenes",
        ]
    elif "dostoevsky" in blob:
        alternates = [
            "Dostoevsky Demons The Idiot polyphonic novel technique",
            "Dostoevsky religious philosophy existential influence literature",
            "Dostoevsky Raskolnikov guilt psychology literary criticism",
        ]
    elif "chekhov" in blob:
        alternates = [
            "Chekhov Ward No 6 Lady with the Little Dog themes modern short story",
            "Chekhov influence on modern drama Ibsen theatre of mood",
            "Chekhov physician laconic style ordinary life literary significance",
        ]
    topic_clean = (topic or "").strip()
    if topic_clean:
        alternates.append(f"{topic_clean[:100]} counterargument lesser known evidence")
    for alt in alternates:
        if not any(part and part.lower() in used_blob for part in alt.split()[:4]):
            return alt
    return alternates[0] if alternates else (topic_clean or "literary evidence debate")


def _debate_fresh_reference_facts(
    speaker: Dict[str, Any],
    topic: str,
    used_evidence: Optional[List[str]] = None,
) -> str:
    """One fresh search query for a new angle when a turn would pass due to repetition."""
    if not _debate_search_enabled():
        return ""
    query = _debate_alternate_search_query(speaker, topic, used_evidence)
    return _debate_search_snippets(query, max_snippets=3)


def _conversation_in_room(conversations_store: Optional[Dict[str, Any]]) -> bool:
    """True when the active conversation has entered the debate room."""
    conv = _get_active_conversation(conversations_store or {}) or {}
    return bool(conv.get("in_room"))


def _last_transcript_speaker(chat_state: Optional[List[Dict[str, Any]]]) -> str:
    completed = _completed_chat_turns(chat_state)
    if not completed:
        return ""
    return _turn_speaker_name(completed[-1])


def _last_opponent_turn(
    chat_state: Optional[List[Dict[str, Any]]],
    speaker_name: str,
) -> Tuple[str, str]:
    """Return (name, content) for the latest completed turn from a different debater."""
    target = str(speaker_name or "").strip()
    for item in reversed(_completed_chat_turns(chat_state)):
        name = _turn_speaker_name(item)
        if not name or name.lower() == "human" or name == target:
            continue
        return name, str(item.get("content") or "").strip()
    return "", ""


_PROFILE_CHALLENGE_KEYWORDS = re.compile(
    r"\b(rebut(?:tal)?|challenge|respond to|push back|counter(?:-?argue)?|refute)\b",
    re.I,
)


def _profile_specifies_challenge_behavior(profile: str) -> bool:
    """True when the speaker profile already encodes rebuttal/challenge instructions."""
    return bool(_PROFILE_CHALLENGE_KEYWORDS.search(profile or ""))


def _prior_debater_in_phase(
    chat_state: Optional[List[Dict[str, Any]]],
    speaker_name: str,
    phase: str,
) -> Tuple[str, str]:
    """Return (name, content) for the immediately prior debater in the given phase."""
    phase_key = str(phase or "opening").strip().lower()
    if phase_key not in ("opening", "proposal"):
        return "", ""
    debater_turns: List[Tuple[str, str]] = []
    for item in _completed_chat_turns(chat_state):
        name = _turn_speaker_name(item)
        if not name or name.lower() == "human":
            continue
        debater_turns.append((name, str(item.get("content") or "").strip()))
    if not debater_turns:
        return "", ""
    return debater_turns[-1]


def _opening_phase_challenge_turn(
    chat_state: Optional[List[Dict[str, Any]]],
    speaker_name: str,
    profile: str,
    *,
    phase: str,
) -> bool:
    """True when proposal/opening phase should inject default challenge-the-prior-speaker behavior."""
    if str(phase or "").strip().lower() not in ("opening", "proposal"):
        return False
    if _profile_specifies_challenge_behavior(profile):
        return False
    prior_name, prior_content = _prior_debater_in_phase(chat_state, speaker_name, phase)
    return bool(prior_name and prior_content)


def _is_repetitive_turn(
    new_text: str,
    chat_state: Optional[List[Dict[str, Any]]],
    speaker_name: str,
    *,
    threshold: float = 0.38,
) -> bool:
    """True when a turn is too similar to an earlier one (common small-model failure)."""
    cleaned = (new_text or "").strip()
    if len(cleaned) < 40:
        return False
    for item in chat_state or []:
        prior = (item.get("content") or "").strip()
        if not prior:
            continue
        if _text_similarity(cleaned, prior) >= threshold:
            return True
        if _is_near_copy(cleaned, prior):
            return True
    head = cleaned[:100]
    for item in chat_state or []:
        prior = (item.get("content") or "").strip()
        if prior and _text_similarity(head, prior[:100]) >= 0.55:
            return True
    return False


def _build_phase_debate_request(
    speaker: Dict[str, Any],
    chat_state: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temp: float,
    opponents: Optional[List[str]] = None,
    *,
    phase: str,
    anti_repeat: bool = False,
    topic_hint: str = "",
    simple_mode: bool = False,
    reference_facts: str = "",
    round_idx: Optional[int] = None,
    max_rounds: Optional[int] = None,
) -> GenerationRequest:
    """Build a structured debate turn for proposal, develop, or conclusion phase."""
    name = str(speaker.get("name") or "Speaker").strip() or "Speaker"
    profile = str(speaker.get("profile") or "").strip()
    stance_lock = _debate_stance_lock(speaker)
    prior_turns = _completed_chat_turns(chat_state)
    phase_key = str(phase or "opening").strip().lower()
    if phase_key not in DEBATE_PHASES and phase_key not in DEBATE_TURN_MODES:
        phase_key = "opening"
    turn_mode = _normalize_turn_mode(phase_key)
    is_proposal = turn_mode == "proposal"
    is_develop = turn_mode == "develop"
    is_conclusion = turn_mode == "conclusion"
    transcript = _chat_transcript_by_name(prior_turns)
    others = [o for o in (opponents or []) if o and o != name]
    last_speaker = _last_transcript_speaker(prior_turns)
    opp_name, opp_content = _last_opponent_turn(prior_turns, name)
    has_opponent_argument = bool(opp_name and opp_content)
    opening_challenge = _opening_phase_challenge_turn(prior_turns, name, profile, phase=phase_key)
    prior_debater_name, prior_debater_content = (
        _prior_debater_in_phase(prior_turns, name, phase_key) if opening_challenge else ("", "")
    )
    points = _points_already_made(prior_turns)
    used_evidence = _evidence_already_used(prior_turns)
    pacing_block = ""
    if round_idx is not None and max_rounds is not None:
        pacing = _debate_pacing_context(int(round_idx), int(max_rounds), name, prior_turns)
        pacing_block = f"\n\n{pacing['pacing_line']}"
    opponents_clause = (
        f"You are debating against: {', '.join(others)}. Pick a DISTINCT position from theirs and defend it."
        if others
        else "Argue your position clearly and rigorously."
    )
    search_awareness = ""
    if _debate_search_enabled() or reference_facts:
        search_awareness = (
            "\n\nYou may rely on reference facts from search (injected below when available). "
            "Use them to support arguments with concrete works, scenes, or historical facts. "
            "Cite each piece of evidence at most once across the whole debate."
        )
    facts_block = ""
    if reference_facts:
        facts_block = (
            "\n\nReference facts (use these; do not invent attributions):\n"
            + reference_facts
        )
    elif _debate_search_enabled():
        facts_block = (
            "\n\nReference facts from search may appear here on later turns — cite them when provided."
        )
    proposal_directive = (
        "The previous speaker stated their position. Challenge that point directly, then advance "
        "your own position with new evidence (CHALLENGE + NEW EVIDENCE + CLAIM)."
        if opening_challenge
        else (
            "State your thesis clearly. Use NEW EVIDENCE + CLAIM. No rebuttal yet — "
            "no opponent has argued against you in this phase. Set the depth bar for the debate."
        )
    )
    develop_directive = (
        "Read the transcript carefully. Choose ONE primary move:\n"
        "A) RESPOND/REBUT — if an opponent raised a provocative claim, question, or left a gap you can exploit.\n"
        "B) NEW EVIDENCE — if rebutting would repeat yourself or no fresh angle exists, deepen with new evidence.\n"
        "Either way, push the conversation deeper than the prior phase. Do not repeat evidence already used."
    )
    phase_directive = {
        "proposal": proposal_directive,
        "develop": develop_directive,
        "conclusion": (
            "SYNTHESIZE your strongest threads into a closing argument deeper than prior phases. "
            "Do not recycle opening wording verbatim. "
            "Use FINAL CLAIM (brief NEW EVIDENCE only if not yet stated)."
        ),
    }[turn_mode]
    phase_status_label = DEBATE_TURN_MODE_LABELS.get(turn_mode, turn_mode.title())
    if simple_mode:
        if is_proposal and opening_challenge:
            turn_shape = (
                "Write ONE short paragraph (3–4 sentences): challenge the previous speaker's point directly, "
                "cite ONE concrete book/scene/fact, then state your thesis (CHALLENGE + NEW EVIDENCE + CLAIM)."
            )
        elif is_proposal:
            turn_shape = (
                "Write ONE short paragraph (3–4 sentences): state your position clearly, "
                "cite ONE concrete book/scene/fact (NEW EVIDENCE + CLAIM). No rebuttal."
            )
        elif is_conclusion:
            turn_shape = (
                "Write ONE short paragraph (3–4 sentences): closing summary of your case "
                "without repeating verbatim; end with FINAL CLAIM."
            )
        else:
            turn_shape = (
                "Write ONE short paragraph (3–4 sentences): either rebut/respond to a gap in the opponent's case "
                "OR add new evidence — whichever deepens the debate more (RESPOND/REBUT or NEW EVIDENCE + CLAIM)."
            )
        system = (
            f"You are \"{name}\" in a live structured debate. {opponents_clause}\n\n"
            f"Phase: {phase_status_label}. {phase_directive}\n\n"
            f"{turn_shape} First person only; "
            "never prefix with a name; never copy prior wording; never attribute a novel to the wrong author. "
            f"If you have nothing substantively new to add, respond with only {DEBATE_PASS_TOKEN}."
            + pacing_block
            + search_awareness
            + facts_block
            + (f"\n\n{stance_lock}" if stance_lock else "")
            + (f"\n\nPersona:\n{profile}" if profile else "")
            + (f"\n\nDebate topic:\n{topic_hint}" if topic_hint else "")
        )
    else:
        if is_proposal and opening_challenge:
            structure = (
                "Your opening turn MUST include exactly these three parts (use short paragraphs):\n"
                "1) CHALLENGE — one direct pushback on the previous speaker's argument.\n"
                "2) NEW EVIDENCE — one concrete example (book, scene, character, or historical fact).\n"
                "3) CLAIM — your thesis in one sharp sentence."
            )
        elif is_proposal:
            structure = (
                "Your opening turn MUST include exactly these two parts (use short paragraphs):\n"
                "1) NEW EVIDENCE — one concrete example (book, scene, character, or historical fact).\n"
                "2) CLAIM — your thesis in one sharp sentence.\n\n"
                "Do NOT include REBUTTAL on your opening turn."
            )
        elif is_conclusion:
            structure = (
                "Your closing turn MUST include:\n"
                "1) A brief synthesis of your strongest points (do NOT repeat prior turns verbatim).\n"
                "2) FINAL CLAIM — your closing thesis in one sharp sentence."
            )
        else:
            structure = (
                "Each develop turn MUST choose ONE primary move (use short paragraphs):\n"
                "1) RESPOND/REBUT — direct pushback on the latest opposing point or a gap they left, OR\n"
                "2) NEW EVIDENCE — a concrete example not mentioned before that deepens your case.\n"
                "3) CLAIM — your updated thesis in one sharp sentence."
            )
        system = (
            f"You are \"{name}\" in a live structured debate. {opponents_clause} "
            "WIN by making the strongest case — do not agree just to be agreeable.\n\n"
            f"Phase: {phase_status_label}. {phase_directive}\n\n"
            + structure
            + "\n\nRules: first person only; never prefix with a name; never copy prior wording; "
            + (
                "never paraphrase the opponent's last message back at them; "
                if (has_opponent_argument and is_develop) or opening_challenge
                else ""
            )
            + "stay under ~120 words; "
            "never attribute a novel to the wrong author.\n"
            + (
                f"If you have no new evidence or rebuttal beyond what is already in the transcript, "
                f"respond with only {DEBATE_PASS_TOKEN} — passing is preferred over repeating yourself.\n"
                if is_develop or opening_challenge
                else f"If you cannot state an opening position, respond with only {DEBATE_PASS_TOKEN}.\n"
                if is_proposal
                else f"If you have nothing new for your closing, respond with only {DEBATE_PASS_TOKEN}.\n"
            )
            + f"End with {DEBATE_STOP_TOKEN} only if the debate is truly finished."
            + pacing_block
            + search_awareness
            + facts_block
            + (f"\n\n{stance_lock}" if stance_lock else "")
            + (f"\n\nPersona:\n{profile}" if profile else "")
            + (f"\n\nDebate topic context:\n{topic_hint}" if topic_hint else "")
        )
    if transcript:
        user_parts = [
            "Transcript so far:\n" + transcript,
        ]
        if points and (is_develop or is_conclusion):
            user_parts.append(
                "Points already made (do NOT repeat these ideas or phrasing):\n"
                + "\n".join(points)
            )
        if used_evidence:
            user_parts.append(
                "Evidence already used (do NOT repeat):\n"
                + "\n".join(f"- {item}" for item in used_evidence)
            )
        if has_opponent_argument and is_develop:
            user_parts.append(
                f"Latest opposing argument ({opp_name}) — read carefully:\n{opp_content[:900]}"
            )
            user_parts.append(
                "Before you write: what did the opponent NOT address? "
                "If you see a gap or provocative claim, RESPOND/REBUT; otherwise deepen with NEW EVIDENCE."
            )
        elif opening_challenge and prior_debater_name and prior_debater_content:
            user_parts.append(
                f"The previous speaker ({prior_debater_name}) argued:\n{prior_debater_content[:900]}\n\n"
                "Challenge that point directly, then advance your own position with new evidence."
            )
        elif last_speaker and last_speaker != name and is_develop:
            user_parts.append(f"Reply directly to {last_speaker}'s latest point.")
        if anti_repeat:
            if used_evidence:
                user_parts.append(
                    "IMPORTANT: Your previous attempt repeated evidence or phrasing. "
                    "Use NEW evidence NOT in the used list above — a different work, theme, "
                    "scene, or historical fact grounded in the reference facts."
                )
            else:
                user_parts.append(
                    "IMPORTANT: Your previous attempt was too repetitive or off-topic. Say something substantively NEW — "
                    "a different work, theme, or line of argument grounded in the reference facts."
                )
        if is_proposal and opening_challenge:
            user_parts.append(
                "Write your opening now (CHALLENGE → NEW EVIDENCE → CLAIM)."
            )
        elif is_proposal:
            user_parts.append(
                "State your position clearly (NEW EVIDENCE + CLAIM only; no rebuttal yet)."
            )
        elif is_conclusion:
            user_parts.append(
                "Write your closing statement now (summarize without repeating verbatim; end with FINAL CLAIM)."
            )
        else:
            user_parts.append(
                "Write your develop turn now — choose RESPOND/REBUT or NEW EVIDENCE, then CLAIM."
                if not simple_mode
                else "Write your develop turn now."
            )
        user = "\n\n".join(user_parts)
    else:
        user = "State your position clearly (NEW EVIDENCE + CLAIM only; no rebuttal yet)."
    eff_temp = float(temp or 0.0)
    if anti_repeat:
        eff_temp = min(0.85, eff_temp + 0.1)
    elif simple_mode:
        eff_temp = max(0.35, eff_temp - 0.1)
    return GenerationRequest(
        messages=[ChatMessage("system", system), ChatMessage("user", user)],
        max_tokens=max(96, int(max_tokens or 768)),
        temperature=eff_temp,
    )


def _build_adjudicator_request(
    adjudicator: Dict[str, Any],
    chat_state: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temp: float,
) -> GenerationRequest:
    """Build the adjudicator-only phase request (narrative summary write-up)."""
    name = str(adjudicator.get("name") or "Adjudicator").strip() or "Adjudicator"
    profile = str(adjudicator.get("profile") or "").strip()
    transcript = _chat_transcript_by_name(chat_state, max_chars=12000)
    topic = _debate_topic_hint(chat_state) or "the debate topic"
    system = (
        f'You are "{name}", the impartial adjudicator for this debate. '
        "You are NOT a debater — read the full transcript and produce a balanced narrative write-up.\n\n"
        f"{_CONVERSATION_SUMMARY_STRUCTURE}\n\n"
        "Be factual, specific, and impartial. Do not invent facts not in the transcript."
        + (f"\n\nAdjudicator profile:\n{profile}" if profile else "")
    )
    user = (
        f"Debate topic: {topic}\n\n"
        f"Full transcript:\n{transcript or '(empty)'}\n\n"
        "Read the full transcript and write the narrative summary using the required structure."
    )
    return GenerationRequest(
        messages=[ChatMessage("system", system), ChatMessage("user", user)],
        max_tokens=max(128, int(max_tokens or 512)),
        temperature=max(0.0, min(0.3, float(temp or 0.0))),
    )


def _build_debate_turn_request(
    speaker: Dict[str, Any],
    chat_state: Optional[List[Dict[str, Any]]],
    max_tokens: int,
    temp: float,
    opponents: Optional[List[str]] = None,
    *,
    anti_repeat: bool = False,
    topic_hint: str = "",
    simple_mode: bool = False,
    reference_facts: str = "",
    phase: Optional[str] = None,
    round_idx: Optional[int] = None,
    max_rounds: Optional[int] = None,
) -> GenerationRequest:
    """Build a debate turn request with explicit anti-repetition and rebuttal structure."""
    name = str(speaker.get("name") or "Speaker").strip() or "Speaker"
    prior_turns = _completed_chat_turns(chat_state)
    if phase:
        resolved_phase = str(phase).strip().lower()
    elif round_idx is not None and max_rounds is not None:
        resolved_phase = _debate_phase_for_round(int(round_idx), int(max_rounds))
    else:
        turn_idx = _speaker_turn_index(prior_turns, name)
        if turn_idx <= 0:
            resolved_phase = "opening" if _is_debate_opening_turn(prior_turns, name) else "rebuttal"
        elif turn_idx == 1:
            resolved_phase = "rebuttal"
        else:
            resolved_phase = "final"
    return _build_phase_debate_request(
        speaker,
        chat_state,
        max_tokens,
        temp,
        opponents=opponents,
        phase=resolved_phase,
        anti_repeat=anti_repeat,
        topic_hint=topic_hint,
        simple_mode=simple_mode,
        reference_facts=reference_facts,
        round_idx=round_idx,
        max_rounds=max_rounds,
    )


def _debate_topic_hint(chat_state: Optional[List[Dict[str, Any]]]) -> str:
    for item in chat_state or []:
        if str(item.get("speaker_name") or item.get("speaker") or "").strip().lower() == "human":
            return str(item.get("content") or "").strip()
    return ""


def _generate_debate_turn_text(
    speaker: Dict[str, Any],
    chat_state: List[Dict[str, Any]],
    opponents: List[str],
    max_tokens: int,
    temp: float,
    *,
    phase: Optional[str] = None,
    round_idx: Optional[int] = None,
    max_rounds: Optional[int] = None,
) -> str:
    """Generate one debate turn; retry with simpler prompts, then curated fallback."""
    names = list(opponents or [])
    prompt_state = _completed_chat_turns(chat_state)
    topic = _debate_topic_hint(prompt_state)
    used_evidence = _evidence_already_used(prompt_state)
    reference_facts = ""
    if _debate_search_enabled():
        reference_facts = _debate_reference_facts(speaker, topic)
    resolved_phase = phase
    if not resolved_phase and round_idx is not None and max_rounds is not None:
        resolved_phase = _debate_phase_for_round(int(round_idx), int(max_rounds))
    for attempt in range(2):
        attempt_facts = reference_facts
        if attempt > 0 and _debate_search_enabled():
            fresh = _debate_fresh_reference_facts(speaker, topic, used_evidence)
            if fresh:
                attempt_facts = fresh
        req = _build_debate_turn_request(
            speaker,
            prompt_state,
            max_tokens,
            temp,
            opponents=names,
            anti_repeat=attempt > 0,
            topic_hint=topic,
            simple_mode=attempt >= 1,
            reference_facts=attempt_facts,
            phase=resolved_phase,
            round_idx=round_idx,
            max_rounds=max_rounds,
        )
        text, _ = _generate_with_speaker_backend(speaker, req)
        cleaned = _clean_debate_reply(sanitize_model_output(text), names)
        if _is_debate_pass_turn(cleaned, text):
            return ""
        if _turn_quality_ok(speaker, cleaned, prompt_state, phase=resolved_phase):
            return cleaned
    fallback = _fallback_debate_turn(speaker, prompt_state)
    if _is_debate_pass_turn(fallback, fallback):
        return ""
    return fallback


def _generate_adjudicator_writeup(
    adjudicator: Dict[str, Any],
    chat_state: List[Dict[str, Any]],
    max_tokens: int,
    temp: float,
) -> str:
    """Generate the phase-4 adjudicator summary; fall back to neutral summarizer on failure."""
    state = _completed_chat_turns(chat_state)
    if not state:
        return ""
    try:
        req = _build_adjudicator_request(adjudicator, state, max_tokens, temp)
        text, _ = _generate_with_speaker_backend(adjudicator, req)
        cleaned = _normalize_summary_text(text)
        if cleaned and len(cleaned) >= 40:
            return cleaned
    except Exception:
        pass
    return _rule_based_conversation_summary(state, None)


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
    for item in _completed_chat_turns(chat_state):
        name = _turn_speaker_name(item)
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{name}: {content}")
    text = "\n\n".join(lines)
    if len(text) <= max_chars:
        return text
    return "[earlier transcript clipped]\n\n" + text[-max_chars:]


def _resolve_speaker_backend(speaker: Dict[str, Any]) -> Tuple[Any, str]:
    """Return (backend, kind) for a speaker, reusing module-scope caches.

    ``kind`` is "frontier" or "local". Backends are cached (`_LOCAL_BACKEND_CACHE` /
    `_FRONTIER_BACKEND_CACHE`) so repeated turns/judgments reuse the same loaded
    model or client instead of reloading weights per turn.
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
        return backend, "frontier"

    adapter = str(speaker.get("local_adapter") or "checkpoints/fe-lora-30m").strip()
    model_id = resolve_local_model_id(adapter, os.environ.get("MODEL"))
    cache_key = (str(model_id), adapter)
    backend = _LOCAL_BACKEND_CACHE.get(cache_key)
    if backend is None:
        backend = LocalMlxBackend(model_id=model_id, adapter_path=adapter)
        _LOCAL_BACKEND_CACHE[cache_key] = backend
    return backend, "local"


def _generate_with_speaker_backend(speaker: Dict[str, Any], req: "GenerationRequest") -> Tuple[str, Optional[int]]:
    """Route a GenerationRequest to a speaker's backend and return (sanitized_text, total_tokens_or_None)."""
    backend, kind = _resolve_speaker_backend(speaker)
    if kind == "frontier":
        text, usage = backend.generate(req)
        return sanitize_model_output(text).strip(), usage.get("total_tokens")
    text = backend.generate(req)
    return sanitize_model_output(text).strip(), None


def _stream_with_speaker_backend(speaker: Dict[str, Any], req: "GenerationRequest"):
    """Yield incremental text deltas from a speaker's backend for a live typing effect.

    Uses the backend's ``stream_generate`` (token-by-token for local MLX, SSE deltas
    for frontier) when available. If streaming yields nothing or is unavailable, falls
    back to a single full-text chunk via the non-streaming path. Deltas are raw model
    output; callers should sanitize the concatenated result when finalizing.
    """
    backend, _kind = _resolve_speaker_backend(speaker)
    stream = getattr(backend, "stream_generate", None)
    if stream is not None:
        produced = False
        try:
            for chunk in stream(req):
                if chunk:
                    produced = True
                    yield chunk
        except Exception:
            if produced:
                raise
            produced = False  # nothing usable streamed; fall back below
        if produced:
            return
    text, _tot = _generate_with_speaker_backend(speaker, req)
    if text:
        yield text


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


_CLAIM_EXTRACT_RE = re.compile(r"\bCLAIM\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

_CONVERSATION_SUMMARY_STRUCTURE = (
    "Write a narrative summary using exactly this structure:\n"
    "1. Opening paragraph (one sentence): Frame the debate topic and who argued what — "
    "name each debater and their opposing position. Do not use inline speaker labels "
    "(e.g. 'Researcher 2:') or meta notes about phases or end conditions.\n"
    "2. Body paragraphs (one per debater, separated by blank lines): Each must begin with "
    "'The basis of [Name]'s argument was...' and summarize substantive points from their "
    "opening and final turns.\n"
    "3. Closing paragraph: Begin with 'In summary...' — give a balanced conclusion, note "
    "any agreement, and state the outcome if judges voted.\n\n"
    "Use blank lines between paragraphs. No bullet lists, no section headings, no run-on "
    "single paragraph, and no meta junk (e.g. 'End condition:', 'debate phases completed')."
)


def _normalize_summary_text(text: str) -> str:
    """Collapse intra-line whitespace but preserve paragraph breaks."""
    raw = (text or "").strip()
    if not raw:
        return ""
    paragraphs = []
    for block in re.split(r"\n\s*\n", raw):
        line = " ".join(block.split())
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _extract_claim_or_snippet(content: str, max_len: int = 120) -> str:
    match = _CLAIM_EXTRACT_RE.search(content or "")
    if match:
        return match.group(1).strip()[:max_len]
    text = " ".join((content or "").split())
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _infer_position_label(name: str, claim: str) -> str:
    """Short position label for opening framing (e.g. 'Tolstoy' from name or claim)."""
    blob = f"{name} {claim}".lower()
    for author in ("tolstoy", "dostoevsky", "chekhov", "pushkin", "gogol"):
        if author in blob:
            return author.capitalize()
    short = " ".join(claim.split()[:8]).rstrip(".,;")
    return short or name


def _extract_speaker_argument_basis(turns: List[Dict[str, Any]], *, max_len: int = 220) -> str:
    """Substantive argument basis from a speaker's opening and final turns."""
    if not turns:
        return ""
    opening = _extract_claim_or_snippet(str(turns[0].get("content") or ""), max_len=max_len)
    if len(turns) == 1:
        return opening.rstrip(".")
    closing = _extract_claim_or_snippet(str(turns[-1].get("content") or ""), max_len=max_len)
    if closing and closing != opening:
        return f"{opening.rstrip('.')}, and in closing emphasized that {closing.rstrip('.')}"
    return opening.rstrip(".")


def _rule_based_conversation_summary(
    chat_state: Optional[List[Dict[str, Any]]],
    room_state: Optional[List[Dict[str, Any]]],
    *,
    stop_reason: Optional[str] = None,
    verdicts: Optional[List[Dict[str, Any]]] = None,
    tally: Optional[Dict[str, Any]] = None,
) -> str:
    """Offline-safe narrative synopsis of a concluded debate."""
    completed = _completed_chat_turns(chat_state)
    topic = (_debate_topic_hint(chat_state) or "the assigned topic").strip().rstrip(".")
    speakers = [n for n in _transcript_speaker_names(chat_state) if n.lower() != "human"]
    paragraphs: List[str] = []

    positions: List[Tuple[str, str]] = []
    for name in speakers[:4]:
        turns = [t for t in completed if _turn_speaker_name(t) == name]
        claim = _extract_claim_or_snippet(str(turns[0].get("content") or "")) if turns else ""
        positions.append((name, _infer_position_label(name, claim)))

    if len(positions) >= 2:
        first_name, first_pos = positions[0]
        second_name, second_pos = positions[1]
        opening = (
            f"This conversation centered around {topic}, with both sides discussing opposing opinions: "
            f"{first_name} proposed {first_pos}, while {second_name} argued for {second_pos}"
        )
        for extra_name, extra_pos in positions[2:]:
            opening += f", and {extra_name} made the case for {extra_pos}"
        opening += "."
    elif positions:
        only_name, only_pos = positions[0]
        opening = (
            f"This conversation centered around {topic}, with {only_name} arguing for {only_pos}."
        )
    else:
        opening = f"This conversation centered around {topic}."

    paragraphs.append(opening)

    for name in speakers[:4]:
        turns = [t for t in completed if _turn_speaker_name(t) == name]
        basis = _extract_speaker_argument_basis(turns)
        if basis:
            paragraphs.append(f"The basis of {name}'s argument was {basis}.")

    closing_parts = ["In summary, the exchange presented contrasting positions"]
    winner = (tally or {}).get("winner") if tally else None
    if winner:
        closing_parts.append(f"and judges rated {winner} as having made the stronger overall case")
    elif verdicts:
        closing_parts.append("and judges reviewed the transcript without a clear winner")
    elif len(speakers) >= 2:
        closing_parts.append("without either side fully conceding")
    paragraphs.append(f"{closing_parts[0]} {closing_parts[1] if len(closing_parts) > 1 else ''}.".replace("  ", " "))

    return "\n\n".join(paragraphs)


def _pick_summarizer_speaker(room_state: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    debaters = [s for s in (room_state or []) if s.get("name") and not _is_judge_speaker(s)]
    for speaker in debaters:
        if str(speaker.get("backend") or "Local") == "Frontier":
            return speaker
    return debaters[0] if debaters else None


def _build_conversation_summary_request(
    chat_state: List[Dict[str, Any]],
    *,
    stop_reason: Optional[str] = None,
    verdicts: Optional[List[Dict[str, Any]]] = None,
    tally: Optional[Dict[str, Any]] = None,
) -> GenerationRequest:
    transcript = _chat_transcript_by_name(chat_state, max_chars=8000)
    topic = _debate_topic_hint(chat_state) or "the debate topic"
    user_parts = [
        f"Debate topic: {topic}",
        f"Transcript:\n{transcript}",
    ]
    if tally and tally.get("winner"):
        user_parts.append(f"Judge outcome (for closing paragraph only): {tally['winner']} won the panel vote.")
    elif tally and tally.get("tie"):
        user_parts.append("Judge outcome (for closing paragraph only): tie.")
    user_parts.append(_CONVERSATION_SUMMARY_STRUCTURE)
    system = (
        "You are a neutral summarizer for a multi-agent debate. "
        "Be factual, specific, and readable. Do not invent facts not in the transcript. "
        "Do not mention end conditions, phase counts, or inline speaker labels."
    )
    return GenerationRequest(
        messages=[ChatMessage("system", system), ChatMessage("user", "\n\n".join(user_parts))],
        max_tokens=256,
        temperature=0.2,
    )


def _generate_conversation_summary(
    room_state: Optional[List[Dict[str, Any]]],
    chat_state: Optional[List[Dict[str, Any]]],
    *,
    stop_reason: Optional[str] = None,
    verdicts: Optional[List[Dict[str, Any]]] = None,
    tally: Optional[Dict[str, Any]] = None,
    max_tokens: int = 256,
    temp: float = 0.2,
) -> str:
    """Summarize a concluded debate; try a speaker backend, else rule-based synthesis."""
    state = _completed_chat_turns(chat_state)
    if not state:
        return ""
    speaker = _pick_summarizer_speaker(room_state)
    if speaker:
        try:
            req = _build_conversation_summary_request(
                state,
                stop_reason=stop_reason,
                verdicts=verdicts,
                tally=tally,
            )
            req.max_tokens = max(96, int(max_tokens))
            req.temperature = float(temp or 0.2)
            text, _ = _generate_with_speaker_backend(speaker, req)
            cleaned = _normalize_summary_text(text)
            if cleaned and len(cleaned) >= 40:
                return cleaned
        except Exception:
            pass
    return _rule_based_conversation_summary(
        chat_state,
        room_state,
        stop_reason=stop_reason,
        verdicts=verdicts,
        tally=tally,
    )


def _build_debate_record(
    *,
    room_state: List[Dict[str, Any]],
    chat_state: List[Dict[str, Any]],
    judge_names: List[str],
    verdicts: List[Dict[str, Any]],
    tally: Dict[str, Any],
    summary: str = "",
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
        "summary": (summary or "").strip(),
    }


def _post_conversation_to_cloud(record: Dict[str, Any], timeout_s: int = 20) -> Optional[str]:
    """Push a booked debate to the hosted Conversation DB (cloud/conversation_db).

    Reads ``FE_CONVERSATION_DB_URL`` (base URL of the deployed service) and optional
    ``FE_CONVERSATION_DB_TOKEN`` (bearer secret). Returns the public conversation URL
    on success, or None when no cloud endpoint is configured. Raises on HTTP errors
    so callers can surface the failure without losing the local JSONL copy.
    """
    base_url = (os.environ.get("FE_CONVERSATION_DB_URL") or "").strip().rstrip("/")
    if not base_url:
        return None
    token = (os.environ.get("FE_CONVERSATION_DB_TOKEN") or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{base_url}/api/conversations",
        data=json.dumps(record).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    conv_path = str(body.get("url") or "").lstrip("/")
    return f"{base_url}/{conv_path}" if conv_path else base_url


# --------------------------------------------------------------------------- #
# Model Chat: persistent conversations + lobby/room state
# --------------------------------------------------------------------------- #

def _default_conversation_fields() -> Dict[str, Any]:
    return {
        "room_state": [],
        "chat_state": [],
        "in_room": False,
        "max_rounds": DEFAULT_DEBATE_PHASES,
        "max_tokens": 768,
        "temperature": 0.5,
        "early_stop_mode": DEFAULT_EARLY_STOP_MODE,
        "summary": "",
        "debate_concluded": False,
        "debate_stop_reason": "",
    }


def _new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


def _create_conversation_record(*, title: str = "New conversation") -> Dict[str, Any]:
    now = utc_now()
    return {
        "id": _new_conversation_id(),
        "title": (title or "New conversation").strip() or "New conversation",
        "created_at": now,
        "updated_at": now,
        **_default_conversation_fields(),
    }


CONVERSATION_TITLE_MAX_LEN = 60


def _truncate_conversation_title(text: str, *, max_len: int = CONVERSATION_TITLE_MAX_LEN) -> str:
    snippet = " ".join(str(text or "").split())
    if not snippet:
        return "New conversation"
    if len(snippet) <= max_len:
        return snippet
    return snippet[: max_len - 1].rstrip() + "…"


def _derive_conversation_title(conv: Dict[str, Any]) -> str:
    if conv.get("title_manual"):
        return _truncate_conversation_title(str(conv.get("title") or ""))
    explicit = str(conv.get("title") or "").strip()
    for item in conv.get("chat_state") or []:
        if str(item.get("speaker_name") or item.get("speaker") or "") == "Human":
            snippet = str(item.get("content") or "").strip().replace("\n", " ")
            if snippet:
                return _truncate_conversation_title(snippet)
    for item in conv.get("chat_state") or []:
        snippet = str(item.get("content") or "").strip().replace("\n", " ")
        if snippet:
            return _truncate_conversation_title(snippet)
    debaters = _debater_names(conv.get("room_state"))
    if debaters:
        joined = ", ".join(debaters[:3]) + ("…" if len(debaters) > 3 else "")
        return _truncate_conversation_title(joined)
    if explicit and explicit != "New conversation":
        return _truncate_conversation_title(explicit)
    return "New conversation"


def _conversation_sidebar_label(conv: Dict[str, Any]) -> str:
    title = _derive_conversation_title(conv)
    status = "in room" if conv.get("in_room") else "lobby"
    turns = len(conv.get("chat_state") or [])
    turn_note = f" · {turns} turn(s)" if turns else ""
    return f"{title} ({status}{turn_note})"


def _conversation_list_choices(store: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    conversations = (store or {}).get("conversations") or {}
    rows: List[Tuple[str, str]] = []
    for conv_id, conv in sorted(
        conversations.items(),
        key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
        reverse=True,
    ):
        rows.append((_conversation_sidebar_label(conv), conv_id))
    return rows


def _empty_conversations_store() -> Dict[str, Any]:
    conv = _create_conversation_record()
    return {"active_id": conv["id"], "conversations": {conv["id"]: conv}}


def _load_conversations_store() -> Dict[str, Any]:
    """Load persisted conversations from local JSON (survives arena restarts)."""
    path = MODEL_CHAT_CONVERSATIONS_INDEX
    if not path.is_file():
        store = _empty_conversations_store()
        _save_conversations_store(store)
        return store
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        store = _empty_conversations_store()
        _save_conversations_store(store)
        return store
    conversations = raw.get("conversations") if isinstance(raw, dict) else None
    if not isinstance(conversations, dict) or not conversations:
        store = _empty_conversations_store()
        _save_conversations_store(store)
        return store
    active_id = str(raw.get("active_id") or "")
    if active_id not in conversations:
        active_id = next(iter(conversations))
    for conv in conversations.values():
        for key, default in _default_conversation_fields().items():
            conv.setdefault(key, default if not callable(default) else default())
        conv.setdefault("title_manual", False)
        if not conv.get("title_manual"):
            conv["title"] = _derive_conversation_title(conv)
    return {"active_id": active_id, "conversations": conversations}


def _save_conversations_store(store: Dict[str, Any]) -> None:
    MODEL_CHAT_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_id": store.get("active_id"),
        "conversations": store.get("conversations") or {},
        "saved_at": utc_now(),
    }
    write_text(MODEL_CHAT_CONVERSATIONS_INDEX, json.dumps(payload, indent=2, ensure_ascii=False))


def _get_active_conversation(store: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not store:
        return None
    conv_id = str(store.get("active_id") or "")
    return (store.get("conversations") or {}).get(conv_id)


def _set_active_conversation(store: Dict[str, Any], conv_id: str) -> Dict[str, Any]:
    if conv_id not in (store.get("conversations") or {}):
        raise KeyError(f"unknown conversation: {conv_id}")
    store["active_id"] = conv_id
    return store


def _touch_conversation(conv: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    conv.update(fields)
    conv["updated_at"] = utc_now()
    if not conv.get("title_manual"):
        conv["title"] = _derive_conversation_title(conv)
    return conv


def _rename_active_conversation(store: Dict[str, Any], new_title: str) -> Dict[str, Any]:
    conv = _get_active_conversation(store)
    if not conv:
        return store
    title = _truncate_conversation_title(new_title)
    conv["title"] = title
    conv["title_manual"] = True
    conv["updated_at"] = utc_now()
    store["conversations"][conv["id"]] = conv
    _save_conversations_store(store)
    return store


def _update_active_conversation(store: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    conv = _get_active_conversation(store)
    if not conv:
        return store
    _touch_conversation(conv, **fields)
    store["conversations"][conv["id"]] = conv
    _save_conversations_store(store)
    return store


def _add_conversation(store: Dict[str, Any], *, title: str = "New conversation") -> Dict[str, Any]:
    conv = _create_conversation_record(title=title)
    store.setdefault("conversations", {})[conv["id"]] = conv
    store["active_id"] = conv["id"]
    _save_conversations_store(store)
    return store


def _delete_conversation(store: Dict[str, Any], conv_id: str) -> Dict[str, Any]:
    """Remove a conversation from the store; ensure at least one remains."""
    target = str(conv_id or "").strip()
    conversations = dict((store or {}).get("conversations") or {})
    if not target or target not in conversations:
        return store or _empty_conversations_store()
    del conversations[target]
    if not conversations:
        fresh = _empty_conversations_store()
        store["active_id"] = fresh["active_id"]
        store["conversations"] = fresh["conversations"]
    elif str(store.get("active_id") or "") == target:
        next_id = max(
            conversations.items(),
            key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
        )[0]
        store["active_id"] = next_id
        store["conversations"] = conversations
    else:
        store["conversations"] = conversations
    _save_conversations_store(store)
    return store


def _collect_conclude_votes(
    chat_state: Optional[List[Dict[str, Any]]],
    debater_names: Optional[List[str]],
) -> set[str]:
    """Return debater names who have signaled conclude via stop token or ended_debate flag."""
    voted: set[str] = set()
    debater_set = {str(n).strip().lower() for n in (debater_names or []) if str(n).strip()}
    for item in chat_state or []:
        speaker = str(item.get("speaker_name") or item.get("speaker") or "").strip()
        if speaker.lower() not in debater_set:
            continue
        content = str(item.get("content") or "")
        if item.get("ended_debate") or _debate_signaled_stop(content):
            voted.add(speaker)
    return voted


def _render_conclude_vote_status(
    conclude_votes: set[str],
    debater_names: List[str],
    *,
    mode: str,
) -> str:
    if not debater_names:
        return ""
    parts = []
    for name in debater_names:
        mark = "✓" if name in conclude_votes else "—"
        parts.append(f"**{name}** {mark}")
    mode_label = {
        "first_signal": "first conclude signal ends debate",
        "unanimous": "all must vote to conclude",
        "majority": "majority must vote to conclude",
    }.get(mode, mode)
    return f"**Vote to conclude** ({mode_label}): " + " · ".join(parts)


def _early_stop_reached(
    conclude_votes: set[str],
    debater_names: List[str],
    *,
    mode: str = DEFAULT_EARLY_STOP_MODE,
    just_signaled: bool = False,
    just_voter: Optional[str] = None,
) -> tuple[bool, str]:
    n = len(debater_names)
    if n == 0:
        return False, ""
    count = len(conclude_votes)
    if mode == "first_signal" and just_signaled:
        who = just_voter or "A debater"
        return True, f"{who} signaled the debate is concluded."
    if mode == "majority" and count > n // 2:
        return True, f"Majority voted to conclude ({count}/{n})."
    if mode == "unanimous" and count >= n:
        return True, f"All debaters voted to conclude ({count}/{n})."
    return False, ""


def _lobby_room_visibility(in_room: bool) -> tuple[bool, bool]:
    """Return (lobby_visible, room_visible) for the lobby ↔ room panels."""
    return (not bool(in_room), bool(in_room))


def _resolve_lobby_room_state(
    room_state: Optional[List[Dict[str, Any]]],
    conversations_store: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prefer live Gradio ``room_state``; fall back to the persisted active conversation."""
    live = [dict(s) for s in (room_state or []) if isinstance(s, dict)]
    if live:
        return live
    conv = _get_active_conversation(conversations_store or {})
    return [dict(s) for s in ((conv or {}).get("room_state") or []) if isinstance(s, dict)]


def _format_room_speaker_summary(room_state: Optional[List[Dict[str, Any]]]) -> str:
    speakers = list(room_state or [])
    if not speakers:
        return "No speakers yet. Add a speaker profile, then enter the room."
    rows = []
    for idx, speaker in enumerate(speakers, start=1):
        tag = " `[JUDGE]`" if _is_judge_speaker(speaker) else ""
        rows.append(
            f"{idx}. `{speaker.get('name')}`{tag} — {speaker.get('backend')} — "
            f"{speaker.get('profile') or 'no profile'}"
        )
    return "\n".join(rows)


def _room_roster_note(speakers: Optional[List[Dict[str, Any]]]) -> str:
    debaters = _debater_names(speakers)
    judges = _judge_names(speakers)
    parts = []
    if debaters:
        parts.append(f"{len(debaters)} debater(s): {', '.join(debaters)}")
    if judges:
        parts.append(f"{len(judges)} judge(s): {', '.join(judges)}")
    return " · ".join(parts) if parts else "no speakers"


def _upsert_room_speaker(
    speakers: List[Dict[str, Any]],
    speaker: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool]:
    roster = [dict(s) for s in speakers or []]
    clean_name = str(speaker.get("name") or "").strip()
    replaced = False
    for idx, existing in enumerate(roster):
        if existing.get("name") == clean_name:
            roster[idx] = dict(speaker)
            replaced = True
            break
    if not replaced:
        roster.append(dict(speaker))
    return roster, replaced


def _append_archetype_speaker(
    speakers: List[Dict[str, Any]],
    preset: Dict[str, Any],
    *,
    local_adapter: str,
    frontier_model: str,
) -> tuple[List[Dict[str, Any]], str, str]:
    roster = [dict(s) for s in speakers or []]
    unique = _unique_speaker_name([s.get("name") for s in roster], str(preset["name"]))
    roster.append({
        "name": unique,
        "backend": preset.get("backend", "Local"),
        "profile": preset.get("profile", ""),
        "local_adapter": (local_adapter or "checkpoints/fe-lora-30m").strip(),
        "frontier_model": (frontier_model or os.environ.get("FRONTIER_MODEL", "")).strip(),
        "is_judge": bool(preset.get("is_judge")),
    })
    kind = "judge" if preset.get("is_judge") else "debater"
    return roster, unique, kind


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
.model-chat-sidebar {
  border-right: 1px solid var(--arena-border) !important;
  padding-right: 0.75rem !important;
}
.model-chat-archetype-row {
  align-items: flex-end !important;
  gap: 0.5rem !important;
}
.model-chat-archetype-row button {
  min-height: 2.25rem !important;
  height: 2.25rem !important;
  padding: 0.25rem 0.75rem !important;
  font-size: 0.8125rem !important;
  line-height: 1.2 !important;
  border-radius: 8px !important;
  white-space: nowrap !important;
}
.model-chat-archetype-row button.primary {
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18) !important;
}
.model-chat-advanced-settings {
  margin-top: 0.25rem !important;
}
.model-chat-advanced-settings > .label-wrap {
  margin-bottom: 0.25rem !important;
}
.model-chat-advanced-settings .model-chat-advanced-row {
  align-items: flex-start !important;
  gap: 1rem !important;
  flex-wrap: wrap !important;
}
.model-chat-advanced-settings .model-chat-advanced-row > * {
  flex: 1 1 14rem !important;
  min-width: 0 !important;
}
.model-chat-advanced-settings input[type="range"] {
  width: 100% !important;
}
.model-chat-advanced-settings input[type="number"] {
  min-width: 4.5rem !important;
  max-width: 6rem !important;
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
/* Gradio adds `.pending` (opacity: 0.2) to outputs while a generator streams.
   That made the whole chat unreadable during typing — keep it fully visible. */
.gradio-container .pending,
.gradio-container .block.pending,
.gradio-container div.pending {
  opacity: 1 !important;
}
.gradio-container .chatbot .message,
.gradio-container .chatbot .bot,
.gradio-container .chatbot .user {
  color: var(--arena-text) !important;
}
.gradio-container .chatbot .user {
  background: rgba(214, 173, 75, 0.12) !important;
}
.gradio-container .chatbot .bot {
  background: rgba(94, 196, 189, 0.08) !important;
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
        return _chat_transcript_by_name(chat_state, max_chars)

    def _chat_request(
        speaker,
        chat_state,
        max_tokens,
        temp,
        opponents=None,
        anti_repeat=False,
        *,
        phase=None,
        round_idx=None,
        max_rounds=None,
    ):
        topic = _debate_topic_hint(chat_state)
        used_evidence = _evidence_already_used(chat_state)
        if _debate_search_enabled():
            if anti_repeat and used_evidence:
                reference_facts = _debate_fresh_reference_facts(speaker, topic, used_evidence)
            else:
                reference_facts = _debate_reference_facts(speaker, topic)
        else:
            reference_facts = ""
        return _build_debate_turn_request(
            speaker,
            chat_state,
            int(max_tokens or 768),
            float(temp or 0.0),
            opponents=opponents,
            anti_repeat=anti_repeat,
            topic_hint=topic,
            simple_mode=anti_repeat,
            reference_facts=reference_facts,
            phase=phase,
            round_idx=round_idx,
            max_rounds=max_rounds,
        ), _speaker_label(speaker)

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

    def chat_clear_ui(conversations_store):
        store = _update_active_conversation(
            dict(conversations_store or {}),
            chat_state=[],
            summary="",
            debate_concluded=False,
            debate_stop_reason="",
        )
        conv = _get_active_conversation(store) or {}
        names = _debater_names(conv.get("room_state"))
        mode = conv.get("early_stop_mode") or DEFAULT_EARLY_STOP_MODE
        return [], [], "", _render_conclude_vote_status(set(), names, mode=mode), store, "Transcript cleared.", ""

    def _active_conv_snapshot(store, room_state, chat_state, **extra):
        fields = {
            "room_state": list(room_state or []),
            "chat_state": list(chat_state or []),
        }
        fields.update(extra)
        return _update_active_conversation(dict(store or {}), **fields)

    def _conv_list_update(store):
        choices = _conversation_list_choices(store)
        active = _get_active_conversation(store)
        active_id = active["id"] if active else (choices[0][1] if choices else None)
        return gr.update(choices=choices, value=active_id)

    def _sync_from_active_conv(store):
        conv = _get_active_conversation(store) or {}
        lobby_vis, room_vis = _lobby_room_visibility(bool(conv.get("in_room")))
        names = _debater_names(conv.get("room_state"))
        mode = conv.get("early_stop_mode") or DEFAULT_EARLY_STOP_MODE
        votes = _collect_conclude_votes(conv.get("chat_state"), names)
        judge_choices, judge_selected = _judge_checkbox_state(conv.get("room_state"))
        return (
            store,
            conv.get("room_state") or [],
            _chat_pairs(conv.get("chat_state")),
            conv.get("chat_state") or [],
            gr.update(visible=lobby_vis),
            gr.update(visible=room_vis),
            room_summary(conv.get("room_state")),
            gr.update(choices=judge_choices, value=judge_selected),
            conv.get("max_rounds", DEFAULT_DEBATE_PHASES),
            conv.get("max_tokens", 768),
            conv.get("temperature", 0.5),
            mode,
            _render_conclude_vote_status(votes, names, mode=mode),
            _conv_list_update(store),
            f"**{_derive_conversation_title(conv)}** — {'in room' if conv.get('in_room') else 'lobby'}.",
            _format_debate_summary_panel(str(conv.get("summary") or "")),
            gr.update(value=_derive_conversation_title(conv)),
        )

    def select_conversation_ui(conv_id, store):
        if not conv_id or conv_id not in (store or {}).get("conversations", {}):
            return _sync_from_active_conv(store)
        store = dict(store or {})
        _set_active_conversation(store, conv_id)
        _save_conversations_store(store)
        return _sync_from_active_conv(store)

    def new_conversation_ui(store):
        store = dict(store or {})
        _add_conversation(store)
        return _sync_from_active_conv(store)

    def rename_conversation_ui(new_title, store):
        store = dict(store or {})
        title = (new_title or "").strip()
        if not title:
            return _sync_from_active_conv(store)
        _rename_active_conversation(store, title)
        return _sync_from_active_conv(store)

    def delete_conversation_ui(conv_id, store):
        store = dict(store or {})
        target = str(conv_id or store.get("active_id") or "").strip()
        if not target:
            return _sync_from_active_conv(store)
        _delete_conversation(store, target)
        return _sync_from_active_conv(store)

    TYPING_CURSOR = DEBATE_TYPING_CURSOR

    def run_agentic_loop_ui(
        opening_message,
        room_state,
        chat_state,
        max_tokens,
        temp,
        max_rounds,
        early_stop_mode,
        conversations_store,
    ):
        """Structured debate: proposal → develop (×N−2) → conclusion (each debater once per phase), then adjudicator.

        Streams each reply into the chatbox. Stops when debaters vote to conclude
        (per ``early_stop_mode``) or ``max_rounds`` debate phases complete. Judges
        sit out the debate phases and produce the phase-4 write-up when present."""
        empty_summary = ""
        state = _hydrate_loop_chat_state(conversations_store, chat_state)
        names = _debater_names(room_state)
        mode = (early_stop_mode or DEFAULT_EARLY_STOP_MODE).strip().lower()
        if mode not in EARLY_STOP_MODES:
            mode = DEFAULT_EARLY_STOP_MODE
        opening = (opening_message or "").strip()
        if not names:
            yield _chat_pairs(state), state, opening_message, "", conversations_store, "Add at least one non-judge speaker, then Enter Room.", empty_summary
            return
        if not opening and not state:
            yield _chat_pairs(state), state, "", "", conversations_store, "Type a message to kick off the debate.", empty_summary
            return
        if opening:
            state.append({"speaker": "Human", "speaker_name": "Human", "content": opening})
            yield _chat_pairs(state), state, "", _render_conclude_vote_status(set(), names, mode=mode), conversations_store, f"Sent to {', '.join(names)}. Debate starting…", empty_summary

        phases_cap = max(1, int(max_rounds or DEFAULT_DEBATE_PHASES))
        conclude_votes = _collect_conclude_votes(state, names)
        stop_reason = None
        for round_idx in range(1, phases_cap + 1):
            phase = _debate_phase_for_round(round_idx, phases_cap)
            for name in names:
                speaker = _find_speaker(room_state, name)
                if not speaker or speaker.get("name") != name:
                    continue
                label = _speaker_label(speaker)
                prompt_state = _completed_chat_turns(state)
                req, _lbl = _chat_request(
                    speaker, prompt_state, int(max_tokens or 768), float(temp or 0.0),
                    opponents=names, phase=phase, round_idx=round_idx, max_rounds=phases_cap,
                )
                state.append({"speaker": label, "speaker_name": name, "content": "", "ended_debate": False})
                vote_status = _render_conclude_vote_status(conclude_votes, names, mode=mode)
                typing_status = _format_debate_phase_status(round_idx, phases_cap, detail=f"{name} is typing…")
                acc = ""
                last_emit = 0.0
                try:
                    for chunk in _stream_with_speaker_backend(speaker, req):
                        acc += chunk
                        if _stream_signals_pass(acc):
                            break
                        state[-1]["content"] = acc + TYPING_CURSOR
                        now = time.perf_counter()
                        if now - last_emit >= 0.05:
                            last_emit = now
                            yield _chat_pairs(state), state, "", vote_status, conversations_store, typing_status, empty_summary
                    prior_state = _completed_chat_turns(state[:-1])
                    final_text, ended, explicit_pass = _parse_debate_model_output(
                        speaker, acc, prior_state, names, phase=phase,
                    )
                    if explicit_pass:
                        state.pop()
                        status = _format_pass_turn_status(name, debug=acc)
                    elif final_text is None:
                        typing_status = _format_debate_phase_status(
                            round_idx, phases_cap, detail=f"{name} retrying (too repetitive)…",
                        )
                        yield _chat_pairs(state), state, "", vote_status, conversations_store, typing_status, empty_summary
                        retry_req, _ = _chat_request(
                            speaker, prior_state, int(max_tokens or 768), float(temp or 0.0),
                            opponents=names, anti_repeat=True,
                            phase=phase, round_idx=round_idx, max_rounds=phases_cap,
                        )
                        acc = ""
                        for chunk in _stream_with_speaker_backend(speaker, retry_req):
                            acc += chunk
                            if _stream_signals_pass(acc):
                                break
                            state[-1]["content"] = acc + TYPING_CURSOR
                            now = time.perf_counter()
                            if now - last_emit >= 0.05:
                                last_emit = now
                                yield _chat_pairs(state), state, "", vote_status, conversations_store, typing_status, empty_summary
                        final_text, ended, explicit_pass = _parse_debate_model_output(
                            speaker, acc, prior_state, names, phase=phase,
                        )
                        if explicit_pass:
                            state.pop()
                            status = _format_pass_turn_status(name, debug=acc)
                        elif final_text is None:
                            if _is_substantive_less_turn("", acc, speaker, prior_state, phase=phase):
                                state.pop()
                                status = _format_pass_turn_status(name, debug=acc)
                            else:
                                fallback = _fallback_debate_turn(speaker, prior_state)
                                if _is_debate_pass_turn(fallback, fallback):
                                    state.pop()
                                    status = _format_pass_turn_status(name, debug=acc or fallback)
                                else:
                                    state[-1]["content"] = fallback
                                    state[-1]["ended_debate"] = ended
                                    status = _format_debate_phase_status(round_idx, phases_cap, detail=f"{label} responded.")
                        else:
                            state[-1]["content"] = final_text
                            state[-1]["ended_debate"] = ended
                            status = _format_debate_phase_status(round_idx, phases_cap, detail=f"{label} responded.")
                    else:
                        state[-1]["content"] = final_text
                        state[-1]["ended_debate"] = ended
                        status = _format_debate_phase_status(round_idx, phases_cap, detail=f"{label} responded.")
                    if ended:
                        conclude_votes.add(name)
                    vote_status = _render_conclude_vote_status(conclude_votes, names, mode=mode)
                    reached, reason = _early_stop_reached(
                        conclude_votes, names, mode=mode, just_signaled=ended, just_voter=name,
                    )
                    if reached:
                        stop_reason = f"{reason} (phase {round_idx}/{phases_cap})."
                        status += " Debate concluded."
                except Exception as exc:
                    state[-1]["content"] = f"_(failed: {type(exc).__name__}: {exc})_"
                    status = _format_debate_phase_status(
                        round_idx, phases_cap, detail=f"{name} failed: `{type(exc).__name__}: {exc}`",
                    )
                store = _update_active_conversation(
                    dict(conversations_store or {}),
                    chat_state=state,
                    room_state=list(room_state or []),
                )
                yield _chat_pairs(state), state, "", vote_status, store, status, empty_summary
                if stop_reason:
                    break
            if stop_reason:
                break

        final_status = stop_reason or f"Completed {phases_cap} debate phase(s) — adjudicator next."
        adjudicator = _resolve_adjudicator(room_state)
        if adjudicator:
            adj_name = str(adjudicator.get("name") or "Adjudicator")
            adj_label = _speaker_label(adjudicator)
            yield (
                _chat_pairs(state),
                state,
                "",
                _render_conclude_vote_status(conclude_votes, names, mode=mode),
                conversations_store,
                _format_adjudicator_status(detail=f"{adj_name} is writing…"),
                empty_summary,
            )
            state.append({"speaker": adj_label, "speaker_name": adj_name, "content": "", "ended_debate": False})
            acc = ""
            last_emit = 0.0
            try:
                req = _build_adjudicator_request(
                    adjudicator, state[:-1], int(max_tokens or 768), float(temp or 0.0),
                )
                for chunk in _stream_with_speaker_backend(adjudicator, req):
                    acc += chunk
                    state[-1]["content"] = acc + TYPING_CURSOR
                    now = time.perf_counter()
                    if now - last_emit >= 0.05:
                        last_emit = now
                        yield (
                            _chat_pairs(state), state, "", _render_conclude_vote_status(conclude_votes, names, mode=mode),
                            conversations_store, _format_adjudicator_status(detail=f"{adj_name} is writing…"), empty_summary,
                        )
                writeup = " ".join((acc or "").split())
                if not writeup or len(writeup) < 40:
                    writeup = _generate_adjudicator_writeup(
                        adjudicator, state[:-1], int(max_tokens or 768), float(temp or 0.0),
                    )
                state[-1]["content"] = writeup
                adj_status = _format_adjudicator_status(detail=f"{adj_label} finished.")
            except Exception as exc:
                writeup = _generate_adjudicator_writeup(
                    adjudicator, state[:-1], int(max_tokens or 768), float(temp or 0.0),
                )
                state[-1]["content"] = writeup or f"_(adjudicator failed: {type(exc).__name__}: {exc})_"
                adj_status = _format_adjudicator_status(detail=f"{adj_name} failed — used fallback summary.")
            summary = writeup or ""
            store = _update_active_conversation(
                dict(conversations_store or {}),
                chat_state=state,
                room_state=list(room_state or []),
                summary=summary,
                debate_concluded=True,
                debate_stop_reason=final_status,
            )
            summary_panel = _format_debate_summary_panel(summary)
            yield (
                _chat_pairs(state), state, "", _render_conclude_vote_status(conclude_votes, names, mode=mode),
                store, f"{final_status} {adj_status}", summary_panel,
            )
            return

        yield (
            _chat_pairs(state),
            state,
            "",
            _render_conclude_vote_status(conclude_votes, names, mode=mode),
            conversations_store,
            "Generating conversation summary…",
            empty_summary,
        )
        summary = _generate_conversation_summary(
            room_state,
            state,
            stop_reason=final_status,
            max_tokens=int(max_tokens or 768),
            temp=float(temp or 0.0),
        )
        store = _update_active_conversation(
            dict(conversations_store or {}),
            chat_state=state,
            room_state=list(room_state or []),
            summary=summary,
            debate_concluded=True,
            debate_stop_reason=final_status,
        )
        summary_panel = _format_debate_summary_panel(summary)
        yield _chat_pairs(state), state, "", _render_conclude_vote_status(conclude_votes, names, mode=mode), store, final_status, summary_panel

    def send_message_if_in_room(
        opening_message,
        room_state,
        chat_state,
        max_tokens,
        temp,
        max_rounds,
        early_stop_mode,
        conversations_store,
    ):
        """Send handler for Enter/submit — only runs the debate loop when in the room."""
        if not _conversation_in_room(conversations_store):
            msg = (opening_message or "").strip()
            status = "Enter the room before sending a message." if msg else ""
            state = list(chat_state or [])
            yield _chat_pairs(state), state, opening_message, "", conversations_store, status, ""
            return
        yield from run_agentic_loop_ui(
            opening_message,
            room_state,
            chat_state,
            max_tokens,
            temp,
            max_rounds,
            early_stop_mode,
            conversations_store,
        )

    def continue_loop_ui(room_state, chat_state, max_tokens, temp, max_rounds, early_stop_mode, conversations_store):
        """Keep the debate going with no new human message (speakers respond to each other)."""
        yield from run_agentic_loop_ui(
            "", room_state, chat_state, max_tokens, temp, max_rounds, early_stop_mode, conversations_store,
        )

    def judge_the_debate_ui(judge_names_selected, room_state, chat_state, max_tokens, temp, conversations_store):
        speakers = list(room_state or [])
        selected = set(judge_names_selected or [])
        judges = [s for s in speakers if s.get("name") in selected and _is_judge_speaker(s)]
        if not judges:
            return "", "Select at least one judge (add a speaker with 'This speaker is a judge' checked, then Enter Room).", ""
        candidates = _transcript_speaker_names(chat_state)
        if len(candidates) < 2:
            return "", "Need at least two participants with turns in the transcript before judging.", ""
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
        conv = _get_active_conversation(conversations_store or {}) or {}
        stop_reason = str(conv.get("debate_stop_reason") or "")
        summary = _generate_conversation_summary(
            speakers,
            chat_state,
            stop_reason=stop_reason or None,
            verdicts=verdicts,
            tally=tally,
            max_tokens=int(max_tokens or 768),
            temp=float(temp or 0.0),
        )
        record = _build_debate_record(
            room_state=speakers,
            chat_state=chat_state,
            judge_names=[j.get("name") for j in judges],
            verdicts=verdicts,
            tally=tally,
            summary=summary,
        )
        try:
            append_jsonl(MODEL_CHAT_DEBATES_PATH, record)
            persist_note = f" Logged to `{MODEL_CHAT_DEBATES_PATH.relative_to(REPO)}` for training data."
        except Exception as exc:
            persist_note = f" (failed to log debate record: `{type(exc).__name__}: {exc}`)"
        cloud_note = ""
        try:
            cloud_url = _post_conversation_to_cloud(record)
            if cloud_url:
                cloud_note = f" Booked to the cloud: [{cloud_url}]({cloud_url})"
        except Exception as exc:
            cloud_note = f" (cloud push failed: `{type(exc).__name__}: {exc}` — local copy is safe)"
        _update_active_conversation(dict(conversations_store or {}), summary=summary)
        status = f"{len(judges)} judge(s) voted." + persist_note + cloud_note
        return scoreboard, status, _format_debate_summary_panel(summary)

    def room_summary(room_state) -> str:
        return _format_room_speaker_summary(room_state)

    def room_reset_ui(room_state, conversations_store):
        speakers = []
        store = _active_conv_snapshot(
            conversations_store,
            speakers,
            (_get_active_conversation(conversations_store or {}) or {}).get("chat_state") or [],
        )
        return (
            speakers,
            room_summary(speakers),
            store,
            "Room cleared.",
        )

    def room_add_speaker_ui(room_state, name, backend, profile, local_adapter, frontier_model, is_judge, conversations_store):
        speakers = _resolve_lobby_room_state(room_state, conversations_store)
        clean_name = (name or "").strip()
        if not clean_name:
            return (
                speakers,
                room_summary(speakers),
                conversations_store,
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
        speakers, replaced = _upsert_room_speaker(speakers, speaker)
        chat_state = (_get_active_conversation(conversations_store or {}) or {}).get("chat_state") or []
        store = _active_conv_snapshot(conversations_store, speakers, chat_state)
        return (
            speakers,
            room_summary(speakers),
            store,
            f"Updated `{clean_name}`." if replaced else f"Added `{clean_name}`.",
        )

    def room_enter_ui(room_state, conversations_store):
        speakers = _resolve_lobby_room_state(room_state, conversations_store)
        if not speakers:
            return (
                speakers,
                room_summary(speakers),
                gr.update(),
                gr.update(),
                conversations_store,
                gr.update(),
                gr.update(),
                "Add at least one speaker before entering the room.",
            )
        chat_state = (_get_active_conversation(conversations_store or {}) or {}).get("chat_state") or []
        store = _active_conv_snapshot(conversations_store, speakers, chat_state, in_room=True)
        judge_choices, judge_selected = _judge_checkbox_state(speakers)
        lobby_vis, room_vis = _lobby_room_visibility(True)
        return (
            speakers,
            room_summary(speakers),
            gr.update(choices=judge_choices, value=judge_selected),
            store,
            gr.update(visible=lobby_vis),
            gr.update(visible=room_vis),
            f"Entered room ({_room_roster_note(speakers)}).",
        )

    def room_exit_ui(room_state, conversations_store):
        speakers = _resolve_lobby_room_state(room_state, conversations_store)
        chat_state = (_get_active_conversation(conversations_store or {}) or {}).get("chat_state") or []
        store = _active_conv_snapshot(conversations_store, speakers, chat_state, in_room=False)
        lobby_vis, room_vis = _lobby_room_visibility(False)
        return (
            store,
            gr.update(visible=lobby_vis),
            gr.update(visible=room_vis),
            "Left the room — adjust setup in the lobby, then Enter Room again.",
        )

    def archetype_picker_sync_ui(name):
        """Preview the selected preset and toggle the custom-speaker form."""
        return (
            _archetype_preview_text(name),
            gr.update(visible=_lobby_custom_fields_visible(name)),
        )

    def room_add_archetype_ui(room_state, name, local_adapter, frontier_model, conversations_store):
        speakers = _resolve_lobby_room_state(room_state, conversations_store)
        if _is_custom_archetype_pick(name):
            return (
                speakers,
                room_summary(speakers),
                conversations_store,
                "Use **Add Or Update Speaker** below for a custom agent.",
                gr.update(visible=True),
            )
        preset = _find_archetype(name)
        if not preset:
            return (
                speakers,
                room_summary(speakers),
                conversations_store,
                "Pick a base archetype first.",
                gr.update(),
            )
        speakers, unique, kind = _append_archetype_speaker(
            speakers,
            preset,
            local_adapter=local_adapter,
            frontier_model=frontier_model,
        )
        chat_state = (_get_active_conversation(conversations_store or {}) or {}).get("chat_state") or []
        store = _active_conv_snapshot(conversations_store, speakers, chat_state)
        return (
            speakers,
            room_summary(speakers),
            store,
            f"Added `{unique}` ({kind}) from the **{preset['name']}** archetype. Press **Enter Room** when ready.",
            gr.update(),
        )

    def lobby_settings_change_ui(room_state, chat_state, max_rounds, max_tokens, temp, early_stop_mode, conversations_store):
        store = _active_conv_snapshot(
            conversations_store,
            room_state,
            chat_state,
            max_rounds=int(max_rounds or DEFAULT_DEBATE_PHASES),
            max_tokens=int(max_tokens or 768),
            temperature=float(temp or 0.5),
            early_stop_mode=(early_stop_mode or DEFAULT_EARLY_STOP_MODE),
        )
        return store

    def archetype_load_into_form_ui(name):
        loaded = _archetype_load_into_form_values(name)
        if not loaded:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                gr.update(),
            )
        picker_value, speaker_name, backend, profile, is_judge = loaded
        return (
            gr.update(value=picker_value),
            gr.update(value=speaker_name),
            gr.update(value=backend),
            gr.update(value=profile),
            gr.update(value=is_judge),
            gr.update(visible=True),
            gr.update(value=_archetype_preview_text(picker_value)),
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
                conversations_store = gr.State(_load_conversations_store())
                _boot = _load_conversations_store()
                _boot_conv = _get_active_conversation(_boot) or {}
                _boot_lobby_vis, _boot_room_vis = _lobby_room_visibility(bool(_boot_conv.get("in_room")))
                room_state = gr.State(_boot_conv.get("room_state") or [])
                chat_state = gr.State(_boot_conv.get("chat_state") or [])

                with gr.Row():
                    with gr.Column(scale=1, min_width=260, elem_classes=["model-chat-sidebar"]):
                        gr.Markdown("### Conversations")
                        conv_list = gr.Radio(
                            choices=_conversation_list_choices(_boot),
                            value=_boot.get("active_id"),
                            label="Switch conversation",
                            show_label=False,
                        )
                        with gr.Row():
                            conv_title_input = gr.Textbox(
                                label="Title",
                                show_label=False,
                                placeholder="Conversation title",
                                value=_derive_conversation_title(_boot_conv),
                                scale=4,
                            )
                            conv_rename_btn = gr.Button("Rename", scale=1, variant="secondary")
                        with gr.Row():
                            new_conv_btn = gr.Button("+ New conversation", variant="secondary", scale=2)
                            delete_conv_btn = gr.Button("Delete conversation", variant="stop", scale=2)

                    with gr.Column(scale=3):
                        conv_header = gr.Markdown(
                            f"**{_derive_conversation_title(_boot_conv)}** — "
                            f"{'in room' if _boot_conv.get('in_room') else 'lobby'}."
                        )

                        with gr.Column(visible=_boot_lobby_vis) as lobby_column:
                            gr.Markdown("## Lobby — set up before entering")
                            with gr.Accordion("Speakers & archetypes", open=True):
                                room_speaker_summary = gr.Markdown(_format_room_speaker_summary(_boot_conv.get("room_state")))
                                gr.Markdown(
                                    "Add debaters and optional judges (Impartial Judge adjudicates after the "
                                    "Debate phases (proposal → deepen → conclusion; adjudicator runs after). Expand **Advanced settings** to tune phases, early stop, "
                                    "tokens, and temperature, then **Enter Room** to start chatting."
                                )
                                _boot_archetype = (
                                    _archetype_picker_choices()[0] if _archetype_picker_choices() else None
                                )
                                with gr.Row(elem_classes=["model-chat-archetype-row"]):
                                    archetype_picker = gr.Dropdown(
                                        choices=_archetype_picker_choices(),
                                        value=_boot_archetype,
                                        label="Base archetype",
                                        scale=6,
                                    )
                                    archetype_add_btn = gr.Button(
                                        "Add specialist",
                                        variant="primary",
                                        scale=1,
                                        min_width=96,
                                    )
                                    archetype_load_btn = gr.Button(
                                        "Load Into Form",
                                        scale=1,
                                        min_width=96,
                                    )
                                archetype_preview = gr.Markdown(_archetype_preview_text(_boot_archetype))
                                with gr.Column(visible=False) as custom_speaker_column:
                                    with gr.Row():
                                        room_speaker_name = gr.Textbox(
                                            label="Speaker name",
                                            placeholder="e.g. Local implementer",
                                        )
                                        room_backend = gr.Dropdown(
                                            ["Local", "Frontier"],
                                            value="Local",
                                            label="Backend",
                                        )
                                    room_profile = gr.Textbox(
                                        label="Profile / personality",
                                        lines=3,
                                        placeholder="Role, style, constraints…",
                                    )
                                    with gr.Row():
                                        room_local_adapter = gr.Textbox(
                                            label="Local adapter",
                                            value="checkpoints/fe-lora-30m",
                                        )
                                        room_frontier_model = gr.Textbox(
                                            label="Frontier model",
                                            value=os.environ.get("FRONTIER_MODEL", ""),
                                        )
                                    room_is_judge = gr.Checkbox(
                                        label="Judge (adjudicates after debate phases; optional panel vote)",
                                        value=False,
                                    )
                                    with gr.Row():
                                        room_add_btn = gr.Button(
                                            "Add Or Update Speaker",
                                            variant="primary",
                                            min_width=120,
                                        )
                                with gr.Row():
                                    room_reset_btn = gr.Button("Clear Speakers", scale=1, min_width=96)
                            with gr.Accordion("Advanced settings", open=False, elem_classes=["model-chat-advanced-settings"]):
                                with gr.Row(elem_classes=["model-chat-advanced-row"]):
                                    loop_max_rounds = gr.Slider(
                                        1,
                                        20,
                                        value=int(_boot_conv.get("max_rounds", DEFAULT_DEBATE_PHASES)),
                                        step=1,
                                        label="Debate phases",
                                        info="Phase 1: proposal; middle phases: deepen (rebut or new evidence); last: conclusion. Adjudicator runs after.",
                                        scale=1,
                                    )
                                    early_stop_mode = gr.Dropdown(
                                        choices=list(EARLY_STOP_MODES),
                                        value=_boot_conv.get("early_stop_mode", DEFAULT_EARLY_STOP_MODE),
                                        label="Early stop when",
                                        info="Debaters vote via [[DEBATE_CONCLUDED]] in their reply.",
                                        scale=1,
                                    )
                                with gr.Row(elem_classes=["model-chat-advanced-row"]):
                                    chat_max_tokens = gr.Number(
                                        label="Max response tokens",
                                        value=int(_boot_conv.get("max_tokens", 768)),
                                        precision=0,
                                        minimum=64,
                                        maximum=8192,
                                        scale=1,
                                    )
                                    chat_temp = gr.Slider(
                                        0,
                                        1,
                                        value=float(_boot_conv.get("temperature", 0.5)),
                                        step=0.05,
                                        label="Temperature",
                                        scale=1,
                                    )
                            with gr.Row():
                                room_enter_btn = gr.Button("Enter Room", variant="primary", scale=2)
                            room_status = gr.Markdown()

                        with gr.Column(visible=_boot_room_vis) as room_column:
                            gr.Markdown("## In the room")
                            with gr.Row():
                                room_exit_btn = gr.Button("Exit room", variant="secondary", scale=1)
                            chat_box = gr.Chatbot(
                                label="Conversation",
                                height=520,
                                type="tuples",
                                value=_chat_pairs(_boot_conv.get("chat_state")),
                            )
                            chat_input = gr.Textbox(
                                label="Message",
                                lines=1,
                                max_lines=4,
                                placeholder="Message the room — press Enter to send…",
                            )
                            conclude_vote_status = gr.Markdown(
                                _render_conclude_vote_status(
                                    _collect_conclude_votes(
                                        _boot_conv.get("chat_state"),
                                        _debater_names(_boot_conv.get("room_state")),
                                    ),
                                    _debater_names(_boot_conv.get("room_state")),
                                    mode=_boot_conv.get("early_stop_mode", DEFAULT_EARLY_STOP_MODE),
                                )
                            )
                            with gr.Row():
                                chat_send_btn = gr.Button("Send & Run Loop", variant="primary", scale=3)
                                chat_continue_btn = gr.Button("Continue Loop", scale=1)
                                loop_stop_btn = gr.Button("Stop", scale=1)
                                chat_clear_btn = gr.Button("Clear transcript", scale=1)
                            chat_status = gr.Markdown()
                            debate_summary = gr.Markdown(
                                value=_format_debate_summary_panel(str(_boot_conv.get("summary") or "")),
                            )

                            with gr.Accordion("Judge the debate", open=False):
                                gr.Markdown(
                                    "Each selected judge reads the transcript and votes. Results are "
                                    "logged locally and pushed to the cloud Conversation DB when configured."
                                )
                                judge_checkboxes = gr.CheckboxGroup(
                                    choices=_judge_names(_boot_conv.get("room_state")),
                                    value=_judge_names(_boot_conv.get("room_state")),
                                    label="Judges to consult",
                                )
                                judge_run_btn = gr.Button("Judge The Debate", variant="primary")
                                judge_status = gr.Markdown()
                                judge_scoreboard = gr.Markdown()

                sync_outputs = [
                    conversations_store,
                    room_state,
                    chat_box,
                    chat_state,
                    lobby_column,
                    room_column,
                    room_speaker_summary,
                    judge_checkboxes,
                    loop_max_rounds,
                    chat_max_tokens,
                    chat_temp,
                    early_stop_mode,
                    conclude_vote_status,
                    conv_list,
                    conv_header,
                    debate_summary,
                    conv_title_input,
                ]
                room_add_outputs = [room_state, room_speaker_summary, conversations_store, room_status]
                room_enter_outputs = [
                    room_state,
                    room_speaker_summary,
                    judge_checkboxes,
                    conversations_store,
                    lobby_column,
                    room_column,
                    room_status,
                ]

                def _boot_model_chat_ui():
                    store = _load_conversations_store()
                    return _sync_from_active_conv(store)

                demo.load(_boot_model_chat_ui, outputs=sync_outputs, show_api=False)

                archetype_picker.change(
                    archetype_picker_sync_ui,
                    inputs=[archetype_picker],
                    outputs=[archetype_preview, custom_speaker_column],
                    show_api=False,
                )
                new_conv_btn.click(new_conversation_ui, inputs=[conversations_store], outputs=sync_outputs, show_api=False)
                delete_conv_btn.click(
                    delete_conversation_ui,
                    inputs=[conv_list, conversations_store],
                    outputs=sync_outputs,
                    show_api=False,
                )
                conv_list.change(select_conversation_ui, inputs=[conv_list, conversations_store], outputs=sync_outputs, show_api=False)
                conv_rename_btn.click(
                    rename_conversation_ui,
                    inputs=[conv_title_input, conversations_store],
                    outputs=sync_outputs,
                    show_api=False,
                )
                conv_title_input.submit(
                    rename_conversation_ui,
                    inputs=[conv_title_input, conversations_store],
                    outputs=sync_outputs,
                    show_api=False,
                )

                archetype_add_btn.click(
                    room_add_archetype_ui,
                    inputs=[room_state, archetype_picker, room_local_adapter, room_frontier_model, conversations_store],
                    outputs=room_add_outputs + [custom_speaker_column],
                    show_api=False,
                )
                archetype_load_btn.click(
                    archetype_load_into_form_ui,
                    inputs=[archetype_picker],
                    outputs=[
                        archetype_picker,
                        room_speaker_name,
                        room_backend,
                        room_profile,
                        room_is_judge,
                        custom_speaker_column,
                        archetype_preview,
                    ],
                    show_api=False,
                )
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
                        conversations_store,
                    ],
                    outputs=room_add_outputs,
                    show_api=False,
                )
                room_reset_btn.click(
                    room_reset_ui,
                    inputs=[room_state, conversations_store],
                    outputs=room_add_outputs,
                    show_api=False,
                )
                lobby_settings_inputs = [
                    room_state, chat_state, loop_max_rounds, chat_max_tokens, chat_temp, early_stop_mode, conversations_store,
                ]
                for setting_widget in (loop_max_rounds, chat_max_tokens, chat_temp, early_stop_mode):
                    setting_widget.change(
                        lobby_settings_change_ui,
                        inputs=lobby_settings_inputs,
                        outputs=[conversations_store],
                        show_api=False,
                    )
                room_enter_btn.click(
                    room_enter_ui,
                    inputs=[room_state, conversations_store],
                    outputs=room_enter_outputs,
                    show_api=False,
                )
                room_exit_btn.click(
                    room_exit_ui,
                    inputs=[room_state, conversations_store],
                    outputs=[conversations_store, lobby_column, room_column, room_status],
                    show_api=False,
                )

                loop_inputs = [
                    chat_input, room_state, chat_state, chat_max_tokens, chat_temp,
                    loop_max_rounds, early_stop_mode, conversations_store,
                ]
                loop_outputs = [chat_box, chat_state, chat_input, conclude_vote_status, conversations_store, chat_status, debate_summary]
                send_click = chat_send_btn.click(
                    send_message_if_in_room,
                    inputs=loop_inputs,
                    outputs=loop_outputs,
                    show_progress="hidden",
                    show_api=False,
                )
                send_submit = chat_input.submit(
                    send_message_if_in_room,
                    inputs=loop_inputs,
                    outputs=loop_outputs,
                    show_progress="hidden",
                    show_api=False,
                )
                continue_click = chat_continue_btn.click(
                    continue_loop_ui,
                    inputs=[room_state, chat_state, chat_max_tokens, chat_temp, loop_max_rounds, early_stop_mode, conversations_store],
                    outputs=loop_outputs,
                    show_progress="hidden",
                    show_api=False,
                )
                loop_stop_btn.click(
                    lambda: "Loop stopped — the current turn finishes, then it halts.",
                    inputs=None,
                    outputs=[chat_status],
                    cancels=[send_click, send_submit, continue_click],
                    show_api=False,
                )
                chat_clear_btn.click(
                    chat_clear_ui,
                    inputs=[conversations_store],
                    outputs=[chat_box, chat_state, chat_input, conclude_vote_status, conversations_store, chat_status, debate_summary],
                    show_api=False,
                )

                judge_run_btn.click(
                    judge_the_debate_ui,
                    inputs=[judge_checkboxes, room_state, chat_state, chat_max_tokens, chat_temp, conversations_store],
                    outputs=[judge_scoreboard, judge_status, debate_summary],
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

        with gr.Accordion("Grade And Complete — human rubric & save (open when you're ready to score a trial)", open=False):
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
