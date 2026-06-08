#!/usr/bin/env python3
"""Execution modes, sandbox scaffolds, BM25 context, and verify commands for economistRL tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

EXECUTION_MODE_ARENA = "arena"
EXECUTION_MODE_INTEGRATION = "integration"
INTEGRATION_KIND_SANDBOX = "sandbox"
INTEGRATION_KIND_STANDARD = "standard"

# Read-only Fallen Empire style context (BM25 ranks globs against the task prompt).
DEFAULT_STYLE_CONTEXT_PATHS: tuple[str, ...] = (
    "README.md",
    "package.json",
    "src/lib/empireEconomy.ts",
    "src/lib/gameLoop.ts",
    "src/types/game.ts",
    "src/store/useGameStore.ts",
    "src/components/ui/GameHUD.tsx",
    "tests/ml-cohort/**/*.ts",
)

FICTIONAL_LIB_PATHS: frozenset[str] = frozenset(
    {
        "src/lib/economy.ts",
        "src/lib/market.ts",
        "src/lib/workers.ts",
    }
)

DEFAULT_CONTEXT_MAX_CHARS = 12_000
DEFAULT_GENERATION_MAX_TOKENS = 4000

# Best local arena apply-contract LoRA (see docs/SESSION_LOG 20260430 apply-sft train).
DEFAULT_ECONOMIST_RL_INIT_ADAPTER = REPO / "checkpoints" / "fe-lora-arena-apply-sft"


def _slug(task_id: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", task_id).strip("_").lower()
    return token[:64] or "task"


def sandbox_paths(task: dict[str, Any]) -> tuple[str, str]:
    """Stable virtual mechanic + test paths for sandbox integration."""
    slug = _slug(str(task.get("id") or "task"))
    lib = f"src/lib/economistRl/{slug}/mechanic.ts"
    test = f"tests/economistRl/{slug}.test.ts"
    return lib, test


def _relevant_files(task: dict[str, Any]) -> list[str]:
    req = task.get("codebase_requirements") if isinstance(task.get("codebase_requirements"), dict) else {}
    return [str(p).strip() for p in (req.get("relevant_files") or []) if str(p).strip()]


def _file_exists_in_repo(repo: Path | None, rel: str) -> bool:
    if repo is None or not repo.is_dir():
        return False
    if any(ch in rel for ch in "*?[]"):
        return bool(list(repo.glob(rel)))
    return (repo / rel).is_file()


def classify_integration_kind(task: dict[str, Any], *, source_repo: Path | None) -> str:
    """sandbox = virtual island; standard = edit existing game files with cohort verify."""
    if str(task.get("execution_mode") or "").strip().lower() == EXECUTION_MODE_ARENA:
        return INTEGRATION_KIND_STANDARD
    files = _relevant_files(task)
    if not files or source_repo is None:
        return INTEGRATION_KIND_SANDBOX
    if any(p in FICTIONAL_LIB_PATHS for p in files):
        return INTEGRATION_KIND_SANDBOX
    if any("economistRl" in p for p in files):
        return INTEGRATION_KIND_SANDBOX
    for rel in files:
        if any(ch in rel for ch in "*?[]"):
            return INTEGRATION_KIND_SANDBOX
        if not _file_exists_in_repo(source_repo, rel):
            return INTEGRATION_KIND_SANDBOX
    return INTEGRATION_KIND_STANDARD


def classify_execution_mode(
    task: dict[str, Any],
    *,
    arena_task_ids: set[str] | None = None,
) -> str:
    """arena = use game_task_arena task spec; integration = economist sandbox or standard."""
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if str(ex.get("mode") or "").strip().lower() in {EXECUTION_MODE_ARENA, EXECUTION_MODE_INTEGRATION}:
        return str(ex["mode"]).strip().lower()
    arena_link = str(task.get("arena_task_id") or ex.get("arena_task_id") or "").strip()
    if arena_link:
        return EXECUTION_MODE_ARENA
    task_id = str(task.get("id") or "")
    if arena_task_ids and task_id in arena_task_ids:
        return EXECUTION_MODE_ARENA
    return EXECUTION_MODE_INTEGRATION


def vitest_run_command(test_path: str) -> str:
    """Run vitest from repo ``node_modules`` (worktree must link/symlink ``node_modules``).

    Avoids ``npx`` interactive install prompts when the worktree has no local install.
    """
    path = str(test_path).strip()
    return f"node_modules/.bin/vitest run {path}"


def verify_commands_for_task(task: dict[str, Any]) -> list[str]:
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    mode = classify_execution_mode(task)
    kind = str(ex.get("integration_kind") or classify_integration_kind(task, source_repo=None))
    if mode == EXECUTION_MODE_INTEGRATION and kind == INTEGRATION_KIND_SANDBOX:
        _lib, test_path = sandbox_paths(task)
        return [vitest_run_command(test_path)]
    if mode == EXECUTION_MODE_ARENA:
        raw = ex.get("verify_commands") or task.get("verify_commands")
        if isinstance(raw, list) and raw:
            return [str(c).strip() for c in raw if str(c).strip()][:4]
        return ["npm run test:ml-cohort"]
    raw = ex.get("verify_commands") or task.get("verify_commands") or task.get("compile_commands")
    if isinstance(raw, list) and raw:
        return [str(c).strip() for c in raw if str(c).strip()][:4]
    if kind == INTEGRATION_KIND_STANDARD:
        return ["npm run test:ml-cohort"]
    _lib, test_path = sandbox_paths(task)
    return [vitest_run_command(test_path)]


def editable_paths_for_task(task: dict[str, Any]) -> list[str]:
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if isinstance(ex.get("starter_paths"), list) and ex["starter_paths"]:
        return [str(p).strip() for p in ex["starter_paths"] if str(p).strip()]
    mode = classify_execution_mode(task)
    if mode == EXECUTION_MODE_ARENA:
        return list(task.get("allowed_paths") or [])
    kind = str(ex.get("integration_kind") or INTEGRATION_KIND_SANDBOX)
    if kind == INTEGRATION_KIND_STANDARD:
        paths = _relevant_files(task)
        if paths:
            return paths
    lib, _test = sandbox_paths(task)
    return [lib]


def apply_allowed_paths_for_execution(task: dict[str, Any]) -> list[str]:
    """Globs the model may write during apply (sandbox: mechanic + env only, never tests/)."""
    mode = classify_execution_mode(task)
    if mode == EXECUTION_MODE_ARENA:
        return [str(p) for p in (task.get("allowed_paths") or []) if str(p).strip()]
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if isinstance(ex.get("apply_allowed_paths"), list) and ex["apply_allowed_paths"]:
        return [str(p) for p in ex["apply_allowed_paths"] if str(p).strip()]
    kind = str(ex.get("integration_kind") or INTEGRATION_KIND_SANDBOX)
    if kind == INTEGRATION_KIND_STANDARD:
        return allowed_paths_for_execution(task)
    lib, _test = sandbox_paths(task)
    from economist_rl_sandbox_envs import env_dir_for_subsection, resolve_subsection

    env_dir = env_dir_for_subsection(resolve_subsection(task))
    return [
        lib,
        f"{env_dir}/**",
        "src/lib/economistRl/**",
    ]


apply_allowed_paths_for_task = apply_allowed_paths_for_execution


def allowed_paths_for_execution(task: dict[str, Any]) -> list[str]:
    """Allowed apply globs: sandbox island, standard relevant trees, or arena allowlist."""
    mode = classify_execution_mode(task)
    if mode == EXECUTION_MODE_ARENA:
        return [str(p) for p in (task.get("allowed_paths") or []) if str(p).strip()]
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if isinstance(ex.get("allowed_paths"), list) and ex["allowed_paths"]:
        return [str(p) for p in ex["allowed_paths"] if str(p).strip()]
    kind = str(ex.get("integration_kind") or INTEGRATION_KIND_SANDBOX)
    if kind == INTEGRATION_KIND_STANDARD:
        globs: set[str] = set()
        for rel in _relevant_files(task):
            globs.add(rel)
            parts = Path(rel).parts
            if len(parts) >= 2:
                globs.add(str(Path(*parts[:2])) + "/**")
            parent = str(Path(rel).parent.as_posix())
            if parent and parent != ".":
                globs.add(parent + "/**")
        return sorted(globs) or ["src/**/*.ts", "src/**/*.tsx", "tests/**/*.ts"]
    lib, test = sandbox_paths(task)
    from economist_rl_sandbox_envs import env_dir_for_subsection, resolve_subsection

    env_dir = env_dir_for_subsection(resolve_subsection(task))
    return [
        lib,
        test,
        f"{env_dir}/**",
        "src/lib/economistRl/**",
        "tests/economistRl/**",
    ]


def context_paths_for_task(task: dict[str, Any]) -> list[str]:
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if isinstance(ex.get("context_paths"), list) and ex["context_paths"]:
        return [str(p) for p in ex["context_paths"] if str(p).strip()]
    mode = classify_execution_mode(task)
    if mode == EXECUTION_MODE_ARENA:
        return [str(p) for p in (task.get("context_paths") or []) if str(p).strip()]
    paths = list(DEFAULT_STYLE_CONTEXT_PATHS)
    kind = str(ex.get("integration_kind") or INTEGRATION_KIND_SANDBOX)
    if kind == INTEGRATION_KIND_STANDARD:
        for rel in _relevant_files(task):
            if rel not in paths and rel not in FICTIONAL_LIB_PATHS:
                paths.insert(0, rel)
    lib, _test = sandbox_paths(task)
    from economist_rl_sandbox_envs import env_file_paths, resolve_subsection

    for env_path in env_file_paths(resolve_subsection(task)):
        if env_path not in paths:
            paths.append(env_path)
    if lib not in paths:
        paths.append(lib)
    return paths


def starter_files_for_task(task: dict[str, Any]) -> list[dict[str, str]]:
    """Starter files materialized in the worktree before apply (sandbox / integration)."""
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    mode = classify_execution_mode(task)
    kind = str(ex.get("integration_kind") or classify_integration_kind(task, source_repo=None))
    if mode == EXECUTION_MODE_INTEGRATION and kind == INTEGRATION_KIND_SANDBOX:
        from economist_rl_coding_contract import sandbox_starter_bodies

        return sandbox_starter_bodies(task)
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    if isinstance(ex.get("starter_files"), list) and ex["starter_files"]:
        out: list[dict[str, str]] = []
        for item in ex["starter_files"]:
            if isinstance(item, dict) and item.get("path") and item.get("content") is not None:
                out.append({"path": str(item["path"]), "content": str(item["content"])})
        return out
    return []


def materialize_starter_files(worktree: Path, task: dict[str, Any]) -> list[str]:
    written: list[str] = []
    for item in starter_files_for_task(task):
        rel = str(item["path"]).strip()
        if not rel or ".." in Path(rel).parts:
            continue
        dest = worktree / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(str(item["content"]), encoding="utf-8")
        written.append(rel)
    return written


def enrich_task_execution(
    task: dict[str, Any],
    *,
    source_repo: Path | None = None,
    arena_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Attach or refresh ``execution`` block and derived top-level fields."""
    out = dict(task)
    mode = classify_execution_mode(out, arena_task_ids=arena_task_ids)
    kind = INTEGRATION_KIND_STANDARD if mode == EXECUTION_MODE_ARENA else classify_integration_kind(out, source_repo=source_repo)
    lib, test = sandbox_paths(out)
    starter_paths = editable_paths_for_task({**out, "execution": {"integration_kind": kind, "mode": mode}})
    if mode == EXECUTION_MODE_INTEGRATION and kind == INTEGRATION_KIND_SANDBOX:
        from economist_rl_sandbox_envs import env_file_paths, resolve_subsection

        starter_paths = env_file_paths(resolve_subsection(out)) + [lib]
    starters: list[dict[str, str]] = []
    if mode == EXECUTION_MODE_INTEGRATION and kind == INTEGRATION_KIND_SANDBOX:
        from economist_rl_coding_contract import sandbox_starter_bodies

        starters = sandbox_starter_bodies(out)
    apply_allowed = apply_allowed_paths_for_execution({**out, "execution": {"mode": mode, "integration_kind": kind}})
    verify = verify_commands_for_task(
        {
            **out,
            "execution": {
                "mode": mode,
                "integration_kind": kind,
                "verify_commands": out.get("verify_commands"),
            },
        }
    )
    allowed = allowed_paths_for_execution({**out, "execution": {"mode": mode, "integration_kind": kind}})
    ctx_paths = context_paths_for_task({**out, "execution": {"mode": mode, "context_paths": out.get("context_paths")}})
    execution = {
        "mode": mode,
        "integration_kind": kind if mode == EXECUTION_MODE_INTEGRATION else None,
        "arena_task_id": str(out.get("arena_task_id") or "").strip() or None,
        "starter_paths": starter_paths,
        "starter_files": starters,
        "context_paths": ctx_paths,
        "context_max_chars": int(
            (out.get("execution") or {}).get("context_max_chars") or DEFAULT_CONTEXT_MAX_CHARS
        ),
        "use_bm25": bool((out.get("execution") or {}).get("use_bm25", True)),
        "verify_commands": verify,
        "allowed_paths": allowed,
        "apply_allowed_paths": apply_allowed,
        "generation_max_tokens": int(
            (out.get("execution") or {}).get("generation_max_tokens") or DEFAULT_GENERATION_MAX_TOKENS
        ),
    }
    out["execution"] = execution
    out["execution_mode"] = mode
    out["integration_kind"] = kind if mode == EXECUTION_MODE_INTEGRATION else None
    out["verify_commands"] = verify
    out["compile_commands"] = verify
    out["allowed_paths"] = allowed
    out["apply_allowed_paths"] = apply_allowed
    out["context_paths"] = ctx_paths
    out["starter_paths"] = starter_paths
    from economist_rl_vitest_goals import sync_targeted_tests_from_goals

    out["targeted_tests"] = sync_targeted_tests_from_goals(out)
    return out


def load_arena_task_specs(path: Path | None = None) -> dict[str, Any]:
    from game_task_arena import TaskSpec, load_task_specs

    specs = load_task_specs(path or (REPO / "benchmarks" / "game_task_arena_examples.json"))
    return {tid: spec for tid, spec in specs.items()}


def merge_arena_task(task: dict[str, Any], arena_specs: dict[str, Any]) -> dict[str, Any]:
    """Overlay arena task fields when execution mode is arena."""
    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    arena_id = str(task.get("arena_task_id") or ex.get("arena_task_id") or "").strip()
    if not arena_id or arena_id not in arena_specs:
        return task
    spec = arena_specs[arena_id]
    out = dict(task)
    out["arena_task_id"] = arena_id
    out["title"] = spec.title
    out["prompt_body"] = spec.prompt
    out["allowed_paths"] = list(spec.allowed_paths)
    out["context_paths"] = list(spec.context_paths)
    out["verify_commands"] = list(spec.verify_commands)
    out["preview_path"] = spec.preview_path
    out["execution"] = {
        **(ex if isinstance(ex, dict) else {}),
        "mode": EXECUTION_MODE_ARENA,
        "arena_task_id": arena_id,
        "verify_commands": list(spec.verify_commands),
        "allowed_paths": list(spec.allowed_paths),
        "context_paths": list(spec.context_paths),
        "integration_kind": None,
    }
    out["execution_mode"] = EXECUTION_MODE_ARENA
    return out


def build_rollout_user_prompt(
    task: dict[str, Any],
    *,
    source_repo: Path | None,
    context_max_chars: int | None = None,
    use_bm25: bool | None = None,
    log_dir: Path | None = None,
    include_starter_bodies_in_prompt: bool = False,
) -> str:
    """Full user prompt: task header, BM25 repo context, optional starter excerpts."""
    from economist_rl_coding_contract import coding_user_prompt
    from game_task_arena import TaskSpec, build_context_pack_for_task_spec

    ex = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    mode = classify_execution_mode(task)
    max_chars = int(context_max_chars or ex.get("context_max_chars") or DEFAULT_CONTEXT_MAX_CHARS)
    bm25 = bool(use_bm25 if use_bm25 is not None else ex.get("use_bm25", True))

    if mode == EXECUTION_MODE_ARENA:
        body = str(task.get("prompt_body") or task.get("prompt") or "")
        title = str(task.get("title") or task.get("id") or "")
        preview = str(task.get("preview_path") or "")
        header = f"""You are editing a disposable Fallen Empire worktree (arena eval task).

Task: {title}
Task id: {task.get("id")}
Preview route: `{preview or "/"}`
Arena task id: `{task.get("arena_task_id") or ""}`

User request:
{body}
"""
    else:
        kind = str(ex.get("integration_kind") or INTEGRATION_KIND_SANDBOX)
        if kind == INTEGRATION_KIND_SANDBOX:
            from economist_rl_coding_contract import sandbox_coding_user_prompt

            header = sandbox_coding_user_prompt(task)
        else:
            from economist_rl_coding_contract import coding_user_prompt

            header = coding_user_prompt(
                task,
                original_prompt=str(task.get("prompt_body") or task.get("prompt_legacy_v1") or ""),
            )

    context_section = ""
    if source_repo is not None and source_repo.is_dir():
        ctx_paths = context_paths_for_task(task)
        allowed = allowed_paths_for_execution(task)
        verify = verify_commands_for_task(task)
        spec = TaskSpec(
            id=str(task.get("id") or "economist"),
            task_type="economist_rl",
            title=str(task.get("title") or ""),
            prompt=str(task.get("prompt_body") or task.get("prompt") or "")[:4000],
            allowed_paths=allowed,
            context_paths=ctx_paths,
            verify_commands=verify,
            preview_path=str(task.get("preview_path") or ""),
            max_tokens=int(ex.get("generation_max_tokens") or DEFAULT_GENERATION_MAX_TOKENS),
        )
        pack = build_context_pack_for_task_spec(
            source_repo,
            spec,
            max_chars=max_chars,
            log_dir=log_dir,
            use_bm25=bm25,
            header_markdown=header,
        )
        context_section = pack.text
    else:
        context_section = header + "\n\n[context pack skipped: no source repo]\n"

    starters = starter_files_for_task(task)
    if include_starter_bodies_in_prompt and starters and mode == EXECUTION_MODE_INTEGRATION:
        blocks = []
        for item in starters:
            path = item["path"]
            content = item["content"].strip()
            blocks.append(f"## Starter `{path}` (pre-seeded in worktree)\n\n```ts path={path}\n{content}\n```")
        context_section += "\n\n# Starter files\n\n" + "\n\n".join(blocks)

    return context_section


def load_eval_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arena_task_as_economist_eval(arena_id: str, spec: Any) -> dict[str, Any]:
    """Wrap a game_task_arena TaskSpec as an economist eval row (execution_mode=arena)."""
    task = {
        "id": f"arena-eval-{arena_id}",
        "title": spec.title,
        "curriculum_track": "arena",
        "arena_task_id": arena_id,
        "prompt_body": spec.prompt,
        "prompt": spec.prompt,
        "allowed_paths": list(spec.allowed_paths),
        "context_paths": list(spec.context_paths),
        "verify_commands": list(spec.verify_commands),
        "preview_path": spec.preview_path,
        "execution_mode": EXECUTION_MODE_ARENA,
    }
    return enrich_task_execution(task)


def resolve_eval_tasks(
    tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
    limit: int,
    *,
    arena_specs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Stratified eval: arena specs from game_task_arena, economist rows by id, then fill."""
    specs = arena_specs if arena_specs is not None else load_arena_task_specs()
    by_id = {str(t.get("id") or ""): t for t in tasks}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(task: dict[str, Any]) -> None:
        tid = str(task.get("id") or "")
        if not tid or tid in seen:
            return
        seen.add(tid)
        selected.append(task)

    for arena_id in manifest.get("tiers", {}).get("arena") or []:
        aid = str(arena_id).strip()
        if aid in specs:
            _add(arena_task_as_economist_eval(aid, specs[aid]))
        elif aid in by_id:
            _add(by_id[aid])

    for key in ("integration_standard", "sandbox", "generalist"):
        for tid in manifest.get("tiers", {}).get(key) or []:
            tid = str(tid).strip()
            if tid in by_id:
                _add(by_id[tid])

    if len(selected) < limit:
        for task in tasks:
            if len(selected) >= limit:
                break
            _add(task)
    return selected[: max(1, min(int(limit), len(selected)))]


def select_eval_tasks_from_manifest(
    tasks: list[dict[str, Any]],
    manifest: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    return resolve_eval_tasks(tasks, manifest, limit)
