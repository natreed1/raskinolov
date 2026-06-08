#!/usr/bin/env python3
"""Shared applyable-output contract for economistRL coding tasks (arena-aligned)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Aligned with `game_task_arena.py` model packet + `build_game_task_pairwise_dataset.py`.
CODING_SYSTEM_PROMPT = (
    "You are a careful TypeScript engineer working in the Fallen Empire codebase "
    "(Next.js, Zustand game state, economy systems under src/lib). "
    "Return only applyable patches: unified diffs or fenced full-file blocks with "
    "repo-relative paths. Preserve exports and schemas. Include bounded formulas, "
    "clamps, and tick-order hooks for economy mechanics. No plan-only prose, "
    "commentary, or shell commands."
)

OUTPUT_FORMAT_RULES = """\
Return exactly one applyable output format:
1. Fenced full-file blocks with repo-relative paths, e.g.:

```ts path=src/lib/economy.ts
// ...
```

2. OR a single unified diff that applies with `git apply`.

Rules:
- Touch only allowed paths.
- Use `export function` / `export const` for logic changes.
- Keep changes minimal and testable; do not weaken existing tests.
- Do not include summaries, bullet plans, or markdown outside fences/diff."""

SANDBOX_OUTPUT_FORMAT_RULES = """\
Return exactly one applyable output format:
1. A single fenced full-file block for the sandbox mechanic only, with a `path=` attribute, e.g.:

```ts path=src/lib/economistRl/<task_slug>/mechanic.ts
// ...
```

2. OR a single unified diff that touches only the mechanic path (and env helpers only if required).

Rules:
- Vitest is **pre-seeded** in the worktree. Do **not** output any path under `tests/` — we grade your **mechanic** against our tests.
- Edit only `{lib}` (and shared env types under `src/lib/economistRl/envs/**` only when the task requires new shared types).
- Use `export function` for the tick entrypoint `{fn}` expected by the seeded Vitest file.
- Include bounded formulas, clamps, and deterministic tick updates matching the implementation request.
- Do not edit production modules (`empireEconomy.ts`, `useGameStore`, HUD) unless this task is explicitly arena/standard integration.
- Do not include summaries, bullet plans, or markdown outside fences/diff."""


def allowed_paths_for_task(task: dict[str, Any]) -> list[str]:
    from economist_rl_task_execution import allowed_paths_for_execution, classify_execution_mode

    if classify_execution_mode(task) == "integration":
        paths = allowed_paths_for_execution(task)
        if paths:
            return paths
    req = task.get("codebase_requirements") if isinstance(task.get("codebase_requirements"), dict) else {}
    files = [str(p).strip() for p in (req.get("relevant_files") or []) if str(p).strip()]
    globs: set[str] = set()
    for rel in files:
        globs.add(rel)
        parts = Path(rel).parts
        if len(parts) >= 2:
            globs.add(str(Path(*parts[:2])) + "/**")
        parent = str(Path(rel).parent.as_posix())
        if parent and parent != ".":
            globs.add(parent + "/**")
    globs.update({"src/**/*.ts", "src/**/*.tsx", "tests/**/*.ts", "tests/**/*.tsx"})
    return sorted(globs)


def compile_commands_for_task(task: dict[str, Any]) -> list[str]:
    from economist_rl_task_execution import verify_commands_for_task

    per_task = verify_commands_for_task(task)
    if per_task:
        return per_task[:3]
    targeted = task.get("targeted_tests") if isinstance(task.get("targeted_tests"), dict) else {}
    preferred = [str(c).strip() for c in (targeted.get("preferred_commands") or []) if str(c).strip()]
    if preferred:
        return preferred[:3]
    raw = task.get("compile_commands")
    if isinstance(raw, list) and raw:
        return [str(c).strip() for c in raw if str(c).strip()][:3]
    return ["npm run test:ml-cohort"]


def _implementation_body(task: dict[str, Any]) -> str:
    for key in ("prompt_legacy_v1", "prompt_body", "prompt_implementation"):
        raw = str(task.get(key) or "").strip()
        if not raw:
            continue
        marker = "Implementation request:"
        if marker in raw:
            chunk = raw.split(marker, 1)[1]
            for stop in ("Primary files:", "Allowed paths:", "Execution mode:"):
                if stop in chunk:
                    chunk = chunk.split(stop, 1)[0]
            return chunk.strip()
        if "You are editing" not in raw[:80]:
            return raw
    return str(task.get("prompt") or "").strip()


def coding_user_prompt(task: dict[str, Any], *, original_prompt: str | None = None) -> str:
    """Arena-style user prompt for rollout generation."""
    title = str(task.get("title") or task.get("id") or "economy task")
    task_id = str(task.get("id") or "")
    body = (original_prompt or _implementation_body(task)).strip()
    allowed = allowed_paths_for_task(task)
    allowed_lines = "\n".join(f"- `{p}`" for p in allowed)
    files = (task.get("codebase_requirements") or {}).get("relevant_files") or []
    file_hint = ", ".join(str(f) for f in files[:4]) if files else "see allowed paths"
    return f"""You are editing the Fallen Empire game repository.

Task: {title}
Task id: {task_id}

Implementation request:
{body}

Primary files: {file_hint}

Allowed paths:
{allowed_lines}

{OUTPUT_FORMAT_RULES}
"""


def sandbox_coding_user_prompt(task: dict[str, Any]) -> str:
    """Sandbox prompt: extend Fallen Empire types/mechanics + task extensions, vitest in worktree."""
    from economist_rl_sandbox_envs import env_dir_for_subsection, env_file_paths, resolve_subsection, tick_function_name
    from economist_rl_task_execution import sandbox_paths

    title = str(task.get("title") or task.get("id") or "economy task")
    task_id = str(task.get("id") or "")
    body = _implementation_body(task)
    lib, test = sandbox_paths(task)
    subsection = resolve_subsection(task)
    env_dir = env_dir_for_subsection(subsection)
    env_files = env_file_paths(subsection)
    fn = tick_function_name(task)
    from economist_rl_task_execution import apply_allowed_paths_for_task

    apply_allowed = apply_allowed_paths_for_task(task)
    allowed_lines = "\n".join(f"- `{p}`" for p in apply_allowed)
    env_lines = "\n".join(f"- `{p}`" for p in env_files)
    sandbox_rules = SANDBOX_OUTPUT_FORMAT_RULES.format(lib=lib, fn=fn)
    return f"""You are editing a disposable Fallen Empire worktree (economistRL sandbox).

Task: {title}
Task id: {task_id}

Implementation request:
{body}

This sandbox **extends the real game codebase** — use ``City``, ``Player``, and imports from
``@/types/game`` and ``@/lib/gameLoop`` (e.g. ``processEconomyTurn``) where the env package does.
Task-only fields live in ``taskExt`` or fixtures; do not invent parallel flat economy state unless
the env ``types.ts`` shows scalar lab mode.

Sandbox mechanic (you must implement or refine ``{fn}`` here):
- `{lib}`

Pre-seeded Vitest (read-only grader — do **not** output this file; make your mechanic pass it):
- `{test}`

Environment package (read seeds/fixtures; edit shared types only when required):
{env_lines}

Paths you may edit (apply):
{allowed_lines}

{sandbox_rules}

Example fence for the mechanic:

```ts path={lib}
export function {fn}(state: EconomistRlState): EconomistRlState {{
  // bounded tick update
}}
```
"""


def _slug(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return token or "mechanic"


def _pick_lib_file(task: dict[str, Any]) -> str:
    from economist_rl_task_execution import classify_execution_mode, sandbox_paths

    if classify_execution_mode(task) != "arena":
        lib, _test = sandbox_paths(task)
        return lib
    req = task.get("codebase_requirements") if isinstance(task.get("codebase_requirements"), dict) else {}
    for rel in req.get("relevant_files") or []:
        text = str(rel)
        if text.startswith("src/lib/") and text.endswith(".ts") and "*" not in text:
            return text
    for rel in req.get("relevant_files") or []:
        text = str(rel)
        if text.endswith(".ts") and "*" not in text:
            return text
    lib, _test = sandbox_paths(task)
    return lib


def _pick_test_file(task: dict[str, Any]) -> str | None:
    from economist_rl_task_execution import classify_execution_mode, sandbox_paths

    if classify_execution_mode(task) != "arena":
        _lib, test = sandbox_paths(task)
        return test
    req = task.get("codebase_requirements") if isinstance(task.get("codebase_requirements"), dict) else {}
    for rel in req.get("relevant_files") or []:
        text = str(rel)
        if text.startswith("tests/") and text.endswith(".ts") and "*" not in text:
            return text
    _lib, test = sandbox_paths(task)
    return test


def sandbox_starter_bodies(task: dict[str, Any]) -> list[dict[str, str]]:
    """Pre-seeded env package + mechanic + vitest for sandbox worktrees."""
    from economist_rl_sandbox_envs import sandbox_starter_bodies as _bodies

    lib_file = _pick_lib_file(task)
    test_file = _pick_test_file(task) or ""
    return _bodies(task, lib_file=lib_file, test_file=test_file)


def format_reference_answer_coding(task: dict[str, Any]) -> str:
    """Fenced applyable reference used for rollout oracle comparisons."""
    lib_file = _pick_lib_file(task)
    test_file = _pick_test_file(task)
    from economist_rl_sandbox_envs import mechanic_stub_body, test_stub_body

    lib_body = mechanic_stub_body(task, lib_file)
    parts = [f"```ts path={lib_file}\n{lib_body.strip()}\n```"]
    if test_file:
        parts.append(f"```ts path={test_file}\n{test_stub_body(task, lib_file=lib_file, test_file=test_file).strip()}\n```")
    return "\n\n".join(parts)


def convert_task_to_coding(task: dict[str, Any], *, source_repo: Path | None = None) -> dict[str, Any]:
    """Return a copy of task with coding prompts, references, and arena metadata."""
    from economist_rl_task_execution import (
        EXECUTION_MODE_INTEGRATION,
        INTEGRATION_KIND_SANDBOX,
        classify_execution_mode,
        classify_integration_kind,
        enrich_task_execution,
    )

    out = dict(task)
    original = str(task.get("prompt") or "")
    if not out.get("prompt_legacy_v1"):
        out["prompt_legacy_v1"] = original
    mode = classify_execution_mode(out)
    kind = classify_integration_kind(out, source_repo=source_repo)
    sandbox = mode == EXECUTION_MODE_INTEGRATION and kind == INTEGRATION_KIND_SANDBOX
    impl = _implementation_body(out) or original
    out["prompt_implementation"] = impl
    if sandbox:
        out["prompt"] = sandbox_coding_user_prompt(out)
    else:
        out["prompt"] = coding_user_prompt(out, original_prompt=impl)
    out["reference_answer_legacy_v1"] = str(task.get("reference_answer") or "")
    out["reference_answer"] = format_reference_answer_coding(out)
    out["allowed_paths"] = allowed_paths_for_task(out)
    out["compile_commands"] = compile_commands_for_task(out)
    out["output_format"] = "arena_fenced_files_or_unified_diff"
    out["apply_contract"] = "game_arena_apply_v1"
    expect = dict(out.get("expect") or {})
    expect["min_chars"] = max(int(expect.get("min_chars") or 0), 120)
    expect["max_chars"] = max(int(expect.get("max_chars") or 0), 8000)
    expect.setdefault("none_contains", [])
    none = [str(x) for x in expect["none_contains"] if str(x).strip()]
    for banned in ("ignore food", "always grow", "plan only", "summary only"):
        if banned not in none:
            none.append(banned)
    expect["none_contains"] = none
    out["expect"] = expect
    return enrich_task_execution(out, source_repo=source_repo)
