"""Final-system ablation runner over task_bank compiled manifest.

This script:
1) validates `final_mass_testing_system_v1.json`,
2) expands task refs from source task banks into arena-compatible task specs,
3) runs three ablation variants through the arena generate/apply/verify loop:
   - advanced_router_with_specialists
   - qwen_7_5b_only
   - gpt_5_5_only
4) writes side-by-side summary metrics from the new eval path only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from game_task_arena import (
    apply_output,
    attempt_applied,
    cleanup,
    create_trial,
    generate_attempt,
    load_trial,
    verify,
)
from model_router import GenerationRequest, RoutingPolicy, messages_from_prompt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "task_bank" / "compiled" / "final_mass_testing_system_v1.json"
DEFAULT_SOURCE_REPO = Path(os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))).expanduser()
DEFAULT_WORKTREE_ROOT = Path(
    os.environ.get("GAME_ARENA_ROOT", str(Path.home() / "fallen-empire-arena"))
).expanduser()
DEFAULT_ADAPTER_REGISTRY = ROOT / "training" / "adapter_registry_v1.json"
DEFAULT_TASKS_JSON = ROOT / "benchmarks" / "results" / "final_system_runtime_tasks_v1.json"
DEFAULT_ROWS_JSONL = ROOT / "benchmarks" / "results" / "final_system_ablation_rows_v1.jsonl"
DEFAULT_SUMMARY_JSON = ROOT / "benchmarks" / "results" / "final_system_ablation_summary_v1.json"
DEFAULT_SUMMARY_MD = ROOT / "benchmarks" / "results" / "final_system_ablation_summary_v1.md"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DEFAULT_FRONTIER_MODEL = "gpt-5.5"
VERIFY_ENV_CHECK_CMD = ["npm", "run", "test:ml-cohort"]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_tasks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return [row for row in tasks if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _normalize_allowed_prefixes(prefixes: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in prefixes:
        pref = str(raw).strip()
        if not pref:
            continue
        if pref.endswith("/"):
            out.append(f"{pref}**/*")
        else:
            out.append(pref)
    return out


def _derive_allowed_paths(task: Dict[str, Any]) -> List[str]:
    hard = task.get("hard_constraints") if isinstance(task.get("hard_constraints"), dict) else {}
    allowed_prefixes = hard.get("allowed_path_prefixes") if isinstance(hard, dict) else []
    if isinstance(allowed_prefixes, list) and allowed_prefixes:
        return _normalize_allowed_prefixes(allowed_prefixes)

    changed_files = task.get("changed_files")
    if isinstance(changed_files, list) and changed_files:
        return [str(p).strip() for p in changed_files if str(p).strip()]

    hints = task.get("canonical_file_hints")
    if isinstance(hints, list) and hints:
        return [str(p).strip() for p in hints if str(p).strip()]

    return [
        "src/**/*.ts",
        "src/**/*.tsx",
        "src/**/*.js",
        "src/**/*.jsx",
        "src/**/*.css",
        "app/**/*.ts",
        "app/**/*.tsx",
        "public/**/*",
        "game-server/**/*",
    ]


def _derive_context_paths(task: Dict[str, Any]) -> List[str]:
    hints = task.get("canonical_file_hints") if isinstance(task.get("canonical_file_hints"), list) else []
    changed = task.get("changed_files") if isinstance(task.get("changed_files"), list) else []
    base = ["README.md", "package.json", *hints, *changed]
    out: List[str] = []
    seen: set[str] = set()
    for raw in base:
        path = str(raw).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _derive_verify_commands(task: Dict[str, Any]) -> List[str]:
    acceptance = task.get("acceptance") if isinstance(task.get("acceptance"), dict) else {}
    verify_commands = acceptance.get("verify_commands") if isinstance(acceptance, dict) else None
    if isinstance(verify_commands, list) and verify_commands:
        return [str(cmd).strip() for cmd in verify_commands if str(cmd).strip()]
    compile_commands = task.get("compile_commands")
    if isinstance(compile_commands, list) and compile_commands:
        return [str(cmd).strip() for cmd in compile_commands if str(cmd).strip()]
    return ["npm run test:ml-cohort"]


def _derive_notes(task: Dict[str, Any]) -> str:
    parts: List[str] = []
    acceptance = task.get("acceptance") if isinstance(task.get("acceptance"), dict) else {}
    required = acceptance.get("required_outcomes") if isinstance(acceptance, dict) else None
    if isinstance(required, list) and required:
        parts.append("Required outcomes: " + "; ".join(str(x) for x in required))
    signals = task.get("acceptance_signals")
    if isinstance(signals, list) and signals:
        parts.append("Acceptance signals: " + "; ".join(str(x) for x in signals))
    anti = str(task.get("anti_overfit_notes") or "").strip()
    if anti:
        parts.append(f"Anti-overfit: {anti}")
    return "\n".join(parts)


def _build_runtime_task(task: Dict[str, Any]) -> Dict[str, Any]:
    acceptance = task.get("acceptance") if isinstance(task.get("acceptance"), dict) else {}
    preview_path = str(acceptance.get("preview_path") or "").strip()
    return {
        "id": str(task["id"]),
        "task_type": str(task.get("task_type") or "state_logic"),
        "title": str(task.get("task_title") or task.get("title") or task["id"]),
        "prompt": str(task.get("prompt") or "").strip(),
        "allowed_paths": _derive_allowed_paths(task),
        "context_paths": _derive_context_paths(task),
        "verify_commands": _derive_verify_commands(task),
        "preview_command": "next dev -H 127.0.0.1 -p {port}",
        "preview_path": preview_path,
        "max_tokens": int(task.get("max_tokens") or 4096),
        "notes": _derive_notes(task),
        "grading": [],
    }


def _compile_manifest_tasks(manifest: Dict[str, Any], manifest_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    refs = manifest.get("task_refs")
    source_files = manifest.get("source_files")
    if not isinstance(refs, list):
        raise SystemExit("Manifest missing `task_refs` list.")
    if not isinstance(source_files, list):
        raise SystemExit("Manifest missing `source_files` list.")

    source_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source_rel in source_files:
        source_path = (ROOT / str(source_rel)).resolve()
        payload = _load_json(source_path)
        tasks = _iter_tasks(payload)
        by_id: Dict[str, Dict[str, Any]] = {}
        for row in tasks:
            tid = str(row.get("id") or "").strip()
            if tid:
                by_id[tid] = row
        source_maps[str(source_rel)] = by_id

    runtime_tasks: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        source_rel = str(ref.get("source_file") or "").strip()
        task_id = str(ref.get("task_id") or "").strip()
        if not source_rel or not task_id:
            continue
        source_bucket = source_maps.get(source_rel, {})
        row = source_bucket.get(task_id)
        if not row:
            missing.append(f"{source_rel}::{task_id}")
            continue
        runtime = _build_runtime_task(row)
        runtime_tasks.append(runtime)
        metadata_rows.append(
            {
                "id": task_id,
                "source_file": source_rel,
                "domain_primary_normalized": str(ref.get("domain_primary_normalized") or ""),
                "subskill": str(ref.get("subskill") or ""),
                "domain_primary": str(row.get("domain_primary") or ""),
            }
        )
    if missing:
        raise SystemExit(
            "Manifest references missing tasks:\n"
            + "\n".join(missing[:20])
            + ("\n..." if len(missing) > 20 else "")
            + f"\nmanifest={manifest_path}"
        )
    return runtime_tasks, metadata_rows


def _load_adapter_paths(registry_path: Path) -> Dict[str, str]:
    payload = _load_json(registry_path)
    rows = payload.get("entries")
    out: Dict[str, str] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        aid = str(row.get("adapter_id") or "").strip()
        apath = str(row.get("adapter_path") or "").strip()
        if aid and apath:
            out[aid] = apath
    return out


def _resolve_adapter_path(raw_path: str) -> str:
    path = str(raw_path or "").strip()
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    return path if candidate.exists() else ""


def _tokenize_simple(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"[a-zA-Z0-9_./*-]+", text or "")]


_PROMPT_DOMAIN_HINTS: Dict[str, Tuple[str, ...]] = {
    "hud_status": (
        "hud",
        "panel",
        "ui",
        "overlay",
        "screen",
        "render",
        "sprite",
        "visual",
    ),
    "economy": (
        "economy",
        "resource",
        "production",
        "income",
        "cost",
        "wage",
        "market",
        "worker",
    ),
    "army_operations": (
        "army",
        "combat",
        "battle",
        "unit",
        "siege",
        "naval",
        "commander",
        "formation",
    ),
    "state_perstitence_integrity": (
        "save",
        "load",
        "snapshot",
        "state",
        "session",
        "auth",
        "middleware",
        "multiplayer",
        "server",
    ),
    "ai_strategy_and_planning": (
        "ai",
        "planner",
        "planning",
        "strategy",
        "tactic",
        "decision",
    ),
}


def _infer_prompt_domains(prompt: str) -> set[str]:
    text = (prompt or "").lower()
    out: set[str] = set()
    for domain, keywords in _PROMPT_DOMAIN_HINTS.items():
        if any(re.search(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", text) for kw in keywords):
            out.add(domain)
    return out


def _sparse_tf(tokens: List[str]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0.0) + 1.0
    for tok in list(vec.keys()):
        vec[tok] = float(math.log1p(vec[tok])) if vec[tok] > 0 else 0.0
    return vec


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for tok, av in a.items():
        bv = b.get(tok)
        if bv is not None:
            dot += av * bv
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _truncate_chars(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    budget = max(0, int(max_chars))
    if budget <= 0:
        return ""
    if len(value) <= budget:
        return value
    if budget <= 3:
        return value[:budget]
    return value[: budget - 3].rstrip() + "..."


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _format_limited_items(items: Iterable[str], *, max_items: int, max_chars: int) -> str:
    values = _dedupe_preserve_order(items)
    if not values:
        return "(none)"
    clipped = values[: max(0, int(max_items))]
    text = ", ".join(clipped)
    if len(text) <= max_chars:
        return text
    while clipped and len(text) > max_chars:
        clipped = clipped[:-1]
        text = ", ".join(clipped)
    if text:
        return text
    return _truncate_chars(values[0], max_chars)


def _rank_context_paths(
    *,
    paths: Iterable[str],
    prompt: str,
    domain: str,
    subskill: str,
    max_items: int,
) -> List[str]:
    unique_paths = _dedupe_preserve_order(paths)
    if not unique_paths:
        return []
    prompt_tokens = set(_tokenize_simple(prompt))
    domain_tokens = set(_tokenize_simple(domain))
    subskill_tokens = set(_tokenize_simple(subskill))

    def _score(path: str) -> float:
        path_tokens = set(_tokenize_simple(path))
        overlap_prompt = len(path_tokens & prompt_tokens)
        overlap_domain = len(path_tokens & domain_tokens)
        overlap_subskill = len(path_tokens & subskill_tokens)
        src_bonus = 1.0 if path.startswith("src/") else 0.0
        extension_bonus = 0.5 if path.endswith((".ts", ".tsx", ".js", ".jsx")) else 0.0
        return (
            overlap_prompt * 2.0
            + overlap_domain * 1.5
            + overlap_subskill * 1.0
            + src_bonus
            + extension_bonus
        )

    ranked = sorted(unique_paths, key=lambda p: (_score(p), -len(p)), reverse=True)
    return ranked[: max(0, int(max_items))]


def _load_current_rag_entries(corpus_path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(corpus_path)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        tags = [str(t).strip().lower() for t in (row.get("tags") or []) if str(t).strip()]
        retrieval_parts: List[str] = []
        retrieval_description = str(row.get("retrieval_description") or "").strip()
        if retrieval_description:
            retrieval_parts.append(retrieval_description)
        for field in (
            "retrieval_aliases",
            "retrieval_symptoms",
            "retrieval_task_signatures",
            "retrieval_examples",
        ):
            values = row.get(field)
            if isinstance(values, list):
                retrieval_parts.extend(str(v).strip() for v in values if str(v).strip())
        retrieval_text = " ".join(retrieval_parts).strip()
        if not retrieval_text:
            retrieval_text = text
        out.append(
            {
                "id": str(row.get("id") or "").strip(),
                "text": text,
                "tags": tags,
                "tokens": set(_tokenize_simple(text)),
                "retrieval_text": retrieval_text,
                "vector": _sparse_tf(_tokenize_simple(retrieval_text)),
            }
        )
    return out


def _retrieve_current_rag_snippets(
    *,
    entries: List[Dict[str, Any]],
    prompt: str,
    domain: str,
    subskill: str,
    top_k: int,
    max_chars: int,
    use_label_signals: bool,
    min_score: float,
    min_margin: float,
    restrict_to_failure_profiles: bool = False,
) -> List[str]:
    if not entries:
        return []
    query_tokens = _tokenize_simple(prompt)
    query_vector = _sparse_tf(query_tokens)
    if not query_vector:
        return []

    candidates: List[Dict[str, Any]]
    if use_label_signals:
        wanted = {domain.lower(), subskill.lower()}
        tagged = [row for row in entries if wanted & set(row.get("tags") or [])]
        candidates = tagged if tagged else entries
    else:
        candidates = entries
    if restrict_to_failure_profiles:
        profiled = [row for row in candidates if "failure_profile" in set(row.get("tags") or [])]
        if profiled:
            candidates = profiled

    prompt_domains = _infer_prompt_domains(prompt)
    cross_domain_prompt = len(prompt_domains) >= 2
    if not use_label_signals:
        filtered_candidates: List[Dict[str, Any]] = []
        for row in candidates:
            tags = set(row.get("tags") or [])
            has_multidomain = "multidomain" in tags
            if has_multidomain and not cross_domain_prompt:
                continue
            if cross_domain_prompt and not has_multidomain:
                domain_tags = tags.intersection(
                    {
                        "ai_strategy_and_planning",
                        "army_operations",
                        "economy",
                        "hud_status",
                        "state_perstitence_integrity",
                    }
                )
                if domain_tags and not (domain_tags & prompt_domains):
                    continue
            filtered_candidates.append(row)
        candidates = filtered_candidates

    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in candidates:
        similarity = _cosine_sparse(query_vector, row.get("vector") or {})
        if similarity <= 0.0:
            continue
        tags = set(row.get("tags") or [])
        if not use_label_signals and "multidomain" in tags and cross_domain_prompt:
            # Mild demotion so explicit domain matches win when available.
            similarity *= 0.95
        scored.append((similarity, row))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    if not scored:
        return []
    best_score = float(scored[0][0])
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    margin = best_score - second_score
    if best_score < float(min_score):
        return []
    if len(scored) > 1 and margin < float(min_margin):
        scored = scored[:1]
    snippets: List[str] = []
    budget = max(0, int(max_chars))
    for _, row in scored[: max(1, int(top_k))]:
        text = row["text"]
        if budget <= 0:
            break
        if len(text) > budget:
            text = text[:budget].rstrip() + "..."
        snippets.append(text)
        budget -= len(text)
    return snippets


def _augment_task_prompt(
    *,
    task: Dict[str, Any],
    task_meta: Dict[str, Any],
    rag_current_entries: List[Dict[str, Any]],
    rag_current_top_k: int,
    rag_current_max_chars: int,
    rag_current_min_score: float,
    rag_current_min_margin: float,
    context_engineering_mode: str,
) -> str:
    prompt = str(task.get("prompt") or "").strip()
    if not prompt:
        return prompt
    additions: List[str] = []
    domain = str(task_meta.get("domain_primary_normalized") or "")
    subskill = str(task_meta.get("subskill") or "")
    if context_engineering_mode == "max_potential":
        allowed_paths = task.get("allowed_paths") or []
        verify_commands = task.get("verify_commands") or []
        context_paths = task.get("context_paths") or []
        notes = _truncate_chars(str(task.get("notes") or ""), 420)
        ranked_context = _rank_context_paths(
            paths=context_paths,
            prompt=prompt,
            domain=domain,
            subskill=subskill,
            max_items=8,
        )
        additions.append(
            "Execution brief (max potential):\n"
            f"- Domain: {domain or 'unknown'}\n"
            f"- Subskill: {subskill or 'unknown'}\n"
            f"- Edit scope: {_format_limited_items(allowed_paths, max_items=8, max_chars=420)}\n"
            f"- Verify commands: {_format_limited_items(verify_commands, max_items=4, max_chars=280)}\n"
            f"- High-value context files: {_format_limited_items(ranked_context, max_items=8, max_chars=420)}"
        )
        if notes:
            additions.append("Task notes:\n- " + notes)
        if rag_current_entries:
            # "Cheat allowed" mode: use labels and broader retrieval budget to maximize useful guidance.
            snippets = _retrieve_current_rag_snippets(
                entries=rag_current_entries,
                prompt=prompt,
                domain=domain,
                subskill=subskill,
                top_k=min(max(1, rag_current_top_k), 4),
                max_chars=min(max(0, rag_current_max_chars), 1800),
                use_label_signals=True,
                min_score=min(float(rag_current_min_score), 0.06),
                min_margin=min(float(rag_current_min_margin), 0.015),
                restrict_to_failure_profiles=False,
            )
            if snippets:
                additions.append(
                    "Advanced guidance (retrieved):\n"
                    + "\n".join(f"- {snippet}" for snippet in snippets)
                )
        additions.append(
            "Execution strategy:\n"
            "- Make the minimum set of edits that fully satisfies the task semantics.\n"
            "- Prefer existing symbols and utilities over introducing new APIs.\n"
            "- If a new symbol is unavoidable, wire imports/exports and all call sites in one pass.\n"
            "- Keep behavior deterministic and preserve existing contracts."
        )
        additions.append(
            "Pre-output self-check:\n"
            "- Patch is applyable and starts directly with diff/fenced-file content.\n"
            "- No files outside declared scope unless task requirements force it.\n"
            "- Verify commands are expected to pass with the proposed changes.\n"
            "- No placeholder TODOs, stubs, or partial wiring left behind."
        )
        additions.append(
            "Output contract (strict):\n"
            "- Return only applyable patch output (unified diff or fenced files with repo-relative paths).\n"
            "- No prose before the first patch line.\n"
            "- Ensure edits are complete and production-grade for this task."
        )
    else:
        if rag_current_entries:
            # For non-max modes, keep current RAG retrieval prompt-only (no label signals).
            snippets = _retrieve_current_rag_snippets(
                entries=rag_current_entries,
                prompt=prompt,
                domain=domain,
                subskill=subskill,
                top_k=rag_current_top_k,
                max_chars=rag_current_max_chars,
                use_label_signals=False,
                min_score=rag_current_min_score,
                min_margin=rag_current_min_margin,
            )
            if snippets:
                additions.append(
                    "Current RAG guidance (retrieved):\n"
                    + "\n\n".join(f"- {snippet}" for snippet in snippets)
                )
    if not additions:
        return prompt
    return prompt + "\n\n---\n\n" + "\n\n".join(additions)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_run_artifact_checkpoint(completed_steps: int, last_checkpoint_at: float) -> float:
    command = os.environ.get("FE_ARTIFACT_CHECKPOINT_COMMAND", "").strip()
    if not command:
        return last_checkpoint_at
    every_steps = int(os.environ.get("FE_ARTIFACT_UPLOAD_EVERY_STEPS", "0") or "0")
    every_seconds = int(os.environ.get("FE_ARTIFACT_UPLOAD_EVERY_SECONDS", "0") or "0")
    now = time.time()
    due_by_steps = every_steps > 0 and completed_steps > 0 and completed_steps % every_steps == 0
    due_by_time = every_seconds > 0 and now - last_checkpoint_at >= every_seconds
    if not due_by_steps and not due_by_time:
        return last_checkpoint_at
    env = os.environ.copy()
    env["FE_ARTIFACT_COMPLETED_STEPS"] = str(completed_steps)
    env["FE_ARTIFACT_CHECKPOINT_KIND"] = "periodic"
    result = subprocess.run(command, shell=True, env=env, text=True, capture_output=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    if result.returncode != 0:
        print(f"artifact_checkpoint_failed exit={result.returncode}", flush=True)
    return now


def _classify_error(error_text: str) -> str:
    msg = (error_text or "").lower()
    if "insufficient_quota" in msg or "http 429" in msg:
        return "frontier_insufficient_quota"
    if "cannot find module 'tsconfig-paths/register'" in msg:
        return "verify_env_missing_tsconfig_paths"
    if "module_not_found" in msg:
        return "module_not_found"
    return "runtime_error"


def _classify_verify_failure(trial_id: str, attempt: str = "local") -> str:
    log_dir = (
        ROOT
        / "benchmarks"
        / "results"
        / "game_task_trials"
        / trial_id
        / "attempts"
        / attempt
        / "logs"
    )
    logs = sorted(log_dir.glob("verify_*.log"))
    if not logs:
        return "verify_failed_unknown"
    text = logs[-1].read_text(encoding="utf-8", errors="replace").lower()
    if "cannot find module 'tsconfig-paths/register'" in text:
        return "verify_env_missing_tsconfig_paths"
    if "module_not_found" in text:
        return "verify_module_not_found"
    if "failed to compile" in text or "type error" in text:
        return "verify_ts_compile_error"
    return "verify_failed_unknown"


def _verify_env_precheck(source_repo: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            VERIFY_ENV_CHECK_CMD,
            cwd=str(source_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"verify_env_precheck_error:{type(exc).__name__}"
    if proc.returncode == 0:
        return None
    output = (proc.stdout or "").lower()
    if "cannot find module 'tsconfig-paths/register'" in output:
        return "verify_env_missing_tsconfig_paths"
    return f"verify_env_precheck_failed_exit_{proc.returncode}"


def _variant_plan(
    *,
    variant: str,
    prompt: str,
    policy: RoutingPolicy,
    adapter_paths: Dict[str, str],
    local_model: str,
    frontier_model: str,
    fallback_adapter_path: str,
    disable_frontier_routing: bool = False,
    single_specialist_adapter_id: str = "",
) -> Dict[str, Any]:
    if variant == "advanced_router_with_specialists":
        decision = policy.decide(
            GenerationRequest(messages=messages_from_prompt(prompt), max_tokens=4096)
        )
        route = decision.route
        adapter_id = decision.adapter_id or "general_fallback"
        requested_adapter = adapter_paths.get(adapter_id, fallback_adapter_path)
        adapter_path = _resolve_adapter_path(requested_adapter)
        reason = decision.reason
        if not adapter_path and requested_adapter:
            reason = f"{reason}; adapter_missing_fallback=base_local"
        if disable_frontier_routing and route != "local":
            reason = f"{reason}; frontier_routing_disabled=forced_local"
            route = "local"
        backend = "local" if route == "local" else "frontier"
        return {
            "backend": backend,
            "route": route,
            "adapter_id": adapter_id,
            "adapter_path": adapter_path,
            "model": frontier_model if backend == "frontier" else local_model,
            "reason": reason,
            "secondary_adapter_id": getattr(decision, "secondary_adapter_id", None),
            "secondary_confidence": float(getattr(decision, "secondary_confidence", 0.0) or 0.0),
        }
    if variant == "qwen_7_5b_only":
        return {
            "backend": "local",
            "route": "local",
            "adapter_id": "none",
            "adapter_path": "",
            "model": local_model,
            "reason": "forced qwen-only local baseline",
            "secondary_adapter_id": None,
            "secondary_confidence": 0.0,
        }
    if variant == "custom_local_adapter":
        adapter_path = _resolve_adapter_path(fallback_adapter_path)
        reason = "forced custom local adapter lane"
        if not adapter_path:
            reason = f"{reason}; adapter_missing_fallback=base_local"
        return {
            "backend": "local",
            "route": "local",
            "adapter_id": str(single_specialist_adapter_id or "custom_local_adapter").strip(),
            "adapter_path": adapter_path,
            "model": local_model,
            "reason": reason,
            "secondary_adapter_id": None,
            "secondary_confidence": 0.0,
        }
    if variant == "gpt_5_5_only":
        return {
            "backend": "frontier",
            "route": "frontier",
            "adapter_id": "none",
            "adapter_path": "",
            "model": frontier_model,
            "reason": "forced gpt-only frontier baseline",
            "secondary_adapter_id": None,
            "secondary_confidence": 0.0,
        }
    if variant == "hud_status_only":
        requested_adapter = adapter_paths.get("hud_status", "")
        adapter_path = _resolve_adapter_path(requested_adapter)
        reason = "forced hud_status-only local specialist lane"
        if not adapter_path and requested_adapter:
            reason = f"{reason}; adapter_missing_fallback=base_local"
        return {
            "backend": "local",
            "route": "local",
            "adapter_id": "hud_status",
            "adapter_path": adapter_path,
            "model": local_model,
            "reason": reason,
            "secondary_adapter_id": None,
            "secondary_confidence": 0.0,
        }
    if variant == "single_specialist_local":
        adapter_id = str(single_specialist_adapter_id or "").strip()
        if not adapter_id:
            raise ValueError("single_specialist_adapter_id is required for single_specialist_local variant")
        requested_adapter = adapter_paths.get(adapter_id, "")
        adapter_path = _resolve_adapter_path(requested_adapter)
        reason = f"forced single-specialist local lane ({adapter_id})"
        if not adapter_path and requested_adapter:
            reason = f"{reason}; adapter_missing_fallback=base_local"
        return {
            "backend": "local",
            "route": "local",
            "adapter_id": adapter_id,
            "adapter_path": adapter_path,
            "model": local_model,
            "reason": reason,
            "secondary_adapter_id": None,
            "secondary_confidence": 0.0,
        }
    raise ValueError(f"Unknown variant: {variant}")


def _run_one_task(
    *,
    variant: str,
    task_id: str,
    task_meta: Dict[str, Any],
    plan: Dict[str, Any],
    runtime_tasks_path: Path,
    source_repo: Path,
    worktree_root: Path,
    timeout_s: int,
    context_chars: int,
    max_tokens: int,
    trial_prefix: str,
    keep_worktrees: bool,
    verify_skip_reason: Optional[str],
) -> Dict[str, Any]:
    trial_id = f"{trial_prefix}-{variant}-{task_id[:28]}-{uuid.uuid4().hex[:6]}"
    row: Dict[str, Any] = {
        "utc": _utc_iso(),
        "variant": variant,
        "task_id": task_id,
        "trial_id": trial_id,
        "source_file": task_meta["source_file"],
        "domain_primary_normalized": task_meta["domain_primary_normalized"],
        "subskill": task_meta["subskill"],
        "backend": plan["backend"],
        "route": plan["route"],
        "adapter_id": plan["adapter_id"],
        "adapter_path": plan["adapter_path"],
        "model": plan["model"],
        "router_reason": plan["reason"],
        "secondary_adapter_id": plan["secondary_adapter_id"],
        "secondary_confidence": round(float(plan["secondary_confidence"]), 4),
        "apply_status": "not_run",
        "verify_status": "not_run",
        "accepted": False,
        "error": "",
        "failure_class": "",
        "infra_blocker": "",
        "generation_elapsed_s": None,
        "generation_total_tokens": None,
        "generation_cost_usd": None,
    }
    created = False
    try:
        create_trial(
            argparse.Namespace(
                tasks=runtime_tasks_path,
                task_id=task_id,
                source_repo=str(source_repo),
                worktree_root=str(worktree_root),
                base_ref="HEAD",
                trial_id=trial_id,
                attempts=["local"],
                local_adapter=plan["adapter_path"] or "none",
                frontier_model=plan["model"] or DEFAULT_FRONTIER_MODEL,
                copy=False,
            )
        )
        created = True
        output_path = generate_attempt(
            argparse.Namespace(
                trial_id=trial_id,
                attempt="local",
                backend=plan["backend"],
                adapter_path=plan["adapter_path"],
                local_model=plan["model"] if plan["backend"] == "local" else None,
                model=plan["model"],
                max_tokens=max_tokens,
                temp=0.0,
                context_chars=context_chars,
                no_context_bm25=False,
                bug_check_loop=True,
                bug_check_rounds=1,
                bug_check_max_tokens=min(2048, max_tokens),
                bug_check_system_prompt=(
                    "You are a strict bug-fix reviewer for code patches. "
                    "Return only a corrected final patch output."
                ),
                bug_check_rag=True,
                bug_check_rag_corpus="data/rag/bug_fix_agent_corpus.json",
                bug_check_rag_top_k=6,
                bug_check_rag_max_chars=2200,
            )
        )
        applied = apply_output(argparse.Namespace(trial_id=trial_id, attempt="local", input=str(output_path)))
        if verify_skip_reason:
            verified = load_trial(trial_id).attempts["local"]
            row["verify_status"] = "infra_precheck_failed"
            row["infra_blocker"] = verify_skip_reason
            row["failure_class"] = verify_skip_reason
        else:
            verified = verify(argparse.Namespace(trial_id=trial_id, attempt="local", timeout=timeout_s))
            row["verify_status"] = verified.verify_status
            if verified.verify_status != "passed":
                row["failure_class"] = _classify_verify_failure(trial_id)
                if row["failure_class"].startswith("verify_env_"):
                    row["infra_blocker"] = row["failure_class"]
        row["apply_status"] = applied.apply_status
        row["accepted"] = bool(attempt_applied(verified) and verified.verify_status == "passed")
        row["generation_elapsed_s"] = verified.generation_elapsed_s
        row["generation_total_tokens"] = verified.generation_total_tokens
        row["generation_cost_usd"] = verified.generation_cost_usd
    except Exception as exc:  # pylint: disable=broad-except
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["failure_class"] = _classify_error(row["error"])
        if row["failure_class"].startswith("frontier_") or row["failure_class"].startswith("verify_env_"):
            row["infra_blocker"] = row["failure_class"]
        row["traceback_tail"] = traceback.format_exc(limit=2).strip()
        if created:
            try:
                latest = load_trial(trial_id).attempts["local"]
                row["apply_status"] = latest.apply_status
                row["verify_status"] = latest.verify_status
                row["generation_elapsed_s"] = latest.generation_elapsed_s
                row["generation_total_tokens"] = latest.generation_total_tokens
                row["generation_cost_usd"] = latest.generation_cost_usd
            except Exception:
                pass
    finally:
        if created and not keep_worktrees:
            try:
                cleanup(argparse.Namespace(trial_id=trial_id, attempt="local", copy=False, keep_branch=False))
            except Exception:
                pass
    return row


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in sorted({str(r["variant"]) for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        total = len(subset)
        applyable = sum(1 for r in subset if str(r.get("apply_status", "")).startswith(("applied", "wrote")))
        verify_pass = sum(1 for r in subset if r.get("verify_status") == "passed")
        accepted = sum(1 for r in subset if r.get("accepted"))
        errors = sum(1 for r in subset if r.get("error"))
        infra_blocked = sum(1 for r in subset if str(r.get("infra_blocker") or "").strip())
        tokens = [int(r["generation_total_tokens"]) for r in subset if isinstance(r.get("generation_total_tokens"), int)]
        costs = [float(r["generation_cost_usd"]) for r in subset if isinstance(r.get("generation_cost_usd"), (int, float))]
        route_counts = Counter(str(r.get("route") or "unknown") for r in subset)
        failure_classes = Counter(str(r.get("failure_class") or "none") for r in subset)
        by_domain: Dict[str, Dict[str, Any]] = {}
        domain_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in subset:
            domain_bucket[str(row.get("domain_primary_normalized") or "unknown")].append(row)
        for domain, drows in sorted(domain_bucket.items()):
            dtotal = len(drows)
            by_domain[domain] = {
                "tasks": dtotal,
                "accepted": sum(1 for r in drows if r.get("accepted")),
                "accept_rate": round(sum(1 for r in drows if r.get("accepted")) / dtotal, 4) if dtotal else 0.0,
                "verify_pass_rate": round(sum(1 for r in drows if r.get("verify_status") == "passed") / dtotal, 4)
                if dtotal
                else 0.0,
            }
        by_variant[variant] = {
            "tasks": total,
            "applyable": applyable,
            "verify_passed": verify_pass,
            "accepted": accepted,
            "errors": errors,
            "infra_blocked": infra_blocked,
            "applyable_rate": round(applyable / total, 4) if total else 0.0,
            "verify_pass_rate": round(verify_pass / total, 4) if total else 0.0,
            "accept_rate": round(accepted / total, 4) if total else 0.0,
            "avg_tokens": round(sum(tokens) / len(tokens), 2) if tokens else 0.0,
            "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
            "route_counts": dict(route_counts),
            "failure_classes": dict(failure_classes),
            "domain_slices": by_domain,
        }

    return {
        "schema_version": "final_system_ablation_v1",
        "generated_utc": _utc_iso(),
        "side_by_side": by_variant,
    }


def _write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    side = summary.get("side_by_side", {})
    lines = [
        "# Final-system ablation summary",
        "",
        f"- Generated UTC: `{summary.get('generated_utc', '')}`",
        "",
        "| Variant | Tasks | Applyable | Verify Passed | Accepted | Infra Blocked | Applyable Rate | Verify Rate | Accept Rate | Avg Tokens | Avg Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, payload in side.items():
        lines.append(
            f"| `{variant}` | {payload['tasks']} | {payload['applyable']} | {payload['verify_passed']} | {payload['accepted']} | {payload.get('infra_blocked', 0)} | "
            f"{payload['applyable_rate']:.2%} | {payload['verify_pass_rate']:.2%} | {payload['accept_rate']:.2%} | "
            f"{payload['avg_tokens']:.2f} | {payload['avg_cost_usd']:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final-system 3-way ablation benchmark.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    parser.add_argument("--adapter-registry", type=Path, default=DEFAULT_ADAPTER_REGISTRY)
    parser.add_argument("--runtime-tasks-json", type=Path, default=DEFAULT_TASKS_JSON)
    parser.add_argument("--rows-jsonl", type=Path, default=DEFAULT_ROWS_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--frontier-model", default=os.environ.get("FRONTIER_MODEL", DEFAULT_FRONTIER_MODEL))
    parser.add_argument(
        "--variants",
        action="append",
        default=[],
        help="Subset of variants to run. Repeatable. Default runs all three.",
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--context-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument(
        "--task-domain",
        action="append",
        default=[],
        help="Filter runtime tasks by normalized domain before --max-tasks. Repeatable.",
    )
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--no-rows-reset", action="store_true", help="Append to rows jsonl instead of replacing.")
    parser.add_argument(
        "--single-specialist-adapter-id",
        default="",
        help=(
            "Adapter id to force when running --variants single_specialist_local "
            "(e.g. loading_screen, combat_risk, economy_tooltip)."
        ),
    )
    parser.add_argument(
        "--custom-local-adapter-path",
        default="",
        help="Adapter path to force when running --variants custom_local_adapter.",
    )
    parser.add_argument(
        "--custom-local-adapter-id",
        default="custom_local_adapter",
        help="Adapter id label to record for --variants custom_local_adapter.",
    )
    parser.add_argument(
        "--abort-on-frontier-quota",
        action="store_true",
        help="Stop issuing frontier requests after first quota error; mark later frontier tasks as infra-blocked skips.",
    )
    parser.add_argument(
        "--verify-precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one verify-environment precheck and mark verify as infra-failed when broken.",
    )
    parser.add_argument(
        "--disable-frontier-routing",
        action="store_true",
        help="For advanced-router variant, force local backend even when router picks frontier/hybrid.",
    )
    parser.add_argument(
        "--prompt-rag-current-corpus",
        type=Path,
        default=None,
        help="Optional current RAG dataset JSON with entries[id,tags,text] for prompt augmentation.",
    )
    parser.add_argument(
        "--prompt-rag-current-top-k",
        type=int,
        default=3,
        help="How many current RAG snippets to attach when rag_current is enabled.",
    )
    parser.add_argument(
        "--prompt-rag-current-max-chars",
        type=int,
        default=1600,
        help="Total char budget for attached current RAG snippets.",
    )
    parser.add_argument(
        "--prompt-rag-current-min-score",
        type=float,
        default=0.08,
        help="Minimum cosine similarity score required to attach current RAG snippets.",
    )
    parser.add_argument(
        "--prompt-rag-current-min-margin",
        type=float,
        default=0.02,
        help="If top-2 cosine scores are within this margin, keep only top-1 snippet.",
    )
    parser.add_argument(
        "--prompt-context-engineering",
        choices=["off", "max_potential"],
        default="off",
        help="Attach explicit execution constraints/checklist to each task prompt.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_json(manifest_path)
    refs = manifest.get("task_refs", [])
    missing_contract = int((manifest.get("validation") or {}).get("missing_contract_count", 0))
    print(f"manifest={manifest_path}")
    print(f"task_count={len(refs)}")
    print(f"missing_contract_count={missing_contract}")
    if missing_contract:
        print("validation_status=FAIL")
        return 1
    print("validation_status=PASS")

    runtime_tasks, task_meta_rows = _compile_manifest_tasks(manifest, manifest_path)
    if args.task_domain:
        wanted_domains = {str(domain).strip() for domain in args.task_domain if str(domain).strip()}
        if wanted_domains:
            meta_by_id = {str(row["id"]): row for row in task_meta_rows}
            runtime_tasks = [
                task
                for task in runtime_tasks
                if str((meta_by_id.get(str(task.get("id") or "")) or {}).get("domain_primary_normalized") or "")
                in wanted_domains
            ]
            task_meta_rows = [
                row for row in task_meta_rows if str(row.get("domain_primary_normalized") or "") in wanted_domains
            ]
    if args.max_tasks and args.max_tasks > 0:
        runtime_tasks = runtime_tasks[: args.max_tasks]
        wanted_ids = {row["id"] for row in runtime_tasks}
        task_meta_rows = [row for row in task_meta_rows if row["id"] in wanted_ids]
    task_meta = {row["id"]: row for row in task_meta_rows}
    rag_current_entries: List[Dict[str, Any]] = []
    if args.prompt_rag_current_corpus:
        rag_current_corpus_path = args.prompt_rag_current_corpus.expanduser().resolve()
        if rag_current_corpus_path.is_file():
            rag_current_entries = _load_current_rag_entries(rag_current_corpus_path)
            print(f"rag_current_entries={len(rag_current_entries)} corpus={rag_current_corpus_path}")
        else:
            print(f"rag_current_entries=0 corpus_missing={rag_current_corpus_path}")
    if rag_current_entries or args.prompt_context_engineering != "off":
        for task in runtime_tasks:
            tid = str(task.get("id") or "")
            task["prompt"] = _augment_task_prompt(
                task=task,
                task_meta=task_meta.get(tid, {}),
                rag_current_entries=rag_current_entries,
                rag_current_top_k=args.prompt_rag_current_top_k,
                rag_current_max_chars=args.prompt_rag_current_max_chars,
                rag_current_min_score=args.prompt_rag_current_min_score,
                rag_current_min_margin=args.prompt_rag_current_min_margin,
                context_engineering_mode=args.prompt_context_engineering,
            )
    args.runtime_tasks_json.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_tasks_json.write_text(
        json.dumps({"version": 1, "tasks": runtime_tasks}, indent=2) + "\n",
        encoding="utf-8",
    )

    variants = args.variants or [
        "advanced_router_with_specialists",
        "qwen_7_5b_only",
        "gpt_5_5_only",
    ]
    allowed = {
        "advanced_router_with_specialists",
        "qwen_7_5b_only",
        "gpt_5_5_only",
        "hud_status_only",
        "single_specialist_local",
        "custom_local_adapter",
    }
    bad = [v for v in variants if v not in allowed]
    if bad:
        raise SystemExit(f"Unsupported variants: {bad}")
    if "single_specialist_local" in variants and not str(args.single_specialist_adapter_id).strip():
        raise SystemExit("--single-specialist-adapter-id is required with --variants single_specialist_local")
    if "custom_local_adapter" in variants and not str(args.custom_local_adapter_path).strip():
        raise SystemExit("--custom-local-adapter-path is required with --variants custom_local_adapter")

    adapter_paths = _load_adapter_paths(args.adapter_registry.expanduser().resolve())
    fallback_adapter_path = adapter_paths.get("general_fallback", "checkpoints/adapters/general_fallback/champion")
    policy = RoutingPolicy(adapter_registry_path=args.adapter_registry.expanduser().resolve())

    rows_path = args.rows_jsonl.expanduser().resolve()
    if not args.no_rows_reset:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("", encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    trial_prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    frontier_quota_blocked = False
    artifact_checkpoint_at = time.time()
    verify_skip_reason = _verify_env_precheck(args.source_repo.expanduser().resolve()) if args.verify_precheck else None
    if verify_skip_reason:
        print(f"verify_precheck={verify_skip_reason}")
    for variant in variants:
        print(f"=== Variant: {variant} ===")
        for idx, task in enumerate(runtime_tasks, start=1):
            tid = task["id"]
            plan = _variant_plan(
                variant=variant,
                prompt=task["prompt"],
                policy=policy,
                adapter_paths=adapter_paths,
                local_model=args.local_model,
                frontier_model=args.frontier_model,
                fallback_adapter_path=(
                    str(args.custom_local_adapter_path or "").strip()
                    if variant == "custom_local_adapter"
                    else fallback_adapter_path
                ),
                disable_frontier_routing=bool(args.disable_frontier_routing),
                single_specialist_adapter_id=(
                    str(args.custom_local_adapter_id or "").strip()
                    if variant == "custom_local_adapter"
                    else str(args.single_specialist_adapter_id or "").strip()
                ),
            )
            if (
                args.abort_on_frontier_quota
                and frontier_quota_blocked
                and plan.get("backend") == "frontier"
            ):
                row = {
                    "utc": _utc_iso(),
                    "variant": variant,
                    "task_id": tid,
                    "trial_id": "",
                    "source_file": task_meta[tid]["source_file"],
                    "domain_primary_normalized": task_meta[tid]["domain_primary_normalized"],
                    "subskill": task_meta[tid]["subskill"],
                    "backend": plan["backend"],
                    "route": plan["route"],
                    "adapter_id": plan["adapter_id"],
                    "adapter_path": plan["adapter_path"],
                    "model": plan["model"],
                    "router_reason": plan["reason"],
                    "secondary_adapter_id": plan["secondary_adapter_id"],
                    "secondary_confidence": round(float(plan["secondary_confidence"]), 4),
                    "apply_status": "not_run",
                    "verify_status": "not_run",
                    "accepted": False,
                    "error": "",
                    "failure_class": "frontier_skipped_after_quota",
                    "infra_blocker": "frontier_skipped_after_quota",
                    "generation_elapsed_s": None,
                    "generation_total_tokens": None,
                    "generation_cost_usd": None,
                }
            else:
                row = _run_one_task(
                    variant=variant,
                    task_id=tid,
                    task_meta=task_meta[tid],
                    plan=plan,
                    runtime_tasks_path=args.runtime_tasks_json.expanduser().resolve(),
                    source_repo=args.source_repo.expanduser().resolve(),
                    worktree_root=args.worktree_root.expanduser().resolve(),
                    timeout_s=int(args.timeout_s),
                    context_chars=int(args.context_chars),
                    max_tokens=int(args.max_tokens),
                    trial_prefix=trial_prefix,
                    keep_worktrees=bool(args.keep_worktrees),
                    verify_skip_reason=verify_skip_reason,
                )
            if row.get("failure_class") == "frontier_insufficient_quota":
                frontier_quota_blocked = True
            all_rows.append(row)
            _append_jsonl(rows_path, row)
            artifact_checkpoint_at = _maybe_run_artifact_checkpoint(len(all_rows), artifact_checkpoint_at)
            status = "PASS" if row.get("accepted") else "FAIL"
            print(
                f"[{status}] {variant} {idx}/{len(runtime_tasks)} task={tid} "
                f"route={row.get('route')} backend={row.get('backend')} "
                f"apply={row.get('apply_status')} verify={row.get('verify_status')}"
            )
            if row.get("error"):
                print(f"  error={row['error']}")

    summary = _summarize(all_rows)
    summary["manifest"] = str(manifest_path)
    summary["runtime_tasks_json"] = str(args.runtime_tasks_json.expanduser().resolve())
    summary["rows_jsonl"] = str(rows_path)
    summary["variants"] = variants
    summary["tasks"] = len(runtime_tasks)
    summary["source_repo"] = str(args.source_repo.expanduser().resolve())
    summary["worktree_root"] = str(args.worktree_root.expanduser().resolve())
    summary["local_model"] = args.local_model
    summary["frontier_model"] = args.frontier_model
    summary["keep_worktrees"] = bool(args.keep_worktrees)

    summary_json = args.summary_json.expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(args.summary_md.expanduser().resolve(), summary)
    print(f"rows_jsonl={rows_path}")
    print(f"summary_json={summary_json}")
    print(f"summary_md={args.summary_md.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
