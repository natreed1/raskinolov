#!/usr/bin/env python3
"""Evaluate custom task acceptance signals with soft + hard checks.

This script is intentionally lightweight and deterministic where possible:
- soft checks: heuristic scoring for visual/style signals (color, accents, styling, alignment, contrast)
- hard checks: changed-file policy checks and compile/test commands

It is designed for custom task entries in:
  benchmarks/task_bank/sources/custom_author_tasks_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASK_FILE = REPO / "benchmarks" / "task_bank" / "sources" / "custom_author_tasks_v1.json"
DEFAULT_OUTPUT = REPO / "benchmarks" / "results" / "custom_task_signal_eval.json"
TEXT_EXTS = {".ts", ".tsx", ".js", ".jsx", ".css", ".scss", ".md", ".json", ".mjs", ".cjs"}
IMAGE_EXTS = {".png", ".webp", ".jpg", ".jpeg"}


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def _git_changed_files(repo: Path) -> list[str]:
    code1, out1 = _run(["git", "diff", "--name-only"], cwd=repo)
    code2, out2 = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo)
    if code1 != 0:
        raise SystemExit(f"Failed to read changed files from git diff:\n{out1}")
    if code2 != 0:
        raise SystemExit(f"Failed to read untracked files:\n{out2}")
    items = [x.strip() for x in (out1 + "\n" + out2).splitlines() if x.strip()]
    return sorted(dict.fromkeys(items))


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() not in TEXT_EXTS:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_task(task_file: Path, task_id: str) -> dict[str, Any]:
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else []
    if not isinstance(tasks, list):
        raise SystemExit(f"Invalid task file format: {task_file}")
    for task in tasks:
        if isinstance(task, dict) and task.get("id") == task_id:
            return task
    raise SystemExit(f"Task id not found in {task_file}: {task_id}")


def _score_visual_heuristics(changed_files: list[str], file_text: dict[str, str]) -> dict[str, Any]:
    combined = "\n".join(file_text.values()).lower()

    # Palette/theme vocabulary
    palette_terms = [
        "royal",
        "medieval",
        "parchment",
        "gold",
        "bronze",
        "charcoal",
        "navy",
        "jewel",
        "accent",
    ]
    palette_hits = [term for term in palette_terms if term in combined]

    # Color declarations in CSS/Tailwind/class strings
    hex_hits = re.findall(r"#[0-9a-fA-F]{3,8}\b", combined)
    rgb_hits = re.findall(r"\brgba?\(", combined)
    hsl_hits = re.findall(r"\bhsla?\(", combined)
    tw_hits = re.findall(r"\b(?:bg|text|border|ring|from|to|via)-(?:slate|gray|zinc|stone|amber|yellow|orange|red|blue|indigo|violet|purple|emerald|teal|cyan|neutral)-\d{2,3}\b", combined)
    color_decl_count = len(hex_hits) + len(rgb_hits) + len(hsl_hits) + len(tw_hits)

    # Accent/styling cues
    accent_tokens = re.findall(r"\b(accent|border|ring|shadow|outline|badge|chip)\b", combined)
    style_tokens = re.findall(
        r"\b(background|gradient|border|shadow|rounded|font|tracking|spacing|opacity|saturat|contrast)\b",
        combined,
    )

    # Alignment + contrast cues
    alignment_tokens = re.findall(r"\b(text-left|text-center|text-right|items-center|justify-center|justify-between|text-align)\b", combined)
    contrast_tokens = re.findall(r"\b(text-|color:|foreground|background|bg-|contrast)\b", combined)

    # 0..1 component scores
    palette_score = min(1.0, len(palette_hits) / 4.0)
    color_score = min(1.0, color_decl_count / 10.0)
    accent_score = min(1.0, len(accent_tokens) / 8.0)
    style_score = min(1.0, len(style_tokens) / 12.0)
    alignment_score = min(1.0, len(alignment_tokens) / 3.0)
    contrast_score = min(1.0, len(contrast_tokens) / 8.0)

    # Weighted soft score
    soft = (
        0.22 * palette_score
        + 0.20 * color_score
        + 0.18 * accent_score
        + 0.16 * style_score
        + 0.12 * alignment_score
        + 0.12 * contrast_score
    )
    return {
        "soft_score": round(soft, 4),
        "components": {
            "palette_score": round(palette_score, 4),
            "color_score": round(color_score, 4),
            "accent_score": round(accent_score, 4),
            "style_score": round(style_score, 4),
            "alignment_score": round(alignment_score, 4),
            "contrast_score": round(contrast_score, 4),
        },
        "evidence": {
            "palette_hits": palette_hits,
            "color_declaration_count": color_decl_count,
            "accent_token_count": len(accent_tokens),
            "style_token_count": len(style_tokens),
            "alignment_token_count": len(alignment_tokens),
            "contrast_token_count": len(contrast_tokens),
            "changed_files": changed_files,
        },
    }


def _hard_file_policy(changed_files: list[str], task: dict[str, Any]) -> dict[str, Any]:
    constraints = task.get("hard_constraints") if isinstance(task.get("hard_constraints"), dict) else {}
    forbidden_prefixes = (
        constraints["forbidden_path_prefixes"]
        if "forbidden_path_prefixes" in constraints
        else ["src/core/", "src/lib/", "src/store/", "src/types/"]
    )
    allowed_prefixes = (
        constraints["allowed_path_prefixes"]
        if "allowed_path_prefixes" in constraints
        else ["src/components/ui/", "src/components/test/", "src/app/", "src/styles/"]
    )

    forbidden_hits = [f for f in changed_files if any(f.startswith(prefix) for prefix in forbidden_prefixes)]
    allowed_hits = [f for f in changed_files if any(f.startswith(prefix) for prefix in allowed_prefixes)]
    return {
        "pass_no_forbidden_paths": len(forbidden_hits) == 0,
        "pass_has_allowed_area_changes": len(allowed_hits) > 0,
        "forbidden_hits": forbidden_hits,
        "allowed_hits": allowed_hits,
    }


def _image_integration_policy(changed_files: list[str], file_text: dict[str, str], task: dict[str, Any]) -> dict[str, Any]:
    constraints = task.get("hard_constraints") if isinstance(task.get("hard_constraints"), dict) else {}
    require_sprite_asset = bool(constraints.get("require_sprite_asset"))
    require_sprite_reference = bool(constraints.get("require_sprite_reference"))
    required_integration_files = constraints.get("required_integration_files") or []
    forbid_placeholder_tokens = [str(x).lower() for x in (constraints.get("forbid_placeholder_tokens") or [])]

    sprite_files = [
        f
        for f in changed_files
        if f.startswith("public/sprites/") and Path(f).suffix.lower() in IMAGE_EXTS
    ]
    sprite_exists = True
    if require_sprite_asset:
        sprite_exists = False
        for rel in sprite_files:
            p = REPO / rel
            if p.is_file() and p.stat().st_size > 0:
                sprite_exists = True
                break

    combined = "\n".join(file_text.values()).lower()
    sprite_reference = True
    reference_hits: list[str] = []
    if require_sprite_reference:
        sprite_reference = False
        for rel in sprite_files:
            normalized = "/" + rel.replace("public/", "")
            token = normalized.lower()
            if token in combined:
                sprite_reference = True
                reference_hits.append(normalized)
        # If asset not changed this run, still allow if any sprite path token is present in changed code.
        if not sprite_files and re.search(r"/sprites/.+\.(?:png|webp|jpg|jpeg)", combined):
            sprite_reference = True

    integration_touched = True
    touched_required: list[str] = []
    if required_integration_files:
        integration_touched = False
        for rel in changed_files:
            if any(rel.startswith(prefix) for prefix in required_integration_files):
                integration_touched = True
                touched_required.append(rel)

    placeholder_ok = True
    placeholder_hits: list[str] = []
    if forbid_placeholder_tokens:
        for token in forbid_placeholder_tokens:
            if token in combined:
                placeholder_ok = False
                placeholder_hits.append(token)

    return {
        "require_sprite_asset": require_sprite_asset,
        "require_sprite_reference": require_sprite_reference,
        "pass_sprite_asset_exists": sprite_exists,
        "pass_sprite_reference_present": sprite_reference,
        "pass_required_integration_files_touched": integration_touched,
        "pass_no_placeholder_tokens": placeholder_ok,
        "sprite_files": sprite_files,
        "reference_hits": reference_hits,
        "touched_required_integration_files": touched_required,
        "placeholder_hits": placeholder_hits,
    }


def _run_compile_checks(repo: Path, task: dict[str, Any], compile_commands: list[str], skip_compile: bool) -> list[dict[str, Any]]:
    if skip_compile:
        return [{"command": "(skipped)", "exit_code": 0, "ok": True, "output_preview": "compile checks skipped"}]

    commands = compile_commands or task.get("compile_commands") or ["npm run test:ml-cohort"]
    results: list[dict[str, Any]] = []
    for cmd in commands:
        code, out = _run(["sh", "-lc", cmd], cwd=repo)
        results.append(
            {
                "command": cmd,
                "exit_code": code,
                "ok": code == 0,
                "output_preview": out[:1200],
            }
        )
    return results


def _evaluate_signals(
    signals: list[str], visual: dict[str, Any], hard_policy: dict[str, Any], image_policy: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for signal in signals:
        s = signal.lower()
        if "royal-medieval" in s or "palette" in s:
            ok = visual["components"]["palette_score"] >= 0.5 and visual["components"]["color_score"] >= 0.4
            reason = "palette/color heuristic threshold"
        elif "contrast" in s or "readable" in s:
            ok = visual["components"]["contrast_score"] >= 0.4
            reason = "contrast heuristic threshold"
        elif "accent" in s or "chip distinction" in s:
            ok = visual["components"]["accent_score"] >= 0.4 and visual["components"]["style_score"] >= 0.35
            reason = "accent/style heuristic threshold"
        elif "no gameplay/state logic files" in s:
            ok = bool(hard_policy.get("pass_no_forbidden_paths"))
            reason = "forbidden-path policy check"
        elif "sprite asset exists" in s or "new non-empty sprite asset exists" in s:
            ok = bool(image_policy.get("pass_sprite_asset_exists"))
            reason = "sprite-asset existence check"
        elif "sprite path is referenced" in s or "mapping conventions" in s or "not orphaned" in s:
            ok = bool(image_policy.get("pass_sprite_reference_present"))
            reason = "sprite-reference integration check"
        elif "placeholder" in s or "fallback-only integration" in s:
            ok = bool(image_policy.get("pass_no_placeholder_tokens"))
            reason = "placeholder-token ban check"
        elif "texture/render pipeline" in s or "visibly correct in hud/game view" in s:
            ok = bool(image_policy.get("pass_required_integration_files_touched")) and visual["components"]["contrast_score"] >= 0.35
            reason = "integration-file + readability heuristic check"
        else:
            ok = None
            reason = "manual review required (no mapped evaluator rule)"
        out.append({"signal": signal, "ok": ok, "reason": reason})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate custom task acceptance signals.")
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--compile-command",
        action="append",
        default=[],
        help="Compile/test command to run (repeatable). Defaults to task.compile_commands or npm run test:ml-cohort.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--min-soft-score", type=float, default=0.55)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task_file = args.task_file.expanduser().resolve()
    target_repo = args.target_repo.expanduser().resolve()
    task = _load_task(task_file, args.task_id)

    changed_files = _git_changed_files(target_repo)
    file_text = {f: _read_text_file(target_repo / f) for f in changed_files}
    visual = _score_visual_heuristics(changed_files, file_text)
    hard_policy = _hard_file_policy(changed_files, task)
    image_policy = _image_integration_policy(changed_files, file_text, task)
    compile_results = _run_compile_checks(
        target_repo, task, compile_commands=args.compile_command, skip_compile=args.skip_compile
    )
    signal_results = _evaluate_signals(task.get("acceptance_signals") or [], visual, hard_policy, image_policy)

    compile_ok = all(r.get("ok") for r in compile_results)
    hard_ok = (
        hard_policy["pass_no_forbidden_paths"]
        and hard_policy["pass_has_allowed_area_changes"]
        and image_policy["pass_sprite_asset_exists"]
        and image_policy["pass_sprite_reference_present"]
        and image_policy["pass_required_integration_files_touched"]
        and image_policy["pass_no_placeholder_tokens"]
    )
    signal_bools = [r["ok"] for r in signal_results if r["ok"] is not None]
    signals_ok = all(signal_bools) if signal_bools else True
    soft_ok = visual["soft_score"] >= args.min_soft_score
    overall_ok = bool(compile_ok and hard_ok and signals_ok and soft_ok)

    result = {
        "schema_version": "custom_task_signal_eval_v1",
        "task_id": task["id"],
        "task_file": str(task_file),
        "target_repo": str(target_repo),
        "changed_files": changed_files,
        "visual": visual,
        "hard_policy": hard_policy,
        "image_integration_policy": image_policy,
        "compile_checks": compile_results,
        "signal_results": signal_results,
        "thresholds": {"min_soft_score": args.min_soft_score},
        "summary": {
            "compile_ok": compile_ok,
            "hard_ok": hard_ok,
            "signals_ok": signals_ok,
            "soft_ok": soft_ok,
            "soft_score": visual["soft_score"],
            "overall_ok": overall_ok,
        },
    }

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

