#!/usr/bin/env python3
"""Generate documentation captures for a changed path (open-source + specialized)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
SPECIALIZED_ADAPTER = REPO / "checkpoints" / "adapters" / "documentation" / "cycle3"
OUT_DIR = REPO / "data" / "documentation_captures"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _estimate_tokens(text: str) -> int:
    return max(0, int(round(len(text.split()) * 1.33)))


def _safe_slug(path: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in path)[:120].strip("-") or "change"


def _git_diff_for_path(path: str, max_chars: int = 5000) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = proc.stdout.strip()
    if not diff:
        return f"(No unstaged diff available for {path}; file may be staged or unchanged.)"
    return diff[:max_chars]


def _strip_leading_headings(text: str) -> str:
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if line.startswith("#"):
            idx += 1
            continue
        break
    return "\n".join(lines[idx:]).strip()


def _extract_summary_bullets(text: str, max_bullets: int = 3) -> list[str]:
    bullets: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("**"):
            continue
        if line.startswith(("- ", "* ")):
            item = line[2:].strip()
        else:
            match = re.match(r"^\d+\.\s+(.*)$", line)
            if not match:
                continue
            item = match.group(1).strip()
        item = item.strip("` ").strip()
        if item and item not in bullets:
            bullets.append(item)
        if len(bullets) >= max_bullets:
            return bullets

    if bullets:
        return bullets

    fallback_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("**"):
            continue
        fallback_lines.append(line[:160])
        if len(fallback_lines) >= max_bullets:
            break
    return fallback_lines


def _normalize_change_doc(
    text: str,
    *,
    changed_path: str,
    trigger_event: str,
    timestamp_iso: str,
) -> str:
    date = timestamp_iso[:10]
    cleaned_body = _strip_leading_headings(text)
    summary = _extract_summary_bullets(cleaned_body)
    if not summary:
        summary = [f"Updated `{changed_path}` from trigger `{trigger_event}`."]

    header_lines = [
        "## Change Documentation Update",
        f"Date: {date}",
        *[f"- {item}" for item in summary],
        "",
    ]
    if cleaned_body:
        return "\n".join(header_lines) + cleaned_body.strip() + "\n"
    return "\n".join(header_lines).strip() + "\n"


def _generate_with_mlx(prompt: str, adapter_path: Optional[Path] = None) -> Dict[str, object]:
    sys.path.insert(0, str(SCRIPTS))
    from mlx_lm import generate, load
    from mlx_qwen_stop_tokens import register_qwen_coder_instruct_extra_stops

    load_kw = {}
    if adapter_path:
        load_kw["adapter_path"] = str(adapter_path)
    t0 = time.perf_counter()
    model, tok = load(MODEL_ID, **load_kw)
    register_qwen_coder_instruct_extra_stops(tok)
    load_s = round(time.perf_counter() - t0, 3)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a documentation writer for fallen-empire-lora. "
                "Write precise implementation notes with exact file paths and commands."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    g0 = time.perf_counter()
    output = generate(model, tok, prompt=chat_prompt, max_tokens=420, verbose=False).strip()
    gen_s = round(time.perf_counter() - g0, 3)
    return {
        "output": output,
        "load_seconds": load_s,
        "generation_seconds": gen_s,
        "estimated_tokens": _estimate_tokens(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentation captures for a changed path.")
    parser.add_argument("--changed-path", required=True)
    parser.add_argument("--event", default="afterFileEdit")
    args = parser.parse_args()

    changed = args.changed_path.strip().lstrip("./")
    diff_excerpt = _git_diff_for_path(changed)
    prompt = (
        "Document this codebase change for dataset capture.\n\n"
        f"Changed path: `{changed}`\n"
        f"Trigger: `{args.event}`\n\n"
        "Requirements:\n"
        "- Start with: title line, date line, then 2-4 bullet summary lines directly under date.\n"
        "- 4 short sections: What changed, Why it matters, Commands/tests run, Next validation steps.\n"
        "- Include exact repo paths and commands if inferable.\n"
        "- Keep it under 220 words.\n\n"
        "Diff excerpt:\n"
        f"{diff_excerpt}\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = _safe_slug(changed)
    out_prefix = OUT_DIR / f"{stamp}_{slug}"

    started = _now()
    base = _generate_with_mlx(prompt, adapter_path=None)
    specialized = _generate_with_mlx(prompt, adapter_path=SPECIALIZED_ADAPTER if SPECIALIZED_ADAPTER.is_dir() else None)
    finished = _now()
    normalized_open = _normalize_change_doc(
        str(base["output"]),
        changed_path=changed,
        trigger_event=args.event,
        timestamp_iso=finished,
    )
    normalized_specialized = _normalize_change_doc(
        str(specialized["output"]),
        changed_path=changed,
        trigger_event=args.event,
        timestamp_iso=finished,
    )
    base["output"] = normalized_open
    specialized["output"] = normalized_specialized

    payload = {
        "schema_version": "change_documentation_capture_v1",
        "started_at": started,
        "finished_at": finished,
        "changed_path": changed,
        "trigger_event": args.event,
        "model": MODEL_ID,
        "specialized_adapter": str(SPECIALIZED_ADAPTER) if SPECIALIZED_ADAPTER.is_dir() else None,
        "opensource": base,
        "specialized": specialized,
        "diff_excerpt_chars": len(diff_excerpt),
    }
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_prefix.with_name(out_prefix.name + ".opensource.md")).write_text(str(base["output"]), encoding="utf-8")
    (out_prefix.with_name(out_prefix.name + ".specialized.md")).write_text(
        str(specialized["output"]), encoding="utf-8"
    )
    print(f"Wrote capture artifacts: {json_path}")


if __name__ == "__main__":
    main()
