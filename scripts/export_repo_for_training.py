#!/usr/bin/env python3
"""
Walk a source repo and emit JSONL: one record per text file suitable for LoRA data prep.

Hygiene (defaults on):
  - Skips oversize files (default 1 MB; EXPORT_MAX_FILE_BYTES), sensitive path segments, credential-like extensions
  - Redacts common secret patterns in file bodies before writing
Adjust EXTENSIONS / SKIP_DIRS / MAX_FILE_BYTES / REDACT_PATTERNS as needed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Tuple

# Default: sibling game repo (override with SOURCE_REPO).
DEFAULT_SOURCE = Path.home() / "fallen-empire"

EXTENSIONS = {
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".css",
    ".html",
    ".sh",
    ".py",
    ".sql",
    ".yml",
    ".yaml",
}

SKIP_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
    "artifacts",
    ".turbo",
    ".parcel-cache",
    "credentials",
    "secrets",
}

SKIP_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
    ".npmrc",
}

# Skip entire relative path if any path component matches (case-insensitive).
SKIP_PATH_SUBSTRINGS = (
    "/.env",
    "secret",
    "credential",
    "private_key",
    "id_rsa",
    ".pem/",
)

SKIP_NAME_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".mobileprovision",
    ".keystore",
    ".jks",
)

# Max bytes per file before excluding. Default raised so long game sources reach
# build_lora_dataset.py chunking instead of being dropped entirely (override with EXPORT_MAX_FILE_BYTES).
MAX_FILE_BYTES = int(os.environ.get("EXPORT_MAX_FILE_BYTES", "1000000"))

# (regex, replacement) — applied to full text; keep patterns tight to avoid false positives.
REDACT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"(?i)(api[_-]?key|password|secret|client_secret|access_token|refresh_token|"
            r"auth_token|bearer|authorization)\s*[:=]\s*([^\s\n\"'#]+)"
        ),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(r"(?i)(-----BEGIN [A-Z ]+PRIVATE KEY-----)([\s\S]*?)(-----END [A-Z ]+PRIVATE KEY-----)"),
        r"\1\n[REDACTED]\n\3",
    ),
    (re.compile(r"(?i)(sk-[A-Za-z0-9]{20,})"), r"[REDACTED_OPENAI_SK]"),
    (re.compile(r"(?i)(ghp_[A-Za-z0-9]{20,})"), r"[REDACTED_GH_PAT]"),
    (re.compile(r"(?i)(xox[baprs]-[A-Za-z0-9-]+)"), r"[REDACTED_SLACK_TOKEN]"),
)


def _should_skip_path(rel: str) -> bool:
    low = rel.lower()
    for frag in SKIP_PATH_SUBSTRINGS:
        if frag.lower() in low.replace("\\", "/"):
            return True
    return False


def _scrub_text(text: str) -> str:
    out = text
    for pat, repl in REDACT_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _looks_binary_sample(text: str) -> bool:
    if "\x00" in text:
        return True
    # Heuristic: high ratio of non-text control chars
    if not text:
        return False
    ctrl = sum(1 for c in text[:8000] if ord(c) < 9 and c not in "\n\r\t")
    return ctrl > max(50, len(text[:8000]) // 100)


def main() -> None:
    root = Path(os.environ.get("SOURCE_REPO", DEFAULT_SOURCE)).resolve()
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repo_text.jsonl"

    if not root.is_dir():
        raise SystemExit(f"SOURCE_REPO is not a directory: {root}")

    skipped: dict[str, int] = {
        "oversize": 0,
        "binaryish": 0,
        "suffix": 0,
        "path_rule": 0,
    }
    count = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            rel_posix = rel.replace("\\", "/")
            if path.name in SKIP_FILES:
                continue
            if path.name.startswith(".env"):
                continue
            if any(path.name.lower().endswith(sfx) for sfx in SKIP_NAME_SUFFIXES):
                skipped["suffix"] += 1
                continue
            if _should_skip_path(rel_posix):
                skipped["path_rule"] += 1
                continue
            parts = set(path.relative_to(root).parts)
            if parts & SKIP_DIRS:
                continue
            if any(p in SKIP_DIRS for p in path.relative_to(root).parts):
                continue
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                sz = path.stat().st_size
            except OSError:
                continue
            if sz > MAX_FILE_BYTES:
                skipped["oversize"] += 1
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _looks_binary_sample(text):
                skipped["binaryish"] += 1
                continue
            text = _scrub_text(text)
            rec = {"path": rel_posix, "text": text}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} files to {out_path}")
    if any(skipped.values()):
        print("Skipped:", ", ".join(f"{k}={v}" for k, v in skipped.items() if v))


if __name__ == "__main__":
    main()
