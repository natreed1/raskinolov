#!/usr/bin/env python3
"""
Build a TypeScript/TSX LoRA corpus from GitHub repositories with compile-safety-oriented filters.

Why this exists:
  The arena bottleneck is often "apply + tsc + exports" correctness. This script curates
  TypeScript code from permissively-licensed GitHub repos that look like non-trivial apps and
  use strict TypeScript settings, then writes MLX-ready JSONL splits under data/lora/.

Output shape:
  - train.jsonl / valid.jsonl / test.jsonl (rows are {"text": "..."} for mlx_lm.lora)
  - samples_metadata.jsonl (one metadata row per emitted sample, including split)
  - manifest.json (accepted/rejected repo details and filter settings)

GitHub API behavior:
  - Supports explicit --repo owner/name input.
  - Optional --discover-query uses GitHub repository search (best effort).
  - Supports unauthenticated usage, but may be rate-limited.
  - If GITHUB_TOKEN (or --github-token) is provided, API limits improve.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fe_lineage as _fe

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "lora" / _fe.LINEAGE_SLUG / "github_ts_compile_safe"
DEFAULT_CACHE = REPO / "data" / "raw" / "github_ts_cache"

DEFAULT_LICENSES = (
    "MIT,Apache-2.0,BSD-2-Clause,BSD-3-Clause,ISC,Unlicense,0BSD,CC0-1.0"
)
DEFAULT_INCLUDE_ROOTS = "src,app,components,pages,lib,utils,server,client,packages"

DEFAULT_EXCLUDE_SUBSTRINGS = (
    "/node_modules/",
    "/dist/",
    "/build/",
    "/coverage/",
    "/storybook-static/",
    "/vendor/",
    "/examples/",
    "/demo/",
    "/demos/",
    "/fixtures/",
    "/__tests__/",
    "/tests/",
    "/test/",
)

DEFAULT_EXCLUDE_SUFFIXES = (
    ".d.ts",
    ".test.ts",
    ".test.tsx",
    ".spec.ts",
    ".spec.tsx",
    ".stories.ts",
    ".stories.tsx",
)

SYNTHETIC_SMOKE_ROWS = [
    (
        "sample-org/strict-ui",
        "MIT",
        "src/hud/status.tsx",
        """import React from 'react';

type StatusProps = { hp: number; morale: number; maxHp: number };

export function StatusBar({ hp, morale, maxHp }: StatusProps): JSX.Element {
  const hpPct = Math.max(0, Math.min(100, Math.floor((hp / Math.max(1, maxHp)) * 100)));
  const moraleTone = morale >= 70 ? 'good' : morale >= 35 ? 'warn' : 'bad';
  return (
    <div data-testid="status-bar" className={`status ${moraleTone}`}>
      <span>HP {hpPct}%</span>
      <span>Morale {morale}</span>
    </div>
  );
}
""",
    ),
    (
        "sample-org/strict-ui",
        "MIT",
        "src/economy/tooltip.ts",
        """export type EconomyState = {
  gold: number;
  upkeep: number;
  income: number;
};

export function projectedGold(state: EconomyState, turns: number): number {
  const net = state.income - state.upkeep;
  return state.gold + net * turns;
}
""",
    ),
    (
        "sample-org/strict-ui",
        "Apache-2.0",
        "src/save/guards.ts",
        """export type SavePayload = Record<string, unknown>;

export function isSavePayload(value: unknown): value is SavePayload {
  if (!value || typeof value !== 'object') return false;
  const rec = value as Record<string, unknown>;
  return typeof rec['version'] === 'string' && typeof rec['createdAt'] === 'number';
}
""",
    ),
]


@dataclass
class RepoOutcome:
    repo: str
    accepted: bool
    reason: str
    license_spdx: str
    default_branch: str
    stars: int
    ts_files_scanned: int
    ts_files_emitted: int
    strict_detected: bool
    strict_evidence: str
    package_json_found: bool
    include_roots_present: List[str]

    def to_json(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "accepted": self.accepted,
            "reason": self.reason,
            "license_spdx": self.license_spdx,
            "default_branch": self.default_branch,
            "stars": self.stars,
            "ts_files_scanned": self.ts_files_scanned,
            "ts_files_emitted": self.ts_files_emitted,
            "strict_detected": self.strict_detected,
            "strict_evidence": self.strict_evidence,
            "package_json_found": self.package_json_found,
            "include_roots_present": self.include_roots_present,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _path_unit(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_json_comments(text: str) -> str:
    # Handles common tsconfig JSONC patterns. Not a full JSONC parser.
    no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    no_line = re.sub(r"(^|\s)//.*$", r"\1", no_block, flags=re.MULTILINE)
    no_trailing = re.sub(r",(\s*[}\]])", r"\1", no_line)
    return no_trailing


def _load_json_tolerant(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        data = json.loads(_strip_json_comments(raw))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _github_json(url: str, token: str, timeout_s: int) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fallen-empire-lora-dataset-builder",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub HTTP {e.code} for {url}: {detail[:320]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GitHub request failed for {url}: {e}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from GitHub for {url}: {e}") from e


def _github_bytes(url: str, token: str, timeout_s: int) -> bytes:
    headers = {"User-Agent": "fallen-empire-lora-dataset-builder"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download HTTP {e.code} for {url}: {detail[:320]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def _discover_repos(queries: Sequence[str], per_query: int, token: str, timeout_s: int) -> List[str]:
    out: List[str] = []
    for q in queries:
        q_full = f"{q} language:TypeScript archived:false"
        encoded = urllib.parse.quote(q_full, safe="")
        url = (
            "https://api.github.com/search/repositories"
            f"?q={encoded}&sort=stars&order=desc&per_page={max(1, min(100, per_query))}"
        )
        try:
            payload = _github_json(url, token, timeout_s)
        except RuntimeError as e:
            _warn(f"discovery query failed ({q!r}): {e}")
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            full = str(item.get("full_name") or "").strip()
            if full:
                out.append(full)
    deduped: List[str] = []
    seen: set[str] = set()
    for r in out:
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped


def _read_repos_file(path: Path) -> List[str]:
    repos: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        repos.append(s)
    return repos


def _normalize_repo_name(repo: str) -> Optional[str]:
    s = repo.strip().strip("/")
    if not s or "/" not in s:
        return None
    owner, name = s.split("/", 1)
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        return None
    return f"{owner}/{name}"


def _extract_zip(zip_bytes: bytes, target_dir: Path) -> Path:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target_dir)
    children = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return target_dir


def _is_ts_candidate(path: Path) -> bool:
    return path.suffix.lower() in {".ts", ".tsx"}


def _to_posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _has_excluded_suffix(p: str) -> bool:
    lp = p.lower()
    return any(lp.endswith(suf) for suf in DEFAULT_EXCLUDE_SUFFIXES)


def _path_is_allowed(rel_posix: str, include_roots: set[str]) -> bool:
    lp = "/" + rel_posix.lower().strip("/")
    if any(x in lp for x in DEFAULT_EXCLUDE_SUBSTRINGS):
        return False
    if _has_excluded_suffix(lp):
        return False
    parts = [x for x in rel_posix.split("/") if x]
    if not parts:
        return False
    first = parts[0].lower()
    if first not in include_roots:
        # Allow monorepo package paths like packages/foo/src/x.ts.
        if "packages" in include_roots and len(parts) >= 3 and first == "packages":
            return True
        return False
    return True


def _scan_ts_files(repo_root: Path, include_roots: set[str]) -> Tuple[List[Path], List[str], bool]:
    ts_files: List[Path] = []
    roots_present: set[str] = set()
    package_json_found = False
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = _to_posix_rel(p, repo_root)
        rel_l = rel.lower()
        if rel_l == "package.json" or rel_l.endswith("/package.json"):
            package_json_found = True
        if not _is_ts_candidate(p):
            continue
        if not _path_is_allowed(rel, include_roots):
            continue
        top = rel.split("/", 1)[0].lower() if "/" in rel else rel.lower()
        roots_present.add(top)
        ts_files.append(p)
    ts_files.sort(key=lambda p: _to_posix_rel(p, repo_root))
    return ts_files, sorted(roots_present), package_json_found


def _detect_strict_ts(repo_root: Path) -> Tuple[bool, str]:
    candidates = list(repo_root.rglob("tsconfig*.json"))
    if not candidates:
        return False, "no tsconfig*.json found"
    for cfg in sorted(candidates):
        rel = _to_posix_rel(cfg, repo_root).lower()
        if any(x in rel for x in ("/node_modules/", "/dist/", "/build/", "/coverage/", "/test/", "/tests/")):
            continue
        payload = _load_json_tolerant(cfg)
        if not payload:
            continue
        opts = payload.get("compilerOptions")
        if not isinstance(opts, dict):
            continue
        if opts.get("strict") is True:
            return True, f"{_to_posix_rel(cfg, repo_root)} compilerOptions.strict=true"
        strict_keys = ("strictNullChecks", "noImplicitAny", "noImplicitThis")
        if all(opts.get(k) is True for k in strict_keys):
            return True, (
                f"{_to_posix_rel(cfg, repo_root)} strictNullChecks+noImplicitAny+noImplicitThis=true"
            )
    return False, "strict=true (or strict key trio) not found in tsconfig"


def _file_text_rows(
    repo_name: str,
    license_spdx: str,
    repo_root: Path,
    files: Sequence[Path],
    min_chars: int,
    max_chars: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    for f in files:
        rel = _to_posix_rel(f, repo_root)
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body.strip()) == 0:
            continue
        if len(body) < min_chars or len(body) > max_chars:
            continue
        if "\x00" in body:
            continue
        header = (
            f"# repo: {repo_name}\n"
            f"# license: {license_spdx}\n"
            f"# path: {rel}\n"
            "# language: TypeScript\n\n"
        )
        sample_text = header + body
        train_rows.append({"text": sample_text})
        metadata.append(
            {
                "repo": repo_name,
                "license_spdx": license_spdx,
                "path": rel,
                "chars": len(body),
                "lines": body.count("\n") + 1,
                "sha1": hashlib.sha1(body.encode("utf-8", errors="ignore")).hexdigest(),
            }
        )
    return train_rows, metadata


def _split_rows(
    rows: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(rows) != len(metadata):
        raise ValueError("rows and metadata must have matching lengths")
    indexed = list(zip(rows, metadata))
    train: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    valid: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    test: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for row, md in indexed:
        key = f"{md.get('repo','?')}::{md.get('path','?')}"
        u = _path_unit(key, seed)
        if u < train_ratio:
            train.append((row, md))
        elif u < train_ratio + valid_ratio:
            valid.append((row, md))
        else:
            test.append((row, md))

    # Keep splits non-empty when possible.
    def _move_one(src: List[Tuple[Dict[str, Any], Dict[str, Any]]], dst: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        if not src:
            return
        src.sort(key=lambda x: f"{x[1].get('repo','')}::{x[1].get('path','')}")
        dst.append(src.pop())

    if indexed:
        if not train:
            _move_one(valid if len(valid) >= len(test) else test, train)
        if not valid and len(indexed) >= 3:
            _move_one(train, valid)
        if not test and len(indexed) >= 4:
            _move_one(train, test)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    def unpack(items: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows_out: List[Dict[str, Any]] = []
        meta_out: List[Dict[str, Any]] = []
        for row, md in items:
            rows_out.append(row)
            meta_out.append({**md, "split": split})
        return rows_out, meta_out

    tr_rows, tr_meta = unpack(train, "train")
    va_rows, va_meta = unpack(valid, "valid")
    te_rows, te_meta = unpack(test, "test")
    all_meta = [*tr_meta, *va_meta, *te_meta]
    return tr_rows, va_rows, te_rows, all_meta


def _parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def _collect_repos(args: argparse.Namespace, token: str) -> Tuple[List[str], List[str]]:
    explicit: List[str] = []
    for r in args.repo:
        n = _normalize_repo_name(r)
        if not n:
            _warn(f"invalid --repo value skipped: {r!r}")
            continue
        explicit.append(n)
    if args.repos_file:
        if not args.repos_file.is_file():
            raise SystemExit(f"--repos-file not found: {args.repos_file}")
        for r in _read_repos_file(args.repos_file):
            n = _normalize_repo_name(r)
            if n:
                explicit.append(n)
            else:
                _warn(f"invalid repos-file line skipped: {r!r}")
    discovered = _discover_repos(args.discover_query, args.discover_per_query, token, args.timeout_s)
    all_repos = [*explicit, *discovered]
    deduped: List[str] = []
    seen: set[str] = set()
    for r in all_repos:
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped, discovered


def _smoke_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    for repo_name, license_spdx, path, body in SYNTHETIC_SMOKE_ROWS:
        sample_text = (
            f"# repo: {repo_name}\n"
            f"# license: {license_spdx}\n"
            f"# path: {path}\n"
            "# language: TypeScript\n\n"
            + body
        )
        rows.append({"text": sample_text})
        meta.append(
            {
                "repo": repo_name,
                "license_spdx": license_spdx,
                "path": path,
                "chars": len(body),
                "lines": body.count("\n") + 1,
                "sha1": hashlib.sha1(body.encode("utf-8")).hexdigest(),
            }
        )
    return rows, meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a strict-TS, permissive-license GitHub dataset for compile-safe TypeScript LoRA training."
    )
    p.add_argument("--repo", action="append", default=[], help="GitHub repo as owner/name (repeatable)")
    p.add_argument("--repos-file", type=Path, default=None, help="Text file with owner/name per line")
    p.add_argument(
        "--discover-query",
        action="append",
        default=[],
        help="GitHub search query (repeatable), e.g. 'react state management stars:>1000'",
    )
    p.add_argument("--discover-per-query", type=int, default=12, help="Max repos per discover query (<=100)")
    p.add_argument("--github-token", default="", help="Optional GitHub token (or use GITHUB_TOKEN env)")
    p.add_argument("--timeout-s", type=int, default=20, help="HTTP timeout seconds")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Cache dir for downloaded zip archives")
    p.add_argument("--refresh-cache", action="store_true", help="Ignore cached archives and re-download")
    p.add_argument("--max-zip-mb", type=int, default=55, help="Reject repositories whose zip archive exceeds this size")

    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output dataset directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)

    p.add_argument("--allowed-licenses", default=DEFAULT_LICENSES, help="CSV SPDX allowlist")
    p.add_argument("--allow-forks", action="store_true", help="Allow forked repositories")
    p.add_argument("--allow-non-strict", action="store_true", help="Allow repos without strict tsconfig evidence")
    p.add_argument(
        "--include-roots",
        default=DEFAULT_INCLUDE_ROOTS,
        help="CSV list of top-level folders to include (e.g. src,app,packages)",
    )
    p.add_argument("--min-ts-files", type=int, default=10, help="Minimum filtered TS/TSX files per repo")
    p.add_argument("--max-files-per-repo", type=int, default=220, help="Cap sampled files per accepted repo")
    p.add_argument("--min-file-chars", type=int, default=120, help="Minimum file character count")
    p.add_argument("--max-file-chars", type=int, default=14000, help="Maximum file character count")
    p.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Write a tiny built-in dataset (no network) to validate local pipeline wiring.",
    )
    args = p.parse_args()

    ratios = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {ratios}")

    token = args.github_token.strip() or os.environ.get("GITHUB_TOKEN", "").strip()

    out_dir = args.out_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    include_roots = _parse_csv_set(args.include_roots.lower())
    allowed_licenses = {x.upper() for x in _parse_csv_set(args.allowed_licenses)}

    all_rows: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []
    outcomes: List[RepoOutcome] = []
    discovered_repos: List[str] = []
    repos: List[str] = []

    if args.synthetic_smoke:
        rows, meta = _smoke_rows()
        all_rows.extend(rows)
        all_meta.extend(meta)
        repos = ["synthetic/smoke"]
        outcomes.append(
            RepoOutcome(
                repo="synthetic/smoke",
                accepted=True,
                reason="synthetic_smoke",
                license_spdx="MIXED",
                default_branch="n/a",
                stars=0,
                ts_files_scanned=len(rows),
                ts_files_emitted=len(rows),
                strict_detected=True,
                strict_evidence="synthetic_smoke",
                package_json_found=True,
                include_roots_present=["src"],
            )
        )
    else:
        repos, discovered_repos = _collect_repos(args, token)
        if not repos:
            raise SystemExit(
                "No repositories provided. Use --repo / --repos-file and/or --discover-query. "
                "Use --synthetic-smoke to validate wiring without network."
            )
        for repo_name in repos:
            meta_url = f"https://api.github.com/repos/{repo_name}"
            try:
                repo_meta = _github_json(meta_url, token, args.timeout_s)
            except RuntimeError as e:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason=f"repo_metadata_error: {e}",
                        license_spdx="UNKNOWN",
                        default_branch="",
                        stars=0,
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue
            if not isinstance(repo_meta, dict):
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason="repo_metadata_not_object",
                        license_spdx="UNKNOWN",
                        default_branch="",
                        stars=0,
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            is_fork = bool(repo_meta.get("fork"))
            if is_fork and not args.allow_forks:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason="fork_rejected",
                        license_spdx="UNKNOWN",
                        default_branch=str(repo_meta.get("default_branch") or ""),
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            license_block = repo_meta.get("license") if isinstance(repo_meta.get("license"), dict) else {}
            spdx = str(license_block.get("spdx_id") or "UNKNOWN")
            if spdx.upper() not in allowed_licenses:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason=f"license_not_allowed:{spdx}",
                        license_spdx=spdx,
                        default_branch=str(repo_meta.get("default_branch") or ""),
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            default_branch = str(repo_meta.get("default_branch") or "").strip() or "HEAD"
            zip_url = str(repo_meta.get("zipball_url") or "").strip()
            if not zip_url:
                archive_tmpl = str(repo_meta.get("archive_url") or "").strip()
                if archive_tmpl and "{archive_format}{/ref}" in archive_tmpl:
                    zip_url = archive_tmpl.replace("{archive_format}{/ref}", f"zipball/{default_branch}")
            if not zip_url:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason="missing_zipball_url",
                        license_spdx=spdx,
                        default_branch=default_branch,
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            cache_path = cache_dir / f"{repo_name.replace('/', '__')}.zip"
            try:
                if cache_path.is_file() and not args.refresh_cache:
                    zip_bytes = cache_path.read_bytes()
                else:
                    zip_bytes = _github_bytes(zip_url, token, args.timeout_s)
                    cache_path.write_bytes(zip_bytes)
            except (RuntimeError, OSError) as e:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason=f"zip_download_error:{e}",
                        license_spdx=spdx,
                            default_branch=default_branch,
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            if len(zip_bytes) > args.max_zip_mb * 1024 * 1024:
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=False,
                        reason=f"zip_too_large:{len(zip_bytes)}",
                        license_spdx=spdx,
                            default_branch=default_branch,
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=0,
                        ts_files_emitted=0,
                        strict_detected=False,
                        strict_evidence="",
                        package_json_found=False,
                        include_roots_present=[],
                    )
                )
                continue

            with tempfile.TemporaryDirectory(prefix="fe_github_ts_") as tmp:
                tmp_root = Path(tmp)
                try:
                    extracted = _extract_zip(zip_bytes, tmp_root)
                except zipfile.BadZipFile:
                    outcomes.append(
                        RepoOutcome(
                            repo=repo_name,
                            accepted=False,
                            reason="bad_zip_file",
                            license_spdx=spdx,
                            default_branch=default_branch,
                            stars=int(repo_meta.get("stargazers_count") or 0),
                            ts_files_scanned=0,
                            ts_files_emitted=0,
                            strict_detected=False,
                            strict_evidence="",
                            package_json_found=False,
                            include_roots_present=[],
                        )
                    )
                    continue

                strict_ok, strict_evidence = _detect_strict_ts(extracted)
                if not strict_ok and not args.allow_non_strict:
                    ts_all, roots_present, pkg_found = _scan_ts_files(extracted, include_roots)
                    outcomes.append(
                        RepoOutcome(
                            repo=repo_name,
                            accepted=False,
                            reason="strict_ts_not_detected",
                            license_spdx=spdx,
                            default_branch=default_branch,
                            stars=int(repo_meta.get("stargazers_count") or 0),
                            ts_files_scanned=len(ts_all),
                            ts_files_emitted=0,
                            strict_detected=False,
                            strict_evidence=strict_evidence,
                            package_json_found=pkg_found,
                            include_roots_present=roots_present,
                        )
                    )
                    continue

                ts_files, roots_present, package_json_found = _scan_ts_files(extracted, include_roots)
                if len(ts_files) < args.min_ts_files:
                    outcomes.append(
                        RepoOutcome(
                            repo=repo_name,
                            accepted=False,
                            reason=f"too_few_ts_files:{len(ts_files)}<{args.min_ts_files}",
                            license_spdx=spdx,
                            default_branch=default_branch,
                            stars=int(repo_meta.get("stargazers_count") or 0),
                            ts_files_scanned=len(ts_files),
                            ts_files_emitted=0,
                            strict_detected=strict_ok,
                            strict_evidence=strict_evidence,
                            package_json_found=package_json_found,
                            include_roots_present=roots_present,
                        )
                    )
                    continue

                # Non-trivial app heuristic: package.json + at least one include root.
                if not package_json_found or not roots_present:
                    outcomes.append(
                        RepoOutcome(
                            repo=repo_name,
                            accepted=False,
                            reason="non_trivial_structure_filter_failed",
                            license_spdx=spdx,
                            default_branch=default_branch,
                            stars=int(repo_meta.get("stargazers_count") or 0),
                            ts_files_scanned=len(ts_files),
                            ts_files_emitted=0,
                            strict_detected=strict_ok,
                            strict_evidence=strict_evidence,
                            package_json_found=package_json_found,
                            include_roots_present=roots_present,
                        )
                    )
                    continue

                if len(ts_files) > args.max_files_per_repo:
                    rng = random.Random(int(hashlib.sha256(f"{args.seed}:{repo_name}".encode()).hexdigest(), 16))
                    ts_files = rng.sample(ts_files, args.max_files_per_repo)
                    ts_files.sort(key=lambda p: _to_posix_rel(p, extracted))

                rows, meta_rows = _file_text_rows(
                    repo_name,
                    spdx,
                    extracted,
                    ts_files,
                    args.min_file_chars,
                    args.max_file_chars,
                )
                if not rows:
                    outcomes.append(
                        RepoOutcome(
                            repo=repo_name,
                            accepted=False,
                            reason="no_rows_after_file_size_filters",
                            license_spdx=spdx,
                            default_branch=default_branch,
                            stars=int(repo_meta.get("stargazers_count") or 0),
                            ts_files_scanned=len(ts_files),
                            ts_files_emitted=0,
                            strict_detected=strict_ok,
                            strict_evidence=strict_evidence,
                            package_json_found=package_json_found,
                            include_roots_present=roots_present,
                        )
                    )
                    continue

                all_rows.extend(rows)
                all_meta.extend(meta_rows)
                outcomes.append(
                    RepoOutcome(
                        repo=repo_name,
                        accepted=True,
                        reason="accepted",
                        license_spdx=spdx,
                        default_branch=default_branch,
                        stars=int(repo_meta.get("stargazers_count") or 0),
                        ts_files_scanned=len(ts_files),
                        ts_files_emitted=len(rows),
                        strict_detected=strict_ok,
                        strict_evidence=strict_evidence,
                        package_json_found=package_json_found,
                        include_roots_present=roots_present,
                    )
                )

    if not all_rows:
        manifest = {
            "generated_at": _now_iso(),
            "status": "no_rows",
            "repos_input_count": len(repos),
            "repos_discovered_count": len(discovered_repos),
            "out_dir": str(out_dir),
            "message": "No rows emitted; inspect rejected_repos in this manifest.",
            "rejected_repos": [o.to_json() for o in outcomes if not o.accepted],
            "accepted_repos": [o.to_json() for o in outcomes if o.accepted],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(
            f"No dataset rows were produced. See {out_dir / 'manifest.json'} for rejection details."
        )

    train, valid, test, split_meta = _split_rows(
        all_rows,
        all_meta,
        args.seed,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
    )
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "valid.jsonl", valid)
    _write_jsonl(out_dir / "test.jsonl", test)
    _write_jsonl(out_dir / "samples_metadata.jsonl", split_meta)

    manifest = {
        "generated_at": _now_iso(),
        "status": "ok",
        "script": "scripts/build_github_ts_dataset.py",
        "repos_input_count": len(repos),
        "repos_discovered_count": len(discovered_repos),
        "repos_discovered": discovered_repos,
        "accepted_repos_count": sum(1 for o in outcomes if o.accepted),
        "rejected_repos_count": sum(1 for o in outcomes if not o.accepted),
        "allowed_licenses": sorted(allowed_licenses),
        "include_roots": sorted(include_roots),
        "filters": {
            "min_ts_files": args.min_ts_files,
            "max_files_per_repo": args.max_files_per_repo,
            "min_file_chars": args.min_file_chars,
            "max_file_chars": args.max_file_chars,
            "require_strict_tsconfig": not args.allow_non_strict,
            "allow_forks": bool(args.allow_forks),
            "max_zip_mb": args.max_zip_mb,
        },
        "rows": {
            "total": len(all_rows),
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "out_dir": str(out_dir),
        "files": {
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test.jsonl",
            "samples_metadata": "samples_metadata.jsonl",
            "manifest": "manifest.json",
        },
        "accepted_repos": [o.to_json() for o in outcomes if o.accepted],
        "rejected_repos": [o.to_json() for o in outcomes if not o.accepted],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "rows_total": len(all_rows),
                "train": len(train),
                "valid": len(valid),
                "test": len(test),
                "accepted_repos": [o.repo for o in outcomes if o.accepted],
                "rejected_count": sum(1 for o in outcomes if not o.accepted),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
