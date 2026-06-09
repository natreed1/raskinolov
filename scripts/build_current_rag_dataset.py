#!/usr/bin/env python3
"""
Build the current RAG dataset from benchmark row JSONL files.

The output keeps optional base entries and appends domain/subskill entries derived
from common failure patterns, so prompt augmentation can include evidence-backed
guidance instead of only handcrafted generic snippets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CORPUS = ROOT / "data" / "rag" / "current_rag_base_dataset.json"
DEFAULT_OUTPUT_CORPUS = ROOT / "data" / "rag" / "current_rag_dataset.json"
DEFAULT_OUTPUT_STATS = ROOT / "benchmarks" / "results" / "current_rag_dataset_stats.json"
DEFAULT_TASKS_GLOB = ["benchmarks/results/cloud_ablation_runtime_tasks_*.json"]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_task_prompt_map(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = _load_json(path)
        except Exception:  # pylint: disable=broad-except
            continue
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id") or "").strip()
            prompt = str(task.get("prompt") or "").strip()
            if tid and prompt and tid not in out:
                out[tid] = prompt
    return out


def _failure_bucket(row: Dict[str, Any]) -> str:
    if row.get("accepted") is True:
        return "accepted"
    if str(row.get("apply_status") or "") == "no_applyable_changes":
        return "no_applyable_changes"
    failure_class = str(row.get("failure_class") or "").strip()
    if failure_class:
        return failure_class
    if str(row.get("verify_status") or "") != "passed":
        return "verify_failed_unknown"
    if str(row.get("error") or "").strip():
        return "runtime_error"
    return "rejected_other"


def _guidance_for_bucket(bucket: str) -> str:
    if bucket == "no_applyable_changes":
        return (
            "Prioritize strict applyable output shape: start with unified diff or fenced files "
            "that include repo-relative file paths; avoid prose before the first patch line."
        )
    if bucket.startswith("verify_"):
        return (
            "Bias toward verify-safe edits: keep changes minimal, preserve exports/imports and "
            "existing type contracts, and avoid introducing new symbols unless fully wired."
        )
    if bucket.startswith("frontier_") or bucket.startswith("runtime_"):
        return (
            "Reduce runtime risk by avoiding broad refactors, keeping deterministic control flow, "
            "and validating assumptions against existing module boundaries."
        )
    return (
        "Keep edits focused and contract-safe, and ensure output is directly applyable by the "
        "arena harness."
    )


def _slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in (text or "").lower())
    out = "_".join(part for part in out.split("_") if part)
    return out or "unknown"


def _tokens_for_alias(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower()) if tok]


def _aliases_for_label(label: str) -> List[str]:
    label = str(label or "").strip().lower()
    if not label:
        return []
    words = [w for w in label.split("_") if w]
    aliases = {
        label,
        " ".join(words),
        "-".join(words),
    }
    if len(words) > 1:
        aliases.add(" ".join(words[:2]))
        aliases.add(" ".join(words[-2:]))
    return sorted(a for a in aliases if a)


def _symptom_for_bucket(bucket: str) -> str:
    if bucket == "no_applyable_changes":
        return "model returns non-applyable output shape (missing file path/diff contract)"
    if bucket.startswith("verify_"):
        return "verification/typecheck failures from unsafe symbol or contract edits"
    if bucket.startswith("frontier_") or bucket.startswith("runtime_"):
        return "runtime-side failures under infra/runtime constraints"
    return "task-level rejection from non-passing output"


def _entry_text(
    *,
    label: str,
    top_buckets: List[Tuple[str, int]],
) -> str:
    primary = top_buckets[0][0] if top_buckets else "rejected_other"
    return f"For {label}: {_guidance_for_bucket(primary)}"


def _build_corpus(
    *,
    base_corpus: Dict[str, Any],
    rows: List[Dict[str, Any]],
    task_prompts: Dict[str, str],
    include_base_entries: bool,
    min_domain_failures: int,
    min_subskill_failures: int,
    max_subskill_entries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_entries = list(base_corpus.get("entries") or []) if include_base_entries else []
    generated_entries: List[Dict[str, Any]] = []

    domain_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    subskill_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    domain_samples: Counter[str] = Counter()
    subskill_samples: Counter[Tuple[str, str]] = Counter()
    domain_task_ids: Dict[str, Counter[str]] = defaultdict(Counter)
    subskill_task_ids: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

    for row in rows:
        domain = str(row.get("domain_primary_normalized") or "unknown").strip() or "unknown"
        subskill = str(row.get("subskill") or "unknown").strip() or "unknown"
        bucket = _failure_bucket(row)
        domain_counts[domain][bucket] += 1
        subskill_counts[(domain, subskill)][bucket] += 1
        domain_samples[domain] += 1
        subskill_samples[(domain, subskill)] += 1
        task_id = str(row.get("task_id") or "").strip()
        if task_id and bucket != "accepted":
            domain_task_ids[domain][task_id] += 1
            subskill_task_ids[(domain, subskill)][task_id] += 1

    # Domain-level entries.
    for domain, counts in sorted(domain_counts.items()):
        total_failures = sum(v for k, v in counts.items() if k != "accepted")
        if total_failures < min_domain_failures:
            continue
        top_buckets = [(k, v) for k, v in counts.most_common(3) if k != "accepted"]
        if not top_buckets:
            continue
        top_task_ids = [tid for tid, _ in domain_task_ids[domain].most_common(4)]
        retrieval_examples = [task_prompts.get(tid, "") for tid in top_task_ids if task_prompts.get(tid, "")]
        retrieval_signatures = sorted(
            {
                str(sub).strip().lower()
                for (dom, sub), c in subskill_counts.items()
                if dom == domain and sum(v for k, v in c.items() if k != "accepted") > 0
            }
        )[:8]
        retrieval_aliases = sorted(set(_aliases_for_label(domain)))
        retrieval_symptoms = [_symptom_for_bucket(bucket) for bucket, _ in top_buckets]
        retrieval_description = (
            f"Domain {domain} failure retrieval profile. "
            f"Focus signatures: {', '.join(retrieval_signatures[:4])}. "
            f"Primary symptoms: {', '.join(retrieval_symptoms[:2])}."
        )
        generated_entries.append(
            {
                "id": f"failure_profile_domain_{_slug(domain)}",
                "tags": [domain, "failure_profile", "domain_failure_profile"],
                "text": _entry_text(
                    label=f"domain '{domain}'",
                    top_buckets=top_buckets,
                ),
                "retrieval_description": retrieval_description,
                "retrieval_aliases": retrieval_aliases,
                "retrieval_symptoms": retrieval_symptoms,
                "retrieval_task_signatures": retrieval_signatures,
                "retrieval_examples": retrieval_examples[:3],
            }
        )

    # Subskill-level entries, sorted by failure volume.
    ranked_subskills: List[Tuple[int, str, str, Counter[str]]] = []
    for (domain, subskill), counts in subskill_counts.items():
        total_failures = sum(v for k, v in counts.items() if k != "accepted")
        if total_failures < min_subskill_failures:
            continue
        ranked_subskills.append((total_failures, domain, subskill, counts))
    ranked_subskills.sort(reverse=True)

    for total_failures, domain, subskill, counts in ranked_subskills[: max_subskill_entries]:
        top_buckets = [(k, v) for k, v in counts.most_common(3) if k != "accepted"]
        if not top_buckets:
            continue
        top_task_ids = [tid for tid, _ in subskill_task_ids[(domain, subskill)].most_common(4)]
        retrieval_examples = [task_prompts.get(tid, "") for tid in top_task_ids if task_prompts.get(tid, "")]
        retrieval_aliases = sorted(set(_aliases_for_label(subskill) + _aliases_for_label(domain)))
        retrieval_symptoms = [_symptom_for_bucket(bucket) for bucket, _ in top_buckets]
        retrieval_signatures = sorted(set([subskill, domain] + _tokens_for_alias(subskill)))[:10]
        retrieval_description = (
            f"Subskill {subskill} in domain {domain}. "
            f"Failure symptoms: {', '.join(retrieval_symptoms[:2])}. "
            f"Related aliases: {', '.join(retrieval_aliases[:5])}."
        )
        generated_entries.append(
            {
                "id": f"failure_profile_subskill_{_slug(domain)}_{_slug(subskill)}",
                "tags": [domain, subskill, "failure_profile", "subskill_failure_profile"],
                "text": _entry_text(
                    label=f"subskill '{subskill}' in domain '{domain}'",
                    top_buckets=top_buckets,
                ),
                "retrieval_description": retrieval_description,
                "retrieval_aliases": retrieval_aliases,
                "retrieval_symptoms": retrieval_symptoms,
                "retrieval_task_signatures": retrieval_signatures,
                "retrieval_examples": retrieval_examples[:3],
            }
        )

    output = {
        "schema_version": "current_rag_dataset",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_schema_version": base_corpus.get("schema_version"),
        "entries": base_entries + generated_entries,
    }

    stats = {
        "input_rows": len(rows),
        "base_entries": len(base_entries),
        "generated_entries": len(generated_entries),
        "output_entries": len(output["entries"]),
        "domains_seen": len(domain_counts),
        "subskills_seen": len(subskill_counts),
        "task_prompts_available": len(task_prompts),
        "top_domains_by_failures": [
            {
                "domain": domain,
                "failures": sum(v for k, v in counts.items() if k != "accepted"),
                "accepted": counts.get("accepted", 0),
                "top_failure_buckets": [
                    {"bucket": k, "count": v}
                    for k, v in counts.most_common(3)
                    if k != "accepted"
                ],
            }
            for domain, counts in sorted(
                domain_counts.items(),
                key=lambda kv: sum(v for k, v in kv[1].items() if k != "accepted"),
                reverse=True,
            )[:20]
        ],
        "top_subskills_by_failures": [
            {
                "domain": domain,
                "subskill": subskill,
                "failures": total_failures,
                "accepted": counts.get("accepted", 0),
                "top_failure_buckets": [
                    {"bucket": k, "count": v}
                    for k, v in counts.most_common(3)
                    if k != "accepted"
                ],
            }
            for total_failures, domain, subskill, counts in ranked_subskills[:50]
        ],
    }
    return output, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the current RAG dataset.")
    parser.add_argument(
        "--rows-jsonl",
        action="append",
        default=[],
        help="Explicit row JSONL path(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--rows-glob",
        action="append",
        default=["benchmarks/results/cloud_ablation_rows_*.jsonl"],
        help="Glob(s) relative to repo root for row JSONL files.",
    )
    parser.add_argument(
        "--tasks-json",
        action="append",
        default=[],
        help="Explicit runtime tasks JSON path(s) with task id/prompt for retrieval examples.",
    )
    parser.add_argument(
        "--tasks-glob",
        action="append",
        default=DEFAULT_TASKS_GLOB,
        help="Glob(s) relative to repo root for runtime tasks JSON files.",
    )
    parser.add_argument(
        "--include-base-entries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include base corpus entries in output (default false).",
    )
    parser.add_argument("--base-corpus", type=Path, default=DEFAULT_BASE_CORPUS)
    parser.add_argument("--out-corpus", type=Path, default=DEFAULT_OUTPUT_CORPUS)
    parser.add_argument("--out-stats", type=Path, default=DEFAULT_OUTPUT_STATS)
    parser.add_argument("--min-domain-failures", type=int, default=20)
    parser.add_argument("--min-subskill-failures", type=int, default=4)
    parser.add_argument("--max-subskill-entries", type=int, default=40)
    args = parser.parse_args()

    base_corpus = _load_json(args.base_corpus.expanduser().resolve())

    explicit_paths = [Path(p).expanduser().resolve() for p in args.rows_jsonl]
    glob_paths: List[Path] = []
    for pattern in args.rows_glob:
        glob_paths.extend(sorted(ROOT.glob(pattern)))
    row_paths = list(dict.fromkeys(explicit_paths + glob_paths))
    rows = _load_rows(row_paths)
    if not rows:
        raise SystemExit("No input rows found.")
    explicit_tasks = [Path(p).expanduser().resolve() for p in args.tasks_json]
    glob_tasks: List[Path] = []
    for pattern in args.tasks_glob:
        glob_tasks.extend(sorted(ROOT.glob(pattern)))
    task_prompt_map = _load_task_prompt_map(list(dict.fromkeys(explicit_tasks + glob_tasks)))

    corpus, stats = _build_corpus(
        base_corpus=base_corpus,
        rows=rows,
        task_prompts=task_prompt_map,
        include_base_entries=bool(args.include_base_entries),
        min_domain_failures=max(1, int(args.min_domain_failures)),
        min_subskill_failures=max(1, int(args.min_subskill_failures)),
        max_subskill_entries=max(1, int(args.max_subskill_entries)),
    )

    out_corpus = args.out_corpus.expanduser().resolve()
    out_stats = args.out_stats.expanduser().resolve()
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_corpus.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    out_stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"row_files={len(row_paths)}")
    print(f"task_prompts={len(task_prompt_map)}")
    print(f"base_entries={stats['base_entries']}")
    print(f"generated_entries={stats['generated_entries']}")
    print(f"output_entries={stats['output_entries']}")
    print(f"out_corpus={out_corpus}")
    print(f"out_stats={out_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
