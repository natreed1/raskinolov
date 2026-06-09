#!/usr/bin/env python3
"""Audit adapter datasets for contamination, duplication, and negative priors."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTERS_DIR = REPO / "data" / "lora" / "adapters"
DEFAULT_OUT_JSON = REPO / "benchmarks" / "results" / "adapter_data_audit_v1.json"
DEFAULT_OUT_MD = REPO / "benchmarks" / "results" / "adapter_data_audit_v1.md"

SPECIALIST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "loading_screen": ("loading", "start", "readiness", "campaign", "screen"),
    "hud_status": ("hud", "chip", "status", "clutter", "visibility"),
    "economy_tooltip": ("economy", "gold", "income", "upkeep", "tooltip", "market"),
    "combat_risk": ("combat", "attacker", "defender", "terrain", "risk"),
    "save_load_api_guard": ("save", "load", "schema", "version", "api", "contract"),
    "ai_planning_explanation": ("planning", "intent", "rationale", "strategy", "turn"),
}

NEGATIVE_CUES = (
    "cannot",
    "can't",
    "unable",
    "not enough context",
    "need more details",
    "placeholder",
    "todo",
    "generic suggestion",
    "i do not know",
    "i don't know",
)


@dataclass
class DatasetAudit:
    dataset_id: str
    rows: int
    parse_errors: int
    duplicate_record_ids: int
    duplicate_prompt_assistant_pairs: int
    negative_cue_rows: int
    contamination_rows: int
    plain_text_violations: int
    one_sentence_violations: int
    expected_specialist: str
    risk_score: float
    top_issues: list[str]
    role_counts: dict[str, int]
    contamination_examples: list[dict[str, str]]
    constraint_violation_examples: list[dict[str, str]]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    if not path.is_file():
        return rows, parse_errors
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            parse_errors += 1
    return rows, parse_errors


def _extract_user_assistant(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", ""
    user = ""
    assistant = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "")
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return user, assistant


def _specialist_from_dataset(dataset_id: str) -> str:
    for specialist in SPECIALIST_KEYWORDS:
        if dataset_id.startswith(specialist):
            return specialist
    return ""


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _contamination_label(text: str, expected_specialist: str) -> str:
    if not expected_specialist or not text.strip():
        return ""
    expected_hits = _keyword_hits(text, SPECIALIST_KEYWORDS[expected_specialist])
    if expected_hits >= 2:
        return ""
    best_other_hits = 0
    best_other = ""
    for specialist, keywords in SPECIALIST_KEYWORDS.items():
        if specialist == expected_specialist:
            continue
        hits = _keyword_hits(text, keywords)
        if hits > best_other_hits:
            best_other_hits = hits
            best_other = specialist
    if best_other_hits >= 2 and best_other_hits > expected_hits:
        return best_other
    return ""


def _is_plain_text_violation(user: str, assistant: str) -> bool:
    if "plain text only" not in user.lower():
        return False
    return "```" in assistant or "<" in assistant or "{" in assistant


def _sentence_count(text: str) -> int:
    chunks = [c.strip() for c in re.split(r"[.!?]+", text) if c.strip()]
    return len(chunks)


def _is_one_sentence_violation(user: str, assistant: str) -> bool:
    prompt = user.lower()
    if "one sentence only" not in prompt and "single sentence" not in prompt:
        return False
    return _sentence_count(assistant) != 1


def _risk_score(
    rows: int,
    duplicate_pairs: int,
    negative_rows: int,
    contamination_rows: int,
    plain_text_violations: int,
    one_sentence_violations: int,
) -> float:
    if rows <= 0:
        return 0.0
    dup_rate = duplicate_pairs / rows
    neg_rate = negative_rows / rows
    contam_rate = contamination_rows / rows
    constraint_rate = (plain_text_violations + one_sentence_violations) / rows
    score = (0.35 * contam_rate) + (0.25 * dup_rate) + (0.25 * constraint_rate) + (0.15 * neg_rate)
    return round(score * 100.0, 2)


def _audit_dataset(dataset_dir: Path) -> DatasetAudit:
    dataset_id = dataset_dir.name
    expected_specialist = _specialist_from_dataset(dataset_id)
    all_rows: list[dict[str, Any]] = []
    parse_errors = 0
    for split in ("train", "valid", "test"):
        rows, split_errors = _read_jsonl(dataset_dir / f"{split}.jsonl")
        all_rows.extend(rows)
        parse_errors += split_errors

    record_ids: list[str] = []
    prompt_assistant_counter: Counter[str] = Counter()
    negative_cue_rows = 0
    contamination_rows = 0
    plain_text_violations = 0
    one_sentence_violations = 0
    role_counts: Counter[str] = Counter()
    contamination_examples: list[dict[str, str]] = []
    constraint_examples: list[dict[str, str]] = []

    for row in all_rows:
        dataset_role = str(row.get("dataset_role") or "unknown").strip() or "unknown"
        role_counts[dataset_role] += 1
        record_id = str(row.get("record_id") or "").strip()
        if record_id:
            record_ids.append(record_id)

        user, assistant = _extract_user_assistant(row)
        fingerprint = f"{user.strip()}||{assistant.strip()}"
        if user.strip() or assistant.strip():
            prompt_assistant_counter[fingerprint] += 1

        lowered_assistant = assistant.lower()
        if any(token in lowered_assistant for token in NEGATIVE_CUES):
            negative_cue_rows += 1

        contamination_from_user = _contamination_label(user, expected_specialist)
        contamination_from_assistant = _contamination_label(assistant, expected_specialist)
        contamination_target = contamination_from_user or contamination_from_assistant
        if contamination_target:
            contamination_rows += 1
            if len(contamination_examples) < 3:
                contamination_examples.append(
                    {
                        "record_id": record_id or "<missing>",
                        "dataset_role": dataset_role,
                        "expected_specialist": expected_specialist or "<none>",
                        "dominant_other_specialist": contamination_target,
                        "user_excerpt": user.strip()[:140],
                        "assistant_excerpt": assistant.strip()[:140],
                    }
                )

        plain_text_violation = _is_plain_text_violation(user, assistant)
        one_sentence_violation = _is_one_sentence_violation(user, assistant)
        if plain_text_violation:
            plain_text_violations += 1
        if one_sentence_violation:
            one_sentence_violations += 1
        if (plain_text_violation or one_sentence_violation) and len(constraint_examples) < 3:
            constraint_examples.append(
                {
                    "record_id": record_id or "<missing>",
                    "dataset_role": dataset_role,
                    "violation": "plain_text_only" if plain_text_violation else "one_sentence_only",
                    "user_excerpt": user.strip()[:140],
                    "assistant_excerpt": assistant.strip()[:140],
                }
            )

    rows = len(all_rows)
    duplicate_record_ids = max(0, len(record_ids) - len(set(record_ids)))
    duplicate_prompt_assistant_pairs = sum(count - 1 for count in prompt_assistant_counter.values() if count > 1)
    score = _risk_score(
        rows=rows,
        duplicate_pairs=duplicate_prompt_assistant_pairs,
        negative_rows=negative_cue_rows,
        contamination_rows=contamination_rows,
        plain_text_violations=plain_text_violations,
        one_sentence_violations=one_sentence_violations,
    )

    issues: list[tuple[str, int]] = [
        ("contamination", contamination_rows),
        ("duplicate_pairs", duplicate_prompt_assistant_pairs),
        ("negative_cues", negative_cue_rows),
        ("plain_text_violations", plain_text_violations),
        ("one_sentence_violations", one_sentence_violations),
        ("parse_errors", parse_errors),
        ("duplicate_record_ids", duplicate_record_ids),
    ]
    top_issues = [f"{name}:{count}" for name, count in sorted(issues, key=lambda x: x[1], reverse=True) if count > 0]

    return DatasetAudit(
        dataset_id=dataset_id,
        rows=rows,
        parse_errors=parse_errors,
        duplicate_record_ids=duplicate_record_ids,
        duplicate_prompt_assistant_pairs=duplicate_prompt_assistant_pairs,
        negative_cue_rows=negative_cue_rows,
        contamination_rows=contamination_rows,
        plain_text_violations=plain_text_violations,
        one_sentence_violations=one_sentence_violations,
        expected_specialist=expected_specialist,
        risk_score=score,
        top_issues=top_issues[:5],
        role_counts=dict(sorted(role_counts.items(), key=lambda kv: kv[1], reverse=True)),
        contamination_examples=contamination_examples,
        constraint_violation_examples=constraint_examples,
    )


def _collect_datasets(adapters_dir: Path) -> list[Path]:
    if not adapters_dir.is_dir():
        return []
    datasets: list[Path] = []
    for child in sorted(adapters_dir.iterdir()):
        if not child.is_dir():
            continue
        has_split = any((child / f"{split}.jsonl").is_file() for split in ("train", "valid", "test"))
        if has_split:
            datasets.append(child)
    return datasets


def _write_markdown(path: Path, audits: list[DatasetAudit], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Adapter Data Audit v1")
    lines.append("")
    lines.append(f"- datasets: {summary['datasets']}")
    lines.append(f"- total_rows: {summary['total_rows']}")
    lines.append(f"- high_risk_datasets (score >= 25): {summary['high_risk_datasets']}")
    lines.append("")
    lines.append("## Top risk datasets")
    lines.append("")
    lines.append("| dataset_id | risk_score | rows | contamination | duplicate_pairs | plain_text_violations | one_sentence_violations |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in audits[:10]:
        lines.append(
            f"| {item.dataset_id} | {item.risk_score:.2f} | {item.rows} | {item.contamination_rows} | "
            f"{item.duplicate_prompt_assistant_pairs} | {item.plain_text_violations} | {item.one_sentence_violations} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Contamination is heuristic keyword drift against expected specialist.")
    lines.append("- Constraint checks currently cover `plain text only` and `one sentence only` prompts.")
    lines.append("- Use this report to prioritize data cleanup before retraining adapters.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit adapter datasets for likely negative-association risk factors.")
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    adapters_dir = args.adapters_dir.expanduser().resolve()
    datasets = _collect_datasets(adapters_dir)
    audits = [_audit_dataset(path) for path in datasets]
    audits.sort(key=lambda x: x.risk_score, reverse=True)

    summary = {
        "schema_version": "adapter_data_audit_v1",
        "adapters_dir": str(adapters_dir),
        "datasets": len(audits),
        "total_rows": sum(item.rows for item in audits),
        "high_risk_datasets": sum(1 for item in audits if item.risk_score >= 25.0),
        "max_risk_score": max((item.risk_score for item in audits), default=0.0),
    }
    payload = {
        "summary": summary,
        "audits": [item.__dict__ for item in audits],
    }

    out_json = args.out_json.expanduser().resolve()
    out_md = args.out_md.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, audits, summary)

    top = audits[:5]
    print(json.dumps(summary, indent=2))
    for item in top:
        print(
            f"{item.dataset_id}: score={item.risk_score:.2f} rows={item.rows} "
            f"contam={item.contamination_rows} dup_pairs={item.duplicate_prompt_assistant_pairs} "
            f"plain_text_viol={item.plain_text_violations} one_sent_viol={item.one_sentence_violations}"
        )


if __name__ == "__main__":
    main()
