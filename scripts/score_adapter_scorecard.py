#!/usr/bin/env python3
"""Create a scorecard and verdict for every adapter dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_JSON = REPO / "benchmarks" / "results" / "adapter_data_audit_v1.json"
DEFAULT_OUT_JSON = REPO / "benchmarks" / "results" / "adapter_scorecard_v1.json"
DEFAULT_OUT_MD = REPO / "benchmarks" / "results" / "adapter_scorecard_v1.md"
DEFAULT_REGISTRY_JSON = REPO / "training" / "adapter_registry_v1.json"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_rate(num: float, den: float) -> float:
    return (num / den) if den > 0 else 0.0


def _component_scores(audit: dict[str, Any]) -> dict[str, float]:
    rows = float(max(1, int(audit.get("rows") or 0)))
    contamination = float(audit.get("contamination_rows") or 0)
    plain_viol = float(audit.get("plain_text_violations") or 0)
    one_sent_viol = float(audit.get("one_sentence_violations") or 0)
    dup_pairs = float(audit.get("duplicate_prompt_assistant_pairs") or 0)
    dup_ids = float(audit.get("duplicate_record_ids") or 0)
    negative_cues = float(audit.get("negative_cue_rows") or 0)

    purity = 1.0 - _safe_rate(contamination, rows)
    constraints = 1.0 - _safe_rate((plain_viol + one_sent_viol), rows)
    # Penalize both record-id duplication and text duplication; cap penalty at 100%.
    hygiene_penalty = min(1.0, _safe_rate((dup_pairs + dup_ids), rows))
    hygiene = 1.0 - hygiene_penalty
    tone = 1.0 - _safe_rate(negative_cues, rows)
    # Coverage reward saturates at 160 rows.
    coverage = min(1.0, _safe_rate(rows, 160.0))

    return {
        "purity": round(_clamp01(purity) * 100.0, 2),
        "constraints": round(_clamp01(constraints) * 100.0, 2),
        "hygiene": round(_clamp01(hygiene) * 100.0, 2),
        "tone": round(_clamp01(tone) * 100.0, 2),
        "coverage": round(_clamp01(coverage) * 100.0, 2),
    }


def _final_score(components: dict[str, float]) -> float:
    # Weighted for specialist skill health:
    # - purity + constraints are the hardest gates (65% total)
    # - hygiene catches data artifacts
    # - tone + coverage are secondary
    score = (
        0.35 * components["purity"]
        + 0.30 * components["constraints"]
        + 0.20 * components["hygiene"]
        + 0.10 * components["tone"]
        + 0.05 * components["coverage"]
    )
    return round(score, 2)


def _verdict(score: float, components: dict[str, float], rows: int) -> str:
    if rows < 60:
        return "insufficient_data"
    purity = components["purity"]
    constraints = components["constraints"]
    hygiene = components["hygiene"]

    # Hard fails first.
    if purity < 60.0 or constraints < 85.0:
        return "quarantine"
    if hygiene < 35.0:
        return "rebuild_dataset"
    if score >= 80.0 and purity >= 80.0 and constraints >= 92.0:
        return "promote_candidate"
    if score >= 65.0:
        return "hold_and_monitor"
    return "rebuild_dataset"


def _ranked_entries(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in audits:
        components = _component_scores(audit)
        final = _final_score(components)
        entry = {
            "dataset_id": str(audit.get("dataset_id") or ""),
            "rows": int(audit.get("rows") or 0),
            "expected_specialist": str(audit.get("expected_specialist") or ""),
            "score": final,
            "verdict": _verdict(final, components, int(audit.get("rows") or 0)),
            "components": components,
            "risk_score": float(audit.get("risk_score") or 0.0),
            "top_issues": list(audit.get("top_issues") or []),
            "role_counts": dict(audit.get("role_counts") or {}),
        }
        rows.append(entry)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def _label_dataset(dataset_id: str) -> str:
    if any(tag in dataset_id for tag in ("mock_aug", "ui_merge", "dual", "ablation", "experiment")):
        return "test_ablation"
    true_specialists = {
        "loading_screen",
        "hud_status",
        "economy_tooltip",
        "combat_risk",
        "save_load_api_guard",
        "ai_planning_explanation",
        "economistRL",
    }
    for specialist in true_specialists:
        if dataset_id in {specialist, f"{specialist}_specialist"}:
            return "true_specialist"
    return "other"


def _load_active_true_specialists(registry_path: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not registry_path.is_file():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload.get("entries") or []
    if not isinstance(rows, list):
        return []
    by_dataset = {e["dataset_id"]: e for e in entries}
    true_ids = {
        "loading_screen",
        "hud_status",
        "economy_tooltip",
        "combat_risk",
        "save_load_api_guard",
        "ai_planning_explanation",
        "economistRL",
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        adapter_id = str(row.get("adapter_id") or "")
        if adapter_id not in true_ids:
            continue
        dataset_id = "economistRL_seed" if adapter_id == "economistRL" else f"{adapter_id}_specialist"
        scored = by_dataset.get(dataset_id)
        out.append(
            {
                "adapter_id": adapter_id,
                "dataset_id": dataset_id,
                "active_adapter_path": str(row.get("adapter_path") or ""),
                "score": (scored or {}).get("score"),
                "verdict": (scored or {}).get("verdict", "no_score"),
                "dataset_label": (scored or {}).get("dataset_label", _label_dataset(dataset_id)),
            }
        )
    return out


def _write_markdown(
    path: Path, entries: list[dict[str, Any]], summary: dict[str, Any], active_true_specialists: list[dict[str, Any]]
) -> None:
    lines: list[str] = []
    lines.append("# Adapter Scorecard v1")
    lines.append("")
    lines.append(f"- datasets_scored: {summary['datasets_scored']}")
    lines.append(f"- promote_candidate: {summary['verdict_counts'].get('promote_candidate', 0)}")
    lines.append(f"- hold_and_monitor: {summary['verdict_counts'].get('hold_and_monitor', 0)}")
    lines.append(f"- rebuild_dataset: {summary['verdict_counts'].get('rebuild_dataset', 0)}")
    lines.append(f"- quarantine: {summary['verdict_counts'].get('quarantine', 0)}")
    lines.append("")
    lines.append("## Active True Specialists")
    lines.append("")
    lines.append("| adapter_id | dataset_id | score | verdict | dataset_label | active_adapter_path |")
    lines.append("|---|---|---:|---|---|---|")
    for row in active_true_specialists:
        score = row.get("score")
        score_txt = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
        lines.append(
            f"| {row['adapter_id']} | {row['dataset_id']} | {score_txt} | {row['verdict']} | "
            f"{row['dataset_label']} | {row['active_adapter_path']} |"
        )
    lines.append("")
    lines.append(
        "| dataset_id | score | verdict | dataset_label | purity | constraints | hygiene | tone | coverage | rows | risk_score |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in entries:
        c = row["components"]
        lines.append(
            f"| {row['dataset_id']} | {row['score']:.2f} | {row['verdict']} | "
            f"{row['dataset_label']} | "
            f"{c['purity']:.2f} | {c['constraints']:.2f} | {c['hygiene']:.2f} | "
            f"{c['tone']:.2f} | {c['coverage']:.2f} | {row['rows']} | {row['risk_score']:.2f} |"
        )
    lines.append("")
    lines.append("## Scoring policy")
    lines.append("")
    lines.append("- score = 0.35*purity + 0.30*constraints + 0.20*hygiene + 0.10*tone + 0.05*coverage")
    lines.append("- hard gates: purity < 60 or constraints < 85 => `quarantine`")
    lines.append("- hygiene < 35 => `rebuild_dataset`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scorecard and verdicts for every adapter dataset.")
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_JSON)
    args = parser.parse_args()

    audit_path = args.audit_json.expanduser().resolve()
    if not audit_path.is_file():
        raise SystemExit(f"Audit JSON not found: {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audits = payload.get("audits") or []
    if not isinstance(audits, list):
        raise SystemExit(f"Invalid audit payload in: {audit_path}")

    entries = _ranked_entries(audits)
    for row in entries:
        row["dataset_label"] = _label_dataset(str(row["dataset_id"]))
    verdict_counts: dict[str, int] = {}
    for row in entries:
        verdict = row["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    label_counts: dict[str, int] = {}
    for row in entries:
        label = row["dataset_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    active_true_specialists = _load_active_true_specialists(
        args.registry_json.expanduser().resolve(), entries
    )

    summary = {
        "schema_version": "adapter_scorecard_v1",
        "datasets_scored": len(entries),
        "verdict_counts": verdict_counts,
        "dataset_label_counts": label_counts,
        "mean_score": round(sum(row["score"] for row in entries) / max(1, len(entries)), 2),
        "min_score": round(min((row["score"] for row in entries), default=0.0), 2),
        "max_score": round(max((row["score"] for row in entries), default=0.0), 2),
    }
    out = {
        "summary": summary,
        "active_true_specialists": active_true_specialists,
        "entries": entries,
    }

    out_json = args.out_json.expanduser().resolve()
    out_md = args.out_md.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, entries, summary, active_true_specialists)

    print(json.dumps(summary, indent=2))
    print("Top 5:")
    for row in entries[:5]:
        print(f"{row['dataset_id']}: {row['score']:.2f} ({row['verdict']})")
    print("Bottom 5:")
    for row in entries[-5:]:
        print(f"{row['dataset_id']}: {row['score']:.2f} ({row['verdict']})")


if __name__ == "__main__":
    main()
