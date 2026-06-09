#!/usr/bin/env python3
"""Generate mock pairwise specialist rows for supervised training.

The output schema matches the fields consumed by specialist dataset builders:
- task.id
- task.prompt
- winner_output

Additional metadata is included for traceability and optional diagnostics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SpecialistProfile:
    specialist_id: str
    role_line: str
    fallback_keywords: List[str]


PROFILES: Dict[str, SpecialistProfile] = {
    "loading_screen": SpecialistProfile(
        specialist_id="loading_screen",
        role_line="Keep loading copy concise, atmospheric, and status-aware.",
        fallback_keywords=["empire", "readiness", "supply", "banner"],
    ),
    "hud_status": SpecialistProfile(
        specialist_id="hud_status",
        role_line="Emit compact HUD labels with strategic signal density.",
        fallback_keywords=["morale", "supply", "risk", "pressure"],
    ),
    "economy_tooltip": SpecialistProfile(
        specialist_id="economy_tooltip",
        role_line="Explain economy cause/effect and concrete turn actions.",
        fallback_keywords=["gold", "income", "upkeep", "market"],
    ),
    "combat_risk": SpecialistProfile(
        specialist_id="combat_risk",
        role_line="Summarize combat posture, risk drivers, and tactical caution.",
        fallback_keywords=["attacker", "defender", "terrain", "morale"],
    ),
    "save_load_api_guard": SpecialistProfile(
        specialist_id="save_load_api_guard",
        role_line="Prioritize typed contracts, schema checks, and compatibility.",
        fallback_keywords=["schema", "version", "errors", "migration"],
    ),
    "ai_planning_explanation": SpecialistProfile(
        specialist_id="ai_planning_explanation",
        role_line="State intent, rationale, and one follow-up action.",
        fallback_keywords=["defense", "expansion", "risk", "reinforce"],
    ),
}

STYLE_SUFFIXES = [
    "Use practical strategy language and keep the response directly actionable.",
    "Favor specific next-turn guidance over generic flavor text.",
    "Keep formatting strict when the prompt requests constrained output shape.",
]

LOSER_SNIPPETS = [
    "I cannot determine context, please provide more details.",
    "General suggestion: improve things gradually over time.",
    "```python\nprint('placeholder')\n```",
]


def _read_tasks(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Expected list in tasks file: {path}")
    rows: List[Dict[str, Any]] = []
    for task in payload:
        if not isinstance(task, dict):
            continue
        prompt = str(task.get("prompt") or "").strip()
        specialists = [str(x) for x in (task.get("specialists") or [])]
        if not prompt or not specialists:
            continue
        rows.append(task)
    return rows


def _keywords(task: Dict[str, Any], specialist: str) -> List[str]:
    expect = task.get("expect") or {}
    tokens: List[str] = [str(x) for x in (expect.get("all_contains") or [])]
    tokens.extend(str(x) for x in (expect.get("any_contains") or []))
    deduped: List[str] = []
    for token in tokens:
        tok = token.strip()
        if tok and tok not in deduped:
            deduped.append(tok)
    profile = PROFILES.get(specialist)
    if len(deduped) < 3 and profile:
        for token in profile.fallback_keywords:
            if token not in deduped:
                deduped.append(token)
            if len(deduped) >= 4:
                break
    return deduped[:4]


def _winner_text(task: Dict[str, Any], specialist: str, variant_idx: int, rng: random.Random) -> str:
    profile = PROFILES[specialist]
    prompt = str(task.get("prompt") or "").strip()
    category = str(task.get("category") or "general")
    words = _keywords(task, specialist)
    anchor = ", ".join(words[:3]) if words else "strategic clarity"
    style = STYLE_SUFFIXES[variant_idx % len(STYLE_SUFFIXES)]
    if "constraints" in category or "exactly" in prompt.lower():
        return (
            f"{profile.role_line} Focus terms: {anchor}. "
            "Respect literal output constraints and avoid extraneous framing. "
            f"{style}"
        )
    return (
        f"{profile.role_line} Explain using {anchor}. "
        "Call out one immediate action and one trade-off so the player can act this turn. "
        f"{style}"
    )


def _loser_text(rng: random.Random, variant_idx: int) -> str:
    return LOSER_SNIPPETS[variant_idx % len(LOSER_SNIPPETS)]


def _iter_task_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        p = path.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Task file not found: {p}")
        yield p


def build_mock_pairwise_rows(
    task_files: Iterable[Path],
    repeats_per_task: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for task_file in _iter_task_files(task_files):
        tasks = _read_tasks(task_file)
        for task in tasks:
            task_id = str(task.get("id") or "")
            prompt = str(task.get("prompt") or "").strip()
            specialists = [str(x) for x in (task.get("specialists") or [])]
            eligible = [s for s in specialists if s in PROFILES]
            for specialist in eligible:
                for idx in range(repeats_per_task):
                    winner = _winner_text(task, specialist, idx, rng)
                    loser = _loser_text(rng, idx)
                    rows.append(
                        {
                            "task": {
                                "id": task_id or f"mock-{specialist}",
                                "category": str(task.get("category") or "general"),
                                "prompt": prompt,
                                "specialists": [specialist],
                            },
                            "winner_model": "mock_specialist_teacher_v1",
                            "winner_output": winner,
                            "loser_model": "mock_baseline_weak_v1",
                            "loser_output": loser,
                            "winner_label": "local",
                            "loser_label": "frontier",
                            "winner_score": 1.0,
                            "loser_score": 0.0,
                            "preference_strength": 0.85,
                            "loser_error": "generic_or_off_target",
                            "source": "mock_pairwise_generator_v1",
                            "metadata": {
                                "specialist_id": specialist,
                                "task_file": str(task_file.relative_to(REPO)),
                                "variant_idx": idx,
                            },
                        }
                    )
    rng.shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mock specialist pairwise training data.")
    parser.add_argument(
        "--task-file",
        type=Path,
        action="append",
        dest="task_files",
        default=[],
        help="Task JSON file(s) with specialist-tagged prompts; can repeat.",
    )
    parser.add_argument(
        "--repeats-per-task",
        type=int,
        default=3,
        help="How many mock winner/loser variants to generate per task/specialist pair.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO / "benchmarks" / "results" / "mock_specialist_pairwise_training_data_v1.jsonl",
    )
    args = parser.parse_args()

    task_files = args.task_files or [REPO / "benchmarks" / "specialist_benchmark_tasks.json"]
    rows = build_mock_pairwise_rows(
        task_files=task_files,
        repeats_per_task=max(1, int(args.repeats_per_task)),
        seed=int(args.seed),
    )
    output = args.output_jsonl.expanduser().resolve()
    _write_jsonl(output, rows)
    summary = {
        "schema_version": "mock_specialist_pairwise_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "output_jsonl": str(output),
        "task_files": [str(Path(p).expanduser().resolve()) for p in task_files],
        "repeats_per_task": int(args.repeats_per_task),
        "seed": int(args.seed),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
