"""Dry-run co-evolution season runner.

The runner creates the repeatable phase/artifact backbone for future real
trace collection, scoring, dataset building, LoRA training, evaluation, and
promotion. It deliberately records skipped dry-run phases instead of silently
doing nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .archetypes import CouncilArchetypeRegistry
from .coevolution import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_STATE_PATH,
    CoevolutionState,
    PhaseResult,
    SeasonManifest,
    bootstrap_state,
    next_season_id,
    utc_iso,
    write_json,
)

PHASES = (
    "resolve_population",
    "collect_interaction_traces",
    "score_traces",
    "build_training_datasets",
    "create_train_requests",
    "train_candidate_adapters",
    "evaluate_candidates",
    "gate_and_promote",
    "update_state",
)


def load_or_bootstrap_state(
    *,
    state_path: Path,
    archetype_registry: CouncilArchetypeRegistry,
    bootstrap: bool = False,
) -> CoevolutionState:
    resolved = state_path.expanduser().resolve()
    if bootstrap or not resolved.is_file():
        state = bootstrap_state(archetype_registry)
        state.save(resolved)
        return state
    return CoevolutionState.load(resolved)


def _phase_artifact(run_dir: Path, index: int, result: PhaseResult) -> str:
    phase_dir = run_dir / "phases"
    path = phase_dir / f"{index:02d}_{result.phase}.json"
    write_json(path, result.to_dict())
    return str(path)


def run_dry_season(
    *,
    archetype_registry: CouncilArchetypeRegistry,
    state_path: Path = DEFAULT_STATE_PATH,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    season_id: str | None = None,
    bootstrap: bool = False,
) -> SeasonManifest:
    state = load_or_bootstrap_state(
        state_path=state_path,
        archetype_registry=archetype_registry,
        bootstrap=bootstrap,
    )
    sid = season_id or next_season_id(results_root)
    run_dir = results_root.expanduser().resolve() / "seasons" / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    archetype_ids = sorted(state.archetypes_by_id())
    reward_ids = sorted(state.reward_population.by_id())
    phases: list[PhaseResult] = []

    phase_payloads: list[tuple[str, dict[str, Any], dict[str, str]]] = [
        (
            "resolve_population",
            {
                "archetype_count": len(archetype_ids),
                "reward_matrix_count": len(reward_ids),
                "active_reward_matrix": state.reward_population.active_matrix_id,
            },
            {
                "state_path": str(state_path.expanduser().resolve()),
                "archetype_registry": "training/council_archetype_registry_v1.json",
            },
        ),
        (
            "collect_interaction_traces",
            {
                "dry_run": True,
                "trace_contracts": {
                    entry.archetype_id: entry.lane.trace_schema for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "score_traces",
            {
                "dry_run": True,
                "active_reward_weights": state.reward_population.active().weights,
            },
            {},
        ),
        (
            "build_training_datasets",
            {
                "dry_run": True,
                "dataset_roots": {
                    entry.archetype_id: entry.lane.dataset_root for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "create_train_requests",
            {
                "dry_run": True,
                "request_roots": {
                    entry.archetype_id: entry.lane.request_root for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "train_candidate_adapters",
            {
                "dry_run": True,
                "candidate_roots": {
                    entry.archetype_id: entry.lane.candidate_root for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "evaluate_candidates",
            {
                "dry_run": True,
                "eval_roots": {
                    entry.archetype_id: entry.lane.eval_root for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "gate_and_promote",
            {
                "dry_run": True,
                "promotion_gates": {
                    entry.archetype_id: entry.lane.promotion_gate for entry in archetype_registry.entries
                },
            },
            {},
        ),
        (
            "update_state",
            {
                "dry_run": True,
                "state_changed": False,
                "reason": "dry-run scaffold only; no candidate adapters trained or promoted",
            },
            {"state_path": str(state_path.expanduser().resolve())},
        ),
    ]

    for index, (phase, summary, outputs) in enumerate(phase_payloads, start=1):
        result = PhaseResult(phase=phase, status="skipped_dry_run", summary=summary, outputs=outputs)
        outputs = dict(result.outputs)
        outputs["phase_artifact"] = _phase_artifact(run_dir, index, result)
        phases.append(PhaseResult(phase=phase, status=result.status, summary=result.summary, outputs=outputs))

    manifest = SeasonManifest(
        season_id=sid,
        run_dir=str(run_dir),
        state_path=str(state_path.expanduser().resolve()),
        archetype_ids=archetype_ids,
        reward_matrix_ids=reward_ids,
        dry_run=True,
        phases=phases,
        started_at_utc=utc_iso(),
        finished_at_utc=utc_iso(),
        status="dry_run_complete",
    )
    write_json(run_dir / "SEASON_MANIFEST.json", manifest.to_dict())
    (run_dir / "RUN.md").write_text(_season_markdown(manifest), encoding="utf-8")
    return manifest


def _season_markdown(manifest: SeasonManifest) -> str:
    lines = [
        f"# Council EQ Coevolution {manifest.season_id}",
        "",
        f"- Status: `{manifest.status}`",
        f"- Dry run: `{manifest.dry_run}`",
        f"- State: `{manifest.state_path}`",
        f"- Archetypes: `{len(manifest.archetype_ids)}`",
        f"- Reward matrices: `{len(manifest.reward_matrix_ids)}`",
        "",
        "## Phases",
        "",
    ]
    for phase in manifest.phases:
        lines.append(f"- `{phase.phase}`: `{phase.status}`")
    return "\n".join(lines) + "\n"
