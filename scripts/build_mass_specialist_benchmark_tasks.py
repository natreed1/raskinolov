#!/usr/bin/env python3
"""Generate 30-task specialist benchmark suites and ACI aggregate set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"


MASS_FILES = {
    "loading_screen": "loading_screen_mass_tasks_v1.json",
    "hud_status": "hud_status_mass_tasks_v1.json",
    "economy_tooltip": "economy_tooltip_mass_tasks_v1.json",
    "combat_risk": "combat_risk_mass_tasks_v1.json",
    "save_load_api_guard": "save_load_api_guard_mass_tasks_v1.json",
    "ai_planning_explanation": "ai_planning_explanation_mass_tasks_v1.json",
}


def _task(
    task_id: str,
    category: str,
    specialist: str,
    prompt: str,
    expect: Dict[str, Any],
    domains: Sequence[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": task_id,
        "category": category,
        "specialists": [specialist],
        "prompt": prompt,
        "expect": expect,
    }
    row["domains"] = list(dict.fromkeys(domains))
    return row


def _write(name: str, rows: List[Dict[str, Any]]) -> None:
    path = BENCH / name
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(path)


def _loading_rows() -> List[Dict[str, Any]]:
    # Keep existing richer suite as canonical for loading.
    path = BENCH / "loading_screen_mass_tasks_v1.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _hud_rows() -> List[Dict[str, Any]]:
    s = "hud_status"
    rows = [
        _task("hud-mass-chips-core-01", "hud_core", s, "List exactly four compact HUD chips including morale and supply plus two risk signals.", {"all_contains": ["morale", "supply"], "any_contains": ["risk", "pressure", "territory", "upkeep"], "min_chars": 35}, ["hud", "status"]),
        _task("hud-mass-terse-values-02", "hud_core", s, "Write three terse HUD labels with values for observer mode.", {"any_contains": ["Morale", "Supply", "Risk", "Pressure", "Territory"], "none_contains": ["```", "function ", "const "], "min_chars": 35}, ["hud", "status"]),
        _task("hud-mass-visibility-note-03", "hud_core", s, "In 2-3 sentences, suggest one HUD visibility improvement for a dark medieval panel.", {"any_contains": ["visibility", "contrast", "readable", "hierarchy"], "min_chars": 95}, ["hud", "ui"]),
        _task("hud-mass-priority-order-04", "hud_ui_change", s, "Propose ordering for morale, supply, and risk in a compact HUD block and explain why.", {"all_contains": ["morale", "supply"], "any_contains": ["risk", "priority", "scan"], "min_chars": 90}, ["hud", "ui"]),
        _task("hud-mass-no-clutter-05", "hud_ui_change", s, "Describe one change that reduces HUD clutter while keeping strategic signal quality.", {"any_contains": ["clutter", "compact", "signal", "density"], "min_chars": 85}, ["hud", "ui"]),
        _task("hud-mass-alert-tone-06", "hud_ui_change", s, "Write one warning-style HUD line and one stable-state line with terse tactical wording.", {"any_contains": ["warning", "stable", "ready", "risk"], "min_chars": 55}, ["hud", "ui"]),
        _task("hud-mass-constraints-07", "hud_constraints", s, "Plain text only: output one comma-separated line of four HUD chips.", {"all_contains": [","], "any_contains": ["Morale", "Supply", "Risk", "Pressure"], "none_contains": ["```"], "min_chars": 35}, ["hud", "constraints"]),
        _task("hud-mass-constraints-08", "hud_constraints", s, "Exactly two lines: first for morale status, second for supply status.", {"all_contains": ["morale", "supply"], "min_chars": 30}, ["hud", "constraints"]),
        _task("hud-mass-constraints-09", "hud_constraints", s, "One sentence only: explain why HUD should keep morale and supply always visible.", {"all_contains": ["morale", "supply"], "min_chars": 55}, ["hud", "constraints"]),
        _task("hud-mass-transfer-load-10", "hud_transfer", s, "Write a loading-screen hint that previews upcoming HUD morale/supply pressure.", {"all_contains": ["morale", "supply"], "any_contains": ["pressure", "preview", "status"], "min_chars": 60}, ["hud", "loading_screen"]),
        _task("hud-mass-transfer-econ-11", "hud_transfer", s, "Provide one HUD tooltip line connecting supply pressure to net income.", {"all_contains": ["supply"], "any_contains": ["income", "gold", "upkeep"], "min_chars": 55}, ["hud", "economy"]),
        _task("hud-mass-transfer-combat-12", "hud_transfer", s, "Give one HUD combat-prep line that includes morale and terrain awareness.", {"all_contains": ["morale", "terrain"], "any_contains": ["risk", "combat", "preview"], "min_chars": 55}, ["hud", "combat"]),
        _task("hud-mass-terse-casing-13", "hud_constraints", s, "Write exactly three short HUD lines with numeric values, and each line must start with one of: Morale, Supply, Risk.", {"all_contains": ["Morale", "Supply"], "any_contains": ["Risk", "Pressure", "%"], "none_contains": ["```"], "min_chars": 45}, ["hud", "constraints"]),
        _task("hud-mass-terse-observer-14", "hud_core", s, "Observer mode is paused. Provide three compact HUD labels with values and keep each label to three words or fewer.", {"any_contains": ["Morale", "Supply", "Risk", "Pressure", "Territory", "Upkeep"], "none_contains": ["```"], "min_chars": 40}, ["hud", "status"]),
        _task("hud-mass-terse-snapshot-15", "hud_constraints", s, "Return one comma-separated HUD snapshot line with four chips. Include Morale and Supply and at least one risk-style chip.", {"all_contains": [",", "Morale", "Supply"], "any_contains": ["Risk", "Pressure", "Territory"], "none_contains": ["```"], "min_chars": 45}, ["hud", "constraints"]),
        _task("hud-mass-terse-ready-16", "hud_core", s, "Write three HUD status labels for a ready-check panel. Include Morale and Supply with values, plus one urgency indicator.", {"all_contains": ["Morale", "Supply"], "any_contains": ["Risk", "Pressure", "Ready"], "min_chars": 45}, ["hud", "status"]),
        _task("hud-mass-terse-contrast-17", "hud_ui_change", s, "In two short lines, show one low-risk HUD state and one high-risk HUD state using compact labels and values.", {"any_contains": ["Risk", "Pressure", "Morale", "Supply"], "min_chars": 50}, ["hud", "ui"]),
        _task("hud-mass-terse-economy-link-18", "hud_transfer", s, "Provide two compact HUD chips that connect Supply pressure to Income trend.", {"all_contains": ["Supply"], "any_contains": ["Income", "gold", "upkeep", "Pressure"], "min_chars": 45}, ["hud", "economy"]),
        _task("hud-mass-terse-combat-link-19", "hud_transfer", s, "Write two terse HUD chips for combat preview that include Morale and one terrain/risk indicator.", {"all_contains": ["Morale"], "any_contains": ["terrain", "Risk", "Pressure", "combat"], "min_chars": 45}, ["hud", "combat"]),
        _task("hud-mass-terse-colon-format-20", "hud_constraints", s, "Output exactly three lines in Label: Value format, using Morale, Supply, and Risk.", {"all_contains": ["Morale", "Supply"], "any_contains": ["Risk", ":"], "none_contains": ["```"], "min_chars": 45}, ["hud", "constraints"]),
        _task("hud-mass-domain-balance-21", "hud_multidomain", s, "Design a compact HUD strip that shows morale, supply, economy pressure, and combat readiness in one row.", {"all_contains": ["morale", "supply"], "any_contains": ["economy", "income", "combat", "risk"], "min_chars": 85}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-tooltip-22", "hud_multidomain", s, "Write one tooltip that links HUD risk level to both net gold trend and expected battle posture.", {"any_contains": ["risk", "gold", "income", "battle", "posture"], "min_chars": 80}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-loading-23", "hud_multidomain", s, "Draft two loading-to-HUD transition lines that mention morale, supply, and AI intent.", {"all_contains": ["morale", "supply"], "any_contains": ["intent", "defend", "expand", "reinforce"], "min_chars": 70}, ["hud", "loading_screen", "planning"]),
        _task("hud-mass-domain-save-24", "hud_multidomain", s, "Provide a HUD system line confirming save-state compatibility plus current morale/supply snapshot.", {"any_contains": ["save", "compatibility", "morale", "supply"], "min_chars": 65}, ["hud", "save_load"]),
        _task("hud-mass-domain-compact-25", "hud_multidomain", s, "One comma-separated line with four chips: morale, supply, economy pressure, combat risk.", {"all_contains": [",", "morale", "supply"], "any_contains": ["economy", "income", "combat", "risk"], "min_chars": 55}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-why-26", "hud_multidomain", s, "In 2 sentences explain why HUD should expose morale/supply alongside economy and combat pressure.", {"all_contains": ["morale", "supply"], "any_contains": ["economy", "combat", "risk"], "min_chars": 85}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-ai-27", "hud_multidomain", s, "Write a war-room HUD note that ties AI intent to morale, supply, and frontline risk.", {"all_contains": ["morale", "supply"], "any_contains": ["intent", "risk", "frontline"], "min_chars": 75}, ["hud", "planning", "combat"]),
        _task("hud-mass-domain-recovery-28", "hud_multidomain", s, "Create one warning line and one recovery line connecting low supply to economy drag and combat risk.", {"all_contains": ["supply"], "any_contains": ["economy", "income", "combat", "risk", "recover"], "min_chars": 80}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-minimal-29", "hud_multidomain", s, "Plain text only: summarize morale, supply, economy trend, and battle readiness in two lines.", {"all_contains": ["morale", "supply"], "any_contains": ["economy", "income", "battle", "readiness"], "none_contains": ["```"], "min_chars": 70}, ["hud", "economy", "combat"]),
        _task("hud-mass-domain-ops-30", "hud_multidomain", s, "Propose a compact command-center HUD block that merges status chips, save integrity, and AI intent tags.", {"any_contains": ["status", "save", "integrity", "intent", "chips"], "min_chars": 85}, ["hud", "save_load", "planning"]),
    ]
    return rows


def _economy_rows() -> List[Dict[str, Any]]:
    s = "economy_tooltip"
    rows = [
        _task("eco-mass-cause-effect-01", "eco_core", s, "In 2-4 sentences, explain why net gold dropped this turn using upkeep, workforce, and markets.", {"all_contains": ["gold"], "any_contains": ["upkeep", "workforce", "market"], "min_chars": 100}, ["economy"]),
        _task("eco-mass-actionable-02", "eco_core", s, "Write a tooltip that explains low income and gives two next-turn actions with one trade-off.", {"any_contains": ["income", "gold", "trade-off", "tradeoff"], "min_chars": 105}, ["economy"]),
        _task("eco-mass-territory-lines-03", "eco_core", s, "Explain in 2-3 sentences how territory and supply lines can pressure income.", {"any_contains": ["territory", "supply"], "all_contains": ["income"], "min_chars": 90}, ["economy"]),
        _task("eco-mass-ui-hint-04", "eco_ui_change", s, "Suggest one tooltip UI change that makes economy cause-and-effect easier to scan.", {"any_contains": ["tooltip", "scan", "cause", "effect", "hierarchy"], "min_chars": 90}, ["economy", "ui"]),
        _task("eco-mass-short-copy-05", "eco_ui_change", s, "Write one compact warning line and one recovery line for economy pressure.", {"any_contains": ["warning", "recover", "pressure", "income"], "min_chars": 55}, ["economy", "ui"]),
        _task("eco-mass-comparison-06", "eco_ui_change", s, "In 2 sentences, contrast short-term income fixes versus long-term economy stability.", {"any_contains": ["short-term", "long-term", "stability", "income"], "min_chars": 80}, ["economy", "ui"]),
        _task("eco-mass-constraints-07", "eco_constraints", s, "Plain text only: one sentence that includes gold, upkeep, and one action verb.", {"all_contains": ["gold", "upkeep"], "any_contains": ["reduce", "shift", "build", "secure"], "none_contains": ["```"], "min_chars": 45}, ["economy", "constraints"]),
        _task("eco-mass-constraints-08", "eco_constraints", s, "Output one comma-separated line with exactly four economy factors.", {"all_contains": [","], "any_contains": ["income", "gold", "upkeep", "market", "supply"], "min_chars": 35}, ["economy", "constraints"]),
        _task("eco-mass-constraints-09", "eco_constraints", s, "Exactly two lines: first describes the drop, second gives one mitigation step.", {"any_contains": ["drop", "mitigate", "income", "gold"], "min_chars": 45}, ["economy", "constraints"]),
        _task("eco-mass-transfer-loading-10", "eco_transfer", s, "Write a loading hint line foreshadowing economy pressure from upkeep.", {"all_contains": ["upkeep"], "any_contains": ["loading", "hint", "pressure"], "min_chars": 45}, ["economy", "loading_screen"]),
        _task("eco-mass-transfer-hud-11", "eco_transfer", s, "Provide a HUD-friendly tooltip snippet tying supply pressure to net gold.", {"all_contains": ["supply", "gold"], "min_chars": 50}, ["economy", "hud"]),
        _task("eco-mass-transfer-ai-12", "eco_transfer", s, "Draft one AI-council rationale line that chooses economy stabilization over expansion.", {"any_contains": ["stabilization", "defense", "expansion", "economy"], "min_chars": 55}, ["economy", "planning"]),
        _task("eco-mass-terse-breakdown-13", "eco_constraints", s, "Give one comma-separated line with income, upkeep, workforce, and market trend.", {"all_contains": [","], "any_contains": ["income", "upkeep", "workforce", "market"], "min_chars": 45}, ["economy", "constraints"]),
        _task("eco-mass-warning-window-14", "eco_ui_change", s, "Write one warning-state economy tooltip and one steady-state tooltip with concise wording.", {"any_contains": ["warning", "steady", "income", "gold", "pressure"], "min_chars": 65}, ["economy", "ui"]),
        _task("eco-mass-turn-forward-15", "eco_core", s, "Explain one next-turn economy fix and one second-order consequence in 2-3 sentences.", {"any_contains": ["next turn", "income", "consequence", "trade-off", "upkeep"], "min_chars": 90}, ["economy"]),
        _task("eco-mass-supply-tax-16", "eco_core", s, "Describe how long supply lines can create an upkeep-like drag on net gold.", {"all_contains": ["gold"], "any_contains": ["supply", "upkeep", "drag", "income"], "min_chars": 75}, ["economy"]),
        _task("eco-mass-compact-three-17", "eco_constraints", s, "Exactly three short lines: income summary, pressure source, mitigation step.", {"any_contains": ["income", "pressure", "mitigate", "gold"], "min_chars": 55}, ["economy", "constraints"]),
        _task("eco-mass-ai-ops-18", "eco_transfer", s, "Write one AI planning note that delays expansion due to upkeep and market contraction.", {"any_contains": ["expansion", "upkeep", "market", "economy"], "min_chars": 60}, ["economy", "planning"]),
        _task("eco-mass-hud-chip-19", "eco_transfer", s, "Provide two HUD chip labels that expose economy trend and supply drag.", {"any_contains": ["economy", "income", "supply", "drag", "trend"], "min_chars": 45}, ["economy", "hud"]),
        _task("eco-mass-combat-link-20", "eco_transfer", s, "Write one tooltip line tying low income to reduced combat readiness.", {"any_contains": ["income", "combat", "readiness", "upkeep"], "min_chars": 55}, ["economy", "combat"]),
        _task("eco-mass-save-note-21", "eco_multidomain", s, "Draft one save/load note that verifies economy fields before applying turn simulation.", {"any_contains": ["save", "load", "economy", "fields", "verify"], "min_chars": 65}, ["economy", "save_load"]),
        _task("eco-mass-domain-hud-combat-22", "eco_multidomain", s, "In 2 sentences connect economy pressure, morale trend, and combat risk visibility.", {"any_contains": ["economy", "morale", "combat", "risk"], "min_chars": 80}, ["economy", "hud", "combat"]),
        _task("eco-mass-domain-loading-ai-23", "eco_multidomain", s, "Write two loading hints that reference economy pressure and AI intent to defend or expand.", {"any_contains": ["economy", "income", "intent", "defend", "expand"], "min_chars": 70}, ["economy", "loading_screen", "planning"]),
        _task("eco-mass-domain-compact-24", "eco_multidomain", s, "One comma-separated line with economy trend, supply pressure, combat readiness, and save integrity.", {"all_contains": [","], "any_contains": ["economy", "supply", "combat", "save", "integrity"], "min_chars": 65}, ["economy", "combat", "save_load"]),
        _task("eco-mass-domain-recovery-25", "eco_multidomain", s, "Provide one fallback plan when income drops and save validation blocks risky migration.", {"any_contains": ["income", "save", "validation", "migration", "fallback"], "min_chars": 75}, ["economy", "save_load"]),
        _task("eco-mass-domain-ai-combat-26", "eco_multidomain", s, "Write a war-council explanation linking gold shortage to delayed attack timing.", {"any_contains": ["gold", "attack", "delay", "council", "risk"], "min_chars": 70}, ["economy", "planning", "combat"]),
        _task("eco-mass-domain-hud-save-27", "eco_multidomain", s, "Create a system tooltip merging economy summary with save compatibility status.", {"any_contains": ["economy", "income", "save", "compatibility", "status"], "min_chars": 70}, ["economy", "hud", "save_load"]),
        _task("eco-mass-domain-ops-28", "eco_multidomain", s, "Plain text only: summarize economy trend, AI plan, and combat readiness in two lines.", {"any_contains": ["economy", "income", "plan", "combat", "readiness"], "none_contains": ["```"], "min_chars": 65}, ["economy", "planning", "combat"]),
        _task("eco-mass-domain-limits-29", "eco_multidomain", s, "Give one sentence on why over-expansion can hurt economy, morale, and save stability.", {"any_contains": ["expansion", "economy", "morale", "save", "stability"], "min_chars": 75}, ["economy", "planning", "save_load"]),
        _task("eco-mass-domain-aci-30", "eco_multidomain", s, "Propose four compact metrics for an economy-health panel spanning finance, risk, and persistence safety.", {"any_contains": ["metric", "economy", "risk", "save", "income", "upkeep"], "min_chars": 75}, ["economy", "combat", "save_load"]),
    ]
    return rows


def _combat_rows() -> List[Dict[str, Any]]:
    s = "combat_risk"
    rows = [
        _task("combat-mass-factors-01", "combat_core", s, "List five combat risk factors. Must include morale, terrain, and walls or fortification.", {"all_contains": ["morale", "terrain"], "any_contains": ["wall", "fortification", "fortified"], "min_chars": 40}, ["combat"]),
        _task("combat-mass-summary-02", "combat_core", s, "Write one combat summary with attacker strength, defender strength, and risk band.", {"all_contains": ["attacker", "defender"], "any_contains": ["low", "medium", "high"], "min_chars": 40}, ["combat"]),
        _task("combat-mass-modifiers-03", "combat_core", s, "Explain in 2-3 sentences how terrain and morale modifiers change engagement risk.", {"all_contains": ["terrain", "morale"], "any_contains": ["risk", "modifier"], "min_chars": 90}, ["combat"]),
        _task("combat-mass-ui-clarity-04", "combat_ui_change", s, "Suggest one UI improvement that makes combat risk preview more legible.", {"any_contains": ["preview", "risk", "legible", "hierarchy"], "min_chars": 85}, ["combat", "ui"]),
        _task("combat-mass-short-lines-05", "combat_ui_change", s, "Write one warning combat line and one favorable combat line.", {"any_contains": ["warning", "favorable", "risk", "odds"], "min_chars": 55}, ["combat", "ui"]),
        _task("combat-mass-bands-06", "combat_ui_change", s, "Define concise wording for low, medium, and high combat risk bands.", {"all_contains": ["low", "medium", "high"], "min_chars": 60}, ["combat", "ui"]),
        _task("combat-mass-constraints-07", "combat_constraints", s, "Plain text only: one sentence with attacker, defender, and terrain.", {"all_contains": ["attacker", "defender", "terrain"], "none_contains": ["```"], "min_chars": 45}, ["combat", "constraints"]),
        _task("combat-mass-constraints-08", "combat_constraints", s, "Output one comma-separated line with four combat preview factors.", {"all_contains": [","], "any_contains": ["morale", "terrain", "wall", "risk", "strength"], "min_chars": 35}, ["combat", "constraints"]),
        _task("combat-mass-constraints-09", "combat_constraints", s, "Exactly two lines: line one strength snapshot, line two risk recommendation.", {"any_contains": ["strength", "risk", "recommend"], "min_chars": 45}, ["combat", "constraints"]),
        _task("combat-mass-transfer-loading-10", "combat_transfer", s, "Write one loading hint line preparing the player for morale/terrain combat checks.", {"all_contains": ["morale", "terrain"], "min_chars": 45}, ["combat", "loading_screen"]),
        _task("combat-mass-transfer-hud-11", "combat_transfer", s, "Provide one HUD chip label related to combat risk and defender posture.", {"any_contains": ["risk", "defender", "posture", "fortified"], "min_chars": 35}, ["combat", "hud"]),
        _task("combat-mass-transfer-ai-12", "combat_transfer", s, "Give one AI planning line that delays attack due to high combat risk.", {"any_contains": ["attack", "delay", "high", "risk"], "min_chars": 45}, ["combat", "planning"]),
        _task("combat-mass-terrain-read-13", "combat_core", s, "Write a short combat brief that compares open terrain vs walled defense risk.", {"any_contains": ["terrain", "wall", "defense", "risk"], "min_chars": 65}, ["combat"]),
        _task("combat-mass-stance-impact-14", "combat_core", s, "In 2 sentences explain how defender stance and morale influence expected losses.", {"any_contains": ["defender", "stance", "morale", "loss"], "min_chars": 75}, ["combat"]),
        _task("combat-mass-compact-triad-15", "combat_constraints", s, "Return three short lines: attacker snapshot, defender snapshot, risk call.", {"all_contains": ["attacker", "defender"], "any_contains": ["risk"], "min_chars": 55}, ["combat", "constraints"]),
        _task("combat-mass-odds-window-16", "combat_ui_change", s, "Suggest a compact UI wording pattern for odds, morale delta, and terrain modifier.", {"any_contains": ["odds", "morale", "terrain", "modifier"], "min_chars": 70}, ["combat", "ui"]),
        _task("combat-mass-hud-badge-17", "combat_transfer", s, "Create two HUD badge labels for siege pressure and frontline risk.", {"any_contains": ["siege", "frontline", "risk", "pressure"], "min_chars": 45}, ["combat", "hud"]),
        _task("combat-mass-econ-link-18", "combat_transfer", s, "Write one note that ties low supply/economy to degraded combat readiness.", {"any_contains": ["supply", "economy", "income", "combat", "readiness"], "min_chars": 60}, ["combat", "economy"]),
        _task("combat-mass-domain-plan-19", "combat_multidomain", s, "Draft a council line linking combat risk, economy constraints, and AI intent.", {"any_contains": ["combat", "risk", "economy", "intent", "defend", "attack"], "min_chars": 75}, ["combat", "economy", "planning"]),
        _task("combat-mass-domain-save-20", "combat_multidomain", s, "Provide one API-safe summary line for combat preview persistence with version guard.", {"any_contains": ["combat", "preview", "version", "guard", "save"], "min_chars": 70}, ["combat", "save_load"]),
        _task("combat-mass-domain-loading-21", "combat_multidomain", s, "Write two loading lines that foreshadow combat risk and AI strategic posture.", {"any_contains": ["loading", "combat", "risk", "intent", "posture"], "min_chars": 70}, ["combat", "loading_screen", "planning"]),
        _task("combat-mass-domain-econ-22", "combat_multidomain", s, "Explain in 2 sentences how income pressure can change attack timing and risk tolerance.", {"any_contains": ["income", "attack", "risk", "tolerance"], "min_chars": 80}, ["combat", "economy", "planning"]),
        _task("combat-mass-domain-hud-save-23", "combat_multidomain", s, "Write one HUD/system line showing combat readiness plus save integrity check.", {"any_contains": ["combat", "readiness", "save", "integrity", "check"], "min_chars": 70}, ["combat", "hud", "save_load"]),
        _task("combat-mass-domain-compact-24", "combat_multidomain", s, "One comma-separated line with morale, terrain, economy drag, and fallback plan.", {"all_contains": [","], "any_contains": ["morale", "terrain", "economy", "plan", "risk"], "min_chars": 65}, ["combat", "economy", "planning"]),
        _task("combat-mass-domain-ops-25", "combat_multidomain", s, "Plain text only: two lines summarizing battle odds and strategic next move.", {"any_contains": ["battle", "odds", "risk", "next move", "defend", "attack"], "none_contains": ["```"], "min_chars": 65}, ["combat", "planning"]),
        _task("combat-mass-domain-support-26", "combat_multidomain", s, "Give one recommendation that balances wall repairs, economy, and morale stabilization.", {"any_contains": ["wall", "economy", "morale", "stabil"], "min_chars": 70}, ["combat", "economy"]),
        _task("combat-mass-domain-riskbar-27", "combat_multidomain", s, "Design compact risk-bar labels that include combat, supply, and save confidence.", {"any_contains": ["risk", "combat", "supply", "save", "confidence"], "min_chars": 65}, ["combat", "save_load", "hud"]),
        _task("combat-mass-domain-ai-check-28", "combat_multidomain", s, "Write one AI check rule that aborts attack when risk high and supply low.", {"any_contains": ["attack", "abort", "risk", "supply", "high"], "min_chars": 60}, ["combat", "planning", "economy"]),
        _task("combat-mass-domain-catalog-29", "combat_multidomain", s, "List four combat readiness factors spanning battlefield, logistics, and persistence safety.", {"any_contains": ["battle", "terrain", "supply", "save", "guard", "morale"], "min_chars": 65}, ["combat", "economy", "save_load"]),
        _task("combat-mass-domain-aci-30", "combat_multidomain", s, "Provide one sentence on why multi-domain combat evaluation beats single-metric risk scoring.", {"any_contains": ["multi-domain", "combat", "risk", "economy", "planning"], "min_chars": 75}, ["combat", "economy", "planning"]),
    ]
    return rows


def _save_rows() -> List[Dict[str, Any]]:
    s = "save_load_api_guard"
    rows = [
        _task("save-mass-checklist-01", "save_core", s, "Provide a 4-item TypeScript save/load guard checklist including schema validation and compatibility.", {"all_contains": ["schema", "compatibility"], "any_contains": ["validate", "version", "migration", "serialize"], "min_chars": 95}, ["save_load"]),
        _task("save-mass-response-shape-02", "save_core", s, "Write a minimal TypeScript type for save response with ok, version, and errors.", {"all_contains": ["ok", "version", "errors"], "any_contains": ["type", "interface"], "min_chars": 45}, ["save_load"]),
        _task("save-mass-migration-note-03", "save_core", s, "In 2 sentences, explain a safe migration path for older save versions.", {"any_contains": ["migration", "version", "backward", "compatibility"], "min_chars": 80}, ["save_load"]),
        _task("save-mass-ui-warning-04", "save_ui_change", s, "Draft one compact warning line for failed save validation and one recovery hint.", {"any_contains": ["validation", "failed", "retry", "recover"], "min_chars": 55}, ["save_load", "ui"]),
        _task("save-mass-error-contract-05", "save_ui_change", s, "Describe one improvement to error contract readability for save/load API responses.", {"any_contains": ["error", "contract", "response", "typed"], "min_chars": 80}, ["save_load", "ui"]),
        _task("save-mass-field-guard-06", "save_ui_change", s, "Explain how to guard unknown fields without breaking existing saves.", {"any_contains": ["unknown", "field", "guard", "compatibility"], "min_chars": 75}, ["save_load", "ui"]),
        _task("save-mass-constraints-07", "save_constraints", s, "Plain text only: one sentence containing schema, version, and errors.", {"all_contains": ["schema", "version", "errors"], "none_contains": ["```"], "min_chars": 45}, ["save_load", "constraints"]),
        _task("save-mass-constraints-08", "save_constraints", s, "Output one comma-separated line with four save/load safeguards.", {"all_contains": [","], "any_contains": ["schema", "version", "migration", "validation", "serialize"], "min_chars": 35}, ["save_load", "constraints"]),
        _task("save-mass-constraints-09", "save_constraints", s, "Exactly two lines: first indicates guard failure, second indicates fallback behavior.", {"any_contains": ["failure", "fallback", "guard", "load"], "min_chars": 45}, ["save_load", "constraints"]),
        _task("save-mass-transfer-loading-10", "save_transfer", s, "Write one loading reassurance line about save-state verification.", {"any_contains": ["save", "state", "verified", "integrity", "version"], "min_chars": 45}, ["save_load", "loading_screen"]),
        _task("save-mass-transfer-hud-11", "save_transfer", s, "Provide one HUD/system message line for save compatibility check success.", {"any_contains": ["save", "compatibility", "check", "success"], "min_chars": 40}, ["save_load", "hud"]),
        _task("save-mass-transfer-ai-12", "save_transfer", s, "Draft one AI system note when persisted plan data fails schema validation.", {"all_contains": ["schema"], "any_contains": ["validation", "plan", "fallback"], "min_chars": 50}, ["save_load", "planning"]),
        _task("save-mass-version-gate-13", "save_core", s, "Write a concise version-gating rule for loading legacy saves safely.", {"any_contains": ["version", "legacy", "safe", "load", "guard"], "min_chars": 65}, ["save_load"]),
        _task("save-mass-migration-map-14", "save_core", s, "In 2 sentences explain mapping old fields into new schema without data loss.", {"any_contains": ["schema", "migration", "fields", "data"], "min_chars": 75}, ["save_load"]),
        _task("save-mass-integrity-check-15", "save_constraints", s, "One comma-separated line with checksum, schema, version, fallback.", {"all_contains": [","], "any_contains": ["checksum", "schema", "version", "fallback"], "min_chars": 50}, ["save_load", "constraints"]),
        _task("save-mass-api-shape-16", "save_core", s, "Provide a minimal typed error payload that includes code, message, and recoverable flag.", {"any_contains": ["code", "message", "recoverable", "type", "interface"], "min_chars": 55}, ["save_load"]),
        _task("save-mass-ui-copy-17", "save_ui_change", s, "Write one short user-facing line for retry-safe save failure and one for success.", {"any_contains": ["retry", "safe", "save", "success", "failure"], "min_chars": 55}, ["save_load", "ui"]),
        _task("save-mass-econ-protect-18", "save_transfer", s, "Explain one guard ensuring economy fields are validated before commit.", {"any_contains": ["economy", "validate", "fields", "commit", "save"], "min_chars": 65}, ["save_load", "economy"]),
        _task("save-mass-combat-protect-19", "save_transfer", s, "Write one note ensuring combat preview state remains compatible across versions.", {"any_contains": ["combat", "preview", "compatible", "version", "state"], "min_chars": 65}, ["save_load", "combat"]),
        _task("save-mass-domain-plan-20", "save_multidomain", s, "Draft a policy line combining save safety, AI plan persistence, and fallback routing.", {"any_contains": ["save", "safety", "plan", "persist", "fallback"], "min_chars": 70}, ["save_load", "planning"]),
        _task("save-mass-domain-hud-21", "save_multidomain", s, "Write one HUD/system banner that reports save integrity and current match status confidence.", {"any_contains": ["save", "integrity", "status", "confidence", "hud"], "min_chars": 70}, ["save_load", "hud"]),
        _task("save-mass-domain-loading-22", "save_multidomain", s, "Create two loading-phase lines indicating state verification and migration progress.", {"any_contains": ["loading", "state", "verification", "migration", "progress"], "min_chars": 70}, ["save_load", "loading_screen"]),
        _task("save-mass-domain-econ-23", "save_multidomain", s, "In 2 sentences describe a rollback-safe plan when saved economy data fails validation.", {"any_contains": ["rollback", "economy", "validation", "save", "fallback"], "min_chars": 80}, ["save_load", "economy", "planning"]),
        _task("save-mass-domain-combat-24", "save_multidomain", s, "Provide one typed response strategy for stale combat state in multiplayer restore.", {"any_contains": ["typed", "response", "combat", "state", "restore", "stale"], "min_chars": 75}, ["save_load", "combat"]),
        _task("save-mass-domain-compact-25", "save_multidomain", s, "One comma-separated line with schema, version, combat snapshot, and economy checksum.", {"all_contains": [","], "any_contains": ["schema", "version", "combat", "economy", "checksum"], "min_chars": 70}, ["save_load", "combat", "economy"]),
        _task("save-mass-domain-ai-26", "save_multidomain", s, "Write one AI-note fallback when persisted intent payload fails schema gate.", {"any_contains": ["AI", "intent", "payload", "schema", "fallback"], "min_chars": 70}, ["save_load", "planning"]),
        _task("save-mass-domain-constraints-27", "save_multidomain", s, "Plain text only: summarize persistence safety across save, HUD visibility, and loading flow.", {"any_contains": ["save", "persistence", "HUD", "loading", "safety"], "none_contains": ["```"], "min_chars": 75}, ["save_load", "hud", "loading_screen"]),
        _task("save-mass-domain-resume-28", "save_multidomain", s, "Explain how to resume safely when one subsystem payload is ahead of schema version.", {"any_contains": ["resume", "schema", "version", "subsystem", "safe"], "min_chars": 75}, ["save_load", "planning"]),
        _task("save-mass-domain-guardrail-29", "save_multidomain", s, "Give one guardrail that prevents corrupted state from propagating into combat and economy.", {"any_contains": ["guardrail", "corrupted", "state", "combat", "economy"], "min_chars": 75}, ["save_load", "combat", "economy"]),
        _task("save-mass-domain-aci-30", "save_multidomain", s, "One sentence on why persistence checks should be weighted higher on multi-domain tasks.", {"any_contains": ["persistence", "weighted", "multi-domain", "task", "checks"], "min_chars": 75}, ["save_load", "combat", "economy"]),
    ]
    return rows


def _ai_rows() -> List[Dict[str, Any]]:
    s = "ai_planning_explanation"
    rows = [
        _task("ai-mass-rationale-01", "ai_core", s, "Write a short AI council rationale choosing defense over expansion this turn.", {"any_contains": ["defense", "expansion", "risk", "supply", "morale"], "min_chars": 90}, ["planning"]),
        _task("ai-mass-intents-02", "ai_core", s, "Provide three concise AI intent labels and one-line explanations.", {"any_contains": ["expansion", "scout", "reinforce", "attack", "defend"], "min_chars": 80}, ["planning"]),
        _task("ai-mass-followup-03", "ai_core", s, "In 2-3 sentences, explain one AI choice and one concrete follow-up action.", {"any_contains": ["choice", "follow-up", "reinforce", "fortify", "scout"], "min_chars": 90}, ["planning"]),
        _task("ai-mass-ui-clarity-04", "ai_ui_change", s, "Suggest one UI wording improvement to make AI rationale easier to inspect.", {"any_contains": ["rationale", "inspect", "clarity", "signal"], "min_chars": 80}, ["planning", "ui"]),
        _task("ai-mass-compact-line-05", "ai_ui_change", s, "Write one compact planning line for a war-room panel.", {"any_contains": ["plan", "risk", "reinforce", "defend", "expand"], "min_chars": 45}, ["planning", "ui"]),
        _task("ai-mass-branching-06", "ai_ui_change", s, "Describe one branching condition where AI swaps from expansion to defense.", {"all_contains": ["expansion", "defense"], "any_contains": ["risk", "morale", "supply"], "min_chars": 80}, ["planning", "ui"]),
        _task("ai-mass-constraints-07", "ai_constraints", s, "Plain text only: one sentence containing defend, expand, and reinforce.", {"all_contains": ["defend", "expand", "reinforce"], "none_contains": ["```"], "min_chars": 45}, ["planning", "constraints"]),
        _task("ai-mass-constraints-08", "ai_constraints", s, "Output one comma-separated line with four AI intent verbs.", {"all_contains": [","], "any_contains": ["defend", "expand", "scout", "reinforce", "attack"], "min_chars": 35}, ["planning", "constraints"]),
        _task("ai-mass-constraints-09", "ai_constraints", s, "Exactly two lines: line one intent, line two reason.", {"any_contains": ["intent", "reason", "risk", "supply", "morale"], "min_chars": 45}, ["planning", "constraints"]),
        _task("ai-mass-transfer-loading-10", "ai_transfer", s, "Write one loading line hinting at current AI council intent.", {"any_contains": ["council", "intent", "defend", "expand", "reinforce"], "min_chars": 45}, ["planning", "loading_screen"]),
        _task("ai-mass-transfer-combat-11", "ai_transfer", s, "Provide one combat-preview note that references AI risk assessment.", {"any_contains": ["combat", "risk", "assessment", "defend"], "min_chars": 45}, ["planning", "combat"]),
        _task("ai-mass-transfer-economy-12", "ai_transfer", s, "Give one AI rationale line that postpones expansion for economy stabilization.", {"all_contains": ["expansion"], "any_contains": ["economy", "stabilization", "income", "upkeep"], "min_chars": 50}, ["planning", "economy"]),
        _task("ai-mass-intent-lexical-13", "ai_constraints", s, "Write exactly three intent labels and keep the literal verbs defend, expand, and reinforce.", {"all_contains": ["defend", "expand", "reinforce"], "none_contains": ["```"], "min_chars": 40}, ["planning", "constraints"]),
        _task("ai-mass-intent-lines-14", "ai_constraints", s, "Output exactly three lines in the form intent - reason, and include scout and attack at least once.", {"any_contains": ["scout", "attack", "defend", "expand", "reinforce"], "none_contains": ["```"], "min_chars": 55}, ["planning", "constraints"]),
        _task("ai-mass-intent-comma-15", "ai_constraints", s, "Return one comma-separated line with four intent verbs using this token style: defend, expand, scout, reinforce, attack.", {"all_contains": [","], "any_contains": ["defend", "expand", "scout", "reinforce", "attack"], "none_contains": ["```"], "min_chars": 40}, ["planning", "constraints"]),
        _task("ai-mass-intent-followup-16", "ai_core", s, "Provide three concise AI intent labels with one short follow-up action each. Use defend or expansion language explicitly.", {"any_contains": ["defend", "expansion", "reinforce", "scout", "attack"], "min_chars": 80}, ["planning"]),
        _task("ai-mass-intent-branch-17", "ai_ui_change", s, "In two short lines, show an expansion intent line and then a defend intent line when risk spikes.", {"all_contains": ["expansion", "defend"], "any_contains": ["risk", "morale", "supply"], "min_chars": 55}, ["planning", "ui"]),
        _task("ai-mass-intent-council-18", "ai_transfer", s, "Write one council line that names current intent and one next intent using verbs from defend, expand, scout, reinforce, attack.", {"all_contains": ["intent"], "any_contains": ["defend", "expand", "scout", "reinforce", "attack"], "min_chars": 55}, ["planning", "loading_screen"]),
        _task("ai-mass-intent-war-room-19", "ai_core", s, "War-room snapshot: list three intent labels and keep each label under three words.", {"any_contains": ["defend", "expand", "scout", "reinforce", "attack"], "none_contains": ["```"], "min_chars": 45}, ["planning"]),
        _task("ai-mass-intent-plain-20", "ai_constraints", s, "Plain text only. Include literal tokens defend and expansion in a short intent explanation.", {"all_contains": ["defend", "expansion"], "none_contains": ["```"], "min_chars": 45}, ["planning", "constraints"]),
        _task("ai-mass-domain-hud-21", "ai_multidomain", s, "Write a compact HUD rationale linking AI intent to morale and supply status.", {"any_contains": ["intent", "morale", "supply", "defend", "expand"], "min_chars": 70}, ["planning", "hud"]),
        _task("ai-mass-domain-combat-22", "ai_multidomain", s, "Explain in 2 sentences why AI delays attack based on combat preview and terrain risk.", {"any_contains": ["attack", "combat", "terrain", "risk", "delay"], "min_chars": 80}, ["planning", "combat"]),
        _task("ai-mass-domain-econ-23", "ai_multidomain", s, "Provide one rationale that trades expansion speed for economy stabilization and save safety.", {"any_contains": ["expansion", "economy", "stabil", "save", "safety"], "min_chars": 75}, ["planning", "economy", "save_load"]),
        _task("ai-mass-domain-loading-24", "ai_multidomain", s, "Draft two loading lines that communicate current AI intent and fallback plan.", {"any_contains": ["loading", "intent", "fallback", "plan", "council"], "min_chars": 70}, ["planning", "loading_screen"]),
        _task("ai-mass-domain-save-25", "ai_multidomain", s, "Write one guard note for persisted AI plans when schema version mismatches.", {"any_contains": ["schema", "version", "plan", "persist", "mismatch"], "min_chars": 70}, ["planning", "save_load"]),
        _task("ai-mass-domain-compact-26", "ai_multidomain", s, "One comma-separated line with intent, economy pressure, combat risk, and save confidence.", {"all_contains": [","], "any_contains": ["intent", "economy", "combat", "risk", "save"], "min_chars": 70}, ["planning", "economy", "combat", "save_load"]),
        _task("ai-mass-domain-threat-27", "ai_multidomain", s, "Provide one sentence on how low supply shifts AI from expansion to reinforcement.", {"any_contains": ["supply", "expansion", "reinforcement", "intent"], "min_chars": 70}, ["planning", "economy"]),
        _task("ai-mass-domain-hud-combat-28", "ai_multidomain", s, "Write one war-room line connecting HUD threat chips to combat decision thresholds.", {"any_contains": ["HUD", "threat", "combat", "decision", "threshold"], "min_chars": 70}, ["planning", "hud", "combat"]),
        _task("ai-mass-domain-ops-29", "ai_multidomain", s, "Plain text only: summarize AI intent, risk posture, and persistence fallback in two lines.", {"any_contains": ["intent", "risk", "fallback", "persist"], "none_contains": ["```"], "min_chars": 70}, ["planning", "save_load", "combat"]),
        _task("ai-mass-domain-aci-30", "ai_multidomain", s, "One sentence on why AI rationale quality should be scored across multiple gameplay domains.", {"any_contains": ["AI", "rationale", "multiple", "domains", "score"], "min_chars": 75}, ["planning", "economy", "combat"]),
    ]
    return rows


def _write_combined_specialist_suite() -> None:
    combined: List[Dict[str, Any]] = []
    for name in MASS_FILES.values():
        path = BENCH / name
        if not path.is_file():
            continue
        combined.extend(json.loads(path.read_text(encoding="utf-8")))
    _write("specialist_benchmark_tasks.json", combined)


def main() -> None:
    _write(MASS_FILES["hud_status"], _hud_rows())
    _write(MASS_FILES["economy_tooltip"], _economy_rows())
    _write(MASS_FILES["combat_risk"], _combat_rows())
    _write(MASS_FILES["save_load_api_guard"], _save_rows())
    _write(MASS_FILES["ai_planning_explanation"], _ai_rows())
    # Ensure loading exists; don't overwrite if maintained manually.
    if not (BENCH / MASS_FILES["loading_screen"]).is_file():
        _write(MASS_FILES["loading_screen"], _loading_rows())
    _write_combined_specialist_suite()


if __name__ == "__main__":
    main()
