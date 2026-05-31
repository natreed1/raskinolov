#!/usr/bin/env python3
"""Manage and score economistRL tasks for economy-focused RL experiments.

This framework keeps the RL task bank appendable while making rewards
deterministic enough for baseline comparisons. It supports:

- `validate`: check task-bank shape and duplicate ids.
- `add-task`: append a new task with a starter reward rubric.
- `score`: score JSONL model outputs against execution-oriented rollout evidence.

Output JSONL rows for `score` should include `task_id`, compile/test/simulation
evidence when available, plus one of `output`, `assistant`, `response`,
`generated_text`, or `text`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO / "benchmarks" / "economistRL_tasks_v1.json"
DEFAULT_OUT_JSON = REPO / "benchmarks" / "results" / "economistRL_scorecard_v1.json"
DEFAULT_OUT_MD = REPO / "benchmarks" / "results" / "economistRL_scorecard_v1.md"

DEFAULT_BASE_REWARD_WEIGHTS = {
    "simulation_behavior": 0.45,
    "targeted_tests": 0.20,
    "static_code_mechanics": 0.15,
    "formula_signal": 0.10,
    "instruction_contract": 0.05,
    "concision": 0.03,
    "anti_overfit": 0.02,
}

VALID_CURRICULUM_TRACKS = {"economy", "generalist"}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {
            "schema_version": "economist_rl_task_bank_v1",
            "adapter_id": "economistRL",
            "tasks": payload,
        }
    if not isinstance(payload, dict):
        raise SystemExit(f"Task bank must be object or list: {path}")
    return payload


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _task_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    task = row.get("task")
    if isinstance(task, dict):
        return str(task.get("id") or "").strip()
    return ""


def _output_text(row: dict[str, Any]) -> str:
    for key in ("output", "assistant", "response", "generated_text", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "").strip()
    return ""


def _contains(text_l: str, token: str) -> bool:
    token_l = str(token or "").strip().lower()
    if not token_l:
        return False
    if re.search(r"\s", token_l):
        return token_l in text_l
    return re.search(rf"(?<![a-z0-9_]){re.escape(token_l)}(?![a-z0-9_])", text_l) is not None


def _rubric_score(text: str, expect: dict[str, Any]) -> tuple[float, list[str]]:
    text_l = text.lower()
    checks = 0
    passed = 0
    failures: list[str] = []

    for token in expect.get("all_contains") or []:
        checks += 1
        if _contains(text_l, str(token)):
            passed += 1
        else:
            failures.append(f"missing_required:{token}")

    any_tokens = [str(t) for t in expect.get("any_contains") or [] if str(t).strip()]
    if any_tokens:
        checks += 1
        if any(_contains(text_l, token) for token in any_tokens):
            passed += 1
        else:
            failures.append("missing_any:" + "|".join(any_tokens[:8]))

    for token in expect.get("none_contains") or []:
        checks += 1
        if _contains(text_l, str(token)):
            failures.append(f"forbidden:{token}")
        else:
            passed += 1

    min_chars = int(expect.get("min_chars") or 0)
    if min_chars:
        checks += 1
        if len(text) >= min_chars:
            passed += 1
        else:
            failures.append(f"too_short:{len(text)}<{min_chars}")

    max_chars = int(expect.get("max_chars") or 0)
    if max_chars:
        checks += 1
        if len(text) <= max_chars:
            passed += 1
        else:
            failures.append(f"too_long:{len(text)}>{max_chars}")

    return (_clamp01(passed / max(1, checks)), failures)


def _mechanics_score(text: str, expect: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    mechanics = expect.get("mechanics") or []
    if not isinstance(mechanics, list) or not mechanics:
        return 1.0, [], []

    text_l = text.lower()
    covered: list[str] = []
    missing: list[str] = []
    for item in mechanics:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "mechanic").strip()
        keywords = [str(k) for k in item.get("keywords") or [] if str(k).strip()]
        if keywords and any(_contains(text_l, kw) for kw in keywords):
            covered.append(name)
        else:
            missing.append(name)
    return (_clamp01(len(covered) / max(1, len(covered) + len(missing))), covered, missing)


def _instruction_score(text: str, prompt: str, expect: dict[str, Any]) -> tuple[float, list[str]]:
    prompt_l = prompt.lower()
    failures: list[str] = []
    score = 1.0
    lines = [line for line in text.splitlines() if line.strip()]

    if any(token in prompt_l for token in ("exactly", "one sentence", "one-line", "one line")):
        if len(lines) > 4:
            score -= 0.25
            failures.append("too_many_lines_for_constrained_prompt")
    if "plain text" in prompt_l and "```" in text:
        score -= 0.35
        failures.append("code_fence_in_plain_text")
    if "formula" in prompt_l and not any(ch in text for ch in ("=", "/", "*", "+", "-", "ratio", "curve")):
        score -= 0.20
        failures.append("formula_shape_not_signaled")
    if int(expect.get("max_chars") or 0) and len(text) > int(expect["max_chars"]):
        score -= 0.20
    return _clamp01(score), failures


def _concision_score(text: str, expect: dict[str, Any]) -> float:
    chars = len(text.strip())
    if chars == 0:
        return 0.0
    max_chars = int(expect.get("max_chars") or 1200)
    if chars <= max_chars:
        return 1.0
    return _clamp01(max_chars / chars)


def _anti_overfit_score(text: str) -> tuple[float, list[str]]:
    failures: list[str] = []
    score = 1.0
    lower = text.lower()
    if lower.count("economy") > 12:
        score -= 0.25
        failures.append("repeated_economy_keyword")
    if any(marker in lower for marker in ("as an ai", "i cannot", "not enough information")):
        score -= 0.35
        failures.append("generic_model_refusal_or_meta")
    if len(set(re.findall(r"[a-zA-Z_]{4,}", lower))) < 12 and len(text) > 220:
        score -= 0.20
        failures.append("low_vocabulary_repetition")
    return _clamp01(score), failures


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        raw = float(value)
        return _clamp01(raw / 100.0 if raw > 1.0 else raw)
    if isinstance(value, str):
        parsed_bool = _boolish(value)
        if parsed_bool is not None:
            return 1.0 if parsed_bool else 0.0
        try:
            raw = float(value)
            return _clamp01(raw / 100.0 if raw > 1.0 else raw)
        except ValueError:
            return None
    return None


def _weighted_goal_score(goals: Any) -> tuple[float, list[str], list[str]]:
    if not isinstance(goals, list) or not goals:
        return 0.0, [], ["missing_goals"]
    total_weight = 0.0
    earned = 0.0
    passed: list[str] = []
    failed: list[str] = []
    for idx, goal in enumerate(goals):
        if not isinstance(goal, dict):
            continue
        name = str(goal.get("name") or goal.get("metric") or f"goal_{idx + 1}")
        weight = float(goal.get("weight") or 1.0)
        score = _numeric_score(goal.get("score"))
        if score is None:
            score = _numeric_score(goal.get("passed"))
        if score is None:
            score = _numeric_score(goal.get("achieved"))
        if score is None:
            score = 0.0
        total_weight += max(0.0, weight)
        earned += max(0.0, weight) * score
        if score >= 0.999:
            passed.append(name)
        else:
            failed.append(name)
    if total_weight <= 0:
        return 0.0, passed, failed or ["missing_goal_weights"]
    return _clamp01(earned / total_weight), passed, failed


def _score_from_result_object(result: Any, *, default_missing: str) -> tuple[float, list[str], list[str]]:
    score = _numeric_score(result)
    if score is not None:
        return score, [], [] if score >= 1.0 else [default_missing]
    if not isinstance(result, dict):
        return 0.0, [], [default_missing]
    for key in ("score", "reward", "pass_rate", "passed_rate"):
        score = _numeric_score(result.get(key))
        if score is not None:
            return score, [], [] if score >= 1.0 else [default_missing]
    if isinstance(result.get("goals"), list):
        return _weighted_goal_score(result["goals"])
    if isinstance(result.get("checks"), list):
        return _weighted_goal_score(result["checks"])
    passed = result.get("passed")
    score = _numeric_score(passed)
    if score is not None:
        return score, [], [] if score >= 1.0 else [default_missing]
    return 0.0, [], [default_missing]


def _simulation_behavior_score(row: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    result = row.get("simulation_behavior")
    if result is None:
        result = row.get("simulation_results")
    if result is None:
        result = row.get("twenty_tick_simulation")
    return _score_from_result_object(result, default_missing="simulation_not_run")


def _targeted_tests_score(row: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    result = row.get("targeted_tests")
    if result is None:
        result = row.get("test_results")
    if result is None:
        verify = str(row.get("verify_status") or "").strip().lower()
        if verify == "passed":
            return 1.0, ["verify_status_passed"], []
        if verify in {"failed", "infra_precheck_failed"}:
            return 0.0, [], ["verify_status_failed"]
        return 0.0, [], ["targeted_tests_not_run"]
    return _score_from_result_object(result, default_missing="targeted_tests_failed")


def _text_for_static_checks(row: dict[str, Any], text: str) -> str:
    parts = [text]
    for key in ("diff", "patch", "code", "changed_code", "model_output"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _static_code_mechanics_score(task: dict[str, Any], row: dict[str, Any], text: str) -> tuple[float, list[str], list[str]]:
    provided = row.get("static_code_mechanics")
    if provided is not None:
        return _score_from_result_object(provided, default_missing="static_code_mechanics_failed")
    spec = task.get("static_code_mechanics") if isinstance(task.get("static_code_mechanics"), dict) else {}
    signals = [str(s) for s in spec.get("flexible_signals") or [] if str(s).strip()]
    hooks = [str(s) for s in spec.get("required_hooks") or [] if str(s).strip()]
    if not signals and not hooks:
        return 0.0, [], ["static_code_mechanics_missing_spec"]
    haystack = _text_for_static_checks(row, text).lower()
    covered = [signal for signal in signals if _contains(haystack, signal)]
    hook_credit = 0
    for hook in hooks:
        hook_tokens = [tok for tok in re.findall(r"[a-zA-Z_]{4,}", hook.lower()) if tok not in {"reads", "uses", "with", "from", "into"}]
        if hook_tokens and any(_contains(haystack, tok) for tok in hook_tokens):
            hook_credit += 1
    signal_score = len(covered) / max(1, min(len(signals), 8))
    hook_score = hook_credit / max(1, len(hooks))
    score = _clamp01((0.55 * signal_score) + (0.45 * hook_score))
    missing = []
    if score < 1.0:
        missing = ["static_code_mechanics_partial"]
    return score, covered[:12], missing


def _formula_signal_score(task: dict[str, Any], row: dict[str, Any], text: str) -> tuple[float, list[str], list[str]]:
    provided = row.get("formula_signal")
    if provided is not None:
        return _score_from_result_object(provided, default_missing="formula_signal_failed")
    spec = task.get("formula_signal") if isinstance(task.get("formula_signal"), dict) else {}
    patterns = [str(p) for p in spec.get("patterns") or [] if str(p).strip()]
    haystack = _text_for_static_checks(row, text).lower()
    generic_markers = ("clamp", "min(", "max(", "ratio", "/", "*", "threshold", "curve", "multiplier", "bounded", "floor", "cap")
    covered = [p for p in patterns if any(_contains(haystack, tok) for tok in re.findall(r"[a-zA-Z_]{4,}", p.lower()))]
    generic_hits = [m for m in generic_markers if m in haystack]
    if patterns:
        score = _clamp01((len(covered) / len(patterns)) * 0.8 + min(0.2, len(generic_hits) * 0.04))
    else:
        score = _clamp01(min(1.0, len(generic_hits) / 4.0))
    return score, covered + generic_hits[:6], [] if score >= 0.8 else ["formula_signal_partial"]


def _changed_files(row: dict[str, Any]) -> list[str]:
    raw = row.get("changed_files")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    raw = row.get("files_changed")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _instruction_contract_score(task: dict[str, Any], row: dict[str, Any], text: str) -> tuple[float, list[str], list[str], float | None]:
    spec = task.get("instruction_contract") if isinstance(task.get("instruction_contract"), dict) else {}
    failures: list[str] = []
    passed: list[str] = []
    cap: float | None = None
    score = 1.0
    changed = _changed_files(row)
    if _boolish(row.get("unnecessary_changes")) is True or _boolish(row.get("broad_rewrite")) is True:
        failures.append("unnecessary_or_broad_changes")
        score -= 0.6
        cap = min(cap if cap is not None else 1.0, float(spec.get("unnecessary_change_cap") or 0.25))
    if _boolish(row.get("tests_weakened")) is True or _boolish(row.get("deleted_tests")) is True:
        failures.append("tests_deleted_or_weakened")
        score -= 0.85
        cap = min(cap if cap is not None else 1.0, float(spec.get("test_weakening_cap") or 0.10))
    unrelated = row.get("unrelated_files_changed")
    if isinstance(unrelated, list) and unrelated:
        failures.append("unrelated_files_changed")
        score -= min(0.7, 0.15 * len(unrelated))
        cap = min(cap if cap is not None else 1.0, float(spec.get("unnecessary_change_cap") or 0.25))
    if changed:
        passed.append("changed_files_declared")
    if "```" in text and len(text) > 0:
        passed.append("structured_output_present")
    return _clamp01(score), passed, failures, cap


def _anti_overfit_guard_score(row: dict[str, Any], text: str) -> tuple[float, list[str], list[str], float | None]:
    score, failures = _anti_overfit_score(text)
    cap: float | None = None
    passed: list[str] = []
    flags: list[str] = []
    raw_flags = row.get("reward_gaming_flags")
    if isinstance(raw_flags, list):
        flags.extend(str(flag) for flag in raw_flags if str(flag).strip())
    for key, flag in (
        ("hardcoded_scenarios", "hardcoded_visible_scenarios"),
        ("deleted_tests", "deleted_or_weakened_tests"),
        ("tests_weakened", "deleted_or_weakened_tests"),
        ("bypassed_mechanics", "mechanic_bypass"),
        ("unrelated_files_changed", "unrelated_file_changes"),
        ("non_general_behavior", "non_general_behavior"),
    ):
        value = row.get(key)
        if _boolish(value) is True or (isinstance(value, list) and value):
            flags.append(flag)
    previous = _numeric_score(row.get("previous_potential"))
    new = _numeric_score(row.get("new_potential"))
    if previous is not None and new is not None:
        if new <= previous + 1e-9:
            flags.append("potential_not_improved")
        else:
            passed.append("potential_improved")
    if flags:
        unique = sorted(set(flags))
        failures.extend(f"reward_gaming:{flag}" for flag in unique)
        score = min(score, 0.25)
        cap = 0.30
        if any(flag in unique for flag in ("deleted_or_weakened_tests", "mechanic_bypass", "hardcoded_visible_scenarios")):
            cap = 0.15
    elif previous is not None and new is not None:
        passed.append("potential_comparison_clean")
    return _clamp01(score), passed, failures, cap


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "passed", "pass", "compiled", "success", "ok"}:
        return True
    if token in {"0", "false", "no", "n", "failed", "fail", "error", "errored", "timeout"}:
        return False
    return None


def _compile_status(row: dict[str, Any]) -> bool | None:
    """Infer compile viability from common rollout/eval result fields."""
    for key in ("compiled", "compile_passed", "typecheck_passed", "tsc_passed"):
        parsed = _boolish(row.get(key))
        if parsed is not None:
            return parsed
    for key in ("compile_status", "typecheck_status", "tsc_status"):
        parsed = _boolish(row.get(key))
        if parsed is not None:
            return parsed
    verify_status = str(row.get("verify_status") or "").strip().lower()
    if verify_status == "passed":
        return True
    if verify_status in {"failed", "infra_precheck_failed"}:
        return False
    return None


def compile_gate_reward(
    *,
    compiled: bool | None,
    rolling_compile_rate: float,
    base_reward: float,
) -> dict[str, Any]:
    """Apply a non-decaying compile-failure gate with decaying compile-success bonus.

    Compile success becomes less rewarding as recent compile rate improves, but
    compile failure always caps or zeroes the final reward. This keeps the RL
    system from regressing after it learns to compile.
    """
    base = _clamp01(base_reward)
    rate = _clamp01(rolling_compile_rate)
    if compiled is None:
        return {
            "reward": base,
            "applied": False,
            "compiled": None,
            "rolling_compile_rate": round(rate, 4),
            "compile_bonus_weight": 0.0,
            "failure_cap": None,
            "reason": "compile status unavailable; base reward used",
        }

    if not compiled:
        if rate < 0.50:
            cap = 0.10
        elif rate < 0.80:
            cap = 0.05
        else:
            cap = 0.0
        return {
            "reward": min(base, cap),
            "applied": True,
            "compiled": False,
            "rolling_compile_rate": round(rate, 4),
            "compile_bonus_weight": 0.0,
            "failure_cap": cap,
            "reason": "compile failed; catastrophic cap applied",
        }

    if rate < 0.50:
        bonus_weight = 0.35
    elif rate < 0.75:
        bonus_weight = 0.20
    elif rate < 0.90:
        bonus_weight = 0.10
    else:
        bonus_weight = 0.03
    reward = bonus_weight + ((1.0 - bonus_weight) * base)
    return {
        "reward": _clamp01(reward),
        "applied": True,
        "compiled": True,
        "rolling_compile_rate": round(rate, 4),
        "compile_bonus_weight": bonus_weight,
        "failure_cap": None,
        "reason": "compile passed; decaying compile-success bonus applied",
    }


def _weights(task: dict[str, Any], payload: dict[str, Any]) -> dict[str, float]:
    raw = task.get("reward_weights")
    if not isinstance(raw, dict):
        raw = (payload.get("scoring_policy") or {}).get("default_reward_weights")
    weights = dict(DEFAULT_BASE_REWARD_WEIGHTS)
    if isinstance(raw, dict):
        for key in weights:
            if key in raw:
                weights[key] = float(raw[key])
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return DEFAULT_BASE_REWARD_WEIGHTS
    return {key: max(0.0, value) / total for key, value in weights.items()}


def _curriculum_track(task: dict[str, Any]) -> str:
    raw = str(task.get("curriculum_track") or "").strip().lower()
    if raw in VALID_CURRICULUM_TRACKS:
        return raw
    # Existing seed tasks predate the field and are economy-focused by default.
    return "economy"


def _curriculum_summary(payload: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    policy = payload.get("curriculum_policy") if isinstance(payload.get("curriculum_policy"), dict) else {}
    target_counts = policy.get("target_counts") if isinstance(policy.get("target_counts"), dict) else {}
    target_total = int(policy.get("target_total_prompts") or sum(int(v) for v in target_counts.values() if isinstance(v, int)) or 0)
    track_counts = Counter(_curriculum_track(task) for task in tasks)
    out: dict[str, Any] = {
        "target_total_prompts": target_total,
        "track_counts": dict(sorted(track_counts.items())),
        "track_rates": {
            track: round(count / max(1, len(tasks)), 4)
            for track, count in sorted(track_counts.items())
        },
        "target_counts": dict(target_counts),
        "remaining_to_target": {},
    }
    for track, target in target_counts.items():
        try:
            target_i = int(target)
        except (TypeError, ValueError):
            continue
        out["remaining_to_target"][track] = max(0, target_i - int(track_counts.get(str(track), 0)))
    return out


def score_output(
    task: dict[str, Any],
    text: str,
    payload: dict[str, Any],
    *,
    rollout_row: dict[str, Any] | None = None,
    compiled: bool | None = None,
    rolling_compile_rate: float = 0.0,
) -> dict[str, Any]:
    expect = task.get("expect") if isinstance(task.get("expect"), dict) else {}
    prompt = str(task.get("prompt") or "")
    row = rollout_row or {}
    prompt_rubric, rubric_failures = _rubric_score(text, expect)
    legacy_mechanics, legacy_covered_mechanics, legacy_missing_mechanics = _mechanics_score(text, expect)
    legacy_instruction, instruction_failures = _instruction_score(text, prompt, expect)
    concision = _concision_score(text, expect)
    simulation_behavior, simulation_passed, simulation_failures = _simulation_behavior_score(row)
    targeted_tests, targeted_tests_passed, targeted_tests_failures = _targeted_tests_score(row)
    static_code_mechanics, static_passed, static_failures = _static_code_mechanics_score(task, row, text)
    formula_signal, formula_passed, formula_failures = _formula_signal_score(task, row, text)
    instruction_contract, instruction_passed, instruction_contract_failures, instruction_cap = _instruction_contract_score(
        task, row, text
    )
    anti_overfit, anti_passed, anti_failures, anti_cap = _anti_overfit_guard_score(row, text)
    weights = _weights(task, payload)
    base_reward = (
        weights["simulation_behavior"] * simulation_behavior
        + weights["targeted_tests"] * targeted_tests
        + weights["static_code_mechanics"] * static_code_mechanics
        + weights["formula_signal"] * formula_signal
        + weights["instruction_contract"] * instruction_contract
        + weights["concision"] * concision
        + weights["anti_overfit"] * anti_overfit
    )
    policy_caps = [cap for cap in (instruction_cap, anti_cap) if cap is not None]
    gated_base_reward = min([base_reward, *policy_caps]) if policy_caps else base_reward
    compile_gate = compile_gate_reward(
        compiled=compiled,
        rolling_compile_rate=rolling_compile_rate,
        base_reward=gated_base_reward,
    )
    reward = min([float(compile_gate["reward"]), *policy_caps]) if policy_caps else float(compile_gate["reward"])
    failures = (
        simulation_failures
        + targeted_tests_failures
        + static_failures
        + formula_failures
        + instruction_contract_failures
        + anti_failures
        + [f"prompt_rubric:{f}" for f in rubric_failures]
        + [f"legacy_missing_mechanic:{m}" for m in legacy_missing_mechanics]
        + [f"legacy_instruction:{f}" for f in instruction_failures]
    )
    if compiled is False:
        failures.append("compile_failed")
    simulation_evaluated = "simulation_not_run" not in simulation_failures
    tests_evaluated = "targeted_tests_not_run" not in targeted_tests_failures
    return {
        "task_id": str(task.get("id") or ""),
        "title": str(task.get("title") or ""),
        "difficulty": str(task.get("difficulty") or "standard"),
        "curriculum_track": _curriculum_track(task),
        "subskill": str(task.get("subskill") or ""),
        "rl_focus": list(task.get("rl_focus") or []),
        "score": round(100.0 * reward, 2),
        "reward": round(reward, 4),
        "base_reward": round(base_reward, 4),
        "gated_base_reward": round(gated_base_reward, 4),
        "passed": (
            reward >= 0.82
            and compiled is not False
            and simulation_behavior >= 0.70
            and (not tests_evaluated or targeted_tests >= 0.50)
            and not policy_caps
        ),
        "components": {
            "simulation_behavior": round(100.0 * simulation_behavior, 2),
            "targeted_tests": round(100.0 * targeted_tests, 2),
            "static_code_mechanics": round(100.0 * static_code_mechanics, 2),
            "formula_signal": round(100.0 * formula_signal, 2),
            "instruction_contract": round(100.0 * instruction_contract, 2),
            "concision": round(100.0 * concision, 2),
            "anti_overfit": round(100.0 * anti_overfit, 2),
            "prompt_rubric_guardrail": round(100.0 * prompt_rubric, 2),
            "legacy_mechanics_guardrail": round(100.0 * legacy_mechanics, 2),
            "legacy_instruction_guardrail": round(100.0 * legacy_instruction, 2),
        },
        "policy_caps": policy_caps,
        "compile_gate": compile_gate,
        "simulation_passed_goals": simulation_passed,
        "targeted_tests_passed": targeted_tests_passed,
        "static_code_mechanics_passed": static_passed,
        "formula_signal_passed": formula_passed,
        "instruction_contract_passed": instruction_passed,
        "anti_overfit_passed": anti_passed,
        "covered_mechanics": legacy_covered_mechanics,
        "missing_mechanics": legacy_missing_mechanics,
        "simulation_evaluated": simulation_evaluated,
        "targeted_tests_evaluated": tests_evaluated,
        "failures": failures,
        "output_chars": len(text),
        "output_preview": text[:500],
    }


def validate_tasks(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"Task bank missing `tasks` list: {path}")
    ids: list[str] = []
    issues: list[str] = []
    task_objs: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            issues.append(f"task[{idx}] is not an object")
            continue
        task_objs.append(task)
        tid = str(task.get("id") or "").strip()
        if not tid:
            issues.append(f"task[{idx}] missing id")
        ids.append(tid)
        if not str(task.get("prompt") or "").strip():
            issues.append(f"{tid or idx}: missing prompt")
        expect = task.get("expect")
        if not isinstance(expect, dict):
            issues.append(f"{tid or idx}: missing expect object")
        elif not expect.get("mechanics"):
            issues.append(f"{tid or idx}: missing expect.mechanics")
        for required_key in (
            "simulation_spec",
            "targeted_tests",
            "static_code_mechanics",
            "formula_signal",
            "instruction_contract",
            "anti_overfit_guards",
        ):
            if not isinstance(task.get(required_key), dict):
                issues.append(f"{tid or idx}: missing {required_key} object")
        sim = task.get("simulation_spec") if isinstance(task.get("simulation_spec"), dict) else {}
        if int(sim.get("tick_count") or 0) != 20:
            issues.append(f"{tid or idx}: simulation_spec.tick_count must be 20")
        track = _curriculum_track(task)
        if track not in VALID_CURRICULUM_TRACKS:
            issues.append(f"{tid or idx}: invalid curriculum_track {track}")
    dupes = sorted([tid for tid, count in Counter(ids).items() if tid and count > 1])
    for tid in dupes:
        issues.append(f"duplicate id: {tid}")
    summary = {
        "task_bank": str(path),
        "adapter_id": str(payload.get("adapter_id") or ""),
        "task_count": len(tasks),
        "curriculum": _curriculum_summary(payload, task_objs),
        "duplicate_ids": dupes,
        "issues": issues,
        "valid": not issues,
    }
    if issues:
        raise SystemExit(json.dumps(summary, indent=2))
    return summary


def add_task(args: argparse.Namespace) -> None:
    path = args.tasks.expanduser().resolve()
    payload = _load_json(path) if path.exists() else {
        "schema_version": "economist_rl_task_bank_v1",
        "adapter_id": "economistRL",
        "created_utc": _utc_iso(),
        "tasks": [],
    }
    tasks = payload.setdefault("tasks", [])
    if not isinstance(tasks, list):
        raise SystemExit("Task bank `tasks` must be a list.")
    if any(str(task.get("id") or "") == args.id for task in tasks if isinstance(task, dict)):
        raise SystemExit(f"Task id already exists: {args.id}")

    mechanics = []
    for raw in args.mechanic:
        name, _, keywords = raw.partition(":")
        mechanics.append(
            {
                "name": name.strip(),
                "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()],
            }
        )
    task = {
        "id": args.id,
        "title": args.title,
        "difficulty": args.difficulty,
        "curriculum_track": args.curriculum_track,
        "rl_focus": args.focus_tag,
        "subskill": args.subskill,
        "prompt": args.prompt,
        "expect": {
            "all_contains": args.required,
            "any_contains": args.optional,
            "none_contains": args.forbidden,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "mechanics": mechanics,
        },
        "reference_answer": args.reference_answer,
        "simulation_spec": {
            "tick_count": 20,
            "purpose": args.simulation_goal,
            "relevant_state": args.relevant_state,
            "scenario": {
                "initial_state": {},
                "change_under_test": args.simulation_goal,
            },
            "goals": [
                {
                    "name": "behavior_changes_in_expected_direction",
                    "metric": "task_specific_metric",
                    "benchmark": args.simulation_goal,
                    "target": "rollout supplies scored 20-tick evidence",
                    "weight": 1.0,
                }
            ],
        },
        "targeted_tests": {
            "goal": args.test_goal,
            "preferred_commands": args.test_command,
            "outcome_checks": args.test_check,
        },
        "static_code_mechanics": {
            "required_hooks": args.static_hook,
            "flexible_signals": args.static_signal,
        },
        "formula_signal": {
            "patterns": args.formula_pattern,
        },
        "instruction_contract": {
            "gate": "large_policy_penalty",
            "allowed_change_policy": "Touch only files required by the task or directly related tests.",
            "unnecessary_change_cap": 0.25,
            "test_weakening_cap": 0.10,
            "forbidden_actions": [
                "delete_or_weaken_tests",
                "bypass_mechanics",
                "hardcode_visible_scenario",
                "change_unrelated_files_for_reward",
                "broad_rewrite_without_need",
            ],
        },
        "anti_overfit_guards": {
            "cap_policy": "Cap reward when rollout appears to game checks instead of improving behavior.",
            "flags": [
                "hardcoded_visible_scenarios",
                "deleted_or_weakened_tests",
                "mechanic_bypass",
                "unrelated_file_changes",
                "keyword_stuffing",
                "non_general_behavior",
                "potential_not_improved",
            ],
            "potential_comparison": "Progress credit requires new_potential > previous_potential when supplied.",
        },
    }
    tasks.append(task)
    payload["updated_utc"] = _utc_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"added": args.id, "tasks": str(path)}, indent=2))


def score_outputs(args: argparse.Namespace) -> None:
    tasks_path = args.tasks.expanduser().resolve()
    outputs_path = args.outputs_jsonl.expanduser().resolve()
    payload = _load_json(tasks_path)
    tasks = {str(task.get("id") or ""): task for task in payload.get("tasks") or [] if isinstance(task, dict)}
    rows = _iter_jsonl(outputs_path)
    scored: list[dict[str, Any]] = []
    missing_task_ids: list[str] = []
    for row in rows:
        tid = _task_id(row)
        task = tasks.get(tid)
        if not task:
            missing_task_ids.append(tid or "<missing>")
            continue
        scored.append(
            score_output(
                task,
                _output_text(row),
                payload,
                rollout_row=row,
                compiled=_compile_status(row),
                rolling_compile_rate=float(args.rolling_compile_rate),
            )
        )

    by_focus: dict[str, list[float]] = defaultdict(list)
    by_subskill: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        for focus in row["rl_focus"]:
            by_focus[str(focus)].append(float(row["score"]))
        by_subskill[row["subskill"]].append(float(row["score"]))

    summary = {
        "schema_version": "economist_rl_scorecard_v1",
        "created_utc": _utc_iso(),
        "adapter_id": str(payload.get("adapter_id") or "economistRL"),
        "tasks_file": str(tasks_path),
        "outputs_jsonl": str(outputs_path),
        "outputs_scored": len(scored),
        "missing_task_ids": missing_task_ids,
        "rolling_compile_rate": round(float(args.rolling_compile_rate), 4),
        "compile_gate_applied": sum(1 for row in scored if row.get("compile_gate", {}).get("applied")),
        "compiled": sum(1 for row in scored if row.get("compile_gate", {}).get("compiled") is True),
        "compile_failed": sum(1 for row in scored if row.get("compile_gate", {}).get("compiled") is False),
        "mean_score": round(sum(row["score"] for row in scored) / max(1, len(scored)), 2),
        "pass_rate": round(sum(1 for row in scored if row["passed"]) / max(1, len(scored)), 4),
        "by_focus": {
            focus: round(sum(vals) / len(vals), 2)
            for focus, vals in sorted(by_focus.items())
        },
        "by_subskill": {
            subskill: round(sum(vals) / len(vals), 2)
            for subskill, vals in sorted(by_subskill.items())
        },
    }
    out = {"summary": summary, "rows": scored}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.out_md, summary, scored)
    print(json.dumps(summary, indent=2))


def _write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# economistRL Scorecard v1",
        "",
        f"- adapter_id: `{summary['adapter_id']}`",
        f"- outputs_scored: {summary['outputs_scored']}",
        f"- rolling_compile_rate: {summary.get('rolling_compile_rate', 0.0)}",
        f"- compile_gate_applied: {summary.get('compile_gate_applied', 0)}",
        f"- compiled: {summary.get('compiled', 0)}",
        f"- compile_failed: {summary.get('compile_failed', 0)}",
        f"- mean_score: {summary['mean_score']}",
        f"- pass_rate: {summary['pass_rate']}",
        "",
        "| task_id | score | base | compiled | passed | difficulty | subskill | missing_mechanics | failures |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        failures = ", ".join(row["failures"][:5])
        missing = ", ".join(row["missing_mechanics"])
        lines.append(
            f"| `{row['task_id']}` | {row['score']:.2f} | {float(row.get('base_reward', 0.0)) * 100.0:.2f} | "
            f"{row.get('compile_gate', {}).get('compiled')} | {row['passed']} | "
            f"{row['difficulty']} | `{row['subskill']}` | {missing} | {failures} |"
        )
    lines.append("")
    lines.append("## Focus Scores")
    lines.append("")
    for focus, score in summary["by_focus"].items():
        lines.append(f"- `{focus}`: {score:.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="economistRL task-bank and scoring utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate economistRL task-bank shape.")
    p_validate.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)

    p_add = sub.add_parser("add-task", help="Append a task to the economistRL task bank.")
    p_add.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--title", required=True)
    p_add.add_argument("--prompt", required=True)
    p_add.add_argument("--subskill", required=True)
    p_add.add_argument("--difficulty", choices=["smoke", "standard", "hard", "adversarial"], default="standard")
    p_add.add_argument("--curriculum-track", choices=sorted(VALID_CURRICULUM_TRACKS), default="economy")
    p_add.add_argument("--focus-tag", action="append", default=[])
    p_add.add_argument("--required", action="append", default=[])
    p_add.add_argument("--optional", action="append", default=[])
    p_add.add_argument("--forbidden", action="append", default=[])
    p_add.add_argument("--mechanic", action="append", default=[], help="name:keyword,keyword")
    p_add.add_argument("--min-chars", type=int, default=160)
    p_add.add_argument("--max-chars", type=int, default=1400)
    p_add.add_argument("--reference-answer", default="")
    p_add.add_argument("--simulation-goal", default="Run a focused 20-tick simulation and score task-specific behavior change.")
    p_add.add_argument("--relevant-state", action="append", default=[])
    p_add.add_argument("--test-goal", default="Executable tests verify the intended behavior changed without requiring exact implementation names.")
    p_add.add_argument("--test-command", action="append", default=[])
    p_add.add_argument("--test-check", action="append", default=[])
    p_add.add_argument("--static-hook", action="append", default=[])
    p_add.add_argument("--static-signal", action="append", default=[])
    p_add.add_argument("--formula-pattern", action="append", default=[])

    p_score = sub.add_parser("score", help="Score economistRL output JSONL.")
    p_score.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p_score.add_argument("--outputs-jsonl", type=Path, required=True)
    p_score.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p_score.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p_score.add_argument(
        "--rolling-compile-rate",
        type=float,
        default=0.0,
        help="Recent compile pass rate used by the decaying compile gate.",
    )

    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_tasks(args.tasks.expanduser().resolve()), indent=2))
    elif args.command == "add-task":
        add_task(args)
    elif args.command == "score":
        score_outputs(args)


if __name__ == "__main__":
    main()
