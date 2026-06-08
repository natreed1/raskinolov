#!/usr/bin/env python3
"""Parse Vitest JSON reports for partial targeted-test credit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def vitest_json_report_command(test_path: str, report_path: str) -> str:
    """Single vitest invocation with JSON output for assertion-level scoring."""
    test = str(test_path).strip()
    report = str(report_path).strip()
    return f"node_modules/.bin/vitest run {test} --reporter=json --outputFile={report}"


def _flatten_assertions(test_results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in test_results:
        if not isinstance(suite, dict):
            continue
        for assertion in suite.get("assertionResults") or []:
            if not isinstance(assertion, dict):
                continue
            title = str(assertion.get("title") or assertion.get("fullName") or "assertion").strip()
            status = str(assertion.get("status") or "").strip().lower()
            passed = status == "passed"
            rows.append({"name": title, "passed": passed, "score": 1.0 if passed else 0.0})
    return rows


def _vitest_it_key(title: str) -> str:
    """Normalize Vitest ``it`` titles for goal lookup (JSON may omit ``goal_`` prefix)."""
    token = str(title or "").strip().lower()
    if not token:
        return ""
    return token if token.startswith("goal_") else f"goal_{token}"


def _per_it_scores(test_results: list[Any]) -> list[dict[str, Any]]:
    """One row per Vitest ``it()`` (top-level test case), not per ``expect``."""
    rows: list[dict[str, Any]] = []
    for suite in test_results:
        if not isinstance(suite, dict):
            continue
        for assertion in suite.get("assertionResults") or []:
            if not isinstance(assertion, dict):
                continue
            title = str(assertion.get("title") or "").strip()
            if not title:
                continue
            status = str(assertion.get("status") or "").strip().lower()
            passed = status == "passed"
            rows.append(
                {
                    "title": title,
                    "lookup_key": _vitest_it_key(title),
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                }
            )
    return rows


def parse_vitest_json_report(path: Path) -> dict[str, Any]:
    """Return partial credit payload from a Vitest JSON report file."""
    if not path.is_file():
        return {
            "score": 0.0,
            "checks": [],
            "assertions_passed": 0,
            "assertions_total": 0,
            "tests_passed": 0,
            "tests_total": 0,
            "vitest_ran": False,
            "error": "missing_report",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "score": 0.0,
            "checks": [],
            "assertions_passed": 0,
            "assertions_total": 0,
            "tests_passed": 0,
            "tests_total": 0,
            "vitest_ran": False,
            "error": f"invalid_report:{exc}",
        }

    test_results = payload.get("testResults") if isinstance(payload, dict) else None
    test_list = test_results if isinstance(test_results, list) else []
    assertions = _flatten_assertions(test_list)
    per_it = _per_it_scores(test_list)

    tests_total = int(payload.get("numTotalTests") or 0) if isinstance(payload, dict) else 0
    tests_passed = int(payload.get("numPassedTests") or 0) if isinstance(payload, dict) else 0

    if per_it:
        passed = sum(1 for row in per_it if row["passed"])
        total = len(per_it)
        score = passed / max(1, total)
        checks = [{"name": row["title"], "score": row["score"]} for row in per_it]
    elif assertions:
        passed = sum(1 for row in assertions if row["passed"])
        total = len(assertions)
        score = passed / max(1, total)
        checks = [{"name": row["name"], "score": row["score"]} for row in assertions]
    elif tests_total > 0:
        passed = tests_passed
        total = tests_total
        score = passed / max(1, total)
        checks = [{"name": f"vitest_test_{idx + 1}", "score": 1.0 if idx < passed else 0.0} for idx in range(total)]
    else:
        passed = total = 0
        score = 0.0
        checks = []

    return {
        "score": round(float(score), 4),
        "checks": checks,
        "assertions_passed": passed if assertions else 0,
        "assertions_total": len(assertions),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "per_it": per_it,
        "vitest_ran": True,
        "source": "vitest_json_report",
    }


def merge_outcome_checks_with_vitest(
    task: dict[str, Any],
    vitest: dict[str, Any],
) -> dict[str, Any]:
    """Score targeted tests; prefer goal-weighted credit when goals map to ``goal_*`` tests."""
    from economist_rl_vitest_goals import goal_it_title, simulation_goals

    goals = simulation_goals(task)
    per_it = list(vitest.get("per_it") or [])
    if not per_it:
        checks_raw = list(vitest.get("checks") or [])
        per_it = [{"title": str(c.get("name") or ""), "passed": float(c.get("score") or 0) >= 1.0, "score": float(c.get("score") or 0)} for c in checks_raw]

    if goals and per_it:
        by_title: dict[str, dict[str, Any]] = {}
        for row in per_it:
            by_title[str(row.get("title") or "")] = row
            key = str(row.get("lookup_key") or _vitest_it_key(str(row.get("title") or "")))
            if key:
                by_title[key] = row
        merged: list[dict[str, Any]] = []
        weight_sum = 0.0
        weighted = 0.0
        for goal in goals:
            title = goal_it_title(str(goal["name"]))
            row = by_title.get(title) or by_title.get(_vitest_it_key(title))
            if row is None:
                goal_slug = _vitest_it_key(str(goal["name"])).removeprefix("goal_")
                for cand_title, cand in by_title.items():
                    cand_slug = _vitest_it_key(cand_title).removeprefix("goal_")
                    if not cand_slug or not goal_slug:
                        continue
                    if cand_slug.startswith(goal_slug[:40]) or goal_slug.startswith(cand_slug[:40]):
                        row = cand
                        break
            w = float(goal.get("weight") or 1.0)
            weight_sum += w
            if row is not None:
                s = float(row.get("score") or 0.0)
                merged.append({"name": str(goal["name"]), "score": s, "vitest_it": title})
                weighted += w * s
            else:
                merged.append({"name": str(goal["name"]), "score": 0.0, "vitest_it": title})
        score = weighted / max(1e-9, weight_sum) if weight_sum > 0 else float(vitest.get("score") or 0.0)
        return {
            "score": round(score, 4),
            "checks": merged,
            "source": "vitest_goal_weighted",
            "assertions_passed": vitest.get("assertions_passed"),
            "assertions_total": vitest.get("assertions_total"),
            "tests_passed": vitest.get("tests_passed"),
            "tests_total": vitest.get("tests_total"),
            "vitest_ran": vitest.get("vitest_ran"),
        }

    spec = task.get("targeted_tests") if isinstance(task.get("targeted_tests"), dict) else {}
    names = [str(item).strip() for item in spec.get("outcome_checks") or [] if str(item).strip()]
    checks = list(vitest.get("checks") or [])
    if names and checks:
        per = float(vitest.get("score") or 0.0)
        merged = [{"name": name, "score": per} for name in names]
    elif names:
        per = float(vitest.get("score") or 0.0)
        merged = [{"name": name, "score": per} for name in names]
    else:
        merged = checks
    return {
        "score": float(vitest.get("score") or 0.0),
        "checks": merged,
        "source": vitest.get("source") or "vitest_json_report",
        "assertions_passed": vitest.get("assertions_passed"),
        "assertions_total": vitest.get("assertions_total"),
        "tests_passed": vitest.get("tests_passed"),
        "tests_total": vitest.get("tests_total"),
        "vitest_ran": vitest.get("vitest_ran"),
    }
