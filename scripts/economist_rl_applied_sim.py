#!/usr/bin/env python3
"""Run 20-tick traces against the applied sandbox mechanic (post-apply worktree)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from economist_rl_game_sandbox import sandbox_integration_mode
from economist_rl_sandbox_envs import env_dir_for_subsection, resolve_subsection, tick_function_name
from economist_rl_sim_harnesses import _score_all_goals, _uses_bounded_mechanics
from economist_rl_task_execution import sandbox_paths


def applied_sim_cli_path() -> str:
    return "src/lib/economistRl/envs/shared/runAppliedSimCli.ts"


def _flatten_trace_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(row.get("cities"), list) and row["cities"]:
        city = row["cities"][0] if isinstance(row["cities"][0], dict) else {}
        storage = city.get("storage") if isinstance(city.get("storage"), dict) else {}
        out["population"] = float(city.get("population") or 0)
        out["storageFood"] = float(storage.get("food") or 0)
        out["storageGoods"] = float(storage.get("goods") or 0)
        out["morale"] = float(city.get("morale") or 0)
        out["productivity"] = float(city.get("productivity") or 0)
    if isinstance(row.get("city"), dict):
        city = row["city"]
        storage = city.get("storage") if isinstance(city.get("storage"), dict) else {}
        out["population"] = float(city.get("population") or 0)
        out["storageFood"] = float(storage.get("food") or 0)
        out["storageGoods"] = float(storage.get("goods") or 0)
        out["morale"] = float(city.get("morale") or 0)
    if isinstance(row.get("players"), list) and row["players"]:
        player = row["players"][0] if isinstance(row["players"][0], dict) else {}
        out["playerGold"] = float(player.get("gold") or 0)
    if isinstance(row.get("taskExt"), dict):
        for key, raw in row["taskExt"].items():
            try:
                out[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
    for key, raw in row.items():
        if key in {"cities", "city", "players", "taskExt", "tiles", "territory", "units"}:
            continue
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _trace_metrics(task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    subsection = resolve_subsection(task)
    tick_count = int(payload.get("tickCount") or (task.get("simulation_spec") or {}).get("tick_count") or 20)
    scar_end = _flatten_trace_row(payload.get("scarcityEnd") if isinstance(payload.get("scarcityEnd"), dict) else {})
    sur_end = _flatten_trace_row(payload.get("surplusEnd") if isinstance(payload.get("surplusEnd"), dict) else {})
    scar_start = _flatten_trace_row(payload.get("scarcityStart") if isinstance(payload.get("scarcityStart"), dict) else {})

    if subsection == "market_elasticity_pricing":
        return {
            "scarcity_price_up": scar_end.get("marketPriceGold", 0) > scar_start.get("marketPriceGold", scar_end.get("marketPriceGold", 0)),
            "surplus_price_down": sur_end.get("marketPriceGold", 0) < scar_end.get("marketPriceGold", 0),
            "price_bounded": 5.0 <= scar_end.get("marketPriceGold", 0) <= 25.0,
            "smooth_adjustment": scar_end.get("marketPriceGold") != scar_start.get("marketPriceGold"),
        }
    if subsection == "food_population_feedback":
        return {
            "birth_rate_decreased": scar_end.get("appliedBirthRate", 1) < scar_start.get("appliedBirthRate", 1),
            "birth_rate_diff_safe_unsafe": sur_end.get("appliedBirthRate", 0) >= scar_end.get("appliedBirthRate", 0),
            "min_food": min(scar_end.get("storageFood", 0), sur_end.get("storageFood", 0)),
            "starvation_handled": scar_end.get("storageFood", -1) >= 0,
            "smooth_population": scar_end.get("population") != scar_start.get("population"),
        }
    if subsection == "labor_wage_productivity":
        return {
            "morale_reacts": scar_end.get("morale", 0) != scar_start.get("morale", 0),
            "wage_cut_saves": True,
            "productivity_penalty": sur_end.get("productivity", 1) >= scar_end.get("productivity", 0),
            "gradual_recovery": sur_end.get("morale", 0) >= scar_end.get("morale", 0),
        }
    if subsection == "inventory_storage_spoilage":
        return {
            "decay_applies": scar_end.get("storageFood", 0) != scar_start.get("storageFood", 0),
            "preservation_reduces_decay": sur_end.get("storageFood", 0) >= scar_end.get("storageFood", 0),
            "capacity_respected": scar_end.get("storageFood", -1) >= 0,
            "no_total_wipe": scar_end.get("storageFood", 0) > 0,
        }
    if subsection == "upkeep_progressive_costs":
        return {
            "small_force_affordable": True,
            "large_force_pressures": True,
            "upkeep_bounded": True,
            "wage_cut_saves": sur_end.get("playerGold", 0) >= scar_end.get("playerGold", 0),
        }
    if subsection == "resource_projection_cache_integrity":
        valid_end = scar_end.get("resourceProjectionValid", 0)
        valid_start = scar_start.get("resourceProjectionValid", 0)
        return {
            "recomputes": 1 if valid_end != valid_start else 0,
            "stale_reads": 0,
            "cache_integrity": valid_end >= valid_start,
        }
    return {
        "bounded_adjustment": scar_end != sur_end,
        "end_gt_start": bool(scar_end),
        "tick_count": tick_count,
    }


def _goal_payload_from_trace(task: dict[str, Any], trace: dict[str, Any], *, mechanic_source: str) -> dict[str, Any]:
    tick_count = int((task.get("simulation_spec") or {}).get("tick_count") or 20)
    bounded = _uses_bounded_mechanics(mechanic_source, task)
    goal_scores = _score_all_goals(task, trace, bounded=bounded, output=mechanic_source)
    goals = []
    for goal in task.get("simulation_spec", {}).get("goals") or []:
        if not isinstance(goal, dict):
            continue
        name = str(goal.get("name") or "")
        goals.append(
            {
                "name": name,
                "score": float(goal_scores.get(name, 0.0)),
                "weight": float(goal.get("weight") or 1.0),
            }
        )
    return {
        "ticks": tick_count,
        "goals": goals,
        "trace_summary": trace,
        "env": "applied_mechanic_v1",
        "bounded_mechanics_detected": bounded,
    }


def run_applied_mechanic_simulation(
    *,
    worktree: Path,
    task: dict[str, Any],
    log_dir: Path,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Execute shared TS CLI against applied mechanic.ts; score simulation_spec goals."""
    subsection = resolve_subsection(task)
    if sandbox_integration_mode(subsection) == "lab_scalar":
        return {"ticks": 0, "goals": [], "error": "lab_scalar_no_applied_sim", "env": "applied_mechanic_v1"}

    lib_rel, _test_rel = sandbox_paths(task)
    mechanic_abs = (worktree / lib_rel).resolve()
    if not mechanic_abs.is_file():
        return {"ticks": 0, "goals": [], "error": "mechanic_missing", "env": "applied_mechanic_v1"}

    mechanic_source = mechanic_abs.read_text(encoding="utf-8")
    env_base = env_dir_for_subsection(subsection)
    config = {
        "mechanicAbs": str(mechanic_abs),
        "runTicksAbs": str((worktree / env_base / "runTicks.ts").resolve()),
        "fixtureAbs": str((worktree / env_base / "fixture.ts").resolve()),
        "tickFunction": tick_function_name(task),
        "tickCount": int((task.get("simulation_spec") or {}).get("tick_count") or 20),
    }
    config_path = log_dir / "applied_sim_config.json"
    log_path = log_dir / "applied_sim.log"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    cli = (worktree / applied_sim_cli_path()).resolve()
    cmd = (
        f"npx ts-node -r tsconfig-paths/register --project tsconfig.train.json "
        f"{cli} {config_path}"
    )
    proc = subprocess.run(
        ["sh", "-lc", cmd],
        cwd=str(worktree),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    log_path.write_text(
        f"$ {cmd}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\nexit={proc.returncode}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return {
            "ticks": 0,
            "goals": [],
            "error": f"applied_sim_cli_failed:{proc.returncode}",
            "env": "applied_mechanic_v1",
            "log_path": str(log_path),
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {
            "ticks": 0,
            "goals": [],
            "error": "applied_sim_cli_invalid_json",
            "env": "applied_mechanic_v1",
            "log_path": str(log_path),
        }
    trace = _trace_metrics(task, payload)
    result = _goal_payload_from_trace(task, trace, mechanic_source=mechanic_source)
    result["log_path"] = str(log_path)
    return result
