#!/usr/bin/env python3
"""Extract outcome-first skills/manifolds artifacts from mass benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent

SPECIALIST_MASS_TASKS: Dict[str, str] = {
    "loading_screen": "loading_screen_mass_tasks_v1.json",
    "hud_status": "hud_status_mass_tasks_v1.json",
    "economy_tooltip": "economy_tooltip_mass_tasks_v1.json",
    "combat_risk": "combat_risk_mass_tasks_v1.json",
    "save_load_api_guard": "save_load_api_guard_mass_tasks_v1.json",
    "ai_planning_explanation": "ai_planning_explanation_mass_tasks_v1.json",
}

CONCEPT_LEXICON: Dict[str, Tuple[str, ...]] = {
    "Loading/Start": ("loading", "start", "queued", "preparing", "ready", "readiness"),
    "UI Hierarchy": ("title", "subtitle", "hierarchy", "layout", "spacing", "contrast", "badge", "chip", "tooltip"),
    "Morale": ("morale",),
    "Supply": ("supply",),
    "Risk": ("risk", "odds"),
    "Terrain/Fort": ("terrain", "wall", "fortification", "fortified"),
    "Economy": ("gold", "income", "upkeep", "market", "workforce", "trade"),
    "Schema/API": ("schema", "version", "errors", "serialize", "migration", "api"),
    "Planning Intent": ("defend", "expand", "scout", "reinforce", "intent", "council"),
}

BUCKETS = ["core", "ui_change", "constraints", "transfer"]
STATUS_RE = re.compile(r"\[(PASS|FAIL)\]\s+([^\s]+)\s+\(([^)]+)\)\s+(\d+) chars\s+([0-9.]+)s\s+cap=([0-9.]+)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bucket_for_category(category: str) -> str:
    cat = (category or "").lower()
    if "constraint" in cat:
        return "constraints"
    if "ui_change" in cat:
        return "ui_change"
    if "transfer" in cat:
        return "transfer"
    return "core"


def _extract_concepts(prompt: str) -> List[str]:
    p = (prompt or "").lower()
    found = [name for name, words in CONCEPT_LEXICON.items() if any(w in p for w in words)]
    return found or ["Generic"]


def _latest_mass_runs_by_specialist() -> Dict[str, Dict[str, Any]]:
    runs_dir = REPO / "benchmarks" / "results" / "runs"
    latest: Dict[str, Dict[str, Any]] = {}
    if not runs_dir.is_dir():
        return latest
    for mf in runs_dir.glob("*/manifest.json"):
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("subcommand")) != "benchmark":
            continue
        argv = payload.get("argv") or []
        if not isinstance(argv, list):
            continue
        specialists: List[str] = []
        tasks_path = ""
        for idx, token in enumerate(argv):
            if token == "--specialist" and idx + 1 < len(argv):
                specialists.append(str(argv[idx + 1]))
            if token == "--tasks" and idx + 1 < len(argv):
                tasks_path = str(argv[idx + 1])
        if len(specialists) != 1:
            continue
        specialist = specialists[0]
        expected_name = SPECIALIST_MASS_TASKS.get(specialist)
        if not expected_name or Path(tasks_path).name != expected_name:
            continue
        current = latest.get(specialist)
        finished = str(payload.get("finished_at") or "")
        if not current or finished > str(current.get("finished_at") or ""):
            latest[specialist] = {
                "run_id": str(payload.get("run_id") or mf.parent.name),
                "finished_at": finished,
            }
    return latest


def _collect_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    latest = _latest_mass_runs_by_specialist()
    task_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for specialist, task_file in SPECIALIST_MASS_TASKS.items():
        path = REPO / "benchmarks" / task_file
        if not path.is_file():
            continue
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id") or "")
            if not tid:
                continue
            prompt = str(task.get("prompt") or "")
            category = str(task.get("category") or "")
            task_lookup[(specialist, tid)] = {
                "prompt": prompt,
                "category": category,
                "bucket": _bucket_for_category(category),
                "concepts": _extract_concepts(prompt),
            }

    rows: List[Dict[str, Any]] = []
    for specialist, run in latest.items():
        log_path = REPO / "benchmarks" / "results" / "runs" / run["run_id"] / "logs" / "run_game_benchmark.log"
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for m in STATUS_RE.finditer(text):
            status, task_id, _category_from_log, chars, seconds, cap = m.groups()
            meta = task_lookup.get((specialist, task_id), {})
            rows.append(
                {
                    "id": f"{specialist}:{task_id}",
                    "specialist": specialist,
                    "task_id": task_id,
                    "passed": 1 if status == "PASS" else 0,
                    "chars": int(chars),
                    "seconds": float(seconds),
                    "capability": float(cap),
                    "bucket": str(meta.get("bucket") or "core"),
                    "concepts": list(meta.get("concepts") or ["Generic"]),
                    "prompt": str(meta.get("prompt") or ""),
                }
            )
    return rows, latest


def _bootstrap_effect(vals: List[int], all_mean: float, rng: random.Random, n_resamples: int) -> Dict[str, Any]:
    if not vals:
        return {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "sign_agreement": 0.0}
    local = float(sum(vals) / len(vals))
    effect = local - all_mean
    samples: List[float] = []
    for _ in range(max(1, n_resamples)):
        draw = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
        samples.append(float(sum(draw) / len(draw)) - all_mean)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    if effect >= 0:
        sign_agreement = sum(1 for v in samples if v >= 0.0) / len(samples)
    else:
        sign_agreement = sum(1 for v in samples if v <= 0.0) / len(samples)
    return {
        "effect": effect,
        "ci_low": lo,
        "ci_high": hi,
        "sign_agreement": sign_agreement,
    }


def _kmeans(points: np.ndarray, k: int, seed: int, max_iter: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if k <= 1 or n == 1:
        return np.zeros(n, dtype=np.int64)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    centers = points[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for j in range(k):
            members = points[labels == j]
            if len(members) == 0:
                centers[j] = points[rng.integers(0, n)]
            else:
                centers[j] = np.mean(members, axis=0)
    return labels


def extract_skills_v1(
    *,
    min_support_tasks_per_skill: int,
    min_support_specialists_per_skill: int,
    bootstrap_resamples: int,
    min_effect_abs: float,
    stability_min_sign_agreement: float,
    region_stability_min: float,
    seed: int,
) -> Dict[str, Any]:
    rows, latest = _collect_rows()
    if not rows:
        raise SystemExit("No mass benchmark rows found. Run specialist mass benchmarks first.")
    rng = random.Random(seed)
    overall_mean = float(sum(r["passed"] for r in rows) / len(rows))

    # Skill candidates (expand beyond original nodes with concept-bucket composites).
    skill_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        skill_members[f"bucket::{r['bucket']}"].append(r)
        for c in r["concepts"]:
            skill_members[f"concept::{c}"].append(r)
            skill_members[f"concept_bucket::{c}::{r['bucket']}"].append(r)

    candidates: List[Dict[str, Any]] = []
    for skill_id, members in skill_members.items():
        vals = [int(m["passed"]) for m in members]
        specialist_support = len({m["specialist"] for m in members})
        task_support = len(members)
        stats = _bootstrap_effect(vals, overall_mean, rng, bootstrap_resamples)
        kind = skill_id.split("::", 1)[0]
        candidates.append(
            {
                "skill_id": skill_id,
                "kind": kind,
                "support_tasks": task_support,
                "support_specialists": specialist_support,
                "pass_rate": float(sum(vals) / len(vals)) if vals else 0.0,
                "effect": float(stats["effect"]),
                "ci_low": float(stats["ci_low"]),
                "ci_high": float(stats["ci_high"]),
                "sign_agreement": float(stats["sign_agreement"]),
            }
        )

    retained: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for c in candidates:
        ci_nonzero = (c["ci_low"] > 0.0 and c["ci_high"] > 0.0) or (c["ci_low"] < 0.0 and c["ci_high"] < 0.0)
        keep = (
            c["support_tasks"] >= min_support_tasks_per_skill
            and c["support_specialists"] >= min_support_specialists_per_skill
            and abs(c["effect"]) >= min_effect_abs
            and ci_nonzero
            and c["sign_agreement"] >= stability_min_sign_agreement
        )
        if keep:
            retained.append(c)
        else:
            rejected.append(c)
    retained.sort(key=lambda x: abs(x["effect"]), reverse=True)
    exploratory = [
        c
        for c in candidates
        if c["support_tasks"] >= 4 and c["support_specialists"] >= 2 and abs(c["effect"]) >= 0.05
    ]
    exploratory.sort(key=lambda x: abs(x["effect"]), reverse=True)

    # Specialist profiles over retained skills.
    by_skill_rows = {k: v for k, v in skill_members.items()}
    specialists = sorted({r["specialist"] for r in rows})
    specialist_profiles: Dict[str, Any] = {}
    for s in specialists:
        srows = [r for r in rows if r["specialist"] == s]
        row: Dict[str, Any] = {
            "specialist": s,
            "task_count": len(srows),
            "overall_pass_rate": float(sum(r["passed"] for r in srows) / len(srows)) if srows else 0.0,
            "skills": {},
        }
        for skill in retained:
            sid = skill["skill_id"]
            members = [r for r in by_skill_rows.get(sid, []) if r["specialist"] == s]
            if not members:
                continue
            pr = float(sum(r["passed"] for r in members) / len(members))
            row["skills"][sid] = {
                "support_tasks": len(members),
                "pass_rate": pr,
                "delta_vs_skill_global": pr - skill["pass_rate"],
            }
        specialist_profiles[s] = row

    # Graph/manifold build.
    task_nodes = [f"task::{r['id']}" for r in rows]
    specialist_nodes = [f"specialist::{s}" for s in specialists]
    skill_nodes = [f"skill::{s['skill_id']}" for s in retained]
    node_ids = task_nodes + specialist_nodes + skill_nodes
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    edges: List[Dict[str, Any]] = []
    # task-specialist edges
    for r in rows:
        u = f"task::{r['id']}"
        v = f"specialist::{r['specialist']}"
        edges.append({"u": u, "v": v, "weight": 1.0 + 0.5 * float(r["passed"]), "kind": "task_specialist"})
    # task-skill edges (retained only)
    retained_ids = {s["skill_id"] for s in retained}
    for sid in retained_ids:
        for r in by_skill_rows.get(sid, []):
            u = f"task::{r['id']}"
            v = f"skill::{sid}"
            edges.append({"u": u, "v": v, "weight": 1.0, "kind": "task_skill"})

    neigh: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        neigh[e["u"]].append(e["v"])
        neigh[e["v"]].append(e["u"])

    for e in edges:
        du = len(neigh[e["u"]])
        dv = len(neigh[e["v"]])
        e["curvature"] = float(4 - du - dv)

    n = len(node_ids)
    A = np.zeros((n, n), dtype=np.float64)
    curvs = np.array([e["curvature"] for e in edges], dtype=np.float64)
    c_mu = float(np.mean(curvs)) if len(curvs) else 0.0
    c_sd = float(np.std(curvs)) if len(curvs) and float(np.std(curvs)) > 1e-12 else 1.0
    for e in edges:
        i = node_idx[e["u"]]
        j = node_idx[e["v"]]
        z = (float(e["curvature"]) - c_mu) / c_sd
        w = float(e["weight"]) * math.exp(0.20 * z)
        if w > A[i, j]:
            A[i, j] = w
            A[j, i] = w
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)
    embed_dims = 2 if n >= 3 else 1
    if embed_dims == 2:
        X = np.column_stack([eigvecs[:, 1], eigvecs[:, 2]])
    else:
        X = np.column_stack([eigvecs[:, 0]])

    # Cluster task manifold regions.
    task_indices = np.array([node_idx[nid] for nid in task_nodes], dtype=np.int64)
    task_points = X[task_indices]
    k = max(3, min(8, int(round(math.sqrt(max(1, len(task_nodes) / 3.0))))))
    labels = _kmeans(task_points, k=k, seed=seed)
    task_region: Dict[str, int] = {task_nodes[i]: int(labels[i]) for i in range(len(task_nodes))}

    # Region stability and specialist-region fit.
    regions: Dict[int, Dict[str, Any]] = {}
    # Build quick edge map for boundary/intra splits.
    edge_list = [(e["u"], e["v"], float(e["curvature"])) for e in edges]
    for region_id in sorted(set(int(x) for x in labels)):
        tset = {tid for tid, rid in task_region.items() if rid == region_id}
        region_nodes = set(tset)
        # include specialist/skill neighbors connected to region tasks
        for t in list(tset):
            region_nodes.update(neigh[t])
        intra_curv: List[float] = []
        boundary_neg: List[float] = []
        for u, v, curv in edge_list:
            u_in = u in region_nodes
            v_in = v in region_nodes
            if u_in and v_in:
                intra_curv.append(curv)
            elif u_in != v_in:
                boundary_neg.append(max(0.0, -curv))
        mean_intra = float(sum(intra_curv) / len(intra_curv)) if intra_curv else 0.0
        boundary_penalty = float(sum(boundary_neg) / len(boundary_neg)) if boundary_neg else 0.0
        stability = mean_intra - boundary_penalty

        # Specialist fit in region (pass rate on tasks in region for that specialist).
        fit: Dict[str, float] = {}
        region_row_ids = {tid.split("task::", 1)[1] for tid in tset}
        for s in specialists:
            s_rows = [r for r in rows if r["specialist"] == s and r["id"] in region_row_ids]
            fit[s] = float(sum(r["passed"] for r in s_rows) / len(s_rows)) if s_rows else 0.0

        regions[region_id] = {
            "region_id": region_id,
            "task_count": len(tset),
            "mean_intra_curvature": mean_intra,
            "boundary_penalty": boundary_penalty,
            "stability_score": stability,
            "specialist_fit": fit,
            "task_ids": sorted(region_row_ids),
        }

    region_vals = [regions[k]["stability_score"] for k in sorted(regions)]
    mu = float(sum(region_vals) / len(region_vals)) if region_vals else 0.0
    sd = float(np.std(np.array(region_vals, dtype=np.float64))) if region_vals else 0.0
    if sd <= 1e-12:
        sd = 1.0
    for k in sorted(regions):
        z = (regions[k]["stability_score"] - mu) / sd
        regions[k]["stability_z"] = z
        regions[k]["stable_for_direct_routing"] = bool(
            regions[k]["stability_score"] >= region_stability_min or z >= 0.0
        )

    top_pos = sorted(retained, key=lambda x: x["effect"], reverse=True)[:12]
    top_neg = sorted(retained, key=lambda x: x["effect"])[:12]

    return {
        "created_utc": _utc_now(),
        "latest_runs": latest,
        "rows_count": len(rows),
        "thresholds": {
            "min_support_tasks_per_skill": min_support_tasks_per_skill,
            "min_support_specialists_per_skill": min_support_specialists_per_skill,
            "bootstrap_resamples": bootstrap_resamples,
            "min_effect_abs": min_effect_abs,
            "stability_min_sign_agreement": stability_min_sign_agreement,
            "region_stability_min": region_stability_min,
            "seed": seed,
        },
        "retained_skills": retained,
        "exploratory_skills": exploratory,
        "rejected_skills_count": len(rejected),
        "top_positive_skills": top_pos,
        "top_negative_skills": top_neg,
        "specialist_profiles": specialist_profiles,
        "manifolds": {
            "node_count": n,
            "edge_count": len(edges),
            "eigenvalues": [float(v) for v in eigvals[: min(16, len(eigvals))]],
            "regions": [regions[k] for k in sorted(regions)],
            "task_region_assignments": {k.split("task::", 1)[1]: v for k, v in task_region.items()},
        },
    }


def _write_report(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Skills Extraction v1 Report")
    lines.append("")
    lines.append(f"- Generated: `{payload['created_utc']}`")
    lines.append(f"- Task outcome rows: `{payload['rows_count']}`")
    lines.append(f"- Retained skills: `{len(payload['retained_skills'])}`")
    lines.append(f"- Exploratory skills (relaxed): `{len(payload.get('exploratory_skills', []))}`")
    lines.append(f"- Rejected skills: `{payload['rejected_skills_count']}`")
    lines.append("")
    lines.append("## Top Positive Skills (effect on pass)")
    lines.append("")
    for s in payload.get("top_positive_skills", [])[:10]:
        lines.append(
            f"- `{s['skill_id']}` effect `{s['effect']:.3f}` "
            f"(CI `{s['ci_low']:.3f}`..`{s['ci_high']:.3f}`), "
            f"support `{s['support_tasks']}` tasks / `{s['support_specialists']}` specialists"
        )
    lines.append("")
    lines.append("## Top Negative Skills (effect on pass)")
    lines.append("")
    for s in payload.get("top_negative_skills", [])[:10]:
        lines.append(
            f"- `{s['skill_id']}` effect `{s['effect']:.3f}` "
            f"(CI `{s['ci_low']:.3f}`..`{s['ci_high']:.3f}`), "
            f"support `{s['support_tasks']}` tasks / `{s['support_specialists']}` specialists"
        )
    lines.append("")
    lines.append("## Exploratory Skills (relaxed thresholds)")
    lines.append("")
    for s in payload.get("exploratory_skills", [])[:16]:
        lines.append(
            f"- `{s['skill_id']}` effect `{s['effect']:.3f}` "
            f"(CI `{s['ci_low']:.3f}`..`{s['ci_high']:.3f}`), "
            f"support `{s['support_tasks']}` / `{s['support_specialists']}`"
        )
    lines.append("")
    lines.append("## Region Stability")
    lines.append("")
    for r in payload.get("manifolds", {}).get("regions", []):
        lines.append(
            f"- Region `{r['region_id']}` tasks `{r['task_count']}` "
            f"stability `{r['stability_score']:.3f}` (z `{r.get('stability_z', 0.0):.3f}`) "
            f"(intra `{r['mean_intra_curvature']:.3f}`, boundary penalty `{r['boundary_penalty']:.3f}`), "
            f"stable=`{r['stable_for_direct_routing']}`"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract outcome-first skills and manifolds artifacts.")
    parser.add_argument("--min-support-tasks-per-skill", type=int, default=8)
    parser.add_argument("--min-support-specialists-per-skill", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--min-effect-abs", type=float, default=0.10)
    parser.add_argument("--stability-min-sign-agreement", type=float, default=0.8)
    parser.add_argument("--region-stability-min", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skills-out", type=Path, default=REPO / "data" / "routing" / "skills_v1.json")
    parser.add_argument(
        "--profiles-out", type=Path, default=REPO / "data" / "routing" / "specialist_skill_profiles_v1.json"
    )
    parser.add_argument("--manifolds-out", type=Path, default=REPO / "data" / "routing" / "skill_manifolds_v1.json")
    parser.add_argument(
        "--report-out", type=Path, default=REPO / "benchmarks" / "results" / "skills_extraction_report_v1.md"
    )
    args = parser.parse_args()

    payload = extract_skills_v1(
        min_support_tasks_per_skill=int(args.min_support_tasks_per_skill),
        min_support_specialists_per_skill=int(args.min_support_specialists_per_skill),
        bootstrap_resamples=int(args.bootstrap_resamples),
        min_effect_abs=float(args.min_effect_abs),
        stability_min_sign_agreement=float(args.stability_min_sign_agreement),
        region_stability_min=float(args.region_stability_min),
        seed=int(args.seed),
    )

    skills_payload = {
        "schema_version": "skills_v1",
        "created_utc": payload["created_utc"],
        "rows_count": payload["rows_count"],
        "thresholds": payload["thresholds"],
        "retained_skills": payload["retained_skills"],
        "exploratory_skills": payload["exploratory_skills"],
        "top_positive_skills": payload["top_positive_skills"],
        "top_negative_skills": payload["top_negative_skills"],
    }
    profiles_payload = {
        "schema_version": "specialist_skill_profiles_v1",
        "created_utc": payload["created_utc"],
        "rows_count": payload["rows_count"],
        "specialists": payload["specialist_profiles"],
    }
    manifolds_payload = {
        "schema_version": "skill_manifolds_v1",
        "created_utc": payload["created_utc"],
        "rows_count": payload["rows_count"],
        "thresholds": payload["thresholds"],
        **payload["manifolds"],
    }

    args.skills_out.parent.mkdir(parents=True, exist_ok=True)
    args.skills_out.write_text(json.dumps(skills_payload, indent=2), encoding="utf-8")
    args.profiles_out.parent.mkdir(parents=True, exist_ok=True)
    args.profiles_out.write_text(json.dumps(profiles_payload, indent=2), encoding="utf-8")
    args.manifolds_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifolds_out.write_text(json.dumps(manifolds_payload, indent=2), encoding="utf-8")
    _write_report(payload, args.report_out)

    print(
        json.dumps(
            {
                "ok": True,
                "skills_out": str(args.skills_out),
                "profiles_out": str(args.profiles_out),
                "manifolds_out": str(args.manifolds_out),
                "report_out": str(args.report_out),
                "retained_skills": len(payload["retained_skills"]),
                "regions": len(payload["manifolds"]["regions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
