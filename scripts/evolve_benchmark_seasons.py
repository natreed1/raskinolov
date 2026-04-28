#!/usr/bin/env python3
"""
Evolve the benchmark task population across **seasons**, with **tiered** evaluation (C / B / A),
aligned with Fallen Empire's sim-system metaphor (anchors, promotion pressure, mutation).

Default: **no MLX** — fast structural evolution (difficulty / tier-scaled rubric).

Use `--with-mlx` to load the model each season and score tasks with real generations (slow, meaningful fitness).

Outputs a JSON **task array** for `run_game_benchmark.py --tasks`.

Example:
  python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json
  python scripts/evolve_benchmark_seasons.py --with-mlx --output benchmarks/results/evolved_tasks.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Import sibling module when executed as script
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_evolution_lib as bel

DEFAULT_CONFIG = REPO_ROOT / "training" / "evolution_config.json"
DEFAULT_SEED = REPO_ROOT / "benchmarks" / "fallen_empire_tasks.json"

IMMIGRANTS = [
    {
        "id": "immigrant-hex-ring",
        "category": "spatial",
        "prompt": "In TypeScript, write a function `hexRing(center: { q: number; r: number }, radius: number): Array<{ q: number; r: number }>` that returns all hexes at exactly that axial distance (not inside the disk). Outline only: explain the loop pattern in comments + empty return [] stub.",
        "expect": {
            "all_contains": ["q", "r", "radius"],
            "any_contains": ["for", "while", "push", "loop"],
            "min_chars": 120,
        },
    },
    {
        "id": "immigrant-tier-essay",
        "category": "evolution",
        "prompt": "Fallen Empire's simulation league uses tiers C, B, and A with seasons and promotion. In 4–6 sentences, explain how that maps to **curriculum learning** for a code LLM (easier tasks early, harder later, anchors unchanged).",
        "expect": {
            "any_contains": ["C", "B", "A", "curriculum", "season", "anchor", "promot", "harder", "easier"],
            "min_chars": 220,
        },
    },
    {
        "id": "immigrant-merge-ai-params",
        "category": "game-systems",
        "prompt": "Write TypeScript that takes `Partial<AiParams>` and returns full `AiParams` by merging over `DEFAULT_AI_PARAMS` (immutable: do not mutate defaults). Code only.",
        "expect": {
            "all_contains": ["DEFAULT_AI_PARAMS", "AiParams"],
            "any_contains": ["spread", "...", "Object.assign"],
            "none_contains": ["python", "def "],
            "min_chars": 60,
        },
    },
]


def _load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_one_mlx(
    task: Dict[str, Any],
    tier: str,
    model,
    tokenizer,
    gen_kwargs: dict,
    system: str,
) -> Tuple[bool, int]:
    ex = bel.scale_expect_for_tier(task["expect"], tier)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    out = __import__("mlx_lm").generate(model, tokenizer, prompt=prompt, verbose=False, **gen_kwargs)
    ok, _ = bel.check_expect(out, ex)
    diff = bel.task_difficulty_score({**task, "expect": ex})
    return ok, diff


def _fitness(
    mode: str,
    passed: bool,
    difficulty: int,
) -> float:
    if mode == "adversarial_benchmark":
        return (10000 if not passed else 0) + difficulty
    # champions_for_curriculum
    return (5000 + difficulty) if passed else difficulty


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve benchmark tasks across seasons / tiers")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed-tasks", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-mlx", action="store_true", help="Score with real model each season (slow)")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    be = cfg["benchmark_evolution"]
    pop_n = int(be["population_size"])
    n_seasons = int(be["seasons"])
    elite_frac = float(be["elite_fraction"])
    offspring = int(be["offspring_per_elite"])
    immigration_n = int(be["immigration_per_season"])
    anchors = frozenset(be["anchor_task_ids"])
    tier_cycle: List[str] = list(be["tier_cycle"])
    mode = str(be["selection_mode"])

    rng = random.Random(args.seed)
    seed_tasks = bel.load_tasks(args.seed_tasks.resolve())
    by_id = {t["id"]: copy.deepcopy(t) for t in seed_tasks}

    missing = [a for a in anchors if a not in by_id]
    if missing:
        raise SystemExit(f"anchor_task_ids not in seed tasks: {missing}")

    population: List[Dict[str, Any]] = copy.deepcopy(seed_tasks)
    while len(population) < pop_n:
        parent = rng.choice(seed_tasks)
        population.append(bel.mutate_task(parent, rng, anchors))

    population = population[:pop_n]

    system = (
        "You are Albert, assisting with the Fallen Empire strategy game (TypeScript / React / Zustand). "
        "Follow each user instruction literally."
    )

    model = tokenizer = None
    gen_kwargs: dict = {}
    if args.with_mlx:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        load_kw = {}
        if args.adapter_path:
            load_kw["adapter_path"] = args.adapter_path
        print(f"Loading {args.model!r} …", file=sys.stderr)
        model, tokenizer = load(args.model, **load_kw)
        gen_kwargs = {"max_tokens": bel.tier_max_tokens("B")}
        if args.temp > 0:
            gen_kwargs["sampler"] = make_sampler(temp=args.temp, top_p=1.0)

    history: List[Dict[str, Any]] = []

    for season in range(n_seasons):
        tier = tier_cycle[season % len(tier_cycle)]
        if args.with_mlx and model is not None:
            gen_kwargs["max_tokens"] = bel.tier_max_tokens(tier)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for task in population:
            if not args.with_mlx:
                diff = bel.task_difficulty_score(task)
                scaled_ex = bel.scale_expect_for_tier(task["expect"], tier)
                diff_tier = bel.task_difficulty_score({**task, "expect": scaled_ex})
                # Structural proxy: no model. Prefer harder rubrics under adversarial; prefer balanced mid-hard under curriculum.
                if mode == "adversarial_benchmark":
                    fit = float(diff_tier)
                else:
                    fit = float(diff_tier) + rng.uniform(0, 12) * (1.0 if task["id"] in anchors else 1.0)
            else:
                passed, diff = _evaluate_one_mlx(
                    task, tier, model, tokenizer, gen_kwargs, system
                )
                fit = _fitness(mode, passed, diff)
            scored.append((fit, task))

        scored.sort(key=lambda x: -x[0])
        elite_n = max(2, int(math.ceil(len(population) * elite_frac)))
        elites = [t for _, t in scored[:elite_n]]

        next_pop: List[Dict[str, Any]] = []
        for t in elites:
            if t["id"] in anchors:
                next_pop.append(copy.deepcopy(by_id[t["id"]]))
            else:
                next_pop.append(copy.deepcopy(t))

        children: List[Dict[str, Any]] = []
        for _ in range(offspring * len(elites)):
            p = rng.choice(elites)
            if p["id"] in anchors:
                continue
            children.append(bel.mutate_task(p, rng, anchors))
        for _ in range(max(0, offspring // 2)):
            if len(elites) < 2:
                break
            a, b = rng.sample(elites, 2)
            if a["id"] in anchors or b["id"] in anchors:
                continue
            children.append(bel.crossover(a, b, rng))

        for _ in range(immigration_n):
            children.append(copy.deepcopy(rng.choice(IMMIGRANTS)))
            children[-1] = bel.mutate_task(children[-1], rng, frozenset())

        merged = next_pop + children
        rng.shuffle(merged)
        population = merged[:pop_n]

        # Ensure anchors always present and canonical
        pop_ids = {t["id"] for t in population}
        for aid in anchors:
            if aid not in pop_ids:
                population.append(copy.deepcopy(by_id[aid]))
        population = population[:pop_n]

        history.append(
            {
                "season": season + 1,
                "tier": tier,
                "top_fitness": scored[0][0] if scored else 0.0,
                "median_fitness": scored[len(scored) // 2][0] if scored else 0.0,
            }
        )

    # Unique ids for benchmark runner
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for t in population:
        tid = t["id"]
        if tid in seen:
            t = bel.mutate_task(t, rng, anchors)
            tid = t["id"]
        seen.add(tid)
        deduped.append(t)
    population = deduped

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(population, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "config": str(args.config),
        "seed_tasks": str(args.seed_tasks),
        "seasons": n_seasons,
        "with_mlx": args.with_mlx,
        "selection_mode": mode,
        "history": history,
        "task_count": len(population),
    }
    print(json.dumps(report, indent=2), file=sys.stderr)
    print(f"Wrote task array: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
