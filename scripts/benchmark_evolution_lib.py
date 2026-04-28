"""
Shared helpers for benchmark tiers, expect checks, and task mutation (evolutionary suite).
Used by run_game_benchmark.py and evolve_benchmark_seasons.py.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

TIER_ORDER = ("C", "B", "A")

# Mirrors sim-system tier idea: stricter rubric + more decode budget at top tier.
TIER_SPECS: Dict[str, Dict[str, float]] = {
    "C": {"max_tokens": 384, "min_chars_mul": 1.0},
    "B": {"max_tokens": 512, "min_chars_mul": 1.2},
    "A": {"max_tokens": 768, "min_chars_mul": 1.45},
}

# Keeps evolved + tier-scaled rubrics within one greedy decode at ~768 tokens.
MAX_EXPECT_MIN_CHARS = 320

CATEGORIES_CODEISH = frozenset(
    {"types", "pure-ts", "api-shape", "serialization", "frontend", "spatial", "game-systems"}
)

VOCAB_FRAGMENTS = [
    "SimResult",
    "hexDistance",
    "DEFAULT_AI_PARAMS",
    "AiParams",
    "runSimulation",
    "useGameStore",
    "siegeChance",
    "wallBuildPriority",
    "Biome",
    "Tile",
    "promotion",
    "relegation",
    "mutation",
    "season",
]


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Tasks file must be a JSON array")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "id" not in item or "prompt" not in item or "expect" not in item:
            raise ValueError(f"Invalid task at index {i}: need id, prompt, expect")
    return data


def check_expect(text: str, expect: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    t = text or ""

    min_chars = expect.get("min_chars")
    if min_chars is not None and len(t) < int(min_chars):
        failures.append(f"min_chars: got {len(t)}, need >= {min_chars}")

    for sub in expect.get("all_contains", []) or []:
        if sub not in t:
            failures.append(f"missing required substring: {sub!r}")

    anys = expect.get("any_contains", []) or []
    if anys and not any(s in t for s in anys):
        failures.append(f"none of any_contains matched: {anys}")

    for sub in expect.get("none_contains", []) or []:
        if sub in t:
            failures.append(f"forbidden substring present: {sub!r}")

    return (len(failures) == 0, failures)


def scale_expect_for_tier(expect: Dict[str, Any], tier: str) -> Dict[str, Any]:
    spec = TIER_SPECS.get(tier, TIER_SPECS["B"])
    out = copy.deepcopy(expect)
    mc = out.get("min_chars")
    if mc is not None:
        out["min_chars"] = min(
            MAX_EXPECT_MIN_CHARS,
            int(math.ceil(int(mc) * spec["min_chars_mul"])),
        )
    return out


def tier_max_tokens(tier: str) -> int:
    return int(TIER_SPECS.get(tier, TIER_SPECS["B"])["max_tokens"])


def task_difficulty_score(task: Dict[str, Any]) -> int:
    """Proxy fitness when no model is available: higher = harder rubric."""
    ex = task.get("expect") or {}
    prompt = task.get("prompt") or ""
    n_all = len(ex.get("all_contains") or [])
    n_any = len(ex.get("any_contains") or [])
    mc = int(ex.get("min_chars") or 0)
    return mc * 3 + n_all * 40 + n_any * 8 + len(prompt) // 5


def mutate_task(
    task: Dict[str, Any],
    rng: random.Random,
    anchor_ids: frozenset[str],
) -> Dict[str, Any]:
    """Offspring from one task; anchors are never mutated (caller should skip)."""
    tid = task["id"]
    if tid in anchor_ids:
        return copy.deepcopy(task)
    t = copy.deepcopy(task)
    suffix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5))
    t["id"] = f"{tid}-evo-{suffix}"
    ex = t.setdefault("expect", {})
    if rng.random() < 0.55 and ex.get("min_chars") is not None:
        ex["min_chars"] = min(
            MAX_EXPECT_MIN_CHARS,
            int(ex["min_chars"]) + rng.randint(8, 72),
        )
    elif rng.random() < 0.45:
        ex.setdefault("min_chars", 40)
        ex["min_chars"] = min(
            MAX_EXPECT_MIN_CHARS,
            int(ex["min_chars"]) + rng.randint(8, 56),
        )
    if ex.get("min_chars") is not None:
        ex["min_chars"] = min(MAX_EXPECT_MIN_CHARS, max(8, int(ex["min_chars"])))
    cat = task.get("category") or ""
    codeish = cat in CATEGORIES_CODEISH or "TypeScript" in (task.get("prompt") or "")
    if rng.random() < 0.4 and codeish:
        ac = list(ex.get("all_contains") or [])
        frag = rng.choice(VOCAB_FRAGMENTS)
        if frag not in ac and len(ac) < 6:
            ac.append(frag)
            ex["all_contains"] = ac
    if rng.random() < 0.25:
        tail = " Be explicit and use Fallen Empire terminology where it fits."
        t["prompt"] = (t.get("prompt") or "") + tail
    return t


def crossover(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Blend expect constraints from two parents (ids from a)."""
    out = copy.deepcopy(a)
    suffix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5))
    out["id"] = f"{a['id']}-x-{b['id']}-{suffix}"[:120]
    ea = a.get("expect") or {}
    eb = b.get("expect") or {}
    ac = list(dict.fromkeys((ea.get("all_contains") or []) + (eb.get("all_contains") or [])))[:8]
    out.setdefault("expect", {})
    out["expect"]["all_contains"] = ac
    mc = max(int(ea.get("min_chars") or 0), int(eb.get("min_chars") or 0))
    out["expect"]["min_chars"] = min(MAX_EXPECT_MIN_CHARS, mc + rng.randint(0, 36))
    return out
