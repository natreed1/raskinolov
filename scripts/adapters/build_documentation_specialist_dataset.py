#!/usr/bin/env python3
"""Build a compact documentation/run-analysis specialist dataset for fallen-empire-lora prose.

Teacher rows teach:** canonical filenames, MLX defaults, orchestrator wording, checkpoint paths,
append-only docs discipline, and training-run artifact reading** so a small LoRA can normalize
README/SESSION notes and summarize workflow runs safely.

Does **not** use game HUD/arena pairwise data or game feature-code supervision (keeps supervision
local to this ML repo).

Output: ``data/lora/adapters/documentation_specialist/{train,valid,test}.jsonl`` + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
OUT_DEFAULT = REPO / "data" / "lora" / "adapters" / "documentation_specialist"
TASK_ID_SYNTH = "mlx-lora-docs-normalize"
LINEAGE = "documentation_specialist:v1"

DOCS_SYSTEM = (
    "You maintain Markdown technical notes and training-run analysis for fallen-empire-lora, "
    "the MLX/LoRA lab adjacent to Fallen Empire. This specialist is for the LoRA lab repo, "
    "not game feature implementation. Prefer plain language and exact identifiers from "
    "`docs/PROJECT_STATE.md`: default models, script names, canonical paths "
    "(`docs/run_history.md` append-only, `docs/SESSION_LOG.md` append-only, "
    "`benchmarks/results/runs/`, `training_trajectory.jsonl`, `SOURCE_REPO`, "
    "`GAME_ARENA_ROOT`), and `python scripts/ml_workflow.py` subcommands."
)

# Draft → polished pairs (minimal hallucination supervision; curator-authored).
GOLD_DOC_PAIRS: Sequence[Tuple[str, str]] = (
    (
        "Train with ml workflow full on fe-lora latest 400 iterations chunk 6k.",
        "**Train:** `python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest -- "
        "--iters 400` — pass-through flags after `--` go to `mlx_lm.lora`; default chunking/stack details live "
        "in **`docs/DATA_LAYOUT.md`** and **`docs/CHUNKED_GAME_TEXT.md`.**",
    ),
    (
        "Run history commits when you iterate ml workflow except dashboard.",
        "Every **`python scripts/ml_workflow.py`** invocation except **`arena-dashboard`** should append **`docs/run_history.md`** "
        "and emit run artifacts under **`benchmarks/results/runs/<run_id>/`** (see `.cursor/rules/precise-ml-documentation.mdc`).",
    ),
    (
        "Arena gate uses qwencoder 25 7 billion base without quant.",
        "Arena local MLX attempts default to **`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`** (MLX 4-bit instruct); adapters must match "
        "the base checkpoint recorded on the run manifest.",
    ),
    (
        "Loading screen champ lives at checkpoints/loading_cycle2.",
        "Loading-screen promoted adapter:** `checkpoints/adapters/loading_screen/cycle2`** (see **`training/adapter_registry_v1.json`**).",
    ),
    (
        "Project facts live README only.",
        "Environment pins, reproduced commands, and known quirks belong in **`docs/PROJECT_STATE.md`**; **`README`** stays pointers only.",
    ),
    (
        "SESSION_LOG replacements when same day edits happen.",
        "`**docs/SESSION_LOG.md`** is **append-only** — add new sections at the top; do not rewrite prior sessions.",
    ),
    (
        "dataset path default game texts under data/lora/game_text/train.jsonl",
        "Default curated JSONL lineage for chunked exports:** `data/lora/qwen25-coder-7b/game_text/{train,valid,test}.jsonl`** "
        "(see **`docs/DATA_LAYOUT.md`**; legacy `data/lora/game_text/` noted as historical).",
    ),
    (
        "activate python3 venv fallen empire",
        "```bash\ncd fallen-empire-lora\nsource .venv/bin/activate\n.venv/bin/python scripts/ml_workflow.py smoke\n```",
    ),
    (
        "how do i rebuild arena pairwise jsonl?",
        "`scripts/build_game_task_pairwise_dataset.py` builds chat JSONL consumed by pairwise trainers; pairwise source "
        "**`benchmarks/results/game_task_pairwise_training_data.jsonl`** feeds specialist builders (paths vary — see **`docs/WORKFLOW.md`**).",
    ),
    (
        "fallen empire game arena root env var?",
        "**`GAME_ARENA_ROOT`** selects disposable arena worktrees; **`SOURCE_REPO`** points at `~/fallen-empire` (see "
        "**`PROJECT_STATE`** + `scripts/ml_workflow.py` defaults).",
    ),
    (
        "what file lists adapter promotion state?",
        "Use **`training/adapter_registry_v1.json`** (taxonomy **`scripts/adapters/taxonomy.py`**) plus narrative context in **`docs/RUNS.md`**.",
    ),
    (
        "Docs canvas skill path users remember wrong.",
        "Cursor Skills live under user `.cursor/skills-cursor/` and plugin-backed skills for docs canvas / PR canvases "
        "**only when the workflow calls for canvases**, not arbitrary ML notes.",
    ),
    (
        "Rename Qwen coder 1.5B mainline for new runs.",
        "Historical 1.5B adapters are comparison-only unless you match that base in YAML + adapter config; current training "
        "defaults target **7B Instruct 4-bit** above.",
    ),
    (
        "Where are arena capability HTML dashboard generated?",
        "`python scripts/ml_workflow.py arena-dashboard` rebuilds **`benchmarks/results/arena_dashboard.html`** from local run JSON "
        "(no new `run_history` row for that subcommand).",
    ),
    (
        "training config filename for qwen 25 coder 7b",
        "LoRA hyperparameters live in **`training/lora_qwen25_coder_7b.yaml`**; pass `-c training/lora_qwen25_coder_7b.yaml` via "
        "`mlx_lm.lora` (usually through `ml_workflow.py train`).",
    ),
    (
        "smoke test script does what",
        "`python scripts/ml_workflow.py smoke` runs a fast synthetic train + benchmark path to validate the venv + MLX stack "
        "(see **`docs/WORKFLOW.md`**).",
    ),
    (
        "Make sure the documentation agent is for this project and not the video game.",
        "Scope the **`documentation`** adapter to **`fallen-empire-lora`**: MLX/LoRA workflow docs, run manifests, "
        "adapter registry notes, dashboards, and token-routing decisions. It may mention `/Users/natreed/fallen-empire`, "
        "`SOURCE_REPO`, or `GAME_ARENA_ROOT` only as lab inputs; it should not act as the game implementation specialist.",
    ),
    (
        "Analyze a training run and tell me whether it should become the routed docs adapter.",
        "For training-run analysis, read **`benchmarks/results/runs/<run_id>/manifest.json`**, **`RUN.md`**, "
        "**`logs/mlx_lm_lora_train.log`**, and **`training_trajectory.jsonl`**. Summarize exit code, adapter path, "
        "data dir, resume source, final/best validation loss, test loss when present, and whether "
        "**`training/adapter_registry_v1.json`** should move the documentation adapter path.",
    ),
    (
        "Token usage optimization flow for Codex docs questions.",
        "Route low-risk **fallen-empire-lora** documentation and training-run-reading prompts to the local "
        "**`documentation`** adapter. Reserve hybrid/frontier paths for broad architecture, security, or cross-repo code "
        "changes, and log adapter id, confidence, route, and resolved checkpoint for later SFT review.",
    ),
    (
        "What files should the run-reading specialist inspect before updating docs?",
        "Use the committed index **`docs/run_history.md`** to identify the run, then inspect ignored artifacts under "
        "**`benchmarks/results/runs/<run_id>/`**: `manifest.json`, `RUN.md`, step logs, and `training_trajectory.jsonl`. "
        "Do not infer training quality from the Fallen Empire game checkout.",
    ),
    (
        "Which dataset backs the project documentation/run analyst adapter?",
        "The project documentation/run-analysis adapter trains on **`data/lora/adapters/documentation_specialist/{train,valid,test}.jsonl`** "
        "from **`scripts/adapters/build_documentation_specialist_dataset.py`**; it is intentionally separate from HUD, economy, "
        "combat, and other game-task specialist corpora.",
    ),
)


def _stable_unit(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _split_rows(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get("record_id"))
        u = _stable_unit(seed, key)
        if u < 0.80:
            train.append(row)
        elif u < 0.90:
            valid.append(row)
        else:
            test.append(row)
    if rows and not train:
        train.append(rows[0])
    if len(rows) > 1 and not valid:
        valid.append(rows[1])
    if len(rows) > 2 and not test:
        test.append(rows[2])
    return train, valid, test


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_repeated(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    if not rows or target <= 0:
        return []
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < target:
        out.append(dict(rows[i % len(rows)]))
        i += 1
    return out


def _gold_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, (draft, polished) in enumerate(GOLD_DOC_PAIRS):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": DOCS_SYSTEM},
                    {"role": "user", "content": draft.strip()},
                    {"role": "assistant", "content": polished.strip()},
                ],
                "task_id": TASK_ID_SYNTH,
                "task_type": "documentation",
                "complexity": "low",
                "risk_class": "low",
                "dataset_role": "core_docs_style",
                "record_id": f"gold-docs:{i}",
                "lineage": LINEAGE,
                "policy_version": "router_policy_v1",
            }
        )
    return rows


def build_dataset(*, out_dir: Path, seed: int, min_train_core_rows: int) -> Dict[str, Any]:
    core_rows = _gold_rows()
    train_core, valid_core, test_core = _split_rows(core_rows, seed)
    target = max(min_train_core_rows, len(train_core))
    train_rows = _sample_repeated(train_core, target)
    valid_rows = valid_core[:] if valid_core else _sample_repeated(train_rows, max(2, len(test_core)))
    test_rows = test_core[:] if test_core else _sample_repeated(valid_rows, max(2, len(test_core)))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    manifest = {
        "schema_version": "documentation_specialist_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "policy_version": "router_policy_v1",
        "lineage": LINEAGE,
        "task_id_alias": TASK_ID_SYNTH,
        "raw_counts": {"core_docs_rows": len(core_rows)},
        "split_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "min_train_core_rows": min_train_core_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build documentation specialist JSONL from curated drafts.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--min-train-core-rows", type=int, default=120)
    args = ap.parse_args()
    manifest = build_dataset(
        out_dir=args.out_dir.expanduser().resolve(),
        seed=args.seed,
        min_train_core_rows=args.min_train_core_rows,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
