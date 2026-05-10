# Routing Dataset Contract

This contract defines the routing dataset shape for dual-label supervision:

- Primary label: `expected_adapter_id`
- Compatibility label: `expected_legacy_route` (`local`, `hybrid`, `frontier`)

The contract supports historical benchmark prompts, curated prompts, and live prompts
captured from interactive routing sessions.

## Canonical locations

- Source datasets: `data/routing/`
- Prompt-lab captures: `benchmarks/results/routing_prompt_lab/`
- Built train/valid/test datasets: `data/lora/routing_classifier/<dataset_version>/`

## Required source row fields

Every labeled source row must include:

- `prompt`
- `expected_adapter_id`
- `expected_legacy_route`
- `source` (`benchmark`, `live`, `curated`)
- `policy_version`
- `lineage`
- `created_at`

Optional but recommended:

- `record_id` (stable id; generated from prompt hash if missing)
- `reviewer`
- `notes`
- `predicted_adapter_id`
- `predicted_legacy_route`
- `confidence`
- `ambiguity`
- `risk_class`
- `complexity`

## Label hierarchy rules

1. Adapter label is authoritative for supervision and confusion analysis.
2. Legacy route label is retained for backward compatibility and trend continuity.
3. If both labels are present, they must be logically compatible with policy expectations.
4. Benchmark-only legacy-route tasks may be auto-labeled with adapter predictions, but must be
   marked with lineage metadata explaining label provenance.

## Training split artifacts

Each dataset build must emit:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `manifest.json`

Each split row includes:

- `record_id`
- `text` (the prompt)
- `labels` with `adapter_id` and `legacy_route`
- `source`
- `policy_version`
- `lineage`
- `priority_hard_example` (bool)
- `created_at`

## Deterministic split policy

Use a stable hash key from `seed + record_id`:

- `< 0.85` -> train
- `< 0.95` -> valid
- otherwise -> test

This avoids train/test leakage and keeps split membership stable across rebuilds.

## Hard-example priority

Mark rows as `priority_hard_example=true` when any is true:

- live row explicitly accepted for training from prompt-lab,
- prediction mismatch (`predicted_*` vs `expected_*`),
- confidence below threshold (default `0.62`),
- source metadata flags uncertainty/edge-case status.

## Validation checks

A dataset build should fail fast if:

- required fields are missing on explicitly labeled rows,
- label values are outside supported adapter ids or legacy routes,
- output split files would be empty.

## Router supervisor completions (`router_chat_gradio`)

`scripts/router_chat_gradio.py` optionally appends JSON Lines supervisor captures (`ROUTER_CHAT_LOG_JSONL` or `--interaction-log-jsonl`).
These are auxiliary *prompt → assistant* artifacts for manual review—they do not replace the routing-label contract above.

**Merge hygiene:** prefer downstream chat/SFT rows where **`recommended_for_sft_assistant_turn`** is **`true`**
—that requires **`mlx_finish_reason == "stop"`**, non-empty **`generation_text_only`**,
no **`generation_budget_hit`**, and **`unbalanced_markdown_fence`** is **`false`**.

Responses with **`generation_budget_hit: true`** (default decode budget ~**2048** new tokens unless you raise `MAX_TOKENS`/`--max-tokens`)
typically look “cut off” for economy/HUD/UI tasks; keep **`likely_incomplete_generation`** for exclusions or iterative completion tooling.
