# Dataset Contract v1

Per-adapter datasets are generated under `data/lora/adapters/<adapter_id>/` with:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `manifest.json`

## Row schema requirements

Each row must carry:

- `task_id`
- `task_type`
- `complexity`
- `risk_class`
- `dataset_role` (`family_specific`, `shared_anchor`, or `hard_negative`)
- `policy_version`
- `lineage`

Rows can be `messages` format (preferred for arena-derived supervision) or `text` format if source data is legacy.

## Mixing policy (locked v1 defaults)

- 85% family-specific rows
- 10% shared anti-overfit anchor rows
- 5% hard-negative/repair rows

The builder enforces this policy for train split composition and records the exact ratio in each adapter manifest.

## Shared anti-overfit corpus

Shared rows are loaded from `data/lora/shared_general_anchor/` (train/valid/test JSONL). If missing, adapter build should fail in strict production mode; current scaffold keeps best-effort behavior for early rollout.

## Leakage guard

Split assignment is deterministic by stable hash key (`seed + task_id/record_id`) to keep rows reproducible and avoid accidental leakage across train/test for identical record identities.

