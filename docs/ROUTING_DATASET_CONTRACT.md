# Routing Dataset Contract (v1)

`scripts/build_routing_training_dataset.py` writes Router V2 classifier data under:

- `data/lora/routing_classifier/<dataset_version>/train.jsonl`
- `data/lora/routing_classifier/<dataset_version>/valid.jsonl`
- `data/lora/routing_classifier/<dataset_version>/test.jsonl`
- `data/lora/routing_classifier/<dataset_version>/manifest.json`

## Row schema (`routing_classifier_row_v1`)

Required keys:

- `schema_version`: string (`routing_classifier_row_v1`)
- `prompt`: natural-language router input
- `expected_adapter_id`: adapter/task family label (`loading_screen`, `hud_status`, `economy_tooltip`, `combat_risk`, `save_load_api_guard`, `ai_planning_explanation`, `documentation`, `general_fallback`)
- `source`: source family (`benchmark_tasks`, `live`, `curated`)
- `case_id`: stable case identifier
- `accepted_for_training`: boolean

Optional keys:

- `expected_legacy_route`: one of `local|hybrid|frontier` for dual-label evaluation
- `expected_keywords`: keyword hints captured with the row
- `label_source`: `human_label`, `bootstrap_policy_v2`, or curator-specific marker

## Split policy

- Deterministic split by stable hash on prompt text + seed
- Target ratios: train/valid/test = 85/10/5
- Very small datasets still guarantee at least one train row and preserve a test row when possible

## Manifest schema (`routing_classifier_dataset_v1`)

Key fields:

- `dataset_version`, `seed`, `created_utc`
- row counts (`rows_total`, `rows_train`, `rows_valid`, `rows_test`)
- adapter distribution (`labels`)
- source distribution (`sources`)
- input files (`benchmark_tasks`, `live_jsonl`, `curated_jsonl`)
- `low_confidence_threshold` used for live-prompt filtering

