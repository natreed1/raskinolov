# Routing data

Versioned routing label sources live here.

Suggested conventions:

- `routing_labels_<version>.jsonl` for curated benchmark/live labels
- `routing_live_review_queue_<date>.jsonl` for pending manual review
- Specialist smoke sets: `loading_screen_eval_prompts_v1.jsonl`, `save_load_api_guard_eval_prompts_v1.jsonl`, `economy_tooltip_eval_prompts_v1.jsonl`, **`documentation_eval_prompts_v1.jsonl`** (pair with `benchmarks/*_eval_tasks_v1.json` + `run_routing_benchmark.py`; regenerate docs tasks via `scripts/build_documentation_eval_tasks_v1.py`)
- Mixed routing regression: `Python scripts/build_mixed_routing_eval_v1.py` → `benchmarks/mixed_routing_eval_v1.json` (loading_screen + save/load + **documentation** eval rows, policy fixtures, specialist probes; seed **42**) + `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/mixed_routing_eval_v1.json --mode both`

Use `scripts/routing_prompt_lab.py` to capture prompts and
`scripts/build_routing_training_dataset.py` to generate deterministic
`train.jsonl` / `valid.jsonl` / `test.jsonl` splits under
`data/lora/routing_classifier/<dataset_version>/`.
