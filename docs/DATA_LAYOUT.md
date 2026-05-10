# Data and checkpoint layout (by model lineage)

This repo’s **default** fine-tuning stack targets `**mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`**. Paths and filenames encode that lineage so local trees stay unambiguous when you add other base models later.

This file owns **path layout only**. For current-vs-historical run interpretation, use `docs/RUNS.md`; for command examples, use `docs/WORKFLOW.md`; for chunking internals, use `docs/CHUNKED_GAME_TEXT.md`.

## Lineage slug


| Concept         | Value                                          |
| --------------- | ---------------------------------------------- |
| Hugging Face id | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` |
| Directory slug  | `qwen25-coder-7b` (hyphens)                    |


Constants for scripts: `**scripts/fe_lineage.py`**.

## Export (base-model agnostic)


| Artifact              | Path                       |
| --------------------- | -------------------------- |
| Raw repo walk (JSONL) | `data/raw/repo_text.jsonl` |


Produced by `scripts/export_repo_for_training.py`. Same export can feed any MLX base; it is not named after a model size.

## LoRA JSONL splits (default game-text corpus)


| Role                       | Path                                                           |
| -------------------------- | -------------------------------------------------------------- |
| Train / valid / test JSONL | `data/lora/qwen25-coder-7b/game_text/{train,valid,test}.jsonl` |


Built by `scripts/build_lora_dataset.py` (default `--out-dir` matches the table above). `**ml_workflow.py prepare**` and `**ml_workflow.py smoke**` write here.

**Chunking (default):** long files become **multiple JSONL rows** with `# path:` / `# part:` / `# chars:` headers so **tail content is not dropped** (see `**docs/CHUNKED_GAME_TEXT.md`**). Pass `**--no-chunk**` only for the old single-row-per-file truncation behavior.

### Historical path

Older notes and runs used `**data/lora/game_text/**` (no model slug). That directory is **not** the default anymore. To reuse existing splits without re-running `build_lora_dataset.py`:

```bash
mkdir -p data/lora/qwen25-coder-7b
# copy or symlink — example:
# ln -sfn "$(pwd)/data/lora/game_text" data/lora/qwen25-coder-7b/game_text
```

## LoRA training config


| File                                 | Purpose                                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `training/lora_qwen25_coder_7b.yaml` | Default `mlx_lm.lora` config: **7B** model id, `data` + `adapter_path` for this lineage |


## Default adapter output (workflow / YAML)


| Purpose                      | Path                                     |
| ---------------------------- | ---------------------------------------- |
| YAML + `ml_workflow.py train | full`default`--adapter-path`             |
| Gradio train UI default      | `checkpoints/fe-lora-qwen25-coder-7b-ui` |


Named experiment dirs (e.g. `checkpoints/fe-lora-game-text-20260428`) are fine; prefer including `**qwen25-coder-7b`** or `**7b**` in new checkpoint folder names when the adapter is for this base.

## Other `data/lora/` trees

Datasets that are not the plain repo export game-text corpus (e.g. arena pairwise chat) keep their own paths, e.g. `data/lora/game_task_pairwise/`, `data/lora/game_text_pairwise_cautious_text/`. They are orthogonal naming; you can still train **7B** LoRA from them by passing `--data` to `mlx_lm.lora`.

These extra datasets are experiment inputs, not current default paths. Record which one a run used in `docs/run_history.md` / run manifests, and summarize any recommended promotion in `docs/RUNS.md`.

### Compare feedback supervision datasets

| Role | Path |
| --- | --- |
| Structured compare pairwise records | `data/lora/compare_feedback/pairwise_feedback.jsonl` |
| Span rewrite supervision rows | `data/lora/compare_feedback/rewrite_feedback.jsonl` |
| Export manifest | `data/lora/compare_feedback/manifest.json` |

Generate with `scripts/export_compare_feedback_training_data.py` from dashboard SQLite (`data/private_dashboard.sqlite3` by default). Optional ingestion into game pairwise SFT builder is available via `scripts/build_game_task_pairwise_dataset.py --compare-feedback-pairwise <path>`.

### Routing classification datasets

| Role | Path |
| --- | --- |
| Raw routing label sources | `data/routing/` |
| Live prompt captures | `benchmarks/results/routing_prompt_lab/*.jsonl` |
| Train / valid / test JSONL | `data/lora/routing_classifier/<dataset_version>/{train,valid,test}.jsonl` |
| Build manifest | `data/lora/routing_classifier/<dataset_version>/manifest.json` |

Contract and field requirements live in `docs/ROUTING_DATASET_CONTRACT.md`.

### Documentation / run-analysis specialist (mlx-lab prose)

| Role | Path |
| --- | --- |
| Train / valid / test JSONL | `data/lora/adapters/documentation_specialist/{train,valid,test}.jsonl` |
| Build manifest | `data/lora/adapters/documentation_specialist/manifest.json` |

Build with **`python scripts/ml_workflow.py documentation-dataset`** (or `scripts/adapters/build_documentation_specialist_dataset.py`). Current routed shadow checkpoint lives at `checkpoints/adapters/documentation/cycle3` (see `training/adapter_registry_v1.json`).

### GitHub strict TypeScript corpus (compile-safe curation)


| Role                       | Path                                                                        |
| -------------------------- | --------------------------------------------------------------------------- |
| Train / valid / test JSONL | `data/lora/qwen25-coder-7b/github_ts_compile_safe/{train,valid,test}.jsonl` |
| Per-sample metadata index  | `data/lora/qwen25-coder-7b/github_ts_compile_safe/samples_metadata.jsonl`   |
| Build manifest             | `data/lora/qwen25-coder-7b/github_ts_compile_safe/manifest.json`            |


Build with `scripts/build_github_ts_dataset.py`. The script can ingest explicit repos (`--repo owner/name`) and/or discover candidates via GitHub search (`--discover-query`), then filter by permissive SPDX license, strict `tsconfig`, and non-trivial TS app structure.
