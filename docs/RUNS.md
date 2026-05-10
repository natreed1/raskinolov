# Current and Historical Runs

Use this file to answer "what should I use now?" without reading the full run archive.

Arena **dashboard** (local runs), **apply KPIs**, **progressive policy**, and **promotion gate**: `**docs/ARENA_ROADMAP.md`**.

---

## Source Of Truth Split


| File                                | Role                                                                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `docs/RUNS.md`                      | Current recommendations plus interpretation notes for historical runs.                                                      |
| `docs/run_history.md`               | Append-only index of every `scripts/ml_workflow.py` invocation. Keep all rows, including failed and superseded experiments. |
| `docs/SESSION_LOG.md`               | Append-only narrative of substantive work, decisions, commands, blockers, and why changes happened.                         |
| `docs/PROJECT_STATE.md`             | Current environment, defaults, known issues, and durable workflow facts. Avoid long run recaps here.                        |
| `benchmarks/results/runs/<run_id>/` | Ignored detailed artifacts: manifests, logs, summaries, trajectories, dashboards.                                           |


## Current Defaults


| Purpose                       | Current value                                          | Notes                                                                        |
| ----------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Fine-tuning lineage           | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`         | Canonical paths live in `scripts/fe_lineage.py` and `docs/DATA_LAYOUT.md`.   |
| Default LoRA YAML             | `training/lora_qwen25_coder_7b.yaml`                   | Replaces the old `training/lora_qwen_coder.yaml` name.                       |
| Default game-text data        | `data/lora/qwen25-coder-7b/game_text/`                 | Built by `scripts/build_lora_dataset.py`; long files are chunked by default. |
| Latest default adapter output | `checkpoints/fe-lora-qwen25-coder-7b-latest`           | Generic train/full output path.                                              |
| Current arena adapter default | `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` | Shared through `scripts/fe_lineage.py`; do not mix with 1.5B adapters.       |


The smoke path can still use the smaller 1.5B model for quick downloads and basic wiring checks. That is separate from the default 7B LoRA lineage.

## Current Run Interpretation


| Run / adapter                                                                                           | Status                                             | Use now?                              | Why                                                                                                |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `20260428-160917_fd9233` / `arena-acceptance` on `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` | ok, `arena 1/1`                                    | Yes for arena smoke/acceptance checks | Latest documented deterministic real-edit gate.                                                    |
| `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428`                                                  | training completed, workflow exit 1 from benchmark | Yes for current arena defaults        | Better validation loss than the 8k chunk pass; still failed one lexical game benchmark task.       |
| `checkpoints/fe-lora-qwen25-coder-7b-chunked-20260428`                                                  | training completed, workflow exit 1 from benchmark | Historical comparison                 | First 7B chunked baseline; useful for A/B notes, not the current default.                          |
| `checkpoints/fe-lora-game-text-20260428`                                                                | ok, `15/15` lexical game benchmark                 | Historical 1.5B-era adapter           | Do not use as a default with 7B arena/model settings unless its base model is explicitly selected. |
| `checkpoints/fe-lora-30m` and descendants                                                               | mixed historical results                           | Historical only                       | Useful for earlier benchmark comparisons; superseded by the 7B lineage for current arena defaults. |


An exit code of `1` on a `train --evaluate` or `full` run can mean training succeeded and the post-train benchmark failed. Check the run row, `RUN.md`, and manifest before treating it as a crashed training job.

## Historical Archive Rules

- Do not delete old rows from `docs/run_history.md`; mark interpretation here instead.
- Keep old commands and outcomes in `docs/SESSION_LOG.md`; it is an audit log, not current guidance.
- Put new "what to run next" guidance in `docs/PROJECT_STATE.md`, `docs/WORKFLOW.md`, or this file, not inside old session entries.
- When a default changes, update `scripts/fe_lineage.py`, `docs/DATA_LAYOUT.md`, `docs/PROJECT_STATE.md`, and this file together.

