# Data Pipeline Map

This map separates active training paths from experiment archives and generated caches. Use it before starting new data work so new artifacts land in the right lane.

## Orchestration Spine

`scripts/ml_workflow.py` is the primary wrapper for repeatable local runs. Normal subcommands create `benchmarks/results/runs/<run_id>/` with a `manifest.json`, `RUN.md`, logs, and a `docs/run_history.md` row. Use the underlying scripts directly only for focused maintenance, one-off inspection, or Lambda worker entrypoints that are not normal local workflow runs.

## Active Data Lanes

| Lane | Canonical inputs | Builders / runners | Primary outputs | Status |
| --- | --- | --- | --- | --- |
| Base code LoRA | Fallen Empire checkout via `SOURCE_REPO` | `scripts/export_repo_for_training.py`, `scripts/build_lora_dataset.py`, `scripts/ml_workflow.py prepare/full` | `data/raw/repo_text.jsonl`, `data/lora/game_text/`, `data/lora/qwen25-coder-7b/game_text*` | Active |
| Specialist adapter SFT | Pairwise winners, curated baselines, mass benchmark prompts, shared anchors | `scripts/ml_workflow.py *-dataset`, `scripts/adapters/build_*_specialist_dataset.py` | `data/lora/adapters/<specialist>/` | Active |
| Documentation specialist SFT | Curated ML-lab documentation examples | `scripts/build_documentation_specialist_dataset.py`, `scripts/ml_workflow.py documentation-dataset` | `data/lora/adapters/documentation_specialist/` | Active |
| Router classifier | Benchmark routing tasks, curated router cases, accepted prompt-lab rows | `scripts/build_routing_training_dataset.py`, `scripts/train_routing_classifier.py`, `scripts/ml_workflow.py routing-dataset/routing-train` | `data/lora/routing_classifier/<dataset_version>/`, `training/router_classifier_v*/` | Active |
| Council orchestration | Router chat and prompt-lab interaction logs | `scripts/build_council_training_dataset.py`, `scripts/ml_workflow.py council-dataset` | `data/lora/council_orchestration/<dataset_version>/` | Active |
| RAG corpora | Docs, run manifests, benchmark failures, router corpora | `scripts/build_current_rag_dataset.py`, `scripts/build_run_analysis_rag_corpus.py`, RAG benchmark runners | `data/rag/*.json` | Active |
| economistRL execution/RL | Execution-tagged task bank, sandbox starters, rollout evidence | `scripts/lambda/run_economist_rl_lambda_cycle.py`, `scripts/lambda/run_economist_rl_split_workers.py`, `scripts/economist_rl_reward_engine.py` | `benchmarks/results/economistRL/`, `checkpoints/adapters/economistRL/rl_pass_*` | Experimental active |

## Archive And Experiment Lanes

| Path family | Meaning | Guidance |
| --- | --- | --- |
| `data/lora/arena_*` | Arena repair, balanced-curriculum, compile-supervision, and apply-contract experiments | Keep for provenance. Do not treat as the current default unless a run or registry entry points to it. |
| `data/lora/game_task_pairwise*` | Pairwise chat SFT and apply-focus experiments for arena gate training | Active only when intentionally running `arena-gate-train`. |
| `data/lora/adapters/*_mock_aug_*` | Synthetic/mock augmentation datasets for low-data specialists | Experimental. Compare against non-mock specialist datasets before promotion. |
| `data/lora/adapters_messages_only/` | Filtered message-only adapter experiments | Historical/experimental. Not dashboard-canonical. |
| `data/lora/adapters/_deprecated_economistRL_stub_sft/` | Retired economistRL seed SFT data | Deprecated. Do not rebuild or train from it. |
| `data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke/` | External compile-safe smoke dataset | Experimental smoke lane. |

## Registry And Cache Files

| File | Role | Source of truth? |
| --- | --- | --- |
| `training/adapter_registry_v1.json` | Adapter identity, stable IDs, promotion state, active checkpoint paths, routing metadata | Yes for adapter selection and promotion state. |
| `data/lora/*/manifest.json` | Dataset provenance, inputs, source counts, caps, and split counts | Yes for that dataset directory. |
| `data/training_dashboard/training_data_catalog.json` | Cached dashboard index of immediate `data/lora/adapters/*` children | No. Regenerate with `python scripts/build_training_data_dashboard_cache.py` after dataset changes. |
| `docs/run_history.md` | Append-only normal workflow run index | Yes for workflow run history, not dataset status. |
| `docs/SPECIALIZED_RUN_HISTORY.md` | Auxiliary evaluator and cross-system run index | Yes for auxiliary run history, not dataset status. |

## Current economistRL Rule

`economist-rl-dataset` and `scripts/adapters/build_economist_rl_dataset.py` are retired stub-SFT paths. economistRL now initializes rollouts from the configured init adapter, gathers execution evidence, scores continuous reward, and trains PPO candidates through the Lambda/local split-worker path. Task-bank conversion and tagging still matter, but they feed rollout/evidence, not seed SFT.

Each current split-worker run must write `benchmarks/results/economistRL/runs/run_NNNN_<UTC>/RUN_MANIFEST.json` with `schema_version: economist_rl_pipeline_run_v1`. Lambda extracts must include the same `run_id`, `run_number`, and `run_date_utc` in `EXTRACT_MANIFEST.json` and `extracts/index.jsonl`.

## Where New Work Should Land

- New specialist supervised datasets: `data/lora/adapters/<adapter_id>[_variant]/` with `train.jsonl`, `valid.jsonl`, `test.jsonl`, and `manifest.json`.
- New router labels: `data/routing/*.jsonl` for curated cases, then `data/lora/routing_classifier/<dataset_version>/`.
- New council labels: prompt-lab/router-chat logs, then `data/lora/council_orchestration/<dataset_version>/`.
- New RAG corpora: `data/rag/<agent_or_purpose>_corpus.json`.
- New economistRL rollouts, scores, queues, and PPO manifests: `benchmarks/results/economistRL/`.
- New economistRL run-level manifests: `benchmarks/results/economistRL/runs/run_NNNN_<UTC>/`.
- Dashboard updates: regenerate `data/training_dashboard/training_data_catalog.json`; do not edit it by hand.
