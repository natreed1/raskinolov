# fallen-empire-lora — project state

Last verified: **2026-05-18** (Apple Silicon macOS; workspace path `/Users/natreed/fallen-empire-lora`). Current defaults target the **Qwen2.5-Coder-7B** MLX lineage. See `docs/RUNS.md` for current-vs-historical run interpretation and `docs/run_history.md` for the complete append-only run table. **Auxiliary** documentation-agent RAG evals and other cross-cutting benchmarks are indexed in **`docs/SPECIALIZED_RUN_HISTORY.md`** (with **`benchmarks/results/documentation_rag_timeseries.jsonl`** for trend JSONL).

## Purpose

Experiment with **LoRA fine-tuning** on open code models using the Fallen Empire game repository as training text, without bloating the main game repo. See repository `README.md` for rationale.

## Canonical local sites (policy)

Only these three local sites are supported:

1. Documentation site (documentation-agent work + cost tracking): `python scripts/private_dashboard_server.py` (`http://127.0.0.1:8787`)
2. Arena training supervision site (specialists vs frontier/GPT): `python scripts/game_task_arena.py ui` (`http://127.0.0.1:7868`)
3. Router prompt site (agentic routing + specialist behavior): `python scripts/router_chat_gradio.py` (`http://127.0.0.1:7864`)

Legacy UI entrypoints (`chat_gradio`, `human_eval_ui`, `train_ui_gradio`, `landing_page_arena ui`) are deprecated and blocked by default unless explicitly launched with `--allow-legacy-ui`.

## Recent documentation-state updates (2026-05-10)

- Change documentation capture outputs are normalized to a strict header format for readability and RAG consistency:
  - title line
  - `Date: YYYY-MM-DD`
  - summary bullets directly below the date
  - implemented in `scripts/generate_change_documentation_capture.py`
- Formatting expectation is codified in `docs/DOCUMENTATION_AGENT_PRACTICES.md` and guarded by `tests/test_change_documentation_capture_format.py`.
- `docs/SESSION_LOG.md` includes a top recency index entry to surface the latest updates first for quick scan.
- `python scripts/ml_workflow.py loading-screen-dataset` now supports loading benchmark prompt ingestion (`benchmarks/loading_screen_mass_tasks_v1.json`) through `scripts/adapters/build_loading_screen_specialist_dataset.py` so benchmark stress prompts can become supervised training rows.
- `scripts/build_mass_specialist_benchmark_tasks.py` now generates mass benchmark suites for all gameplay specialists (`loading_screen`, `hud_status`, `economy_tooltip`, `combat_risk`, `save_load_api_guard`, `ai_planning_explanation`), and `ml_workflow.py` dataset subcommands can ingest those benchmark prompt files for specialist dataset expansion.
- As of 2026-05-17, all specialist mass benchmark suites are normalized to **30 tasks each** (180 total in aggregate), and `scripts/build_mass_specialist_benchmark_tasks.py` refresh now also rebuilds `benchmarks/specialist_benchmark_tasks.json` from those mass suites for benchmark/ACI alignment.
- `scripts/run_game_benchmark.py` now reports an **Advanced ACI** headline in addition to the legacy capability index, blending weighted task capability + domain-balance + multi-domain mastery from task metadata/category inference.
- `python scripts/ml_workflow.py mock-specialist-pairwise` now generates synthetic pairwise rows (`winner_output` + weak loser baseline) from specialist-tagged benchmark task files, enabling quick bootstrap/refresh corpora for low-data specialists (default output: `benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl`).
- `scripts/model_router.py` defaults the local backend to `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`; this keeps Linux `LOCAL_BACKEND=transformers` smoke/eval runs compatible with the current 7B specialist adapters.
- As of 2026-05-30, stable adapter IDs remain unchanged for checkpoint compatibility, but user-facing specialist roles use measured names from the full crossdomain 121-task baseline: `loading_screen` -> `ui_surface_composer`, `hud_status` -> `ui_state_signals`, `economy_tooltip` -> `resource_ui_projection`, `combat_risk` -> `army_ui_flow`, `save_load_api_guard` -> `state_contract_guard`, and `ai_planning_explanation` -> `crossdomain_state_patch`.
- `economistRL` is a new experimental LoRA adapter lane for RL-scored hard economy mechanics plus common RL/generalist skills. Seed tasks live in `benchmarks/economistRL_tasks_v1.json`; deterministic reward scoring and task append tools live in `scripts/economist_rl_tasks.py`; seed SFT data is built with `python scripts/ml_workflow.py economist-rl-dataset`; training config is `training/economistRL_lora_qwen25_coder_7b.yaml`. The curriculum target is `500` prompts split `70%` economy (`350`) and `30%` generalist/common RL framework skills (`150`).

## Environment


| Item                        | Value                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| OS                          | darwin 24.x (from prior session metadata)                                                        |
| Python baseline             | **3.11+ recommended for new environments**                                                       |
| Current verified local venv | **3.9.6** (`/Users/natreed/fallen-empire-lora/.venv`; legacy working snapshot)                   |
| Reproducibility constraints | `requirements.lock.txt` records exact top-level ML/UI package pins from the verified environment |


### Pinned Python packages (venv)

Recorded via `pip freeze` on 2026-04-23; `evalplus==0.3.1` added on 2026-04-26:

```
mlx==0.29.3
mlx-lm==0.29.1
mlx-metal==0.29.3
huggingface_hub==0.36.2
transformers==4.57.6
numpy==2.0.2
gradio==4.44.1
gradio_client==1.3.0
evalplus==0.3.1
```

Install source: `requirements.txt`; resolver pins include `gradio>=4.44,<5` for the chat UI. For the recorded environment constraints, run `pip install -r requirements.txt -c requirements.lock.txt`.

## Default base model (smoke / LoRA / arena local base)


| Field           | Value                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Hugging Face id | `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`                                                                            |
| Role            | Default instruct **code** model, MLX 4-bit, used for current LoRA training, benchmarks, chat UI, and arena local attempts |


Smaller alternatives (same org): `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` (set `MODEL` env var for scripts or pass `--model`, and use a matching adapter for that base).

## Current Run Status


| Item                                                | Current interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current arena adapter default                       | `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` via `scripts/fe_lineage.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Latest deterministic arena gate                     | `20260428-160917_fd9233` scored `arena 1/1` against the current arena adapter                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Latest arena acceptance runner check (restored entrypoint) | `20260518-041101_2920b1` (custom subset: `loading-screen-polish`, `hud-status-summary`) now runs via restored `scripts/run_arena_acceptance_tests.py`; result **0/2**, ACI **20.0** (`loading-screen-polish`: applied + preview ready but verify fail; `hud-status-summary`: no-apply + verify fail). |
| Arena six-task A/B (progressive) chunk6k            | `20260428-171007_517f3e` (`auto`) vs `20260428-180627_42befd` (`off`): both **1/6** preview passes; ACI **53.4** vs **56.3**. See `**docs/ARENA_PROGRESSION.md`**                                                                                                                                                                                                                                                                                                                                                                                                 |
| Arena six-task A/B (progressive) best-val300        | `20260429-025429_9793ff` (`auto`, **1/6**, ACI **55.2**) vs `20260429-034932_0812a4` (`off`, **0/6**, ACI **47.8**). Adapter `checkpoints/fe-lora-qwen25-coder-7b-best-val300` = iter-**300** weights from `20260428-195934_d451c9`. Off run used worktree `benchmarks/results/arena_worktrees`. Failed quick retry: `20260429-034918_34e0cb` (`PermissionError` on `~/fallen-empire-arena`).                                                                                                                                                                     |
| Latest apply-contract + SFT six-task run            | `20260430-183012_121be1` (`auto`, **3/6**, ACI **68.1**) on `checkpoints/fe-lora-arena-apply-sft`; tiers **100.0 / 35.9 / 78.1** (smoke/std/high). Working attribution: improvement likely comes from combined apply-contract context injection + targeted arena baseline SFT; still below promotion thresholds due to weak worst-task tail.                                                                                                                                                                                                                      |
| Latest reverse-recovery six-task run                | `20260430-231450_487deb` (`auto`, **3/6**, ACI **75.0**) on `checkpoints/fe-lora-arena-reverse-recover` after retraining from the overfit standard-dev adapter on six-task apply-contract baselines. Tiers **100.0 / 45.1 / 85.4**; standard-dev remains the active bottleneck (`hud-status-summary`, `economy-tooltip`).                                                                                                                                                                                                                                         |
| Latest repair-set SFT + six-task check              | Trained `checkpoints/fe-lora-arena-apply-repair-stddev-20260502` on combined repair-set corpus `data/lora/arena_apply_repair_stddev_20260502` (`20260502-190626_aa03bc`, exit 0), then ran six-task acceptance `20260502-192506_b1c69f` (`auto`) and got **0/6**, ACI **52.54**. Applyability improved on most tasks, but final `tsc` remained the gating failure mode.                                                                                                                                                                                           |
| Latest mixed-curriculum + repairs six-task check    | Trained `checkpoints/fe-lora-arena-mixed-repair-20260502` on `data/lora/arena_balanced_with_repairs_20260502` (broad curriculum + boosted HUD/economy repair rows) via `20260502-201848_c45345` (exit 0), then ran six-task acceptance `20260502-211555_ce7aa2` (`auto`) and got **1/6**, ACI **53.33**. `combat-risk-preview` passed; standard-dev still failed with `tsc`/export instability.                                                                                                                                                                   |
| Latest compile-supervision checkpoint sweep + core6 | Trained `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502` on `data/lora/arena_balanced_with_repairs_compile_20260502` (broad+repair corpus plus explicit compile/export repair rows) via `20260502-220731_da066d` (exit 0), then gated checkpoints every 20 iters: `ckpt20` (`std-dev 0/2`, `guard 2/4`), `ckpt40` (`0/2`, `0/4`), `ckpt60` (`0/2`, `0/4`). Core6 confirmation on `ckpt20` (`20260502-234543_08b06c`) reached **2/6**, ACI **61.97** (`loading-screen-polish`, `save-load-api-guard` passed), with standard-dev still `tsc`-limited. |
| Overnight specialist cycle2 (`economy` / HUD / combat) single-task gates | **`economy_tooltip`:** dataset `20260504-035847_b88a85`, train **`20260504-035848_5bfba2`** (exit 0, ~8969 s); arena **`20260504-062823_8b3903`**: **`0/1`** — **apply_ok**, exports_ok, **`tsc` exit 2** both rounds.<br>**`hud_status`:** dataset `20260504-063042_18ea7e`, train **`20260504-063044_ba7a5e`** (exit 0, ~9763 s); arena **`20260504-091330_8c7651`**: **`0/1`** — round0 `tsc` fail + export gap; round1 **`no_applyable_changes`**.<br>**`combat_risk`:** `adapter-datasets` `20260504-091830_32b592`; first train **`20260504-091837_00c86b`** failed (empty **`valid.jsonl`** / **`test.jsonl`** until **`dataset_builder`** fix); rerun train **`20260504-091854_ee9a2d`** (exit 0); arena **`20260504-110442_ee9296`**: **`0/1`** — apply_ok, **`tsc` exit 2** both rounds. **`adapter_registry_v1`** now defaults these three specialists to **`cycle2`** (**shadow**, not promoted on gate pass). |
| Router registry default paths (2026-05-18) | `training/adapter_registry_v1.json` now points **`hud_status`** back to **`checkpoints/adapters/hud_status/cycle3`** (`lineage: hud_status:v3:cycle3`, `promotion_state: champion`) while strict specialist-only dataset guardrails are enforced. `combat_risk` remains champion on `cycle4`; `economy_tooltip`, `save_load_api_guard`, and `ai_planning_explanation` remain shadow-candidate specialists for next targeted promotion loops under strict data policy. |
| Loading-screen specialist lock-down (v2 data)      | Built `data/lora/adapters/loading_screen_specialist` with `scripts/ml_workflow.py loading-screen-dataset` (core loading + UI transfer rows), trained `checkpoints/adapters/loading_screen/cycle2` (`20260503-180149_d2ed85`, 120 iters, exit 0), then evaluated `loading-screen-polish` + transfer set (`20260503-184315_a608df`) at **1/3**: loading-screen passed (`100`), transfer tasks (`hud-status-summary`, `economy-tooltip`) still failed on `tsc`. Registry promotion applied: `training/adapter_registry_v1.json` now maps `loading-screen-polish` to `loading_screen` with `promotion_state: champion` and adapter path `checkpoints/adapters/loading_screen/cycle2`. |
| Documentation/run-analysis specialist (mlx-lab prose) | `python scripts/ml_workflow.py documentation-dataset` builds `data/lora/adapters/documentation_specialist/` (gold draft→canonical pairs; task alias `mlx-lora-docs-normalize`) for this **LoRA lab repo**, not game implementation. Registry **`documentation`** adapter path **`checkpoints/adapters/documentation/cycle3`** (**shadow**). Latest scope-guard dataset: **`20260505-012316_bc96ac`** (21 core rows, 160 train rows). Latest train: **`20260505-012327_c24601`** (exit 0, **40** iters from `cycle2`, final val **1.045**, best val **0.957** at iter 30, test loss **0.781**) wrote **`adapters.safetensors`** (~154MB). Routing eval **`benchmarks/documentation_eval_tasks_v1.json`** (22 prompts) scored **100%** route+adapter locally; regenerate prompts via **`scripts/build_documentation_eval_tasks_v1.py`**; **`python -m unittest discover -s tests`**. Two RAG lanes: **documentation RAG** (`data/rag/documentation_agent_corpus.json`, `run_documentation_agent_benchmark.py`) and **run-analysis RAG** (`data/rag/run_analysis_agent_corpus.json`, `run_run_analysis_agent_benchmark.py` + `build_run_analysis_rag_corpus.py`). |
| Latest 7B chunked train runs                        | Training completed for both Apr 28 7B chunk passes; workflow exit `1` came from one failed post-train lexical game benchmark task, not a training crash                                                                                                                                                                                                                                                                                                                                                                                                           |
| Optional arena SFT corpus                           | `data/arena_task_baselines/*.assistant.txt` + `scripts/build_arena_baseline_dataset.py` → `data/lora/arena_task_baselines/` (messages JSONL for supervised arena outputs). See `data/arena_task_baselines/README.md`.                                                                                                                                                                                                                                                                                                                                             |
| Historical 1.5B adapters                            | Keep for comparison only unless scripts are explicitly pointed at the matching 1.5B base model                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |


Detailed row-by-row run history belongs in `docs/run_history.md`; narrative command context belongs in `docs/SESSION_LOG.md`. **Arena A/B baselines** (same task suite, progressive on vs off) are tracked in `docs/ARENA_PROGRESSION.md`.

## Verified commands

### Lambda Cloud eval launchers

`scripts/launch_lambda_parallel_ablation.py` and `scripts/launch_lambda_council_eval.py` use the Lambda Cloud instance API (`LAMBDA_API_KEY`, `LAMBDA_CLOUD_BASE_URL`; legacy `LAMBDA_API_BASE` is still accepted as a fallback) plus SSH to start remote `tmux` jobs. The parallel ablation launcher retries transient Lambda API failures (`429`, `5xx`) with short backoff, which matters for multi-worker specialist-suite launches. Instances created with `--launch-instances` / `--launch-instance` default to `auto_terminate=1`: the remote eval script calls `/instance-operations/terminate` when the script exits. Reused instances passed with `--instance-ids` / `--instance-id` default to non-terminating unless `--auto-terminate` is provided. Use `--no-auto-terminate` only when intentionally leaving a worker alive for inspection or manual artifact recovery. The remote idle-log watchdog defaults on when auto-termination is on; configure with `--watchdog-idle-minutes` (default `60.0`), `--watchdog-check-minutes` (default `5.0`), `--watchdog`, and `--no-watchdog`.

For new parallel ablation launches, `scripts/launch_lambda_parallel_ablation.py` queries Lambda `GET /instance-types` before launch and uses `regions_with_capacity_available` to avoid regions with no reported capacity. `--instance-type` and `--region` remain the first preference; use repeatable or comma-separated `--fallback-region` and `--fallback-instance-type` to let the launcher try capacity-backed alternatives. If Lambda reports insufficient capacity or returns a partial/wrong-quantity launch, the launcher best-effort terminates any partial instances before trying the next candidate.

Artifact preservation is part of the Lambda lifecycle. Eval runners append rows locally after each completed task, and the launchers set a checkpoint hook so workers collect artifacts every `--artifact-upload-every-steps` (default `5`) or `--artifact-upload-every-minutes` (default `10.0`). Before each checkpoint/final tarball, the remote collector parses `~/cloud-eval-logs/gpu-smi*.csv` into `~/cloud-eval-logs/gpu-telemetry-summary.json` and `~/cloud-eval-logs/gpu-telemetry-summary.md` with sample counts, average/p95/max GPU utilization, memory MiB, power draw, and temperature. Before any cleanup/watchdog termination, the remote worker bundles `~/cloud-eval-logs/`, `~/fallen-empire-lora/benchmarks/results/`, routing registry/config, and eval/router scripts into `~/cloud-eval-artifacts/*.tar.gz`, then runs `--artifact-export-command` / `FE_ARTIFACT_EXPORT_COMMAND` with `FE_ARTIFACT_TARBALL` set. Final termination requires a successful artifact export by default; pass `--allow-terminate-without-artifact-export` only when cost control is more important than preserving recoverable data.

Lambda file system attachment defaults to file system id `5795235886dd4e45a5adcdb5637de9d6` (`Evaluation-Runs`, mount point `/lambda/nfs/Evaluation-Runs`, region `us-west-1`) for `scripts/launch_lambda_parallel_ablation.py`. When the file system is attached, the launcher stages artifact bundles under `/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts` by default and uses the file system's region unless `--region` is explicitly supplied. Controls: `--file-system-id`, `--file-system-name`, `--no-file-system`, and `--artifact-staging-dir`.

Operational note: a 2026-05-28 six-worker specialist-suite smoke initially looked like the file system did not attach because terminated/terminating Lambda instance records report `file_system_names: []`. A follow-up probe launched one `gpu_1x_a10` in `us-west-1` with `file_system_names=["Evaluation-Runs"]`; while active, Lambda reported `file_system_mounts=[{"mount_point": "/lambda/nfs/Evaluation-Runs", "file_system_id": "5795235886dd4e45a5adcdb5637de9d6"}]`, SSH `mountpoint /lambda/nfs/Evaluation-Runs` passed, and a marker file was written. The launcher also checks `mountpoint -q /lambda/nfs/Evaluation-Runs` during bootstrap when a file system is configured, so future runs fail before eval work if the persistent file system is not actually mounted.

For one-task Lambda smoke runs, `scripts/launch_lambda_parallel_ablation.py` supports `--only-cell`, `--max-tasks`, `--task-domain`, and `--single-specialist-adapter-id`. When forcing `--variant single_specialist_local`, the launcher syncs the requested adapter checkpoint directory from `training/adapter_registry_v1.json` after the normal repo sync because bulk `checkpoints/` are excluded from the main rsync.

For pre-training specialist benchmark sweeps, `scripts/launch_lambda_parallel_ablation.py --specialist-suite` creates one Lambda worker per game specialist and forces `single_specialist_local` with the mapped task domain. Supported game specialists are `loading_screen` (`hud_status` domain), `hud_status` (`hud_status`), `economy_tooltip` (`economy`), `combat_risk` (`army_operations`), `save_load_api_guard` (`state_perstitence_integrity`), and `ai_planning_explanation` (`ai_strategy_and_planning`). Use repeatable `--specialist-id` to run a subset. The `documentation` specialist is excluded from this game-patch suite and should be evaluated with the documentation-specific benchmark path.

Measured specialist-role metadata is stored in `training/adapter_registry_v1.json` and `data/routing/*`. Do not rename the stable `adapter_id` values unless checkpoint paths, router labels, training data, and historical artifact readers are migrated together. User-facing names/tags should prefer the measured roles documented in `docs/ROUTER_TASK_TAXONOMY.md`.

`economistRL` is intentionally marked `promotion_state: experimental` in `training/adapter_registry_v1.json`. Use it for economy RL task-bank rollouts and reward scoring before considering default routing promotion; compare against `resource_ui_projection`, `crossdomain_state_patch`, and the base model on the same `benchmarks/economistRL_tasks_v1.json` tasks.

Smoke outcome: a 2026-05-28 `--specialist-suite --max-tasks 1` run launched six `gpu_1x_a10` workers in `us-west-1` with `Evaluation-Runs` attached, passed bootstrap mount checks, synced all specialist adapters (including registry fallback paths for `save_load_api_guard` and `ai_planning_explanation`), started all six remote tmux jobs, confirmed durable cleanup artifact staging under `/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts` while workers were active, and returned Lambda active instance count to `0`.

### Create venv and install

```bash
cd /Users/natreed/fallen-empire-lora
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Smoke test: load + one generation

```bash
source .venv/bin/activate
python scripts/smoke_base_model.py --max-tokens 96
```

**Outcome (2026-04-23):** First full run downloaded 9 files from Hugging Face (~20 s on a fast connection); subsequent load ~0.7 s from cache. Generation ~0.8 s for 96 new tokens. Output contained correct TypeScript `clamp` implementation; end of string may include model-specific control tokens (harmless for smoke; strip in production UI if needed).

### Interactive chat (mlx-lm CLI)

```bash
source .venv/bin/activate
python -m mlx_lm chat --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

CLI entry points also installed in `.venv/bin/` (e.g. `mlx_lm.lora`, `mlx_lm.generate`).

### Gradio browser chat (project UI)

```bash
source .venv/bin/activate
pip install -r requirements.txt   # if gradio not installed yet
python scripts/chat_gradio.py
```

Defaults: bind **[http://127.0.0.1:7860](http://127.0.0.1:7860)**, same base model as smoke script, `concurrency_limit=1` (one generation at a time). Uses `stream_generate` and yields cumulative text for token streaming. Gradio title and default system prompt name the assistant **Albert**.

Useful flags: `--port`, `--model`, `--adapter-path` (LoRA dir when you have one), `--max-tokens`, `--temp`, `--share` (temporary public Gradio URL). Environment mirrors: `MODEL`, `ADAPTER_PATH`, `SYSTEM_PROMPT`, `MAX_TOKENS`, `TEMP`.

**Outcome (2026-04-23):** `--help` and import-time test for `_history_to_messages` succeeded; full browser run not re-logged here (same model stack as smoke).

### Router prompt compare lab (specialist router vs frontier + grading)

`scripts/router_chat_gradio.py` is now the single prompt-comparison site for router evaluation: each prompt runs a **specialist routed local lane** (MLX + registry LoRA from `training/adapter_registry_v1.json`) against a **frontier API lane** (`OpenAICompatibleBackend`, default model from `FRONTIER_MODEL`). The UI shows routed adapter metadata and both outputs side by side, then saves human grading in the same page.

Specialist lane still uses policy+adapter metadata (`adapter_id`, confidence, ambiguity, risk/complexity) and supports **Auto** vs **Codebase OSS** (`force_route='local'`) plus optional **LoRA adapter lock** override.

`router_chat_gradio.py` now auto-loads repo-root `.env` on startup (without overriding already exported shell variables), so `OPENAI_API_KEY` / `FRONTIER_API_KEY` and related router env defaults can be provided via local `.env`.

```bash
source .venv/bin/activate
export ROUTER_CHAT_DEFAULT_BACKBONE=codebase_oss   # optional UI default (or `auto`)
export ROUTER_CHAT_DEFAULT_ADAPTER_LOCK=auto      # registry id overrides classifier when not `auto`
export FRONTIER_MODEL=gpt-4o-mini                 # or your preferred frontier id
python scripts/router_chat_gradio.py
```

Regression guardrails: **`tests/test_router_backbone_controls.py`** (policy parity for forced-local vs security/long-prompt escalation).

Specialist lane now supports optional **multi-agent subtask orchestration** for one prompt:

- split into primary + secondary adapter subtasks from router metadata,
- run both subtasks with their resolved adapter weights,
- merge into one final answer for the same prompt.

Runtime knobs:

- `ROUTER_CHAT_MULTI_AGENT_LANE` (`--multi-agent-specialist-lane`)
- `ROUTER_CHAT_MULTI_AGENT_SECONDARY_MIN_CONFIDENCE` (`--multi-agent-secondary-min-confidence`)
- `ROUTER_CHAT_MULTI_AGENT_MERGE_MAX_TOKENS` (`--multi-agent-merge-max-tokens`)
- `ROUTER_CHAT_COUNCIL_LANE` (`--council-specialist-lane`) for profile+specialist council execution

When `ROUTER_COUNCIL_ENABLED=1`, the specialist lane can run a council execution path:
- 3 generalist profiles (`wide_compressed`, `precise_short`, `sliding_window`),
- top specialists from active roster metadata in `data/routing/council_roster_v1.json`,
- optional roster-defined personalities per selected specialist (same base adapter, different trait profiles) via `ROUTER_COUNCIL_SPECIALIST_PERSONALITY_VARIANTS`,
- blind equal-prior adjudication metadata (confidence/disagreement/escalation candidate),
- bounded iterative debate rounds with caps:
  - policy cap: `ROUTER_COUNCIL_DEBATE_MAX_ROUNDS`
  - runtime hard cap: `ROUTER_CHAT_COUNCIL_DEBATE_MAX_ROUNDS` / `--council-debate-max-rounds`

Per-expert council knobs are configurable in roster-level `traits` and `personalities` in `data/routing/council_roster_v1.json`:
- `assertiveness`, `verbosity`, `risk_tolerance`, `creativity`, `skepticism`, `decisiveness`,
- `personalities` is a list of named possible expert voices, each with `state`, direct `traits` values, optional `trait_deltas`, and offline/online outcome metrics,
- carried into council participant metadata + prompt shaping ("how to voice ideas"),
- used only for inference-time participant prompting and expert interaction; adjudication assigns blind ids, scores answer content with a deterministic equal-prior rubric, and does not give router-selected specialists or personality traits a winner-selection bonus. `task_outcome_score` is neutral (`0.5`) in council traces unless a future verifier writes measured outcomes.

Council promotion/elimination is now two-level:
- expert-level states: `candidate`, `active`, `probation`, `demoted`,
- personality-level states: `candidate`, `active`, `cooldown`, `retired`,
- `scripts/router_promotion_gate.py` updates expert scores from adapter metrics and personality scores from selected-personality benchmark rows or council conversation winners,
- retired personalities are filtered out of planning while the parent expert can remain active.

Personality optimization is a council interaction policy, not separate domain-knowledge training:
- `ROUTER_COUNCIL_PERSONALITY_SELECTION_POLICY=bandit` ranks personality options by offline/online correctness with an exploration bonus,
- `ROUTER_COUNCIL_PERSONALITY_EXPLORATION_RATE` defaults to `0.15` and raises/lowers under-sampled voice exploration,
- `scripts/router_promotion_gate.py` emits `personality_credit` with attempts, wins, top-2 proxy rate, correctness, and escalation-help counts,
- `scripts/run_council_conversation_eval.py --selective-generation` spends generation only on high-risk, ambiguous, or low-confidence tasks before full training.

Router specialist lane interactions continue to log to `benchmarks/results/router_chat_interactions.jsonl` by default (override with `ROUTER_CHAT_LOG_JSONL` or `--interaction-log-jsonl`). Prompt compare runs + saved grades append to `benchmarks/results/router_compare_feedback.jsonl` (override `--compare-log-jsonl`).

As of 2026-05-15, the optional **post-generation bug-check loop** (default on) runs on both specialist and frontier lanes and now surfaces loop telemetry in the UI status (rounds run + whether output changed). Flags remain: **`--bug-check-loop`** / **`--no-bug-check-loop`**, **`--bug-check-rounds`** (default `1`), **`--bug-check-max-tokens`**, and **`--bug-check-system-prompt`**. Env mirrors: **`ROUTER_CHAT_BUG_CHECK_LOOP`**, **`ROUTER_CHAT_BUG_CHECK_ROUNDS`**, **`ROUTER_CHAT_BUG_CHECK_MAX_TOKENS`**, **`ROUTER_CHAT_BUG_CHECK_SYSTEM_PROMPT`**.

### Terminal lab runner + static optimization page

Prefer **non-interactive** orchestration (`scripts/fe_ml_lab_runner.py`) for the recurring “kick off MLX training smoke / full workflow” flows so Composer sessions stay short. Runner events land in **`lab_dashboard/agent_events.jsonl`** (optional heuristic `FE_ML_LAB_SPARED_USD`). Rebuild the deployable KPI bundle with **`python scripts/build_lab_optimization_dashboard.py`** → **`lab_dashboard/index.html`** (+ committed `docs/run_history.md`; optional manual **`lab_dashboard/cursor_usage.jsonl`** ledger; optional **`--cursor-usage`** / **`--agent-events`** overrides for scripted builds). Deploy instructions: **`lab_dashboard/README.md`**. Lightweight MLX Cursor skill **`.cursor/skills/fe-mlx-lab/SKILL.md`** uses **`disable-model-invocation: true`**; **`@fe-mlx-lab`** on MLX/router tasks so unrelated sessions skip that context. Regression tests **`python3 -m unittest discover -s tests -p test_fe_ml_lab_tools.py`**. Optional hooks: **Cursor `stop`** (**`FE_LAB_CURSOR_HOOK_APPEND=1`**) appends **`lab_dashboard/cursor_hook_events.jsonl`**; **`afterShellExecution`** autodoc (enabled in `.cursor/hooks.json`) appends **`lab_dashboard/shell_command_events.jsonl`** and compact test-run rows in **`docs/SPECIALIZED_RUN_HISTORY.md`** with git-change snapshots (`m/u/d` + touched paths). Set **`FE_LAB_REMOTE_INGEST_URL`** and **`FE_LAB_REMOTE_INGEST_TOKEN`** to mirror those events to a private hosted dashboard (`scripts/private_dashboard_server.py`; see **`docs/PRIVATE_DASHBOARD_DEPLOY.md`**). Hooks stay lightweight (no MLX training inside hooks).

## Game-oriented model benchmark + evolution


| Item                           | Location                                                                                                                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task definitions (specialist benchmark) | `benchmarks/specialist_benchmark_tasks.json` (aggregate suite generated from all six 30-task specialist mass files; use repeated `--specialist <id>` filters)                                                                 |
| Arena capability index         | `scripts/arena_capability_index.py` + `scripts/run_arena_acceptance_tests.py` / `ml_workflow.py arena-acceptance` (deterministic apply/typecheck/export/preview task-completion score) |
| Execution benchmark            | `scripts/run_evalplus_benchmark.py` (EvalPlus HumanEval+/MBPP+; use `ml_workflow.py evalplus`)                                                                                         |
| Routing benchmark              | `benchmarks/task_routing_tasks.json` + `scripts/run_routing_benchmark.py` (`--mode route|adapter|both`; adapter-first Router V2 metadata + summary JSON)                                  |
| Specialist routing eval sets     | `data/routing/*_eval_prompts_v*.jsonl` (+ `benchmarks/*_eval_tasks_v*.json`; loading_screen / save_load / economy_tooltip / **documentation** v1), mixed suite `benchmarks/mixed_routing_eval_v1.json` (incl. docs shard; + `scripts/build_mixed_routing_eval_v1.py`)                                              |
| Live routing prompt lab        | `scripts/routing_prompt_lab.py` + `benchmarks/results/routing_prompt_lab/*.jsonl` (manual prompt intake with predicted adapter/route lineage rows)                                      |
| Routing training dataset build | `scripts/build_routing_training_dataset.py` + `data/lora/routing_classifier/<dataset_version>/` + `docs/ROUTING_DATASET_CONTRACT.md`                                                     |
| Routing classifier train       | `scripts/train_routing_classifier.py` + `training/router_classifier_v1/` + `docs/ROUTER_CLASSIFIER_ARTIFACT_CONTRACT.md`                                                                  |
| Adapter registry + taxonomy    | `training/adapter_registry_v1.json` + `scripts/adapters/taxonomy.py` + `docs/adapter_taxonomy_v1.md`                                                                                    |
| Per-adapter datasets           | `scripts/adapters/dataset_builder.py` + `data/lora/adapters/<adapter_id>/` + `docs/dataset_contract_v1.md`                                                                             |
| Adapter gates + drift          | `scripts/adapters/gates.py` + `scripts/adapters/drift_monitor.py` + `docs/gate_policy_v1.md`                                                                                            |
| Control plane scaffold         | `scripts/control_plane/*.py` + `ml_workflow.py control-plane-schedule`                                                                                                                   |
| Ops report/dashboard outputs   | `scripts/build_multi_adapter_report.py` + `ml_workflow.py multi-adapter-report` (`benchmarks/results/multi_adapter_dashboard.json`, `benchmarks/results/multi_adapter_report.md`)     |
| Visual human arena             | `scripts/landing_page_arena.py` + `benchmarks/landing_page_brief.md` (standalone landing-page trials; no game repo edits)                                                              |
| Game task arena                | `scripts/game_task_arena.py` + `benchmarks/game_task_arena_examples.json` (disposable worktree/copy trials against the game repo)                                                      |
| Shared tier / mutation helpers | `scripts/benchmark_evolution_lib.py`                                                                                                                                                   |
| Runner                         | `scripts/run_game_benchmark.py` (`--specialist <id>` optional; defaults to all specialist benchmark tasks)                                                                             |
| Final-system ablation runner   | `scripts/run_final_mass_testing_system.py` (compiled final task-bank manifest -> arena generate/apply/verify evaluation for `advanced_router_with_specialists`, `qwen_7_5b_only`, `gpt_5_5_only`; prompt controls include style-RAG plus context modes `off` and `max_potential`) |
| Season evolution               | `scripts/evolve_benchmark_seasons.py` + `training/evolution_config.json`                                                                                                               |
| Training notes                 | `training/README.md`                                                                                                                                                                   |
| Notes                          | `benchmarks/README.md`                                                                                                                                                                 |


`run_game_benchmark.py` scoring is heuristic (substring rules), intended to regress specialist prompt behavior—not to replace game unit tests. Arena scoring is split into **Smoke Test** (low complexity), **Standard Dev Benchmark** (low-medium/medium), and **High-Reasoning Architecture Benchmark** (medium-high/high). The deterministic arena score is the preferred non-human signal for real game-edit completion: it scores apply success, TypeScript checks, export preservation, preview readiness, retry count, token/time efficiency, and task complexity from arena artifacts. Low-only runs are capped as smoke tests so a loading-screen pass is not treated as full arena capability. As of 2026-05-01, arena capability output also includes a **diversified** headline index (`diversified_arena_capability_index`) that blends task-type-balanced performance, worst-tail robustness, and task-catalog coverage.

**Dashboard + gates:** Regenerate `**benchmarks/results/arena_dashboard.html`** with `**python scripts/ml_workflow.py arena-dashboard**` after local six-task runs; gate adapter promotion with `**python scripts/arena_promotion_gate.py …/arena_capability.json**` (see `**docs/ARENA_ROADMAP.md**`). Promotion gating now enforces minimum evaluated-task count and coverage ratio by default (`--min-tasks 6`, `--min-coverage-ratio 1.0`) and can require suite metadata (`--require-suite core6|all|custom`). Acceptance summaries mark explicit `--task-id` subsets as `suite: "custom"` so partial checks are not mislabeled as core-six runs. Progressive `**auto`/off** norms for keyed adapters live in `**scripts/fe_lineage.py`**.

**Router V3 council adapter-first baseline (2026-05-25):** `scripts/model_router.py` defaults to manifold/similarity adapter selection (`ROUTER_ADAPTER_SELECTION_MODE=similarity`) and route derivation as before, with optional council metadata (`ROUTER_COUNCIL_ENABLED=1`). Router V3 emits `council_plan`, `council_disagreement`, and `council_escalation_candidate` while preserving existing route/adapter fields. Policy tag now reports `router_policy_v3_council_adapter_first`.

**Hierarchical mixed-task stage (v1):** optional two-stage coarse-to-fine selection is enabled by default (`ROUTER_HIERARCHICAL_ROUTING_ENABLED=1`):

- Stage 1 (taxonomy/rule coarse bucket): choose candidate adapters by mixed-intent keyword families.
- Stage 2 (manifold rerank): run cosine similarity within candidate adapters first; fallback to global similarity if coarse evidence is weak.
- Runtime metadata now includes `secondary_adapter_id`, `secondary_confidence`, `coarse_bucket`, `candidate_adapters`, and `hierarchy_stage`.
- Key knobs: `ROUTER_HIERARCHY_CANDIDATE_WIDTH`, `ROUTER_HIERARCHY_MIN_COARSE_HITS`, `ROUTER_SIMILARITY_MIN_SCORE`, `ROUTER_SIMILARITY_MIN_MARGIN`.


Routing test+train pipeline now supports dual-label + council evaluation and dataset generation:

- `scripts/run_routing_benchmark.py --mode route|adapter|both|council` scores baseline labels and council metrics (specialist selection recall, escalation accuracy, quality proxy) and emits baseline-vs-council comparisons in summary JSON.
- `scripts/routing_prompt_lab.py` captures live prompts (vibe-coding style) with predicted adapter/route and council metadata (`council_plan`, disagreement, escalation candidate) into JSONL.
- `scripts/build_routing_training_dataset.py` merges benchmark/live/curated labels into deterministic train/valid/test splits with `manifest.json`.
- `scripts/build_council_training_dataset.py` builds deterministic council-orchestration train/valid/test splits from router chat and prompt-lab logs.
- `scripts/train_routing_classifier.py` trains OSS classifier artifacts (vocab + weights + manifest) from routing dataset splits.
- `scripts/router_promotion_gate.py` enforces combined offline+online promotion thresholds (route/adapter/council metrics, high-risk misroutes, optional online task outcome gate) and can apply roster transitions when `--roster-json` is provided.
- `scripts/optimize_manifold_routing.py` runs score/margin/unknown-threshold grid search, emits merged benchmark summaries, and evaluates gate pass/fail for each manifold config.
- `scripts/init_council_roster.py` bootstraps/refreshes `data/routing/council_roster_v1.json` with default per-expert traits (`assertiveness`, `verbosity`, `risk_tolerance`, `creativity`, `skepticism`, `decisiveness`).
- `scripts/ml_workflow.py` includes `routing-prompt-lab`, `routing-benchmark`, `routing-dataset`, `council-dataset`, `routing-train`, `routing-gate`, and `council-roster-init` so these runs follow normal workflow artifact + `docs/run_history.md` audit paths.

Game task arena task specs now support `preview_path`. Preview URLs are composed as `http://127.0.0.1:<port><preview_path>` and previews are marked `ready` only after the URL responds. The current standardized visual tasks point at game-owned `/test-env/...` sandbox routes in `/Users/natreed/fallen-empire`, including `/test-env/loading-screen` and `/test-env/combat-risk-preview`, so human review opens directly on deterministic test screens.

Preview startup uses `next dev -H 127.0.0.1 -p {port}` rather than the package `npm run dev` wrapper. This avoids the old disposable-worktree failure where `npm run dev -- --host ... --port ...` resolved to `next: command not found`; the arena now reuses source checkout dependencies through `PATH`/`NODE_PATH` and a best-effort `node_modules` symlink. Generation artifacts include `generation_metrics.json` plus manifest fields for elapsed seconds and token counts.

Progressive context is now enabled by default for non-low-complexity arena generation (`--progressive-context auto`). The model first requests exact files it needs, the harness appends bounded targeted snippets, and logs are written to `logs/progressive_context.json` / `.md`. This is intended to move HUD/economy/combat/architecture tasks away from one static oversized context packet.

As of 2026-05-01, standard-dev tasks (`hud-status-summary`, `economy-tooltip`) use tighter retrieval controls in `scripts/game_task_arena.py`: noisy file exclusion (`.bak`, `.DS_Store`), task-specific priority/allowlisted context paths, deduped packed files with a small cap, and larger reserved body budget for code context (with `package.json` omitted for that path).

Applyability contract is now codified in `docs/GAME_ARENA_APPLY_CONTRACT.md` and injected into arena model packets plus pairwise/baseline SFT prompts. For targeted `no_applyable_changes` training loops, use `build_game_task_pairwise_dataset.py --focus-apply-failures` (or `ml_workflow.py arena-gate-train --rebuild-dataset --focus-apply-failures`).

```bash
source .venv/bin/activate
python scripts/run_game_benchmark.py
python scripts/run_game_benchmark.py --specialist combat_risk --specialist ai_planning_explanation
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 5
python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 --task-id loading-screen-polish
python scripts/run_routing_benchmark.py
python scripts/landing_page_arena.py ui
python scripts/game_task_arena.py ui
python scripts/run_game_benchmark.py --tier A
python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json
```

**Outcome (2026-04-24):** On `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` with greedy `temp=0` and default `--max-tokens` 512, legacy benchmark suites reached full pass on that snapshot. (2026-04-23) Evolved task files are experimental; cap `min_chars` at **320** after mutation/tier scaling to stay decodable.

### Visual arena parser / UI notes

As of 2026-04-26, `scripts/landing_page_arena.py ui` presents a single IDE-style workflow: one build prompt fans out to Local and Frontier lanes, generated code appears in editable `index.html` / `styles.css` / `script.js` panes, previews render side by side in sandboxed iframes, and grading writes paired comparison/training rows. Parser behavior is intentionally strict to safe static filenames but tolerant of common model response formats (`path=`, `filename=`, language-only fences, bare fenced filenames, labeled sections, and raw HTML). Unparsed output is preserved in `model_output.md` / `parse_error.html` without replacing `index.html`.

## Dataset export + hygiene

- Script: `scripts/export_repo_for_training.py`
- Default source: `~/fallen-empire` unless `SOURCE_REPO` is set
- Output: `data/raw/repo_text.jsonl` (gitignored under `data/raw/`)
- **Hygiene:** max file size **400 KB** (override with `EXPORT_MAX_FILE_BYTES`), extra skip dirs (`.turbo`, `credentials`, …), skip credential-like suffixes (`.pem`, `.key`, …), path substring rules (`secret`, `/.env`, …), NUL / control-char heuristic for “binary-ish” text, and **regex redaction** for common API keys / PEM blocks / obvious GitHub+Slack token shapes.
- Script: `scripts/build_github_ts_dataset.py` (optional TypeScript corpus intake from GitHub repos/search with permissive-license + strict-tsconfig + path filters; default output `data/lora/qwen25-coder-7b/github_ts_compile_safe/`).

## Human evaluation UI

- Script: `scripts/human_eval_ui.py` (default **[http://127.0.0.1:7861](http://127.0.0.1:7861)**)
- Appends JSONL rows to `benchmarks/results/human_eval.jsonl` (gitignored under `benchmarks/results/`).

## Landing page trial arena

- Script: `scripts/landing_page_arena.py` (Gradio default **[http://127.0.0.1:7863](http://127.0.0.1:7863)**)
- Shared context: `benchmarks/landing_page_brief.md`
- Trial artifacts: `benchmarks/results/landing_page_trials/<trial_id>/` plus JSONL indexes under `benchmarks/results/`
- Purpose: compare local LoRA/base/frontier attempts by viewing standalone static pages in a browser, then record visual/product ratings. This is intentionally separate from the game repo and does not mutate `/Users/natreed/fallen-empire`.
- UI: single IDE-style arena. One prompt splits into Local and Frontier attempts, shows editable `index.html` / `styles.css` / `script.js` panes for both, previews current code, then saves grades/training data. **Local** calls `LocalMlxBackend`; **Frontier** can call an OpenAI-compatible API via `FRONTIER_API_KEY` / `OPENAI_API_KEY`, `FRONTIER_MODEL`, and optional `FRONTIER_API_BASE_URL`, or import pasted Cursor/frontier output.
- Token budget: arena Local + Frontier split defaults to **8192** max tokens. If either lane still truncates static-site output, escalate deliberately to **12000**, then **16000** only for larger landing-page trials.
- Local API config: `scripts/landing_page_arena.py` loads ignored repo-root `.env` files; `.env.example` documents keys. Do not commit `.env`.
- Training data: comparison records append to `benchmarks/results/landing_page_comparisons.jsonl` and `benchmarks/results/landing_page_training_data.jsonl`, including task, brief, winner, scores, notes, and generated files.
- Default local adapter: `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` (shared via `scripts/fe_lineage.py`; do not use 1.5B adapters with the 7B base).

## Training UI (Gradio)

- Script: `scripts/train_ui_gradio.py` (default **[http://127.0.0.1:7862](http://127.0.0.1:7862)**)
- Convenience panel for export + dataset build + `mlx_lm.lora` with live logs; for committed run documentation use `scripts/ml_workflow.py` (see `docs/WORKFLOW.md`).

## Game-side ML cohort tests

- **Game repo:** `scripts/ml-lora-cohort-guard.ts` — `npm run test:ml-cohort`
- **ML repo wrapper:** `scripts/run_game_ml_tests.sh` (honours `SOURCE_REPO`, default `~/fallen-empire`)

## Game task arena

- Script: `scripts/game_task_arena.py` (Gradio default **[http://127.0.0.1:7868](http://127.0.0.1:7868)**)
- Task specs: `benchmarks/game_task_arena_examples.json`
- Bug-fix RAG checklist corpus: `data/rag/bug_fix_agent_corpus.json` (primary source `docs/BUG_FIX_COMMON_ERRORS.md`) for common patch-failure review prompts.
- Standardized task coverage: low-complexity loading/start screen polish, visible HUD/status UI, economy tooltip, combat risk preview, save/load/API guard, and AI planning rationale. Tasks are designed to span UI/backend/combat/economics/planning and prefer changes that are easy to locate quickly in a preview.
- Grading: each standardized task defines five task-specific grading criteria in `benchmarks/game_task_arena_examples.json`; the UI updates local/frontier slider labels when the selected task changes and saved ratings include `rubric_labels` / `rubric_scores`.
- Default source repo: `SOURCE_REPO` or `/Users/natreed/fallen-empire`
- Default disposable tree root: `GAME_ARENA_ROOT` or `~/fallen-empire-arena`
- Default local adapter: `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` (shared via `scripts/fe_lineage.py`; avoids loading old 1.5B adapters against the 7B base)
- Results: `benchmarks/results/game_task_trials/<trial_id>/`, append-only `benchmarks/results/game_task_index.jsonl`, training rows in `benchmarks/results/game_task_training_data.jsonl`, summaries under `benchmarks/results/game_task_reports/`
- Purpose: evaluate local/frontier attempts on real game-code tasks in disposable worktrees/copies, run fixed verification commands, record diffs/logs/previews/ratings, compare adapters over time, and clean up worktrees without deleting indexed artifacts.
- UI: standardized task selector is the primary control; `Source repo`, `Worktree root`, and `Base ref` are advanced settings. Main flow creates worktrees, generates/applies both model attempts, starts playable Local/Frontier preview links, records grades, and optionally deletes disposable worktrees. Packet text, manifests, verification output, and reports are hidden in a details drawer.
- Safety: never edit the main game checkout; never run model-suggested shell commands; only run task-spec verification commands; reject file writes outside allowed path globs.
- Generation timeout: the UI runs Local and Frontier generation in bounded subprocesses (default **360s**, override with `GAME_TASK_ARENA_GENERATION_TIMEOUT_S` or the UI field). Timeouts are recorded as `generation_timeout` instead of leaving the UI apparently running forever.
- Post-generation bug-check loop (2026-05-10): `game_task_arena.py generate` supports `--bug-check-loop` / `--no-bug-check-loop`, `--bug-check-rounds`, `--bug-check-max-tokens`, and `--bug-check-system-prompt` (env mirrors `GAME_TASK_ARENA_BUG_CHECK_*`) to run one or more patch-review repair passes before apply/verify; generation metrics record bug-check fields.
- Bug-check RAG retrieval (2026-05-10): bug-check rounds can pull targeted context from `data/rag/bug_fix_agent_corpus.json` using `--bug-check-rag` / `--no-bug-check-rag`, `--bug-check-rag-corpus`, `--bug-check-rag-top-k`, and `--bug-check-rag-max-chars` (env mirrors `GAME_TASK_ARENA_BUG_CHECK_RAG*`); metrics include retrieved-hit totals.

## Known issues / API notes

1. **urllib3 / LibreSSL:** At import time, urllib3 may warn that LibreSSL is older than OpenSSL 1.1.1; training/inference still proceeded.
2. `**mlx-lm` 0.29.1 generation:** `generate()` forwards kwargs to `generate_step()`, which does **not** accept `temp=`. Use `sampler=make_sampler(temp=..., top_p=1.0)` for non-zero temperature; `temp=0` omits sampler (greedy argmax). Implemented in `scripts/smoke_base_model.py` and `scripts/chat_gradio.py`.
3. **EvalPlus dataset availability:** First `ml_workflow.py evalplus` verification attempts on 2026-04-26 failed before model load because GitHub returned HTTP 502 for `HumanEvalPlus.jsonl.gz`. Retry later or set `HUMANEVAL_OVERRIDE_PATH` / `MBPP_OVERRIDE_PATH` to a local EvalPlus JSONL.
4. **Qwen2.5 chat EOS under mlx_lm streaming:** Wrapped tokenizers expose `<|endoftext|>` (151643) in `eos_token_ids` only. Assistant turns terminate with `<|im_end|>` (**151645**); if omitted, ``stream_generate`` never stops on IM-end and chats show repeated sentinel strings (looks like a “failed” docs LoRA). After every `mlx_lm.load`, runners call **`scripts/mlx_qwen_stop_tokens.register_qwen_coder_instruct_extra_stops`** (Gradio chats, **`router_chat_gradio.py`**, **`human_eval_ui.py`**, **`LocalMlxBackend`**).
5. **Specialist adapter model mismatch:** For local arena generation with specialist adapters, resolve base model from adapter metadata (`adapter_config.json`) or pass `--local-model` / `$MODEL`. Mismatched base+adapter pairs can throw LoRA matmul shape errors.
## Training / LoRA (wired)


| Piece                                | Current role                                                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/ml_workflow.py`             | Preferred repeatable entry point; re-execs into `.venv/bin/python`, writes ignored per-run artifacts, and appends `docs/run_history.md`.    |
| `docs/WORKFLOW.md`                   | Command reference for workflow subcommands and examples.                                                                                    |
| `docs/RUNS.md`                       | Current-vs-historical run interpretation and adapter recommendations.                                                                       |
| `docs/run_history.md`                | Append-only committed run index.                                                                                                            |
| `scripts/build_lora_dataset.py`      | Builds the default 7B `game_text` splits; layout details live in `docs/DATA_LAYOUT.md` and chunking details in `docs/CHUNKED_GAME_TEXT.md`. |
| `training/lora_qwen25_coder_7b.yaml` | Current 7B LoRA config. The old `training/lora_qwen_coder.yaml` name is historical only.                                                    |
| `scripts/train_lora.sh`              | Legacy convenience wrapper; prefer `scripts/ml_workflow.py` for documented runs.                                                            |


For historical run outcomes, use `docs/RUNS.md` first, then `docs/run_history.md` for the full table and `docs/SESSION_LOG.md` for command narrative. This file should stay focused on current defaults and durable workflow behavior.

## Git / large artifacts

`.gitignore` excludes `.venv/`, `models/`, `checkpoints/`, `data/raw/`, `data/lora/`, `benchmarks/results/`, common weight extensions. Do not commit weights or raw exports.
