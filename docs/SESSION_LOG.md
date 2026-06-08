# Session log (append-only)

Newest entries at the **top**.

---

## 2026-06-07 — PPO logprob window alignment (rollout attach + train)

**Goal:** Keep PPO `old_logprob` / `new_logprob` on the same bounded-forward scale and avoid rollout attach OOM on long sequences.

**Changes:**

- `scripts/lambda/run_economist_rl_lambda_cycle.py` — pass `ppo_config.max_logprob_window_tokens` into rollout attach, eval attach, and PPO train in the same cycle.
- `scripts/economist_rl_ppo_trainer.py` — clarify `refresh_old_logprobs=False` comment (rollout attach is windowed).
- `tests/test_economist_rl_lambda_ppo_pipeline.py` — window/full equivalence + long-sequence tail tests.
- `tests/test_economist_rl_shared_inference_logprob.py` — assert attach forwards `max_window_tokens`.
- `docs/ECONOMIST_RL_ADAPTER.md` — PPO logprob windowing section.

**Verification:** `python3 -m unittest tests.test_economist_rl_lambda_ppo_pipeline tests.test_economist_rl_shared_inference_logprob`

**Docs:** `docs/ECONOMIST_RL_ADAPTER.md` — new **RL cycle pipeline** section (artifact map + Phase 5 PPO optimization detail + Lambda full-run command).

---

## 2026-06-01 — Full v2 prep: seed retrain + cycle 012 (500 rollouts)

**Prep:** `economistRL_tasks_v2_coding.json`; `build_economist_rl_dataset.py` (1500 train rows, fenced assistants); backed up `seed_bootstrap` → `seed_bootstrap_v1_plan_20260601`; `mlx_lm.lora` 160 iters (final val loss ~0.215).

**Run:** cycle **012**, `ECONOMIST_RL_SOURCE_REPO=/Users/natreed/fallen-empire`, 500 rollouts, execution evidence on, `max_tokens=1024`, `temperature=0.2`. Logs: `benchmarks/results/economistRL/logs/cycle_012_full_500.log`, `cycle_012_nohup.out`.

---

## 2026-06-01 — economistRL coding task bank v2 (arena-aligned)

**Goal:** Convert 500 economistRL tasks from plan stubs to applyable coding tasks using game-arena / LoRA eval prompt shape.

**Changes:**
- `scripts/economist_rl_coding_contract.py` — shared coding system prompt, arena user prompts, fenced TS `reference_answer` stubs.
- `scripts/adapters/convert_economist_rl_tasks_to_coding.py` → `benchmarks/economistRL_tasks_v2_coding.json` (500/500 fenced references).
- Defaults: `run_economist_rl_lambda_cycle.py`, `build_economist_rl_dataset.py`, `economist_rl_reward_engine.py` → v2 bank + `CODING_SYSTEM_PROMPT`.
- `model_router.LocalMlxBackend` registers Qwen `im_end` stop tokens on MLX load.
- `tests/test_economist_rl_coding_tasks.py`.

**Next:** Re-run `build_economist_rl_dataset.py` + LoRA seed train so rollouts emit fences (current `seed_bootstrap` was trained on v1 plan text).

---

## 2026-06-01 — Design A execution evidence (apply + compile gate)

**Goal:** Wire real apply/compile into economistRL cycles so `compiled` is set and the compile gate applies; cancel in-flight 500-rollout run (PID 32247).

**Changes:**
- `scripts/economist_rl_execution_evidence.py` — disposable worktree per cycle, arena apply (diff/fences), shell compile commands, `compiled` / `compile_evidence` on rollout rows.
- `run_economist_rl_lambda_cycle.py` — step after toy evidence; CLI `--skip-execution`, `--execution-source-repo`, `--execution-compile-command`, `--no-require-execution-source`; eval path uses same pool; default requires `ECONOMIST_RL_SOURCE_REPO`.
- `tests/test_economist_rl_execution_evidence.py`, `docs/ECONOMIST_RL_ADAPTER.md`, `.env.example`.

**Verified:** `python3 -m unittest tests.test_economist_rl_execution_evidence -v` (apply+`true` → `compiled=True`; plan-only → `compiled=False`).

---

## 2026-06-01 — Dual eval comparisons (registry baseline + working source)

**Goal:** Report candidate vs both `seed_bootstrap` and the cycle's rollout/PPO source (e.g. `rl_pass_010` vs `rl_pass_009`).

**Changes:** `run_eval_and_compare` emits `comparisons.vs_registry_baseline` and `comparisons.vs_working_source`; skips duplicate baseline eval when working source equals registry.

---

## 2026-06-01 — Remove automatic registry promotion from economistRL cycle runner

**Goal:** Research-phase cycles should not mutate `adapter_registry_v1.json`; eval remains telemetry-only.

**Changes:** Removed `--promote-if-better`, `update_registry_adapter()`, and `promotion_decision` manifest field. Added `cycle_status` (`completed` / `dry_run` / failure states) and `registry_auto_update: false` on manifests and eval comparison payloads.

---

## 2026-06-01 — PEFT weight detection for multi-cycle chaining

**Goal:** Cycle 2+ rollouts chain to prior `rl_pass_*` when only `adapter_model.safetensors` exists (PEFT save layout).

**Changes:** `economist_rl_peft.adapter_dir_has_weights` / `adapter_weights_file`; `adapter_from_resolved_path` + `resolve_cycle_current_adapter` use them; tests in `test_economist_rl_peft.py` and `test_economist_rl_lambda_ppo_pipeline.py`.

---

## 2026-06-01 — PEFT LoRA for economistRL Transformers path (save_pretrained)

**Goal:** Stop baking MLX LoRA into base `Linear.weight` on Lambda; use Hugging Face PEFT side adapters for rollouts, eval, and PPO.

**Changes:**

- `scripts/economist_rl_peft.py`: MLX→PEFT conversion (`adapter_model.safetensors`), `load_peft_causal_lm`, `save_peft_adapter` (`PeftModel.save_pretrained`).
- `model_router.LocalMlxBackend` (transformers): `PeftModel.from_pretrained` instead of `_apply_mlx_lora_adapter_to_transformers_model`.
- `train_ppo_batch_transformers`: train PEFT params only; save candidate via `save_pretrained`.
- Lambda pip: `peft`; `requirements.txt` notes optional dep; `tests/test_economist_rl_peft.py`.

**Note:** `seed_bootstrap` MLX checkpoints convert once on load; new PPO passes write native PEFT layout.

---

## 2026-06-01 — Reuse cached inference backend for rollout old_logprob attach

**Goal:** Stop per-rollout full `from_pretrained` + LoRA merge during `attach_old_logprob_to_rollout`.

**Changes:**

- `economist_rl_ppo_trainer.py`: `attach_old_logprob_to_rollout` / `compute_sequence_logprob` accept optional `inference_backend` or `model`+`tokenizer`; helpers `_sequence_logprob_from_backend` / `_sequence_logprob_from_loaded`.
- `run_economist_rl_lambda_cycle.py`: `CachedAdapterGenerator.inference_backend()`; rollouts + eval pass `generator.inference_backend()` into attach.
- `tests/test_economist_rl_shared_inference_logprob.py`: mock backend `_ensure_loaded` called once.

**Note:** Standalone callers without `inference_backend` still load a fresh backend (legacy path).

---

## 2026-06-01 — Lambda economistRL run extracted and terminated (inefficient Transformers path)

**Goal:** Stop the slow `50×2` Lambda cycle, preserve telemetry, and free the GPU for a rebuild.

**Actions:**

- Rsynced `fe-economist-rl-cycle.log` and `gpu-smi-economist-rl.csv` from `64.181.239.58` to `benchmarks/results/economistRL/lambda_extract_20260601_terminated/`.
- Saved redacted remote `/tmp/run_economist_rl_cycle.sh` + `MANIFEST.json` / `README.md` (progress: **2/50** rollouts; no rollout JSONL on disk — `run_rollouts()` buffers until phase end; `FE_ARTIFACT_EXPORT_COMMAND` was empty).
- Terminated instance `8200031ec23e43b39544b8484995b9e9` (`fe-economist-rl-1780336744`, `gpu_1x_a10` us-west-1) via Lambda API (`manual_extract_and_rebuild`).

**Telemetry:** ~1122 GPU samples; mean GPU util **0.98%**, max **98%**; **~22 GB** VRAM resident; log shows CPU offload + per-rollout full model reload.

**Next:** Rebuild with shared backend for generate+logprob, per-rollout JSONL flush, and artifact export before long runs.

---

## 2026-05-31 — economistRL PPO/reward semantics cleanup (diagnostics vs training signal)

**Goal:** Separate continuous PPO reward from diagnostic failures and strict scorecard labels; tighten compile evidence semantics.

**Changes:**

- `score_output()` now emits `training_usable`, `hard_cap_applied`, `cap_reason`, `diagnostics`, `strict_scorecard_pass` (+ `high_reward` alias).
- Evidence runner: `compiled` only from real compile/test results; added `has_code_fence`, `has_export_function`, `looks_code_like`, `compile_checked`.
- PPO `build_ppo_samples()` documents/uses continuous `score.reward` only; skips `training_usable=False` rows.
- Lambda eval/promotion uses `hard_cap_applied` for regression flags; scorecard uses `strict_scorecard_pass`.
- Added `scripts/audit_economist_rl_ppo_reward_semantics.py` and `tests/test_economist_rl_ppo_reward_semantics.py`.

**Verified:** 23 economistRL tests pass; audit sample shows `compiled=none` for oracle rows without compile commands and 16/16 legacy-exclusion rows remain `training_usable`.

---

## 2026-05-31 — Align economistRL PPO old/new log-prob scale

**Goal:** Fix PPO loss blow-up from mismatched `old_logprob` / `new_logprob` scales.

**Changes:**

- Added shared helpers in `scripts/economist_rl_ppo_trainer.py`: `_encode_prompt_and_completion`, `_mean_completion_logprob_*`, `refresh_ppo_old_logprobs_{mlx,torch}`, `ppo_clip_loss_mlx`.
- Rollout attach, transformers path, and MLX PPO now use the same mean-per-token log-prob computation (`add_special_tokens=False` on rendered prompts).
- PPO train refreshes `old_logprob` from the loaded source adapter before updates; MLX loss stays fully differentiable (removed `float()` detach).
- Fixed `proxy_old_logprob` to mean-token scale (was incorrectly total-scale).

**Verified:** `tests/test_economist_rl_lambda_ppo_pipeline.py` (8 tests) pass.

---

## 2026-05-31 — Rename economistRL `seed_sft` → `seed_bootstrap`

**Goal:** Stop RL cycle logs/docs from implying ongoing SFT when the bootstrap checkpoint is only a one-time warm-start.

**Changes:**

- Renamed `checkpoints/adapters/economistRL/seed_sft/` → `seed_bootstrap/`.
- Updated registry, YAML, PPO guard, tests, dataset lineage (`economistRL:v1:seed_bootstrap`), and economistRL docs/workflow strings.

---

## 2026-05-31 — economistRL seed SFT + two 10-task smoke RL cycles

**Goal:** Bootstrap `economistRL/seed_sft`, then run two end-to-end smoke cycles (10 rollouts → evidence → reward → PPO → eval each).

**Seed:**

- Dataset: `.venv/bin/python scripts/ml_workflow.py economist-rl-dataset` → `data/lora/adapters/economistRL_seed/` (run `20260531-232653_76fd26`).
- SFT: `.venv/bin/python -m mlx_lm.lora --train -c training/economistRL_lora_qwen25_coder_7b.yaml` — stopped at **iter 40** (~154MB) so smoke could proceed; checkpoint at `checkpoints/adapters/economistRL/seed_sft/`.

**Smoke cycles** (`.venv/bin/python scripts/lambda/run_economist_rl_lambda_cycle.py --rollouts-per-cycle 10 --eval-limit 5 --ppo-min-samples 4 --cycles 1 --promote-if-better --temperature 0.0 --max-tokens 256`):

| Cycle | Manifest | Rollout mean reward | PPO | Eval candidate vs seed | Promotion |
|---|---|---:|---|---|---|
| 003 | `benchmarks/results/economistRL/manifests/cycle_003_manifest.json` | 0.8078 | `rl_pass_003` trained (MLX) | 0.096 vs 0.7105 | kept_current |
| 004 | `benchmarks/results/economistRL/manifests/cycle_004_manifest.json` | 0.8078 | `rl_pass_004` trained (MLX) | 0.096 vs 0.7105 | kept_current |

**Fixes during smoke:** PPO MLX save → LoRA-only `adapters.safetensors` via `tree_flatten(trainable_parameters)`; import `eval_report_high_reward` in lambda cycle.

**Open issues:** PPO loss magnitudes ~1e8; post-PPO eval collapses on generalist-safety slice (tasks 011–015) — likely PPO gradient/sign bug or eval/task mismatch, not rollout signal (rollouts strong at ~0.81).

---

## 2026-05-31 — Remove misleading per-task promotion scoring from economistRL

**Goal:** Stop implying task-level `passed` / `promotion_floor` drive RL when only continuous `reward` matters.

**Changes:**

- Removed unused `promotion_floor` / `hard_fail_floor` from `benchmarks/economistRL_tasks_v1.json`.
- Dropped `passed` from `score_output()`; added `eval_report_high_reward()` for optional scorecard summaries (`high_reward_rate`).
- Lambda eval/registry compare uses `mean_reward` instead of mean score for `--promote-if-better`.
- Updated scorecard markdown, smoke tests, and `docs/ECONOMIST_RL_ADAPTER.md`.

---

## 2026-05-31 — economistRL toy env seed normalization (spoilage/upkeep/adversarial)

**Goal:** Mass economy tasks should pass simulation scoring reliably when oracle reference answers describe bounded mechanics.

**Changes:**

- Extended `normalize_scenario_state()` in `scripts/economist_rl_sim_harnesses.py` with subsection-specific seed shaping (surplus pressure for spoilage, garrison-sized armies for upkeep, multidomain seeds for adversarial).
- Reworked spoilage sim: guaranteed over-capacity stock, preservation A/B comparison, no-total-wipe guard.
- Improved goal-name inference for mass task goal ids (`surplus_decays_over_ticks`, `small_garrisons_remain_affordable`, `dependent_projections_update`, etc.).
- Fixed upkeep/adversarial trace semantics so bounded oracle rollouts score all sim goals.
- Added `test_economy_mass_oracle_sim_meets_floor` in `tests/test_economist_rl_sim_harness_coverage.py`.

**Verified:** spoilage/upkeep/resource/adversarial oracle sim behavior now 100% (≥70 floor); all 13 economistRL tests pass.

---

## 2026-05-31 — economistRL toy sim environments (full 500-task harness coverage)

**Goal:** Lab-side 20-tick environments that score economistRL tasks even when prompts describe mechanics not present in production game code.

**Changes:**

- Reworked `scripts/economist_rl_sim_harnesses.py` (`economist_rl_toy_v2`):
  - Resolves `task_defined_seed_value` placeholders into deterministic numeric seeds (task-id jitter).
  - Handles nested `initial_state` dicts via numeric coercion.
  - Per-subsection toy envs: food/population, market elasticity, labor/wage, inventory spoilage, upkeep scaling, cache projection, adversarial multidomain.
  - Goal scoring inferred from goal names + rollout mechanics signal (not exact game implementation).
  - Simulator errors return structured `error` payloads instead of crashing the evidence runner.
- Added `tests/test_economist_rl_sim_harness_coverage.py` (500/500 harness run, oracle sim > 0).

**Verified:**

```bash
python3 -m unittest tests.test_economist_rl_sim_harness_coverage tests.test_economist_rl_lambda_ppo_pipeline tests.test_economist_rl_market_elasticity tests.test_economist_rl_food_steady_state -v
python3 scripts/lambda/run_economist_rl_lambda_cycle.py --specialization economist_rl --dry-run --rollouts-per-cycle 5 --cycles 1
```

**Outcome:** 500/500 tasks run harness without error; oracle `reference_answer` rows score `simulation_behavior > 0` for all 500 (mean reward ~0.79; 202/500 strict `passed`).

---

## 2026-05-31 — economistRL evidence runner + PPO lambda cycle

**Goal:** Replace SFT-only lambda training with a real RL loop: rollout → evidence → reward → PPO → eval.

**Changes:**

- Added `scripts/economist_rl_sim_harnesses.py` (subsection 20-tick simulators).
- Added `scripts/economist_rl_evidence_runner.py` (simulation/test/compile evidence on rollout rows).
- Added `scripts/economist_rl_ppo_trainer.py` (PPO batch builder + MLX LoRA update path).
- Rewired `scripts/lambda/run_economist_rl_lambda_cycle.py`:
  - `SpecializationConfig` now carries `train_mode`, evidence runner, and PPO config.
  - Cycle manifest v2 artifacts: `evidence/`, `ppo/`, no reference-answer SFT replay.
- Added `tests/test_economist_rl_lambda_ppo_pipeline.py`.

**Verified:**

```bash
python3 -m py_compile scripts/economist_rl_sim_harnesses.py scripts/economist_rl_evidence_runner.py scripts/economist_rl_ppo_trainer.py scripts/lambda/run_economist_rl_lambda_cycle.py
python3 -m unittest tests.test_economist_rl_lambda_ppo_pipeline tests.test_economist_rl_market_elasticity tests.test_economist_rl_food_steady_state -v
```

---

## 2026-05-31 — economistRL market elasticity smoke test

**Goal:** Wire and run a 20-tick reference price simulation for `economistRL-market-elasticity-02`, mirroring the food steady-state harness.

**Changes:**

- Added `tests/test_economist_rl_market_elasticity.py` with bounded elasticity price ticks, good/bad rollout scoring cases.

**Verified:**

```bash
python3 -m unittest tests.test_economist_rl_market_elasticity -v
```

Both tests pass. Example good rollout: scarcity 10.0 → 20.92, surplus end 7.77, reward score 100.0.

---

## 2026-05-31 — economistRL reward engine rename

**Goal:** Rename `scripts/economist_rl_tasks.py` to better reflect that it validates task-bank schema and scores rollout rewards, rather than being the task bank itself.

**Changes:**

- Renamed `scripts/economist_rl_tasks.py` -> `scripts/economist_rl_reward_engine.py`.
- Updated live references in:
  - `docs/ECONOMIST_RL_ADAPTER.md`
  - `docs/PROJECT_STATE.md`
  - `docs/WORKFLOW.md`
  - `training/adapter_registry_v1.json`

---

## 2026-05-31 — economistRL 500-task curriculum expansion

**Goal:** Expand `economistRL` from the 6 seed tasks to a 500-task RL curriculum with a 70/30 economy/generalist anti-overfit split.

**Changes:**

- Expanded `benchmarks/economistRL_tasks_v1.json` to exactly `500` tasks:
  - `350` economy tasks,
  - `150` generalist/common RL tasks.
- Economy subsection counts:
  - `food_population_feedback`: `60`
  - `market_elasticity_pricing`: `60`
  - `labor_wage_productivity`: `55`
  - `inventory_storage_spoilage`: `50`
  - `upkeep_progressive_costs`: `50`
  - `resource_projection_cache_integrity`: `50`
  - `adversarial_multidomain_economy`: `25`
- Generalist subsection counts:
  - `instruction_following`: `25`
  - `structured_reasoning`: `25`
  - `code_patch_planning`: `25`
  - `test_design_and_invariants`: `25`
  - `debugging_and_root_cause`: `20`
  - `concise_explanation`: `15`
  - `safety_and_scope_control`: `15`
- Each generated task includes:
  - `simulation_spec` with 20-tick goals,
  - `targeted_tests`,
  - `static_code_mechanics`,
  - `formula_signal`,
  - `instruction_contract`,
  - `anti_overfit_guards`,
  - `progress_guard`,
  - codebase requirement hints.
- Rebuilt `data/lora/adapters/economistRL_seed/`:
  - `train`: `1500`
  - `valid`: `12`
  - `test`: `12`
- Updated `training/adapter_registry_v1.json` task IDs and `docs/ECONOMIST_RL_ADAPTER.md` / `docs/PROJECT_STATE.md`.

**Verification:**

- `python3 scripts/economist_rl_tasks.py validate` passed with `500` tasks and exact `350/150` track counts.
- Distribution check passed across difficulty and subsection buckets.
- `python3 scripts/adapters/build_economist_rl_dataset.py` rebuilt the seed dataset successfully.

---

## 2026-05-31 — economistRL progress guard split from anti-overfit

**Goal:** Avoid classifying no-change potential outcomes as malicious reward gaming.

**Changes:**

- Updated `scripts/economist_rl_tasks.py`:
  - removed `potential_not_improved` from anti-overfit reward-gaming flags,
  - added `_progress_guard_score()` for `previous_potential` vs `new_potential`,
  - `no_change_exploratory_failure` caps final reward at `0.45` but is not reward gaming,
  - `regression` caps final reward at `0.25` but is not reward gaming unless separate anti-overfit flags fire,
  - improved potential receives progress credit and no cap.
- Updated `benchmarks/economistRL_tasks_v1.json` task metadata with `progress_guard` and revised anti-overfit text.
- Updated `docs/ECONOMIST_RL_ADAPTER.md`.

---

## 2026-05-31 — economistRL execution-oriented reward schema

**Goal:** Replace the old keyword-dominant `economistRL` reward with a schema that rewards executable behavior while preserving small prompt/guardrail checks.

**Changes:**

- Updated `benchmarks/economistRL_tasks_v1.json`:
  - default reward weights now target `simulation_behavior` (`45%`), `targeted_tests` (`20%`), `static_code_mechanics` (`15%`), `formula_signal` (`10%`), `instruction_contract` (`5%`), `concision` (`3%`), and `anti_overfit` (`2%`),
  - each seed task now has a 20-tick `simulation_spec` with relevant state, scenario, scored goals, and benchmarks,
  - each task now defines `targeted_tests`, `static_code_mechanics`, `formula_signal`, `instruction_contract`, and `anti_overfit_guards`.
- Updated `scripts/economist_rl_tasks.py`:
  - renamed the base weights to `DEFAULT_BASE_REWARD_WEIGHTS`,
  - scoring now consumes rollout JSON evidence for simulation results, targeted tests, static code mechanics, formula signals, instruction policy caps, and reward-gaming/potential checks,
  - legacy `expect.*` keyword checks remain guardrails/debug signals, not the main reward,
  - `add-task` now emits the full execution-oriented task scaffold.
- Updated `docs/ECONOMIST_RL_ADAPTER.md` with the new reward components and example rollout JSONL.

---

## 2026-05-31 — economistRL compile gate reward

**Goal:** Add a compile gate where compile success gives a decaying curriculum bonus, but compile failure always remains a catastrophic penalty even after the compile bonus weight becomes small.

**Changes:**

- Updated `scripts/economist_rl_tasks.py`:
  - added `compile_gate_reward(compiled, rolling_compile_rate, base_reward)`,
  - compile failures cap final reward at `0.10`, `0.05`, or `0.0` depending on maturity,
  - compile success bonus decays from `0.35` to `0.03` as rolling compile rate rises,
  - `score` now infers compile status from rollout fields like `compiled`, `compile_passed`, `typecheck_status`, `tsc_status`, or `verify_status`,
  - `score` accepts `--rolling-compile-rate`.
- Updated `benchmarks/economistRL_tasks_v1.json` with compile-gate scoring policy metadata.
- Updated `docs/ECONOMIST_RL_ADAPTER.md` with the exact gate equation.

---

## 2026-05-31 — economistRL 70/30 curriculum target

**Goal:** Make `economistRL` expand toward a 500-prompt RL curriculum with 70% economy specialization and 30% common/generalist RL framework skills.

**Changes:**

- Updated `benchmarks/economistRL_tasks_v1.json` with `curriculum_policy`:
  - target total: `500` prompts,
  - `350` economy prompts,
  - `150` generalist prompts,
  - economy and generalist subsection targets.
- Added `curriculum_track` to current seed tasks (`economy`).
- Updated `scripts/economist_rl_tasks.py`:
  - `validate` now reports current track counts, rates, and remaining tasks to target,
  - `add-task` accepts `--curriculum-track economy|generalist`,
  - scoring rows include `curriculum_track`.
- Updated `scripts/adapters/build_economist_rl_dataset.py` to preserve `curriculum_track` and report split counts in the generated manifest.
- Updated `docs/ECONOMIST_RL_ADAPTER.md` and `docs/PROJECT_STATE.md`.

---

## 2026-05-30 — economistRL adapter framework created

**Goal:** Start a new experimental LoRA adapter, `economistRL`, to test whether RL-style reward scoring can improve elusive Fallen Empire economy mechanics beyond the current economy tooltip/UI specialist.

**Changed files:**

- Added `benchmarks/economistRL_tasks_v1.json` with seed hard/standard economy tasks for:
  - food-supported population steady state,
  - market elasticity from stock pressure,
  - worker wage/productivity tradeoffs,
  - warehouse spoilage/food decay,
  - progressive army upkeep,
  - resource projection cache invalidation.
- Added `scripts/economist_rl_tasks.py`:
  - `validate` checks task-bank shape and duplicate ids,
  - `add-task` appends new tasks with rubric/mechanics scaffolding,
  - `score` grades rollout JSONL with rubric, mechanics, instruction, concision, and anti-overfit reward components.
- Added `scripts/adapters/build_economist_rl_dataset.py` and `python scripts/ml_workflow.py economist-rl-dataset` for seed SFT data under `data/lora/adapters/economistRL_seed`.
- Added `training/economistRL_lora_qwen25_coder_7b.yaml` targeting `checkpoints/adapters/economistRL/seed_sft`.
- Updated `training/adapter_registry_v1.json` with `adapter_id: economistRL`, `promotion_state: experimental`, routing tags, and framework paths.
- Updated router keyword/coarse-policy metadata so explicit economy simulation/RL prompts can select `economistRL`.
- Added `docs/ECONOMIST_RL_ADAPTER.md` and updated project/taxonomy docs.

**Verification:**

- `python3 scripts/economist_rl_tasks.py validate` passed (`6` tasks, no duplicate ids).
- `python3 scripts/ml_workflow.py economist-rl-dataset` succeeded and wrote run artifacts under `benchmarks/results/runs/20260531-053043_2b4179`.
- Direct rebuild via `python3 scripts/adapters/build_economist_rl_dataset.py` wrote `data/lora/adapters/economistRL_seed/` with `96` train, `4` valid, and `4` test rows.
- Reference-answer scorer smoke passed all six seed tasks at `100.0`.
- Router probe selected `economistRL` for explicit economy simulation / market elasticity / food-buffer prompts.
- `python3 -m py_compile scripts/economist_rl_tasks.py scripts/adapters/build_economist_rl_dataset.py scripts/model_router.py scripts/router/policy.py scripts/ml_workflow.py scripts/launch_lambda_parallel_ablation.py scripts/score_adapter_scorecard.py` passed.
- JSON validation passed for the task bank, adapter registry, routing prototypes, and generated seed dataset manifest.
- `ReadLints` reported no diagnostics for edited Python files.

---

## 2026-05-30 — Automatic Lambda GPU telemetry summaries

**Goal:** Ensure future Lambda eval artifacts include parsed GPU usage numbers without a separate post-run probe.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py` shared remote lifecycle:
  - artifact collector now parses `~/cloud-eval-logs/gpu-smi*.csv` before each checkpoint/final tarball,
  - writes `cloud-eval-logs/gpu-telemetry-summary.json` and `cloud-eval-logs/gpu-telemetry-summary.md`,
  - records sample counts, average/median/p95/max GPU utilization, memory MiB, power draw, and temperature,
  - uses GNU tar `--warning=no-file-changed --ignore-failed-read` so live GPU CSV writes do not fail checkpoint tarballs.
- Updated `docs/PROJECT_STATE.md` and `docs/WORKFLOW.md` with the new artifact contract.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py` (pass).
- Local synthetic `gpu-smi` CSV smoke of the generated shell summarizer produced expected `avg_gpu_util_pct=50.0` and `max_gpu_util_pct=90.0`.
- `ReadLints` reported no diagnostics for the edited files.

---

## 2026-05-30 — Measured specialist role renames and routing tags

**Goal:** Rename user-facing specialist identities and tags to match the full crossdomain 121-task skill map without breaking stable adapter IDs or checkpoint paths.

**Changed files:**

- Updated `training/adapter_registry_v1.json`:
  - kept stable `adapter_id` values unchanged,
  - added measured `display_name`, `role`, `routing_tags`, `demoted_tags`, `measured_strengths`, and `measured_weaknesses`,
  - recorded `measured_role_source=full_crossdomain_121_task_baseline_20260530`.
- Updated routing metadata:
  - `data/routing/specialist_skill_profiles_v1.json`
  - `data/routing/specialist_skill_profiles_v2.json`
  - `data/routing/council_roster_v1.json`
  - `data/routing/manifold_prototype_prompts_v2.json`
- Updated deterministic router wording:
  - `scripts/model_router.py` specialist keyword surfaces now favor `ui_surface_composer`, `ui_state_signals`, `resource_ui_projection`, `army_ui_flow`, `state_contract_guard`, and `crossdomain_state_patch` concepts.
  - `scripts/router/policy.py` coarse buckets now use measured role names while still selecting the legacy adapter IDs.
- Updated docs:
  - `docs/ROUTER_TASK_TAXONOMY.md`
  - `docs/PROJECT_STATE.md`

**Measured role map:**

- `loading_screen` -> `ui_surface_composer`
- `hud_status` -> `ui_state_signals`
- `economy_tooltip` -> `resource_ui_projection`
- `combat_risk` -> `army_ui_flow`
- `save_load_api_guard` -> `state_contract_guard`
- `ai_planning_explanation` -> `crossdomain_state_patch`

**Compatibility note:**

- Adapter IDs are intentionally unchanged because they key checkpoint paths, router labels, training data, and historical artifact recovery.

---

## 2026-05-30 — Full crossdomain 121-task specialist baseline launched

**Goal:** Establish baselines by running every game specialist against the full 121-task manifest (`--specialist-suite --specialist-suite-all-tasks --max-tasks 121`), rather than only its mapped domain slice.

**Command:**

- Loaded Lambda credentials from `/Users/natreed/.ssh/fallen-empirelora.env`.
- Ran:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --specialist-suite --specialist-suite-all-tasks --max-tasks 121 --name-prefix fe-specialist-crossdomain-full121-baseline --region us-west-1 --instance-type gpu_1x_a10 --fallback-region us-east-1 --fallback-instance-type gpu_1x_a100_sxm4 --artifact-upload-every-steps 5 --artifact-upload-every-minutes 10 --watchdog-idle-minutes 60 --watchdog-check-minutes 5 --setup-command-retries 1 --setup-retry-sleep-seconds 10`

**Launch state:**

- Lambda capacity preflight selected `gpu_1x_a10@us-west-1` after transient HTTP `429` retries.
- Six `gpu_1x_a10` workers launched in `us-west-1` with `Evaluation-Runs` attached.
- Bootstrap mount checks passed for `/lambda/nfs/Evaluation-Runs`.
- Repo sync and all six adapter syncs completed.
- Adapter fallback paths were used for missing local `champion` directories:
  - `save_load_api_guard`: `checkpoints/adapters/save_load_api_guard/cycle2`
  - `ai_planning_explanation`: `checkpoints/adapters/ai_planning_explanation/mock_aug_v2_cycle1`
- All six remote tmux jobs reached `parallel_ablation_started`.
- Remote progress sample confirmed active tmux sessions, `task_count=121`, `task_domains=` empty for all-task coverage, first rows written, and checkpoint artifacts artifacts

---

## 2026-05-30 — Specialist 121 crossdomain smoke with capacity fallback

**Goal:** Run a bounded smoke for the full specialist eval shape (`--specialist-suite --specialist-suite-all-tasks`) without running all 121 tasks per worker.

**Command:**

- Loaded Lambda credentials from `/Users/natreed/.ssh/fallen-empirelora.env`.
- Ran:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --specialist-suite --specialist-suite-all-tasks --max-tasks 1 --name-prefix fe-specialist-crossdomain-smoke121-fallback --region us-west-1 --instance-type gpu_1x_a10 --fallback-region us-east-1 --fallback-instance-type gpu_1x_a100_sxm4 --artifact-upload-every-steps 1 --artifact-upload-every-minutes 0 --watchdog-idle-minutes 30 --watchdog-check-minutes 2 --setup-command-retries 1 --setup-retry-sleep-seconds 10`

**Outcome:**

- Lambda capacity preflight reported `gpu_1x_a10` capacity in `us-east-1,us-west-1`; `gpu_1x_a100_sxm4` capacity in `asia-south-1,us-east-1,us-west-2`.
- Candidate order was `gpu_1x_a10@us-west-1`, `gpu_1x_a10@us-east-1`, then `gpu_1x_a100_sxm4@us-east-1`; the first candidate was selected after two transient HTTP 429 retries.
- Six `gpu_1x_a10` workers launched in `us-west-1` with `Evaluation-Runs` attached and reached `parallel_ablation_started`.
- Remote eval path started with `task_count=121`, `validation_status=PASS`, `variant=single_specialist_local`, `max_tasks=1`, and empty `task_domains` for all-task specialist coverage.
- Workers completed the one-task smoke and auto-terminated; final Lambda poll returned `active_like_instances 0`.

**Instance IDs:**

- `fc2b63928d434540ba575c6a04ad9908` — `loading_screen`
- `4c7e5c09aa9648aa8b6c2d1ee1ecaede` — `hud_status`
- `b0efafee70bf4dd8aa6dceb5c002f94c` — `economy_tooltip`
- `5cfd1f05cd0245009e0b049f94640101` — `combat_risk`
- `9f8df797a5a64501ba3e81ae52fce59e` — `save_load_api_guard`
- `1b5276318350443ba2e2c3607f7c0fdd` — `ai_planning_explanation`

**Artifacts:**

- Staging root: `/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts`.
- Artifact probe confirmed `artifact_count=66` total and two new tarballs for each smoke worker:
  - `checkpoint_1-<instance_id>-20260530T0550xxZ.tar.gz`
  - `cleanup_exit_0-<instance_id>-20260530T0550xxZ.tar.gz` (last worker at `20260530T055118Z`)
- Inspected worker tarballs contain `cloud-eval-logs/fe-ablation-specialist_*.log`, `cloud-eval-logs/gpu-smi-specialist_*.csv`, `benchmarks/results/cloud_ablation_rows_specialist_*.jsonl`, and one `benchmarks/results/game_task_trials/20260530-05501*-single_specialist_local-`* trial directory.
- Temporary artifact probe instance `c81d040f39664dc2875cf843c8ab264e` was terminated after inspection; final active-like Lambda instance count was `0`.

**Next intent:**

- The bounded all-task specialist-suite path is now smoke-verified. A full `--specialist-suite --specialist-suite-all-tasks --max-tasks 121` run can use the same fallback shape when budget/capacity allows.

---

## 2026-05-29 — Lambda capacity-aware launch fallback

**Goal:** Reduce Lambda insufficient-capacity failures for parallel ablation launches while preserving partial-launch cleanup safety.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added Lambda `GET /instance-types` capacity preflight parsing for `regions_with_capacity_available`,
  - added repeatable/comma-separated `--fallback-region` and `--fallback-instance-type`,
  - orders launch candidates with the requested instance type/region first, then requested type in fallback/available regions, then fallback types in preferred regions,
  - retries insufficient-capacity and partial/wrong-quantity launches on the next candidate after best-effort terminating any partial instances,
  - logs capacity decisions and the selected launch type/region to stdout.
- Updated `docs/PROJECT_STATE.md` with the durable launcher behavior.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- `python3 scripts/launch_lambda_parallel_ablation.py --help` inspection confirms the new fallback flags are exposed.

---

## 2026-05-29 — Specialist 121 crossdomain smoke cleanup

**Goal:** Run a bounded Lambda smoke for the specialist 121-task eval shape without launching the full 121-task suite.

**Command:**

- Loaded Lambda credentials from `/Users/natreed/.ssh/fallen-empirelora.env`.
- Ran:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --specialist-suite --specialist-suite-all-tasks --max-tasks 1 --name-prefix fe-specialist-crossdomain-smoke121-retry --instance-type gpu_1x_a10 --artifact-upload-every-steps 1 --artifact-upload-every-minutes 0 --watchdog-idle-minutes 30 --watchdog-check-minutes 2 --setup-command-retries 1 --setup-retry-sleep-seconds 10`

**Outcome:**

- Initial launch attempt with `fe-specialist-crossdomain-smoke121` failed immediately with Lambda `insufficient-capacity`.
- Retry launched six `gpu_1x_a10` instances in `us-west-1`, but setup failed before remote tmux/eval start when adapter checkpoint sync lost SSH connectivity across workers (`Can't assign requested address`, then `Network is unreachable`).
- Setup cleanup triggered before remote start and terminated all six launched instances:
  - `a47480cfd62c4e66bd464939134dbe13`
  - `d96a6314a116402a85d0bf1ba33be6a1`
  - `15ecd2df77c743a4b6d322f78cc2d728`
  - `c492b13ca9c742459022945c3eb2d3ff`
  - `2c5004f5c64a4965ba008123a1eb133f`
  - `ca0946018bf44ea9a3608036126f538f`
- Final Lambda instance poll returned `count: 0`; no workers were left active.
- Because the failure happened before remote lifecycle start, no eval rows, remote run artifacts, or `gpu-smi-*.csv` telemetry were produced for this smoke attempt.

**Next intent:**

- Retry later when Lambda networking/capacity is healthier, or run the already-proven mapped-domain smoke (`--specialist-suite --max-tasks 1` without `--specialist-suite-all-tasks`) before another crossdomain smoke/full run.

---

## 2026-05-29 — Lambda GPU telemetry artifacts

**Goal:** Capture per-worker GPU utilization in future Lambda eval artifacts.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - starts a background GPU monitor in the shared remote lifecycle prelude,
  - writes `nvidia-smi` CSV samples every 5 seconds to `~/cloud-eval-logs/gpu-smi-<cell>.csv`,
  - stops the monitor during remote cleanup so the existing artifact collector includes the CSV.
- Updated `scripts/launch_lambda_council_eval.py` to pass a council GPU monitor log path into the shared lifecycle helper.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py` (pass).

---

## 2026-05-29 — Lambda rsync timeout guard

**Goal:** Prevent Lambda setup from hanging indefinitely when repo or adapter `rsync` stalls over SSH.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added shared rsync timeout defaults: connect timeout 30 seconds, idle I/O timeout 120 seconds,
  - added `--contimeout=30` and `--timeout=120` to ML repo sync, game repo sync, and adapter checkpoint sync,
  - added SSH transport options for rsync: `ConnectTimeout=30`, `ServerAliveInterval=30`, `ServerAliveCountMax=4`.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).

---

## 2026-05-29 — Rename current RAG dataset surface

**Goal:** Rename the failure-guided style RAG surface to the current RAG naming requested for prompts, ablations, and data artifacts.

**Changed files:**

- Renamed `data/rag/style_prompt_rag_v2_failure_guided.json` to `data/rag/current_rag_dataset.json` and updated its `schema_version` to `current_rag_dataset`.
- Renamed the optional base corpus to `data/rag/current_rag_base_dataset.json` and updated its schema reference.
- Renamed `scripts/build_style_prompt_rag_from_failures.py` to `scripts/build_current_rag_dataset.py`:
  - default output now writes `data/rag/current_rag_dataset.json`,
  - default stats output now writes `benchmarks/results/current_rag_dataset_stats.json`,
  - CLI description now describes the current RAG dataset.
- Updated `scripts/run_final_mass_testing_system.py`:
  - renamed helper functions and local variables from style-RAG terms to `rag_current` terms,
  - replaced `--prompt-style-rag-`* flags with `--prompt-rag-current-`*,
  - changed runtime logging from `style_rag_entries` to `rag_current_entries`.
- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - default matrix cell `style_rag` is now `rag_current`,
  - combined cell `style_plus_max_potential` is now `rag_current_plus_max_potential`,
  - default corpus path now points to `data/rag/current_rag_dataset.json`,
  - pass-through argument is now `--rag-current-corpus-path`.

**Verification:**

- `python3 -m py_compile scripts/build_current_rag_dataset.py scripts/run_final_mass_testing_system.py scripts/launch_lambda_parallel_ablation.py` (pass).
- IDE lint check on edited scripts and `data/rag/current_rag_dataset.json` (pass).

---

## 2026-05-29 — Crossdomain rerun killed after adapter-sync hang

**Goal:** Stop the second full crossdomain specialist rerun and diagnose why it did not reach eval start.

**Observed:**

- Launched `fe-specialist-crossdomain-full121-rerun-`* with `--specialist-suite --specialist-suite-all-tasks --max-tasks 121`.
- Six `gpu_1x_a10` instances became active in `us-west-1`.
- Bootstrap/repo sync progressed, and adapter directories existed on all six workers.
- The launcher never printed `parallel_ablation_started`; no remote tmux sessions, eval logs, or `run_final_mass_testing_system.py` processes existed.
- Local process inspection showed three adapter `rsync` commands hung for ~40 minutes (`loading_screen/cycle2`, `hud_status/cycle3`, `economy_tooltip/cycle2`), even though remote probes showed no matching remote rsync/eval processes.
- Manually terminated the six rerun instances through the Lambda API; they moved to `terminating`.
- Manually stopped the hung local launcher and child rsync/ssh processes.

**Diagnosis:**

- The previous successful specialist matrix used `--specialist-suite --max-tasks 121` without `--specialist-suite-all-tasks`, so each specialist ran its mapped domain slice and reached `start_ok`.
- The failed reruns used `--specialist-suite-all-tasks` for the crossdomain matrix. They still failed before eval work started, but the setup surface was now longer/more expensive and exposed that setup command retries do not cover hung `rsync`; no hard subprocess timeout exists yet.

**Next intent:**

- Add hard setup subprocess timeouts for rsync/ssh/scp so a hung command exits, retries once, and then triggers setup cleanup termination.

---

## 2026-05-29 — Lambda setup-failure cleanup guard

**Goal:** Prevent newly launched Lambda specialist workers from being stranded when launcher setup fails before the remote eval script installs its own watchdog/cleanup trap.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added best-effort parent-side termination for incomplete bulk launches and partial one-by-one launch failures,
  - added default `--cleanup-on-setup-failure` behavior with `--no-cleanup-on-setup-failure` escape hatch,
  - wrapped wait/bootstrap/repo sync/adapter sync/start setup so newly launched instances that never reach remote start are terminated on setup exceptions,
  - added bounded setup command retries: each failed bootstrap/sync/adapter-sync/start command retries once by default (`--setup-command-retries 1`), then setup cleanup terminates launched instances if the retry also fails.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).

---

## 2026-05-28 — Full 121-task specialist matrix launched

**Goal:** Start the full six-specialist Lambda matrix evaluation across the 121-task final mass-testing manifest.

**Command:**

- Loaded Lambda credentials from `/Users/natreed/.ssh/fallen-empirelora.env`.
- Ran:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --specialist-suite --max-tasks 121 --name-prefix fe-specialist-matrix-full121 --instance-type gpu_1x_a10 --artifact-upload-every-steps 5 --artifact-upload-every-minutes 10 --watchdog-idle-minutes 60 --watchdog-check-minutes 5`

**Launch state:**

- Lambda initially returned transient `429` responses; the launcher retry/backoff path continued and completed the six-instance launch.
- Six `gpu_1x_a10` workers are active in `us-west-1`, all with `file_system_names=["Evaluation-Runs"]`.
- Bootstrap mount checks passed for `/lambda/nfs/Evaluation-Runs`.
- Repo sync and all six adapter syncs completed.
- Adapter fallback paths were again used where local registry `champion` paths were absent:
  - `save_load_api_guard`: `checkpoints/adapters/save_load_api_guard/cycle2`
  - `ai_planning_explanation`: `checkpoints/adapters/ai_planning_explanation/mock_aug_v2_cycle1`
- All six remote tmux jobs reached `start_ok`.
- Remote progress sample confirmed `mount True`, active tmux sessions, manifest `task_count=121`, and row files already being written:
  - `loading_screen`: 3 rows
  - `hud_status`: 3 rows
  - `economy_tooltip`: 1 row
  - `combat_risk`: 4 rows
  - `save_load_api_guard`: 3 rows
  - `ai_planning_explanation`: 4 rows

**Runtime controls:**

- Auto-termination enabled.
- Idle-log watchdog enabled: 60 minute idle threshold, 5 minute check interval.
- Artifact staging: `/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts`.
- Checkpoints: every 5 completed rows or every 10 minutes.

---

## 2026-05-28 — Specialist matrix smoke completed

**Goal:** Run the six-worker Lambda specialist-suite smoke before attempting the full mass specialist benchmark.

**Command:**

- Loaded Lambda credentials from `/Users/natreed/.ssh/fallen-empirelora.env`.
- Ran:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --specialist-suite --max-tasks 1 --name-prefix fe-specialist-matrix-smoke2 --instance-type gpu_1x_a10 --artifact-upload-every-steps 1 --artifact-upload-every-minutes 0 --watchdog-idle-minutes 30 --watchdog-check-minutes 2`

**Observed:**

- Six `gpu_1x_a10` workers launched in `us-west-1` with `file_system_names=["Evaluation-Runs"]`.
- Bootstrap mount checks passed for `/lambda/nfs/Evaluation-Runs`.
- Repo sync and all six adapter syncs completed.
- Adapter fallback paths were used where registry `champion` paths were absent locally:
  - `save_load_api_guard`: `checkpoints/adapters/save_load_api_guard/cycle2`
  - `ai_planning_explanation`: `checkpoints/adapters/ai_planning_explanation/mock_aug_v2_cycle1`
- All six remote tmux jobs reached `start_ok`.
- Remote inspection confirmed `mount True` and durable artifact staging under `/lambda/nfs/Evaluation-Runs/fallen-empire-lora-artifacts`.
- Completed-worker logs showed one-task pass output and final cleanup artifact collection before Lambda termination; all workers then disappeared from the active instance list.
- Lambda instance poll after completion returned `count 0`; no GPUs were left running.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py` to retry transient Lambda API failures (`429`, `5xx`) with short backoff during API calls.
- Updated `docs/PROJECT_STATE.md` with the retry behavior and smoke outcome.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- Lint check for `scripts/launch_lambda_parallel_ablation.py` returned no errors.

---

## 2026-05-28 — Lambda file system mount probe resolved

**Goal:** Resolve whether `Evaluation-Runs` actually mounts on Lambda instances before running the mass specialist benchmark.

**Observed:**

- Launched a single probe instance with payload:
  - `region_name=us-west-1`
  - `instance_type_name=gpu_1x_a10`
  - `ssh_key_names=["lambda-cloud-cursor"]`
  - `file_system_names=["Evaluation-Runs"]`
- While active, Lambda reported:
  - `file_system_names=["Evaluation-Runs"]`
  - `file_system_mounts=[{"mount_point": "/lambda/nfs/Evaluation-Runs", "file_system_id": "5795235886dd4e45a5adcdb5637de9d6"}]`
- SSH verification passed:
  - `/lambda/nfs/Evaluation-Runs is a mountpoint`
  - `df -h /lambda/nfs/Evaluation-Runs` reported the mounted filesystem,
  - wrote `fs_probe_marker_20260528T215740Z.txt`.
- Probe instance was terminated through Lambda API.

**Conclusion:**

- The file system attachment path works. The earlier `file_system_names: []` observation came from inspecting terminating/detached instance records, not active mounted workers.
- Keep the new bootstrap `mountpoint -q` guard because it is the strongest runtime proof that artifact staging is durable before eval work starts.

**Changed files:**

- Updated `docs/PROJECT_STATE.md` to replace the earlier blocker note with the successful mount-probe result.

---

## 2026-05-28 — Specialist matrix smoke attempt and file-system blocker

**Goal:** Run a one-task smoke for each game specialist before the mass specialist benchmark.

**Observed:**

- First `--specialist-suite --launch-instances --max-tasks 1` attempt partially launched 5 workers, then Lambda/Cloudflare returned HTTP 429 on the sixth launch.
- A sixth worker was launched after cooldown; all six became active.
- Retried with `--instance-ids ... --specialist-suite --max-tasks 1 --auto-terminate`.
- Bootstrap/sync completed and all six remote sessions started:
  - `loading_screen`
  - `hud_status`
  - `economy_tooltip`
  - `combat_risk`
  - `save_load_api_guard`
  - `ai_planning_explanation`
- Adapter sync surfaced missing local registry paths:
  - `save_load_api_guard` expected `checkpoints/adapters/save_load_api_guard/champion`; fallback used `cycle2`.
  - `ai_planning_explanation` expected `checkpoints/adapters/ai_planning_explanation/champion`; fallback used `mock_aug_v2_cycle1`.
- Remote logs showed all six workers entering model/Hugging Face load.
- Auto-termination fired; final Lambda API state had only one lingering `terminating` instance, and a repeat terminate request returned Lambda HTTP 500 because the VM was already in termination flow.

**Blocker:**

- The persistent file system did not actually attach. Lambda launch payloads included `file_system_names=["Evaluation-Runs"]`, but later `GET /instances` reported `file_system_names: []`, and `GET /file-systems` still reported `bytes_used: 7`.
- Because the file system was not mounted, result tarballs were not preserved on `Evaluation-Runs`. This blocks trusting a mass unattended run until mount attachment is verified at bootstrap.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - adapter sync now creates absolute remote paths instead of quoted `~` paths,
  - missing registry checkpoint paths fall back to the newest available local adapter cycle and copy it into the registry-expected remote path,
  - bootstrap now verifies the configured Lambda file system mount with `mountpoint -q` before doing eval setup.
- Updated `docs/PROJECT_STATE.md` with the file-system attachment finding and bootstrap mount guard.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).

---

## 2026-05-28 — Lambda file system artifact staging

**Goal:** Auto-attach the persistent Lambda file system for benchmark workers and write artifact bundles there before termination.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added default Lambda file system id `5795235886dd4e45a5adcdb5637de9d6`,
  - resolves the id/name through `GET /file-systems`,
  - passes `file_system_names` to the Lambda launch payload,
  - defaults launch region to the file system region (`us-west-1`) when attached,
  - stages artifact bundles at the file-system mount by default,
  - added `--file-system-id`, `--file-system-name`, `--no-file-system`, and `--artifact-staging-dir`.
- Updated `scripts/launch_lambda_council_eval.py` for compatibility with the shared launch/lifecycle helper signature.
- Updated `docs/PROJECT_STATE.md` with file system id, mount point, region, and controls.

**Verification:**

- Lambda API resolved file system id `5795235886dd4e45a5adcdb5637de9d6` to `Evaluation-Runs`, mount `/lambda/nfs/Evaluation-Runs`, region `us-west-1`.
- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py` (pass).
- Launcher `--help` includes file-system controls.

---

## 2026-05-28 — Lambda specialist matrix mode

**Goal:** Add a safe pre-training Lambda benchmark mode that runs all active game specialists in parallel before additional training.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added `--specialist-suite` to create one worker per game specialist,
  - added repeatable `--specialist-id` to run a subset,
  - maps specialists to normalized task domains:
    - `loading_screen` -> `hud_status`,
    - `hud_status` -> `hud_status`,
    - `economy_tooltip` -> `economy`,
    - `combat_risk` -> `army_operations`,
    - `save_load_api_guard` -> `state_perstitence_integrity`,
    - `ai_planning_explanation` -> `ai_strategy_and_planning`,
  - rejects `documentation` in this game-patch suite because it uses a documentation-specific eval path,
  - syncs each selected worker's adapter checkpoint independently.
- Updated `docs/PROJECT_STATE.md` with specialist-suite usage and scope.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- Launcher `--help` includes `--specialist-suite` and `--specialist-id`.
- Unsupported `--specialist-id documentation` is rejected with the expected game-domain mapping error.

---

## 2026-05-28 — Lambda economy specialist smoke

**Goal:** Observe one Lambda GPU smoke run for the `economy_tooltip` specialist path, including Hugging Face model load, row/artifact capture, and termination behavior.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added `--task-domain` filtering before `--max-tasks`, enabling one-task domain-specific smoke runs.
- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added `--only-cell`, `--max-tasks`, `--task-domain`, and `--single-specialist-adapter-id` pass-through,
  - normalizes copied `LAMBDA_CLOUD_BASE_URL` values ending in `/instances`,
  - remote termination now sends a normal `User-Agent`,
  - periodic artifact checkpoint labels now defer `FE_ARTIFACT_COMPLETED_STEPS` expansion until checkpoint time,
  - forced `single_specialist_local` runs sync the selected adapter checkpoint directory after the normal repo sync.
- Updated `docs/PROJECT_STATE.md` with one-task Lambda smoke controls and adapter sync behavior.

**Observed run:**

- Command shape: one `baseline` cell, `--variant single_specialist_local`, `--single-specialist-adapter-id economy_tooltip`, `--task-domain economy`, `--max-tasks 1`, `gpu_1x_a10` in `us-east-1`.
- Instance: `20856e1900c74c86a793b2e77cefe030` / `fe-economy-tooltip-smoke-1779997510`.
- Local artifact copy: `benchmarks/results/lambda_smoke_economy_tooltip_20260528/`.
- Hugging Face model path worked unauthenticated but logged the expected HF_TOKEN warning.
- Eval selected `patch-builder-automation-rebalance-15` (`economy`, `resource_projection_cache_invalidation`), wrote `src/lib/builderAutomation.ts`, and failed verification (`verify_failed_unknown`).
- Smoke exposed that the first run missed adapter weights because `checkpoints/` are excluded from bulk sync; the launcher now syncs the selected adapter for future forced-specialist runs.
- Remote termination hit Lambda HTTP 403 before the `User-Agent` fix; artifacts were copied back manually, and the instance was terminated from the local Lambda API.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py scripts/run_final_mass_testing_system.py scripts/run_council_conversation_eval.py` (pass).
- Generated remote lifecycle shell syntax checks (`bash -n`) pass.

---

## 2026-05-28 — Lambda artifact checkpoints and termination gating

**Goal:** Preserve maximum eval/conversation/routing data before any Lambda GPU termination.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py` and `scripts/run_council_conversation_eval.py`:
  - after each JSONL row append, runners can invoke `FE_ARTIFACT_CHECKPOINT_COMMAND`,
  - checkpoint cadence is controlled by `FE_ARTIFACT_UPLOAD_EVERY_STEPS` and `FE_ARTIFACT_UPLOAD_EVERY_SECONDS`.
- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - generated remote scripts now create `/tmp/fe_collect_lambda_artifacts.sh`,
  - artifact bundles include cloud logs, benchmark results, adapter registry, eval scripts, router scripts, and the final mass task manifest,
  - added `--artifact-export-command`, `--artifact-upload-every-steps`, `--artifact-upload-every-minutes`, `--require-artifact-export-before-terminate`, and `--allow-terminate-without-artifact-export`,
  - cleanup/watchdog termination now runs artifact collection/export before calling Lambda terminate.
- Updated `scripts/launch_lambda_council_eval.py` with the same artifact export controls.
- Updated `docs/PROJECT_STATE.md` with the artifact preservation lifecycle.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py scripts/run_final_mass_testing_system.py scripts/run_council_conversation_eval.py` (pass).
- `bash -n` on generated lifecycle prelude and collector script (pass).
- Launcher `--help` output includes artifact export flags.

---

## 2026-05-28 — Lambda idle-log watchdog

**Goal:** Reduce runaway Lambda GPU billing risk when a detached remote eval hangs without exiting.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added a shared remote idle-log watchdog in `_remote_lifecycle_prelude`,
  - added `--watchdog`, `--no-watchdog`, `--watchdog-idle-minutes`, and `--watchdog-check-minutes`,
  - watchdog defaults on when auto-termination is on,
  - generated ablation runners pass their `~/cloud-eval-logs/fe-ablation-*.log` path to the watchdog.
- Updated `scripts/launch_lambda_council_eval.py` with the same watchdog flags and log-idle termination behavior.
- Updated `docs/PROJECT_STATE.md` with the watchdog defaults and controls.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py` (pass).
- `python3 scripts/launch_lambda_parallel_ablation.py --help` and `python3 scripts/launch_lambda_council_eval.py --help` include watchdog flags.

---

## 2026-05-28 — Lambda max-runtime removal

**Goal:** Remove max-runtime timeout behavior from Lambda Cloud launchers.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - removed `--max-runtime-hours`,
  - removed `FE_MAX_RUNTIME_SECONDS`,
  - removed the remote `timeout --foreground` wrapper.
- Updated `scripts/launch_lambda_council_eval.py` with the same timeout removal.
- Updated `docs/PROJECT_STATE.md` so Lambda lifecycle docs now describe exit-based auto-termination only.

---

## 2026-05-28 — Lambda Cloud env naming alignment

**Goal:** Make Lambda Cloud launcher environment variables match the active local env file.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - defaults `--api-base` from `LAMBDA_CLOUD_BASE_URL`,
  - keeps legacy `LAMBDA_API_BASE` as a fallback,
  - renames the remote cleanup base-url env to `FE_LAMBDA_CLOUD_BASE_URL`.
- Updated `scripts/launch_lambda_council_eval.py` to use the same base-url resolver.
- Updated `docs/PROJECT_STATE.md` with the canonical Lambda Cloud env names.

---

## 2026-05-26 — Unbiased council adjudication

**Goal:** Remove router-selection bias from council winner selection and make adapter loading auditable in traces.

**Changed files:**

- Updated `scripts/router/council.py`:
  - adjudication now assigns blind answer ids and scores drafts with an equal-prior deterministic rubric,
  - removed dependency on participant identity, router-selected specialist status, and inherited confidence for winner ranking,
  - contribution metadata now includes `blind_id` and `rubric_scores`.
- Updated `scripts/run_council_conversation_eval.py`:
  - council traces now use neutral `confidence` / `task_outcome_score` metadata,
  - specialist outputs explicitly record that this batch eval path does not load per-specialist adapters.
- Updated `scripts/router_chat_gradio.py`:
  - live council lane now uses neutral scoring metadata and records `adapter_requested`, `resolved_mlx_adapter`, and `adapter_loaded`.
- Added `tests/test_council_adjudication.py` for equal-prior and blind-scoring smoke coverage.
- Updated `docs/ROUTER_ARCHITECTURE.md` and `docs/PROJECT_STATE.md` with the new adjudication semantics.

---

## 2026-05-26 — Council credit optimization implementation

**Goal:** Complete the staged council credit optimization plan without editing the plan file.

**Changed files:**

- Updated `scripts/router_promotion_gate.py`:
  - emits `personality_credit` with attempts, wins, top-2 proxy rate, correctness, and escalation-help counts,
  - feeds personality offline scores and sample counts into roster gating.
- Updated `scripts/router/roster.py`:
  - persists `offline_sample_count` per personality,
  - keeps expert-level online metrics optional during early offline exploration.
- Updated `scripts/router/council.py` and `scripts/model_router.py`:
  - added bandit-style personality ranking with observed offline/online correctness, sample-count exploration bonus, and state bias,
  - added `ROUTER_COUNCIL_PERSONALITY_SELECTION_POLICY` and `ROUTER_COUNCIL_PERSONALITY_EXPLORATION_RATE`.
- Updated `scripts/run_council_conversation_eval.py`:
  - added `--selective-generation`, `--selective-min-ambiguity`, `--selective-max-confidence`, and repeatable `--selective-risk` to spend generation on high-value cases first.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md`, `docs/PROJECT_STATE.md`, and `docs/WORKFLOW.md` now document personality credit, bandit selection, and selective generation.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/run_routing_benchmark.py scripts/router_promotion_gate.py scripts/run_council_conversation_eval.py` (pass).
- `python3 -m unittest tests/test_council_adjudication.py` (pass).
- Lint check for touched Python files returned no errors.
- Smoke-tested `scripts/router_promotion_gate.py` with temporary summary/rows/roster files; output included `personality_credit` and promoted `combat_risk::risk_auditor` to `active`.
- Smoke-tested `scripts/run_council_conversation_eval.py --mock-generation --selective-generation --max-tasks 1` with `/tmp` outputs; selective path completed and wrote one mock council row.

---

## 2026-05-26 — Council behavior baselines from Lambda traces

**Goal:** Preserve the recovered council run's winning persona behaviors as reusable baselines for later real-adapter council runs.

**Changed files:**

- Added `benchmarks/council_behavior_baselines/lambda_council_behavior_baselines_20260526.json`:
  - records per-base-expert behavior stats, registry lineage, winner counts, contribution scores, variants, domains, terms, and example winning outputs.
- Added `benchmarks/council_behavior_baselines/lambda_council_behavior_baselines_20260526.md`:
  - human-readable ranking and example excerpts for the recovered Lambda council traces.

**Interpretation note:**

- These are behavior/persona baselines from one-base-model generation. Lineage fields come from `training/adapter_registry_v1.json`; the recovered run does not prove per-participant LoRA weights were loaded.

---

## 2026-05-26 — RAG improvements/failures strategy writeup

**Goal:** Capture the current state of RAG improvements, the new failure modes revealed by retrieval audits and specialist failures, and the next experiment direction.

**Changed files:**

- Added `docs/RAG_IMPROVEMENTS_FAILURES_NEXT_STEPS.md`:
  - summarizes the current RAG lanes (`bug_fix`, `run_analysis`, `router`, and failure-guided style RAG),
  - records the main style-RAG improvements: retrieval-surface separation, prompt-only versus `max_potential` split, cosine ranking/margin gates, multidomain suppression, and targeted repair entries,
  - highlights remaining failures: compile/type errors still dominate acceptance loss, prompt-only cross-domain ambiguity persists, failure buckets are too coarse, and one-shot style guidance is weaker than verifier-aware repair,
  - recommends a 3-cell ablation: baseline, prompt-only RAG v2, and prompt-only RAG v2 plus verifier/error-text repair loop.

**Verification:**

- Documentation-only change; no runtime tests run.

---

## 2026-05-26 — Recovered all active Lambda host artifacts

**Goal:** Pull all remaining Lambda worker outputs before terminating instances from the dashboard.

**Recovered artifacts:**

- Copied per-host result files and logs into `benchmarks/results/lambda_recovery/hosts-20260526-184501/`.
- Host `150.136.71.190`: recovered `style_rag` ablation (`cloud_ablation_rows`_*, summary JSON/MD, runtime tasks JSON, and `fe-ablation-style_rag.log`). Summary reports 121 tasks, 39 accepted, 59 verify-passed.
- Host `150.136.244.198`: recovered `context_max_potential` ablation. Summary reports 121 tasks, 30 accepted, 60 verify-passed.
- Host `157.151.155.138`: recovered `style_plus_max_potential` ablation. Summary reports 121 tasks, 26 accepted, 65 verify-passed.
- Host `129.80.20.32`: recovered council conversation eval outputs/log again. Summary reports 121 rows.

**Operational status:**

- All listed active Lambda eval hosts have recoverable output copied locally and are safe to terminate from a data-preservation standpoint.

---

## 2026-05-26 — Recovered Lambda council eval artifacts before shutdown

**Goal:** Preserve remote Lambda eval output before stopping instances that would erase local VM disks.

**Recovered artifacts:**

- Copied council run outputs from `129.80.20.32` into `benchmarks/results/lambda_recovery/council121-20260526/`:
  - `council_conversation_eval_rows_council121-20260526.jsonl`
  - `council_conversation_eval_summary_council121-20260526.json`
  - `council_conversation_remote_smoke_rows.jsonl`
  - `council_conversation_remote_smoke_summary.json`
  - `fe-council-eval-council121-20260526.log`
- Summary confirms the council eval completed all 121 rows with 95 escalations, average 6.3 participants, and average 2.0 rounds.

**Blocker:**

- The ablation worker terminal was killed during bootstrap before final host mappings were printed. Local logs only preserve the stopped baseline IP and the council IP. Recovering ablation artifacts requires the active instance IPs from the Lambda dashboard or a working Lambda instance-control API key.

---

## 2026-05-26 — Lambda eval auto-termination guard

**Goal:** Stop Lambda Cloud eval workers from continuing to bill after detached remote runs finish or hang.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - newly launched workers now default to remote `auto_terminate=1`,
  - reused `--instance-ids` remain non-terminating by default unless `--auto-terminate` is set,
  - remote cell scripts call Lambda `/instance-operations/terminate` on exit when auto-termination is enabled. Note: a later 2026-05-28 update removed the temporary max-runtime timeout guard.
- Updated `scripts/launch_lambda_council_eval.py` with the same auto-terminate/default reuse policy.
- Updated `.env.example` to remove a real-looking Lambda token from the example and document separate `LAMBDA_API_KEY` / `LAMBDA_API_BASE` instance-control settings.
- Updated `docs/PROJECT_STATE.md` with the Lambda launcher lifecycle policy.

**Operational finding:**

- Existing Lambda launchers started detached `tmux` sessions and returned locally; they did not previously terminate the VM after eval completion.
- Current local shell did not have `LAMBDA_API_KEY`; reading the repo `.env` value reached Lambda Cloud but returned `403 Forbidden` for `/instances`, so this session could not list or terminate active instances from the available credentials.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py scripts/launch_lambda_council_eval.py` (pass).
- `python3 scripts/launch_lambda_parallel_ablation.py --help` (pass).
- `python3 scripts/launch_lambda_council_eval.py --help` (pass).
- Lint check for both touched launcher scripts returned no errors.

---

## 2026-05-26 — Personality-level council credit gate

**Goal:** Add a cheap optimization loop for council personalities before scaling conversation generation or training.

**Changed files:**

- Updated `scripts/router/roster.py`:
  - personality entries now carry `state`, `offline_score`, `online_task_outcome`, and `online_sample_count`,
  - added nested personality gate transitions: `candidate`, `active`, `cooldown`, `retired`,
  - retired personalities are excluded from planning via `personalities_map()`.
- Updated `scripts/router_promotion_gate.py`:
  - aggregates personality offline credit from routing benchmark `council_selected_personalities`,
  - also supports richer council conversation rows using `rounds[].participants` and `adjudication.winner_ids`,
  - accepts optional online personality metrics under `online.personalities`.
- Updated `scripts/run_routing_benchmark.py`:
  - emits `council_selected_personalities` so benchmark rows can seed personality credit.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md`, `docs/PROJECT_STATE.md`, and `docs/WORKFLOW.md` describe two-level expert/personality elimination.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/run_routing_benchmark.py scripts/router_promotion_gate.py` (pass).
- Lint check for touched Python files returned no errors.
- Smoke-tested `ExpertRoster.apply_combined_gate(...)` with synthetic personality credit; the parent `combat_risk` expert became active while only the passing `risk_auditor` personality became active.

---

## 2026-05-26 — Roster-defined expert personalities

**Goal:** Move council personality options into expert roster metadata instead of relying only on fixed cautious/balanced/assertive deltas.

**Changed files:**

- Updated `scripts/router/roster.py`:
  - added `personalities` to roster entries,
  - normalizes named personalities with direct trait knobs and optional `trait_deltas`,
  - exposes `personalities_map()` for council planning.
- Updated `scripts/router/council.py`:
  - council planning now prefers roster-defined personalities for selected specialists,
  - `ROUTER_COUNCIL_SPECIALIST_PERSONALITY_VARIANTS` now caps per-expert personalities,
  - generic cautious/balanced/assertive variants remain only as fallback.
- Updated `scripts/model_router.py` and `scripts/run_routing_benchmark.py` to pass roster personalities into council planning.
- Updated docs:
  - `docs/PROJECT_STATE.md`, `docs/ROUTER_ARCHITECTURE.md`, and `docs/WORKFLOW.md` describe roster-defined personalities and correctness-only adjudication boundaries.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/run_routing_benchmark.py scripts/init_council_roster.py` (pass).
- Lint check for touched Python files returned no errors.
- Smoke-tested `build_council_plan(...)` with custom `probability_hawk` and `field_commander` personalities; participant ids used the roster-defined names.

---

## 2026-05-26 — Correctness-only council adjudication

**Goal:** Separate council inference policy from final adjudication so expert traits shape interaction but do not decide winners.

**Changed files:**

- Updated `scripts/router/council.py`:
  - removed assertiveness and personality trait terms from adjudication scoring,
  - limited adjudication contribution metadata to correctness inputs (`task_outcome_score`, confidence, answer presence via score),
  - kept traits available on participant traces for inference/prompt shaping.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md`, `docs/PROJECT_STATE.md`, and `docs/WORKFLOW.md` now describe variants as inference-time interaction behavior and adjudication as correctness-only.

**Verification:**

- `python3 -m py_compile scripts/router/council.py` (pass).
- Lint check for `scripts/router/council.py` returned no errors.

---

## 2026-05-25 — Skip baseline by default in parallel Lambda ablations

**Goal:** Avoid rerunning the unchanged baseline cell on every cloud prompt ablation and stop the currently running baseline worker to save GPU cost.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - removed `baseline` from the default cell list,
  - added `--include-baseline` for explicit fresh-control runs,
  - default parallel runs now launch only `style_rag`, `context_max_potential`, and `style_plus_max_potential`.

**Cloud operation:**

- Stopped `fe-ablation-baseline` on `158.101.101.166`.
- Terminated the idle baseline Lambda instance (`7d54b63d95f944809f2226a8cab80abc`).
- Confirmed the remaining three non-baseline sessions are still running.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- CLI help confirms `--include-baseline`.

---

## 2026-05-25 — Parallel Lambda launcher recovery and active 4-worker run

**Goal:** Restore cloud-parallel ablation execution after fresh Lambda workers stalled during serial bootstrap/sync.

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - added fallback one-by-one instance launches for Lambda accounts that reject `quantity > 1`,
  - parallelized worker bootstrap and repo sync with `ThreadPoolExecutor`,
  - reduced ML repo rsync payload by excluding large local-only paths (`checkpoints`, `models`, `data/raw`, `data/lora`),
  - installed global TS tooling on workers during sync (`ts-node`, `tsconfig-paths`, `typescript`),
  - added progress prints (`bootstrap_ok`, `sync_ok`, `start_ok`) for faster confirmation.

**Cloud execution:**

- Reused/started 4 active `gpu_1x_a10` Lambda workers.
- Confirmed active tmux sessions:
  - `baseline` -> `fe-ablation-baseline`
  - `style_rag` -> `fe-ablation-style_rag`
  - `context_max_potential` -> `fe-ablation-context_max_potential`
  - `style_plus_max_potential` -> `fe-ablation-style_plus_max_potential`

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- Remote session/log check confirmed all four `fe-ablation-`* sessions and log files exist.

---

## 2026-05-25 — Parallel Lambda launcher aligned with final prompt modes

**Goal:** Ensure cloud-parallel ablation infrastructure remains runnable after context-engineering mode consolidation (`off` + `max_potential`).

**Changed files:**

- Updated `scripts/launch_lambda_parallel_ablation.py`:
  - replaced deprecated matrix cells:
    - `context_heavy` -> `context_max_potential`
    - `style_plus_context` -> `style_plus_max_potential`
  - switched context-engineering args from `heavy` to `max_potential`,
  - updated default style corpus to `data/rag/style_prompt_rag_v2_failure_guided.json`.
- Updated `docs/PROJECT_STATE.md`:
  - corrected final-system prompt context modes to `off` and `max_potential`.

**Verification:**

- `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).
- `python3 scripts/launch_lambda_parallel_ablation.py --help` (pass).

---

## 2026-05-25 — Add `max_potential` context-engineering prompt mode

**Goal:** Support an explicitly non-realistic, "maximize model ceiling" prompt profile for context-engineering experiments, independent from RAG realism constraints.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added `--prompt-context-engineering max_potential`,
  - `max_potential` prompt injection now includes:
    - domain/subskill-aware execution brief,
    - expanded ranked context/verify/edit-scope sections,
    - label-aware advanced style guidance retrieval with wider budget,
    - explicit execution strategy + pre-output self-check checklist,
    - strict output contract block.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).
- Lint check for updated file (no issues).

---

## 2026-05-25 — Context engineering `sharp_v2` mode for prompt precision

**Goal:** Replace broad context-heavy prompt injection with a sharper, budgeted context mode that prioritizes failure-targeted guidance and strict output-shape compliance.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added `--prompt-context-engineering sharp_v2`,
  - added `--prompt-sharp-v2-components` (comma-separated `style,constraints,contract`) for component-level prompt ablations,
  - implemented failure-profile-focused style retrieval in `sharp_v2` (stronger score/margin gates, reduced snippet count/char budget),
  - added ranked/truncated context path selection and tighter constraints formatting for high-signal context only,
  - added concise strict output-contract section to reduce non-applyable output drift.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).
- Lint check for updated file (no issues).

---

## 2026-05-25 — Prompt-only miss artifact + targeted repair entry generation

**Goal:** Materialize the remaining prompt-only retrieval misses into an auditable artifact and automatically add targeted subskill repair entries for uncovered misses.

**Changed files:**

- Generated `benchmarks/results/style_prompt_rag_prompt_only_misses_v1.json`:
  - full list of prompt-only top-retrieval misses (`40` rows).
- Updated `data/rag/style_prompt_rag_v2_failure_guided.json`:
  - auto-added `34` targeted `subskill_failure_profile` repair entries (`targeted_repair` tag) for missing/under-covered miss subskills.
- Generated `benchmarks/results/style_prompt_rag_targeted_repair_stats.json`:
  - `miss_count=40`, `unique_missing_subskills=39`, `repairs_added=34`.

**Verification (121-task retrieval audit):**

- Prompt-only mode improved from previous state to:
  - `strict (domain+subskill): 80/121`
  - `broad (domain+subskill or domain-only): 105/121`
  - `no-label-match: 16/121`
- Heavy mode:
  - `strict: 88/121`
  - `domain-only: 33/121`
  - no `no_snippet` rows observed in this pass.

---

## 2026-05-25 — Prompt-only multidomain suppression policy for style-RAG retrieval

**Goal:** Reduce broad `multidomain` over-selection in realistic prompt-only retrieval while preserving label-aware heavy-mode behavior.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py` retrieval policy:
  - in prompt-only mode (`use_label_signals=False`), suppress `multidomain` entries unless prompt-domain inference indicates cross-domain intent,
  - apply mild multidomain score demotion for cross-domain prompt-only retrieval so explicit domain matches win when available,
  - keep heavy/tag-aware mode permissive (no extra multidomain filter) to avoid reducing retrieval coverage in prompting experiments.

**Verification (121-task local retrieval audit):**

- Prompt-only mode:
  - `wrong_multidomain_top`: `18` -> `5`
  - `domain+subskill`: `51` -> `46`
  - `domain-only`: `30` -> `35`
  - `no-label-match`: `40` (unchanged)
- Heavy mode preserved strong coverage:
  - `domain+subskill=51`, `domain-only=66`, `subskill-only=1`, `no-snippet=3`.

---

## 2026-05-25 — Retrieval-surface RAG entries + prompt-only audit rerun

**Goal:** Improve prompt-only style-RAG sorting by separating retrieval features from final guidance text (examples/symptoms/aliases/signatures), mirroring specialist-router-style discriminative retrieval surfaces.

**Changed files:**

- Updated `scripts/build_style_prompt_rag_from_failures.py`:
  - added retrieval-surface fields to generated entries:
    - `retrieval_description`
    - `retrieval_aliases`
    - `retrieval_symptoms`
    - `retrieval_task_signatures`
    - `retrieval_examples`
  - added runtime-task prompt intake (`--tasks-json` / `--tasks-glob`) to attach representative example prompts per domain/subskill.
  - default output now excludes base entries unless explicitly enabled (`--include-base-entries`).
- Updated `scripts/run_final_mass_testing_system.py`:
  - style-RAG vectors now embed retrieval surface fields (fallback to `text` if missing) rather than embedding guidance text only.
- Regenerated `data/rag/style_prompt_rag_v2_failure_guided.json` and `benchmarks/results/style_prompt_rag_v2_failure_stats.json`.

**Commands / verification:**

- `python3 scripts/build_style_prompt_rag_from_failures.py` (pass):
  - `rows=484`, `task_prompts=121`, `generated_entries=46`, `output_entries=46`.
- Retrieval audit over 121 tasks (local, no generation):
  - prompt-only mode:
    - before: `domain+subskill=3`, `domain-only=18`, `no-label-match=78`, `no-snippet=22`
    - after: `domain+subskill=51`, `domain-only=30`, `no-label-match=40`, `no-snippet=0`
  - heavy mode (label-aware): `domain+subskill=51`, `domain-only=66`, `subskill-only=1`, `no-snippet=3`.
- `python3 -m py_compile scripts/build_style_prompt_rag_from_failures.py scripts/run_final_mass_testing_system.py` (pass).

**Outcome:**

- Prompt-only retrieval quality improved materially after adding discriminative retrieval surfaces, reducing arbitrary cross-domain picks caused by near-duplicate guidance text embeddings.

---

## 2026-05-25 — Router-inspired style-RAG sorting (cosine + margin gates)

**Goal:** Reuse proven router ranking patterns for style-RAG sorting to reduce noisy snippet selection and ambiguous top-k retrieval.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added sparse TF + cosine helpers (`_sparse_tf`, `_cosine_sparse`),
  - style corpus entries now cache sparse vectors (`vector`) for retrieval scoring,
  - `_retrieve_style_snippets(...)` now ranks candidates by cosine similarity instead of raw token-overlap count,
  - added calibration gates:
    - `--prompt-style-rag-min-score` (default `0.08`) to suppress weak matches,
    - `--prompt-style-rag-min-margin` (default `0.02`) to collapse near ties to top-1,
  - preserved the realism split:
    - prompt-only retrieval by default,
    - label/tag-aware candidate narrowing only in `--prompt-context-engineering heavy`.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).

---

## 2026-05-25 — RAG retrieval realism split: prompt-only by default, label-aware in heavy context mode

**Goal:** Prevent label leakage in style-RAG retrieval for realistic runs while preserving label-aware retrieval in explicit context-heavy prompting experiments.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - `_retrieve_style_snippets(...)` now accepts `use_label_signals: bool`,
  - when `use_label_signals=False`, retrieval query uses prompt text only and does not apply domain/subskill tag bonus,
  - when `use_label_signals=True`, existing domain/subskill-aware behavior is preserved.
  - `_augment_task_prompt(...)` now sets:
    - `use_label_signals = (context_engineering_mode == "heavy")`.

**Behavioral impact:**

- Style-RAG-only runs now retrieve using prompt-only information (realistic setting).
- Context-heavy prompt experiments (`--prompt-context-engineering heavy`) still allow label/tag-informed retrieval.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).

---

## 2026-05-25 — Rewrite failure-guided RAG text to actionable-only guidance

**Goal:** Remove non-actionable provenance prose from failure-guided RAG entries so retrieved snippets stay concise and instruction-focused for generation.

**Changed files:**

- Updated `scripts/build_style_prompt_rag_from_failures.py`:
  - changed `_entry_text(...)` to emit compact action guidance (`For domain/subskill ...`) instead of counts/sample prose.
- Regenerated `data/rag/style_prompt_rag_v2_failure_guided.json`:
  - rewrote all entry text to concise imperative guidance, no `Observed failure profile...` metrics in snippet text.
- Updated `benchmarks/results/style_prompt_rag_v2_failure_stats.json` notes to reflect text rewrite.

**Commands / verification:**

- `python3 scripts/build_style_prompt_rag_from_failures.py` (pass)
- post-process cleanup retained failure-only entries (`final_entries 46`).

---

## 2026-05-25 — Clean v2 style RAG corpus to failure-only entries

**Goal:** Remove carried-over v1 handcrafted entries from the generated v2 corpus so retrieval uses only failure-guided guidance.

**Changed files:**

- Updated `data/rag/style_prompt_rag_v2_failure_guided.json`:
  - removed the original v1 entries,
  - retained only generated `failure_profile`_* entries (`46` total).
- Updated `benchmarks/results/style_prompt_rag_v2_failure_stats.json`:
  - adjusted metadata to reflect `base_entries=0`,
  - updated `output_entries` and added cleanup note.

**Verification:**

- One-shot cleanup script output: `entries_after_cleanup 46`.

---

## 2026-05-25 — Failure-guided style RAG corpus generation from cloud ablation rows

**Goal:** Replace sparse static-only style guidance with retrieval snippets grounded in observed failure patterns by domain + subskill from the 2x2 Lambda prompt ablation outputs.

**Changed files:**

- Added `scripts/build_style_prompt_rag_from_failures.py`:
  - ingests benchmark row JSONL files (default `benchmarks/results/cloud_ablation_rows_*.jsonl`),
  - classifies failures (`no_applyable_changes`, `verify_`*, runtime/rejected buckets),
  - aggregates dominant failures by domain and domain+subskill,
  - emits a merged corpus with original v1 entries plus generated failure-profile entries,
  - writes a machine-readable stats report for auditability.
- Generated `data/rag/style_prompt_rag_v2_failure_guided.json`:
  - includes the 10 original v1 entries plus 46 generated failure-profile entries (`56` total).
- Generated `benchmarks/results/style_prompt_rag_v2_failure_stats.json`:
  - captures top failure domains/subskills and bucket counts used to build v2.

**Commands / verification:**

- `python3 scripts/build_style_prompt_rag_from_failures.py` (pass):
  - `rows=484`, `row_files=4`, `base_entries=10`, `generated_entries=46`, `output_entries=56`.

**Outcome:**

- Produced a failure-informed style corpus that can be used for the next ablation cycle to provide targeted guidance (especially where `no_applyable_changes` and `verify_failed_unknown` dominate).

---

## 2026-05-24 — Parallel Lambda ablation launcher (multi-worker cell fanout)

**Goal:** Enable true cloud-parallel experiment execution by dispatching one ablation cell per Lambda GPU worker instead of running cells sequentially on one instance.

**Changed files:**

- Added `scripts/launch_lambda_parallel_ablation.py`:
  - launches Lambda workers (or reuses provided instance ids),
  - waits for active instances + SSH reachability,
  - bootstraps runtime dependencies on each worker,
  - syncs ML + game repos (optional `--skip-sync`),
  - starts one tmux session per ablation cell (`baseline`, `style_rag`, `context_heavy`, `style_plus_context`),
  - writes isolated per-cell artifacts under `benchmarks/results/cloud_ablation_`*.

**Usage shape:**

- New instances:
  - `python scripts/launch_lambda_parallel_ablation.py --launch-instances --instance-type gpu_1x_a10 --region us-east-1 --ssh-key-name lambda-cloud-cursor`
- Existing workers:
  - `python scripts/launch_lambda_parallel_ablation.py --instance-ids <id1,id2,id3,id4> --skip-sync`

**Verification:**

- Local syntax check:
  - `python3 -m py_compile scripts/launch_lambda_parallel_ablation.py` (pass).

---

## 2026-05-24 — Arena generation backend caching for faster cloud ablations

**Goal:** Remove per-task model reload overhead during ablation runs by reusing initialized generation backends across tasks.

**Changed files:**

- Updated `scripts/game_task_arena.py`:
  - added in-process backend caches:
    - `_LOCAL_BACKEND_CACHE` keyed by `(local_model_id, adapter_path)`
    - `_FRONTIER_BACKEND_CACHE` keyed by `(model, base_url)`
  - `generate_attempt()` now reuses cached backend instances instead of constructing/loading a new backend on every task.

**Cloud follow-up:**

- Synced updated `game_task_arena.py` to Lambda worker.
- Restarted tmux session `fe-prompt-ablation` so the 2x2 matrix uses the caching behavior from task 1 onward.

**Expected impact:**

- Eliminates repeated local model weight loading for every single task within a run process.
- Improves throughput for long 121-task experiment sweeps and ablation matrices.

---

## 2026-05-24 — Prompt ablation matrix (style-RAG + context engineering) setup and cloud launch

**Goal:** Evaluate whether quality improves without additional training by adding (a) style-based RAG snippets and (b) heavier prompt context engineering.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added optional style-RAG prompt augmentation:
    - `--prompt-style-rag-corpus`
    - `--prompt-style-rag-top-k`
    - `--prompt-style-rag-max-chars`
  - added optional context-engineering augmentation:
    - `--prompt-context-engineering off|heavy`
  - implemented lightweight lexical retrieval + prompt augmentation pipeline that attaches style snippets and/or structured execution constraints to each task prompt before generation.
- Added `data/rag/style_prompt_rag_v1.json`:
  - style guidance entries for global apply-contract discipline plus domain-specific tags (HUD/economy/combat/save-load/planning).

**Cloud execution launched:**

- Started tmux session `fe-prompt-ablation` on Lambda with a sequential 2x2 matrix (all on `qwen_7_5b_only`, full 121-task manifest):
  1. `baseline`
  2. `style_rag`
  3. `context_heavy`
  4. `style_plus_context`
- Artifacts per cell are written under `benchmarks/results/`:
  - `cloud_ablation_rows_<cell>.jsonl`
  - `cloud_ablation_summary_<cell>.json`
  - `cloud_ablation_summary_<cell>.md`
  - `cloud_ablation_runtime_tasks_<cell>.json`

**Verification:**

- Local syntax check passed for `scripts/run_final_mass_testing_system.py`.
- Cloud run log confirms matrix start and successful baseline initialization/load.

---

## 2026-05-24 — Specialist cloud failure playbook (verify-signature + output-shape analysis)

**Goal:** Convert the completed six-specialist cloud 121-task runs into actionable failure buckets for targeted retraining and prompt/guardrail fixes.

**Changed files:**

- Added `docs/SPECIALIST_FAILURE_PLAYBOOK_20260524.md`:
  - aggregated failure mix across six specialist runs (`495` failed rows),
  - ranked top recurring TypeScript verify signatures (top 20),
  - identified dominant non-apply output-shape bucket (`fenced_non_contract_output`),
  - documented concrete remediation plan (prompt controls, repair-set training data, symbol-validation guardrail loop, and instrumentation split for `verify_failed_unknown`).

**Commands / analysis run:**

- Queried cloud artifacts and trial logs under `benchmarks/results/` and `benchmarks/results/game_task_trials/` on the Lambda worker.
- Parsed verify logs to normalize recurring TS signatures and counted frequency.
- Parsed `model_output.md` for `no_applyable_changes` trials to classify output-shape failures.

**Outcome:**

- Dominant failure bucket confirmed as compile/type verification failures (`419/495`), with output-shape non-apply failures secondary (`76/495`).
- Produced a prioritized remediation sequence for next training/eval cycle.

---

## 2026-05-23 — Linux local-backend fallback for cloud Qwen runs (model_router)

**Goal:** Unblock Lambda/Linux execution for local Qwen lanes by removing the hard MLX-only dependency in `LocalMlxBackend`.

**Changed files:**

- Updated `scripts/model_router.py`:
  - `LocalMlxBackend` now supports backend selection via `LOCAL_BACKEND` (`mlx` or `transformers`), with automatic fallback to transformers when MLX import/load is unavailable.
  - added a Linux transformers generation path (`AutoModelForCausalLM` + `AutoTokenizer`) with greedy decode when `temperature=0.0` and sampled decode when `temperature>0`.
  - added model-id normalization for MLX-style model ids (for example `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` -> `Qwen/Qwen2.5-Coder-7B-Instruct`) in transformers fallback.
  - local-route telemetry now reports backend as `local_transformers` when the fallback is active.
  - added explicit guard that MLX adapter paths are not yet supported under transformers fallback.

**Cloud execution notes:**

- Launched Lambda worker `fe-specialist-eval-worker` (`gpu_1x_a10`, `us-east-1`) and configured detached tmux runners.
- `gpt_5_5_only` cloud ablation run completed (`121` rows written).
- Linux fallback smoke was validated (`LOCAL_BACKEND=transformers`, local route output returned expected one-line response).
- `qwen_7_5b_only` cloud run was started with `LOCAL_BACKEND=transformers` and progressed past model load.

**Verification:**

- Local syntax check: `source .venv/bin/activate && python -m py_compile scripts/model_router.py` (pass).
- Remote smoke test (Lambda, transformers fallback): model loaded + generated expected single-line response.
- Remote `qwen_7_5b_only` run startup confirmed in `~/cloud-eval-logs/fe-qwen-eval.log`.

---

## 2026-05-23 — Add Lambda Cloud env template for OpenAI-compatible frontier lane

**Goal:** Make provider switching explicit by documenting Lambda Cloud-compatible environment values in the repo-level env template.

**Changed files:**

- Updated `.env.example`:
  - added a provider-oriented layout with a default OpenAI block and a commented Lambda Cloud block,
  - documented `FRONTIER_API_BASE_URL=https://api.lambda.ai/v1` and placeholder model/id values for quick copy into local `.env`.

**Verification:**

- Manual config check: variable names remain unchanged (`FRONTIER_API_KEY`, `FRONTIER_MODEL`, `FRONTIER_API_BASE_URL`), so existing scripts that load `.env` continue to work without code changes.

---

## 2026-05-17 — Prompt-site consolidation: embedded arena split/merge validation controls

**Goal:** Consolidate testing workflow into the primary prompt/dashboard site so arena split/merge validation can be viewed and executed without switching to the separate arena Gradio app.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added new dashboard tab/panel **Arena Validation** in the main web UI,
  - added validation artifact APIs:
    - `GET /api/arena/validation/list`
    - `GET /api/arena/validation/content?path=...`
    - `POST /api/arena/validation/run`
  - added backend helpers for:
    - listing `benchmarks/results/multi_agent_orchestration_validation*.json`,
    - loading/summarizing selected artifact metrics,
    - running expanded multi-source validation from dashboard and writing:
      - `benchmarks/results/multi_agent_orchestration_validation_dashboard_latest.json`
      - `benchmarks/results/game_task_reports/dashboard_validation_run.log`
  - added front-end controls inside dashboard panel:
    - artifact selector,
    - refresh button,
    - run-expanded-validation button,
    - summary + raw JSON viewer with auto-refresh.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- `source .venv/bin/activate && python3 - <<'PY' ... _validation_artifact_list(limit=3) ... PY` (pass, artifacts discovered).
- `source .venv/bin/activate && python3 - <<'PY' ... _validation_artifact_payload(...) ... PY` (pass, summary generated).

---

## 2026-05-17 — High-tier run on 50-task HUD+combat field-flow suite

**Goal:** Execute the renamed 50-task field-flow benchmark under high complexity tier scoring.

**Run command:**

- `.venv/bin/python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --tier A --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 --output-jsonl benchmarks/results/hud_combat_field_flow_high_tier_rows_20260518.jsonl`

**Outcome:**

- Summary: `19/50` (`38%`) with non-zero exit because benchmark requires full pass for exit `0`.
- Capability Index: `78.1/100` (`correctness 86.1`, `instruction 100.0`, `concision 24.1`, `speed 31.0`).
- Advanced ACI: `64.1/100` (`weighted 78.1`, `domain-balance 38.0`, `multi-domain 38.0`).
- Dominant misses: strict required-keyword misses (`hud`, `risk`, `attacker/defender`, `wall`, `reinforcement`, `skirmish`) and occasional fenced-output token violations (````` present).

---

## 2026-05-17 — Massive split/merge validation expansion (suite builder + validator v2 + multi-source audits)

**Goal:** Materially increase split/merge validation coverage (volume, variety, edge controls) with auditable and repeatable artifacts.

**Changed files:**

- Added `scripts/build_multi_agent_validation_suite.py`:
  - deterministic generator for large validation corpus with:
    - dual-specialist coverage across all specialist pairs,
    - tri-specialist cross-domain prompts,
    - single-specialist controls,
    - high-risk frontier controls,
    - mechanical/ambiguous controls.
- Updated `scripts/validate_multi_agent_orchestration.py` (backwards-compatible v2 output):
  - supports repeatable multi-source input via repeated `--tasks` and optional `--tasks-glob`,
  - preserves legacy top-level fields (`tasks_path`, `rows`, split/merge metrics),
  - adds robustness metrics:
    - `expected_multi_agent_rows/hit_rows/hit_rate`,
    - `unexpected_multi_agent_rows/rate`,
    - `priority_valid_rows/rate`,
    - `unique_adapter_rows/rate`,
    - `empty_subtasks_rows`,
    - `coarse_bucket_counts`, `source_counts`, `category_metrics`,
    - per-row `validation_errors`,
  - adds optional gates (`--expected-multi-agent-min-rate`, `--max-unexpected-multi-agent-rate`) while keeping default behavior compatible.

**Commands run:**

- `source .venv/bin/activate && python3 -m py_compile scripts/validate_multi_agent_orchestration.py scripts/build_multi_agent_validation_suite.py`
- `source .venv/bin/activate && python scripts/build_multi_agent_validation_suite.py --output benchmarks/multi_agent_orchestration_mass_tasks_v1.json`
- Baseline validation:
  - `source .venv/bin/activate && python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_baseline_v2.json`
- Expanded validation (multi-source + gates):
  - `source .venv/bin/activate && python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --tasks benchmarks/multi_agent_orchestration_mass_tasks_v1.json --tasks benchmarks/task_routing_mixed_tasks_v1.json --expected-multi-agent-min-rate 0.6 --max-unexpected-multi-agent-rate 0.25 --output-json benchmarks/results/multi_agent_orchestration_validation_expanded_v2.json`
- Expanded validation with stricter secondary confidence:
  - `source .venv/bin/activate && python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --tasks benchmarks/multi_agent_orchestration_mass_tasks_v1.json --tasks benchmarks/task_routing_mixed_tasks_v1.json --secondary-min-confidence 0.35 --expected-multi-agent-min-rate 0.55 --max-unexpected-multi-agent-rate 0.35 --output-json benchmarks/results/multi_agent_orchestration_validation_expanded_v2_conf35.json`
- Expanded validation under similarity mode:
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --tasks benchmarks/multi_agent_orchestration_mass_tasks_v1.json --tasks benchmarks/task_routing_mixed_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_expanded_v2_similarity.json`

**Outcomes:**

- Generated suite `benchmarks/multi_agent_orchestration_mass_tasks_v1.json`:
  - `rows=284`,
  - `expected_multi_agent_rows=180`,
  - `expected_single_agent_rows=104`.
- Baseline (single-source legacy set, 50 rows):
  - `split_valid_rate=1.0`, `merge_valid_rate=1.0`,
  - `multi_agent_rows=35`,
  - `expected_multi_agent_hit_rate=0.70`.
- Expanded multi-source run (`rows=346`) surfaced major orchestration-policy gaps:
  - split/merge structure still perfect (`1.0/1.0`),
  - `expected_multi_agent_hit_rate=0.6478`,
  - `unexpected_multi_agent_rate=0.5776` (high over-splitting on control categories),
  - useful per-category failure visibility added via `category_metrics`.
- Stricter secondary confidence (`0.35`) reduced unexpected splitting sharply (`0.0345`) but collapsed expected split coverage (`0.2826`), revealing confidence-threshold tradeoff.
- Similarity-mode expanded run increased splitting (`multi_agent_rows=190`) but retained high unexpected split rate (`0.5603`), confirming mode-level policy imbalance.

---

## 2026-05-17 — Arena UI integration for split/merge validity viewing + rerun

**Goal:** Integrate expanded split/merge validation into the arena website so results can be viewed and re-tested directly in UI.

**Changed files:**

- Updated `scripts/game_task_arena.py`:
  - added validation artifact helpers:
    - `validation_artifact_paths()`
    - `summarize_validation_artifact()`
  - added arena UI controls under new accordion **Split/Merge Validity Testing**:
    - artifact picker for `benchmarks/results/multi_agent_orchestration_validation*.json`,
    - refresh button for artifact list,
    - run button to execute expanded validation suite from UI,
    - human-readable summary panel + raw JSON viewer.
  - wired run action to execute:
    - `scripts/validate_multi_agent_orchestration.py` over low-task + mass-suite + mixed tasks
    - writes latest output to `benchmarks/results/multi_agent_orchestration_validation_arena_latest.json`
    - logs command output to `benchmarks/results/game_task_reports/arena_validation_run.log`

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/game_task_arena.py` (pass).

---

## 2026-05-17 — Dual-usage optimization pass #2 (expanded routing corpus + retrain + comparative eval)

**Goal:** Improve dual-usage routing outcomes by expanding supervised routing labels beyond the tiny benchmark-only split.

**Commands run:**

- Expanded dataset build:
  - `source .venv/bin/activate && python scripts/ml_workflow.py routing-dataset --benchmark-tasks benchmarks/task_routing_mixed_tasks_v1.json --curated-jsonl data/routing/router_cases_v1.jsonl --curated-jsonl data/routing/router_cases_v2.jsonl --dataset-version 20260518_dualopt --low-confidence-threshold 0.58`
- Retrain classifier:
  - `source .venv/bin/activate && python scripts/ml_workflow.py routing-train --data-dir data/lora/routing_classifier/20260518_dualopt --out-dir training/router_classifier_v1`
- Routing benchmark (hybrid policy, classifier-enabled):
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=hybrid ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 python scripts/ml_workflow.py routing-benchmark --mode both --tasks benchmarks/task_routing_mixed_tasks_v1.json`
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=hybrid ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 python scripts/ml_workflow.py routing-benchmark --mode both --tasks benchmarks/task_routing_tasks.json`
- Multi-agent split/merge validation:
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=hybrid ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_dual_posttrain_v2.json`
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_dual_posttrain_similarity_v2.json`
- Similarity-mode mixed benchmark cross-check:
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD=0.58 python scripts/ml_workflow.py routing-benchmark --mode both --tasks benchmarks/task_routing_mixed_tasks_v1.json`

**Outcomes:**

- Expanded routing dataset succeeded (`run_id=20260518-061115_4383c5`):
  - `rows_total=52` (`train=44`, `valid=5`, `test=3`) with all 8 adapter labels represented.
- Retrain succeeded (`run_id=20260518-061124_88148f`) but emitted numeric warnings in classifier training (`overflow/invalid matmul`), likely due aggressive LR on sparse features:
  - summary: `train_acc=1.0`, `valid_acc=0.4`, `test_acc=0.3333`.
- Hybrid-policy benchmarks improved materially vs prior pass:
  - mixed (`run_id=20260518-061138_459207`): **overall `10/12`**, route `11/12`, adapter `10/12`.
  - core routing tasks (`run_id=20260518-061143_8acfcd`): **overall `11/12`**, route `12/12`, adapter `11/12`.
- Multi-agent orchestration contract remained valid in both modes:
  - hybrid: `split_valid=1.0`, `merge_valid=1.0`, `multi_agent_rows=10`.
  - similarity: `split_valid=1.0`, `merge_valid=1.0`, `multi_agent_rows=35`.
- Similarity-mode mixed benchmark stayed weaker (`run_id=20260518-061208_57da6b`): overall `6/12`.

**Interpretation:**

- For optimizing benchmark correctness now, **hybrid mode + updated classifier is best**.
- For maximizing dual-split frequency, similarity mode still yields more secondary subtasks, but at substantial routing accuracy cost.
- Next technical fix should lower classifier LR / add gradient clipping to eliminate numeric instability and improve validation/test accuracy.

---

## 2026-05-17 — Dual-usage routing optimization run (dataset/train/benchmark + orchestration validation)

**Goal:** Execute dual-usage optimization workflow and verify whether task-building/routing outcomes improved.

**Commands run:**

- `source .venv/bin/activate && python scripts/ml_workflow.py routing-dataset`
- `source .venv/bin/activate && python scripts/ml_workflow.py routing-train --data-dir data/lora/routing_classifier/20260518 --out-dir training/router_classifier_v1`
- `source .venv/bin/activate && python scripts/ml_workflow.py routing-benchmark --mode both --tasks benchmarks/task_routing_mixed_tasks_v1.json`
- Tuned retry:
  - `source .venv/bin/activate && ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD=0.58 python scripts/ml_workflow.py routing-benchmark --mode both --tasks benchmarks/task_routing_mixed_tasks_v1.json`
- `source .venv/bin/activate && python scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_dual_posttrain.json`

**Outcomes:**

- `routing-dataset` succeeded (`run_id=20260518-060422_1404f5`):
  - dataset written under `data/lora/routing_classifier/20260518`,
  - rows: `train=10`, `valid=1`, `test=1` (very small split).
- `routing-train` succeeded (`run_id=20260518-060438_347778`):
  - artifacts refreshed in `training/router_classifier_v1`,
  - reported `train/valid/test accuracy=1.0` on tiny split.
- Mixed dual-label benchmark failed baseline (`run_id=20260518-060443_d31396`):
  - route `8/12`, adapter `5/12`, overall `5/12`.
- Tuned-threshold benchmark improved slightly but still failed (`run_id=20260518-060516_5e94c2`):
  - route `9/12`, adapter `6/12`, overall `6/12`.
- Multi-agent orchestration contract validation succeeded:
  - `benchmarks/results/multi_agent_orchestration_validation_dual_posttrain.json`
  - `rows=50`, `split_valid_rate=1.0`, `merge_valid_rate=1.0`, `multi_agent_rows=35`.

**Interpretation:**

- Task split/merge mechanics are healthy.
- Mixed routing accuracy remains below promotion quality despite small gains from threshold tuning.
- Main blocker is label/data coverage (current routing-train split is too small for reliable generalization).

---

## 2026-05-17 — Added “Most Common Skills Demonstrated” to knob topology

**Goal:** Show what skills a selected knob expresses most often (frequency view), in addition to effect-based center/boundary skill analysis.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` knob topology panel:
  - added new `Most Common Skills Demonstrated` block,
  - computes and displays top frequencies for:
    - `concept::<...>` signals,
    - `bucket::<...>` signals,
    - `concept_bucket::<concept>::<bucket>` subskills,
  - included explicit note clarifying this section is **frequency-based**, not effect-based.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Added Map 2 + Map 3 with tabbed capability explorer flow and relation hotlinks

**Goal:** Build additional manifold/evidence maps with a cleaner, organized UI flow and quick relation navigation.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability explorer (`/view/capability-map`):
  - introduced tabbed map flow:
    - `Map 1: Capability`
    - `Map 2: Skill Manifold`
    - `Map 3: Region Evidence`
  - added `Map 2` manifold visualization:
    - skill points by effect/sign-agreement with support-weighted marker size,
    - region bubbles (`R<id>`) sized by task count and colored by stability sign,
    - click region bubble to open `Map 3` evidence for that region,
    - click skill point to hotlink back to `Map 1` node-level skill inspection.
  - added `Map 3` region evidence panel:
    - region stability/curvature/boundary metrics,
    - specialist-fit summary,
    - task evidence list with concepts and buckets.
  - added relation hotlinks in evidence pane:
    - `Open selected node skills` (returns to node skill topology)
    - `Open selected region in manifold`
  - synchronized tab events and map render lifecycle so map 2/3 update alongside map 1 selection state.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Preserve edge polarity colors on selection; use glow emphasis

**Goal:** Keep positive/negative edge color cues readable when a knob is selected, while still clearly emphasizing connected edges.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map CSS:
  - changed `.edge-line.active` from forced gold stroke to glow-only emphasis,
  - kept each edge’s original stroke color so polarity remains visually obvious.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Explicit positive/negative correlation labeling in knob inspector

**Goal:** Make selected-knob diagnostics explicitly state polarity (positive/negative), not just signed numbers.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map UI:
  - selected-knob correlation list now labels each row as **Positive correlation** or **Negative correlation**,
  - selected-knob header now includes positive/negative correlation counts,
  - knob skill topology rows now label each effect as **Positive effect** or **Negative effect**,
  - skill decoder copy now clearly explains effect polarity semantics.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Knob skill topology decoder for human-readable docs insight

**Goal:** Convert raw code-like skill IDs in capability-map topology into understandable documentation output, including clear center-vs-boundary interpretation.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map UI:
  - added a **Skill decoder** block in `Knob Skill Topology` that explains skill types and metrics in plain language,
  - added per-skill human-readable descriptions (`concept`, `bucket`, `concept_bucket`) instead of ID-only rows,
  - added inferred-center fallback when direct center rows are missing:
    - reports stable-region coverage,
    - reports dominant specialists and buckets from selected-knob task rows,
  - kept boundary-subskill + cross-knob task traceability while improving wording/labels.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Capability map knob deselect UX (Esc + top-right clear button)

**Goal:** Make selected knob state easy to exit using keyboard and explicit UI control.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map view:
  - added top-right clear-selection `×` button inside the map viewport,
  - button appears only when a knob is selected,
  - added `Escape` key handler to clear current knob selection,
  - unified selection-control visibility updates with selection state (`syncSelectionControls`).

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Knob skill topology: center vs boundary skills + cross-knob subskill tasks

**Goal:** Extend capability-map knob selection so each selected knob shows center skills, boundary skills, and related subskill tasks crossing into other knobs.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - extended capability-map payload to include `skills_v1.task_region_assignments` from `skill_manifolds_v1.json`,
  - added new diagnostics dropdown: **Knob Skill Topology**,
  - implemented per-selected-knob skill parsing and grouping:
    - **Center skills**: direct `concept::` / `bucket::` skills for the selected knob,
    - **Boundary skills**: `concept_bucket::...` cross-knob skills touching the selected knob,
  - added **Related subskill tasks from other knobs** listing (sample tasks per boundary skill) and annotated region ids, including unstable boundary-region markers.
  - auto-opens the new skill-topology menu when a knob is selected.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Capability map crop + dropdown menus + stronger knob selection highlight

**Goal:** Fix capability-map viewport cropping, collapse side sections into dropdown-style menus by default, and make selected knob state unmistakable.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map page:
  - added dynamic SVG `viewBox` fitting from rendered node/label bounds to remove large trailing empty map area,
  - converted diagnostics/similarity/skills/region blocks into `<details>` dropdown menus (collapsed by default),
  - upgraded selected-node styling (strong yellow ring + glow + label emphasis),
  - added edge-state styling to emphasize connected edges for the selected node and dim unrelated edges,
  - auto-opens the Selected Knob Correlations dropdown when a node is clicked.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-17 — Capability map layout: move skills/region below map

**Goal:** Improve docs-site capability-map structure by moving secondary analytics sections below the map area.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map page:
  - moved `Skills Correlation (v1)` and `Region Stability (v1)` out of the right diagnostics stack,
  - added a new `below-grid` section rendered under the map/diagnostics row,
  - made the new section responsive (two columns on wide screens, one column on narrow screens).

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Restarted docs server; confirmed it is serving on `http://0.0.0.0:8787`.

---

## 2026-05-17 — Non-routed baseline run on 180-task ACI suite

**Goal:** Run the same 180-task specialist benchmark corpus without routed-policy env overrides and compare against the prior hierarchical-policy-tagged run.

**Command run:**

- `.venv/bin/python scripts/run_game_benchmark.py --tasks benchmarks/specialist_benchmark_tasks.json --output-jsonl benchmarks/results/aci_180_non_routed_rows_latest.jsonl`

**Outcome:**

- Summary: `133/180` (`74%`)
- Capability Index: `88.5/100` (`correctness 88.2`, `instruction 99.4`, `concision 72.1`, `speed 92.4`)
- Advanced ACI: `81.8/100` (`weighted 88.8`, `domain-balance 67.0`, `multi-domain 71.7`)
- Runtime: ~180s

**Comparison vs prior run (`aci_180_hier_policy_rows_latest`):**

- Pass count: unchanged (`133/180`)
- Advanced ACI: `81.9` -> `81.8` (effectively unchanged)
- Main delta was speed subscore (`94.7` -> `92.4`) from longer generation latency; quality dimensions stayed the same.

---

## 2026-05-17 — Re-run 180-task specialist ACI load under default hierarchical routing policy

**Goal:** Re-run the full 180-task specialist benchmark corpus after confirming hierarchical adapter-first routing remains default policy.

**Commands run:**

- Attempted workflow run with unsupported passthrough arg (rejected by CLI):
  - `python3 scripts/ml_workflow.py benchmark --tasks benchmarks/specialist_benchmark_tasks.json --output-jsonl ...`
- Executed full benchmark directly with venv Python and hierarchical-routing env:
  - `ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_HIERARCHICAL_ROUTING_ENABLED=1 .venv/bin/python scripts/run_game_benchmark.py --tasks benchmarks/specialist_benchmark_tasks.json --output-jsonl benchmarks/results/aci_180_hier_policy_rows_latest.jsonl`

**Outcomes:**

- Full 180-task run completed in ~149s.
- Summary: `133/180` (`74%`).
- Capability Index: `88.7/100` (`correctness 88.2`, `instruction 99.4`, `concision 72.1`, `speed 94.7`).
- Advanced ACI: `81.9/100` (`weighted 88.9`, `domain-balance 67.0`, `multi-domain 71.7`).
- Per-task rows written to:
  - `benchmarks/results/aci_180_hier_policy_rows_latest.jsonl`

---

## 2026-05-17 — Expand all specialist suites to 30 + advanced multi-domain ACI scoring

**Goal:** Bring every specialist benchmark suite to `30` tasks (matching loading-screen), feed those tasks into the main benchmark run, and upgrade ACI-style scoring to reward multi-domain performance.

**Changed files:**

- Updated `scripts/build_mass_specialist_benchmark_tasks.py`:
  - expanded generated suites to `30` rows each for `hud_status`, `economy_tooltip`, `combat_risk`, `save_load_api_guard`, and `ai_planning_explanation`,
  - added explicit `domains` metadata to generated tasks,
  - emits an aggregate `benchmarks/specialist_benchmark_tasks.json` from all six `*_mass_tasks_v1.json` files (now `180` total rows).
- Regenerated benchmark suites:
  - `benchmarks/hud_status_mass_tasks_v1.json`
  - `benchmarks/economy_tooltip_mass_tasks_v1.json`
  - `benchmarks/combat_risk_mass_tasks_v1.json`
  - `benchmarks/save_load_api_guard_mass_tasks_v1.json`
  - `benchmarks/ai_planning_explanation_mass_tasks_v1.json`
  - `benchmarks/specialist_benchmark_tasks.json`
- Updated `scripts/run_game_benchmark.py`:
  - added task domain inference + task weighting (difficulty/category/multi-domain),
  - added `advanced_aci` summary combining weighted capability, domain-balance, and multi-domain mastery,
  - includes `task_weight`, `domains`, and `multi_domain` in per-task JSONL rows.

**Commands run:**

- `python3 scripts/build_mass_specialist_benchmark_tasks.py`
- `python3 -m py_compile scripts/build_mass_specialist_benchmark_tasks.py scripts/run_game_benchmark.py`
- `python3 -c "import json,glob; ..."` task-count checks
- `python3 -c "import json,collections; ..."` specialist + multi-domain distribution checks

**Outcomes:**

- All specialist mass suites now report `30` rows each.
- `benchmarks/specialist_benchmark_tasks.json` now contains `180` rows (`30` per specialist).
- Aggregate benchmark set now includes substantial cross-domain coverage (`134` rows flagged as transfer/multi-domain-like by validation script).

---

## 2026-05-17 — Restore arena-acceptance runner + re-check split/merge routing

**Goal:** Unblock `ml_workflow.py arena-acceptance` after missing script error, then re-verify multi-agent split/merge routing behavior.

**Changed files:**

- Added `scripts/run_arena_acceptance_tests.py`:
  - restored expected `ml_workflow.py arena-acceptance` entrypoint (`create -> generate -> apply -> verify -> preview -> cleanup`),
  - added deterministic acceptance summary output (`arena_acceptance_summary_v2`) with `passed_count` and `arena_capability` payload,
  - printed ACI headline line compatible with workflow parsing (`=== Arena Capability Index: ... ===`),
  - supported existing workflow flags (`--task-id`, `--suite`, `--summary`, `--no-cleanup`, retries/context/token controls).

**Verification / runs:**

- `python3 -m py_compile scripts/run_arena_acceptance_tests.py` (pass).
- Focused arena acceptance check:
  - `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 --task-id loading-screen-polish --task-id hud-status-summary --tsc-retries 0 --max-tokens 4096`
  - run id: `20260518-041101_2920b1`
  - result: `passed_count=0/2`, `ACI=20.0`; row details written to `benchmarks/results/runs/20260518-041101_2920b1/arena_acceptance_summary.json`.
- Multi-agent split/merge validation:
  - `python3 scripts/validate_multi_agent_orchestration.py --tasks benchmarks/ui_merge_dual_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_20260518_focus.json`
  - result: `42/42` split-valid, `42/42` merge-valid, `31` prompts using multi-agent (secondary subtask emitted).

---

## 2026-05-17 — Add low-complexity loading+combat dual-domain task set (50)

**Goal:** Create a larger low-complexity mixed-domain benchmark focused on strongest specialists (`loading_screen` + `combat_risk`) for split-routing and multi-agent orchestration checks.

**Changed files:**

- Added `benchmarks/hud_combat_field_flow_low_tasks_v1.json` with **50** tasks:
  - each task is dual-tagged with specialists `loading_screen` + `combat_risk`,
  - domains are explicitly multi-domain (`loading`, `combat`, `ui`),
  - prompts are low-complexity microcopy/UI guidance spanning both domains in one request,
  - expectation blocks enforce dual-domain signal (`loading` + combat/risk keywords), min-char checks, and plain-text guardrails where relevant.

**Verification / runs:**

- `python3 scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_loading_combat_dual_low_20260518.json`
- Result: `rows=50`, `split_valid_rows=50`, `merge_valid_rows=50`, `multi_agent_rows=40` (split/merge valid rate `1.0`).

---

## 2026-05-17 — Replace synthetic loading+combat set with real field-flow HUD+combat tasks

**Goal:** Replace overly synthetic low-complexity prompts with realistic, in-game graphical field needs using tightly related specialists.

**Changed files:**

- Replaced `benchmarks/hud_combat_field_flow_low_tasks_v1.json`:
  - switched specialist pairing from broad loading/combat copy to `**hud_status` + `combat_risk`** (shared in-match field UI domain),
  - rebuilt as 50 low-complexity prompts grounded in real gameplay flow states (`scout_to_contact`, `march_to_engage`, `terrain_warning`, `morale_breakpoint`, `flank_exposure`, `siege_pressure`, `supply_strain`, `reinforcement_arrival`, `post_skirmish`, `night_visibility`),
  - each task now enforces practical constraints for existing UI surfaces (in-match HUD/battle panel, no new routes/systems), with metadata `real_need: in_match_field_ui`.

**Verification / runs:**

- `python3 scripts/validate_multi_agent_orchestration.py --tasks benchmarks/hud_combat_field_flow_low_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_loading_combat_dual_low_20260518_v2.json`
- Result: `rows=50`, `split_valid_rows=50`, `merge_valid_rows=50`, `multi_agent_rows=35` (split/merge valid rate `1.0`).

## 2026-05-17 — Concision-focused specialist tuning run (`ai_planning_explanation`) + aggregate recheck

**Goal:** Execute a concision-focused tuning loop and measure impact on overall specialist capability.

**Plan executed:**

1. Baseline `ai_planning_explanation` specialist benchmark on current `cycle1`.
2. Refresh `ai_planning` dataset with benchmark-ingested rows emphasizing constrained prompts.
3. Train `ai_planning_explanation/cycle2` from `cycle1`.
4. Re-benchmark `ai_planning_explanation` and then re-run full six-specialist aggregate with:
  - `save_load_api_guard=cycle2`
  - `ai_planning_explanation=cycle2`

**Commands run:**

- `.venv/bin/python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --tasks benchmarks/specialist_benchmark_tasks.json --specialist ai_planning_explanation --adapter-path checkpoints/adapters/ai_planning_explanation/cycle1 --output-jsonl benchmarks/results/ai_planning_before_concision_rows_20260518.jsonl`
- `python3 scripts/ml_workflow.py ai-planning-dataset --benchmark-tasks-json benchmarks/ai_planning_explanation_mass_tasks_v1.json --max-benchmark-rows 30 --min-train-core-rows 140`
- `python3 scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ai_planning_explanation/cycle2 -- --data data/lora/adapters/ai_planning_explanation_specialist --iters 80 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/ai_planning_explanation/cycle1/adapters.safetensors`
- `.venv/bin/python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --tasks benchmarks/specialist_benchmark_tasks.json --specialist ai_planning_explanation --adapter-path checkpoints/adapters/ai_planning_explanation/cycle2 --output-jsonl benchmarks/results/ai_planning_after_concision_rows_20260518.jsonl`
- Full aggregate sweep with overrides, outputting `benchmarks/results/agi_specialist_summary_20260518_after_concision_tuning.json`.

**Run artifacts:**

- Dataset run: `benchmarks/results/runs/20260518-000944_c11c26/` (exit `0`)
- Train run: `benchmarks/results/runs/20260518-000944_7cd478/` (exit `0`)
- New adapter: `checkpoints/adapters/ai_planning_explanation/cycle2/adapters.safetensors`
- Aggregate summary after tuning: `benchmarks/results/agi_specialist_summary_20260518_after_concision_tuning.json`

**Observed impact:**

- `ai_planning_explanation` benchmark:
  - before (`cycle1`): `1/2`, capability `63.1`
  - after (`cycle2`): `2/2`, capability `78.6`
- Full specialist aggregate (with save-load `cycle2` + ai-planning `cycle2`):
  - pass rate: `75.0%` (`9/12`)
  - capability index: `75.8`
- Concision remained the main bottleneck (`18.27/100`) despite correctness gains.

---

## 2026-05-17 — Targeted save/load retrain (`cycle2`) + all-specialist aggregate recheck

**Goal:** Test whether boosting `save_load_api_guard` lifts overall specialist aggregate capability.

**Commands run:**

- `python3 scripts/ml_workflow.py save-load-dataset --benchmark-tasks-json benchmarks/save_load_api_guard_mass_tasks_v1.json --max-benchmark-rows 12 --min-train-core-rows 120`
- `python3 scripts/ml_workflow.py train --adapter-path checkpoints/adapters/save_load_api_guard/cycle2 -- --data data/lora/adapters/save_load_api_guard_specialist --iters 80 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/save_load_api_guard/cycle1/adapters.safetensors`
- Full specialist aggregate sweep with `.venv/bin/python scripts/run_game_benchmark.py` over all six specialist families, overriding save/load adapter to `cycle2`.

**Run artifacts:**

- Dataset run: `benchmarks/results/runs/20260518-000021_7b4b01/` (exit `0`)
- Train run: `benchmarks/results/runs/20260518-000021_4a7f02/` (exit `0`)
- New adapter: `checkpoints/adapters/save_load_api_guard/cycle2/adapters.safetensors`
- Aggregate summary (before): `benchmarks/results/agi_specialist_summary_20260517.json`
- Aggregate summary (after save/load `cycle2`): `benchmarks/results/agi_specialist_summary_20260518_after_save_load_cycle2.json`

**Observed impact:**

- Save/load specialist moved from `0/2` (`cycle1`) to `1/2` (`cycle2`) with capability `64.9` → `70.4`.
- Full specialist aggregate moved:
  - pass rate: `58.33%` (`7/12`) → `66.67%` (`8/12`)
  - capability index: `72.3` → `73.22`
- Main remaining bottleneck is concision (aggregate `~18/100`) plus unresolved misses in combat summary-line and ai-planning intent-label tasks.

---

## 2026-05-17 — Targeted documentation training pass (`cycle4`)

**Goal:** Run a targeted documentation specialist refresh and new training cycle from the current `cycle3` adapter.

**Commands run:**

- `python3 scripts/ml_workflow.py documentation-dataset`
- `python3 scripts/ml_workflow.py train --adapter-path checkpoints/adapters/documentation/cycle4 -- --data data/lora/adapters/documentation_specialist --iters 60 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/documentation/cycle3/adapters.safetensors`

**Run artifacts:**

- Dataset refresh: `benchmarks/results/runs/20260517-230055_22ed67/` (exit `0`)
- Train run: `benchmarks/results/runs/20260517-230055_5e5bc9/` (exit `0`)
- New adapter: `checkpoints/adapters/documentation/cycle4/adapters.safetensors`

**Train metrics (from manifest trajectory):**

- Final train loss: `0.063` @ iter `60`
- Final val loss: `1.456` @ iter `60`
- Best val loss: `0.071` @ iter `40`
- Peak memory: `6.198 GB`

---

## 2026-05-17 — Implement skills/manifold extraction artifacts + dashboard visibility

**Goal:** Implement data + skills extraction first (before routing changes) and expose specific skill correlations and region stability in the capability visualizer.

**Changed files:**

- Added `scripts/extract_skills_v1.py`:
  - parses latest specialist mass benchmark outcomes,
  - builds candidate skills (bucket, concept, concept×bucket),
  - computes support/effect/bootstrap stability stats,
  - emits retained and exploratory skill sets,
  - builds task/specialist/skill manifold graph with curvature-aware weighting,
  - computes region stability (`stability_score`, `stability_z`) and specialist-region fit.
- Updated `scripts/private_dashboard_server.py`:
  - loads `skills_v1` and manifold artifacts into `/api/capability-map`,
  - renders new `Skills Correlation (v1)` and `Region Stability (v1)` sections in `/view/capability-map`.

**Commands run:**

- `python3 scripts/extract_skills_v1.py`
- `python3 -m py_compile scripts/private_dashboard_server.py scripts/extract_skills_v1.py`

**Artifacts generated:**

- `data/routing/skills_v1.json`
- `data/routing/specialist_skill_profiles_v1.json`
- `data/routing/skill_manifolds_v1.json`
- `benchmarks/results/skills_extraction_report_v1.md`

**Current extraction snapshot:**

- Retained strict skills: `1` (`bucket::constraints`).
- Exploratory skills (relaxed): `22` (includes `concept_bucket::Planning Intent::constraints`, `concept_bucket::UI Hierarchy::ui_change`, `concept::Schema/API`, etc.).
- Regions discovered: `5`, with stability-z ranking available in the visualizer.

**Verification:**

- Lint check returned no errors for changed scripts.
- Capability map payload smoke check confirms skills/manifold fields are present (`skills_available=True`).

---

## 2026-05-17 — Extend skills spec with curvature manifolds + hierarchical routing

**Goal:** Incorporate curvature-based stability regions and specialist/task manifold routing into the skills extraction plan, plus a hierarchical output-composition policy.

**Changed files:**

- Updated `docs/SKILLS_EXTRACTION_V1.md`:
  - added curvature/manifold section (edge curvature, region stability, specialist-region fit matrix),
  - added region-aware routing behavior for unstable boundaries,
  - added hierarchical routing + composable output plan (sub-operation assignment + deterministic merge rules),
  - added new thresholds/deliverables for manifold artifacts and routing policy artifacts,
  - added evaluation gate/data requirements for stability-calibrated routing and multi-skill composition tasks.

**Notes:**

- This reframes routing around stable behavior regions and skill manifolds rather than prompt-angle similarity alone.
- Spec now explicitly supports multi-specialist output assembly without naive adapter stacking.

---

## 2026-05-17 — Draft `skills_v1` extraction spec (outcome-first)

**Goal:** Define a concrete skill extraction framework that moves routing from prompt-similarity toward behavior/skill evidence, including expanded skill families and data requirements.

**Changed files:**

- Added `docs/SKILLS_EXTRACTION_V1.md` with:
  - outcome-first extraction pipeline (tensor -> graph -> Laplacian -> candidate skills -> effect filtering),
  - support/stability thresholds for retained skills,
  - specialist skill profiling and unknown/OOD routing hooks,
  - expanded candidate skill families beyond current nodes (format discipline, schema/API guarding, planning rationale, UI information design, cross-domain transfer),
  - deliverables (`skills_v1.json`, specialist skill profiles, extraction report) and adoption gates.

**Notes:**

- Spec explicitly treats prompt-angle similarity as diagnostic only, not ownership ground truth.
- Spec calls out likely need for additional benchmark data and curated unknown prompt reviews to avoid overfitting current node granularity.

---

## 2026-05-17 — Similarity routing fixes: fallback/docs prototypes + tie-margin gate

**Goal:** Implement two corrective changes requested for similarity routing: (1) include `general_fallback` + `documentation` prototype support, and (2) add a near-tie cosine margin gate to avoid forced misassignment.

**Changed files:**

- Updated `scripts/model_router.py`:
  - expanded similarity prototype sources to include route-labeled task files (`expected_adapter_id`) so `general_fallback` and `documentation` build lexical prototypes,
  - added similarity calibration knobs:
    - `ROUTER_SIMILARITY_MIN_SCORE` (default `0.10`)
    - `ROUTER_SIMILARITY_MIN_MARGIN` (default `0.03`)
  - added tie-margin rejection in `_classify_adapter_similarity(...)`:
    - if `best_score < min_score` -> fallback,
    - if `best_score - second_score < min_margin` -> fallback with explicit ambiguous reason.

**Verification:**

- `python3 -m py_compile scripts/model_router.py` (pass).
- Lint check for `scripts/model_router.py` returned no errors.
- Prototype presence check:
  - `general_fallback` prototype terms: `71`
  - `documentation` prototype terms: `31`

**Post-fix reruns (similarity mode):**

- AGI routing suite (`benchmarks/task_routing_tasks.json`):
  - `route 12/12`, `adapter 9/12`, `overall 9/12` (`75%`)
  - artifact: `benchmarks/results/routing_agi_similarity_summary_v2.json`
- Mixed routing suite (`benchmarks/task_routing_mixed_tasks_v1.json`):
  - `route 9/12`, `adapter 10/12`, `overall 8/12` (`66.7%`)
  - artifact: `benchmarks/results/routing_mixed_similarity_summary_v2.json`

**Impact:** Fixes recover major regression from prior similarity run and materially improve adapter correctness on both AGI and mixed suites.

---

## 2026-05-17 — Mixed + AGI routing benchmark under similarity alignment

**Goal:** Evaluate whether prompt-similarity alignment (`ROUTER_ADAPTER_SELECTION_MODE=similarity`) improves routing on (1) a new mixed prompt suite and (2) the existing AGI-style routing task suite, then identify task-level agent reassignment deltas.

**Changed files:**

- Added `benchmarks/task_routing_mixed_tasks_v1.json` (12 mixed-domain routing tasks).

**Commands run:**

- Baseline mixed:
  - `python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_mixed_tasks_v1.json --mode both --output-jsonl benchmarks/results/routing_mixed_baseline_rows.jsonl --summary-json benchmarks/results/routing_mixed_baseline_summary.json`
- Similarity mixed:
  - `ROUTER_ADAPTER_SELECTION_MODE=similarity python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_mixed_tasks_v1.json --mode both --output-jsonl benchmarks/results/routing_mixed_similarity_rows.jsonl --summary-json benchmarks/results/routing_mixed_similarity_summary.json`
- Baseline AGI-style routing:
  - `python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_tasks.json --mode both --output-jsonl benchmarks/results/routing_agi_baseline_rows.jsonl --summary-json benchmarks/results/routing_agi_baseline_summary.json`
- Similarity AGI-style routing:
  - `ROUTER_ADAPTER_SELECTION_MODE=similarity python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_tasks.json --mode both --output-jsonl benchmarks/results/routing_agi_similarity_rows.jsonl --summary-json benchmarks/results/routing_agi_similarity_summary.json`

**Run artifacts:**

- Mixed baseline run: `benchmarks/results/runs/20260517-222755_492722/`
- Mixed similarity run: `benchmarks/results/runs/20260517-222755_17fbe6/`
- AGI baseline run: `benchmarks/results/runs/20260517-222755_f3a594/`
- AGI similarity run: `benchmarks/results/runs/20260517-222756_3e45bb/`

**Outcomes:**

- Mixed suite (`task_routing_mixed_tasks_v1.json`):
  - Baseline: route `7/12` (58.3%), adapter `7/12` (58.3%), overall `5/12` (41.7%)
  - Similarity: route `9/12` (75.0%), adapter `5/12` (41.7%), overall `4/12` (33.3%)
- AGI suite (`task_routing_tasks.json`):
  - Baseline: route `12/12`, adapter `12/12`, overall `12/12` (100%)
  - Similarity: route `9/12` (75.0%), adapter `0/12`, overall `0/12`

**Reassignment summary:**

- Mixed suite: `7/12` tasks changed adapter and/or route vs baseline.
- AGI suite: `12/12` tasks changed adapter and/or route vs baseline.
- Similarity mode over-assigns to game specialists (`loading_screen`, `save_load_api_guard`) for many general/docs tasks, indicating prototype coverage/class balance mismatch for routing labels.

**Conclusion:** Similarity alignment improved route-only hit rate on mixed prompts but degraded adapter assignment robustness overall, and severely regressed the AGI suite vs baseline deterministic policy. Dataset/prototype rework + calibration gates are required before promotion.

---

## 2026-05-17 — Add unknown-prompt review queue to router decisions

**Goal:** Automatically capture low-confidence or fallback-routed prompts into a review queue so unmarked/ambiguous prompts can be triaged and labeled.

**Changed files:**

- Updated `scripts/model_router.py`:
  - added unknown-review queue appender in `RoutingPolicy.decide()` with JSONL rows,
  - queue triggers when:
    - predicted adapter is `general_fallback`, or
    - confidence is below configured threshold, or
    - similarity mode explicitly reports low cosine similarity,
  - appends routing metadata (`adapter`, `route`, `confidence`, `ambiguity`, token estimates, reason, selection mode),
  - tags decision reason with `unknown_review=queued` (or `queue_failed` on write error).

**New/used env controls:**

- `ROUTER_UNKNOWN_REVIEW_ENABLED` (default `1`)
- `ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD` (default `0.62`)
- `ROUTER_UNKNOWN_REVIEW_JSONL` (default `data/routing/unknown_prompts_review_queue.jsonl`)

**Verification:**

- `python3 -m py_compile scripts/model_router.py` (pass).
- Lint check for `scripts/model_router.py` returned no errors.
- Smoke test with high review threshold confirmed queue growth and row append at:
  - `data/routing/unknown_prompts_review_queue.jsonl`

---

## 2026-05-17 — Add prompt-similarity adapter routing mode

**Goal:** Enable routing to specialists by prompt similarity (cosine over lexical prototypes) so adapter selection can run from direct prompt-domain similarity instead of keyword/classifier-only selection.

**Changed files:**

- Updated `scripts/model_router.py`:
  - added sparse lexical cosine prototype builder from benchmark prompt corpora,
  - added `_classify_adapter_similarity(...)` with confidence scoring from top/second cosine scores,
  - added router mode toggle `ROUTER_ADAPTER_SELECTION_MODE`:
    - `hybrid` (default; existing classifier+lexical fallback behavior),
    - `similarity` (new cosine prototype mode),
    - `lexical` (keyword-only mode).

**Verification:**

- `python3 -m py_compile scripts/model_router.py` (pass).
- Lint check for `scripts/model_router.py` returned no errors.
- Similarity-mode smoke run (`PYTHONPATH=scripts ROUTER_ADAPTER_SELECTION_MODE=similarity`) returned expected specialist picks for loading/combat/save-load/AI-planning prompts.

---

## 2026-05-17 — Add task-angle cosine similarity to capability visualizer

**Goal:** Extend the capability map with exact task-to-task similarity inspection using cosine-angle similarity (`cos(theta) = dot(a,b)/(||a||*||b||)`).

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - extended `_capability_map_payload()` to build per-task vectors (concept one-hot + bucket one-hot),
  - added `task_similarity` payload section with:
    - formula metadata,
    - dimensions (`concepts`, `buckets`),
    - full task vectors,
    - top task pairs by cosine similarity,
  - updated `/view/capability-map` UI with interactive similarity inspector:
    - task A / task B selectors,
    - exact cosine + angle readout in degrees,
    - nearest-neighbor list for selected task A.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.
- Payload smoke test:
  - `ok=True`,
  - task vectors present (`90`),
  - top similarity pairs present (`240`),
  - formula string returned as expected.

---

## 2026-05-17 — Move capability visualizer to dedicated dashboard tab

**Goal:** Make the capability visualizer a first-class top-level tab instead of a link inside Documentation Explorer.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` dashboard HTML:
  - added top nav tab: `Capability Map`,
  - removed capability-map link from `Docs Snapshot`,
  - added new panel `panel-capability-map` with embedded iframe to `/view/capability-map`,
  - kept an optional `Open in full page` link for standalone view.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.

---

## 2026-05-17 — Targeted AI-planning lexical patch + successful retrain (`2/2`)

**Goal:** Apply the same targeted lexical-anchor strategy used for HUD to `ai_planning_explanation`, after the prior `mock_aug_v1` run plateaued at `1/2`.

**Changed files:**

- Updated `benchmarks/ai_planning_explanation_mass_tasks_v1.json`:
  - expanded from 12 to 20 tasks,
  - added intent-label lexical constraints (`defend`, `expand`, `reinforce`, `scout`, `attack`) and stricter plain-text/comma/line-shape prompts.

**Commands run:**

- Rebuild AI-planning dataset with expanded benchmark prompts:
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py ai-planning-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --benchmark-tasks-json benchmarks/ai_planning_explanation_mass_tasks_v1.json --out-dir data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v2 --max-core-rows 160 --max-benchmark-rows 240 --min-train-core-rows 220`
- Train + evaluate patched AI-planning dataset:
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ai_planning_explanation/mock_aug_v2_cycle1 --evaluate --bench-specialist ai_planning_explanation -- --data data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v2 --iters 80`

**Outcomes:**

- Patched dataset `data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v2`:
  - `core_pairwise=160`, `core_benchmark_synth=20`,
  - split `train/valid/test=220/18/18`,
  - run: `benchmarks/results/runs/20260517-222532_9c9985/`.
- Retrained adapter `checkpoints/adapters/ai_planning_explanation/mock_aug_v2_cycle1`:
  - benchmark improved to `**2/2`** (`100%`),
  - capability `**78.6/100`**,
  - run: `benchmarks/results/runs/20260517-222540_b1feba/`.

---

## 2026-05-17 — Targeted training batch (`ai_planning_explanation`, `combat_risk`, `economy_tooltip`)

**Goal:** Execute the requested specialist training runs in priority order with immediate benchmark evaluation.

**Commands run:**

- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ai_planning_explanation/mock_aug_v1_cycle1 --evaluate --bench-specialist ai_planning_explanation -- --data data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v1 --iters 80`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/combat_risk/mock_aug_v1_cycle1 --evaluate --bench-specialist combat_risk -- --data data/lora/adapters/combat_risk_specialist_mock_aug_v1 --iters 80`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/economy_tooltip/mock_aug_v1_cycle1 --evaluate --bench-specialist economy_tooltip -- --data data/lora/adapters/economy_tooltip_specialist_mock_aug_v1 --iters 80`

**Outcomes (all trains exit `0`; benchmark step determines final run status):**

- `ai_planning_explanation`:
  - run: `benchmarks/results/runs/20260517-213906_cbe76a/`
  - benchmark: `1/2`, capability `63.1/100`
  - fail mode: `ai-planning-intent-labels` missing strict intent keyword tokens.
- `combat_risk`:
  - run: `benchmarks/results/runs/20260517-214124_27cdfd/`
  - benchmark: `1/2`, capability `70.0/100`
  - fail mode: `combat-risk-summary-line` missing literal `attacker`/`defender`.
- `economy_tooltip`:
  - run: `benchmarks/results/runs/20260517-214335_e31191/`
  - benchmark: `0/2`, capability `60.6/100`
  - fail modes: missing required literal economy anchors (`gold`, `income`/`trade-off` token family).

**Next correction direction:** all three now present the same pattern seen before HUD patching—strong instruction/speed, but failing strict lexical anchors. Use targeted benchmark-task augmentation (literal token/casing constraints) per specialist before the next retrain cycle.

---

## 2026-05-17 — Promote HUD specialist champion to mock-aug v2 checkpoint

**Goal:** Promote the successful HUD retrain output to active routing registry default and identify the next highest-value specialist improvement target.

**Changed files:**

- Updated `training/adapter_registry_v1.json`:
  - `hud_status.adapter_path`: `checkpoints/adapters/hud_status/mock_aug_v2_cycle1`
  - `hud_status.lineage`: `hud_status:v4:mock_aug_v2_cycle1`
  - kept `promotion_state: champion`.
- Updated `docs/PROJECT_STATE.md` router registry status row to reflect HUD promotion and current next-candidate specialists.

**Rationale / outcomes:**

- HUD now has verified benchmark pass on the specialist suite (`2/2`, capability `85.5/100`) in `benchmarks/results/runs/20260517-213417_7b1585/`.
- Next targeted specialist run by urgency remains `ai_planning_explanation` (latest specialist benchmark `3/12`), followed by `combat_risk` (`5/12`) and `economy_tooltip` (`8/12`).

---

## 2026-05-17 — Disable dashboard viewer password gate

**Goal:** Remove the HTTP Basic auth gate so the visualization dashboard opens directly without login prompts.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - changed `_basic_ok(...)` to always allow viewer access,
  - removed stale startup warning about missing viewer password,
  - kept ingestion auth behavior unchanged (`/api/ingest` still requires bearer token).

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.

---

## 2026-05-17 — Targeted HUD data patch (terse/casing) + successful retrain

**Goal:** Address HUD benchmark stagnation (`1/2` despite capability gains) by increasing HUD baseline-task density and adding tighter terse/casing/value constraints.

**Changed files:**

- Updated `benchmarks/hud_status_mass_tasks_v1.json`:
  - expanded HUD mass tasks from 12 to 20 prompts,
  - added targeted terse-label tasks focused on exact casing (`Morale`, `Supply`, `Risk`) and compact value formats.

**Commands run:**

- Baseline benchmark (pre-patch):
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/hud_status/cycle3 --specialist hud_status`
- Rebuild HUD dataset with patched task file:
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py hud-status-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --benchmark-tasks-json benchmarks/hud_status_mass_tasks_v1.json --out-dir data/lora/adapters/hud_status_specialist_mock_aug_v2 --max-core-rows 160 --max-benchmark-rows 240 --min-train-core-rows 220`
- Train + evaluate patched HUD dataset:
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/hud_status/mock_aug_v2_cycle1 --evaluate --bench-specialist hud_status -- --data data/lora/adapters/hud_status_specialist_mock_aug_v2 --iters 80`

**Outcomes:**

- Pre-patch baseline (`cycle3`): `1/2`, capability `62.5/100` (`benchmarks/results/runs/20260517-211625_62d13e/`).
- Intermediate mock-aug v1 training (before this patch) had `1/2` with better capability `79.3/100` (`benchmarks/results/runs/20260517-211713_100bb4/`), indicating quality gains despite one strict lexical miss.
- Patched HUD dataset `data/lora/adapters/hud_status_specialist_mock_aug_v2` now has:
  - source counts: `core_pairwise=160`, `core_benchmark_synth=20`,
  - split counts: `train/valid/test=220/18/18`
  - run: `benchmarks/results/runs/20260517-213405_b8b5ba/`.
- Post-patch retrain adapter `checkpoints/adapters/hud_status/mock_aug_v2_cycle1` reached:
  - benchmark `2/2` (pass),
  - capability `85.5/100`,
  - run: `benchmarks/results/runs/20260517-213417_7b1585/`.

---

## 2026-05-17 — Mock specialist training corpus generator + workflow wiring

**Goal:** Create trainable mock work for specialist dataset building, especially where pairwise winner coverage is thin (`hud_status`, `combat_risk`, `ai_planning_explanation`).

**Changed files:**

- Added `scripts/build_mock_specialist_training_data.py`:
  - emits pairwise-compatible JSONL rows (`task`, `winner_output`, weak `loser_output`, metadata),
  - supports repeated `--task-file`,
  - supports deterministic generation via `--seed` and `--repeats-per-task`,
  - defaults to `benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl`.
- Updated `scripts/ml_workflow.py`:
  - new subcommand `mock-specialist-pairwise`,
  - run artifacts + `docs/run_history.md` integration through normal workflow path.
- Updated docs:
  - `docs/WORKFLOW.md` command table + usage examples,
  - `docs/PROJECT_STATE.md` recent updates section.

**Commands run:**

- `python3 -m py_compile scripts/build_mock_specialist_training_data.py scripts/ml_workflow.py`
- `.venv/bin/python scripts/ml_workflow.py mock-specialist-pairwise --task-file benchmarks/specialist_benchmark_tasks.json --task-file benchmarks/loading_screen_mass_tasks_v1.json --task-file benchmarks/hud_status_mass_tasks_v1.json --task-file benchmarks/economy_tooltip_mass_tasks_v1.json --task-file benchmarks/combat_risk_mass_tasks_v1.json --task-file benchmarks/save_load_api_guard_mass_tasks_v1.json --task-file benchmarks/ai_planning_explanation_mass_tasks_v1.json --repeats-per-task 4 --output-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl`
- `.venv/bin/python scripts/ml_workflow.py hud-status-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --out-dir data/lora/adapters/hud_status_specialist_mock_aug_v1 --max-core-rows 140 --min-train-core-rows 180`
- `.venv/bin/python scripts/ml_workflow.py economy-tooltip-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --out-dir data/lora/adapters/economy_tooltip_specialist_mock_aug_v1 --max-core-rows 140 --min-train-core-rows 180`
- `.venv/bin/python scripts/ml_workflow.py combat-risk-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --out-dir data/lora/adapters/combat_risk_specialist_mock_aug_v1 --max-core-rows 140 --min-train-core-rows 180`
- `.venv/bin/python scripts/ml_workflow.py ai-planning-dataset --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --out-dir data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v1 --max-core-rows 140 --min-train-core-rows 180`

**Outcomes:**

- Workflow run recorded at `benchmarks/results/runs/20260517-073412_c4cd4d/` (exit `0`).
- Generated mock pairwise corpus contains `408` rows:
  - `benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl`.
- Output is immediately consumable by specialist dataset builders via `--pairwise-jsonl`.
- Built augmented specialist datasets (all exit `0` and tracked in `docs/run_history.md`):
  - `benchmarks/results/runs/20260517-073503_012dc5/` → `data/lora/adapters/hud_status_specialist_mock_aug_v1` (`train/valid/test`: `180/15/15`, core pairwise `140`)
  - `benchmarks/results/runs/20260517-073503_ce1c06/` → `data/lora/adapters/economy_tooltip_specialist_mock_aug_v1` (`180/15/15`, core pairwise `140`)
  - `benchmarks/results/runs/20260517-073503_3a8621/` → `data/lora/adapters/combat_risk_specialist_mock_aug_v1` (`180/15/15`, core pairwise `140`)
  - `benchmarks/results/runs/20260517-073503_3b02ab/` → `data/lora/adapters/ai_planning_explanation_specialist_mock_aug_v1` (`180/15/15`, core pairwise `140`)

---

## 2026-05-16 — Dual-specialist merge experiment (`loading_screen` + `hud_status`)

**Goal:** Execute a concrete merge test for two specialists with three variants: (1) rank-8 merged baseline, (2) double-rank method, and (3) weighted-insert method; then benchmark all on a unified task suite.

**Changed files/artifacts:**

- Added `training/lora_qwen25_coder_7b_rank8.yaml` (LoRA rank-8 training config).
- Created merged specialist datasets:
  - `data/lora/adapters/ui_merge_dual_rank8/` (balanced 50/50 source mix)
  - `data/lora/adapters/ui_merge_dual_weighted/` (weighted insert 70/30 toward `loading_screen`)
- Created merged benchmark task suite:
  - `benchmarks/ui_merge_dual_tasks_v1.json` (loading + HUD mass tasks, unified under specialist id `ui_merge_dual`)
- Trained adapters:
  - `checkpoints/adapters/ui_merge_dual/rank8_balanced`
  - `checkpoints/adapters/ui_merge_dual/rank16_double_rank`
  - `checkpoints/adapters/ui_merge_dual/rank8_weighted_insert`

**Commands run:**

- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ui_merge_dual/rank8_balanced -- --data data/lora/adapters/ui_merge_dual_rank8 --iters 60 -c training/lora_qwen25_coder_7b_rank8.yaml`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/ui_merge_dual/rank8_balanced --specialist ui_merge_dual --tasks benchmarks/ui_merge_dual_tasks_v1.json`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ui_merge_dual/rank16_double_rank -- --data data/lora/adapters/ui_merge_dual_rank8 --iters 60`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/ui_merge_dual/rank16_double_rank --specialist ui_merge_dual --tasks benchmarks/ui_merge_dual_tasks_v1.json`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/ui_merge_dual/rank8_weighted_insert -- --data data/lora/adapters/ui_merge_dual_weighted --iters 60 -c training/lora_qwen25_coder_7b_rank8.yaml`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/ui_merge_dual/rank8_weighted_insert --specialist ui_merge_dual --tasks benchmarks/ui_merge_dual_tasks_v1.json`

**Run artifacts + outcomes:**

- Rank-8 merged baseline:
  - Train: `benchmarks/results/runs/20260517-053022_f4da6f/` (final train/val loss: `0.641 / 0.642`)
  - Bench: `benchmarks/results/runs/20260517-053533_02d24f/` → `24/42`, capability `73.6/100`
- Double-rank method (rank 16):
  - Train: `benchmarks/results/runs/20260517-055021_9c6a85/` (final train/val loss: `0.454 / 0.554`)
  - Bench: `benchmarks/results/runs/20260517-055731_89c513/` → `23/42`, capability `72.0/100`
- Weighted-insert method (rank 8, 70/30):
  - Train: `benchmarks/results/runs/20260517-061008_dae45e/` (final train/val loss: `0.734 / 0.663`)
  - Bench: `benchmarks/results/runs/20260517-061620_13c65d/` → `16/42`, capability `66.7/100`

**Takeaway:** For this pair and short 60-iter runs, merged rank-8 baseline outperformed both tested alternatives; increasing rank did not improve benchmark pass-rate, and weighted insertion degraded both pass-rate and capability index.

---

## 2026-05-16 — Documentation-site capability map explorer (pan/zoom)

**Goal:** Add a docs-website visualization system analogous to the prior canvas map, but with stronger interactive navigation (scroll/zoom/pan) for capability diagnostics.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added mass-benchmark capability graph data pipeline (`_capability_map_payload`) that scans latest specialist mass benchmark runs and emits concept↔capability edge weights,
  - added `GET /api/capability-map` JSON endpoint,
  - added `GET /view/capability-map` standalone interactive map page with drag-pan, wheel zoom, zoom/reset controls, and diagnostics/sample panels,
  - added a docs panel link in the dashboard (`/` -> Documentation Explorer) to open the map page.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.

---

## 2026-05-16 — Fix documentation-dataset builder wiring

**Goal:** Repair repeated `documentation-dataset` trigger failures by wiring `ml_workflow.py` to the active documentation dataset builder path.

**Root cause confirmed:**

- `python3 scripts/ml_workflow.py documentation-dataset` failed with exit `2`.
- Run log showed `Errno 2` for missing file: `scripts/adapters/build_documentation_specialist_dataset.py`.

**Changes made:**

- Added `scripts/build_documentation_specialist_dataset.py` (root-level builder) that:
  - rebuilds `data/lora/adapters/documentation_specialist/{train,valid,test}.jsonl`,
  - preserves the existing schema (`documentation_specialist_dataset_v1`, task alias `mlx-lora-docs-normalize`),
  - keeps deterministic split behavior and `--min-train-core-rows` upsampling.
- Updated `scripts/ml_workflow.py` `documentation-dataset` subcommand to call:
  - `scripts/build_documentation_specialist_dataset.py`
  - instead of missing `scripts/adapters/build_documentation_specialist_dataset.py`.

**Verification:**

- `python3 -m py_compile scripts/ml_workflow.py scripts/build_documentation_specialist_dataset.py` (pass).
- `python3 scripts/ml_workflow.py documentation-dataset` (pass; run `20260517-022917_689389`).
- New run log confirms successful builder invocation and manifest write with expected counts (`core_docs_rows: 21`, `train/valid/test: 120/3/1`).

## 2026-05-16 — Loading-screen capability sweep + ACI lane check

**Goal:** Investigate loading-screen specialist capability across broader UI prompt slices and run the ACI benchmark path for the same adapter.

**Commands run:**

- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/loading_screen/cycle2 --specialist loading_screen`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/loading_screen/cycle2 --specialist loading_screen --specialist hud_status`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/loading_screen/cycle2`
- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/loading_screen/cycle2 --task-id loading-screen-polish`

**Run artifacts:**

- Loading-only benchmark: `benchmarks/results/runs/20260517-022240_28453e/` → summary `1/2`, capability `71.5/100`.
- UI-broad benchmark (loading + HUD): `benchmarks/results/runs/20260517-022338_498b1f/` → summary `1/4`, capability `63.8/100`.
- Full specialist benchmark: `benchmarks/results/runs/20260517-022446_4549f0/` → summary `6/12`, capability `71.4/100`.
- ACI attempt: `benchmarks/results/runs/20260517-022801_96a8cf/` exited `2` because `scripts/run_arena_acceptance_tests.py` is missing in this checkout (`Errno 2`).

**Notes / blockers:**

- Adapter/model compatibility required explicit `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`; default benchmark model (`1.5B`) produced LoRA shape mismatch for `checkpoints/adapters/loading_screen/cycle2`.
- `ml_workflow.py arena-acceptance` currently cannot execute without `scripts/run_arena_acceptance_tests.py`.

---

## 2026-05-16 — Mass loading-screen benchmark suite (v1)

**Goal:** Create a broader, high-volume benchmark set for the `loading_screen` specialist and measure what it can actually do across copy, UI-change, strict-format, and transfer-style prompts.

**Changed files:**

- Added `benchmarks/loading_screen_mass_tasks_v1.json` with 30 `loading_screen` tasks across four categories:
  - `loading_copy_core` (8)
  - `loading_ui_change` (11)
  - `loading_constraints` (6)
  - `loading_transfer_ui` (5)

**Command run:**

- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/adapters/loading_screen/cycle2 --specialist loading_screen --tasks benchmarks/loading_screen_mass_tasks_v1.json`

**Run artifacts:**

- `benchmarks/results/runs/20260517-023102_f3a27f/`
- Benchmark summary: `19/30`
- Capability index: `73.7/100` (`correctness 85.2`, `instruction 85.2`, `concision 18.3`, `speed 30.0`)

**Category pass-rate snapshot:**

- `loading_copy_core`: `7/8` (`87.5%`)
- `loading_ui_change`: `6/11` (`54.5%`)
- `loading_constraints`: `1/6` (`16.7%`)
- `loading_transfer_ui`: `5/5` (`100%`)

**Observed behavior:**

- The adapter remains strong on thematic loading copy and cross-domain transfer hints.
- It underperforms on strict constraint prompts (exact short shape/comma-line/two-lines-only contracts), largely due verbose outputs and missing literal token requirements.

## 2026-05-16 — Router V2 adapter-first classifier + routing docs/RAG system

- Implemented Router V2 classifier stack:
  - Added `scripts/router/classifier.py` (OSS bag-of-words + linear softmax train/load/predict contract).
  - Added `scripts/router/policy.py` (adapter-first route derivation with high-risk and long-prompt overrides).
  - Updated `scripts/model_router.py` to use classifier-first adapter prediction with low-confidence lexical fallback; now emits `policy_version=router_policy_v2_adapter_first`.
- Restored/added missing routing pipeline scripts referenced by workflow:
  - `scripts/routing_prompt_lab.py`
  - `scripts/build_routing_training_dataset.py`
  - `scripts/train_routing_classifier.py`
  - `scripts/router_rag.py`
- Upgraded routing benchmark + labels:
  - Updated `scripts/run_routing_benchmark.py` for `--mode route|adapter|both`, summary JSON output, and adapter-level scoring.
  - Extended `benchmarks/task_routing_tasks.json` with `expected_adapter_id` labels for dual-label evaluation.
- Added routing schema/docs and RAG corpus:
  - `docs/ROUTING_DATASET_CONTRACT.md`
  - `docs/ROUTER_CLASSIFIER_ARTIFACT_CONTRACT.md`
  - `docs/ROUTER_ARCHITECTURE.md`
  - `docs/ROUTER_TASK_TAXONOMY.md`
  - `docs/ROUTER_PROMPT_CASEBOOK.md`
  - `data/routing/router_cases_v1.jsonl`
  - `data/rag/router_agent_corpus.json`
- Workflow/docs updates:
  - `scripts/ml_workflow.py` adds `routing-train` and `routing-gate`.
  - `docs/WORKFLOW.md` updated with `routing-train`, `routing-gate`, and router RAG command.
  - `docs/PROJECT_STATE.md` updated for Router V2 adapter-first policy and new routing commands/contracts.
- Validation runs:
  - `python3 -m py_compile scripts/model_router.py scripts/run_routing_benchmark.py scripts/routing_prompt_lab.py scripts/build_routing_training_dataset.py scripts/train_routing_classifier.py scripts/router/classifier.py scripts/router/policy.py scripts/router_rag.py` (pass).
  - `python3 scripts/build_routing_training_dataset.py --benchmark-tasks benchmarks/task_routing_tasks.json --curated-jsonl data/routing/router_cases_v1.jsonl --dataset-version router-v2-smoke2 --out-root data/lora/routing_classifier --seed 42` (pass).
  - `python3 scripts/train_routing_classifier.py --data-dir data/lora/routing_classifier/router-v2-smoke2 --out-dir training/router_classifier_v1 --epochs 30 --lr 0.3` (pass).
  - `python3 scripts/run_routing_benchmark.py --tasks benchmarks/task_routing_tasks.json --mode both --summary-json benchmarks/results/routing_policy_summary_v2.json` (pass, overall 12/12).
  - `python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_tasks.json --mode both --output-jsonl benchmarks/results/routing_policy_rows_v2.jsonl --summary-json benchmarks/results/routing_policy_summary_v2.json` (pass; run docs appended).
  - `python3 scripts/ml_workflow.py routing-gate --summary-json benchmarks/results/routing_policy_summary_v2.json --rows-jsonl benchmarks/results/routing_policy_rows_v2.jsonl --output benchmarks/results/routing_gate_result_v2.json` (pass; run docs appended).
  - `python3 scripts/ml_workflow.py routing-dataset --benchmark-tasks benchmarks/task_routing_tasks.json --curated-jsonl data/routing/router_cases_v1.jsonl --dataset-version router-v2-workflow --out-root data/lora/routing_classifier --seed 42` (pass; run docs appended).
  - `python3 scripts/ml_workflow.py routing-train --data-dir data/lora/routing_classifier/router-v2-workflow --out-dir training/router_classifier_v1 --epochs 30 --lr 0.3` (pass; run docs appended).

---

## 2026-05-15 — Router compare output URL views (arena-style inspection)

- Updated `scripts/router_chat_gradio.py` compare flow to write per-run artifacts under `benchmarks/results/router_compare_trials/`.
- Each compare run now emits clickable URL links in status for:
  - side-by-side compare page (`compare_view.html`),
  - specialist-only page (`specialist_view.html`),
  - frontier-only page (`frontier_view.html`).
- Persisted text artifacts per run: `prompt.txt`, `specialist_output.md`, `frontier_output.md`.
- Added CLI override `--compare-artifacts-dir` for alternate artifact root.

**Verify:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-31 — economistRL simulation-goal cleanup

**Goal:** Remove a meta-scoring goal from the first economistRL food steady-state task so simulation reward focuses on behavior rather than evaluator scope.

**Changed files:**

- Updated `benchmarks/economistRL_tasks_v1.json`:
  - removed the `localized_simulation` / `simulation_scope` goal from `economistRL-food-steady-state-01`,
  - renormalized remaining 20-tick behavior goal weights to `0.4118`, `0.3529`, and `0.2353`,
  - confirmed no remaining exact `localized_simulation` or `simulation_scope` entries in the task bank.

**Verification:**

- `python3 scripts/economist_rl_reward_engine.py validate --tasks benchmarks/economistRL_tasks_v1.json` (pass; 500 tasks, 350 economy / 150 generalist, no duplicate ids, no issues).

---

## 2026-05-31 — economistRL food steady-state scoring smoke test

**Goal:** Add a regression test proving `economistRL-food-steady-state-01` can move from an example patch response through 20-tick rollout evidence into the reward scorer.

**Changed files:**

- Added `tests/test_economist_rl_food_steady_state.py`:
  - builds an example food-buffer birth-taper patch response and code-shaped diff,
  - runs a deterministic 20-tick food/population simulation from the task's initial state,
  - feeds `simulation_results`, targeted-test evidence, changed files, compile status, and progress evidence into `score_output`,
  - asserts `localized_simulation` is absent and the resulting score passes from behavior evidence.

**Verification:**

- `python3 -m unittest tests.test_economist_rl_food_steady_state` (pass).
- Example good-patch score summary: final score `100.0`, `passed: true`, `simulation_behavior: 100.0`, no failures; trace birth rate `0.0308 -> 0.0`, min food `19.2097`, final population `132.9248`.
- Added a bad-patch rollout case that ignores food, always grows population, permits silent negative food, and reports failed targeted tests; final score `25.0`, `passed: false`, `simulation_behavior: 0.0`, with failures for all three behavior goals plus targeted tests and potential regression.

---

## 2026-05-31 — Lambda economistRL smoke extraction and termination

**Goal:** Extract partial Lambda smoke data for the `economistRL` 10-task run and terminate the worker on request.

**Changed files / artifacts:**

- Added local extraction artifacts under `benchmarks/results/lambda_economistRL_smoke10_20260531/`:
  - `cloud_ablation_rows_specialist_economistRL.jsonl`,
  - `cloud_ablation_runtime_tasks_specialist_economistRL.json`,
  - `gpu-smi-specialist_economistRL.csv`,
  - `gpu-telemetry-summary.json` / `.md`,
  - `fe-ablation-specialist_economistRL.log`,
  - `checkpoint_2-fd5c5ea6028c45c5a8da1ffedee47467-20260531T213832Z.tar.gz`,
  - `EXTRACTION_SUMMARY.json` / `.md`,
  - `LOCAL_INVENTORY.json`.

**Outcome:**

- Lambda instance `fd5c5ea6028c45c5a8da1ffedee47467` (`gpu_1x_a10`, `us-west-1`) was terminated after extraction.
- The run had completed `3/10` requested rows before manual termination; no final run summary existed yet.
- Quality snapshot: accepted `0/3`, verify passed `1/3`, applyable wrote files `2/3`.
- Adapter caveat: all `3/3` rows fell back to base local model (`adapter_path: ""`, `adapter_missing_fallback=base_local`) because `checkpoints/adapters/economistRL/seed_sft` was not present.
- GPU telemetry: `107` samples, average GPU utilization `90.61%`, p95 `98.0%`, max `99.0%`, average memory `15482.79 MiB`, max memory `16573.0 MiB`, average power `141.43 W`, max power `150.32 W`.

---

## 2026-05-31 — RL Lambda Runner economistRL specialization

**Goal:** Add a Lambda-compatible RL cycle runner pattern, currently specialized for economistRL rollout/score/train/eval/promotion, that avoids per-task model reloads and never promotes in place.

**Changed files:**

- Added `scripts/lambda/run_economist_rl_lambda_cycle.py`:
  - tags manifests and training rows as `rl_lambda_runner` with specialization `economist_rl`,
  - resolves the active `economistRL` adapter from `training/adapter_registry_v1.json`,
  - uses `LocalMlxBackend` as a cached generator so the base model + adapter load once per rollout/eval batch,
  - pulls tasks from `benchmarks/economistRL_tasks_v1.json`,
  - writes versioned rollout JSONL, scored JSONL, training data, train config, eval summary, and cycle manifest files,
  - builds candidate training data from high-scoring rollouts plus seed-reference replay while excluding frozen eval task ids from training,
  - trains candidates into `checkpoints/adapters/economistRL/rl_pass_XXX` without overwriting `seed_sft` or the promoted adapter path,
  - promotes by atomically updating the registry only when `--promote-if-better` is set, the candidate beats current eval score, and no major regression flags are present,
  - includes `--dry-run` and `--lambda-mode`.

**Verification:**

- `python3 -m py_compile scripts/lambda/run_economist_rl_lambda_cycle.py` (pass).
- Dry-run smoke with one rollout and one eval task completed and wrote a manifest, then temporary dry-run artifacts were removed.
- Lint check for `scripts/lambda/run_economist_rl_lambda_cycle.py` returned no errors.

**Notes:**

- Live runs now fail fast if the registry adapter path is missing, preventing accidental base-model fallback from being treated as `economistRL`.
- Rollout/eval can use `LOCAL_BACKEND=transformers` on Lambda; the training step still uses the repo's existing `mlx_lm.lora` path and therefore needs a host where that training command is available.
- Follow-up abstraction: added `SpecializationConfig` and `--specialization` support so the orchestration is tagged as `rl_lambda_runner` while `economist_rl` supplies the current adapter id, task DB, eval set, train config, output roots, scorer identity, and system prompt. This leaves the runner ready for future reward-specialist plug-ins without changing the cycle control flow.

---

## 2026-05-26 — Lambda 7B specialist smoke and 121-task run

**Goal:** Run HUD/economy specialist LoRA smoke tests on Lambda, then start full 121-task single-specialist evaluations and prevent the instance from being left idle.

**Changed files:**

- Updated `scripts/model_router.py` so the default local model is `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, matching the repo's documented/current specialist adapter lineage.
- Updated `docs/PROJECT_STATE.md` to note the 7B default for Linux `LOCAL_BACKEND=transformers` smoke/eval runs.

**Remote run notes:**

- Instance: `138.2.238.190`.
- Smoke: initial transformers fallback load failed because the old default was 1.5B-shaped while `hud_status/cycle3` and `economy_tooltip/cycle2` are 7B adapters.
- Smoke retry with explicit `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` succeeded for both `hud_status` and `economy_tooltip`; LoRA tensor pairs applied successfully and generated short outputs.
- Started retry run `lambda_121_hud_economy_retry_20260527-041600` under `benchmarks/results/runs/` on the remote instance with `LOCAL_BACKEND=transformers`, `SOURCE_REPO=~/fallen-empire`, and forced `single_specialist_local` variants for `hud_status` then `economy_tooltip`.

**Verification:**

- `python3 -m py_compile scripts/model_router.py` (pass).
- Lint check for `scripts/model_router.py` returned no errors.

---

## 2026-05-26 — Lambda council conversation eval data collection

**Goal:** Start a 121-task Router V3 council data-collection run on Lambda to gather conversation flow, participant rounds, and adjudication traces for future interactions-system training.

**Changed files:**

- Added `scripts/run_council_conversation_eval.py`:
  - expands `benchmarks/task_bank/compiled/final_mass_testing_system_v1.json`,
  - forces `ROUTER_COUNCIL_ENABLED=1`,
  - records per-task routing metadata, council plan, participant outputs, debate rounds, adjudication, final output, and training eligibility,
  - supports `--mock-generation` for fast structural smoke tests.
- Added `scripts/launch_lambda_council_eval.py`:
  - launches or reuses one Lambda instance,
  - syncs the ML repo and Linux transformers environment,
  - runs a mocked remote smoke first,
  - starts the real 121-task council conversation collector in `tmux`.

**Verification / launch status:**

- Local structural smoke passed:
  - `python3 scripts/run_council_conversation_eval.py --max-tasks 1 --mock-generation --participant-max-tokens 64 --rows-jsonl /tmp/council_smoke_rows.jsonl --summary-json /tmp/council_smoke_summary.json`
  - produced 1 row with 9 participants, 2 debate rounds, adjudication, and summary JSON.
- Lambda launch started with label `council121-20260526` on one `gpu_1x_a10` instance:
  - instance id `1dc361b8ee774e56a33ed6d146cfbf19`,
  - host `129.80.20.32`,
  - remote session `fe-council-eval-council121-20260526`,
  - remote log `~/cloud-eval-logs/fe-council-eval-council121-20260526.log`.
- Remote mocked smoke passed before the full run.
- Real run confirmed healthy after model download/load:
  - remote rows file `~/fallen-empire-lora/benchmarks/results/council_conversation_eval_rows_council121-20260526.jsonl`,
  - observed progress: 3/121 real rows completed,
  - summary file pending until completion.

---

## 2026-05-26 — Specialist personality variants per selected adapter

**Goal:** Encode support for evaluating multiple personality variants of the same specialist adapter in one council run.

**Changed files:**

- Updated `scripts/router/council.py`:
  - council participants now include `base_expert_id` and `variant`,
  - council plan now records `selected_specialist_variants`,
  - planner can expand each selected specialist into personality variants (`cautious`, `balanced`, `assertive`) via `specialist_personality_variants`.
- Updated `scripts/model_router.py`:
  - new env knob `ROUTER_COUNCIL_SPECIALIST_PERSONALITY_VARIANTS` (default 3),
  - passes specialist variant fanout into council planning.
- Updated `scripts/router_chat_gradio.py`:
  - specialist participants resolve adapter weights using `base_expert_id` while preserving unique variant ids for adjudication,
  - logs include `variant` + `base_expert_id`, enabling same-specialist multi-personality comparisons.
- Updated docs:
  - `docs/PROJECT_STATE.md`, `docs/ROUTER_ARCHITECTURE.md`, `docs/WORKFLOW.md` with specialist variant fanout behavior and knobs.

**Verification:**

- `python3 -m py_compile scripts/router/council.py scripts/model_router.py scripts/router_chat_gradio.py` (pass).
- Lint check on touched files returned no errors.

---

## 2026-05-25 — Bounded iterative council debate (EQ compute cap)

**Goal:** Support back-and-forth council deliberation while capping rounds as a trainable/operational compute control.

**Changed files:**

- Updated `scripts/router/council.py`:
  - `CouncilPlan` now includes `debate_max_rounds`,
  - planner accepts `debate_max_rounds` input.
- Updated `scripts/model_router.py`:
  - new planner env knob `ROUTER_COUNCIL_DEBATE_MAX_ROUNDS`,
  - passes debate-round cap into council plan metadata.
- Updated `scripts/router_chat_gradio.py`:
  - added runtime hard-cap flag `--council-debate-max-rounds` (`ROUTER_CHAT_COUNCIL_DEBATE_MAX_ROUNDS`),
  - council lane now runs iterative rounds with peer-summary critique/revision,
  - early-stop on high-confidence / low-disagreement convergence,
  - logs round traces (`rounds`, `debate_rounds_run`, `debate_max_rounds`) for downstream training.
- Updated `scripts/build_council_training_dataset.py`:
  - includes round traces in council dataset rows.
- Updated docs:
  - `docs/PROJECT_STATE.md`, `docs/ROUTER_ARCHITECTURE.md`, `docs/WORKFLOW.md` with debate cap controls.

**Verification:**

- `python3 -m py_compile scripts/router/council.py scripts/model_router.py scripts/router_chat_gradio.py scripts/build_council_training_dataset.py` (pass).
- Lint check on touched files returned no errors.

---

## 2026-05-25 — Council training infrastructure wired into existing workflow

**Goal:** Ensure existing relevant systems can train/evaluate the new council orchestration without ad-hoc scripts.

**Changed files:**

- Added `scripts/init_council_roster.py`:
  - bootstrap or merge-refresh `data/routing/council_roster_v1.json` from adapter registry,
  - writes per-expert default trait bundle.
- Added `scripts/build_council_training_dataset.py`:
  - builds deterministic `train/valid/test` council dataset from router chat and prompt-lab JSONL logs,
  - writes `manifest.json` under dataset version root.
- Updated `scripts/routing_prompt_lab.py`:
  - emits council metadata fields (`council_plan`, disagreement, escalation candidate) for downstream council dataset construction.
- Updated `scripts/ml_workflow.py`:
  - routing benchmark now exposes council flags (`--mode council`, roster path, compare toggle),
  - routing gate now exposes council/online/roster gate args,
  - added `council-dataset` and `council-roster-init` subcommands,
  - command docs updated to reflect council-ready workflow.
- Updated docs:
  - `docs/WORKFLOW.md` routing/council command matrix and examples,
  - `docs/PROJECT_STATE.md` council dataset + roster init + traits and workflow coverage.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/router_chat_gradio.py scripts/routing_prompt_lab.py scripts/build_council_training_dataset.py scripts/init_council_roster.py scripts/ml_workflow.py` (pass).
- `python3 scripts/init_council_roster.py --merge-existing` created `data/routing/council_roster_v1.json` with specialists + 3 generalist profiles and default trait knobs.

---

## 2026-05-25 — Initial per-expert trait knobs wired

**Goal:** Add initial per-expert council knobs and wire them end-to-end so expert behavior can be tuned from roster data.

**Changed files:**

- Updated `scripts/router/roster.py`:
  - added trait schema (`assertiveness`, `verbosity`, `risk_tolerance`, `creativity`, `skepticism`, `decisiveness`),
  - introduced default trait presets by expert/profile,
  - normalized/clamped trait loading + serialization,
  - added `traits_map()` for planner/runtime wiring.
- Updated `scripts/router/council.py`:
  - council participants now carry full `traits`,
  - planner consumes per-expert traits when building participants/weights,
  - adjudicator scoring now includes bounded trait effects.
- Updated `scripts/model_router.py`:
  - passes roster `traits_map()` into council planning.
- Updated `scripts/router_chat_gradio.py`:
  - council lane prompt now includes all trait directives,
  - runtime confidence/task-outcome simulation includes trait influences,
  - participant payload includes trait values for adjudication telemetry.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md` and `docs/PROJECT_STATE.md` with trait knob list and behavior.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/router_chat_gradio.py` (pass).
- Lint check on touched router/docs files returned no errors.

---

## 2026-05-25 — Per-expert council assertiveness

**Goal:** Add per-expert assertiveness so each council participant can voice ideas more cautiously or strongly.

**Changed files:**

- Updated `scripts/router/roster.py`:
  - added `assertiveness` field to roster entries (0..1),
  - defaults for generalist profiles + specialist entries,
  - added `assertiveness_map()` helper for planner/runtime.
- Updated `scripts/router/council.py`:
  - council participant schema now includes `assertiveness`,
  - planner consumes per-expert assertiveness and adjusts participant weight,
  - adjudication scoring now includes a bounded assertiveness term.
- Updated `scripts/model_router.py`:
  - passes roster assertiveness map into council plan construction.
- Updated `scripts/router_chat_gradio.py`:
  - council lane prompt now includes assertiveness instruction,
  - participant confidence simulation and adjudication payload include assertiveness.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md` and `docs/PROJECT_STATE.md` with assertiveness behavior.

**Verification:**

- `python3 -m py_compile scripts/router/roster.py scripts/router/council.py scripts/model_router.py scripts/router_chat_gradio.py` (pass).

---

## 2026-05-25 — Router V3 council scaffolding + combined gate

**Goal:** Implement the Specialist Council + Cascade architecture scaffold: council contracts, profile shaping, top-3 specialist planning, combined offline/online gate, and benchmark extensions.

**Changed files:**

- Added `scripts/router/council.py`:
  - council plan schema (`router_council_plan_v1`) with 3 fixed generalist profiles + specialist slots,
  - profile context-shaping helpers (`wide_compressed`, `precise_short`, `sliding_window`),
  - deterministic adjudication schema (`router_council_adjudication_v1`) and escalation recommendation logic.
- Added `scripts/router/roster.py`:
  - roster schema (`router_council_roster_v1`) with `candidate -> active -> probation -> demoted` state machine,
  - combined gate transition logic using offline + online task-outcome thresholds.
- Updated `scripts/model_router.py`:
  - Router V3 metadata fields on `RouteDecision` (`council_plan`, disagreement, escalation candidate),
  - council env knobs (`ROUTER_COUNCIL_*`) and roster bootstrap/load behavior.
- Updated `scripts/router_chat_gradio.py`:
  - council metadata in specialist lane logs and header,
  - optional council execution lane (`--council-specialist-lane`) with per-participant drafts + adjudication.
- Updated `scripts/run_routing_benchmark.py`:
  - new `--mode council`,
  - council metrics (selection recall/precision proxy, escalation accuracy, quality proxy),
  - baseline-vs-council comparison block in summary schema `routing_benchmark_summary_v3_council`.
- Updated `scripts/router_promotion_gate.py`:
  - support for v2/v3 benchmark summaries,
  - combined checks including council metrics + optional online gate input,
  - optional roster transition application/writeback via `--roster-json --write-roster`.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md` (Router V3 council architecture and key files),
  - `docs/PROJECT_STATE.md` (Router V3 defaults, council knobs, updated benchmark/gate notes),
  - `docs/WORKFLOW.md` (routing benchmark/gate commands including council flows).

**Verification:**

- `python3 -m py_compile scripts/router/council.py scripts/router/roster.py scripts/model_router.py scripts/router_chat_gradio.py scripts/run_routing_benchmark.py scripts/router_promotion_gate.py` (pass).
- `python3 scripts/run_routing_benchmark.py --mode council --tasks benchmarks/task_routing_tasks.json --output-jsonl /tmp/router_council_rows.jsonl --summary-json /tmp/router_council_summary.json` (runs; emits council metrics and summary schema v3).
- `python3 scripts/router_promotion_gate.py --summary-json /tmp/router_council_summary.json --rows-jsonl /tmp/router_council_rows.jsonl --output /tmp/router_gate_report.json` (runs; expected fail under default strict thresholds on current baseline metrics).

---

## 2026-05-23 — Replay preview refusal diagnosis + UX hardening

**Goal:** Diagnose why one-click replay opened a refused localhost URL and ensure the dashboard reports actionable failure reasons instead of opening dead previews.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - fixed replay trial creation to use the original trial base commit (`base_git_ref` / `base_ref`) when reconstructing worktrees,
  - changed replay endpoint behavior to return `ok=false` unless preview reaches `preview_status=ready`,
  - propagated `last_error` details (e.g., TypeScript preflight failures) in API response,
  - updated trial-page replay button handler to display detailed failure text.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Replay smoke checks:
  - trial `20260523-164956-advanced_router_with_specialists-custom-hud-unit-overlay-repl-df7d46` now returns structured failure `preview_not_ready:preflight_failed` with TS parse diagnostics (instead of opening a dead URL blindly),
  - trial `20260523-164956-advanced_router_with_specialists-patch-home-screen-multiplaye-12650b` likewise reports preflight failure details.

---

## 2026-05-23 — One-click trial rehydrate + preview automation

**Goal:** Eliminate repeated manual replay steps by adding a trial-page action that reconstructs a disposable arena environment and boots a live preview automatically.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added `_rehydrate_preview_from_trial(...)` helper to:
    - create a single-task replay manifest from the original trial task spec,
    - create a fresh replay trial,
    - apply the saved `model_output.md`,
    - run verify (optional),
    - start preview and return preview URL/status,
  - added `POST /api/arena/trials/rehydrate-preview`,
  - updated `/view/trial` page with **Rehydrate + Start Preview** button that calls the new endpoint and opens the preview URL.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.

---

## 2026-05-23 — Trial explorer specialist identifiers + row sorting

**Goal:** Improve trial-table readability by exposing specialist identifiers per row and adding sorting controls for specialist-centric review.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` Trial Explorer UI:
  - added row index column (`#`),
  - added specialist identifier column (`adapter_id`),
  - added sort option `Specialist A-Z`,
  - expanded table/loading/error placeholder column spans to match new layout.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).

---

## 2026-05-23 — Trial explorer direct open/replay UX

**Goal:** Make accepted trial inspection actionable from the docs site by adding direct hyperlinks to trial pages and explicit replay/mock-condition guidance.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added `/view/trial?trial_id=...` page with trial status, artifact links, and replay guidance for preview/mock game conditions,
  - added `/view/repo-file?path=...` to render artifact/code files directly in-browser,
  - updated Trial Explorer `Inspect` column to include a direct `open trial` hyperlink,
  - retained in-panel details workflow while adding direct navigation.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- `python3 - <<'PY' ... _trial_detail_payload(...) ... _render_trial_view(...) ... PY` confirmed valid trial payload rendering.

---

## 2026-05-23 — Trial Explorer default source + sorting usability fix

**Goal:** Make accepted trials discoverable by default in the docs dashboard without requiring manual file guessing or ad-hoc filters.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added rows-source metadata helpers (`_ablation_rows_file_stats`, `_ablation_rows_sources_payload`),
  - changed automatic source selection to prefer the most recent rows file with accepted results (fallback to most recent non-empty),
  - added `GET /api/arena/trials/sources`,
  - expanded Trial Explorer UI with source selector and sort options,
  - wired Trial Explorer JS to fetch sources, pass `source_path`, and sort row results client-side.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- `python3 - <<'PY' ... _ablation_rows_sources_payload ... _load_ablation_rows_payload ... PY`:
  - confirmed source list includes row/accepted counts,
  - confirmed auto source now resolves to `benchmarks/results/final_system_ablation_rows_20260523_run4.jsonl` (`accepted=26`, `total=121`) when no source is selected.

---

## 2026-05-23 — Forced HUD-status full benchmark lane

**Goal:** Start a full-system benchmark pass with every task forced through the local `hud_status` specialist lane to measure cross-domain behavior without router frontier escalation.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added a new ablation variant `hud_status_only` in `_variant_plan`,
  - forced this variant to `backend=local`, `route=local`, `adapter_id=hud_status`,
  - included fallback note when the adapter path is missing,
  - expanded allowed variant set to include `hud_status_only`.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).
- Started full run:
  - `source .venv/bin/activate && python -u scripts/run_final_mass_testing_system.py --variants hud_status_only --verify-precheck --rows-jsonl benchmarks/results/final_system_ablation_rows_20260523_hud_status_full.jsonl --summary-json benchmarks/results/final_system_ablation_summary_20260523_hud_status_full.json --summary-md benchmarks/results/final_system_ablation_summary_20260523_hud_status_full.md --runtime-tasks-json benchmarks/results/final_system_runtime_tasks_20260523_hud_status_full.json`
  - startup confirmed (`validation_status=PASS`, variant banner `hud_status_only` printed).

---

## 2026-05-23 — Documentation dashboard trial explorer

**Goal:** Add an automated dashboard workflow to inspect ablation trial rows, filter outcomes, and open trial artifact previews (patch/output/verify snippets) from one UI.

**Changed files:**

- Updated `scripts/private_dashboard_server.py`:
  - added backend helpers to load/filter `final_system_ablation_rows_*.jsonl`,
  - added trial-detail payload loader for `benchmarks/results/game_task_trials/<trial_id>/attempts/local/*`,
  - added two APIs:
    - `GET /api/arena/trials/list`
    - `GET /api/arena/trials/detail?trial_id=<id>`
  - added new dashboard tab **Trial Explorer** with:
    - filters (variant, accepted, verify status, domain, free-text),
    - row table showing task outcome/routing/failure columns,
    - click-to-open trial detail pane with file preview snippets (`model_output.md`, `diff.patch`, `verify_results.json`, `preview.json`).

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- `python3 - <<'PY' ...` helper smoke check:
  - `_load_ablation_rows_payload(limit=5)` returned `ok=True`,
  - `_trial_detail_payload(<trial_id>)` returned `ok=True` with task id.

---

## 2026-05-23 — Force advanced-router local-only execution

**Goal:** Prevent `advanced_router_with_specialists` from escalating to frontier/hybrid backends so ablations can run fully local when quota or policy requires it.

**Changed files:**

- Updated `scripts/run_final_mass_testing_system.py`:
  - added `--disable-frontier-routing` CLI flag,
  - updated `_variant_plan(...)` to accept `disable_frontier_routing`,
  - when enabled and router chooses `frontier`/`hybrid`, force `route=local` and append `frontier_routing_disabled=forced_local` to `router_reason`.

**Verification:**

- `python3 -m py_compile scripts/run_final_mass_testing_system.py` (pass).
- `python -u scripts/run_final_mass_testing_system.py --max-tasks 1 --variants advanced_router_with_specialists --disable-frontier-routing ...` (pass).
- Smoke row confirms forced-local behavior on formerly frontier-prone task:
  - `task_id=patch-node-path-runtime-resolution-01`
  - `backend=local`, `route=local`
  - `router_reason` contains `frontier_routing_disabled=forced_local`
  - `verify_status=passed`.

---

## 2026-05-23 — Controlled checkpoint stop at 121 rows

**Goal:** Stop the `run_final_mass_testing_system.py` ablation cleanly at the end of variant 1 (`121` rows) so progress can be resumed later without losing generated signals.

**Changed files:**

- None (runtime-operation only).

**Verification:**

- Row watchdog output: `ROW_TARGET_REACHED n=121 signal=INT`.
- `wc -l benchmarks/results/final_system_ablation_rows_20260523_run4.jsonl` -> `121`.
- Active runner process was confirmed stopped after signal handling.
- Summary artifacts for run4 were not written yet (expected for mid-run checkpoint).

---

## 2026-05-17 — Adapter data quality diagnosis (negative association risk)

**Goal:** Run a concrete data-quality diagnosis pass to test whether adapter-side training data could be driving negative or off-target associations.

**Changed files:**

- Added `scripts/audit_adapter_data_quality.py`:
  - audits every adapter dataset under `data/lora/adapters/*`,
  - computes risk signals per dataset: contamination (cross-specialist keyword drift), duplicate prompt/assistant pairs, duplicate record ids, negative cue language, and basic constraint adherence failures,
  - writes structured and human-readable outputs to:
    - `benchmarks/results/adapter_data_audit_v1.json`
    - `benchmarks/results/adapter_data_audit_v1.md`.

**Verification:**

- `python3 scripts/audit_adapter_data_quality.py` (pass).
- Audit summary:
  - datasets scanned: `21`
  - rows scanned: `2744`
  - high-risk datasets (score >= 25): `11`
  - highest-risk dataset: `hud_status_specialist_mock_aug_v2` (score `38.87`).
- Top recurring failure pattern in high-risk sets: large cross-specialist contamination + high duplicate pair counts.
- `ReadLints` check for `scripts/audit_adapter_data_quality.py` returned no errors.

---

## 2026-05-18 — Enforce strict specialist-only dataset assembly (rollback of transfer mixing)

**Goal:** Keep cross-domain/test corpora for evaluation only and prevent specialist LoRA training sets from ingesting transfer or mixed-domain rows by default.

**Changed files:**

- Updated `scripts/adapters/build_specialist_dataset.py`:
  - added strict specialist-only assembly controls (default on),
  - pairwise ingestion now keeps only rows that belong to the target specialist (`task.specialists` or canonical `task_id` match),
  - transfer rows are disallowed by default (`transfer_ratio=0`, `max_transfer_rows=0`),
  - cross-domain benchmark tasks (`transfer` / `multidomain` tagged) are skipped by default in strict mode,
  - manifest now records `rejected_foreign_pairwise` and `skipped_cross_domain_benchmark` counts.
- Updated specialist dataset wrappers:
  - `scripts/adapters/build_hud_status_specialist_dataset.py`
  - `scripts/adapters/build_combat_risk_specialist_dataset.py`
  - `scripts/adapters/build_economy_tooltip_specialist_dataset.py`
  - `scripts/adapters/build_save_load_api_guard_specialist_dataset.py`
  - `scripts/adapters/build_ai_planning_explanation_specialist_dataset.py`
  - each now defaults to no transfer (`transfer_ratio=0`, `max_transfer_rows=0`, empty `transfer_task_ids`) and calls strict builder mode.
- Updated `scripts/adapters/build_loading_screen_specialist_dataset.py` with matching strict specialist-only behavior and manifest counters.
- Updated `scripts/ml_workflow.py` dataset-subcommand defaults to remove transfer mixing by default for specialist dataset builds.
- Updated `training/adapter_registry_v1.json` to move `hud_status` champion path from mock-aug back to `checkpoints/adapters/hud_status/cycle3`.
- Updated `docs/PROJECT_STATE.md` registry status row to match the strict-policy rollback.

**Verification:**

- `python3 -m py_compile` for all updated adapter builders and `scripts/ml_workflow.py` (pass).
- Strict build smoke check:
  - `python3 scripts/adapters/build_hud_status_specialist_dataset.py --pairwise-jsonl benchmarks/results/mock_specialist_pairwise_training_data_v1.jsonl --out-dir benchmarks/results/tmp_hud_status_pure_check --max-core-rows 40 --max-benchmark-rows 40 --min-train-core-rows 60` (pass).
  - Result manifest reports strict filtering behavior:
    - `transfer_pairwise=0`
    - `rejected_foreign_pairwise=352`
    - `skipped_cross_domain_benchmark=15`
- `ReadLints` check for all edited scripts returned no errors.

---

## 2026-05-18 — Adapter scorecard + per-adapter verdicts

**Goal:** Create a concrete scorecard and explicit judgment for every adapter dataset in `data/lora/adapters`.

**Changed files:**

- Added `scripts/score_adapter_scorecard.py`:
  - reads `benchmarks/results/adapter_data_audit_v1.json`,
  - computes component metrics per adapter dataset (`purity`, `constraints`, `hygiene`, `tone`, `coverage`),
  - computes weighted final score,
  - assigns verdict (`insufficient_data`, `rebuild_dataset`, `quarantine`) with hard gates,
  - writes artifacts:
    - `benchmarks/results/adapter_scorecard_v1.json`
    - `benchmarks/results/adapter_scorecard_v1.md`.

**Verification:**

- `python3 scripts/score_adapter_scorecard.py` (pass).
- Summary from generated scorecard:
  - datasets scored: `21`
  - verdict counts:
    - `insufficient_data`: `6`
    - `rebuild_dataset`: `8`
    - `quarantine`: `7`
- Lowest-scoring datasets are all mock-aug specialist variants (`hud_status_specialist_mock_aug_v2`, `hud_status_specialist_mock_aug_v1`, `combat_risk_specialist_mock_aug_v1`, `ai_planning_explanation_specialist_mock_aug_v2`, `economy_tooltip_specialist_mock_aug_v1`).
- `ReadLints` check for `scripts/score_adapter_scorecard.py` returned no errors.

---

## 2026-05-18 — HUD status dataset expansion runbook (strict-only)

**Goal:** Draft a clear, executable expansion plan for `hud_status` that avoids transfer contamination and supports promotion-quality retraining.

**Changed files:**

- Added `docs/HUD_STATUS_DATASET_EXPANSION_PLAN.md`:
  - explicit objective and acceptance criteria,
  - prompt/data mix targets,
  - hard authoring rules for HUD-only pairwise rows,
  - command to create `benchmarks/hud_status_mass_tasks_v2_pure.json`,
  - strict build command for `hud_status_specialist_cycle4_pure`,
  - manifest validation checks (`transfer_pairwise=0`, strict flags, rejected-foreign counters),
  - train + benchmark command block and promotion gate criteria,
  - explicit `test_ablation` labeling policy.

---

## 2026-05-18 — Clean 7B benchmark execution + mismatch hardening

**Goal:** Run a clean final HUD benchmark with fixed 7B env and prevent recurring model/python mismatch failures.

**Changed files:**

- Updated `scripts/run_game_benchmark.py`:
  - switched default benchmark model from legacy 1.5B to canonical 7B via `fe_lineage.HF_MODEL_ID`,
  - registered Qwen extra stop tokens (`register_qwen_coder_instruct_extra_stops`) after load to avoid hanging generations on `<|im_end|>` handling.
- Updated `scripts/ml_workflow.py`:
  - `_cmd_benchmark(...)` now launches benchmark with repo venv python via `_venv_exe("python")` instead of inheriting potentially non-venv `sys.executable`.

**Commands run and outcomes:**

- Clean benchmark command:
  - `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/run_game_benchmark.py --tasks benchmarks/specialist_benchmark_tasks.json --specialist hud_status --adapter-path checkpoints/adapters/hud_status/cycle4_pure --max-tokens 256`
- Final benchmark result for `checkpoints/adapters/hud_status/cycle4_pure`:
  - summary: `10/30` (33%)
  - capability index: `83.8/100`
  - advanced ACI: `65.5/100`
- This run confirms execution path is now clean (no `mlx_lm` import mismatch and no 1.5B/7B shape mismatch when `MODEL` is set to 7B).

**Verification:**

- `python3 -m py_compile scripts/run_game_benchmark.py scripts/ml_workflow.py` (pass).
- `ReadLints` for both edited files returned no errors.

---

## 2026-05-18 — Router prompt-type gap fill, manifold v2, and classifier v2 training

**Goal:** Load workflow/routing history, enumerate prompt families, identify structure gaps, and materialize new routing nodes/manifolds plus a new trained router classifier artifact.

**Changed files:**

- Added `data/routing/router_cases_v2.jsonl` with 32 curated prompt-label rows covering:
  - constraint-heavy prompts,
  - mixed-intent prompts,
  - incident/security escalation prompts,
  - docs/run-analysis prompts,
  - specialist transfer prompts.
- Added `data/routing/manifold_prototype_prompts_v2.json` with expanded per-adapter prototype prompts.
- Updated `scripts/model_router.py` prototype builder:
  - now includes `manifold_prototype_prompts_v2.json`,
  - now supports loading `.jsonl` prompt prototype files in addition to JSON arrays.
- Generated new extracted skill/manifold artifacts:
  - `data/routing/skills_v2.json`
  - `data/routing/specialist_skill_profiles_v2.json`
  - `data/routing/skill_manifolds_v2.json`
- Added synthesis doc `docs/ROUTER_PROMPT_TYPES_AND_GAPS.md` (prompt taxonomy, gap map, outcomes, and next directions).

**Verification / runs:**

- `python3 scripts/extract_skills_v1.py --min-support-tasks-per-skill 4 --min-effect-abs 0.05 --skills-out data/routing/skills_v2.json --profiles-out data/routing/specialist_skill_profiles_v2.json --manifolds-out data/routing/skill_manifolds_v2.json --report-out benchmarks/results/skills_extraction_report_v2.md` (pass; retained skills=4, regions=5).
- `python3 scripts/ml_workflow.py routing-dataset --benchmark-tasks benchmarks/task_routing_tasks.json --curated-jsonl data/routing/router_cases_v1.jsonl --curated-jsonl data/routing/router_cases_v2.jsonl --dataset-version router-v3-gapfill-20260518 --out-root data/lora/routing_classifier` (pass; run `20260518-052420_dfb6e7`).
- `python3 scripts/ml_workflow.py routing-train --data-dir data/lora/routing_classifier/router-v3-gapfill-20260518 --out-dir training/router_classifier_v2 --epochs 160 --lr 0.18` (pass; run `20260518-052423_ef205b`).
- `ROUTER_ADAPTER_SELECTION_MODE=hybrid ROUTER_CLASSIFIER_DIR=training/router_classifier_v2 python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_tasks.json --mode both --output-jsonl benchmarks/results/routing_policy_rows_v3_classifier.jsonl --summary-json benchmarks/results/routing_policy_summary_v3_classifier.json` (pass; run `20260518-052428_2613fe`; overall `12/12`).

---

## 2026-05-17 — Capability map knob-selection correlation inspector

**Goal:** When selecting a knob/node in the documentation site's capability map, show the strongest related knob correlations immediately in the diagnostics panel.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` capability-map full-page renderer:
  - added a new **Selected Knob Correlations** panel (`nodeCorrelations`) in the diagnostics column,
  - made map nodes clickable/selectable with visible selected-node highlight,
  - added correlation ranking logic for the selected node using existing graph edges (`weight`, `signed`, `sample_count`),
  - rendered top related correlations with sign (+/-), magnitude, and sample counts.

**Verification:**

- `source .venv/bin/activate && python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Restarted docs server and confirmed healthy startup on `http://0.0.0.0:8787`.

---

## 2026-05-17 — Multi-agent hierarchical execution + split/merge validation

**Goal:** Execute hierarchical recommendations (not just metadata), add multi-agent subtask routing for a single prompt, and validate that split + merge produces one combined output reliably.

**Changed files:**

- Added `scripts/router/multi_agent.py`:
  - `build_multi_agent_subtasks(...)` constructs primary + optional secondary adapter subtasks from hierarchy recommendations.
  - `merge_multi_agent_outputs(...)` deterministically combines per-adapter drafts into one integrated seed payload.
- Updated `scripts/router_chat_gradio.py` specialist lane:
  - added runtime knobs:
    - `--multi-agent-specialist-lane` / `ROUTER_CHAT_MULTI_AGENT_LANE`
    - `--multi-agent-secondary-min-confidence` / `ROUTER_CHAT_MULTI_AGENT_SECONDARY_MIN_CONFIDENCE`
    - `--multi-agent-merge-max-tokens` / `ROUTER_CHAT_MULTI_AGENT_MERGE_MAX_TOKENS`
  - implemented execution path:
    - build subtasks from routing decision (`adapter_id` + `secondary_adapter_id`),
    - run each subtask with resolved adapter weights,
    - run merge pass to synthesize one final response for the same prompt,
    - log orchestration telemetry (`multi_agent.subtasks`, merge stats, subtask ids).
- Added `scripts/validate_multi_agent_orchestration.py`:
  - validates subtask split contract and merge coverage on mixed routing prompts,
  - writes machine-readable validation summary JSON.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md` (multi-agent execution stage and key file),
  - `docs/PROJECT_STATE.md` (router chat multi-agent controls),
  - `docs/WORKFLOW.md` (validation command).

**Verification:**

- `python3 -m py_compile scripts/router/multi_agent.py scripts/router_chat_gradio.py scripts/validate_multi_agent_orchestration.py` (pass).
- `python3 scripts/validate_multi_agent_orchestration.py --tasks benchmarks/task_routing_mixed_tasks_v1.json --output-json benchmarks/results/multi_agent_orchestration_validation_20260517.json`
  - rows: `12`
  - split valid: `12/12` (`1.0`)
  - merge valid: `12/12` (`1.0`)
  - prompts with true multi-agent subtasks: `8/12`
  - artifact: `benchmarks/results/multi_agent_orchestration_validation_20260517.json`.

---

## 2026-05-17 — Hierarchical mixed-task routing (taxonomy -> manifold rerank) v1

**Goal:** Implement stable two-stage mixed-task routing (taxonomy/rule coarse stage + manifold rerank), emit secondary adapter recommendation metadata, and validate A/B impact against the non-hierarchical manifold baseline.

**Changed files:**

- Updated `scripts/router/policy.py`:
  - added coarse taxonomy rule families and `infer_coarse_adapter_candidates(...)`,
  - returns `coarse_bucket`, candidate adapter set, coarse-hit count, and reason text.
- Updated `scripts/model_router.py`:
  - added hierarchical routing controls:
    - `ROUTER_HIERARCHICAL_ROUTING_ENABLED` (default on),
    - `ROUTER_HIERARCHY_CANDIDATE_WIDTH`,
    - `ROUTER_HIERARCHY_MIN_COARSE_HITS`,
  - added two-stage similarity path:
    - Stage 1 taxonomy candidate narrowing,
    - Stage 2 cosine rerank in candidate set,
    - fallback to global similarity if coarse stage is weak,
  - added `AdapterSelection` internal struct and decision metadata:
    - `secondary_adapter_id`, `secondary_confidence`,
    - `coarse_bucket`, `candidate_adapters`, `hierarchy_stage`,
  - updated policy tag to `router_policy_v2_hierarchical_adapter_first`.
- Updated `scripts/run_routing_benchmark.py`:
  - benchmark rows now include optional hierarchy diagnostics (`secondary_adapter_id`, `coarse_bucket`, `candidate_adapters`, `hierarchy_stage`) without changing summary/gate schemas.
- Updated `scripts/optimize_manifold_routing.py`:
  - added hierarchy grid knobs:
    - `--hierarchy-width-grid`,
    - `--hierarchy-min-hits-grid`,
  - propagates hierarchy env vars to each benchmark run and records them in optimization artifacts.
- Updated `data/routing/manifold_prototype_prompts_v1.json`:
  - added mixed-intent disambiguation prototypes for `ai_planning_explanation` vs `general_fallback` and `loading_screen` vs `hud_status`.
- Updated docs:
  - `docs/ROUTER_ARCHITECTURE.md`,
  - `docs/PROJECT_STATE.md`,
  - `docs/WORKFLOW.md`.

**Verification:**

- Syntax:
  - `python3 -m py_compile scripts/model_router.py scripts/router/policy.py scripts/run_routing_benchmark.py scripts/optimize_manifold_routing.py` (pass).
- Hierarchical optimizer sweep:
  - `python3 scripts/optimize_manifold_routing.py --out-dir benchmarks/results/routing_manifold_hier_opt_20260517`,
  - best case: `s0p08_m0p02_u0p58_w3_h2`,
  - merged metrics: overall `0.8333`, route `0.9583`, adapter `0.8333`,
  - gate: fail on overall/adapter thresholds (same gating bottleneck as prior manifold runs).
- A/B (same thresholds, hierarchy off vs on):
  - hierarchy **off** (`ROUTER_HIERARCHICAL_ROUTING_ENABLED=0`):
    - merged: overall `0.8333`, route `1.0000`, adapter `0.8333`,
    - gate output: `benchmarks/results/routing_hier_ab_off_20260517_gate.json` (failed overall/adapter thresholds).
  - hierarchy **on** (`ROUTER_HIERARCHICAL_ROUTING_ENABLED=1`, width `3`, min_hits `2`):
    - merged: overall `0.8333`, route `0.9583`, adapter `0.8333`,
    - gate output: `benchmarks/results/routing_hier_ab_on_20260517_gate.json` (failed overall/adapter thresholds).

**Observed deltas / residual failures:**

- Hierarchy-on fixed one AGI ambiguity (`performance-review`) but introduced one mixed-route regression (`mixed-econ-planning-explain`) in this A/B setting.
- Common remaining misses are still concentrated in:
  - `ai_planning_explanation` vs `general_fallback` boundary (`architecture-plan`, `code-review`),
  - loading/hud crossover (`mixed-loading-hud-status`).

---

## 2026-05-17 — Promote manifold-first routing and optimize threshold sweep

**Goal:** Make manifold/similarity routing the default adapter selector, improve mixed-prompt routing behavior, and re-run benchmark + gate with tuned manifold thresholds.

**Changed files:**

- Updated `scripts/model_router.py`:
  - defaulted `ROUTER_ADAPTER_SELECTION_MODE` to `similarity` (manifold-first),
  - included `data/routing/manifold_prototype_prompts_v1.json` in similarity prototype corpus loading.
- Updated `scripts/router/policy.py`:
  - escalated `save_load_api_guard` to `frontier` route policy to match high-risk API-guard benchmark intent.
- Added `data/routing/manifold_prototype_prompts_v1.json`:
  - curated prototype prompts for adapter anchors, especially mixed/failure intents (`ai_planning_explanation`, `loading_screen`, `save_load_api_guard`).
- Added `scripts/optimize_manifold_routing.py`:
  - grid-searches `ROUTER_SIMILARITY_MIN_SCORE`, `ROUTER_SIMILARITY_MIN_MARGIN`, and `ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD`,
  - writes merged summary/rows per case and evaluates each config with `router_promotion_gate.py`.
- Updated docs:
  - `docs/PROJECT_STATE.md` router baseline section now states manifold/similarity is default and classifier mode is override-only,
  - `docs/WORKFLOW.md` now includes a manifold optimization command example.

**Commands + outcomes:**

- `python3 scripts/optimize_manifold_routing.py --out-dir benchmarks/results/routing_manifold_optimization_20260517`
  - best case: `s0p08_m0p02_u0p58`,
  - merged weighted metrics: overall `0.7917`, route `0.9583`, adapter `0.7917`,
  - artifact: `benchmarks/results/routing_manifold_optimization_20260517/optimization_report.json`.
- `ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD=0.58 python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_tasks.json --mode both --output-jsonl benchmarks/results/routing_manifold_promotion_20260517_agi_rows.jsonl --summary-json benchmarks/results/routing_manifold_promotion_20260517_agi_summary.json`
  - AGI summary: overall `0.75`, route `1.00`, adapter `0.75`.
- `ROUTER_ADAPTER_SELECTION_MODE=similarity ROUTER_SIMILARITY_MIN_SCORE=0.08 ROUTER_SIMILARITY_MIN_MARGIN=0.02 ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD=0.58 python3 scripts/ml_workflow.py routing-benchmark --tasks benchmarks/task_routing_mixed_tasks_v1.json --mode both --output-jsonl benchmarks/results/routing_manifold_promotion_20260517_mixed_rows.jsonl --summary-json benchmarks/results/routing_manifold_promotion_20260517_mixed_summary.json`
  - mixed summary: overall `0.8333`, route `0.9167`, adapter `0.8333`.
- merged the AGI+mixed outputs into:
  - `benchmarks/results/routing_manifold_promotion_20260517_merged_summary.json`,
  - `benchmarks/results/routing_manifold_promotion_20260517_merged_rows.jsonl`.
- `python3 scripts/ml_workflow.py routing-gate --summary-json benchmarks/results/routing_manifold_promotion_20260517_merged_summary.json --rows-jsonl benchmarks/results/routing_manifold_promotion_20260517_merged_rows.jsonl --output benchmarks/results/routing_manifold_promotion_20260517_gate.json`
  - gate failed on overall/adapter thresholds (current `0.7917` vs required `0.90`/`0.88`),
  - route and high-risk checks passed (`route=0.9583`, high-risk misroutes `0`).

**Net result / remaining misses:**

- Compared with earlier manifold baseline (overall `0.7084`), tuned manifold run improved to `0.7917` (+`0.0833` absolute).
- Biggest gain came from mixed prompts (`0.6667` -> `0.8333`, +`0.1666`).
- Remaining failures are adapter-selection ambiguity concentrated in:
  - `ai_planning_explanation` prompts (`performance-review`, `architecture-plan`, `code-review`, `mixed-econ-planning-explain`),
  - one mixed UI crossover prompt (`mixed-loading-hud-status`).

---

## 2026-05-16 — Local generated-file cleanup pass

**Goal:** Remove local/generated artifacts that are not source-of-truth project files and keep them out of future commits.

**Changed files:**

- Updated `.gitignore` to exclude:
  - `.cursor/hooks/.doc_training_trigger_state.json`
  - `data/private_dashboard.sqlite3`
  - `data/private_dashboard.local.sqlite3`
  - `data/training_triggers/documentation_training_queue.jsonl`
  - `lab_dashboard/shell_command_events.jsonl`
- Deleted the local/generated files above from the working tree.

**Verification:**

- `git status --short` no longer reports those generated files as untracked.

---

## 2026-05-16 — Repeat mass-agent process for all specialist agents

**Goal:** Repeat the loading-screen mass benchmark + data-pipeline expansion process across every gameplay specialist agent.

**Changed files:**

- Added mass benchmark generator:
  - `scripts/build_mass_specialist_benchmark_tasks.py`
- Added generic specialist dataset builder:
  - `scripts/adapters/build_specialist_dataset.py`
- Added specialist wrappers:
  - `scripts/adapters/build_hud_status_specialist_dataset.py`
  - `scripts/adapters/build_economy_tooltip_specialist_dataset.py`
  - `scripts/adapters/build_combat_risk_specialist_dataset.py`
  - `scripts/adapters/build_save_load_api_guard_specialist_dataset.py`
  - `scripts/adapters/build_ai_planning_explanation_specialist_dataset.py`
- Generated new mass benchmark task files:
  - `benchmarks/hud_status_mass_tasks_v1.json`
  - `benchmarks/economy_tooltip_mass_tasks_v1.json`
  - `benchmarks/combat_risk_mass_tasks_v1.json`
  - `benchmarks/save_load_api_guard_mass_tasks_v1.json`
  - `benchmarks/ai_planning_explanation_mass_tasks_v1.json`
- Updated workflow plumbing and docs:
  - `scripts/ml_workflow.py` (benchmark-ingestion args for specialist dataset commands + new `save-load-dataset` and `ai-planning-dataset`)
  - `docs/WORKFLOW.md`
  - `docs/PROJECT_STATE.md`

**Dataset build runs (benchmark-ingested) via `ml_workflow.py`:**

- `20260517-031042_7b71df` (`loading-screen-dataset`)
- `20260517-031042_ecfaf4` (`hud-status-dataset`)
- `20260517-031042_2e96cf` (`economy-tooltip-dataset`)
- `20260517-031043_e73c2e` (`combat-risk-dataset`)
- `20260517-031043_532c3a` (`save-load-dataset`)
- `20260517-031043_f0f517` (`ai-planning-dataset`)

**Resulting specialist dataset manifests:**

- `data/lora/adapters/loading_screen_specialist/manifest.json` (`core_benchmark_synth=30`)
- `data/lora/adapters/hud_status_specialist/manifest.json` (`core_benchmark_synth=12`)
- `data/lora/adapters/economy_tooltip_specialist/manifest.json` (`core_benchmark_synth=12`)
- `data/lora/adapters/combat_risk_specialist/manifest.json` (`core_benchmark_synth=12`)
- `data/lora/adapters/save_load_api_guard_specialist/manifest.json` (`core_benchmark_synth=12`)
- `data/lora/adapters/ai_planning_explanation_specialist/manifest.json` (`core_benchmark_synth=12`)

**Mass benchmark runs (per specialist adapter):**

- Loading: `20260517-031050_21d5dd` → `19/30`, capability `73.7/100`
- HUD: `20260517-031849_d268f2` → `5/12`, capability `66.8/100`
- Economy: `20260517-032204_f14b76` → `8/12`, capability `76.4/100`
- Combat: `20260517-032511_b38f4b` → `5/12`, capability `66.9/100`
- Save/load:
  - registry path attempt failed (`20260517-032802_ef46a5`) because `checkpoints/adapters/save_load_api_guard/champion` missing,
  - rerun on existing adapter `checkpoints/adapters/save_load_api_guard/cycle1` (`20260517-032832_0bc2cb`) → `7/12`, capability `68.2/100`
- AI-planning:
  - registry path attempt failed (`20260517-032808_adff5f`) because `checkpoints/adapters/ai_planning_explanation/champion` missing,
  - rerun on existing adapter `checkpoints/adapters/ai_planning_explanation/cycle1` (`20260517-033205_9d5944`) → `3/12`, capability `66.4/100`

**Verification:**

- `.venv/bin/python -m py_compile scripts/build_mass_specialist_benchmark_tasks.py scripts/adapters/build_specialist_dataset.py scripts/adapters/build_hud_status_specialist_dataset.py scripts/adapters/build_economy_tooltip_specialist_dataset.py scripts/adapters/build_combat_risk_specialist_dataset.py scripts/adapters/build_save_load_api_guard_specialist_dataset.py scripts/adapters/build_ai_planning_explanation_specialist_dataset.py scripts/ml_workflow.py` (pass).

---

## 2026-05-16 — Loading dataset pipeline expansion with mass benchmark prompts

**Goal:** Expand the loading-screen data pipeline so the new mass benchmark prompt set can directly feed supervised dataset generation.

**Changed files:**

- Added `scripts/adapters/build_loading_screen_specialist_dataset.py`:
  - ingests pairwise winners from `benchmarks/results/game_task_pairwise_training_data.jsonl`,
  - ingests loading benchmark prompts from `benchmarks/loading_screen_mass_tasks_v1.json`,
  - synthesizes benchmark-target assistant replies that satisfy rubric constraints (`all_contains` / `any_contains` / `none_contains` / `min_chars`),
  - supports configurable caps and ratios (`core/transfer/shared`) with split generation and manifest output.
- Updated `scripts/ml_workflow.py` loading dataset subcommand:
  - new args `--benchmark-tasks-json`, `--max-benchmark-rows`, `--max-shared-rows`,
  - forwards those args to the loading dataset builder.
- Updated docs:
  - `docs/WORKFLOW.md` loading dataset description and example now include benchmark ingestion flags.
  - `docs/PROJECT_STATE.md` notes benchmark-ingestion support in loading dataset flow.

**Command run:**

- `MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit .venv/bin/python scripts/ml_workflow.py loading-screen-dataset --benchmark-tasks-json benchmarks/loading_screen_mass_tasks_v1.json --max-benchmark-rows 30 --min-train-core-rows 120`

**Run artifacts and outcome:**

- Workflow run: `benchmarks/results/runs/20260517-024103_69fc86/` (exit `0`)
- Dataset manifest: `data/lora/adapters/loading_screen_specialist/manifest.json`
- Source counts:
  - `core_pairwise`: `7`
  - `transfer_pairwise`: `0`
  - `core_benchmark_synth`: `30`
  - `shared_anchor`: `0`
- Split counts:
  - `train`: `120`
  - `valid`: `3`
  - `test`: `3`

**Verification:**

- `.venv/bin/python -m py_compile scripts/adapters/build_loading_screen_specialist_dataset.py scripts/ml_workflow.py` (pass).
- Lint check for edited files returned no errors.

---

## 2026-05-15 — Router compare prompt presets from arena tasks

- Updated `scripts/router_chat_gradio.py` to add an `Arena task prompt` dropdown next to the prompt textbox.
- Dropdown options are auto-loaded from `benchmarks/game_task_arena_examples.json` (`id — title`) and selecting one pre-fills the prompt textbox for faster repeated retests.
- Added safe fallback behavior: if the arena task file is missing/invalid, UI still loads with `Custom prompt` only.
- Follow-up: switched dropdown choices to plain task IDs (same source set as old `game_task_arena.py ui` task picker) for reliable option visibility on this Gradio build.
- Follow-up: added explicit preset preview + load flow (`Selected arena prompt preview` + `Load selected preset into Prompt`) and forced dropdown option text to dark for readability on light panels.
- Follow-up: added `Arena prompt quick pick` radio selector (always-visible task-id list) wired to update dropdown + preview, so preset selection still works even when dropdown menu rendering is flaky.

**Verify:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Router prompt lab auto-loads repo `.env`

- Updated `scripts/router_chat_gradio.py` to load repo-root `.env` at startup using an internal lightweight parser.
- Behavior is non-destructive: already exported shell variables still win (`.env` only fills missing keys).
- This enables frontier credentials/defaults (for example `OPENAI_API_KEY` / `FRONTIER_API_KEY`) to be picked up automatically when launching the router prompt site.

**Verify:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Docs Explorer model-track tabs (Open-source only filter)

- Updated `scripts/private_dashboard_server.py` Documentation Explorer to add model-track tabs, including an explicit **Open-source model only** tab.
- Added backend catalog metadata (`model_track`) so docs list filtering can reliably distinguish generated `opensource` / `specialized` outputs from reference/manual docs.
- Extended docs dropdown/meta labels to show track classification and applied the track filter before category/type/search sorting.

**Changed:** `scripts/private_dashboard_server.py`

**Verify:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Dashboard manual check: Docs Snapshot tab now includes `Open-source model only`; selecting it narrows list to `*.opensource.md` generated outputs.

---

## 2026-05-15 — Router winner-pill contrast fix

- Updated `scripts/router_chat_gradio.py` CSS so the Winner fieldset label stays dark on white, while radio-pill labels (`specialist`/`frontier`/`tie`/`neither`) render light text for high contrast on dark pill buttons.

**Verify:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Router prompt/label contrast tweak

- Updated `scripts/router_chat_gradio.py` CSS so prompt textbox entered text renders black and white-panel form labels/titles (Prompt, Specialist output, Frontier output, related labels) render black.
- Kept placeholder styling muted and preserved existing code-block contrast behavior.

**Verify:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-10 — Compare feedback UX: bottom placement + Save and Exit + winner labels

- Moved structured span indicator/review block (`spanList`) below both compare code panels for easier bottom-up review.
- Moved structured submit action below the span list and renamed it to **Save and Exit**.
- Winner selector now shows track-facing labels (e.g., **Frontier**, **Specialized**) instead of generic left/right text.
- On successful structured save, compare view now returns to dashboard automatically.

**Changed:** `scripts/private_dashboard_server.py`

**Verify:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Compare page checks passed after restart: Save and Exit present, winner options show Frontier/Specialized, and verdict section renders below grid.

---

## 2026-05-10 — Compare-view corruption label for broken output spans

- Added explicit span label for corrupted output in compare view: `Mark Corrupt (delete/replace)`.
- Preserved existing `good`/`bad` flow while enabling corruption-specific highlighting and notes.
- Updated training export logic so `corrupt` spans are treated as negative spans in pairwise weighting and rewrite export.

**Changed:** `scripts/private_dashboard_server.py`, `scripts/export_compare_feedback_training_data.py`

- Extended span label set to include `corrupt` in compare-feedback validation.
- Added UI controls/styles for corruption marks (button, highlight color, span-card style, and copy updates).
- Updated structured-span rendering + submission logic to support `corrupt` alongside `good`/`bad`.
- Updated `export_compare_feedback_training_data.py` to treat both `bad` and `corrupt` as negative spans and include span label metadata on rewrite rows.

**Verify:**

- `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py` (pass).

---

## 2026-05-10 — Recency repair: latest documentation updates indexed at top

**Goal:** Restore quick human readability so the newest documentation work is visible first.

**Recency index (latest changes):**

- `2026-05-10` — change-doc captures now enforce `title -> date -> summary bullets` (`scripts/generate_change_documentation_capture.py`, `docs/DOCUMENTATION_AGENT_PRACTICES.md`, `tests/test_change_documentation_capture_format.py`).
- `2026-05-10` — docs explorer labels use content-derived titles (`scripts/private_dashboard_server.py`).
- `2026-05-10` — docs snapshot replaced with live Documentation Explorer (`scripts/private_dashboard_server.py`).
- `2026-05-10` — compare-page curation workflow and base-vs-adapter clarity updates (`scripts/private_dashboard_server.py`).
- `2026-05-05` — scoring clarity with full-doc links and side-by-side compare view (`scripts/private_dashboard_server.py`).

**Note:** Older sections remain append-only below; this top index is the canonical “what changed most recently” summary for fast scan and retrieval context.

---

## 2026-05-10 — Bug-check RAG A/B (retrieval impact on economy specialist)

Ran a focused A/B on `economy-tooltip` with `checkpoints/adapters/economy_tooltip/cycle2`, keeping bug-check loop enabled and toggling RAG retrieval:

- **RAG off** (`GAME_TASK_ARENA_BUG_CHECK_RAG=0`): trial `20260511-053124_economy-tooltip`
  - `apply_status`: `no_applyable_changes`
  - `verify_status`: `failed`
  - generation: `460.59s`, total tokens `3297`
  - bug-check metrics: `bug_check_changed_output=true`, `bug_check_rag_hits_total=0`
- **RAG on** (`GAME_TASK_ARENA_BUG_CHECK_RAG=1`, corpus `data/rag/bug_fix_agent_corpus.json`): trial `20260511-060014_economy-tooltip`
  - `apply_status`: `no_applyable_changes`
  - `verify_status`: `failed`
  - generation: `723.81s`, total tokens `3478`
  - bug-check metrics: `bug_check_changed_output=true`, `bug_check_rag_hits_total=6`

Result for this task/adapter sample: retrieval context changed output but did not improve pass outcome; latency increased materially.

---

## 2026-05-10 — Bug-check loop now retrieves bug-fix RAG context

Wired `scripts/game_task_arena.py` bug-check loop to retrieve per-round guidance from `data/rag/bug_fix_agent_corpus.json` before review/repair generation.

Implementation details:

- Added retrieval wiring via `scripts/documentation_rag.py` helpers (`load_corpus`, `retrieve`, `build_context`).
- Added cached corpus loader + per-round context builder (`build_bug_fix_rag_context`) in `game_task_arena.py`.
- Bug-check review prompt now injects retrieved "Reference bug-fix context" when available.
- Added generate flags/env:
  - `--bug-check-rag` / `--no-bug-check-rag` (`GAME_TASK_ARENA_BUG_CHECK_RAG`)
  - `--bug-check-rag-corpus` (`GAME_TASK_ARENA_BUG_CHECK_RAG_CORPUS`)
  - `--bug-check-rag-top-k` (`GAME_TASK_ARENA_BUG_CHECK_RAG_TOP_K`)
  - `--bug-check-rag-max-chars` (`GAME_TASK_ARENA_BUG_CHECK_RAG_MAX_CHARS`)
- Added generation metrics fields:
  - `bug_check_rag_enabled`
  - `bug_check_rag_corpus`
  - `bug_check_rag_top_k`
  - `bug_check_rag_max_chars`
  - `bug_check_rag_hits_total`

Validation:

- `python3 -m py_compile scripts/game_task_arena.py` (exit 0)
- retrieval smoke test: `build_bug_fix_rag_context(...)` returned hits (`4`) and non-empty context.
- `ReadLints` on `scripts/game_task_arena.py`: no linter errors.

---

## 2026-05-10 — Bug-fix RAG corpus for patch review loop

Created a dedicated bug-fix checklist doc and RAG corpus manifest for repair/review prompts:

- `docs/BUG_FIX_COMMON_ERRORS.md`
- `data/rag/bug_fix_agent_corpus.json`

Corpus uses `documentation_agent_corpus_v1` schema for compatibility with existing `scripts/documentation_rag.py` loader and includes weighted sources for:

- bug-fix checklist (`docs/BUG_FIX_COMMON_ERRORS.md`)
- arena harness implementation (`scripts/game_task_arena.py`)
- generation backend (`scripts/model_router.py`)
- current/project context (`docs/PROJECT_STATE.md`, `docs/SESSION_LOG.md`)

Validation:

- `python3 -c "from pathlib import Path; from scripts.documentation_rag import load_corpus; c=load_corpus(Path('data/rag/bug_fix_agent_corpus.json')); print(len(c))"` → `5`

---

## 2026-05-10 — Game-task arena bug-check loop + specialist A/B

Added a post-generation bug-check repair loop directly to `scripts/game_task_arena.py` generation flow (local and frontier backends). New generate flags/env:

- `--bug-check-loop` / `--no-bug-check-loop` (`GAME_TASK_ARENA_BUG_CHECK_LOOP`, default on)
- `--bug-check-rounds` (`GAME_TASK_ARENA_BUG_CHECK_ROUNDS`, default `1`)
- `--bug-check-max-tokens` (`GAME_TASK_ARENA_BUG_CHECK_MAX_TOKENS`, default `4096`)
- `--bug-check-system-prompt` (`GAME_TASK_ARENA_BUG_CHECK_SYSTEM_PROMPT`)

Also fixed local model/adaptor mismatch during specialist eval by resolving local model id from `adapter_config.json` when available (or `--local-model` / `$MODEL`) before loading `LocalMlxBackend`. This avoids LoRA matmul shape errors when specialist adapters are 7B and default backend model differs.

Recorded generation metrics now include bug-check fields:
`bug_check_loop_enabled`, `bug_check_rounds_requested`, `bug_check_rounds_run`, `bug_check_changed_output`.

### A/B test (specialist task, same adapter)

Task: `economy-tooltip`  
Adapter: `checkpoints/adapters/economy_tooltip/cycle2`

- **Bug-check off**: trial `20260511-041607_economy-tooltip`
  - `apply_status`: `no_applyable_changes`
  - `verify_status`: `failed`
  - generation: `4274` tokens, `276.29s`
- **Bug-check on (1 round)**: trial `20260511-042100_economy-tooltip`
  - `apply_status`: `no_applyable_changes`
  - `verify_status`: `failed`
  - bug-check changed output: `true`
  - generation: `3297` tokens, `456.09s`

Result for this A/B: no pass-rate improvement on this specialist task; additional latency observed.

Verification:

- `python3 -m py_compile scripts/game_task_arena.py` (exit 0)

---

## 2026-05-10 — Router specialist bug-check loop stage

Implemented a post-specialist bug-check stage in `scripts/router_chat_gradio.py` so routed generation now supports:

1. user prompt
2. routed specialist generation
3. bug-check review loop over the candidate patch output

New CLI/env controls:

- `--bug-check-loop` / `--no-bug-check-loop` (`ROUTER_CHAT_BUG_CHECK_LOOP`, default on)
- `--bug-check-rounds` (`ROUTER_CHAT_BUG_CHECK_ROUNDS`, default `1`)
- `--bug-check-max-tokens` (`ROUTER_CHAT_BUG_CHECK_MAX_TOKENS`, default `2048`)
- `--bug-check-system-prompt` (`ROUTER_CHAT_BUG_CHECK_SYSTEM_PROMPT`)

Behavior: after initial routed output, the app runs a strict review prompt that attempts to catch likely patch bugs (compile/import/export/path issues) and rewrites the candidate when needed; it emits the reviewed candidate as the final answer. Interaction logs now record `bug_check_loop_enabled`, `bug_check_rounds_run`, and `bug_check_changed_output`.

Verification: `python3 -m py_compile scripts/router_chat_gradio.py` (exit 0).

---

## 2026-05-04 — Private dashboard scoring tab (Cursor vs model outputs)

Updated `scripts/private_dashboard_server.py` to add a tabbed UI with a new **Scoring vs Cursor Work** panel. The server now reads latest `docs/generated/*.comparison.json` (e.g., duplicate-doc artifacts), computes winner by keyword-hit score, and renders side-by-side tracks for `codex_authored`, `opensource`, and `specialized`.

Added private API endpoint `GET /api/scoring` (same Basic-auth guard as other viewer APIs) and HTML tab wiring (`Overview`, `Scoring`, `Docs Snapshot`).

Docs updated: `docs/PRIVATE_DASHBOARD_DEPLOY.md` now includes `/api/scoring` and scoring-tab verification step.

---

## 2026-05-04 — Code-change training trigger + dual RAG lanes

Added a second RAG lane for run documentation/analysis and wired training prep to codebase edits.

**Dual RAG:**

- Documentation RAG (existing): `data/rag/documentation_agent_corpus.json` + `scripts/run_documentation_agent_benchmark.py`
- Run-analysis RAG (new): `data/rag/run_analysis_agent_corpus.json` + `scripts/run_analysis_rag.py` + `scripts/run_run_analysis_agent_benchmark.py` + tasks `benchmarks/run_analysis_rag_tasks_v1.json`

**Trigger on codebase changes (training prep):**

- New hook: `.cursor/hooks/lab_hook_after_fileedit_training_trigger.py` wired via `.cursor/hooks.json` `afterFileEdit`
- New trigger script: `scripts/trigger_doc_training_on_changes.py`
- On watched edits (`docs/`, `scripts/`, `benchmarks/`, `tests/`, `training/`), it refreshes run-analysis corpus and (cooldown-gated) runs `python scripts/ml_workflow.py documentation-dataset`
- Queue/audit output: `data/training_triggers/documentation_training_queue.jsonl`

Controls:

- `FE_LAB_DOC_TRIGGER_MIN_SECONDS` (default 900)
- `FE_LAB_AUTODOC_DATASET_ON_CHANGE` (default on)

---

## 2026-05-04 — Duplicate docs phase: open-source vs documentation-specialist output

Ran a paired generation pass for the new private dashboard docs using the same prompt/reference source (`docs/PRIVATE_DASHBOARD_DEPLOY.md`) across:

- base open-source model (`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`)
- specialized adapter (`checkpoints/adapters/documentation/cycle3`)

Artifacts written to `**docs/generated/`**:

- `private_dashboard_deploy.opensource.md`
- `private_dashboard_deploy.specialized.md`
- `private_dashboard_deploy.comparison.json`
- `private_dashboard_deploy.comparison.md`

Comparison includes a third anchor score for **Codex-authored** `docs/PRIVATE_DASHBOARD_DEPLOY.md` (keyword coverage + word-count). This establishes the requested duplicate-supervision baseline for tuning documentation behavior.

---

## 2026-05-04 — Private live telemetry dashboard (Railway-first, secure)

Implemented `**scripts/private_dashboard_server.py`**: a no-extra-deps WSGI service with HTTP Basic auth for viewers (`/`, `/api/summary`, `/api/events`), bearer-token ingest (`POST /api/ingest`), SQLite persistence (`FE_DASHBOARD_DB_PATH`), and docs snapshots (`PROJECT_STATE`, `SESSION_LOG`, `SPECIALIZED_RUN_HISTORY`) for personal project observability.

Added `**railway.json`** for one-command deploy (`python3 scripts/private_dashboard_server.py`) and wrote `**docs/PRIVATE_DASHBOARD_DEPLOY.md`** with env/volume/security setup plus local hook->remote ingest wiring.

Hook updates: `**.cursor/hooks/lab_hook_after_shell_autodoc.py`** and `**.cursor/hooks/lab_hook_stop_append.py`** now optionally POST events when `**FE_LAB_REMOTE_INGEST_URL`** + `**FE_LAB_REMOTE_INGEST_TOKEN`** are set (fail-open on network errors). `afterShellExecution` autodoc remains local-first (`shell_command_events.jsonl` + `SPECIALIZED_RUN_HISTORY`).

---

## 2026-05-04 — Cursor test-run autodoc hook (shell + code-change snapshot)

Added project hook `**.cursor/hooks/lab_hook_after_shell_autodoc.py`** and wired `**afterShellExecution`** in `**.cursor/hooks.json`**. Hook command now enables autodoc by default for this repo (`FE_LAB_AUTODOC_APPEND=1` inline), so Cursor shell test/benchmark commands are auto-recorded with exit code + git status snapshot (`m/u/d` counts + touched paths) to `**lab_dashboard/shell_command_events.jsonl`** and appended as compact rows in `**docs/SPECIALIZED_RUN_HISTORY.md**`.

Filter is command-based (`pytest`, `unittest`, benchmark runners, `scripts/ml_workflow.py`) unless `**FE_LAB_AUTODOC_INCLUDE_ALL=1**` is set.

---

## 2026-05-04 — Documentation-agent RAG: specialized run history + orchestrated workflow

**Goal:** keep auxiliary evals legible alongside `docs/run_history.md`, and make the documentation-agent RAG benchmark a one-command habit with aggregate telemetry.

**Added:** `docs/SPECIALIZED_RUN_HISTORY.md` — append-only table for cross-cutting benchmarks (retrospective rows for the manual **2026-05-04** `documentation_agent_*_7b.jsonl` runs + convention for linking `ml_workflow_run_id` in Notes).

**Orchestration:** `python scripts/ml_workflow.py documentation-rag-benchmark` — `benchmarks/results/runs/<run_id>/` (manifest, per-task JSONL), `**docs/run_history.md`** row, `**docs/SPECIALIZED_RUN_HISTORY.md`** row, tail `**benchmarks/results/documentation_rag_timeseries.jsonl`**. **Exit code** follows the **with-RAG** pass; optional no-RAG baseline uses `run_documentation_agent_benchmark.py --no-fail`.

**Runner:** `scripts/fe_ml_lab_runner.py documentation-rag-benchmark` (optional passthrough flags) appends `lab_dashboard/agent_events.jsonl` with `kind: documentation_rag_benchmark`.

**Code:** `scripts/run_documentation_agent_benchmark.py` gains `--no-fail`; `scripts/ml_workflow.py` gains helpers `_append_specialized_history_row`, `_count_doc_benchmark_jsonl`, `_append_documentation_rag_timeseries`. Docs: `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `benchmarks/README.md`, `.cursor/rules/precise-ml-documentation.mdc`. Test: `tests/test_fe_ml_lab_tools.py::test_cmd_documentation_rag_benchmark_invokes_ml_workflow`.

**Verify:** `python -m py_compile scripts/ml_workflow.py …`, `python scripts/ml_workflow.py documentation-rag-benchmark --help`, `python -m unittest tests.test_fe_ml_lab_tools tests.test_documentation_rag -v`.

---

## 2026-05-04 — Cursor `stop` hook ledger (opt-in)

Added `**.cursor/hooks.json`** plus `**.cursor/hooks/lab_hook_stop_append.py`**: Agents hitting `**stop`** append `**lab_dashboard/cursor_hook_events.jsonl**` when `**FE_LAB_CURSOR_HOOK_APPEND=1**` (optional `**FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD=1**` reruns `**build_lab_optimization_dashboard.py**`). `**--cursor-hooks**` override added on the dashboard script. `**lab_dashboard/README.md**` documents hooks versus scheduled MLX versus git syncing across clones/windows.

---

## 2026-05-04 — fe-mlx-lab skill · Cursor token footprint

Shrunk `**.cursor/skills/fe-mlx-lab/SKILL.md**`, `**disable-model-invocation: true**` so MLX guidance is mainly on explicit `**@fe-mlx-lab**` mentions; playbook says terminal-only MLX and **path citations instead of log dumps**.

---

## 2026-05-04 — Router Gradio OSS backbone switch (`force_route=local`)

`scripts/router_chat_gradio.py` now exposes accordion **OSS / routing controls**: **Backbone** (policy **Auto** vs **Codebase OSS** → `GenerationRequest(force_route='local')`), optional registry **LoRA adapter lock**, plus defaults via `**ROUTER_CHAT_DEFAULT_BACKBONE`**, `**ROUTER_CHAT_DEFAULT_ADAPTER_LOCK`**, or `**--default-backbone**` / `**--default-adapter-lock**`. Listener default `**--port` is `7864**` (avoids `**train_ui_gradio.py`'s** `**7862`** collision). `**tests/test_router_backbone_controls.py`** locks regressions vs security-keyword frontier escalation. Revised `**.cursor/skills/fe-mlx-lab/SKILL.md`** stating Composer cannot load repo LoRA; OSS path is MLX UIs described in `**docs/PROJECT_STATE.md`**.

Added `**tests/test_fe_ml_lab_tools.py`** (mocked subprocess coverage for `**fe_ml_lab_runner`**, deterministic fixtures for `**build_lab_optimization_dashboard`**) + CLI overrides `**--cursor-usage**` / `**--agent-events**` on `**scripts/build_lab_optimization_dashboard.py**`. Narrative SKILL note: Cursor integration is Markdown skill metadata consumption, **not** a runtime plugin.

Added `**scripts/fe_ml_lab_runner.py`** with default task `**learning` → `ml_workflow.py smoke`** (switch `--sequence full` for end-to-end; passthrough MLX flags **after `--`**). Each invocation appends JSON lines to `**lab_dashboard/agent_events.jsonl**` (`FE_ML_LAB_SPARED_USD` optional heuristic). `**scripts/build_lab_optimization_dashboard.py**` renders `**lab_dashboard/index.html**` from `**docs/run_history.md**` plus optional `**lab_dashboard/cursor_usage.jsonl**`. Supporting docs: `**lab_dashboard/README.md**`, Cursor skill `**.cursor/skills/fe-mlx-lab/SKILL.md**`. Verified `**python3 -m py_compile scripts/fe_ml_lab_runner.py scripts/build_lab_optimization_dashboard.py**`, `**python scripts/build_lab_optimization_dashboard.py**`, and `**python3 -m unittest discover -s tests**`.

---

## 2026-05-04 — Combat `cycle3` resume train + HUD / combat arena acceptance

**Combat resume:** Finished `**python scripts/ml_workflow.py train`** with `**--resume-adapter-file checkpoints/adapters/combat_risk/cycle3/adapters.safetensors --iters 300`**. Run `**benchmarks/results/runs/20260504-175440_9e0edb`** — `**final_exit_code` 0** (~12644 s).

**Arena (single-task, `auto`, `benchmarks/game_task_arena_examples.json`):**


| Adapter                                   | Run id                       | Task                  | Result                                                                                                         |
| ----------------------------------------- | ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `checkpoints/adapters/hud_status/cycle2`  | `**20260504-212542_a576dd`** | `hud-status-summary`  | **Failed** — `**generate_failed`**, `**generate_exit` 124** (never reached apply)                              |
| `checkpoints/adapters/combat_risk/cycle3` | `**20260504-214556_a37ce5`** | `combat-risk-preview` | **Failed** — round0 edited `**GameHUD.tsx`** (export `**GameHUD`** missing); round1 `**no_applyable_changes`** |


**Registry:** Bumped `**combat_risk` `adapter_path`** to `**checkpoints/adapters/combat_risk/cycle3`** now that resume train completed (`**training/adapter_registry_v1.json`**).

---

## 2026-05-04 — Registry: route `economy_tooltip`, `hud_status`, `combat_risk` to `cycle2`

Updated `training/adapter_registry_v1.json` so `**adapter_path`** resolves to the adapters trained in the overnight lockdown run: `checkpoints/adapters/economy_tooltip/cycle2`, `checkpoints/adapters/hud_status/cycle2`, `checkpoints/adapters/combat_risk/cycle2`; lineages bumped accordingly. **Promotion remains `shadow`** (single-task arena gates for these three were **not** passing at last documented runs); this only aligns the router / local UIs with the newest on-disk weights.

---

## 2026-05-04 — Overnight specialist lockdown (economy / HUD / combat cycle2)

Executed the agreed runbook: `**ml_workflow`** dataset → `**train`** → single-task `**arena-acceptance`** (`--progressive-context auto`, `**benchmarks/game_task_arena_examples.json`**, `**SOURCE_REPO`** `~/fallen-empire`, `**GAME_ARENA_ROOT**` `~/fallen-empire-arena`). **No registry promotion** (none of the three gates passed end-to-end).


| Stage                         | Run id / adapter                                                         | Outcome                                                                                                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**economy_tooltip`** dataset | `benchmarks/results/runs/20260504-035847_b88a85`                         | exit **0**                                                                                                                                                                                                                                                                              |
| `**economy_tooltip`** train   | `20260504-035848_5bfba2` → `checkpoints/adapters/economy_tooltip/cycle2` | exit **0** (~8970 s)                                                                                                                                                                                                                                                                    |
| `**economy_tooltip`** arena   | `20260504-062823_8b3903`, `--task-id economy-tooltip`                    | exit **1**, **0/1** — `**apply_final_ok`**, `**exports_final_ok`**, `**tsc**` `tsc_exit_2` rounds 0–1                                                                                                                                                                                   |
| `**hud_status**` dataset      | `20260504-063042_18ea7e`                                                 | exit **0**                                                                                                                                                                                                                                                                              |
| `**hud_status`** train        | `20260504-063044_ba7a5e` → `checkpoints/adapters/hud_status/cycle2`      | exit **0** (~9763 s)                                                                                                                                                                                                                                                                    |
| `**hud_status`** arena        | `20260504-091330_8c7651`, `--task-id hud-status-summary`                 | exit **1**, **0/1** — mixed rounds (export gap + retry `**no_applyable_changes`**)                                                                                                                                                                                                      |
| `**adapter-datasets`**        | `20260504-091830_32b592`                                                 | exit **0** (refreshed `data/lora/adapters/*`)                                                                                                                                                                                                                                           |
| `**combat_risk`** train (1st) | `20260504-091837_00c86b`                                                 | exit **1** — MLX `**IndexError`** on empty `**valid.jsonl`** / `**test.jsonl`** for `**combat_risk`** when only two synthetic train lines existed                                                                                                                                       |
| `**combat_risk`** train (2nd) | `20260504-091854_ee9a2d`                                                 | exit **0** after duplicating train rows into `**valid.jsonl`** / `**test.jsonl`** for the immediate run; `**scripts/adapters/dataset_builder.py`** `**_split_rows`** now guarantees non-empty valid+test for tiny families so future `**adapter-datasets`** builds load in `**mlx_lm`** |
| `**combat_risk`** arena       | `20260504-110442_ee9296`, `--task-id combat-risk-preview`                | exit **1**, **0/1** — `**apply_final_ok`**, `**tsc_exit_2`** both rounds                                                                                                                                                                                                                |


**Next:** widen `**shared_general_anchor`** or add `**build_combat_*` / pairwise combat rows** before another combat cycle2 pass; chase `**tsc`** deltas on `**economy`** and `**combat`** with curator-aligned repair JSONL or lower LR / fewer iters smoke.

---

## 2026-05-04 — HUD cycle2 arena smoke (`hud-status-summary`)

Ran `arena-acceptance` on `checkpoints/adapters/hud_status/cycle2`. Run `**benchmarks/results/runs/20260504-031825_0f7a64`**: `**exit_code` 1**, **0 / 1** passed. Applied `CompactEmpireStatus.tsx` + `TestEnvironmentShell.tsx` but `**tsc` failed** both rounds (`goldPile`, `goldHold`, `morale` on `Player`, missing imports like `countVillagesInPlayerTerritory` / supply helper, bogus `provinceHexKeys`) — schema hallucination vs curator baseline.

---

## 2026-05-04 — Retrain HUD adapter (v2 specialist JSONL)

**Outcome:** Completed successfully (`exit 0`, ~80 min). Run `benchmarks/results/runs/20260504-015258_141e5c`. MLX iter **200** (val loss ~**0.39**, train loss ~**0.005**); final weights written to `checkpoints/adapters/hud_status/cycle2/adapters.safetensors` and `0000200_adapters.safetensors`. `run_history.md` appended by orchestrator.

**Command:** Background `ml_workflow.py train --adapter-path checkpoints/adapters/hud_status/cycle2` with `--data data/lora/adapters/hud_status_specialist` (200 iters, hyphenated mlx args).

**Next:** `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/hud_status/cycle2 --task-id hud-status-summary`

---

## 2026-05-04 — HUD specialist dataset v2 (alignment)

**Goal:** align HUD training targets with curator gold + real `useGameStore` slices; reduce TSC-hallucination pressure from pairwise.

**Implementation (`scripts/adapters/build_hud_status_specialist_dataset.py`):**

- Lineage `**hud_status_specialist:v2`**; manifest `**schema_version`:** `hud_status_specialist_dataset_v2`.
- Arena-aligned `**HUD_TASK_USER`** for all baseline shards (paths, spectate bootstrap, no invented APIs).
- `**--baseline-shards` (default 12):** duplicate `hud-status-summary.assistant.txt` core rows with distinct `record_id`s.
- **Guardrail rows (3):** short instruction replies anchoring valid store selectors (`players`, `cities`, `units`, …), bootstrap order, JSX closing-tag caution; omit with `**--no-guardrails`**.
- **HUD pairwise filter:** skip winners containing known bad fragments (`matchState`, `myId`, …); disable via `**--no-filter-pairwise-hud`**.
- `**--max-core-rows` default 0:** no HUD pairwise core unless explicitly increased.

**Orchestration (`scripts/ml_workflow.py`):** `hud-status-dataset` forwards `**--baseline-shards`**, `**--no-hud-guardrails`**, `**--no-filter-pairwise-hud**`, default `**--max-core-rows 0**`.

**Docs:** `docs/WORKFLOW.md` row for `hud-status-dataset`.

**Rebuild:** Ran builder with `--min-train-core-rows 120` → **15 core** (12 baseline + 3 guard, 0 pairwise HUD), manifest under `data/lora/adapters/hud_status_specialist/manifest.json`.

---

## 2026-05-04 — HUD cycle2 smoke arena: background launcher PATH

**What happened:** A background `arena-acceptance` for `hud-status-summary` wrote `command not found: python` to `benchmarks/results/hud_cycle2_smoke_arena.log` (non-login shell had no `python` on `PATH`).

**Fix:** Re-launched the same gate using the repo venv: `.venv/bin/python scripts/ml_workflow.py arena-acceptance ...` (PID noted in shell; log appended to the same file).

---

## 2026-05-03 — Promoted loading-screen specialist as pipeline-ready

**Goal:** mark `loading_screen` as the primary pipeline adapter for `loading-screen-polish` after successful lock-down run.

**Changes:**

- Updated taxonomy mapping in `scripts/adapters/taxonomy.py`:
  - `loading-screen-polish` now routes to `loading_screen` (no longer `general_fallback`).
- Updated `training/adapter_registry_v1.json`:
  - `general_fallback.task_ids` cleared.
  - `loading_screen.task_ids` now includes `loading-screen-polish`.
  - `loading_screen.adapter_path` set to `checkpoints/adapters/loading_screen/cycle2`.
  - `loading_screen.lineage` set to `loading_screen:v2:cycle2`.
  - `loading_screen.promotion_state` set to `champion`.

**Rationale:**

- `loading-screen-polish` achieved deterministic acceptance pass (`score 100`) on the focused specialist evaluation run (`20260503-184315_a608df`), which is sufficient for the current narrow-scope promotion target.
- Transfer-domain behavior remains experimental and is intentionally not part of this promotion decision.

---

## 2026-05-03 — Loading-screen specialist dataset v2 + cycle2 training/acceptance

**Goal:** lock down `loading_screen` first by improving specialist data quality (core loading rows + transfer UI rows) before moving to the next domain.

**Implemented:**

- Added `scripts/adapters/build_loading_screen_specialist_dataset.py` to build `data/lora/adapters/loading_screen_specialist/{train,valid,test}.jsonl` + `manifest.json`.
- Added `loading-screen-dataset` subcommand wiring in `scripts/ml_workflow.py`.
- Dataset builder sources:
  - `benchmarks/results/game_task_pairwise_training_data.jsonl` winner outputs for `loading-screen-polish`.
  - curated baseline `data/arena_task_baselines/loading-screen-polish.assistant.txt`.
  - transfer task rows (`hud-status-summary`, `economy-tooltip`) to test domain translation.
- Added deterministic split/backfill behavior and minimum train-size expansion (`--min-train-core-rows`) to avoid tiny specialist runs.

**Commands run:**

- `python scripts/ml_workflow.py loading-screen-dataset --transfer-task-id hud-status-summary --transfer-task-id economy-tooltip --min-train-core-rows 120` (pass; run `20260503-180139_940384`).
- `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/loading_screen/cycle2 -- --data data/lora/adapters/loading_screen_specialist --iters 120 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 40 --max-seq-length 2048` (pass; run `20260503-180149_d2ed85`).
- `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/loading_screen/cycle2 --source-repo /Users/natreed/fallen-empire --preview-port 5300 --timeout-s 14400 --task-id loading-screen-polish --task-id hud-status-summary --task-id economy-tooltip` (run `20260503-184315_a608df`, `1/3` pass).

**Outcome:**

- `loading-screen-polish` passed (`score 100`, apply/tsc/exports/preview all true), so the specialized agent can complete its core task in this run.
- Transfer-domain probes remain unstable (`hud-status-summary` and `economy-tooltip` failed with `tsc`), so scope is currently **core-only reliable** and **transfer experimental**.

---

## 2026-05-03 — Built routing test+train pipeline (prompt lab + dual benchmark + dataset builder + workflow wiring)

**Goal:** implement the routing development pipeline for specialization selection: support live typed prompts, preserve historical route benchmark coverage, evaluate adapter+route labels, and generate reproducible training datasets.

**Added files:**

- `docs/ROUTING_DATASET_CONTRACT.md`: dual-label schema (`expected_adapter_id` + `expected_legacy_route`), lineage fields, deterministic split policy, hard-example criteria.
- `scripts/routing_prompt_lab.py`: interactive/batch prompt intake that predicts adapter + legacy route and writes lineage-rich JSONL under `benchmarks/results/routing_prompt_lab/`.
- `scripts/build_routing_training_dataset.py`: merges benchmark/live/curated labeled prompts into deterministic `train/valid/test` splits + `manifest.json` under `data/lora/routing_classifier/<dataset_version>/`.
- `data/routing/README.md`: versioning conventions for routing source labels.

**Updated runtime wiring:**

- `scripts/run_routing_benchmark.py` now supports `--mode route|adapter|both`, dual-label scoring, confusion summaries, and optional `--summary-json`.
- `scripts/ml_workflow.py` added subcommands:
  - `routing-prompt-lab`
  - `routing-benchmark`
  - `routing-dataset`
  All three are included in standard run-artifact + `docs/run_history.md` workflow behavior.

**Docs updated:**

- `docs/WORKFLOW.md`: command table + examples for prompt lab, dual-label routing benchmark, and routing dataset builds.
- `docs/PROJECT_STATE.md`: routing pipeline status and script/index paths.
- `docs/DATA_LAYOUT.md`: canonical routing source/build artifact paths.

**Verification commands run:**

- `python3 -m py_compile scripts/routing_prompt_lab.py scripts/run_routing_benchmark.py scripts/build_routing_training_dataset.py scripts/ml_workflow.py` (pass).
- `python3 scripts/run_routing_benchmark.py --mode both` (pass; route accuracy `12/12`, adapter labeled rows `0/0` on legacy benchmark file).
- `python3 scripts/routing_prompt_lab.py --prompt "Please review this combat risk tooltip behavior" --output-jsonl benchmarks/results/routing_prompt_lab/smoke.jsonl` (pass; wrote one routing row with adapter/route prediction).
- `python3 scripts/build_routing_training_dataset.py --live-jsonl benchmarks/results/routing_prompt_lab/smoke.jsonl --dataset-version routing-smoke --out-root data/lora/routing_classifier` (pass; wrote deterministic split files + manifest).

---

## 2026-05-03 — Implemented v1 multi-adapter architecture scaffold end-to-end

**Goal:** implement the approved v1 architecture plan without replacing the existing ML workflow/arena stack; deliver practical scaffolding across taxonomy, datasets, gates, routing policy, control plane, drift automation, reporting, docs, and verification.

**Added architecture modules:**

- `scripts/adapters/taxonomy.py`: locked v1 family taxonomy, task mapping, registry schema, promotion state metadata, canary defaults, and `policy_version` lineage fields.
- `scripts/adapters/dataset_builder.py`: per-adapter dataset flow with shared anti-overfit corpus + manifests (`85/10/5` default mix) into `data/lora/adapters/<adapter_id>/`.
- `scripts/adapters/gates.py`: per-adapter gate verdicts with primary average-gain objective and hard rollback guards.
- `scripts/adapters/drift_monitor.py`: warn/freeze/reduce/rollback drift checks separating adapter vs router movement.
- `scripts/router/classifier.py` + `scripts/router/policy.py`: confidence/ambiguity/risk/complexity classifier and adapter-aware routing policy with council modes + API escalation ladder.
- `scripts/control_plane/models.py`, `scripts/control_plane/auth.py`, `scripts/control_plane/scheduler.py`: worker heartbeat/capability schemas, auth token primitives, scheduling/retry scaffolding for multi-Mac workers.
- `scripts/build_multi_adapter_report.py`: dashboard/report output for routing distribution, adapter gate quality, and SLO drift signals.

**Integrated with existing stack:**

- Updated `scripts/model_router.py` to keep legacy `local/frontier/hybrid` routing while adding v1 metadata fields (`adapter_id`, `execution_tier`, `council_mode`, `confidence`, `ambiguity`, `risk_class`, `complexity`, `policy_version`, lineage tags).
- Updated `scripts/run_routing_benchmark.py` to emit the new routing metadata in benchmark rows.
- Extended `scripts/ml_workflow.py` with v1 subcommands while preserving run-artifact and run-history behavior:
  - `adapter-registry`
  - `adapter-datasets`
  - `adapter-gate`
  - `drift-check`
  - `control-plane-schedule`
  - `multi-adapter-report`

**Docs + contracts added/updated:**

- Added: `docs/adapter_taxonomy_v1.md`, `docs/dataset_contract_v1.md`, `docs/gate_policy_v1.md`.
- Added baseline registry: `training/adapter_registry_v1.json`.
- Updated: `docs/WORKFLOW.md` (new subcommands/examples + routing metadata), `docs/PROJECT_STATE.md` (v1 architecture status and defaults).

**Verification commands run:**

- `.venv/bin/python -m py_compile scripts/model_router.py scripts/run_routing_benchmark.py scripts/ml_workflow.py scripts/build_multi_adapter_report.py scripts/adapters/taxonomy.py scripts/adapters/dataset_builder.py scripts/adapters/gates.py scripts/adapters/drift_monitor.py scripts/control_plane/models.py scripts/control_plane/auth.py scripts/control_plane/scheduler.py`
- `.venv/bin/python scripts/run_routing_benchmark.py` (policy-only benchmark, now writing v1 metadata fields)
- `.venv/bin/python scripts/run_routing_benchmark.py --output-jsonl benchmarks/results/routing_policy_rows.jsonl` (12/12 pass with v1 route metadata rows)
- `.venv/bin/python scripts/ml_workflow.py adapter-registry --path training/adapter_registry_v1.json`
- `.venv/bin/python scripts/ml_workflow.py adapter-registry --write-default --path /tmp/adapter_registry_v1.smoke.json`
- `.venv/bin/python scripts/ml_workflow.py adapter-datasets --sources-dir data/arena_task_baselines --shared-anchor-dir data/lora/game_text --out-dir data/lora/adapters`

---

## 2026-05-02 — Added GitHub strict-TS ingestion pipeline for compile-safe LoRA data

**Goal:** create a practical now-usable ingestion/curation path for GitHub TypeScript repos that better match arena apply+`tsc`+exports bottlenecks.

**Added / changed:**

- Added `scripts/build_github_ts_dataset.py`.
  - Inputs: explicit `--repo owner/name`, optional `--repos-file`, optional discovery via `--discover-query`.
  - Filters: permissive SPDX allowlist, strict `tsconfig` evidence (`strict=true` or strict key trio), non-trivial app heuristic (`package.json` + include-root TS paths), TS/TSX path filters and test/build exclusions.
  - Outputs: `train.jsonl`, `valid.jsonl`, `test.jsonl`, `samples_metadata.jsonl`, `manifest.json`.
  - Robustness: best-effort discovery (warnings on query/API failure), optional GitHub token, zip cache under `data/raw/github_ts_cache`, manifest-level accepted/rejected reasons.
- Added docs: `docs/GITHUB_TS_DATASET.md` (usage, filters, output layout, training invocation).
- Updated `docs/DATA_LAYOUT.md` with canonical path section for `data/lora/qwen25-coder-7b/github_ts_compile_safe/`.
- Updated `docs/PROJECT_STATE.md` dataset section to reference the new optional GitHub TypeScript builder.

**Commands run:**

- `.venv/bin/python scripts/build_github_ts_dataset.py --repo pmndrs/zustand --repo reduxjs/redux-toolkit --out-dir data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke`
  - First run: exited with no rows; manifest showed `missing_zipball_url` for both repos.
  - Fixed script to fall back from `zipball_url` to `archive_url` template.
- `.venv/bin/python scripts/build_github_ts_dataset.py --repo pmndrs/zustand --repo reduxjs/redux-toolkit --out-dir data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke --refresh-cache`
  - Success: emitted **145** rows total (`train=124`, `valid=10`, `test=11`), accepted both repos.
- `.venv/bin/python -m py_compile scripts/build_github_ts_dataset.py`
  - Success (syntax check clean).

**Smoke output path:** `data/lora/qwen25-coder-7b/github_ts_compile_safe_smoke/` (contains dataset JSONL + manifest + metadata index).

---

## 2026-05-02 — Compile supervision rows + checkpoint-gated sweep (20/40/60) + anti-regression

**Goal:** recover capability lost in recent mixed-repair runs by adding explicit compile/export supervision rows and selecting a safer checkpoint via gated evaluation every 20 iterations.

**Prepared compile-supervision datasets:**

- Built `data/lora/arena_compile_repair_rows_20260502` from recent core6 failures in `20260502-211555_ce7aa2`:
  - tasks: `loading-screen-polish`, `hud-status-summary`, `economy-tooltip`, `save-load-api-guard`, `ai-planning-explanation`
  - each row includes failure diagnostics (`apply_status`, `tsc_status`, `missing_exports`) and uses the curated `data/arena_task_baselines/<task>.assistant.txt` as the corrected target.
  - split sizes: `train=3`, `valid=1`, `test=1`.
- Built mixed corpus `data/lora/arena_balanced_with_repairs_compile_20260502` by combining:
  - `data/lora/arena_balanced_with_repairs_20260502`
  - `data/lora/arena_compile_repair_rows_20260502` (compile rows repeated 5x per split)
  - final sizes: `train=127`, `valid=21`, `test=21`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-mixed-compile-supervision-20260502 -- --data data/lora/arena_balanced_with_repairs_compile_20260502 --iters 60 --batch-size 1 --val-batches 2 --steps-per-eval 20 --steps-per-report 10 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors`
- `ml_workflow` run id: `20260502-220731_da066d` (exit 0).
- Materialized checkpoint adapters:
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt20`
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt40`
  - `checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt60`

**Checkpoint gate sweep (standard-dev + anti-regression):**

- `ckpt20`:
  - std-dev run `20260502-223623_9e2673`: **0/2**, ACI **37.56**.
  - anti-regression run `20260502-224810_d58935`: **2/4**, ACI **72.78** (`loading-screen-polish`, `save-load-api-guard` passed).
- `ckpt40`:
  - std-dev run `20260502-230130_539fa0`: **0/2**, ACI **44.17**.
  - anti-regression run `20260502-230941_4d9206`: **0/4**, ACI **49.22**.
- `ckpt60`:
  - std-dev run `20260502-232636_e02b4a`: **0/2**, ACI **44.03**.
  - anti-regression run `20260502-233256_847399`: **0/4**, ACI **50.48**.

**Core6 confirmation on selected checkpoint (`ckpt20`):**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-mixed-compile-supervision-20260502-ckpt20 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto --task-id loading-screen-polish --task-id hud-status-summary --task-id economy-tooltip --task-id combat-risk-preview --task-id save-load-api-guard --task-id ai-planning-explanation`
- `ml_workflow` run id: `20260502-234543_08b06c` (exit 1), result **2/6** accepted, ACI **61.97**.
- Passes: `loading-screen-polish`, `save-load-api-guard`.
- Remaining failures are still largely `tsc_exit_2` final-round failures; `hud-status-summary` / `economy-tooltip` did not recover acceptance.

**Conclusion:** compile-supervision rows improved selected anti-regression tasks at the earlier checkpoint (`ckpt20`) and raised full-core6 acceptance versus the prior mixed-repair run, but standard-dev recovery remains incomplete and compile correctness is still the primary gate.

---

## 2026-05-02 — Mixed-curriculum (baseline + repairs) SFT and core6 ACI check

**Goal:** validate whether mixing broad arena curriculum with targeted HUD/economy repair rows preserves broader capability while improving applyability failures.

**Prepared mixed dataset:**

- Built `data/lora/arena_balanced_with_repairs_20260502` from:
  - `data/lora/arena_balanced_curriculum_v1`
  - `data/lora/arena_apply_repair_stddev_20260502` (repair rows repeated 3x)
- Final counts: `train=112`, `valid=16`, `test=16`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-mixed-repair-20260502 -- --data data/lora/arena_balanced_with_repairs_20260502 --iters 60 --batch-size 1 --val-batches 2 --steps-per-eval 20 --steps-per-report 10 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors`
- `ml_workflow` run id: `20260502-201848_c45345` (exit 0).

**Core6 acceptance / ACI:**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-mixed-repair-20260502 --suite core6 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto`
- `ml_workflow` run id: `20260502-211555_ce7aa2` (exit 1), result **1/6** accepted, ACI **53.33**.
- Passing task: `combat-risk-preview`; standard-dev tasks (`hud-status-summary`, `economy-tooltip`) still failed final `tsc` and dropped exports in retry rounds.

---

## 2026-05-02 — Combined repair-set SFT pass + six-task acceptance

**Goal:** run one focused SFT pass on both newly created repair sets (`hud-status-summary` + `economy-tooltip`) and immediately validate on the core six-task arena suite.

**Prepared combined dataset:**

- Built `data/lora/arena_apply_repair_stddev_20260502` by combining:
  - `data/lora/arena_apply_repair_hud_20260502`
  - `data/lora/arena_apply_repair_economy_20260502`
- Combined split sizes: `train=16`, `valid=4`, `test=4`.

**Training run:**

- `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-arena-apply-repair-stddev-20260502 -- --data data/lora/arena_apply_repair_stddev_20260502 --iters 40 --batch-size 1 --val-batches 1 --steps-per-eval 10 --steps-per-report 5 --max-seq-length 4096 --save-every 20 --resume-adapter-file checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428/adapters.safetensors`
- `ml_workflow` run id: `20260502-190626_aa03bc` (exit 0).

**Six-task acceptance check:**

- `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-arena-apply-repair-stddev-20260502 --suite core6 --worktree-root benchmarks/results/arena_worktrees --preview-port 5274 --timeout-s 7200 --progressive-context auto`
- `ml_workflow` run id: `20260502-192506_b1c69f` (exit 1), result **0/6** accepted, ACI **52.54**.
- Pattern: repair-set SFT improved applyability on several tasks (`apply_final_ok` true for 5/6) but all those tasks still failed final `tsc`; `save-load-api-guard` remained `no_applyable_changes`.

---

## 2026-05-02 — Added economy-tooltip apply-repair supervision artifact

**Goal:** capture the newest economy tooltip dual failure (`20260502-173012_economy-tooltip`) as targeted repair supervision.

**Changed:**

- Added `data/arena_task_baselines/economy-tooltip.apply-repair-20260502.json` with local/frontier failure metadata:
  - local: `no_applyable_changes`
  - frontier (`gpt-5.5`): `apply_check_failed` (`error: corrupt patch at line 45`)
- Kept `data/arena_task_baselines/economy-tooltip.assistant.txt` as the corrected applyable supervision target for this task.

**Built dataset:**

- `.venv/bin/python scripts/build_arena_baseline_dataset.py --task-id economy-tooltip --sources-dir data/arena_task_baselines --out-dir data/lora/arena_apply_repair_economy_20260502 --repeat 12`
- Result: `expanded_records=12` (`train=8`, `valid=2`, `test=2`) with no missing baselines.

---

## 2026-05-02 — Added hud-status-summary apply-repair supervision artifact

**Goal:** capture the `20260502-161308_hud-status-summary` dual failure (`no_applyable_changes` local, `apply_check_failed` frontier `gpt-5.5`) as a targeted training example with a corrected applyable answer.

**Changed:**

- Updated `data/arena_task_baselines/hud-status-summary.assistant.txt` with a corrected fenced-file response suitable for direct SFT supervision (parse-safe apply format; no malformed diff hunks).
- Added `data/arena_task_baselines/hud-status-summary.apply-repair-20260502.json` to record trial id, failure statuses, and the corrected supervision target path.

**Built dataset:**

- `.venv/bin/python scripts/build_arena_baseline_dataset.py --task-id hud-status-summary --sources-dir data/arena_task_baselines --out-dir data/lora/arena_apply_repair_hud_20260502 --repeat 12`
- Result: `expanded_records=12` (`train=8`, `valid=2`, `test=2`) with no missing baselines.

---

## 2026-05-02 — Patched suite metadata + promotion coverage gate

**Goal:** close two benchmark hardening gaps: misleading suite labels for explicit task subsets and promotion-gate acceptance of partial benchmark runs.

**Changed (`scripts/run_arena_acceptance_tests.py`):**

- Explicit `--task-id` runs now write `suite: "custom"` in `summary.json` instead of inheriting CLI default `core6`.
- `--suite all` and default runs now emit canonical suite labels (`all`, `core6`) based on resolved task selection.

**Changed (`scripts/arena_promotion_gate.py`):**

- Added minimum coverage guardrails with defaults:
  - `--min-tasks` (default **6**),
  - `--min-coverage-ratio` (default **1.0**),
  - optional `--require-suite core6|all|custom`.
- Gate output now prints evaluated-task count, coverage ratio, and detected suite label alongside worst-task stats.
- Added backward-compatible coverage fallback for older `arena_capability.json` files that do not include `task_coverage_ratio`.

**Validation:**

- `python3 -m py_compile scripts/run_arena_acceptance_tests.py scripts/arena_promotion_gate.py` passed.
- `python3 scripts/arena_promotion_gate.py benchmarks/results/runs/20260501-184217_6f6e9a/arena_capability.json` now correctly fails partial runs on task-count/coverage guards.
- `python3 scripts/arena_promotion_gate.py benchmarks/results/runs/20260430-231450_487deb/arena_capability.json` uses compatibility fallback (`task_coverage_ratio=1.0000` from legacy metadata) and fails only on configured worst-task threshold.

---

## 2026-05-01 — Guardrailed standard-dev SFT attempt (no safe promotion checkpoint)

**Goal:** run a compile-repair-oriented SFT pass while explicitly checking for overfitting regressions on non-standard tasks.

**Training run:** `ml_workflow.py train` `20260501-182241_8f2f0a` on mixed dataset `data/lora/arena_balanced_curriculum_v1`, adapter `checkpoints/fe-lora-arena-guarded-stddev-v1`, resumed from `checkpoints/fe-lora-arena-reverse-recover/adapters.safetensors` (60 iters; final/best val **0.003**).

**Guardrail evals (latest checkpoint, iter 60):**

- Standard-dev pair `20260501-184217_6f6e9a`: **0/2**, ACI **54.9** (still `tsc`-blocked).
- Non-standard guard set `20260501-184605_5ac7f5` (`loading`, `combat`, `save-load`, `ai`): **2/4**; `ai-planning-explanation` regressed (previous reverse-recover had this passing).

**Checkpoint sweep (anti-overfit fallback):**

- Materialized checkpoint-20 adapter (`checkpoints/fe-lora-arena-guarded-stddev-v1-ckpt20`):
  - Standard-dev `20260501-185755_db8e58`: **0/2**, ACI **44.61** (worse).
  - Guard set `20260501-190206_da70d1`: **3/4**, preserving prior non-standard behavior profile.
- Materialized checkpoint-40 adapter (`checkpoints/fe-lora-arena-guarded-stddev-v1-ckpt40`):
  - Standard-dev `20260501-190606_5eccc9`: **0/2**, ACI **52.33** (still no std-dev pass).

**Conclusion:** this mixed SFT recipe did not produce a checkpoint that improves standard-dev tasks without collateral risk. Keep `fe-lora-arena-reverse-recover` (or ckpt20 for guarded behavior) as the safer baseline; next cycle should add explicit compile-repair supervision examples for standard-dev tasks rather than relying on baseline-text repetition.

---

## 2026-05-01 — Diversified arena capability metric + broader suite selector

**Goal:** make arena capability harder to game with narrow task subsets and enable straightforward full-catalog acceptance runs.

**Changed (`scripts/arena_capability_index.py`):**

- Added task-type breakdown from task metadata (`task_type_breakdown`).
- Added diversified scoring outputs:
  - `task_type_balanced_index` (equal-weight across active task types),
  - `tail_robustness_index` (bottom quartile mean, min two tasks),
  - `task_coverage_ratio` (evaluated tasks vs catalog tasks),
  - `raw_diversified_index` and `diversified_arena_capability_index`.
- Added markdown/report output sections for task-type breakdown and diversified metrics.
- CLI summary line now prints headline ACI and diversified ACI together.

**Changed (`scripts/run_arena_acceptance_tests.py`, `scripts/ml_workflow.py`):**

- Added `--suite core6|all` (default `core6`).
- `--suite all` runs all task ids in the selected tasks JSON when no explicit `--task-id` list is provided.
- Workflow manifest now records arena suite under `arena_context.suite`.

**Docs updated:** `docs/ARENA_ROADMAP.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`.

**Validation:**

- `python3 -m py_compile scripts/arena_capability_index.py scripts/run_arena_acceptance_tests.py scripts/ml_workflow.py` passed.
- `python3 scripts/arena_capability_index.py --help` passed.
- `python3 scripts/run_arena_acceptance_tests.py --help` passed.
- `python3 scripts/ml_workflow.py arena-acceptance --help` passed.

---

## 2026-05-01 — Standard-dev RAG tightening in arena context pipeline

**Goal:** reduce noisy/low-signal retrieval for `hud-status-summary` and `economy-tooltip`, and increase actual code context budget for those tasks.

**Changed (`scripts/game_task_arena.py`):**

- Added standard-dev retrieval controls:
  - task-specific priority context files (`STANDARD_DEV_PRIORITY_CONTEXT_FILES`)
  - task-specific progressive prefix allowlists (`STANDARD_DEV_PROGRESSIVE_ALLOWED_PREFIXES`)
  - noisy file exclusion (`.bak`, `.DS_Store`) from context collection
- Tightened context packing for standard-dev tasks:
  - dedupe final packed paths, cap to top 6 files
  - raise per-file minimum packed chars (1200 vs 900)
  - skip `package.json` in prefix for standard-dev tasks
  - reserve larger body budget for code context (prefix cap lowered; body budget increased)
- Tightened progressive retrieval behavior:
  - added standard-dev retrieval rules in probe prompt
  - post-filter progressive paths through task-specific prefix allowlists plus priority seeds

**Validation:**

- `python3 -m py_compile scripts/game_task_arena.py` passed.
- Focused acceptance after first tightening: `20260501-161805_01a0d2` (**0/2**, ACI **53.33**).
- Focused acceptance after body-budget tightening: `20260501-162712_9623fb` (**0/2**, ACI **54.32**), with improved context pack quality (`prefix_chars` ~4046 vs ~6444 and larger code-file coverage in `context_pack.json`), but final failures remain `tsc`-driven.

**Interpretation:** retrieval quality improved measurably (more relevant code included, less prompt crowding), but model-side compile correctness on standard-dev tasks is still the gating issue.

---

## 2026-04-30 — Reverse recovery pass completed + six-task validation

**Reverse pass training:** `ml_workflow.py train` run `20260430-224503_0280f4` finished **exit 0** on `checkpoints/fe-lora-arena-reverse-recover`, resuming from the overfit adapter (`checkpoints/fe-lora-arena-standard-dev-sft/adapters.safetensors`) and training on `data/lora/arena_task_baselines_apply_contract` for 80 iters. Final train loss **0.005**, final/best val loss **0.002** (best at iter 60).

**Immediate six-task check:** `ml_workflow.py arena-acceptance` run `20260430-231450_487deb` on `checkpoints/fe-lora-arena-reverse-recover` scored **3/6**, ACI **74.97** (smoke/std/high: **100.0 / 45.09 / 85.39**). This recovers from the earlier 0/6 regression and preserves strong high-tier behavior.

**Remaining weakness:** `hud-status-summary` and `economy-tooltip` still fail due to `tsc` instability (and HUD export break), so standard-dev remains the bottleneck.

**Prepared better-system dataset:** built `data/lora/arena_balanced_curriculum_v1` as a mixed curriculum (all six tasks + light standard-dev boost) for the next SFT cycle.

---

## 2026-04-30 — Control checks: context vs SFT regression attribution

**Control A (same harness/context, non-standard tasks):** `ml_workflow.py arena-acceptance` on `checkpoints/fe-lora-arena-apply-sft` for `loading-screen-polish` + `ai-planning-explanation` → run `20260430-221251_84b70d`, **1/2** pass. `loading-screen-polish` remained fully healthy (`apply/tsc/exports` true), indicating the harness/context path is not globally broken.

**Control B (same harness/context, standard-dev tasks):** `ml_workflow.py arena-acceptance` on `checkpoints/fe-lora-arena-apply-sft` for `hud-status-summary` + `economy-tooltip` → run `20260430-221703_cfce32`, **0/2** pass (ACI **41.93**). `hud-status-summary` failed on repeated `no_applyable_changes`; `economy-tooltip` applied but failed `tsc`, then dropped `GameHUD` export on retry.

**Interpretation:** Evidence still points to narrow standard-dev SFT over-specialization as the primary 0/6 collapse driver, with standard-dev prompt/context tightening acting as a fragility amplifier on the standard-dev pair.

---

## 2026-04-30 — Wrote frontier playbook for 1/6 → 3/6 jump

**Docs:** Expanded `docs/ARENA_ROADMAP.md` with a dedicated write-up section covering: observed improvement (`20260430-183012_121be1`, **3/6**, ACI **68.05**), explicit working attribution (apply-contract context + targeted SFT), causal/mechanical rationale, repeatable runbook, and ordered frontier experiments to raise worst-tail tasks.

**Intent:** make the capability increase and repeat strategy durable for future cycles without relying on chat memory.

---

## 2026-04-30 — Apply-contract SFT run + six-task acceptance jump (1/6 → 3/6)

**Trained:** `ml_workflow.py train` on `data/lora/arena_task_baselines_apply_contract` → run `20260430-172957_435fb1`, adapter `checkpoints/fe-lora-arena-apply-sft`, final val loss **0.006**.

**Evaluated:** `ml_workflow.py arena-acceptance` on that adapter → valid run `20260430-183012_121be1` (**3/6**, ACI **68.05**, tiers **100.0 / 35.9 / 78.1**). Prior immediate attempt `20260430-182924_4a1732` is not comparable because it used system Python and failed generation imports (`ModuleNotFoundError: mlx_lm`).

**Attribution note (explicit):** current working interpretation is that the capability jump from historical **1/6** runs to **3/6** is driven by the **combination** of (a) apply-contract context injection (`docs/GAME_ARENA_APPLY_CONTRACT.md`) and (b) targeted SFT. This is recorded as provisional until repeated across additional acceptance batches.

**Gate status:** `scripts/arena_promotion_gate.py` still fails on worst-tail thresholds (`worst=22.87`, `mean_two_worst=27.57`), so promotion remains blocked.

---

## 2026-04-30 — Apply contract context + apply-failure-focused SFT hooks

**Changed runtime context:** Added `docs/GAME_ARENA_APPLY_CONTRACT.md` (strict output schema for arena applyability) and injected it into `scripts/game_task_arena.py` packet context (`build_context_pack` prefix + packet prompt reminder). This gives every generate/apply round a shared contract aimed at reducing `no_applyable_changes`.

**Changed SFT data path:** Updated `scripts/build_game_task_pairwise_dataset.py` with `--focus-apply-failures` (filters loser records to apply-format failures), plus contract text injection in user prompts and metadata tags. Updated `scripts/build_arena_baseline_dataset.py` to include the same contract in baseline SFT prompts.

**Workflow/docs:** Added `--focus-apply-failures` passthrough in `scripts/ml_workflow.py arena-gate-train` when `--rebuild-dataset` is used. Documented the loop in `docs/ARENA_ROADMAP.md`, `docs/WORKFLOW.md`, and `docs/PROJECT_STATE.md`.

**Verified:** `python3 -m py_compile scripts/game_task_arena.py scripts/build_game_task_pairwise_dataset.py scripts/build_arena_baseline_dataset.py scripts/ml_workflow.py`.

---

## 2026-04-29 — Arena dashboard, promotion gate, roadmap, fe_lineage policies

**Added:** `scripts/build_arena_dashboard.py` → `benchmarks/arena_dashboard.html` (six-task × runs table, combat/ai KPI columns); `**python scripts/ml_workflow.py arena-dashboard`** alias (no run artifacts/history append). `**scripts/arena_promotion_gate.py`** for worst-of-six thresholds on `arena_capability.json`. `**scripts/fe_lineage.py`**: `DEFAULT_ARENA_PROGRESSIVE_CONTEXT`, `ARENA_ADAPTER_PROGRESSIVE_POLICY` (chunk6k + best-val300 `**auto`**). Docs: `docs/ARENA_ROADMAP.md`; `**docs/ARENA_PROGRESSION.md**`, `**docs/PROJECT_STATE.md`**, `**docs/WORKFLOW.md`**, `**scripts/ml_workflow.py**` updated for dashboards + policy pointers. Verified: `python3 -m py_compile` on touched scripts; `build_arena_dashboard.py --last 5`; `arena_promotion_gate.py` rejects `20260429-025429_9793ff` at `--min-worst-score 40`.

---

**Ran:** `ml_workflow.py arena-acceptance` → `20260429-024422_e7aec2` on `checkpoints/fe-lora-qwen25-coder-7b-latest` (default six-task suite, progressive auto, worktree `~/fallen-empire-arena`, ~~31 min). **Exit 1**, **0/6** preview passes, headline ACI **~~48.28**; artifacts `benchmarks/results/runs/20260429-024422_e7aec2/` (row in `docs/run_history.md`). Distinct from **best-val300** A/B above (different adapter snapshot; best-val300 auto **55.24** / off **47.84**).

---

## 2026-04-29 — Best-val300 six-task arena (progressive auto vs off)

**Adapter:** Created `checkpoints/fe-lora-qwen25-coder-7b-best-val300` = `adapter_config.json` + `0000300_adapters.safetensors` from `fe-lora-qwen25-coder-7b-latest` (best validation at iter **300**, train run `20260428-195934_d451c9`).

**Ran:** Two `ml_workflow.py arena-acceptance` six-task batches (same `SOURCE_REPO` / task ids). **Auto:** `--worktree-root ~/fallen-empire-arena` → `20260429-025429_9793ff`, **1/6** pass, ACI **55.24**, tier indices smoke/std/high **32.3 / 48.7 / 64.3**. **Off (first try):** `20260429-034918_34e0cb` failed all tasks at `create` with `PermissionError` on `~/fallen-empire-arena`. **Off (retry):** `--worktree-root benchmarks/results/arena_worktrees`, `--preview-port 5200` → `20260429-034932_0812a4`, **0/6** pass, ACI **47.84**, tier **33.2 / 50.9 / 49.7**.

**Docs:** `docs/ARENA_PROGRESSION.md` table + narrative; `docs/PROJECT_STATE.md` arena row; canvases `arena-capability-index-tracker` and `arena-three-tier-progression` embedded with both JSON snapshots (best-ACI sort: **auto** ahead of **off** for this adapter).

---

## 2026-04-29 — Canvas: run logs sorted by best ACI

**Changed:** `arena-capability-index-tracker.canvas.tsx` and `arena-three-tier-progression.canvas.tsx` — run log tables and chart categories ordered by **headline ACI descending** (best first); canonical **Run #** unchanged; tier canvas registry gains an **ACI** column and **best-first** tier series order.

---

## 2026-04-29 — Canvas: omit smoke-only run from ACI tracker

**Changed:** `canvases/arena-capability-index-tracker.canvas.tsx` and `canvases/arena-three-tier-progression.canvas.tsx` no longer include **20260428-160917** (single-task acceptance); run numbers reset to **Run 1–2** for the comparable **six-task** batches (`171007` auto, `180627` off). Tracker intro + callout state that smoke-only rows are out of scope for this trend line.

---

## 2026-04-28 — Full arena acceptance run and model pin

**Ran:** `.venv/bin/python scripts/run_arena_acceptance_tests.py --adapter-path checkpoints/fe-lora-game-text-20260428 --timeout-s 7200` → `benchmarks/results/arena_acceptance/acceptance-20260428-031445/`, exit **1**, **0/4** tasks passed (`loading-screen-polish`, `hud-status-summary`, `economy-tooltip`, `combat-risk-preview`).

**Outcome:** Loading/economy produced no applyable changes or corrupt patches; HUD wrote `src/components/ui/GameHUD.tsx` but failed `tsc` and dropped the `GameHUD` export; combat crashed with a LoRA/base-model hidden-size mismatch (`3584` vs `1536`), indicating the gate inherited a 7B default with a 1.5B adapter.

**Changed:** `scripts/run_arena_gate_benchmark.py` and `scripts/run_arena_acceptance_tests.py` now accept/pass `--model` and infer the base model from `adapter_config.json` when present, avoiding 7B/1.5B adapter mismatches.

---

## 2026-04-26 — Game arena preview connectivity + metrics

**Changed:** Fixed Game Task Arena preview startup for disposable Fallen Empire worktrees. Task specs now use `next dev -H 127.0.0.1 -p {port}` instead of the previous `npm run dev -- --host ... --port ...` wrapper that failed with `sh: next: command not found`; preview startup now reuses the source checkout `node_modules` through `PATH`/`NODE_PATH` plus a best-effort worktree symlink.

**Changed:** `preview()` now waits for the sandbox URL and records `preview_status`, `ready_elapsed_s`, and preview metadata before showing links as ready. It also carries the local `/test-env` sandbox files into new disposable worktrees while those game-side files are still uncommitted. Generation now records elapsed seconds, max tokens, input/output/total token counts, estimated frontier cost, provider usage, and writes per-attempt `generation_metrics.json`.

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py`; `load_task_specs` confirms `loading-screen-polish` uses `next dev -H 127.0.0.1 -p {port}` and `/test-env/loading-screen`; fresh smoke trial `preview-smoke-1777253838` previewed `/test-env/loading-screen` on port **5187** with `preview_status=ready` and HTTP **200**, then the preview process was terminated and the disposable worktree cleaned up.

---

## 2026-04-26 — Game sandbox preview links

**Changed:** Added `preview_path` support to `scripts/game_task_arena.py` so preview metadata and UI links can open a task-specific local path while still starting the task's existing `preview_command`. Updated `benchmarks/game_task_arena_examples.json` so standardized tasks deep-link to `/test-env/...` sandbox routes in the game repo.

**Game repo:** Added `/Users/natreed/fallen-empire/src/app/test-env/[envId]/page.tsx`, `/Users/natreed/fallen-empire/src/components/test/TestEnvironmentShell.tsx`, and `/Users/natreed/fallen-empire/src/lib/testEnvironments.ts`. The loading-screen route renders only the loading shell; combat risk preview starts the existing `battle_test`, opens the battle report, and pauses the sim; other visual tasks boot deterministic paused observer states.

**Verified:** `npx tsc --noEmit` and `npm run test:ml-cohort` in `/Users/natreed/fallen-empire` passed; IDE lints reported no errors for the new game route/components or `scripts/game_task_arena.py`; `.venv/bin/python -m py_compile scripts/game_task_arena.py` passed; loading `benchmarks/game_task_arena_examples.json` through `load_task_specs` returned all six `preview_path` values.

---

## 2026-04-26 — Game task arena contrast fix

**Changed:** Tightened `scripts/game_task_arena.py` Gradio CSS so the Game Task Arena uses explicit readable text, panel, input, table, and code colors in light and dark mode while preserving the blue primary-button styling.

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py`; `.venv/bin/python -c "from scripts import game_task_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning); IDE lints report no errors for `scripts/game_task_arena.py`; started a fresh UI server on **[http://127.0.0.1:7874](http://127.0.0.1:7874)** and verified HTTP 200.

---

## 2026-04-26 — Safer arena token default

**Changed:** Raised the landing-page arena Local + Frontier split max-token default from 4096 to **8192** via `DEFAULT_ARENA_MAX_TOKENS`, including the matching `local-attempt --max-tokens` default. Documented the escalation path: start at **8192**, raise to **12000** if static-site output still truncates, then **16000** only for larger trials.

**Verified:** `.venv/bin/python -m py_compile scripts/landing_page_arena.py`; `.venv/bin/python -c "from scripts import landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning). Started updated arena at **[http://127.0.0.1:7867](http://127.0.0.1:7867)** and verified HTTP 200.

---

## 2026-04-26 — Landing page arena parser and IDE UX fixes

**Changed:** Hardened `scripts/landing_page_arena.py` file extraction for common model output styles (`path=`, `filename=`, language-only fences, bare fenced filenames, labeled sections, and raw HTML). Generated attempts now save unparsed raw output to `model_output.md` / `parse_error.html` without replacing `index.html` with fallback parse-error HTML, and generated links to missing local HTML pages such as `game.html` are neutralized to avoid preview 404s.

**Changed:** Tightened the landing-page generation prompt to require only `index.html`, `styles.css`, and `script.js`, with no local secondary HTML links. Simplified the Gradio surface into a single IDE-style workflow with local/frontier code panes, side-by-side iframe previews, paste-import into the Frontier lane, and grading/training-data save.

**Verified:** `.venv/bin/python -m py_compile scripts/landing_page_arena.py`; synthetic parser cases for multiple fence/label/raw-HTML formats; `.venv/bin/python -c "from scripts import landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` → `Blocks` (with the existing urllib3/LibreSSL warning). No local MLX generation was run.

---

## 2026-04-26 — Workflow documentation and audit hardening

**Changed:** `scripts/ml_workflow.py` now re-execs into `.venv/bin/python` when the repo venv exists, keeping export/build/benchmark/EvalPlus steps on the same interpreter as the MLX stack. Future manifests and `RUN.md` files include separate `trained_adapter_path` and `benchmark_adapter_path` fields.

**Changed:** `docs/run_history.md` is now one well-formed table with `Exit`, `Status`, `Trained adapter`, and `Benchmarked adapter/model` columns. Historical rows were normalized so failed runs are visible in the committed index and smoke runs show that the synthetic adapter is trained while the base model is benchmarked.

**Docs:** updated `README.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `requirements.txt`; added `requirements.lock.txt` as lightweight reproducibility constraints for the verified ML/UI package versions.

**Verified:** `.venv/bin/python -m py_compile scripts/ml_workflow.py scripts/visualize_results.py`; `.venv/bin/python scripts/ml_workflow.py --help`; `python3 scripts/ml_workflow.py --help`. No training or benchmark run was launched.

---

## 2026-04-26 — Game task arena V1 (`game_task_arena.py`)

**Added:** `scripts/game_task_arena.py`, a disposable worktree/copy arena for real Fallen Empire game-code trials. It creates local/frontier attempts from versioned task specs, writes context/model packets, applies unified diffs or fenced repo-relative file blocks with path allowlists, runs fixed verification commands, records preview metadata, saves human grades/training records, generates adapter/model comparison reports, and cleans up disposable trees without deleting indexed artifacts.

**Added:** `benchmarks/game_task_arena_examples.json` with starter `ui`, `combat`, `economy`, and `refactor` task specs. Results are filed under `benchmarks/results/game_task_trials/<trial_id>/`, indexed in `benchmarks/results/game_task_index.jsonl`, and summarized in `benchmarks/results/game_task_reports/game_task_summary.md`.

**Updated:** simplified the Gradio UI so standardized tests are primary. Source/worktree/base-ref fields moved into an advanced drawer; the main output after trial creation is just Local and Frontier worktree links. Packet text, manifests, verification output, and reports now live in an optional details drawer.

**Updated:** changed the Game Task Arena UI to match the intended evaluation flow: create disposable worktrees, generate/apply both local and frontier model attempts, start two playable preview links, grade both attempts, and complete the trial with optional cleanup. Worktree links are no longer the primary output.

**Updated:** restyled the Game Task Arena UI with a forced dark theme: black background, readable white text, dark panels/inputs, blue primary buttons, rounded controls, and reduced visual noise.

**Updated:** revised `benchmarks/game_task_arena_examples.json` to six standardized tasks spanning complexity and subsystem coverage: loading/start screen polish, HUD/status UI, economy tooltip, combat risk preview, save/load/API guard, and AI planning rationale. The suite prioritizes quick-to-locate preview changes where possible while still including backend/combat/planning tasks.

**Updated:** added task-specific grading rubrics to every standardized game arena task. The Gradio grading sliders now relabel for the selected task, and saved ratings include `rubric_labels` / `rubric_scores` alongside the generic score fields for comparison reports.

**Updated:** collapsed create/generate/preview into a single **Run Full Trial** button. Manual create/generate/preview/packet/verify/report controls remain in the details drawer for debugging, but the primary workflow is now one action followed by playtesting and grading.

**Verified:** `python3 -m py_compile scripts/game_task_arena.py`; created a temporary mock game repo under `/tmp`, created local/frontier copy attempts, applied a fenced file write to `src/ui/Hud.tsx`, ran fixed `npm run test:ml-cohort` verification (mock pass), recorded preview metadata, saved a human grade/training record, generated the comparison report, and cleaned both disposable copy attempts.

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — Landing page human trial arena

**Added:** `scripts/landing_page_arena.py`, a standalone visual/product evaluation arena for Fallen Empire landing-page trials. It creates shared briefs, isolated static-site attempt folders, Cursor-ready frontier packets, local MLX attempts via `LocalMlxBackend`, preview serving via `python -m http.server`, Gradio UI on port **7863**, and human rating JSON.

**Updated:** the Gradio arena now has separate **Local** and **Frontier** lanes. Local runs the MLX adapter directly. Frontier can call an OpenAI-compatible Codex/frontier API from environment variables (`FRONTIER_API_KEY` / `OPENAI_API_KEY`, `FRONTIER_MODEL`, optional `FRONTIER_API_BASE_URL`) or import pasted Cursor/frontier output for immediate preview and rating.

**Updated:** added a **Compare** tab that takes one shared prompt, splits it into Local and Frontier attempts, runs both lanes in parallel when Frontier API mode is enabled, shows both final previews side by side, and saves paired comparison/training records with winner, scores, notes, brief, and generated files.

**Updated:** promoted the arena to an IDE-style first tab: one prompt, split Local/Frontier generation, editable code panes for `index.html` / `styles.css` / `script.js`, a single **Preview Current Code** button, and grading/training-data save after visual inspection. Trial/attempt names remain internal storage details instead of required user inputs.

**Updated:** removed redundant Gradio tabs from the arena UI and made the IDE workflow the whole product surface. Cursor/frontier paste import now writes directly into the Frontier code panes and preview.

**Updated:** added ignored `.env` support for the landing page arena so `FRONTIER_API_KEY`, `FRONTIER_MODEL`, and optional `FRONTIER_API_BASE_URL` can persist locally without being committed. `.env.example` documents the expected keys and `.gitignore` excludes `.env*` except the example.

**Added:** `benchmarks/landing_page_brief.md` as the shared context packet generated from current repo docs. Trial artifacts are written under `benchmarks/results/landing_page_trials/` and rating/index rows under `benchmarks/results/` (gitignored).

**Verified:** `python3 scripts/landing_page_arena.py brief --overwrite`; created smoke trial `smoke-landing-page`, generated a Cursor packet, scaffolded `manual_smoke`, and saved a smoke rating. `python3 -m py_compile scripts/landing_page_arena.py scripts/model_router.py`; `.venv/bin/python -m py_compile scripts/landing_page_arena.py scripts/model_router.py`; `.venv/bin/python -c "import scripts.landing_page_arena as arena; app = arena.build_app(); print(type(app).__name__)"` (Gradio app builds).

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — EvalPlus benchmark + cost-aware routing framework

**Added:** `scripts/run_evalplus_benchmark.py` for execution-based EvalPlus HumanEval+/MBPP+ subsets/full suites; `scripts/ml_workflow.py evalplus` records those runs under the normal `benchmarks/results/runs/<id>/` artifact structure. Added `evalplus>=0.3.1` to `requirements.txt`.

**Added:** `scripts/model_router.py` with deterministic `RoutingPolicy`, `LocalMlxBackend`, `OpenAICompatibleBackend`, `ModelRouter`, cost metadata, and dry-run support. Added `benchmarks/task_routing_tasks.json` plus `scripts/run_routing_benchmark.py`; dry-run routing benchmark scored **12/12**.

**Adapter selection:** `scripts/select_best_adapter.py` writes `benchmarks/results/adapter_selection.md`; current recommendation is `checkpoints/fe-lora-30m` because it retains game benchmark **15/15** and has much better validation behavior than `checkpoints/fe-lora-800-from-30m`, which regressed to game **14/15** and final validation loss **1.943**.

**Verification:** `python3 -m py_compile scripts/ml_workflow.py scripts/run_evalplus_benchmark.py scripts/model_router.py scripts/run_routing_benchmark.py scripts/select_best_adapter.py`; `python3 scripts/run_routing_benchmark.py` → **12/12**; `python3 scripts/select_best_adapter.py` → recommends `checkpoints/fe-lora-30m`. Two `ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-30m --limit 2` attempts failed before model load because GitHub returned HTTP 502 while EvalPlus downloaded `HumanEvalPlus.jsonl.gz`; retry later or set `HUMANEVAL_OVERRIDE_PATH`.

**Docs:** updated `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, and `benchmarks/README.md`.

---

## 2026-04-26 — 800-iteration resumed LoRA pass + general benchmark

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --evaluate --bench-profile general --adapter-path checkpoints/fe-lora-800-from-30m -- --iters 800 --batch-size 1 --val-batches 8 --resume-adapter-file checkpoints/fe-lora-30m/adapters.safetensors --save-every 100` produced run artifacts at `benchmarks/results/runs/20260426-175720_fb7d5b/`. Training resumed from `checkpoints/fe-lora-30m/adapters.safetensors`, completed successfully in ~52.97 minutes, saved final weights plus 100/200/300/400/500/600/700/800 snapshots, and ended at train loss **0.178**, validation loss **1.943**, peak memory **8.598 GB**. Best validation reading during this pass was **1.400** at iteration 150 with `--val-batches 8`.

**Benchmarks:** automatic **general** profile benchmark on `checkpoints/fe-lora-800-from-30m` scored **8/8**. Follow-up game profile benchmark via `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-800-from-30m --profile game` produced `benchmarks/results/runs/20260426-185136_24ec8d/` and scored **14/15**; failure was `siege-wall-priority-chain` missing `wallBuildPriority`.

**Dashboard:** refreshed `benchmarks/results/run_dashboard.html` with `python3 scripts/visualize_results.py` (7 runs included).

---

## 2026-04-25 — 300-iteration LoRA training pass (`fe-lora-30m`)

**Ran:** `python3 scripts/ml_workflow.py train --evaluate --adapter-path checkpoints/fe-lora-30m -- --iters 300 --batch-size 1 --val-batches 1` produced run artifacts at `benchmarks/results/runs/20260425-202718_459eb4/`. Training completed successfully in ~19.75 minutes, saved `checkpoints/fe-lora-30m/adapters.safetensors` plus 100/200/300-iteration snapshots, and ended at train loss **0.799**, validation loss **0.964**, peak memory **8.598 GB**.

**Benchmark retry:** The workflow's inline benchmark failed because the command was launched with system `python3` instead of the venv Python (`ModuleNotFoundError: No module named 'mlx_lm'`). Reran with `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-30m`, which produced `benchmarks/results/runs/20260425-204711_1efcd0/` and scored **15/15** on the game profile.

**Dashboard:** refreshed `benchmarks/results/run_dashboard.html` with `python3 scripts/visualize_results.py` (5 runs included).

---

## 2026-04-25 — Workflow metadata reliability review

**Changed:** `scripts/ml_workflow.py` now records true workflow start/finish timestamps, total elapsed seconds, and `final_exit_code` in each future manifest; `RUN.md` mirrors those fields and step logs use shell-quoted commands for more reproducible copy/paste. `scripts/visualize_results.py` remains compatible with older manifests and uses `final_exit_code` when present, with an explicit Exit column in the dashboard table.

**Safety:** Detected an active `python scripts/ml_workflow.py train --evaluate --adapter-path checkpoints/fe-lora-30m -- --iters 300 --batch-size 1 --val-batches 1` / `mlx_lm.lora` training process and did not run workflow commands or modify generated run artifacts.

**Verified:** `python3 -m py_compile scripts/ml_workflow.py scripts/visualize_results.py`; `python3 scripts/ml_workflow.py --help`; `tmp_runs="$(mktemp -d /tmp/fe-lora-empty-runs.XXXXXX)" && python3 scripts/visualize_results.py --runs-dir "$tmp_runs" --history /tmp/fe-lora-empty-history.md --out /tmp/fe-lora-dashboard.html` (writes outside repo generated artifacts; output: `Runs included: 0`).

**Docs:** updated `docs/WORKFLOW.md` and `docs/PROJECT_STATE.md`.

---

## 2026-04-25 — Run results visualizer (`visualize_results.py`)

**Added:** `scripts/visualize_results.py` to generate a static HTML dashboard from `benchmarks/results/runs/*/manifest.json`, enriched with `docs/run_history.md` metadata when present. Dashboard includes KPI cards (success/failure, average duration, benchmark pass rate), subcommand/day bar summaries, and a sortable-style run table with status, duration, benchmark summary, adapter path, and step failure counts.

**Verified:** `python3 scripts/visualize_results.py` → wrote `benchmarks/results/run_dashboard.html` with **3 runs included**.

**Docs:** updated `docs/WORKFLOW.md` (new “Results visualizer” section) and `docs/PROJECT_STATE.md` (wired tools table + command examples).

---

## 2026-04-24 — General coding benchmark (`general_coding_tasks.json`)

**Added:** `benchmarks/general_coding_tasks.json` (8 tasks: HTTP GET, JSONL, SQL `SELECT`, digit regex, UTF-8, API expansion, Markdown fence, semver) and `--profile game|general` on `scripts/run_game_benchmark.py` (general → neutral system prompt + default general task file). `json` import fixed for `--output-jsonl`. `scripts/ml_workflow.py` forwards `--profile` on `benchmark`; `full` and `train --evaluate` accept `--bench-profile`.

**Verified:** `python scripts/run_game_benchmark.py --profile general` → **8/8**; default game run → **15/15** on `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, greedy, default max tokens.

**Docs:** `benchmarks/README.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`.

---

## 2026-04-24 — Train UI Gradio compatibility fix

Fixed startup failures in `scripts/train_ui_gradio.py` for older `gradio` APIs in this venv: removed unsupported `Textbox(..., monospace=True)` argument and added missing `Tuple` import required by Gradio's runtime type-hint inspection. Verified app construction with `python -c "import scripts.train_ui_gradio as t; t.build_app(); print('build_app ok')"`.

---

## 2026-04-24 — Gradio training UI (`train_ui_gradio.py`)

Added `scripts/train_ui_gradio.py` (default port **7862**): export + build dataset buttons area, LoRA hyperparameter sliders/fields, live training log stream, Stop, optional post-train benchmark. Documented in `docs/WORKFLOW.md`. Does not replace `ml_workflow.py` for committed `run_history.md` / `manifest.json` audit trail.

---

## 2026-04-24 — Built-in `ml_workflow.py` + run documentation

**Added:** `scripts/ml_workflow.py` subcommands `smoke`, `prepare`, `train` (optional `--evaluate`), `benchmark`, `full`. Each run writes `benchmarks/results/runs/<id>/{manifest.json,RUN.md,logs/}` and appends `**docs/run_history.md`**. `**docs/WORKFLOW.md`** describes usage. README + `training/README.md` + `.cursor/rules/precise-ml-documentation.mdc` + `docs/PROJECT_STATE.md` updated to prefer the orchestrator. `train_lora.sh` comment points to workflow.

**Verified:** `python scripts/ml_workflow.py smoke` exit 0; `docs/run_history.md` gains a row; sample `RUN.md` lists steps and timings.

---

## 2026-04-23 — Game-side tests, human eval UI, export hygiene

**Game repo (`fallen-empire`):** Added `scripts/ml-lora-cohort-guard.ts` (Biome / map presets / Tile+SimResult shape / `runSimulation` smoke) and `package.json` script `**npm run test:ml-cohort`**.

**ML repo:** `scripts/run_game_ml_tests.sh` runs that npm script against `SOURCE_REPO` or `~/fallen-empire`. `scripts/human_eval_ui.py` — Gradio UI on port **7861** by default: pick benchmark task, generate with MLX (optional adapter), 1–5 sliders + notes, append JSONL to `benchmarks/results/human_eval.jsonl`.

**Export:** Expanded `scripts/export_repo_for_training.py` with size cap, path/suffix skips, binary-ish skip, and redaction regexes for keys/tokens/PEM blocks; env `EXPORT_MAX_FILE_BYTES` overrides cap.

**Verified:** `npm run test:ml-cohort` in game repo passes; `SOURCE_REPO=/Users/natreed/fallen-empire python scripts/export_repo_for_training.py` reports skip counts; `human_eval_ui.py --help` ok.

---

## 2026-04-23 — mlx_lm.lora wiring

**Added:** `scripts/build_lora_dataset.py` (export JSONL → `text` JSONL splits), `training/lora_qwen_coder.yaml`, `scripts/train_lora.sh`. `.gitignore`: `data/lora/`. `training/README.md` documents end-to-end flow.

**Verified:** `build_lora_dataset.py --synthetic-smoke`; `mlx_lm.lora --train -c training/lora_qwen_coder.yaml --iters 2 --batch-size 1 --val-batches 1 --steps-per-eval 1 --steps-per-report 1 --max-seq-length 1024 --adapter-path checkpoints/_smoke_lora` → training completes, adapters saved.

---

## 2026-04-23 — Ambitious benchmark + season / evolutionary harness

**Goal:** Harder game-aligned eval tasks; **tiered** scoring (C/B/A); **season-based evolution** of the benchmark population mirroring sim-system; ambitious **LoRA curriculum** defaults as JSON (not executed).

**Added / changed:**

- `benchmarks/fallen_empire_tasks.json`: +8 tasks (SimResult fields, axial distance, map presets, mutation buckets prose, season↔curriculum metaphor, Next route sketch, siege/wall chain, headless core path).
- `scripts/benchmark_evolution_lib.py`: tier specs, `scale_expect_for_tier`, `check_expect`, `mutate_task` / `crossover`, `MAX_EXPECT_MIN_CHARS=320`, vocab injection only for codeish categories.
- `scripts/run_game_benchmark.py`: imports lib; `--tier C|B|A`.
- `training/evolution_config.json`, `training/README.md`: population 18, 8 seasons, tier cycle `C,C,B,B,A,A,B,A`, anchors, selection modes, `lora_ambitious_defaults`, named curriculum seasons.
- `scripts/evolve_benchmark_seasons.py`: season loop, immigration templates, optional `--with-mlx`, dedupe ids, writes flat task JSON array.

**Verified:** `python scripts/run_game_benchmark.py` → 15/15; `python scripts/evolve_benchmark_seasons.py --output benchmarks/results/evolved_tasks.json` (structural mode, fast).

---

## 2026-04-23 — Fallen Empire model benchmark suite

**Goal:** A repeatable, game-flavored eval harness for the MLX model (compare base vs future LoRA).

**Added:** `benchmarks/fallen_empire_tasks.json` (7 tasks: biomes, hex coords, pure TS, API shape, AI param prose, Zustand, neighbor offsets), `scripts/run_game_benchmark.py` (load model once, score with `expect` rules, optional JSONL log), `benchmarks/README.md`. `.gitignore`: `benchmarks/results/`.

**Verified:** `python scripts/run_game_benchmark.py` → **7/7** on Qwen2.5-Coder-1.5B-Instruct-4bit after tightening biome prompt and broadening Zustand `any_contains` keywords.

---

## 2026-04-23 — Assistant name “Albert”

Renamed the Gradio window title and default system persona to **Albert** in `scripts/chat_gradio.py`; aligned smoke script system line in `scripts/smoke_base_model.py`.

---

## 2026-04-23 — Gradio chat UI

**Goal:** Browser interface to confirm you can hold a multi-turn conversation with the local MLX model (and optional future LoRA adapters).

**Changes:**

- Added `scripts/chat_gradio.py`: loads `mlx_lm.load` once; `gr.ChatInterface` with a generator that streams `stream_generate` token deltas; builds prompts from system prompt + Gradio tuple history + current user message via `apply_chat_template`.
- `requirements.txt`: `gradio>=4.44,<5` (venv resolved to `gradio==4.44.1`).

**Commands run:** `python scripts/chat_gradio.py --help`; small Python snippet importing `_history_to_messages` (success).

**Next:** Run `python scripts/chat_gradio.py`, open the printed URL, chat; then proceed to dataset export / LoRA when ready.

---

## 2026-04-23 — Base MLX model smoke + documentation rule

**Goal:** Confirm an open-source MLX code model downloads and generates; establish strict documentation practice for the repo.

**Changes:**

- Added `scripts/smoke_base_model.py`: loads default HF repo `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`, builds instruct prompt via `tokenizer.apply_chat_template`, calls `mlx_lm.generate` with `max_tokens`; uses `make_sampler` when `--temp > 0` (required for `mlx-lm==0.29.1`).
- Added `.cursor/rules/precise-ml-documentation.mdc` (`alwaysApply: true`): mandates maintaining `docs/PROJECT_STATE.md` and append-only `docs/SESSION_LOG.md` with precise versions, ids, and verified commands.
- Added `docs/PROJECT_STATE.md` and this file.

**Commands run (representative):**

- `python3 -m venv .venv` → `pip install -r requirements.txt` (success; resolved to `mlx-lm==0.29.1`, `mlx==0.29.3`, etc.).
- First `python scripts/smoke_base_model.py`: download OK; **failed** once with `TypeError: generate_step() got an unexpected keyword argument 'temp'` — fixed by removing `temp=` and using `make_sampler` only when needed.
- Second run: **success** (~0.7 s load from cache, ~0.8 s generation for 96 tokens).

**Next (suggested):** Run `export_repo_for_training.py` with `SOURCE_REPO` pointing at the game tree; sketch LoRA invocation for `mlx_lm.lora` with the same base model id and document exact CLI in `PROJECT_STATE.md` after verification.

---

## 2026-04-28 — 7B chunked `game_text` baseline run

**Goal:** Get a first documented baseline for the larger **Qwen2.5-Coder-7B-Instruct-4bit** LoRA path using chunked long-file data.

**Ran:** `export SOURCE_REPO=/Users/natreed/fallen-empire && .venv/bin/python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunked-20260428 -- --iters 200 --batch-size 1 --val-batches 4 --max-seq-length 2048 --steps-per-eval 25 --steps-per-report 10 --save-every 50`.

**Outcome:** Workflow run `20260428-041532_007bdf` exited **1** because the automatic game benchmark failed one task; **training completed successfully**. Export wrote **160** files; chunked builder wrote **426** rows (`377/24/25` train/valid/test). Training ended at iter **200**, final train loss **1.168**, final validation loss **1.109**, peak memory **9.825 GB**, trained tokens **326,024**, and saved final adapter plus 50/100/150/200 snapshots under `checkpoints/fe-lora-qwen25-coder-7b-chunked-20260428/`.

**Benchmark:** Game profile scored **14/15**; failed `siege-wall-priority-chain` for missing `siegeChance` and `wallBuildPriority`. Capability Index **81.1/100** (correctness **96.7**, instruction **75.0**, concision **32.4**, speed **38.3**).

---

## 2026-04-28 — 7B LoRA 400-iter train on `fe-lora-qwen25-coder-7b-latest`

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest -- --iters 400`

**Outcome:** Exit **0**; ~~**3.7 h**; `benchmarks/results/runs/20260428-195934_d451c9/`. Trajectory: **final train loss ~0.87**, **final val ~1.574**, **best val ~1.275 @ iter 300**, **~~768k** trained tokens, **~12.97 GB** peak. No automatic lexical benchmark in this subcommand.

**Docs:** Repaired `docs/run_history.md` table row that had merged `arena-acceptance` + `train` on one line.

---

## 2026-04-29 — Game benchmark on `fe-lora-qwen25-coder-7b-latest` (post 400-iter train)

**Ran:** `.venv/bin/python scripts/ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest --profile game`

**Outcome:** Run `20260429-023956_396046`, exit **1** (one heuristic task failed). **14/15** tasks passed (**93%**). Failed task: `siege-wall-priority-chain` — missing required substrings `siegeChance` and `wallBuildPriority`. Capability Index **84.2/100** (correctness **96.7**, instruction **79.7**, concision **45.9**, speed **48.0**). ~**160 s** wall time.

---

## 2026-04-29 — Full arena acceptance default (six tasks / three ACI tiers)

**Changed:** `scripts/run_arena_acceptance_tests.py` `DEFAULT_TASKS` now includes `**save-load-api-guard`** and `**ai-planning-explanation`** so default `arena-acceptance` runs populate **Smoke / Standard Dev / High-Reasoning** tier breakdown in `arena_capability_index.py`. Increased default `--timeout-s` to **14400** here and in `ml_workflow.py arena-acceptance` for six-task preview gates.

**Started:** `.venv/bin/python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/fe-lora-qwen25-coder-7b-latest --source-repo /Users/natreed/fallen-empire --worktree-root /Users/natreed/fallen-empire-arena` (background on agent host; can take up to 4 h).

---

## 2026-04-29 — Canvas: training + lexical progression charts

**Updated** Cursor canvas `canvases/arena-three-tier-progression.canvas.tsx`: embedded **val/train loss lines** from `20260428-195934_d451c9/training_trajectory.jsonl`, **bar chart** of final vs best val across three **7B** passes (chunked / chunk6k / latest), **line chart** of lexical capability index + pass rate across those passes, rubric table, footnote for **1.5B game_text** run, retained latest-run three-band task table.

---

## 2026-04-29 — Canvas: Arena ACI three tiers (Smoke / Standard / High-Reasoning)

**Updated** `canvases/arena-three-tier-progression.canvas.tsx` to chart **Arena Capability Index** `tier_breakdown` only: **Smoke Test**, **Standard Dev Benchmark**, **High-Reasoning Architecture Benchmark** from `arena_capability.json` runs **20260428-171007** vs **20260428-180627**, plus tier definition table and note on smoke-only **20260428-160917**; removed lexical-band / training-loss focus from this canvas.

**Follow-up:** Canvas adds chronological **Run 1–3** registry table, **“latest”** pill (Run 3), run numbers on chart axes and stats, and **Run 4** hint for the next embedded acceptance batch.

---

## 2026-04-29 — Canvas: Arena Capability Index tracker

**Added** `canvases/arena-capability-index-tracker.canvas.tsx`: run-numbered log table, headline ACI line chart, six-task-only ACI bar chart, completion/integration/efficiency lines, tier_breakdown columns; seeds from `arena_capability.json` for runs **20260428-160917**, **171007**, **180627**; documents how to append **Run 4+** after new acceptance batches.

---

## 2026-05-03 — Workflow cleanup + generated artifact path separation

**Goal:** Reduce workflow/docs drift and keep generated outputs out of source-doc paths under `benchmarks/`.

**Changed (code defaults):**

- `scripts/ml_workflow.py`
  - `arena-dashboard` now defaults to `benchmarks/results/arena_dashboard.html`.
  - `multi-adapter-report` now defaults to `benchmarks/results/multi_adapter_dashboard.json` and `benchmarks/results/multi_adapter_report.md`.
  - Updated command help text/docstring references to match the new artifact locations.
- `scripts/build_arena_dashboard.py` default `--out` now points at `benchmarks/results/arena_dashboard.html`.
- `scripts/build_multi_adapter_report.py` default `--out-json`/`--out-md` now point at `benchmarks/results/...`.

**Changed (docs):**

- `docs/WORKFLOW.md` updated command table/example paths for `arena-dashboard` and clarified that `multi-adapter-report` artifacts belong under `benchmarks/results/`.
- `docs/PROJECT_STATE.md` updated ops-report and arena-dashboard path references to `benchmarks/results/...`.
- `docs/ARENA_ROADMAP.md` and `docs/ARENA_PROGRESSION.md` updated to reference `benchmarks/results/arena_dashboard.html`.

**Outcome:** Workflow defaults now route generated HTML/JSON/MD outputs to `benchmarks/results/` (already gitignored), leaving `benchmarks/*.md` and `docs/*.md` focused on source documentation.

---

## 2026-05-03 — Routing: loading_screen false positives explained + eval re-run

**Context:** Hybrid/frontier mismatches on the 30-prompt loading-screen eval were not coming from ambiguous complexity in `router/policy.py` alone—they were substring overrides in `scripts/model_router.py` (`RoutingPolicy.frontier_keywords` / `hybrid_keywords`) firing before the classifier ladder outcome.

**Specific triggers:**

- `**performance`** matched “perceived **performance**”.
- `**review`** substring-matched “**review**ers”.
- `**production`** matched “**production**-ready”.
- (`**optimize`** is in `hybrid_keywords` too; prompts with the exact contiguous substring trigger hybrid when that path runs.)

**Code:** Loading-screen prompts already short-circuit in `RoutingPolicy.decide` when `adapter_id=="loading_screen"` and `risk_class!="high"` to use `plan_to_legacy_route(build_plan(...))`, suppressing generic keyword shortcuts while still honoring `force_route` and classifier-derived high-risk.

**Verification:** `python scripts/run_routing_benchmark.py --tasks benchmarks/loading_screen_eval_tasks_v1.json --mode both` → route **30/30**, adapter **30/30**; summaries in `benchmarks/results/routing_policy_summary_loading_eval_v1_keyword_override_fix.json` and row dump `benchmarks/results/routing_policy_rows_loading_eval_v1_keyword_override_fix.jsonl`.

---

## 2026-05-03 — Arena acceptance: `save-load-api-guard` (save_load_api_guard adapter)

**Command:** `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/save_load_api_guard/cycle1 --task-id save-load-api-guard` (registry still names `…/champion` but only **cycle1** exists on disk locally).

**Outcome:** **1/1** passed (`passed_all: true`), preview gate **ok**, ACI headline **100**/100 for the single task. Run dir: `benchmarks/results/runs/20260503-191922_510e7d/`.

---

## 2026-05-03 — Save/load specialist: 30-prompt routing eval + router alignment

**Data:** Added `data/routing/save_load_api_guard_eval_prompts_v1.jsonl` (30 curated prompts; mix of serialization/API/auth/production/review-ish wording). Benchmark tasks: `benchmarks/save_load_api_guard_eval_tasks_v1.json` (SHA256[:16] ids).

**Routing:** Extended `scripts/router/policy.py` with `**save_load_api_guard_specialist_fastpath`** so in-domain `**auth` / `security` / `api`** classifier signals don’t escalate to council/API; mirrored `**scripts/model_router.py`** specialist branch (suppress generic frontier/hybrid substring overrides when classifier selects this adapter).

**Prompt hygiene:** A few originals mis-scored (`**loading`** in “loading persisted” → `**loading_screen`**, `**serializing`** vs taxonomy `**serialization**`, stray `**compare**`, ambiguity on missing save keywords)—rewritten in the JSONL.

**Benchmark:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/save_load_api_guard_eval_tasks_v1.json --mode both` → **30/30** route + adapter (`benchmarks/results/routing_policy_summary_save_load_eval_v1.json`, `routing_policy_rows_save_load_eval_v1.jsonl`).

---

## 2026-05-03 — Mixed routing regression (specialists + policy fixtures)

**Artifact:** `benchmarks/mixed_routing_eval_v1.json` (**76** rows): shuffled (**seed 42**) mix of loading-screen (**30**) + save/load (**30**) curated eval tasks, `**task_routing_tasks.json`** policy rows (**12**), and four **specialist probes** (**hud / economy_tooltip / combat_risk / ai_planning_explanation**). Regenerate with `python scripts/build_mixed_routing_eval_v1.py`.

**Router:** Added `_force_frontier_over_save_specialist()` in `scripts/model_router.py` before the save-specialist shortcut: `**cryptography`** always escalates `**frontier`**; `**audit`** + `**authentication**`/`**authorization**` without `**save`/`serialization**` lexicon escapes false `**save_load_api_guard**` classification (fixes policy rows like security audit + crypto signing design).

**Run:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/mixed_routing_eval_v1.json --mode both` → route **76/76**, labeled adapter buckets **64/64**, overall **100%**; row trace `benchmarks/results/routing_policy_rows_mixed_eval_v1.jsonl`, summary `benchmarks/results/routing_policy_summary_mixed_eval_v1.json`. Re-checked save-only `**30/30`** after the frontier guard (`routing_policy_summary_save_load_after_mixed_guard.json`).

---

## 2026-05-03 — Third specialist integrated: `economy_tooltip` (routing + dataset path)

**Router:** `**economy_tooltip_specialist_fastpath`** in `scripts/router/policy.py`; matching specialist branch in `scripts/model_router.py` (suppresses generic `**frontier`** / `**hybrid`** substring overrides after `**save_load_api_guard`** and before blanket frontier keywords).

**Data:** Curated `**data/routing/economy_tooltip_eval_prompts_v1.jsonl`** (**30**) + `**benchmarks/economy_tooltip_eval_tasks_v1.json`**. Routing check: `**30/30`** route + adapter (`benchmarks/results/routing_policy_summary_economy_eval_v1.json`). Mixed `**76/76**` regression still passes; mixed `**economy_tooltip**` probe now reports `**economy_tooltip specialist route**`.

**Train path:** `**scripts/adapters/build_economy_tooltip_specialist_dataset.py`** outputs `**data/lora/adapters/economy_tooltip_specialist/`**; orchestrated via `**python scripts/ml_workflow.py economy-tooltip-dataset`** (defaults: transfer `**loading-screen-polish**` + `**hud-status-summary**`, `--min-train-core-rows` **100**).

**Registry:** `training/adapter_registry_v1.json` `**economy_tooltip`** `**adapter_path`** → `**checkpoints/adapters/economy_tooltip/cycle1`**, lineage `**economy_tooltip:v1:cycle1+routing_fastpath_v1`**, `**promotion_state**` remains `**shadow**` until arena `**economy-tooltip**` is proven independently of routing-only readiness.

**Docs:** `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `data/routing/README.md`.

---

## 2026-05-04 — Documentation specialist scaffolding (dataset + registry + orchestrator)

**Goal:** Simple “documentation agent” LoRA: canonical mlx-lab prose (paths, append-only docs, `ml_workflow` wording) without HUD/arena pairwise.

**Scripts:** Fixed bash newline escaping in `**scripts/adapters/build_documentation_specialist_dataset.py`**. `**scripts/ml_workflow.py`** new subcommand `**documentation-dataset**` → runs that builder with run manifest + `**docs/run_history.md**` row.

**Taxonomy/registry:** `**documentation`** added to `**LOCKED_ADAPTER_FAMILIES`**; `**TASK_TO_ADAPTER`** maps `**mlx-lora-docs-normalize**` → `**documentation**`. `**training/adapter_registry_v1.json**` entry: `**checkpoints/adapters/documentation/cycle1**`, lineage `**documentation_specialist:v1:cycle1**`, `**shadow**`. `**checkpoints/adapters/documentation/cycle1/adapter_config.json**` seeded (data → `**documentation_specialist/train.jsonl**`).

**Verify:** `.venv/bin/python scripts/ml_workflow.py documentation-dataset` (run `**20260504-032701_489e1b`**) wrote `**data/lora/adapters/documentation_specialist/`** (**120**/3/1 train/valid/test rows).

**Docs:** `**docs/WORKFLOW.md`** table row; `**docs/DATA_LAYOUT.md`** documentation specialist paths.

---

## 2026-05-04 — Documentation routing keyword eval + trained cycle1 weights

**Routing:** Expanded `**scripts/router/classifier.py`** phrase bank for `**documentation`**; `**build_plan`** + `**model_router**` specialist fastpaths mirror economy/loading semantics so frontier/hybrid substring hooks don’t steal mlx-doc prompts.

**Eval sources:** `**scripts/build_documentation_eval_tasks_v1.py`** → `**data/routing/documentation_eval_prompts_v1.jsonl`** + `**benchmarks/documentation_eval_tasks_v1.json`** (SHA256-prefix ids; regenerate with the script).

**Regression:** `**tests/test_documentation_routing_eval.py`** + `**tests/__init__.py`** exercise each task via `**RoutingPolicy`**, subprocess `**run_routing_benchmark.py`**, and (after train) `**adapters.safetensors**` presence.

**Train:** `**python scripts/ml_workflow.py documentation-dataset`** then `**train --adapter-path checkpoints/adapters/documentation/cycle1 -- --data …/documentation_specialist --iters 80 …`** (`**20260504-032912_d8cd21`**, exit 0).

**Docs:** `**data/routing/README.md`** listed the docs eval JSONL beside other specialists.

---

## 2026-05-04 — Mixed routing eval includes documentation shard

**Builder:** `scripts/build_mixed_routing_eval_v1.py` now merges `benchmarks/documentation_eval_tasks_v1.json` as `**mixed-doc-{id}`** rows (`shard: documentation_eval`) before the fixed specialist probes and policy fixtures; **94** tasks (**seed 42**).

**Regression:** `PYTHONPATH=scripts python scripts/run_routing_benchmark.py --tasks benchmarks/mixed_routing_eval_v1.json --mode both` → **94/94** route + labeled adapter (**82/82** adapter-scored rows), summary `benchmarks/results/routing_policy_summary_mixed_eval_v2_docs.json`.

---

## 2026-05-04 — Router supervisor Gradio (human try-out)

**Added:** `scripts/router_chat_gradio.py` — chats with **MLX** while each user turn runs `RoutingPolicy`; loads `adapters.safetensors` from `training/adapter_registry_v1.json` when present (documentation / economy_tooltip / loading_screen on this machine).

**Run:** `python scripts/router_chat_gradio.py` (default `**http://127.0.0.1:7862`**, distinct from `human_eval_ui.py` `**7861`** and `chat_gradio.py` `**7860**`).

**Fix:** (1) `mlx_lm.stream_generate` leaked `verbose` into `generate_step` — removed bogus `verbose=False`. (2) MLX `tokenizer.eos_token_ids` omitted `**151645` (`<|im_end|>`)**, so `stream_generate` never stopped cleanly on assistant end and Gradio echoed repeated sentinel strings (looked like a broken documentation LoRA). Added `**scripts/mlx_qwen_stop_tokens.py`** + post-`load()` registration wherever we stream (`**router_chat_gradio.py**`, `**chat_gradio.py**`, `**human_eval_ui.py**`, `**model_router.LocalMlxBackend**`, `**smoke_base_model.py**`). Optional supervisor JSONL (`**ROUTER_CHAT_LOG_JSONL**` or `**--interaction-log-jsonl**`) captures routing + prompts + generations.

---

## 2026-05-04 — Supervisor JSONL + decode budget tuned for reusable SFT rows

`**router_chat_gradio.py`:** default assistant decode ceiling raised (**2048** new tokens vs **896**) so economy/HUD/UI turns stop mid-structure less often; override with `**MAX_TOKENS`** / `**--max-tokens`**. Completed responses log `**mlx_finish_reason`**, token counts, `**generation_budget_hit**`, Markdown fence imbalance, `**recommended_for_sft_assistant_turn**`, `**schema_version: router_chat_supervisor_v1**`. UI emits a truncation banner when MLX hits `**length**` limits.

**System prompt:** nudges complete structured Markdown/fenced replies for gameplay/UI workloads.

**Docs:** `**docs/ROUTING_DATASET_CONTRACT.md`** — appendix on merging supervisor completions into downstream datasets.

---

## 2026-05-10 — Combat-risk specialist rank-16 pilot (cycle4) + harness blockers

**Goal:** Run a fresh `combat_risk` specialist training pass at rank 16 and validate it with post-train tests.

**Environment repair before training:**

- Restored missing workflow helper `scripts/fe_lineage.py` (required by `scripts/ml_workflow.py` import).
- Restored missing 7B LoRA config `training/lora_qwen25_coder_7b.yaml` (workflow was failing `FileNotFoundError` before this).

**Training attempts:**

- `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/combat_risk/cycle4 -- --data data/lora/adapters/combat_risk_specialist --iters 120 ... --resume-adapter-file checkpoints/adapters/combat_risk/cycle3/adapters.safetensors`
  - run `20260511-055208_6579da`: failed (`exit -6`) with Metal OOM on first validation.
  - run `20260511-055258_ec5f0e`: failed (`exit -6`) with Metal OOM again at reduced `max_seq_length=3072`.
  - run `20260511-055346_55c856`: failed (`exit -6`) even with `max_seq_length=2048` and `val_batches=0` (mlx still executed iter-1 val pass).
- Constrained successful pilot:
  - `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/combat_risk/cycle4 -- --data data/lora/adapters/combat_risk_specialist --iters 40 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/combat_risk/cycle3/adapters.safetensors`
  - run `20260511-055420_4d4ef6`: success (`exit 0`), duration ~402.6s, final train loss `0.019`, final val loss `0.995`, best val `0.011` at iter 1.

**Post-train test attempts:**

- `python scripts/ml_workflow.py arena-acceptance --adapter-path checkpoints/adapters/combat_risk/cycle4 --task-id combat-risk-preview`
  - run `20260511-060110_dadf82`: failed immediately (`exit 2`) because `scripts/run_arena_acceptance_tests.py` is missing from this checkout.
- `python scripts/run_game_benchmark.py --adapter-path checkpoints/adapters/combat_risk/cycle4 --profile game`
  - failed due base/adapter mismatch (benchmark defaulted to 1.5B model; adapter is 7B lineage).
- Re-ran benchmark with matching model:
  - `python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --adapter-path checkpoints/adapters/combat_risk/cycle4 --profile game`
  - launched and running at session close.

---

## 2026-05-11 — Test-policy cleanup: ACI + multi-task specialist benchmarks

**Goal:** Drop legacy generic benchmark-test flow and align workflow/testing toward ACI + specialist-focused benchmark checks.

**Changed files:**

- Added `benchmarks/specialist_benchmark_tasks.json` with specialist-tagged benchmark prompts and **multiple tasks per specialist** (loading_screen, hud_status, economy_tooltip, combat_risk, save_load_api_guard, ai_planning_explanation).
- Updated `scripts/run_game_benchmark.py` to:
  - default to `benchmarks/specialist_benchmark_tasks.json`,
  - support repeated `--specialist <id>` filters,
  - remove legacy `--profile game|general` branch from the runner surface.
- Updated `scripts/ml_workflow.py` benchmark wiring:
  - `benchmark` now targets specialist tasks (`--specialist` and `--tasks`),
  - `train --evaluate` and `full` now use repeated `--bench-specialist` filters instead of `--bench-profile`,
  - internal benchmark invocation forwards specialist filters and specialist task file path.
- Updated docs:
  - `docs/WORKFLOW.md` command table/examples now describe specialist benchmark filters.
  - `docs/PROJECT_STATE.md` benchmark section now points to specialist task definitions and specialist-filter benchmark commands.

**Intent/result:** Primary acceptance remains arena ACI (`ml_workflow.py arena-acceptance`), while lexical checks are now specialist-scoped and no longer centered on old game/general profile splits.

---

## 2026-05-11 — Economy + HUD rank-16 specialist retrains and targeted tests

**Goal:** Run fresh rank-16 specialist passes for failing `economy_tooltip` and `hud_status`, then test each with specialist benchmarks plus ACI-style task verification.

**Training (rank 16, constrained memory profile):**

- Economy:
  - `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/economy_tooltip/cycle3 -- --data data/lora/adapters/economy_tooltip_specialist --iters 40 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/economy_tooltip/cycle2/adapters.safetensors`
  - Run `20260511-172447_07db29` exit 0; final train `0.005`, final val `1.073`, best val `0.003` @ iter 20.
- HUD:
  - `python scripts/ml_workflow.py train --adapter-path checkpoints/adapters/hud_status/cycle3 -- --data data/lora/adapters/hud_status_specialist --iters 40 --batch-size 1 --val-batches 1 --steps-per-eval 20 --steps-per-report 10 --save-every 20 --learning-rate 1e-5 --max-seq-length 1024 --resume-adapter-file checkpoints/adapters/hud_status/cycle2/adapters.safetensors`
  - Run `20260511-173046_2c7554` exit 0; final train `0.04`, final val `0.933`, best val `0.002` @ iter 20.

**Specialist benchmark tests (multi-task per specialist):**

- Economy `cycle3`:
  - `python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --adapter-path checkpoints/adapters/economy_tooltip/cycle3 --specialist economy_tooltip`
  - **2/2 pass**, capability **92.9/100**.
- HUD `cycle3`:
  - `python scripts/run_game_benchmark.py --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --adapter-path checkpoints/adapters/hud_status/cycle3 --specialist hud_status`
  - **1/2 pass**, capability **62.5/100** (`hud-status-signal-selection` failed required lexical fields).

**ACI-style task verification via `game_task_arena` (task-specific generate/apply/verify):**

- Economy task `economy-tooltip` with `cycle3`:
  - trial `20260511-173710_economy-tooltip`
  - `apply_status: no_applyable_changes` (model output lacked repo-relative fenced path), so not a valid applyable patch.
  - verify originally failed due missing deps in disposable worktree (`tsconfig-paths/register`), then passed after `npm install`; final signal still blocked by non-applyable output.
- HUD task `hud-status-summary` with `cycle3`:
  - trial `20260511-173846_hud-status-summary`
  - `apply_status: wrote_files:src/components/test/overlays/HudStatusSummary.tsx`
  - verify originally failed due missing deps in disposable worktree, then passed after `npm install`.

---

## 2026-05-11 — Promote HUD/combat + mixed routing check for promoted specialists

**Registry promotion updates:**

- `training/adapter_registry_v1.json`
  - `hud_status` → `adapter_path: checkpoints/adapters/hud_status/cycle3`, `promotion_state: champion`, lineage `hud_status:v3:cycle3`.
  - `combat_risk` → `adapter_path: checkpoints/adapters/combat_risk/cycle4`, `promotion_state: champion`, lineage `combat_risk:v2_tsc_resume:cycle4`.

**Mixed routing prompt check (promoted specialists):**

- Ran an inline mixed prompt harness against `scripts/model_router.py` policy for promoted specialist intents (`loading_screen`, `hud_status`, `combat_risk`), expecting `local`.
- Result: **4/6** passed.
  - Loading-screen prompts: local/local.
  - HUD prompts: local/local.
  - Combat prompts: routed **hybrid** (2 failures) because substring keyword matching in router treats `preview` as containing `review`.

**Interpretation:** promotion metadata is updated, but deterministic routing policy still has a `preview`→`review` false-positive path for combat-preview phrasing and should be hardened before claiming routing is fully working for all promoted specialists.

---

## 2026-05-11 — Documentation site: auto-refresh training-data tab on training-page edits

**Goal:** Make documentation-site updates follow training-page changes and expose specialist training data in an organized tabbed view.

**Changed files:**

- Added `scripts/build_training_data_dashboard_cache.py`:
  - scans `data/lora/adapters/`*,
  - captures split metadata (`train/valid/test` row counts + size + mtime) and `manifest.json`,
  - writes `data/training_dashboard/training_data_catalog.json` for dashboard reads.
- Updated `scripts/trigger_doc_training_on_changes.py`:
  - introduced `TRAINING_PAGE_PREFIXES` for training-page edits (`docs/WORKFLOW.md`, `training/README.md`, `training/`, `scripts/ml_workflow.py`),
  - runs the new cache builder on matching edits,
  - records `training_dashboard_cache` action status in queue rows.
- Updated `scripts/private_dashboard_server.py`:
  - new endpoints: `GET /api/training/catalog`, `GET /api/training/content`,
  - new **Training Data** top-level dashboard tab,
  - dataset-tab + split-tab UX (`train`, `valid`, `test`, `manifest`) with row pagination and auto-refresh.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py scripts/trigger_doc_training_on_changes.py scripts/build_training_data_dashboard_cache.py` → success.
- `python3 scripts/build_training_data_dashboard_cache.py` wrote `data/training_dashboard/training_data_catalog.json`.

---

## 2026-05-15 — Added centralized local site command list

**Goal:** Provide one copy-paste command reference for launching/opening each local web UI/site in this repo.

**Changed files:**

- Added `docs/SITE_COMMANDS.md` with:
  - shared venv setup commands,
  - launch + open commands for `chat_gradio`, `human_eval_ui`, `train_ui_gradio`,
  - launch + open commands for `landing_page_arena` UI, `router_chat_gradio`, and `game_task_arena` UI,
  - private dashboard startup env vars + launch/open command,
  - static dashboard build/open commands for arena and run dashboards.

**Result:** Site startup instructions are now centralized instead of scattered across `docs/PROJECT_STATE.md` and `docs/WORKFLOW.md`.

---

## 2026-05-15 — Site-surface cleanup to three canonical UIs

**Goal:** Reduce UI drift by making only three local sites supported: documentation dashboard, arena training supervision, and router prompt lab.

**Changed files:**

- Updated `docs/SITE_COMMANDS.md` to only list:
  - `scripts/private_dashboard_server.py` (docs + cost dashboard),
  - `scripts/game_task_arena.py ui` (specialist-vs-frontier arena supervision),
  - `scripts/router_chat_gradio.py` (agentic router/specialist prompt testing).
- Added canonical site policy notes in `docs/PROJECT_STATE.md` and `docs/WORKFLOW.md`.
- Deprecated legacy UI entrypoints by default:
  - `scripts/chat_gradio.py`
  - `scripts/human_eval_ui.py`
  - `scripts/train_ui_gradio.py`
  - `scripts/landing_page_arena.py ui`
  - each now requires explicit `--allow-legacy-ui` to run.

**Verification:**

- `python3 -m py_compile scripts/chat_gradio.py scripts/human_eval_ui.py scripts/train_ui_gradio.py scripts/landing_page_arena.py` (pass).
- `python3 scripts/chat_gradio.py` now exits with deprecation message listing the three supported sites.
- `python3 scripts/human_eval_ui.py` now exits with deprecation message listing the three supported sites.
- `python3 scripts/train_ui_gradio.py` now exits with deprecation message listing the three supported sites.
- `python3 scripts/landing_page_arena.py ui` now exits with deprecation message unless `--allow-legacy-ui` is provided.

---

## 2026-05-15 — Prompt-driven adapter routing fix for router site

**Goal:** Ensure router prompt classification uses the adapter system directly so specialist prompts (including loading-screen requests) do not silently drop to `general_fallback`.

**Changed files:**

- Updated `scripts/model_router.py`:
  - extended `RouteDecision` with adapter metadata (`adapter_id`, `confidence`, `ambiguity`, `risk_class`, `complexity`),
  - added prompt-driven specialist classification against registry-backed adapter ids,
  - added explicit loading-screen phrase handling (`loading screen`, `load screen`, `splash screen`) including common typo `medival`,
  - merged route + adapter rationale into decision reason.
- Updated `scripts/router_chat_gradio.py`:
  - default supervisor log sink is now `benchmarks/results/router_chat_interactions.jsonl` (unless overridden),
  - when a specialist is selected but adapter weights are missing locally, UI reason now records a base-model fallback note.
- Updated docs:
  - `docs/PROJECT_STATE.md` router section now documents prompt-driven adapter routing and default supervisor logging.

**Verification:**

- `python3 -m py_compile scripts/model_router.py scripts/router_chat_gradio.py` (pass).
- Reproduced prompt routing:
  - prompt: `create a new loading screen for the game. Make it in theme with royal colors and medival looking graphics.`
  - decision: `route=local`, `adapter_id=loading_screen`, `confidence=0.92`, reason includes `explicit loading-screen phrase match`.

---

## 2026-05-15 — Router UI polish: cleaner shell + collapsed docs/examples

**Goal:** Make the router site feel cleaner and move descriptive content below the active chat interface.

**Changed files:**

- Updated `scripts/router_chat_gradio.py` UI layout:
  - introduced a sleeker visual theme (lighter glass-card styling, softer shadows, tighter typography),
  - moved long-form description and example prompts into a collapsed `Details and examples` accordion **below** the chat interface,
  - kept routing controls available via a collapsed `Routing controls` accordion in-chat,
  - removed inline top-level examples list from `ChatInterface` for a cleaner first screen.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Confirmed installed Gradio accepts `additional_inputs_accordion` usage in this environment.

---

## 2026-05-15 — Router accordion label contrast follow-up

**Goal:** Fix remaining low-contrast accordion trigger text (`Routing controls`) on the light theme.

**Changed files:**

- Updated `scripts/router_chat_gradio.py` theme CSS with explicit dark text selectors for:
  - `.label-wrap` and `button.label-wrap`,
  - accordion trigger/header buttons and nested spans,
  - accordion icon/label wrappers across instances.
- Applied `color: #0f172a`, `opacity: 1`, and `-webkit-text-fill-color: #0f172a` as `!important` for those controls.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Training dashboard: hide legacy datasets + low-row warnings

**Goal:** Fix misleading training-data explorer presentation by defaulting to specialist datasets, hiding legacy/base datasets, and flagging tiny datasets.

**Changed files:**

- Updated `scripts/private_dashboard_server.py` training UI:
  - added `Hide legacy datasets` toggle (default on),
  - marked base datasets as legacy when a matching `*_specialist` dataset exists,
  - filtered dataset tabs to hide legacy rows by default,
  - sorted tabs with specialist-first + larger datasets first,
  - added warning badges (`⚠ under 10`) on low-row datasets,
  - added legacy badges when legacy rows are shown,
  - improved meta summary to include visible count, hidden legacy count, and low-row flagged count.

**Verification:**

- `python3 -m py_compile scripts/private_dashboard_server.py` (pass).
- Lint check for `scripts/private_dashboard_server.py` returned no errors.

---

## 2026-05-15 — Router site unified with frontier compare + grading

**Goal:** Merge prompt router and GPT-vs-specialized workflow into one site, and make bug-loop behavior explicit per lane.

**Changed files:**

- Updated `scripts/router_chat_gradio.py`:
  - replaced single-lane chat flow with one-prompt compare flow (specialist router lane vs frontier lane),
  - added frontier controls (`--frontier-model`, `--frontier-base-url`) and compare feedback sink (`--compare-log-jsonl`),
  - kept specialist routing controls (Auto / Codebase OSS, adapter lock) and reused registry-based adapter resolution,
  - added explicit bug-check execution telemetry for both lanes (rounds + changed/not-changed),
  - added in-UI grading (`winner`, specialist/frontier scores, notes) with append-only save rows.
- Updated `docs/PROJECT_STATE.md` router section to document the new unified compare+grade workflow and log paths.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

## 2026-05-15 — Combat preview prompt routing correction

**Goal:** Fix router misclassification where combat-preview prompts were routed to fallback/hybrid due keyword collisions.

**Changed files:**

- Updated `scripts/model_router.py`:
  - switched keyword checks from raw substring matching to boundary-aware regex matching (prevents `review` matching inside `previewing`),
  - expanded `combat_risk` specialist signals (`combat preview`, `combat previewing`, `enemy stats`, `versus your own`, `vs your own`).

**Verification:**

- `python3 -m py_compile scripts/model_router.py` (pass).
- Prompt check:
  - `create a new system for combat previewing where you can view the enemies stats versus your own`
  - decision now: `route=local`, `adapter_id=combat_risk`, confidence `0.79`.

---

## 2026-05-15 — Router typed/user message visibility follow-up

**Goal:** Fix remaining white/invisible chat text symptoms where typed user content and generated content could disappear after submit.

**Changed files:**

- Updated `scripts/router_chat_gradio.py` CSS:
  - forced dark text + text-fill on all message descendants (`.message` *, user/bot variants),
  - preserved readable code-block contrast by re-overriding `pre/code` descendants,
  - added explicit input text fill + caret color for `input/textarea/select`.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Router generation text visibility fix

**Goal:** Fix chat output where generated text became effectively invisible (white-on-white appearance) after sending a prompt.

**Changed files:**

- Updated `scripts/router_chat_gradio.py` CSS:
  - explicitly forced dark readable text inside `.message` markdown containers,
  - stabilized user/bot message bubble backgrounds to near-white for consistent contrast,
  - kept dedicated dark-theme overrides for `pre/code` blocks intact.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-15 — Router output panel contrast/cropping follow-up

**Goal:** Resolve output-pane readability issues after prior UI color overrides (dark panel + low-contrast code text appearance).

**Changed files:**

- Updated `scripts/router_chat_gradio.py` CSS scope:
  - removed broad global text-color overrides that affected all markdown/message content,
  - limited forced dark text to the top hero copy via `elem_id="router-hero"`,
  - kept accordion label-specific dark selectors,
  - explicitly restored readable light text for assistant code blocks (`.message pre/code`) to match dark code backgrounds.

**Verification:**

- `python3 -m py_compile scripts/router_chat_gradio.py` (pass).
- Lint check for `scripts/router_chat_gradio.py` returned no errors.

---

## 2026-05-31 — economistRL MLX PPO logprob + LoRA-only save fix

**Goal:** Stop PPO from NaN-corrupting adapters (NaN grads on first step, `!!!!` generation on next cycle).

**Root cause:** `_mean_completion_logprob_mlx` used `log(softmax(...))`, which yields non-finite MLX autograd; `train_ppo_batch_mlx` optimized/saved all 533 `trainable_parameters()` (LoRA + layernorms/biases). Passing zero-grad updates for non-LoRA keys via `optimizer.update(model, …)` also corrupted MLX state after the first step.

**Changed files:**

- `scripts/economist_rl_ppo_trainer.py` — stable log-softmax via `logsumexp`; LoRA-only `value_and_grad`, in-place `optimizer.update(lora_params, lora_grads)` + merge back into model; save 392 LoRA tensors only; manifest `saved_lora_tensors`.
- `tests/test_economist_rl_lambda_ppo_pipeline.py` — unit tests for finite MLX logprob grads and LoRA key filter.

**Verification:**

- `PYTHONPATH=scripts:tests .venv/bin/python -m unittest discover -s tests -p 'test_economist_rl*.py' -q` — 27 tests OK.
- Integration: 4-sample MLX PPO from `seed_bootstrap` → 392-key safetensors, 0 NaN tensors, finite batch loss.

**Note:** Delete or ignore corrupted `rl_pass_003`–`006` checkpoints before re-smoking cycles.

**Follow-up smoke (cycles 007–008):** Local 2×10-rollout run (~19 min). PPO losses finite; `saved_lora_tensors=392`, 0 NaN in `rl_pass_007`/`008`; cycle 008 chained from `rl_pass_007` with coherent rollouts (no `!!!!` collapse). Promotion `kept_current` (registry stays `seed_bootstrap`; eval on generalist tail hurt vs seed/007).

---

## 2026-06-03 — economistRL sandbox vitest environments (subsection envs + v3 retag)

**Goal:** Per-subsection TypeScript env packages so vitest can execute fictional mechanics (`lastPrice`, wage curves, etc.) without touching production `empireEconomy.ts`; align stubs/tests/prompts.

**Added:** `scripts/economist_rl_sandbox_envs.py` — `src/lib/economistRl/envs/<subsection>/{types,runTicks,index}.ts`, subsection-aware `mechanic.ts` + vitest starters, scarcity/surplus seeds.

**Changed:** `economist_rl_coding_contract.py` (`sandbox_coding_user_prompt`, narrow allowed paths); `economist_rl_task_execution.py` (env paths in allowed/starter/context); `tag_economist_rl_task_execution.py` (`--refresh-all`); tests `test_economist_rl_sandbox_envs.py`, updated `test_economist_rl_task_execution.py`.

**Regenerated:** `benchmarks/economistRL_tasks_v3_execution.json` (500 sandbox tasks, 5 starter files each) via `--refresh-all` from v2.

**Verify:** `unittest` sandbox + task_execution + execution_evidence tests OK.

---

## 2026-06-02 — economistRL P0: stricter vitest deltas + v3 retag

**Goal:** Close cycle-015 false compile pass (`economistRL-production-cache-invalid-06` no-op game loops on numeric sandbox state); make `compiled` reflect real mechanic motion.

**Changed:**
- `scripts/economist_rl_sandbox_envs.py` — `runTicks` clones input each tick; all subsection `test_stub_body` branches assert trace deltas + cross-seed checks; cache stub clears `projectionCacheDirty` after scopes drain; cache seeds/tick counts tuned for mid-trace vs end comparisons.
- `benchmarks/economistRL_tasks_v3_execution.json` — `--refresh-all` retag (500 tasks, new frozen starters/tests).
- `tests/test_economist_rl_sandbox_envs.py` — delta template asserts, reference stub vitest (market + cache), cycle-015 no-op patch must fail vitest.
- `docs/ECONOMIST_RL_ADAPTER.md` — document delta grading + in-place v3 refresh command.

**Verification:** `PYTHONPATH=scripts:. .venv/bin/python -m unittest discover -s tests -p test_economist_rl_sandbox_envs.py` — 11 OK (includes vitest against `/Users/natreed/fallen-empire`).

**Next:** Re-run 20-rollout preflight (`cycle_016`) and compare `compiled` rate vs batch 015 (was 1/20).

---

## 2026-06-04 — economistRL cycle 016 Lambda launch (20-rollout P0 preflight)

**Command:** `.venv/bin/python scripts/launch_economist_rl_lambda_cycle.py --launch-instances --game-repo /Users/natreed/fallen-empire -- --cycles 1 --rollouts-per-cycle 20 --eval-limit 18 --init-adapter-path checkpoints/fe-lora-arena-apply-sft --task-db benchmarks/economistRL_tasks_v3_execution.json --execution-source-repo /home/ubuntu/fallen-empire --temperature 0.2 --max-tokens 4000 --skip-ppo --skip-eval`

**Worker:** `gpu_1x_a10` `us-west-1`; instance `c924eac12d44434eafa04af670e3c859` @ `146.235.197.204`; tmux `fe-economist-rl`; log `logs/cycle_016_lambda_launch.log` + remote `~/cloud-eval-logs/fe-economist-rl-cycle.log`. `auto_terminate=1`, watchdog 180 min idle.

**Outcome:** Launcher exit 0 (`economist_rl_lambda_cycle_started`). Await remote `rollout_batch_016.jsonl` / `scored_batch_016.jsonl` rsync.

**2026-06-04 follow-up:** Execution evidence hung ~10 min/task on `npx vitest` interactive install (worktrees lack `node_modules`). Killed remote cycle; rsynced partial 6-task logs to `benchmarks/results/economistRL/lambda_extract_20260604_partial/`. Patched `vitest_run_command()` → `node_modules/.bin/vitest`, worktree `node_modules` symlink, compile subprocess killpg on timeout.

---

## 2026-06-03 — economistRL: remove stub SFT, arena apply-sft init, cycle 014 smoke

**Goal:** Drop stub seed SFT from the RL pipeline; initialize rollouts from `checkpoints/fe-lora-arena-apply-sft`; 10-task smoke without PPO/eval OOM.

**Changed:**
- Deprecated `data/lora/adapters/economistRL_seed` → `_deprecated_economistRL_stub_sft/`; moved `seed_bootstrap` checkpoints under `_deprecated_seed_bootstrap_stub/`.
- Disabled `build_economist_rl_dataset.py` + `ml_workflow.py economist-rl-dataset`; `training/economistRL_lora_qwen25_coder_7b.yaml` marked `train: false`.
- Registry `economistRL` → `adapter_path: checkpoints/fe-lora-arena-apply-sft`.
- `run_economist_rl_lambda_cycle.py`: default arena init adapter, `--skip-ppo`/`--skip-eval`, no starter bodies in prompt.
- Cycle **014** (`--skip-ppo --skip-eval`, 10 rollouts): `load_reason=init_adapter_path`, `cycle_status=rollouts_only`.

**014 vs 013 rollouts:** 0/10 stub comments (was 6/10); longer outputs (~4.6k avg vs ~1.9k); real `empireEconomy`/`useGameStore` imports; fence format often wrong (```ts path= missing — ` ```ts src/...` only). Apply/compile not re-audited in this pass (mean_reward still 0.1).

**Artifacts:** `benchmarks/results/economistRL/rollouts/rollout_batch_014.jsonl`, `manifests/cycle_014_manifest.json`, log `logs/cycle_014_arena_init_10.log`.

---

## 2026-06-02 — economistRL: gate simulation on compile

**Change:** `simulation_behavior` scores 0 unless `compiled is True` when execution evidence ran; `runAppliedSimCli` skipped if Vitest did not run. Failure tag `simulation_gated_on_compile`.

**Files:** `economist_rl_reward_engine.py`, `economist_rl_execution_evidence.py`, `tests/test_economist_rl_ppo_reward_semantics.py`, `docs/ECONOMIST_RL_ADAPTER.md`.

---

## 2026-06-02 — economistRL reward: applied sim (38%) + Vitest partial credit (27%)

**Goal:** Lower text-sim weight; score `simulation_behavior` from applied `mechanic.ts` traces; grade Vitest by assertion pass rate.

**Changed files:**

- `scripts/economist_rl_reward_engine.py` — weights 38% / 27%; ignore `simulation_source=completion_text` when execution ran.
- `scripts/economist_rl_applied_sim.py`, `scripts/economist_rl_vitest_scoring.py` — applied sim CLI + JSON vitest parse.
- `scripts/economist_rl_execution_evidence.py` — vitest `--reporter=json`, applied sim, partial `targeted_tests`.
- `scripts/economist_rl_game_sandbox.py` — `runAppliedSimCli.ts` in shared env package.
- `tests/test_economist_rl_vitest_scoring.py`, `docs/ECONOMIST_RL_ADAPTER.md`.

**Verification:** `unittest tests.test_economist_rl_vitest_scoring tests.test_economist_rl_sandbox_envs`.

---

## 2026-06-02 — economistRL sandboxes extend game codebase (City + processEconomyTurn)

**Goal:** Match original intent — sandboxes extend Fallen Empire mechanics for vitest, with `taskExt` only where production has no field (market price, projection invalidation).

**Changed files:**

- `scripts/economist_rl_game_sandbox.py` — `game_turn` (food/inventory/labor → `processEconomyTurn`), `hybrid_city` (market/upkeep/cache/adversarial → `City` + `taskExt`), `envs/shared/fixture.ts`.
- `scripts/economist_rl_sandbox_envs.py`, `scripts/economist_rl_coding_contract.py` — wire generators + prompts.
- `docs/SANDBOX_GAME_FIELD_ALIGNMENT.md` — architecture (intent vs gaps).
- `tests/test_economist_rl_sandbox_envs.py`.

**Verification:** `unittest tests.test_economist_rl_sandbox_envs` (11 OK, vitest reference stubs). `--refresh-all` on v3 bank (500 tasks).

**Gaps (documented):** empty territory in default fixtures; upkeep not wired to `upkeepTick`; no single-phase export from `gameLoop`.

---

## 2026-06-02 — economistRL sandbox field names ↔ Fallen Empire vocabulary

**Goal:** Align sandbox env `types.ts`, mechanic/test stubs, and sim harness seeds with production naming (`City.storage.food`, `POP_BIRTH_RATE`, …) so rollouts that use game-shaped language can map to vitest state without a second fictional schema.

**Changed files:**

- `scripts/economist_rl_field_names.py` — `SCENARIO_FIELD_ALIASES`, `SUBSECTION_CANONICAL_DEFAULTS`, `canonicalize_state()`.
- `scripts/economist_rl_sandbox_envs.py` — canonical defaults, seeds, stubs, tests (e.g. `storageFood`, `marketPriceGold`, `resourceProjectionValid`).
- `scripts/economist_rl_sim_harnesses.py` — `FIELD_DEFAULTS` + `resolve_scenario_state` use canonical keys.
- `docs/SANDBOX_GAME_FIELD_ALIGNMENT.md`, `docs/ECONOMIST_RL_ADAPTER.md` (pointer).
- `tests/test_economist_rl_field_names.py`, `tests/test_economist_rl_sandbox_envs.py` (assertions).

**Verification:** `python3 -m unittest tests.test_economist_rl_field_names tests.test_economist_rl_sandbox_envs` (13 OK). Retag: `tag_economist_rl_task_execution.py --refresh-all` → 500 sandbox starters refreshed.

---

## 2026-06-02 — economistRL execution tagging (v3 bank, BM25 context, sandbox verify)

**Goal:** Tag tasks with arena vs integration execution, starter paths/worktree scaffolds, 4000-token rollouts with BM25 Fallen Empire context, and per-task `verify_commands` (sandbox vitest vs cohort).

**Changed files:**

- `scripts/economist_rl_task_execution.py` — execution modes, sandbox paths under `src/lib/economistRl/<slug>/`, starters, `build_rollout_user_prompt`, eval manifest resolver.
- `scripts/adapters/tag_economist_rl_task_execution.py` — v2 → `benchmarks/economistRL_tasks_v3_execution.json`.
- `benchmarks/economistRL_eval_manifest_v1.json` — stratified eval (6 arena + 8 sandbox + 4 generalist).
- `scripts/game_task_arena.py` — `build_context_pack_for_task_spec()` for economist BM25 packs.
- `scripts/lambda/run_economist_rl_lambda_cycle.py` — default `--max-tokens 4000`, context packs, eval manifest, `generation_prompt` on rollouts.
- `scripts/economist_rl_execution_evidence.py` — seed starters before apply; per-task verify commands.
- `scripts/economist_rl_coding_contract.py` — sandbox path stubs + `sandbox_starter_bodies`.
- `tests/test_economist_rl_task_execution.py`, `tests/test_economist_rl_execution_evidence.py`.

**Verification:** `python tests/test_economist_rl_task_execution.py` (5 OK); `python tests/test_economist_rl_execution_evidence.py` (2 OK). Tag summary: 500 tasks, all `integration` + `sandbox`, 500 with `starter_files` (`ECONOMIST_RL_SOURCE_REPO=/Users/natreed/fallen-empire`).

---

## 2026-06-02 — Vitest encodes task-bank goals; drop simulation_behavior reward

**Goal:** Remove redundant `simulation_behavior` training slice; load every `simulation_spec.goals[]` entry into generated Vitest `it('goal_*')` blocks; redistribute weight to `targeted_tests` (55%).

**Changed files:**

- `scripts/economist_rl_vitest_goals.py` — goal → assert templates, fallback for generalist rubric goals, `render_vitest_stub`.
- `scripts/economist_rl_sandbox_envs.py` — `test_stub_body` delegates to goal-driven Vitest.
- `scripts/economist_rl_vitest_scoring.py` — per-`it()` parsing; `vitest_goal_weighted` merge using goal weights.
- `scripts/economist_rl_reward_engine.py` — weights: `targeted_tests` 55%, no `simulation_behavior` in base sum or scorecard threshold.
- `scripts/economist_rl_execution_evidence.py` — stop `runAppliedSimCli` / applied sim on rollouts.
- `scripts/economist_rl_evidence_runner.py` — no text `simulation_results`; `simulation_source=vitest_goals`.
- `docs/ECONOMIST_RL_ADAPTER.md` — reward docs updated.
- Tests: `test_economist_rl_vitest_goals.py`, harness/sandbox/food/market/PPO semantics updates.

**Verification:** `python3 -m unittest tests.test_economist_rl_vitest_goals tests.test_economist_rl_vitest_scoring tests.test_economist_rl_ppo_reward_semantics tests.test_economist_rl_sim_harness_coverage tests.test_economist_rl_sandbox_envs tests.test_economist_rl_food_steady_state tests.test_economist_rl_market_elasticity` — pass (sandbox vitest integration test skipped when FE repo absent).

**Next:** Re-tag v3 `reference_answer` test fences via `tag_economist_rl_task_execution.py --refresh-all` so bank examples match multi-`goal_*` Vitest stubs.

---

## 2026-06-05 — Lambda overnight economistRL PPO (3×50 chained)

**Launch:** `scripts/launch_economist_rl_lambda_cycle.py --launch-instances --region us-west-1 …`  
**Instance:** `fe1acd40cee64b80876badbfa9f9b501` @ `159.54.181.215`  
**Remote:** `tmux` `fe-economist-rl`, log `~/cloud-eval-logs/fe-economist-rl-cycle.log`  
**Cycle cmd:** `--lambda-mode --cycles 3 --rollouts-per-cycle 50 --skip-eval --ppo-max-samples 12 --ppo-epochs 1` (Transformers rollouts + CUDA PPO; `ECONOMIST_RL_SOURCE_REPO=~/fallen-empire`, Vitest via `npm ci`).  
**Local log:** `logs/launch_economist_rl_overnight.log`  
**Verified:** tmux running; cycle 001 loading HF+PEFT from `fe-lora-arena-apply-sft`.

---

## 2026-06-05 — PPO cycles 019–020 (cancelled after cycle 020 rollouts)

**Run:** 5×50 PPO attempt; cycle 019 rollouts+score; PPO completed manually (`ppo_train_019.json`, status `trained`, 12 samples, ~46m). Cycle 020 rollouts+score complete; PPO killed (exit 137) before manifest. Cycles 021+ cancelled.

**Fixes during run:** `refresh_old_logprobs=False` by default; `PPOConfig.max_samples=16`; `--ppo-max-samples` CLI; skip redundant MLX logprob refresh on rollout-attached `old_logprob`.

**Extract:** `benchmarks/results/economistRL/extracts/cycles_019_020_summary.json` via `scripts/extract_economist_rl_cycle_data.py`.

**Note:** Separate `--cycles 4` invocation for 020+ did **not** chain `rl_pass_019` (used default init again) — multi-cycle must be one invocation or omit `--init-adapter-path` after cycle 1.

---

## 2026-06-02 — Step-2 cycle 018 mechanic-only 20-rollout smoke

**Command:** `run_economist_rl_lambda_cycle.py --cycles 1 --rollouts-per-cycle 20 --skip-ppo --skip-eval --init-adapter-path checkpoints/fe-lora-arena-apply-sft --task-db benchmarks/economistRL_tasks_v3_execution.json --execution-source-repo /Users/natreed/fallen-empire --temperature 0.2 --max-tokens 4000` (~9.7 min MLX).

**Artifacts:** `rollout_batch_018.jsonl`, `scored_batch_018.jsonl`, `manifests/cycle_018_manifest.json`, log `logs/cycle_018_smoke_20.log`.

**Results:** mean_reward **0.433**; `compiled` **20/20**; `vitest_goal_weighted` **20/20**; `targeted_tests` **>0 on 5/20**, **≥50 on 4/20** (mean component **18.7%**); **0/20** rollouts wrote `tests/` (mechanic-only apply). Best: worker-03 **100%** tt / reward **0.80**; food-01 **58.8%** tt. Compare cycle 017 (3 rollouts, temp 0.0): **0/3** behavioral pass — variance is task/sample dependent, not infra.

**Step-2 gate:** infra OK for PPO trial; behavioral signal sparse but non-zero.

---

## 2026-06-02 — Step-1 oracle sanity + game_turn Vitest imports

**Command:** `PYTHONPATH=scripts ECONOMIST_RL_SOURCE_REPO=/Users/natreed/fallen-empire python scripts/oracle_reference_sanity.py --limit 3`

**Finding:** `game_turn` reference mechanics used `@/lib/gameLoop`; Vitest in disposable worktrees failed before any `it()` ran (`per_it` 0/0) while `hybrid_city` (relative imports) passed 4/4.

**Fix:** `mechanist_stub_body` game_turn branch uses relative `gameLoop` import; `ensure_worktree_vitest_config()` writes `vitest.config.mts` alias as belt-and-suspenders.

**After fix:** Step-1 gate **PASS** — food 76% / market 100% / labor 65% targeted_tests on bank reference stubs (`scripts/oracle_reference_sanity.py`).

---

## 2026-06-02 — Mechanic-only apply + goal Vitest scoring alignment

**Goal:** Task bank owns Vitest; model edits `mechanic.ts` only; `targeted_tests` grades goal-weighted Vitest on applied mechanic.

**Changed:**

- `scripts/economist_rl_task_execution.py` — `apply_allowed_paths_for_execution` (no `tests/**`); sandbox starters always from `sandbox_starter_bodies()`; `apply_allowed_paths_for_task` alias; `enrich_task_execution` sets `apply_allowed_paths`, syncs `targeted_tests` via `sync_targeted_tests_from_goals`.
- `scripts/economist_rl_coding_contract.py` — sandbox prompt uses apply paths + formatted `SANDBOX_OUTPUT_FORMAT_RULES`; pre-seeded test path shown read-only.
- `scripts/economist_rl_vitest_scoring.py` — truncated `it()` title fallback for goal merge.
- `tests/test_economist_rl_mechanic_only_apply.py`, `test_economist_rl_vitest_goals.py` (truncated title case).
- `docs/ECONOMIST_RL_ADAPTER.md` — mechanic-only + rescore note.

**Verification:** `PYTHONPATH=scripts python3 -m unittest discover -s tests -p 'test_economist_rl_mechanic*.py' -p 'test_economist_rl_vitest*.py'` — pass. Rescore cycle 016: mean_reward **0.43**, `compiled` **20/20**, `vitest_goal_weighted` **20/20**, `targeted_tests` component **>0 on 5/20** (model quality, not infra).

---

## 2026-06-05 — Cycle 016 local rescore after Vitest install

**Command:** `.venv/bin/python scripts/rescore_economist_rl_rollouts.py --rollout-file benchmarks/results/economistRL/rollouts/rollout_batch_016.jsonl --cycle-id 16 --execution-source-repo /Users/natreed/fallen-empire`

**Before (no vitest binary):** mean_reward **0.10**, compiled **0/20**, no `vitest_report.json`.

**After rescore:** mean_reward **~0.36**, `compiled` **20/20**, `vitest_ran` **20/20**, `vitest_goal_weighted` **18/20** (~9s). Vitest behavioral pass rate still **0/20** on goal asserts (model patches wrong shape / overwrote tests with Jest on market-02); rewards rose mainly because compile gate no longer hard-caps at 0.1.

**Added:** `scripts/rescore_economist_rl_rollouts.py`; fixed `economist_rl_vitest_scoring.py` goal title lookup when Vitest JSON omits `goal_` prefix.

---

## 2026-06-01 — economistRL Lambda launcher (Transformers/CUDA)

**Goal:** Wire `run_economist_rl_lambda_cycle.py` through Lambda Cloud using the Transformers/CUDA PPO path, not only local MLX.

**Changed files:**

- `scripts/launch_economist_rl_lambda_cycle.py` — new launcher: bootstrap/sync, `economistRL` adapter rsync, remote `tmux` with `LOCAL_BACKEND=transformers`, `PPO_TRAIN_BACKEND=transformers`, `--lambda-mode` cycle runner.
- `scripts/lambda/run_economist_rl_lambda_cycle.py` — set `PPO_TRAIN_BACKEND=transformers` when `--lambda-mode`.
- `scripts/launch_lambda_parallel_ablation.py` — artifact collector paths for economistRL results/adapters.
- `docs/ECONOMIST_RL_ADAPTER.md` — Lambda launch examples.
- `tests/test_launch_economist_rl_lambda_cycle.py` — argv/command wiring tests.

**Verification:** `python3 -m py_compile scripts/launch_economist_rl_lambda_cycle.py`; `unittest` launcher tests pass.

---

## 2026-06-05 — PPO subprocess isolation after rollouts (VRAM fix)

**Goal:** Avoid Lambda/Mac CUDA OOM when PPO starts after 50 rollout generations in the same process.

**Changed files:**

- `scripts/lambda/run_economist_rl_ppo_train.py` — child entry point; reads `scored_batch_*.jsonl` + JSON request, runs `train_ppo_batch`, writes `ppo_train_*.json`.
- `scripts/lambda/run_economist_rl_lambda_cycle.py` — default live PPO spawns child via `run_ppo_train_subprocess()`; parent calls `_release_accelerator_memory()` before spawn; `--ppo-in-process` escape hatch for debug.
- `tests/test_economist_rl_lambda_ppo_pipeline.py` — subprocess request wiring + child dry-run tests.

**Verification:** `python3 -m unittest discover -s tests -p 'test_economist_rl_lambda_ppo_pipeline.py'` — new subprocess tests pass.

---

## 2026-06-05 — Lambda smoke: 4 cycles × 10 rollouts

**Command:** `python scripts/launch_economist_rl_lambda_cycle.py --launch-instances --region us-west-1 --watchdog-idle-minutes 360 -- --cycles 4 --rollouts-per-cycle 10 --skip-eval --ppo-epochs 1 --ppo-max-samples 10 --ppo-min-samples 4`

**Worker:** instance `5ffe20c52cd846aa996cd60390f90daf` @ `170.9.11.242` (gpu_1x_a10, us-west-1); tmux `fe-economist-rl`; log `~/cloud-eval-logs/fe-economist-rl-cycle.log`; local launch log `logs/launch_economist_rl_smoke_4x10.log`.

**Intent:** Subprocess-isolated PPO smoke with adapter chaining across 4 cycles; eval skipped for speed.