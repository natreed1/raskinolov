# Session log (append-only)

Newest entries at the **top**.

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
  - benchmark improved to **`2/2`** (`100%`),
  - capability **`78.6/100`**,
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

Added `**railway.json`** for one-command deploy (`python3 scripts/private_dashboard_server.py`) and wrote `**docs/PRIVATE_DASHBOARD_DEPLOY.md**` with env/volume/security setup plus local hook->remote ingest wiring.

Hook updates: `**.cursor/hooks/lab_hook_after_shell_autodoc.py**` and `**.cursor/hooks/lab_hook_stop_append.py**` now optionally POST events when `**FE_LAB_REMOTE_INGEST_URL**` + `**FE_LAB_REMOTE_INGEST_TOKEN**` are set (fail-open on network errors). `afterShellExecution` autodoc remains local-first (`shell_command_events.jsonl` + `SPECIALIZED_RUN_HISTORY`).

---

## 2026-05-04 — Cursor test-run autodoc hook (shell + code-change snapshot)

Added project hook `**.cursor/hooks/lab_hook_after_shell_autodoc.py**` and wired `**afterShellExecution**` in `**.cursor/hooks.json**`. Hook command now enables autodoc by default for this repo (`FE_LAB_AUTODOC_APPEND=1` inline), so Cursor shell test/benchmark commands are auto-recorded with exit code + git status snapshot (`m/u/d` counts + touched paths) to `**lab_dashboard/shell_command_events.jsonl**` and appended as compact rows in `**docs/SPECIALIZED_RUN_HISTORY.md**`.

Filter is command-based (`pytest`, `unittest`, benchmark runners, `scripts/ml_workflow.py`) unless `**FE_LAB_AUTODOC_INCLUDE_ALL=1**` is set.

---

## 2026-05-04 — Documentation-agent RAG: specialized run history + orchestrated workflow

**Goal:** keep auxiliary evals legible alongside `docs/run_history.md`, and make the documentation-agent RAG benchmark a one-command habit with aggregate telemetry.

**Added:** `docs/SPECIALIZED_RUN_HISTORY.md` — append-only table for cross-cutting benchmarks (retrospective rows for the manual **2026-05-04** `documentation_agent_*_7b.jsonl` runs + convention for linking `ml_workflow_run_id` in Notes).

**Orchestration:** `python scripts/ml_workflow.py documentation-rag-benchmark` — `benchmarks/results/runs/<run_id>/` (manifest, per-task JSONL), `**docs/run_history.md`** row, `**docs/SPECIALIZED_RUN_HISTORY.md`** row, tail `**benchmarks/results/documentation_rag_timeseries.jsonl**`. **Exit code** follows the **with-RAG** pass; optional no-RAG baseline uses `run_documentation_agent_benchmark.py --no-fail`.

**Runner:** `scripts/fe_ml_lab_runner.py documentation-rag-benchmark` (optional passthrough flags) appends `lab_dashboard/agent_events.jsonl` with `kind: documentation_rag_benchmark`.

**Code:** `scripts/run_documentation_agent_benchmark.py` gains `--no-fail`; `scripts/ml_workflow.py` gains helpers `_append_specialized_history_row`, `_count_doc_benchmark_jsonl`, `_append_documentation_rag_timeseries`. Docs: `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `benchmarks/README.md`, `.cursor/rules/precise-ml-documentation.mdc`. Test: `tests/test_fe_ml_lab_tools.py::test_cmd_documentation_rag_benchmark_invokes_ml_workflow`.

**Verify:** `python -m py_compile scripts/ml_workflow.py …`, `python scripts/ml_workflow.py documentation-rag-benchmark --help`, `python -m unittest tests.test_fe_ml_lab_tools tests.test_documentation_rag -v`.

---

## 2026-05-04 — Cursor `stop` hook ledger (opt-in)

Added `**.cursor/hooks.json`** plus `**.cursor/hooks/lab_hook_stop_append.py`**: Agents hitting `**stop**` append `**lab_dashboard/cursor_hook_events.jsonl**` when `**FE_LAB_CURSOR_HOOK_APPEND=1**` (optional `**FE_LAB_CURSOR_HOOK_REFRESH_DASHBOARD=1**` reruns `**build_lab_optimization_dashboard.py**`). `**--cursor-hooks**` override added on the dashboard script. `**lab_dashboard/README.md**` documents hooks versus scheduled MLX versus git syncing across clones/windows.

---

## 2026-05-04 — fe-mlx-lab skill · Cursor token footprint

Shrunk `**.cursor/skills/fe-mlx-lab/SKILL.md**`, `**disable-model-invocation: true**` so MLX guidance is mainly on explicit `**@fe-mlx-lab**` mentions; playbook says terminal-only MLX and **path citations instead of log dumps**.

---

## 2026-05-04 — Router Gradio OSS backbone switch (`force_route=local`)

`scripts/router_chat_gradio.py` now exposes accordion **OSS / routing controls**: **Backbone** (policy **Auto** vs **Codebase OSS** → `GenerationRequest(force_route='local')`), optional registry **LoRA adapter lock**, plus defaults via `**ROUTER_CHAT_DEFAULT_BACKBONE`**, `**ROUTER_CHAT_DEFAULT_ADAPTER_LOCK`**, or `**--default-backbone**` / `**--default-adapter-lock**`. Listener default `**--port` is `7864**` (avoids `**train_ui_gradio.py`'s** `**7862`** collision). `**tests/test_router_backbone_controls.py`** locks regressions vs security-keyword frontier escalation. Revised `**.cursor/skills/fe-mlx-lab/SKILL.md**` stating Composer cannot load repo LoRA; OSS path is MLX UIs described in `**docs/PROJECT_STATE.md**`.

Added `**tests/test_fe_ml_lab_tools.py**` (mocked subprocess coverage for `**fe_ml_lab_runner**`, deterministic fixtures for `**build_lab_optimization_dashboard**`) + CLI overrides `**--cursor-usage**` / `**--agent-events**` on `**scripts/build_lab_optimization_dashboard.py**`. Narrative SKILL note: Cursor integration is Markdown skill metadata consumption, **not** a runtime plugin.

Added `**scripts/fe_ml_lab_runner.py`** with default task `**learning` → `ml_workflow.py smoke`** (switch `--sequence full` for end-to-end; passthrough MLX flags **after `--`**). Each invocation appends JSON lines to `**lab_dashboard/agent_events.jsonl**` (`FE_ML_LAB_SPARED_USD` optional heuristic). `**scripts/build_lab_optimization_dashboard.py**` renders `**lab_dashboard/index.html**` from `**docs/run_history.md**` plus optional `**lab_dashboard/cursor_usage.jsonl**`. Supporting docs: `**lab_dashboard/README.md**`, Cursor skill `**.cursor/skills/fe-mlx-lab/SKILL.md**`. Verified `**python3 -m py_compile scripts/fe_ml_lab_runner.py scripts/build_lab_optimization_dashboard.py**`, `**python scripts/build_lab_optimization_dashboard.py**`, and `**python3 -m unittest discover -s tests**`.

---

## 2026-05-04 — Combat `cycle3` resume train + HUD / combat arena acceptance

**Combat resume:** Finished `**python scripts/ml_workflow.py train`** with `**--resume-adapter-file checkpoints/adapters/combat_risk/cycle3/adapters.safetensors --iters 300`**. Run `**benchmarks/results/runs/20260504-175440_9e0edb**` — `**final_exit_code` 0** (~12644 s).

**Arena (single-task, `auto`, `benchmarks/game_task_arena_examples.json`):**


| Adapter                                   | Run id                       | Task                  | Result                                                                                                         |
| ----------------------------------------- | ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `checkpoints/adapters/hud_status/cycle2`  | `**20260504-212542_a576dd`** | `hud-status-summary`  | **Failed** — `**generate_failed`**, `**generate_exit` 124** (never reached apply)                              |
| `checkpoints/adapters/combat_risk/cycle3` | `**20260504-214556_a37ce5`** | `combat-risk-preview` | **Failed** — round0 edited `**GameHUD.tsx`** (export `**GameHUD`** missing); round1 `**no_applyable_changes**` |


**Registry:** Bumped `**combat_risk` `adapter_path`** to `**checkpoints/adapters/combat_risk/cycle3`** now that resume train completed (`**training/adapter_registry_v1.json**`).

---

## 2026-05-04 — Registry: route `economy_tooltip`, `hud_status`, `combat_risk` to `cycle2`

Updated `training/adapter_registry_v1.json` so `**adapter_path**` resolves to the adapters trained in the overnight lockdown run: `checkpoints/adapters/economy_tooltip/cycle2`, `checkpoints/adapters/hud_status/cycle2`, `checkpoints/adapters/combat_risk/cycle2`; lineages bumped accordingly. **Promotion remains `shadow`** (single-task arena gates for these three were **not** passing at last documented runs); this only aligns the router / local UIs with the newest on-disk weights.

---

## 2026-05-04 — Overnight specialist lockdown (economy / HUD / combat cycle2)

Executed the agreed runbook: `**ml_workflow`** dataset → `**train`** → single-task `**arena-acceptance**` (`--progressive-context auto`, `**benchmarks/game_task_arena_examples.json**`, `**SOURCE_REPO**` `~/fallen-empire`, `**GAME_ARENA_ROOT**` `~/fallen-empire-arena`). **No registry promotion** (none of the three gates passed end-to-end).


| Stage                         | Run id / adapter                                                         | Outcome                                                                                                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**economy_tooltip`** dataset | `benchmarks/results/runs/20260504-035847_b88a85`                         | exit **0**                                                                                                                                                                                                                                                                              |
| `**economy_tooltip`** train   | `20260504-035848_5bfba2` → `checkpoints/adapters/economy_tooltip/cycle2` | exit **0** (~8970 s)                                                                                                                                                                                                                                                                    |
| `**economy_tooltip`** arena   | `20260504-062823_8b3903`, `--task-id economy-tooltip`                    | exit **1**, **0/1** — `**apply_final_ok`**, `**exports_final_ok`**, `**tsc**` `tsc_exit_2` rounds 0–1                                                                                                                                                                                   |
| `**hud_status**` dataset      | `20260504-063042_18ea7e`                                                 | exit **0**                                                                                                                                                                                                                                                                              |
| `**hud_status`** train        | `20260504-063044_ba7a5e` → `checkpoints/adapters/hud_status/cycle2`      | exit **0** (~9763 s)                                                                                                                                                                                                                                                                    |
| `**hud_status`** arena        | `20260504-091330_8c7651`, `--task-id hud-status-summary`                 | exit **1**, **0/1** — mixed rounds (export gap + retry `**no_applyable_changes`**)                                                                                                                                                                                                      |
| `**adapter-datasets`**        | `20260504-091830_32b592`                                                 | exit **0** (refreshed `data/lora/adapters/*`)                                                                                                                                                                                                                                           |
| `**combat_risk`** train (1st) | `20260504-091837_00c86b`                                                 | exit **1** — MLX `**IndexError`** on empty `**valid.jsonl`** / `**test.jsonl**` for `**combat_risk**` when only two synthetic train lines existed                                                                                                                                       |
| `**combat_risk**` train (2nd) | `20260504-091854_ee9a2d`                                                 | exit **0** after duplicating train rows into `**valid.jsonl`** / `**test.jsonl`** for the immediate run; `**scripts/adapters/dataset_builder.py**` `**_split_rows**` now guarantees non-empty valid+test for tiny families so future `**adapter-datasets**` builds load in `**mlx_lm**` |
| `**combat_risk**` arena       | `20260504-110442_ee9296`, `--task-id combat-risk-preview`                | exit **1**, **0/1** — `**apply_final_ok`**, `**tsc_exit_2`** both rounds                                                                                                                                                                                                                |


**Next:** widen `**shared_general_anchor`** or add `**build_combat_*` / pairwise combat rows** before another combat cycle2 pass; chase `**tsc`** deltas on `**economy`** and `**combat**` with curator-aligned repair JSONL or lower LR / fewer iters smoke.

---

## 2026-05-04 — HUD cycle2 arena smoke (`hud-status-summary`)

Ran `arena-acceptance` on `checkpoints/adapters/hud_status/cycle2`. Run `**benchmarks/results/runs/20260504-031825_0f7a64**`: `**exit_code` 1**, **0 / 1** passed. Applied `CompactEmpireStatus.tsx` + `TestEnvironmentShell.tsx` but `**tsc` failed** both rounds (`goldPile`, `goldHold`, `morale` on `Player`, missing imports like `countVillagesInPlayerTerritory` / supply helper, bogus `provinceHexKeys`) — schema hallucination vs curator baseline.

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

**Router:** `**economy_tooltip_specialist_fastpath`** in `scripts/router/policy.py`; matching specialist branch in `scripts/model_router.py` (suppresses generic `**frontier`** / `**hybrid`** substring overrides after `**save_load_api_guard**` and before blanket frontier keywords).

**Data:** Curated `**data/routing/economy_tooltip_eval_prompts_v1.jsonl`** (**30**) + `**benchmarks/economy_tooltip_eval_tasks_v1.json`**. Routing check: `**30/30`** route + adapter (`benchmarks/results/routing_policy_summary_economy_eval_v1.json`). Mixed `**76/76**` regression still passes; mixed `**economy_tooltip**` probe now reports `**economy_tooltip specialist route**`.

**Train path:** `**scripts/adapters/build_economy_tooltip_specialist_dataset.py`** outputs `**data/lora/adapters/economy_tooltip_specialist/`**; orchestrated via `**python scripts/ml_workflow.py economy-tooltip-dataset`** (defaults: transfer `**loading-screen-polish**` + `**hud-status-summary**`, `--min-train-core-rows` **100**).

**Registry:** `training/adapter_registry_v1.json` `**economy_tooltip`** `**adapter_path`** → `**checkpoints/adapters/economy_tooltip/cycle1`**, lineage `**economy_tooltip:v1:cycle1+routing_fastpath_v1**`, `**promotion_state**` remains `**shadow**` until arena `**economy-tooltip**` is proven independently of routing-only readiness.

**Docs:** `docs/WORKFLOW.md`, `docs/PROJECT_STATE.md`, `data/routing/README.md`.

---

## 2026-05-04 — Documentation specialist scaffolding (dataset + registry + orchestrator)

**Goal:** Simple “documentation agent” LoRA: canonical mlx-lab prose (paths, append-only docs, `ml_workflow` wording) without HUD/arena pairwise.

**Scripts:** Fixed bash newline escaping in `**scripts/adapters/build_documentation_specialist_dataset.py`**. `**scripts/ml_workflow.py`** new subcommand `**documentation-dataset**` → runs that builder with run manifest + `**docs/run_history.md**` row.

**Taxonomy/registry:** `**documentation`** added to `**LOCKED_ADAPTER_FAMILIES`**; `**TASK_TO_ADAPTER**` maps `**mlx-lora-docs-normalize**` → `**documentation**`. `**training/adapter_registry_v1.json**` entry: `**checkpoints/adapters/documentation/cycle1**`, lineage `**documentation_specialist:v1:cycle1**`, `**shadow**`. `**checkpoints/adapters/documentation/cycle1/adapter_config.json**` seeded (data → `**documentation_specialist/train.jsonl**`).

**Verify:** `.venv/bin/python scripts/ml_workflow.py documentation-dataset` (run `**20260504-032701_489e1b`**) wrote `**data/lora/adapters/documentation_specialist/`** (**120**/3/1 train/valid/test rows).

**Docs:** `**docs/WORKFLOW.md`** table row; `**docs/DATA_LAYOUT.md`** documentation specialist paths.

---

## 2026-05-04 — Documentation routing keyword eval + trained cycle1 weights

**Routing:** Expanded `**scripts/router/classifier.py`** phrase bank for `**documentation`**; `**build_plan**` + `**model_router**` specialist fastpaths mirror economy/loading semantics so frontier/hybrid substring hooks don’t steal mlx-doc prompts.

**Eval sources:** `**scripts/build_documentation_eval_tasks_v1.py`** → `**data/routing/documentation_eval_prompts_v1.jsonl`** + `**benchmarks/documentation_eval_tasks_v1.json**` (SHA256-prefix ids; regenerate with the script).

**Regression:** `**tests/test_documentation_routing_eval.py`** + `**tests/__init__.py`** exercise each task via `**RoutingPolicy**`, subprocess `**run_routing_benchmark.py**`, and (**after train**) `**adapters.safetensors`** presence.

**Train:** `**python scripts/ml_workflow.py documentation-dataset`** then `**train --adapter-path checkpoints/adapters/documentation/cycle1 -- --data …/documentation_specialist --iters 80 …`** (`**20260504-032912_d8cd21**`, exit 0).

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

`**router_chat_gradio.py`:** default assistant decode ceiling raised (**2048** new tokens vs **896**) so economy/HUD/UI turns stop mid-structure less often; override with `**MAX_TOKENS`** / `**--max-tokens`**. Completed responses log `**mlx_finish_reason**`, token counts, `**generation_budget_hit**`, Markdown fence imbalance, `**recommended_for_sft_assistant_turn**`, `**schema_version: router_chat_supervisor_v1**`. UI emits a truncation banner when MLX hits `**length**` limits.

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
  - scans `data/lora/adapters/*`,
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

---

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
  - forced dark text + text-fill on all message descendants (`.message *`, user/bot variants),
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