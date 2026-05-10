# Arena roadmap: metrics, tooling, improvement cycles

Companion to `**docs/ARENA_PROGRESSION.md**` (comparable batches) and `**docs/PROJECT_STATE.md**` (environment facts). Use this doc for interpreted goals, dashboards, gates, and the next frontier of apply-focused work.

---

## 1. Single-page dashboard (runs × tasks)

**Generate** after local `arena-acceptance` passes (artifacts live under `**benchmarks/results/runs/`**, gitignored):

```bash
python scripts/ml_workflow.py arena-dashboard --last 35
# or:
python scripts/build_arena_dashboard.py --last 35 --out benchmarks/results/arena_dashboard.html
```

**Output:** `benchmarks/results/arena_dashboard.html` — newest runs first; columns include adapter shorthand, `**--progressive-context`**, pass count, **ACI**, **combat KPI** / **ai KPI** (score + coarse failure tag for `combat-risk-preview` + `ai-planning-explanation`), wall time, and **per-task** score/pass/tag.

Interpret tier rollups skeptically — see `**arena_capability_index.py`** and the analysis of single-task-vs-tier averages in `**docs/ARENA_PROGRESSION.md`** notes.

---

## 2. Apply-focused improvement cycle (North Star KPIs)

**Primary observation (Apr 2026 `best-val300` progressive run):** several tasks stalled on `**apply_final_ok`** (`no_applyable_changes` / `apply_check_failed`) while TypeScript occasionally passed anyway; `**loading-screen-polish`** instead failed **TSC/export** after apply; `**save-load-api-guard`** achieved a full-stack pass alone.

**KPI tasks for the next tightening loop** — track regressions/improvements explicitly:


| KPI                           | Purpose                                                                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `**combat-risk-preview`**     | Medium-high complexity; penalizes speculative architecture when patch application is the bottleneck.                               |
| `**ai-planning-explanation`** | High labeled complexity; same apply symptoms as HUD/economy in logged runs — good pair for qualitative “did edits land?” progress. |


**Concrete levers (pick in order of cost):**

- **Orchestration / parsing**: stricter constrained patch extraction; retry prompting when `**apply_final_ok`** is false.
- **SFT corpus**: pairwise + `**data/arena_task_baselines/`** → `**build_arena_baseline_dataset.py`** (Messages JSONL) mixed with `**build_lora_dataset.py`** outputs; evaluate deltas on **same six-task** suite **before** promoting weights.
- **Progressive retrieval**: paired **auto/off** batches on identical adapter weights (`**scripts/fe_lineage.py`** frozen policies); log to `**docs/ARENA_PROGRESSION.md`**.

Avoid treating **headline ACI** alone as “good”; one strong task (**e.g.** save/load) can mask weak tails.

---

## 3. Promotion gate: worst-of-six + optional two-worst mean

**Script:** `scripts/arena_promotion_gate.py` — reads `**arena_capability.json`** from any completed run folder.

Examples:

```bash
python scripts/arena_promotion_gate.py \
  benchmarks/results/runs/<run_id>/arena_capability.json \
  --min-worst-score 40

python scripts/arena_promotion_gate.py \
  benchmarks/results/runs/<run_id>/arena_capability.json \
  --min-worst-score 42 --min-mean-two-worst 46
```

- `**--min-worst-score**`: rejects promotions when **any** single task collapses (**default 40**; tune after seeing your distribution — the Apr 29 example fails at 40 because `**loading-screen-polish` ≈ 32**).
- `**--min-mean-two-worst`**: optional second bar on `**mean(lowest_two_scores)`**.

Use in CI hooks only after aligning thresholds with a few historical `**arena_capability.json**` snapshots.

---

## 4. Frozen progressive policy (canonical pairings)

**Default for scripted comparisons:** `**DEFAULT_ARENA_PROGRESSIVE_CONTEXT`** = `**auto`** (see `**scripts/fe_lineage.py`**).

**Adapter pairings tracked for regressions:**


| Checkpoint path (relative to repo root)                | Documented progressive default                                               |
| ------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428` | `auto` (use `off` only for deltas on identical weights).                     |
| `checkpoints/fe-lora-qwen25-coder-7b-best-val300`      | `auto` (Apr 2026: materially better than `off` on logged best-val six-pack). |


When you change `**DEFAULT_ARENA_ADAPTER_RELPATH`**, repeat a paired batch and append `**docs/ARENA_PROGRESSION.md`**.

---

## 5. Apply-contract + focused SFT loop

`docs/GAME_ARENA_APPLY_CONTRACT.md` is now injected into arena model packets and baseline/pairwise SFT user prompts. Use it as the canonical format contract when updating parser behavior or training prompts.

### Recommended loop for the current bug (`no_applyable_changes`)

```bash
# 1) Build apply-failure-focused pairwise SFT rows
python scripts/build_game_task_pairwise_dataset.py \
  --input benchmarks/results/game_task_pairwise_training_data.jsonl \
  --out-dir data/lora/game_task_pairwise_apply_focus \
  --focus-apply-failures \
  --repeat 6

# 2) Train + gate in cycles (rebuild each cycle from latest pairwise file)
python scripts/ml_workflow.py arena-gate-train \
  --adapter-path checkpoints/fe-lora-arena-apply-focus \
  --data-dir data/lora/game_task_pairwise_apply_focus \
  --rebuild-dataset \
  --focus-apply-failures \
  --pairwise-repeat 6 \
  --max-cycles 12 \
  --iters-per-cycle 80 \
  --task-id combat-risk-preview \
  --task-id ai-planning-explanation \
  -- --batch-size 1
```

Promotion remains blocked unless worst-of-six gates pass (`scripts/arena_promotion_gate.py`) even if headline ACI improves.

---

## 6. Capability jump write-up (Apr 30)

This section summarizes what changed, what improved, and how to repeat the process to push the frontier.

### Observed improvement

- Before this change set, comparable six-task runs were mostly **1/6** accepted (or worse) in the tracked batches.
- After introducing apply-contract context + targeted SFT, run `20260430-183012_121be1` reached **3/6** accepted with headline ACI **68.05**.
- Passing tasks in that run: `loading-screen-polish`, `save-load-api-guard`, `ai-planning-explanation`.

### What likely caused the jump

Current working attribution is the **combination** of:

1. **Context contract tightening**
  - Added `docs/GAME_ARENA_APPLY_CONTRACT.md`.
  - Injected contract text into arena packet context (`scripts/game_task_arena.py`) so every generation round sees strict applyability rules.
2. **Targeted SFT on apply-shaped artifacts**
  - Built and trained on `data/lora/arena_task_baselines_apply_contract`.
  - This concentrated training on “return applyable output only” behavior instead of generic repo continuation.

Interpretation note: this is a strong directional signal but still a **provisional causal claim** until repeated on additional comparable batches.

### Why this likely worked mechanically

- Historical failures were dominated by `no_applyable_changes` / `apply_check_failed`.
- The contract reduces format ambiguity at inference time (fewer invalid output shapes).
- The focused SFT increases probability mass on valid diff/fenced-file responses.
- Combined, these changes improve **apply_final_ok** rates enough to move multiple tasks from fail to pass.

### Repeatable frontier playbook

1. **Freeze run comparability**
  - Keep the six-task suite fixed.
  - Keep `--progressive-context auto` unless explicitly testing A/B.
  - Keep one adapter lineage per campaign.
2. **Encode contract first**
  - Update `docs/GAME_ARENA_APPLY_CONTRACT.md` when parser/output requirements evolve.
  - Ensure packet + SFT prompts include that contract.
3. **Train small, evaluate often**
  - Run short focused SFT passes (`80–160` iters scale) on apply-focused corpora.
  - Immediately run six-task `arena-acceptance` after each pass.
4. **Promote only on tails**
  - Continue using `scripts/arena_promotion_gate.py` with worst-task thresholds.
  - Headline ACI increases do not promote unless worst-tail rises too.
5. **Target weak tasks explicitly**
  - Current weakest tail after the 3/6 jump: `hud-status-summary`, `combat-risk-preview`, `economy-tooltip`.
  - Bias next data refresh toward these failure modes (failed outputs + corrected winners).

### Next frontier experiments (ordered)

1. **Apply-failure-only pairwise refresh**
  - Populate `benchmarks/results/game_task_pairwise_training_data.jsonl` with fresh winner/loser rows from latest runs.
  - Rebuild with `--focus-apply-failures` and run `arena-gate-train` cycles.
2. **Task-local repair corpus**
  - Add targeted correction examples for the three weak tasks where round-0 fails and round-1 recovers (or fails differently).
3. **Parser-contract co-evolution**
  - Tighten parser acceptance heuristics only when new winning formats are safe.
  - Mirror each parser change into the contract doc and SFT prompts.
4. **Reproducibility check**
  - Repeat the exact train/eval recipe across at least two more batches before treating attribution as settled.

---

## 7. Diversified capability index + broader suite mode (May 1)

To reduce score inflation from narrow subsets, `scripts/arena_capability_index.py` now also reports:

- `diversified_arena_capability_index` — blended headline for broad capability.
- `task_type_balanced_index` — equal-weight mean across active `task_type` groups.
- `tail_robustness_index` — mean of the bottom quartile task scores (minimum two tasks).
- `task_coverage_ratio` — evaluated tasks divided by catalog size (`--tasks` JSON).
- `task_type_breakdown` — per-type acceptance and sub-scores.

The diversified score is coverage-adjusted and includes task-type caps when only a small number of capability families are present.

For broader acceptance runs, `run_arena_acceptance_tests.py` and `ml_workflow.py arena-acceptance` now support:

```bash
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/<adapter> \
  --suite all
```

`--suite all` runs every task defined in the selected tasks JSON (unless explicit `--task-id` filters are provided).

