# economistRL Adapter Experiment

`economistRL` is a new experimental LoRA adapter lane for testing whether reward-scored RL improves economy mechanics that previous specialists handled poorly. It is intentionally separate from `economy_tooltip` / `resource_ui_projection`, which is mostly an economy UI/copy specialist.

## Scope

Curriculum target: **500 prompts total** with a **70% / 30% split**:

- **350 economy prompts** across economy subsections
- **150 generalist prompts** for common RL/framework skills

Economy subsections:

- `economy_simulation`
- `market_dynamics`
- `population_dynamics`
- `food_economy`
- `labor_economy`
- `progressive_costs`
- `feedback_control`
- `resource_projection`

Generalist subsections:

- `instruction_following`
- `structured_reasoning`
- `code_patch_planning`
- `test_design_and_invariants`
- `debugging_and_root_cause`
- `concise_explanation`
- `safety_and_scope_control`

Initial task bank: `benchmarks/economistRL_tasks_v1.json`

Scoring/tooling: `scripts/economist_rl_tasks.py`

Seed SFT builder: `scripts/adapters/build_economist_rl_dataset.py`

Training config: `training/economistRL_lora_qwen25_coder_7b.yaml`

Registry entry: `training/adapter_registry_v1.json` (`adapter_id: economistRL`, `promotion_state: experimental`)

## Task Format

Each task contains:

- `id`, `title`, `difficulty`, `subskill`, and `rl_focus`
- `curriculum_track`: `economy` or `generalist`
- `prompt`
- `simulation_spec`: a focused 20-tick behavior scenario with scored goals
- `targeted_tests`: executable outcome expectations; tests should verify behavior, not exact names
- `static_code_mechanics`: flexible hooks/signals for state fields, tick integration, bounded updates, cache invalidation, etc.
- `formula_signal`: formula/curve/bounds patterns that indicate real computation
- `instruction_contract`: policy gates for unnecessary/broad changes and test weakening
- `anti_overfit_guards`: reward-gaming caps for hardcoding, deleting tests, bypassing mechanics, unrelated changes, or non-general behavior
- `expect.all_contains`, `expect.any_contains`, `expect.none_contains`
- `expect.min_chars` / `expect.max_chars`
- `expect.mechanics`, where each mechanic has a `name` and keyword set
- optional `reference_answer` for seed SFT rows

Add a task:

```bash
python scripts/economist_rl_tasks.py add-task \
  --id economistRL-new-task-07 \
  --title "New Economy Mechanic" \
  --subskill market_inventory_pressure \
  --difficulty hard \
  --curriculum-track economy \
  --focus-tag economy_simulation \
  --focus-tag market_dynamics \
  --required price \
  --required stock \
  --optional elasticity \
  --mechanic stock_pressure:stock,scarcity,surplus \
  --prompt "Describe the mechanic and its invariant."
```

Validate:

```bash
python scripts/economist_rl_tasks.py validate
```

Validation reports current track counts and remaining tasks needed for the 350 economy / 150 generalist target.

## Scoring

`score` expects JSONL rows with `task_id` and one generated text field: `output`, `assistant`, `response`, `generated_text`, or `text`.

```bash
python scripts/economist_rl_tasks.py score \
  --outputs-jsonl benchmarks/results/economistRL_rollouts.jsonl
```

Reward components:

- `simulation_behavior` (`45%`): deterministic 20-tick task scenario outcomes. This is the main reward and should trace whether the patch changes the relevant game state in the intended direction.
- `targeted_tests` (`20%`): executable tests verify the behavior changed. Tests should not require exact names or one fixed implementation.
- `static_code_mechanics` (`15%`): code/diff evidence for required hooks such as state fields, tick-order integration, bounded update functions, and cache invalidation.
- `formula_signal` (`10%`): code/diff evidence that the patch actually computes with formulas, thresholds, bounds, curves, clamps, or scoped invalidation.
- `instruction_contract` (`5%`): policy gate for applyable, scoped patches. Unnecessary broad changes or test weakening cap the reward.
- `concision` (`3%`): small penalty for overly wordy or noisy responses.
- `anti_overfit` (`2%`): reward-gaming guard. Hardcoding visible scenarios, deleting tests, bypassing mechanics, changing unrelated files, keyword stuffing, or failing to improve system potential caps reward.

The legacy `expect.*` prompt checks remain as guardrails and seed/debug signals, but they are no longer the main reward. Serious RL rollouts should provide executable evidence in JSONL, for example:

```json
{
  "task_id": "economistRL-food-steady-state-01",
  "compiled": true,
  "simulation_results": {
    "ticks": 20,
    "goals": [
      {"name": "birth_rate_tapers_near_food_floor", "score": 1.0, "weight": 0.35},
      {"name": "food_not_silently_negative", "score": 0.8, "weight": 0.30}
    ]
  },
  "targeted_tests": {"score": 0.75},
  "changed_files": ["src/lib/economy.ts"],
  "previous_potential": 0.40,
  "new_potential": 0.68
}
```

The base reward is wrapped by a compile gate when rollout rows include compile status fields such as `compiled`, `compile_passed`, `typecheck_passed`, `compile_status`, `typecheck_status`, `tsc_status`, or `verify_status`.

Compile rule:

- Compile failure always caps the final reward.
- Compile success gets a positive bonus that decays as recent compile rate rises.
- This means the model is rewarded heavily for learning to compile early, but once it compiles reliably, most reward pressure moves to simulation behavior and economy correctness.

Current gate:

```text
if not compiled:
  rolling_compile_rate < 0.50 -> final_reward <= 0.10
  rolling_compile_rate < 0.80 -> final_reward <= 0.05
  rolling_compile_rate >= 0.80 -> final_reward = 0.0

if compiled:
  rolling_compile_rate < 0.50 -> final_reward = 0.35 + 0.65 * base_reward
  rolling_compile_rate < 0.75 -> final_reward = 0.20 + 0.80 * base_reward
  rolling_compile_rate < 0.90 -> final_reward = 0.10 + 0.90 * base_reward
  rolling_compile_rate >= 0.90 -> final_reward = 0.03 + 0.97 * base_reward
```

Use `--rolling-compile-rate` when scoring a rollout batch:

```bash
python scripts/economist_rl_tasks.py score \
  --outputs-jsonl benchmarks/results/economistRL_rollouts.jsonl \
  --rolling-compile-rate 0.83
```

Outputs:

- `benchmarks/results/economistRL_scorecard_v1.json`
- `benchmarks/results/economistRL_scorecard_v1.md`

## Seed Dataset And Training

Build seed SFT data:

```bash
python scripts/ml_workflow.py economist-rl-dataset
```

Direct builder:

```bash
python scripts/adapters/build_economist_rl_dataset.py
```

Train seed LoRA:

```bash
mlx_lm.lora --train -c training/economistRL_lora_qwen25_coder_7b.yaml
```

This is only the bootstrap phase. The RL phase should generate multiple rollouts per task, score them with `scripts/economist_rl_tasks.py score`, and feed high-reward trajectories or pairwise preferences into the next adapter cycle.

## Promotion Discipline

Do not promote `economistRL` into default routing until it beats `resource_ui_projection`, `crossdomain_state_patch`, and the base model on the same economy task bank. Treat early scores as RL research telemetry, not production routing evidence.
