# Skills Extraction v1 (Outcome-First)

This spec defines how to extract a richer specialist skill map from benchmark outcomes, not prompt wording alone.

## Why this exists

Prompt-angle similarity (`cos(theta)` over prompt features) is useful for diagnostics, but it is not enough for routing ownership. We need skills that are:

- behavior-grounded (tied to pass/fail outcomes),
- stable across specialist/task variants,
- interpretable enough to gate promotion and routing decisions.

## Scope

Build a `skills_v1` artifact that:

- proposes candidate skill dimensions from benchmark outcomes,
- quantifies each skill's effect on pass probability,
- computes specialist skill profiles,
- exposes confidence and support for each skill,
- supports unknown/OOD routing when skill evidence is weak.

## Data inputs

- Task-level benchmark rows (all relevant suites):
  - `benchmarks/results/runs/*/logs/run_game_benchmark.log`
  - `benchmarks/results/runs/*/manifest.json`
- Task definitions:
  - `benchmarks/*_mass_tasks_v1.json`
  - `benchmarks/task_routing_tasks.json`
  - `benchmarks/task_routing_mixed_tasks_v1.json`
- Optional curated labels:
  - unknown review queue and manually reviewed routing rows.

## Core entities

- **Task outcome row**: `{task_id, specialist, pass, capability, category, prompt}`
- **Base nodes**:
  - specialist nodes,
  - task nodes,
  - concept nodes,
  - capability bucket nodes (`core`, `ui_change`, `constraints`, `transfer`).
- **Skill candidate node**:
  - latent or explicit behavior cluster derived from outcomes.

## Extraction pipeline

1. **Build outcome tensor**
  - Matrix `Y[s, t]` = specialist `s` pass/fail on task `t` (or pass-rate over repeats).
  - Add side channels:
    - `capability_score[s, t]`,
    - constraints/format failure flags when available.
2. **Construct behavior graph**
  - Bipartite core:
    - task <-> specialist edges weighted by success signal.
  - Augment:
    - task <-> concept edges,
    - task <-> capability bucket edges.
  - Weight each edge by support-aware effect (not raw counts only).
3. **Spectral decomposition**
  - Build weighted Laplacian `L = D - A`.
  - Use non-trivial eigenvectors (`v2+`) for structure discovery.
  - Candidate skill axes come from stable partitions / clusters in this space.
4. **Candidate skill synthesis**
  - Create candidate skills from:
    - dense co-success communities,
    - high-curvature conflict zones,
    - repeated failure-mode motifs (especially constraints).
  - Keep only candidates with support >= `min_support_tasks`.
5. **Skill effect estimation**
  - Fit regularized logistic model on pass outcome:
    - features: candidate skills + specialist controls + bucket controls.
  - Retain skills with:
    - effect magnitude above threshold,
    - bootstrap confidence interval not crossing zero,
    - split-stable sign.
6. **Specialist skill profiling**
  - For each specialist, compute:
    - mean success per retained skill,
    - variance and support counts,
    - transfer and constraints reliability.
  - Emit profile confidence and blind spots.
7. **Unknown routing support**
  - For incoming prompt, infer required skill vector.
  - Score specialist fit against skill profiles.
  - Route to unknown review if:
    - low top score,
    - low top-vs-second margin,
    - low support in required skills.

## Curvature + manifold extension (v1.1)

To identify stable routing regions and specialist/task manifolds, extend the graph math beyond plain Laplacian coordinates.

1. **Edge curvature**
  - Compute edge curvature on the weighted behavior graph.
  - Initial pragmatic choice: Forman-style curvature:
    - `F(e=u-v) ~= 4 - deg(u) - deg(v)` (unweighted form)
    - weighted variant scales by adjacent edge weights.
  - Interpretation:
    - high negative curvature = conflict/bridge edge, unstable routing boundary.
    - near-zero or positive curvature = locally coherent region.
2. **Region stability score**
  - For each manifold region `R` (cluster/community), compute:
    - `stability(R) = mean(curvature of intra-region edges) - boundary_penalty`
  - Boundary penalty increases when many high-negative edges cross region borders.
3. **Task manifold + specialist manifold**
  - Task manifold: cluster task nodes in spectral space.
  - Specialist manifold: project specialist profiles into same latent space.
  - Build assignment matrix:
    - `A[s, r] = specialist s fit to region r` (skill coverage weighted by stability).
4. **Routing with region awareness**
  - First pick manifold region(s) for prompt-required skill vector.
  - Then choose specialist with highest `A[s, r]`.
  - If region confidence low or boundary curvature very negative, send to unknown/hybrid.

## Hierarchical routing + output composition

Use a two-level policy:

1. **Level 1: Region / skill family**
  - Identify primary and optional secondary region.
2. **Level 2: Specialist selection inside region**
  - Select one primary specialist.
  - Optionally select one secondary specialist for constrained subtasks.
3. **Composable output plan**
  - Decompose request into sub-operations tied to skills:
    - e.g., `schema_guard` + `ui_wording` + `planning_rationale`.
  - Assign each sub-operation to the best specialist.
  - Merge with deterministic assembly rules:
    - required sections/order,
    - conflict resolution precedence,
    - final constraints validator pass.

This avoids naive adapter stacking while still using multiple specialists per response when needed.

## Initial thresholds (v1 defaults)

- `min_support_tasks_per_skill = 8`
- `min_support_specialists_per_skill = 2`
- `bootstrap_resamples = 200`
- `min_effect_abs = 0.10` (logit-space, after regularization)
- `stability_min_sign_agreement = 0.8`
- `region_stability_min = 0.0` (below this, treat region as unstable for direct routing)
- unknown gate:
  - `top_score < 0.35` OR
  - `(top_score - second_score) < 0.05` OR
  - `required_skill_support < 5`
  - OR selected region stability below `region_stability_min`

Tune by benchmark gate results, not manually.

## Expected new skill families (beyond current nodes)

These are target families to test as candidates, not final labels:

- **Format Discipline**
  - exact-line-count,
  - token-lock,
  - no-extra-text compliance.
- **Schema/API Guarding**
  - versioning compatibility,
  - response-shape correctness,
  - migration safety language.
- **Planning Rationale**
  - intent articulation,
  - defend/expand/scout/reinforce coherence,
  - evidence-backed explanation.
- **UI Information Design**
  - hierarchy clarity,
  - compact signal density,
  - alert prioritization.
- **Cross-Domain Transfer**
  - loading<->HUD transfer,
  - economy<->planning transfer,
  - combat<->save/load transfer safety.

## Deliverables

- `data/routing/skills_v1.json`
  - retained skills, definitions, support, effect stats.
- `data/routing/specialist_skill_profiles_v1.json`
  - per-specialist skill strengths + confidence.
- `data/routing/skill_manifolds_v1.json`
  - region assignments, curvature stats, stability scores, specialist-region fit matrix.
- `data/routing/hierarchical_routing_policy_v1.json`
  - region selection thresholds, specialist selection weights, composition rules.
- `benchmarks/results/skills_extraction_report_v1.md`
  - extraction summary, retained/rejected skills, risk notes.

## Evaluation gates for adoption

Do not promote skill-based routing unless all pass:

- AGI routing suite: no regression vs current promoted baseline.
- Mixed routing suite: overall (`both`) >= baseline + configured margin.
- High-risk route miss rate within gate.
- Unknown queue precision improves (more true ambiguous/OOD captures).
- Region stability calibration: unstable-region prompts should preferentially route to unknown/hybrid and reduce hard misroutes.

## Data requirements note

Yes: this likely needs more data.

- Current node set is too coarse for robust skill extraction.
- Add harder constraints tasks and cross-domain transfer tasks per specialist.
- Add reviewed unknown prompts into curated routing sets each cycle.
- Add compositional tasks requiring multi-skill outputs (schema + planning + UI) to validate hierarchical assembly.

Without this expansion, skill estimates will overfit to current benchmark shape.