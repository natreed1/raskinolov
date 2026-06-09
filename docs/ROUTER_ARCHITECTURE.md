# Router V3 Architecture (Council + Adapter-First)

Router V3 keeps adapter-first routing and adds an optional parallel council:
- 3 generalist attention profiles (`wide_compressed`, `precise_short`, `sliding_window`)
- top-3 eligible specialists from the active roster
- blind equal-prior adjudication + escalation candidate metadata

## Runtime flow

1. Build prompt text from chat messages.
2. Select adapter using `ROUTER_ADAPTER_SELECTION_MODE`:
   - default `similarity`: cosine similarity across adapter prototypes.
   - optional hierarchy stage (`ROUTER_HIERARCHICAL_ROUTING_ENABLED=1`): taxonomy/rule coarse bucket -> candidate adapters -> manifold rerank.
   - fallback modes: `hybrid` (classifier + lexical fallback) or `lexical`.
3. Derive route from adapter policy + high-risk overrides:
   - high-risk keywords => `frontier`
   - long prompt threshold => `frontier`
   - adapter route map => local/hybrid/frontier default
4. Build council plan (if enabled):
   - load active roster (`data/routing/council_roster_v1.json`)
   - include 3 generalist profiles + top-3 specialists by ranked adapter candidates
   - optionally expand each selected specialist into roster-defined personalities that share the same base adapter
   - include bounded debate round cap (`debate_max_rounds`)
   - estimate disagreement and escalation candidate (`either_trigger` default)
5. Emit metadata:
   - `adapter_id`, `confidence`, `ambiguity`
   - `secondary_adapter_id`, `secondary_confidence`
   - `coarse_bucket`, `candidate_adapters`, `hierarchy_stage`
   - `risk_class`, `complexity`
   - `council_plan`, `council_disagreement`, `council_escalation_candidate`
   - `policy_version`
6. Optional specialist-lane orchestration (`router_chat_gradio.py`):
   - split one prompt into adapter subtasks (primary + optional secondary),
   - or run full council lane (profiles + specialists + adjudication),
   - run each subtask with its resolved adapter weights,
   - merge drafts into one final output for the same prompt.

## Key files

- `scripts/model_router.py` (primary policy + inference wiring)
- `scripts/router/classifier.py` (OSS linear classifier train/load/predict)
- `scripts/router/council.py` (council planning, profile shaping, adjudication)
- `scripts/router/roster.py` (expert roster states + combined gate transitions)
- `scripts/init_council_roster.py` (bootstrap/refresh council roster with trait defaults)
- `scripts/router/policy.py` (adapter-to-route policy helpers)
- `scripts/router/multi_agent.py` (subtask split + deterministic merge helpers)
- `scripts/run_routing_benchmark.py` (route/adapter/both/council scoring + comparisons)
- `scripts/build_council_training_dataset.py` (council orchestration dataset build from logs)
- `scripts/router_promotion_gate.py` (combined offline+online gate, optional roster updates)
- `scripts/build_routing_training_dataset.py` (dataset build and split)
- `scripts/train_routing_classifier.py` (classifier training artifact writer)

## Safety behavior

- Routing defaults to deterministic fallback behavior if classifier artifacts are missing.
- Hierarchical pass does not force multi-adapter execution; it only influences primary adapter selection and optional secondary recommendation metadata.
- Council execution is opt-in at runtime (`ROUTER_COUNCIL_ENABLED=1`) and preserves base route metadata fields.
- Per-expert knobs are roster-driven and carried into council participants via `traits`:
  - `assertiveness`, `verbosity`, `risk_tolerance`, `creativity`, `skepticism`, `decisiveness` (all in [0,1]),
  - these influence inference-time participant prompting and expert interaction, not final adjudication scoring.
- Per-expert personalities are also roster-driven:
  - each entry may provide `personalities: [{ "name": "...", "state": "candidate", "traits": { ... } }]`,
  - optional `trait_deltas` are supported for personalities that should offset the expert base traits,
  - offline/online outcome fields (`offline_score`, `offline_sample_count`, `online_task_outcome`, `online_sample_count`) feed bandit-style selection,
  - personality states are `candidate`, `active`, `cooldown`, `retired`; retired personalities are excluded from council planning,
  - `ROUTER_COUNCIL_SPECIALIST_PERSONALITY_VARIANTS` caps how many personalities per selected specialist enter a council run.
- Promotion/elimination is two-level:
  - expert state transitions remain `candidate -> active -> probation -> demoted` via combined offline/online gates,
  - personality state transitions run inside each expert from benchmark/conversation credit, so a weak voice can cool down or retire without demoting the whole adapter.
- Personality optimization is policy training, not domain-knowledge training:
  - `ROUTER_COUNCIL_PERSONALITY_SELECTION_POLICY=bandit` ranks voices by observed correctness plus exploration,
  - `ROUTER_COUNCIL_PERSONALITY_EXPLORATION_RATE` controls how much under-sampled personalities are tried,
  - promotion gate reports expose per-personality attempts, wins, top-2 proxy rate, correctness, and escalation-help counts.
- Adjudication is blind and equal-prior:
  - router-selected specialists do not receive an outcome-score bonus,
  - candidate drafts are assigned blind ids before scoring, then mapped back to participant ids after ranking,
  - deterministic rubric components score answer presence, prompt keyword coverage, concrete references, risk awareness, verification plan, and mock/refusal penalties,
  - `task_outcome_score` remains neutral metadata (`0.5`) in council traces and is not capability evidence,
  - assertiveness, verbosity, creativity, skepticism, risk tolerance, and decisiveness shape generation only, not winner selection.
- Iterative "back-and-forth" council debate is bounded by round caps to contain compute:
  - planner cap: `ROUTER_COUNCIL_DEBATE_MAX_ROUNDS`,
  - runtime hard cap: `ROUTER_CHAT_COUNCIL_DEBATE_MAX_ROUNDS`.
- Specialist variant fanout is configurable with `ROUTER_COUNCIL_SPECIALIST_PERSONALITY_VARIANTS` (default 3); roster personalities produce independent drafts through the same underlying specialist adapter weights when loaded, and traces record `adapter_requested`, `resolved_mlx_adapter`, and `adapter_loaded`.
- Low-confidence classifier outputs never hard-force specialist routing; lexical fallback decides.
- Route overrides for high-risk keywords and long prompts remain active even with confident adapter predictions.

