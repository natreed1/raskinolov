# Gate Policy v1

Per-adapter promotion uses an average-gain-first objective with rollback hard guards.

## Primary objective

Promote only when candidate average score gain versus champion is positive and above threshold:

- `min_average_gain = +1.5`
- No gate dimension may violate floor/ceiling constraints.

## Required gate dimensions

- apply-contract pass rate floor (`>= 0.80`)
- compile/export integrity floor (`>= 0.80`)
- latency p95 ceiling (`<= 80s`)
- minimum evaluated coverage (`>= 6`)

## Hard rollback guards (canary defaults)

Rollback immediately if any condition triggers:

- apply hard-failure rate worsens by `>= 8` percentage points versus champion
- latency p95 regresses by `>= 40%`
- severe safety/contract events exceed `2`

## Gate output

`scripts/adapters/gates.py` emits `adapter_gate_result_v1`:

- `decision`: `promote` | `reject` | `rollback`
- `reasons`: explicit rejection reasons
- `rollback_triggers`: explicit trigger list
- `signals`: computed deltas
- `thresholds`: policy snapshot for audit

