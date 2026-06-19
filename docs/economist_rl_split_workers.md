# economistRL Split-Worker PPO Launcher

`scripts/launch_economist_rl_lambda_split_workers.py` is the preferred Lambda
entry point for economistRL RL training. When referring to an economistRL
"Lambda run", this is the launcher that should be used.

It starts `scripts/lambda/run_economist_rl_split_workers.py`, which maintains a
constant rollout queue and feeds PPO updates back into the next rollout batch.
The older cycle launcher remains available for controlled smoke runs and
debugging, but it is deprecated and should not be used for new Lambda runs.

## Runtime Shape

1. The runner writes a numbered run descriptor under
   `benchmarks/results/economistRL/runs/run_NNNN_<UTC>/`.
2. The Rollout Worker runs one scored rollout batch using the latest approved
   adapter.
3. Scored rows with `score.training_usable != false` are appended to a durable
   queue at `benchmarks/results/economistRL/split_worker/training_queue.jsonl`.
4. The PPO Optimisation Worker consumes untrained queue rows when at least
   `--ppo-min-samples` are available.
5. PPO runs in the existing isolated subprocess trainer.
6. A trained candidate with `.economist_rl_ppo_trained` is promoted into
   `state.json` as the next `latest_approved_adapter`.
7. The next rollout batch loads that promoted adapter.
8. On completion or failure, `RUN_MANIFEST.json` and `RUN.md` are finalized with
   the run status and phase summaries.

## Run Schema And Status

Every split-worker run writes:

- `benchmarks/results/economistRL/runs/run_NNNN_<UTC>/RUN_MANIFEST.json`
- `benchmarks/results/economistRL/runs/run_NNNN_<UTC>/RUN.md`

`RUN_MANIFEST.json` uses `schema_version: economist_rl_pipeline_run_v1` and
records `run_id`, `run_number`, `run_date_utc`, `started_at_utc`,
`finished_at_utc`, `entrypoint`, `pipeline_lane`, `status`,
`success_definition`, `obsolete_paths`, `usable_outputs`, `artifact_roots`,
`extraction`, `task_db`, `init_adapter`, `queue_file`, `state_file`, and
phase summaries.

Status values distinguish usable data from PPO success:

- `completed_trained`: at least one PPO update trained and was promoted.
- `completed_no_ppo`: rollout/scored data exists, but no PPO update trained.
- `failed_rollout`: rollout, evidence, scoring, or runner setup failed.
- `failed_ppo`: PPO phase failed.
- `completed_no_data`: the runner exited without rollouts.

PPO success means a candidate adapter has `status: trained` and the PPO training
marker. Rollout-only completion is usable data for analysis or future PPO, but
is not a successful PPO run.

Lambda watcher extracts are linked to the same run identity. Extract manifests
use `schema_version: economist_rl_extract_manifest_v2` and include `run_id`,
`run_number`, and `run_date_utc`; the same fields are appended to
`benchmarks/results/economistRL/extracts/index.jsonl`.

## Preserved Policies

- Quantized model default remains
  `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`.
- New Lambda split-worker runs must initialize from the latest trusted PPO
  adapter when one exists. "Latest trusted" means a successfully trained and
  intentionally accepted/promoted `rl_pass_NNN` with usable LoRA weights and PPO
  training metadata. Do not fall back to `checkpoints/fe-lora-arena-apply-sft`
  unless no trusted PPO adapter exists, validation fails, or the run is
  explicitly a baseline/SFT restart.
- Lambda exports `LOCAL_BACKEND=transformers` and
  `PPO_TRAIN_BACKEND=transformers`.
- Rollout/scoring still uses the proven Lambda cycle runner, preserving context
  packing, execution evidence, reward scoring, old-logprob windowing, and
  artifact paths.
- Lambda lifecycle, watchdog, GPU telemetry, NFS artifact staging, and cleanup
  are inherited from the existing launcher utilities.

## Example

```bash
python scripts/launch_economist_rl_lambda_split_workers.py --launch-instances -- \
  --run-number 7 \
  --run-id run_0007_20260613T220000Z \
  --max-ppo-updates 2 \
  --rollout-batch-size 25 \
  --bootstrap-rollout-batch-size 50 \
  --ppo-min-samples 25 \
  --ppo-max-samples 64 \
  --ppo-mini-batch-size 16 \
  --execution-source-repo /home/ubuntu/fallen-empire
```

The launcher accepts model, task DB, adapter, and threshold overrides so future
model families or adapters can use the same orchestration without changing the
launcher code.
