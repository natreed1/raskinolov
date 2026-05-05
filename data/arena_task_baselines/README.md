# Arena task baseline answers (training sources)

Place one UTF-8 text file per arena task:

```text
data/arena_task_baselines/<task_id>.assistant.txt
```

`task_id` must match `id` in `benchmarks/game_task_arena_examples.json` (e.g. `loading-screen-polish`).

The file body should look like a **winning arena `model_output.md`**: fenced code blocks with repo-relative paths (and/or a valid unified diff), no surrounding chat prose.

## Build LoRA JSONL

From the repo root (venv active):

```bash
python scripts/build_arena_baseline_dataset.py \
  --sources-dir data/arena_task_baselines \
  --out-dir data/lora/arena_task_baselines \
  --repeat 8
```

- Use `--require-all` to fail if any task in the JSON is missing a baseline.
- Point **`mlx_lm.lora`** at `data/lora/arena_task_baselines` for a **chat / messages** fine-tune (same row shape as `build_game_task_pairwise_dataset.py`).

## Merging several shell-based baselines into the game repo

Tasks that patch `src/components/test/TestEnvironmentShell.tsx` are written so **each** baseline carries a shell that adds **only that task’s** overlay import + conditional. If you copy several baselines into the real repo, **merge** the imports and `{environment.id === '…' && <…/>}` lines instead of overwriting the file repeatedly. A single combined shell with all environment overlays is equivalent training signal but easier to apply once.

## Refreshing baselines from a passing trial

After a green local or frontier attempt:

1. Open `benchmarks/results/game_task_trials/<trial_id>/attempts/<attempt>/model_output.md` (path from your machine).
2. Copy the raw model body into the matching `<task_id>.assistant.txt`.
3. Re-run `build_arena_baseline_dataset.py`.

Human-authored baselines in git should be **verified** with `npm run test:ml-cohort` / arena gate when possible.
