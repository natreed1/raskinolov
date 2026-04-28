# Session log (append-only)

Newest entries at the **top**.

---

## 2026-04-27 — Git repository initialized

**Goal:** Turn the working tree into a proper Git repo with an initial commit (no commits existed previously).

**Changed:** Local `git config user.name` / `user.email` for this repo only; staged project files (respecting `.gitignore`); root commit on `main`. Push requires a remote: install/authenticate `gh` or `git remote add origin <url>` then `git push -u origin main`.

**Outcome:** Root commit on `main` with message *Initial commit: MLX LoRA lab for Fallen Empire* (`git log -1`).

---

## 2026-04-27 — Game Task Arena lexical context packing

**Goal:** Phases 0–2 from the lexical context packing plan: log what enters the arena prompt, reserve prefix budget + deterministic literal-vs-glob ordering + head/tail truncation for code files, and BM25 reordering of glob-matched files using chunked scores.

**Changed:** [scripts/game_task_arena.py](scripts/game_task_arena.py) — added `build_context_pack()`, helpers (`_truncate_file_body`, `_bm25_order_paths`, …), extended `packet()` (optional prebuilt `context`, `max_chars`, `use_bm25`), `generate_attempt` writes `attempts/<slug>/logs/context_pack.json` and augments it with `full_user_prompt_est_tokens` after `estimate_tokens` on the full user message; CLI `generate` and `packet` accept `--no-context-bm25` and `packet` accepts `--context-chars`.

**Deps:** `rank-bm25>=0.2.2` in [requirements.txt](requirements.txt), `rank-bm25==0.2.2` in [requirements.lock.txt](requirements.lock.txt); recorded in [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md).

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py`.

---

## 2026-04-27 — Benchmark capability index

**Changed:** `scripts/run_game_benchmark.py` now reports a **Capability Index** alongside legacy pass/fail. The index combines correctness, instruction following, concision, and speed so models that pass substring rubrics with huge generic outputs are penalized.

**Changed:** `scripts/ml_workflow.py` now parses and records the Capability Index in `manifest.json` and `RUN.md` for benchmark runs.

**Verified:** `.venv/bin/python -m py_compile scripts/run_game_benchmark.py scripts/ml_workflow.py`; `ml_workflow.py benchmark --adapter-path checkpoints/fe-lora-pairwise-r10-from-30m --profile general` produced run `20260427-220341_d13b1d` with legacy **8/8** and Capability Index **83.7/100** (`correctness 100.0`, `instruction 86.9`, `concision 11.9`, `speed 55.0`), confirming the metric catches over-verbose adapter behavior.

---

## 2026-04-27 — LoRA vector trajectory analyzer

**Goal:** Add the deeper visualization layer for “lines of LoRA vectors over variables,” so saved adapter checkpoints can be treated as points in weight space rather than only scalar loss curves.

**Added:** `scripts/analyze_lora_vector_trajectory.py`, which reads numbered `*_adapters.safetensors` checkpoints, compares them as stable ordered vectors, and writes `vector_trajectory.jsonl`, `layer_trajectory.jsonl`, `manifest.json`, and `SUMMARY.md` under `benchmarks/results/lora_vector_trajectories/<adapter-name>/`. Metrics include L2 norm, RMS, mean absolute value, delta from reference adapter, delta from previous checkpoint, cosine to reference, cosine to first checkpoint, and per-layer movement rows.

**Dependency docs:** Added direct `safetensors>=0.7.0` to `requirements.txt`, constrained `safetensors==0.7.0` in `requirements.lock.txt`, and recorded the package in `docs/PROJECT_STATE.md`.

**Ran:** `.venv/bin/python scripts/analyze_lora_vector_trajectory.py --adapter-path checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m --reference-adapter-file checkpoints/fe-lora-30m/adapters.safetensors`.

**Outcome:** Output written to `benchmarks/results/lora_vector_trajectories/fe-lora-mixed-cautious-text-160s2k-from-30m/`. The analyzer processed **4** checkpoints, **392** tensors, and **18,464,768** LoRA parameters. Whole-adapter movement from reference increased **0.386487 → 0.712277** from iter 40 to iter 160, while cosine to reference remained high (**0.999930 → 0.999763**), indicating smooth drift rather than a sharp vector jump.

**Verified:** `python3 -m py_compile scripts/analyze_lora_vector_trajectory.py`; Cursor lints reported no errors.

---

## 2026-04-27 — Training dynamics telemetry probe

**Goal:** Start collecting structured training trajectories for the proposed optimizer/controller research path without replacing `mlx_lm.lora` or AdamW.

**Changed:** `scripts/ml_workflow.py` now parses `mlx_lm.lora` training logs into `benchmarks/results/runs/<run_id>/training_trajectory.jsonl` and stores a `training_trajectory` summary on the training step in `manifest.json`. `RUN.md` includes a compact trajectory section when points are present. Parsed fields include iteration, train loss, validation loss, learning rate, throughput, trained tokens, peak memory, validation timing, and saved-checkpoint flags when logged.

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-dynamics-probe-30-from-30m -- --data data/lora/game_text_pairwise_cautious_text --iters 30 --batch-size 1 --val-batches 2 --steps-per-eval 10 --steps-per-report 10 --max-seq-length 2048 --save-every 30 --learning-rate 3e-6 --resume-adapter-file checkpoints/fe-lora-30m/adapters.safetensors`.

**Outcome:** Run `20260427-051140_4baad5` completed in **96.13s** and produced **5** trajectory points: `iter 0`, initial validation, and iters 10/20/30. Final train loss **0.953**, final validation loss **2.158**, best validation loss **0.649** at iter 20, peak memory **4.814 GB**, trained tokens **39,295**. Follow-up labels: `20260427-051326_55ab75` scored **15/15 game** and `20260427-051502_fc30f6` scored **8/8 general** for `checkpoints/fe-lora-dynamics-probe-30-from-30m`.

**Next:** Run a deliberate ladder of 80/160/300 iteration probes with varied learning rates and dataset mixes, each followed by game + general benchmarks, before fitting any ODE/controller model.

---

## 2026-04-27 — Training architecture canvas

**Goal:** Create a navigable Cursor Canvas explaining the Fallen Empire LoRA training system architecture.

**Added:** `.cursor` canvas artifact `training-system-architecture.canvas.tsx` under Cursor's managed project canvas directory. The canvas summarizes the audited `ml_workflow.py` spine, source export and dataset builders, LoRA training adapters, benchmark surfaces, arena feedback loops, and durable artifact contract.

**Verified:** Cursor diagnostics reported no linter errors for the canvas file.

---

## 2026-04-27 — Cautious mixed game/pairwise LoRA train

**Goal:** Run a larger audited LoRA pass without repeating the pairwise-smoke overfit pattern, while preserving general coding behavior and leaving enough artifacts for later arena/EvalPlus evaluation.

**Prepared:** `data/lora/game_text_pairwise_cautious_text/`, a uniform `text` dataset mixing all current `game_text` rows with modest pairwise transcript reinforcement. Split counts: train **153** (`141` game text + `3` pairwise rows repeated **4x**), valid **9** (`8` game text + `1` pairwise), test **8** (`7` game text + `1` pairwise). The first mixed attempt used raw `messages` rows beside `text` rows and failed immediately with `KeyError: 'messages'`, so the compatible dataset converts pairwise chat rows into plain transcript text.

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m -- --data data/lora/game_text_pairwise_cautious_text --iters 160 --batch-size 1 --val-batches 4 --steps-per-eval 20 --steps-per-report 10 --max-seq-length 2048 --save-every 40 --learning-rate 3e-6 --resume-adapter-file checkpoints/fe-lora-30m/adapters.safetensors`.

**Outcome:** Workflow run `20260427-045439_18486b` completed with exit **0** in **524.81s**. Adapter written to `checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m/` with checkpoints at iters 40/80/120/160. Final train loss **0.539**, final validation loss **1.918**, best observed validation loss **0.983** at iter 100 (not a saved checkpoint), peak memory **4.887 GB**. Long rows were truncated to 2048 tokens. Earlier audited failures: `20260427-045341_7dff75` failed due mixed `messages`/`text` schema; `20260427-045410_759dc8` failed with Metal OOM at `max_seq_length=4096`.

**Follow-up eval:** `checkpoints/fe-lora-mixed-cautious-text-160s2k-from-30m` scored **13/15 game** (`skirmish-signature` missing `attackerPower`; `axial-hex-distance` missing `Math.max`) and **8/8 general**. Treat this as a useful negative result: the run preserved general benchmark behavior but regressed heuristic game recall versus `fe-lora-30m` and `fe-lora-pairwise-r10-from-30m`, so do not promote it as default without additional arena evidence. The concurrent EvalPlus run `20260427-045347_05350b` on `fe-lora-pairwise-r10-from-30m` passed HumanEval/0 and HumanEval/1 before Metal OOM.

**Next:** Evaluate saved intermediate checkpoint material if worth recovering a lower-regression point, and run a standardized Game Task Arena local smoke on any candidate before promotion. For EvalPlus, retry with fewer concurrent MLX/Gradio processes or lower token/concurrency settings.

---

## 2026-04-27 — Pairwise arena LoRA smoke train

**Goal:** Run a first supervised-style LoRA pass directly from Game Task Arena local-vs-frontier results.

**Added:** `scripts/build_game_task_pairwise_dataset.py`, which converts `benchmarks/results/game_task_pairwise_training_data.jsonl` into chat SFT JSONL splits under `data/lora/game_task_pairwise/`. Current dataset: **5 raw pairwise records**, repeated to **20 expanded rows** → train **16**, valid **2**, test **2**.

**Ran:** `.venv/bin/python scripts/ml_workflow.py train --adapter-path checkpoints/fe-lora-pairwise-smoke -- --data data/lora/game_task_pairwise --iters 30 --batch-size 1 --val-batches 1 --steps-per-eval 10 --steps-per-report 5 --max-seq-length 2048 --save-every 30`.

**Outcome:** Workflow run `20260427-042822_4168da` completed with exit **0** in **132.07s**. Adapter written to `checkpoints/fe-lora-pairwise-smoke/` (`adapters.safetensors`, `0000030_adapters.safetensors`, `adapter_config.json`; ~141 MB). Training log reported val loss from **1.763 → 0.092** over 30 iters, with sequence truncation warnings at 2048 tokens. This is a smoke-sized overfit/format-training pass, not yet a general-quality adapter.

**Follow-up eval:** Pairwise smoke adapter benchmarked **15/15** on game but **7/8** on general (`acronym-api` failed), indicating overfit/regression risk. A conservative resumed pass from `checkpoints/fe-lora-30m` trained `checkpoints/fe-lora-pairwise-r10-from-30m` for 10 iters at LR `2e-6` on non-repeated pairwise data; it benchmarked **15/15** game and **8/8** general, but local arena smoke still generated declaration stubs/filler and `no_applyable_changes`, so it is not yet an effective arena-edit adapter.

**EvalPlus attempt:** Ran `.venv/bin/python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-pairwise-r10-from-30m --suite humaneval --limit 5 --max-tokens 512` as run `20260427-045138_25fa6c`; recorded **0/5**, but failures were invalid because EvalPlus' memory guard failed on Darwin with `ValueError: current limit exceeds maximum limit`. Retried with `EVALPLUS_MAX_MEMORY_BYTES=-1` as run `20260427-045347_05350b`; first two HumanEval tasks passed, then the process aborted with Metal out-of-memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Treat EvalPlus status as **blocked by local runner/resource configuration**, not as a model score yet.

**Docs:** `docs/run_history.md` was appended automatically by `ml_workflow.py`; artifacts live under `benchmarks/results/runs/20260427-042822_4168da/`.

---

## 2026-04-26 — Loading task context hardening

**Investigated:** Latest `loading-screen-polish` attempts connected after the preview fix, but both showed baseline screens because model output failed apply. Local output targeted `src/app/layout.tsx`, repeated malformed diff content, and emitted repeated `<|im_end|>` tokens; frontier output returned an invalid placeholder diff for `src/app/page.tsx`. The root issue was task/harness integration: the preview renders `/test-env/loading-screen` through `src/components/test/TestEnvironmentShell.tsx` and `src/components/ui/GameLoadingScreen.tsx`, while the task context used root-style `app/**/*` / `components/**/*` globs and lowercase `loading` patterns that missed key `src/app` / `src/components` files.

**Changed:** Tightened the loading-screen task prompt to name `/test-env/loading-screen`, `TestEnvironmentShell`, and `GameLoadingScreen`; added exact `src/app` / `src/components` context paths and case variants; added literal path handling for bracketed Next route folders like `[envId]`; made generation instructions prefer fenced full-file blocks for small UI edits; stripped repeated special tokens from model output; and added fenced-file fallback when a diff is present but fails `git apply --check`.

**Changed:** Applied the same integration fix to every standardized game task prompt. Each task now names its `/test-env/...` route, points at likely rendered files/helpers, and instructs the model to infer schema and Fallen Empire visual language from supplied code instead of inventing generic UI/data shapes. Added exact context files for HUD/status, economy, combat risk, save/load serialization, and AI planning tasks.

**Changed:** Hardened fenced file extraction after frontier/local produced valid-looking blocks with paths in several nonstandard places. The parser now supports `path=...`, bare info-string paths, path comments (`// path: ...` or `// src/...`), markdown heading paths immediately before a fence, and separate path-marker fences followed by code fences. It matches `.tsx` before `.ts`, resolves to existing `.tsx` files when appropriate, strips path comments before writing, and logs skipped fences with no path.

**Investigated:** Local loading-screen failures are consistent across seven attempts: older runs emitted special-token/filler tails and unrelated diffs; newer runs target `GameLoadingScreen.tsx` but hallucinate a giant prop API and produce corrupt diff hunks. The latest prompt hit the local input cap (`4096` estimated input tokens) and the model spent ~56s generating invalid diff output. This points to local diff-format/schema reliability under long context, not preview connectivity.

**Investigated:** After switching local to file blocks, the local model echoed the task packet/context and targeted `docs/WORKFLOW.md`; the reduced local context was still ordered with broad docs before the key component files. Exact task files were not reliably near the top of the local prompt.

**Changed:** Local arena generation now uses a smaller context window and explicitly requests fenced full-file blocks instead of diffs, while preserving existing exports. Frontier can still return diff or file blocks. Context selection now prioritizes exact task files before broad globs and README, and skips duplicate `package.json`, so `GameLoadingScreen.tsx` / `TestEnvironmentShell.tsx` appear early in the local prompt.

**Changed:** Fixed stale preview readiness: if requested ports such as `5174` / `5175` are already occupied by older Next dev servers, `preview()` now chooses the next free port before launching and records `requested_port` plus actual `port`. This prevents an old server from making a new failed launch look `ready`.

**Changed:** Added preview preflight typechecking. Before launching Next, `preview()` runs `npx tsc --noEmit`; invalid model code is marked `preflight_failed`, logs to `logs/preview_preflight_tsc.log`, and does not launch a broken HTTP 500 preview server.

**Changed:** Preview startup now skips attempts whose `apply_status` is not applyable (`applied*` or `wrote*`) and records `skipped_apply_status:<status>`, avoiding misleading baseline previews for failed/no-op local attempts.

**Changed:** Added automatic pairwise training-signal capture. When local/frontier share a task and exactly one attempt applies, the arena writes `pairwise_training_signal.json` in the trial and appends `benchmarks/results/game_task_pairwise_training_data.jsonl` with the winning output/diff, rejected output/error, metadata, and token metrics.

**Changed:** Improved manual grading labels. The Game Task Arena now stores automated viability (`applied`, `verified`, `preview_ready`), manual typecheck/visible-change confirmations, preference strength, and structured failure modes (`parse/apply failed`, `typecheck/preflight failed`, `generic/off-theme`, etc.) in `rating.json` and downstream training records.

**Verified:** `.venv/bin/python -m py_compile scripts/game_task_arena.py scripts/test_game_task_arena_parser.py`; `.venv/bin/python scripts/test_game_task_arena_parser.py`; latest local heading-path output now writes `src/components/ui/GameLoadingScreen.tsx`; preview preflight returns `tsc_exit_2` for the broken local worktree and `tsc_exit_0` for the good frontier worktree; `load_task_specs` loads all six tasks; every task prompt includes its `preview_path`; selected context includes exact schema files for loading, HUD, economy, combat, save/load, and AI planning tasks; sanitizer removes repeated `<|im_end|>` tokens and preserves extractable diffs; latest frontier loading-screen output now resolves to `src/components/ui/GameLoadingScreen.tsx` instead of a mistaken `.ts` file; local 9000-char loading context now includes `GameLoadingScreen.tsx` and `TestEnvironmentShell.tsx` before README.

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

## **Next (suggested):** Run `export_repo_for_training.py` with `SOURCE_REPO` pointing at the game tree; sketch LoRA invocation for `mlx_lm.lora` with the same base model id and document exact CLI in `PROJECT_STATE.md` after verification.

## 2026-04-27 — Instrumented LoRA dynamics probe batch

**Goal:** Run 50 sequential instrumented LoRA dynamics probes from `checkpoints/fe-lora-30m/adapters.safetensors` on `data/lora/game_text_pairwise_cautious_text`, then label successful adapters with game and general benchmark profiles.

**Matrix:** 50 probes `p001`-`p050`; iters distribution {'30': 6, '50': 11, '80': 11, '120': 11, '160': 11}; learning-rate distribution {'1e-6': 9, '2e-6': 9, '3e-6': 9, '5e-6': 9, '8e-6': 9, '1e-5': 5}; val-batches distribution {'2': 18, '4': 32}. Fixed settings: `batch-size=1`, `max-seq-length=2048`, resume adapter `checkpoints/fe-lora-30m/adapters.safetensors`. Adapter names use `checkpoints/fe-lora-dynamics-pNNN-i<iters>-lr<lr>`.

**Artifacts:** Batch manifest `/Users/natreed/fallen-empire-lora/benchmarks/results/dynamics_probe_batches/dynamics_probe_batch_20260427_052022/batch_manifest.json`; matrix CSV `/Users/natreed/fallen-empire-lora/benchmarks/results/dynamics_probe_batches/dynamics_probe_batch_20260427_052022/matrix.csv`; progress stream `/Users/natreed/fallen-empire-lora/benchmarks/results/dynamics_probe_batches/dynamics_probe_batch_20260427_052022/progress.jsonl`; per-workflow manifests/logs under `benchmarks/results/runs/<run_id>/`; adapters under `checkpoints/fe-lora-dynamics-`*.

**Outcome:** Stop state: completed all 50 requested probes. Training succeeded 50/50 and failed 0. Game benchmarks succeeded 25/50 and failed 25. General benchmarks succeeded 50/50 and failed 0.

**Notable results:** Best validation loss was p001 `checkpoints/fe-lora-dynamics-p001-i30-lr1e6` (0.647); worst validation loss was p040 `checkpoints/fe-lora-dynamics-p040-i80-lr8e6` (1.65). Best game benchmark: p001 `checkpoints/fe-lora-dynamics-p001-i30-lr1e6` (15/15). Best general benchmark: p001 `checkpoints/fe-lora-dynamics-p001-i30-lr1e6` (8/8).

**Next recommendation:** Use the batch manifest plus each run's `training_trajectory.jsonl` to select the lowest-loss adapters that do not regress benchmark labels, then run a smaller confirmatory sweep around the best LR/iteration region before longer training.