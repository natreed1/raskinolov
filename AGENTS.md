# AGENTS.md

## Cursor Cloud specific instructions

### Platform constraint

This project is designed for **macOS Apple Silicon** (M-series). The core ML
runtime (`mlx`, `mlx-lm`, `mlx-metal`) requires Apple Metal GPU acceleration.
On Cursor Cloud Linux x86_64 VMs, these packages install but **cannot execute**
MLX operations (model loading, training, inference). All Gradio UIs, smoke
tests, and `ml_workflow.py` subcommands that invoke model generation will fail
at runtime on Linux.

### What works on Linux / Cloud VMs

The following scripts and workflows run correctly without Apple Silicon:

- **Routing benchmark:** `python scripts/run_routing_benchmark.py` (deterministic policy scoring, 12/12)
- **Benchmark evolution:** `python scripts/evolve_benchmark_seasons.py --output /tmp/evolved.json`
- **Results dashboard:** `python scripts/visualize_results.py --out /tmp/dashboard.html`
- **Arena parser tests:** `python scripts/test_game_task_arena_parser.py`
- **Best adapter selector:** `python scripts/select_best_adapter.py`
- **RAG corpus builder:** `python scripts/build_run_analysis_rag_corpus.py`
- **Unit tests** (partial): `python -m unittest discover -s tests` — `test_benchmark_fixture_shape` passes; others require generated data or MLX runtime.
- **Router policy** can be imported and tested programmatically: `from scripts.model_router import RoutingPolicy, GenerationRequest, messages_from_prompt`

### What requires Apple Silicon

- `python scripts/ml_workflow.py smoke` (or any `ml_workflow.py` subcommand that loads/trains/generates with MLX)
- All Gradio UIs (`chat_gradio.py`, `train_ui_gradio.py`, `human_eval_ui.py`, `landing_page_arena.py`, `game_task_arena.py`, `router_chat_gradio.py`)
- `python scripts/smoke_base_model.py`
- `python scripts/run_game_benchmark.py` (loads MLX model for generation)

### Known repo issues

- `docs/RUNS.md` is referenced by the RAG corpus builder and tests but does not exist in the repository. This causes `test_corpus_manifest_points_to_existing_sources` and `test_retriever_returns_manifest_and_history_context` to fail.

### Environment setup

See `README.md` Quick Start for canonical setup commands. The venv is at `.venv/` in the repo root. Dependencies are in `requirements.txt` (with optional reproducibility constraints in `requirements.lock.txt`). On Linux, `python3.12-venv` system package may be needed before creating the venv.

### Developing and testing

- For routing/policy/dataset/benchmark work that doesn't need MLX, use the scripts listed above.
- For ML training/inference work, this must run on macOS Apple Silicon. See `docs/WORKFLOW.md` for the full subcommand reference and `docs/PROJECT_STATE.md` for pinned versions and current defaults.
