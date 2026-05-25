# Raskinolov Repository Analysis Report

**Repository:** https://github.com/natreed1/raskinolov  
**Analysis Date:** 2026-05-25

---

## 1. Project Purpose

**Raskinolov** (also known as `fallen-empire-lora`) is a specialized ML experimentation project for **LoRA fine-tuning** of open-source code models using Apple's MLX framework. The project fine-tunes models on code from the "Fallen Empire" game repository (a TypeScript/Next.js strategy game) to create specialized coding assistants.

**Key Goals:**
- Train LoRA adapters that understand game-specific code patterns (hex grids, Zustand state, AI systems)
- Keep large ML dependencies, weights, and checkpoints separate from the main game repository
- Compare local fine-tuned models against frontier/commercial models (OpenAI Codex, etc.)
- Build specialized routing systems that select the right model/adapter for different coding tasks

---

## 2. Main Directories & Modules

### Core Directories

| Directory | Purpose |
|-----------|---------|
| **`scripts/`** | ~30 Python scripts - the heart of the project. Includes training orchestration, benchmarking, data preparation, Gradio UIs, and evaluation harnesses |
| **`training/`** | LoRA configuration files (`lora_qwen_coder.yaml`), evolution config for benchmark mutation |
| **`benchmarks/`** | Task definitions (JSON), landing page briefs, task routing specs. Results written to gitignored `benchmarks/results/` |
| **`docs/`** | Comprehensive documentation: `PROJECT_STATE.md` (versions, defaults), `WORKFLOW.md` (CLI reference), `SESSION_LOG.md` (append-only work log), `run_history.md` (training run index), `SPECIALIZED_RUN_HISTORY.md` (RAG/auxiliary evals) |
| **`tests/`** | Python unit tests (currently minimal - only 1 file: `test_run_analysis_rag.py`) |
| **`lab_dashboard/`** | Static optimization dashboard for visualizing training ROI, with `README.md` for deployment instructions |
| **`.cursor/`** | Cursor IDE integration: hooks for auto-documentation on file edits/shell commands, project-specific rules, skills |
| **`data/`** | (gitignored) Raw exports, LoRA training splits, RAG corpora, routing datasets |
| **`checkpoints/`** | (gitignored) Trained LoRA adapters |

### Key Scripts

| Script | Responsibility |
|--------|---------------|
| **`ml_workflow.py`** | Unified orchestration CLI (74KB) - `smoke`, `prepare`, `train`, `benchmark`, `full`, `arena-acceptance`, etc. Writes run manifests and appends `docs/run_history.md` |
| **`game_task_arena.py`** | Disposable worktree testing harness for real game code tasks (79KB) |
| **`landing_page_arena.py`** | Visual arena for comparing local vs frontier models on standalone web pages (54KB) |
| **`chat_gradio.py`** | Browser chat UI with LoRA adapter support |
| **`train_ui_gradio.py`** | Browser training UI with live logs |
| **`build_lora_dataset.py`** | Converts raw repo exports to mlx-lm format (train/valid/test splits) |
| **`export_repo_for_training.py`** | Exports game repo code with secret redaction, size caps |
| **`run_game_benchmark.py`** | Lexical benchmark runner (substring matching) |
| **`run_evalplus_benchmark.py`** | Execution-based Python benchmark (HumanEval+/MBPP+) |
| **`run_documentation_agent_benchmark.py`** | RAG benchmark for documentation tasks |
| **`model_router.py`** | Cost-aware routing policy (local/frontier/hybrid) with adapter selection |
| **`private_dashboard_server.py`** | WSGI server for live telemetry dashboard (Railway deployment) |
| **`fe_ml_lab_runner.py`** | Non-interactive orchestration wrapper for recurring workflows |

---

## 3. Key Entrypoints & Commands

### Environment Setup
```bash
# Create venv (Python 3.11+ recommended, tested on 3.9.6)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Or with reproducibility constraints:
pip install -r requirements.txt -c requirements.lock.txt
```

### Quick Smoke Test
```bash
source .venv/bin/activate
python scripts/ml_workflow.py smoke  # Synthetic data + 4 train iters + benchmark
```

### Full Training Pipeline
```bash
export SOURCE_REPO=/path/to/fallen-empire  # Game repo path
source .venv/bin/activate

# Prepare data
python scripts/ml_workflow.py prepare

# Train + benchmark (400 iterations)
python scripts/ml_workflow.py full --adapter-path checkpoints/fe-lora-latest -- --iters 400

# Arena acceptance test (deterministic game-edit capability)
python scripts/ml_workflow.py arena-acceptance \
  --adapter-path checkpoints/fe-lora-qwen25-coder-7b-chunk6k-20260428 \
  --task-id loading-screen-polish
```

### Interactive UIs
```bash
# Chat with LoRA adapter
python scripts/chat_gradio.py --adapter-path checkpoints/fe-lora-latest

# Router supervisor (multi-adapter + OSS/frontier switching)
python scripts/router_chat_gradio.py  # Port 7864

# Training UI with live logs
python scripts/train_ui_gradio.py  # Port 7862

# Landing page visual arena
python scripts/landing_page_arena.py ui  # Port 7863

# Game task disposable worktree arena
python scripts/game_task_arena.py ui  # Port 7868
```

### Benchmarking
```bash
# Lexical game benchmark
python scripts/run_game_benchmark.py --adapter-path checkpoints/fe-lora-latest

# General coding benchmark
python scripts/run_game_benchmark.py --profile general --adapter-path checkpoints/fe-lora-latest

# EvalPlus execution benchmark
python scripts/ml_workflow.py evalplus --adapter-path checkpoints/fe-lora-latest --limit 5

# Documentation RAG benchmark
python scripts/ml_workflow.py documentation-rag-benchmark
```

### Tests
```bash
# Unit tests
python -m unittest discover -s tests

# Game-side ML cohort tests (requires SOURCE_REPO)
./scripts/run_game_ml_tests.sh
```

---

## 4. Dependencies & External Services

### Python Dependencies (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| **mlx-lm** | ≥0.21.0 | Apple Silicon MLX framework for LLM inference/training |
| **huggingface_hub** | ≥0.24.0 | Download models from Hugging Face |
| **safetensors** | ≥0.7.0 | Efficient tensor serialization |
| **gradio** | ≥4.44,<5 | Web UIs for chat, training, arenas |
| **evalplus** | ≥0.3.1 | HumanEval+/MBPP+ execution benchmarks |
| **rank-bm25** | ≥0.2.2 | BM25 retrieval for RAG systems |

**Current Verified Environment:**
- Python 3.9.6 (local venv), 3.11+ recommended
- mlx==0.29.3, mlx-lm==0.29.1
- transformers==4.57.6, numpy==2.0.2
- Tested on macOS (darwin 24.x), Apple Silicon (M4 Pro + 24GB RAM)

### External Services

1. **Hugging Face Hub** - Downloads base models:
   - Default: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
   - Alternatives: 1.5B/3B variants

2. **OpenAI-compatible APIs** (optional):
   - For frontier model comparisons in arenas
   - Configured via `FRONTIER_API_KEY`, `FRONTIER_MODEL`, `FRONTIER_API_BASE_URL`

3. **Railway** (optional):
   - Deploy private dashboard server (`railway.json` included)
   - Requires `FE_DASHBOARD_DB_PATH`, auth tokens

4. **GitHub** (indirect):
   - EvalPlus downloads datasets from GitHub releases
   - Fallback to local overrides via env vars

5. **Source Repository** (required for training):
   - `SOURCE_REPO` env var points to `fallen-empire` game repo
   - Read-only access for code export

---

## 5. ML/LLM Components

### Models

**Base Model (Default):**
- `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
- 4-bit quantized for Apple Silicon efficiency
- Used for training, chat, benchmarks, arena attempts

**Historical Models:**
- 1.5B and 3B Qwen2.5-Coder variants (for comparison)

### LoRA Training

**Architecture:**
- LoRA layers injected into all (`-1`) transformer layers
- Adapter-only training (base model frozen)
- Config: `training/lora_qwen_coder.yaml`

**Training Data Pipeline:**
1. Export game repo → `data/raw/repo_text.jsonl`
2. Build LoRA splits → `data/lora/game_text/{train,valid,test}.jsonl`
3. Each line: `{"text": "# path\n\n<file body>"}` for causal LM
4. Optional: `messages` format for chat-style SFT (supervised fine-tuning)

**Hygiene Features:**
- Max file size: 400KB (configurable)
- Secret redaction: regex patterns for API keys, PEM blocks, tokens
- Binary detection: skips non-text files
- Path filters: excludes `.turbo`, `credentials`, `.env`, etc.

### Specialized Systems

**1. Multi-Adapter Routing:**
- `training/adapter_registry_v1.json` - registry of specialized adapters
- Deterministic classifier routes prompts to adapters:
  - `loading_screen` - UI loading screens
  - `save_load_api_guard` - Save/load APIs
  - `economy_tooltip` - Economy UI
  - `hud_status` - HUD status UI
  - `combat_risk` - Combat systems
  - `documentation` - MLX lab prose documentation
- Policy: `local`/`frontier`/`hybrid` with confidence/ambiguity/risk scores
- Gradio UI: `router_chat_gradio.py` (port 7864)

**2. Progressive Context Retrieval:**
- Arena tasks use 2-stage prompting:
  1. Model requests exact files it needs
  2. Harness appends bounded code snippets
- Reduces context window pressure for complex tasks
- Enabled by default for non-low-complexity arena tasks

**3. RAG Systems (Dual Lanes):**

**Documentation RAG:**
- Corpus: `data/rag/documentation_agent_corpus.json`
- Tasks: `benchmarks/documentation_agent_rag_tasks_v1.json`
- Answers repo practices/policy questions
- Runner: `scripts/run_documentation_agent_benchmark.py --use-rag`

**Run-Analysis RAG:**
- Corpus: `data/rag/run_analysis_agent_corpus.json` (from run manifests/history)
- Tasks: `benchmarks/run_analysis_rag_tasks_v1.json`
- Documents/analyzes training runs
- Runner: `scripts/run_run_analysis_agent_benchmark.py --use-rag`

### Arena Evaluation Framework

**Game Task Arena:**
- Disposable git worktrees for real game-code edits
- Deterministic scoring: apply success, TypeScript checks, export preservation, preview readiness
- **Arena Capability Index (ACI):** 0-100 score combining correctness, integration, efficiency
- Tasks span UI/backend/combat/economics (6 standardized core tasks)
- Progressive context, TypeScript retry loops, export guards
- Artifacts: diffs, logs, generation metrics, ratings
- Training data: pairwise comparisons (winner/loser) for SFT

**Landing Page Arena:**
- Standalone static website generation
- Visual/product quality ratings (5-point scales)
- Local vs Frontier side-by-side comparison
- No game repo edits (isolated `benchmarks/results/landing_page_trials/`)

**Benchmark Evolution:**
- `scripts/evolve_benchmark_seasons.py` - mutate eval tasks over "seasons"
- Tightens suite as models improve
- Config: `training/evolution_config.json`

---

## 6. Tests & How to Run Them

### Test Coverage (Limited)

**Current Tests:**
- `tests/test_run_analysis_rag.py` - Run-analysis RAG correctness

**Missing Coverage:**
- No tests for core training/inference logic
- No tests for arena harnesses
- No tests for data preparation scripts
- No tests for router policy

### Running Tests

```bash
# All unit tests
python -m unittest discover -s tests

# Specific test
python -m unittest tests.test_run_analysis_rag -v

# Documentation RAG benchmark (treated as integration test)
python scripts/ml_workflow.py documentation-rag-benchmark

# Game-side ML cohort tests (requires SOURCE_REPO)
cd ~/fallen-empire && npm run test:ml-cohort
# Or from ML repo:
./scripts/run_game_ml_tests.sh
```

### Smoke Testing

```bash
# Fast sanity check (synthetic data, 4 iters, benchmark)
python scripts/ml_workflow.py smoke  # ~20-30s

# Base model generation test
python scripts/smoke_base_model.py --max-tokens 96
```

### Benchmarking as Tests

The project treats comprehensive benchmarking as the primary validation:

```bash
# Lexical benchmark
python scripts/run_game_benchmark.py --profile game
python scripts/run_game_benchmark.py --profile general

# Execution benchmark
python scripts/ml_workflow.py evalplus --limit 5

# Arena acceptance (deterministic game-edit capability)
python scripts/ml_workflow.py arena-acceptance --task-id loading-screen-polish

# Routing accuracy
python scripts/run_routing_benchmark.py --mode both
```

---

## 7. Architectural Improvements

### 1. **Test Coverage & CI Pipeline**

**Current State:** Minimal unit tests, no CI

**Recommendations:**
- Add pytest with fixtures for common test scenarios (model loading, dataset generation)
- Unit tests for critical paths:
  - `build_lora_dataset.py` - validate train/valid/test splits, chunking logic
  - `export_repo_for_training.py` - test secret redaction, size caps
  - `model_router.py` - routing policy correctness
  - `game_task_arena.py` parser - already has `test_game_task_arena_parser.py`, expand coverage
- Integration tests:
  - End-to-end smoke workflow with tiny synthetic data
  - Arena apply/verify/preview pipeline
- GitHub Actions CI:
  - Linting (ruff/black)
  - Type checking (mypy)
  - Unit tests on push
  - Nightly smoke training run (Apple Silicon runner)
- Pre-commit hooks for code quality

**Benefits:** Catch regressions before production runs, enable confident refactoring, document expected behavior

---

### 2. **Modularize Monolithic Scripts**

**Current State:** `ml_workflow.py` (74KB), `game_task_arena.py` (79KB), `landing_page_arena.py` (54KB) are massive single-file scripts

**Recommendations:**
- Create Python packages:
  ```
  src/
    fe_lora/
      __init__.py
      workflow/
        orchestrator.py  # ml_workflow.py core logic
        steps.py  # prepare, train, benchmark steps
        manifest.py  # run manifest serialization
      arena/
        game_task.py  # game task arena core
        landing_page.py  # landing page arena core
        parser.py  # fenced file block parsing
        worktree.py  # git worktree management
      routing/
        policy.py  # routing policy
        classifier.py  # adapter classification
        registry.py  # adapter registry
      data/
        export.py  # repo export
        dataset.py  # LoRA dataset building
        rag.py  # RAG corpus building
      ui/
        chat.py  # chat Gradio UI
        training.py  # training Gradio UI
        arena.py  # shared arena UI components
  ```
- Keep CLI scripts thin (argparse + package imports)
- Benefits:
  - Easier to test individual components
  - Shared code reuse (e.g., worktree management across arenas)
  - Clearer dependency boundaries
  - Faster IDE navigation/intellisense

---

### 3. **Configuration Management System**

**Current State:** Env vars scattered across scripts, YAML for LoRA only, JSON for various configs

**Recommendations:**
- Adopt `pydantic-settings` for typed configuration:
  ```python
  from pydantic_settings import BaseSettings
  
  class TrainingConfig(BaseSettings):
      source_repo: Path
      adapter_path: Path
      model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
      max_tokens: int = 512
      temperature: float = 0.0
      
      class Config:
          env_prefix = "FE_LORA_"
          env_file = ".env"
  ```
- Centralized config schema:
  - `config/base.yaml` - defaults for all environments
  - `config/dev.yaml` - development overrides
  - `config/prod.yaml` - production settings
  - Support env var overrides (12-factor app pattern)
- Validation at startup (fail fast on misconfiguration)
- Config versioning in manifests
- Benefits:
  - Type safety, autocomplete in IDEs
  - Clear documentation of all configuration options
  - Easier to reproduce runs from manifests
  - Reduces "works on my machine" issues

---

### 4. **Observability & Monitoring Infrastructure**

**Current State:** `docs/run_history.md` append-only table, per-run manifests, private dashboard server (basic)

**Recommendations:**
- **Structured Logging:**
  - Replace `print()` with `structlog` or `loguru`
  - JSON logs with trace IDs linking related operations
  - Log levels: DEBUG (model internals), INFO (workflow steps), WARNING (degraded perf), ERROR (failures)
  
- **Metrics Collection:**
  - Track key metrics:
    - Training: loss curves, throughput (tokens/s), memory usage, GPU utilization
    - Inference: latency (p50/p95/p99), token counts, cache hit rates
    - Arena: apply success rate, TypeScript pass rate, retry counts
    - Routing: adapter selection distribution, confidence scores
  - Export to Prometheus/Grafana or lightweight SQLite + Plotly dashboards
  
- **Alerting:**
  - Training divergence (loss spikes, NaN gradients)
  - Arena regression (ACI drops >10pp)
  - Routing drift (adapter distribution shifts unexpectedly)
  
- **Distributed Tracing:**
  - For complex workflows (arena trials with retries), trace spans:
    - `arena_trial` → `generate` → `apply` → `verify` → `preview`
  - OpenTelemetry for vendor-neutral tracing
  
- **Enhanced Dashboard:**
  - Expand `private_dashboard_server.py`:
    - Time-series plots (ACI over runs, loss curves)
    - Adapter comparison matrices
    - Cost tracking (API calls, compute hours)
    - Alerting UI

**Benefits:** Debug production issues faster, understand training dynamics, catch regressions early, optimize costs

---

### 5. **Dependency & Environment Management**

**Current State:** `requirements.txt` + optional `requirements.lock.txt`, venv-based

**Recommendations:**
- **Migration to `uv` or `poetry`:**
  - `uv` (modern, fast):
    ```bash
    uv init
    uv add mlx-lm huggingface_hub gradio evalplus rank-bm25 safetensors
    uv lock  # reproducible lockfile
    uv sync  # install exact versions
    ```
  - Benefits:
    - Faster dependency resolution
    - Better conflict detection
    - Built-in lockfile support
    - Virtual env management
  
- **Docker Images:**
  - Base image for Apple Silicon (`mlx` dependencies):
    ```dockerfile
    FROM ghcr.io/ml-explore/mlx-base:latest
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    ```
  - Development image (includes Gradio UIs, notebooks)
  - CI image (minimal for tests)
  - Benefits:
    - Consistent environments across machines
    - Cloud Agent onboarding (mentioned in workspace rules)
    - Easy deployment to Railway/other platforms
  
- **Dev Containers:**
  - `.devcontainer/devcontainer.json` for VS Code/Cursor
  - Pre-configured Python, extensions, env vars
  - One-click environment setup for contributors
  
- **Python Version Management:**
  - Use `pyproject.toml` to specify Python version requirements:
    ```toml
    [project]
    requires-python = ">=3.11,<3.13"
    ```
  - Document Apple Silicon compatibility constraints

**Benefits:** Eliminate "works on my machine", faster onboarding, easier Cloud Agent configuration

---

## Summary

**Raskinolov** is a sophisticated MLX-based LoRA fine-tuning lab with:
- Comprehensive training orchestration (`ml_workflow.py`)
- Multi-adapter routing with specialized task handlers
- Dual RAG systems (documentation + run analysis)
- Rigorous arena evaluation (deterministic game-edit scoring)
- Rich observability (run history, manifests, dashboards)

**Strengths:**
- Well-documented (extensive `docs/` directory)
- Modular workflows (prepare → train → benchmark)
- Apple Silicon optimized
- Strong evaluation framework (arenas, RAG, execution benchmarks)

**Areas for Growth:**
- Test coverage (currently minimal)
- Code organization (monolithic scripts)
- Configuration management (env vars scattered)
- Observability tooling (basic structured metrics)
- Environment reproducibility (manual venv setup)

The proposed architectural improvements target these gaps while preserving the project's strengths. Implementing them incrementally (tests first, then modularization, then config/observability) will improve maintainability, reliability, and contributor velocity.
