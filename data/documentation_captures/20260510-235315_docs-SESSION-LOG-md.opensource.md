## Change Documentation Update
Date: 2026-05-10
- Implemented structured supervision in `scripts/private_dashboard_server.py` for side-by-side compare:
- Added SQLite tables `compare_feedback` and `compare_feedback_spans` (kept legacy `feedback_entries` intact).
- Added private endpoints:
- Implemented structured supervision in `scripts/private_dashboard_server.py` for side-by-side compare:
  - Added SQLite tables `compare_feedback` and `compare_feedback_spans` (kept legacy `feedback_entries` intact).
  - Added private endpoints:
    - `POST /api/compare-feedback` (winner/strength + validated span labels)
    - `GET /api/compare-feedback` (query structured compare history)
- Reworked `/view/compare` UI:
  - Winner + preference strength controls
  - Green/red span labeling from selected text
  - Editable annotation list (reason + optional rewrite for red spans)
  - Submit one structured compare payload
  - Retained legacy per-track curation block as optional transition path
- Added `scripts/export_compare_feedback_training_data.py`:
  - Exports `pairwise_feedback.jsonl` with hybrid weights (strength + bad-span penalty signal)
  - Exports `rewrite_feedback.jsonl` from red spans with provided rewrite text
  - Writes `data/lora/compare_feedback/manifest.json`
- Integrated optional ingestion into `scripts/build_game_task_pairwise_dataset.py`:
  - New `--compare-feedback-pairwise` and `--compare-feedback-repeat` flags append compare-derived chat SFT rows.

### Why it matters
- Enhanced dataset quality through structured supervision and feedback mechanisms.
- Improved UI for better user interaction and data visualization.
- Added new export functionality for training data, facilitating further model training and improvement.

### Commands/tests run
- `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py`

### Next validation steps
- Verify that the new endpoints (`/api/compare-feedback`) are functioning correctly.
- Test the export functionality to ensure the generated files are in the correct format and contain the expected data.
- Review the UI changes to ensure they
