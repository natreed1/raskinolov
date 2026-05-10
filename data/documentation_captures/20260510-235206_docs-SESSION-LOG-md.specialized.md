## 2026-05-10 — Structured compare feedback: winner/strength + span labels + training export

**What changed:** Implemented structured supervision for compare feedback, adding new SQLite tables, private API endpoints, and UI changes in `/view/compare`. Added scripts to export training data and integrate optional ingestion into the pairwise dataset builder.

**Why it matters:** This enables more precise training signals for pairwise trainers, improving the quality of the adapter stack. Structured feedback also supports more sophisticated cursor vs. model comparisons.

**Commands/tests run:** 
- `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py`
- Unit tests for `scripts/export_compare_feedback_training_data.py` (see `scripts/tests/export_compare_feedback_training_data.py`).

**Next validation steps:** Verify training runs with the new pairwise dataset. Assess adapter performance on cursor vs. model comparisons.
