## 2026-05-10 — Structured compare feedback: winner/strength + span labels + training export

**What changed:** Implemented structured supervision for compare feedback, adding new SQLite tables, private API endpoints, and updated UI for the compare view.

**Why it matters:** This change enables more precise training signals for pairwise trainers, improving the quality of the adapter stack.

**Commands/tests run:** 
- `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py` (pass)
- `pytest tests/private_dashboard_server.py::test_compare_feedback_api` (pass)
- `pytest tests/export_compare_feedback_training_data.py::test_export_compare_feedback_training_data` (pass)

**Next validation steps:** Verify training runs with exported pairwise and rewrite data, and validate that adapter stack improves on pairwise benchmarks.
