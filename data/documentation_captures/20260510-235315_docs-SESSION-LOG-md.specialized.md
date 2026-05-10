## Change Documentation Update
Date: 2026-05-10
- **What changed:** Implemented structured supervision for compare feedback in `scripts/private_dashboard_server.py`, including new SQLite tables, private endpoints, and updated UI for the `/view/compare` route.
- **Why it matters:** Enables more precise training signals for pairwise task models, facilitating better performance on game task pairwise comparisons.
- **Commands/tests run:** `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py` (pass); `pytest tests/private_dashboard_server.py` (pass).
- **What changed:** Implemented structured supervision for compare feedback in `scripts/private_dashboard_server.py`, including new SQLite tables, private endpoints, and updated UI for the `/view/compare` route.
- **Why it matters:** Enables more precise training signals for pairwise task models, facilitating better performance on game task pairwise comparisons.
- **Commands/tests run:** `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py` (pass); `pytest tests/private_dashboard_server.py` (pass).
- **Next validation steps:** Verify training export quality with `scripts/analyze_pairwise_training_data.py` and run pairwise task adapter training to observe performance improvements.
