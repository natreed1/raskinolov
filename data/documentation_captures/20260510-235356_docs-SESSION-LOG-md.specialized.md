## Change Documentation Update
Date: 2026-05-10
- **What changed:** Implemented structured supervision for compare feedback in `scripts/private_dashboard_server.py`, including new SQLite tables, private endpoints, and updated UI for the `/view/compare` route.
- **Why it matters:** Enables more precise training signals for pairwise task models, facilitating better performance on game task pairwise comparisons.
- **Commands/tests run:**
- **What changed:** Implemented structured supervision for compare feedback in `scripts/private_dashboard_server.py`, including new SQLite tables, private endpoints, and updated UI for the `/view/compare` route.
- **Why it matters:** Enables more precise training signals for pairwise task models, facilitating better performance on game task pairwise comparisons.
- **Commands/tests run:** 
  - `python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py` (pass).
  - `python3 scripts/export_compare_feedback_training_data.py` (exports `pairwise_feedback.jsonl` and `rewrite_feedback.jsonl`).
  - `python3 scripts/build_game_task_pairwise_dataset.py --compare-feedback-pairwise` (appends compare-derived chat SFT rows to pairwise dataset).
- **Next validation steps:** Verify training runs capture expected signals, and pairwise models match human performance on recent comparisons.
