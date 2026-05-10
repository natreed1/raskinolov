## Change Documentation Update
Date: 2026-05-10
- **What changed:** Implemented structured supervision in `scripts/private_dashboard_server.py` for side-by-side compare. Added SQLite tables `compare_feedback` and `compare_feedback_spans` (kept legacy `feedback_entries` intact). Added private endpoints for winner/strength + validated span labels and query structured compare history. Reworked `/view/compare` UI with winner + preference strength controls, green/red span labeling, editable annotation list, submit one structured compare payload, and retained legacy per-track curation block as an optional transition path.
- **Why it matters:** Enhanced the system's ability to capture and analyze structured feedback, improving the quality and reliability of training data.
- **Commands/tests run:**
- **What changed:** Implemented structured supervision in `scripts/private_dashboard_server.py` for side-by-side compare. Added SQLite tables `compare_feedback` and `compare_feedback_spans` (kept legacy `feedback_entries` intact). Added private endpoints for winner/strength + validated span labels and query structured compare history. Reworked `/view/compare` UI with winner + preference strength controls, green/red span labeling, editable annotation list, submit one structured compare payload, and retained legacy per-track curation block as an optional transition path.
- **Why it matters:** Enhanced the system's ability to capture and analyze structured feedback, improving the quality and reliability of training data.
- **Commands/tests run:** 
  ```bash
  python3 -m py_compile scripts/private_dashboard_server.py scripts/export_compare_feedback_training_data.py scripts/build_game_task_pairwise_dataset.py
  ```
- **Next validation steps:** Verify that the new endpoints and UI components function as expected. Test the export functionality to ensure the training data is correctly formatted and includes the necessary signals.

## 2026-05-05 — Clarity upgrade: KPI-first analytics + always-on change documentation captures

- **What changed:** Upgraded the private dashboard to focus on performance/reliability/accuracy essentials. Added KPI cards, recent workflow runs, and documentation captures tables. Improved scoring tab clarity and automation for dataset quality.
- **Why it matters:** Enhanced the usability and reliability of the dashboard, providing clearer insights and automating critical processes.
- **Commands/tests run:** 
  ```bash
  python3 -m py_compile scripts/private_dashboard_server.py scripts/trigger_doc_training_on_changes.py scripts/generate_change_documentation_capture.py
  ```
- **Next validation steps:** Verify that the KPI cards and tables display accurate data. Test the automation to ensure it triggers documentation captures on watched edits.

## 202
