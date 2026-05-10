## What Changed
The codebase received updates to enhance dataset capture and improve the user interface of the private dashboard. The primary changes include:

1. **KPI-First Analytics**: The private dashboard now focuses on key performance indicators (KPIs) such as run reliability, accuracy, average run time, and documentation capture reliability.
2. **Automation for Dataset Quality**: A new script `scripts/trigger_doc_training_on_changes.py` was added to automatically capture change documentation on watched edits, writing artifacts to `data/documentation_captures/`.
3. **Modern Dashboard UI**: The `scripts/private_dashboard_server.py` was updated to a modern tabbed experience with a blue/white theme and dark-mode toggle. It includes a feedback mode and token-usage estimation.
4. **Scoring Tab Clarity**: The scoring tab was improved to include deltas, missing-keyword visibility, and a clearer compare panel. Full-document links and side-by-side views were added for better exploration of scoring outputs.
5. **Curation Workflow**: The compare page was updated to include per-output curation panels with actions like feedback, training inclusion/exclusion, and deletion.

## Why It Matters
These changes improve the efficiency and usability of the dataset capture process. By automating documentation captures and providing clearer visualizations and controls, users can more effectively manage and analyze their data. The modernized dashboard and scoring tab enhancements provide better insights and actionable insights, leading to more informed decision-making.

## Commands/Tests Run
To verify the changes, the following commands were run:

1. **Python Compilation**:
   ```bash
   .venv/bin/python -m py_compile scripts/private_dashboard_server.py
   ```

2. **Local Server Verification**:
   - Started the local dashboard server:
     ```bash
     .venv/bin/python scripts/private_dashboard_server.py
     ```
   - Accessed the server at `http://127.0.0.1:8787` with Basic auth (`admin/localdev`) to verify:
     - The scoring page includes compare links
