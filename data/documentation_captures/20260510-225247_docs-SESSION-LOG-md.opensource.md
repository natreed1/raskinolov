## What Changed
The codebase received updates to enhance dataset capture and improve the user interface of the private dashboard. Specifically, the dashboard now includes KPI cards for performance metrics, automated documentation captures on file edits, and a modernized UI with tabs and improved scoring clarity.

## Why It Matters
These changes improve the efficiency and usability of the dataset capture process, providing real-time insights and automated documentation generation. The updated UI makes it easier for users to review and curate captured data, leading to higher quality datasets.

## Commands/Tests Run
To verify the changes, the following commands were run:
1. Compile the `private_dashboard_server.py` script:
   ```bash
   .venv/bin/python -m py_compile scripts/private_dashboard_server.py
   ```
2. Start the local server and access it via `http://127.0.0.1:8787` with Basic auth (`admin/localdev`):
   ```bash
   .venv/bin/python scripts/private_dashboard_server.py
   ```
3. Navigate to the scoring page and verify that compare links and read-full links are present.
4. Access the `/view/doc` and `/view/compare` endpoints to ensure they render full documents and side-by-side comparisons correctly.

## Next Validation Steps
1. Review the KPI cards on the dashboard to ensure they accurately reflect performance metrics.
2. Test the automated documentation capture feature to ensure it triggers on file edits and saves artifacts in `data/documentation_captures/`.
3. Verify that the modernized UI is intuitive and easy to use, with clear tabs and actionable elements.
4. Confirm that the scoring tab provides detailed insights and actionable curation options.
