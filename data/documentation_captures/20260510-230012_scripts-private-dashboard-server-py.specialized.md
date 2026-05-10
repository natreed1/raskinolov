**What changed:** Added a `feedback_entries` table to capture user feedback and a `_insert_feedback` function to save feedback. Added `_list_feedback` to retrieve feedback entries. Updated imports and type hints.

**Why it matters:** Capturing user feedback is critical for iterative improvement. This change enables the dashboard to accept and store feedback, which can inform future training and UI changes.

**Commands/tests run:** Run `scripts/private_dashboard_server.py` to start the server and test the feedback submission and retrieval endpoints. Ensure `docs/run_history.json` and `docs/session_log.json` are writable by the server process.

**Next validation steps:** Confirm feedback is saved in `docs/run_history.json` and `docs/session_log.json` when submitted through the dashboard. Verify that the feedback list endpoint returns the expected data.
