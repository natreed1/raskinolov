**What changed:** Added a `feedback_entries` table to capture user feedback and a `_insert_feedback` function to save feedback. Added `_list_feedback` to retrieve feedback entries. Updated imports and type hints.

**Why it matters:** Capturing user feedback is critical for iterative improvement. This change enables the dashboard to accept and store feedback, which can inform future training and UI changes.

**Commands/tests run:** `python scripts/private_dashboard_server.py` to rebuild the dashboard. `scripts/run_tests.sh` to validate the backend. Manual inspection of the `feedback_entries` table in `docs/run_history.db` is recommended.

**Next validation steps:** Confirm feedback appears in the dashboard UI. Verify that the `feedback_entries` table is populated with expected data.
