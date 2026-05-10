**What changed:** The script `scripts/private_dashboard_server.py` was updated to include new tables for capturing user feedback and compare results, and to add utility functions for hashing text and working with UTC timestamps.

**Why it matters:** These changes enable the private dashboard to capture and store user feedback and compare results, which are essential for training and iterating on the ML models. The new tables provide a structured way to store this data, and the utility functions ensure that the data is stored consistently and efficiently.

**Commands/tests run:** Run `python scripts/private_dashboard_server.py` to rebuild the dashboard. No additional tests are required at this stage.

**Next validation steps:** Verify that the new tables are created in the database and that the script runs without errors. Check that the new tables are populated with data when user feedback and compare results are captured.
