### What Changed
The codebase was updated to include a new table `feedback_entries` in the SQLite database to store user feedback. Additionally, functions were added to insert and list feedback entries.

### Why it Matters
This change enhances the system's ability to capture and manage user feedback, which is crucial for improving the quality and usability of the system.

### Commands/tests run
1. **Run the server to ensure it starts without errors:**
   ```bash
   python scripts/private_dashboard_server.py
   ```

2. **Insert a test feedback entry:**
   ```python
   import sqlite3
   from scripts.private_dashboard_server import _connect, _insert_feedback

   conn = _connect()
   _insert_feedback(
       conn,
       track="test_track",
       rating=5,
       notes="This is a test feedback entry.",
       source="user_input",
   )
   conn.close()
   ```

3. **List feedback entries to verify insertion:**
   ```python
   import sqlite3
   from scripts.private_dashboard_server import _connect, _list_feedback

   conn = _connect()
   feedback = _list_feedback(conn)
   print(feedback)
   conn.close()
   ```

### Next validation steps
1. **Manual verification:**
   - Log in to the dashboard and manually submit a feedback entry to ensure it appears in the list.
   - Check that the feedback entry is stored correctly in the database.

2. **Automated tests:**
   - Write unit tests for the `_insert_feedback` and `_list_feedback` functions.
   - Ensure that the database schema migration works as expected when the script is run multiple times.

3. **Performance testing:**
   - Test the performance of the feedback insertion and listing functions under load to ensure they remain efficient.
