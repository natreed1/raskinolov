### What Changed
The codebase was updated to include new tables and columns in the SQLite database for capturing feedback and comparing artifacts. The `feedback_entries` table stores individual feedback entries, while the `compare_feedback` and `compare_feedback_spans` tables store comparisons between artifacts.

### Why it Matters
This change is crucial for enhancing the dataset capture functionality by allowing users to provide detailed feedback and compare artifacts, which will be essential for training and improving the models.

### Commands/tests run
To apply these changes, run the following commands:
```bash
# Navigate to the root directory of the repository
cd /path/to/fallen-empire-lora

# Apply the database migrations
python scripts/private_dashboard_server.py --migrate

# Run tests to ensure everything is working as expected
pytest tests/
```

### Next validation steps
1. **Verify Database Schema**: Check if the new tables and columns have been created correctly by querying the database.
    ```sql
    PRAGMA table_info(feedback_entries);
    PRAGMA table_info(compare_feedback);
    PRAGMA table_info(compare_feedback_spans);
    ```

2. **Test Feedback Entry Insertion**: Insert a sample feedback entry and verify that it is stored correctly in the database.
    ```python
    from scripts.private_dashboard_server import _insert_feedback, _connect

    conn = _connect()
    _insert_feedback(conn, track="codex_authored", rating=5, notes="Great work!", source="user1")
    conn.close()
    ```

3. **Test Artifact Comparison**: Insert a sample comparison entry and verify that it is stored correctly in the database.
    ```python
    from scripts.private_dashboard_server import _insert_compare_feedback, _connect

    conn = _connect()
    _insert_compare_feedback(conn, left_track="codex_authored", right_track="opensource", winner="left", strength="strong", notes="Left is better!", source="user2")
    conn.close()
    ```

4. **Run Integration Tests**: Execute the integration tests to ensure that the new
