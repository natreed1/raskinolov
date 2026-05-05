# Private Telemetry Dashboard Deployment and Operations Doc

## Purpose

This document outlines the deployment and operations for a private telemetry dashboard using Railway. The dashboard is designed to store incoming telemetry in SQLite and serve a private page with HTTP Basic auth. It includes sections for architecture, deployment, local hook wiring, security, and operations checklist.

## Architecture

- **Server**: `scripts/private_dashboard_server.py`
  - `GET /` private HTML dashboard (HTTP Basic auth)
  - `GET /api/summary` private JSON
  - `GET /api/events?limit=100` private JSON
  - `POST /api/ingest` bearer-token ingest endpoint
  - `GET /healthz` public health probe
- **Deployment config**: `railway.json`
- **Cursor hooks**: `.cursor/hooks/lab_hook_after_shell_autodoc.py`, `.cursor/hooks/lab_hook_stop_append.py`

## Deploy (Railway)

1. **Create a new Railway project** from this repo.
2. **Add environment variables**:

```bash
FE_DASHBOARD_USER=your_username
FE_DASHBOARD_PASSWORD=your_long_random_password
FE_DASHBOARD_INGEST_TOKEN=your_long_random_ingest_token
FE_DASHBOARD_DB_PATH=/data/private_dashboard.sqlite3
```

3. **Add a persistent volume** and mount at `/data`.
4. **Deploy** (Railway uses `railway.json` start command automatically).
5. **Open your service URL** and log in via HTTP Basic auth.

## Local Hook Wiring

Set these in your shell profile (`~/.zshrc`) on your machine:

```bash
export FE_LAB_REMOTE_INGEST_URL="https://<your-railway-domain>/api/ingest"
export FE_LAB_REMOTE_INGEST_TOKEN="<same-as-FE_DASHBOARD_INGEST_TOKEN>"
```

Your existing hooks will then:

- Keep writing local ledgers (`lab_dashboard/*.jsonl`, `docs/SPECIALIZED_RUN_HISTORY.md`).
- Also POST each captured event to your private dashboard.

## Security

- **Use strong random values** for `FE_DASHBOARD_PASSWORD` and `FE_DASHBOARD_INGEST_TOKEN`.
- **Treat ingest token** like an API secret.
- **Rotate both secrets periodically**.
- If needed, add an IP allowlist at Railway/proxy layer.

## Operations Checklist

1. **Verify end-to-end**:
   - Run a tracked command in Cursor terminal:

```bash
python -m unittest tests.test_documentation_rag -v
```

   - Visit dashboard URL and confirm:
     - `events_total` increased
     - Recent row shows your command + exit code
     - Docs snapshot sections render

2. **Monitor**:
   - Check the health endpoint (`/healthz`) to ensure the server is running.
   - Review the SQLite database for any anomalies.

3. **Rotate secrets**:
   - Update `FE_DASHBOARD_PASSWORD` and `FE_DASHBOARD_INGEST_TOKEN` with new values.
   - Update the environment variables in Railway.

4. **Update hooks**:
   - If new hooks are added, ensure they are correctly configured to forward events to the remote ingest endpoint.

5. **Backup**:
   - Regularly back up the SQLite database (`FE_DASHBOARD_DB_PATH`) to prevent data loss.

By following these steps, you can ensure the private telemetry dashboard is deployed, configured,
